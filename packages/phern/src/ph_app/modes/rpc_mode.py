"""`--mode rpc`: JSON-RPC over stdio, in the shape of the dsh SDK (D13, I-7).

The mode deliberately uses the method names and payload shapes of dsh
(`initialize`, `session/prompt`, `session.event`, `session.status`) and not a
protocol specific to pH. dsh already ships a Python client for this protocol,
and the daemon of Phase 5 extends the same surface and does not invent a second
one.

Notifications are the envelopes of the log itself, in camelCase, so an RPC
client and a log reader parse one format.

@module ph_app.modes.rpc_mode
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, TextIO

import anyio
from anyio.abc import TaskGroup

from ph.agent.types import AgentDriver
from ph.cordis import DEPLOYMENT, Context, Profile, releasing
from ph.json import dumps
from ph.keys import AGENTS, SESSIONS, TOOLS
from ph.llm.types import user_text
from ph.persistence import unwritten_at_unwind
from ph.seams.models import ModelChoice, start_on
from ph.session import Session, SessionEvent, SessionForkError
from ph.wire import WireModel

from .. import verbs
from ..daemon.recovery import resume_children
from ..payloads import (
    SessionEventNotice,
    SessionNotice,
    SessionStatusNotice,
)
from ..protocol import (
    Frame,
    MethodResult,
    NoParams,
    SessionId,
    SessionParams,
    UnknownMethod,
    capabilities,
    notification,
    parse_params,
    respond,
)
from ..runtime import mount_session, mounted, open_root, read_start
from ..sessions import RecordedStart

__all__ = ["RpcServer", "run_rpc"]

log = logging.getLogger("ph_app.modes.rpc")


# The params of the stdio transport itself (P8-07). These are the two whose
# contract differs from the contract of the daemon. Here `session/prompt` may
# omit the id, and it names the route, because there is no supervisor that holds
# one. A shared model would have to make that optional for the daemon too, and
# the daemon would then silently lose its `invalid_params` refusal.
# `session/new` may omit the id on both transports since protocol 8, because
# the server names a new session on either transport. But the `session/new` of
# the daemon also carries the directory and trust of the client, which stdio
# does not need.
#
# The shapes that both transports need belong to the protocol and are not
# copied here. `NoParams` and `SessionParams` come from `..protocol` beside
# `Cursor`, for the reason that `Cursor` states. "Each server owns its method
# table" is about the table. It does not mean that each server declares an
# empty model again.


class _NewParams(WireModel):
    session_id: SessionId | None = None


class _PromptParams(WireModel):
    session_id: SessionId | None = None
    prompt: str = ""
    provider: str | None = None
    model: str | None = None


@dataclass(slots=True)
class _Served:
    """One session that this server serves: its own mount, and its agent after
    the first prompt."""

    ctx: Context
    start: RecordedStart
    """What the session is opened with (`open_root`): its start as read before
    the mount and, after it is open, its own id and family."""
    exits: AsyncExitStack
    """The unwind of this mount, pushed on the unwind of the server
    (`RpcServer.exits`). The mount has its own unwind so that the mount of a new
    session that opened nothing can be removed immediately (`RpcServer._open`)."""
    agent: AgentDriver | None = None
    driving: anyio.Lock = field(default_factory=anyio.Lock)
    """Held while this server drives the agent (`RpcServer._drive`). A prompt and
    a wake take turns: one waits for the `run()` of the other to finish, and does
    not meet "already has active work"."""


@dataclass(slots=True)
class RpcServer:
    """One stdio JSON-RPC endpoint. Each session that it serves has its own mount (S5).

    There is one mount for each session, as the daemon has one for each root.
    The overrides of a session are realized on its mount. A shared mount
    therefore made the `/model` or allowance of one session apply to the next
    session too, with nothing in the log of the second session to record it.
    Each session is mounted from the environment in its own log
    (`mount_session`), and `profile` is what a new session starts on. All the
    mounts unwind with `exits`.
    """

    profile: Profile
    exits: AsyncExitStack
    out: TextIO
    choice: ModelChoice = field(default_factory=ModelChoice)
    """What `phern --mode rpc --provider/--model` asked for. Empty means the
    profile default."""
    tasks: TaskGroup | None = None
    """Where a turn that nobody prompted runs: a parent that the ending notice of
    a child woke between prompts (P13-05). When this is `None`, as in a server
    that a test builds, no such turn runs."""
    ended: bool = False
    """Whether the peer is done (`end`). Nothing is written to the peer after that."""
    _served: dict[str, _Served] = field(default_factory=dict)
    _deployment: Context | None = None

    async def _mount(self, session_id: str | None) -> _Served:
        """Return the mount of the session, which is mounted the first time it is
        asked for. `None` means a session that is about to be made. It has no log
        to read an environment from, and its mount is filed under its id after it
        opens (`_open`)."""
        served = self._served.get(session_id) if session_id else None
        if served is None:
            start = await read_start(session_id)
            own = AsyncExitStack()
            try:
                # An adopted version that the loader refuses is withdrawn in
                # `mount_session`, as it is for the roots of the daemon and for
                # `phern -p`. The log of the session records the reason.
                ctx, _ = await mount_session(own, start, self.profile)
            except BaseException:
                with releasing():
                    await own.aclose()
                raise
            self.exits.push_async_callback(own.aclose)
            served = _Served(ctx, start, own)
            if session_id:
                self._served[session_id] = served
        return served

    async def _open(self, served: _Served) -> Session:
        """Open a session on its mount, and file the mount under the id of the session.

        A session that is about to be made gets its id from `open_session`, as a
        new session of `phern -p` does. `open_session` does not search the store
        for an id that it made.

        **The mount of a new session that opened nothing is removed now**
        (Holdings rule 2). The mount is filed under the id that the session opens
        with, so nothing would find it again. It stayed mounted (rows, plugins,
        everything that a profile mounts) until the server ended. The mount of a
        named session stays, filed, for the next request to open."""
        try:
            session = await open_root(served.ctx, served.start)
        except BaseException:
            if served.start.session_id is None:
                with releasing():
                    await served.exits.aclose()
            raise
        served.start = replace(served.start, session_id=session.id, family=session.header.family)
        self._served[session.id] = served
        self._attach(served.ctx, session)
        return session

    def unwritten(self) -> dict[str, str]:
        """What the last write of the mount of each session could not write, by
        session id, with the reason (`unwritten_at_unwind`). It is read after the
        mounts are down."""
        return {
            session_id: reason
            for served in self._served.values()
            for session_id, reason in unwritten_at_unwind(served.ctx).items()
        }

    async def _listing(self) -> Context:
        """Return the profile itself, mounted one time. It serves what the
        deployment offers before any session exists: `tools/list`."""
        if self._deployment is None:
            self._deployment = await self.exits.enter_async_context(mounted(self.profile))
        return self._deployment

    def end(self) -> None:
        """Record that the peer is done: it sent `shutdown`, or its end of stdin closed.

        Nothing more is written to the peer, and a turn that the notice of a child
        woke ends with the tasks of the server. `ended` is set first, because a
        turn that is canceled here still reports `idle` on its way out. That frame
        would come after the end, to a peer that may have closed the pipe."""
        self.ended = True
        if self.tasks is not None:
            self.tasks.cancel_scope.cancel()

    def _write(self, payload: Frame) -> None:
        if self.ended:
            return
        self.out.write(f"{dumps(payload)}\n")
        self.out.flush()

    def _notify(self, notice: SessionNotice) -> None:
        """Send one notification, under the name that its payload declares.

        These are the same two shapes that the daemon sends. A client that reads
        this transport and a client that reads the socket parse the frame in the
        same way. "They are the same protocol" must mean that at the payload as
        well as at the envelope.
        """
        self._write(notification(notice.METHOD, notice.to_wire()))

    async def handle(self, request: dict[str, Any]) -> None:
        reply = await respond(request, self._dispatch)
        if reply is not None:
            self._write(reply)

    async def _dispatch(self, method: str, params: dict[str, Any]) -> MethodResult:
        if method in ("initialize", "daemon/hello"):
            # This is the same block that the daemon answers with, minus what
            # stdio cannot do: one process, one peer, no supervision. The params
            # are parsed and discarded. The capability block of a client means
            # nothing to a transport that will never ask the client anything,
            # but a stray field is still a stray field.
            #
            # **The parse goes through the verb, because `NoParams` contradicted
            # that comment.** `WireModel` is `extra="forbid"`, so a parse
            # against `NoParams` refused the `capabilities` list that every
            # client sends, including the frame of `DaemonClient.initialize`
            # itself. `initialize` was then two contracts under one name, while
            # `protocol.py` says "they are the same protocol". Verified:
            # `parse_params("initialize", NoParams, {"capabilities": []})` came
            # back `invalid_params: capabilities: Extra inputs are not
            # permitted`.
            verbs.INITIALIZE.parse(params)
            return capabilities("tools")
        if method == "session/new":
            # Open, not create. A peer that names a stored id resumes it. An id
            # that another process holds is refused by name (P5-03).
            opened = parse_params(method, _NewParams, params)
            session = await self._open(await self._mount(opened.session_id))
            return {"sessionId": session.id}
        if method == "session/prompt":
            return await self._prompt(parse_params(method, _PromptParams, params))
        if method == "session/events":
            asked = parse_params(method, SessionParams, params)
            # A session is only ever opened on its own mount. A session with no
            # mount here is therefore one that this server never opened.
            served = self._served.get(asked.session_id)
            if served is None:
                raise SessionForkError(
                    f'session "{asked.session_id}" not found', "SESSION_NOT_FOUND"
                )
            session = served.ctx.require(SESSIONS).require(asked.session_id)
            return {"events": [event.to_wire() for event in session.events]}
        if method == "tools/list":
            # `DEPLOYMENT` (P6-32): RPC mode advertises what the deployment
            # offers, before any agent exists to narrow it.
            parse_params(method, NoParams, params)
            schemas = (await self._listing()).require(TOOLS).schemas(scope=DEPLOYMENT)
            return {"tools": [schema.to_wire() for schema in schemas]}
        if method == "shutdown":
            verbs.SHUTDOWN.parse(params)
            # Return nothing, like the daemon. `shutdown` is a `Notify`, so
            # there is no reply model to build and `respond` sends no frame.
            # The `{"ok": True}` literal that this replaced was therefore
            # pointless, and not merely duplicated.
            return None
        raise UnknownMethod(f'unknown method "{method}"')

    def _attach(self, ctx: Context, session: Session) -> None:
        def emit(source: Session, event: SessionEvent) -> None:
            if source.id != session.id:
                return
            self._notify(SessionEventNotice(session_id=source.id, event=event.to_wire()))

        ctx.on("session/event", emit)

    def _woken(self, served: _Served, agent: object) -> None:
        """A waking message reached the inbox of a served agent (P13-05).

        This is the ring of the daemon, for a transport that drives the agent
        itself. Between prompts the agent is idle and nothing else would run it.
        Before this listener, the result of a readmitted child stayed in the
        inbox until the next prompt of the peer ran both as one turn."""
        if agent is served.agent and self.tasks is not None:
            self.tasks.start_soon(self._run_woken, served)

    async def _run_woken(self, served: _Served) -> None:
        """Run `_drive` on a task of the server itself, where a raise would end the
        server. A failure is logged, and the next wake or prompt drives again."""
        try:
            await self._drive(served)
        except Exception:
            log.exception("ph_app.modes.rpc: a woken turn failed")

    async def _drive(self, served: _Served) -> None:
        """Run the served agent until its inbox is empty, and tell the peer.

        A prompt and a wake both do this, one at a time (`_Served.driving`). When
        nothing is pending, nothing runs. That is the case for a wake whose
        message a turn in flight already took, or for a prompt that a woken turn
        already answered."""
        async with served.driving:
            agent = served.agent
            if agent is None or agent.session is None or not agent.inbox.has_pending:
                return
            session = agent.session
            self._notify(SessionStatusNotice(session_id=session.id, status="running"))
            try:
                await agent.run()
                await served.ctx.require(SESSIONS).flush(session)
            finally:
                self._notify(SessionStatusNotice(session_id=session.id, status="idle"))

    async def _prompt(self, params: _PromptParams) -> dict[str, Any]:
        served = await self._mount(params.session_id)
        ctx = served.ctx
        session = ctx.require(SESSIONS).get(params.session_id) if params.session_id else None
        if session is None:
            session = await self._open(served)
        agent = served.agent
        if agent is None:
            # A prompt that names a route asks for it in the same way as the
            # flags. A prompt that names neither takes the route of the server.
            # Both go through the one rule.
            named = bool(params.provider or params.model)
            asked = ModelChoice.from_flags(params.provider, params.model) if named else self.choice
            # This is an override of the model of the session where it differs
            # (S4). The route of the prompt itself comes from a verb, and the
            # flags of the server are a start option.
            entry = await start_on(
                ctx,
                session,
                asked,
                source="verb" if named else "cli",
                command=f"session/prompt {asked.spelled}" if named else asked.flags,
            )
            agents = ctx.require(AGENTS)
            agent = agents.create(session, entry.options())
            # The listener is attached before the sweep, because the sweep may
            # itself deliver the notice of a readmitted child. The listener reads
            # `served.agent`, so a ring before that is set runs nothing, and the
            # notice waits in the inbox for the drive of this prompt.
            unheard = ctx.on("agent/woken", partial(self._woken, served))
            try:
                await resume_children(ctx, agent)
            except BaseException:
                # The agent is served only after it is swept (D9, Holdings rule
                # 2). When it was recorded first, a sweep that raised was never
                # run again, because every later prompt found an agent and
                # skipped both steps. The agent is ended here instead, so the
                # next prompt makes it again and sweeps.
                unheard()
                with releasing():
                    await agents.dispose(agent.id)
                raise
            served.agent = agent
        # The prompt is delivered, and then driven through the one path that a
        # wake also takes. A woken turn in flight therefore takes this prompt
        # too, and does not race it.
        agent.followup(user_text(params.prompt))
        await self._drive(served)
        return {"sessionId": session.id, "events": len(session.events)}


async def run_rpc(
    profile: Profile,
    *,
    choice: ModelChoice = ModelChoice(),
    stdin: TextIO | None = None,
    out: TextIO | None = None,
) -> Mapping[str, str]:
    """Serve JSON-RPC until stdin closes, then return `RpcServer.unwritten`."""
    source = stdin if stdin is not None else sys.stdin
    sink = out if out is not None else sys.stdout
    async with AsyncExitStack() as exits, anyio.create_task_group() as tasks:
        server = RpcServer(profile=profile, exits=exits, out=sink, choice=choice, tasks=tasks)
        try:
            await _serve_lines(server, source)
        finally:
            # A turn that the notice of a child woke (P13-05) ends with the
            # peer. It is canceled before the group waits for it, and the mounts
            # unwind after.
            server.end()
    return server.unwritten()


async def _serve_lines(server: RpcServer, source: TextIO) -> None:
    """Answer each line of `source` until it closes or says `shutdown`."""
    while True:
        # The read is abandoned on cancel. A signal then stops a server that
        # waits for the next line of its peer, and does not also wait for that
        # line. The thread stays blocked in `readline`, and it ends when the
        # host exits through the signal (`ph.resources.run_until_signaled`).
        line = await anyio.to_thread.run_sync(source.readline, abandon_on_cancel=True)
        if not line:
            return
        text = line.strip()
        if not text:
            continue
        try:
            request = json.loads(text)
        except json.JSONDecodeError:
            continue
        await server.handle(request)
        if request.get("method") == "shutdown":
            return
