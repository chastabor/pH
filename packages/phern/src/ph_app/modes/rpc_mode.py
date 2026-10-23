"""`--mode rpc` — JSON-RPC over stdio, in the dsh SDK's shape (D13, I-7).

Deliberately dsh's method names and payload shapes (`initialize`,
`session/prompt`, `session.event`, `session.status`) rather than a pH-specific
protocol: dsh already ships a Python client for this, and Phase 5's daemon
extends the same surface rather than inventing a second one.

Notifications are the log's own envelopes, camelCase, so an RPC client and a log
reader parse one format.

@module ph_app.modes.rpc_mode
"""

from __future__ import annotations

import json
import logging
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, TextIO

import anyio
from anyio.abc import TaskGroup

from ph.agent.types import AgentDriver
from ph.cordis import DEPLOYMENT, Context, Profile
from ph.json import dumps
from ph.keys import AGENTS, SESSIONS, TOOLS
from ph.llm.types import user_text
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


# The stdio transport's own params (P8-07) — the two whose contract differs from
# the daemon's: here `session/prompt` may omit the id and names the route, because
# there is no supervisor holding one. A shared model would have to make that
# optional for the daemon too, which is the daemon's `invalid_params` refusal
# quietly given away. `session/new` may omit the id on both since protocol 8 — the
# door names a new session, on either transport — but the daemon's also carries the
# client's directory and trust, which stdio has no use for.
#
# The shapes both transports need are the protocol's, not a copy here:
# `NoParams` and `SessionParams` come from `..protocol` beside `Cursor`, for
# `Cursor`'s stated reason. "Each server owns its method table" is about the
# table, not about re-declaring an empty model per server.


class _NewParams(WireModel):
    session_id: SessionId | None = None


class _PromptParams(WireModel):
    session_id: SessionId | None = None
    prompt: str = ""
    provider: str | None = None
    model: str | None = None


@dataclass(slots=True)
class _Served:
    """One session this server serves: its own mount, and its agent once prompted."""

    ctx: Context
    start: RecordedStart
    """What it is opened with (`open_root`): its start as read before the mount, and
    once it is open, its own id and family."""
    agent: AgentDriver | None = None
    driving: anyio.Lock = field(default_factory=anyio.Lock)
    """Held while this server drives the agent (`RpcServer._drive`): a prompt and a
    wake take turns, so one waits out the other's `run()` rather than meeting
    "already has active work"."""


@dataclass(slots=True)
class RpcServer:
    """One stdio JSON-RPC endpoint, each session it serves on a mount of its own (S5).

    One mount per session, as the daemon has one per root: a session's overrides are
    realized on its mount, so a shared one made one session's `/model` or allowance
    the next session's too, with nothing in the second's log to say so. Each is
    mounted from its own log's environment (`mount_session`), `profile` being what a
    new one starts on; all of them unwind with `exits`.
    """

    profile: Profile
    exits: AsyncExitStack
    out: TextIO
    choice: ModelChoice = field(default_factory=ModelChoice)
    """What `phern --mode rpc --provider/--model` asked for; empty is the profile default."""
    tasks: TaskGroup | None = None
    """Where a turn nobody prompted runs — a parent woken by a child's ending notice
    between prompts (P13-05). `None`, as a test builds one, runs none."""
    _served: dict[str, _Served] = field(default_factory=dict)
    _deployment: Context | None = None

    async def _mount(self, session_id: str | None) -> _Served:
        """The session's own mount, mounted the first time it is asked for. `None` is a
        session about to be made, which has no log to read an environment from and is
        filed under its id once it opens (`_open`)."""
        served = self._served.get(session_id) if session_id else None
        if served is None:
            start = await read_start(session_id)
            # An adopted version the loader refuses is taken back there, as the
            # daemon's roots and `phern -p` do; its log says why.
            ctx, _ = await mount_session(self.exits, start, self.profile)
            served = _Served(ctx, start)
            if session_id:
                self._served[session_id] = served
        return served

    async def _open(self, served: _Served) -> Session:
        """Open a session on its mount, and file the mount under its id. A session about
        to be made gets its id from `open_session`, which does not search the store for
        an id it made, as `phern -p`'s does."""
        session = await open_root(served.ctx, served.start)
        served.start = replace(served.start, session_id=session.id, family=session.header.family)
        self._served[session.id] = served
        self._attach(served.ctx, session)
        return session

    async def _listing(self) -> Context:
        """The profile itself, mounted once, for what the deployment offers before any
        session exists — `tools/list`."""
        if self._deployment is None:
            self._deployment = await self.exits.enter_async_context(mounted(self.profile))
        return self._deployment

    def _write(self, payload: Frame) -> None:
        self.out.write(f"{dumps(payload)}\n")
        self.out.flush()

    def _notify(self, notice: SessionNotice) -> None:
        """One notification, under the name its payload declares.

        The same two shapes the daemon sends — a client reading this transport
        and one reading the socket parse the frame the same way, which is what
        "they are the same protocol" has to mean at the payload as well as at
        the envelope.
        """
        self._write(notification(notice.METHOD, notice.to_wire()))

    async def handle(self, request: dict[str, Any]) -> None:
        reply = await respond(request, self._dispatch)
        if reply is not None:
            self._write(reply)

    async def _dispatch(self, method: str, params: dict[str, Any]) -> MethodResult:
        if method in ("initialize", "daemon/hello"):
            # The same block the daemon answers with, minus what stdio cannot
            # do: one process, one peer, no supervision. The params are parsed
            # and discarded: a client's capability block means nothing to a
            # transport that will never ask it anything, but a stray field is
            # still a stray field.
            #
            # **Through the verb, because `NoParams` contradicted that comment.**
            # `WireModel` is `extra="forbid"`, so parsing against `NoParams`
            # refused the `capabilities` list every client sends — including
            # `DaemonClient.initialize`'s own frame — which made `initialize`
            # two contracts under one name while `protocol.py` says "they are
            # the same protocol". Verified: `parse_params("initialize",
            # NoParams, {"capabilities": []})` came back `invalid_params:
            # capabilities: Extra inputs are not permitted`.
            verbs.INITIALIZE.parse(params)
            return capabilities("tools")
        if method == "session/new":
            # Open, not create: a peer naming a stored id resumes it, and one
            # another process holds is refused by name (P5-03).
            opened = parse_params(method, _NewParams, params)
            session = await self._open(await self._mount(opened.session_id))
            return {"sessionId": session.id}
        if method == "session/prompt":
            return await self._prompt(parse_params(method, _PromptParams, params))
        if method == "session/events":
            asked = parse_params(method, SessionParams, params)
            # A session is only ever opened on its own mount, so one without one
            # here is one this server never opened.
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
            # Nothing, like the daemon's. `shutdown` is a `Notify`, so there is
            # no reply model to build and `respond` sends no frame — which was
            # the whole argument for the `{"ok": True}` literal this replaced
            # being pointless rather than merely duplicated.
            return None
        raise UnknownMethod(f'unknown method "{method}"')

    def _attach(self, ctx: Context, session: Session) -> None:
        def emit(source: Session, event: SessionEvent) -> None:
            if source.id != session.id:
                return
            self._notify(SessionEventNotice(session_id=source.id, event=event.to_wire()))

        ctx.on("session/event", emit)

    def _woken(self, served: _Served, agent: object) -> None:
        """A waking message reached a served agent's inbox (P13-05): the daemon's
        ring, for a transport that drives the agent itself. Between prompts the
        agent is idle and nothing else would run it — a readmitted child's result
        sat there until the peer's next prompt ran both as one turn."""
        if agent is served.agent and self.tasks is not None:
            self.tasks.start_soon(self._run_woken, served)

    async def _run_woken(self, served: _Served) -> None:
        """`_drive` on the server's own task, where a raise would end the server: a
        failure is logged, and the next wake or prompt drives again."""
        try:
            await self._drive(served)
        except Exception:
            log.exception("ph_app.modes.rpc: a woken turn failed")

    async def _drive(self, served: _Served) -> None:
        """Run the served agent until its inbox is empty, telling the peer — what a
        prompt and a wake both do, one at a time (`_Served.driving`). Nothing pending
        runs nothing: a wake whose message a turn in flight already took, or a prompt
        a woken turn already answered."""
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
            # A prompt that names a route asks for it the way the flags do, and
            # one that names neither takes the server's — through the one rule.
            named = bool(params.provider or params.model)
            asked = ModelChoice.from_flags(params.provider, params.model) if named else self.choice
            # An override of the session's model where it differs (S4): the prompt's
            # own route is a verb's, the server's flags a start option.
            entry = await start_on(
                ctx,
                session,
                asked,
                source="verb" if named else "cli",
                command=f"session/prompt {asked.spelled}" if named else asked.flags,
            )
            agent = served.agent = ctx.require(AGENTS).create(session, entry.options())
            ctx.on("agent/woken", partial(self._woken, served))
            await resume_children(ctx, agent)
        # Delivered, then driven through the one path a wake also takes, so a woken
        # turn in flight takes this prompt too rather than racing it.
        agent.followup(user_text(params.prompt))
        await self._drive(served)
        return {"sessionId": session.id, "events": len(session.events)}


async def run_rpc(
    profile: Profile,
    *,
    choice: ModelChoice = ModelChoice(),
    stdin: TextIO | None = None,
    out: TextIO | None = None,
) -> None:
    """Serve JSON-RPC until stdin closes."""
    source = stdin if stdin is not None else sys.stdin
    sink = out if out is not None else sys.stdout
    async with AsyncExitStack() as exits, anyio.create_task_group() as tasks:
        server = RpcServer(profile=profile, exits=exits, out=sink, choice=choice, tasks=tasks)
        try:
            await _serve_lines(server, source)
        finally:
            # A turn a child's notice woke (P13-05) ends with the peer: cancelled
            # before the group waits for it, and the mounts unwind after.
            tasks.cancel_scope.cancel()


async def _serve_lines(server: RpcServer, source: TextIO) -> None:
    """Answer each line of `source` until it closes or says `shutdown`."""
    while True:
        # Abandoned on cancel, so a signal stops a server waiting for its peer's
        # next line rather than waiting for it as well. The thread stays blocked
        # in `readline`, and the host leaving through the signal is what ends it
        # (`ph.resources.run_until_signaled`).
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
