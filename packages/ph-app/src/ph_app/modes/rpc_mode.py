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
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

import anyio

from ph.agent.types import AgentOptions
from ph.cordis import DEPLOYMENT, Profile
from ph.keys import AGENTS, SESSIONS, TOOLS
from ph.session import Session, SessionEvent, dumps
from ph.wire import WireModel

from ..payloads import SessionEventNotice, SessionNotice, SessionStatusNotice
from ..protocol import (
    Frame,
    NoParams,
    SessionParams,
    UnknownMethod,
    capabilities,
    notification,
    parse_params,
    respond,
)
from ..runtime import mounted, open_session

__all__ = ["RpcServer", "run_rpc"]


# The stdio transport's own params (P8-07) — the two, and only the two, whose
# contract genuinely differs from the daemon's: here `session/new` may omit the
# id (one process, one peer, so "a fresh session" needs no name) and
# `session/prompt` names the route, because there is no supervisor holding one.
# A shared model would have to make both optional for the daemon too, which is
# the daemon's `invalid_params` refusal quietly given away.
#
# The shapes both transports need are the protocol's, not a copy here:
# `NoParams` and `SessionParams` come from `..protocol` beside `Cursor`, for
# `Cursor`'s stated reason. "Each server owns its method table" is about the
# table, not about re-declaring an empty model per server.


class _NewParams(WireModel):
    session_id: str | None = None


class _PromptParams(WireModel):
    session_id: str | None = None
    prompt: str = ""
    provider: str | None = None
    model: str | None = None


@dataclass(slots=True)
class RpcServer:
    """One stdio JSON-RPC endpoint over a mounted pH."""

    ctx: Any
    out: TextIO
    provider: str = "fake"
    model: str = "fake-1"
    _agents: dict[str, Any] = field(default_factory=dict)

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

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method in ("initialize", "daemon/hello"):
            # The same block the daemon answers with, minus what stdio cannot
            # do: one process, one peer, no supervision. The params are parsed
            # and discarded: a client's capability block means nothing to a
            # transport that will never ask it anything, but a stray field is
            # still a stray field.
            parse_params(method, NoParams, params)
            return capabilities("tools")
        if method == "session/new":
            # Open, not create: a peer naming a stored id resumes it, and one
            # another process holds is refused by name (P5-03).
            opened = parse_params(method, _NewParams, params)
            session = await open_session(self.ctx, opened.session_id)
            self._attach(session)
            return {"sessionId": session.id}
        if method == "session/prompt":
            return await self._prompt(parse_params(method, _PromptParams, params))
        if method == "session/events":
            asked = parse_params(method, SessionParams, params)
            session = self.ctx.require(SESSIONS).require(asked.session_id)
            return {"events": [event.to_wire() for event in session.events]}
        if method == "tools/list":
            # `DEPLOYMENT` (P6-32): RPC mode advertises what the deployment
            # offers, before any agent exists to narrow it.
            parse_params(method, NoParams, params)
            schemas = self.ctx.require(TOOLS).schemas(scope=DEPLOYMENT)
            return {"tools": [schema.to_wire() for schema in schemas]}
        if method == "shutdown":
            parse_params(method, NoParams, params)
            return {"ok": True}
        raise UnknownMethod(f'unknown method "{method}"')

    def _attach(self, session: Session) -> None:
        def emit(source: Session, event: SessionEvent) -> None:
            if source.id != session.id:
                return
            self._notify(SessionEventNotice(session_id=source.id, event=event.to_wire()))

        self.ctx.on("session/event", emit)

    async def _prompt(self, params: _PromptParams) -> dict[str, Any]:
        session_id = params.session_id
        session = self.ctx.require(SESSIONS).get(session_id) if session_id else None
        if session is None:
            session = await open_session(self.ctx, session_id)
            self._attach(session)
        agent = self._agents.get(session.id)
        if agent is None:
            agent = self.ctx.require(AGENTS).create(
                session,
                AgentOptions(
                    provider=params.provider or self.provider,
                    model=params.model or self.model,
                ),
            )
            self._agents[session.id] = agent
        self._notify(SessionStatusNotice(session_id=session.id, status="running"))
        await agent.prompt(params.prompt)
        await self.ctx.require(SESSIONS).flush(session)
        self._notify(SessionStatusNotice(session_id=session.id, status="idle"))
        return {"sessionId": session.id, "events": len(session.events)}


async def run_rpc(
    profile: Profile,
    *,
    provider: str,
    model: str,
    stdin: TextIO | None = None,
    out: TextIO | None = None,
) -> None:
    """Serve JSON-RPC until stdin closes."""
    source = stdin if stdin is not None else sys.stdin
    sink = out if out is not None else sys.stdout
    async with mounted(profile) as ctx:
        server = RpcServer(ctx=ctx, out=sink, provider=provider, model=model)
        while True:
            line = await anyio.to_thread.run_sync(source.readline)
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
