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
from ph.cordis import DEPLOYMENT, ProfileLayer
from ph.session import Session, SessionEvent, dumps

from ..protocol import capabilities, notification, respond
from ..runtime import mounted

__all__ = ["RpcServer", "run_rpc"]


@dataclass(slots=True)
class RpcServer:
    """One stdio JSON-RPC endpoint over a mounted pH."""

    ctx: Any
    out: TextIO
    provider: str = "fake"
    model: str = "fake-1"
    _agents: dict[str, Any] = field(default_factory=dict)

    def _write(self, payload: dict[str, Any]) -> None:
        self.out.write(f"{dumps(payload)}\n")
        self.out.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write(notification(method, params))

    async def handle(self, request: dict[str, Any]) -> None:
        reply = await respond(request, self._dispatch)
        if reply is not None:
            self._write(reply)

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        if method in ("initialize", "daemon/hello"):
            # The same block the daemon answers with, minus what stdio cannot
            # do: one process, one peer, no supervision.
            return capabilities("tools")
        if method == "session/new":
            session = self.ctx.sessions.create(params.get("sessionId"))
            self._attach(session)
            return {"sessionId": session.id}
        if method == "session/prompt":
            return await self._prompt(params)
        if method == "session/events":
            session = self.ctx.sessions.require(params["sessionId"])
            return {"events": [event.to_wire() for event in session.events]}
        if method == "tools/list":
            # `DEPLOYMENT` (P6-32): RPC mode advertises what the deployment
            # offers, before any agent exists to narrow it.
            schemas = self.ctx.tools.schemas(scope=DEPLOYMENT)
            return {"tools": [schema.to_wire() for schema in schemas]}
        if method == "shutdown":
            return {"ok": True}
        raise ValueError(f'unknown method "{method}"')

    def _attach(self, session: Session) -> None:
        def emit(source: Session, event: SessionEvent) -> None:
            if source.id != session.id:
                return
            self._notify("session.event", {"sessionId": source.id, "event": event.to_wire()})

        self.ctx.on("session/event", emit)

    async def _prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = params.get("sessionId")
        session = self.ctx.sessions.get(session_id) if session_id else None
        if session is None:
            session = self.ctx.sessions.create(session_id)
            self._attach(session)
        agent = self._agents.get(session.id)
        if agent is None:
            agent = self.ctx.agents.create(
                session,
                AgentOptions(
                    provider=params.get("provider") or self.provider,
                    model=params.get("model") or self.model,
                ),
            )
            self._agents[session.id] = agent
        self._notify("session.status", {"sessionId": session.id, "status": "running"})
        await agent.prompt(str(params.get("prompt", "")))
        await self.ctx.sessions.flush(session)
        self._notify("session.status", {"sessionId": session.id, "status": "idle"})
        return {"sessionId": session.id, "events": len(session.events)}


async def run_rpc(
    documents: list[ProfileLayer],
    *,
    provider: str,
    model: str,
    stdin: TextIO | None = None,
    out: TextIO | None = None,
) -> None:
    """Serve JSON-RPC until stdin closes."""
    source = stdin if stdin is not None else sys.stdin
    sink = out if out is not None else sys.stdout
    async with mounted(documents) as run:
        server = RpcServer(ctx=run.ctx, out=sink, provider=provider, model=model)
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
