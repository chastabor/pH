"""A client for the supervisor's socket (P5-01).

Small on purpose: it is what the tests drive and what `ph agents` (P5-10) will
drive, and every method here is one frame. Anything richer — reconnection
policy, cursors, command journaling — belongs to P5-02, which is where the
protocol grows.

@module ph_app.daemon.client
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream

from ..protocol import notification, request
from .framing import read_frames, write_frame

__all__ = ["DaemonClient"]

Notification = Callable[[str, dict[str, Any]], None]


@dataclass(slots=True)
class DaemonClient:
    """One connection, with replies matched to requests by id."""

    stream: ByteStream
    on_notify: Notification | None = None
    id: str = field(default_factory=lambda: f"client-{secrets.token_hex(6)}")
    """This connection's identity, for idempotence. Minted rather than asked
    for: a caller that had to supply one would supply the same one twice."""
    _commands: int = 0
    _next_id: int = 0
    _replies: dict[int, dict[str, Any]] = field(default_factory=dict)
    _events: dict[int, anyio.Event] = field(default_factory=dict)

    @classmethod
    async def connect(cls, path: Path, on_notify: Notification | None = None) -> DaemonClient:
        stream: ByteStream = await anyio.connect_unix(str(path))
        return cls(stream=stream, on_notify=on_notify)

    async def pump(self) -> None:
        """Read frames until the socket closes. Run this in a task group.

        A closed stream ends the pump rather than raising: closing the client is
        how a caller says it is done, and a teardown that reports the thing it
        asked for as an error is one every caller has to write a `try` around.
        """
        try:
            await self._pump()
        except (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
            return

    async def _pump(self) -> None:
        async for frame in read_frames(self.stream):
            reply_id = frame.get("id")
            if reply_id is None:
                if self.on_notify is not None:
                    self.on_notify(str(frame.get("method", "")), frame.get("params") or {})
                continue
            self._replies[int(reply_id)] = frame
            waiting = self._events.pop(int(reply_id), None)
            if waiting is not None:
                waiting.set()

    async def prompt(self, session_id: str, text: str) -> dict[str, Any]:
        """Queue a turn, idempotently, without the caller minting ids.

        Idempotence was a pair of parameters a caller had to remember, which
        means it held for the test that hand-rolled them and for nothing else.
        A client knows its own identity and can count its own commands, so it
        does both — and a retry after a reconnect is safe by default rather
        than by discipline.
        """
        self._commands += 1
        return await self.call(
            "session/prompt",
            sessionId=session_id,
            prompt=text,
            clientId=self.id,
            commandId=str(self._commands),
        )

    async def call(self, method: str, **params: Any) -> dict[str, Any]:
        """One request, awaited to its reply. Raises what the server refused."""
        self._next_id += 1
        request_id = self._next_id
        waiting = anyio.Event()
        self._events[request_id] = waiting
        await write_frame(self.stream, request(request_id, method, params))
        await waiting.wait()
        frame = self._replies.pop(request_id, {})
        if "error" in frame:
            raise RuntimeError(str(frame["error"].get("message", "the daemon refused")))
        result: dict[str, Any] = frame.get("result") or {}
        return result

    async def notify(self, method: str, **params: Any) -> None:
        """Send a request that expects no reply.

        `shutdown` is the one that matters: a request-with-reply would have the
        caller waiting on a frame the daemon is in the middle of tearing down
        the ability to send. "Stop" is not a question, so it does not get an id.
        """
        await write_frame(self.stream, notification(method, params))

    async def aclose(self) -> None:
        await self.stream.aclose()
