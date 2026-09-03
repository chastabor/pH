"""A client for the supervisor's socket (P5-01).

Small on purpose: it is what the tests drive and what `ph agents` (P5-10) will
drive, and every method here is one frame. Anything richer — reconnection
policy, cursors, command journaling — belongs to P5-02, which is where the
protocol grows.

@module ph_app.daemon.client
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream

from ..protocol import notification
from .duplex import Handler, Notification, Peer

__all__ = ["DaemonClient"]


@dataclass(slots=True)
class DaemonClient:
    """One connection to the daemon, and what this end can answer back.

    The framing, the pending table and the write ordering all live in `Peer`,
    which the daemon's own connection object is built from too. What is left here
    is pH's client vocabulary: the identity a command is idempotent under, and
    the two verbs a caller actually says.
    """

    stream: ByteStream
    on_notify: Notification | None = None
    handlers: dict[str, Handler] = field(default_factory=dict)
    """What this client will answer when the *daemon* asks it something (P5-13).

    The protocol was one-directional for anything expecting a reply, but a front
    end's whole contract is two calls that wait for a person — an approval and a
    question — so the daemon has to be able to ask. A method here is how this
    client says it can answer one."""
    id: str = field(default_factory=lambda: f"client-{secrets.token_hex(6)}")
    """This connection's identity, for idempotence. Minted rather than asked for:
    a caller that had to supply one would supply the same one twice."""
    _commands: int = 0
    _peer: Peer | None = None

    @classmethod
    async def connect(cls, path: Path, on_notify: Notification | None = None) -> DaemonClient:
        stream: ByteStream = await anyio.connect_unix(str(path))
        return cls(stream=stream, on_notify=on_notify)

    @property
    def peer(self) -> Peer:
        """This connection's duplex end, built on first use.

        Lazily, because a `DaemonClient` is constructed in places that are not yet
        inside a running loop, and `Peer` holds an `anyio.Event`.
        """
        if self._peer is None:
            self._peer = Peer(
                stream=self.stream,
                dispatch=self._answer,
                on_notify=self.on_notify,
                id_prefix="c",
            )
        return self._peer

    @property
    def closed(self) -> anyio.Event:
        """Set when the pump stops, whichever end ended it."""
        return self.peer.closed

    async def _answer(self, method: str, params: dict[str, Any]) -> Any:
        handler = self.handlers.get(method)
        if handler is None:
            raise LookupError(f'this client cannot answer "{method}"')
        return await handler(params)

    async def pump(self) -> None:
        """Read frames until the socket closes. Run this in a task group."""
        await self.peer.serve()

    async def initialize(self, *capabilities: str) -> dict[str, Any]:
        """Trade capability blocks: what the daemon serves, what this end answers.

        Both directions in one call, because they are one negotiation. A client
        that can put a question in front of a person says `asks` here — once, for
        the connection — and every root it attaches to afterwards may ask it. The
        alternative, a flag on each `session/attach`, let one client answer for
        one session and not another, which is not a thing a UI can be.
        """
        return await self.call("initialize", capabilities=list(capabilities))

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
        return await self.peer.ask(method, params)

    async def notify(self, method: str, **params: Any) -> None:
        """Send a request that expects no reply.

        `shutdown` is the one that matters: a request-with-reply would have the
        caller waiting on a frame the daemon is in the middle of tearing down the
        ability to send. "Stop" is not a question, so it does not get an id.

        Waits for room rather than refusing, unlike the daemon's `tell`: a client
        that cannot write has nobody to drop but itself.
        """
        await self.peer.send(notification(method, params))

    async def aclose(self) -> None:
        await self.stream.aclose()
