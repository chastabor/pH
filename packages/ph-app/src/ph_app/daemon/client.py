"""A client for the supervisor's socket (P5-01).

Small on purpose: it is what the tests drive and what `ph agents` (P5-10) will
drive, and every method here is one frame. Anything richer — reconnection
policy, cursors, command journaling — belongs to P5-02, which is where the
protocol grows.

@module ph_app.daemon.client
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream

from ph.wire import WireModel

from ..params import InitializeParams, MutationParams, PromptParams
from ..protocol import notification
from .duplex import Handler, Notification, Peer

__all__ = ["DaemonClient", "Exchange", "connected"]

type Exchange[T] = Callable[[DaemonClient], Awaitable[T]]
"""One caller's business with the daemon, given a connected client."""


@dataclass(slots=True)
class DaemonClient:
    """One connection to the daemon, and what this end can answer back.

    The framing, the pending table and the write ordering all live in `Peer`,
    which the daemon's own connection object is built from too. What is left here
    is pH's client vocabulary: the identity a command is idempotent under, and
    the two verbs a caller actually says.
    """

    stream: ByteStream
    on_notify: InitVar[Notification | None] = None
    """What to call for a notification the daemon sends. **Init-only**: the
    `Peer` is the one place it lives, so `client.peer.on_notify = ...` is how a
    caller changes it later and there is no second copy to go stale."""
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
    peer: Peer = field(init=False)
    """This connection's duplex end. Built eagerly — `connect` is the only way
    to make one, and it is already inside a running loop."""

    def __post_init__(self, on_notify: Notification | None) -> None:
        self.peer = Peer(
            stream=self.stream, dispatch=self._answer, on_notify=on_notify, id_prefix="c"
        )

    @classmethod
    async def connect(cls, path: Path, on_notify: Notification | None = None) -> DaemonClient:
        stream: ByteStream = await anyio.connect_unix(str(path))
        return cls(stream=stream, on_notify=on_notify)

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
        return await self.call("initialize", InitializeParams(capabilities=list(capabilities)))

    async def mutate(self, method: str, params: MutationParams) -> dict[str, Any]:
        """One mutating call, stamped with this client's idempotence key.

        Every method in the daemon's `MUTATIONS` table needs a
        `clientId`/`commandId` pair, and a caller that forgot one silently lost
        the write-ahead guard: a reconnecting client re-sends what it cannot know
        landed, and an unkeyed retry runs the effect twice. So the stamp is
        applied here, once, rather than at each verb — `prompt` was the only verb
        that had it, and `session/command`, `session/shell`, `session/stage`,
        `session/preset` and `credentials/store` are all in the same table.

        The counter is this client's own, which is what makes a retry after a
        reconnect safe by default rather than by discipline.

        Takes the method's own params model (P8-09), typed as `MutationParams`
        so a verb that is *not* in the daemon's table cannot be sent through the
        door that claims a key. The stamp goes on by `model_copy` rather than by
        two more keyword arguments every caller had to remember.
        """
        self._commands += 1
        keyed = params.model_copy(update={"client_id": self.id, "command_id": str(self._commands)})
        return await self.call(method, keyed)

    async def prompt(self, session_id: str, text: str) -> dict[str, Any]:
        """Queue a turn. Keyed by `mutate`, which says why."""
        return await self.mutate("session/prompt", PromptParams(session_id=session_id, prompt=text))

    async def call(
        self, method: str, params: WireModel | None = None, /, **fields: Any
    ) -> dict[str, Any]:
        """One request, awaited to its reply. Raises what the server refused.

        Two doors, and the model is the one to use. `client.call("session/status",
        SessionParams(session_id=s))` is checked against what the daemon accepts
        at *this* end, so a misspelled field is a type error here rather than the
        `invalid_params` refusal P8-07 taught the daemon to send. The `**fields`
        form stays for callers that have no model to build — a test sending a
        deliberately malformed frame to exercise a refusal, which is most of what
        `test_daemon_methods` does, and which a typed-only signature would make
        unwritable.

        Positional-only, so a model can never be confused with a field named
        `params` on some future method.
        """
        return await self.peer.ask(method, params.to_wire() if params is not None else fields)

    async def notify(self, method: str, params: WireModel | None = None, /, **fields: Any) -> None:
        """Send a request that expects no reply.

        `shutdown` is the one that matters: a request-with-reply would have the
        caller waiting on a frame the daemon is in the middle of tearing down the
        ability to send. "Stop" is not a question, so it does not get an id.

        Waits for room rather than refusing, unlike the daemon's `tell`: a client
        that cannot write has nobody to drop but itself.
        """
        await self.peer.send(
            notification(method, params.to_wire() if params is not None else fields)
        )

    async def aclose(self) -> None:
        await self.stream.aclose()


async def connected[T](path: Path, work: Exchange[T]) -> T:
    """Connect, run one exchange, and close — with the pump alongside it.

    The pump has to be a task rather than something the caller drives, because
    replies and notifications arrive on the same stream: a caller that read its
    own reply directly would consume a `session.event` it had no way to hand
    back. Closing the stream is what ends the pump, so there is no cancel here —
    a teardown that cancelled would race the last frame it asked for.

    Here rather than in `ph agents`, because a one-shot exchange is not a CLI
    shape: `ph_app.web.serve` stages a browser's upload this way too, and the
    copy it started with re-derived every subtlety below.

    **What leaves this function is the exception, not the wrapper.** anyio wraps
    whatever comes out of a task group, so a `DaemonError` the server sent would
    reach a caller as a group of one and match no `except` written for it — a
    hazard every caller was solving for itself, one with `_alone` and one with
    `except*`. The group is *this* function's, so unwrapping it is too, and a
    caller writes the plain `except DaemonError` it meant. A group holding more
    than one is re-raised whole: two simultaneous failures want neither caller's
    sentence.
    """
    client = await DaemonClient.connect(path)
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(client.pump)
            try:
                outcome = await work(client)
            finally:
                await client.aclose()
    except BaseExceptionGroup as group:
        raise _alone(group) from None
    # Assigned inside the group and returned outside it: a task group's
    # `__aexit__` is typed as one that may suppress, so a `return` in the block
    # leaves the function with a path that falls off the end.
    return outcome


def _alone(raised: BaseException) -> BaseException:
    """The single exception inside a task group's wrapper, if that is all it holds.

    Nested groups are unwrapped too — a task group inside a task group is two
    layers of one exception, and a caller cares about neither.
    """
    while isinstance(raised, BaseExceptionGroup) and len(raised.exceptions) == 1:
        raised = raised.exceptions[0]
    return raised
