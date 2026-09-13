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
from typing import Any, overload

import anyio
from anyio.abc import ByteStream

from ph.wire import WireModel

from .. import verbs
from ..params import InitializeParams, MutationParams, PromptParams
from ..payloads import MutationRepeated, RootDescription
from ..protocol import CapabilityBlock, Notify, Verb, notification
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

    async def _answer(self, method: str, params: dict[str, Any]) -> Any:  # noqa: ANN401
        handler = self.handlers.get(method)
        if handler is None:
            raise LookupError(f'this client cannot answer "{method}"')
        return await handler(params)

    async def pump(self) -> None:
        """Read frames until the socket closes. Run this in a task group."""
        await self.peer.serve()

    async def initialize(self, *capabilities: str) -> CapabilityBlock:
        """Trade capability blocks: what the daemon serves, what this end answers.

        Both directions in one call, because they are one negotiation. A client
        that can put a question in front of a person says `asks` here — once, for
        the connection — and every root it attaches to afterwards may ask it. The
        alternative, a flag on each `session/attach`, let one client answer for
        one session and not another, which is not a thing a UI can be.
        """
        return await self.call(verbs.INITIALIZE, InitializeParams(capabilities=list(capabilities)))

    async def mutate[P: MutationParams, R: WireModel](
        self, verb: Verb[P, R], params: P
    ) -> R | MutationRepeated:
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

        **Takes the `Verb`, so the params cannot belong to another method** and
        the reply arrives typed (issue 74). `P` is bound to `MutationParams`
        rather than `WireModel`, which is what keeps a verb that is *not* in the
        daemon's table out of the door that claims a key — the bound is the
        check, where before it was the declared parameter type and a caller
        passing the name of a non-mutation got no complaint. The stamp goes on
        by `model_copy` rather than by two more keyword arguments every caller
        had to remember.

        **The reply is `R | MutationRepeated`, and that union is the honest
        one.** A repeat answers with one shape for every verb (`MUTATIONS` says
        why), and for the verbs whose own reply is a `RootDescription` the
        repeat is a subtype — but `session/shell` answers with an exit code and
        `credentials/store` with a name, so a caller that ignored the union
        would be reading fields off a description that is not there. Narrow on
        `isinstance(reply, MutationRepeated)`, which is the branch
        `MutationRepeated.repeated` was always for.
        """
        self._commands += 1
        keyed = params.model_copy(update={"client_id": self.id, "command_id": str(self._commands)})
        wire = await self.peer.ask(verb.name, keyed.to_wire())
        # The repeat is decided by the frame, not by the verb: the same verb
        # answers either way, and `repeated` is the field that says which.
        if wire.get("repeated"):
            return MutationRepeated.model_validate(wire)
        return verb.read(wire)

    async def prompt(self, session_id: str, text: str) -> RootDescription:
        """Queue a turn. Keyed by `mutate`, which says why.

        `RootDescription` and not `RootDescription | MutationRepeated`, which is
        what this said: `MutationRepeated` subclasses `RootDescription`, so that
        union normalises to the left side and asked every caller to narrow
        something the checker had already flattened. The repeat still arrives
        here — it is simply already the type it claims to be, which is the
        property `MutationRepeated`'s own docstring is about.
        """
        return await self.mutate(
            verbs.SESSION_PROMPT, PromptParams(session_id=session_id, prompt=text)
        )

    @overload
    async def call[P: WireModel, R: WireModel](self, verb: Verb[P, R], params: P, /) -> R: ...

    @overload
    async def call(self, method: str, /, **fields: Any) -> dict[str, Any]: ...  # noqa: ANN401

    async def call(
        self, verb: str | Verb[Any, Any], params: WireModel | None = None, /, **fields: Any
    ) -> Any:
        """One request, awaited to its reply. Raises what the server refused.

        **Two doors, and the `Verb` is the one to use.** `client.call(
        SESSION_STATUS, SessionParams(session_id=s))` checks three things at
        *this* end that were unchecked before: the field spellings (P8-09 bought
        that), that the params belong to this method, and what comes back. The
        old form `client.call("session/status", SessionParams(...))` type-checked
        and so did `client.call("sessions/list", SessionParams(...))` — a name
        and a model that had to agree with nothing to make them.

        The reply is validated here, which is what retires
        `AttachReply.model_validate(await client.call(...))`: the model was
        chosen by hand beside the method name, so it could be the wrong one and
        nothing would say so.

        The `**fields` form stays, and is now the only door taking a bare name:
        a caller with no model to build — a test sending a deliberately
        malformed frame to exercise a refusal, which is most of what
        `test_daemon_methods` does — would find a typed-only signature
        unwritable. Because it takes no positional model, the mismatched pair
        the first paragraph describes is not merely discouraged but
        unexpressible.

        Positional-only, so a model can never be confused with a field named
        `params` on some future method.
        """
        if isinstance(verb, Verb):
            return await self._exchange(verb, params)
        return await self.peer.ask(verb, fields)

    async def _exchange[P: WireModel, R: WireModel](self, verb: Verb[P, R], params: P | None) -> R:
        """The verb door's body, where it can be checked.

        An `@overload`ed function's implementation is typed against the union of
        its signatures and returns `Any`, so nothing in `call` above checks that
        the verb branch validates at all: sabotaged to `return wire`, dropping
        the `verb.read` that is the whole point of the typed door, `mypy
        --strict` passed on all 266 files. Here `R` is bound, so the body is
        checked and that sabotage is an error.

        `params` is `P | None` rather than `P` only because the implementation
        signature above cannot express "required on this branch"; both overloads
        declare it required, so the `{}` is unreachable through the public door.
        """
        return verb.read(await self.peer.ask(verb.name, params.to_wire() if params else {}))

    @overload
    async def notify[P: WireModel](self, verb: Notify[P], params: P, /) -> None: ...

    @overload
    async def notify(self, method: str, /, **fields: Any) -> None: ...  # noqa: ANN401

    async def notify(
        self, verb: str | Notify[Any], params: WireModel | None = None, /, **fields: Any
    ) -> None:
        """Send a request that expects no reply.

        `shutdown` is the one that matters: a request-with-reply would have the
        caller waiting on a frame the daemon is in the middle of tearing down the
        ability to send. "Stop" is not a question, so it does not get an id.

        **A `Notify`, not a `Verb`, and that is the pairing this door adds.**
        While `SHUTDOWN` was a `Verb` with an unread reply model, the rule lived
        in prose: `client.call(SHUTDOWN, NoParams())` type-checked, all 19 sites
        used `notify` by convention, and the hang the paragraph above describes
        was one edit away. The two doors now take different types, so neither
        mistake is expressible — `call` refuses a `Notify` and this refuses a
        `Verb`, which would otherwise run a method on the daemon and silently
        drop its answer.

        Waits for room rather than refusing, unlike the daemon's `tell`: a client
        that cannot write has nobody to drop but itself.
        """
        name = verb.name if isinstance(verb, Notify) else verb
        body = params.to_wire() if params is not None else fields
        await self.peer.send(notification(name, body))

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
