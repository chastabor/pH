"""A client for the socket of the supervisor (P5-01).

The client is small on purpose. The tests drive it, `phern agents` (P5-10)
drives it, and every method here is one frame. Anything richer, such as a
reconnection policy, cursors or command journaling, belongs to P5-02, which is
where the protocol grows.

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
from ..protocol import CapabilityBlock, MethodResult, Notify, Verb, notification
from .duplex import Handler, Notification, Peer

__all__ = ["DaemonClient", "Exchange", "connected"]

type Exchange[T] = Callable[[DaemonClient], Awaitable[T]]
"""One caller's business with the daemon, given a connected client."""


@dataclass(slots=True)
class DaemonClient:
    """One connection to the daemon, and what this end can answer.

    The framing, the pending table and the write ordering are all in `Peer`. The
    connection object of the daemon is built from `Peer` too. What remains here
    is the client vocabulary of pH: the identity under which a command is
    idempotent, and the two verbs that a caller actually uses.
    """

    stream: ByteStream
    on_notify: InitVar[Notification | None] = None
    """What to call for a notification that the daemon sends. **Init-only**: the
    `Peer` is the one place that holds it. A caller changes it later with
    `client.peer.on_notify = ...`, and there is no second copy that can go
    stale."""
    handlers: dict[str, Handler] = field(default_factory=dict)
    """What this client answers when the *daemon* asks it something (P5-13).

    The protocol went one way for anything that expected a reply. But the whole
    contract of a front end is two calls that wait for a person: an approval and
    a question. So the daemon must be able to ask. A method here is how this
    client states that it can answer one."""
    id: str = field(default_factory=lambda: f"client-{secrets.token_hex(6)}")
    """The identity of this connection, for idempotence. It is minted and not
    requested: a caller that had to supply one would supply the same one twice."""
    _commands: int = 0
    peer: Peer = field(init=False)
    """The duplex end of this connection. It is built eagerly: `connect` is the
    only way to make a client, and `connect` already runs inside a running loop."""

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

    async def _answer(self, method: str, params: dict[str, Any]) -> MethodResult:
        handler = self.handlers.get(method)
        if handler is None:
            raise LookupError(f'this client cannot answer "{method}"')
        return await handler(params)

    async def pump(self) -> None:
        """Read frames until the socket closes. Run this in a task group."""
        await self.peer.serve()

    async def initialize(self, *capabilities: str) -> CapabilityBlock:
        """Exchange capability blocks: what the daemon serves, and what this end
        answers.

        Both directions are in one call, because they are one negotiation. A
        client that can put a question in front of a person says `asks` here,
        one time for the connection. Every root that it attaches to afterwards
        may then ask it. The alternative was a flag on each `session/attach`.
        That let one client answer for one session and not for another, and a UI
        cannot work that way.
        """
        return await self.call(verbs.INITIALIZE, InitializeParams(capabilities=list(capabilities)))

    async def mutate[P: MutationParams, R: WireModel](
        self, verb: Verb[P, R], params: P
    ) -> R | MutationRepeated:
        """One mutating call, stamped with the idempotence key of this client.

        Every method in the `MUTATIONS` table of the daemon needs a
        `clientId`/`commandId` pair. A caller that forgot one silently lost the
        write-ahead guard: a reconnecting client sends again what it cannot know
        arrived, and a retry without a key runs the effect twice. So the stamp
        is applied here, in one place, and not at each verb. `prompt` was the
        only verb that had it, and `session/command`, `session/shell`,
        `session/stage`, `session/preset` and `session/model` are all in the
        same table.

        The counter belongs to this client. A retry after a reconnect is
        therefore safe by default, and not only when a caller is disciplined.

        **This takes the `Verb`, so the params cannot belong to another
        method**, and the reply arrives typed (issue 74). `P` is bound to
        `MutationParams` and not to `WireModel`. The bound keeps a verb that is
        *not* in the table of the daemon out of the method that claims a key.
        The bound is the check. Before, the check was the declared parameter
        type, and a caller that passed the name of a non-mutation got no
        complaint. The stamp is applied by `model_copy`, and not by two more
        keyword arguments that every caller had to remember.

        **The reply is `R | MutationRepeated`, and that union is the correct
        one.** A repeat answers with one shape for every verb (`MUTATIONS` says
        why). For the verbs whose own reply is a `RootDescription`, the repeat
        is a subtype. But `session/shell` answers with an exit code, so a caller
        that ignored the union would read fields from a description that is not
        there. Narrow on `isinstance(reply, MutationRepeated)`. That is the
        branch that `MutationRepeated.repeated` was always for.
        """
        self._commands += 1
        keyed = params.model_copy(update={"client_id": self.id, "command_id": str(self._commands)})
        wire = await self.peer.ask(verb.name, keyed.to_wire())
        # The frame decides whether this is a repeat, not the verb. The same verb
        # answers in both ways, and the `repeated` field says which.
        if wire.get("repeated"):
            return MutationRepeated.model_validate(wire)
        return verb.read(wire)

    async def prompt(self, session_id: str, text: str) -> RootDescription:
        """Queue a turn. `mutate` keys the call, and says why.

        The return type is `RootDescription` and not
        `RootDescription | MutationRepeated`, which it used to be.
        `MutationRepeated` subclasses `RootDescription`, so that union
        normalizes to the left side. It asked every caller to narrow something
        that the checker had already flattened. The repeat still arrives here.
        It is already the type that it claims to be, which is the property that
        the docstring of `MutationRepeated` describes.
        """
        return await self.mutate(
            verbs.SESSION_PROMPT, PromptParams(session_id=session_id, prompt=text)
        )

    @overload
    async def call[P: WireModel, R: WireModel](self, verb: Verb[P, R], params: P, /) -> R: ...

    @overload
    async def call(self, method: str, /, **fields: object) -> dict[str, Any]: ...

    async def call(
        self, verb: str | Verb[Any, Any], params: WireModel | None = None, /, **fields: object
    ) -> Any:
        """One request, awaited to its reply. Raises what the server refused.

        **There are two forms, and the `Verb` form is the one to use.**
        `client.call(SESSION_STATUS, SessionParams(session_id=s))` checks three
        things at *this* end that were unchecked before: the field spellings
        (P8-09 provided that), that the params belong to this method, and what
        comes back. The old form
        `client.call("session/status", SessionParams(...))` type-checked, and so
        did `client.call("sessions/list", SessionParams(...))`: a name and a
        model that had to agree, with nothing to make them agree.

        The reply is validated here. That retires
        `AttachReply.model_validate(await client.call(...))`, where the model
        was chosen by hand next to the method name. It could be the wrong model,
        and nothing would report it.

        The `**fields` form stays, and is now the only form that takes a bare
        name. A caller with no model to build could not write a typed-only
        signature. An example is a test that sends a deliberately malformed
        frame to exercise a refusal, which is most of what
        `test_daemon_methods` does. Because this form takes no positional
        model, the mismatched pair that is described above is not only
        discouraged: it cannot be written.

        The arguments are positional-only, so a model can never be confused with
        a field named `params` on some future method.
        """
        if isinstance(verb, Verb):
            return await self._exchange(verb, params)
        return await self.peer.ask(verb, fields)

    async def _exchange[P: WireModel, R: WireModel](self, verb: Verb[P, R], params: P | None) -> R:
        """The body of the `Verb` form of `call`, in a place where it can be
        checked.

        The implementation of an `@overload`ed function is typed against the
        union of its signatures and returns `Any`. So nothing in `call` above
        checks that the verb branch validates at all. When the branch was
        sabotaged to `return wire`, which drops the `verb.read` that is the
        purpose of the typed form, `mypy --strict` passed on all 266 files.
        Here `R` is bound, so the body is checked and that sabotage is an error.

        `params` is `P | None` and not `P` only because the implementation
        signature above cannot express "required on this branch". Both overloads
        declare it required, so the `{}` cannot be reached through the public
        method.
        """
        return verb.read(await self.peer.ask(verb.name, params.to_wire() if params else {}))

    @overload
    async def notify[P: WireModel](self, verb: Notify[P], params: P, /) -> None: ...

    @overload
    async def notify(self, method: str, /, **fields: object) -> None: ...

    async def notify(
        self, verb: str | Notify[Any], params: WireModel | None = None, /, **fields: object
    ) -> None:
        """Send a request that expects no reply.

        `shutdown` is the case that matters. A request with a reply would make
        the caller wait for a frame while the daemon is tearing down its ability
        to send. "Stop" is not a question, so it does not get an id.

        **This takes a `Notify`, not a `Verb`, and that pairing is what this
        method adds.** While `SHUTDOWN` was a `Verb` with an unread reply model,
        the rule existed only in prose. `client.call(SHUTDOWN, NoParams())`
        type-checked, all 19 sites used `notify` by convention, and the hang
        that the paragraph above describes was one edit away. The two methods
        now take different types, so neither mistake can be written. `call`
        refuses a `Notify`. This method refuses a `Verb`, which would otherwise
        run a method on the daemon and silently drop its answer.

        This waits for room and does not refuse, unlike `tell` on the daemon: a
        client that cannot write has nobody to drop except itself.
        """
        name = verb.name if isinstance(verb, Notify) else verb
        body = params.to_wire() if params is not None else fields
        await self.peer.send(notification(name, body))

    async def aclose(self) -> None:
        await self.stream.aclose()


async def connected[T](path: Path, work: Exchange[T]) -> T:
    """Connect, run one exchange, and close, with the pump running alongside.

    The pump must be a task, and not something that the caller drives, because
    replies and notifications arrive on the same stream. A caller that read its
    own reply directly would consume a `session.event` that it could not give
    back. The close of the stream ends the pump, so nothing is canceled here. A
    teardown that canceled would race the last frame that it asked for.

    This is here and not in `phern agents`, because a one-shot exchange is not
    specific to a CLI. `ph_app.web.serve` stages the upload of a browser in this
    way too, and the copy that it started with derived every subtlety below
    again.

    **The exception leaves this function, not the wrapper.** anyio wraps
    whatever comes out of a task group. A `DaemonError` that the server sent
    would therefore reach a caller as a group of one, and match no `except`
    that was written for it. Every caller solved that hazard for itself, one
    with `_alone` and one with `except*`. The group belongs to *this* function,
    so this function also unwraps it, and a caller writes the plain
    `except DaemonError` that it meant. A group that holds more than one
    exception is raised again whole: two simultaneous failures want neither
    caller's sentence.
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
    # `outcome` is assigned inside the group and returned outside it. The
    # `__aexit__` of a task group is typed as one that may suppress, so a `return`
    # in the block leaves the function with a path that falls off the end.
    return outcome


def _alone(raised: BaseException) -> BaseException:
    """The single exception inside the wrapper of a task group, if the wrapper
    holds nothing else.

    Nested groups are unwrapped too. A task group inside a task group is two
    layers around one exception, and a caller cares about neither layer.
    """
    while isinstance(raised, BaseExceptionGroup) and len(raised.exceptions) == 1:
        raised = raised.exceptions[0]
    return raised
