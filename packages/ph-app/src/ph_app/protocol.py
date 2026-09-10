"""The wire vocabulary both transports speak (P5-02, I-7).

pH answers on two: `--mode rpc` over stdio, for a caller that owns the process,
and `$PH_RUNTIME/daemon.sock`, for one that does not. **They are the same
protocol**, and this module is what makes that checkable rather than intended.

**The names are dsh's**, deliberately: `initialize`, `session/prompt`,
`session.event`, `session.status`. dsh already ships a Python client for this
shape, and a second vocabulary would make "use the client you have" false in
exactly the deployment Phase 5 exists for.

**Envelope here, methods there.** What is genuinely transport-independent is the
request/reply/error shaping and the version; what a transport serves — one session
over a pipe, or many supervised roots over a socket — is its own. So this module
owns `respond`, `notify` and the capability block, and each server owns its method
table — the request params both servers check against are `ph_app.params`, and
what the daemon emits is `ph_app.payloads`.

**Two kinds of frame, typed two ways (P8-07).** A frame this side *builds* is one
of the `TypedDict`s below: `notification`, `request` and `respond` return them,
the outbox carries them, and a builder that forgot `jsonrpc` or spelled `params`
wrong is a type error rather than a peer's parse error. A frame the peer *sent*
is a `dict[str, Any]` and stays one: it is a claim, read with `.get`, and giving
it one of these names would be asserting what has not been checked — the same
line `ph_rlm.kernel.codec` draws for the fd-3 channel (C10). What *is* checked
about an inbound request is its `params`, and that check is `parse_params`: one
`WireModel` per method, `extra="forbid"`, refusing with a named reason.

@module ph_app.protocol
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from pydantic import Field, ValidationError

from ph.wire import WireModel, validation_errors

__all__ = [
    "PROTOCOL_VERSION",
    "SNAPSHOT_EVENTS",
    "CapabilityBlock",
    "Cursor",
    "DaemonError",
    "DaemonGone",
    "Dispatch",
    "ErrorBody",
    "ErrorFrame",
    "Frame",
    "InvalidParams",
    "NoParams",
    "NotificationFrame",
    "Notify",
    "Refusal",
    "ReplyFrame",
    "RequestFrame",
    "ResultFrame",
    "SeamAbsent",
    "SessionParams",
    "UnknownMethod",
    "Verb",
    "capabilities",
    "cursor_of",
    "cursor_text",
    "notification",
    "parse_cursor",
    "parse_params",
    "request",
    "respond",
    "result_of",
    "resume_at",
    "served",
]

PROTOCOL_VERSION = 1
"""One number, in one place.

It was declared twice — once per transport — which is how two servers come to
claim the same version for two different vocabularies.
"""

SNAPSHOT_EVENTS = 2048
"""How many events one `session/snapshot` reply carries.

A resumed root can hold hundreds of thousands of events, and a client that asked
for its history should not be handed a frame that trips the transport's own
`MAX_LINE` — or a reply it must buffer whole before rendering a line. The cursor
in each reply is what asks for the next page.

**A count, not a byte budget.** Measuring each event with its own `dumps` to fill
a byte page costs more than the encode it exists to bound, and all of it is
discarded. A count needs no measuring pass, and the transport's `MAX_LINE` is the
real protection against an oversized frame.
"""

Dispatch = Callable[[str, dict[str, Any]], Awaitable[Any]]
"""A server's method table: `(method, params) -> result`, raising to refuse.

`params` arrives as the peer sent it. Narrowing it to the method's model is the
table's first move — `parse_params` — and not this signature's, because the
model depends on the method and the envelope does not know the vocabulary."""


# ---------------------------------------------------------------- envelopes --
# JSON-RPC 2.0's four shapes, as the `TypedDict`s this side builds. `Literal["2.0"]`
# on each rather than a shared base: a `TypedDict` base with only `jsonrpc` would
# let a reader accept "any frame" where every reader here wants one direction.


class NotificationFrame(TypedDict):
    """A frame with no `id`, which is what makes it a notification."""

    jsonrpc: Literal["2.0"]
    method: str
    params: dict[str, Any]


class RequestFrame(TypedDict):
    """A frame that expects a reply. `id` is `int | str` because both ends mint
    them — `request` says why."""

    jsonrpc: Literal["2.0"]
    id: int | str
    method: str
    params: dict[str, Any]


class ErrorData(TypedDict):
    """The one member pH puts under `error.data`: the refusal's name."""

    reason: str


class ErrorBody(TypedDict):
    """JSON-RPC's error object. `data` is present exactly when the refusal named
    itself — `respond` says why the code stays generic and the name does not."""

    code: int
    message: str
    data: NotRequired[ErrorData]


class ResultFrame(TypedDict):
    jsonrpc: Literal["2.0"]
    id: int | str
    result: Any


class ErrorFrame(TypedDict):
    jsonrpc: Literal["2.0"]
    id: int | str
    error: ErrorBody


ReplyFrame: TypeAlias = ResultFrame | ErrorFrame
"""What `respond` returns for a request that wanted an answer."""

Frame: TypeAlias = NotificationFrame | RequestFrame | ResultFrame | ErrorFrame
"""Every frame this side builds — what the outbox carries and `write_frame` takes."""


class Refusal(Exception):
    """A refusal that names itself, for a server to raise.

    `code` is a class attribute here because these refusals are *kinds* — an
    unknown method is one thing whatever the method was. An error that computes
    its code per instance sets `self.code` instead, which is what `ph-core`'s
    coded errors do (`HarnessError`, `SessionForkError`, `CompactionError`) and
    what `respond` reads, so both shapes reach the wire through one path.

    Declared rather than duck-typed, for the reason the seams give for their
    provider Protocols: a probe finds whatever happens to be called `code`, and
    an error whose attribute drifted would go out unnamed with nothing to say
    so.
    """

    code = ""


class SeamAbsent(Refusal):
    """This deployment did not mount the seam that method needs.

    Its own code, because `unknown_method` is a different sentence: that one
    means "this daemon is older than you think" and a client responds by
    disabling the feature everywhere. This one means "this deployment does not
    do that", which is a per-root fact and the right thing to grey out one
    button over.

    The read-side projections answer absence with an empty list for the same
    reason stated the other way round — see `projections.py` — so the two halves
    agree that a missing seam is a fact about the profile, not a fault.

    **Here rather than in the daemon**, because the in-process front end refuses
    for the same reason and cannot import from `daemon` without a cycle. A
    fourth refusal class per absent seam is the accumulation this one was
    generalized to stop.
    """

    code = "seam_absent"


class UnknownMethod(Refusal):
    """This server does not serve that name.

    Here rather than in either server, because both raise it and a client
    branching on `unknown_method` must not have to know which transport
    answered. It lived in `daemon/server.py`, which a front end may not import
    (`test_app_layering`), so the stdio transport had grown a byte-identical
    second copy — one refusal code, two definitions, nothing comparing them.
    """

    code = "unknown_method"


class InvalidParams(Refusal):
    """The request named a method this server serves, with params it does not take.

    Its own code because the client's next move differs from `unknown_method`'s:
    that one means "this daemon is older than you think", this one means "you
    spelled the call wrong" — a missing `sessionId`, a field the method does not
    take, a cursor that is not a cursor. Before P8-07 the first of those was a
    `KeyError: 'sessionId'` rendered as the error message, which named the field
    and nothing else, and the second was silently ignored.
    """

    code = "invalid_params"


class DaemonError(RuntimeError):
    """A refusal the server sent back, with its name where it had one.

    The other end of `respond`'s `data.reason`. Without it a client that wanted
    to tell `session_already_active` (I-5) from a mistyped method had to match
    on the message text — which is a contract nobody wrote down and every
    rewording breaks. `reason` is `""` when the server did not name one.
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason

    @classmethod
    def of(cls, error: dict[str, Any]) -> DaemonError:
        """Rebuild the refusal from an error frame."""
        data = error.get("data")
        reason = data.get("reason", "") if isinstance(data, dict) else ""
        return cls(str(error.get("message", "the daemon refused")), str(reason))


class DaemonGone(DaemonError):
    """The connection ended without an answer. Nobody refused anything.

    A subclass, so a caller that only wants "the call did not succeed" still
    catches `DaemonError` — and a distinct type, because the two are opposite
    diagnoses and a client renders them differently. Reported as a refusal, the
    message read "the daemon refused: the daemon closed the connection", which
    is precisely the confusion `ph agents`' absent-socket / stale-socket split
    exists to prevent.
    """

    def __init__(self, message: str = "the daemon closed the connection") -> None:
        super().__init__(message, "connection_closed")


class CapabilityBlock(WireModel):
    """`initialize` and `daemon/hello` — the version, and what this end serves.

    Both names answer with one handler, so both verbs name this one reply. The
    values are all `True` by construction (`protocol.capabilities` says why a
    capability is present or absent rather than a flag), which is what makes
    `dict[str, bool]` the honest declaration instead of a field per name.
    """

    protocol_version: int
    capabilities: dict[str, bool] = Field(default_factory=dict)


def capabilities(*names: str) -> CapabilityBlock:
    """The `initialize` reply, with whatever this transport adds.

    `sessions` and `streaming` are true of both; a transport that supervises
    adds `roots`, `attach`, `cursors`, `snapshots`, and one that does not simply
    omits them — a client reads the block rather than inferring from which
    socket it happened to open. Names rather than `**kwargs`, because every
    value is `True` by construction: a capability is present or absent.
    """
    return CapabilityBlock(protocol_version=PROTOCOL_VERSION, capabilities=served(*names))


def served(*names: str) -> dict[str, bool]:
    """The capability map alone, for a reply that *is* a `CapabilityBlock`.

    `DaemonStatusReply` subclasses it, so `daemon/status` was building a whole
    `CapabilityBlock` and unpacking two fields out of it — which named both a
    second time, so a third field added to the block would reach `initialize`
    and silently not reach `daemon/status`. Here the only thing said twice is
    `PROTOCOL_VERSION`, which is a constant reference and cannot disagree with
    itself. Split out rather than spreading `model_dump()` because a `**` spread
    would take `pid`, `socket` and the ten cadences with it and stop mypy
    checking any of them.
    """
    return {"sessions": True, "streaming": True, **dict.fromkeys(names, True)}


def notification(method: str, params: dict[str, Any]) -> NotificationFrame:
    """A frame with no id, which is what makes it a notification."""
    return {"jsonrpc": "2.0", "method": method, "params": params}


def request(request_id: int | str, method: str, params: dict[str, Any]) -> RequestFrame:
    """A frame that expects a reply — from either side.

    `int | str` because both ends mint ids now (P5-13): a client counts its own
    calls, and the daemon mints `"s<n>"` for the questions it puts *to* a client.
    Strings and ints cannot collide, which is what lets a reader of a frame log
    tell at a glance which side asked.

    **Which means `id` cannot say which direction a frame is going, and `method`
    can.** A frame carrying one is somebody asking; a frame without one is an
    answer. Both pumps route on that, and both were written assuming the
    opposite — the reason it is stated here rather than twice over there.
    """
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def result_of(frame: dict[str, Any]) -> dict[str, Any]:
    """One reply frame as its result, raising whatever the peer refused.

    Both directions unwrap a reply the same way, so this is the one place that
    knows an `error` member outranks a `result` — written twice, the two copies
    disagreed about what an empty frame meant.
    """
    if "error" in frame:
        raise DaemonError.of(frame["error"])
    result: dict[str, Any] = frame.get("result") or {}
    return result


class Cursor(WireModel):
    """Where a reader has got to: `{generation, sequence}`.

    A sequence alone is only meaningful against the log that counted it, so it
    travels with the identity of that log. `generation` is the header's
    `createdAt` as a string: durable, already on the wire, stable across a resume
    — which continues the same log — and different for anything that is not
    that log.

    A model rather than the dict it was, so the shape has one spelling: built
    here by `cursor_of`, sent back inside a method's params, and read by
    `resume_at` off the model rather than through two `isinstance` checks. A
    client sends it as the dict `to_wire()` produced and the method's params
    model rebuilds it — so a cursor that is not one is refused as
    `invalid_params` at the edge, where a *stale* one (right shape, another
    log's generation) is still answered by `resume_at` with 0.
    """

    generation: str
    sequence: int


def cursor_of(session: Any, sequence: int | None = None) -> Cursor:
    """A session's position as a `Cursor`; `to_wire()` puts it in a reply.

    Here rather than on the daemon's `Root`, because it is a fact about a
    *session*: the stdio transport serves the same protocol and would otherwise
    have to re-derive it.
    """
    return Cursor(
        generation=str(session.header.created_at),
        sequence=session.seq if sequence is None else sequence,
    )


def cursor_text(cursor: Cursor) -> str:
    """A cursor as `GENERATION:SEQ` — the form a person or a script hands back.

    Beside `cursor_of` for `cursor_of`'s own stated reason: the shape is a fact
    about a session, not about one transport or one command, and the printed form
    is the same fact spelled for a terminal. It lived as an f-string in the CLI's
    status table and a `rpartition` in its option parser, 350 lines apart, with
    nothing tying the two.

    **Takes the model, and this used to be `Any`** — a `dict`-or-nothing test
    whose `else` branch returned `":"`. That was written when every caller held
    a reply dict, and it went wrong the moment one held a `Cursor`: `ph agents
    status` printed `--since :` and nothing complained, because `Any` accepts a
    model and the `isinstance(cursor, dict)` test quietly failed. Issue 74's own
    lesson, arriving inside issue 74 — a widened parameter is a check deleted,
    and the deletion surfaces at the call site that changes.
    """
    return f"{cursor.generation}:{cursor.sequence}"


def parse_cursor(text: str, current: Any) -> Cursor | None:
    """`GENERATION:SEQ` or a bare `SEQ`, as a cursor — or `None` if it is neither.

    Two spellings, and only one of them can be checked. The full form is a cursor
    a reader kept from an earlier read, passed through intact so `resume_at` can
    refuse it when the log has since been forked or rebuilt. A bare sequence is
    stamped with `current`'s generation — the only thing that makes a typed number
    mean anything, and exactly what makes it unverifiable.

    `generation` is `SessionHeader.created_at`, an integer, so the split is
    unambiguous. `None` rather than a raise: what to do about an unparseable
    cursor is the caller's — the CLI exits 2, and a front end reading a stored
    position would rather start from the beginning than fail to open.

    Returns the model, not its dict: a caller putting it on the wire says
    `.to_wire()` there, which is the one place the shape becomes JSON.
    """
    generation, separator, sequence = text.rpartition(":")
    if not separator:
        fields = current if isinstance(current, dict) else {}
        generation, sequence = str(fields.get("generation", "")), text
    if not (sequence.isdigit() and generation.isdigit()):
        return None
    return Cursor(generation=generation, sequence=int(sequence))


def resume_at(session: Any, cursor: Cursor | None) -> int:
    """The index a cursor asks to resume from, or 0 when it cannot say.

    A cursor from another incarnation of the log is neither honoured nor
    refused: honouring it would skip events the client never saw, refusing it
    would strand a client that did nothing wrong. So a stale generation reads as
    "you have seen nothing of *this* log" — the only safe reading of the two,
    and the reply says where it actually started so the client is not left
    inferring it from sequence numbers.

    Shape is the params model's business, which is why there is no `isinstance`
    here any more: a cursor that is not a cursor never reaches this.
    """
    if cursor is None or cursor.generation != str(session.header.created_at):
        return 0
    seq: int = session.seq
    return max(0, min(cursor.sequence, seq))


def parse_params[P: WireModel](method: str, model: type[P], params: object) -> P:
    """A request's params as the method's model, or the refusal that names why.

    One place for every server, so the sentence a client reads for a bad call is
    the same over the socket and over stdio. `extra="forbid"` is inherited from
    `WireModel`: a field the method does not take is refused rather than
    ignored, because the ignored field was a client that believed it had said
    something. Pydantic's error text is kept — it names the field and the
    shape — under the method's name, which is what the reader is missing.
    """
    try:
        return model.model_validate(params)
    except ValidationError as error:
        raise InvalidParams(
            f"{method}: {'; '.join(validation_errors(error, root='params'))}"
        ) from error


@dataclass(frozen=True, slots=True)
class _Spoken[P: WireModel]:
    """A wire method name bound to the params model it carries.

    The half `Verb` and `Notify` share. Private because neither door takes it:
    `DaemonClient.call` wants a `Verb` and `notify` wants a `Notify`, and the
    whole point of there being two types is that the wrong one is refused. A
    base they *both* satisfy would be a third door accepting either.
    """

    name: str
    params: type[P]

    def parse(self, params: object) -> P:
        """The wire params as this method's model, or the refusal naming why.

        On the method rather than at the dispatch, because the method is what
        knows the model: the caller used to pass both, which is the pair these
        types exist to stop being a pair.
        """
        return parse_params(self.name, self.params, params)


@dataclass(frozen=True, slots=True)
class Notify[P: WireModel](_Spoken[P]):
    """A method with **no reply**: a name, its params, and nothing coming back.

    `shutdown` is the one, and its hazard is why this type exists rather than a
    `Verb` with an unread reply model. A request-with-reply would leave the
    caller waiting on a frame the daemon is concurrently losing the ability to
    write — so "stop" is not a question and does not get an id. But while
    `shutdown` was a `Verb`, `client.call(SHUTDOWN, NoParams())` type-checked:
    every one of the 19 call sites happened to use `notify`, and nothing made
    that a rule. Now `call` cannot take this and `notify` cannot take a `Verb`,
    so the hang is unexpressible rather than merely avoided.

    It also deletes a model. `ShutdownAck` existed because a `Verb` must name a
    reply type and the handler must return *something*, for a frame `respond`
    computes and then drops because the id is `None` — eight lines describing a
    reply nobody reads. A handler for one of these returns `None`, which is the
    honest shape.

    **Not a subtype of `Verb`, and `Verb` not a subtype of this.** Making one
    extend the other would let `notify(SESSION_STATUS, …)` through, which runs
    a method on the daemon and silently discards its answer — a different way
    to be wrong about the same distinction.
    """


@dataclass(frozen=True, slots=True)
class Verb[P: WireModel, R: WireModel](_Spoken[P]):
    """One method: its name, what it takes, and what it answers — as one value.

    **The three used to be three.** A name was a string literal at the call
    site, the params model was named again in the server's table, and the reply
    was whatever the handler happened to return — so
    `client.call("session/status", SessionParams(session_id=s))` type-checked and
    so did `client.call("sessions/list", SessionParams(...))`. P8-09 bought
    field-spelling inside the params; it could not check that the params
    belonged to the method, because nothing held the two together. A `Verb` is
    that holding: `client.call(SESSION_STATUS, SessionParams(...))` cannot be
    given another verb's params, and its reply arrives as
    `RootStatusReply` rather than as a dict the caller re-narrows by hand.

    Declared in `ph_app.verbs`, which both ends import — that is the whole
    point, and the reason this class lives here rather than there: `protocol.py`
    is what a transport may depend on, and a verb table that imported the
    server's handlers could not be read by a client.

    **Notifications are not verbs, and `METHOD` is not replaced.** A notice
    already binds its name to its payload on the payload itself
    (`SessionScoped.METHOD`), which is what let `Root.publish` take one argument
    instead of two that had to agree. A `Verb` owning those names would put that
    back, so it does not: verbs cover the request/reply half, and the emitted
    half keeps the binding it has.
    """

    reply: type[R]

    def read(self, wire: dict[str, Any]) -> R:
        """A reply frame as this verb's model.

        What retires `AttachReply.model_validate(await client.call(...))` — a
        dump-then-reparse whose model was chosen by hand at the call site, and
        so could be the wrong one with nothing to say so.
        """
        return self.reply.model_validate(wire)


class NoParams(WireModel):
    """A method that takes none.

    A model rather than skipping the parse, so that a stray field on one of
    these is refused like a stray field anywhere else — the rule has no
    exceptions, which is what makes it a rule. Here rather than in either
    server's own table for `Cursor`'s reason: both transports have methods that
    take nothing, so the shape is the protocol's.
    """


class SessionParams(WireModel):
    """Every method about one session starts here.

    The protocol's, for the same reason: `sessionId` is what a method about a
    session is *about*, on either transport. What each server adds to it is its
    own — the daemon's idempotence key, stdio's optional id — which is where
    the two vocabularies genuinely differ.
    """

    session_id: str


async def respond(request_frame: dict[str, Any], dispatch: Dispatch) -> ReplyFrame | None:
    """Run one request and shape its reply, or `None` if it wanted none.

    A failing method is *this call's* failure, not the connection's: an unknown
    method or a bad argument comes back as an error frame and the peer keeps
    talking. Framing errors are the transport's and end the stream, because
    after a bad frame there is no way to know where the next one starts.

    The body runs **before** the `id` check, not after: an id-less frame is a
    method whose answer nobody wants (`shutdown`), and its body still has to run.
    """
    request_id: int | str | None = request_frame.get("id")
    method = str(request_frame.get("method", ""))
    params = request_frame.get("params") or {}
    try:
        answer = await dispatch(method, params)
        # A handler that answers with a model is dumped **here**, at the one
        # point a result becomes a frame — so every handler either returns its
        # reply's type or a plain dict, and none of them spells `.to_wire()`
        # at the `return`. `dumps` cannot encode a model, so this is also the
        # only place that could have gone wrong.
        result = answer.to_wire() if isinstance(answer, WireModel) else answer
    except Exception as error:
        failure: ErrorBody = {"code": -32000, "message": str(error)}
        # A refusal a client is expected to *branch* on carries a name rather
        # than making the client match message text: `session_already_active`
        # (I-5) is a thing to retry elsewhere, and telling it from a typo in a
        # method name should not mean grepping prose. JSON-RPC's own code stays
        # generic, because these are pH's vocabulary and not the transport's.
        #
        # Read off the *instance*, which is what already carries a code here —
        # `HarnessError`, `SessionForkError` and friends all set `self.code` in
        # `__init__`, so keying on `type(error)` would make those four families
        # structurally invisible.
        reason = getattr(error, "code", "")
        if isinstance(reason, str) and reason:
            failure["data"] = {"reason": reason}
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "error": failure}
    if request_id is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}
