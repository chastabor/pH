"""`ctx.llm`: the model adapter seam.

Definition, Provider, Consumer (invariant I5). The *definition* is `LlmAdapter`.
A *provider* is any plugin that calls `ctx.llm.register_adapter`. The *consumer*
is the loop, which never learns which adapter answered.

Every call goes through the `llm/stream` waterfall and then through `llm/send`.
A row that must check or change a request before it is sent listens on
`llm/stream`: `agent-loop-invariant` and `media-degrade` attach there. A barrier
waits on `llm/send`. If an adapter raises, the seam normalizes the error into a
terminal `finish{error}` chunk before any consumer sees it. So a consumer never
handles two shapes for the same failure.

A route may also *decide* (`DecisionAdapter`). It answers typed questions about
a state with probabilities, and not with text. `ctx.llm.decide` is the call for
that.

@module ph.llm.adapter
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

import anyio

from ..cordis import Context, Running, events, plugin, running
from ..json import JsonValue
from ..keys import LLM
from ..session.writers import log_writer
from .types import (
    BAD_DECISION,
    CONTEXT_WINDOW_EXCEEDED,
    NO_DECISIONS,
    TIMEOUT,
    CallSetting,
    Finish,
    FinishReason,
    GenerateOptions,
    LlmCallConfig,
    LlmFailure,
    Sampling,
    SamplingSetting,
    StreamChunk,
    TokenUsage,
    unmapped_effort,
)

if TYPE_CHECKING:
    from ..session import Session

__all__ = [
    "DECIDED",
    "DECISION_FAILED",
    "AdapterHandle",
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "DecisionAdapter",
    "DecisionRequest",
    "Decisions",
    "LlmAdapter",
    "LlmError",
    "LlmRuntime",
    "MediaRoute",
    "ModelResolver",
    "Noul",
    "NoulAnswer",
    "Question",
    "ResolvedModel",
    "Score",
    "ScoreAnswer",
    "apply",
    "failure_of",
    "input_too_long",
    "refused_share",
    "resolved",
    "too_long_facts",
]

log = logging.getLogger("ph.llm")

_LOG = log_writer(__name__)

DECIDED = "llm/decided"
"""A decision model answered questions on a route for the agent of a session. The
record carries the route, the number of questions, and the `usage` of the
provider when the provider reports one. It is a record that spends tokens
(`ph.seams.token_meter.SPENDING_TYPES`)."""

DECISION_FAILED = "llm/decision-failed"
"""A route would not decide. The record carries the route, the number of
questions, and the `code` and `reason` of the failure."""

events.declare(
    "llm/stream",
    "waterfall",
    GenerateOptions,
    owner="ph.llm",
    doc="Wraps every model call. Media degradation and the stream invariant attach here.",
)
events.declare(
    "llm/send",
    "serial",
    GenerateOptions,
    owner="ph.llm",
    doc="A request about to reach its adapter, once every `llm/stream` listener has had its "
    "say. A barrier that must cover what they appended waits here.",
)
events.declare(
    "llm/adapters-updated",
    "emit",
    owner="ph.llm",
    doc="The provider topology changed; consumers re-read list_providers().",
)


class LlmError(Exception):
    """A structured model-call failure.

    It carries the facts of the provider and not a flattened string, because
    `turn/end{error}` records them verbatim and a later reader needs the code.
    """

    def __init__(self, message: str, code: str, failure: LlmFailure | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.failure = failure or failure_of(message, code)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Exact-route metadata an adapter can answer for one provider/model."""

    context_window: int | None = None
    default_max_tokens: int | None = None
    accepts: frozenset[str] = frozenset()
    """The MIME types that this route takes as message *content* (P7-01).

    Empty means text only. That is every route until an adapter says otherwise,
    and it is the safe default: the failure that it prevents is a media block
    that reaches a wire that silently drops it. A caller that finds a block
    outside this set degrades it to a text pointer and logs a notice. It never
    sends the block in the hope that it works. It also never fails the turn,
    because a session that began on a vision model and is resumed on a text
    model must still open."""
    max_attachment_bytes: int | None = None
    """The ceiling for one attachment that goes *inline*, when the route
    publishes one.

    It is a limit and not only `accepts`, because with it an over-sized block
    degrades to a pointer here and is not rejected at the provider. It does not
    apply to a MIME type in `uploads`."""
    uploads: frozenset[str] = frozenset()
    """The MIME types that the adapter of this route sends by reference, through
    the file API of its provider (P7-03).

    A file of one of these types is not held to `max_attachment_bytes`, because
    that number is the ceiling of an inline block. On the Google route it is
    the 20 MB limit of a whole request, and a video of 300 MB is fine by
    reference. `media-degrade` used to apply the inline ceiling to every block
    before the adapter ran, so such a video became a pointer and was never
    uploaded. The adapter now decides: it references the file, and when the
    upload gives no handle it sends a file under the ceiling inline and a file
    over it as a pointer.

    Empty means that nothing is uploaded. That is the answer of a route with no
    file API, and of any route in a deployment where `ctx.uploads` is not
    mounted."""
    max_image_edge: int | None = None
    """The longest edge in pixels that the route will *accept*, when it publishes one.

    An earlier docstring beside this one said that a pixel ceiling had to wait
    for `media-transform` (P7-02), because "declaring one now would be a knob
    wired to nothing". Two things changed that (P7-03). `ph.llm.dimensions` reads
    the size of an image from its header with no dependency. And a limit does
    not need a resizer to be worth declaring: a refusal with a sentence that the
    model can read is an action, and `max_attachment_bytes` already takes the
    same action."""
    structured_output: bool = False
    """Whether this route enforces `GenerateOptions.response_schema` on the wire.

    It is declared and not assumed, and it defaults to *no*, for the same reason
    as `accepts`: the failure that it prevents is a caller that believes a shape
    was guaranteed when the provider ignored the field. OpenAI-compatible
    endpoints take `response_format: {"type": "json_schema", …}`. Anthropic has
    no equivalent today, so a caller there gets the instruction and the
    validation, and not the guarantee. That is a real difference, and
    `ph.llm.structured` states it and does not hide it."""
    usable_image_edge: int | None = None
    """The longest edge that the route can *use*, when that is less than it accepts.

    Providers scale a large image down before the model sees it. Anthropic
    scales to 1568 px, and it accepts up to 8000. So pixels above this edge are
    uploaded, paid for, and discarded at the far end. This field is separate
    from `max_image_edge` because the outcomes differ in kind. Over
    `max_image_edge`, nothing is sent. Over this edge, everything works and the
    user quietly overpays. That failure lasts for a whole session, because
    nothing announces it."""
    sampling: frozenset[SamplingSetting] | None = None
    """The sampling settings that the API of this route takes (A1), or `None` when
    its adapter has not said.

    A setting outside this set is dropped before the request reaches the adapter.
    The seam warns one time for each route and setting, and the `models` row
    names the setting in `phern doctor`. This is graceful degradation: a route
    that runs without a recommended setting still works, and a route that is
    refused does not. `None` drops nothing and warns about nothing, because a
    guess would either warn falsely or send a field that a strict server
    refuses."""
    credential: str | None = None
    """The credential that this route resolves at the adapter edge, **by name**
    (I-3). It is the `apiKeyEnv` that the row of the route names, or `None` for a
    route that needs none.

    With the name, anything above the edge can ask "can this route run here,
    now?" without holding a value. The resume check (T5) reads the name and asks
    `ctx.credentials.has`. So a session whose key is missing waits for the key
    by name, and does not start and then fail at its first request. The check
    reads the adapters of the mounted profile and never a name that was copied
    into the log, because a profile that renamed the variable must be asked
    about the new name."""
    max_decision_tokens: int | None = None
    """The most tokens that one decision on this route may carry, when its row
    gives a number.

    This is a fact about the server and not about the model, and a llama.cpp
    server does not publish it. That server reads a decision in one physical
    batch, so its limit is `--ubatch-size`, which is 512 unless the server was
    started with more. The limit is not its context window. The value is `None`
    for a route that gives no number, and for every route that does not decide.
    An asker reads `LlmRuntime.decision_limit`, because a refusal can teach a
    lower limit than this field gives."""


class MediaRoute(Protocol):
    """The route facts that a config declares and that `ResolvedModel` carries.

    The members are read-only on purpose. The three configs disagree about the
    *types*: Anthropic states a pixel edge as `int`, and the other two allow
    `None`. A property is covariant and a mutable attribute is not, so each
    config may narrow its own field and still satisfy this Protocol.

    The Protocol is declared and not duck-typed. That makes `resolved` below
    more than a shortcut: a config that falls out of step with `ResolvedModel`
    fails at the call site, and does not quietly report a default.

    **It is beside `ResolvedModel` and not beside the adapters that satisfy
    it.** Its members are the six fields of that dataclass that `resolved` reads
    from a config, so the two must change together. In a private module of an
    application package, a third-party adapter would have had to import
    `ph_app.adapters._media` or write a fourth copy of the projection, and
    that drift is what this Protocol exists to stop. This is also the only side
    of the boundary where the invariant can be checked at all: a rename here and
    a matching edit there cannot pass one suite and fail the other.
    """

    @property
    def context_window(self) -> int | None: ...
    @property
    def default_max_tokens(self) -> int | None: ...
    @property
    def accepts(self) -> tuple[str, ...]: ...
    @property
    def max_attachment_bytes(self) -> int | None: ...
    @property
    def max_image_edge(self) -> int | None: ...
    @property
    def usable_image_edge(self) -> int | None: ...


def resolved(
    route: MediaRoute,
    *,
    structured_output: bool,
    credential: str | None,
    sampling: frozenset[SamplingSetting],
    max_decision_tokens: int | None,
    uploads: frozenset[str],
) -> ResolvedModel:
    """The config of one route as the `ResolvedModel` that every layer above reads.

    **This is the projection, not the values.** What a route accepts, how large
    a file it takes, and what it does with pixels are facts about each provider.
    They stay in each config, with the paragraph that argues them. What was
    copied three times is the *field list*: six lines that map one name onto the
    same name. That list is the vocabulary of `ResolvedModel` and not of any
    adapter.

    The cost of the copies was that a seventh capability took three edits. An
    adapter that missed one reported the default and not the number of the
    route, and `media-degrade` then applied the wrong rule for that provider
    alone. Now it takes one edit here, and a config that has not caught up fails
    to type-check.

    The other five arguments are passed and not read from the config:

    * `structured_output` is the one field that is not a config value. It is a
      claim about the *wire*, argued at each call site. A route that declared it
      in config could promise a guarantee that its adapter does not implement.
    * `credential` is not a media fact. It is the name that the adapter resolves
      at its edge, and it is **required** so that an adapter cannot forget to
      give it (T5).
    * `sampling` also belongs to the wire, and it is required for the same
      reason.
    * `max_decision_tokens` is config but not a media fact, and it is required
      too. An adapter that does not decide passes `None`, and an adapter that
      does decide cannot drop the limit of its route unnoticed.
    * `uploads` is what the adapter *will* reference, which is less than what
      the config lists when no upload service is mounted or the wire has no
      reference shape for a type. It is required, so that an adapter with a
      file API cannot leave its large files to the inline ceiling.
    """
    return ResolvedModel(
        context_window=route.context_window,
        default_max_tokens=route.default_max_tokens,
        accepts=frozenset(route.accepts),
        max_attachment_bytes=route.max_attachment_bytes,
        max_image_edge=route.max_image_edge,
        usable_image_edge=route.usable_image_edge,
        structured_output=structured_output,
        credential=credential,
        sampling=sampling,
        max_decision_tokens=max_decision_tokens,
        uploads=uploads,
    )


class LlmAdapter(Protocol):
    """`stream` is the one required method. `resolve_model` is optional, and
    `ModelResolver` declares it."""

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]: ...


@runtime_checkable
class ModelResolver(Protocol):
    """An adapter that knows what its routes can do. **Optional capability.**

    `LlmRuntime.resolve_model` asks with `isinstance`, for the reason that
    `DecisionAdapter` gives. An adapter that does not implement this is not an
    instance, and every route of it resolves to the defaults of `ResolvedModel`.
    """

    def resolve_model(self, provider: str, model: str) -> ResolvedModel: ...


@dataclass(frozen=True, slots=True)
class Noul:
    """A yes-or-no question, answered with how likely *yes* is (`NoulAnswer`).

    The three kinds of question carry the names of the decision API: `noul`,
    `choice`, `score`. Their fields carry its names too. So a reader of the wire
    and a reader of this module read one vocabulary."""

    instructions: str


@dataclass(frozen=True, slots=True)
class Choice:
    """A question answered with one of its `criteria` (`ChoiceAnswer`)."""

    instructions: str
    criteria: Mapping[str, str]
    """The key of each option, and what choosing it means. The meaning is `""` for
    a key that explains itself. There are at least two options, in the order
    that breaks a tie."""


@dataclass(frozen=True, slots=True)
class Score:
    """A question answered with a place on a scale (`ScoreAnswer`)."""

    instructions: str
    criteria: tuple[str, ...]
    """What each level means, lowest first. There are two to ten levels."""


Question: TypeAlias = Noul | Choice | Score
"""One thing a decision model is asked about a state."""


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Questions about one state, on one route.

    This is not a conversation. `state` is the thing that is judged, as text.
    Each question is answered by itself, so one answer never influences another,
    and several questions are one request."""

    provider: str
    model: str
    state: str
    questions: Mapping[str, Question]
    """Each question under the name its answer comes back under."""


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """A `Noul`'s answer."""

    noul: float
    """How likely *yes* is, from 0 to 1. This is a probability and never a verdict.
    The asker has the rule for where to draw the line."""


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """A `Choice`'s answer."""

    choice: str
    """The likeliest option's key."""
    probabilities: Mapping[str, float]
    """Each option's share, adding up to 1, in the order the criteria were given."""
    confidence: float | None = None
    """How far the shares are from an even split, from 0 to 1, when the provider
    reports it. It does not measure how often the answer is right."""


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A `Score`'s answer."""

    score: float
    """The expected level, counted from 0 at the lowest. It can fall between two levels."""
    probabilities: tuple[float, ...]
    """The share of each level, lowest first. The wire keys them `"0"`, `"1"`, ….
    Here the order is the key, and the `criteria` of the question are the
    legend."""
    confidence: float | None = None
    """As `ChoiceAnswer.confidence`."""


Answer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer

_ANSWERS: Mapping[type[Question], type[Answer]] = {
    Noul: NoulAnswer,
    Choice: ChoiceAnswer,
    Score: ScoreAnswer,
}
"""The kind of answer each kind of question is owed."""


@dataclass(frozen=True, slots=True)
class Decisions:
    """One answer for each question, under the name of the question.

    Each answer is the kind that its question is owed, and `LlmRuntime.decide`
    holds every adapter to that. A caller that asked a `Noul` reads a
    `NoulAnswer`."""

    answers: Mapping[str, Answer]
    input_tokens: int | None = None
    """What the provider reports that the request cost, when it reports a cost. A
    decision model scores and writes nothing, so there are no output tokens to
    report."""

    def answer[A: Answer](self, name: str, kind: type[A]) -> A:
        """The answer to `name`, as the `kind` that its question was owed. With this
        reader, a caller does not narrow what `decide` already checked.

        :raises LlmError: `BAD_DECISION`, for a question with no answer or with
            an answer of another kind.
        """
        found = self.answers.get(name)
        if isinstance(found, kind):
            return found
        got = "no answer" if found is None else f"a {type(found).__name__}"
        raise LlmError(
            f'the question "{name}" has {got}, where a {kind.__name__} is owed', BAD_DECISION
        )


@runtime_checkable
class DecisionAdapter(Protocol):
    """An adapter that can decide. **Optional capability.**

    It is its own Protocol beside `LlmAdapter`, and a caller asks with
    `isinstance` and never with a `getattr` probe. This tree states that rule
    for every optional capability: a probe reports an adapter whose method is
    misnamed as an adapter that cannot decide. An adapter for a provider with no
    decision API implements none of this, and is not an instance.

    The credential is resolved inside `decide`, at the edge of the adapter, as
    it is for `stream`. A caller hands in a state and questions and gets answers
    back, and at no point holds a key.
    """

    async def decide(self, request: DecisionRequest) -> Decisions: ...


_TOO_LONG = (
    "context length",
    "context window",
    "context size",
    "exceeds the available context",
    "exceed context",
    "too many tokens",
    "maximum number of tokens",
    "too large to process",
    "too long",
)
"""How embeddings servers and decision servers report that an input did not fit.
llama.cpp answers *"input (1602 tokens) is too large to process"* with a 500.
OpenAI answers *"maximum context length is 8192 tokens"* with a 400. The list
holds phrases, because the status says nothing either way."""


def input_too_long(body: str) -> bool:
    """Whether an embeddings or decision server refused a request because an
    input did not fit.

    Every client of such an API uses this one list. If one client read a refusal
    as "too long" and another read it as a plain failure, the same answer from
    one server would have two outcomes: one client sends a smaller input and
    continues, and the other ends the run. The refusal is classified as
    `CONTEXT_WINDOW_EXCEEDED`, which a caller can act on."""
    lowered = body.lower()
    return any(phrase in lowered for phrase in _TOO_LONG)


_SENT = (r"input \((\d+) tokens\)", r"requested (\d+) tokens", r"resulted in (\d+) tokens")
_READS = (r"batch size: (\d+)", r"maximum context length is (\d+)")


def too_long_facts(body: str) -> tuple[int | None, int | None]:
    """The numbers in a refusal that `input_too_long` recognizes: how many tokens
    the input was, and how many the server takes. Either may be missing.

    A llama.cpp server gives both: *"input (702 tokens) is too large to process
    … (current batch size: 512)"*. OpenAI gives both too. They are worth reading
    because they are exact, and the count of a caller is an estimate. The first
    number says how far over the input was, so the next input is aimed and not
    halved. The second number is the real limit of the server, which is not
    always the limit that it reports when asked. This function is beside
    `_TOO_LONG` because both read the prose of the same servers. A refusal in
    other words has no numbers."""

    def found(patterns: tuple[str, ...]) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, body)
            if match is not None:
                return int(match.group(1))
        return None

    return found(_SENT), found(_READS)


def failure_of(message: str, code: str, *, said: str = "", status: int | None = None) -> LlmFailure:
    """The failure of a provider, with `message` as it is shown.

    A refusal as too long carries the two numbers that it gave (`too_long_facts`)
    as fields. They are read here, one time, from everything that the provider
    `said`. That is the message, unless a builder cut the message down for
    showing. So no caller reads the numbers again out of a shortened message."""
    sent, reads = (
        too_long_facts(said or message) if code == CONTEXT_WINDOW_EXCEEDED else (None, None)
    )
    return LlmFailure(
        message=message, code=code, status=status, sent_tokens=sent, limit_tokens=reads
    )


def refused_share(sent: int | None, reads: int | None) -> float | None:
    """The share of a refused input that the server reads, from the two numbers
    that its refusal gave (`too_long_facts`). It is `None` when the refusal did
    not give both, or gave a limit that the input was within. A caller aims its
    next, shorter input by this share, less a margin."""
    return reads / sent if sent and reads and sent > reads else None


@dataclass(slots=True)
class AdapterHandle:
    """One adapter registration, and the routes it claims."""

    adapter: LlmAdapter
    by: Running
    """Who registered it (P6-29). The `stream` of an adapter is row code that this
    registry invokes, and it used to run unbound. That is the same category as
    the `execute` of a tool.

    P6-29 missed it because both P6-30 walks look for a `Callable`. An adapter is
    an *object that satisfies a Protocol*, so `AdapterHandle.adapter: LlmAdapter`
    names no `Callable`, and this registers without a `claim_*` helper at all.
    The value is resolved from the ambient binding and not from a `scope=`
    argument, because `register_adapter` has never taken one. A row that calls
    it from its own `apply` is the only shape of caller, and that is exactly
    what `running_for(None)` reads."""
    providers: tuple[str, ...]
    _runtime: LlmRuntime
    _disposed: bool = False

    def dispose(self) -> None:
        if self._disposed:
            return
        self._disposed = True
        self._runtime._registrations.remove(self)
        self._runtime._reindex()


@dataclass(slots=True)
class LlmRuntime:
    """The service published as `ctx.llm`."""

    ctx: Context
    _registrations: list[AdapterHandle] = field(default_factory=list)
    _routes: dict[str, AdapterHandle] = field(default_factory=dict)
    _warned: set[str] = field(default_factory=set)
    """Each warning about a withheld setting that was already given. So a session
    warns one time, and not on every request."""
    _decision_limits: dict[tuple[str, str], int] = field(default_factory=dict)
    """The most tokens that a route was found to read in one decision, by route, as
    its refusals reported (`too_long_facts`). It is kept so that the next asker
    sizes its state to fit, and does not spend a refusal to learn the limit
    again. A limit is only ever lowered, and it is kept for the life of the
    process. A server that is restarted with a bigger batch is read at the
    smaller limit until the process ends. That shows an asker less and fails
    nothing."""

    def register_adapter(self, providers: Sequence[str], adapter: LlmAdapter) -> AdapterHandle:
        """Claim one or more provider routes for `adapter`.

        The caller adds the `dispose` of the handle to its own scope, so unloading
        the plugin unregisters the routes.
        """
        handle = AdapterHandle(
            adapter=adapter,
            by=self.ctx.running_for(),
            providers=tuple(providers),
            _runtime=self,
        )
        self._registrations.append(handle)
        self._reindex()
        return handle

    def _reindex(self) -> None:
        self._routes = {
            provider: handle for handle in self._registrations for provider in handle.providers
        }
        self.ctx.emit("llm/adapters-updated")

    def list_providers(self) -> list[str]:
        return sorted(self._routes)

    def _route(self, provider: str) -> AdapterHandle:
        """The whole registration, for the three paths that call into the adapter:
        `stream`, `decide` and `resolve_model`.

        It is the registration and not the adapter alone, because the binding
        needs to know who registered the adapter. Raises `LlmError` with
        `NO_ADAPTER` when no adapter is registered for `provider`.
        """
        handle = self._routes.get(provider)
        if handle is None:
            raise LlmError(f'no adapter is registered for provider "{provider}"', "NO_ADAPTER")
        return handle

    def decision_limit(self, provider: str, model: str) -> int | None:
        """The most tokens that one decision on this route may carry, when anything
        gives a number. It is the `max_decision_tokens` of the route or what a
        refusal taught, whichever is less. It is `None` when neither gives a
        number, and the asker then assumes a limit."""
        said = (
            self.resolve_model(provider, model).max_decision_tokens,
            self._decision_limits.get((provider, model)),
        )
        return min((one for one in said if one is not None), default=None)

    def decides(self, provider: str) -> bool:
        """Whether the adapter that is registered for `provider` can decide. It is
        also `False` for a provider with no registered adapter: there is no
        adapter to ask."""
        handle = self._routes.get(provider)
        return handle is not None and isinstance(handle.adapter, DecisionAdapter)

    async def decide(
        self,
        request: DecisionRequest,
        *,
        session: Session | None = None,
        timeout_s: float | None = None,
    ) -> Decisions:
        """Ask `request.questions` of the adapter that owns its route.

        The call runs as the row that registered the adapter, like `stream`. It
        is refused with a code that names the provider in three cases:

        * The route has no adapter (`NO_ADAPTER`).
        * Its adapter cannot decide (`NO_DECISIONS`).
        * The answers are not the ones that the questions are owed
          (`BAD_DECISION`).

        **`timeout_s` is how long the asker will wait.** A route that takes
        longer is refused as `TIMEOUT`. The timeout is applied here, and not by
        the asker around its call, so that a decision that the asker gave up on
        is recorded like any other failed decision. The asker is usually holding
        something up until the answer comes, such as a turn that is about to
        end. The patience of an adapter itself is that of a model call.

        **`session` says whose spend it is.** A decision is not a request of a
        turn, so nothing else in a log says that a route was used. With a
        session, the cost of the request is recorded in it (`DECIDED`), where a
        budget and the account of a child read every other spend. A request that
        the route refused or failed is recorded there too (`DECISION_FAILED`),
        with the reason. Without a session, the caller decides for no account,
        and nothing is recorded. The asker records what was *asked* and
        *answered*, because it knows what the answer was for."""
        account: dict[str, JsonValue] = {
            "provider": request.provider,
            "model": request.model,
            "questions": len(request.questions),
        }
        try:
            handle = self._route(request.provider)
            adapter = handle.adapter
            if not isinstance(adapter, DecisionAdapter):
                raise LlmError(
                    f'the adapter for provider "{request.provider}" does not decide; name '
                    "a provider whose adapter does",
                    NO_DECISIONS,
                )
            try:
                with anyio.fail_after(timeout_s), running(handle.by):
                    result = await adapter.decide(request)
            except TimeoutError:
                raise LlmError(
                    f'provider "{request.provider}" did not decide within {timeout_s:g} s',
                    TIMEOUT,
                ) from None
            _owed(request, result)
        except Exception as error:
            self._learn(request, error)
            if session is not None:
                failure = _failure_of(error)
                _LOG.append(
                    session,
                    DECISION_FAILED,
                    {**account, "code": failure.code, "reason": failure.message},
                )
            raise
        if session is not None:
            tokens = result.input_tokens
            usage = (
                None
                if tokens is None
                else TokenUsage(input_tokens=tokens, output_tokens=0).to_wire()
            )
            _LOG.append(session, DECIDED, {**account, "usage": usage})
        return result

    def _learn(self, request: DecisionRequest, error: Exception) -> None:
        """Keep the limit that a refusal names (`decision_limit`). The refusal comes
        from a route that reported that its input was too long, and reported how
        long an input it reads."""
        if not isinstance(error, LlmError) or error.code != CONTEXT_WINDOW_EXCEEDED:
            return
        reads = error.failure.limit_tokens
        if reads is not None:
            route = (request.provider, request.model)
            self._decision_limits[route] = min(reads, self._decision_limits.get(route, reads))

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        """Ask the owning adapter what it knows about one exact route."""
        try:
            handle = self._route(provider)
        except LlmError:
            return ResolvedModel()
        adapter = handle.adapter
        if not isinstance(adapter, ModelResolver):
            return ResolvedModel()
        with running(handle.by):
            return adapter.resolve_model(provider, model)

    def withhold(
        self, config: LlmCallConfig, route: ResolvedModel, efforts: Collection[str]
    ) -> tuple[LlmCallConfig, dict[CallSetting, float | int | str]]:
        """`config` as `route` can carry it, and what it withholds, with the values that
        were asked. It withholds a sampling setting that the API of the route does
        not take (`Sampling.uncarried`), and a reasoning effort that the `efforts`
        of its profile do not define (`unmapped_effort`). Each is warned about the
        first time, because someone asked for it.

        The agent loop calls this before it logs the header of the request that it
        sends. So the log says what was sent, and what was asked and not sent
        (`EpochHeader.withheld`). A request that reaches the seam another way is
        held to the sampling half at the edge (`_carried`), and it sends an effort
        only as the `effort_fields` that it is given."""
        withheld: dict[CallSetting, float | int | str] = {}
        if route.sampling is not None:
            dropped = self._uncarried(
                config.provider, config.model, config.sampling, route.sampling
            )
            withheld.update(dropped.items())
        effort = config.reasoning_effort
        if unmapped_effort(effort, efforts):
            withheld["reasoningEffort"] = effort
            self._warn_once(
                "ph.llm: reasoning effort %s is not sent to %s/%s, so the provider's default "
                "applies: its profile says how to send %s",
                effort,
                config.provider,
                config.model,
                ", ".join(efforts) or "none",
            )
        return (config.without(withheld) if withheld else config), withheld

    def _carried(self, request: GenerateOptions) -> GenerateOptions:
        """`request` as its route can carry it. A sampling setting that the API of the
        route does not take is dropped, with a warning the first time
        (`ResolvedModel.sampling`)."""
        if request.sampling.is_empty():
            return request
        carried = self.resolve_model(request.provider, request.model).sampling
        if carried is None:
            return request
        dropped = self._uncarried(request.provider, request.model, request.sampling, carried)
        return replace(request, sampling=request.sampling.without(dropped)) if dropped else request

    def _uncarried(
        self,
        provider: str,
        model: str,
        sampling: Sampling,
        carried: frozenset[SamplingSetting],
    ) -> dict[SamplingSetting, float | int]:
        """The part of `sampling` that a route carrying `carried` cannot send, with the
        values that were asked. Each setting is warned about the first time. Both
        drop paths share this one step."""
        dropped = sampling.uncarried(carried)
        for setting in dropped:
            self._warn_once(
                "ph.llm: %s/%s does not take %s, so it is not sent; it takes %s",
                provider,
                model,
                setting,
                ", ".join(sorted(carried)) or "none",
            )
        return dropped

    def _warn_once(self, message: str, *args: object) -> None:
        said = message % args
        if said not in self._warned:
            self._warned.add(said)
            log.warning("%s", said)

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        """Dispatch one request through the `llm/stream` waterfall."""

        async def inner(request: GenerateOptions) -> AsyncIterator[StreamChunk]:
            handle = self._route(request.provider)
            request = self._carried(request)
            await self.ctx.serial("llm/send", request)
            # The binding goes *inside* the generator (C10). Both
            # `adapter.stream(...)` and `_normalized(...)` are async-generator
            # calls, so this line only constructs them, and no adapter code runs
            # here. Before, the whole stream was pulled later, outside any
            # `running`. Every registration that the adapter made mid-stream then
            # landed on the seam and not on the row, and outlived the row.
            return _normalized(handle.adapter.stream(request), request, handle.by)

        return await self.ctx.waterfall("llm/stream", options, inner=inner)


def _owed(request: DecisionRequest, result: Decisions) -> None:
    """Refuse `result` unless every question has the kind of answer that it is owed.

    The check is here and not in each adapter, so that what a caller may assume
    of `Decisions` is one rule. If an answer that was a choice were read as a
    likelihood, a decision would be made on a number that nobody gave."""
    for name, question in request.questions.items():
        try:
            result.answer(name, _ANSWERS[type(question)])
        except LlmError as error:
            raise LlmError(f'provider "{request.provider}": {error}', BAD_DECISION) from None


def _failure_of(error: Exception) -> LlmFailure:
    """What the raise of an adapter says, as a failure. For an `LlmError` it is the
    failure of the error. For any other exception it is the message under
    `UNKNOWN`. A stream that ends in error and a decision that ends in error are
    read the same way, so a route that fails both ways is recorded with one
    code."""
    if isinstance(error, LlmError):
        return error.failure
    return LlmFailure(message=str(error) or type(error).__name__, code="UNKNOWN")


async def _normalized(
    source: AsyncIterator[StreamChunk], request: GenerateOptions, by: Running
) -> AsyncIterator[StreamChunk]:
    """Turn an adapter raise into a terminal `finish{error}`.

    The contract of the loop is that a stream always ends with a finish. Without
    this function the loop would need a second failure path, and
    `agent/request-error` would not see provider failures uniformly.

    **`by` is entered around each pull, and not around construction** (C10). The
    body of an adapter is row code that this seam drives, so it runs as an
    effect of the row that registered it (P6-25). The body of a generator runs
    at `__anext__`, on whatever task consumes it. So the binding wraps the
    *await*, and not the `__anext__()` call, which only builds the coroutine.
    To wrap that call would be the same mistake one level down. The binding is
    entered and left for each chunk, and not held across the `yield`, because a
    binding that is held across a suspension is released on whichever task
    resumes it. `_hooks` returns a list and not a generator for the same reason.

    **This uses `running`, and not the inline `_ACTIVATING.set/reset` that
    `_invoke` uses.** The difference between the two is a layer and not a
    preference. `_invoke` is inside cordis and may touch its own ContextVar. To
    reach for that ContextVar from here would be a private import across two
    packages, to save a measured 291 ns per chunk. That is 0.3 ms on a stream of
    a thousand chunks, against a stream that takes seconds. The note of
    `_invoke` about the frame of a context manager is about a path with no such
    boundary to cross.
    """
    pull = source.__anext__
    try:
        while True:
            with running(by):
                try:
                    chunk = await pull()
                except StopAsyncIteration:
                    return
            yield chunk
    except Exception as error:
        failure = _failure_of(error)
        log.debug("ph.llm: adapter for %s failed: %s", request.provider, failure.message)
        yield Finish(reason=FinishReason(kind="error", failure=failure))


@plugin("llm", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the model adapter seam."""
    ctx.provide(LLM, LlmRuntime(ctx=ctx))
