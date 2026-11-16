"""`ctx.llm` — the model adapter seam.

Definition, Provider, Consumer (invariant I5). The *definition* is
`LlmAdapter`; a *provider* is any plugin calling `ctx.llm.register_adapter`;
the *consumer* is the loop, which never learns which adapter answered.

Every call goes through the `llm/stream` waterfall, which is where retry,
replay and session-title attach, then `llm/send`, where a barrier waits. An adapter that
raises is normalized into a terminal `finish{error}` chunk before any consumer
sees it, so a consumer never handles two shapes for the same failure.

A route may also *decide* (`DecisionAdapter`): typed questions about a state,
answered with probabilities rather than text. `ctx.llm.decide` is that door.

@module ph.llm.adapter
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

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
"""A decision model answered questions on a route for a session's agent: the
route, how many questions, and the provider's `usage` when it reports one. A
record that spends tokens (`ph.seams.token_meter.SPENDING_TYPES`)."""

DECISION_FAILED = "llm/decision-failed"
"""A route would not decide: the route, how many questions, and the failure's
`code` and `reason`."""

events.declare(
    "llm/stream",
    "waterfall",
    GenerateOptions,
    owner="ph.llm",
    doc="Wraps every model call. Retry, replay and recording attach here.",
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

    Carries the provider's own facts rather than a flattened string, because
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
    """MIME types this route takes as message *content* (P7-01).

    Empty means text only, which is every route until an adapter says otherwise —
    the safe default, since the failure it prevents is a media block reaching a
    wire that silently drops it. A caller that finds a block outside this set
    degrades it to a text pointer and logs a notice; it never sends it and hopes,
    and never fails the turn, because a session begun on a vision model and
    resumed on a text one must still open."""
    max_attachment_bytes: int | None = None
    """Per-attachment ceiling, when the route publishes one.

    A limit and not just `accepts`, because it is what lets an over-sized block
    degrade to a pointer here rather than be rejected at the provider."""
    max_image_edge: int | None = None
    """Longest edge in pixels the route will *accept*, when it publishes one.

    This docstring's neighbor used to say a pixel ceiling had to wait for
    `media-transform` (P7-02) because "declaring one now would be a knob wired to
    nothing". Two things wired it (P7-03): `ph.llm.dimensions` reads an image's
    size from its header with no dependency, and a limit does not need a resizer
    to be worth declaring — refusing with a sentence the model can read is an
    action, and it is the same one `max_attachment_bytes` already takes."""
    structured_output: bool = False
    """Whether this route enforces `GenerateOptions.response_schema` on the wire.

    Declared rather than assumed, and defaulting to *no*, for the reason `accepts`
    does: the failure it prevents is a caller believing a shape was guaranteed
    when the provider ignored the field. OpenAI-compatible endpoints take
    `response_format: {"type": "json_schema", …}`; Anthropic has no equivalent
    today, so a caller there gets the instruction and the validation and not the
    guarantee — which is a real difference and one `structured` says out loud
    rather than papering over."""
    usable_image_edge: int | None = None
    """Longest edge the route can actually *use*, when it is smaller than it accepts.

    Providers scale a large image down before the model sees it — Anthropic at
    1568 px, and it accepts up to 8000 — so pixels above this are uploaded, paid
    for, and discarded at the far end. Separate from `max_image_edge` because the
    outcomes differ in kind: over that, nothing is sent; over this, everything
    works and a person is quietly overpaying, which is the failure that lasts for
    a whole session because nothing announces it."""
    sampling: frozenset[SamplingSetting] | None = None
    """The sampling settings this route's API takes (A1), or `None` when its adapter has
    not said.

    A setting outside it is dropped before the request reaches the adapter, with a
    warning once per route and setting, and the `models` row names it in `phern
    doctor`: graceful degradation, since a route that runs without a recommended
    setting still works, where one refused would not. `None` drops and warns about
    nothing, since a guess would either warn falsely or send a field a strict server
    refuses."""
    credential: str | None = None
    """The credential this route resolves at the adapter edge, **by name** (I-3) —
    the `apiKeyEnv` its row names — or `None` for a route that needs none.

    What lets anything above the edge ask "can this route run here, now?" without
    holding a value: the resume check (T5) reads it and asks `ctx.credentials.has`,
    so a session whose key is missing waits for it by name rather than starting and
    failing at its first request. Checked against the mounted profile's adapters,
    never a name copied into the log, because a profile that renamed the variable
    must be asked about the new name."""
    max_decision_tokens: int | None = None
    """The most tokens one decision on this route may carry, where its row says.

    A fact about the server rather than the model, and one a llama.cpp does not
    publish: it reads a decision in one physical batch, so its limit is
    `--ubatch-size` — 512 unless it was started with more — and not its context
    window. `None` for a route that says nothing, and for every route that does
    not decide. `LlmRuntime.decision_limit` is what an asker reads, since a
    refusal can teach less than this says."""


class MediaRoute(Protocol):
    """The route facts a config declares and `ResolvedModel` carries.

    Read-only members on purpose: the three configs disagree about the *types* —
    Anthropic states a pixel edge as `int` where the other two allow `None` — and
    a property is covariant where a mutable attribute would not be, so each may
    narrow its own field and still satisfy this.

    Declared rather than duck-typed, which is what makes `resolved` below more
    than a shortcut: a config that grows out of step with `ResolvedModel` fails at
    the call site instead of quietly reporting a default.

    **Beside `ResolvedModel` rather than beside the adapters that satisfy it.**
    Its member list *is* that dataclass's field list, so the two have to move
    together — and living in an application package's private module would have
    meant a third-party adapter importing `ph_app.adapters._media` or writing the
    seventh copy of the projection, which is the drift this exists to stop. It is
    also the only side of the boundary where the invariant can be checked at all:
    a rename here and a matching edit there cannot pass one suite and fail the
    other.
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
) -> ResolvedModel:
    """One route's config as the `ResolvedModel` every layer above reads.

    **The projection, not the values.** What a route accepts, how large a file it
    takes and what it does with pixels are per-provider facts that stay in each
    config with the paragraph arguing them; what was copied three times is the
    *field list* — six lines mapping one name onto the same name — and that list
    is `ResolvedModel`'s vocabulary rather than any adapter's.

    The cost of the copies was a seventh capability being three edits, where an
    adapter that missed one reports the default instead of the route's number and
    `media-degrade` then applies the wrong rule for that provider alone. Now it is
    one edit here, and a config that has not caught up fails to type-check.

    `structured_output` is passed rather than read, because it is the one field
    that is not a config value: it is a claim about the *wire*, argued at each
    call site, and a route that declared it in config could promise a guarantee
    its adapter does not implement. `credential` is passed for the plainer reason
    that it is not a media fact: it is the name the adapter resolves at its edge,
    and **required** so an adapter cannot forget to say (T5). `sampling` is the
    wire's too, and required for the same reason. So is `max_decision_tokens`,
    which is config but not a media fact: an adapter that does not decide says
    `None`, and one that does cannot drop its route's limit unnoticed.
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
    )


class LlmAdapter(Protocol):
    """The one required method is `stream`; `resolve_model` is optional."""

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]: ...


@dataclass(frozen=True, slots=True)
class Noul:
    """A yes-or-no question, answered with how likely *yes* is (`NoulAnswer`).

    The three kinds of question carry the decision API's own names — `noul`,
    `choice`, `score` — and so do their fields, so a reader of the wire and a
    reader of this module are reading one vocabulary."""

    instructions: str


@dataclass(frozen=True, slots=True)
class Choice:
    """A question answered with one of its `criteria` (`ChoiceAnswer`)."""

    instructions: str
    criteria: Mapping[str, str]
    """Each option's key, and what choosing it means; `""` for a key that says
    it. Two at least, in the order a tie is broken in."""


@dataclass(frozen=True, slots=True)
class Score:
    """A question answered with a place on a scale (`ScoreAnswer`)."""

    instructions: str
    criteria: tuple[str, ...]
    """What each level means, lowest first. Two to ten."""


Question: TypeAlias = Noul | Choice | Score
"""One thing a decision model is asked about a state."""


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Questions about one state, on one route.

    Not a conversation: `state` is what is judged, as text, and each question is
    answered by itself, so one answer never colors another and several are one
    request."""

    provider: str
    model: str
    state: str
    questions: Mapping[str, Question]
    """Each question under the name its answer comes back under."""


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """A `Noul`'s answer."""

    noul: float
    """How likely *yes* is, from 0 to 1. A probability, never a verdict: where
    to draw the line is the asker's rule."""


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """A `Choice`'s answer."""

    choice: str
    """The likeliest option's key."""
    probabilities: Mapping[str, float]
    """Each option's share, adding up to 1, in the order the criteria were given."""
    confidence: float | None = None
    """How far from an even split the shares are, from 0 to 1, when the
    provider says. Not a measure of how often it is right."""


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A `Score`'s answer."""

    score: float
    """The expected level, counted from 0 at the lowest; it can fall between two."""
    probabilities: tuple[float, ...]
    """Each level's share, lowest first. The wire keys them `"0"`, `"1"`, …; the
    order is the key, and the question's `criteria` are the legend."""
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
    """One answer a question, under the question's name.

    Each is the kind its question is owed, which `LlmRuntime.decide` holds every
    adapter to: a caller that asked a `Noul` reads a `NoulAnswer`."""

    answers: Mapping[str, Answer]
    input_tokens: int | None = None
    """What the provider says the request cost, when it says. A decision model
    scores and writes nothing, so there are no output tokens to report."""

    def answer[A: Answer](self, name: str, kind: type[A]) -> A:
        """The answer to `name`, as the `kind` its question was owed — the reader
        that lets a caller skip narrowing what `decide` already checked.

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

    Its own Protocol beside `LlmAdapter`, and asked with `isinstance`, never a
    `getattr` probe — the rule this tree states for every optional capability:
    a probe reports an adapter whose method is misnamed as one that cannot. An
    adapter for a provider with no decision API implements none of this and is
    simply not an instance.

    The credential is resolved inside `decide`, at the adapter's edge, as it is
    for `stream`: a caller hands a state and questions in and gets answers
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
"""How embeddings and decision servers say an input did not fit. llama.cpp:
*"input (1602 tokens) is too large to process"*, with a 500; OpenAI: *"maximum
context length is 8192 tokens"*, with a 400. Phrases, because the status says
nothing either way."""


def input_too_long(body: str) -> bool:
    """Whether an embeddings or decision server refused a request because an
    input did not fit.

    The one list, for every client of such an API, because a refusal one of them
    reads as "too long" and another as a plain failure sends the same server's
    answer down two roads: one sends a smaller input and goes on, the other ends
    the run. Classified as `CONTEXT_WINDOW_EXCEEDED`, which is what a caller can
    act on."""
    lowered = body.lower()
    return any(phrase in lowered for phrase in _TOO_LONG)


_SENT = (r"input \((\d+) tokens\)", r"requested (\d+) tokens", r"resulted in (\d+) tokens")
_READS = (r"batch size: (\d+)", r"maximum context length is (\d+)")


def too_long_facts(body: str) -> tuple[int | None, int | None]:
    """What a refusal `input_too_long` recognizes says in numbers: how many
    tokens the input was, and how many the server takes. Either may be missing.

    A llama.cpp says both — *"input (702 tokens) is too large to process …
    (current batch size: 512)"* — and so does OpenAI. They are worth reading
    because they are exact where a caller's own count is an estimate: the first
    says how far over the input was, so the next one is aimed rather than
    halved, and the second is the server's real limit, which is not always the
    one it reports when asked. Beside `_TOO_LONG` because both read the same
    servers' prose; a refusal in other words simply has no numbers."""

    def found(patterns: tuple[str, ...]) -> int | None:
        for pattern in patterns:
            match = re.search(pattern, body)
            if match is not None:
                return int(match.group(1))
        return None

    return found(_SENT), found(_READS)


def failure_of(message: str, code: str, *, said: str = "", status: int | None = None) -> LlmFailure:
    """A provider's failure, `message` as it is shown. A refusal as too long
    carries the two numbers it gave (`too_long_facts`) as fields, read here once
    from all of what the provider `said` — the message, unless a builder cut it
    down for showing — so no caller reads them again out of a shortened one."""
    sent, reads = (
        too_long_facts(said or message) if code == CONTEXT_WINDOW_EXCEEDED else (None, None)
    )
    return LlmFailure(
        message=message, code=code, status=status, sent_tokens=sent, limit_tokens=reads
    )


def refused_share(sent: int | None, reads: int | None) -> float | None:
    """The share of a refused input the server reads, by the two numbers its
    refusal gave (`too_long_facts`), or `None` when it did not give both or gave
    a limit the input was within. What a caller aims its next, shorter input by,
    a margin under."""
    return reads / sent if sent and reads and sent > reads else None


@dataclass(slots=True)
class AdapterHandle:
    """One adapter registration, and the routes it claims."""

    adapter: LlmAdapter
    by: Running
    """Who registered it (P6-29). An adapter's `stream` is row code this registry
    invokes, and it ran unbound — the same category as a tool's `execute`.

    Missed by P6-29 because both P6-30 walks look for a `Callable`: an adapter is
    an *object satisfying a Protocol*, so `AdapterHandle.adapter: LlmAdapter`
    names none, and this registers without a `claim_*` helper at all. Resolved
    from the ambient binding rather than a `scope=`, because `register_adapter`
    has never taken one — a row calling it from its own `apply` is the only
    caller shape there is, and that is exactly what `running_for(None)` reads."""
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
    """Each warning about a withheld setting already given, so a session warns once
    rather than on every request."""
    _decision_limits: dict[tuple[str, str], int] = field(default_factory=dict)
    """The most tokens a route was found to read in one decision, by route, as
    its refusals said (`too_long_facts`). Kept so the next asker sizes its state
    to fit rather than spending a refusal to learn it again; only ever lowered,
    and kept for the life of the process — a server restarted with a bigger
    batch is read at the smaller one until then, which shows an asker less and
    fails nothing."""

    def register_adapter(self, providers: Sequence[str], adapter: LlmAdapter) -> AdapterHandle:
        """Claim one or more provider routes for `adapter`.

        The caller's scope owns the handle's `dispose`, so unloading the plugin
        unregisters the routes.
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

    def adapter_for(self, provider: str) -> LlmAdapter:
        return self._route(provider).adapter

    def _route(self, provider: str) -> AdapterHandle:
        """The whole registration, for the two paths that call into the adapter.

        `adapter_for` stays the public spelling — one out-of-package caller reads
        it — and the binding needs what it discards, which is who registered the
        thing it returns.
        """
        handle = self._routes.get(provider)
        if handle is None:
            raise LlmError(f'no adapter is registered for provider "{provider}"', "NO_ADAPTER")
        return handle

    def decision_limit(self, provider: str, model: str) -> int | None:
        """The most tokens one decision on this route may carry, where anything
        says: the route's own `max_decision_tokens`, or what a refusal taught,
        whichever is less. `None` when neither says, and the asker assumes."""
        said = (
            self.resolve_model(provider, model).max_decision_tokens,
            self._decision_limits.get((provider, model)),
        )
        return min((one for one in said if one is not None), default=None)

    def decides(self, provider: str) -> bool:
        """Whether the adapter registered for `provider` can decide. `False` for a
        provider nothing is registered for, too: there is no adapter to ask."""
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

        Run as the row that registered the adapter, like `stream`. Refused with a
        code when the route has no adapter (`NO_ADAPTER`), its adapter cannot
        decide (`NO_DECISIONS`), or the answers are not the ones the questions
        are owed (`BAD_DECISION`), each naming the provider.

        **`timeout_s` is how long the asker will wait**, and a route that takes
        longer is refused as `TIMEOUT`. Here rather than around the call, so a
        decision given up on is recorded like any other that failed: whoever
        asks is usually holding something up until the answer comes — a turn
        about to end — and an adapter's own patience is a model call's.

        **`session` is whose spend it is.** A decision is not a turn's request,
        so nothing else in a log says a route was used: given a session, what
        the request cost is recorded in it (`DECIDED`), where a budget and a
        child's account read every other spend, and so is a request the route
        refused or failed (`DECISION_FAILED`), with why. Without one the caller
        is deciding for nobody's account, and nothing is recorded. What was
        *asked* and *answered* is the asker's to record: it knows what the
        answer was for."""
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
        """Keep the limit a refusal names (`decision_limit`): a route that said
        its input was too long, and said how long a one it reads."""
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
        resolver: Callable[..., Any] | None = getattr(handle.adapter, "resolve_model", None)
        if resolver is None:
            return ResolvedModel()
        with running(handle.by):
            resolved = resolver(provider, model)
        return resolved if isinstance(resolved, ResolvedModel) else ResolvedModel()

    def withhold(
        self, config: LlmCallConfig, route: ResolvedModel, efforts: Collection[str]
    ) -> tuple[LlmCallConfig, dict[CallSetting, float | int | str]]:
        """`config` as `route` can carry it, and what it withholds, by the values asked:
        a sampling setting its API does not take (`Sampling.uncarried`), and a reasoning effort
        its profile's `efforts` do not define (`unmapped_effort`). Each with a warning the
        first time, since someone asked for it.

        The agent loop's, before it logs the header of the request it sends, so the log
        says what was sent and what was asked and not (`EpochHeader.withheld`). A request
        that reaches the seam another way is held to the sampling half at the edge
        (`_carried`), and sends an effort only as the `effort_fields` it is given."""
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
        """`request` as its route can carry it: a sampling setting its API does not take
        is dropped, with a warning the first time (`ResolvedModel.sampling`)."""
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
        """What of `sampling` a route carrying `carried` cannot send, by the value asked,
        each warned about the first time: both drop paths' one step."""
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
            # calls, so this line only constructs them — nothing of the adapter
            # ran here, and the whole stream used to be pulled later, outside any
            # `running`. Every registration the adapter made mid-stream landed on
            # the seam instead of the row, and outlived it.
            return _normalized(handle.adapter.stream(request), request, handle.by)

        return await self.ctx.waterfall("llm/stream", options, inner=inner)


def _owed(request: DecisionRequest, result: Decisions) -> None:
    """Refuse `result` unless every question has the kind of answer it is owed.

    Here rather than in each adapter, so what a caller may assume of `Decisions`
    is one rule: an answer read as a likelihood that was a choice would be a
    decision made on a number nobody gave."""
    for name, question in request.questions.items():
        try:
            result.answer(name, _ANSWERS[type(question)])
        except LlmError as error:
            raise LlmError(f'provider "{request.provider}": {error}', BAD_DECISION) from None


def _failure_of(error: Exception) -> LlmFailure:
    """What an adapter's raise says, as a failure: an `LlmError`'s own, and any
    other exception's message under `UNKNOWN`. One reading for a stream that
    ends in error and a decision that does, so a route that fails both ways is
    recorded with one code."""
    if isinstance(error, LlmError):
        return error.failure
    return LlmFailure(message=str(error) or type(error).__name__, code="UNKNOWN")


async def _normalized(
    source: AsyncIterator[StreamChunk], request: GenerateOptions, by: Running
) -> AsyncIterator[StreamChunk]:
    """Turn an adapter raise into a terminal `finish{error}`.

    The loop's contract is that a stream always ends with a finish. Without this
    the loop would need a second failure path, and `agent/request-error` would
    not see provider failures uniformly.

    **`by` is entered around each pull rather than around construction** (C10).
    An adapter's body is row code this seam drives, so it runs as an effect of
    the row that registered it (P6-25) — and a generator's body runs at
    `__anext__`, on whatever task is consuming it. The binding therefore wraps
    the *await*, not the `__anext__()` call that merely builds the coroutine,
    which is the same mistake one level down. Entered and left per chunk rather
    than held across the `yield`, because a binding held across a suspension is
    released on whichever task resumes it — `_hooks`' reason for returning a
    list rather than a generator.

    **`running` and not the inline `_ACTIVATING.set/reset` `_invoke` uses**, and
    the difference between the two is a layer rather than a preference. That one
    is inside cordis and may touch its own ContextVar; reaching for it from here
    would be a private import across two packages to save a measured 291 ns per
    chunk — 0.3 ms on a thousand-chunk stream, against a stream that takes
    seconds. `_invoke`'s note about the context manager's own frame is about a
    path with no such boundary to cross.
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
