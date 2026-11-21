"""Agent-facing vocabulary: cancellation causes, decisions, turn endings, and
the payloads that the agent waterfalls carry.

The waterfall payloads are frozen dataclasses and not string-keyed dicts, so
that the signature of a listener *is* the contract. The limits and compaction
plugins (Phase 4) that listen on `agent/pre-step`, and the retry plugin on
`agent/request-error`, read fields that the type checker knows about.
`phern events` can also name the payload beside the event.

@module ph.agent.types
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from ..cancel import Cancellation
from ..llm.types import LlmCallConfig, LlmFailure, Message, Sampling
from ..wire import WireDataclass

if TYPE_CHECKING:
    from ..cordis import Context
    from ..session import Session
    from .inbox import Inbox

__all__ = [
    "AgentCancelCause",
    "AgentDriver",
    "AgentHandle",
    "AgentOptions",
    "AgentStatus",
    "PreStepDecision",
    "PreStepRequest",
    "RequestErrorAction",
    "RequestFailure",
    "RequestProposal",
    "TurnEndReason",
]

AgentStatus: TypeAlias = Literal["idle", "running"]


class AgentHandle(Protocol):
    """What a seam, a tool body or a waterfall payload may assume about an agent.

    The handle is six read-only facts, and deliberately no more. This is the
    surface that the seams read. It is also the surface that
    `ph.testing.StubAgent` has, so a test can stand in an agent without a loop.
    It is a Protocol and not the driver class for two reasons that support each
    other. Two implementations exist (the loop and the stub). And the driver
    imports this module, so naming the driver here would be an import cycle.

    **`signal` is the sixth fact, and it is here because two seams already
    reached for it.** `AgentDriver.cancel` is the verb, and the token that it
    trips is the fact. A row that *asks a human* has to know whether the work
    is still wanted before it puts a question on the screen of a person.
    `ph_stabilize.permissions_fs` and the RLM harness both wrote
    `getattr(agent, "signal", None)` for it. No agent had the attribute, so the
    default answered every call.

    The type is `Cancellation` and not `CancelToken`: a seam may ask whether
    the work is still wanted, and may not end it. It is not optional, because
    every driver owns one. `_Phase.token` is built at construction, and it is
    what `session/cancel` trips, through the verb that the TUI and RPC share.

    **`status` is a fact and not a verb**, which is why it is here and not on
    `AgentDriver` beside `cancel`. It was on the driver, so a seam that held a
    handle could not ask whether the agent was busy. `compaction` asked with
    `getattr(agent, "status", "idle")`. That default answered *idle* for every
    stub, in the one place that refuses to compact a working agent.

    The members are properties and not attributes, so that a driver whose
    `session` is a `Session` satisfies a reader that accepts `Session | None`.
    A Protocol attribute is settable and therefore invariant, and it would
    refuse exactly the implementation that this describes.
    """

    @property
    def id(self) -> str: ...
    @property
    def ctx(self) -> Context: ...
    @property
    def session(self) -> Session | None: ...
    @property
    def options(self) -> AgentOptions: ...
    @property
    def status(self) -> AgentStatus: ...
    @property
    def signal(self) -> Cancellation: ...


class AgentDriver(AgentHandle, Protocol):
    """What the registry, the daemon and a supervising row may *do* to an agent.

    This is the handle plus the verbs. `AgentRegistry.create` returns this
    surface, and the supervisor of `ph_app` drives it. It is separate from
    `AgentHandle` because most callers have no reason to steer. A seam that
    could reach `cancel` through a parameter that is typed for reading will
    eventually do so.
    """

    @property
    def inbox(self) -> Inbox: ...
    def steer(self, message: Message) -> None: ...
    def inject(self, message: Message) -> None: ...
    def followup(self, message: Message) -> None: ...
    def interject(self, message: Message) -> None: ...
    def cancel(self, cause: AgentCancelCause, *, keep_inbox: bool = False) -> None: ...
    def reroute(self, options: AgentOptions) -> None: ...
    async def run(self) -> None: ...
    async def prompt(self, text: str) -> None: ...
    async def dispose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentCancelCause(WireDataclass):
    """Why an active driver was canceled: the user stopped it (`user`), its parent
    released it (`parent`), or it was disposed (`disposed`)."""

    kind: Literal["user", "parent", "disposed"]


@dataclass(frozen=True, slots=True)
class TurnEndReason(WireDataclass):
    """Why a turn ended.

    `error` always carries structured facts and not a flattened string. A retry
    policy and a later reader both route on the code.
    """

    kind: Literal["completed", "aborted", "blocked", "error", "max-tokens", "interrupted"]
    reason: AgentCancelCause | None = None
    """Who stopped an `aborted` turn, when the `cancel` of the loop was told. Absent
    for a stop from outside the loop, whose author the loop cannot see."""
    detail: str | None = None
    """What that stop from outside said of itself, in place of `reason`: the reason
    of a token or of a scope, or else the type of the exception. It is kept for a
    reader and for telemetry. Nothing routes on it."""
    error: LlmFailure | None = None


@dataclass(frozen=True, slots=True)
class PreStepRequest:
    """`agent/pre-step`: the batch that is about to enter a step, before the decision.

    **`session` is not optional here**, though `AgentHandle.session` is. A
    pre-step is a step that is about to happen, so there is a turn, so there is
    a session. The one production constructor is the driver, whose own
    `session` is a `Session`. The handle keeps its `| None` for the callers
    that hold one outside a turn, and for `StubAgent`. Three listeners on this
    chain each derived the session again from the handle, and each guarded the
    `None` differently.
    """

    agent: AgentHandle
    session: Session
    messages: tuple[Message, ...]
    turn: int
    step: int


@dataclass(frozen=True, slots=True)
class PreStepDecision:
    """The authoritative decision about whether a step happens, and with what.

    `reject` closes the turn as `blocked`. `enter` supplies the exact messages
    that the step will log. A limits or permissions plugin owns this decision
    by returning without calling `next()`.
    """

    kind: Literal["reject", "enter"]
    messages: tuple[Message, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RequestProposal:
    """`agent/request`: the call config the loop proposes for one request."""

    agent: AgentHandle
    session: Session
    """Not optional here, for the reason that `PreStepRequest.session` gives (L5).

    `input-offload` is the listener that had reached through the handle. It
    paid the same price that `llm-retry` did: a `None` that its only producer
    cannot create, guarded by an early return. The feature therefore turned
    itself off and did not fail."""
    turn: int
    step: int
    config: LlmCallConfig


@dataclass(frozen=True, slots=True)
class RequestFailure:
    """`agent/request-error`: a request that ended in `error` or `aborted`."""

    agent: AgentHandle
    session: Session
    """Not optional here, for the reason that `PreStepRequest.session` gives (L5).

    When `llm-retry` reached through the handle instead, it had to guard a
    `None` that its only producer cannot create. That guard does not fail
    loudly: it logs at debug and declines the retry. A route that had retries
    configured therefore had none, and nothing said so."""
    turn: int
    step: int
    provider: str
    failure: LlmFailure
    retries_by: Mapping[str, int]
    """How many times the loop has already retried this step, by the row that asked (G13).

    The key is the plugin whose listener granted each retry, as
    `Context.waterfall_attributed` reports it. It is `""` for a listener
    outside any plugin. The driver counts, because every retry passes through
    it. The count is handed to the policy and not derived again there. A row
    can therefore budget against its own retries (`retries_by["llm-retry"]`) or
    against all of them (`retries`) by choosing a number, and need not fold a
    log that only some rows write.

    **The count is per step and per agent, by construction.** It is a local
    variable in the retry loop of the driver, so two subagents at the same
    coordinates cannot share it (I4). A count that was held on the listener had
    that bug. Nothing is lost on a resume either, because a resumed session
    starts a new turn and never enters again the step that it was retrying."""

    @property
    def retries(self) -> int:
        """Every retry this step has had, whichever row asked."""
        return sum(self.retries_by.values())


@dataclass(frozen=True, slots=True)
class RequestErrorAction:
    """What to do about a failed model request: retry, or let it stand."""

    kind: Literal["retry"]
    delay_ms: int = 0
    """How long the loop waits before the retry. The loop sleeps, after the
    waterfall has settled, and the row never does. A listener that refuses the
    retry therefore costs no wait."""
    granted: Callable[[], None] | None = None
    """What the row that granted this retry records of it. The loop runs it when the
    retry is final, for the same reason as `delay_ms`: a later row may still
    refuse the retry, and a record made at the time of the grant would name a
    retry that never ran."""


@dataclass(frozen=True, slots=True)
class AgentOptions:
    """Per-agent settings resolved at creation."""

    provider: str = ""
    model: str = ""
    max_tokens: int | None = None
    sampling: Sampling = field(default_factory=Sampling)
    reasoning_effort: str | None = None
    model_key: str = ""
    """The name that the session has for this route (`ph.seams.models`), or `""`
    for a route that the session does not list. It is carried with the route and
    not beside it, so whoever moves the agent moves both. It never goes on the
    wire to a provider: `seed_config` leaves it out."""

    def seed_config(self) -> LlmCallConfig:
        return LlmCallConfig(
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            sampling=self.sampling,
            max_tokens=self.max_tokens,
        )
