"""Agent-facing vocabulary: cancellation causes, decisions, turn endings, and
the payloads the agent waterfalls carry.

The waterfall payloads are frozen dataclasses rather than string-keyed dicts so
a listener's signature *is* the contract: the limits and permissions plugins
(Phase 4) that own `agent/pre-step`, and the retry plugin on
`agent/request-error`, read fields the type checker knows about, and
`ph events` can name the payload beside the event.

@module ph.agent.types
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ..llm.types import LlmCallConfig, LlmFailure, Message
from ..wire import WireDataclass

__all__ = [
    "AgentCancelCause",
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


@dataclass(frozen=True, slots=True)
class AgentCancelCause(WireDataclass):
    """Why an active driver was cancelled."""

    kind: Literal["user", "parent", "hook", "disposed", "legacy"]
    reason: str | None = None
    """Set only for `hook`: which listener objected, and why."""


@dataclass(frozen=True, slots=True)
class TurnEndReason(WireDataclass):
    """Why a turn ended.

    `error` always carries structured facts rather than a flattened string: the
    code is what a retry policy and a later reader both route on.
    """

    kind: Literal["completed", "aborted", "blocked", "error", "max-tokens", "interrupted"]
    reason: AgentCancelCause | None = None
    error: LlmFailure | None = None


@dataclass(frozen=True, slots=True)
class PreStepRequest:
    """`agent/pre-step`: the batch about to enter a step, before the decision."""

    agent: Any
    messages: tuple[Message, ...]
    turn: int
    step: int


@dataclass(frozen=True, slots=True)
class PreStepDecision:
    """The authoritative decision about whether a step happens, and with what.

    `reject` closes the turn as `blocked`; `enter` supplies the exact messages
    the step will log. A limits or permissions plugin owns this decision by
    returning without calling `next()`.
    """

    kind: Literal["reject", "enter"]
    messages: tuple[Message, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RequestProposal:
    """`agent/request`: the call config the loop proposes for one request."""

    agent: Any
    turn: int
    step: int
    config: LlmCallConfig


@dataclass(frozen=True, slots=True)
class RequestFailure:
    """`agent/request-error`: a request that ended in `error` or `aborted`."""

    agent: Any
    turn: int
    step: int
    provider: str
    failure: LlmFailure


@dataclass(frozen=True, slots=True)
class RequestErrorAction:
    """What to do about a failed model request: retry, or let it stand."""

    kind: Literal["retry"]
    delay_ms: int = 0


@dataclass(frozen=True, slots=True)
class AgentOptions:
    """Per-agent settings resolved at creation."""

    provider: str = ""
    model: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None

    def seed_config(self) -> LlmCallConfig:
        return LlmCallConfig(
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
