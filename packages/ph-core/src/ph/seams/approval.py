"""`ctx.approval` — asking a human, and failing closed when you cannot.

The pipeline turns an `ask` decision into exactly one of four outcomes, and only
`allowed-once` proceeds (B3). The other three are distinct on purpose: a model
that is told "the user rejected this" can re-plan, while one told "there is no
approval channel" knows the deployment is misconfigured rather than that a human
said no. Collapsing them would make a missing UI look like a decision.

**Fail closed** is the whole design. No answerer, an unmounted seam, a cancelled
prompt, an exception inside an answerer — every one of them denies. A permission
system whose failure mode is "allow" is not a permission system.

**Re-asking on resume** falls out of the log rather than being remembered:
`approval/asked` without a matching `approval/decided` *is* the pending state, so
a crash between the two leaves a question a resumed session can find and put
back to the human.

@module ph.seams.approval
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast, get_args

from pydantic import Field

from ..cancel import CancelToken, is_cancelled
from ..cordis import Context, Disposer, events, plugin
from ..session import Session
from ..wire import WireModel

__all__ = [
    "DENIAL_REASONS",
    "ApprovalAnswer",
    "ApprovalDecisionName",
    "ApprovalOutcome",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalService",
    "Edited",
    "PendingApproval",
    "Responded",
    "answer_kind",
    "apply",
    "denial_reason",
    "pending_approvals",
]

log = logging.getLogger("ph.seams.approval")

ApprovalOutcome: TypeAlias = Literal["allowed-once", "rejected", "cancelled", "unavailable"]
"""The four answers that carry no data. Only `allowed-once` proceeds (B3)."""

ApprovalPolicy: TypeAlias = Literal["ask", "never"]

DENIAL_REASONS: dict[str, str] = {
    "rejected": "the user rejected {subject}",
    "cancelled": "approval for {subject} was cancelled",
    "unavailable": "{subject} requires approval, but no approval channel is available",
}
"""What a consumer tells the model when an ask did not grant.

Three sentences rather than one, and that distinction is the reason this table
exists at all: a model told "the user rejected this" can re-plan, while one told
"there is no approval channel" knows the deployment is misconfigured rather than
that a human said no. Collapsing them makes a missing UI look like a decision.

Here rather than in each consumer because there are three of them — the tools
pipeline, `permissions-fs`, and the RLM harness service — and the *first* copy to
disagree was the one that collapsed all four outcomes into one string. `{subject}`
is the caller's, since a tool call, a path and a refinement are named differently
and only the caller knows which it is holding."""


def denial_reason(outcome: Any, subject: str) -> str:
    """The sentence for one non-grant outcome. Unknown answers read as absence,
    which is the fail-closed direction and the honest one."""
    template = DENIAL_REASONS.get(str(outcome), DENIAL_REASONS["unavailable"])
    return template.format(subject=subject)


ApprovalDecisionName: TypeAlias = Literal["approve", "edit", "reject", "respond"]
"""The four things a human may *do* about a prompt, as a row and a front end
name them.

A closed vocabulary, so it is a `Literal` — the rule `ApprovalOutcome`,
`CardKind`, `ReadingLevel` and `PresetName` are already held to. Distinct from
`ApprovalOutcome`, which names what an answerer *returned*: `approve` is the
button, `allowed-once` is the verdict, and the two are the same decision seen
from either side of the prompt."""

_ANSWER_DECISIONS: dict[str, ApprovalDecisionName] = {
    "allowed-once": "approve",
    "edited": "edit",
    "responded": "respond",
}
"""Which answers a restricted ask has to check. `rejected` is absent because
refusing is always available — a row that withheld every button would still be
refused by a dismissal — and `cancelled`/`unavailable` are failures rather than
decisions."""


@dataclass(frozen=True, slots=True)
class Edited:
    """Run it, but with these arguments instead (P4-05).

    The human corrected the call rather than refusing it — a path or a flag was
    wrong, and stopping the turn to say so costs a round trip that changing it
    does not.

    **Both versions end up in the log, and they have to.** `tool/call` is
    appended before the pipeline runs (B4), so it already records what the
    *model* asked for; the substitution is recorded on `approval/decided`. A
    reader sees the request, the correction, and who made it — where quietly
    rewriting the call's own record would have attributed the human's arguments
    to the model, which is the falsehood this codebase refuses everywhere else.
    """

    arguments: Any
    kind: Literal["edited"] = "edited"


@dataclass(frozen=True, slots=True)
class Responded:
    """Do not run it; tell the model this instead (P4-05).

    The answer to "why are you calling that?" — the human replies in the tool's
    own voice, the body never runs, and the model reads a *successful* result
    rather than a refusal it has to interpret. A rejection says no; this says
    what to do instead, in the one place the model is already looking.
    """

    message: str
    kind: Literal["responded"] = "responded"


ApprovalAnswer: TypeAlias = "ApprovalOutcome | Edited | Responded"
"""What an answerer may return.

The four bare outcomes stay strings so the fail-closed reading is unchanged and
every existing answerer keeps working; the two that carry data are objects
because they have data to carry."""


def answer_kind(answer: ApprovalAnswer) -> str:
    """The one word that names an answer, whichever shape it took."""
    return answer if isinstance(answer, str) else answer.kind


events.declare(
    "approval/request",
    "waterfall",
    owner="ph.seams.approval",
    doc="Routes one approval prompt to an answerer (TUI, RPC). Fails closed.",
)


class ApprovalRequest(WireModel):
    """One prompt, as an answerer receives it."""

    tool_name: str
    call_id: str | None = None
    reason: str | None = None
    agent_id: str | None = None
    allowed_decisions: list[ApprovalDecisionName] = Field(default_factory=list)
    """What the asking row will accept. Empty means all four.

    A `list`, not the `tuple` the config uses, because this model is appended:
    `freeze_json_value` refuses a tuple outright (A1 — "would come back as a
    list"), and a wire model that cannot be logged is one whose first append
    fails in someone's session rather than here."""
    arguments: Any = None
    """The call as it stands, so a human can correct it rather than only refuse.

    **Deliberately not recorded on `approval/asked`.** `tool/call` is appended
    before the pipeline runs (B4) and already holds them; a second copy in the
    log is two statements of one fact that can disagree, and this one is here to
    be *shown*, not stored."""


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """An `asked` with no `decided` — a question a resume must put back."""

    seq: int
    tool_name: str
    call_id: str | None
    reason: str | None


def pending_approvals(session: Session) -> list[PendingApproval]:
    """Approvals this log asked and never recorded an answer for.

    Derived, not tracked: the log is the pending state, so a crash between the
    two events cannot lose the question.
    """
    asked: dict[str, PendingApproval] = {}
    for event in session.events:
        if event.type == "approval/asked":
            key = str(event.data.get("callId") or event.data.get("toolName"))
            asked[key] = PendingApproval(
                seq=event.seq,
                tool_name=str(event.data.get("toolName", "")),
                call_id=event.data.get("callId"),
                reason=event.data.get("reason"),
            )
        elif event.type == "approval/decided":
            asked.pop(str(event.data.get("callId") or event.data.get("toolName")), None)
    return sorted(asked.values(), key=lambda pending: pending.seq)


def approval_policy(session: Session) -> ApprovalPolicy:
    """The policy in force: the last `approval/policy` event, else `ask`."""
    event = session.latest("approval/policy")
    if event is None:
        return "ask"
    return "never" if event.data.get("policy") == "never" else "ask"


@dataclass(slots=True)
class ApprovalService:
    """The service published as `ctx.approval`."""

    ctx: Context

    def register_answerer(
        self, answerer: Callable[..., Any], *, scope: Context | None = None
    ) -> Disposer:
        """Claim the right to answer prompts. A front-end registers one of these.

        Sugar over `ctx.on("approval/request", answerer)`: there is one routing
        mechanism, the waterfall, and this is only a discoverable name for it. The
        answerer receives `(request, next_)` and returns an `ApprovalOutcome`.
        """
        return (scope or self.ctx).on("approval/request", answerer)

    async def request(
        self,
        *,
        agent: Any,
        tool_name: str,
        call_id: str | None = None,
        reason: str | None = None,
        cancel: CancelToken | None = None,
        allowed_decisions: tuple[ApprovalDecisionName, ...] = (),
        arguments: Any = None,
    ) -> ApprovalAnswer:
        """Ask, record both halves, and return the outcome. Never raises."""
        session: Session | None = getattr(agent, "session", None)
        request = ApprovalRequest(
            tool_name=tool_name,
            call_id=call_id,
            reason=reason,
            agent_id=getattr(agent, "id", None),
            allowed_decisions=list(allowed_decisions),
            arguments=arguments,
        )
        if session is not None and approval_policy(session) == "never":
            # A deployment that turned prompting off has answered in advance;
            # the decision is still recorded so the log says why.
            self._record_asked(session, request)
            self._record_decided(session, request, "rejected", automatic=True)
            return "rejected"

        if session is not None:
            self._record_asked(session, request)

        outcome = await self._route(request, cancel)
        if session is not None:
            self._record_decided(session, request, outcome, automatic=False)
        return outcome

    async def _route(self, request: ApprovalRequest, cancel: CancelToken | None) -> ApprovalAnswer:
        if is_cancelled(cancel):
            return "cancelled"

        async def inner(_request: ApprovalRequest) -> ApprovalOutcome:
            # No answerer took the prompt. Absence is not consent.
            return "unavailable"

        try:
            outcome = await self.ctx.waterfall("approval/request", request, inner=inner)
        except Exception:
            log.exception("ph.seams.approval: an answerer failed; denying")
            return "unavailable"
        if not isinstance(outcome, (Edited, Responded)) and outcome not in get_args(
            ApprovalOutcome
        ):
            log.error("ph.seams.approval: answerer returned %r; denying", outcome)
            return "unavailable"
        answer = cast(ApprovalAnswer, outcome)
        # What *may* be decided is the asking row's policy, so the seam that owns
        # the fail-closed reading is the one that has to hold it: enforcing this
        # only in the front end would let a second answerer — an RPC one, a
        # test's — return an `Edited` for a tool whose arguments the row said
        # must not be hand-written.
        decision = _ANSWER_DECISIONS.get(answer_kind(answer))
        if request.allowed_decisions and decision not in (None, *request.allowed_decisions):
            log.error(
                "ph.seams.approval: answerer chose %r, which %r does not allow; denying",
                decision,
                request.tool_name,
            )
            return "unavailable"
        return answer

    def _record_asked(self, session: Session, request: ApprovalRequest) -> None:
        """The ask, as the log keeps it.

        Built field by field rather than by subtracting `arguments` from
        `to_wire()`, and symmetric with `_record_decided` for the reason: the
        request is the *answerer's* view and will grow fields for its benefit — a
        diff preview, a risk label — and every one of them would otherwise land
        in the log by default, silently, with the filter needing to be remembered
        at each new serialization site. The arguments themselves stay out because
        `tool/call` recorded them before the pipeline ran (B4); two statements of
        one fact are two that can disagree.
        """
        data: dict[str, Any] = {"toolName": request.tool_name}
        if request.call_id is not None:
            data["callId"] = request.call_id
        if request.reason is not None:
            data["reason"] = request.reason
        if request.agent_id is not None:
            data["agentId"] = request.agent_id
        if request.allowed_decisions:
            data["allowedDecisions"] = list(request.allowed_decisions)
        session.append("approval/asked", data)

    def _record_decided(
        self,
        session: Session,
        request: ApprovalRequest,
        outcome: ApprovalAnswer,
        *,
        automatic: bool,
    ) -> None:
        data: dict[str, Any] = {"toolName": request.tool_name, "outcome": answer_kind(outcome)}
        if isinstance(outcome, Edited):
            # The substitution itself, because `tool/call` already recorded what
            # the model asked for and the model is about to run something else.
            data["arguments"] = outcome.arguments
        elif isinstance(outcome, Responded):
            data["message"] = outcome.message
        if request.call_id is not None:
            data["callId"] = request.call_id
        if automatic:
            data["automatic"] = True
        session.append("approval/decided", data)

    def set_policy(self, session: Session, policy: ApprovalPolicy) -> None:
        """Record a policy change. The last one recorded is the one in force."""
        session.append("approval/policy", {"policy": policy})


@plugin("approval")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the approval seam."""
    ctx.provide("approval", ApprovalService(ctx=ctx))
