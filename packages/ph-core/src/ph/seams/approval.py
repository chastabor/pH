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

from ..cancel import CancelToken, is_cancelled
from ..cordis import Context, Disposer, events, plugin
from ..session import Session
from ..wire import WireModel

__all__ = [
    "ApprovalOutcome",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalService",
    "PendingApproval",
    "apply",
    "pending_approvals",
]

log = logging.getLogger("ph.seams.approval")

ApprovalOutcome: TypeAlias = Literal["allowed-once", "rejected", "cancelled", "unavailable"]
ApprovalPolicy: TypeAlias = Literal["ask", "never"]

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
    ) -> ApprovalOutcome:
        """Ask, record both halves, and return the outcome. Never raises."""
        session: Session | None = getattr(agent, "session", None)
        request = ApprovalRequest(
            tool_name=tool_name,
            call_id=call_id,
            reason=reason,
            agent_id=getattr(agent, "id", None),
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

    async def _route(self, request: ApprovalRequest, cancel: CancelToken | None) -> ApprovalOutcome:
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
        if outcome in get_args(ApprovalOutcome):
            return cast(ApprovalOutcome, outcome)
        log.error("ph.seams.approval: answerer returned %r; denying", outcome)
        return "unavailable"

    def _record_asked(self, session: Session, request: ApprovalRequest) -> None:
        session.append("approval/asked", request.to_wire())

    def _record_decided(
        self,
        session: Session,
        request: ApprovalRequest,
        outcome: ApprovalOutcome,
        *,
        automatic: bool,
    ) -> None:
        data: dict[str, Any] = {"toolName": request.tool_name, "outcome": outcome}
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
