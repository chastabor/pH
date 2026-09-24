"""`ctx.approval` — asking a human, and failing closed when you cannot.

The pipeline turns an `ask` decision into exactly one of four outcomes, and only
`allowed-once` proceeds (B3). The other three are distinct on purpose: a model
that is told "the user rejected this" can re-plan, while one told "there is no
approval channel" knows the deployment is misconfigured rather than that a human
said no. Collapsing them would make a missing UI look like a decision.

**Fail closed** is the whole design. No answerer, an unmounted seam, a canceled
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pydantic import Field

from ..agent.types import AgentHandle
from ..cancel import Cancellation, is_canceled
from ..cordis import Context, Disposer, events, plugin
from ..json import JsonObject, JsonValue, as_str
from ..keys import APPROVAL
from ..session import (
    Claim,
    IntentKind,
    IntentNotDurable,
    Session,
    SessionEvent,
    Unsettled,
    declare_intent,
    intents_of,
    open_intents,
)
from ..wire import WireModel, literal_lookup

__all__ = [
    "APPROVAL_ASK",
    "APPROVAL_OUTCOMES",
    "DENIAL_REASONS",
    "INTERRUPTED",
    "ApprovalAnswer",
    "ApprovalDecisionName",
    "ApprovalOutcome",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalService",
    "Edited",
    "PendingApproval",
    "Responded",
    "answer_from_wire",
    "answer_kind",
    "answer_to_wire",
    "apply",
    "denial_reason",
    "pending_approvals",
]

log = logging.getLogger("ph.seams.approval")

ApprovalOutcome: TypeAlias = Literal[
    "allowed-once", "rejected", "canceled", "unavailable", "interrupted"
]
"""The four answers that carry no data. Only `allowed-once` proceeds (B3)."""

INTERRUPTED: ApprovalOutcome = "interrupted"
"""The process died while a person was being asked (P5-13).

**Written only by repair, never by an answerer**, which is what separates it from
the four beside it. Those four say what happened when the question was *put*:
somebody allowed it, somebody refused, the work was canceled, or nobody could
be asked. This one says the question was never resolved at all, because the
process holding it stopped existing — and it is recorded on resume so that
`pending_approvals` stops reporting a question no one can answer.

Not `canceled`, which claims somebody stopped the work; not `unavailable`,
which is the live answer when no front end takes the prompt and is a *denial* a
turn continues from. Naming it apart is the point: a person reading a transcript
can tell "I was asked and the daemon died" from "I was asked and said no".
"""

APPROVAL_OUTCOMES: Mapping[str, ApprovalOutcome] = literal_lookup(ApprovalOutcome)
"""Every `ApprovalOutcome` by its own spelling — the read-side check. See
`literal_lookup`.

Named for its alias rather than the bare `OUTCOMES` it started as, because
`ph.seams.goals` declares one too and `permission_presets` imports from both
seams — which is the collision the `as_int` rename existed to stop
repeating."""

ApprovalPolicy: TypeAlias = Literal["ask", "never"]

DENIAL_REASONS: dict[str, str] = {
    "rejected": "the user rejected {subject}",
    "canceled": "approval for {subject} was canceled",
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


def denial_reason(outcome: object, subject: str) -> str:
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
refused by a dismissal — and `canceled`/`unavailable` are failures rather than
decisions."""


@dataclass(frozen=True, slots=True)
class Edited:
    """Run it, but with these arguments instead (P4-05).

    The human corrected the call rather than refusing it — a path or a flag was
    wrong, and stopping the turn to say so costs a round trip that changing it
    does not.

    **Every version ends up in the log, attributed.** The assistant message
    holds what the *model* asked for; the substitution is recorded here, on
    `approval/decided`; and `tool/call`, written after the gate (P7-15), records
    what ran. A reader sees the request, the correction, and who made it — where
    rewriting one record in place would have put the human's arguments in the
    model's mouth, which is the falsehood this codebase refuses everywhere else.
    """

    arguments: JsonValue
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


def answer_from_wire(raw: object) -> ApprovalAnswer:
    """One answer as it arrives from a front end that is not in this process.

    `answer_kind`'s inverse, and here rather than in whatever transport happens
    to need it first: the four bare outcomes and the two that carry data are this
    seam's vocabulary, so a second decoder in `ph_app` would be a second opinion
    about what `{"kind": "edited"}` means.

    **Anything unrecognized is `unavailable`**, never a denial and never a pass.
    The outcomes are distinct on purpose — a model told "the user rejected this"
    can re-plan, one told "there is no approval channel" knows the deployment is
    misconfigured — and a garbled frame is the second of those, not the first.
    """
    if isinstance(raw, str):
        return APPROVAL_OUTCOMES.get(raw, "unavailable")
    if isinstance(raw, dict):
        kind = raw.get("kind")
        if kind == "edited":
            return Edited(arguments=raw.get("arguments"))
        if kind == "responded":
            return Responded(message=as_str(raw.get("message")))
    return "unavailable"


def answer_to_wire(answer: ApprovalAnswer) -> str | dict[str, JsonValue]:
    """One answer on its way *out* of the process that decided it.

    `answer_from_wire`'s inverse, and the pair has to be a pair: four of the six
    answers are bare strings and travel as themselves, but `Edited` and
    `Responded` carry data and are frozen dataclasses — so a front end that put
    one straight into a JSON frame produced `TypeError: Object of type Responded
    is not JSON serializable`, and P4-05's whole point (answer in the tool's
    voice rather than refusing) simply could not cross a socket.

    Beside its inverse rather than at the transport, for the reason stated
    there: what `{"kind": "edited"}` means is this seam's to say once.
    """
    if isinstance(answer, str):
        return answer
    if isinstance(answer, Edited):
        return {"kind": answer.kind, "arguments": answer.arguments}
    return {"kind": answer.kind, "message": answer.message}


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

    `Any` and not `JsonValue` like `Edited.arguments` below, though it is the
    same value: `JsonValue` is a *recursive* alias, and pydantic builds a
    validator from a model field where a dataclass never evaluates its
    annotation — so this field spelled that way sends schema generation into
    unbounded recursion and every `ApprovalRequest(...)` raises.

    **Deliberately not recorded on `approval/asked`.** The assistant message
    already holds them; a second copy in the log is two statements of one fact
    that can disagree, and this one is here to be *shown*, not stored."""


@dataclass(frozen=True, slots=True)
class PendingApproval:
    """An `asked` with no `decided` — a question a resume must put back."""

    seq: int
    tool_name: str
    call_id: str | None
    reason: str | None


def pending_approvals(events: Sequence[SessionEvent]) -> list[PendingApproval]:
    """Approvals this log asked and never recorded an answer for.

    Derived, not tracked: the log is the pending state, so a crash between the
    two events cannot lose the question.

    **Events rather than a `Session`**, so `ph.persistence.repair` can call it
    with the raw sequence it is handed. Repair is the one caller that has no
    session — it runs while the log is being rebuilt — and it needs *this* rule,
    not a copy of it: it writes the `approval/decided` that makes a pending ask
    stop being pending, so a second spelling of the key here is a second
    spelling of what repair must settle.
    """
    return [
        PendingApproval(
            seq=intent.opened.seq,
            tool_name=as_str(intent.opened.data.get("toolName")),
            call_id=_str_or_none(intent.opened.data.get("callId")),
            reason=_str_or_none(intent.opened.data.get("reason")),
        )
        for intent in open_intents(events, APPROVAL_ASK)
    ]


def _str_or_none(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _ask_key(event: SessionEvent) -> str:
    """An ask is keyed by its call id, or by the tool's name when it has none —
    the rule both halves of the pair have always been read by."""
    return as_str(event.data.get("callId") or event.data.get("toolName"))


def _closed(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The decision nobody made, as it has always been written.

    `not-started` is the barrier failing: the ask could not be written, so nobody
    was asked, and the live answer for that is `unavailable` — a denial the turn
    continues from. `outcome-unknown` is repair's: the process died while a
    person may have been deciding, which is `INTERRUPTED`, `automatic` because
    no person made it — the flag a policy of `never` sets for the same reason.
    """
    request = ApprovalRequest(
        tool_name=as_str(opened.data.get("toolName")),
        call_id=_str_or_none(opened.data.get("callId")),
    )
    if why == "not-started":
        return ApprovalService._decided_data(request, "unavailable", automatic=False)
    return ApprovalService._decided_data(request, INTERRUPTED, automatic=True)


APPROVAL_ASK = declare_intent(
    IntentKind(
        opened="approval/asked",
        settled="approval/decided",
        opened_key=_ask_key,
        settled_key=_ask_key,
        # A person may have decided on a screen whose answer never reached the
        # log: the question's outcome is what is unknown.
        orphan="outcome-unknown",
        # On disk before anybody is asked (F8).
        barrier="durable",
        closer=_closed,
        owner="ph.seams.approval",
        # Keyed by tool name when there is no call id, so one key is asked many
        # times in a session; each ask is its own.
        dedupe=False,
    )
)
"""An approval: asked, then decided — by a person, a policy, or repair (P10-09)."""


def approval_policy(session: Session) -> ApprovalPolicy:
    """The policy in force: the last `approval/policy` event, else `ask`.

    **Not enforced: who wrote it** — the same statement as `SandboxSeam.logged_mode`,
    for the same reason. The writers are checked where they are shipped, not at the
    append.
    """
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
        return self.ctx.owner_for(scope).on("approval/request", answerer)

    async def request(
        self,
        *,
        agent: AgentHandle,
        tool_name: str,
        call_id: str | None = None,
        reason: str | None = None,
        cancel: Cancellation | None = None,
        allowed_decisions: tuple[ApprovalDecisionName, ...] = (),
        arguments: JsonValue = None,
    ) -> ApprovalAnswer:
        """Ask, record both halves, and return the outcome.

        Raises only what the wait itself raises — a cancellation, or a
        `KeyboardInterrupt` — and records the pair closed before it does (K7).
        It answers rather than raising for everything it can answer for.

        **`cancel` narrows; it does not enable.** The floor is the agent's own
        token, defaulted below, because "a question is not put to a person about
        work they already stopped" is a property of asking rather than of any one
        caller remembering to say so — and three callers did not. Pass it only to
        supply something *narrower*: `ToolRuntime` passes `execution.signal`,
        which a Code Mode cell narrows to a child that settles with the cell
        while the agent keeps running.
        """
        session = agent.session
        request = ApprovalRequest(
            tool_name=tool_name,
            call_id=call_id,
            reason=reason,
            agent_id=agent.id,
            allowed_decisions=list(allowed_decisions),
            arguments=arguments,
        )
        if session is not None and approval_policy(session) == "never":
            # A deployment that turned prompting off has answered in advance;
            # the decision is still recorded so the log says why.
            self._record_asked(session, request)
            self._record_decided(session, request, "rejected", automatic=True)
            return "rejected"

        journal = intents_of(self.ctx)
        held: Claim | None = None
        if session is not None:
            # **On disk before anybody is asked** (F8) — the kind's barrier. The
            # wait may be hours, and the last flush was before the model request,
            # so an ask held only in memory was lost with the `assistant/message`
            # whose call it gates. A log that cannot be written is not a question
            # worth putting to a person: nobody would be able to see it was asked.
            # The journal closes that pair `unavailable` itself, and does the same
            # if the write is canceled.
            try:
                opened = await journal.open(session, APPROVAL_ASK, self._asked_data(request))
            except IntentNotDurable:
                return "unavailable"
            if not isinstance(opened, Claim):
                raise RuntimeError("APPROVAL_ASK does not dedupe, so no ask has a prior")
            held = opened

        # **The pair closes even when the wait does not return** (K7).
        # `_route` checks the token on the way *in*, so `"canceled"` was reachable
        # only for a caller that was already cancelled — a cancel arriving while
        # the person is being asked unwound straight past the record, leaving an
        # `approval/asked` with nothing answering it. The crash repair then stamps
        # it `interrupted` on the next open, which is a different and less true
        # story than "somebody pressed escape": the run was not interrupted, the
        # question was.
        #
        # Stated once, in a `finally`, rather than in two arms that have to agree:
        # the sentinel is the answer for *every* way out that is not `_route`
        # returning one, including a `return` added between here and the record
        # later. Only the log is touched on the way through, and `append` is
        # synchronous so it completes inside a cancelled scope.
        outcome: ApprovalAnswer = "canceled"
        try:
            outcome = await self._route(request, cancel if cancel is not None else agent.signal)
            return outcome
        finally:
            if session is not None and held is not None:
                journal.settle(session, held, self._decided_data(request, outcome, automatic=False))

    async def _route(self, request: ApprovalRequest, cancel: Cancellation | None) -> ApprovalAnswer:
        if is_canceled(cancel):
            return "canceled"

        # `ApprovalAnswer`, not `ApprovalOutcome`: `waterfall` reads the chain's
        # type from here, and an answerer may return an `Edited` or a `Responded`
        # as well as a bare outcome. Declaring the default's own narrower type
        # would make the branch below statically dead for every answerer.
        async def inner(_request: ApprovalRequest) -> ApprovalAnswer:
            # No answerer took the prompt. Absence is not consent.
            return "unavailable"

        try:
            outcome = await self.ctx.waterfall("approval/request", request, inner=inner)
        except Exception:
            log.exception("ph.seams.approval: an answerer failed; denying")
            return "unavailable"
        # `waterfall` carries the chain's type, but its last step is a `cast`: a
        # listener arrives through an entry point and no checker sees what it
        # returns, so this is where the claim is checked. Through the lookup for
        # the same reason as everywhere else — `literal_lookup` says why.
        # `str(outcome)` rather than an `isinstance` guard: the only job that
        # guard had was keeping an unhashable value out of `.get`, and the next
        # line already `%r`s the same object.
        answer: ApprovalAnswer | None = (
            outcome
            if isinstance(outcome, (Edited, Responded))
            else APPROVAL_OUTCOMES.get(str(outcome))
        )
        if answer is None:
            log.error("ph.seams.approval: answerer returned %r; denying", outcome)
            return "unavailable"
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
        the assistant message holds them; two statements of one fact are two that
        can disagree.
        """
        session.append("approval/asked", self._asked_data(request))

    @staticmethod
    def _asked_data(request: ApprovalRequest) -> dict[str, Any]:
        data: dict[str, Any] = {"toolName": request.tool_name}
        if request.call_id is not None:
            data["callId"] = request.call_id
        if request.reason is not None:
            data["reason"] = request.reason
        if request.agent_id is not None:
            data["agentId"] = request.agent_id
        if request.allowed_decisions:
            data["allowedDecisions"] = list(request.allowed_decisions)
        return data

    def _record_decided(
        self,
        session: Session,
        request: ApprovalRequest,
        outcome: ApprovalAnswer,
        *,
        automatic: bool,
    ) -> None:
        session.append(
            "approval/decided", self._decided_data(request, outcome, automatic=automatic)
        )

    @staticmethod
    def _decided_data(
        request: ApprovalRequest, outcome: ApprovalAnswer, *, automatic: bool
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"toolName": request.tool_name, "outcome": answer_kind(outcome)}
        if isinstance(outcome, Edited):
            # The substitution itself: the assistant message holds what the model
            # asked for, and `tool/call` will record what ran (P7-15).
            data["arguments"] = outcome.arguments
        elif isinstance(outcome, Responded):
            data["message"] = outcome.message
        if request.call_id is not None:
            data["callId"] = request.call_id
        if automatic:
            data["automatic"] = True
        return data

    def set_policy(self, session: Session, policy: ApprovalPolicy) -> None:
        """Record a policy change. The last one recorded is the one in force."""
        session.append("approval/policy", {"policy": policy})


@plugin("approval")
async def apply(ctx: Context, config: None) -> None:
    """Mount the approval seam."""
    ctx.provide(APPROVAL, ApprovalService(ctx=ctx))
