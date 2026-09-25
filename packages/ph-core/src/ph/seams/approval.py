"""`ctx.approval` — asking a human, and failing closed when you cannot.

The pipeline turns an `ask` decision into exactly one of four outcomes, and only
`allowed-once` proceeds (B3). The other three are distinct on purpose: a model
that is told "the user rejected this" can re-plan, while one told "there is no
approval channel" knows the deployment is misconfigured rather than that a human
said no. Collapsing them would make a missing UI look like a decision.

**Fail closed** is the whole design. No answerer, an unmounted seam, a canceled
prompt, an exception inside an answerer — every one of them denies. A permission
system whose failure mode is "allow" is not a permission system.

**An ask a crash left open is settled, not re-asked**: `approval/asked` without a
matching `approval/decided` is `APPROVAL_ASK` open in the log, and repair closes
it `interrupted` on resume (P5-13), since the turn that was waiting on it is gone.

@module ph.seams.approval
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from pydantic import Field

from ..agent.types import AgentHandle
from ..cancel import Cancellation, is_canceled
from ..cordis import Context, Disposer, events, plugin
from ..json import JsonObject, JsonValue, as_str
from ..keys import APPROVAL
from ..session import Claim, IntentNotDurable, Session, intents_of
from ..session.kinds import APPROVAL_ASK, INTERRUPTED, approval_decided
from ..session.writers import log_writer
from ..wire import WireModel, literal_lookup

_LOG = log_writer(__name__)

__all__ = [
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
    "Responded",
    "answer_from_wire",
    "answer_kind",
    "answer_to_wire",
    "apply",
    "denial_reason",
]

log = logging.getLogger("ph.seams.approval")

ApprovalOutcome: TypeAlias = Literal[
    "allowed-once", "rejected", "canceled", "unavailable", "interrupted"
]
"""The four answers that carry no data. Only `allowed-once` proceeds (B3)."""

# `INTERRUPTED` — the answer only repair writes — is declared with the kind in
# `ph.session.kinds` (T4), and re-exported here with the seam's vocabulary.

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


def approval_policy(session: Session) -> ApprovalPolicy:
    """The policy in force: the last `approval/policy` event, else `ask`.

    **Who wrote it is this seam** — the same statement as `SandboxSeam.logged_mode`:
    `approval/policy` is written only through this module's writer (T6), and what is
    not enforced is the same deliberate bypass.
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
            # Asked and decided together, through the journal (T6): one batch, so
            # the pair lands whole.
            intents_of(self.ctx).open_settled(
                session,
                APPROVAL_ASK,
                self._asked_data(request),
                lambda asked: self._decided_data(
                    request, "rejected", automatic=True, ask_seq=asked.seq
                ),
            )
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
                held = await journal.open(session, APPROVAL_ASK, self._asked_data(request))
            except IntentNotDurable:
                return "unavailable"

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
                journal.settle(
                    session,
                    held,
                    self._decided_data(request, outcome, automatic=False, ask_seq=held.opened.seq),
                )

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

    @staticmethod
    def _asked_data(request: ApprovalRequest) -> dict[str, Any]:
        """The ask, as the log keeps it.

        Built field by field rather than by subtracting `arguments` from
        `to_wire()`, and symmetric with `_decided_data` for the reason: the
        request is the *answerer's* view and will grow fields for its benefit — a
        diff preview, a risk label — and every one of them would otherwise land
        in the log by default, silently, with the filter needing to be remembered
        at each new serialization site. The arguments themselves stay out because
        the assistant message holds them; two statements of one fact are two that
        can disagree.
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
        return data

    @staticmethod
    def _decided_data(
        request: ApprovalRequest, outcome: ApprovalAnswer, *, automatic: bool, ask_seq: int
    ) -> dict[str, Any]:
        """The decision, built by the kind's own builder — the one repair's closer
        calls too, so the two cannot drift."""
        answer: JsonObject | None = None
        if isinstance(outcome, Edited):
            # The substitution itself: the assistant message holds what the model
            # asked for, and `tool/call` will record what ran (P7-15).
            answer = {"arguments": outcome.arguments}
        elif isinstance(outcome, Responded):
            answer = {"message": outcome.message}
        return approval_decided(
            tool_name=request.tool_name,
            call_id=request.call_id,
            outcome=answer_kind(outcome),
            ask_seq=ask_seq,
            automatic=automatic,
            answer=answer,
        )

    def set_policy(self, session: Session, policy: ApprovalPolicy) -> None:
        """Record a policy change. The last one recorded is the one in force."""
        _LOG.append(session, "approval/policy", {"policy": policy})


@plugin("approval")
async def apply(ctx: Context, config: None) -> None:
    """Mount the approval seam."""
    ctx.provide(APPROVAL, ApprovalService(ctx=ctx))
