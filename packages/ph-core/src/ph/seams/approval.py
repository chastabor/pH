"""`ctx.approval`: asking a human, and failing closed when nobody can be asked.

The pipeline turns an `ask` decision into one answer. Only two answers proceed
(B3): `allowed-once`, and `Edited`, which is the call with the arguments that
the user edited. `Responded` runs nothing, and the model gets the text of the
user instead. The three refusals are distinct on purpose. A model that is told
"the user rejected this" can re-plan. A model that is told "there is no
approval channel" knows that the deployment is misconfigured, and not that a
human said no. One shared refusal would make a missing UI look like a decision.

**Fail closed** is the design. No answerer, an unmounted seam, a canceled
prompt, an exception inside an answerer: each of them denies. A permission
system that allows when it fails is not a permission system.

**An ask that a crash left open is settled, not asked again.** An
`approval/asked` record without a matching `approval/decided` is an open
`APPROVAL_ASK` in the log. Repair closes it as `interrupted` on resume (P5-13),
because the turn that waited for it is gone.

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
from ..session import (
    Claim,
    IntentNotDurable,
    Session,
    SessionBatch,
    intents_of,
    session_written,
)
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
"""The answers that carry no data. Of these, only `allowed-once` proceeds (B3), and
only repair writes `interrupted`."""

# `INTERRUPTED` is the answer that only repair writes. It is declared with the
# kind in `ph.session.kinds` (T4) and exported again here, with the vocabulary
# of the seam.

APPROVAL_OUTCOMES: Mapping[str, ApprovalOutcome] = literal_lookup(ApprovalOutcome)
"""Every `ApprovalOutcome` by its own spelling. This is the check on the read
side. See `literal_lookup`.

The name carries its alias. It started as the bare `OUTCOMES`, and
`ph.seams.goals` declares a lookup of outcomes too. A module that imports from
both seams would then hold two names that collide. That is the collision that
the `as_int` rename was made to stop from repeating."""

ApprovalPolicy: TypeAlias = Literal["ask", "never"]

DENIAL_REASONS: dict[str, str] = {
    "rejected": "the user rejected {subject}",
    "canceled": "approval for {subject} was canceled",
    "unavailable": "{subject} requires approval, but no approval channel is available",
}
"""What a consumer tells the model when an ask did not grant.

The table has three sentences and not one, and that distinction is the reason
why the table exists. A model that is told "the user rejected this" can
re-plan. A model that is told "there is no approval channel" knows that the
deployment is misconfigured, and not that a human said no. One shared sentence
makes a missing UI look like a decision.

The table is here and not in each consumer, because three callers ask through
this seam: the tools pipeline, `permissions-fs`, and the harness service of
`ph-rlm`. The tools pipeline and `permissions-fs` build their refusal from this
table. The harness service only tests for `allowed-once` and raises a refusal
of its own. The *first* copy that disagreed was the copy that collapsed all
four outcomes into one string. `{subject}` comes from the caller, because a
tool call, a path and a refinement have different names, and only the caller
knows which one it holds."""


def denial_reason(outcome: object, subject: str) -> str:
    """The sentence for one outcome that did not grant. An unknown answer reads
    as a missing channel (`unavailable`), which is the fail-closed direction
    and the correct one."""
    template = DENIAL_REASONS.get(str(outcome), DENIAL_REASONS["unavailable"])
    return template.format(subject=subject)


ApprovalDecisionName: TypeAlias = Literal["approve", "edit", "reject", "respond"]
"""The four things that a human may *do* about a prompt, as a row and a front
end name them.

The vocabulary is closed, so it is a `Literal`. `ApprovalOutcome`, `CardKind`,
`ReadingLevel` and `PresetName` already follow the same rule. It is distinct
from `ApprovalOutcome`, which names what an answerer *returned*. `approve` is
the button and `allowed-once` is the verdict. The two are the same decision,
seen from the two sides of the prompt."""

_ANSWER_DECISIONS: dict[str, ApprovalDecisionName] = {
    "allowed-once": "approve",
    "edited": "edit",
    "responded": "respond",
}
"""The answers that a restricted ask has to check. `rejected` is absent because
a refusal is always available: a row that withheld every button would still be
refused by a dismissal. `canceled` and `unavailable` are failures and not
decisions."""


@dataclass(frozen=True, slots=True)
class Edited:
    """Run it, but with these arguments instead (P4-05).

    The human corrected the call and did not refuse it. A path or a flag was
    wrong. To stop the turn and say so costs a round trip, and to change the
    argument does not.

    **Every version ends up in the log, attributed.** The assistant message
    holds what the *model* asked for. The substitution is recorded here, on
    `approval/decided`. `tool/call`, which is written after the gate (P7-15),
    records what ran. A reader sees the request, the correction, and who made
    the correction. A rewrite of one record in place would have attributed the
    arguments of the human to the model, and this codebase refuses that kind
    of false record everywhere else.
    """

    arguments: JsonValue
    kind: Literal["edited"] = "edited"


@dataclass(frozen=True, slots=True)
class Responded:
    """Do not run it. Tell the model this instead (P4-05).

    This is the answer to "why are you calling that?". The human replies in
    place of the tool. The body never runs, and the model reads a *successful*
    result and not a refusal that it has to interpret. A rejection says no.
    This answer says what to do instead, in the one place where the model
    already looks.
    """

    message: str
    kind: Literal["responded"] = "responded"


ApprovalAnswer: TypeAlias = "ApprovalOutcome | Edited | Responded"
"""What an answerer may return.

The bare outcomes stay strings, so the fail-closed reading is unchanged and
every existing answerer keeps working. The two answers that carry data are
objects, because they have data to carry."""


def _proceeds(answer: ApprovalAnswer) -> bool:
    """Whether the caller acts on `answer`: it runs the call, as asked or as
    edited. A `Responded` runs nothing, and the model gets the text of the user
    instead."""
    return answer == "allowed-once" or isinstance(answer, Edited)


def answer_from_wire(raw: object) -> ApprovalAnswer:
    """One answer as it arrives from a front end that is not in this process.

    This is the inverse of `answer_to_wire`. It is here and not in whichever
    transport needs it first, because the bare outcomes and the two answers
    that carry data are the vocabulary of this seam. A second decoder in
    `ph_app` would be a second opinion about what `{"kind": "edited"}` means.

    **Anything unrecognized is `unavailable`**, never a rejection by the user
    and never a pass. The outcomes are distinct on purpose. A model that is
    told "the user rejected this" can re-plan, and a model that is told "there
    is no approval channel" knows that the deployment is misconfigured. A
    garbled frame is the second case, not the first.
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

    This is the inverse of `answer_from_wire`, and the two must stay a pair. A
    bare outcome is a string and travels as itself. `Edited` and `Responded`
    carry data and are frozen dataclasses. A front end that put one straight
    into a JSON frame got `TypeError: Object of type Responded is not JSON
    serializable`. So the purpose of P4-05, an answer in place of the tool and
    not a refusal, could not cross a socket.

    The function is beside its inverse and not at the transport, for the reason
    that `answer_from_wire` gives: this seam says one time what
    `{"kind": "edited"}` means.
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
    """What the asking row accepts. Empty means all four.

    This is a `list` and not the `tuple` that the config uses, because this
    field is appended to the log. `freeze_json_value` refuses a tuple outright
    (A1: a tuple would come back as a list). A wire model that cannot be logged
    fails at its first append, in the session of a user and not here."""
    arguments: Any = None
    """The call as it stands, so that a human can correct it and not only
    refuse it.

    The type is `Any` and not `JsonValue`, although `Edited.arguments` holds
    the same value as a `JsonValue`. `JsonValue` is a *recursive* alias.
    Pydantic builds a validator from the field of a model, where a dataclass
    never evaluates its annotation. With `JsonValue` here, schema generation
    recurses without bound and every `ApprovalRequest(...)` raises.

    **Deliberately not recorded on `approval/asked`.** The assistant message
    already holds the arguments. A second copy in the log would be two
    statements of one fact, and the two can disagree. This copy is here to be
    *shown*, not stored."""


def approval_policy(session: Session) -> ApprovalPolicy:
    """The policy in force: the last `approval/policy` event, or `ask` when the
    log holds none.

    **Who wrote it is this seam**, as the docstring of `SandboxSeam.logged_mode`
    says for `sandbox/mode`. `approval/policy` is written only through the
    writer of this module (T6). The bypasses that are not prevented at run time
    are the same deliberate ones.
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
        """Claim the right to answer prompts. A front end registers one of these.

        This is sugar over `ctx.on("approval/request", answerer)`. There is one
        routing mechanism, the waterfall, and this method is only a name for it
        that is easy to find. The answerer receives `(request, next_)` and
        returns an `ApprovalAnswer`.
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

        The method raises only what the wait itself raises: a cancellation, or
        a `KeyboardInterrupt`. It records the pair as closed before it raises
        (K7). For everything that it can answer, it answers and does not raise.

        **`cancel` narrows and does not enable.** The floor is the token of the
        agent itself, which is the default below. "A question is not put to a
        person about work they already stopped" is a property of asking. It
        must not depend on each caller remembering to say so, and three callers
        did not. Pass `cancel` only to supply something *narrower*.
        `ToolRuntime` passes `execution.signal`, which a Code Mode cell narrows
        to a child that settles with the cell while the agent keeps running.
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
            # A deployment that turned prompting off has answered in advance.
            # The decision is still recorded, so the log says why. The ask and
            # the decision go through the journal together (T6), as one batch,
            # so the pair is written whole.
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
            # **On disk before anybody is asked** (F8). This is the barrier of
            # the kind. The wait may last hours, and the last flush was before
            # the model request. An ask that was held only in memory was lost
            # together with the `assistant/message` whose call it gates. When
            # the log cannot be written, the question is not worth putting to a
            # user, because nobody could see that it was asked. The journal
            # closes that pair as `unavailable` itself, and it does the same
            # when the write is canceled.
            try:
                held = await journal.open(session, APPROVAL_ASK, self._asked_data(request))
            except IntentNotDurable:
                return "unavailable"

        # **The pair closes even when the wait does not return** (K7).
        # `_route` checks the token on the way *in*. So `"canceled"` was
        # reachable only for a caller that was already canceled. A cancel that
        # arrived while the user was being asked unwound past the record and
        # left an `approval/asked` with no answer. The crash repair then marks
        # it `interrupted` on the next open. That is a different and less true
        # account than "somebody pressed escape": the run was not interrupted,
        # the question was.
        #
        # The rule is stated one time, in a `finally`, and not in two arms that
        # have to agree. The sentinel is the answer for *every* way out that is
        # not `_route` returning an answer, including a `return` that somebody
        # adds later between here and the record. Only the log is touched on
        # the way through, and `append` is synchronous, so it completes inside
        # a canceled scope.
        outcome: ApprovalAnswer = "canceled"
        try:
            outcome = await self._route(request, cancel if cancel is not None else agent.signal)
        finally:
            if session is not None and held is not None:
                journal.settle(
                    session,
                    held,
                    self._decided_data(request, outcome, automatic=False, ask_seq=held.opened.seq),
                )
        # **A decision that lets something happen is on disk before it does**
        # (S8). The ask was on disk, and the decision was only in memory. The
        # barrier of a tool call flushed the decision for an ask that the
        # pipeline makes. But an ask from *inside* a body, such as
        # `permissions-fs` gating a write, is acted on the moment this method
        # returns, and nothing there had to remember a barrier of its own. A
        # host that died next left a write that happened, beside a log that
        # says after repair that nobody approved it.
        # This is fail-closed, as the barrier of the ask is: a decision that
        # cannot be written to disk allows nothing.
        if (
            held is not None
            and session is not None
            and _proceeds(outcome)
            and not await session_written(self.ctx, session)
        ):
            return "unavailable"
        return outcome

    async def _route(self, request: ApprovalRequest, cancel: Cancellation | None) -> ApprovalAnswer:
        if is_canceled(cancel):
            return "canceled"

        # The type is `ApprovalAnswer` and not `ApprovalOutcome`. `waterfall`
        # reads the type of the chain from here, and an answerer may return an
        # `Edited` or a `Responded` as well as a bare outcome. If this function
        # declared the narrower type of the default itself, the branch below
        # would be statically dead for every answerer.
        async def inner(_request: ApprovalRequest) -> ApprovalAnswer:
            # No answerer took the prompt. Absence is not consent.
            return "unavailable"

        try:
            outcome = await self.ctx.waterfall("approval/request", request, inner=inner)
        except Exception:
            log.exception("ph.seams.approval: an answerer failed; denying")
            return "unavailable"
        # `waterfall` carries the type of the chain, but its last step is a
        # `cast`. A listener arrives through an entry point, and no checker
        # sees what it returns, so the claim is checked here. The check goes
        # through the lookup for the same reason as everywhere else, which
        # `literal_lookup` gives. It uses `str(outcome)` and not an
        # `isinstance` guard. The only job of that guard was to keep an
        # unhashable value out of `.get`, and the error line below already
        # formats the same object with `%r`.
        answer: ApprovalAnswer | None = (
            outcome
            if isinstance(outcome, (Edited, Responded))
            else APPROVAL_OUTCOMES.get(str(outcome))
        )
        if answer is None:
            log.error("ph.seams.approval: answerer returned %r; denying", outcome)
            return "unavailable"
        # What *may* be decided is the policy of the asking row. The seam owns
        # the fail-closed reading, so the seam has to enforce that policy. If
        # only the front end enforced it, a second answerer (an RPC answerer,
        # the answerer of a test) could return an `Edited` for a tool whose
        # arguments the row said must not be written by hand.
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

        The data is built field by field, and not by removing `arguments` from
        `to_wire()`. `_decided_data` is built the same way, for the same reason.
        The request is the view of the *answerer*, and it will grow fields for
        the answerer: a diff preview, a risk label. Each such field would
        otherwise reach the log by default and in silence, and somebody would
        have to remember the filter at each new place that serializes the
        request. The arguments themselves stay out because the assistant
        message holds them. Two statements of one fact are two that can
        disagree.
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
        """The decision, built by the builder of the kind (`approval_decided`).
        The closer of repair calls the same builder, so the two cannot drift."""
        answer: JsonObject | None = None
        if isinstance(outcome, Edited):
            # The substitution itself. The assistant message holds what the
            # model asked for, and `tool/call` will record what ran (P7-15).
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

    def set_policy(self, log: Session | SessionBatch, policy: ApprovalPolicy) -> None:
        """Record a policy change. The last one recorded is the one in force."""
        _LOG.append(log, "approval/policy", {"policy": policy})


@plugin("approval", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the approval seam."""
    ctx.provide(APPROVAL, ApprovalService(ctx=ctx))
