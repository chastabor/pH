"""`ctx.user_questions` — asking the human something that is not an approval.

Distinct from `ctx.approval` because the shapes differ: an approval is a
one-shot yes/no about a *specific pending call* and must fail closed, while a
question is free-form and its failure mode is "no answer", which a caller
handles however it likes. Sharing one seam would force one of those two
behaviors onto the other.

**A question is logged only when it is actually put to a person** (P7-09). That
is the one rule here that is not obvious, and it follows from the failure mode
above rather than from tidiness: since "nobody could answer" resolves instantly,
appending around it would write a question-and-refusal pair into the
log of every unattended run — an `/autonomous` turn inside an interactive
profile, a `phern -p` against a profile that armed the row — for an exchange that
never happened. The log would then say a person was asked and declined, which is
a different and false claim.

So attendance is decided **first**, and an unattended ask appends nothing and
returns at once. A deliverable one appends `question/asked` *before* the
waterfall runs and `question/answered` after, which is §5 rule 2 in the order
approvals use: a crash between them leaves the question in the log with no answer
— `QUESTION_ASK` open — which repair settles on resume.

Reachability is the answerer's own claim, because only the answerer knows. An
in-process front end *is* the person's screen and says so by saying nothing —
the default is reachable. A daemon's ask desk is reachable when some front end
is attached and not otherwise, which is the same distinction `AskDesk` already
makes between watching and answering.

@module ph.seams.user_questions
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cancel import Cancellation, is_canceled
from ..cordis import Context, Disposer, events, plugin, settled_or_none
from ..keys import USER_QUESTIONS
from ..session import IntentNotDurable, Session, intents_of
from ..session.kinds import QUESTION_ASK, AskResolution, question_answered
from ..wire import WireModel
from ._registry import claim_entry

__all__ = [
    "AskOutcome",
    "AskResolution",
    "UserQuestion",
    "UserQuestionService",
    "apply",
]

# `AskResolution` — every way one question can end — is declared with the kind in
# `ph.session.kinds` (T4), since the answered payload carries it, and re-exported here.


@dataclass(frozen=True, slots=True)
class AskOutcome:
    """What became of one question, decided here rather than reconstructed.

    Every distinction this carries is one the seam already made internally and
    then threw away: attendance is checked before anything is recorded,
    cancellation before that, and a raising answerer is caught and logged. The
    caller could not re-derive any of them afterwards, which is the whole reason
    it is a value rather than a `str | None`.
    """

    resolution: AskResolution
    answer: str | None = None
    """The person's words, and only ever set when `resolution` is `answered` —
    so a caller that reads this without checking cannot mistake a refusal for an
    empty answer."""

    def __post_init__(self) -> None:
        # The docstring above is a promise, and one line makes it one: an
        # `AskOutcome("declined", "hi")` would put words in the mouth of
        # somebody who declined.
        if (self.answer is None) == (self.resolution == "answered"):
            raise ValueError(f"an answer belongs to `answered` alone: {self}")


log = logging.getLogger("ph.seams.user_questions")

events.declare(
    "user-question/ask",
    "waterfall",
    owner="ph.seams.user_questions",
    doc="Routes one free-form question to a front-end. Resolves to None unanswered.",
)


class UserQuestion(WireModel):
    """One question put to the human."""

    question: str
    options: list[str] | None = None
    header: str | None = None
    multi_select: bool = False
    ask_id: str | None = None
    """This question's identity, for correlating an answer with the ask.

    Carried on the question rather than passed beside it so that every route to
    an answerer keeps them together: the log's `askId`, the wire frame's, and the
    key a re-posed question is recognized by are then one string by construction
    rather than three that agree while someone remembers to make them.

    `None` for a caller that has no natural key; `ask()` fills it in.
    """


def _always() -> bool:
    return True


@dataclass(slots=True)
class UserQuestionService:
    """The service published as `ctx.user_questions`."""

    ctx: Context
    _reachable: list[Callable[[], bool]] = field(default_factory=list)

    def register_answerer(
        self,
        answerer: Callable[..., Any],
        *,
        scope: Context | None = None,
        reachable: Callable[[], bool] | None = None,
    ) -> Disposer:
        """Sugar over `ctx.on("user-question/ask", answerer)`; one mechanism.

        `reachable` is how an answerer says whether it can reach a person *right
        now*, and defaults to "yes": an in-process front end registers one
        answerer for one screen, so its presence is the answer. A transport that
        may have nobody behind it — a daemon whose front ends have all closed —
        passes a probe, and `attended` consults it at ask time rather than at
        registration, because that is when the answer can have changed.
        """
        owner = self.ctx.owner_for(scope)
        off = owner.on("user-question/ask", answerer)
        # `claim_entry` rather than `append` plus a hand-written `remove`: it is
        # the helper this package grew because that release was written out six
        # times and drifted, and it removes by *identity*, so two answerers with
        # equal probes cannot take each other's disposer.
        forgotten = claim_entry(
            owner,
            self._reachable,
            reachable if reachable is not None else _always,
            label="user-questions(reachable)",
        )

        def dispose() -> None:
            off()
            forgotten()

        return dispose

    @property
    def attended(self) -> bool:
        """Whether some registered answerer says it can reach a person."""
        return any(probe() for probe in self._reachable)

    async def ask(
        self,
        question: UserQuestion,
        *,
        session: Session | None = None,
        cancel: Cancellation | None = None,
    ) -> AskOutcome:
        """Ask, and say what became of the question.

        An `AskOutcome` rather than `str | None` because the four ways to get no
        answer are four different things to tell a model, and this is the only
        place that can tell them apart — see `AskResolution`.

        `cancel` is the caller's cancellation, and a canceled ask is not put to
        anybody — the rule `ApprovalService.request` states for the other seam
        that interrupts a person. Not recorded either: this seam already only
        logs a question that was *delivered*, and an abandoned one never was.

        `session` is optional and its absence means "do not record", which is the
        right default for a caller that is not part of a conversation. When it is
        given, both halves are appended — but only for a question that was
        actually delivered; see the module docstring.

        Identity travels on the question (`UserQuestion.ask_id`) rather than
        beside it, so there is one place a caller can put it and one place every
        route — the log record, the wire frame, a re-posed ask — reads it from.
        """
        if is_canceled(cancel):
            return AskOutcome("canceled")
        if not self.attended:
            return AskOutcome("unattended")
        # Minted only when the caller had no natural key of its own. `ask_user`
        # passes the tool call id, which is the string the rest of the log
        # already joins the exchange by; a counter would restart at 1 after a
        # resume and answer a question the log was still holding open.
        asked = (
            question
            if question.ask_id
            else question.model_copy(update={"ask_id": f"q-{secrets.token_hex(4)}"})
        )
        if session is None:
            return await self._deliver(asked)
        # On disk before it is delivered (F8), which is the whole reason the ask
        # is appended ahead of the waterfall: a crash while somebody was deciding
        # must leave the question open in the log, for repair to settle. The
        # kind's barrier does it; one that cannot be written is not asked, and
        # the journal closes it `failed`.
        #
        # `open` and not `claim`: a question canceled while somebody is looking
        # at it — a root passivated while parked on a person — **stays pending**,
        # so the next resume re-poses it. `claim` would settle it on the way out,
        # and the fold would say a released root had been answered.
        journal = intents_of(self.ctx)
        try:
            held = await journal.open(session, QUESTION_ASK, asked.to_wire())
        except IntentNotDurable:
            return AskOutcome("failed")
        outcome = await self._deliver(asked)
        journal.settle(session, held, self._answered_data(asked, outcome, ask_seq=held.opened.seq))
        return outcome

    async def _deliver(self, asked: UserQuestion) -> AskOutcome:
        async def inner(_question: UserQuestion) -> str | None:
            return None

        try:
            raw = await self.ctx.waterfall("user-question/ask", asked, inner=inner)
            answer = settled_or_none("user-question/ask", raw, str)
        except Exception:
            # An answerer that raises and one that answers the wrong shape are the
            # same failure to this seam, and either way the ask has to be closed
            # in the log — the `asked`/`answered` pair is `QUESTION_ASK`, and an
            # open one is what repair settles on resume.
            log.exception("ph.seams.user_questions: an answerer failed")
            return AskOutcome("failed")
        # Delivered and unanswered is a *person* declining; the failure above is
        # the machinery not reaching one. Both close the log pair, and they are
        # told apart here because nothing downstream can.
        return AskOutcome("answered", answer) if answer is not None else AskOutcome("declined")

    @staticmethod
    def _answered_data(
        question: UserQuestion, outcome: AskOutcome, *, ask_seq: int
    ) -> dict[str, Any]:
        """Close the pair, and say *how* it closed.

        The ask itself is `question.to_wire()` whole, rather than built field by
        field — the opposite of `ApprovalService._asked_data`, and for the
        opposite reason: an approval request grows fields for the *answerer's*
        benefit that have no business in the log, while a question is nothing but
        what was asked. Every field it gains is part of the question.

        **The resolution reaches the log, not just the model.** `declined` alone
        was the same collapse `AskResolution` exists to undo: a front end that
        fell over mid-ask recorded "asked and not answered", and the transcript
        renders that as *"No answer given."* — a person choosing not to answer a
        question they were never shown. The closer repair writes through says
        `interrupted` rather than `declined` for exactly this reason, so the log's
        vocabulary already knows the distinction is load-bearing; a live ask has to
        supply it too.

        Built by the kind's own `question_answered`, which the closer calls as well.
        """
        return question_answered(
            ask_id=question.ask_id,
            ask_seq=ask_seq,
            resolution=outcome.resolution,
            answer=outcome.answer,
        )


@plugin("user-questions")
async def apply(ctx: Context, config: None) -> None:
    """Mount the user-question seam."""
    ctx.provide(USER_QUESTIONS, UserQuestionService(ctx=ctx))
