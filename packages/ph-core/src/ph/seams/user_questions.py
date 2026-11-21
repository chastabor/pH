"""`ctx.user_questions`: ask the user something that is not an approval.

This seam is distinct from `ctx.approval` because the shapes differ. An approval
is a one-shot yes/no about a *specific pending call*, and it must fail closed. A
question is free-form, and its failure mode is "no answer", which a caller
handles however it likes. One seam for both would force one of those two
behaviors onto the other.

**A question is logged only when it is actually put to a person** (P7-09). This
is the one rule here that is not obvious. It follows from the failure mode
above and not from tidiness. "Nobody could answer" resolves instantly, so an
append around it would write a pair of question and refusal into the log of
every unattended run, for an exchange that never happened. Examples of such a
run are an `/autonomous` turn inside an interactive profile, and a `phern -p`
against a profile that armed the row. The log would then say that a person was
asked and declined, which is a different claim and a false one.

So attendance is decided **first**. An unattended ask appends nothing and
returns at once. A deliverable ask appends `question/asked` *before* the
waterfall runs and `question/answered` after it. That is §5 rule 2, in the
order that approvals use. A crash between the two leaves the question in the
log with no answer (`QUESTION_ASK` open), and repair settles it on resume.

Reachability is the claim of the answerer itself, because only the answerer
knows. An in-process front end *is* the screen of the user. It says so by
saying nothing, because the default is reachable. The ask desk of a daemon is
reachable when some front end is attached and not otherwise. `AskDesk` already
makes the same distinction between watching and answering.

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

# `AskResolution` is every way that one question can end. It is declared with
# the kind in `ph.session.kinds` (T4), because the answered payload carries it,
# and it is re-exported here.


@dataclass(frozen=True, slots=True)
class AskOutcome:
    """What became of one question. The seam decides this, and nobody
    reconstructs it afterwards.

    The seam already made every distinction that this value carries, and used
    to discard them. Attendance is checked before anything is recorded,
    cancellation is checked before that, and an answerer that raises is caught
    and logged. The caller could not derive any of them again afterwards. That
    is why the result is a value and not a `str | None`.
    """

    resolution: AskResolution
    answer: str | None = None
    """The words of the user. It is set only when `resolution` is `answered`, so
    a caller that reads it without a check cannot mistake a refusal for an empty
    answer."""

    def __post_init__(self) -> None:
        # The docstring above is a promise, and this check keeps it. An
        # `AskOutcome("declined", "hi")` would attribute words to somebody who
        # declined.
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
    """One question put to the user."""

    question: str
    options: list[str] | None = None
    header: str | None = None
    multi_select: bool = False
    ask_id: str | None = None
    """The identity of this question, which correlates an answer with the ask.

    It is carried on the question and not passed beside it, so that every route
    to an answerer keeps the two together. The `askId` of the log, the `askId`
    of the wire frame, and the key by which a question that is posed again is
    recognized are then one string by construction. They are not three strings
    that agree only while someone remembers to make them agree.

    `None` for a caller that has no natural key. `ask()` fills it in.
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
        """Sugar over `ctx.on("user-question/ask", answerer)`. It is one mechanism.

        With `reachable`, an answerer says whether it can reach a person *right
        now*. The default is "yes": an in-process front end registers one
        answerer for one screen, so its presence is the answer. A transport
        that may have nobody behind it, such as a daemon whose front ends have
        all closed, passes a probe. `attended` consults the probe at ask time
        and not at registration, because ask time is when the answer can have
        changed.
        """
        owner = self.ctx.owner_for(scope)
        off = owner.on("user-question/ask", answerer)
        # `claim_entry` and not `append` with a hand-written `remove`. This
        # package grew the helper because that release was written out six
        # times and drifted. It removes by *identity*, so two answerers with
        # equal probes cannot take the disposer of each other.
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
        """Ask, and report what became of the question.

        The result is an `AskOutcome` and not a `str | None`, because the four
        ways to get no answer are four different things to tell a model, and
        this is the only place that can tell them apart. See `AskResolution`.

        `cancel` is the cancellation of the caller. A canceled ask is not put
        to anybody, which is the rule that `ApprovalService.request` states for
        the other seam that interrupts a person. It is not recorded either.
        This seam already logs only a question that was *delivered*, and an
        abandoned question never was.

        `session` is optional, and its absence means "do not record". That is
        the right default for a caller that is not part of a conversation. When
        it is given, both halves are appended, but only for a question that was
        actually delivered. See the module docstring.

        The identity is carried on the question (`UserQuestion.ask_id`) and not
        beside it. A caller then has one place to put it, and every route (the
        log record, the wire frame, an ask that is posed again) has one place
        to read it from.
        """
        if is_canceled(cancel):
            return AskOutcome("canceled")
        if not self.attended:
            return AskOutcome("unattended")
        # An id is minted only when the caller had no natural key of its own.
        # `ask_user` passes the tool call id, which is the string by which the
        # rest of the log already joins the exchange. A counter would restart at
        # 1 after a resume, and would answer a question that the log still held
        # open.
        asked = (
            question
            if question.ask_id
            else question.model_copy(update={"ask_id": f"q-{secrets.token_hex(4)}"})
        )
        if session is None:
            return await self._deliver(asked)
        # The ask is on disk before it is delivered (F8). That is the reason
        # why the ask is appended ahead of the waterfall: a crash while somebody
        # was deciding must leave the question open in the log, for repair to
        # settle. The barrier of the kind does it. An ask that cannot be
        # written is not asked, and the journal closes it `failed`.
        #
        # **`claim`, so that the pair closes even when the wait does not
        # return** (S15). A question that was canceled while somebody was
        # looking at it used to stay pending, on the premise that the next
        # resume poses it again. Nothing does that across a restart: `AskDesk`
        # poses a question again only while the daemon runs, and repair settles
        # what is left. So the question stayed open in the live log and read as
        # still waiting. `claim` settles it on the way out with the own closer
        # of the kind. That is the record that repair would have written,
        # `outcome-unknown` and by the process, and it deliberately does not
        # say that anybody *declined* the question.
        journal = intents_of(self.ctx)
        try:
            async with journal.claim(session, QUESTION_ASK, asked.to_wire()) as held:
                outcome = await self._deliver(asked)
                journal.settle(
                    session, held, self._answered_data(asked, outcome, ask_seq=held.opened.seq)
                )
        except IntentNotDurable:
            return AskOutcome("failed")
        return outcome

    async def _deliver(self, asked: UserQuestion) -> AskOutcome:
        async def inner(_question: UserQuestion) -> str | None:
            return None

        try:
            raw = await self.ctx.waterfall("user-question/ask", asked, inner=inner)
            answer = settled_or_none("user-question/ask", raw, str)
        except Exception:
            # To this seam, an answerer that raises and an answerer that answers
            # with the wrong shape are the same failure. In either case the ask
            # must be closed in the log. The `asked`/`answered` pair is
            # `QUESTION_ASK`, and an open pair is what repair settles on resume.
            log.exception("ph.seams.user_questions: an answerer failed")
            return AskOutcome("failed")
        # Delivered and unanswered is a *person* who declines. The failure above
        # is the machinery that does not reach a person. Both close the log
        # pair, and they are told apart here because nothing downstream can do
        # it.
        return AskOutcome("answered", answer) if answer is not None else AskOutcome("declined")

    @staticmethod
    def _answered_data(
        question: UserQuestion, outcome: AskOutcome, *, ask_seq: int
    ) -> dict[str, Any]:
        """Close the pair, and say *how* it closed.

        The ask itself is the whole of `question.to_wire()`, and is not built
        field by field. That is the opposite of `ApprovalService._asked_data`,
        and the reason is the opposite too. An approval request grows fields
        for the *answerer* that do not belong in the log. A question is nothing
        but what was asked, so every field that it gains is part of the
        question.

        **The resolution reaches the log, not just the model.** `declined`
        alone was the same loss of detail that `AskResolution` exists to undo.
        A front end that failed in the middle of an ask recorded "asked and not
        answered", and the transcript renders that as *"No answer given."*.
        That reads as a person who chose not to answer a question that they
        were never shown. The closer that repair writes through writes no
        `declined` and carries the `unsettled` marker, for exactly this reason.
        The vocabulary of the log therefore already depends on the
        distinction, and a live ask has to supply it too.

        The payload is built by `question_answered` of the kind, which the
        closer also calls.
        """
        return question_answered(
            ask_id=question.ask_id,
            ask_seq=ask_seq,
            resolution=outcome.resolution,
            answer=outcome.answer,
        )


@plugin("user-questions", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the user-question seam."""
    ctx.provide(USER_QUESTIONS, UserQuestionService(ctx=ctx))
