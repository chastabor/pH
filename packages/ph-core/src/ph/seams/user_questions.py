"""`ctx.user_questions` — asking the human something that is not an approval.

Distinct from `ctx.approval` because the shapes differ: an approval is a
one-shot yes/no about a *specific pending call* and must fail closed, while a
question is free-form and its failure mode is "no answer", which a caller
handles however it likes. Sharing one seam would force one of those two
behaviours onto the other.

**A question is logged only when it is actually put to a person** (P7-09). That
is the one rule here that is not obvious, and it follows from the failure mode
above rather than from tidiness: since "nobody could answer" resolves instantly
to `None`, appending around it would write a question-and-refusal pair into the
log of every unattended run — an `/autonomous` turn inside an interactive
profile, a `ph -p` against a profile that armed the row — for an exchange that
never happened. The log would then say a person was asked and declined, which is
a different and false claim.

So attendance is decided **first**, and an unattended ask appends nothing and
returns at once. A deliverable one appends `question/asked` *before* the
waterfall runs and `question/answered` after, which is §5 rule 2 in the order
`ApprovalService._record_asked` already uses: a crash between them leaves the
question in the log with no answer, which is exactly the pending state
`pending_questions` folds.

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

from ..cordis import Context, Disposer, events, plugin
from ..session import Session
from ..wire import WireModel
from ._registry import claim_entry

__all__ = [
    "PendingQuestion",
    "UserQuestion",
    "UserQuestionService",
    "apply",
    "pending_questions",
]

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
    key a re-posed question is recognised by are then one string by construction
    rather than three that agree while someone remembers to make them.

    `None` for a caller that has no natural key; `ask()` fills it in.
    """


@dataclass(frozen=True, slots=True)
class PendingQuestion:
    """An `asked` with no `answered` — a question a resume should put back.

    Carries the `UserQuestion` itself rather than a copy of its fields. The
    record *is* a serialized question (`_record_asked` writes `to_wire()`
    whole), so re-listing the fields here would be a second field list to keep
    in step — and the one that fails silently, by dropping whatever the model
    gains next rather than by not compiling. This is where it differs from
    `PendingApproval`, whose request deliberately does not reach the log intact.
    """

    seq: int
    question: UserQuestion

    @property
    def ask_id(self) -> str:
        return self.question.ask_id or ""


def pending_questions(session: Session) -> list[PendingQuestion]:
    """Questions this log put to a person and never recorded an answer for.

    Derived rather than tracked, for the reason `pending_approvals` is: the log
    *is* the pending state, so a crash between the two events cannot lose the
    question. Nothing re-poses these across a restart yet — see the module note
    in `ph_app.daemon.frontend` — but the fold is what that will read.
    """
    asked: dict[str, PendingQuestion] = {}
    for event in session.events:
        ask_id = str(event.data.get("askId", ""))
        if event.type == "question/asked":
            # `model_validate` off the event data, the way `RequestContext` and
            # `Message` are already rehydrated: `WireModel` owns the camelCase
            # aliases and the log's frozen mapping, so neither is spelled here.
            asked[ask_id] = PendingQuestion(
                seq=event.seq, question=UserQuestion.model_validate(event.data)
            )
        elif event.type == "question/answered":
            asked.pop(ask_id, None)
    return sorted(asked.values(), key=lambda pending: pending.seq)


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

    async def ask(self, question: UserQuestion, *, session: Session | None = None) -> str | None:
        """Ask, and return the answer or `None` when nobody could answer.

        `session` is optional and its absence means "do not record", which is the
        right default for a caller that is not part of a conversation. When it is
        given, both halves are appended — but only for a question that was
        actually delivered; see the module docstring.

        Identity travels on the question (`UserQuestion.ask_id`) rather than
        beside it, so there is one place a caller can put it and one place every
        route — the log record, the wire frame, a re-posed ask — reads it from.
        """
        if not self.attended:
            return None
        # Minted only when the caller had no natural key of its own. `ask_user`
        # passes the tool call id, which is the string the rest of the log
        # already joins the exchange by; a counter would restart at 1 after a
        # resume and answer a question the log was still holding open.
        asked = (
            question
            if question.ask_id
            else question.model_copy(update={"ask_id": f"q-{secrets.token_hex(4)}"})
        )
        if session is not None:
            self._record_asked(session, asked)

        async def inner(_question: UserQuestion) -> str | None:
            return None

        try:
            raw = await self.ctx.waterfall("user-question/ask", asked, inner=inner)
        except Exception:
            log.exception("ph.seams.user_questions: an answerer failed")
            raw = None
        answer = raw if isinstance(raw, str) else None
        if session is not None:
            self._record_answered(session, asked, answer)
        return answer

    def _record_asked(self, session: Session, question: UserQuestion) -> None:
        """The ask, as the log keeps it.

        Built from `to_wire()` rather than field by field — the opposite of
        `ApprovalService._record_asked`, and for the opposite reason: an approval
        request grows fields for the *answerer's* benefit that have no business
        in the log, while a question is nothing but what was asked. Every field
        it gains is part of the question and belongs here.
        """
        session.append("question/asked", question.to_wire())

    def _record_answered(
        self, session: Session, question: UserQuestion, answer: str | None
    ) -> None:
        data: dict[str, Any] = {"askId": question.ask_id}
        if answer is None:
            # Asked and *not* answered: somebody was there and declined, or the
            # answerer failed. Distinct from never being asked, which appends
            # nothing at all, and recorded so the fold stops calling it pending.
            data["declined"] = True
        else:
            data["answer"] = answer
        session.append("question/answered", data)


@plugin("user-questions")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the user-question seam."""
    ctx.provide("user_questions", UserQuestionService(ctx=ctx))
