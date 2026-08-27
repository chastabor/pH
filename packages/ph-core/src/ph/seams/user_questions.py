"""`ctx.user_questions` — asking the human something that is not an approval.

Distinct from `ctx.approval` because the shapes differ: an approval is a
one-shot yes/no about a *specific pending call* and must fail closed, while a
question is free-form and its failure mode is "no answer", which a caller
handles however it likes. Sharing one seam would force one of those two
behaviours onto the other.

@module ph.seams.user_questions
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..cordis import Context, Disposer, events, plugin
from ..wire import WireModel

__all__ = ["UserQuestion", "UserQuestionService", "apply"]

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


@dataclass(slots=True)
class UserQuestionService:
    """The service published as `ctx.user_questions`."""

    ctx: Context

    def register_answerer(
        self, answerer: Callable[..., Any], *, scope: Context | None = None
    ) -> Disposer:
        """Sugar over `ctx.on("user-question/ask", answerer)`; one mechanism."""
        return (scope or self.ctx).on("user-question/ask", answerer)

    async def ask(self, question: UserQuestion) -> str | None:
        """Ask, and return the answer or `None` when nobody could answer."""

        async def inner(_question: UserQuestion) -> str | None:
            return None

        try:
            answer = await self.ctx.waterfall("user-question/ask", question, inner=inner)
        except Exception:
            log.exception("ph.seams.user_questions: an answerer failed")
            return None
        return answer if isinstance(answer, str) else None


@plugin("user-questions")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the user-question seam."""
    ctx.provide("user_questions", UserQuestionService(ctx=ctx))
