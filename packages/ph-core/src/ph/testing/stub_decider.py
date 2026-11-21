"""A decision adapter that answers from a script.

Two suites had each written one: ph-core for the records of the seam, and
ph-stabilize for the row that asks. The second gained what the first lacked.
With one adapter here, a test of a new asker starts from the same double that
the seam is tested against.

@module ph.testing.stub_decider
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TypeAlias

import anyio

from ..llm.adapter import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    Decisions,
    LlmError,
    Noul,
    NoulAnswer,
    Question,
    ResolvedModel,
    Score,
    ScoreAnswer,
)
from ..llm.types import GenerateOptions, StreamChunk

__all__ = ["StubDecider"]

_Reply: TypeAlias = float | Decisions | LlmError
"""One scripted reply. A number answers every question in the form that its
kind requires (see `StubDecider`). A `Decisions` is sent as it is, for a reply
in another shape. An `LlmError` is raised."""


class StubDecider:
    """A decision model, as far as `ctx.llm` is concerned.

    Each request gets the next of `replies`. When they run out, every later
    request gets the last reply again. Every request is kept in `asked`.

    A number reports that the request cost `input_tokens`, and answers each
    kind of question as follows:

    - a `Noul` with that likelihood
    - a `Choice` with its first criterion certain
    - a `Score` at its lowest level, certain

    `hangs` never answers, for a test of the timeout of a caller. `route` is
    the answer of `resolve_model`. It is a whole `ResolvedModel`, as
    `FakeAdapter.route` is, so a route fact that a test needs has no second
    declaration here.
    """

    def __init__(
        self,
        *replies: _Reply,
        input_tokens: int | None = 120,
        route: ResolvedModel | None = None,
        hangs: bool = False,
    ) -> None:
        self.replies = replies
        self.input_tokens = input_tokens
        self.route = route
        self.hangs = hangs
        self.asked: list[DecisionRequest] = []

    def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("a decision model does not chat")

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return self.route or ResolvedModel()

    async def decide(self, request: DecisionRequest) -> Decisions:
        self.asked.append(request)
        if self.hangs:
            await anyio.sleep_forever()
        reply = self.replies[min(len(self.asked), len(self.replies)) - 1]
        if isinstance(reply, LlmError):
            raise reply
        if isinstance(reply, Decisions):
            return reply
        answers = {name: _answer(one, reply) for name, one in request.questions.items()}
        return Decisions(answers, self.input_tokens)


def _answer(question: Question, likelihood: float) -> Answer:
    match question:
        case Noul():
            return NoulAnswer(likelihood)
        case Choice():
            first = next(iter(question.criteria))
            return ChoiceAnswer(first, {key: float(key == first) for key in question.criteria})
        case Score():
            return ScoreAnswer(
                0.0, tuple(float(not index) for index in range(len(question.criteria)))
            )
