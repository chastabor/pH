"""A decision adapter that answers from a script.

Two suites had each written one — ph-core's for the seam's records and ph-stabilize's
for the row that asks — and the second grew what the first lacked. One here, so a
test of a new asker starts from the same double the seam is tested against.

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
"""One scripted reply: a number answers every question as its kind is owed
(`StubDecider`), a `Decisions` is sent as it is — for a reply in another shape — and
an `LlmError` is raised."""


class StubDecider:
    """A decision model, as far as `ctx.llm` is concerned.

    Each request gets the next of `replies`, and the last again once they run out;
    every request is kept in `asked`. A number answers a `Noul` with that likelihood,
    a `Choice` with its first criterion certain and a `Score` at its lowest level
    certain, and says the request cost `input_tokens`. `hangs` never answers at all,
    for a caller's timeout. `route` is what `resolve_model` answers — a whole
    `ResolvedModel`, as `FakeAdapter.route` is, so a route fact a test needs is no
    second declaration here.
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
