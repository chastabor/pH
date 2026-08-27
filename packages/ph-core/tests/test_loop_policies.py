"""P1-10 and P1-13 — retry classification, and the two token numbers.

Retry gate: *overflow classified; retry bounded.* The interesting case is the
one retry **declines**: a context-window overflow will not fit on the second
attempt either, and swallowing that signal here would take away the one failure
that has a real remedy (compaction, G4).

Token gate: *the baseline switches from estimate to usage after the first
response.* Two numbers exist because one of them cannot: there is no reported
usage for a request that has not been made, so a pre-flight pressure check has
to guess.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.llm.retry import is_transient
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    LlmFailure,
    TextBlock,
    TextDelta,
    TokenUsage,
    create_user_message,
)
from ph.seams.token_meter import TokenMeter
from ph.session import Session, SurfaceIntent
from ph.testing import assistant_payload, user_payload

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (LlmFailure(message="slow down", code="RATE_LIMIT"), True),
        (LlmFailure(message="boom", code="SERVER_ERROR"), True),
        (LlmFailure(message="nothing", code="EMPTY_RESPONSE"), True),
        (LlmFailure(message="429", code="UNKNOWN", status=429), True),
        (LlmFailure(message="503", code="UNKNOWN", status=503), True),
        (LlmFailure(message="too long", code=CONTEXT_WINDOW_EXCEEDED), False),
        (LlmFailure(message="bad key", code="AUTHENTICATION"), False),
        (LlmFailure(message="400", code="UNKNOWN", status=400), False),
    ],
)
def test_transient_classification(failure: LlmFailure, expected: bool) -> None:
    assert is_transient(failure) is expected


def test_a_context_overflow_is_never_retried() -> None:
    # Stated as its own test because it is a decision, not a side effect of the
    # code table: compaction keys off this failure and needs to see it.
    assert not is_transient(LlmFailure(message="x", code=CONTEXT_WINDOW_EXCEEDED, status=429))


FAST_RETRY = {"id": "llm-retry", "config": {"maxAttempts": 3, "baseDelayMs": 1}}


async def test_retry_recovers_a_transient_failure_and_records_it(mount: Any) -> None:
    ctx = await mount(FAST_RETRY)
    attempts = {"count": 0}

    class Flaky:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            attempts["count"] += 1
            if attempts["count"] < 3:
                yield Finish(
                    reason=FinishReason(
                        kind="error",
                        failure=LlmFailure(message="slow down", code="RATE_LIMIT"),
                    )
                )
                return
            yield BlockStart(index=0, block_type="text")
            yield TextDelta(index=0, text="recovered")
            yield BlockEnd(index=0, block=TextBlock(text="recovered"))
            yield Finish(reason=FinishReason(kind="stop"))

    ctx.llm.register_adapter(["flaky"], Flaky())
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, AgentOptions(provider="flaky", model="m")).prompt("hi")

    assert attempts["count"] == 3
    retries = [event for event in session.events if event.type == "llm/retry"]
    assert [event.data["attempt"] for event in retries] == [1, 2]
    assert retries[0].data["code"] == "RATE_LIMIT"
    assert session.events[-1].data["reason"]["kind"] == "completed"


async def test_retry_is_bounded(mount: Any) -> None:
    ctx = await mount(FAST_RETRY)
    attempts = {"count": 0}

    class AlwaysFailing:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            attempts["count"] += 1
            yield Finish(
                reason=FinishReason(
                    kind="error", failure=LlmFailure(message="down", code="SERVER_ERROR")
                )
            )

    ctx.llm.register_adapter(["down"], AlwaysFailing())
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, AgentOptions(provider="down", model="m")).prompt("hi")

    # max_attempts=3: two retries, then the failure stands.
    assert attempts["count"] == 3
    assert session.events[-1].data["reason"]["kind"] == "error"


async def test_an_overflow_reaches_the_turn_instead_of_being_retried(mount: Any) -> None:
    ctx = await mount(FAST_RETRY)
    attempts = {"count": 0}

    class Overflowing:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            attempts["count"] += 1
            yield Finish(
                reason=FinishReason(
                    kind="error",
                    failure=LlmFailure(message="too long", code=CONTEXT_WINDOW_EXCEEDED),
                )
            )

    ctx.llm.register_adapter(["big"], Overflowing())
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, AgentOptions(provider="big", model="m")).prompt("hi")

    assert attempts["count"] == 1
    assert session.events[-1].data["reason"]["error"]["code"] == CONTEXT_WINDOW_EXCEEDED


def test_the_baseline_switches_from_estimate_to_usage() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    session.append("user/message", user_payload("hello there", "m1"), SurfaceIntent("append"))

    before = meter.baseline(session)
    assert before.source == "estimate"
    assert before.tokens > 0

    payload = assistant_payload("a reply", "m2")
    payload["usage"] = TokenUsage(input_tokens=1_000, output_tokens=50).to_wire()
    session.append("assistant/message", payload, SurfaceIntent("append", ()))

    after = meter.baseline(session)
    # The provider counted the prefix exactly; only a later tail is guessed.
    assert after.source == "usage"
    assert after.tokens == 1_050


def test_pressure_needs_a_known_window() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    payload = assistant_payload("hi", "m1")
    payload["usage"] = TokenUsage(input_tokens=500, output_tokens=0).to_wire()
    session.append("assistant/message", payload, SurfaceIntent("append", ()))
    assert meter.baseline(session).pressure is None

    session.append("request/context", {"provider": "p", "model": "m", "contextWindow": 1_000})
    assert meter.baseline(session).pressure == 0.5


def test_cached_tokens_count_toward_the_baseline() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    payload = assistant_payload("hi", "m1")
    payload["usage"] = TokenUsage(
        input_tokens=100, output_tokens=10, cache_read_tokens=900
    ).to_wire()
    session.append("assistant/message", payload, SurfaceIntent("append", ()))
    # Counts are disjoint, so the window's occupancy is their sum.
    assert meter.baseline(session).tokens == 1_010


def test_measuring_a_message_covers_every_text_carrying_block() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    message = create_user_message(
        content=[
            {"type": "text", "text": "a" * 40},
            {
                "type": "tool-result",
                "toolCallId": "c",
                "content": [{"type": "text", "text": "b" * 40}],
            },
        ],
        source={"kind": "user"},
    )
    # Nested tool-result content counts: it is context the model reads.
    assert meter.measure(message) >= 2
