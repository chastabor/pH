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

import anyio
import pytest

from ph.agent.types import AgentOptions, RequestErrorAction
from ph.json import as_obj
from ph.keys import AGENTS, LLM, SESSIONS
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
from ph.testing import MountProfile, assistant_payload, log_event, text_chunks, user_payload

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


FAST_RETRY: dict[str, Any] = {"id": "llm-retry", "config": {"maxAttempts": 3, "baseDelayMs": 1}}


async def test_retry_recovers_a_transient_failure_and_records_it(mount: MountProfile) -> None:
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

    ctx.require(LLM).register_adapter(["flaky"], Flaky())
    session = ctx.require(SESSIONS).create("s")
    await (
        ctx.require(AGENTS).create(session, AgentOptions(provider="flaky", model="m")).prompt("hi")
    )

    assert attempts["count"] == 3
    retries = [event for event in session.events if event.type == "llm/retry"]
    assert [event.data["attempt"] for event in retries] == [1, 2]
    assert retries[0].data["code"] == "RATE_LIMIT"
    assert as_obj(session.events[-1].data["reason"])["kind"] == "completed"


async def test_retry_is_bounded(mount: MountProfile) -> None:
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

    ctx.require(LLM).register_adapter(["down"], AlwaysFailing())
    session = ctx.require(SESSIONS).create("s")
    # Under a deadline: the claim is that the retries stop, and a regression that
    # never stops must fail rather than hang the suite.
    with anyio.fail_after(10):
        await (
            ctx.require(AGENTS)
            .create(session, AgentOptions(provider="down", model="m"))
            .prompt("hi")
        )

    # max_attempts=3: two retries, then the failure stands.
    assert attempts["count"] == 3
    assert as_obj(session.events[-1].data["reason"])["kind"] == "error"


async def test_two_agents_do_not_share_a_retry_budget(mount: MountProfile) -> None:
    """One row, many agents: the budget belongs to the step, not to the listener.

    The row mounts once per root and every agent beneath it dispatches to the
    same listener, so a count kept on the row was keyed by `turn:step` across all
    of them. Two subagents both working their first step shared `1:1` — and the
    entry was dropped only on give-up, so the first agent's *successful* retries
    were still on the books when the second one failed. The second was then
    refused a retry it had never used, which is precisely the storm this row
    exists for.

    Counted by each agent's driver instead, in a local of the step's retry loop,
    so the budget is per agent by construction. Sabotage: keep the count on the
    row, keyed by `turn:step`, and the second agent ends in `error` after one
    call.
    """
    ctx = await mount(FAST_RETRY)
    calls: list[str] = []

    class Flaky:
        """Fails a set number of times per session, then answers."""

        def __init__(self, plan: dict[str, int]) -> None:
            self.left = dict(plan)

        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            session_id = options.session_id or ""
            calls.append(session_id)
            if self.left.get(session_id, 0) > 0:
                self.left[session_id] -= 1
                yield Finish(
                    reason=FinishReason(
                        kind="error", failure=LlmFailure(message="429", code="RATE_LIMIT")
                    )
                )
                return
            for chunk in text_chunks("ok"):
                yield chunk

    # The first spends both its retries and recovers; the second needs one.
    ctx.require(LLM).register_adapter(["flaky"], Flaky({"first": 2, "second": 1}))
    options = AgentOptions(provider="flaky", model="m")
    sessions = ctx.require(SESSIONS)
    agents = ctx.require(AGENTS)
    first, second = sessions.create("first"), sessions.create("second")

    await agents.create(first, options).prompt("hi")
    await agents.create(second, options).prompt("hi")

    assert as_obj(first.events[-1].data["reason"])["kind"] == "completed"
    assert as_obj(second.events[-1].data["reason"])["kind"] == "completed", (
        "the second agent was refused a retry the first had spent"
    )
    assert calls.count("second") == 2, "one failure, one retry"


async def test_an_overflow_reaches_the_turn_instead_of_being_retried(mount: MountProfile) -> None:
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

    ctx.require(LLM).register_adapter(["big"], Overflowing())
    session = ctx.require(SESSIONS).create("s")
    await ctx.require(AGENTS).create(session, AgentOptions(provider="big", model="m")).prompt("hi")

    assert attempts["count"] == 1
    assert (
        as_obj(as_obj(session.events[-1].data["reason"])["error"])["code"]
        == CONTEXT_WINDOW_EXCEEDED
    )


def test_the_baseline_switches_from_estimate_to_usage() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    log_event(session, "user/message", user_payload("hello there", "m1"), SurfaceIntent("append"))

    before = meter.baseline(session)
    assert before.source == "estimate"
    assert before.tokens > 0

    payload = assistant_payload("a reply", "m2")
    payload["usage"] = TokenUsage(input_tokens=1_000, output_tokens=50).to_wire()
    log_event(session, "assistant/message", payload, SurfaceIntent("append", ()))

    after = meter.baseline(session)
    # The provider counted the prefix exactly; only a later tail is guessed.
    assert after.source == "usage"
    assert after.tokens == 1_050


def test_a_rewritten_message_does_not_erase_the_last_reported_usage() -> None:
    """The fold keeps what it had when a message carries no usage.

    Argument truncation rewrites an `assistant/message` in place, and the
    replacement reports none — it is the same request, not a new one. Clearing
    on it would drop `baseline` back to an *estimate* mid-conversation, which is
    the one thing the estimate exists not to be once a provider has counted.
    """
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    counted = assistant_payload("hi", "m1")
    counted["usage"] = TokenUsage(input_tokens=1_000, output_tokens=50).to_wire()
    log_event(session, "assistant/message", counted, SurfaceIntent("append", ()))

    rewritten = assistant_payload("hi", "m2")
    log_event(session, "assistant/message", rewritten, SurfaceIntent("append", ()))

    usage = meter.last_usage(session)
    assert usage is not None and usage.input_tokens == 1_000
    assert meter.baseline(session).source == "usage"


def test_pressure_needs_a_known_window() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    payload = assistant_payload("hi", "m1")
    payload["usage"] = TokenUsage(input_tokens=500, output_tokens=0).to_wire()
    log_event(session, "assistant/message", payload, SurfaceIntent("append", ()))
    assert meter.baseline(session).pressure is None

    log_event(session, "request/context", {"provider": "p", "model": "m", "contextWindow": 1_000})
    assert meter.baseline(session).pressure == 0.5


def test_cached_tokens_count_toward_the_baseline() -> None:
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    payload = assistant_payload("hi", "m1")
    payload["usage"] = TokenUsage(
        input_tokens=100, output_tokens=10, cache_read_tokens=900
    ).to_wire()
    log_event(session, "assistant/message", payload, SurfaceIntent("append", ()))
    # Counts are disjoint, so the window's occupancy is their sum.
    assert meter.baseline(session).tokens == 1_010


def test_the_cache_reading_says_nothing_until_a_provider_reports_cache() -> None:
    """`StatusField.read`'s rule, against the shape every session starts with.

    A first request has no cache to hit and most routes report no cache fields
    at all, so a footer field that rendered `cache 0%` there would occupy the
    one line a person reads for the whole session to say nothing.
    """
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    assert meter.cache_reading(session) is None, "an empty log says nothing"

    payload = assistant_payload("hi", "m1")
    payload["usage"] = TokenUsage(input_tokens=1_781, output_tokens=64).to_wire()
    log_event(session, "assistant/message", payload, SurfaceIntent("append", ()))

    assert meter.cache_reading(session) is None, "and so does a route that reports no cache"


def test_the_cache_reading_is_the_share_of_the_prompt_the_provider_reused() -> None:
    """The real shape, from a llama.cpp session: 1 777 of 1 810 prompt tokens.

    The percentage is of the *prompt* and not of the request, because output
    tokens are never cacheable — counting them would make a perfect hit rate
    read as a falling one on a long answer, which is the opposite of what this
    field is for.
    """
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    payload = assistant_payload("a reply", "m1")
    payload["usage"] = TokenUsage(
        input_tokens=33, output_tokens=391, cache_read_tokens=1_777
    ).to_wire()
    log_event(session, "assistant/message", payload, SurfaceIntent("append", ()))

    reading = meter.cache_reading(session)
    assert reading is not None
    assert reading.text == "cache 1.8k hit (98%)"


def test_a_stored_prefix_is_not_reported_as_a_hit() -> None:
    """Anthropic's first cached request: everything written, nothing read.

    Two different facts — what this request paid to store, and what it saved by
    reusing — and a field that added them would report the most expensive
    request of a session as its best-cached one.
    """
    meter = TokenMeter(ctx=None)  # type: ignore[arg-type]
    session = Session("s")
    payload = assistant_payload("a reply", "m1")
    payload["usage"] = TokenUsage(
        input_tokens=12, output_tokens=40, cache_write_tokens=2_048
    ).to_wire()
    log_event(session, "assistant/message", payload, SurfaceIntent("append", ()))

    reading = meter.cache_reading(session)
    assert reading is not None
    assert reading.text == "cache 2.0k stored"


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


@pytest.mark.parametrize(("count_all", "model_calls"), [(False, 4), (True, 3)])
async def test_a_compaction_retry_spends_the_budget_only_when_asked(
    mount: MountProfile, count_all: bool, model_calls: int
) -> None:
    """G13 — two rows retry, and only one of them wrote down that it had.

    `compaction` returns `retry` after shrinking an overflowing context, and the
    driver counts it under that row's name. `llm-retry` budgets against its own
    share by default: "the context was too big" and "the provider was busy" are
    unrelated problems, and a step that was compacted still gets every transient
    retry it was configured for. `count_all_retries` opts into one
    ceiling instead — `RequestFailure.retries`, every row's share.

    Here the first call overflows and a stand-in for `compaction` retries it;
    every call after fails transiently. With `maxAttempts: 3` the default allows
    two transient retries on top of the compaction one (four calls); counting it
    leaves one (three calls).

    Sabotage: read `retries_by["llm-retry"]` whatever the option says and the
    `True` case makes four calls; count the compaction retry under `llm-retry`
    and the `False` case makes three; hand the policy an empty count and the
    budget never runs out — which is why the adapter answers after ten calls, so
    that fails instead of hanging.
    """
    ctx = await mount(
        {
            **FAST_RETRY,
            "config": {**FAST_RETRY["config"], "countAllRetries": count_all},
        }
    )
    calls = {"count": 0}

    class OverflowThenBusy:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            calls["count"] += 1
            if calls["count"] > 10:
                # A budget that never runs out retries forever, and a regression
                # must fail rather than hang: answer, so the count below reports it.
                for chunk in text_chunks("gave up"):
                    yield chunk
                return
            code = CONTEXT_WINDOW_EXCEEDED if calls["count"] == 1 else "SERVER_ERROR"
            yield Finish(
                reason=FinishReason(kind="error", failure=LlmFailure(message="no", code=code))
            )

    async def compacting(failure: Any, next_: Any) -> Any:  # noqa: ANN401
        # `compaction`'s shape: retry an overflow. Registered outside any
        # plugin, so it is counted under `""` — any row but `llm-retry`.
        if failure.failure.code == CONTEXT_WINDOW_EXCEEDED:
            return RequestErrorAction(kind="retry")
        return await next_()

    ctx.require(LLM).register_adapter(["flaky"], OverflowThenBusy())
    ctx.on("agent/request-error", compacting)
    session = ctx.require(SESSIONS).create("s")
    await (
        ctx.require(AGENTS).create(session, AgentOptions(provider="flaky", model="m")).prompt("hi")
    )

    assert calls["count"] == model_calls
