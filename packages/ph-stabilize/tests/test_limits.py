"""`limits` (P4-04): hard boundaries on a loop that stopped making progress (G5).

The gate of the row, with one test for each part: *end / continue / error; the
breaker trips.*

Two things are worth reading beyond the behaviors.

The counts are a **fold over the log**. The first tests here therefore check
that a resumed session gets the same answer as a live one. A restart would
forget a limit that was held in a field. Nothing about the shipped behavior
would look different until someone resumed a session that had already spent its
budget.

Every ceiling is **unset by default**. Layering this bundle must not start to
refuse a long legitimate turn. The tests that exercise a limit therefore state
the number, and one test holds the default.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from stabilize_helpers import PROFILE, bash_call, events_of, row, run_tool_calls

from ph.agent.types import AgentDriver, AgentOptions, RequestErrorAction, RequestFailure
from ph.cordis import Context
from ph.json import as_obj, as_seq
from ph.keys import AGENTS, LLM, SESSIONS, SUBAGENTS, TOOLS, TUI_STATUS
from ph.llm.types import Finish, FinishReason, GenerateOptions, LlmFailure, ToolCallBlock
from ph.seams.subagents import ADMITTED, ChildCounts, SubagentRequest, SubagentSpawnError
from ph.session import Session, SurfaceIntent
from ph.session.known_event_types import IGNORABLE_SESSION_EVENT_TYPES
from ph.testing import (
    FAKE_OPTIONS,
    HoldingProvider,
    MountProfile,
    StubSubagentProvider,
    assert_fold_laws,
    log_event,
    result_text,
    session_of,
    simple_tool,
    tool_result_payload,
)
from ph.tools.code_mode import CodeDispatchLog
from ph.tools.errors import TOOL_BUDGET_SPENT, TOOL_DENIED
from ph_stabilize.limits import (
    BREAKER_DENIAL,
    SIBLING_STOPPED,
    TOOL_DENIAL,
    ModelCallLimitExceeded,
    _extend,
    counts_of,
)

pytestmark = pytest.mark.anyio


async def _pre_step(ctx: Context, agent: AgentDriver, *, turn: int, step: int) -> Any:  # noqa: ANN401
    """The decision that `agent/pre-step` reaches, with the `inner` of the loop itself."""
    from ph.agent.types import PreStepDecision, PreStepRequest

    async def inner(request: PreStepRequest) -> PreStepDecision:
        return PreStepDecision(kind="enter", messages=request.messages)

    return await ctx.waterfall(
        "agent/pre-step",
        PreStepRequest(agent=agent, session=session_of(agent), messages=(), turn=turn, step=step),
        inner=inner,
    )


def _failing_read(call_id: str) -> ToolCallBlock:
    """A call that really *errors*, which a non-zero shell exit does not.

    When `bash` returns `exit 3`, that is a successful tool call that reports a
    failed command. `is_error` says that the tool raised, and that is what the
    breaker counts. A read of a path that does not exist raises.
    """
    return ToolCallBlock(id=call_id, name="read", arguments=json.dumps({"path": "no/such/file"}))


def _settled_event(session: Session, call_id: str) -> Any:  # noqa: ANN401
    """The `tool/result` event for one call, found by its id.

    The search is by `toolCallId`. It does not look for the id in the serialized
    message. `new_message_id()` is random, so a substring test matches whenever
    the generated id happens to contain the id of the call. Such a test fails
    about one run in four and blames the code.
    """
    for event in events_of(session, "tool/result"):
        content = as_seq(as_obj(event.data.get("message")).get("content"))
        if any(as_obj(block).get("toolCallId") == call_id for block in content):
            return event
    raise AssertionError(f"no tool/result for {call_id!r}")


def _denied(session: Session, call_id: str, reason: str) -> bool:
    """Whether this call was refused with `reason`.

    The check is containment, not equality. The pipeline renders a `Deny` as
    `Error: <reason>` (B5), so the ported sentence is what the model reads
    *inside* a normalized error and is not the whole error.
    """
    return reason in result_text(session, call_id)


# ------------------------------------------------------------- the counting --


def test_the_counts_are_a_fold_a_resume_reproduces() -> None:
    """The property that the whole design depends on.

    A counter in memory answers the same question until the process restarts.
    These counts are folded from the log, so a session that is reopened from
    disk has spent exactly what it spent.
    """
    session = Session("counted")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    log_event(session, "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "read"})
    log_event(session, "turn/start", {"turn": 2})
    log_event(session, "step/start", {"turn": 2, "step": 1})

    counts = counts_of(session)

    assert (counts.session_calls, counts.turn_calls) == (2, 1), "turn/start resets the turn"
    assert (counts.session_tools, counts.turn_tools) == (1, 0)
    assert counts.per_tool_session == {"read": 1}
    # Reproduced from the same log by a reader that was never running.
    assert counts_of(Session("reopened", seed=list(session.events))) == counts


def test_a_failure_run_is_counted_per_tool_and_reset_by_a_success() -> None:
    """What the breaker reads. A tool that works intermittently never trips."""
    session = Session("failing")
    for index, is_error in enumerate([True, True, False, True], start=1):
        call_id = f"c{index}"
        log_event(session, "tool/call", {"turn": 1, "step": 1, "callId": call_id, "name": "bash"})
        log_event(
            session,
            "tool/result",
            tool_result_payload("out", f"m{index}", call_id, is_error=is_error),
            SurfaceIntent("append"),
        )

    assert counts_of(session).consecutive_failures == {"bash": 1}


def test_the_counts_obey_the_fold_laws() -> None:
    """`counts_of` is `_extend` from empty, so the law holds by construction.
    This test checks that and does not assume it. It checks every prefix of a
    log that exercises each counted type and the reset that a `turn/start`
    performs. (Children are not in this fold. They are counted in their own
    logs when a spawn is judged. See the tests at the end.)"""
    session = Session("lawful")
    for turn in (1, 2):
        log_event(session, "turn/start", {"turn": turn})
        log_event(session, "step/start", {"turn": turn, "step": 1})
        log_event(session, "step/retry", {"turn": turn, "step": 1, "attempt": 1, "by": "llm-retry"})
        for index, is_error in enumerate([True, False], start=1):
            call_id = f"t{turn}c{index}"
            log_event(
                session, "tool/call", {"turn": turn, "step": 1, "callId": call_id, "name": "bash"}
            )
            log_event(
                session,
                "tool/result",
                tool_result_payload("out", f"m{call_id}", call_id, is_error=is_error),
                SurfaceIntent("append"),
            )
        log_event(session, "assistant/chunk", {"text": "…"})

    assert_fold_laws(session, counts_of, _extend)


# -------------------------------------------------------------- model calls --


def test_a_retried_call_is_a_model_call() -> None:
    """P1: `step/start` is written one time per step, before the retry loop.

    A step that was retried twice was therefore one model call to this fold.
    The loop records each retry that it accepts as `step/retry`, whichever row
    asked for it, and a call is either kind. Sabotage: drop `step/retry` from
    `_COUNTED` and the calls are the steps.
    """
    session = Session("retried")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    log_event(session, "step/retry", {"turn": 1, "step": 1, "attempt": 1, "by": "llm-retry"})
    log_event(
        session, "step/retry", {"turn": 1, "step": 1, "attempt": 2, "by": "compaction-summarize"}
    )
    log_event(session, "turn/start", {"turn": 2})
    log_event(session, "step/start", {"turn": 2, "step": 1})

    counts = counts_of(session)

    assert (counts.session_calls, counts.turn_calls) == (4, 1), "turn/start resets the turn"


async def test_the_step_gate_counts_the_retries_already_made(mount: MountProfile) -> None:
    """P1: a step that was retried spent the calls that a next step would need."""
    ctx = await mount(row("limits", modelCalls={"turnLimit": 2}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("capped")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    log_event(session, "step/retry", {"turn": 1, "step": 1, "attempt": 1, "by": "llm-retry"})

    decision = await _pre_step(ctx, agent, turn=1, step=2)

    assert decision.kind == "reject"


@pytest.mark.parametrize("retrier", ["llm-retry", "another row"])
async def test_a_retry_past_the_ceiling_is_refused_whoever_asks(
    mount: MountProfile, retrier: str
) -> None:
    """P1: the ceiling is checked where a retry is decided.

    `agent/pre-step` fires one time per step, so a ceiling that was checked only
    there could not stop a retry at all. A model that failed transiently was
    called up to `maxAttempts` times per step against a `turnLimit` of 2. The
    ceiling is now held at `agent/request-error`, where it wraps every row that
    would grant a retry. That includes `llm-retry`, which answers without
    calling `next_`.

    Sabotage: drop the `agent/request-error` registration and both cases make
    five calls; register it without `prepend` and the `llm-retry` case does.
    """
    ctx = await mount(
        row("llm-retry", maxAttempts=5, baseDelayMs=1),
        row("limits", modelCalls={"turnLimit": 2}),
        profile=PROFILE,
    )
    calls = 0

    class Failing:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            nonlocal calls
            calls += 1
            # `llm-retry` retries `SERVER_ERROR`. It does not retry `TRANSIENT`,
            # so for that code the stand-in below is the row that asks.
            code = "SERVER_ERROR" if retrier == "llm-retry" else "TRANSIENT"
            yield Finish(
                reason=FinishReason(kind="error", failure=LlmFailure(message="no", code=code))
            )

    async def stand_in(failure: RequestFailure, next_: Any) -> Any:  # noqa: ANN401
        if failure.failure.code == "TRANSIENT" and failure.retries < 4:
            return RequestErrorAction(kind="retry")
        return await next_()

    ctx.require(LLM).register_adapter(["failing"], Failing())
    ctx.on("agent/request-error", stand_in)
    session = ctx.require(SESSIONS).create("s")
    with anyio.fail_after(10):
        await (
            ctx.require(AGENTS)
            .create(session, AgentOptions(provider="failing", model="m"))
            .prompt("hi")
        )

    assert calls == 2
    (breach,) = events_of(session, "limits/exceeded")
    assert breach.data["retry"] is True
    assert breach.data["message"] == "Model call limits exceeded: turn limit (2/2)"
    assert as_obj(session.events[-1].data["reason"])["kind"] == "error"


async def test_the_model_call_limit_ends_the_turn_and_says_why(mount: MountProfile) -> None:
    """`exit: end`: the step is rejected, and the log carries the reason.

    The test asks `agent/pre-step` about a session that has already spent its
    budget. That is where the decision is made and how it is made: from the log.
    To drive a multi-step turn through the fake adapter, the test would need a
    scripted tool loop to reach the state that the two `step/start` events below
    state directly.

    pH deviates from upstream in one visible way. Upstream injects an artificial
    `AIMessage`, so that a reader sees the agent announce its own limit. pH
    records `limits/exceeded` instead, because this codebase refuses to invent
    model speech everywhere else. The user still sees the breach, as a notice.
    """
    ctx = await mount(row("limits", modelCalls={"turnLimit": 2}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("capped")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    log_event(session, "step/start", {"turn": 1, "step": 2})

    decision = await _pre_step(ctx, agent, turn=1, step=3)

    assert decision.kind == "reject"
    (breach,) = events_of(session, "limits/exceeded")
    assert breach.data["message"] == "Model call limits exceeded: turn limit (2/2)"
    assert breach.ignorable


async def _fail_once_at_the_ceiling(mount: MountProfile, code: str, **retry: object) -> Session:
    """A turn whose only model call fails with `code`, under a `turnLimit` of 1."""
    ctx = await mount(
        row("llm-retry", **retry),
        row("limits", modelCalls={"turnLimit": 1}),
        profile=PROFILE,
    )

    class FailsOnce:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            yield Finish(
                reason=FinishReason(kind="error", failure=LlmFailure(message="no", code=code))
            )

    ctx.require(LLM).register_adapter(["once"], FailsOnce())
    session = ctx.require(SESSIONS).create("s")
    await ctx.require(AGENTS).create(session, AgentOptions(provider="once", model="m")).prompt("hi")
    return session


async def test_a_failure_nobody_would_retry_is_not_blamed_on_the_ceiling(
    mount: MountProfile,
) -> None:
    """P1 review: the ceiling refuses the retry that the chain granted, and
    nothing else.

    When the ceiling was checked before the rest of the chain, it recorded "not
    retried: model call limits" for an authentication failure that no row was
    ever going to retry. Sabotage: check the ceiling before `next_` and this
    finds one record.
    """
    session = await _fail_once_at_the_ceiling(mount, "AUTHENTICATION")

    assert events_of(session, "limits/exceeded") == []
    assert as_obj(session.events[-1].data["reason"])["kind"] == "error"


async def test_a_refused_retry_costs_no_backoff(mount: MountProfile) -> None:
    """P1 review: the loop sleeps the backoff, after the waterfall settles.

    `llm-retry` used to sleep inside the waterfall and then grant, so a ceiling
    that refused afterwards had already paid the wait. Here the wait would be
    ten seconds. Sabotage: sleep in `llm-retry` again and the deadline trips.

    The refused retry is also not recorded as a retry. `llm-retry` used to
    record `llm/retry` as it granted, so the log showed a retry that never ran.
    Sabotage: append it in the listener again.
    """
    with anyio.fail_after(2):
        session = await _fail_once_at_the_ceiling(
            mount, "SERVER_ERROR", baseDelayMs=10_000, maxDelayMs=10_000
        )

    (breach,) = events_of(session, "limits/exceeded")
    assert breach.data["retry"] is True
    assert events_of(session, "llm/retry") == events_of(session, "step/retry") == []


async def test_the_session_limit_outlives_the_turn(mount: MountProfile) -> None:
    """The *thread* limit of upstream, under the pH name. It does not reset at a
    turn boundary, and that is the whole difference between the two limits."""
    ctx = await mount(row("limits", modelCalls={"sessionLimit": 3}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("session-capped")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await agent.prompt("one")
    await agent.prompt("two")
    await agent.prompt("three")
    await agent.prompt("four")

    assert len(events_of(session, "step/start")) == 3
    assert events_of(session, "limits/exceeded")


async def test_error_raises_instead_of_ending(mount: MountProfile) -> None:
    """`exit: error`. The turn does not end quietly. A deployment that prefers a
    crash to a truncated turn can say so.

    **It also records the breach before it raises** (N2). D7 made this change on
    the tool-call limit and left the model-call limit behind. A raise out of
    `agent/pre-step` unwinds the turn, and an unwound turn appends nothing. The
    posture that reports most loudly to the deployment was therefore the one
    that left nothing for `phern doctor`, a reviewer or a resumed session to
    fold.

    Sabotage: put the raise back above `_record` and the event is gone.
    """
    ctx = await mount(row("limits", modelCalls={"turnLimit": 1, "exit": "error"}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("raising")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    engine = ctx.get("limits")

    await agent.prompt("go")
    # The driver contains a failed turn and does not propagate it. The test
    # therefore asks the listener directly for the raise, which is where the
    # handler of a deployment sees it.
    with pytest.raises(ModelCallLimitExceeded):
        await _pre_step(ctx, agent, turn=2, step=1)
    assert engine is None, "the row provides no service; it is listeners only"

    breaches = events_of(session, "limits/exceeded")
    assert [str(one.data.get("limit")) for one in breaches] == ["model-calls"]
    assert str(breaches[0].data.get("posture")) == "error"


async def test_no_limit_is_the_default(mount: MountProfile) -> None:
    """Layering the bundle must not start refusing anyone's long turn."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("uncapped")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    for _ in range(6):
        await agent.prompt("keep going")

    assert not events_of(session, "limits/exceeded")


# --------------------------------------------------------------- tool calls --


async def test_continue_denies_the_call_and_keeps_the_turn(mount: MountProfile) -> None:
    """`exit: continue`, the default. The model is told, in the words of
    upstream, not to call that tool again, and the turn goes on.

    **The breach is recorded all the same** (N2). This posture used to write
    nothing, on the reasoning that a breach which does not end the turn is not
    worth a record. That reasoning was backwards. `continue` is the shipped
    default, so it is the posture that a deployment is most likely to run when
    a budget first applies, and a turn that goes on makes the denial easy to
    miss. `posture` now tells the three readings apart. The event name alone
    used to mean two of them and omit the third.

    Sabotage: drop the `_record` from the `continue` branch and the log is silent.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("tool-capped")

    await run_tool_calls(ctx, session, bash_call("c1"))
    await run_tool_calls(ctx, session, bash_call("c2"), step=2)

    assert "exit status" not in result_text(session, "c1"), "the first call ran"
    assert _denied(session, "c2", TOOL_DENIAL.format(tool="bash"))
    breaches = events_of(session, "limits/exceeded")
    assert [str(one.data.get("posture")) for one in breaches] == ["continue"]
    assert not events_of(session, "turn/end"), "and the turn is still going"


async def test_error_records_the_breach_and_ends_the_turn_like_its_sibling(
    mount: MountProfile,
) -> None:
    """`exit: error` spends the budget, says so, and stops (D7).

    Two things were wrong, and they were the same mistake. First, it left no
    durable record. Only `end` wrote `limits/exceeded`, so the posture that
    reports a breach most loudly to the *model* said nothing to `phern doctor`,
    a reviewer or a resumed session. Second, it did not end the turn, although
    the model-call setting of the same name does. It raised out of
    `tools/pre-execute`, where the pipeline turns any exception from a policy
    row into a failed result. The ceiling inherited a posture that was meant to
    stop a *broken row* from taking down the call that it was asked about.

    Now it returns `Deny(concludes_turn=True, failure_kind="failed")`. The turn
    ends as `end` ends it, and the model reads a spent budget and not a policy
    refusal. That reading is everything that this word adds over `end`.

    Sabotage: drop `concludes_turn=True` and the second turn runs a third call.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "error"}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("tool-error")

    await run_tool_calls(ctx, session, bash_call("c1"))
    outcome = await run_tool_calls(ctx, session, bash_call("c2"), step=2)

    breaches = events_of(session, "limits/exceeded")
    assert [str(one.data.get("limit")) for one in breaches] == ["tool-calls"]
    assert str(breaches[0].data.get("posture")) == "error"
    assert "call limit reached" in str(breaches[0].data.get("message"))
    assert "Error" in result_text(session, "c2")

    # A spent budget, not policy refusing: nothing judged the call.
    settled = _settled_event(session, "c2")
    assert str(settled.data.get("failureKind")) == "failed", "a ceiling did not judge this call"
    assert dict(settled.data.get("error") or {}).get("code") == TOOL_BUDGET_SPENT

    # And the turn stops, as `end` stops it and as the model-call sibling does.
    assert outcome.concluded, "the turn ran on past a budget it had spent"


async def test_a_per_tool_budget_is_checked_beside_the_aggregate(mount: MountProfile) -> None:
    """One table where upstream mounts one middleware per tool."""
    ctx = await mount(
        row("limits", toolCalls={"perTool": {"bash": {"turnLimit": 1}}}), profile=PROFILE
    )
    session = ctx.require(SESSIONS).create("per-tool")

    await run_tool_calls(ctx, session, bash_call("c1"))
    await run_tool_calls(ctx, session, bash_call("c2"), step=2)

    assert _denied(session, "c2", TOOL_DENIAL.format(tool="bash"))


async def test_end_denies_the_breach_and_the_batch_skips_what_is_behind_it(
    mount: MountProfile,
) -> None:
    """`exit: end`, and the point where the mechanics of pH stopped showing
    through (C13).

    Upstream jumps to the end of the graph and synthesizes results for the calls
    that it skipped. pH could not, and this test used to say so. The batch kept
    opening the next group after a result concluded the turn. A later call was
    therefore *dispatched*, and the row had to refuse it with the sibling
    sentence of upstream. Now the batch stops, which is what upstream does. A
    later call is skipped and not denied, and its result says that the turn
    ended and says nothing about the call itself.

    `bash` is exclusive, so these three calls are three groups and not one
    batch. `SIBLING_STOPPED` is still the answer for a real sibling in flight,
    which is a *parallel* group that is gated concurrently. The next test covers
    that case.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("ending")

    await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"), bash_call("c3"))

    assert _denied(session, "c2", TOOL_DENIAL.format(tool="bash"))
    assert _denied(session, "c3", "the turn ended before this call ran"), (
        "a call the turn never reached was dispatched and refused instead of skipped"
    )
    (breach,) = events_of(session, "limits/exceeded")
    assert breach.data["limit"] == "tool-calls"


async def test_a_sibling_already_in_flight_gets_upstreams_own_sentence(
    mount: MountProfile,
) -> None:
    """The half that the batch cannot skip, and why `_is_first_over` is still
    needed (C13).

    A *parallel* group is gated call by call while the whole pool is already in
    flight. There is no "what comes after it" for the batch to stop: the
    siblings are here now. They did nothing wrong, so the model should read the
    sentence of upstream and not a denial that is phrased as being about them.

    Arithmetic tells them apart, not a latch: the breaching call is the one that
    found the count *at* the ceiling, and a follower finds it past the ceiling.

    Sabotage: drop `_is_first_over` and the third call reads as the breach.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    ctx.require(TOOLS).register(simple_tool("twin", safe=True))
    session = ctx.require(SESSIONS).create("siblings")

    calls = [ToolCallBlock(id=f"p{n}", name="twin", arguments="{}") for n in (1, 2, 3)]
    await run_tool_calls(ctx, session, *calls)

    assert _denied(session, "p2", TOOL_DENIAL.format(tool="twin")), "the breach was not named"
    assert _denied(session, "p3", SIBLING_STOPPED), (
        "a sibling already in flight read the refusal as being about itself"
    )


async def test_an_end_breach_concludes_the_batch(mount: MountProfile) -> None:
    """The other half of `end`: the turn stops, through the flag of the loop
    itself.

    `Deny.concludes_turn` is the same mechanism that a *successful* result uses
    to end a turn (`ToolRunContext.conclude_turn`), reached from the denial
    side. The loop therefore closes the turn at the batch boundary that it
    already has. The first shape of this row latched a `limits/exceeded` event
    and read it again at the next pre-step. That made an **ignorable** event
    into control state. A build that skipped the event, as the vocabulary says
    a build may, would not have ended the turn.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("closing")

    outcome = await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"))

    assert outcome.concluded, "the batch did not tell the loop the turn is over"


async def test_continue_leaves_the_turn_running(mount: MountProfile) -> None:
    """The pair to the test above, and the whole difference between the modes."""
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("continuing")

    outcome = await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"))

    assert not outcome.concluded


async def test_a_later_turn_is_not_ended_by_an_earlier_breach(mount: MountProfile) -> None:
    """Nothing carries the breach forward, and that is the purpose.

    The first shape read a logged breach again at every step and had to scope
    it to a turn by comparing seqs. Without that scope, one `end` rejected every
    step for the rest of the session, and the agent never spoke again. Ending
    through the batch outcome removes that whole class of question. The counts
    of a new turn are reset by `turn/start`, and there is no latch to expire.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("later")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    log_event(session, "turn/start", {"turn": 1})
    await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"))
    log_event(session, "turn/start", {"turn": 2})

    assert (await _pre_step(ctx, agent, turn=2, step=1)).kind == "enter"


# ------------------------------------------------------------- the breaker --


async def test_the_breaker_trips_after_repeated_failure(mount: MountProfile) -> None:
    """The row's second gate. Five identical failures is not a long task."""
    ctx = await mount(row("limits", breaker={"consecutiveFailures": 3}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("stuck")

    for step in range(1, 5):
        await run_tool_calls(ctx, session, _failing_read(f"c{step}"), step=step)

    assert _denied(session, "c4", BREAKER_DENIAL.format(tool="read", count=3))
    (tripped,) = events_of(session, "limits/breaker-tripped")
    assert dict(tripped.data) == {"tool": "read", "failures": 3, "limit": 3}
    assert tripped.ignorable


async def test_a_success_resets_the_breaker(mount: MountProfile) -> None:
    """Consecutive, not cumulative. A tool that works intermittently is not the
    failure that this catches."""
    ctx = await mount(row("limits", breaker={"consecutiveFailures": 2}), profile=PROFILE)
    session = ctx.require(SESSIONS).create("recovering")

    await run_tool_calls(ctx, session, _failing_read("c1"), step=1)
    await run_tool_calls(ctx, session, bash_call("c2"), step=2)
    await run_tool_calls(ctx, session, _failing_read("c3"), step=3)
    await run_tool_calls(ctx, session, bash_call("c4"), step=4)

    assert not events_of(session, "limits/breaker-tripped")
    assert not _denied(session, "c4", "is not being called again")


def test_the_event_types_are_ignorable() -> None:
    """Accounting that a reader may skip: the turn that it bounded is in the log
    either way. (`test_log_writers.py` checks that the types are *known*, for
    every package.)"""
    for event_type in ("limits/exceeded", "limits/breaker-tripped"):
        assert event_type in IGNORABLE_SESSION_EVENT_TYPES


def test_the_denial_text_is_upstreams() -> None:
    """The model-facing sentences are the ported ones, not paraphrases."""
    assert TOOL_DENIAL.format(tool="x") == "Tool call limit exceeded. Do not call 'x' again."
    assert SIBLING_STOPPED.startswith("Execution stopped before this tool call could run")


# --------------------------------------------------------------- the footer --


def _limits_reading(ctx: Context, session: Session) -> Any:  # noqa: ANN401
    """The reading of this row, by id.

    The lookup is by id and not "the only one". These tests asserted "the only
    one" before `StatusReading` had an id. Every profile now mounts rows that
    contribute a posture and a sandbox mode, so "the single reading" is no
    longer this one.
    """
    return next(
        (one for one in ctx.require(TUI_STATUS).readings(session) if one.id == "limits"), None
    )


async def test_the_footer_shows_the_tightest_budget(mount: MountProfile) -> None:
    """A live reading, not only the notice that arrives when the budget is spent.

    Upstream announces a limit on the step where it stops you, which is the one
    moment when the information can no longer change anything. The gauge beside
    it makes the same argument about context, and this reading makes that
    argument for a budget.
    """
    ctx = await mount(
        row("limits", modelCalls={"turnLimit": 10}, toolCalls={"turnLimit": 4}), profile=PROFILE
    )
    session = ctx.require(SESSIONS).create("gauged")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    for index in range(3):
        log_event(
            session, "tool/call", {"turn": 1, "step": 1, "callId": f"c{index}", "name": "bash"}
        )

    reading = _limits_reading(ctx, session)

    # 3/4 tools is tighter than 1/10 model calls, so it is the one that is shown.
    assert reading.text == "tools 3/4"
    assert reading.level == "normal", "0.75 is short of the gauge's own 0.85"

    log_event(session, "tool/call", {"turn": 1, "step": 1, "callId": "c3", "name": "bash"})
    reading = _limits_reading(ctx, session)

    assert reading.text == "tools 4/4"
    assert reading.level == "warning"


async def test_the_footer_says_nothing_when_no_budget_is_set(mount: MountProfile) -> None:
    """The shipped default. A row that always occupies the line teaches a user
    to stop reading it."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("ungauged")

    assert _limits_reading(ctx, session) is None


# ------------------------------------------------------------ code dispatches --


async def test_a_code_mode_dispatch_counts_as_a_tool_call(mount: MountProfile) -> None:
    """C1 made every dispatch a governed evaluation, and the fold now counts it.

    A `tools.glob(...)` call from inside a cell logs `tool/code-dispatch-start`,
    not `tool/call`. Per-tool budgets and the breaker were therefore checked on
    each dispatch against a count that never moved. The payloads go through the
    real wire type, so a renamed field fails here and does not silently stop
    the count.
    """
    ctx = await mount(row("limits"), profile=PROFILE)
    session = ctx.require(SESSIONS).create("cell")
    ref = {"root_call_id": "r1", "parent_call_id": "p1", "name": "glob"}
    for sub, failed in (("d1", False), ("d2", True)):
        start = CodeDispatchLog(**ref, sub_call_id=sub, is_error=failed).to_wire()
        del start["isError"]
        log_event(session, "tool/code-dispatch-start", {**start, "arguments": {}})
        settle = CodeDispatchLog(**ref, sub_call_id=sub, is_error=failed).to_wire()
        log_event(session, "tool/code-dispatch", {**settle, "content": []})

    current = counts_of(session)
    assert (current.turn_tools, current.session_tools) == (2, 2)
    assert current.per_tool_turn["glob"] == 2
    assert current.consecutive_failures["glob"] == 1, "the failed settle fed the breaker"


# ------------------------------------------------------------------ children --


async def _parent(ctx: Context, session_id: str = "parent") -> tuple[Any, Any, Any]:
    """A parent agent and a stub provider. The stub opens a log for each child
    that names its parent, as a real provider does, and the seam writes the
    admission of the child into that log. A spawn here therefore leaves what a
    real spawn leaves."""
    session = ctx.require(SESSIONS).create(session_id)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    provider = StubSubagentProvider(root=ctx)
    ctx.require(SUBAGENTS).register_provider("stub", provider)
    return session, agent, provider


async def test_the_children_budget_refuses_the_spawn_that_would_cross_it(
    mount: MountProfile,
) -> None:
    """The last piece of P4-04: a cap on children, as a guard on `ctx.subagents`.

    The spawn is refused *before* the provider is asked, so nothing is created.
    That is the contract that `SubagentSpawnError` states. The refusal is
    recorded as `limits/exceeded`, like the other two ceilings.
    """
    ctx = await mount(row("limits", children={"turnLimit": 1}), profile=PROFILE)
    session, parent, provider = await _parent(ctx)

    await ctx.require(SUBAGENTS).start("stub", SubagentRequest(prompt="go", parent=parent))

    with pytest.raises(SubagentSpawnError, match=r"turn limit exceeded \(2/1 children\)"):
        await ctx.require(SUBAGENTS).start("stub", SubagentRequest(prompt="again", parent=parent))

    assert len(provider.requests) == 1, "refused before the provider was asked"
    breach = events_of(session, "limits/exceeded")[-1].data
    assert breach["limit"] == "children" and breach["turn"] == 1


@pytest.mark.parametrize(
    ("exit_", "code", "kind", "ends"),
    [
        ("continue", TOOL_DENIED, "denied", False),
        ("end", TOOL_DENIED, "denied", True),
        ("error", TOOL_BUDGET_SPENT, "failed", True),
    ],
)
async def test_the_child_ceiling_has_the_postures_its_siblings_have(
    mount: MountProfile, exit_: str, code: str, kind: str, ends: bool
) -> None:
    """D15: the last ceiling that still ran the posture that D7 replaced.

    `SubagentSpawnError` was a bare `Exception`, so a refusal carried no code
    and no `failure_kind`. Both tool bodies also flattened it into a
    `ValueError`/`ToolCallError`. Three facts were dropped at the one boundary
    that had them. The refusal reached the model as a generic failure, the turn
    continued, and every later spawn got the same refusal.

    `ChildLimits` defended its one posture with "a child that is not created
    cannot be continued past or raised about". That is equally true of a tool
    call that is not executed, so it never distinguished the two. `ChildLimits`
    now carries the same `exit` as the other ceilings, and a refusal says which
    reading it is on the way out.

    `continue` is the default and is the old behavior, so a deployment that
    chose nothing keeps what it had.

    The code is *derived* from the reading and not chosen beside it, exactly as
    `registry._gate` derives it for the tool ceiling. Each code therefore
    carries one reading. `SPAWN_REFUSED` still means what it always meant: a
    misconfiguration that a caller can fix and retry.
    `test_an_ordinary_refusal_still_reads_as_one_the_model_may_retry` in
    `packages/ph-core/tests/test_subagent_task.py` covers that case.

    Sabotage: drop `concludes_turn` from the `Deny` that `refuse_child` returns
    and `end`/`error` stop ending the turn.
    """
    ctx = await mount(row("limits", children={"turnLimit": 1, "exit": exit_}), profile=PROFILE)
    session, parent, _provider = await _parent(ctx)

    await ctx.require(SUBAGENTS).start("stub", SubagentRequest(prompt="go", parent=parent))

    with pytest.raises(SubagentSpawnError) as caught:
        await ctx.require(SUBAGENTS).start("stub", SubagentRequest(prompt="again", parent=parent))

    refused = caught.value
    assert refused.code == code
    assert refused.failure_kind == kind
    assert refused.concludes_turn is ends
    assert str(events_of(session, "limits/exceeded")[-1].data["posture"]) == exit_


async def test_the_turn_cap_counts_children_admitted_this_turn(mount: MountProfile) -> None:
    """Phase 11: the admission of a child is in the log of the child, so it is
    counted there.

    The caps used to count `subagent/admitted` in the log of the parent, and the
    log of a parent no longer holds one. The seam writes the admission into the
    log of the child, stamped with `parentTurn`, the seq of the latest
    `turn/start` of the parent. With the count in the old place, every cap read
    zero and nothing was ever refused. A child from an earlier turn is off the
    count of this turn and still on the count of the session.

    Sabotage: count the parent's log for `subagent/admitted` again and the second
    spawn of turn one is made; count every child as this turn's and the first
    spawn of turn two is refused.
    """
    ctx = await mount(row("limits", children={"turnLimit": 1, "sessionLimit": 2}), profile=PROFILE)
    session, parent, provider = await _parent(ctx)
    subagents = ctx.require(SUBAGENTS)

    log_event(session, "turn/start", {"turn": 1})
    await subagents.start("stub", SubagentRequest(prompt="one", parent=parent))
    with pytest.raises(
        SubagentSpawnError, match=r"reached: turn limit exceeded \(2/1 children\)\."
    ):
        await subagents.start("stub", SubagentRequest(prompt="two", parent=parent))

    log_event(session, "turn/start", {"turn": 2})
    await subagents.start("stub", SubagentRequest(prompt="three", parent=parent))

    log_event(session, "turn/start", {"turn": 3})
    with pytest.raises(
        SubagentSpawnError, match=r"reached: session limit exceeded \(3/2 children\)\."
    ):
        await subagents.start("stub", SubagentRequest(prompt="four", parent=parent))

    assert [one.prompt for one in provider.requests] == ["one", "three"]
    assert not events_of(session, ADMITTED), "the premise: the parent's log holds no admission"
    turns = [one.seq for one in events_of(session, "turn/start")]
    admitted = subagents.children(session.id).values()
    assert [child.parent_turn for child in admitted] == turns[:2]
    breach = events_of(session, "limits/exceeded")[-1].data
    assert (breach["turn"], breach["session"]) == (0, 2)


async def test_two_spawns_at_once_cannot_both_pass_the_cap(mount: MountProfile) -> None:
    """The cap counts a spawn that is in progress (`SubagentService.child_counts`).
    While the first of two spawns from one step is being built, the second is
    refused.

    Sabotage: count only the admitted children in `refuse_child` and the second spawn
    is made.
    """
    ctx = await mount(row("limits", children={"turnLimit": 1}), profile=PROFILE)
    session, parent, provider = await _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    held = HoldingProvider(provider)
    subagents.register_provider("held", held)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(subagents.start, "held", SubagentRequest(prompt="one", parent=parent))
        await held.building.wait()
        with pytest.raises(SubagentSpawnError, match=r"turn limit exceeded \(2/1 children\)"):
            await subagents.start("stub", SubagentRequest(prompt="two", parent=parent))
        held.go.set()

    assert [one.prompt for one in provider.requests] == ["one"]
    assert subagents.child_counts(session) == ChildCounts(turn=1, session=1)


async def test_a_deleted_child_still_counts(mount: MountProfile) -> None:
    """A delete of a child frees none of the budget: the spawn was requested and
    made.

    The tombstone is now in the log of the child itself, beside its admission.
    The count is a count of admissions. It was the same when the log of the
    parent held the admissions, and a delete never removed one.

    Sabotage: leave deleted children out of `SubagentService.child_counts` and the second spawn
    is made.
    """
    ctx = await mount(row("limits", children={"sessionLimit": 1}), profile=PROFILE)
    session, parent, _provider = await _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    first = await subagents.start("stub", SubagentRequest(prompt="one", parent=parent))

    assert await subagents.delete(session, first.id, reason="not needed")
    assert subagents.children(session.id)[first.id].deleted

    with pytest.raises(SubagentSpawnError, match=r"session limit exceeded \(2/1 children\)"):
        await subagents.start("stub", SubagentRequest(prompt="two", parent=parent))


async def test_a_fork_starts_with_no_children(mount: MountProfile) -> None:
    """Decision 4: a fork is a new root, so none of this budget is spent yet.

    The count was folded from the log of the parent from seq 0. The log of a
    fork begins with the prefix of its source, so a fork inherited the children
    of its source and was refused spawns that it never made. The children of a
    fork are the children whose logs name it as their parent, and it has none.
    That is also true in its seeded turn, whose `turn/start` carries the same
    seq as the `turn/start` of the source.

    Sabotage: count the children of the log the fork was cut from
    (`header.parent_session`) and the fork's spawn is refused; count the parent's
    log for `subagent/admitted` and the source's is not.
    """
    ctx = await mount(row("limits", children={"turnLimit": 1, "sessionLimit": 1}), profile=PROFILE)
    source, parent, _provider = await _parent(ctx, "source")
    subagents = ctx.require(SUBAGENTS)
    log_event(source, "turn/start", {"turn": 1})
    await subagents.start("stub", SubagentRequest(prompt="one", parent=parent))
    with pytest.raises(SubagentSpawnError):
        await subagents.start("stub", SubagentRequest(prompt="two", parent=parent))
    log_event(source, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    fork = ctx.require(SESSIONS).fork(source)
    forked = ctx.require(AGENTS).create(fork, FAKE_OPTIONS)
    run = await subagents.start("stub", SubagentRequest(prompt="again", parent=forked))

    assert list(subagents.children(fork.id)) == [run.id]
    assert len(subagents.children(source.id)) == 1


async def test_no_children_budget_registers_no_guard(mount: MountProfile) -> None:
    """Unset by default, like every other ceiling here. The guard is not even
    registered, so a spawn pays nothing for a cap that nobody chose."""
    ctx = await mount(row("limits"), profile=PROFILE)
    assert ctx.require(SUBAGENTS)._guards == []
