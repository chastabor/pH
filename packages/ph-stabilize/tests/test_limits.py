"""P4-04 — `limits`: hard boundaries on a loop that stopped making progress (G5).

The row's gate, one test each: *end / continue / error; the breaker trips.*

Two things are worth reading past the behaviours. The counts are a **fold over
the log**, so the first tests here are about a resumed session getting the same
answer as a live one — a limit that lived in a field would be a limit a restart
forgets, and nothing about the shipped behaviour would look different until
someone resumed a session that had already spent its budget.

The other is that every ceiling is **unset by default**. Layering this bundle
must not start refusing a long legitimate turn, so the tests that exercise a
limit say the number out loud, and one test holds the default.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from stabilize_helpers import PROFILE, bash_call, events_of, result_text, row, run_tool_calls

from ph.llm.types import ToolCallBlock
from ph.session import Session, SurfaceIntent
from ph.session.known_event_types import (
    IGNORABLE_SESSION_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
)
from ph.testing import FAKE_OPTIONS, tool_result_payload
from ph_stabilize.limits import (
    BREAKER_DENIAL,
    SIBLING_STOPPED,
    TOOL_DENIAL,
    ModelCallLimitExceeded,
    counts_of,
)

pytestmark = pytest.mark.anyio


async def _pre_step(ctx: Any, agent: Any, *, turn: int, step: int) -> Any:
    """The decision `agent/pre-step` reaches, with the loop's own `inner`."""
    from ph.agent.types import PreStepDecision, PreStepRequest

    return await ctx.waterfall(
        "agent/pre-step",
        PreStepRequest(agent=agent, messages=(), turn=turn, step=step),
        inner=lambda request: PreStepDecision(kind="enter", messages=request.messages),
    )


def _failing_read(call_id: str) -> ToolCallBlock:
    """A call that genuinely *errors*, which a non-zero shell exit does not.

    `bash` returning `exit 3` is a successful tool call reporting a failed
    command — `is_error` is about the tool raising, and the breaker counts the
    former. Reading a path that is not there raises.
    """
    return ToolCallBlock(id=call_id, name="read", arguments=json.dumps({"path": "no/such/file"}))


def _denied(session: Session, call_id: str, reason: str) -> bool:
    """Whether this call was refused with `reason`.

    Containment, not equality: the pipeline renders a `Deny` as `Error: <reason>`
    (B5), so the ported sentence is what the model reads *inside* a normalized
    error rather than the whole of it.
    """
    return reason in result_text(session, call_id)


# ------------------------------------------------------------- the counting --


def test_the_counts_are_a_fold_a_resume_reproduces() -> None:
    """The property the whole design rests on.

    A counter in memory answers the same question until the process restarts;
    this is folded from the log, so a session reopened from disk has spent
    exactly what it spent.
    """
    session = Session("counted")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "read"})
    session.append("turn/start", {"turn": 2})
    session.append("step/start", {"turn": 2, "step": 1})

    counts = counts_of(session)

    assert (counts.session_steps, counts.turn_steps) == (2, 1), "turn/start resets the turn"
    assert (counts.session_tools, counts.turn_tools) == (1, 0)
    assert counts.per_tool_session == {"read": 1}
    # Reproduced from the same log by a reader that was never running.
    assert counts_of(Session("reopened", seed=list(session.events))) == counts


def test_a_failure_run_is_counted_per_tool_and_reset_by_a_success() -> None:
    """What the breaker reads. A tool that works intermittently never trips."""
    session = Session("failing")
    for index, is_error in enumerate([True, True, False, True], start=1):
        call_id = f"c{index}"
        session.append("tool/call", {"turn": 1, "step": 1, "callId": call_id, "name": "bash"})
        session.append(
            "tool/result",
            tool_result_payload("out", f"m{index}", call_id, is_error=is_error),
            SurfaceIntent("append"),
        )

    assert counts_of(session).consecutive_failures == {"bash": 1}


# -------------------------------------------------------------- model calls --


async def test_the_model_call_limit_ends_the_turn_and_says_why(mount: Any) -> None:
    """`exit: end` — the step is rejected and the log carries the reason.

    Asked of `agent/pre-step` over a session that has already spent its budget,
    which is where the decision is made and how it is made: from the log. Driving
    a multi-step turn through the fake adapter would need a scripted tool loop to
    reach the same state the two `step/start` events below say plainly.

    Deviating from upstream in one visible way: it injects an artificial
    `AIMessage` so a reader sees the agent announce its own limit. pH records
    `limits/exceeded` instead, because inventing model speech is the thing this
    codebase refuses everywhere else — the person still sees it, as a notice.
    """
    ctx = await mount(row("limits", modelCalls={"turnLimit": 2}), profile=PROFILE)
    session = ctx.sessions.create("capped")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("step/start", {"turn": 1, "step": 2})

    decision = await _pre_step(ctx, agent, turn=1, step=3)

    assert decision.kind == "reject"
    (breach,) = events_of(session, "limits/exceeded")
    assert breach.data["message"] == "Model call limits exceeded: turn limit (2/2)"
    assert breach.ignorable


async def test_the_session_limit_outlives_the_turn(mount: Any) -> None:
    """Upstream's *thread* limit under pH's name: it does not reset at a turn
    boundary, which is the whole difference between the two."""
    ctx = await mount(row("limits", modelCalls={"sessionLimit": 3}), profile=PROFILE)
    session = ctx.sessions.create("session-capped")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await agent.prompt("one")
    await agent.prompt("two")
    await agent.prompt("three")
    await agent.prompt("four")

    assert len(events_of(session, "step/start")) == 3
    assert events_of(session, "limits/exceeded")


async def test_error_raises_instead_of_ending(mount: Any) -> None:
    """`exit: error`. The turn does not end quietly — a deployment that would
    rather crash than truncate gets to say so."""
    ctx = await mount(row("limits", modelCalls={"turnLimit": 1, "exit": "error"}), profile=PROFILE)
    session = ctx.sessions.create("raising")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    engine = ctx.get("limits")

    await agent.prompt("go")
    # The driver contains a failed turn rather than propagating, so the raise is
    # asked of the listener directly — where a deployment's own handler sees it.
    with pytest.raises(ModelCallLimitExceeded):
        await _pre_step(ctx, agent, turn=2, step=1)
    assert engine is None, "the row provides no service; it is listeners only"


async def test_no_limit_is_the_default(mount: Any) -> None:
    """Layering the bundle must not start refusing anyone's long turn."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("uncapped")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    for _ in range(6):
        await agent.prompt("keep going")

    assert not events_of(session, "limits/exceeded")


# --------------------------------------------------------------- tool calls --


async def test_continue_denies_the_call_and_keeps_the_turn(mount: Any) -> None:
    """`exit: continue`, the default. The model is told, in upstream's own
    words, not to call that tool again — and the turn goes on."""
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1}), profile=PROFILE)
    session = ctx.sessions.create("tool-capped")

    await run_tool_calls(ctx, session, bash_call("c1"))
    await run_tool_calls(ctx, session, bash_call("c2"), step=2)

    assert "exit status" not in result_text(session, "c1"), "the first call ran"
    assert _denied(session, "c2", TOOL_DENIAL.format(tool="bash"))
    assert not events_of(session, "limits/exceeded"), "continue is not a turn-ending breach"


async def test_a_per_tool_budget_is_checked_beside_the_aggregate(mount: Any) -> None:
    """One table where upstream mounts one middleware per tool."""
    ctx = await mount(
        row("limits", toolCalls={"perTool": {"bash": {"turnLimit": 1}}}), profile=PROFILE
    )
    session = ctx.sessions.create("per-tool")

    await run_tool_calls(ctx, session, bash_call("c1"))
    await run_tool_calls(ctx, session, bash_call("c2"), step=2)

    assert _denied(session, "c2", TOOL_DENIAL.format(tool="bash"))


async def test_end_denies_the_siblings_in_upstreams_words(mount: Any) -> None:
    """`exit: end`, and the one place pH's mechanics show through.

    Upstream jumps to the graph's end and synthesizes results for the calls it
    skipped. pH's batch is already dispatched, so the breaching call is denied
    and its siblings get upstream's own sentence — which matters because those
    calls did nothing wrong and the model must not read the refusal as being
    about them.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    session = ctx.sessions.create("ending")

    await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"), bash_call("c3"))

    assert _denied(session, "c2", TOOL_DENIAL.format(tool="bash"))
    assert _denied(session, "c3", SIBLING_STOPPED)
    (breach,) = events_of(session, "limits/exceeded")
    assert breach.data["limit"] == "tool-calls"


async def test_an_end_breach_concludes_the_batch(mount: Any) -> None:
    """The other half of `end`: the turn stops — through the loop's own flag.

    `Deny.concludes_turn` is the same mechanism a *successful* result uses to end
    a turn (`ToolRunContext.conclude_turn`), reached from the denial side, so the
    loop closes the turn at the batch boundary it already has. The first shape of
    this row latched a `limits/exceeded` event and re-read it at the next
    pre-step, which made an **ignorable** event into control state — a build that
    skipped it, as the vocabulary says one may, would not have ended the turn.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    session = ctx.sessions.create("closing")

    outcome = await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"))

    assert outcome.concluded, "the batch did not tell the loop the turn is over"


async def test_continue_leaves_the_turn_running(mount: Any) -> None:
    """The pair to the test above, and the whole difference between the modes."""
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1}), profile=PROFILE)
    session = ctx.sessions.create("continuing")

    outcome = await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"))

    assert not outcome.concluded


async def test_a_later_turn_is_not_ended_by_an_earlier_breach(mount: Any) -> None:
    """Nothing carries the breach forward, which is the point.

    The first shape re-read a logged breach at every step and had to scope it to
    a turn by comparing seqs — without which one `end` rejected every step for
    the rest of the session and the agent never spoke again. Ending through the
    batch outcome makes that whole class of question disappear: a new turn's
    counts are reset by `turn/start` and there is no latch to expire.
    """
    ctx = await mount(row("limits", toolCalls={"turnLimit": 1, "exit": "end"}), profile=PROFILE)
    session = ctx.sessions.create("later")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    session.append("turn/start", {"turn": 1})
    await run_tool_calls(ctx, session, bash_call("c1"), bash_call("c2"))
    session.append("turn/start", {"turn": 2})

    assert (await _pre_step(ctx, agent, turn=2, step=1)).kind == "enter"


# ------------------------------------------------------------- the breaker --


async def test_the_breaker_trips_after_repeated_failure(mount: Any) -> None:
    """The row's second gate. Five identical failures is not a long task."""
    ctx = await mount(row("limits", breaker={"consecutiveFailures": 3}), profile=PROFILE)
    session = ctx.sessions.create("stuck")

    for step in range(1, 5):
        await run_tool_calls(ctx, session, _failing_read(f"c{step}"), step=step)

    assert _denied(session, "c4", BREAKER_DENIAL.format(tool="read", count=3))
    (tripped,) = events_of(session, "limits/breaker-tripped")
    assert dict(tripped.data) == {"tool": "read", "failures": 3, "limit": 3}
    assert tripped.ignorable


async def test_a_success_resets_the_breaker(mount: Any) -> None:
    """Consecutive, not cumulative — a tool that works intermittently is not
    the failure this catches."""
    ctx = await mount(row("limits", breaker={"consecutiveFailures": 2}), profile=PROFILE)
    session = ctx.sessions.create("recovering")

    await run_tool_calls(ctx, session, _failing_read("c1"), step=1)
    await run_tool_calls(ctx, session, bash_call("c2"), step=2)
    await run_tool_calls(ctx, session, _failing_read("c3"), step=3)
    await run_tool_calls(ctx, session, bash_call("c4"), step=4)

    assert not events_of(session, "limits/breaker-tripped")
    assert not _denied(session, "c4", "is not being called again")


def test_the_event_types_are_in_the_vocabulary() -> None:
    """The proof a producer outside ph-core owes through its own bundle."""
    for event_type in ("limits/exceeded", "limits/breaker-tripped"):
        assert event_type in KNOWN_SESSION_EVENT_TYPES
        assert event_type in IGNORABLE_SESSION_EVENT_TYPES


def test_the_denial_text_is_upstreams() -> None:
    """The model-facing sentences are the ported ones, not paraphrases."""
    assert TOOL_DENIAL.format(tool="x") == "Tool call limit exceeded. Do not call 'x' again."
    assert SIBLING_STOPPED.startswith("Execution stopped before this tool call could run")


# --------------------------------------------------------------- the footer --


async def test_the_footer_shows_the_tightest_budget(mount: Any) -> None:
    """A live reading, not just the notice that lands when the budget is spent.

    Upstream announces a limit on the step it stops you — the one moment the
    information can no longer change anything. The gauge beside it makes the
    same argument about context, and this is that argument for a budget.
    """
    ctx = await mount(
        row("limits", modelCalls={"turnLimit": 10}, toolCalls={"turnLimit": 4}), profile=PROFILE
    )
    session = ctx.sessions.create("gauged")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    for index in range(3):
        session.append("tool/call", {"turn": 1, "step": 1, "callId": f"c{index}", "name": "bash"})

    (reading,) = ctx.tui_status.readings(session)

    # 3/4 tools is tighter than 1/10 steps, and that is the one worth the line.
    assert reading.text == "tools 3/4"
    assert reading.level == "normal", "0.75 is short of the gauge's own 0.85"

    session.append("tool/call", {"turn": 1, "step": 1, "callId": "c3", "name": "bash"})
    (reading,) = ctx.tui_status.readings(session)

    assert reading.text == "tools 4/4"
    assert reading.level == "warning"


async def test_the_footer_says_nothing_when_no_budget_is_set(mount: Any) -> None:
    """The shipped default. A row that always occupies the line teaches a person
    to stop reading it."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("ungauged")

    assert ctx.tui_status.readings(session) == []
