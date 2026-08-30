"""P5-07 — goals, budgets and quality gates.

The seam is a fold plus four appends, so most of this drives it directly. The
two that need a mounted profile — the gate memo and the command — go through
`mount`, because a gate is a real shell run against a real worktree and the
fingerprint is a real `git write-tree`.
"""

from __future__ import annotations

import pytest

from ph.agent.types import AgentOptions
from ph.seams.goals import (
    Budget,
    Goal,
    GoalService,
    Spent,
    goals,
)
from ph.session import Session, SurfaceIntent
from ph.testing import assistant_payload


def _open(session: Session, service: GoalService, gates: list[str] | None = None) -> Goal:
    """Open a goal. Typed, because the `**over` version needed a
    `type: ignore[arg-type]` that switched off argument checking for the whole
    call — which is how three call sites passed `gates=["pytest"]`, the tuple
    the field docstring says the log refuses outright."""
    return service.set(session, Goal(id="g1", objective="make the tests pass", gates=gates or []))


def test_budget_exhaustion_is_named_not_merely_reported() -> None:
    """The row's first gate, and the reason `Spent.exhausted` returns a string.

    "It stopped" and "it ran out of turns" are different things to whoever reads
    the trace, and only one of them suggests raising a number. A boolean could
    not tell them apart, and four budgets that all reported the same word would
    hide which one actually bound.
    """
    budget = Budget(max_continuations=3, max_turns=12, max_tokens=80_000, timeout_ms=1_800_000)

    assert Spent().exhausted(budget, elapsed_ms=0) is None
    assert Spent(continuations=3).exhausted(budget, elapsed_ms=0) == "max_continuations"
    assert Spent(turns=12).exhausted(budget, elapsed_ms=0) == "max_turns"
    assert Spent(tokens=80_000).exhausted(budget, elapsed_ms=0) == "max_tokens"
    # The clock is the caller's, not the log's: the hung run this budget exists
    # to stop is precisely the one that appends nothing.
    assert Spent().exhausted(budget, elapsed_ms=1_800_000) == "timeout"


def test_spend_is_folded_from_the_log_not_carried_by_the_loop() -> None:
    """A run that survives a restart must not come back with a fresh allowance.

    So continuations, turns and tokens are all counted from the session's own
    events. The loop holds nothing: hand it the same log and it reaches the same
    conclusion, which is what lets a daemon resume one it did not start.
    """
    session, service = Session("g"), GoalService()
    goal = _open(session, service)

    service.continued(session, goal.id)
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append(
        "assistant/message",
        {
            **assistant_payload("done", "m1"),
            "usage": {
                "inputTokens": 900,
                "outputTokens": 100,
                "cacheReadTokens": 300,
                "cacheWriteTokens": 100,
            },
        },
        SurfaceIntent("append"),
    )
    service.continued(session, goal.id)

    spent = goals(session)[goal.id].spent
    assert spent.continuations == 2
    assert spent.turns == 1
    # All four terms, through `TokenUsage.total`: a two-term count would put
    # most of a cache-heavy run's input outside the budget.
    assert spent.tokens == 1_400


def test_a_second_goal_is_refused_while_one_is_open() -> None:
    """One open goal per session, which is what makes unaddressed events countable.

    `turn/end` and `assistant/message` carry no goal id and never will — they
    are the agent's records, not this seam's — so attributing them to *the* open
    goal is the only way turns and tokens are counted at all.
    """
    session, service = Session("g"), GoalService()
    _open(session, service)

    with pytest.raises(ValueError, match="already open"):
        service.set(session, Goal(id="g2", objective="something else"))

    service.settle(session, "g1", "achieved")
    assert service.open(session) is None
    assert service.set(session, Goal(id="g2", objective="now allowed")).id == "g2"


def test_a_failed_gate_is_not_re_run_against_a_tree_nobody_touched() -> None:
    """The row's second gate.

    A gate that failed against a tree the agent has not changed cannot have
    changed its mind, and re-running a slow suite to learn nothing spends the
    very budget that decides whether the run continues. An edit anywhere changes
    the hash, and then it runs.
    """
    session, service = Session("g"), GoalService()
    goal = _open(session, service, gates=["pytest"])

    service.record_gate(session, goal.id, gate="pytest", tree="tree-aaa", passed=False)

    assert service.unchanged_failure(session, goal.id, gate="pytest", tree="tree-aaa")
    assert not service.unchanged_failure(session, goal.id, gate="pytest", tree="tree-bbb")
    assert not service.unchanged_failure(session, goal.id, gate="mypy", tree="tree-aaa")


def test_a_passing_gate_is_never_memoized_as_a_reason_to_skip() -> None:
    """Only failures are remembered, deliberately.

    The rule is "an unchanged *failed* gate is not re-run". A pass is the last
    thing to take on trust when deciding a goal is achieved — and a tree hash
    covers the agent's work, not the clock, the network, or a dependency that
    changed underneath it.
    """
    session, service = Session("g"), GoalService()
    goal = _open(session, service, gates=["pytest"])

    service.record_gate(session, goal.id, gate="pytest", tree="tree-aaa", passed=True)
    assert not service.unchanged_failure(session, goal.id, gate="pytest", tree="tree-aaa")


def test_a_gate_with_no_fingerprint_is_always_re_run() -> None:
    """No worktree, no memo.

    An advisory-tier agent has no tree to hash, so `tree` is empty — and an
    empty fingerprint must never match another empty one, or every gate in every
    unfingerprintable session would answer from the first run forever.
    """
    session, service = Session("g"), GoalService()
    goal = _open(session, service, gates=["pytest"])

    service.record_gate(session, goal.id, gate="pytest", tree="", passed=False)
    assert not service.unchanged_failure(session, goal.id, gate="pytest", tree="")


def test_an_unknown_outcome_leaves_the_goal_open() -> None:
    """Read back against the declaration, not against a list restated here.

    `Outcome` constrains writers and dies at `event.data`, so the fold checks
    `get_args` — a log written by a build with a fifth outcome leaves the goal
    open rather than silently settling it as something this build invented.
    """
    session, service = Session("g"), GoalService()
    goal = _open(session, service)
    session.append("goal/settled", {"id": goal.id, "outcome": "who-knows"})

    assert goals(session)[goal.id].outcome is None
    assert service.open(session) is not None


@pytest.mark.anyio
async def test_the_command_opens_a_goal_and_reports_its_spend(mount: object) -> None:
    """`/autonomous` is a command, not a host handler (C2).

    It reaches `ctx.goals` and `ctx.shell` — the same governed surfaces a tool
    would use — so there is no path that starts an autonomous run without a
    record. The bare form reports rather than starting a second one, because
    "what is it doing" is what a person types when they come back to a session.
    """
    ctx = await mount()  # type: ignore[operator]
    session = ctx.sessions.create("cmd")

    opened = await ctx.commands.dispatch(
        "/autonomous make the tests pass -- pytest -q; mypy", session=session
    )
    assert "make the tests pass" in opened
    assert "pytest -q, mypy" in opened

    state = ctx.goals.open(session)
    assert state is not None and state.goal.gates == ["pytest -q", "mypy"]

    status = await ctx.commands.dispatch("/autonomous", session=session)
    assert "continuations 0/3" in status and "turns 0/12" in status

    again = await ctx.commands.dispatch("/autonomous something else", session=session)
    assert "already open" in again
    assert len(ctx.goals.states(session)) == 1, "a second goal was recorded anyway"


@pytest.mark.anyio
async def test_the_loop_continues_a_turn_until_a_budget_stops_it(mount: object) -> None:
    """The driver, which is a policy plugin on `agent/turn-stopping` (§6.7).

    Not a daemon and not a second loop: the hook the agent loop already fires,
    where a listener objects *by steering*. This is what makes every outcome
    reachable — before it, `/autonomous` could open a goal and report the spend
    of a loop that did not exist.

    A gate that always fails means the run can only end one way, which is the
    row's first acceptance gate: `budget_limited`, naming the limit that bound.
    """
    ctx = await mount({"id": "autonomous", "config": {"maxContinuations": 2}})  # type: ignore[operator]
    session = ctx.sessions.create("loop")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="fake-1"))

    await ctx.commands.dispatch("/autonomous fix it -- false", session=session, agent=agent)
    await agent.prompt("go")

    types = [event.type for event in session.events_from(0)]
    assert types.count("goal/continued") == 2, "the loop did not steer the agent onward"
    # *Steps*, not turns: `steer` delivers at the next step boundary, so a
    # continuation keeps the current turn alive rather than starting another.
    # That is what makes `max_continuations` the limit that binds an automatic
    # run — `max_turns` bounds the goal's whole life, including the turns a
    # person starts by prompting again.
    assert types.count("step/start") == 3, "continuations did not become steps"
    assert types.count("turn/start") == 1

    settled = next(e for e in session.events_from(0) if e.type == "goal/settled")
    assert settled.data["outcome"] == "budget_limited"
    assert settled.data["detail"] == "max_continuations", "the trace does not say which budget"
    assert ctx.goals.open(session) is None


@pytest.mark.anyio
async def test_a_run_whose_gates_pass_is_achieved_and_stops(mount: object) -> None:
    """The other ending, and the one that must not be reachable by accident.

    A gate that passes settles the goal as `achieved` and the turn is allowed to
    end — no continuation is spent, because there is nothing left to do.
    """
    ctx = await mount()  # type: ignore[operator]
    session = ctx.sessions.create("done")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="fake-1"))

    await ctx.commands.dispatch("/autonomous ship it -- true", session=session, agent=agent)
    await agent.prompt("go")

    types = [event.type for event in session.events_from(0)]
    assert "goal/continued" not in types, "a passing run spent a continuation anyway"
    settled = next(e for e in session.events_from(0) if e.type == "goal/settled")
    assert settled.data["outcome"] == "achieved"


@pytest.mark.anyio
async def test_stop_abandons_the_open_goal(mount: object) -> None:
    """`/autonomous stop` reaches the third outcome.

    It was unreachable: the "a goal is already open" branch returned first, so
    the message telling a person to use `stop` pointed at a verb that could not
    run, and `abandoned` had no production caller at all.
    """
    ctx = await mount()  # type: ignore[operator]
    session = ctx.sessions.create("stopped")

    await ctx.commands.dispatch("/autonomous something long", session=session)
    said = await ctx.commands.dispatch("/autonomous stop", session=session)

    assert "something long" in said
    assert ctx.goals.open(session) is None
    settled = next(e for e in session.events_from(0) if e.type == "goal/settled")
    assert settled.data["outcome"] == "abandoned"
