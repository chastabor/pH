"""`/autonomous` — work toward a goal until its gates pass or a budget stops it.

**A policy plugin on `agent/turn-stopping`** (§6.7), which is the whole design:
the driver is not a daemon, a scheduler or a second loop, it is a listener on
the boundary the agent loop already fires. When a turn is about to end with a
goal still open, this asks three questions in order and either lets the turn
stop or steers the agent into another one. `agent/turn-stopping` is `serial` and
a listener objects *by steering* rather than by reaching into loop state, which
is why continuing costs no new machinery.

The three questions, in the order that makes the answers honest:

1. **Is a budget spent?** If so the run stops as `budget_limited`, naming which
   one. Asked first, because a run that has exhausted its allowance must not get
   one more "free" gate pass smuggled in after the fact.
2. **Do the gates pass?** All of them, against the current tree → `achieved`.
3. **Otherwise** record a continuation and steer, which is what turns "the
   model thinks it is done" into "the gates disagree, keep going".

Everything reaches `ctx.goals` and `ctx.shell`, so the run's whole history — the
objective, each continuation, each gate verdict and the ending — is in the
session log where `/revert` and the trajectory can see it (C2).

**Gates are memoized against the worktree's tree hash.** A gate that failed
against a tree the agent has not touched cannot have changed its mind, and
re-running a test suite to learn nothing spends the very budget that decides
whether the run continues. Where there is no worktree to fingerprint — the
`shared` tier — the memo simply never fires and every gate runs, which is the
honest degradation rather than a memo keyed on nothing.

@module ph.commands.autonomous
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

from ..cordis import Context, plugin
from ..llm.types import create_user_message
from ..seams.commands import CommandContext, CommandDefinition
from ..seams.goals import Goal, GoalService, GoalState
from ..seams.workspace import workspace_of
from ..seams.workspace_git import tree_hash
from ..session import Session, now_ms
from ..wire import WireModel

log = logging.getLogger("ph.commands.autonomous")

__all__ = ["Config", "apply", "run_gates"]

USAGE = "/autonomous <objective> [-- <gate>; <gate>]  ·  /autonomous stop"
"""The syntax, once. It was written twice — the hint and the empty-goal reply —
and the two had already drifted about whether the gates are optional."""


class Config(WireModel):
    """Row config, so a deployment can set the budgets without editing this file.

    They were pydantic field defaults, which made `base.yaml`'s own promise —
    "a later bundle or the user's profile addresses these rows by id" —
    unkeepable for this row: there was nothing to address. `limits` takes every
    ceiling as a field for exactly this reason.
    """

    max_continuations: int = 3
    max_turns: int = 12
    max_tokens: int = 80_000
    timeout_ms: int = 30 * 60 * 1000


async def run_gates(
    ctx: Context, session: Session, agent: Any, state: GoalState, goals: GoalService
) -> tuple[bool, list[str]]:
    """Run this goal's gates against the current tree. Returns `(passed, notes)`.

    The tree hash is taken **once**, before any gate runs: a suite that writes a
    cache file would otherwise change the fingerprint underneath the gate that
    follows it, and two gates in one pass would be recorded against two
    different trees.
    """
    workspace = workspace_of(ctx, agent)
    tree = "" if workspace is None else (await tree_hash(ctx, workspace) or "")
    notes: list[str] = []
    passed = True
    for gate in state.goal.gates:
        if goals.unchanged_failure(session, state.goal.id, gate=gate, tree=tree):
            notes.append(f"{gate}: still failing (unchanged since it last ran)")
            passed = False
            continue
        result = await ctx.shell.run(gate, agent=agent)
        ok = result.exit_code == 0
        goals.record_gate(session, state.goal.id, gate=gate, tree=tree, passed=ok)
        notes.append(f"{gate}: {'passed' if ok else 'failed'}")
        passed = passed and ok
    return passed, notes


@plugin("autonomous", inject=["commands", "goals", "shell"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register `/autonomous`, and the turn-stopping policy that drives it."""

    def budget_of() -> dict[str, int]:
        return config.model_dump()

    async def keep_going(agent: Any, turn: int) -> None:
        """The driver: decide whether this turn is allowed to be the last one."""
        session = getattr(agent, "session", None)
        if session is None:
            return
        goals: GoalService = ctx.goals
        state = goals.open(session)
        if state is None:
            return

        spent = state.spent.exhausted(state.goal.budget, elapsed_ms=now_ms() - state.started_at)
        if spent is not None:
            goals.settle(session, state.goal.id, "budget_limited", detail=spent)
            return

        passed, notes = await run_gates(ctx, session, agent, state, goals)
        if passed:
            goals.settle(session, state.goal.id, "achieved", detail="; ".join(notes))
            return

        # Write-ahead: the continuation is recorded before the steer that spends
        # it, so a crash between the two costs an allowance rather than handing
        # a resumed run a free one.
        goals.continued(session, state.goal.id)
        agent.steer(
            create_user_message(
                content=[{"type": "text", "text": _nudge(state, notes)}],
                source={"kind": "user"},
            )
        )

    ctx.on("agent/turn-stopping", keep_going)

    async def autonomous(argument: str, invocation: CommandContext) -> str:
        session = invocation.session
        if session is None:
            return "no session"
        goals: GoalService = ctx.goals
        objective, _, gate_text = argument.partition(" -- ")
        objective = objective.strip()
        state = goals.open(session)

        if objective == "stop":
            if state is None:
                return "no goal is open"
            goals.settle(session, state.goal.id, "abandoned")
            return f'stopped: "{state.goal.objective}"'
        if not objective:
            return _status(state)
        if state is not None:
            return f'a goal is already open: "{state.goal.objective}". Use `/autonomous stop`.'

        goal = goals.set(
            session,
            Goal(
                id=f"goal-{secrets.token_hex(4)}",
                objective=objective,
                gates=[part.strip() for part in gate_text.split(";") if part.strip()],
                budget=budget_of(),  # type: ignore[arg-type]
            ),
        )
        return _status(goals.states(session)[goal.id])

    ctx.commands.register(
        CommandDefinition(
            name="autonomous",
            summary="Work toward a goal until its gates pass or a budget stops it.",
            argument_hint=USAGE.partition(" ")[2],
            run=autonomous,
        ),
        scope=ctx,
    )


def _nudge(state: GoalState, notes: list[str]) -> str:
    """What the agent is told when the gates disagree that it is finished."""
    said = "; ".join(notes) if notes else "no gates are defined"
    return (
        f'Not done yet — the goal is "{state.goal.objective}".\n'
        f"Gate results: {said}.\nKeep working."
    )


def _status(state: GoalState | None) -> str:
    """What the open goal has spent, or that there is none."""
    if state is None:
        return f"no goal is open — {USAGE}"
    spent, budget = state.spent, state.goal.budget
    gates = ", ".join(state.goal.gates) if state.goal.gates else "none — the model decides"
    return (
        f'working toward: "{state.goal.objective}"\n'
        f"gates: {gates}\n"
        f"continuations {spent.continuations}/{budget.max_continuations}"
        f"  turns {spent.turns}/{budget.max_turns}"
        f"  tokens {spent.tokens}/{budget.max_tokens}"
    )
