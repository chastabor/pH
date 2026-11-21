"""`/autonomous`: work toward a goal until its gates pass or a budget stops it.

**A policy plugin on `agent/turn-stopping`** (§6.7). That is the whole design.
The driver is not a daemon, a scheduler or a second loop. It is a listener on
the boundary that the agent loop already fires. When a turn is about to end with
a goal still open, the listener asks three questions in order. It then lets the
turn stop, or steers the agent into another turn. `agent/turn-stopping` is
`serial`, and a listener objects *by steering* and does not reach into loop
state. That is why continuing needs no new machinery.

The three questions, in the order that keeps the answers correct:

1. **Is a budget spent?** If so, the run stops as `budget_limited` and names the
   budget. This question is first, because a run that has used up its allowance
   must not get one more "free" gate pass afterwards.
2. **Do the gates pass?** If all of them pass against the current tree, the run
   is `achieved`.
3. **Otherwise** the listener records a continuation and steers. This step turns
   "the model thinks it is done" into "the gates disagree, keep going".

Everything goes through `ctx.goals` and `ctx.shell`. The whole history of the
run is therefore in the session log, where `/revert` and the trajectory can see
it (C2): the objective, each continuation, each gate verdict and the ending.

**Gates are memoized against the tree hash of the worktree.** A gate that failed
against a tree that the agent has not touched cannot have a different result. To
run a test suite again and learn nothing spends the same budget that decides
whether the run continues. Where there is no worktree to fingerprint (the
`shared` tier), the memo never fires and every gate runs. That is the correct
degradation, and it is better than a memo keyed on nothing.

@module ph.commands.autonomous
"""

from __future__ import annotations

import secrets

from pydantic import Field

from ..agent.types import AgentDriver
from ..cordis import Context, plugin
from ..keys import COMMANDS, GOALS, SHELL, WORKSPACE
from ..llm.types import PluginSource, create_user_message
from ..seams.commands import CommandContext, CommandDefinition
from ..seams.goals import (
    Budget,
    Goal,
    GoalService,
    GoalState,
    TokenSource,
    own_tokens_only,
)
from ..seams.workspace import workspace_of
from ..session import Session, now_ms
from ..wire import WireModel

__all__ = ["Config", "apply", "run_gates"]

USAGE = "/autonomous <objective> [-- <gate>; <gate>]  ·  /autonomous stop"
"""The syntax, written one time for both the hint and the empty-goal reply. Two
spellings drift on whether the gates are optional."""


class Config(WireModel):
    """The budgets of an `/autonomous` run, as row config, so that a deployment
    can set them without editing this file.

    The budgets were pydantic field defaults before. That made the promise of
    `base.yaml`, "A later bundle or the profile of the user addresses these rows
    by id", impossible to keep for this row: there was nothing to address.
    `limits` takes every ceiling as a field for the same reason.
    """

    max_continuations: int = 3
    max_turns: int = 12
    max_tokens: int = 80_000
    timeout_ms: int = 30 * 60 * 1000
    token_sources: list[TokenSource] = Field(default_factory=own_tokens_only)
    """The sources whose tokens `maxTokens` charges. See `Budget.token_sources`
    (P2)."""


async def run_gates(
    ctx: Context, session: Session, agent: AgentDriver, state: GoalState, goals: GoalService
) -> tuple[bool, list[str]]:
    """Run the gates of this goal against the current tree. Returns `(passed, notes)`.

    The tree hash is taken **one time**, before any gate runs. Otherwise a suite
    that writes a cache file would change the fingerprint before the next gate
    runs, and two gates in one pass would be recorded against two different
    trees.
    """
    workspace = workspace_of(ctx, agent)
    # The hash comes through the seam and not from the git tier. The fingerprint
    # and the restore point are one token by design, so a call to `capture`
    # keeps a gate memo and a `/revert` from disagreeing about whether the work
    # changed. It is also what gives a jj deployment a fingerprint.
    tree = "" if workspace is None else (await ctx.require(WORKSPACE).capture(workspace) or "")
    notes: list[str] = []
    passed = True
    for gate in state.goal.gates:
        if goals.unchanged_failure(session, state.goal.id, gate=gate, tree=tree):
            notes.append(f"{gate}: still failing (unchanged since it last ran)")
            passed = False
            continue
        # The gate is recorded before it runs (S16), as the `!!` of a user is. A
        # gate that takes the daemon down still shows that it was started, and a
        # gate that the log cannot record does not run.
        result, _settled = await ctx.require(SHELL).run_recorded(
            gate, agent=agent, session=session, gate=state.goal.id
        )
        ok = result.exit_code == 0
        goals.record_gate(session, state.goal.id, gate=gate, tree=tree, passed=ok)
        notes.append(f"{gate}: {'passed' if ok else 'failed'}")
        passed = passed and ok
    return passed, notes


@plugin("autonomous", affects="environment", inject=[COMMANDS, GOALS, SHELL], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register `/autonomous`, and the turn-stopping policy that drives it."""

    def budget_of() -> Budget:
        """The budgets of the row, as the model that a `Goal` carries.

        `Config` and `Budget` declare the same five fields for different
        reasons. `Config` is what a deployment addresses by row id, and `Budget`
        is what a goal records. So this function validates again and does not
        map field by field. If it returned the dump, the two could drift and
        nothing would report it.
        """
        return Budget.model_validate(config.model_dump())

    async def keep_going(agent: AgentDriver, turn: int) -> None:
        """The driver: decide whether this turn is allowed to be the last one."""
        session = agent.session
        if session is None:
            return
        goals: GoalService = ctx.require(GOALS)
        state = goals.open(session)
        if state is None:
            return

        spent = goals.spent(session, state).exhausted(
            state.goal.budget, elapsed_ms=now_ms() - state.started_at
        )
        if spent is not None:
            goals.settle(session, state.goal.id, "budget_limited", detail=spent)
            return

        passed, notes = await run_gates(ctx, session, agent, state, goals)
        if passed:
            goals.settle(session, state.goal.id, "achieved", detail="; ".join(notes))
            return

        # Write-ahead: the continuation is recorded before the steer that spends
        # it. A crash between the two therefore costs an allowance, and does not
        # give a resumed run a free one.
        goals.continued(session, state.goal.id)
        agent.steer(
            create_user_message(
                content=[{"type": "text", "text": _nudge(state, notes)}],
                # The message is tagged as the message of this plugin, not of
                # the user. Every other message from a plugin to the inbox
                # carries a `PluginSource`, and the TUI draws an untagged
                # message as a **user** row. Without the tag, the transcript
                # would attribute the steering of the harness to whoever reads
                # it.
                source=PluginSource(
                    plugin="ph.commands.autonomous",
                    form="notice",
                    summary="the goal's gates are still failing",
                ).to_wire(),
            )
        )

    ctx.on("agent/turn-stopping", keep_going)

    async def autonomous(argument: str, invocation: CommandContext) -> str:
        session = invocation.session
        if session is None:
            return "no session"
        goals: GoalService = ctx.require(GOALS)
        objective, _, gate_text = argument.partition(" -- ")
        objective = objective.strip()
        state = goals.open(session)

        if objective == "stop":
            if state is None:
                return "no goal is open"
            goals.settle(session, state.goal.id, "abandoned")
            return f'stopped: "{state.goal.objective}"'
        if not objective:
            return _status(goals, session, state)
        if state is not None:
            return f'a goal is already open: "{state.goal.objective}". Use `/autonomous stop`.'

        goal = goals.set(
            session,
            Goal(
                id=f"goal-{secrets.token_hex(4)}",
                objective=objective,
                gates=[part.strip() for part in gate_text.split(";") if part.strip()],
                budget=budget_of(),
            ),
        )
        opened = goals.states(session)[goal.id]
        return _status(goals, session, opened)

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="autonomous",
            summary="Work toward a goal until its gates pass or a budget stops it.",
            argument_hint=USAGE.partition(" ")[2],
            run=autonomous,
        ),
        scope=ctx,
    )


def _nudge(state: GoalState, notes: list[str]) -> str:
    """What the agent is told when the gates disagree that it is finished.

    **The text deliberately leaves out the todo list**, which P5-16 added and
    then removed. The prompt layer already delivers the list. `tool-todo`
    registers a `todos` context, which reaches the model as a snapshot whenever
    the list changes, and `tool-todo` is the only producer of that context. A
    nudge can therefore fire only when the model already has the list in
    context, and a second copy here duplicated model-visible tokens on every
    continuation. The premise that this is "the one moment the loop composes a
    message" was false. The prompt layer composes the same message.
    """
    said = "; ".join(notes) if notes else "no gates are defined"
    return (
        f'Not done yet — the goal is "{state.goal.objective}".\n'
        f"Gate results: {said}.\nKeep working."
    )


def _status(goals: GoalService, session: Session, state: GoalState | None) -> str:
    """What the open goal has spent, including the spend of its sub-agents, or a
    statement that no goal is open."""
    if state is None:
        return f"no goal is open — {USAGE}"
    spent, budget = goals.spent(session, state), state.goal.budget
    gates = ", ".join(state.goal.gates) if state.goal.gates else "none — the model decides"
    return (
        f'working toward: "{state.goal.objective}"\n'
        f"gates: {gates}\n"
        f"continuations {spent.continuations}/{budget.max_continuations}"
        f"  turns {spent.turns}/{budget.max_turns}"
        f"  tokens {spent.charged_tokens(budget)}/{budget.max_tokens}"
    )
