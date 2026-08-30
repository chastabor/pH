"""`ctx.goals` — an objective, a budget, and the gates that decide it (P5-07).

An autonomous run is a loop with three ways to stop: the gates pass, a budget
runs out, or a person cancels it. This seam holds all three as facts in the log,
so the loop that reads them can be restarted, resumed, or driven by a daemon
that was not running when the goal was set.

**Budgets are counted, not trusted.** `max_continuations`, `max_turns`,
`max_tokens` and a wall-clock timeout each stop the loop, and every one of them
is folded from the session's own events rather than from a counter the loop
keeps: a run that survives a daemon restart must not come back with a fresh
allowance. Exhaustion settles the goal as `budget_limited` — a named outcome
rather than silence, because "it stopped" and "it stopped because it ran out"
are different things to the person reading the trace, and only one of them
suggests raising the budget.

**A gate is a shell command and a fingerprint.** Quality gates are the reason an
autonomous run can be trusted to stop on its own: `pytest`, `mypy`, whatever the
deployment says "done" means. Each result is recorded against the **tree hash**
of the agent's work at the moment it ran (P4-09's `write-tree`, extracted for
exactly this), so a gate that failed against a tree the agent has not changed is
not run again — the answer cannot have changed, and re-running a slow suite to
learn nothing is how a budget is spent on nothing. An edit anywhere in the
worktree changes the hash and the gate runs.

**Tools and commands, never host handlers** (C2). Everything here is reached
through `ctx.goals` by a *command* a person types or a *tool* the model calls,
so the governed pipeline sees it. There is no host-side path that sets a goal
without a record.

@module ph.seams.goals
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast, get_args

from pydantic import Field

from ..cordis import Context, plugin
from ..llm.types import TokenUsage
from ..session import Session, SessionFoldCache
from ..wire import WireModel

__all__ = [
    "CONTINUED",
    "GATE",
    "SET",
    "SETTLED",
    "Budget",
    "Goal",
    "GoalService",
    "GoalState",
    "Limit",
    "Outcome",
    "Spent",
    "apply",
    "fold_goal_event",
    "goals",
    "open_goal",
]

log = logging.getLogger("ph.seams.goals")

SET = "goal/set"
CONTINUED = "goal/continued"
GATE = "goal/gate"
SETTLED = "goal/settled"

Limit: TypeAlias = Literal["max_continuations", "max_turns", "max_tokens", "timeout"]
"""Which budget stopped a run. Closed, and named for the field it belongs to.

A `Literal` rather than the bare strings it started as, for the reason `Outcome`
below is one: this name reaches a person through `goal/settled`'s `detail`, and
renaming `Budget.max_turns` would otherwise leave `"max_turns"` behind with
nothing to catch it.
"""

Outcome: TypeAlias = Literal["achieved", "budget_limited", "abandoned"]
"""How an autonomous run ended. Three, and they are not interchangeable.

`achieved` means every gate passed; `budget_limited` means one of the four
budgets ran out with gates still failing; `abandoned` means a person stopped it.
A loop that reported the second as the first would be claiming work it did not
do, which is the failure this whole layer exists to make impossible.
"""


class Budget(WireModel):
    """What one autonomous run may spend before it must stop.

    Four limits rather than one, because they fail differently: a model looping
    on the same edit exhausts continuations, a model that cannot stop talking
    exhausts turns, a long context exhausts tokens, and a hung gate exhausts the
    clock. One number could only catch whichever happened to bind first.
    """

    max_continuations: int = 3
    max_turns: int = 12
    max_tokens: int = 80_000
    timeout_ms: int = 30 * 60 * 1000


class Goal(WireModel):
    """An objective, the gates that decide it, and what it may spend."""

    id: str
    objective: str
    gates: list[str] = Field(default_factory=list)
    """Shell commands. All must pass for the goal to be `achieved`; an empty
    list means the model's own judgement decides, which is weaker and says so.

    A list, not a tuple: the session log refuses a tuple outright — *"tuple
    would come back as a list"* — because a value that changes type across a
    round trip makes a resumed session differ from the live one it replaced."""
    budget: Budget = Budget()


@dataclass(slots=True)
class Spent:
    """What a run has used, counted from the log.

    Mutable, and accumulated in place: the frozen version rebuilt a dataclass
    per event, which measured **259 ms against 5.2 ms** over a 200 000-event
    fold — fifty times the budget this seam's own cache exists to keep. The
    enclosing `GoalState` is already mutable and reassigned by the same loop, so
    freezing the inner record bought nothing but allocations.
    """

    continuations: int = 0
    turns: int = 0
    tokens: int = 0

    def exhausted(self, budget: Budget, *, elapsed_ms: int) -> Limit | None:
        """The first limit this run has reached, or `None`.

        Named rather than boolean: the trace says *which* budget stopped it,
        which is the difference between "raise the turn limit" and "this model
        is looping".

        `elapsed_ms` is passed in rather than folded, and that is the whole
        point of the fourth budget. The docstring's own example for why it
        exists — a hung gate — is precisely the case that appends **nothing**,
        so a wall-clock limit answered from event timestamps is blind to the one
        run it was added to stop. The caller reads a clock; the log supplies
        only the start.
        """
        if self.continuations >= budget.max_continuations:
            return "max_continuations"
        if self.turns >= budget.max_turns:
            return "max_turns"
        if self.tokens >= budget.max_tokens:
            return "max_tokens"
        if elapsed_ms >= budget.timeout_ms:
            return "timeout"
        return None


@dataclass(slots=True)
class GoalState:
    """A goal and what its log says has happened to it."""

    goal: Goal
    started_at: int = 0
    spent: Spent = field(default_factory=Spent)
    """Counted from the log. The wall clock is not here — see `exhausted`."""
    outcome: Outcome | None = None
    gates: dict[tuple[str, str], bool] = field(default_factory=dict)
    """`(command, tree) -> passed`. The memo that stops a gate re-running
    against work nobody has touched."""

    @property
    def settled(self) -> bool:
        return self.outcome is not None


_COUNTED = frozenset({SET, CONTINUED, GATE, SETTLED, "turn/end", "assistant/message"})
"""The six types this fold reads. Everything else is skipped on a set test.

A real log is mostly `assistant/chunk`, and without this every one of them paid
a `data.get`, an open-goal scan and a six-branch `elif` to be discarded —
measured at **597 ms against 53 ms** over a 500 000-event log.
"""


def fold_goal_event(found: dict[str, GoalState], event: Any) -> None:
    """Fold one event into the goal table, in place. The rules, in one place.

    Exported as a step rather than only as a loop, for `fold_subagent_event`'s
    reason: there are two consumers with different shapes — the whole-log fold
    and the cache's incremental `extend` — and sharing the loop instead of the
    step is what makes them two implementations of one projection (A11).
    """
    if event.type not in _COUNTED:
        return
    data = event.data
    if event.type == SET:
        goal = Goal.model_validate(dict(data))
        found[goal.id] = GoalState(goal=goal, started_at=int(event.time))
        return
    # A dict is insertion-ordered, so the fold's "which goal does an unaddressed
    # event belong to" and the service's "is one open" are the same question
    # over the same order — asked once, here, rather than written twice and
    # required to agree.
    current = found.get(str(data.get("id", ""))) if data.get("id") else open_goal(found)
    if current is None:
        return
    if event.type == CONTINUED:
        current.spent.continuations += 1
    elif event.type == GATE:
        current.gates[(str(data.get("gate", "")), str(data.get("tree", "")))] = bool(
            data.get("passed")
        )
    elif event.type == SETTLED:
        # Against `get_args`, the way `sandbox` checks a mode read back out of
        # `event.data`: the `Literal` constrains writers and dies at the payload
        # boundary, so the declaration is what a reader checks rather than a
        # list it restates.
        outcome = str(data.get("outcome", ""))
        current.outcome = cast(Outcome, outcome) if outcome in get_args(Outcome) else None
    elif event.type == "turn/end":
        current.spent.turns += 1
    elif "usage" in data:
        # Through `TokenUsage.total`: reading `inputTokens + outputTokens` off
        # the raw payload made this the third hand-written definition of
        # "tokens" in the tree and the only two-term one, so a cache-heavy run
        # spent most of its input outside the budget and `/autonomous`'s status
        # disagreed with the footer showing the same word.
        current.spent.tokens += TokenUsage.model_validate(data["usage"] or {}).total


def goals(session: Session) -> dict[str, GoalState]:
    """Every goal this log knows about — a fold.

    From `seed_length`, so a forked child does not inherit its parent's spend;
    and short-circuited on `Session.latest`, because most sessions have no goal
    at all and this is read from a loop between turns.
    """
    if session.latest(SET) is None:
        return {}
    found: dict[str, GoalState] = {}
    for event in session.events_from(session.header.seed_length or 0):
        fold_goal_event(found, event)
    return found


def extend_goals(
    previous: dict[str, GoalState], session: Session, from_index: int
) -> dict[str, GoalState]:
    """Fold only what arrived since the last read.

    The cache keys on `session.seq`, which moves on *every* event — so without
    this, the driver's per-turn read and `run_gates`' own `record_gate` between
    gates each re-folded the whole log: a three-gate pass measured 1 136 ms at
    500 000 events. The state is mutable and folded in place, so continuing is
    the same step applied to a shorter slice.
    """
    for event in session.events_from(from_index):
        fold_goal_event(previous, event)
    return previous


def open_goal(states: Mapping[str, GoalState]) -> GoalState | None:
    """The goal still being worked on, if there is one.

    `turn/end` and `assistant/message` carry no goal id and never will: they are
    the agent's records, not this seam's. Attributing them to *the* open goal is
    what makes turns and tokens countable at all, and there is only ever one
    because `set` refuses a second.
    """
    return next((state for state in reversed(list(states.values())) if not state.settled), None)


@dataclass(slots=True)
class GoalService:
    """`ctx.goals` — set a goal, record what it spends, decide when it is done.

    Cached like its siblings: the autonomous loop reads this between every turn,
    and a whole-log fold per read is what `Root.accepted` measured at 4.9 ms per
    call at 200 000 events.
    """

    _states: SessionFoldCache[dict[str, GoalState]] = field(
        default_factory=lambda: SessionFoldCache(goals, extend=extend_goals)
    )

    def states(self, session: Session) -> dict[str, GoalState]:
        return self._states.read(session)

    def forget_session(self, session_id: str) -> None:
        self._states.forget(session_id)

    def open(self, session: Session) -> GoalState | None:
        """The goal still being worked on, if there is one."""
        return open_goal(self.states(session))

    def set(self, session: Session, goal: Goal) -> Goal:
        """Record a goal. Refuses a second while one is open."""
        if self.open(session) is not None:
            raise ValueError("a goal is already open in this session")
        session.append(SET, goal.to_wire())
        return goal

    def continued(self, session: Session, goal_id: str) -> None:
        """Record that the loop is going round again — before it does.

        Write-ahead, like every other claim here: a continuation that ran and
        was not recorded is a budget the next daemon hands back.
        """
        session.append(CONTINUED, {"id": goal_id})

    def record_gate(
        self, session: Session, goal_id: str, *, gate: str, tree: str, passed: bool
    ) -> None:
        """Record a gate's verdict against the tree it ran on."""
        session.append(GATE, {"id": goal_id, "gate": gate, "tree": tree, "passed": passed})

    def settle(self, session: Session, goal_id: str, outcome: Outcome, *, detail: str = "") -> None:
        session.append(SETTLED, {"id": goal_id, "outcome": outcome, "detail": detail})

    def unchanged_failure(self, session: Session, goal_id: str, *, gate: str, tree: str) -> bool:
        """Whether this gate already failed against this exact tree.

        The question the caller actually asks, rather than a tri-state where
        `True` was unrepresentable and the one call site had to write
        `is False` — an identity test, because both `if x` and `if not x` were
        wrong.

        A **passing** result is deliberately not remembered: the rule is that an
        *unchanged failed* gate is not re-run, and a pass is the last thing to
        take on trust when declaring a goal achieved. An empty `tree` never
        matches, or every gate in an unfingerprintable session would answer
        forever from its first run.
        """
        state = self.states(session).get(goal_id)
        if state is None or not tree:
            return False
        return state.gates.get((gate, tree)) is False


@plugin("goals")
async def apply(ctx: Context, _config: Any) -> None:
    """Publish `ctx.goals`."""
    service = GoalService()
    ctx.provide("goals", service)
    ctx.on("session/disposed", lambda session: service.forget_session(session.id))
