"""`ctx.goals`: an objective, a budget, and the gates that decide it (P5-07).

An autonomous run is a loop with three ways to stop: the gates pass, a budget
runs out, or the user cancels it. This seam holds all three as facts in the
log. The loop that reads them can therefore be restarted, resumed, or driven by
a daemon that was not running when the goal was set.

**Budgets are counted, not trusted.** `max_continuations`, `max_turns`,
`max_tokens` and a wall-clock timeout each stop the loop. The first three are
folded from the events of the session itself, and the timeout runs from the
time of the `goal/set` record. None of them is a counter that the loop keeps,
because a run that survives a daemon restart must not come back with a fresh
allowance. Exhaustion settles the goal as `budget_limited`. That is a named
outcome and not silence, because "it stopped" and "it stopped because it ran
out" are different things to a reader of the trace, and only the second
suggests a larger budget.

**A gate is a shell command and a fingerprint.** Quality gates are the reason
why an autonomous run can be trusted to stop on its own: `pytest`, `mypy`, or
whatever the deployment defines as "done". Each result is recorded against the
**tree hash** of the work of the agent at the moment the gate ran. The hash is
the token that `ctx.workspace.capture` returns (P4-09): the hash of
`git write-tree` on the git tier, and a commit id on the jj tier. A gate that
failed against a tree that the agent has not changed is not run again. The
answer cannot have changed, and a slow suite that runs again to learn nothing
spends a budget on nothing. An edit anywhere in the worktree changes the hash,
and the gate runs.

**Tools and commands, never host handlers** (C2). A goal is set through
`ctx.goals` by a *command* that the user types or a *tool* that the model
calls, so the governed pipeline sees it. The one writer today is the
`autonomous` row (`ph.commands.autonomous`). Its `/autonomous` command sets and
stops a goal, and its `agent/turn-stopping` listener records the gates, the
continuations and the ending. The daemon and the subagent seam only read the
open goal. No host-side path sets a goal without a record.

@module ph.seams.goals
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, TypeAlias

from pydantic import Field

from ..cordis import Context, plugin
from ..json import as_bool, as_str
from ..keys import GOALS, SUBAGENTS
from ..session import Session, SessionEvent, SessionFoldCache
from ..session.writers import log_writer
from ..wire import WireModel, literal_lookup
from .invariants import contribute_fold_cache
from .token_meter import SPENDING_TYPES, event_tokens

_LOG = log_writer(__name__)

__all__ = [
    "CONTINUED",
    "GATE",
    "GOAL_OUTCOMES",
    "SET",
    "SETTLED",
    "Budget",
    "Goal",
    "GoalService",
    "GoalState",
    "Limit",
    "Outcome",
    "Spent",
    "TokenSource",
    "apply",
    "fold_goal_event",
    "goals",
    "open_goal",
    "own_tokens_only",
]

log = logging.getLogger("ph.seams.goals")

SET = "goal/set"
CONTINUED = "goal/continued"
GATE = "goal/gate"
SETTLED = "goal/settled"

Limit: TypeAlias = Literal["max_continuations", "max_turns", "max_tokens", "timeout"]
"""Which budget stopped a run. The set is closed, and each limit is named for
the field that it belongs to.

It is a `Literal` and not the bare strings that it started as, for the same
reason as `Outcome` below. This name reaches the user through the `detail` of
`goal/settled`. A rename of `Budget.max_turns` would otherwise leave
`"max_turns"` behind, and nothing would catch it.
"""

Outcome: TypeAlias = Literal["achieved", "budget_limited", "abandoned"]
"""How an autonomous run ended. There are three outcomes, and they are not
interchangeable.

`achieved` means that every gate passed. `budget_limited` means that one of the
four budgets ran out with gates still failing. `abandoned` means that the user
stopped the run. A loop that reported the second as the first would claim work
that it did not do. This whole layer exists to make that failure impossible.
"""

TokenSource: TypeAlias = Literal["own", "compaction", "decision", "children"]
"""Who spent the tokens that the log of a goal records (P2).

- `own`: the replies of the agent (`assistant/message`).
- `compaction`: the summarize calls that shortened its context
  (`compaction/summarized`).
- `decision`: the questions that a row had a decision model answer about it
  (`llm/decided`).
- `children`: the sub-agents admitted under the goal, and everything beneath
  them. This is read from their own logs (Phase 11), because nothing about a
  child is in this log.

The fold counts the first three, and `GoalService.spent` adds the fourth.
`Budget.token_sources` decides which of them `max_tokens` charges."""


def own_tokens_only() -> list[TokenSource]:
    """The default of `Budget.token_sources`: what the token budget has always counted."""
    return ["own"]


GOAL_OUTCOMES: Mapping[str, Outcome] = literal_lookup(Outcome)
"""Every `Outcome` by its own spelling. This is the read-side check. See
`literal_lookup`.

It is named for its alias and not the bare `OUTCOMES` that it started as,
because `ph.seams.approval` declares one too (`APPROVAL_OUTCOMES`). A module
that imported from both seams would otherwise have two names that collide,
and that collision is what the `as_int` rename existed to stop."""


class Budget(WireModel):
    """What one autonomous run may spend before it must stop.

    There are four limits and not one, because they fail differently. A model
    that loops on the same edit exhausts continuations. A model that cannot
    stop talking exhausts turns. A long context exhausts tokens. A hung gate
    exhausts the clock. One number could catch only the limit that happened to
    bind first.
    """

    max_continuations: int = 3
    max_turns: int = 12
    max_tokens: int = 80_000
    timeout_ms: int = 30 * 60 * 1000
    token_sources: list[TokenSource] = Field(default_factory=own_tokens_only)
    """The sources whose tokens `max_tokens` charges (P2). The default is the
    tokens of the agent itself, which is what this budget has always counted.
    Add `compaction`, `decision` and `children` for a ceiling on everything
    that the run cost.

    **This is decided here, not by the fold.** The fold used to answer the
    question by the record types that it read, which was only
    `assistant/message`. The summary of a compaction and the work of a child
    were then free, and nobody had chosen that."""


class Goal(WireModel):
    """An objective, the gates that decide it, and what it may spend."""

    id: str
    objective: str
    gates: list[str] = Field(default_factory=list)
    """Shell commands. All of them must pass for the goal to be `achieved`. An
    empty list means that the judgement of the model decides. That is weaker,
    and the status reply of `/autonomous` says so.

    A list and not a tuple. The session log refuses a tuple outright (*"tuple
    would come back as a list"*), because a value that changes type across a
    round trip makes a resumed session differ from the live session that it
    replaced."""
    budget: Budget = Budget()


@dataclass(slots=True)
class Spent:
    """What a run has used, counted from the log.

    It is mutable and accumulated in place. A frozen version would rebuild a
    dataclass for each event. The enclosing `GoalState` is already mutable and
    is reassigned by the same loop, so a frozen inner record would add
    allocations and nothing else.
    """

    continuations: int = 0
    turns: int = 0
    tokens: Counter[TokenSource] = field(default_factory=Counter)
    """What each source spent. See `TokenSource`. Every source is counted, and
    `charged_tokens` decides which of them the budget charges."""

    def charged_tokens(self, budget: Budget) -> int:
        """The tokens that `max_tokens` counts: the sources that the budget names,
        each counted one time."""
        return sum(self.tokens[source] for source in set(budget.token_sources))

    def exhausted(self, budget: Budget, *, elapsed_ms: int) -> Limit | None:
        """The first limit that this run has reached, or `None`.

        The answer is a name and not a boolean. The trace says *which* budget
        stopped the run, and that is the difference between "raise the turn
        limit" and "this model is looping".

        `elapsed_ms` is passed in and not folded. That is the purpose of the
        fourth budget. The docstring of `Budget` gives a hung gate as the
        reason for the clock, and a hung gate is exactly the case that appends
        **nothing**. A wall-clock limit that was answered from event timestamps
        would not see the one run that it was added to stop. The caller reads a
        clock, and the log supplies only the start.
        """
        if self.continuations >= budget.max_continuations:
            return "max_continuations"
        if self.turns >= budget.max_turns:
            return "max_turns"
        if self.charged_tokens(budget) >= budget.max_tokens:
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
    """Counted from the log. The wall clock is not here: see `exhausted`."""
    outcome: Outcome | None = None
    gates: dict[tuple[str, str], bool] = field(default_factory=dict)
    """`(command, tree) -> passed`. This memo stops a gate from running again
    against work that nobody has touched."""

    @property
    def settled(self) -> bool:
        return self.outcome is not None


_SOURCES: Mapping[str, TokenSource] = {
    "assistant/message": "own",
    "compaction/summarized": "compaction",
    "llm/decided": "decision",
}
"""The source that each token-spending record in the own log of a goal belongs
to. `children` is not a record here: see `GoalService.spent`.

The keys are exactly `SPENDING_TYPES`, which the account of a sub-agent also
reads. The import refuses when they differ, because a record that was added
there and not here would be counted in the account of a child and charged to no
budget."""

if _SOURCES.keys() != SPENDING_TYPES:
    raise RuntimeError(
        "ph.seams.goals: every record that spends tokens (SPENDING_TYPES) needs a "
        f"TokenSource here; differing: {sorted(_SOURCES.keys() ^ SPENDING_TYPES)}"
    )

_COUNTED = frozenset({SET, CONTINUED, GATE, SETTLED, "turn/end", *_SOURCES})
"""The types that this fold reads. Everything else is skipped by a set test.

A real log is mostly `assistant/chunk`. Without this set, every chunk pays for
a `data.get`, a scan for the open goal and the whole `elif` chain, and is then
discarded.
"""


def fold_goal_event(found: dict[str, GoalState], event: SessionEvent) -> None:
    """Fold one event into the goal table, in place. The rules are in this one
    place.

    It is exported as a step and not only as a loop, as `fold_child_event` is.
    Its consumers have different shapes: the whole-log fold (`goals`), the
    incremental `extend` of the cache (`extend_goals`), and `holds_of` in
    `ph.seams.live_index`. They share this step. If each had a loop with its
    own rules, they would be separate implementations of one projection (A11).
    """
    if event.type not in _COUNTED:
        return
    data = event.data
    if event.type == SET:
        goal = Goal.model_validate(dict(data))
        found[goal.id] = GoalState(goal=goal, started_at=int(event.time))
        return
    # A dict keeps insertion order. So "which goal does an event with no id
    # belong to" in the fold and "is a goal open" in the service are the same
    # question over the same order. It is asked one time, here, and is not
    # written twice and required to agree.
    current = found.get(as_str(data.get("id"))) if data.get("id") else open_goal(found)
    if current is None:
        return
    if event.type == CONTINUED:
        current.spent.continuations += 1
    elif event.type == GATE:
        current.gates[(as_str(data.get("gate")), as_str(data.get("tree")))] = as_bool(
            data.get("passed")
        )
    elif event.type == SETTLED:
        # The outcome goes through the lookup, as every other `Literal` that is
        # read from a payload does. `literal_lookup` gives the reason.
        current.outcome = GOAL_OUTCOMES.get(as_str(data.get("outcome")))
    elif event.type == "turn/end":
        current.spent.turns += 1
    elif event.type in _SOURCES:
        current.spent.tokens[_SOURCES[event.type]] += event_tokens(event)


def goals(session: Session) -> dict[str, GoalState]:
    """Every goal that this log knows about, as a fold.

    The fold reads `own_events`, which start after the inherited seed
    (`seed_length`), so a forked child does not inherit the spend of its
    parent. It returns early on `Session.latest`, because most sessions have no
    goal at all and a loop reads this between turns.
    """
    if session.latest(SET) is None:
        return {}
    found: dict[str, GoalState] = {}
    for event in session.own_events():
        fold_goal_event(found, event)
    return found


def extend_goals(
    previous: dict[str, GoalState], session: Session, from_index: int
) -> dict[str, GoalState]:
    """Fold only the events that arrived since the last read.

    The cache keys on `session.seq`, which moves on *every* event. Without this
    function, the read of the driver for each turn and the `record_gate` of
    `run_gates` between gates each fold the whole log again. The state is
    mutable and folded in place, so to continue is to apply the same step to a
    shorter slice.
    """
    for event in session.events_from(from_index):
        fold_goal_event(previous, event)
    return previous


def open_goal(states: Mapping[str, GoalState]) -> GoalState | None:
    """The goal that is still being worked on, if there is one.

    `turn/end` and `assistant/message` carry no goal id and never will. They
    are records of the agent and not of this seam. Turns and tokens can be
    counted only because these records are attributed to *the* open goal.
    There is only ever one open goal, because `set` refuses a second.
    """
    return next((state for state in reversed(list(states.values())) if not state.settled), None)


@dataclass(slots=True)
class GoalService:
    """`ctx.goals`: set a goal, record what it spends, decide when it is done.

    It is cached as its sibling services are. The autonomous loop reads it
    between every turn, and a fold of the whole log for each read is the cost
    that the daemon measured for `Root.accepted`. That set is now the
    `CLIENT_COMMAND` index of the journal.
    """

    ctx: Context | None = None
    """The context through which `spent` asks for the sub-agents whose spend a
    goal includes."""
    _states: SessionFoldCache[dict[str, GoalState]] = field(
        default_factory=lambda: SessionFoldCache(goals, extend=extend_goals)
    )

    def states(self, session: Session) -> dict[str, GoalState]:
        return self._states.read(session)

    def open(self, session: Session) -> GoalState | None:
        """The goal still being worked on, if there is one."""
        return open_goal(self.states(session))

    def spent(self, session: Session, state: GoalState) -> Spent:
        """What the goal of `state` has spent: the fold of its own log, and what the
        sub-agents admitted under it spent, including everything beneath them
        (`children`).

        The two are joined here, at read time, and not folded. The answers of a
        child are in the own log of the child (Phase 11). A fold of this log
        that read another log would be a projection that the cache could not
        key. The fold of the goal itself stays pure. This value is a copy of
        it, with the spend of the children beside it.
        """
        subagents = self.ctx.get(SUBAGENTS) if self.ctx is not None else None
        tokens = Counter(state.spent.tokens)
        if subagents is not None:
            tokens["children"] = subagents.delegated_tokens(session.id, state.goal.id)
        return replace(state.spent, tokens=tokens)

    def set(self, session: Session, goal: Goal) -> Goal:
        """Record a goal. Refuses a second while one is open."""
        if self.open(session) is not None:
            raise ValueError("a goal is already open in this session")
        _LOG.append(session, SET, goal.to_wire())
        return goal

    def continued(self, session: Session, goal_id: str) -> None:
        """Record that the loop goes round again, before it does.

        The record is written ahead, as every other claim here is. A
        continuation that ran and was not recorded is an allowance that the
        next daemon hands back.
        """
        _LOG.append(session, CONTINUED, {"id": goal_id})

    def record_gate(
        self, session: Session, goal_id: str, *, gate: str, tree: str, passed: bool
    ) -> None:
        """Record a gate's verdict against the tree it ran on."""
        _LOG.append(session, GATE, {"id": goal_id, "gate": gate, "tree": tree, "passed": passed})

    def settle(self, session: Session, goal_id: str, outcome: Outcome, *, detail: str = "") -> None:
        _LOG.append(session, SETTLED, {"id": goal_id, "outcome": outcome, "detail": detail})

    def unchanged_failure(self, session: Session, goal_id: str, *, gate: str, tree: str) -> bool:
        """Whether this gate already failed against this exact tree.

        This is the question that the caller actually asks. An earlier version
        returned a tri-state in which `True` could not occur, and the one call
        site had to write `is False`. That was an identity test, because both
        `if x` and `if not x` were wrong.

        A **passing** result is deliberately not answered from the memo. The
        rule is that an *unchanged failed* gate is not run again, and a pass is
        the last thing to take on trust when a goal is declared achieved. An
        empty `tree` never matches. Otherwise every gate in a session that
        cannot be fingerprinted would answer forever from its first run.
        """
        state = self.states(session).get(goal_id)
        if state is None or not tree:
            return False
        return state.gates.get((gate, tree)) is False


@plugin("goals", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Publish `ctx.goals`."""
    service = GoalService(ctx=ctx)
    ctx.provide(GOALS, service)
    # This fold is worth polling, because it decides when a run has spent its
    # budget. A table that drifted low gives an autonomous loop turns that it
    # has already used.
    contribute_fold_cache(
        ctx, id="goal-fold-cache", subject="goal table", stale=service._states.stale
    )
