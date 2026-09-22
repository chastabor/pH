"""`skill-steps` — a skill that is a procedure, and a loop that finishes it (P7-18).

A `SKILL.md` may declare `steps:`. Reading such a skill seeds them into the todo
list as entries the model **may mark done and may not delete**, and a listener on
`agent/turn-stopping` objects while any of them is still startable. That is the
whole of the feature, and none of it is new machinery.

**The loop already had the walker.** `commands/autonomous.py` says it best about
its own row: *"the driver is not a daemon, a scheduler or a second loop, it is a
listener on the boundary the agent loop already fires… a listener objects **by
steering** rather than by reaching into loop state, which is why continuing costs
no new machinery."* `/autonomous` asks whether a goal's gates pass; this asks
whether the procedure is finished. Same boundary, same `agent.steer`, same
`PluginSource` tag so the transcript does not attribute the harness's nudge to
the person reading it.

**What had to be added was provenance, not a walker.** `write_todos` replaces the
*whole* list, so steps seeded into it were the model's to delete on its next
write — and a listener enforcing against that list would have been enforcing
against nothing. `tool-todo`'s `SKILL` source and `_carried` are the rule that
closes it; this row is what puts entries there for it to protect.

**Two limits, and they are the point rather than an apology.** A *loop* step is
not a *plan* step: `agent/turn-stopping` fires when a turn is about to end, and a
plan step spans many, so this asserts a boundary invariant — work remains that
can begin — and never claims to know which step is running. And it holds the
model to *its own accepted plan*, not to reality (P5-16): marking a step done is
still the model's word. Gates that check the world are `ctx.goals` and
`ctx.approval`, and a procedure that needs one says so in the step's own text.

@module ph_stabilize.skill_steps
"""

from __future__ import annotations

import logging
from itertools import islice
from typing import Any

from pydantic import Field, ValidationError

from ph.agent.types import AgentDriver
from ph.cordis import Context, plugin
from ph.json import as_int, as_obj, as_str
from ph.llm.types import PluginSource, create_user_message
from ph.session import Session
from ph.text import count_of
from ph.wire import WireModel

from .todo import (
    MAX_TODOS,
    SKILL,
    PlanError,
    TodoItem,
    _checked,
    outstanding_steps,
    startable,
    steps_of,
    todos_of,
)

__all__ = ["Config", "apply", "latest_skill_budget", "nudges_since", "seeded", "steer_text"]

log = logging.getLogger("ph_stabilize.skill_steps")

PLUGIN = "ph_stabilize.skill_steps"
"""This row's name on the messages it steers with — the key `nudges_since`
reads them back by, so the tag and the fold cannot drift apart."""

MAX_NAMED = 3
"""How many outstanding steps the steer names before it counts the rest.

A nudge is read by a model that already has the list in its context — it is a
pointer, not a second copy of the plan, and a twenty-line reminder every time a
turn tries to end is how a steer becomes noise the model learns to skim."""

MAX_NUDGES = 3
"""How many times this row will steer without the plan moving, by default.

**The ceiling the shape this row copies has and it did not.** `/autonomous` is
the same listener on the same boundary and carries four — continuations, turns,
tokens, wall clock — because a driver whose only exit is the model complying is
not bounded at all. The loop has no step cap of its own (`limits` ships with
`turn_limit` unset, deliberately), so without this a model that will not mark a
step done is steered for as long as the session lives.

Counted **since the list last changed**, not since the turn began, which is what
makes it a stall detector rather than a quota: any `write_todos` — marking a step
done, adding an entry, re-planning — resets it, so a run making progress is never
cut off, and one going in circles stands down and lets the person see the list.

The default, not the rule (D16): a profile sets its own with `maxNudges`, and a
skill may set one for its procedure with `max-nudges` — see `keep_going`."""

BUDGET = "skill-steps/budget"
"""The event recording the budget of the procedure seeded last (D16).

A record of its own rather than a field on the seeding `todo/write`: the model's
next `write_todos` replaces that event, so a budget stored on it would last
exactly one plan. Nor a field on the seeded entries, which `tool-todo` echoes
back to the model on every write — a harness number paid for in tokens forever."""


class Config(WireModel):
    """Row config: the nudge budget a profile gives every procedure (D16)."""

    max_nudges: int = Field(default=MAX_NUDGES, ge=0)
    """Nudges without the plan moving before the row stands down. A skill's own
    `max-nudges` overrides it for that skill's procedure; `0` seeds steps but
    never steers."""


def seeded(current: list[dict[str, Any]], steps: list[str]) -> list[dict[str, Any]] | None:
    """`current` with `steps` appended as a skill's, or `None` with a reason logged.

    **Idempotent by content**, because reading a skill twice is ordinary — a
    model re-reads instructions it half-remembers — and a second copy of the
    procedure would be a plan that can never be finished.

    Appended rather than merged into position: the model's own entries stay where
    it put them, and a procedure that arrives mid-session is work added to the
    end rather than a plan rewritten underneath somebody.

    Sequential `requires` within the skill's own steps, and only within them —
    two procedures read in one session are two orderings, not one queue. The
    chain follows the order the *skill* declared rather than the order of what is
    missing, so a re-read that finds one step already present links the rest
    around the gap instead of over it (D13).

    **Built through `TodoItem` and checked by `_checked`, because this is the
    list's second writer.** `write_todos` is bounded by its own schema and its own
    coherence rule; an entry appended straight to the log is bounded by neither,
    and the failure is not a cosmetic one: a step longer than `MAX_TODO_CONTENT`,
    or one that pushes the list past `MAX_TODOS`, seeds fine and then makes
    *every* later `write_todos` fail validation — while `_carried` refuses any
    write that drops it. The model would be locked out of its own plan with the
    row steering it to keep going. Refusing to seed is the recoverable answer,
    and the author gets a warning naming the skill's file.

    All or nothing for the same reason: half a procedure is a plan nobody wrote.
    """
    already = set(steps_of(current))
    wanted = [step for step in steps if step not in already]
    if not wanted:
        return None
    if len(current) + len(wanted) > MAX_TODOS:
        log.warning(
            "ph_stabilize.skill_steps: seeding %s would put the list past %s entries; not seeding",
            count_of(len(wanted), "step"),
            MAX_TODOS,
        )
        return None
    grown = list(current)
    # **The chain follows the skill's own order, not the gap** (D13). Built over
    # `wanted`, a procedure whose middle step was already in the list linked the
    # step after it to the step *before* — `[one, three]` with `three` waiting on
    # `one` — so the ordering the skill declared was quietly replaced by one that
    # let `three` start while `two` was still pending. The predecessor is always
    # nameable: every step is either already present or being added now.
    predecessor = {step: steps[index - 1] for index, step in enumerate(steps) if index}
    for step in wanted:
        try:
            entry = TodoItem(
                content=step,
                status="pending",
                requires=[predecessor[step]] if step in predecessor else [],
            ).model_dump(mode="json")
        except ValidationError as error:
            log.warning("ph_stabilize.skill_steps: a step cannot be a todo entry: %s", error)
            return None
        # `source` is out of band on the dict rather than a `TodoItem` field, the
        # same shape `worked` uses: a field the model could write is a field it
        # could label away.
        grown.append({**entry, "source": SKILL})
    try:
        # One authority for "is this list coherent", rather than a second
        # collision check here. `_checked` already refuses two entries sharing
        # content — which is what a step whose text the model has used would be,
        # and which `requires` could not then name unambiguously.
        _checked(grown)
    except PlanError as refusal:
        log.warning("ph_stabilize.skill_steps: not seeding — %s", refusal)
        return None
    return grown


def steer_text(outstanding: list[str], blocked: int) -> str:
    """What the model is told when it tries to stop with a procedure unfinished."""
    named = ", ".join(repr(one) for one in outstanding[:MAX_NAMED])
    rest = len(outstanding) - MAX_NAMED
    more = f", and {rest} more" if rest > 0 else ""
    waiting = ""
    if blocked:
        verb = "waits" if blocked == 1 else "wait"
        waiting = f" {count_of(blocked, 'further step')} {verb} on these."
    return (
        f"A skill you read set out a procedure and it is not finished: {named}{more} "
        f"can be started now.{waiting} Continue with it, or if a step genuinely does not "
        "apply, mark it completed and say why in your next message."
    )


def latest_skill_budget(session: Session) -> int | None:
    """The budget the latest seeded skill recorded, or `None` — a fact, not a rule.

    `None` both when no skill has seeded and when the latest one set no budget
    (it records `null`). Which record governs, and what `None` falls back to, is
    `keep_going`'s to decide (P4). Folded from the log, for `nudges_since`'s
    reason: a budget held on the listener would be gone after a resume.
    """
    event = session.latest(BUDGET)
    if event is None:
        return None
    recorded = event.data.get("maxNudges")
    return None if recorded is None else as_int(recorded)


def nudges_since(session: Session, seq: int, *, up_to: int | None = None) -> int:
    """How many times this row has steered after `seq` — a count, not a window.

    Folded from the log rather than counted on the row, for P5-04's reason: a
    counter on a listener starts at zero after a resume or a passivation, and the
    stall it bounds is exactly the kind of run that outlives a process. The steer
    lands in the log as a `user/message` carrying this row's `PluginSource`, so
    the tag *is* the record — there is nothing to keep in step with it.

    Where the count starts is the caller's rule (P4): it was "since the todo list
    last changed", written in here, where it read as a fact about the log rather
    than the stall detector's policy. `up_to` stops the walk once the count
    reaches it — the only question `keep_going` asks is "is the budget spent",
    and a run that stood down (or a skill whose budget is `0`) would otherwise
    rescan an ever-growing tail on every turn.
    """
    # By index through `Session.at`, not `events_from`: that copies the whole
    # tail before the first event is looked at, so `up_to` saved the filtering
    # and not the walk — and the tail of a run that stood down only grows.
    events = (session.at(index) for index in range(seq + 1, session.seq))
    nudges = (
        event
        for event in events
        if event is not None
        and event.type == "user/message"
        and as_str(as_obj(event.data.get("source")).get("plugin")) == PLUGIN
    )
    return sum(1 for _ in islice(nudges, up_to))


@plugin("skill-steps", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Seed a read skill's steps, and object while they are unfinished.

    No `inject`: the body registers two listeners and touches no service. `inject`
    is an activation gate — a row is reported inactive on an unmet key and its
    scope unwinds when the service goes — so naming `tools` here would tie this
    row's life to something it never calls.
    """

    def on_skill_read(payload: Any) -> None:  # noqa: ANN401
        session = payload.get("session")
        # The *rendered* steps off the payload, not `skill.steps`: an author may
        # write `Run {{parameters.gate}}` in a step, and the arguments that fill
        # it in belong to the call that read the skill — which this listener,
        # firing after the fact, does not have.
        steps = [str(one) for one in payload.get("steps") or ()]
        if not steps or session is None:
            return
        grown = seeded(todos_of(session), steps)
        if grown is None:
            return
        # The same event the tool writes, because it means one thing — "the list
        # is now this" — and a second type would give `todos_of` two things to
        # fold and the sidebar two things to draw.
        session.append("todo/write", {"todos": grown})
        # Recorded even when `null`: that is what hands a later procedure back
        # to the profile's budget rather than the previous skill's.
        session.append(
            BUDGET, {"skill": payload["skill"].name, "maxNudges": payload.get("max_nudges")}
        )

    async def keep_going(agent: AgentDriver, turn: int) -> None:
        session = agent.session
        if session is None:
            return
        # Frozen, and before anything else: this fires once per turn for every
        # session in the deployment, and a session that never read a skill must
        # not pay a thaw of somebody's hundred-entry list to find that out.
        outstanding = outstanding_steps(session)
        if not outstanding:
            return
        # The stall detector's two rules, here where they are applied (P4).
        # **The procedure seeded last governs**, because a nudge names whatever
        # is startable and cannot be charged to one skill — and the latest read
        # is the procedure the model was most recently asked to follow. One that
        # set no budget hands the row back to the profile's rather than
        # inheriting the one before it.
        recorded = latest_skill_budget(session)
        budget = config.max_nudges if recorded is None else recorded
        # **Counted since the plan last changed**, so any `write_todos` — a step
        # marked done, an entry added, a re-plan — earns fresh nudges, and only
        # a run going in circles reaches the budget.
        plan = session.latest("todo/write")
        if nudges_since(session, plan.seq if plan else -1, up_to=budget) >= budget:
            # Stood down rather than steering into a wall.
            log.info(
                "ph_stabilize.skill_steps: %s with the plan unchanged; standing down",
                count_of(budget, "nudge"),
            )
            return
        todos = todos_of(session)
        ready = [step for step in startable(todos) if step in outstanding]
        if not ready:
            # A shape guard, and said as one rather than dressed up as a policy:
            # `seeded` writes a *sequential* chain, so while anything is
            # unfinished its earliest link is waiting on nothing and this branch
            # cannot be reached by the entries this row writes. It is here
            # because a steer naming no step is worse than no steer at all, and
            # the list is a fold of a log that a hand-edit, a future seeding
            # shape or a second row could put another arrangement into.
            return
        agent.steer(
            create_user_message(
                content=[
                    {"type": "text", "text": steer_text(ready, len(outstanding) - len(ready))}
                ],
                source=PluginSource(
                    plugin=PLUGIN,
                    form="notice",
                    summary="a skill's procedure is unfinished",
                ).to_wire(),
            )
        )

    ctx.on("skills/read", on_skill_read)
    ctx.on("agent/turn-stopping", keep_going)
