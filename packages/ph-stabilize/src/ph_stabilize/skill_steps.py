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
from typing import Any

from pydantic import ValidationError

from ph.cordis import Context, plugin
from ph.llm.types import PluginSource, create_user_message
from ph.session import Session
from ph.text import count_of

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

__all__ = ["apply", "nudges_since_plan", "seeded", "steer_text"]

log = logging.getLogger("ph_stabilize.skill_steps")

PLUGIN = "ph_stabilize.skill_steps"
"""This row's name on the messages it steers with — the key `nudges_since_plan`
reads them back by, so the tag and the fold cannot drift apart."""

MAX_NAMED = 3
"""How many outstanding steps the steer names before it counts the rest.

A nudge is read by a model that already has the list in its context — it is a
pointer, not a second copy of the plan, and a twenty-line reminder every time a
turn tries to end is how a steer becomes noise the model learns to skim."""

MAX_NUDGES = 3
"""How many times this row will steer without the plan moving.

**The ceiling the shape this row copies has and it did not.** `/autonomous` is
the same listener on the same boundary and carries four — continuations, turns,
tokens, wall clock — because a driver whose only exit is the model complying is
not bounded at all. The loop has no step cap of its own (`limits` ships with
`turn_limit` unset, deliberately), so without this a model that will not mark a
step done is steered for as long as the session lives.

Counted **since the list last changed**, not since the turn began, which is what
makes it a stall detector rather than a quota: any `write_todos` — marking a step
done, adding an entry, re-planning — resets it, so a run making progress is never
cut off, and one going in circles stands down and lets the person see the list."""


def seeded(current: list[dict[str, Any]], steps: list[str]) -> list[dict[str, Any]] | None:
    """`current` with `steps` appended as a skill's, or `None` with a reason logged.

    **Idempotent by content**, because reading a skill twice is ordinary — a
    model re-reads instructions it half-remembers — and a second copy of the
    procedure would be a plan that can never be finished.

    Appended rather than merged into position: the model's own entries stay where
    it put them, and a procedure that arrives mid-session is work added to the
    end rather than a plan rewritten underneath somebody.

    Sequential `requires` within the skill's own steps, and only within them —
    two procedures read in one session are two orderings, not one queue.

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
    for index, step in enumerate(wanted):
        try:
            entry = TodoItem(
                content=step,
                status="pending",
                requires=[wanted[index - 1]] if index else [],
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


def nudges_since_plan(session: Session) -> int:
    """How many times this row has steered since the todo list last changed.

    Folded from the log rather than counted on the row, for P5-04's reason: a
    counter on a listener starts at zero after a resume or a passivation, and the
    stall it bounds is exactly the kind of run that outlives a process. The steer
    lands in the log as a `user/message` carrying this row's `PluginSource`, so
    the tag *is* the record — there is nothing to keep in step with it.

    From the tail: the previous `todo/write` is at most a few turns back, and
    this is only asked when the row is about to steer.
    """
    previous = session.latest("todo/write")
    return sum(
        1
        for event in session.events_from((previous.seq if previous else -1) + 1)
        if event.type == "user/message"
        and str((event.data.get("source") or {}).get("plugin")) == PLUGIN
    )


@plugin("skill-steps")
async def apply(ctx: Context, config: Any) -> None:
    """Seed a read skill's steps, and object while they are unfinished.

    No `inject`: the body registers two listeners and touches no service. `inject`
    is an activation gate — a row is reported inactive on an unmet key and its
    scope unwinds when the service goes — so naming `tools` here would tie this
    row's life to something it never calls.
    """

    def on_skill_read(payload: Any) -> None:
        session = payload.get("session")
        # The *rendered* steps off the payload, not `skill.steps`: an author may
        # write `Run {{parameters.gate}}` in a step, and the arguments that fill
        # it in belong to the call that read the skill — which this listener,
        # firing after the fact, does not have.
        steps = [str(one) for one in payload.get("steps") or ()]
        if not steps or not isinstance(session, Session):
            return
        grown = seeded(todos_of(session), steps)
        if grown is None:
            return
        # The same event the tool writes, because it means one thing — "the list
        # is now this" — and a second type would give `todos_of` two things to
        # fold and the sidebar two things to draw.
        session.append("todo/write", {"todos": grown})

    async def keep_going(agent: Any, turn: int) -> None:
        session = getattr(agent, "session", None)
        if not isinstance(session, Session):
            return
        # Frozen, and before anything else: this fires once per turn for every
        # session in the deployment, and a session that never read a skill must
        # not pay a thaw of somebody's hundred-entry list to find that out.
        outstanding = outstanding_steps(session)
        if not outstanding:
            return
        if nudges_since_plan(session) >= MAX_NUDGES:
            # Stood down rather than steering into a wall. Any `write_todos`
            # resets this, so the run that is making progress never reaches it.
            log.info(
                "ph_stabilize.skill_steps: %s with the plan unchanged; standing down",
                count_of(MAX_NUDGES, "nudge"),
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
