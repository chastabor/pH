"""Auto-refine: when the harness refines itself, and what can stop it (H7).

Prime-agent refines every 25 assistant turns or after a compaction, behind a
cheap review call and a 20-minute cooldown. All four are ported. The difference
is where the *state* is kept. Prime-agent keeps counters and a last-run
timestamp beside the session. pH derives every one of them from the log.

**The clock is the clock of the log.** `due()` compares against the time of the
event that triggered the check, and not against `time.time()`. The whole
decision is therefore a fold over a prefix. It gives the same answer on a
resumed session, on a fork, and in a test, with no clock to freeze. The
triggering event *is* now, because it was appended moments ago, so nothing is
lost when the wall clock is not read.

**A consideration is recorded, and a refinement is not the only outcome.**
`harness/refine-considered` advances the cooldown. A review that says "nothing
here" therefore costs one cheap call, and then there is no review for twenty
minutes. The event is ignorable: a reader that skips it gets one extra review,
not a wrong harness. That is exactly the line that
`IGNORABLE_SESSION_EVENT_TYPES` draws.

**A veto costs no tokens.** `harness/before-refine` runs before the review call,
so a plugin that refuses spends nothing. The refusal is recorded as a
consideration, for two reasons. A user who typed `/refine` must get an answer.
Also, a refusal that left no trace would look the same as a pass that never
fired. The payload carries the trigger, so a listener can refuse the automatic
passes and still let the `/refine` of a user through.

@module ph_rlm.harness.auto
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias

from ph.agent.types import AgentHandle
from ph.cordis import Context, events, settled_or_none
from ph.session import Session, SessionEvent

from .state import REFINED, HarnessScope

__all__ = [
    "BEFORE_REFINE",
    "CONSIDERED",
    "COOLDOWN_MINUTES",
    "TURNS_BETWEEN_REFINEMENTS",
    "RefineRequest",
    "RefineTrigger",
    "due",
    "veto_reason",
]

CONSIDERED = "harness/refine-considered"
BEFORE_REFINE = "harness/before-refine"

TURNS_BETWEEN_REFINEMENTS = 25
COOLDOWN_MINUTES = 20
"""The numbers of prime-agent. The turn count makes refinement routine. The
cooldown stops a burst of short turns from making it constant."""

RefineTrigger: TypeAlias = Literal["user", "turns", "compaction"]

events.declare(
    BEFORE_REFINE,
    "waterfall",
    owner="ph_rlm.harness",
    doc="Before a refinement is planned. A listener that returns a reason vetoes it.",
)


@dataclass(frozen=True, slots=True)
class RefineRequest:
    """What a veto listener is shown."""

    session: Session
    agent: AgentHandle
    scope: HarnessScope
    trigger: RefineTrigger
    instructions: str = ""


async def veto_reason(ctx: Context, request: RefineRequest) -> str | None:
    """The first listener's objection, or `None` if none objected."""

    async def inner(_request: RefineRequest) -> str | None:
        return None

    reason = await ctx.waterfall(BEFORE_REFINE, request, inner=inner)
    return settled_or_none(BEFORE_REFINE, reason, str)


class EventLog(Protocol):
    """The events of a log, which is all that `due` and its fold read.

    The protocol is narrower than `Session` on purpose. Both functions below
    are pure folds over the event list. The docstring of
    `_since_last_consideration` says "one backwards pass". A parameter of the
    whole `Session` overstated what they touch. The test double of the harness
    planner is three lines that hold a list. It satisfied this contract from
    the start, and it failed the declared one. Nobody noticed, because the test
    trees were outside mypy (issue 32).
    """

    @property
    def events(self) -> Sequence[SessionEvent]: ...


def _since_last_consideration(session: EventLog) -> tuple[int, int | None, bool]:
    """`(turns, milliseconds, compacted)` since the last consideration.

    One backwards pass is enough, because all three answers stop at the same
    event: the most recent `harness/refine-considered` or `harness/refined`.
    The cooldown runs from whichever came last. An explicit `/refine` should
    quiet the automatic pass exactly as a declined review does.

    The elapsed time is `None` when there has been no consideration at all. A
    session that has never refined is not in a cooldown. If its whole age were
    reported as the elapsed time, a session that reached 25 turns quickly would
    miss its first pass.
    """
    turns = 0
    compacted = False
    now = session.events[-1].time if session.events else 0
    for event in reversed(session.events):
        if event.type in (CONSIDERED, REFINED):
            return turns, now - event.time, compacted
        if event.type == "turn/end":
            turns += 1
        elif event.type == "compaction/summarized":
            # This is the record that compaction itself writes, not
            # `is_replacement_surface_event`. A surface `replace` is a
            # *mechanism*. `input-offload` (P4-02) uses it to relocate a pasted
            # blob, which teaches the harness nothing and is not the moment
            # that this trigger exists for. Compaction reports itself (P4-03),
            # so the discriminator is the attribution in the log.
            compacted = True
    return turns, None, compacted


def due(
    session: EventLog,
    *,
    turns_between: int = TURNS_BETWEEN_REFINEMENTS,
    cooldown_minutes: int = COOLDOWN_MINUTES,
) -> RefineTrigger | None:
    """Why an automatic refinement is due, or `None`.

    Compaction is a separate trigger because it is the one moment when the
    conversation gets *shorter*. What the summary dropped is exactly what the
    harness should have kept, and after the compaction the planner can no
    longer read it.
    """
    turns, elapsed, compacted = _since_last_consideration(session)
    if elapsed is not None and elapsed < cooldown_minutes * 60_000:
        return None
    if compacted:
        return "compaction"
    return "turns" if turns >= turns_between else None
