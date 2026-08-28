"""Auto-refine: when the harness refines itself, and what can stop it (H7).

Prime Agent refines every 25 assistant turns or after a compaction, behind a
cheap review call and a 20-minute cooldown. All four numbers are ported. What is
different is where the *state* lives: prime-agent keeps counters and a last-run
timestamp beside the session, and pH derives every one of them from the log.

**The clock is the log's.** `due()` compares against the time of the event that
triggered the check rather than `time.time()`, so the whole decision is a fold
over a prefix: it gives the same answer on a resumed session, on a fork, and in a
test, with no clock to freeze. The triggering event *is* now — it was appended
moments ago — so nothing is lost by not asking the wall.

**Considering is recorded, refining is not the only outcome.**
`harness/refine-considered` is what advances the cooldown, so a review that says
"nothing here" costs one cheap call and then stays quiet for twenty minutes. It
is ignorable: a reader that skips it gets one extra review, not a wrong harness —
which is exactly the line `IGNORABLE_SESSION_EVENT_TYPES` draws.

**A veto costs no tokens.** `harness/before-refine` runs before the review call,
so a plugin that refuses spends nothing — it is recorded as a consideration, both
because a user who typed `/refine` is owed an answer and because a refusal that
left no trace would be indistinguishable from a pass that never fired. The
payload carries the trigger, so a listener can refuse the automatic passes and
still let a human's `/refine` through.

@module ph_rlm.harness.auto
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ph.cordis import Context, events
from ph.session import Session, is_replacement_surface_event

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
"""Prime Agent's numbers. The turn count is what makes refinement routine; the
cooldown is what stops a burst of short turns making it constant."""

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
    agent: Any
    scope: HarnessScope
    trigger: RefineTrigger
    instructions: str = ""


async def veto_reason(ctx: Context, request: RefineRequest) -> str | None:
    """The first listener's objection, or `None` if none objected."""

    async def inner(_request: RefineRequest) -> str | None:
        return None

    reason = await ctx.waterfall(BEFORE_REFINE, request, inner=inner)
    return None if reason is None else str(reason)


def _since_last_consideration(session: Session) -> tuple[int, int | None, bool]:
    """`(turns, milliseconds, compacted)` since the last consideration.

    One backwards pass, because all three answers stop at the same event: the
    most recent `harness/refine-considered` or `harness/refined`. Whichever came
    last is what the cooldown runs from — an explicit `/refine` should quiet the
    automatic pass exactly as a declined review does.

    The elapsed time is `None` when there has been no consideration at all: a
    session that has never refined is not serving a cooldown, and reporting its
    whole age as one would let a fast 25 turns slip past the first pass.
    """
    turns = 0
    compacted = False
    now = session.events[-1].time if session.events else 0
    for event in reversed(session.events):
        if event.type in (CONSIDERED, REFINED):
            return turns, now - event.time, compacted
        if event.type == "turn/end":
            turns += 1
        elif is_replacement_surface_event(event):
            compacted = True
    return turns, None, compacted


def due(
    session: Session,
    *,
    turns_between: int = TURNS_BETWEEN_REFINEMENTS,
    cooldown_minutes: int = COOLDOWN_MINUTES,
) -> RefineTrigger | None:
    """Why an automatic refinement is due, or `None`.

    Compaction is a trigger in its own right because it is the one moment the
    conversation gets *shorter*: what the summary dropped is exactly what the
    harness should have kept, and after it the planner can no longer read it.
    """
    turns, elapsed, compacted = _since_last_consideration(session)
    if elapsed is not None and elapsed < cooldown_minutes * 60_000:
        return None
    if compacted:
        return "compaction"
    return "turns" if turns >= turns_between else None
