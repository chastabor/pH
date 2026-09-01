"""`ctx.schedule` — work a root will do later, folded from its own log (P5-06).

Three kinds, one mechanism: `once` at a moment, `interval` every so often, and
`cron` on an expression. What makes this a seam rather than a timer is that a
schedule outlives the process holding it.

**Everything is in the log, including the claim.** A schedule is
`schedule/created` until a matching `schedule/cancelled`; a firing is
`schedule/tick`, appended *before* the work is delivered. That ordering is A10's
write-ahead applied to time: a tick recorded and then lost to a crash costs one
skipped run, while a tick delivered and then lost costs a *repeated* run — and
for a schedule whose payload sends a prompt, repeating bills twice and confuses
the transcript. **At-most-once, deliberately**, and the log says which.

**Missed ticks coalesce**, which is why `due_at` takes the last claimed time
rather than counting from creation. A five-minute schedule on a laptop that slept
for three hours has thirty-six fire times behind it; delivering all thirty-six
turns a nap into a stampede, and delivering the *oldest* works through the backlog
for another three hours. So the answer is one tick naming the most recent due
moment, and the gap stays visible in the log because the previous tick is still
there.

**The clock is a parameter.** `now` is passed in rather than read here, so a test
can advance three hours without sleeping and a caller can drive the whole thing
from one stamp.

@module ph.seams.schedule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias

from ..cordis import Context, plugin
from ..session import Session, SessionFoldCache
from ..wire import WireModel

__all__ = [
    "CANCELLED",
    "CREATED",
    "HEARTBEAT",
    "TICK",
    "Schedule",
    "ScheduleKind",
    "ScheduleService",
    "ScheduleState",
    "apply",
    "due_at",
    "next_at",
    "schedules",
    "state_to_wire",
]

log = logging.getLogger("ph.seams.schedule")

CREATED = "schedule/created"
CANCELLED = "schedule/cancelled"
TICK = "schedule/tick"
HEARTBEAT = "schedule/heartbeat"

ScheduleKind: TypeAlias = Literal["once", "cron", "interval"]


class Schedule(WireModel):
    """One piece of future work, as the log records it.

    `spec` means something different per kind — an epoch-ms instant for `once`,
    seconds for `interval`, a cron expression for `cron` — which is one field
    with a kind-dependent reading rather than three mutually exclusive ones. The
    kind is right beside it and `parse` is the only thing that reads it.
    """

    id: str
    kind: ScheduleKind
    spec: str
    """The timing, read according to `kind`.

    Milliseconds for the numeric kinds — an epoch instant for `once`, a duration for
    `interval` — and an expression for `cron`. **Milliseconds because every other
    duration crossing this wire is** (`grace_ms`, `timeout_ms`, `duration_ms`, and this
    seam's own `dueAt`/`firedAt`): one unit, no conversion, nothing to get backwards.
    """
    prompt: str
    """What to say to the agent when this fires.

    Required. It was optional, documented as "empty means the caller delivers
    something of its own" — and no caller did: an empty-prompt schedule was
    claimed, recorded a tick, counted as fired, and delivered nothing, forever.
    At-most-once made that permanent, since the claim is never retried. A
    schedule with nothing to do is not a schedule."""


@dataclass(slots=True)
class ScheduleState:
    """A schedule and what its log says has happened to it."""

    schedule: Schedule
    created_at: int = 0
    """When `schedule/created` was appended, taken from the event's own `time`.

    **From the log rather than from a field on the wire model.** A caller obligation
    here is one nobody honours: a schedule anchored at epoch 0 is due the moment it
    exists, and a cron's first claim walks forward from 1970. The event carries the
    timestamp already, and `last_tick` comes from the same place.
    """
    last_tick: int | None = None
    """The *due* moment of the last claimed tick, not when it was claimed.

    Due rather than claimed, because that is what the next one is computed from:
    a tick claimed four minutes late must not push the following one four
    minutes later, or a five-minute schedule drifts into a six-minute one over a
    day."""
    cancelled: bool = False

    @property
    def anchor(self) -> int:
        """What the next fire time is computed from.

        The last claimed *due* moment, or creation if it has never fired. One
        name for a rule that was spelled out twice, once per branch of
        `due_at`."""
        return self.last_tick if self.last_tick is not None else self.created_at


def schedules(session: Session) -> dict[str, ScheduleState]:
    """Every schedule this log knows about, live or cancelled — a fold.

    Cancelled ones are kept rather than dropped, for `subagent_roster`'s reason:
    a caller asking what happened to the schedule it revoked deserves an answer
    other than silence, and a reader of the log can see the cancellation.
    """
    # The common root has no schedules at all — `schedule` is a `base.yaml` row,
    # so every mounted root carries the seam — and the walk below costs the same
    # whether or not it finds anything, where `latest` is an incremental fold.
    # It scans from zero while the walk starts at the seed boundary, which is the
    # safe direction: no `schedule/created` anywhere means none after the seed.
    if session.latest(CREATED) is None:
        return {}
    found: dict[str, ScheduleState] = {}
    for event in session.events_from(session.header.seed_length or 0):
        data = event.data
        if event.type == CREATED:
            schedule = Schedule.model_validate(dict(data))
            found[schedule.id] = ScheduleState(schedule=schedule, created_at=int(event.time))
        elif event.type == CANCELLED:
            state = found.get(str(data.get("id", "")))
            if state is not None:
                state.cancelled = True
        elif event.type == TICK:
            state = found.get(str(data.get("id", "")))
            if state is not None:
                state.last_tick = int(data.get("dueAt", 0))
    return found


def due_at(state: ScheduleState, *, now: int) -> int | None:
    """The moment this schedule should fire *for*, or `None` if it should not.

    Returns the most recent due moment at or before `now` that has not been
    claimed — so a run of missed fire times collapses to one, naming the latest.
    The caller records that moment as the tick's `dueAt`, which is what the next
    call reads back.
    """
    if state.cancelled:
        return None
    schedule = state.schedule
    anchor = state.anchor
    if schedule.kind == "once":
        at = _int(schedule.spec)
        return at if state.last_tick is None and at is not None and at <= now else None
    if schedule.kind == "interval":
        every = _int(schedule.spec)
        if every is None or every <= 0:
            return None
        elapsed = now - anchor
        if elapsed < every:
            return None
        # The latest whole interval that has passed, not the first: this is the
        # coalescing, and it is one multiplication rather than a loop that would
        # take a step per missed tick on a log that slept for a week.
        return anchor + (elapsed // every) * every
    return _last_cron_before(schedule.spec, after=anchor, now=now)


def state_to_wire(state: ScheduleState, *, now: int) -> dict[str, Any]:
    """One schedule as a client is told about it — the fold's own projection.

    Here rather than in the daemon that first needed it: this is a fact about a
    *schedule*, and every transport serves the same one. `schedule` is a `base.yaml`
    row, so a stdio session has schedules too, and the moment a second front end
    lists them `createdAt`/`lastTick`/`nextAt` would be spelled twice with nothing
    able to see them disagree.

    `nextAt` is computed rather than stored, because it is derived from the anchor and
    the spec and a second carrier is a second answer (A11). `None` means nothing
    further — a `once` that has fired, or a cron whose expression no longer yields.
    """
    return {
        **state.schedule.to_wire(),
        "createdAt": state.created_at,
        "lastTick": state.last_tick,
        "nextAt": next_at(state, now=now),
    }


def next_at(state: ScheduleState, *, now: int) -> int | None:
    """When this schedule will next fire, or `None` if it never will again.

    The forward-looking twin of `due_at`, and deliberately a second function
    rather than a sign on the first: `due_at` answers "should this fire *now*,
    and for which missed moment", which is what the tick needs and the opposite
    of what a person reading `ph agents schedule` wants. One function answering
    both would have every caller branching on which question it got back.

    An overdue schedule answers with the moment it is overdue *for*, not with a
    later one: the next tick claims exactly that, and reporting a time an hour
    out for work about to run in five seconds is the small lie a listing exists
    to prevent.
    """
    if state.cancelled:
        return None
    overdue = due_at(state, now=now)
    if overdue is not None:
        return overdue
    schedule = state.schedule
    if schedule.kind == "once":
        # Not `due_at`'s `None`, which conflates "already fired" with "not yet":
        # a `once` that has never ticked still has a future, and it is its spec.
        return None if state.last_tick is not None else _int(schedule.spec)
    if schedule.kind == "interval":
        every = _int(schedule.spec)
        return None if every is None or every <= 0 else state.anchor + every
    return _next_cron_after(schedule.spec, now=now)


def _next_cron_after(spec: str, *, now: int) -> int | None:
    """The first cron moment after `now`, or `None` if the expression is unusable.

    Lazy `croniter` for `_last_cron_before`'s measured reason, and the same
    refusal shape: an expression nobody can parse is logged and declines, so a
    listing loses one row rather than the command.
    """
    from croniter import croniter

    try:
        moment: datetime = croniter(spec, _local(now)).get_next(datetime)
    except (ValueError, KeyError) as error:
        log.warning("ph.seams.schedule: unusable cron %r (%s)", spec, error)
        return None
    return int(moment.timestamp() * 1000)


def _local(moment: int) -> datetime:
    """An epoch-ms instant as this machine's naive wall-clock time.

    **Cron expressions are local, which croniter only does if you ask in datetimes.**
    Handed a float it works in UTC; handed a naive datetime it works in the machine's
    zone — the same expression meaning two different things depending on the argument
    type. Local is what every crontab a person has ever written means, and there is
    nowhere on this wire to say otherwise.

    The cost is the one every local cron has: an hour that repeats or does not exist
    at a DST boundary resolves by `datetime.timestamp()`'s rule rather than by ours.
    """
    return datetime.fromtimestamp(moment / 1000)


def _last_cron_before(spec: str, *, after: int, now: int) -> int | None:
    """The newest cron moment in `(after, now]`, or `None`.

    **Sought backwards from `now`, not walked forwards from `after`.** The coalescing
    answer is the *last* matching moment, so one `get_prev` finds it whatever the gap,
    where walking forwards costs a step per missed moment — and an anchor far enough
    back blocks the event loop for minutes.

    `croniter` is imported here rather than at module scope: `schedule` is a
    `base.yaml` row, so the module loads in every host, and the import is paid on
    every `ph -p` and every TUI start that never sees a cron expression.
    """
    from croniter import croniter

    try:
        # From `now + 1ms`, because `get_prev` is strictly *before* its start:
        # seeking from `now` itself skips a moment that lands exactly on `now`,
        # which a five-second ticker hits whenever a tick coincides with the
        # cron boundary — the schedule then fires a tick late, every time. The
        # walk this replaced included that moment, so the offset keeps the two
        # equivalent.
        found: datetime = croniter(spec, _local(now + 1)).get_prev(datetime)
    except (ValueError, KeyError) as error:
        log.warning("ph.seams.schedule: unusable cron %r (%s)", spec, error)
        return None
    moment = int(found.timestamp() * 1000)
    return moment if moment > after else None


def _int(value: str) -> int | None:
    try:
        return int(float(value))
    except ValueError:
        return None


@dataclass(slots=True)
class ScheduleService:
    """`ctx.schedule` — create, cancel, and claim what is due.

    The service writes to the log and reads back through the fold; it holds no
    schedule state of its own beyond a cache of that fold, so a resumed session and a
    live one answer the same question the same way (A11).

    **Cached, like `SubagentService._rosters`.** `live()` is on the daemon's
    five-second tick path, so the bare fold ran tens of thousands of times a day per
    root. `SessionFoldCache` keys on `session.seq`, and a schedule only changes when
    something is appended, so every read between appends is a dict hit.

    No `ctx`, because nothing here reads one — a service that takes an argument it
    never uses is one a test has to lie about to construct.
    """

    _states: SessionFoldCache[dict[str, ScheduleState]] = field(
        default_factory=lambda: SessionFoldCache(schedules)
    )

    def states(self, session: Session) -> dict[str, ScheduleState]:
        """Every schedule this log knows about, folded at most once per append."""
        return self._states.read(session)

    def forget_session(self, session_id: str) -> None:
        """Drop what this service cached about one session."""
        self._states.forget(session_id)

    def create(self, session: Session, schedule: Schedule) -> Schedule:
        """Record a schedule. It is live from the moment the event lands."""
        session.append(CREATED, schedule.to_wire())
        return schedule

    def cancel(self, session: Session, schedule_id: str) -> bool:
        """Record a cancellation. `False` when there was nothing to cancel.

        Nothing to cancel covers both an id this log never knew *and* one it
        already cancelled — the fold keeps cancelled schedules visible
        (`subagent_roster`'s reason), so the second case would otherwise report
        success and append a redundant `schedule/cancelled` on every retry. To a
        person running `ph agents schedule --cancel` twice, "cancelled" the
        second time is a claim about work that was already stopped.
        """
        state = self.states(session).get(schedule_id)
        if state is None or state.cancelled:
            return False
        session.append(CANCELLED, {"id": schedule_id})
        return True

    def live(self, session: Session) -> list[ScheduleState]:
        """The schedules that could still fire."""
        return [state for state in self.states(session).values() if not state.cancelled]

    def claim(self, session: Session, *, now: int) -> list[Schedule]:
        """Claim everything due, write-ahead, and return it for delivery.

        The append happens here and the delivery happens in the caller, which is
        the ordering that makes a crash cost a skipped run rather than a
        repeated one. A caller that never delivers has still recorded that it
        meant to, which is what an operator needs to see.
        """
        claimed: list[Schedule] = []
        for state in self.live(session):
            moment = due_at(state, now=now)
            if moment is None:
                continue
            # The due moment goes in the event and nowhere else: one fact with two
            # carriers is one that can disagree.
            session.append(TICK, {"id": state.schedule.id, "dueAt": moment, "firedAt": now})
            claimed.append(state.schedule)
        return claimed

    def heartbeat(self, session: Session, *, now: int, live: int) -> None:
        """Record that the scheduler is still watching this root.

        `live` is passed rather than re-derived: the caller has just asked
        whether this root has any, and folding again to count them was the
        second of two folds per beat."""
        session.append(HEARTBEAT, {"at": now, "live": live})


@plugin("schedule")
async def apply(ctx: Context, _config: Any) -> None:
    """Publish `ctx.schedule`."""
    service = ScheduleService()
    ctx.provide("schedule", service)
    # The cache is bounded by live sessions, and this is what makes that true —
    # the same line `subagents` uses for the same reason.
    ctx.on("session/disposed", lambda session: service.forget_session(session.id))
