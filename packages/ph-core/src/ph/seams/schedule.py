"""`ctx.schedule` — work a root will do later, folded from its own log (P5-06).

Three kinds, one mechanism: `once` at a moment, `interval` every so often, and
`cron` on an expression. What makes this a seam rather than a timer is that a
schedule outlives the process holding it — the daemon is the one thing in pH
built to run for weeks, and a machine that reboots between Tuesday and Wednesday
must not lose Wednesday's run.

**Everything is in the log, including the claim.** A schedule is
`schedule/created` until a matching `schedule/cancelled`; a firing is
`schedule/tick`, appended *before* the work is delivered. That ordering is A10's
write-ahead applied to time: a tick recorded and then lost to a crash costs one
skipped run, while a tick delivered and then lost costs a *repeated* run — and
for a schedule whose payload sends a prompt, repeating is the failure that bills
twice and confuses the transcript. At-most-once, deliberately, and the log says
which.

**Missed ticks coalesce**, which is the row's gate and the reason `due_at` takes
the last claimed time rather than counting from creation. A five-minute schedule
on a laptop that slept for three hours has thirty-six fire times behind it; a
scheduler that delivered all thirty-six would turn a nap into a stampede, and
one that delivered the *oldest* would work through the backlog for another three
hours before catching up. So the answer is one tick naming the most recent due
moment, and the gap is visible in the log because the previous tick is still
there.

**The clock is a parameter.** `now` is passed in rather than read here, so a
test can advance three hours without sleeping and a caller can drive the whole
thing from one stamp — the same seam `Supervisor.sweep` uses for the same
reason.

@module ph.seams.schedule
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    "schedules",
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

    Milliseconds for the numeric kinds — an epoch instant for `once`, a duration
    for `interval` — and an expression for `cron`. Milliseconds because every
    other duration crossing this wire is (`grace_ms`, `timeout_ms`,
    `duration_ms`, and this seam's own `dueAt`/`firedAt`), and because the first
    version read `interval` as seconds while comparing it against millisecond
    stamps, which made a five-minute schedule fire every 300 ms. One unit,
    no conversion, nothing to get backwards.
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

    From the log rather than from a field on the wire model: `created_at` was a
    caller obligation nobody honoured, so every schedule the daemon created
    anchored at epoch 0 — an hourly interval was due the moment it existed, and
    a cron's first claim walked forward from 1970 (measured at six and a half
    minutes of blocked event loop for `* * * * *`). The event carries the
    timestamp already, and `last_tick` comes from the same place."""
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
    # so every mounted root carries the seam — and the walk costs the same
    # either way: 22.6 ms at 500 000 events, whether or not it finds anything.
    # `latest` is an incremental fold, so this answers in 84 ns. It scans from
    # zero while the walk below starts at the seed boundary, which is the safe
    # direction: no `schedule/created` anywhere means none after the seed.
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


def _last_cron_before(spec: str, *, after: int, now: int) -> int | None:
    """The newest cron moment in `(after, now]`, or `None`.

    **Sought backwards from `now`, not walked forwards from `after`.** The
    coalescing answer is the *last* matching moment, so one `get_prev` finds it
    whatever the gap; walking forwards costs a step per missed moment, which is
    11.6 µs each. Measured: a week's gap of `* * * * *` took 122 ms walking and
    40 µs seeking, and the seek is cheaper even in the steady state (40 µs
    against 53 µs) because it takes one step rather than two. The pathological
    case is what settles it — an anchor at epoch 0 walked 29 million steps, some
    391 s of blocked event loop, against the same 55 µs.

    `croniter` is imported here rather than at module scope: this is one of
    three kinds, `schedule` is a `base.yaml` row so the module loads in every
    host, and the import measured 25.4 ms — about what pydantic costs — paid on
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
        moment = int(croniter(spec, (now + 1) / 1000).get_prev(float) * 1000)
    except (ValueError, KeyError) as error:
        log.warning("ph.seams.schedule: unusable cron %r (%s)", spec, error)
        return None
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
    schedule state of its own beyond a cache of that fold, so a resumed session
    and a live one answer the same question the same way (A11).

    **Cached, like `SubagentService._rosters`, and for the reason that seam
    gives.** `live()` is on the daemon's five-second tick path, so the bare fold
    ran 34 500 times a day per root — 22.6 ms each at 500 000 events, which is
    49% of a core permanently at fifty roots. `SessionFoldCache` keys on
    `session.seq`, and a schedule only changes when something is appended, so
    every read between appends is a dict hit: 104 ns, a 237 000x reduction.

    No `ctx`: the first version took one and never read it, which cost seven
    `type: ignore` lines in the tests to construct a service the tests had to
    lie about.
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
        """Record a cancellation. `False` if this log never knew that id."""
        if schedule_id not in self.states(session):
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
            # The due moment goes in the event and nowhere else: the only
            # production caller discarded the copy this used to return, and one
            # fact with two carriers is one that can disagree.
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
