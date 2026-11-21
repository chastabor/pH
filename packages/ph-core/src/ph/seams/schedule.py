"""`ctx.schedule`: work that a root will do later, folded from its own log (P5-06).

There are three kinds and one mechanism: `once` at a moment, `interval` after
each period, and `cron` on an expression. This is a seam and not a timer,
because a schedule outlives the process that holds it.

**Everything is in the log, including the claim.** A schedule is
`schedule/created` until a matching `schedule/canceled`. A firing is
`schedule/tick`, and it is appended *before* the work is delivered. That order
is the write-ahead rule of A10, applied to time. A tick that is recorded and
then lost to a crash costs one skipped run. A tick that is delivered and then
lost costs a *repeated* run. For a schedule whose payload sends a prompt, a
repeated run bills twice and confuses the transcript. **At-most-once,
deliberately**, and the log says which.

**Missed ticks coalesce.** That is why `due_at` takes the last claimed time and
does not count from creation. Take a five-minute schedule on a laptop that slept
for three hours. It has thirty-six fire times behind it. To deliver all
thirty-six turns a short sleep into a flood of runs. To deliver the *oldest*
works through the backlog for another three hours. So the answer is one tick
that names the most recent due moment. The gap stays visible in the log, because
the previous tick is still there.

**The clock is a parameter.** `now` is passed in and is not read here. A test
can therefore advance three hours without sleeping, and a caller can drive
everything from one stamp.

@module ph.seams.schedule
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, TypeAlias

from ..cordis import Context, plugin
from ..json import JsonValue, as_int, as_obj, as_str
from ..keys import SCHEDULE
from ..llm.types import PluginSource
from ..paths import resolve_roots
from ..session import Session, SessionEvent, SessionFoldCache, now_ms, session_written
from ..session.writers import log_writer
from ..session_profile import BASE, CLEARED, OVERRIDE, REFUSED, environment, fold_environment
from ..wire import WireModel
from .invariants import contribute_fold_cache
from .schedule_index import Appointment, IndexRecorder, IndexWriter, ScheduleIndex

_LOG = log_writer(__name__)

__all__ = [
    "CANCELED",
    "CREATED",
    "FOLDED",
    "REBUILD_TYPES",
    "TICK",
    "Schedule",
    "ScheduleKind",
    "ScheduleService",
    "ScheduleState",
    "apply",
    "due_at",
    "fire_due",
    "next_at",
    "rebuild_index",
    "scheduled_source",
    "schedules",
    "state_to_wire",
]

log = logging.getLogger("ph.seams.schedule")

CREATED = "schedule/created"
CANCELED = "schedule/canceled"
TICK = "schedule/tick"

FOLDED = frozenset({CREATED, CANCELED, TICK})
"""The records that `schedules` folds. A reader needs nothing else from a log to
know its schedules."""

ScheduleKind: TypeAlias = Literal["once", "cron", "interval"]


class Schedule(WireModel):
    """One piece of future work, as the log records it.

    The meaning of `spec` depends on the kind: an epoch instant in milliseconds
    for `once`, a duration in milliseconds for `interval`, and a cron expression
    for `cron`. So this is one field that is read by kind, and not three fields
    of which only one may be set. The kind is beside it, and `due_at` and
    `next_at` are the only functions that read it.
    """

    id: str
    kind: ScheduleKind
    spec: str
    """The timing, read according to `kind`.

    The numeric kinds use milliseconds: an epoch instant for `once` and a
    duration for `interval`. `cron` uses an expression. **The unit is
    milliseconds because every other duration on this wire uses it**
    (`grace_ms`, `timeout_ms`, `duration_ms`, and the `dueAt` and `firedAt` of
    this seam). With one unit there is no conversion, and so no conversion to get
    backwards.
    """
    prompt: str
    """What to say to the agent when this schedule fires.

    Required. The field was optional, and its documentation said "empty means
    the caller delivers something of its own". No caller did. A schedule with an
    empty prompt was claimed, recorded a tick, was counted as fired, and
    delivered nothing, at every firing. At-most-once made that permanent,
    because the claim is never retried. A schedule with nothing to do is not a
    schedule."""


@dataclass(slots=True)
class ScheduleState:
    """A schedule and what its log says has happened to it."""

    schedule: Schedule
    created_at: int = 0
    """When `schedule/created` was appended, taken from the `time` of the event.

    **The value comes from the log and not from a field on the wire model.** A
    field would be an obligation on the caller, and no caller honors it. A
    schedule that is anchored at epoch 0 is due the moment it exists, and the
    first claim of a cron is computed from 1970. The event already carries the
    timestamp, and `last_tick` comes from the same place.
    """
    last_tick: int | None = None
    """The *due* moment of the last claimed tick, and not the moment of the claim.

    The next tick is computed from this value, so it must be the due moment. A
    tick that is claimed four minutes late must not move the next tick four
    minutes later. Otherwise a five-minute schedule drifts into a six-minute
    schedule over a day."""
    canceled: bool = False

    @property
    def anchor(self) -> int:
        """The moment that the next fire time is computed from.

        It is the last claimed *due* moment, or the creation time if the
        schedule has never fired. The property gives one name to a rule that
        was written out twice, one time in each branch of `due_at`."""
        return self.last_tick if self.last_tick is not None else self.created_at


def schedules(session: Session) -> dict[str, ScheduleState]:
    """Every schedule that this log knows about, live or canceled. This is a fold.

    Canceled schedules are kept and not dropped. A caller that asks what
    happened to a schedule that it revoked must get an answer and not silence,
    and a reader of the log can see the cancellation.
    """
    # Most roots have no schedules at all. Every mounted root still carries the
    # seam, because `schedule` is a `base.yaml` row. The walk below costs the
    # same whether or not it finds anything, and `latest` is an incremental
    # fold. `latest` scans from zero, and the walk starts at the seed boundary.
    # That is the safe direction: no `schedule/created` anywhere means none
    # after the seed.
    if session.latest(CREATED) is None:
        return {}
    return _fold(session.own_events())


def _fold(events: Iterable[SessionEvent]) -> dict[str, ScheduleState]:
    """`schedules` over events alone, for a reader that holds the records of a
    log and not a `Session` (`rebuild_index`). An event outside `FOLDED` is
    skipped."""
    found: dict[str, ScheduleState] = {}
    for event in events:
        data = event.data
        if event.type == CREATED:
            schedule = Schedule.model_validate(dict(data))
            found[schedule.id] = ScheduleState(schedule=schedule, created_at=int(event.time))
        elif event.type == CANCELED:
            state = found.get(as_str(data.get("id")))
            if state is not None:
                state.canceled = True
        elif event.type == TICK:
            state = found.get(as_str(data.get("id")))
            if state is not None:
                state.last_tick = as_int(data.get("dueAt"))
    return found


def due_at(state: ScheduleState, *, now: int) -> int | None:
    """The moment that this schedule should fire *for*, or `None` if it should
    not fire.

    The result is the most recent due moment at or before `now` that has not
    been claimed. A run of missed fire times therefore collapses to one, and
    the result names the latest. The caller records that moment as the `dueAt`
    of the tick, and the next call reads it back.
    """
    if state.canceled:
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
        # This is the latest whole interval that has passed, and not the first.
        # That is the coalescing. It is one multiplication, where a loop would
        # take one step for each missed tick on a log that slept for a week.
        return anchor + (elapsed // every) * every
    return _last_cron_before(schedule.spec, after=anchor, now=now)


def state_to_wire(state: ScheduleState, *, now: int) -> dict[str, JsonValue]:
    """One schedule as a client is told about it: the projection of the fold.

    The function is here and not in the daemon, which needed it first. It
    states a fact about a *schedule*, and every transport serves the same fact.
    `schedule` is a `base.yaml` row, so a stdio session has schedules too. If a
    second front end listed them with code of its own, `createdAt`, `lastTick`
    and `nextAt` would be written twice, and nothing could see the two disagree.

    `nextAt` is computed and not stored. It is derived from the anchor and the
    spec, and a second carrier is a second answer (A11). `None` means that
    nothing further will fire: a `once` that has fired, or a cron whose
    expression no longer yields a moment.
    """
    return {
        **state.schedule.to_wire(),
        "createdAt": state.created_at,
        "lastTick": state.last_tick,
        "nextAt": next_at(state, now=now),
    }


def next_at(state: ScheduleState, *, now: int) -> int | None:
    """When this schedule will next fire, or `None` if it will never fire again.

    This is the forward-looking counterpart of `due_at`. It is a second function
    on purpose, and not a flag on the first. `due_at` answers "should this fire
    *now*, and for which missed moment". The tick needs that answer, and it is
    the opposite of what a user who reads `phern agents schedule` wants. With
    one function for both questions, every caller would branch on which answer
    it received.

    An overdue schedule answers with the moment that it is overdue *for*, and
    not with a later moment. The next tick claims exactly that moment. A listing
    exists to prevent the small error of a time an hour ahead for work that is
    about to run.
    """
    if state.canceled:
        return None
    overdue = due_at(state, now=now)
    if overdue is not None:
        return overdue
    schedule = state.schedule
    if schedule.kind == "once":
        # This is not the `None` of `due_at`, which means both "already fired"
        # and "not yet". A `once` that has never ticked still has a future
        # moment, and that moment is its spec.
        return None if state.last_tick is not None else _int(schedule.spec)
    if schedule.kind == "interval":
        every = _int(schedule.spec)
        return None if every is None or every <= 0 else state.anchor + every
    return _next_cron_after(schedule.spec, now=now)


def _next_cron_after(spec: str, *, now: int) -> int | None:
    """The first cron moment after `now`, or `None` if the expression is unusable.

    `croniter` is imported lazily, for the reason that `_last_cron_before`
    gives. The refusal has the same form too: an expression that cannot be
    parsed is logged, and the function declines. A listing then has no next
    time for that one row, and the command does not fail.
    """
    from croniter import croniter  # noqa: PLC0415

    try:
        moment: datetime = croniter(spec, _local(now)).get_next(datetime)
    except (ValueError, KeyError) as error:
        log.warning("ph.seams.schedule: unusable cron %r (%s)", spec, error)
        return None
    return int(moment.timestamp() * 1000)


def _local(moment: int) -> datetime:
    """An epoch instant in milliseconds, as the naive wall-clock time of this
    machine.

    **Cron expressions are local, and croniter treats them as local only when it
    is given datetimes.** Given a float, it works in UTC. Given a naive
    datetime, it works in the zone of the machine. So the same expression means
    two different things, and the type of the argument decides which. Local time
    is what every crontab that a person has written means, and this wire has no
    place to say otherwise.

    The cost is the cost of every local cron. At a DST boundary, an hour that
    repeats or does not exist is resolved by the rule of `datetime.timestamp()`,
    and not by a rule of this module.
    """
    return datetime.fromtimestamp(moment / 1000)


def _last_cron_before(spec: str, *, after: int, now: int) -> int | None:
    """The newest cron moment in `(after, now]`, or `None`.

    **The search goes backwards from `now`, and does not walk forwards from
    `after`.** The coalescing answer is the *last* matching moment, so one
    `get_prev` finds it, however long the gap is. A forward walk costs one step
    for each missed moment, and an anchor that is far enough back blocks the
    event loop for minutes.

    `croniter` is imported here and not at module scope. `schedule` is a
    `base.yaml` row, so this module loads in every host. An import at module
    scope would be paid on every `phern -p` and on every TUI start that never
    sees a cron expression.
    """
    from croniter import croniter  # noqa: PLC0415

    try:
        # The search starts from `now + 1ms`, because `get_prev` is strictly
        # *before* its start. A search from `now` itself skips a moment that is
        # exactly `now`. A scheduler pass that runs exactly on the cron boundary
        # hits that case, and the schedule then fires one pass late, every time.
        # The forward walk that this search replaced included that moment, so
        # the offset keeps the two equivalent.
        found: datetime = croniter(spec, _local(now + 1)).get_prev(datetime)
    except (ValueError, KeyError) as error:
        log.warning("ph.seams.schedule: unusable cron %r (%s)", spec, error)
        return None
    moment = int(found.timestamp() * 1000)
    return moment if moment > after else None


def _int(value: str) -> int | None:
    """A spec as milliseconds, or `None` when it is not a number (K2).

    The function catches `OverflowError` as well as `ValueError`, because
    `int(float(...))` raises `OverflowError` for `1e400` and `inf`. A client can
    send those values, and every numeric reader in this module goes through this
    function. When the error was not handled, it escaped the *fold* itself. One
    bad row then stopped every healthy schedule on that root.
    """
    try:
        return int(float(value))
    except (ValueError, OverflowError):
        return None


def _refuse_unfireable(schedule: Schedule, *, now: int) -> None:
    """Refuse a schedule that the fold will never fire (K2).

    **The function asks the fold and does not restate its rules.** `create`
    used to append and *then* reindex. An unreadable spec therefore left a
    `schedule/created` behind, and it came back on every resume. The fix is to
    refuse before the append. To write out *which* specs never fire would be the
    second half of the fix, and a second set of rules. The first draft of that
    set had already drifted. It refused a `once` at a negative epoch, and
    `due_at` treats that spec as overdue and fires it immediately. That was two
    answers about one row, on the first day.

    The function builds a temporary state and asks `next_at`. That costs one
    `int(float(...))` for the numeric kinds and one `croniter` construction for
    a cron. The answer cannot disagree with the tick, because it *is* what the
    tick asks. It also covers cron at no extra cost. An expression that
    `croniter` cannot read never yields a moment, so its row would stay in the
    log and look live. The first draft skipped cron on purpose, so that it did
    not need a second parser.
    """
    probe = ScheduleState(schedule=schedule, created_at=now)
    if next_at(probe, now=now) is None:
        raise ValueError(
            f"schedule {schedule.id!r} has spec {schedule.spec!r}, which a "
            f"{schedule.kind!r} schedule can never fire from"
        )


@dataclass(slots=True)
class ScheduleService:
    """`ctx.schedule`: create, cancel, and claim what is due.

    The service writes to the log and reads back through the fold. It holds no
    schedule state of its own except a cache of that fold. A resumed session and
    a live session therefore answer the same question in the same way (A11).

    **The fold is cached, like `SubagentService._folds`.** The daemon calls
    `live()` in every scheduler pass, in every plan for the next pass, and at
    every turn transition (`Supervisor.tick`, `Supervisor.next_wake`,
    `Supervisor.booked`). When the daemon ticked every five seconds, the bare
    fold ran tens of thousands of times a day for each root. `SessionFoldCache`
    keys on `session.seq`, and a schedule changes only when something is
    appended. So every read between two appends is a dict lookup.

    The service takes no `ctx`, because nothing here reads one. A test would
    have to invent a value for an argument that the service never uses.

    `index` is how an appointment survives the *process*, and not only the log
    (P6-23). It is optional, and the row passes one in every mode. A service
    built with none (a test) still schedules and leaves nothing for a later
    daemon to find. When the index is present, every write here updates it,
    because the seam is the only party that knows the moment at which an
    appointment changes.
    """

    index: IndexRecorder | None = None
    indexed: bool = True
    """Whether the appointments of this mount belong in the index
    (`Config.index`). When they do not, every write here removes the entry of
    the session instead, so an entry written while the setting was on does not
    outlive it."""
    _states: SessionFoldCache[dict[str, ScheduleState]] = field(
        default_factory=lambda: SessionFoldCache(schedules)
    )

    def states(self, session: Session) -> dict[str, ScheduleState]:
        """Every schedule that this log knows about, folded at most one time for
        each append."""
        return self._states.read(session)

    def create(self, session: Session, schedule: Schedule) -> Schedule:
        """Record a schedule. It is live from the moment the event is appended.

        **The schedule is checked before it is written** (K2). `create` used to
        append and *then* reindex. A spec that the fold could not read left a
        `schedule/created` in the log, and nothing could fold it. The log is
        the state, so the bad row came back on every resume. With the refusal
        first, a schedule that can never fire never becomes a fact about the
        session.
        """
        _refuse_unfireable(schedule, now=now_ms())
        _LOG.append(session, CREATED, schedule.to_wire())
        # This runs straight after the append, with no await between the two.
        # The claim that this call takes must be on disk before the log can
        # flush that record.
        self.reindex(session, new=True)
        return schedule

    def cancel(self, session: Session, schedule_id: str) -> bool:
        """Record a cancellation. Returns `False` when there was nothing to cancel.

        "Nothing to cancel" covers an id that this log never knew *and* an id
        that it already canceled. The fold keeps canceled schedules visible
        (`schedules`). Without the second case, a retry would report success and
        append a redundant `schedule/canceled` each time. To a user who runs
        `phern agents schedule --cancel` twice, "canceled" on the second run is
        a statement about work that was already stopped.
        """
        state = self.states(session).get(schedule_id)
        if state is None or state.canceled:
            return False
        _LOG.append(session, CANCELED, {"id": schedule_id})
        self.reindex(session)
        return True

    def live(self, session: Session) -> list[ScheduleState]:
        """The schedules that could still fire."""
        return [state for state in self.states(session).values() if not state.canceled]

    def next_due(self, session: Session, *, now: int) -> int | None:
        """The earliest moment that any live schedule here wants, or `None` when
        none will fire again. For an overdue schedule, the moment is the one
        that it is overdue for."""
        return _soonest(self.live(session), now=now)

    def claim(self, session: Session, *, now: int) -> list[Schedule]:
        """Claim everything that is due, write-ahead, and return it for delivery.

        The append happens here, and the delivery happens in the caller. With
        that order, a crash costs a skipped run and not a repeated run. A caller
        that never delivers has still recorded that it meant to deliver, and an
        operator needs to see that.

        **A host fires through `fire_due`**, which keeps the order that the rest
        of the rule needs: deliver, write, then tell the index (P13-07). The
        index used to be told here, straight after the append, while the tick
        that it answered for was still in the buffer of this log. A kill between
        the two left an index that said "later" over a log that said "due", and
        the run was lost.
        """
        claimed: list[Schedule] = []
        for state in self.live(session):
            moment = due_at(state, now=now)
            if moment is None:
                continue
            # The due moment goes in the event and nowhere else. One fact with
            # two carriers is a fact that can disagree.
            _LOG.append(session, TICK, {"id": state.schedule.id, "dueAt": moment, "firedAt": now})
            claimed.append(state.schedule)
        return claimed

    def reindex(self, session: Session, *, now: int | None = None, new: bool = False) -> None:
        """Tell the index when this session is next due, or that it is not due.

        The value is the earliest moment that any live schedule of the session
        wants. One entry therefore answers for a session, however many
        appointments it holds. The question of the daemon is "is this root
        worth mounting", and not "which schedule".

        The value is `None` when nothing is outstanding, and that removes the
        entry. A canceled schedule and a `once` that has fired both mean that a
        daemon should stop waking for this session.

        `new` is for `create`. A new schedule is the one change whose loss would
        silence a schedule (`ScheduleIndex.claim`). A lost cancel or a lost tick
        costs one wake at most.
        """
        if self.index is None:
            return
        stamp = now if now is not None else now_ms()
        if not self.indexed:
            self.index.record(session.id, next_at=None, now=stamp)
            return
        self.index.record(session.id, next_at=self.next_due(session, now=stamp), now=stamp, new=new)


async def fire_due(
    ctx: Context,
    session: Session,
    *,
    now: int,
    deliver: Callable[[Schedule], Awaitable[object]],
) -> list[Schedule]:
    """Fire whatever is due on `session`: claim, deliver, write, then tell the
    index (P13-07). Every host fires a schedule through this function, so the
    order belongs to the seam and not to each host. Returns what came due and
    was delivered, in the order in which it was claimed.

    **The tick is before the prompt in the log.** `claim` appends the tick
    before `deliver` can put anything in the inbox. A reader therefore never
    finds a scheduled prompt without the record that says why it came.

    **There is one write, after the delivery.** A prompt has no effect until a
    turn sends it. The first model request of that turn is a barrier
    (`session-checkpoint-policy`). The barrier writes the whole log in order,
    with the tick first, and refuses the request when it cannot write. So no run
    bills without its tick on disk. Because of the write here, a kill before
    that request causes a late run and not a skipped run.

    **The index is told last, and only when the log is written.** When the index
    was told at the claim, it said "later" over a log whose tick was still in a
    buffer, and a kill at that point lost the run. A write that fails leaves the
    index where it was. The moment is then still due to whatever reads the index
    next, and the store reports that the log is unwritten (`session/unwritten`).
    """
    service = ctx.require(SCHEDULE)
    claimed = service.claim(session, now=now)
    if not claimed:
        return []
    for schedule in claimed:
        await deliver(schedule)
    if await session_written(ctx, session):
        service.reindex(session, now=now)
    return claimed


def scheduled_source(schedule: Schedule) -> PluginSource:
    """The source under which a scheduled prompt is delivered. It is the source
    of this seam, and it names the schedule. A prompt that the user sends says
    `user` (P13-04).

    A scheduled turn is an ordinary turn (`Supervisor.tick`), and its prompt is
    the words of the user. But the schedule delivers them hours after they were
    typed, with nobody at the keyboard. A reader of the log tells the two apart
    by the source. A transcript draws the prompt as a notice of the schedule,
    and not as something that the user just said. A resumer that finds the turn
    cut short knows that nobody is waiting to send it again.
    """
    return PluginSource(
        plugin="ph.seams.schedule",
        form="notice",
        summary=f"scheduled: {schedule.id}",
    )


def _soonest(states: Iterable[ScheduleState], *, now: int) -> int | None:
    """The earliest moment that any of `states` wants, or `None` when none wants
    one."""
    moments = [moment for state in states if (moment := next_at(state, now=now)) is not None]
    return min(moments) if moments else None


REBUILD_TYPES = FOLDED | {BASE, OVERRIDE, CLEARED, REFUSED}
"""The records that `rebuild_index` reads from a log: its schedules, and the
environment records that say whether the `schedule` row of the session keeps
the index (`Config.index`)."""


def _indexed(events: Iterable[SessionEvent]) -> bool:
    """Whether the environment that a log records keeps its appointments in the
    index. A log with no `profile/base`, or with no `schedule` row, keeps the
    default, which is on."""
    logged = fold_environment((event.seq, event.type, event.data) for event in events)
    for row in environment(logged):
        if row.name == "schedule":
            return as_obj(row.config).get("index") is not False
    return True


def rebuild_index(
    index: ScheduleIndex,
    logs: Iterable[tuple[str, Sequence[SessionEvent]]],
    *,
    now: int,
    vouch: bool = True,
) -> bool:
    """Rebuild `index` from every stored log, for an index that cannot vouch for
    itself (S18). The function blocks: it reads every log that `logs` yields.
    Returns whether the index was written. `vouch` says whether `logs` come from
    the store of the deployment itself (`ScheduleIndex.replace`).

    `logs` yields the id of each session and its `REBUILD_TYPES` records. A log
    with no schedule yields no schedule records, and that withdraws any entry
    that it had. So does a log whose environment turns the index off
    (`Config.index`): the setting is in its `profile/base`, so a rebuild honors
    it as the mount does. A log that `logs` leaves out keeps its entry
    (`ScheduleIndex.replace`).

    **`updated` is the time at which the log last said anything about its
    schedules**, and not the time at which this function ran. `wake_within`
    judges an abandoned appointment by that value, and a rebuild would otherwise
    make every appointment look freshly confirmed.
    """
    survey = index.survey()
    read: dict[str, Appointment | None] = {}
    for session_id, events in logs:
        if not _indexed(events):
            read[session_id] = None
            continue
        own = [event for event in events if event.type in FOLDED]
        soonest = _soonest(_fold(own).values(), now=now)
        read[session_id] = (
            None
            if soonest is None
            else Appointment(session_id, soonest, max(event.time for event in own))
        )
    return index.replace(read, since=survey, vouch=vouch)


class Config(WireModel):
    """The row config for the schedule seam."""

    index: bool = True
    """Whether the schedules of this session go in the index of what is due, one
    for each `$PH_HOME` (P6-23).

    The index is on by default. Its cost is a small write when an appointment
    changes, and its benefit is a schedule that survives a restart: a daemon
    reads the index and wakes the session when it is due. With the index off, a
    daemon never wakes the session, and its schedules fire only while something
    has it open. The mount removes the entry of the session instead of writing
    one. A daemon that rebuilds the index from the stored logs (S18) reads this
    setting from the `profile/base` of each log and leaves the session out too.
    """


@plugin("schedule", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Publish `ctx.schedule`."""
    # The index is written through the writer, so a change to an appointment
    # never waits for the file lock on the event loop (K10).
    index = IndexWriter(ScheduleIndex(resolve_roots().home), ctx)
    service = ScheduleService(index=index, indexed=config.index)
    ctx.provide(SCHEDULE, service)
    # Opening a session reconciles its entry (P6-23, I-6). An entry that a tick
    # or a cancel left stale, because its write was lost, is corrected here. An
    # index that cannot vouch for itself is for a daemon to rebuild
    # (`rebuild_index`, S18). The write is skipped when nothing moved, so an
    # ordinary open costs a fold that the cache already has.
    ctx.on("session/created", lambda session: service.reindex(session))
    # A table that drifted is how a canceled schedule keeps firing, or how a
    # live schedule stops being claimed. Both are silent, because the log would
    # still say the right thing. The cache itself is bounded by live sessions,
    # because `SessionFoldCache` keeps its entries in a `SessionTable`.
    contribute_fold_cache(
        ctx, id="schedule-fold-cache", subject="schedule table", stale=service._states.stale
    )
