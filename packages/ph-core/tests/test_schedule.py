"""P5-06 — `ctx.schedule`: what a root will do later, folded from its own log.

The seam is arithmetic over a log plus one write-ahead append, so these drive it
directly. `now` is a parameter precisely so three hours can pass without three
hours passing.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

from ph.seams.schedule import (
    CANCELLED,
    TICK,
    Schedule,
    ScheduleService,
    due_at,
    next_at,
    schedules,
)
from ph.session import Session, now_ms

MINUTE = 60_000
HOUR = 60 * MINUTE


@contextmanager
def _in_timezone(name: str) -> Iterator[None]:
    """Run a block as if this machine were somewhere else.

    `time.tzset()` rather than `$TZ` alone: `datetime.fromtimestamp` reads the C
    library's cached zone, not the environment, so setting the variable without
    the reset changes nothing and the test passes for no reason.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _sched(kind: str, spec: str) -> tuple[Session, ScheduleService, int]:
    """A session holding one schedule, the service, and when it was created.

    Through `create` rather than appending the event by hand: a helper that
    wrote its own `schedule/created` would keep passing if `create` grew a
    default, while testing a write path the seam no longer owns.

    The creation stamp comes back because every `now` below is relative to it —
    a schedule is anchored at the moment it was made, so a test clock starting
    at zero is decades *before* its own schedule exists.
    """
    session = Session("sched")
    service = ScheduleService()
    service.create(session, Schedule(id="s1", kind=kind, spec=spec, prompt="go"))
    return session, service, schedules(session)["s1"].created_at


def test_a_run_of_missed_ticks_collapses_to_one() -> None:
    """The row's gate.

    A five-minute schedule on a laptop that slept for three hours has
    thirty-six fire times behind it. Delivering all of them turns a nap into a
    stampede; delivering the oldest works through the backlog for another three
    hours before catching up. One tick, naming the most recent due moment — and
    the gap stays visible because the previous tick is still in the log.
    """
    session, service, made = _sched("interval", str(5 * MINUTE))

    claimed = service.claim(session, now=made + 3 * HOUR)
    assert len(claimed) == 1, "a slept-through backlog delivered more than one run"
    assert _last_due(session) == made + 3 * HOUR, "the tick named an old due moment, not the latest"

    assert service.claim(session, now=made + 3 * HOUR + MINUTE) == []
    assert len(service.claim(session, now=made + 3 * HOUR + 5 * MINUTE)) == 1


def test_the_claim_is_written_before_the_caller_can_deliver() -> None:
    """A10, applied to time.

    A tick recorded and then lost to a crash costs one skipped run; a tick
    delivered and then lost costs a *repeated* one — and repeating a scheduled
    prompt bills twice and puts a turn in the transcript nobody asked for. The
    append happens inside `claim`, before it returns anything to deliver, so a
    caller cannot get the order wrong: it has nothing to deliver until the
    record exists.
    """
    session, service, made = _sched("interval", str(MINUTE))

    service.claim(session, now=made + 90_000)
    ticks = [event for event in session.events_from(0) if event.type == TICK]
    assert len(ticks) == 1
    assert ticks[0].data["dueAt"] == made + MINUTE
    assert ticks[0].data["firedAt"] == made + 90_000


def test_an_interval_does_not_drift_when_a_tick_lands_late() -> None:
    """Due, not claimed, is what the next one counts from.

    A tick claimed four seconds late must not push the following one four
    seconds later, or a one-minute schedule becomes a sixty-four-second one and
    then a sixty-eight-second one.
    """
    session, service, made = _sched("interval", str(MINUTE))

    service.claim(session, now=made + 64_000)
    service.claim(session, now=made + 125_000)
    assert _last_due(session) == made + 2 * MINUTE, (
        "the schedule drifted by the previous tick's lateness"
    )


def test_an_interval_is_anchored_at_its_creation_not_at_the_epoch() -> None:
    """The bug a real caller would have hit on its first schedule.

    `created_at` was a field on the wire model that no caller set, so every
    schedule the daemon made anchored at epoch 0 — an hourly interval was due
    the moment it existed, and a cron's first claim walked forward from 1970.
    The anchor now comes from the `schedule/created` event's own timestamp,
    which is where `last_tick` already came from.
    """
    session = Session("fresh")
    service = ScheduleService()
    service.create(session, Schedule(id="s1", kind="interval", spec=str(HOUR), prompt="go"))
    created = schedules(session)["s1"].created_at

    assert created > 0, "the schedule was not anchored to when it was created"
    assert service.claim(session, now=created + MINUTE) == [], "fired immediately on creation"
    assert len(service.claim(session, now=created + HOUR)) == 1


def test_once_fires_once_and_then_never() -> None:
    """`once` names an absolute instant, so it is built from one.

    The other kinds are relative to creation; this one is not, which is exactly
    why `spec` says so on the field and why this test does not go through the
    shared helper — a `once` spec of `10 * MINUTE` means ten past midnight in
    1970, and would fire on its first claim.
    """
    session = Session("once")
    service = ScheduleService()
    fire_at = now_ms() + 10 * MINUTE
    service.create(session, Schedule(id="s1", kind="once", spec=str(fire_at), prompt="go"))

    assert service.claim(session, now=fire_at - MINUTE) == []
    assert len(service.claim(session, now=fire_at + MINUTE)) == 1
    assert service.claim(session, now=fire_at + 12 * HOUR) == [], "a once schedule fired twice"


def test_cron_coalesces_the_same_way() -> None:
    """Every minute, asleep for an hour: one tick, naming the latest minute."""
    session, service, made = _sched("cron", "* * * * *")

    assert len(service.claim(session, now=made + HOUR)) == 1
    assert _last_due(session) <= made + HOUR


def test_a_cron_moment_is_never_claimed_twice() -> None:
    """At-most-once, which is what the tick's memory is for.

    A second pass at the same instant must find nothing: without reading the
    last claimed `dueAt`, the seek returns the same moment again — a duplicate
    delivery of one scheduled run, which for a prompt means the model answers
    twice and the transcript shows a turn nobody asked for.
    """
    session, service, made = _sched("cron", "* * * * *")
    # Aligned to a minute boundary, because cron matches wall-clock moments and
    # `made` is a real timestamp: a `now` landing at :59.7 puts "thirty seconds
    # later" into the *next* minute, so the between-minutes assertion below
    # would fire and the test would fail a few times an hour.
    base = (made // MINUTE + 60) * MINUTE

    service.claim(session, now=base)
    assert _last_due(session) == base
    assert service.claim(session, now=base) == [], "the same cron moment fired twice"
    assert service.claim(session, now=base + 30_000) == [], "fired between minutes"

    service.claim(session, now=base + MINUTE)
    assert _last_due(session) == base + MINUTE


def test_a_cancelled_schedule_stops_firing_and_stays_visible() -> None:
    """Kept as a record, for `subagent_roster`'s reason.

    A caller asking what happened to the schedule it revoked deserves an answer
    other than silence, and the log shows the cancellation either way.
    """
    session, service, made = _sched("interval", str(MINUTE))

    assert service.cancel(session, "s1") is True
    assert service.claim(session, now=made + HOUR) == []
    assert service.live(session) == []
    assert schedules(session)["s1"].cancelled is True
    assert [event.type for event in session.events_from(0)].count(CANCELLED) == 1

    assert service.cancel(session, "never-existed") is False
    # And cancelling twice is not a second cancellation: the fold keeps a
    # cancelled schedule visible, so a membership test alone would report
    # success and append a redundant record on every retry.
    before = session.seq
    assert service.cancel(session, "s1") is False
    assert session.seq == before


def test_a_cron_hour_is_the_hour_on_this_machine_s_clock() -> None:
    """`0 9 * * *` is nine in the morning where the person who wrote it is.

    croniter reads a *float* start time as UTC and a naive *datetime* as local,
    so the same expression meant two different things depending on which the
    caller happened to pass — and this seam passed floats. `0 9 * * *` fired at
    09:00 UTC, which is four in the morning on US Central, with nothing on the
    wire or in the listing to say so.

    Pinned under a fixed zone rather than the machine's, because on a UTC
    builder the two readings agree and the test would pass either way.
    """
    with _in_timezone("America/Chicago"):
        session, service, made = _sched("cron", "0 9 * * *")
        state = service.states(session)["s1"]
        moment = next_at(state, now=made)
        assert moment is not None
        local = datetime.fromtimestamp(moment / 1000)
        assert local.hour == 9, f"fired at {local}, not nine in the morning"
        assert datetime.fromtimestamp(moment / 1000, UTC).hour != 9, "read as UTC"


def test_an_unusable_cron_declines_rather_than_raising() -> None:
    """A bad expression is one dead schedule, not a dead scheduler.

    The daemon fires every root's schedules from one loop, so an expression a
    person typed wrong must not take the others with it.
    """
    session, _, made = _sched("cron", "not a cron")
    assert due_at(schedules(session)["s1"], now=made + HOUR) is None


@pytest.mark.parametrize("spec", ["0", "-30", "nonsense"])
def test_a_nonsensical_interval_never_fires(spec: str) -> None:
    """Zero would be a busy loop and negative would fire forever."""
    session, _, made = _sched("interval", spec)
    assert due_at(schedules(session)["s1"], now=made + HOUR) is None


def test_a_log_with_no_schedules_is_not_walked() -> None:
    """The common root has none, and `schedule` is a base-bundle row.

    So the fold answers from `Session.latest`, which is incremental, rather than
    walking a log that measured 22.6 ms at 500 000 events to return `{}`.
    """
    session = Session("empty")
    for index in range(50):
        session.append("assistant/chunk", {"i": index})
    assert schedules(session) == {}


def _last_due(session: Session) -> int:
    """The `dueAt` of the most recent tick — the log's copy, the only copy."""
    ticks = [event for event in session.events_from(0) if event.type == TICK]
    return int(ticks[-1].data["dueAt"])
