"""The retry ladder, and the source of its state (P5-04).

A root is the one thing in pH that is built to outlive whatever started it. So
"the turn failed" must not mean "the work stops and somebody notices later".
This ladder retries after 250 ms, 1 s and 5 s, and restores the tree between
attempts. When the third retry also fails, the ladder stops and *reports that*.
It does not retry forever.

**Every part of that state is folded from the log.** The number of attempts
that a root has made, and whether the root gave up, are facts about the session.
They are not fields on the supervisor. If they were, a daemon that crashed in
the middle of the ladder and came back would start the count from zero. It would
then retry a failing turn for as long as the process lives.

The state is folded **one time**, when the root starts. After that,
`Root.retry`, `Root.give_up` and `Root.recovered` maintain it. The state must
not be folded from `Root.status`, because `describe()` reads `Root.status` for
every root on every `sessions/list`.

**The ladder retries the *task* of the root, not a failed turn.** The design
depends on this distinction. `ReactLoopAgent.run` contains its own failures.
`llm-retry` has already retried everything that it makes sense to retry after a
model failure: rate limits, 5xx responses and timeouts. It deliberately does not
retry the rest, because a retry of an unknown failure is an unknown failure that
is billed twice. So a `turn/end{error}` that arrives here is a failure on which
the layer that understands model failures decided to stop.

A retry of a failed turn would also not work. The failed turn already *claimed*
its message from the inbox. A second `run()` therefore finds nothing pending and
ends at `phase.step == 0` with `kind="completed"`. That is an empty turn that
succeeds trivially. It clears the ladder and reports a healthy root that
answered nothing.

What remains is what the row names: the task of the root crashes *around* the
turn. Examples are a flush that cannot write, a disposed context, and a bug. In
these cases the work is still in the inbox, and a second run is meaningful.

@module ph_app.daemon.recovery
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ph.keys import SUBAGENTS
from ph.persistence import Resumption
from ph.seams.live_index import Ladder
from ph.session import Session, SessionEvent

if TYPE_CHECKING:
    from ph.agent.types import AgentDriver
    from ph.cordis import Context

__all__ = [
    "CHILD_RETRY_LIMIT",
    "FAILED",
    "LADDER",
    "RECOVERED",
    "RETRY",
    "RETRY_DELAYS",
    "ROOT_RESUME_LIMIT",
    "UNREACHABLE",
    "VERIFY_AFTER",
    "VIOLATED",
    "WAKE_RETRY_DELAYS",
    "Recovery",
    "floored",
    "interrupted_resumes",
    "recovery_from",
    "recovery_of",
    "resume_children",
]

RETRY = "supervisor/retry"
"""A crashed task is being run again. The record holds the attempt, the delay and
the reason. It is written before any part of the attempt runs. The restore that
can follow is recorded as its own pair (`WORKSPACE_RESTORE`, which the seam that
restores opens)."""

FAILED = "supervisor/failed"
"""The ladder is spent. This root is not working, and did not stop quietly."""

RECOVERED = "supervisor/recovered"
"""A retry worked. The ladder is clear, and the record says how many attempts it
took.

**This is the only record that resets the count**, and it must be a record of
its own and not an existing one. A reset on any `turn/end` is unsound, because
the retry *produces* a `turn/end`. A `run()` that is entered again finds an
empty inbox and appends `turn/start` and `turn/end{completed}` before the same
crash happens again. The ladder would then clear the counter that limits it.
**The failure cannot forge a marker that only *success* writes.**

The records are `supervisor/*` and not `agent/*`. `ph.agent` owns the `agent/*`
namespace, and these are the records of the supervisor *about* an agent, not the
records of the agent itself. Also, `agent/failed` is easy to confuse with
`agent/error`, which means something different.
"""

RETRY_DELAYS: tuple[float, ...] = (0.25, 1.0, 5.0)
"""Three attempts after the first, then failed.

The delays go from short to long on purpose. The only failures that are worth a
retry are transient: a dropped connection, a rate limit, a brief provider fault.
Those clear in milliseconds or not at all. A ladder that started at five seconds
would make the common recovery feel like a hang. A ladder that never gave up
would turn a permanently broken root into a process that retries forever a turn
that cannot succeed, and reports itself busy.
"""


CHILD_RETRY_LIMIT = 3
"""How many times an interrupted **child** is put back to work before it is failed.

The limit is here, next to the ladder of the root, because it answers the same
question one level down: *how many times does this harness run work again after
its own stop interrupted that work*. When the answer was in `ph-core`, one
question became two questions, in two packages, with two vocabularies, for
whoever came to tune it. `ctx.subagents` owns the sweep, the fold and the
records. The host owns the number of attempts that they are worth.
`ctx.subagents.resume_children` takes that number with no default, so a host
cannot get the number without stating it. `--mode rpc` and the one-shot modes
also sweep with this limit, through `resume_children` below. A second host is
not a second policy.

The limit is three for the same reason that `RETRY_DELAYS` has three rungs.
What a harness stop interrupts is transient by construction. A child that was
caught in the middle of a turn three times is not unlucky. It is in front of
something that keeps stopping. The number is written out and is not
`len(RETRY_DELAYS)`, because the two are *independent* policies that happen to
agree. This ladder has no delays to take a length from, and a shorter root
ladder must not silently shorten this one.

**This ladder has no delays, unlike the ladder of the root.** The root ladder
retries a crash that just happened, and waits before it tries again. This ladder
runs only while a harness starts, and the start is already the wait. For the
same reason the two ladders do not share `Recovery`. The `total` of `Recovery`
is `len(RETRY_DELAYS)` and its `delay` indexes them. A child would either report
the bound of the root, or force delays into a type that has none. `spent` is the
word that is worth sharing, and it costs nothing to write it twice.
"""


ROOT_RESUME_LIMIT = 3
"""How many times the interrupted turn of a **root** is continued before the root
is failed (P13-04). It is the counterpart of `CHILD_RETRY_LIMIT` for a root, and
it is the same number for the same reason. A turn that a restart cut short three
times in a row is not unlucky. It is in front of something that keeps stopping.

`interrupted_resumes` does the count: the resumes that had to close a turn since
the root last answered. An answer forgives the restarts before it, as the answer
of a child does (`restarts_since_progress`). So a long run that met two
incidents an hour apart counts as two attempts, and not as every restart of its
lifetime. Past the limit, the root is failed in its log (`supervisor/failed`).
`Root.status` reads that record as `failed`, and the live-roots index carries
it. A resumer that comes back every minute therefore does not spend one more
attempt for each restart."""


def interrupted_resumes(session: Session) -> int:
    """How many resumes in this log cut a turn short since the root last answered.
    This is the ladder of the root, folded from records that the resume already
    writes.

    A `session/resumed` that names a `turn` is a restart that cut a turn short
    (`Resumption.turn`). A `session/resumed` that only settled an intent between
    turns is not. An `assistant/message` is an answer, and a turn that did
    nothing cannot produce one. The function walks back from the tail to the last
    answer, so its cost is the events since that answer and not the whole log. It
    reads own events only, for the reason that `recovery_of` gives: a fork
    inherits the transcript of its source, not the restarts of its source.
    """
    count = 0
    for seq in range(session.seq - 1, session.header.first_own_seq - 1, -1):
        event = session.at(seq)
        if event is None or event.type == "assistant/message":
            break
        if event.type == "session/resumed" and Resumption.of(event).turn is not None:
            count += 1
    return count


async def resume_children(ctx: Context, agent: AgentDriver) -> Sequence[str]:
    """Settle what the children of the session of `agent` are owed (P5-04). Each
    child is readmitted, held, or ended, and each in its own log. Returns the
    revived children.

    Every host calls this after it has made the agent of a root: the supervisor
    of the daemon, `--mode rpc`, and the one-shot modes
    (`ph_app.runtime.prompted`). The reason is that a readmitted child is
    attached to the agent of its parent. `CHILD_RETRY_LIMIT` bounds the call: one
    harness has one answer to how many restarts the work of a child is worth.
    """
    subagents = ctx.get(SUBAGENTS)
    if subagents is None:
        return ()
    return await subagents.resume_children(agent, retry_limit=CHILD_RETRY_LIMIT)


@dataclass(frozen=True, slots=True)
class Recovery:
    """What this session's log says about the state of the ladder."""

    attempts: int
    """Crashes retried since the ladder last cleared."""
    failed: bool
    """The ladder was exhausted and nothing has succeeded since."""

    @property
    def total(self) -> int:
        """How many retries the ladder allows, read at the time of the call.

        This is a property, and not a value that is copied into each record at
        import time. A deployment or a test that shortens the ladder is then
        described by the number that is in force, and not by the number that
        was in force when this module was first imported.
        """
        return len(RETRY_DELAYS)

    @property
    def spent(self) -> bool:
        return self.attempts >= self.total

    @property
    def delay(self) -> float:
        """How long to wait before the next attempt."""
        return RETRY_DELAYS[min(self.attempts, self.total - 1)]


def recovery_of(session: Session) -> Recovery:
    """Fold the state of the ladder from the log. This is done one time for each
    root, at start.

    The fold starts at `seed_length`, not at zero. A fork seeds the child with
    the transcript of the parent. If the retries of the parent counted as the
    retries of the child, a fresh root would start with a spent ladder.
    `workspace_leaks` folds from the same boundary for the same reason.
    """
    return recovery_from(session.own_events())


def recovery_from(events: Iterable[SessionEvent]) -> Recovery:
    """The state of the ladder, folded from the own events of a root. This is the
    fold of `recovery_of`. It is also the fold of the live-roots index (`LADDER`),
    which reads a stored log and has no `Session` to ask.

    **Only `supervisor/recovered` resets the count.** Nothing that the ladder
    itself can write may clear it. See `RECOVERED` for the unbounded loop that
    followed from a reset on `turn/end`. This fold does not read `turn/end` at
    all. A `turn/end` means that the driver reached its own containment and
    returned, and that is exactly what this ladder does not count. A failed turn
    is the concern of `llm-retry` and of the transcript, not of the supervisor.
    """
    attempts = 0
    failed = False
    for event in events:
        if event.type == RECOVERED:
            attempts, failed = 0, False
        elif event.type == RETRY:
            attempts += 1
        elif event.type == FAILED:
            failed = True
    return Recovery(attempts=attempts, failed=failed)


LADDER = Ladder(
    types=frozenset({RETRY, FAILED, RECOVERED}), spent=lambda events: recovery_from(events).failed
)
"""This ladder, for the rebuild of the live-roots index (`rebuild_live_index`). A
root whose log says that it gave up is written back as `failed`, so a resumer
skips it."""


PASSIVATED = "supervisor/passivated"
"""A root was released for being idle (P5-05), with how long it had been."""

UNREACHABLE = "supervisor/unreachable"
"""The daemon lost the socket that it was bound to. The record is written into
every live root (P5-11).

The record is here with the other `supervisor/*` records, because this module
holds the records of the supervisor *about* a root. It has the same shape as
`PASSIVATED`: a fact that the log must carry, because the alternative is an
unexplained gap. The subject is the unusual part. This is the record of the
supervisor about *itself*, copied to every root. It must be so, because the
surfaces that would otherwise carry it are exactly the surfaces that just went
away. Nobody can connect to be told, and nobody who is attached is guaranteed to
still be there. What remains is the transcript of each root, which is where
somebody looks in the end.

The record is not a reason to stop. The roots keep working, which is the
inversion that P5-01 made: their tasks hold no reference to a connection. To
kill them because the socket went away would lose an hour of work in progress to
a socket problem.
"""

VIOLATED = "supervisor/violated"
"""A pollable invariant did not hold when the daemon last asked (I6).

The record is here for the same reason as `UNREACHABLE`, and it has the same
unusual subject: it is the record of the supervisor about *itself*, written into
the root that it concerns. The difference is the reader. Whoever opens the
transcript reads an unreachable socket afterwards. Whoever is looking at the
session right now reads this record. A projection that no longer equals its fold
means that what they are reading may not be what the model saw.

**The finding is recorded, and not only logged.** That is the reason why this
constant exists. `ph.seams.invariants` states that a check is "at best unknown
and at worst the failure itself". A finding that exists only in the stderr of a
process is lost exactly when the process is the suspect. The log is the surface
that survives the thing that it reports on.

The record is not a reason to stop, as with `UNREACHABLE`. A stale projection is
a serious bug, but it is not grounds to kill an hour of the work of a user. The
record is the alarm. The user decides what to do about it.
"""

PASSIVATE_AFTER = 90 * 60.0
"""Seconds of quiet before a root is released. `None` turns release off.

The value is ninety minutes because passivation is not a resource emergency. Its
purpose is that a daemon which has collected a month of sessions does not hold a
mounted profile, a kernel and a workspace for every one of them. The value is
short enough that an abandoned root does not outlive the day. It is long enough
that the root is still there when a user who stepped away from a session comes
back, or when an agent between two scheduled runs comes back.

The quiet time is measured from the last event of the log itself, and not from a
timer. So it means "nothing has happened". It survives a restart: a root that is
rehydrated from a three-day-old log is eligible immediately, which is correct.
It also cannot drift from what the transcript shows.
"""


PASS_FLOOR: float = 1.0
"""The least time, in seconds, between two passes of one kind that the clock
wakes: scheduler passes, invariant checks, or root releases.

Each kind sleeps until its next moment and does not tick, so nothing else limits
how often it runs. An `interval` of a millisecond, or a root whose release
raised, would be a busy loop. A second is finer than any schedule that a person
sets, and a pass costs a turn.
"""


VERIFY_AFTER: float = 30.0
"""Seconds that a root must have been settled, with nothing written in its mount,
before its pollable invariants are checked (I6, P12-04).

This is a debounce, not a cadence. Nothing is checked until something was
written, and then one check runs after the writing stops. A refold is O(events):
161 ms on a 500 000-event log, on the event loop. So an autonomous root that
wakes back to back, or a client that stages five files, pays for one check and
not for one check per wake or per call. The value is short enough that a drift
is on the record within a minute of the work that caused it.
"""


def floored(moment: int, last_pass: int) -> int:
    """`moment`, or `PASS_FLOOR` after `last_pass` if that is later (epoch ms)."""
    return max(moment, last_pass + int(PASS_FLOOR * 1000))


WAKE_RETRY_DELAYS: tuple[float, ...] = (30.0, 60.0, 120.0, 300.0, 600.0)
"""Seconds between a scheduler pass that could not mount a session for its
appointment and the next pass that tries again. Each consecutive failing pass
moves one step up this ladder, and the last step repeats (`Supervisor.rehydrate`,
`next_wake`).

This is a one-shot backoff of the same kind as `RETRY_DELAYS`, not a cadence.
Nothing is tried while no wake has failed. The moment is planned, `phern agents
doctor` names it, and it is dropped the first time that a pass wakes everything
that it meant to wake. The backoff exists because the usual reason for a failed
wake is a lease that another process holds on the log of the session: a
`phern -p` run on it at the appointed minute. This daemon cannot be told when
that process releases the lease. Without the backoff, nothing planned for an
appointment that the pass left behind. `next_wake` drops moments that were
already attempted, so a quiet daemon never looked again, and `booked()` kept the
daemon up for a schedule that it would never keep. A pass that declines an
appointment as too stale (`WAKE_WITHIN`) has not failed, and the appointment is
not retried: it will not be younger next time.
"""

WAKE_WITHIN: float | None = None
"""How stale an indexed appointment may be and still wake its root, or `None`.

**The default is `None`: a daemon catches up on whatever it missed, however long
it was down.** That is all that the scheduler of pH promises. It keeps an
appointment while it runs. On start it continues from where it stopped, and
`claim` coalesces what it missed to one run for each missed window. A default
bound would be a second policy on top of the one that `claim` already settled.
The argument of the row itself is that pH does not try to be a better cron than
cron.

The setting exists because a schedule is attached to a *conversation* and not to
a crontab entry. "Resurrect a session that was abandoned in March" is therefore
a case that the OS tools do not have. A deployment that does not want that can
set a bound here. The default assumes that a schedule which somebody created
deliberately is a schedule that they meant.

The bound only ever applies to appointments that nobody has kept. A daemon that
serves an appointment refreshes its entry every time that the appointment fires.
"""
