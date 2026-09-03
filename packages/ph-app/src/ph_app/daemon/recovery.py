"""The retry ladder, and where it reads its own state from (P5-04).

A root is the one thing in pH built to outlive whatever started it, so "the turn
failed" cannot mean "the work stops and somebody notices eventually". This ladder
retries at 250 ms, 1 s and 5 s, restores the tree between attempts, and after the
third failure stops and *says so* rather than retrying forever.

**Every part of that state is folded from the log.** How many attempts a root has
made and whether it gave up are facts about the session, not fields on the
supervisor — a daemon that crashed mid-ladder and came back would otherwise start
the count from zero and retry a failing turn for as long as the process lives.

Folded **once**, at root start, and then maintained through `Root.retry` /
`give_up` / `recovered`, which is `Root.commands`' arrangement exactly. It must
not be folded from `Root.status`, which `describe()` reads for every root on every
`sessions/list`.

**What this ladder retries is the root's *task*, not a failed turn**, and the
distinction is the whole design. `ReactLoopAgent.run` contains its own failures
and `llm-retry` has already retried everything a model failure makes sense to
retry — rate limits, 5xx, timeouts — while deliberately declining the rest,
because an unknown failure retried is an unknown failure billed twice. So a
`turn/end{error}` arriving here is a failure the layer that understands model
failures decided to stop on.

It would also not work: the failed turn already *claimed* its message from the
inbox, so a second `run()` finds nothing pending and ends at `phase.step == 0`
with `kind="completed"` — a trivially successful empty turn that clears the ladder
and reports a healthy root which answered nothing.

What is left is what the row names: the root's task crashing *around* the turn —
a flush that cannot write, a disposed context, a bug — where the work is still in
the inbox and running again is meaningful.

@module ph_app.daemon.recovery
"""

from __future__ import annotations

from dataclasses import dataclass

from ph.session import Session

__all__ = [
    "FAILED",
    "RECOVERED",
    "RETRY",
    "RETRY_DELAYS",
    "UNREACHABLE",
    "Recovery",
    "recovery_of",
]

RETRY = "supervisor/retry"
"""A crashed task is being run again — attempt, delay, and what was restored."""

FAILED = "supervisor/failed"
"""The ladder is spent. This root is not working, and did not stop quietly."""

RECOVERED = "supervisor/recovered"
"""A retry worked. The ladder is clear, and says how many attempts it took.

**The one thing that resets the count**, and it has to be its own record rather
than an existing one. Resetting on any `turn/end` is unsound because the retry
*manufactures* one: a re-entered `run()` finds an empty inbox and appends
`turn/start` + `turn/end{completed}` before the same crash happens again, so the
ladder would clear the counter that bounds it. **A marker only *success* writes
cannot be forged by the failure.**

`supervisor/*`, not `agent/*`: `ph.agent` owns that namespace, and these are the
supervisor's records *about* an agent rather than the agent's own — `agent/failed`
sits one letter from `agent/error`, which means something else entirely.
"""

RETRY_DELAYS: tuple[float, ...] = (0.25, 1.0, 5.0)
"""Three attempts after the first, then failed.

Short-then-long on purpose: the failures worth retrying at all are transient —
a dropped connection, a rate limit, a provider hiccup — and those clear in
milliseconds or not at all. A ladder that started at five seconds would make the
common recovery feel like a hang, and one that never gave up would turn a
permanently broken root into a process that retries a doomed turn forever while
reporting itself busy.
"""


@dataclass(frozen=True, slots=True)
class Recovery:
    """What this session's log says about the state of the ladder."""

    attempts: int
    """Crashes retried since the ladder last cleared."""
    failed: bool
    """The ladder was exhausted and nothing has succeeded since."""

    @property
    def total(self) -> int:
        """How many retries the ladder allows, read when asked.

        A property rather than a value copied into each record at import time,
        so a deployment that shortens the ladder — or a test that does — is
        described by the number actually in force rather than the one that was
        in force when this module was first imported.
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
    """Fold the ladder's state out of the log. Once per root, at start.

    From `seed_length`, not from zero: a fork seeds the child with the parent's
    transcript, and counting the parent's retries as the child's would hand a
    fresh root a spent ladder. `workspace_leaks` folds from the same boundary
    for the same reason.

    **Only `supervisor/recovered` resets the count.** Nothing the ladder itself
    can write may clear it — see `RECOVERED` for the unbounded loop that
    followed from resetting on `turn/end`. `turn/end` is not consulted at all
    here: it means the driver reached its own containment and returned, which is
    exactly what this ladder does not count, and a failed turn is `llm-retry`'s
    business and the transcript's rather than the supervisor's.
    """
    attempts = 0
    failed = False
    for event in session.events_from(session.header.seed_length or 0):
        if event.type == RECOVERED:
            attempts, failed = 0, False
        elif event.type == RETRY:
            attempts += 1
        elif event.type == FAILED:
            failed = True
    return Recovery(attempts=attempts, failed=failed)


PASSIVATED = "supervisor/passivated"
"""A root was released for being idle (P5-05), with how long it had been."""

UNREACHABLE = "supervisor/unreachable"
"""The daemon lost the socket it was bound to, written into every live root (P5-11).

Here beside the other three because this module is where the supervisor's
records *about* a root live, and the shape is the same one `PASSIVATED` argued
for: a fact the log has to carry because the alternative is an unexplained gap.
The subject is the odd part — this is the supervisor's record about *itself*,
fanned out to every root — and it has to be, because the surfaces that would
otherwise carry it are exactly the ones that just went away. Nobody can connect
to be told; nobody attached is guaranteed to still be there. What remains is
each root's own transcript, which is where somebody eventually looks.

Not a reason to stop. The roots keep working, which is P5-01's whole inversion —
their tasks hold no reference to a connection — and killing them because the
front door fell off would lose an hour of in-flight work to a socket problem.
"""

PASSIVATE_AFTER = 90 * 60.0
"""Seconds of quiet before a root is released. `None` turns the sweeper off.

Ninety minutes because passivation is not a resource emergency — it is what
keeps a daemon that has accumulated a month of sessions from holding a mounted
profile, a kernel and a workspace for every one of them. Short enough that an
abandoned root does not outlive the day; long enough that a person who stepped
away from a session, or an agent between two scheduled runs, comes back to a
root that never went anywhere.

Measured from the log's own last event rather than from a timer, so it means
"nothing has happened", survives a restart — a root rehydrated from a
three-day-old log is immediately eligible, which is correct — and cannot drift
from what the transcript shows.
"""


EPHEMERAL_QUIET = 60.0
"""Seconds of quiet before an *auto-started* daemon releases a root — and, one
sweep later, before it exits (P7-08).

The same number for both because it is the same question asked twice: "is anyone
still using this?" A minute, against `PASSIVATE_AFTER`'s ninety, and the
difference is a difference of intent rather than of tuning. A service daemon
holds a warm root *because* holding it is the point — a person who stepped away
comes back to a mounted profile and a live kernel. A daemon a UI spawned behind
somebody's back has no such mandate: it exists for as long as the UI does, and
the cheapest honest thing it can do afterwards is leave.

**This is the interaction that decides the whole design.** Left at ninety
minutes, an ephemeral daemon could not reach an empty `roots` — and so could not
satisfy its own exit predicate — until ninety minutes after the last turn, which
would make "ephemeral" a word with no behaviour behind it for an hour and a half.

Reattaching afterwards costs a mount plus resuming the log, which is why this is
not smaller: a person flipping between two terminals should not pay a rehydrate
for the pause between them.
"""

WAKE_WITHIN: float | None = None
"""How stale an indexed appointment may be and still wake its root, or `None`.

**`None` by default: a daemon catches up on whatever it missed, however long it
was down.** That is the whole of what pH's scheduler promises — it keeps an
appointment while it runs, and on start it picks up where it left off, coalesced
by `claim` to one run per missed window. Bounding that by default would be a
second policy on top of the one `claim` already settled, and the row's own
argument is that pH is not in the business of out-cronning cron.

The knob exists because a schedule is attached to a *conversation* rather than to
a crontab entry, so "resurrect a session abandoned in March" is a shape the OS
tools do not have. A deployment that would rather not can set a bound here; the
default is that a schedule somebody deliberately created is a schedule they meant.

It only ever bites on appointments nobody has kept: one a daemon is serving
refreshes its entry every time it fires.
"""
