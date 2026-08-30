"""The retry ladder, and where it reads its own state from (P5-04).

A root is the one thing in pH built to outlive whatever started it, so "the turn
failed" cannot mean "the work stops and somebody notices eventually". This is the
ladder that decides otherwise: retry at 250 ms, 1 s and 5 s, restore the tree
between attempts, and after the third failure stop and *say so* rather than
retrying forever.

**Every part of that state is folded from the log.** How many attempts a root has
made and whether it gave up are facts about the session, not fields on the
supervisor — a daemon that crashed mid-ladder and came back would otherwise start
the count from zero and retry a failing turn for as long as the process lives.

Folded *once*, at root start, and then maintained through `Root.retry` /
`give_up` / `recovered` — which is `Root.commands`' arrangement exactly, and for
the measurement `Root.accepted` records: a whole-log scan per read cost 4.9 ms
at 200 000 events on the daemon's event loop. The first draft called this fold
from `Root.status`, which `describe()` reads for every root on every
`sessions/list`; measured at 2.3 ms per root at 100 000 events, 12.5 ms at
500 000, and 128 ms for fifty roots at once, with no await point between them.

**What this ladder retries is the root's *task*, not a failed turn**, and the
distinction is the whole design. `ReactLoopAgent.run` contains its own failures
("a driver that propagated would take the process down with one bad turn") and
`llm-retry` has already retried everything a model failure makes sense to retry —
rate limits, 5xx, timeouts — while deliberately declining the rest, because "an
unknown failure retried is an unknown failure billed twice". So a
`turn/end{error}` arriving here is a failure the layer that understands model
failures decided to stop on, and re-running it from up here would be that exact
mistake one level higher.

It would also not work. The failed turn already *claimed* its message from the
inbox, so a second `run()` finds nothing pending and the driver ends the turn at
`phase.step == 0` with `kind="completed"` — a trivially successful empty turn
that clears the ladder and reports a healthy root which answered nothing. That is
strictly worse than not retrying, and it is what the first draft of this module
did until the row's own tests showed the retry turn completing with no request
made. Re-splicing the claimed message instead would append a second
`user/message` and show the model the same prompt twice.

What is left is what the row names: the root's task crashing *around* the turn — a
flush that cannot write, a disposed context, a bug — where the work is still in
the inbox and running again is meaningful.

@module ph_app.daemon.recovery
"""

from __future__ import annotations

from dataclasses import dataclass

from ph.session import Session

__all__ = ["FAILED", "RECOVERED", "RETRY", "RETRY_DELAYS", "Recovery", "recovery_of"]

RETRY = "supervisor/retry"
"""A crashed task is being run again — attempt, delay, and what was restored."""

FAILED = "supervisor/failed"
"""The ladder is spent. This root is not working, and did not stop quietly."""

RECOVERED = "supervisor/recovered"
"""A retry worked. The ladder is clear, and says how many attempts it took.

**The one thing that resets the count**, and the reason it has to be its own
record rather than an existing one. The first version reset on any `turn/end`,
which the retry *manufactures*: a re-entered `run()` finds an empty inbox and
appends `turn/start` + `turn/end{completed}` before the same crash happens
again, so the ladder cleared the counter that bounds it. Measured against a
persistently failing flush: **165 retries in two seconds, no give-up, the fold
pinned at one attempt and the root reporting "idle"** — the unbounded retry this
row exists to prevent, growing the log by three events an iteration. A marker
only *success* writes cannot be forged by the failure.

`supervisor/*`, not `agent/*`: `ph.agent` owns that namespace (its registry
declares `agent/status`, `agent/error`, `agent/inbox/*` with `owner="ph.agent"`)
and these are the supervisor's records about an agent, not the agent's own —
`agent/failed` sat one letter from `agent/error`, which means something else
entirely.
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
