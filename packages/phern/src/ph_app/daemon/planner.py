"""One loop for every job the daemon does at a moment rather than on an event (Phase 12).

Three jobs end on a clock. Firing what is scheduled (`Supervisor.scheduler`),
releasing a root that has been quiet long enough (`Supervisor.releaser`), and the
daemon's own keep-alive and spawn window (`DaemonServer.lifetime_clock`). Each used
to run on a cadence, waking every few seconds or every minute to ask whether
anything had come due. A `Planner` asks instead *when* the next thing comes due,
sleeps until then on the wall clock (`ph.wall_clock.Alarm`), and is woken early by
a notice from whatever can move that moment.

The rules the three would otherwise each carry, held once:

* **A fresh `moved` before the pass**, so a change the pass makes or meets wakes
  the next sleep instead of being swallowed by this one.
* **A failing pass is logged and the loop goes on.** A housekeeping pass that
  raised would otherwise take the daemon's task group, and every root, with it.
  **So is a failing plan**, which sleeps until told: a job that cannot say when it
  is next due waits for the next notice rather than ending the process.
* **`stop` is read before the alarm**, so a daemon stopping at the moment
  something came due stops, instead of counting a due wake.

What each job keeps is its own `plan`: which moments count, and how soon after a
pass the next may come.

**The notices are the risk.** A change that can make something due sooner, with no
notice behind it, waits for an unrelated wake, and on a quiet daemon that may be
never. So each job's notices are listed where they are wired, and each has a test
that changes only that one thing.

@module ph_app.daemon.planner
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio

from ph.cancel import first_of
from ph.session import now_ms
from ph.wall_clock import Alarm

__all__ = ["Planner"]

log = logging.getLogger("ph_app.daemon")


@dataclass(slots=True)
class Planner:
    """A pass, then sleep until the next moment it has work, or until told."""

    what: str
    """Named in the log when a pass fails."""
    run: Callable[[], Awaitable[object]]
    """The pass. Idempotent, so a wake with nothing to do costs one pass."""
    plan: Callable[[int], int | None]
    """When the next pass has work, epoch ms, given now; `None` sleeps until told."""
    rang: int = 0
    """How many times the clock, rather than a notice, ended the sleep. The scheduler
    counts these as reasons a failed index rebuild may now succeed."""
    planned: int | None = None
    """When this means to wake next, or `None` while it sleeps until told. Read by
    `phern agents doctor` rather than worked out again."""
    _moved: anyio.Event = field(default_factory=anyio.Event)

    def notice(self) -> None:
        """The plan may be stale: wake and plan again."""
        self._moved.set()

    async def keep(self, stop: anyio.Event) -> None:
        """Run until `stop`. The first pass is at once."""
        while True:
            self._moved = moved = anyio.Event()
            try:
                await self.run()
                self.planned = self.plan(now_ms())
            except Exception:
                log.exception("ph_app.daemon: %s failed", self.what)
                self.planned = None
            alarm = Alarm(self.planned)
            await first_of(stop, moved, alarm)
            if stop.is_set():
                return
            if alarm.rang:
                self.rang += 1
