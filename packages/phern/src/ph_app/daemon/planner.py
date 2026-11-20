"""One loop for every job that the daemon does at a moment and not on an event (Phase 12).

Four jobs end on a clock:

- firing what is scheduled (`Supervisor.scheduler`)
- releasing a root that has been quiet long enough (`Supervisor.releaser`)
- checking the invariants of a root that has settled (`Supervisor.verifier`)
- the keep-alive and spawn window of the daemon itself
  (`DaemonServer.lifetime_clock`)

Each job used to run on a cadence: it woke every few seconds or every minute to
ask whether anything had come due. A `Planner` asks *when* the next thing comes
due. It sleeps until then on the wall clock (`ph.wall_clock.Alarm`), and a
notice from whatever can move that moment wakes it early.

A `Planner` holds, in one place, the rules that each of the four jobs would
otherwise carry:

- **A fresh `moved` before the pass.** A change that the pass makes or meets
  then wakes the next sleep, and this sleep does not swallow it.
- **A failing pass is logged and the loop continues.** Otherwise a housekeeping
  pass that raised would end the task group of the daemon, and every root with
  it. **A failing plan is handled the same way**, and the loop then sleeps until
  told. A job that cannot say when it is next due waits for the next notice and
  does not end the process.
- **`stop` is read before the alarm.** A daemon that stops at the moment when
  something came due stops, and does not count a due wake.

Each job keeps its own `plan`: which moments count, and how soon after a pass
the next pass may come.

**The notices are the risk.** A change can make something due sooner. If no
notice follows that change, it waits for an unrelated wake, and on a quiet
daemon that wake may never come. The notices of each job are therefore listed
where they are wired, and each notice has a test that changes only that one
thing.

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
    """Run a pass, then sleep until the next moment that has work, or until told."""

    what: str
    """Named in the log when a pass fails."""
    run: Callable[[], Awaitable[object]]
    """The pass. Idempotent, so a wake with nothing to do costs one pass."""
    plan: Callable[[int], int | None]
    """When the next pass has work, in epoch ms, given the current time. `None`
    means sleep until told."""
    rang: int = 0
    """How many times the clock, and not a notice, ended the sleep. The scheduler
    counts these as reasons why a failed index rebuild may now succeed."""
    planned: int | None = None
    """When this planner means to wake next, or `None` while it sleeps until told.
    `phern agents doctor` reads this value and does not work it out again."""
    _moved: anyio.Event = field(default_factory=anyio.Event)

    def notice(self) -> None:
        """The plan may be stale: wake and plan again."""
        self._moved.set()

    async def keep(self, stop: anyio.Event) -> None:
        """Run until `stop`. The first pass runs immediately."""
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
