"""`Planner` (Phase 12): the one loop behind every job the daemon does at a moment.

Its rules are the ones each converted loop would otherwise carry on its own, and
the one tested here is the one whose omission ends the process: a pass or a plan
that raises is logged and the loop goes on. `keep` runs as a task in the daemon's
task group, so an exception out of it cancels every root.
"""

from __future__ import annotations

import anyio
import pytest
from daemon_helpers import until

from ph_app.daemon.planner import Planner

pytestmark = pytest.mark.anyio


async def test_a_plan_that_raises_leaves_the_loop_asleep_until_told() -> None:
    """The pass was guarded from the start; the plan one line below it was not, and
    the four plans are the lines that read the schedule index and the folds.

    Sabotage: move `self.plan(...)` back out of the `try`, and the task group
    raises instead of the second pass running.
    """
    passes = 0
    plans: list[int] = []

    async def run() -> None:
        nonlocal passes
        passes += 1

    def plan(now: int) -> int | None:
        plans.append(now)
        if len(plans) == 1:
            raise RuntimeError("the index could not be read")
        return None

    planner = Planner("a test job", run=run, plan=plan)
    stop = anyio.Event()
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(planner.keep, stop)
        await until(lambda: len(plans) == 1, what="the first plan to be asked")
        assert passes == 1
        assert planner.planned is None, "a plan that failed sleeps until told"

        planner.notice()
        await until(lambda: passes == 2, what="the loop to go on after a failing plan")
        assert len(plans) == 2
        stop.set()
