"""The CPU budget's arming and disarming, which nothing downstream can observe.

`RLIMIT_CPU` is the one limit that is **not** a one-shot. Linux delivers
`SIGXCPU` when the soft limit is first crossed and then again every CPU-second
for as long as the process stays over it — and a cumulative counter never goes
back down, so "over it" is permanent for the life of the process.

That is invisible from the kernel tests: the first delivery ends the cell, which
is what they assert, and the second one lands in whatever the guest does next.
When that was `_snapshot` it escaped the `try` meant to report failures, the run
sent no `done` frame, and the host — which has no wall clock on a run — waited
for one forever. A test that spins a real cell and hopes to lose the race is not
a test, so the mechanism is pinned here instead.

Both functions mutate this process's own `RLIMIT_CPU`, so every test restores it.
"""

from __future__ import annotations

import sys

import pytest

from ph_runtime.limits import arm_cpu_budget, cpu_seconds_used, relax_cpu_budget

resource = pytest.importorskip("resource", reason="RLIMIT_CPU is POSIX")

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")


@pytest.fixture(autouse=True)
def restore_cpu_limit() -> object:
    """Put this process's own budget back, whatever a test did to it."""
    before = resource.getrlimit(resource.RLIMIT_CPU)
    yield
    resource.setrlimit(resource.RLIMIT_CPU, before)


def test_arming_bounds_the_budget_from_what_is_already_spent() -> None:
    """The per-run budget, out of a counter that only goes up.

    A generous number, because this arms the *test process*: what is asserted is
    the arithmetic — the soft limit lands above what has been spent by about the
    budget — and not a limit anything is meant to reach.
    """
    arm_cpu_budget(3600)

    soft, _hard = resource.getrlimit(resource.RLIMIT_CPU)
    spent = cpu_seconds_used()
    assert soft >= spent + 3600
    # And bounded by it: the whole point of re-arming is that the fortieth cell
    # does not inherit what the first one spent.
    assert soft <= spent + 3601 + 1


def test_relaxing_puts_the_soft_limit_back_and_stops_re_delivery() -> None:
    """The disarm, which is what makes one breach one signal.

    Without it the soft limit stays where `arm_cpu_budget` left it, the process
    stays over it, and the kernel keeps being signalled every CPU-second through
    the teardown that is trying to report the breach.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    arm_cpu_budget(3600)
    assert resource.getrlimit(resource.RLIMIT_CPU)[0] != hard, "the budget was not armed"

    relax_cpu_budget()

    assert resource.getrlimit(resource.RLIMIT_CPU) == (hard, hard)
