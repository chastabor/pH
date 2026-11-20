"""Arming and disarming the CPU budget, which nothing downstream can observe.

`RLIMIT_CPU` is the one limit that does **not** fire only one time. Linux
delivers `SIGXCPU` when the soft limit is first crossed, and then again every
CPU-second while the process stays over the limit. A cumulative counter never
goes down, so the process stays over the limit for the rest of its life.

The kernel tests cannot see this. The first delivery ends the cell, which is what
they assert, and the second delivery lands in whatever the guest does next. When
that was `_snapshot`, the signal escaped the `try` that was meant to report
failures. The run sent no `done` frame, and the host, which has no wall clock for
a run, waited for the frame forever. A test that spins a real cell and hopes to
lose the race is not a test, so the mechanism is tested here directly.

Both functions change the `RLIMIT_CPU` of this process, so every test restores
it. Every test also installs a handler first, because `arm_cpu_budget` now
refuses to arm where the breach would kill the process and not reach a handler
(M1). Satisfying that rule is the correct way to test the armer. Bypassing it
would leave this module as the one place that still sets an unhandled timer on
pytest.
"""

from __future__ import annotations

import signal
import sys

import pytest

from ph_runtime.limits import arm_cpu_budget, cpu_seconds_used, relax_cpu_budget

resource = pytest.importorskip("resource", reason="RLIMIT_CPU is POSIX")

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")


@pytest.fixture(autouse=True)
def armable() -> object:
    """A caught `SIGXCPU` and a restored budget, around every test here.

    The handler is what makes `arm_cpu_budget` willing to arm. Its guard reads
    `signal.getsignal`, so a module that arms the test process on purpose must be
    a process that catches the breach. A no-op handler is sufficient, because
    none of these tests uses 3600 CPU-seconds. The handler is removed afterward,
    so the invariant that `test_run_settlement` asserts about this process still
    holds.
    """
    before = resource.getrlimit(resource.RLIMIT_CPU)
    handler = signal.getsignal(signal.SIGXCPU)
    signal.signal(signal.SIGXCPU, lambda *_: None)
    yield
    signal.signal(signal.SIGXCPU, handler)
    resource.setrlimit(resource.RLIMIT_CPU, before)


def test_arming_bounds_the_budget_from_what_is_already_spent() -> None:
    """The per-run budget, made from a counter that only goes up.

    The number is generous, because this arms the *test process*. The test
    asserts the arithmetic: the soft limit is above the CPU already spent by
    about the budget. Nothing is meant to reach the limit.
    """
    arm_cpu_budget(3600)

    soft, _hard = resource.getrlimit(resource.RLIMIT_CPU)
    spent = cpu_seconds_used()
    assert soft >= spent + 3600
    # The limit is also bounded by the budget. The purpose of arming again is
    # that the fortieth cell does not inherit what the first cell spent.
    assert soft <= spent + 3601 + 1


def test_relaxing_puts_the_soft_limit_back_and_stops_re_delivery() -> None:
    """The disarm, which makes one breach produce one signal.

    Without the disarm, the soft limit stays where `arm_cpu_budget` left it and
    the process stays over it. The kernel then signals the process every
    CPU-second, through the teardown that tries to report the breach.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    arm_cpu_budget(3600)
    assert resource.getrlimit(resource.RLIMIT_CPU)[0] != hard, "the budget was not armed"

    relax_cpu_budget()

    assert resource.getrlimit(resource.RLIMIT_CPU) == (hard, hard)
