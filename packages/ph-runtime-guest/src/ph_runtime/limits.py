"""Resource limits, applied in the child process before it reports ready (D3).

Two of the three limits are simple. The CPU limit is not, and the reason matters
because it changes what the number *means*.

**`RLIMIT_CPU` is cumulative over the process, and this process is persistent.**
If the limit were set one time to `cpu_seconds`, the whole kernel would have one
budget for its whole life. The fortieth cell in a session would then die on a
limit that the first cell nearly used up. This module instead arms the limit
again at each run, starting from the CPU already consumed. That turns the
cumulative counter into a budget for each cell, which is what the caller means by
`cpu_seconds` and what the gate tests.

A cell that exceeds the budget gets `CpuBudgetExceeded`. That class derives from
`BaseException` on purpose. Like a denial (C3), a budget is not for the program
to catch. If a cell could pass the limit with `except Exception`, the limit would
be only advisory.

**The runner decides who raises the exception, not this module.** `SIGXCPU` can
be turned into an exception safely only when the main thread is executing the
cell. In any other case the exception lands in whatever library frame the
interpreter is in. A guest mostly waits, so that frame is usually inside
`asyncio`. This module therefore arms and disarms the limit, and `Runner`
installs the handler that knows where the cell is.

@module ph_runtime.limits
"""

from __future__ import annotations

import contextlib
import math
import signal
import time
from typing import Any

__all__ = [
    "CPU_BUDGET_MESSAGE",
    "CpuBudgetExceeded",
    "apply_limits",
    "arm_cpu_budget",
    "cpu_seconds_used",
    "relax_cpu_budget",
]


CPU_BUDGET_MESSAGE = "this cell used its CPU budget"
"""The message for both routes out of a budget breach.

The handler raises on one path and cancels the run on the other. The docstring
of `_on_cpu_budget` promises that both end in the same `cpu` error. When the text
was written at each site, that promise depended on two string literals staying
equal."""


class CpuBudgetExceeded(BaseException):
    """The cell used its CPU budget. By policy this is not an `Exception`, so
    `except Exception` does not catch it."""


try:
    import resource
except ImportError:  # pragma: no cover — Windows has no `resource` module
    resource = None  # type: ignore[assignment]


def cpu_seconds_used() -> float:
    """The CPU seconds that this process has consumed: user time plus system time."""
    if resource is None:  # pragma: no cover
        return time.process_time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def apply_limits(*, address_space_bytes: int) -> dict[str, Any]:
    """Apply the limits that last for the life of the process, and report what
    took effect.

    The result is reported and not assumed. A hard limit that is lower than the
    request cannot be raised again, so the host logs the number that is in force
    and not the number that it asked for.
    """
    applied: dict[str, Any] = {"addressSpaceBytes": None, "cpu": "per-run"}
    if resource is None:  # pragma: no cover — Windows uses a Job Object instead
        applied["addressSpaceBytes"] = "job-object"
        return applied
    if address_space_bytes > 0:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = (
            address_space_bytes
            if hard == resource.RLIM_INFINITY
            else min(address_space_bytes, hard)
        )
        try:
            resource.setrlimit(resource.RLIMIT_AS, (target, hard))
            applied["addressSpaceBytes"] = target
        except (ValueError, OSError):
            # The limit is *not* in force, and the report must say so with a number
            # that the host can read. macOS refuses `RLIMIT_AS` outright
            # (`ValueError: current limit exceeds maximum limit`). This branch
            # used to report `soft`, which on macOS is `RLIM_INFINITY`, 2**63-1.
            # That is above the bound for lossless integers in the codec. The
            # host dropped the whole `boot-ack` as unreadable and waited for
            # `boot_timeout` on every kernel start (measured 2026-09-07: 60 s of
            # silence, then "did not report ready"). In the report, `None` already
            # means "no limit". A finite soft limit that was already in force is
            # still the number in force.
            applied["addressSpaceBytes"] = None if soft == resource.RLIM_INFINITY else soft
    return applied


def _uncaught_budget() -> bool:
    """Whether a `SIGXCPU` would kill this process and not reach a handler.

    **Never arm a limit when nothing here will catch its breach** (M1).
    `RLIMIT_CPU` is process-wide, and the default action of `SIGXCPU` is to
    terminate. An arm with no handler installed therefore sets a timer on
    whatever process this is. The tests of this package build `Runner`
    in-process, and there the process is pytest. This happened for some time. The
    per-run arm left the limit at `used + cpuSeconds` on the test runner. A long
    run of the suite reached that limit and died, and it looked like a run that
    finished and printed no summary.

    The function asks `signal` and does not rely on the caller to remember. A
    flag on the one caller that exists today would be a rule that every later
    caller must be told about. Asking `signal` also covers a handler that was
    reset after install, and an `install_signal_handlers` that partly failed.

    `getattr` is used because `SIGXCPU` is POSIX only. Where it does not exist,
    there is no budget to arm and nothing to catch.
    """
    sigxcpu = getattr(signal, "SIGXCPU", None)
    if sigxcpu is None:  # pragma: no cover - Windows
        return True
    return signal.getsignal(sigxcpu) in (signal.SIG_DFL, signal.SIG_IGN)


def arm_cpu_budget(cpu_seconds: int) -> None:
    """Give the *next* run `cpu_seconds` of CPU, counted from the CPU that is
    already spent."""
    if resource is None or cpu_seconds <= 0:  # pragma: no cover
        return
    # Use `ceil`, not `int`. Rounding down the CPU already spent gives the next
    # cell less than `cpu_seconds`. A cell that burned 1.9s rounds down to 1, so a
    # budget of 1 leaves 0.1s, and the *next* trivial cell dies because of the
    # previous cell.
    if _uncaught_budget():
        return
    used = math.ceil(cpu_seconds_used())
    soft = used + cpu_seconds
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY:
        soft = min(soft, hard)
    with contextlib.suppress(ValueError, OSError):  # pragma: no cover
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))


def relax_cpu_budget() -> None:
    """Set the soft limit back to the hard limit. **The handler calls this
    first.**

    `RLIMIT_CPU` does not fire only one time. Linux delivers `SIGXCPU` when the
    soft limit is first crossed, and then **again every CPU-second** while the
    process stays over the limit. The process does stay over it, because the
    limit is cumulative and nothing gives CPU back. One budget breach therefore
    became one signal per second for the rest of the teardown of the run. One
    signal landed in the cell and unwound it as intended. The next signal landed
    in `_snapshot` or `_settle` and escaped the `try` that was meant to report
    the failure, so the `done` frame was never sent. The host has no wall clock
    of its own for a run, so it waited for a frame that would never come.

    Disarming at the first delivery makes the budget fire one time, which the
    rest of the design assumes. The next `arm_cpu_budget` arms it again, and that
    is also how a budget for each run is made.
    """
    if resource is None:  # pragma: no cover
        return
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    with contextlib.suppress(ValueError, OSError):  # pragma: no cover
        resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
