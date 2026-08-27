"""Resource limits, applied in the child before it reports ready (D3).

Two of the three are straightforward. The CPU limit is not, and the reason is
worth stating because it changes what the number *means*:

**`RLIMIT_CPU` is cumulative over the process, and this process is persistent.**
Setting it once to `cpu_seconds` would give the whole kernel one budget for its
whole life — so the fortieth cell in a session would die on a limit the first
cell nearly spent. Re-arming it at each run, from the CPU already consumed,
turns the cumulative counter into a per-cell budget, which is what the caller
means by `cpu_seconds` and what the gate tests.

Exceeding it raises `CpuBudgetExceeded`, which derives from `BaseException` on
purpose: like a denial (C3), a budget is not the program's to catch. A cell that
could `except Exception` its way past the limit would make the limit advisory.

@module ph_runtime.limits
"""

from __future__ import annotations

import contextlib
import signal
from typing import Any

__all__ = ["CpuBudgetExceeded", "apply_limits", "arm_cpu_budget", "cpu_seconds_used"]


class CpuBudgetExceeded(BaseException):
    """The cell used its CPU budget. Not an `Exception`: not catchable by policy."""


try:
    import resource
except ImportError:  # pragma: no cover — Windows has no `resource` module
    resource = None  # type: ignore[assignment]


def cpu_seconds_used() -> float:
    """CPU seconds this process has consumed, user + system."""
    if resource is None:  # pragma: no cover
        import time

        return time.process_time()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def apply_limits(*, address_space_bytes: int) -> dict[str, Any]:
    """Apply the process-lifetime limits and report what took effect.

    Reported rather than assumed: a hard limit lower than the request cannot be
    raised back, so the host logs the number in force instead of the number it
    asked for.
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
        except (ValueError, OSError):  # pragma: no cover
            applied["addressSpaceBytes"] = soft
    return applied


def _on_sigxcpu(_signum: int, _frame: Any) -> None:
    raise CpuBudgetExceeded("this cell used its CPU budget")


def arm_cpu_budget(cpu_seconds: int) -> None:
    """Give the *next* run `cpu_seconds` of CPU, from whatever is spent so far."""
    if resource is None or cpu_seconds <= 0:  # pragma: no cover
        return
    signal.signal(signal.SIGXCPU, _on_sigxcpu)
    used = int(cpu_seconds_used())
    soft = used + cpu_seconds
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if hard != resource.RLIM_INFINITY:
        soft = min(soft, hard)
    with contextlib.suppress(ValueError, OSError):  # pragma: no cover
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
