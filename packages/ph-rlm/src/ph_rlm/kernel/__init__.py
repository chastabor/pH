"""pH's own CPython code runtime: the fd-3 protocol and the host that drives it.

`ph_runtime` (a separate distribution) is the other half, and it lives in a
different venv on purpose.

@module ph_rlm.kernel
"""

from __future__ import annotations

from .journal import OrphanJournal, SweepReport
from .manager import Config, Kernel, KernelLimits, PythonCodeRuntime
from .protocol import PROTOCOL_FD, PROTOCOL_VERSION
from .venv import RuntimeEnvironment, resolve_interpreter

__all__ = [
    "PROTOCOL_FD",
    "PROTOCOL_VERSION",
    "Config",
    "Kernel",
    "KernelLimits",
    "OrphanJournal",
    "PythonCodeRuntime",
    "RuntimeEnvironment",
    "SweepReport",
    "resolve_interpreter",
]
