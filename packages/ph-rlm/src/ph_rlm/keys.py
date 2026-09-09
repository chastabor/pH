"""Every service the RLM bundle provides, as a typed key — for `ph.keys`'s reason.

@module ph_rlm.keys
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ph.cordis import ServiceKey

if TYPE_CHECKING:
    from .context_loader import ContextService
    from .harness.service import HarnessService
    from .kernel.manager import PythonCodeRuntime
    from .snapshot import KernelSnapshotPolicy
    from .subagents import RlmChildProvider

__all__ = [
    "CONTEXT_CORPUS",
    "HARNESS",
    "KERNEL_SNAPSHOTS",
    "PYTHON_RUNTIME",
    "RLM_CHILDREN",
]

CONTEXT_CORPUS: ServiceKey[ContextService] = ServiceKey("context_corpus")
HARNESS: ServiceKey[HarnessService] = ServiceKey("harness")
KERNEL_SNAPSHOTS: ServiceKey[KernelSnapshotPolicy] = ServiceKey("kernel_snapshots")
PYTHON_RUNTIME: ServiceKey[PythonCodeRuntime] = ServiceKey("python_runtime")
RLM_CHILDREN: ServiceKey[RlmChildProvider] = ServiceKey("rlm_children")
