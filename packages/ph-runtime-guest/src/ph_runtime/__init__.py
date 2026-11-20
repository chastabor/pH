"""The guest half of the Python code runtime of pH.

The guest runs inside `$PH_CACHE/runtime-venv`, in a subprocess that the host
starts for each agent. It reaches the host over one framed channel on fd 3. It
imports neither `ph-core` nor `ph-rlm`. The process boundary exists so that model
code cannot reach the harness, and an import of the harness would put the harness
back inside the boundary.

@module ph_runtime
"""

from __future__ import annotations

from .errors import RunStopped, ToolFailed
from .protocol import PROTOCOL_FD, PROTOCOL_VERSION
from .skill import wrap_skill_module

__all__ = [
    "PROTOCOL_FD",
    "PROTOCOL_VERSION",
    "RunStopped",
    "ToolFailed",
    "wrap_skill_module",
]
