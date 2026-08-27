"""The guest half of pH's Python code runtime.

Runs inside `$PH_CACHE/runtime-venv`, in a subprocess the host spawns per agent,
and reaches the host over one framed channel on fd 3. It imports neither
`ph-core` nor `ph-rlm`: the process boundary exists so that model code cannot
reach the harness, and importing the harness would put it back inside.

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
