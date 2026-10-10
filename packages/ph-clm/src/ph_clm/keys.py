"""Every service the CLM bundle provides, as a typed key — for `ph.keys`'s reason.

@module ph_clm.keys
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ph.cordis import ServiceKey

if TYPE_CHECKING:
    from .edits import Editor
    from .mirror import MirrorService

__all__ = ["CLM", "CLM_MIRROR"]

CLM: ServiceKey[Editor] = ServiceKey("clm")
"""The context editor `clm-context` provides: the section map, and the three verbs
every front end — the tools, the context file — lands its edits through."""

CLM_MIRROR: ServiceKey[MirrorService] = ServiceKey("clm_mirror")
"""The context file `clm-mirror` keeps: where it lives, and its render and read-back."""
