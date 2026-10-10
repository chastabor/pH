"""Every service the CLM bundle provides, as a typed key — for `ph.keys`'s reason.

@module ph_clm.keys
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ph.cordis import ServiceKey

if TYPE_CHECKING:
    from .edits import Editor

__all__ = ["CLM"]

CLM: ServiceKey[Editor] = ServiceKey("clm")
"""The context editor `clm-context` provides: the section map, and the three verbs
every front end — the tools, the Phase 2 mirror — lands its edits through."""
