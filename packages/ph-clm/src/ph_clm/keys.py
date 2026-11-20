"""Each service that the CLM bundle provides, as a typed key. `ph.keys` gives the reason.

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
"""The context editor that `clm-context` provides. It holds the section map and
the edit verbs. Every front end (the tools and the context file) lands its edits
through it."""

CLM_MIRROR: ServiceKey[MirrorService] = ServiceKey("clm_mirror")
"""The context file that `clm-mirror` keeps: its location, and the functions that
render it and read it back."""
