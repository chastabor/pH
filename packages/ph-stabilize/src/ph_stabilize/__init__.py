"""pH stabilization bundle: todo, offload, compaction, limits, HITL and permissions.

Deep Agents' features are *algorithms plus prompts, not runtime* (§1.3 of the
port plan), so each one lands here as a row on a seam `ph-core` already
publishes. Reserved in Phase 0 (P0-01); built from Phase 4.

@module ph_stabilize
"""

from __future__ import annotations

from pathlib import Path

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The rows the `stabilize` layer adds over `ph-base`."""

__all__ = ["BUNDLE"]
