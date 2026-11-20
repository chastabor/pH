"""The pH stabilization bundle: todo, skill steps, offload, compaction, limits, HITL
and permissions.

The features of Deep Agents are *algorithms plus prompts, not runtime* (§1.3 of
the port plan). Each feature is therefore a row here, on a seam that `ph-core`
already publishes. The package was reserved in Phase 0 (P0-01) and built from
Phase 4.

@module ph_stabilize
"""

from __future__ import annotations

from pathlib import Path

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The rows that the `stabilize` layer adds over `ph-base`."""

__all__ = ["BUNDLE"]
