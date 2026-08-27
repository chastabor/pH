"""pH's RLM bundle: Prime Agent's design implemented on pH's seams.

The rule the whole package follows (§6.8): take prime-agent's *semantics* — the
RLM loop, non-blocking admission, the nuclear-family boundary, the Continual
Harness, the doctrine prompts — and implement them on pH's seams. Do not take its
runtime. `ph_rlm.kernel` is pH's own, and `ph_runtime` is its guest half.

@module ph_rlm
"""

from __future__ import annotations

from pathlib import Path

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The rows the `rlm` profile layers over `ph-base`."""

__all__ = ["BUNDLE"]
