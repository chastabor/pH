"""The RLM bundle of pH: the design of prime-agent, implemented on the seams of pH.

The whole package follows one rule (§6.8). Take the *semantics* of prime-agent
and implement them on the seams of pH: the RLM loop, non-blocking admission, the
nuclear-family boundary, the Continual Harness, and the doctrine prompts. Do not
take its runtime. `ph_rlm.kernel` belongs to pH, and `ph_runtime` is its guest
half.

**An import of any part of the bundle declares its intent kinds** (T4). `kinds`
is imported here, so a process that loaded any part of the bundle has
`HARNESS_PROBE` for repair to settle. See `ph_rlm.kinds`.

@module ph_rlm
"""

from __future__ import annotations

from pathlib import Path

from . import kinds as kinds

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The rows that the `rlm` profile layers over `ph-base`."""

__all__ = ["BUNDLE", "kinds"]
