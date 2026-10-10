"""pH's CLM bundle: the model manages its own context.

After *Context Language Models* (Shao et al., 2026): a model keeps the context it
works best with when it may edit that context itself. Here an edit is never a
rewrite of the log. Every one is a surface `replace` ph-core already folds — a
**tombstone** or a **replace** stands in for a run of whole sections, a
**rewrite** changes one result or reply in place — so the model's view changes,
the log keeps every original, and `transcript()` still shows the person what
happened. `context_diff` and `context_recall` read the originals back.

Ideas from the paper's harness and from `pi-clm`; no code from either — the CLM
repository is CC BY-NC, and this package is MIT.

@module ph_clm
"""

from __future__ import annotations

from pathlib import Path

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The rows the `clm` bundle layers over a profile."""

__all__ = ["BUNDLE"]
