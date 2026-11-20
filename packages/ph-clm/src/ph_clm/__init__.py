"""The CLM bundle of pH: the model manages its own context.

This package follows *Context Language Models* (Shao et al., 2026). The idea of
that paper is that a model keeps the context it works best with when it can edit
that context itself. Here, an edit never rewrites the log. Each edit is a surface
`replace`, a core event that ph-core already folds. A **tombstone** or a
**replace** stands in for a run of whole sections. A **rewrite** changes one
result or one reply in place. The view of the model changes, the log keeps every
original, and `transcript()` still shows the user what happened. `context_diff`
and `context_recall` read the originals back. `context_restore` undoes the latest
edit of a section.

The ideas come from the harness of the paper and from `pi-clm`. No code comes
from either. The CLM repository is CC BY-NC, and this package is MIT.

@module ph_clm
"""

from __future__ import annotations

from pathlib import Path

from . import kinds as kinds

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The rows that the `clm` bundle layers over a profile."""

__all__ = ["BUNDLE", "kinds"]
