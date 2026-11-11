"""`ph-text-index` — semantic retrieval over the agent's own documents.

Two tools. `text_index` cuts documents into passages that remember their line
numbers, embeds them and puts them in a turbovec index on disk. `text_search`
answers a question with the passages themselves plus the `path:start-end` each
came from — so a hit is both an answer and a pointer the agent can `read`.

Three groups of modules, by which process loads them:

* `row` — what a session mounts. Its rows register the package's UTCP manual
  with `ph-utcp`, so the tools, turbovec and the embedding model run in the
  command rather than in the session.
* `cli` and `__main__` — the `ph-text-index` command, which mounts `engine`.
* `engine`, `_chunk`, `_embed`, `_store` — the tools themselves.

This module is kept to the bundle's path because `python -m ph_text_index`
imports it before every call: whatever it imports, every call pays for.

@module ph_text_index
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["BUNDLE"]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""This distribution's profile layer, for the `ph.bundles` group.

Discovered rather than imported, which is what lets `ph-app` compose a
profile that layers this package without depending on it.
"""
