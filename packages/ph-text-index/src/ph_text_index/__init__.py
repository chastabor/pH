"""`ph-text-index`: semantic retrieval over the agent's own documents.

The package has two tools. `text_index` cuts documents into passages that keep
their line numbers, embeds them, and puts them in a turbovec index on disk.
`text_search` answers a question with the passages themselves, and with the
`path:start-end` that each passage came from. A hit is therefore both an answer
and a pointer that the agent can `read`.

There are three groups of modules, by the process that loads them:

* `row` is what a session mounts. Its rows register the UTCP manual of the
  package with `ph-utcp`, so the tools, turbovec, and the embedding model run in
  the command and not in the session.
* `cli` and `__main__` are the `ph-text-index` command, which mounts `engine`.
* `engine`, `_chunk`, `_embed`, and `_store` are the tools themselves.

This module holds only the path of the bundle, because `python -m ph_text_index`
imports it before every call. Every call pays for whatever this module imports.

@module ph_text_index
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["BUNDLE"]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The profile layer of this distribution, for the `ph.bundles` group.

The bundle is discovered, not imported. That lets `ph-app` compose a profile that
layers this package without depending on it.
"""
