"""`ph-code-graph` — a Python-native code graph, so the RLM can ask before it reads.

Two tools. `code_index` walks a tree, parses it with tree-sitter and records what
each file defines and references. `code_graph` answers six questions over that
index — and answers every one of them with `path:start-end`, so a hit is a
pointer the agent can hand straight to `read`.

Three modules, by which process loads them:

* `row` — what a session mounts. It registers the package's UTCP manual with
  `ph-utcp`, so the tools run in the command rather than in the session.
* `cli` and `__main__` — the `ph-code-graph` command, which mounts `engine`.
* `engine`, `_extract`, `_store` — the tools themselves.

This module is kept to the bundle's path because `python -m ph_code_graph`
imports it before every call: whatever it imports, every call pays for.

@module ph_code_graph
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["BUNDLE"]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""This distribution's profile layer, for the `ph.bundles` group.

Discovered rather than imported, which is what lets `ph-app` compose a
profile that layers this package without depending on it.
"""
