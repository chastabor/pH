"""`ph-code-graph`: a Python-native code graph, so that the RLM can ask before it reads.

The package has two tools. `code_index` walks a tree, parses it with tree-sitter,
and records what each file defines and references. `code_graph` answers six
questions over that index. Every answer is a `path:start-end`, so the agent can
pass a hit directly to `read`.

There are three groups of modules, by the process that loads them:

* `row` is what a session mounts. It registers the UTCP manual of the package
  with `ph-utcp`, so the tools run in the command and not in the session.
* `cli` and `__main__` are the `ph-code-graph` command, which mounts `engine`.
* `engine`, `_extract`, and `_store` are the tools themselves.

This module holds only the path of the bundle, because `python -m ph_code_graph`
imports it before every call. Every call pays for whatever this module imports.

@module ph_code_graph
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["BUNDLE"]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The profile layer of this distribution, for the `ph.bundles` group.

The bundle is discovered, not imported. That lets `ph-app` compose a profile that
layers this package without depending on it.
"""
