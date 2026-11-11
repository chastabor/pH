"""The `code-graph` row's config, apart from both rows that read it.

Two rows take this one model. The `code-graph` row a session mounts validates it
and hands it across to the command; the `code-graph-engine` row the command
mounts indexes by it. In a module of its own so the session's row can import the
model without importing the engine — and with it the extractor and the store.

So are the facts both processes need about where the engine keeps things: the
row id the config travels under, the directory name its index takes in an agent's
scratch, and the shared paths the engine writes, which the session's row grants
to the confined command. One derivation each, so a grant cannot name a directory
the engine never writes.

@module ph_code_graph._config
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from ph.paths import default_cache_path, resolve_roots
from ph.seams.fs import IndexScope
from ph.toolcli import state_home
from ph.wire import WireModel

__all__ = [
    "GRAMMAR_CACHE_ENV",
    "ROW",
    "STATE",
    "Config",
    "Scope",
    "grammar_cache",
    "grammar_default",
    "index_base",
    "index_path",
]

ROW = "code-graph"
"""The session row's id, the id the command mounts the engine under — so one
config serves both — and the name of the provisioning command both offer."""

STATE = "code-graph"
"""The directory, in the calling agent's scratch, a session's index is kept in."""

DATABASE = "graph.db"

GRAMMAR_CACHE_ENV = "TREE_SITTER_LANGUAGE_PACK_CACHE_DIR"
"""The pack's own override for where grammars are materialized. An operator who
set it means it for the session and for every command alike."""


class Scope(IndexScope):
    """`IndexScope`, and whether a session starts on it unasked."""

    on_mount: bool = True
    """Start indexing this scope when a root session's agent takes its first
    step, in the background, so the graph is being built before anything asks
    for it. The footer says how far it has got and the agent is told when it
    finishes.

    `false` leaves indexing to `code_index` calls. Either way the command that
    answers the tools stays up between calls: `on_mount` decides only whether it
    starts unasked."""


class Config(WireModel):
    """Row config for `code-graph`."""

    index: Scope = Field(default_factory=Scope)
    """What `code_index` looks at when the call does not say: which trees, which
    files in them, which to leave out.

    `include` defaults to everything, because the language filter is the real
    one: a path whose extension names no supported code language is skipped, so
    `**/*` here means "all the code" rather than "all the files". `paths` is how
    this row is pointed at one package of a monorepo, and `exclude` is how
    vendored or generated code is left out without a glob that has to describe
    the same intent negatively."""
    languages: list[str] = Field(default_factory=list)
    """Restrict indexing to these languages. Empty means every bundled one.

    A monorepo with a `node_modules` of vendored JavaScript and one Python
    package wants `[python]`, and saying so is cheaper than a glob that has to
    describe the same intent negatively."""
    max_bytes: int = 2 * 1024 * 1024
    """The largest file parsed. One past it is skipped and reported, not an
    error: a tree walk should not fail because it found a minified bundle or a
    generated parser, and a caller who never learns what was skipped cannot tell
    a thin graph from a broken one."""
    max_files: int = 20_000
    """How many files one `code_index` call will consider."""
    wait_s: float = 15.0
    """How long `code_index` waits for its run before it answers "indexing" and
    lets the run go on in the background. Where the tools are answered by a host,
    which is in a session; a call that runs by itself always waits to the end.

    Long enough that an ordinary run — a package, a second run over a tree —
    answers with its counts. What it bounds is the run nobody should sit through:
    a first pass over a large tree."""


def index_base() -> Path:
    """Where the per-workspace indexes live, for a caller that names no state
    directory."""
    return default_cache_path(None, "code-graph")


def index_path(root: Path) -> Path:
    """The database `root`'s index lives in (`ph.toolcli.state_home`).

    **Where the caller said**: a session's `ph-utcp` names a directory of the
    calling agent's scratch, so the index is that agent's — writable in a
    read-only session, which the workspace is not, and the one the same agent
    finds after a restart. A person names one with `--state`.

    **Otherwise one per workspace under `$PH_CACHE`**. Under the cache root
    because the index is rebuildable: the source is the truth and this is
    derived (Q1)."""
    return state_home(index_base(), root) / DATABASE


def grammar_default() -> Path:
    """Where the engine points the pack's grammar cache, under the cache root."""
    return resolve_roots().cache / "tree-sitter"


def grammar_cache() -> Path:
    """Where grammars are materialized in fact: the operator's override, or the
    engine's default (`_extract.use_cache` applies the same rule)."""
    override = os.environ.get(GRAMMAR_CACHE_ENV)
    return Path(override) if override else grammar_default()
