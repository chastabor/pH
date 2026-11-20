"""The config of the `code-graph` row, in a module apart from both rows that read it.

Two rows use this one model. The `code-graph` row, which a session mounts,
validates the config and passes it to the command. The `code-graph-engine` row,
which the command mounts, indexes by it. The model is in its own module so that
the session row can import it without importing the engine, and with the engine
the extractor and the store.

The module also holds the facts that both processes need about where the engine
keeps things: the row id that the config travels under, the directory name of the
index in the scratch directory of an agent, and the shared paths that the engine
writes. The session row grants those shared paths to the confined command. Each
fact has one derivation, so a grant cannot name a directory that the engine never
writes.

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
"""The id of the session row. The command mounts the engine under the same id, so
one config serves both. It is also the name of the provisioning command that both
offer."""

STATE = "code-graph"
"""The directory, in the scratch directory of the calling agent, that holds the
index of a session."""

DATABASE = "graph.db"

GRAMMAR_CACHE_ENV = "TREE_SITTER_LANGUAGE_PACK_CACHE_DIR"
"""The override of the pack itself for where grammars are materialized. An
operator who set it means it for the session and for every command."""


class Scope(IndexScope):
    """`IndexScope`, plus whether a session starts to index it without a request."""

    on_mount: bool = True
    """Start to index this scope in the background when the agent of a root
    session takes its first step. The graph is then in progress before a call
    asks for it. The footer shows the progress, and the agent is told when the
    run finishes.

    `false` leaves indexing to `code_index` calls."""


class Config(WireModel):
    """The row config for `code-graph`."""

    index: Scope = Field(default_factory=Scope)
    """What `code_index` looks at when the call does not say: the trees, the files
    in them, and the files to leave out.

    `include` defaults to everything, because the language filter is the real
    filter. A path whose extension names no supported code language is skipped,
    so `**/*` here means "all the code" and not "all the files". `paths` points
    this row at one package of a monorepo. `exclude` leaves out vendored or
    generated code, without a glob that must describe the same intent
    negatively."""
    languages: list[str] = Field(default_factory=list)
    """Restrict indexing to these languages. An empty list means every language
    for which a grammar can be obtained. `/code-graph install` then fetches
    `engine.DEFAULT_LANGUAGES`.

    A monorepo with one Python package and a `node_modules` of vendored
    JavaScript needs `[python]`. That setting is simpler than a glob that must
    describe the same intent negatively."""
    max_bytes: int = 2 * 1024 * 1024
    """The largest file that is parsed. A larger file is skipped and reported,
    and is not an error. A tree walk must not fail because it found a minified
    bundle or a generated parser. A caller who never learns what was skipped
    cannot tell a thin graph from a broken one."""
    max_files: int = 20_000
    """The number of files that one `code_index` call considers."""
    wait_s: float = 15.0
    """How long a `code_index` call waits for a run before it answers
    "indexing". In a session, it waits for the run that the session started for
    it in the background. In the command, it waits for another run that holds the
    index.

    The value is long enough that an ordinary run (one package, or a second run
    over a tree) answers with its counts. It limits the wait for a run that
    nobody should wait through: a first pass over a large tree."""


def index_base() -> Path:
    """The directory of the per-workspace indexes, for a caller that names no
    state directory."""
    return default_cache_path(None, "code-graph")


def index_path(root: Path) -> Path:
    """The database that holds the index of `root` (`ph.toolcli.state_home`).

    **Where the caller said.** In a session, `ph-utcp` names a directory in the
    scratch directory of the calling agent, so the index belongs to that agent.
    That directory is writable in a read-only session, which the workspace is
    not, and the same agent finds it again after a restart. A person names a
    directory with `--state`.

    **Otherwise, one index for each workspace under `$PH_CACHE`.** It is under
    the cache root because the index can be rebuilt. The source is the truth,
    and the index is derived from it (Q1)."""
    return state_home(index_base(), root) / DATABASE


def grammar_default() -> Path:
    """The directory, under the cache root, that the engine gives the pack as its
    grammar cache."""
    return resolve_roots().cache / "tree-sitter"


def grammar_cache() -> Path:
    """The directory where grammars are really materialized: the override of the
    operator, or the default of the engine (`_extract.use_cache` applies the same
    rule)."""
    override = os.environ.get(GRAMMAR_CACHE_ENV)
    return Path(override) if override else grammar_default()
