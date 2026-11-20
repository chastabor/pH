"""`code-graph`: a Python-native code graph, so that the RLM can ask before it reads.

The package has two tools. `code_index` walks a tree through `ctx.fs`, parses it
with tree-sitter, and records what each file defines and references.
`code_graph` answers six questions over that index. Every answer is a
`path:start-end`, so the agent can pass a hit directly to `read`.

That is the purpose of the package. An agent in an unfamiliar repository uses its
first several thousand tokens to read files, only to learn which two files
mattered. This package turns that into one call.

## Python-native, and what that decided

The alternative was a wrapper around the Rust/Node CodeGraph, and the
investigation rejected it. The Rust kernel of that project is a *tree-sitter
extractor behind a Node-API boundary*. Its whole export surface is
`extract_file`, `contract_info`, `grammar_info`, and two `cfnptr` helpers. The
intelligence is TypeScript: 29 708 lines of cross-file resolution, plus graph,
search, and context layers. maturin also cannot build a `#[napi]` crate. A
wrapper would have given a parser that Python already has, and left the
intelligence behind.

The package therefore uses `tree-sitter-language-pack` for extraction: a 3.7 MB
wheel, with 371 languages that are fetched one time into a cache. It uses
`sqlite3` from the standard library for the graph, because two of the six
queries here are exactly the two that `pyturso` cannot serve (`_store` has the
measurements).

## Name-based, and it says so

A reference records the *name* that it used, and `callers` and `callees` join on
that name. Two `register` methods in two classes are one name to this index.
Every result that could be ambiguous carries `definitions`, the number of places
that define the name. The model can then see the ambiguity, and is not given one
of the definitions without notice. `code_graph mode=define` lets the model tell
the definitions apart.

A resolved graph needs an import graph, scope, and type inference for each
language. Those are the 29 708 lines that this package did not port. A
name-based graph answers most of what an agent asks, and reports where it
cannot.

## One run at a time

`code_index` is one command, and it finishes before it answers. A run holds the
lock of the index while it works. A second run waits up to `wait_s` for its
turn. After that it answers `state: indexing`, which means that another run is
building the index, and it does not continue to wait. `code_graph` never waits.
It answers from what is indexed so far, and says that the index is still being
built. With `--progress`, the command reports its progress on stderr
(`ph.toolcli.reported`). A session uses that to show a run that it started in
the background. Each file is its own transaction. A run that is stopped
therefore keeps every file that it finished, and the next run finds those files
by their digests.

## Where this row is mounted

This row is mounted in the process of the `ph-code-graph` command, not in the
process of a session. A session mounts the `code-graph` row
(`ph_code_graph.row`), which registers the UTCP manual of this package. The
commands of the manual run `ph-code-graph`. That command mounts *this* row and
runs a tool through the registry (`ph.toolcli`). The `ctx.fs` below therefore
belongs to the command. It is rooted at the working directory that the sandbox
of the calling agent gave the command, and `run.agent` is always `None`.

@module ph_code_graph.engine
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.characters import head
from ph.cordis import Context, MountRefusal, ServiceKey, plugin
from ph.json import JsonObject, as_int
from ph.keys import COMMANDS, FS, TOOLS
from ph.llm.types import ContentBlock, text_of
from ph.locks import file_lock_held
from ph.seams._registry import contribute_item
from ph.seams.changes import tree_state
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.fs import FsService, skipped_lines
from ph.text import BRIEFLY, count_of
from ph.toolcli import RunProgress, claim_home, reported, run_lock
from ph.tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ph.tools.errors import HarnessError
from ph.tools.presentation import simple_views

from ._config import (
    GRAMMAR_CACHE_ENV,
    ROW,
    Config,
    grammar_cache,
    grammar_default,
    index_base,
    index_path,
)
from ._extract import (
    cache_release,
    detect_language,
    ensure,
    extract,
    indexable,
    obtained,
    readiness,
)
from ._store import CodeGraphStore, Hit, SymbolRow, digest_of

__all__ = ["CODE_GRAPH", "CodeGraphSeam", "apply"]

log = logging.getLogger("ph_code_graph")

CODE_GRAPH_FAILED = "CODE_GRAPH_FAILED"

MISSING = (
    "the code-graph row needs `tree-sitter` and `tree-sitter-language-pack`, which "
    "ph-code-graph depends on; install them (`uv sync`) or remove the row"
)

INDEX_DESCRIPTION = """Index a code tree so `code_graph` can answer questions about it.

Parses each file and records what it defines and what it references. Run it once
on a tree, then again after edits — it re-parses only files whose contents
changed, so re-running is cheap and keeps answers current.

Point it at a package rather than a whole monorepo. Non-code files are skipped."""

GRAPH_DESCRIPTION = """Ask about a codebase's structure instead of reading it.

Every answer carries `path:start-end`, so use this to find what matters and then
`read` exactly that.

- `search`  — find symbols by name or docstring wording (fuzzy, ranked)
- `define`  — where a name is defined, exactly
- `callers` — what calls this, with the calling line
- `callees` — what this calls
- `impact`  — everything transitively affected by changing this, ring by ring
- `entities`— the largest definitions in a path, biggest first

Matching is by name, so a name defined in several places reports `definitions`
greater than 1 — use `define` to see which is which. Needs `code_index` to have
run first."""


# --------------------------------------------------------------------- seam ----


_LOCKED = "the code index"


def _lock_of(store: CodeGraphStore) -> Path:
    """The lock file beside a database. It orders the writers of the database."""
    return store.path.with_name(f"{store.path.name}.lock")


CODE_GRAPH: ServiceKey[CodeGraphSeam] = ServiceKey("code_graph")
"""The code graph, for `/code-graph` and for the tools that it backs."""


@dataclass(slots=True)
class CodeGraphSeam:
    """The service published as `ctx.code_graph`.

    It holds the location of the index and orders its writers. The ordering is
    why `code_index` declares itself not concurrency-safe. Two calls that index
    overlapping trees would interleave their per-file transactions. SQLite would
    keep each transaction atomic, but the *pair* would report counts that neither
    call produced.
    """

    ctx: Context
    grammars: Path
    """The directory of the tree-sitter grammar cache. See `_extract.use_cache`."""

    def store_for(self, root: Path) -> CodeGraphStore:
        """The index of this workspace: where the caller said, or one directory
        for each root (`_config.index_path`).

        Two trees never share an index. A worktree at another revision has
        different line numbers, so one index for both would give pointers that
        are stale without any sign. In a session, the caller names a directory in
        the scratch directory of the agent, which is removed with that scratch
        directory. The per-root directory under `$PH_CACHE` records its
        workspace. When that workspace is gone, a later index removes the
        directory (`ph.workspace_caches`).
        """
        return CodeGraphStore(path=index_path(root))

    def indexing(self, store: CodeGraphStore) -> bool:
        """Whether a run is building `store` now. A run holds the lock of the
        index for as long as it works, and only a run takes that lock."""
        return file_lock_held(_lock_of(store))

    @asynccontextmanager
    async def writing(self, store: CodeGraphStore, *, wait_s: float) -> AsyncIterator[bool]:
        """Hold `store` for one indexing run, against every other writer of it.
        Yields whether the lock is held. The value is `False` when another run
        still held the lock after `wait_s` (`ph.toolcli.run_lock`).

        **The lock is a file beside the database, not a lock in this process.**
        Everything on one workspace shares the index: two roots in a daemon, a
        `phern -p`, and a `ph-code-graph` command. An `anyio.Lock` here ordered
        only the calls of one seam. Two processes each stayed atomic for each
        file, but they still raced on the stored version-control token. The
        second process could record a token that vouched for files that the
        first was still writing. `flock` refuses a second holder even inside one
        process, so this one lock also orders the calls of this seam.

        A second run waits and is not refused. It is expected to queue behind
        the first run, and it then finds most of the tree already current.
        """
        async with run_lock(_lock_of(store), what=_LOCKED, wait_s=wait_s) as held:
            yield held

    def report(self) -> list[tuple[str, str]]:
        """The section for `phern doctor`."""
        # Not a declared dependency: the row checks for it before this runs.
        import tree_sitter  # noqa: PLC0415

        # There is no `hasattr` guard. The row declares `inject=[FS]`, so `ctx.fs`
        # is present for as long as the row is active. A `Path.cwd()` fallback
        # gave an index that was keyed to the process directory and not to the
        # workspace, which is worse than the traceback that it avoided.
        root = self.ctx.require(FS).root_for(None)
        store = self.store_for(root)
        rows = [
            ("tree-sitter", getattr(tree_sitter, "__version__", "installed")),
            ("grammars", str(self.grammars)),
            ("index", str(store.path)),
        ]
        if not store.exists():
            rows.append(("state", "not built — run code_index"))
            return rows
        stats = store.stats()
        rows.extend(
            [
                ("files", str(stats["files"])),
                ("symbols", str(stats["symbols"])),
                ("references", str(stats["refs"])),
                ("languages", ", ".join(stats["languages"][:8]) or "none"),
            ]
        )
        return rows


# ------------------------------------------------------------------ schemas ----


class IndexArgs(ToolModel):
    paths: list[str] | None = Field(
        None,
        description=(
            "Directories or files to index, relative to the workspace root. "
            "Defaults to the row's setting, which is the whole workspace unless a "
            "profile pointed it elsewhere."
        ),
    )
    glob: str | None = Field(
        None, description="Which files to take from a directory. Defaults to the row's setting."
    )
    exclude: list[str] | None = Field(
        None,
        description=(
            "Glob patterns for files to leave out, read against the workspace root "
            "(`**/tests/**`). Added to the row's own exclusions."
        ),
    )
    forget: bool = Field(False, description="Remove these paths from the index instead.")


class SkippedValue(ToolModel):
    path: str
    reason: str


class IndexValue(ToolModel):
    state: Literal["done", "indexing"] = "done"
    """`indexing`: another run holds the index, and it was still working after
    this call had waited `wait_s`. This call did nothing below. The totals are
    what the graph holds so far."""
    indexed: int = 0
    """The files that this call parsed: changed or new files only."""
    unchanged: int = 0
    """The files that were already current, by content digest."""
    removed: int = 0
    symbols: int = 0
    """The symbols that this call wrote."""
    skipped: list[SkippedValue] = Field(default_factory=list)
    total_files: int
    total_symbols: int
    total_refs: int
    languages: list[str]


class GraphArgs(ToolModel):
    mode: Literal["search", "define", "callers", "callees", "impact", "entities"] = Field(
        "search", description="Which question to ask. See the tool description."
    )
    query: str | None = Field(
        None,
        description=(
            "The symbol name for define/callers/callees/impact, or the search "
            "wording for search. Not used by entities."
        ),
    )
    path: str | None = Field(None, description="For entities: restrict to this file or directory.")
    kind: str = Field(
        "any", description="For entities: keep only this kind (function, class, method, ...)."
    )
    distance: int = Field(2, ge=1, le=6, description="For impact: how many hops to follow.")
    limit: int = Field(30, ge=1, le=300, description="Return at most this many results.")
    offset: int = Field(0, ge=0, description="For entities: skip this many (paging).")


class SymbolValue(ToolModel):
    name: str
    kind: str
    path: str
    start_line: int
    end_line: int
    lines: int
    doc: str | None = None


class EdgeValue(SymbolValue):
    ref_line: int
    """The line where the reference is written. This is the line to open."""
    ref_path: str
    """The file that holds `ref_line`. **It is not always `path`.** For `callees`,
    the symbol is the definition of the callee, and the reference is a line in
    the caller. See `_store.Hit`."""
    via: str
    """The name that the reference used."""


class RingValue(ToolModel):
    distance: int
    symbols: list[SymbolValue]


class GraphValue(ToolModel):
    mode: str
    query: str | None
    definitions: int
    """The number of places that define `query`. **A value above 1 means that the
    answer is ambiguous.** This index matches by name. See the module docstring."""
    total: int
    offset: int
    truncated: bool
    indexed_files: int
    indexing: bool = False
    """A run is still building the index. The answer comes from what is indexed
    so far."""
    symbols: list[SymbolValue] = Field(default_factory=list)
    edges: list[EdgeValue] = Field(default_factory=list)
    rings: list[RingValue] = Field(default_factory=list)


# ------------------------------------------------------------------- render ----


def _where(one: dict[str, Any]) -> str:
    return f"{one['path']}:{one['start_line']}-{one['end_line']}"


def _doc(one: dict[str, Any]) -> str:
    doc = (one.get("doc") or "").strip().splitlines()
    return f"  — {head(doc[0], 70)}" if doc else ""


def _render_index(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("state") == "indexing":
        return text_content(
            "An indexing run is already building this graph, so this call started "
            "none. `code_graph` answers now, from what is indexed already, and says "
            "while the index is still being built. Call `code_index` again once that "
            "run has finished.\n"
            f"The graph holds {count_of(value['total_symbols'], 'symbol')} "
            f"from {count_of(value['total_files'], 'file')} so far."
        )
    if args.get("forget"):
        return text_content(
            f"Removed {count_of(value['removed'], 'file')} from the index. "
            f"It now holds {count_of(value['total_symbols'], 'symbol')} "
            f"from {count_of(value['total_files'], 'file')}."
        )
    return text_content(_index_summary(value))


SHOWN = 20
"""The number of skipped files that a result names before it only counts the rest."""


def _index_summary(value: Mapping[str, Any], *, shown: int = SHOWN) -> str:
    """What a finished run did: its counts, and each file that it skipped with the
    reason, up to `shown` files. The note about a run that nobody waited for says
    the same. A grammar that could not be fetched leaves an empty graph, and the
    word "finished" alone would not show that."""
    lines = [
        f"Indexed {count_of(value['indexed'], 'file')} "
        f"({value['unchanged']} already current), {count_of(value['symbols'], 'symbol')}."
    ]
    lines += skipped_lines(value["skipped"], shown)
    lines.append(
        f"The graph holds {count_of(value['total_symbols'], 'symbol')} and "
        f"{count_of(value['total_refs'], 'reference')} across "
        f"{count_of(value['total_files'], 'file')} "
        f"({', '.join(value['languages'][:6]) or 'no languages'})."
    )
    return "\n".join(lines)


def _said(value: Mapping[str, Any]) -> str:
    """The note of a run, for a caller that did not wait for it
    (`ph.toolcli.reported`)."""
    return _index_summary(value, shown=BRIEFLY)


def _changed(value: Mapping[str, Any]) -> bool:
    """Whether a run changed the graph: it parsed a file or removed one."""
    return as_int(value.get("indexed")) + as_int(value.get("removed")) > 0


def _busy(value: Mapping[str, Any]) -> bool:
    """Whether another run held the graph, so this one did nothing."""
    return value.get("state") == "indexing"


def _ambiguity(value: Any) -> str:  # noqa: ANN401
    if value["definitions"] <= 1:
        return ""
    return (
        f"\n[{value['query']!r} is defined in {value['definitions']} places and this "
        "index matches by name, so these edges may belong to more than one of them; "
        "`mode=define` lists them]"
    )


def _render_graph(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    """Render the answer. While the index is still being built, add the warning
    below the answer."""
    answer = _render_answer(args, value)
    indexing = value.get("indexing")
    if not indexing:
        return answer
    return text_content(
        f"{text_of(answer)}\n"
        "[Not finished indexing. This answer covers what is indexed so far; ask "
        "again once it finishes.]"
    )


def _render_answer(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    mode = value["mode"]
    if mode in ("search", "define"):
        if not value["symbols"]:
            return text_content(
                f"Nothing matched {value['query']!r} in "
                f"{count_of(value['indexed_files'], 'indexed file')}. "
                "Run `code_index` first if this tree was never indexed."
            )
        lines = [f"{count_of(value['total'], 'match', 'matches')} for {value['query']!r}:"]
        for one in value["symbols"]:
            lines.append(f"  {one['kind']:<10} {one['name']:<24} {_where(one)}{_doc(one)}")
    elif mode in ("callers", "callees"):
        if not value["edges"]:
            verb = "calls" if mode == "callers" else "is called by"
            return text_content(
                f"Nothing {verb} {value['query']!r} in the index."
                + (
                    ""
                    if value["definitions"]
                    else f" No symbol named {value['query']!r} is indexed at all."
                )
            )
        header = (
            f"calling {value['query']!r}"
            if mode == "callers"
            else f"where {value['query']!r} calls out"
        )
        lines = [f"{count_of(value['total'], 'site')} — {header}:"]
        for one in value["edges"]:
            # Use `ref_path`, not `path`. For `callees`, the reference and the
            # definition are in different files. A line that is printed against
            # the wrong file points to nothing (`_store.Hit`).
            at = f"{one['ref_path']}:{one['ref_line']}"
            lines.append(
                f"  {one['name']:<24} {at:<44} (defined {_where(one)})"
                + (f" via {one['via']}" if one["via"] != value["query"] else "")
            )
    elif mode == "impact":
        if not value["rings"]:
            return text_content(
                f"Nothing depends on {value['query']!r} within "
                f"{count_of(as_int(args.get('distance'), 2), 'hop')}."
            )
        lines = [f"Changing {value['query']!r} reaches:"]
        for ring in value["rings"]:
            lines.append(
                f"\n  {count_of(ring['distance'], 'hop')} — "
                f"{count_of(len(ring['symbols']), 'symbol')}:"
            )
            for one in ring["symbols"]:
                lines.append(f"    {one['name']:<24} {_where(one)}")
    else:
        if not value["symbols"]:
            return text_content("No definitions matched.")
        lines = [f"{count_of(value['total'], 'definition')}, biggest first:"]
        lines.append("\nlines  kind       name                     where")
        for one in value["symbols"]:
            lines.append(f"{one['lines']:>5}  {one['kind']:<10} {one['name']:<24} {_where(one)}")
    if value["truncated"]:
        shown = value["offset"] + max(len(value["symbols"]), len(value["edges"]))
        lines.append(f"\n[{value['total'] - shown} more; re-run with offset={shown}]")
    text = "\n".join(lines) + _ambiguity(value)
    return text_content(text)


# --------------------------------------------------------------------- body ----


DEFAULT_LANGUAGES = (
    "python",
    "typescript",
    "tsx",
    "javascript",
    "rust",
    "go",
    "java",
    "csharp",
    "c",
    "cpp",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "dart",
    "lua",
    "r",
    "sql",
    "bash",
)
"""The languages that `/code-graph install|status` covers when the row names none.

This is a convenience list, and it is **filtered through `indexable` before
use**. `bash` and `sql` are in the list and have no tags query. An unfiltered
list reported them as ready, and then gave the extractor a language that it
could not use. The indexer does not read this list. It asks `indexable` and
`obtained` for each file, so a language outside this list still works when the
pack has it.
"""


@plugin("code-graph-engine", affects="environment", inject=[TOOLS, FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam and register both tools."""
    for module in ("tree_sitter", "tree_sitter_language_pack"):
        if importlib.util.find_spec(module) is None:
            raise MountRefusal(MISSING)

    # Do this before anything parses. The pack keeps every grammar that it uses
    # in a writable cache directory, and fails hard without one. The row
    # therefore names a path that it can vouch for, and does not inherit whatever
    # `HOME` is. See `use_cache`.
    try:
        # This goes through `ctx.effect` and runs in a worker thread, for two
        # separate reasons. The effect makes the process-global cache setting of
        # the library unwind with this row, so that it does not outlive the row
        # (§4.9, I2; see `cache_release`). The thread is needed because
        # `use_cache` imports `tree_sitter_language_pack`. That is a 20 ms
        # dlopen, and a profile that mounts many rows would otherwise run those
        # one after the other on the loop.
        await ctx.effect(
            lambda: anyio.to_thread.run_sync(cache_release, grammar_default()),
            label="tree-sitter-cache",
        )
    except OSError as error:
        # Raise `MountRefusal`, not the `OSError`. Every command that mounts a
        # profile turns this one type into a sentence and an exit code, and
        # leaves anything else as the traceback that a bug deserves. A read-only
        # `$PH_CACHE` is a fact of the deployment that an operator can fix. It is
        # not a bug. It was measured inside a sandbox that made `~/.cache`
        # read-only, where it arrived as fourteen frames of `pathlib`.
        raise MountRefusal(
            f"code-graph cannot write the tree-sitter grammar cache at {error.filename}: "
            f"{error.strerror}. The grammars are materialized there on first use, so this "
            "path must be writable — point $PH_CACHE somewhere it is, or set "
            f"{GRAMMAR_CACHE_ENV}."
        ) from error

    seam = CodeGraphSeam(ctx=ctx, grammars=grammar_cache())
    ctx.provide(CODE_GRAPH, seam)

    def store(run: ToolRunContext) -> CodeGraphStore:
        return seam.store_for(ctx.require(FS).root_for(run.agent))

    def _forget_missing(
        fs: FsService,
        book: CodeGraphStore,
        known: Mapping[str, tuple[str, str]],
        run: ToolRunContext,
    ) -> int:
        """Remove every indexed path that is no longer on disk. This takes one
        thread hop."""
        gone = [path for path in known if not fs.resolve(path, agent=run.agent).exists()]
        return book.forget(gone) if gone else 0

    async def index_tool(args: IndexArgs, run: ToolRunContext) -> dict[str, Any]:
        return await reported(
            partial(index_run, args, run),
            what="files",
            said=_said,
            changed=_changed,
            busy=_busy,
        )

    async def index_run(
        args: IndexArgs, run: ToolRunContext, progress: RunProgress
    ) -> dict[str, Any]:
        """One indexing run, to its end, or the answer that another run holds the
        index. The cancellation of the caller is checked at each file."""
        fs = ctx.require(FS)
        book = store(run)
        skipped: list[dict[str, str]] = []
        indexed = unchanged = symbols = removed = 0

        async with seam.writing(book, wait_s=config.wait_s) as held:
            if not held:
                stats: dict[str, Any] = await anyio.to_thread.run_sync(book.stats)
                return IndexValue(
                    state="indexing",
                    total_files=stats["files"],
                    total_symbols=stats["symbols"],
                    total_refs=stats["refs"],
                    languages=stats["languages"],
                ).model_dump()
            # This is under the lock, so a call that will answer "indexing" walks
            # nothing.
            await anyio.to_thread.run_sync(claim_home, index_base(), fs.root_for(run.agent))
            wanted = config.index.narrowed(paths=args.paths, glob=args.glob, exclude=args.exclude)
            paths = await fs.collect(
                wanted.paths,
                wanted.include,
                scope=run.scope,
                limit=config.max_files,
                agent=run.agent,
                exclude=wanted.exclude,
            )
            total = len(paths)
            progress.begin(total)
            await anyio.to_thread.run_sync(book.prepare)
            if args.forget:
                # This falls through to the one `stats` read and the one return
                # below. `indexed`, `unchanged`, and `symbols` are already 0, and
                # `skipped` is already empty. A second copy of the payload was
                # eleven lines that repeated that. It was also a second shape for
                # one `IndexValue` schema, in the mode that no test asserts.
                removed = await anyio.to_thread.run_sync(book.forget, paths)
                paths = []
            known = await anyio.to_thread.run_sync(book.known)
            # **Ask the version control what changed before reading anything.**
            # `tree_state` never raises, and it returns an empty state for a tree
            # with no backend. The loop below is correct in both cases. It reads
            # every file when nothing can vouch for one. The workspace provider
            # decides which backend answers (`ph.seams.changes`).
            parsers: dict[str, bool] = {}
            stored_token = await anyio.to_thread.run_sync(book.token)
            state = await tree_state(ctx, fs.root_for(run.agent), since=stored_token)
            for count, path in enumerate(paths):
                run.raise_if_canceled()
                progress.tick(count)
                language = detect_language(path)
                # This is derived from the pack and not checked against a list of
                # names. `.txt`, `.ini`, `.proto`, and a dozen others are
                # languages that the detector recognizes and that the tags
                # queries do not cover. The default `include` of the row reaches
                # all of them.
                if language is None or not indexable(language):
                    continue
                if config.languages and language not in config.languages:
                    continue
                if language not in parsers:
                    # **One thread hop for each language, not for each file.** The
                    # first file of a language is worth a worker thread:
                    # `obtained` loads a grammar, and fetches one that is not on
                    # disk, which takes seconds the first time. For every later
                    # file of that language the answer is a cached boolean, so
                    # the 60.6 µs hop was the entire cost. Over a tree of 20 000
                    # files that was 1.2 s, which is the whole saving that the
                    # change filter exists to give.
                    #
                    # The answers are kept for one call, not cached for the
                    # process with `@cache`. A person who runs `/code-graph
                    # install` between two calls must see the grammar that
                    # appeared.
                    parsers[language] = await anyio.to_thread.run_sync(obtained, language)
                if not parsers[language]:
                    skipped.append(
                        {
                            "path": path,
                            "reason": (
                                f"no {language} parser available — grammars come from "
                                "GitHub the first time; /code-graph install"
                            ),
                        }
                    )
                    continue
                stored_digest, stored_vcs = known.get(path, ("", ""))
                if stored_digest and state.vouches_for(path, stored_vcs):
                    # git or jj proved that the file is unchanged, so the file is
                    # never opened. The saving is the whole read: 5.2 ms of I/O
                    # plus 3.0 ms of sha256 for 136 files, which is about 1.2 s
                    # on a tree of 20 000 files.
                    unchanged += 1
                    continue
                refused = fs.skip_reason(path, max_bytes=config.max_bytes, agent=run.agent)
                if refused:
                    skipped.append({"path": path, "reason": refused})
                    continue
                slice_ = await fs.read(
                    path,
                    limit=None,
                    # `skip_reason` above is the limit for this walk. It skips and
                    # does not raise, on purpose: an indexing pass must not fail
                    # because it found a minified bundle. The default of `read`
                    # would be a second, larger limit that raises.
                    max_bytes=None,
                    scope=run.scope,
                    agent=run.agent,
                    session=run.session,
                )
                # **There is deliberately no prose test here** (X6 review). The
                # text index refuses documents that hold no prose. Source code is
                # not prose. The same predicate refused a module whose only
                # statement is a long `__all__`, and a `TABLE = {...}` literal
                # beside `def lookup()`. `lookup` would then have been lost from
                # this index, with only a skip line to show it. The shape of a
                # file does not reliably separate a minified bundle from a
                # legitimate data module, and a false positive here loses
                # symbols silently. The size limit of `skip_reason` stays the
                # guard of this walk. The content hash stays the authority on
                # *whether* a file changed. The filter above only decides whether
                # to open the file, so the "content, not clock" guarantee holds.
                digest = digest_of(slice_.text)
                if stored_digest == digest:
                    unchanged += 1
                    continue
                try:
                    extraction = await anyio.to_thread.run_sync(
                        extract, path, slice_.text, language
                    )
                except Exception as error:
                    # The failure is reported and the file is skipped. It is not
                    # raised. A tree walk that stopped on one file that does not
                    # parse was a real failure of the package that this one
                    # replaced.
                    log.debug("ph_code_graph: %s did not parse", path, exc_info=True)
                    skipped.append({"path": path, "reason": f"did not parse ({error})"})
                    continue
                symbols += await anyio.to_thread.run_sync(
                    book.put, path, digest, extraction, state.id_for(path)
                )
                indexed += 1
            if not args.forget:
                # **A file that is gone from disk is removed from the index**
                # (X3). A deleted file used to keep its symbols until someone ran
                # `forget` by hand. `code_graph` then continued to name
                # definitions at paths that did not exist. A rename showed the
                # symbol twice, one time under each name, and nothing said which
                # one was real. A pointer into a file that does not exist is the
                # one answer that this tool must not give, because the next step
                # of the model is to open that file.
                #
                # **The question goes to the filesystem, not to the walk.**
                # "Absent from `paths`" looks like the same question, and it is
                # not. `collect` stops at `max_files` and reports nothing, and
                # the policy screen prunes whole directories. On a tree past the
                # cap, or with a `permissions-fs` rule that hides a subtree, the
                # walk is a *subset* of the scope. A difference against the walk
                # would remove everything that the walk did not reach. The
                # filesystem check also handles the narrowed walk at no cost.
                # There the question is not "did the walk cover the scope" but
                # "is this file still there".
                removed = await anyio.to_thread.run_sync(_forget_missing, fs, book, known, run)
            # Store the token last, after the loop. If the token were recorded
            # before the writes, a crash between the two would leave a token that
            # vouches for files that this run never indexed.
            if state.token:
                await anyio.to_thread.run_sync(book.remember, state.token)
            stats = await anyio.to_thread.run_sync(book.stats)

        return {
            "state": "done",
            "indexed": indexed,
            "unchanged": unchanged,
            "removed": removed,
            "symbols": symbols,
            "skipped": skipped,
            "total_files": stats["files"],
            "total_symbols": stats["symbols"],
            "total_refs": stats["refs"],
            "languages": stats["languages"],
        }

    async def graph_tool(args: GraphArgs, run: ToolRunContext) -> dict[str, Any]:
        book = store(run)
        if not await anyio.to_thread.run_sync(book.exists):
            raise HarnessError(
                "no code index exists for this workspace yet; run `code_index` first",
                CODE_GRAPH_FAILED,
            )
        if args.mode != "entities" and not args.query:
            raise HarnessError(f"mode={args.mode} needs `query`", CODE_GRAPH_FAILED)
        indexing = await anyio.to_thread.run_sync(seam.indexing, book)

        # Use `file_count`, not `stats()`. The header needs one number. `stats`
        # counts symbols and refs and groups files by language: three full scans
        # for each query, which grow with the corpus (1.97 ms against 0.22 ms on
        # an index of 2 000 files).
        indexed = await anyio.to_thread.run_sync(book.file_count)
        name = args.query or ""
        # Use `COUNT(*)`, not `len(define(name, 100))`. That call materialized a
        # hundred rows, and for any name past the cap it reported the *cap* as
        # the count.
        defined = (
            await anyio.to_thread.run_sync(book.definition_count, name)
            if args.mode != "search" and name
            else 0
        )
        body: dict[str, Any] = {
            "symbols": [],
            "edges": [],
            "rings": [],
            "total": 0,
            "truncated": False,
        }

        if args.mode == "search":
            found = await anyio.to_thread.run_sync(book.search, _fts(name), args.limit)
            body |= {"symbols": _symbols(found), "total": len(found)}
        elif args.mode == "define":
            found = await anyio.to_thread.run_sync(book.define, name, args.limit)
            body |= {"symbols": _symbols(found), "total": len(found)}
        elif args.mode in ("callers", "callees"):
            call = book.callers if args.mode == "callers" else book.callees
            hits: list[Hit] = await anyio.to_thread.run_sync(call, name, args.limit + 1)
            body |= {
                "edges": [one.as_value() for one in hits[: args.limit]],
                "total": len(hits),
                "truncated": len(hits) > args.limit,
            }
        elif args.mode == "impact":
            reached = await anyio.to_thread.run_sync(book.impact, name, args.distance, args.limit)
            rings: dict[int, list[dict[str, Any]]] = {}
            for depth, symbol in reached:
                rings.setdefault(depth, []).append(symbol.as_value())
            body |= {
                "rings": [
                    {"distance": depth, "symbols": found} for depth, found in sorted(rings.items())
                ],
                "total": len(reached),
            }
        else:
            found, total = await anyio.to_thread.run_sync(
                book.entities, args.path, args.kind, args.limit, args.offset
            )
            body |= {
                "symbols": _symbols(found),
                "total": total,
                "truncated": args.offset + args.limit < total,
            }

        return {
            "mode": args.mode,
            "query": args.query,
            "definitions": defined,
            "offset": args.offset,
            "indexed_files": indexed,
            "indexing": indexing,
            **body,
        }

    ctx.require(TOOLS).register(
        define_tool(
            "code_index",
            INDEX_DESCRIPTION,
            parameters=IndexArgs,
            output=ToolOutput(schema=IndexValue, render=_render_index),
            execute=index_tool,
            # See `CodeGraphSeam`. The lock makes the outcome correct, and this
            # setting keeps the scheduler from queueing two runs against each
            # other (B6).
            is_concurrency_safe=False,
            self_limits=True,
            **simple_views("search", "Index code", "paths"),
        )
    )
    ctx.require(TOOLS).register(
        define_tool(
            "code_graph",
            GRAPH_DESCRIPTION,
            parameters=GraphArgs,
            output=ToolOutput(schema=GraphValue, render=_render_graph),
            execute=graph_tool,
            is_concurrency_safe=True,
            self_limits=True,
            **simple_views("search", "Code graph", "query"),
        )
    )

    async def grammar_command(verb: Provisioning, _invocation: CommandContext) -> str:
        """`/code-graph install|status`: make the grammars ready, deliberately.

        This is a **command** and not a tool, by the rule of the seam: a person
        asks the harness to provision, and that costs no model turn. A confined
        agent cannot fetch a grammar, so this command is what makes the index of
        such an agent work.
        """
        wanted = [one for one in (config.languages or DEFAULT_LANGUAGES) if indexable(one)]
        if verb == "status":
            ready, missing = await anyio.to_thread.run_sync(readiness, wanted)
            line = f"grammars under {seam.grammars}: {len(ready)} of {len(wanted)} ready"
            return line + (f"; missing {', '.join(missing)}" if missing else "")
        ready, missing = await anyio.to_thread.run_sync(ensure, wanted)
        if missing:
            return (
                f"{len(ready)} of {len(wanted)} grammars ready; could not fetch "
                f"{', '.join(missing)} — grammars come from GitHub, so this needs "
                "network the first time."
            )
        return f"all {count_of(len(ready), 'grammar')} ready under {seam.grammars}."

    command = install_or_status(
        ROW,
        "Make the tree-sitter grammars ready, or report whether they are.",
        grammar_command,
    )
    contribute_item(
        ctx,
        COMMANDS,
        command,
        label="code-graph command",
    )

    contribute(ctx, Diagnostic(id="code-graph", title="Code graph", order=60, read=seam.report))


def _symbols(rows: list[SymbolRow]) -> list[dict[str, Any]]:
    return [one.as_value() for one in rows]


def _fts(query: str) -> str:
    """Convert the words of a model to an FTS5 expression, with the FTS5 syntax
    neutralized.

    FTS5 reads `-`, `"`, `*`, `(`, `:`, and `NEAR` as operators. A query such as
    `read-before-edit` is therefore a syntax error, and `foo:bar` is a column
    filter that matches nothing. A model that writes prose means neither. Each
    term is quoted and the terms are joined with OR, which is the correct meaning
    of "find symbols about these words". The expression also cannot raise on the
    wording of the model.
    """
    terms = [one for one in query.replace('"', " ").split() if one]
    if not terms:
        return '""'
    return " OR ".join(f'"{one}"' for one in terms)
