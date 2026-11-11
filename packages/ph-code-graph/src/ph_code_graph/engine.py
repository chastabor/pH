"""`code-graph` — a Python-native code graph, so the RLM can ask before it reads.

Two tools. `code_index` walks a tree through `ctx.fs`, parses it with
tree-sitter and records what each file defines and references. `code_graph`
answers five questions over that index — and answers every one of them with
`path:start-end`, so a hit is a pointer the agent can hand straight to `read`.

That last part is the whole point. An agent dropped into an unfamiliar
repository spends its first several thousand tokens reading files to discover
which two mattered. This turns that into one call.

## Python-native, and what that decided

The alternative was wrapping the Rust/Node CodeGraph, and the investigation said
no: that project's Rust kernel is a *tree-sitter extractor behind a Node-API
boundary* (its whole export surface is `extract_file`, `contract_info`,
`grammar_info` and two `cfnptr` helpers), while the intelligence — 29 708 lines
of cross-file resolution, plus graph, search and context layers — is TypeScript.
maturin could not build a `#[napi]` crate anyway. Wrapping it would have bought
a parser Python already has, and left the intelligence behind.

So: `tree-sitter-language-pack` for extraction — a 3.7 MB wheel, 371
languages fetched once into a cache — and `sqlite3` from the
standard library for the graph, because two of the five queries here are exactly
the two `pyturso` cannot serve (`_store` has the measurements).

## Name-based, and it says so

A reference records the *name* it used; `callers` and `callees` join on that
name. Two `register` methods in two classes are one name to this index. Every
result that could be ambiguous carries `definitions`, the number of places that
name is defined, so the model can see the ambiguity instead of being handed one
of them — and `code_graph mode=define` is how it disambiguates.

Resolving properly means an import graph, scope, and per-language type
inference. That is the 29 708 lines this package declined to port. Name-based
answers most of what an agent actually asks and reports where it cannot.

## One call, or a host

Run for one call, `code_index` finishes before it answers. Mounted in a host
(`ph-code-graph serve`, `ph.toolcli.TOOL_HOST`) the process outlives the call, so
the tool answers at once when a run is long — `state: indexing`, with how far it
has got — and the run goes on in the background; `code_graph` answers meanwhile
from what is indexed so far, and says the index is still being built. Either way
the run reports to `progress.json` beside the database. Nothing else differs:
each file is its own transaction, so a run that is stopped keeps every file it
finished, and the next one finds them by their digests.

## Where this row is mounted

In the `ph-code-graph` command's own process, and not in a session's. A session
mounts the `code-graph` row (`ph_code_graph.__init__`), which registers this
package's UTCP manual; the manual's commands run `ph-code-graph`, and that
command mounts *this* row and runs a tool through the registry (`ph.toolcli`).
So `ctx.fs` below is the command's own — rooted at the working directory the
calling agent's sandbox gave it — and `run.agent` is always `None`.

@module ph_code_graph.engine
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.cordis import Context, MountRefusal, ServiceKey, plugin
from ph.json import JsonObject, as_int
from ph.keys import COMMANDS, FS, TOOLS
from ph.llm.types import ContentBlock, text_of
from ph.locks import waited_file_lock
from ph.seams._registry import contribute_item
from ph.seams.changes import tree_state
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.fs import FsService
from ph.text import count_of
from ph.toolcli import TOOL_HOST, BackgroundRuns, Progress, RunProgress, claim_home
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
    parseable,
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
    """The lock file beside a database that orders its writers."""
    return store.path.with_name(f"{store.path.name}.lock")


CODE_GRAPH: ServiceKey[CodeGraphSeam] = ServiceKey("code_graph")
"""The code graph, for `/code-graph` and the tools it backs."""


@dataclass(slots=True)
class CodeGraphSeam:
    """The service published as `ctx.code_graph`.

    Holds where the index lives, and orders its writers. The order is why
    `code_index` declares itself not concurrency-safe: two calls indexing
    overlapping trees would interleave their per-file transactions, and while
    SQLite would keep each one atomic the *pair* would report counts neither of
    them produced.
    """

    ctx: Context
    grammars: Path
    """Where the tree-sitter grammar cache was pointed. See `_extract.use_cache`."""
    runs: BackgroundRuns = field(default_factory=BackgroundRuns)
    """The indexing runs, by index directory, and how far each has got."""

    def store_for(self, root: Path) -> CodeGraphStore:
        """This workspace's index: where the caller said, or a directory per root
        (`_config.index_path`).

        Never shared between two trees — a worktree at another revision holds
        different line numbers, so one index for both would hand out pointers
        that are quietly stale. A session's caller names a directory of the
        agent's own scratch, which goes when the agent's scratch does; the
        per-root directory under `$PH_CACHE` records its workspace, and one whose
        workspace is gone is swept on a later index (`ph.workspace_caches`).
        """
        return CodeGraphStore(path=index_path(root))

    def indexing(self, store: CodeGraphStore) -> Progress | None:
        """The run building `store` right now, or `None`
        (`BackgroundRuns.working`): a run holds the index's lock while it works."""
        return self.runs.working(store.path.parent, _lock_of(store), what=_LOCKED)

    @asynccontextmanager
    async def writing(self, store: CodeGraphStore) -> AsyncIterator[None]:
        """Hold `store` for one indexing run, against every other writer of it.

        **A lock file beside the database, not a lock in this process.** The index
        is shared by everything on one workspace — two roots in a daemon, a
        `phern -p`, a `ph-code-graph` command — and an `anyio.Lock` here ordered
        only the calls of one seam. Two processes each stayed atomic per file and
        still raced on the stored version-control token, so the second could
        record a token vouching for files the first was still writing. `flock`
        refuses a second holder even in one process, so this one lock orders this
        seam's own calls too.

        Waited for rather than refused: a second indexing run is expected to queue
        behind the first, and then finds most of the tree already current. The
        directory needs no making here — taking the lock makes it.
        """
        async with waited_file_lock(_lock_of(store), what=_LOCKED):
            yield

    def report(self) -> list[tuple[str, str]]:
        """`phern doctor`'s section."""
        # Not a declared dependency: the row checks for it before this runs.
        import tree_sitter  # noqa: PLC0415

        # No `hasattr` guard: the row declares `inject=[FS]`, so `ctx.fs` is
        # present for as long as it is active — and a `Path.cwd()` fallback
        # answered with an index keyed to the process directory rather than the
        # workspace, which is worse than the traceback it avoided.
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
    """`indexing`: the run goes on in the background, and the counts below are
    how far it has got. Only a host answers that; a call that ran by itself
    answers `done`."""
    done: int = 0
    total: int = 0
    """Files this run has been through, and how many it will go through."""
    indexed: int = 0
    """Files parsed this call — changed or new only."""
    unchanged: int = 0
    """Files already current, by content digest."""
    removed: int = 0
    symbols: int = 0
    """Symbols written this call."""
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
    """The line the reference is written on — what to open."""
    ref_path: str
    """Which file `ref_line` is in. **Not always `path`**: for `callees` the
    symbol is the callee's definition while the reference is a line in the
    caller. See `_store.Hit`."""
    via: str
    """The name the edge was written as."""


class RingValue(ToolModel):
    distance: int
    symbols: list[SymbolValue]


class IndexingValue(ToolModel):
    done: int
    total: int


class GraphValue(ToolModel):
    mode: str
    query: str | None
    definitions: int
    """How many places `query` is defined. **Above 1 means the answer is
    ambiguous** — this index matches by name; see the module docstring."""
    total: int
    offset: int
    truncated: bool
    indexed_files: int
    indexing: IndexingValue | None = None
    """Set while a run is still building the index: the answer is from what it
    has indexed so far."""
    symbols: list[SymbolValue] = Field(default_factory=list)
    edges: list[EdgeValue] = Field(default_factory=list)
    rings: list[RingValue] = Field(default_factory=list)


# ------------------------------------------------------------------- render ----


def _where(one: dict[str, Any]) -> str:
    return f"{one['path']}:{one['start_line']}-{one['end_line']}"


def _doc(one: dict[str, Any]) -> str:
    doc = (one.get("doc") or "").strip().splitlines()
    return f"  — {doc[0][:70]}" if doc else ""


def _render_index(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("state") == "indexing":
        return text_content(
            f"Indexing in the background: {value['done']} of "
            f"{count_of(value['total'], 'file')} so far. `code_graph` answers now, from "
            "what is indexed already, and says while the index is still being built. "
            "You will be told when it finishes; call `code_index` again to see how far "
            "it has got.\n"
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
"""How many skipped files a result names before it counts the rest."""

BRIEFLY = 8
"""How many the note about a run nobody waited for names (`Progress.note`)."""


def _index_summary(value: Mapping[str, Any], *, shown: int = SHOWN) -> str:
    """What a finished run did: its counts, and each file it skipped and why, up
    to `shown`. The note about a run nobody waited for says the same — a grammar
    that could not be fetched leaves a graph of nothing, and "finished" alone
    would not say so."""
    lines = [
        f"Indexed {count_of(value['indexed'], 'file')} "
        f"({value['unchanged']} already current), {count_of(value['symbols'], 'symbol')}."
    ]
    for skipped in value["skipped"][:shown]:
        lines.append(f"  skipped {skipped['path']}: {skipped['reason']}")
    if len(value["skipped"]) > shown:
        lines.append(f"  [and {len(value['skipped']) - shown} more skipped]")
    lines.append(
        f"The graph holds {count_of(value['total_symbols'], 'symbol')} and "
        f"{count_of(value['total_refs'], 'reference')} across "
        f"{count_of(value['total_files'], 'file')} "
        f"({', '.join(value['languages'][:6]) or 'no languages'})."
    )
    return "\n".join(lines)


def _ambiguity(value: Any) -> str:  # noqa: ANN401
    if value["definitions"] <= 1:
        return ""
    return (
        f"\n[{value['query']!r} is defined in {value['definitions']} places and this "
        "index matches by name, so these edges may belong to more than one of them; "
        "`mode=define` lists them]"
    )


def _render_graph(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    """The answer, and under it the warning it carries while its index is still
    being built."""
    answer = _render_answer(args, value)
    indexing = value.get("indexing")
    if not indexing:
        return answer
    return text_content(
        f"{text_of(answer)}\n"
        f"[Not finished indexing: {indexing['done']} of "
        f"{count_of(indexing['total'], 'file')}. This answer covers what is indexed so "
        "far; ask again once it finishes.]"
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
            # `ref_path`, not `path`: the reference and the definition are in
            # different files for `callees`, and printing the line against the
            # wrong one is a pointer to nothing (`_store.Hit`).
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
"""What `/code-graph install|status` covers when the row names no languages.

A convenience list, and **filtered through `indexable` before use** — `bash` and
`sql` are in it and have no tags query, so an unfiltered list reported them
ready and then handed the extractor a language it could not use. Not read by the
indexer, which asks `indexable`/`local` per file, so a language outside this
list still works when the pack has it.
"""


@plugin("code-graph-engine", affects="environment", inject=[TOOLS, FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam and register both tools."""
    for module in ("tree_sitter", "tree_sitter_language_pack"):
        if importlib.util.find_spec(module) is None:
            raise MountRefusal(MISSING)

    # Before anything parses. The pack keeps every grammar it uses in a writable
    # cache directory and fails hard without one, so the row
    # names a path it can vouch for rather than inheriting whatever `HOME`
    # happens to be — see `use_cache`.
    try:
        # Through `ctx.effect` and in a worker thread, for two reasons that
        # happen to share one line. The effect is so the library's
        # process-global cache setting unwinds with this row rather than
        # outliving it (§4.9, I2 — see `cache_release`); the thread is because
        # `use_cache` imports `tree_sitter_language_pack`, a 20 ms dlopen that
        # a profile mounting many rows would otherwise serialize on the loop.
        await ctx.effect(
            lambda: anyio.to_thread.run_sync(cache_release, grammar_default()),
            label="tree-sitter-cache",
        )
    except OSError as error:
        # `MountRefusal`, not the `OSError`: every command that mounts a profile
        # turns this one type into a sentence and an exit code, and leaves
        # anything else as the traceback a bug deserves. A read-only `$PH_CACHE`
        # is a deployment fact an operator can fix, not a bug — measured inside
        # a sandbox that made `~/.cache` read-only, where this arrived as
        # fourteen frames of `pathlib`.
        raise MountRefusal(
            f"code-graph cannot write the tree-sitter grammar cache at {error.filename}: "
            f"{error.strerror}. The grammars are materialized there on first use, so this "
            "path must be writable — point $PH_CACHE somewhere it is, or set "
            f"{GRAMMAR_CACHE_ENV}."
        ) from error

    seam = CodeGraphSeam(
        ctx=ctx, grammars=grammar_cache(), runs=BackgroundRuns(grace=config.wait_s)
    )
    ctx.provide(CODE_GRAPH, seam)

    def store(run: ToolRunContext) -> CodeGraphStore:
        return seam.store_for(ctx.require(FS).root_for(run.agent))

    def _forget_missing(
        fs: FsService,
        book: CodeGraphStore,
        known: Mapping[str, tuple[str, str]],
        run: ToolRunContext,
    ) -> int:
        """Drop every indexed path that is no longer on disk. One thread hop."""
        gone = [path for path in known if not fs.resolve(path, agent=run.agent).exists()]
        return book.forget(gone) if gone else 0

    async def index_tool(args: IndexArgs, run: ToolRunContext) -> dict[str, Any]:
        # In a host the run outlives this call, which answers with the run's
        # counts if it ends within the grace and with how far it has got if not.
        book = store(run)
        value: dict[str, Any] | None = await seam.runs.run(
            ctx.get(TOOL_HOST),
            book.path.parent,
            args,
            partial(index_run, args, run),
            what="files",
            said=partial(_index_summary, shown=BRIEFLY),
            raise_if_canceled=run.raise_if_canceled,
        )
        if value is not None:
            return value
        progress = seam.indexing(book)
        stats: dict[str, Any] = (
            await anyio.to_thread.run_sync(book.stats)
            if await anyio.to_thread.run_sync(book.exists)
            else {"files": 0, "symbols": 0, "refs": 0, "languages": []}
        )
        return IndexValue(
            state="indexing",
            done=0 if progress is None else progress.done,
            total=0 if progress is None else progress.total,
            total_files=stats["files"],
            total_symbols=stats["symbols"],
            total_refs=stats["refs"],
            languages=stats["languages"],
        ).model_dump()

    async def index_run(
        args: IndexArgs, run: ToolRunContext, progress: RunProgress
    ) -> dict[str, Any]:
        """One indexing run, to its end. Its caller's cancellation is asked at
        each `progress.tick`, for as long as the run is that caller's own."""
        fs = ctx.require(FS)
        book = store(run)
        wanted = config.index.narrowed(paths=args.paths, glob=args.glob, exclude=args.exclude)
        paths = await fs.collect(
            wanted.paths,
            wanted.include,
            scope=run.scope,
            limit=config.max_files,
            agent=run.agent,
            exclude=wanted.exclude,
        )
        skipped: list[dict[str, str]] = []
        indexed = unchanged = symbols = removed = 0
        total = len(paths)

        # Before the lock: the record and the sweep touch nothing an index holds.
        await anyio.to_thread.run_sync(claim_home, index_base(), fs.root_for(run.agent))
        async with seam.writing(book):
            # Under the lock, so the report is this run's and not one waiting.
            progress.begin(total)
            await anyio.to_thread.run_sync(book.prepare)
            if args.forget:
                # Falls through to the one `stats` read and the one return
                # below: `indexed`, `unchanged` and `symbols` are already 0 and
                # `skipped` already empty, so a second copy of the payload was
                # eleven lines saying that again — and a second shape for one
                # `IndexValue` schema, in the mode with no test asserting it.
                removed = await anyio.to_thread.run_sync(book.forget, paths)
                paths = []
            known = await anyio.to_thread.run_sync(book.known)
            # **Ask the version control what changed before reading anything.**
            # `tree_state` never raises and answers an empty state for a tree
            # with no backend, so the loop below is correct either way — it just
            # reads every file when nothing can vouch for one. Which backend
            # answers is the workspace provider's to say (`ph.seams.changes`).
            parsers: dict[str, bool] = {}
            stored_token = await anyio.to_thread.run_sync(book.token)
            state = await tree_state(ctx, fs.root_for(run.agent), since=stored_token)
            for count, path in enumerate(paths):
                progress.tick(count)
                language = detect_language(path)
                # Derived from the pack rather than checked against a name
                # list: `.txt`, `.ini`, `.proto` and a dozen others are
                # languages the detector claims and the tags queries do not
                # cover, and the row's default `include` reaches all of them.
                if language is None or not indexable(language):
                    continue
                if config.languages and language not in config.languages:
                    continue
                if language not in parsers:
                    # **One hop per language, not per file.** The first file of a
                    # language is worth a worker thread — `parseable`
                    # materializes a grammar out of the wheel, 6.4 s the first
                    # time — and every later file of it is a cached boolean, so
                    # the 60.6 µs hop was the entire cost: 1.2 s across a
                    # 20 000-file tree, which is the whole saving the change
                    # filter exists to deliver, handed back.
                    #
                    # Keyed per call rather than `@cache`d for the process: a
                    # person who runs `/code-graph install` between two calls has
                    # to see the grammar that appeared.
                    parsers[language] = await anyio.to_thread.run_sync(parseable, language)
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
                    # Proved unchanged by git or jj, so the file is never opened.
                    # The saving is the whole read: 5.2 ms of I/O plus 3.0 ms of
                    # sha256 per 136 files, which is ~1.2 s on a 20 000-file tree.
                    unchanged += 1
                    continue
                refused = fs.skip_reason(path, max_bytes=config.max_bytes, agent=run.agent)
                if refused:
                    skipped.append({"path": path, "reason": refused})
                    continue
                slice_ = await fs.read(
                    path,
                    limit=None,
                    # `skip_reason` above is this walk's bound, and it is a skip
                    # rather than a raise on purpose: an indexing pass must not
                    # fail because it found a minified bundle. `read`'s own
                    # default would be a second, larger bound that raises.
                    max_bytes=None,
                    scope=run.scope,
                    agent=run.agent,
                    session=run.session,
                )
                # **No prose test here, deliberately** (X6 review). The text
                # index refuses documents that hold no prose; source code is not
                # prose, and the same predicate refused a module whose only
                # statement is a long `__all__`, and a `TABLE = {...}` literal
                # beside `def lookup()` — so `lookup` would have vanished from
                # this index with nothing but a skip line to say so. Telling a
                # minified bundle from a legitimate data module by shape is not
                # reliable for code, and a false positive here loses symbols
                # silently. `skip_reason`'s size bound stays this walk's guard.
                # The content hash stays the authority on *whether* a file
                # changed — the filter above only decides whether to open it, so
                # the "content, not clock" guarantee is untouched.
                digest = digest_of(slice_.text)
                if stored_digest == digest:
                    unchanged += 1
                    continue
                try:
                    extraction = await anyio.to_thread.run_sync(
                        extract, path, slice_.text, language
                    )
                except Exception as error:
                    # Reported and skipped rather than raised: a tree walk that
                    # died on one unparseable file would be the failure mode the
                    # package this replaced actually had.
                    log.debug("ph_code_graph: %s did not parse", path, exc_info=True)
                    skipped.append({"path": path, "reason": f"did not parse ({error})"})
                    continue
                symbols += await anyio.to_thread.run_sync(
                    book.put, path, digest, extraction, state.id_for(path)
                )
                indexed += 1
            if not args.forget:
                # **Gone from disk is gone from the index** (X3). A deleted file
                # kept its symbols until somebody ran `forget` by hand, so
                # `code_graph` went on naming definitions at paths that are not
                # there — and a rename showed the symbol twice, once under each
                # name, with nothing saying which was real. A pointer into a file
                # that does not exist is the one answer this tool must not give,
                # because the model's next move is to open it.
                #
                # **Asked of the filesystem, not of the walk.** "Absent from
                # `paths`" looks like the same question and is not: `collect`
                # stops at `max_files` and reports nothing, and the policy screen
                # prunes whole directories — so on a tree past the cap, or with a
                # `permissions-fs` rule hiding a subtree, the walk is a *subset*
                # of the scope and differencing against it would forget
                # everything it did not reach. It also settles the narrowed walk
                # for free, where the question is not "did we cover the scope"
                # but "is this file still there".
                removed = await anyio.to_thread.run_sync(_forget_missing, fs, book, known, run)
            # Stored last, and only after the loop: a token recorded before the
            # writes would, on a crash between the two, vouch for files this run
            # never actually indexed.
            if state.token:
                await anyio.to_thread.run_sync(book.remember, state.token)
            stats = await anyio.to_thread.run_sync(book.stats)

        return {
            "state": "done",
            "done": total,
            "total": total,
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
        progress = seam.indexing(book)

        # `file_count`, not `stats()`: the header needs one number and `stats`
        # counts symbols and refs and groups files by language — three full
        # scans per query, growing with the corpus (1.97 ms against 0.22 ms on a
        # 2 000-file index).
        indexed = await anyio.to_thread.run_sync(book.file_count)
        name = args.query or ""
        # `COUNT(*)`, not `len(define(name, 100))`, which materialized a hundred
        # rows and then reported the *cap* as the count for anything past it.
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
            "indexing": (
                None if progress is None else {"done": progress.done, "total": progress.total}
            ),
            **body,
        }

    ctx.require(TOOLS).register(
        define_tool(
            "code_index",
            INDEX_DESCRIPTION,
            parameters=IndexArgs,
            output=ToolOutput(schema=IndexValue, render=_render_index),
            execute=index_tool,
            # See `CodeGraphSeam` — the lock makes the outcome correct, and this
            # keeps the scheduler from queueing two against each other (B6).
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
        """`/code-graph install|status` — make the grammars ready, on purpose.

        A **command** rather than a tool, per the seam's own rule: a person asks
        the harness to provision, and it costs no model turn. A confined agent
        cannot fetch a grammar, so this is what makes its index work.
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
    """A model's words as an FTS5 expression, with its syntax neutralized.

    FTS5 reads `-`, `"`, `*`, `(`, `:` and `NEAR` as operators, so a query like
    `read-before-edit` is a syntax error and `foo:bar` is a column filter that
    matches nothing. A model writing prose did not mean either. Every term is
    quoted and the set is OR-ed, which is what "find symbols about these words"
    should do — and it cannot raise on the model's phrasing.
    """
    terms = [one for one in query.replace('"', " ").split() if one]
    if not terms:
        return '""'
    return " OR ".join(f'"{one}"' for one in terms)
