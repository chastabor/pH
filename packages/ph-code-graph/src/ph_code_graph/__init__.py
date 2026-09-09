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

So: `tree-sitter-language-pack` for extraction — 26 languages bundled in a
3.7 MB wheel and working offline, 371 available — and `sqlite3` from the
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

@module ph_code_graph
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.cordis import Context, MountRefusal, plugin
from ph.paths import default_cache_path, resolve_roots
from ph.seams._registry import contribute_via
from ph.seams.changes import tree_state
from ph.seams.commands import CommandDefinition
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.skills import discover_skills
from ph.text import count_of
from ph.tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ph.tools.errors import HarnessError
from ph.tools.presentation import simple_views
from ph.wire import WireModel

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

__all__ = ["BUNDLE", "CodeGraphSeam", "Config", "apply"]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""This distribution's profile layer, for the `ph.bundles` group.

Discovered rather than imported, which is what lets `ph-app` compose a
profile that layers this package without depending on it.
"""

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


class Config(WireModel):
    """Row config for `code-graph`."""

    path: str = ""
    """Where the index lives. Defaults to `$PH_CACHE/code-graph/<root digest>.db`.

    Under the cache root because the index is **rebuildable** — the source is the
    truth and this is derived, which is the lifecycle `$PH_CACHE` names (Q1).
    Keyed by a digest of the workspace root so two checkouts do not share one
    index and answer each other's questions."""
    glob: str = "**/*"
    """What `code_index` takes from a directory when the call does not say.

    Everything, because the language filter is the real one: a path whose
    extension names no supported code language is skipped, so a `**/*` here
    means "all the code" rather than "all the files"."""
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


# --------------------------------------------------------------------- seam ----


@dataclass(slots=True)
class CodeGraphSeam:
    """The service published as `ctx.code_graph`.

    Holds the store and the lock. The lock is why `code_index` declares itself
    not concurrency-safe: two calls indexing overlapping trees would interleave
    their per-file transactions, and while SQLite would keep each one atomic the
    *pair* would report counts neither of them produced.
    """

    ctx: Context
    config: Config
    grammars: Path
    """Where the tree-sitter grammar cache was pointed. See `_extract.use_cache`."""
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    def store_for(self, root: Path) -> CodeGraphStore:
        """This workspace's index. One file per root, and **nothing prunes them.**

        Keyed per root on purpose — a worktree at another revision holds
        different line numbers, so sharing one index would hand out pointers
        that are quietly stale. The cost is that under a profile whose children
        run in worktrees, every throwaway tree leaves a database behind, and
        `ph doctor` names only the current one. Stated here rather than left to
        be discovered, which is this codebase's rule for a cache nothing
        collects (`ph.seams.uploads` says the same about attachments): deleting
        `$PH_CACHE/code-graph` reclaims all of them and costs a re-index.
        """
        digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
        return CodeGraphStore(
            path=default_cache_path(self.config.path, "code-graph", f"{digest}.db")
        )

    def locked(self) -> anyio.Lock:
        return self._lock

    def report(self) -> list[tuple[str, str]]:
        """`ph doctor`'s section."""
        import tree_sitter

        # No `hasattr` guard: the row declares `inject=["fs"]`, so `ctx.fs` is
        # present for as long as it is active — and a `Path.cwd()` fallback
        # answered with an index keyed to the process directory rather than the
        # workspace, which is worse than the traceback it avoided.
        root = self.ctx.fs.root_for(None)
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
    paths: list[str] = Field(
        default_factory=lambda: ["."],
        description="Directories or files to index. Relative to the workspace root.",
    )
    glob: str | None = Field(
        None, description="Which files to take from a directory. Defaults to the row's setting."
    )
    forget: bool = Field(False, description="Remove these paths from the index instead.")


class SkippedValue(ToolModel):
    path: str
    reason: str


class IndexValue(ToolModel):
    indexed: int
    """Files parsed this call — changed or new only."""
    unchanged: int
    """Files already current, by content digest."""
    removed: int
    symbols: int
    """Symbols written this call."""
    skipped: list[SkippedValue]
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
    symbols: list[SymbolValue] = Field(default_factory=list)
    edges: list[EdgeValue] = Field(default_factory=list)
    rings: list[RingValue] = Field(default_factory=list)


# ------------------------------------------------------------------- render ----


def _where(one: dict[str, Any]) -> str:
    return f"{one['path']}:{one['start_line']}-{one['end_line']}"


def _doc(one: dict[str, Any]) -> str:
    doc = (one.get("doc") or "").strip().splitlines()
    return f"  — {doc[0][:70]}" if doc else ""


def _render_index(args: Any, value: Any) -> Any:
    if args.get("forget"):
        return text_content(
            f"Removed {count_of(value['removed'], 'file')} from the index. "
            f"It now holds {count_of(value['total_symbols'], 'symbol')} "
            f"from {count_of(value['total_files'], 'file')}."
        )
    lines = [
        f"Indexed {count_of(value['indexed'], 'file')} "
        f"({value['unchanged']} already current), {count_of(value['symbols'], 'symbol')}."
    ]
    for skipped in value["skipped"][:20]:
        lines.append(f"  skipped {skipped['path']}: {skipped['reason']}")
    if len(value["skipped"]) > 20:
        lines.append(f"  [and {len(value['skipped']) - 20} more skipped]")
    lines.append(
        f"The graph holds {count_of(value['total_symbols'], 'symbol')} and "
        f"{count_of(value['total_refs'], 'reference')} across "
        f"{count_of(value['total_files'], 'file')} "
        f"({', '.join(value['languages'][:6]) or 'no languages'})."
    )
    return text_content("\n".join(lines))


def _ambiguity(value: Any) -> str:
    if value["definitions"] <= 1:
        return ""
    return (
        f"\n[{value['query']!r} is defined in {value['definitions']} places and this "
        "index matches by name, so these edges may belong to more than one of them; "
        "`mode=define` lists them]"
    )


def _render_graph(args: Any, value: Any) -> Any:
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
                f"{count_of(int(args.get('distance') or 1), 'hop')}."
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


def offer_skills(ctx: Context) -> None:
    """Install this package's `SKILL.md` files, if a skills registry is mounted.

    `discover_skills` already globs `<root>/*/SKILL.md`, validates each one and
    skips a malformed one with a logged reason — the whole of what a
    hand-written loader was doing here in 28 lines, duplicated byte-for-byte
    into the sibling package. Reaching for the seam's own reader instead is the
    rule `contribute_via`'s docstring states about itself.

    **Through `contribute_via`, so the skill arrives exactly when the plugin
    does** and leaves with it. `skills-progressive` ships an empty `paths` on
    purpose — scanning a well-known directory would make "install a skill" mean
    "drop a file somewhere", and a skill is something a distribution installs
    deliberately (I7). A distribution registering its own is that act.

    Only the one-line description rides the prompt every turn; the body stays on
    disk until the model asks for it by name (G9).
    """
    root = Path(__file__).parent / "skills"
    for skill in discover_skills([str(root)], source="ph-code-graph"):
        contribute_via(ctx, "skills", skill, label=f"skill({skill.name})")


@plugin("code-graph", inject=["tools", "fs"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam and register both tools."""
    import importlib.util

    for module in ("tree_sitter", "tree_sitter_language_pack"):
        if importlib.util.find_spec(module) is None:
            raise MountRefusal(MISSING)

    # Before anything parses. The pack materialises even its *bundled* grammars
    # into a writable cache directory and fails hard without one, so the row
    # names a path it can vouch for rather than inheriting whatever `HOME`
    # happens to be — see `use_cache`.
    grammar_cache = resolve_roots().cache / "tree-sitter"
    try:
        # Through `ctx.effect` and in a worker thread, for two reasons that
        # happen to share one line. The effect is so the library's
        # process-global cache setting unwinds with this row rather than
        # outliving it (§4.9, I2 — see `cache_release`); the thread is because
        # `use_cache` imports `tree_sitter_language_pack`, a 20 ms dlopen that
        # a profile mounting many rows would otherwise serialise on the loop.
        await ctx.effect(
            lambda: anyio.to_thread.run_sync(cache_release, grammar_cache),
            label="tree-sitter-cache",
        )
        grammars = grammar_cache
    except OSError as error:
        # `MountRefusal`, not the `OSError`: every command that mounts a profile
        # turns this one type into a sentence and an exit code, and leaves
        # anything else as the traceback a bug deserves. A read-only `$PH_CACHE`
        # is a deployment fact an operator can fix, not a bug — measured inside
        # a sandbox that made `~/.cache` read-only, where this arrived as
        # fourteen frames of `pathlib`.
        raise MountRefusal(
            f"code-graph cannot write the tree-sitter grammar cache at {error.filename}: "
            f"{error.strerror}. The grammars are materialised there on first use, so this "
            "path must be writable — point $PH_CACHE somewhere it is, or set "
            "TREE_SITTER_LANGUAGE_PACK_CACHE_DIR."
        ) from error

    seam = CodeGraphSeam(ctx=ctx, config=config, grammars=grammars)
    ctx.provide("code_graph", seam)

    def store(run: ToolRunContext) -> CodeGraphStore:
        return seam.store_for(ctx.fs.root_for(run.agent))

    async def index_tool(args: IndexArgs, run: ToolRunContext) -> Any:
        book = store(run)
        paths = await ctx.fs.collect(
            args.paths,
            args.glob or config.glob,
            scope=run.scope,
            limit=config.max_files,
            agent=run.agent,
        )
        skipped: list[dict[str, str]] = []
        indexed = unchanged = symbols = removed = 0

        async with seam.locked():
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
            state = await tree_state(ctx, ctx.fs.root_for(run.agent), since=stored_token)
            for path in paths:
                run.raise_if_cancelled()
                language = detect_language(path)
                # Derived from the pack rather than checked against a name
                # list: `.txt`, `.ini`, `.proto` and a dozen others are
                # languages the detector claims and the tags queries do not
                # cover, and the row's default `glob` reaches all of them.
                if language is None or not indexable(language):
                    continue
                if config.languages and language not in config.languages:
                    continue
                if language not in parsers:
                    # **One hop per language, not per file.** The first file of a
                    # language is worth a worker thread — `parseable`
                    # materialises a grammar out of the wheel, 6.4 s the first
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
                                f"no {language} parser available — a language outside the "
                                "wheel's bundled set comes from GitHub; /code-graph install"
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
                refused = ctx.fs.skip_reason(path, max_bytes=config.max_bytes, agent=run.agent)
                if refused:
                    skipped.append({"path": path, "reason": refused})
                    continue
                slice_ = await ctx.fs.read(
                    path,
                    limit=None,
                    scope=run.scope,
                    agent=run.agent,
                    session=run.session,
                )
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
            # Stored last, and only after the loop: a token recorded before the
            # writes would, on a crash between the two, vouch for files this run
            # never actually indexed.
            if state.token:
                await anyio.to_thread.run_sync(book.remember, state.token)
            stats = await anyio.to_thread.run_sync(book.stats)

        return {
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

    async def graph_tool(args: GraphArgs, run: ToolRunContext) -> Any:
        book = store(run)
        if not await anyio.to_thread.run_sync(book.exists):
            raise HarnessError(
                "no code index exists for this workspace yet; run `code_index` first",
                CODE_GRAPH_FAILED,
            )
        if args.mode != "entities" and not args.query:
            raise HarnessError(f"mode={args.mode} needs `query`", CODE_GRAPH_FAILED)

        # `file_count`, not `stats()`: the header needs one number and `stats`
        # counts symbols and refs and groups files by language — three full
        # scans per query, growing with the corpus (1.97 ms against 0.22 ms on a
        # 2 000-file index).
        indexed = await anyio.to_thread.run_sync(book.file_count)
        name = args.query or ""
        # `COUNT(*)`, not `len(define(name, 100))`, which materialised a hundred
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
            **body,
        }

    ctx.tools.register(
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
    ctx.tools.register(
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

    async def install(argument: str, command: Any) -> str:
        """`/code-graph install|status` — make the grammars ready, on purpose.

        A **command** rather than a tool, per the seam's own rule: a person asks
        the harness to provision, and it costs no model turn. It matters less
        here than for `text-index` — 26 languages are inside the wheel and only
        the long tail fetches — but "is this ready" should have one answer per
        plugin, asked the same way.
        """
        verb = argument.strip().lower() or "status"
        if verb not in ("install", "status"):
            return f"/code-graph takes `install` or `status`, not {argument.strip()!r}."
        wanted = [one for one in (config.languages or DEFAULT_LANGUAGES) if indexable(one)]
        if verb == "status":
            ready, missing = await anyio.to_thread.run_sync(readiness, wanted)
            line = f"grammars under {seam.grammars}: {len(ready)} of {len(wanted)} ready"
            return line + (f"; missing {', '.join(missing)}" if missing else "")
        ready, missing = await anyio.to_thread.run_sync(ensure, wanted)
        if missing:
            return (
                f"{len(ready)} of {len(wanted)} grammars ready; could not fetch "
                f"{', '.join(missing)} — a language outside the wheel's bundled set comes "
                "from GitHub, so this needs network the first time."
            )
        return f"all {count_of(len(ready), 'grammar')} ready under {seam.grammars}."

    # `contribute_via` rather than a hand-written `ctx.inject(["commands"], ...)`:
    # that is this helper's body verbatim, and writing it out in a downstream
    # package is the failure its own docstring predicts. It still waits for the
    # key rather than hard-injecting it, so a headless `ph -p` that mounts no
    # command registry loads the row anyway.
    contribute_via(
        ctx,
        "commands",
        CommandDefinition(
            name="code-graph",
            summary="Make the tree-sitter grammars ready, or report whether they are.",
            argument_hint="[install|status]",
            run=install,
        ),
        label="code-graph command",
    )

    offer_skills(ctx)
    contribute(ctx, Diagnostic(id="code-graph", title="Code graph", order=60, read=seam.report))


def _symbols(rows: list[SymbolRow]) -> list[dict[str, Any]]:
    return [one.as_value() for one in rows]


def _fts(query: str) -> str:
    """A model's words as an FTS5 expression, with its syntax neutralised.

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
