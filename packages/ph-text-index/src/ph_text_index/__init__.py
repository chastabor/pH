"""`text-index` — semantic retrieval over the agent's own documents.

Two rows and two tools. `text_index` reads documents through `ctx.fs`, cuts them
into passages that remember their line numbers, embeds them and puts them in a
turbovec index on disk. `text_search` answers a question with the passages
themselves plus the `path:start-end` each came from — so a hit is both an answer
and a pointer the agent can `read` for the rest.

## Why the tools wait for an embedder

`text-index` publishes the seam; `text-index-local` provides the model. The
tools are registered by the *claim* — `TextIndexSeam.register` offers them on
the provider's own scope — so a profile that mounts the seam and no provider
advertises **nothing**, and unmounting the provider takes the tools with it.
That is the rule `subagent-task` states, kept here for the same reason: a tool
named in every system prompt and refused on every call spends context teaching
the model a capability the deployment does not have.

It is a callback rather than a `ctx.inject` because there is no key to wait on —
an embedder is claimed into this seam, not provided as a service — and because
the provider row may sit either side of this one in the profile, so "what has
been provided so far" is the wrong question.

## Reading through `ctx.fs`, and what that buys

Every byte indexed arrives through `ctx.fs.read`, which means `fs/read-intent`
fires, `permissions-fs` decides, the workspace tier bounds the path and every
registered screen gets its say — the same door `read` goes through, for the same
reason `tool-attach` insists on it (I-9). It matters more here than there,
because indexing is a *bulk* read: a tool that walked the tree with `Path.open`
would be an exfiltration primitive with a glob argument.

The corollary is that the index is per-deployment and not per-agent. Two agents
with different workspaces share `$PH_CACHE/text-index/<embedder>` unless a
profile says otherwise, so a passage one agent indexed is retrievable by
another. That is the right default for a documentation corpus and the wrong one
for anything private; `path:` in the row's config is how a deployment separates
them.

@module ph_text_index
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.cordis import Context, Disposer, MountRefusal, Running, ServiceKey, plugin
from ph.keys import COMMANDS, FS, SKILLS, TOOLS
from ph.llm.types import ContentBlock
from ph.paths import default_cache_path, resolve_roots
from ph.seams._registry import claim_slot, contribute_item
from ph.seams.changes import TreeState, tree_state
from ph.seams.commands import CommandDefinition
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.skills import discover_skills
from ph.text import count_of
from ph.tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ph.tools.errors import HarnessError
from ph.tools.presentation import simple_views
from ph.wire import WireModel

from ._chunk import Chunk, chunk_text
from ._embed import Embedder, LocalWeights, SentenceTransformerEmbedder
from ._store import Record, TextIndex

__all__ = [
    "BUNDLE",
    "Config",
    "Embedder",
    "LocalConfig",
    "TextIndexSeam",
    "apply",
    "local",
]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""This distribution's profile layer, for the `ph.bundles` group.

Discovered rather than imported, which is what lets `ph-app` compose a
profile that layers this package without depending on it.
"""

log = logging.getLogger("ph_text_index")

TEXT_INDEX_FAILED = "TEXT_INDEX_FAILED"

MISSING_TURBOVEC = (
    "the text-index row needs `turbovec`, which ph-text-index depends on; "
    "install it (`uv sync`, or `pip install turbovec`) or remove the row"
)
MISSING_MODEL = (
    "the text-index-local row needs `sentence-transformers`, which ph-text-index "
    "depends on; install it (`uv sync`) or replace the row with an embedder of "
    "your own registered on `ctx.text_index`"
)

INDEX_DESCRIPTION = """Add documents to the searchable text index.

Reads each file, cuts it into passages and embeds them so `text_search` can find
them by meaning rather than by substring. Use it on documentation, notes, specs
— prose where the words you would grep for are not the words in the file. Point
it at a directory and it takes everything matching `glob`.

Re-indexing a file replaces its passages, so running this again after an edit is
correct and cheap. `forget: true` removes a document instead."""

SEARCH_DESCRIPTION = """Search the text index by meaning and get the passages back.

Returns each hit's text along with the `path` and line range it came from, so
you can answer from the passage or `read` the file around it for more. Prefer
`grep` when you know the exact string; prefer this when you know the idea.

Only finds what `text_index` has already indexed — an empty result may mean the
corpus was never indexed rather than that nothing matches."""


# ------------------------------------------------------------------- config ----


class Config(WireModel):
    """Row config for `text-index`."""

    path: str = ""
    """Where the index lives. Defaults to `$PH_CACHE/text-index/<embedder>`.

    Under the cache root because the index is **rebuildable** — the documents are
    the truth and this is a derived artifact, which is exactly the lifecycle
    `$PH_CACHE` names (Q1). A deployment indexing a corpus that is expensive to
    re-embed points this at `$PH_HOME` instead, and gets it backed up with the
    sessions."""
    glob: str = "**/*.md"
    """What `text_index` takes from a directory when the call does not say.

    Markdown by default because that is what a repository's prose is in, and
    because a default of `**/*` would index a `node_modules` on the first call
    somebody made without thinking."""
    max_chars: int = 1_200
    """The largest passage, in characters. Roughly 300 tokens.

    Big enough to hold an argument, small enough that a hit is mostly the
    relevant part — the cost of a large chunk is not storage but that the
    matching sentence is diluted by everything else in it."""
    overlap_chars: int = 200
    """How much of a passage's tail is repeated at the head of the next.

    See `_chunk`: the concession to a paragraph whose meaning depends on the one
    before it."""
    bit_width: Literal[2, 4] = 4
    """turbovec's quantization width: 4, or 2 for half the memory and less recall.

    A closed set, so the **profile** refuses a third value rather than the row
    mounting happily and `IdMapIndex` meeting an 8 at the first index — which is
    the cookbook's refuse-at-mount rule, and the docstring already said what the
    type now says."""
    calibrate: bool = True
    """Whether to fit turbovec's TQ+ calibration when a first batch is big enough
    to sample fairly. See `_store` — it is committed in one situation on purpose."""
    max_bytes: int = 2 * 1024 * 1024
    """The largest document indexed. One past it is skipped and reported, not an
    error: a call over a directory should not fail because it found a minified
    bundle, and a caller that never learns which files were skipped cannot tell a
    quiet corpus from a quiet failure."""
    max_files: int = 5_000
    """How many files one `text_index` call will consider.

    Config rather than a constant hidden in the walk, which is where it was —
    and where it had already drifted from the sibling package's `max_files`, so
    two rows doing the same walk disagreed about the bound for no reason anybody
    chose."""


class LocalConfig(WireModel):
    """Row config for `text-index-local`."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """The `sentence-transformers` model. Downloaded on first use, then cached.

    A small, symmetric, 384-dimensional default: it needs no prefixes, it is
    about 90 MB, and it runs on a CPU at a speed that makes indexing a
    documentation tree a coffee break rather than an afternoon. A deployment
    wanting recall over speed names a bigger one — and pays for it twice, in
    load time and in a fresh index, because vectors do not transfer between
    models.

    **Switching is safe to try.** The index directory is keyed by a digest of
    the embedder's identity — the model name *and* both prefixes — so a new
    model gets its own index and switching back finds the old one intact. The
    README has a worked `nomic-embed-text-v1.5` configuration."""
    query_prefix: str = ""
    """What an asymmetric model wants in front of a question.

    `e5` wants `query: `; `nomic-embed-text` wants `search_query: `. Part of the
    embedder's identity, because an index built with a prefix and searched
    without one is a different space and the failure is quiet — slightly wrong
    neighbours rather than an error."""
    document_prefix: str = ""
    """What it wants in front of a passage (`e5`: `passage: `, nomic:
    `search_document: `)."""
    trust_remote_code: bool = False
    """Whether the model may bring its own modelling code — see the embedder.

    **A trust decision, not a compatibility flag**: `true` executes Python
    downloaded from the model's repository in this process. Required by
    `nomic-embed-text-v1.5` and every other model whose `config.json` carries an
    `auto_map`, and off by default so that granting it is something a profile
    says rather than something a load failure suggests."""
    batch_size: int = 32
    """How many passages are embedded per forward pass."""
    cache: str = ""
    """Where the weights land. Defaults to `$PH_CACHE/models`.

    Under a pH root on purpose: left to `sentence-transformers` the weights go
    to `$HF_HOME` or `~/.cache/huggingface`, which no `ph doctor` mentions and
    no `rm -rf $PH_CACHE` reclaims. Rebuildable and large is the lifecycle
    `$PH_CACHE` names (Q1), and the runtime venv is there for the same reason."""
    preload: bool = False
    """Load the model at **mount** rather than on the first call.

    Off by default because a person at a TUI would rather the harness start now
    and pay for the model when they use it. On for an **unattended** run — a
    daemon, a scheduled tick, `ph -p` in CI — where the alternative is finding
    out mid-turn that the weights cannot be fetched, and where nobody is present
    to type `/text-index install`.

    When it fails it refuses the mount (`MountRefusal`), which is the cookbook's
    own rule: refuse at mount, not at first use, because by then the agent is
    running and "refuse to start" has already been disobeyed."""


def _ready(provider: Embedder) -> bool:
    """Whether `provider` has its model loaded, if it is the kind that has one.

    `isinstance` against `LocalWeights` rather than a `hasattr` probe — see that
    Protocol for the reason, which is this tree's own rule three times over. An
    endpoint-backed embedder has nothing to load and is therefore always ready.
    """
    return provider.ready() if isinstance(provider, LocalWeights) else True


async def provision(provider: Embedder) -> tuple[bool, str]:
    """Load the model now. `(ok, sentence)` — never raises.

    A person triggered this and is waiting for words, so a failure is a sentence
    rather than a traceback — and the *upstream* sentence is kept verbatim,
    because the one failure this exists to catch says exactly what to do:
    "requires the following packages that were not found in your environment:
    einops". Rewording that would be replacing the useful half.
    """
    if not isinstance(provider, LocalWeights):
        return True, f"{provider.name} needs no download."
    try:
        dimension = await anyio.to_thread.run_sync(provider.load)
    except Exception as error:
        log.warning("ph_text_index: %s did not load", provider.name, exc_info=True)
        return False, f"{provider.name} did not load: {error}"
    return True, f"{provider.name} is ready ({dimension}-dimensional)."


# --------------------------------------------------------------------- seam ----


TEXT_INDEX: ServiceKey[TextIndexSeam] = ServiceKey("text_index")
"""The text index, for `/text-index` and the tools it backs."""


@dataclass(slots=True)
class TextIndexSeam:
    """The service published as `ctx.text_index`.

    Holds the one index and the one embedder, and serialises every mutation
    behind a lock. The lock is not defensive tidiness: `text_index` is not
    concurrency-safe precisely *because* of it — two calls that both loaded, both
    mutated and both saved would leave the sidecar describing one of them.
    """

    ctx: Context
    config: Config
    provider: Embedder | None = None
    provider_by: Running | None = None
    on_provider: Callable[[Context], None] | None = None
    """What to do once something can produce a vector — registering the tools.

    A callback rather than an `inject` on some key, because there is no key: an
    embedder is claimed *into* this seam, so nothing appears in the service tree
    for a waiter to wait on. Held here so the two facts stay one decision — the
    tools exist exactly while a provider does, and both leave on the same scope.
    """
    _index: TextIndex | None = None
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    def register(self, provider: Embedder, *, scope: Context | None = None) -> Disposer:
        """Claim the embedder slot, and offer the tools it makes possible.

        The tools are registered on the *provider's* scope, so unmounting the
        embedder row takes them with it. A profile that mounts this seam and no
        provider therefore advertises nothing at all, which is the point: see the
        module docstring.
        """
        disposer = claim_slot(
            self.ctx.running_for(scope), self, "provider", provider, label="text_index_embedder"
        )
        if self.on_provider is not None:
            self.on_provider(scope if scope is not None else self.ctx)
        return disposer

    @property
    def embedder(self) -> Embedder:
        if self.provider is None:
            raise HarnessError(
                "no embedder is registered on ctx.text_index; mount `text-index-local` "
                "or a row of your own",
                TEXT_INDEX_FAILED,
            )
        return self.provider

    def root(self) -> Path:
        """Where this embedder's index lives.

        **Keyed by the embedder**, so switching models does not walk into the
        `IndexMismatch` it would otherwise cause — the two indexes simply do not
        collide, and switching back finds the old one intact. A digest rather
        than the name because a model id has slashes in it."""
        digest = hashlib.sha256(self.embedder.name.encode("utf-8")).hexdigest()[:16]
        return default_cache_path(self.config.path, "text-index", digest)

    async def index(self) -> TextIndex:
        """The open index, loaded once."""
        if self._index is None:
            store = TextIndex(
                root=self.root(),
                model=self.embedder.name,
                bit_width=self.config.bit_width,
                calibrate=self.config.calibrate,
            )
            await anyio.to_thread.run_sync(store.open)
            self._index = store
        return self._index

    async def embed(self, texts: list[str], *, query: bool) -> Any:
        embedder = self.embedder
        return await anyio.to_thread.run_sync(
            lambda: embedder.encode(texts, query=query), abandon_on_cancel=True
        )

    def locked(self) -> anyio.Lock:
        return self._lock

    def report(self) -> list[tuple[str, str]]:
        """`ph doctor`'s section: what is indexed, and by what."""
        if self.provider is None:
            return [("embedder", "none registered — text_index is not offered")]
        rows = [
            ("embedder", self.provider.name),
            ("model", "loaded" if _ready(self.provider) else "not loaded — /text-index install"),
            (
                "weights",
                self.provider.cache_folder
                if isinstance(self.provider, LocalWeights)
                else "the library default",
            ),
            ("path", str(self.root())),
        ]
        if self._index is None:
            rows.append(("state", "not opened yet"))
            return rows
        stats = self._index.stats()
        rows.extend(
            [
                ("documents", str(stats["documents"])),
                ("chunks", str(stats["chunks"])),
                ("vectors", f"{stats['dim']} dim at {stats['bit_width']} bit"),
                ("calibration", str(stats["calibration"])),
            ]
        )
        return rows


# ------------------------------------------------------------------ schemas ----


class IndexArgs(ToolModel):
    paths: list[str] = Field(
        description="Files or directories to index. Relative to the workspace root."
    )
    glob: str | None = Field(
        None, description="Which files to take from a directory. Defaults to the row's setting."
    )
    forget: bool = Field(False, description="Remove these documents from the index instead.")


class SkippedValue(ToolModel):
    path: str
    reason: str


class IndexValue(ToolModel):
    documents: list[str]
    chunks_added: int
    chunks_removed: int
    unchanged: int = 0
    """Documents the version control proved unchanged, so never re-embedded."""
    skipped: list[SkippedValue]
    total_documents: int
    total_chunks: int


class SearchArgs(ToolModel):
    query: str = Field(description="What you are looking for, in your own words.")
    k: int = Field(5, ge=1, le=50, description="How many passages to return.")
    paths: list[str] | None = Field(
        None, description="Restrict the search to these files or directories."
    )


class HitValue(ToolModel):
    path: str
    start_line: int
    end_line: int
    score: float
    text: str


class SearchValue(ToolModel):
    query: str
    hits: list[HitValue]
    searched: int
    """Chunks the query was scored against — the whole index, or the filtered subset."""


# ------------------------------------------------------------------- render ----


def _render_index(args: Any, value: Any) -> list[ContentBlock]:
    verb = "Removed" if args.get("forget") else "Indexed"
    lines = [
        f"{verb} {count_of(len(value['documents']), 'document')}: "
        f"{count_of(value['chunks_added'], 'passage')} added, "
        f"{value['chunks_removed']} removed"
        # Said only when it happened, and said as "not re-embedded" because that
        # is the cost avoided — the read is the cheap half.
        + (
            f"; {value['unchanged']} unchanged and not re-embedded."
            if value.get("unchanged")
            else "."
        )
    ]
    for path in value["documents"][:40]:
        lines.append(f"  {path}")
    if len(value["documents"]) > 40:
        lines.append(f"  [and {len(value['documents']) - 40} more]")
    for skipped in value["skipped"]:
        lines.append(f"  skipped {skipped['path']}: {skipped['reason']}")
    lines.append(
        f"The index now holds {count_of(value['total_chunks'], 'passage')} "
        f"from {count_of(value['total_documents'], 'document')}."
    )
    return text_content("\n".join(lines))


def _render_search(args: Any, value: Any) -> list[ContentBlock]:
    if not value["hits"]:
        return text_content(
            f"Nothing matched {value['query']!r} among "
            f"{count_of(value['searched'], 'indexed passage')}. "
            "Index the documents first if this corpus is empty."
        )
    lines = [f"{count_of(len(value['hits']), 'passage')} for {value['query']!r}:"]
    for hit in value["hits"]:
        lines.append(
            f"\n{hit['path']}:{hit['start_line']}-{hit['end_line']}  (score {hit['score']:.3f})"
        )
        lines.append(hit["text"])
    return text_content("\n".join(lines))


# --------------------------------------------------------------------- body ----


def offer_skills(ctx: Context) -> None:
    """Install this package's `SKILL.md` files, if a skills registry is mounted.

    `discover_skills` already globs `<root>/*/SKILL.md`, validates each one and
    skips a malformed one with a logged reason — the whole of what a
    hand-written loader was doing here in 28 lines, duplicated byte-for-byte
    into the sibling package.

    **Called from the provider's claim, not from this row's `apply`.** The tools
    appear only once an embedder exists, and the skill has to follow them: with
    the seam mounted and no provider — a configuration this package explicitly
    supports — the catalog line for `text-search` rode every prompt while
    `text_index`/`text_search` were absent, which is the exact failure the
    module docstring says the tools avoid.

    Only the one-line description rides the prompt; the body stays on disk until
    the model asks for it by name (G9).
    """
    root = Path(__file__).parent / "skills"
    for skill in discover_skills([str(root)], source="ph-text-index"):
        contribute_item(ctx, SKILLS, skill, label=f"skill({skill.name})")


@plugin("text-index", inject=[TOOLS, FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam, and register the tools once an embedder exists."""
    import importlib.util

    if importlib.util.find_spec("turbovec") is None:
        raise MountRefusal(MISSING_TURBOVEC)

    seam = TextIndexSeam(ctx=ctx, config=config)
    ctx.provide(TEXT_INDEX, seam)

    async def index_tool(args: IndexArgs, run: ToolRunContext) -> Any:
        store = await seam.index()
        documents = await ctx.require(FS).collect(
            args.paths,
            args.glob or config.glob,
            scope=run.scope,
            limit=config.max_files,
            agent=run.agent,
        )
        added = removed = unchanged = 0
        indexed: list[str] = []
        skipped: list[dict[str, str]] = []
        # **Ask the version control before reading or embedding anything.** The
        # saving here is larger than for a code index: this loop has no digest
        # short-circuit of its own, so an unchanged document was re-read, re-cut
        # and — the expensive part — **re-embedded** on every call. Measured on
        # `docs/`: 20 s of MiniLM for 539 passages, or 185 s under nomic.
        #
        # Which backend answers is the workspace provider's to say
        # (`ph.seams.changes`), and a tree with none answers an empty state, so
        # the loop below is correct either way — it just does the work again.
        state = TreeState()
        if not args.forget:
            state = await tree_state(ctx, ctx.require(FS).root_for(run.agent), since=store.token)

        async with seam.locked():
            for path in documents:
                run.raise_if_cancelled()
                if store.holds(path) and state.vouches_for(path, store.vcs_id(path)):
                    unchanged += 1
                    continue
                removed += await anyio.to_thread.run_sync(store.forget, path)
                if args.forget:
                    indexed.append(path)
                    continue
                chunks, reason = await _passages(ctx, seam, path, run)
                if reason is not None:
                    skipped.append({"path": path, "reason": reason})
                    continue
                vectors = await seam.embed([chunk.text for chunk in chunks], query=False)
                added += await anyio.to_thread.run_sync(
                    store.add, path, chunks, vectors, state.id_for(path)
                )
                indexed.append(path)
            # After the writes: a token stored first would, on a crash between
            # the two, vouch for documents this run never indexed.
            moved = bool(state.token) and state.token != store.token
            if state.token:
                store.remember(state.token)
            # Nothing written, nothing to persist. `save` serialises every
            # record — 7.1 MB and 16 ms at 6 000 chunks — plus a turbovec fsync,
            # and a call that proved the whole corpus unchanged is the case this
            # filter exists to make free.
            if added or removed or moved:
                await anyio.to_thread.run_sync(store.save)
        stats = store.stats()
        return {
            "documents": indexed,
            "chunks_added": added,
            "chunks_removed": removed,
            "unchanged": unchanged,
            "skipped": skipped,
            "total_documents": stats["documents"],
            "total_chunks": stats["chunks"],
        }

    async def search_tool(args: SearchArgs, run: ToolRunContext) -> Any:
        store = await seam.index()
        vector = await seam.embed([args.query], query=True)
        run.raise_if_cancelled()
        # The id set is built **once** and handed to `search`, which used to
        # rebuild it internally while this computed it for the count — two
        # O(chunks) passes per filtered query (7.5 ms against a 0.22 ms
        # unfiltered search on a 10 000-chunk index).
        allowed: list[int] | None = None
        if args.paths is not None:
            allowed = await anyio.to_thread.run_sync(store.ids_under, args.paths)
            searched = len(allowed)
        else:
            searched = int(store.stats()["chunks"])
        hits: list[tuple[float, Record]] = await anyio.to_thread.run_sync(
            lambda: store.search(vector, args.k, allowed=allowed)
        )
        return {
            "query": args.query,
            "searched": searched,
            "hits": [
                HitValue(
                    path=record.path,
                    start_line=record.start_line,
                    end_line=record.end_line,
                    score=score,
                    text=record.text,
                ).model_dump()
                for score, record in hits
            ],
        }

    def register(scope: Context) -> None:
        """Offer the tools, now that something can produce a vector."""
        scope.require(TOOLS).register(
            define_tool(
                "text_index",
                INDEX_DESCRIPTION,
                parameters=IndexArgs,
                output=ToolOutput(schema=IndexValue, render=_render_index),
                execute=index_tool,
                # Two of these in one batch would both load, both mutate and both
                # save; see `TextIndexSeam`. The lock makes the outcome correct
                # rather than racy, and this makes the scheduler not queue them
                # against each other in the first place (B6).
                is_concurrency_safe=False,
                self_limits=True,
                # The passages are a payload this call delivered to the index, not
                # something the model refers back to.
                arguments_disposable=True,
                **simple_views("search", "Index text", "paths"),
            ),
            scope=scope,
        )
        # The skill goes on the same scope as the tools it describes.
        offer_skills(scope)
        scope.require(TOOLS).register(
            define_tool(
                "text_search",
                SEARCH_DESCRIPTION,
                parameters=SearchArgs,
                output=ToolOutput(schema=SearchValue, render=_render_search),
                execute=search_tool,
                is_concurrency_safe=True,
                self_limits=True,
                **simple_views("search", "Search text", "query"),
            ),
            scope=scope,
        )

    async def install(argument: str, command: Any) -> str:
        """`/text-index install` — fetch and load the model, now, on purpose.

        A **command** and not a tool, per the seam's own rule: this is a thing
        the *person* asks the harness to do, and routing it through a model turn
        would put the model in the log as having decided it. It costs no turn.
        """
        verb = argument.strip().lower() or "status"
        if verb not in ("install", "status"):
            return f"/text-index takes `install` or `status`, not {argument.strip()!r}."
        if seam.provider is None:
            return (
                "No embedder is registered, so there is nothing to install — "
                "mount `text-index-local` or a provider row of your own."
            )
        # Asked of the *provider*, not of this row's config: where weights land
        # is the embedder's business, and this seam takes any `Embedder` —
        # including one backed by an endpoint, which has no weights and answers
        # the empty string. Duck-typed for the reason `_ready` is.
        weights = (
            seam.provider.cache_folder
            if isinstance(seam.provider, LocalWeights)
            else "the library default"
        )
        if verb == "status":
            state = "loaded" if _ready(seam.provider) else "not loaded"
            return f"{seam.provider.name}: {state}. Weights under {weights}."
        ok, sentence = await provision(seam.provider)
        return sentence if ok else f"{sentence}\n(weights would go under {weights})"

    command = CommandDefinition(
        name="text-index",
        summary="Download and load the embedding model, or report whether it is ready.",
        argument_hint="[install|status]",
        run=install,
    )
    contribute_item(
        ctx,
        COMMANDS,
        command,
        label="text-index command",
    )

    # Set after both bodies are defined, and read by `TextIndexSeam.register`:
    # the embedder row may sit either side of this one in the profile, so the
    # tools cannot be conditional on what has been provided *so far*. A provider
    # claimed later still finds this, because it is the claim that calls it.
    seam.on_provider = register

    contribute(
        ctx,
        Diagnostic(id="text-index", title="Text index", order=61, read=seam.report),
    )


async def _passages(
    ctx: Context, seam: TextIndexSeam, path: str, run: ToolRunContext
) -> tuple[list[Chunk], str | None]:
    """One document's passages, or the reason it was skipped."""
    refused = ctx.require(FS).skip_reason(path, max_bytes=seam.config.max_bytes, agent=run.agent)
    if refused:
        return [], refused
    slice_ = await ctx.require(FS).read(
        path,
        # The whole file, which is what `limit=None` means here — a passage index
        # over a truncated document would answer confidently about the first two
        # thousand lines and silently about the rest.
        limit=None,
        scope=run.scope,
        agent=run.agent,
        session=run.session,
    )
    chunks = chunk_text(
        slice_.text,
        max_chars=seam.config.max_chars,
        overlap_chars=seam.config.overlap_chars,
    )
    return chunks, None if chunks else "is empty"


@plugin("text-index-local", inject=[TEXT_INDEX], config=LocalConfig)
async def local(ctx: Context, config: LocalConfig) -> None:
    """Register a local `sentence-transformers` model as the embedder."""
    import importlib.util

    # The package, not the weights: see `SentenceTransformerEmbedder`. Refusing
    # here is the difference between "this deployment cannot embed" — which an
    # operator can fix — and a tool that fails on its first call in a session
    # that had already committed to using it.
    if importlib.util.find_spec("sentence_transformers") is None:
        raise MountRefusal(MISSING_MODEL)

    embedder = SentenceTransformerEmbedder(
        model_name=config.model,
        query_prefix=config.query_prefix,
        document_prefix=config.document_prefix,
        batch_size=config.batch_size,
        # `$PH_CACHE/models` unless the operator said otherwise. See the field.
        cache_folder=config.cache or str(resolve_roots().cache / "models"),
        trust_remote_code=config.trust_remote_code,
    )

    if config.preload:
        # At mount, and refusing the mount if it fails — the cookbook's rule,
        # and the reason this option exists: an unattended run has nobody to
        # type `/text-index install`, so the alternative is discovering
        # mid-turn that the weights cannot be fetched or the model cannot
        # import. `MountRefusal` is what every command that mounts a profile
        # turns into a sentence and an exit code.
        ok, sentence = await provision(embedder)
        if not ok:
            raise MountRefusal(f"text-index-local was asked to preload and could not: {sentence}")
        log.info("ph_text_index: %s", sentence)

    contribute_item(ctx, TEXT_INDEX, embedder, label=f"embedder({config.model})")
