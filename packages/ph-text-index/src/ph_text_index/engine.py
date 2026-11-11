"""`text-index` — semantic retrieval over the agent's own documents.

Two rows and two tools. `text_index` reads documents through `ctx.fs`, cuts them
into passages that remember their line numbers, embeds them and puts them in a
turbovec index on disk. `text_search` answers a question with the passages
themselves plus the `path:start-end` each came from — so a hit is both an answer
and a pointer the agent can `read` for the rest.

## Where these rows are mounted

In the `ph-text-index` command's own process, and not in a session's. A session
mounts the `text-index` and `text-index-local` rows (`ph_text_index.row`), which
register this package's UTCP manual; the manual's commands run `ph-text-index`,
and that command mounts *these* rows and runs a tool through the registry
(`ph.toolcli`). So `ctx.fs` below is the command's own — rooted at the working
directory the calling agent's sandbox gave it — and `run.agent` is always `None`.

## The seam and the embedder

`text-index-engine` publishes the seam and the tools; one embedder row claims
it — `text-index-local-engine`, `-endpoint-engine` or `-adapter-engine`. The
command always mounts exactly one (`ph.toolcli.ToolCli.providers`), and one that
cannot mount refuses the command's mount, so here there is no seam without an
embedder. The rule that a profile with no embedder offers no tools is the
session's, and lives with the rows a session mounts (`ph_text_index.row`).

## Reading through `ctx.fs`

Every byte indexed arrives through `ctx.fs.read`: the walk's ignore rules, the
size bound and the skip reasons are that seam's, said once. In a session the
command's `ctx.fs` is not the session's — `permissions-fs` governs calls through
a session's `ctx.fs`, and a command is not one — so what bounds these reads
there is the calling agent's sandbox, as it bounds `bash`.

## One call, or a host

Run for one call, `text_index` finishes before it answers. Mounted in a host
(`ph-text-index serve`, `ph.toolcli.TOOL_HOST`) the process outlives the call, so
the tool answers at once — `state: indexing`, with how far it has got — and the
run goes on in the background; `text_search` answers meanwhile from what is
indexed so far, and says the index is still being built. Either way the run
reports to `progress.json` beside the index, keeps a digest of each document so
an unchanged one is never embedded twice, and saves as it goes, so a run that is
stopped keeps what it embedded.

## A passage fits its embedder

A model reads so many tokens and no more. Past that a local model drops the rest
without a word — the tail of the passage is in the index and can never be found
— and a server refuses the request. So passages are packed by what the embedder
counts as well as by `max_chars` (`_chunk.Weight`, `_embed.TokenBounded`): its
own tokens where it has a tokenizer, the text's bytes at the corpus's density
where it has not. Each passage is as long as both allow, so a table is cut small
and the prose around it is not, whatever script either is in. What is sent is
checked, and a passage refused as too long all the same is cut smaller and sent
again rather than failing the run; the result says when any of this happened.
A question is held to the same limit: one longer than the embedder reads is
searched by its head, and the answer says so.

## Whose index

Where the caller said (`ph.toolcli.state_dir`): a session names a directory of
the calling agent's scratch, so each agent has its own index of its own tree,
writable in a read-only session and found again after a restart. With no caller
saying, one per workspace under `$PH_CACHE/text-index`. Either way a directory
per embedder beneath it, and a change is made under a file lock in it
(`TextIndexSeam.writing`), because two processes can share one.

@module ph_text_index.engine
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.cordis import Context, Disposer, MountRefusal, Running, ServiceKey, plugin
from ph.indexable import Paragraph, triage
from ph.json import JsonObject, as_bool
from ph.keys import COMMANDS, FS, TOOLS
from ph.llm.adapter import too_long_facts
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED, ContentBlock
from ph.locks import try_file_lock, wait_for_file_lock, waited_file_lock
from ph.seams._registry import claim_slot, contribute_item
from ph.seams.changes import TreeState, tree_state
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.fs import FsService
from ph.text import count_of, one_line
from ph.toolcli import (
    AGENT_ENV,
    TOOL_ENV,
    TOOL_HOST,
    BackgroundRuns,
    Progress,
    RunProgress,
    claim_home,
    state_home,
)
from ph.tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ph.tools.errors import HarnessError
from ph.tools.presentation import simple_views
from ph.workspace_caches import digest

from ._chunk import Chunk, Weight, chunk_paragraphs, head_of
from ._config import (
    ADAPTER_ROW,
    EMBEDDER_ENV,
    ENDPOINT_ROW,
    LOCAL_ROW,
    MISSING_MODEL,
    ROW,
    AdapterConfig,
    Config,
    EndpointConfig,
    Limit,
    LocalConfig,
    index_base,
    weights_cache,
)
from ._embed import (
    BYTES_PER_TOKEN,
    Embedder,
    EndpointEmbedder,
    LocalWeights,
    SentenceTransformerEmbedder,
    SessionEmbedder,
    TokenBounded,
    Vectors,
    weigh_by_bytes,
)
from ._store import Record, TextIndex

__all__ = [
    "TEXT_INDEX",
    "Config",
    "Embedder",
    "LocalConfig",
    "TextIndexSeam",
    "adapter",
    "apply",
    "endpoint",
    "local",
    "provision",
]

log = logging.getLogger("ph_text_index")

TEXT_INDEX_FAILED = "TEXT_INDEX_FAILED"
TEXT_INDEX_MODEL_MISSING = "TEXT_INDEX_MODEL_MISSING"
TEXT_INDEX_MODEL_FAILED = "TEXT_INDEX_MODEL_FAILED"

MIN_PASSAGE_TOKENS = 16
"""The fewest tokens of its own text a passage is held to in order to fit an
embedder. Under this a "passage" is a phrase, and a document the embedder still
refuses is skipped and said."""

BRIEFLY = 8
"""How many of a list a sentence names before it says how many more there are."""

ECHOED_CHARS = 200
"""How much of a question a search result repeats back."""

MISSING_TURBOVEC = (
    "the text-index row needs `turbovec`, which ph-text-index depends on; "
    "install it (`uv sync`, or `pip install turbovec`) or remove the row"
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


def _model(provider: Embedder) -> str:
    """Where `provider`'s model stands, if it is the kind that has one.

    `isinstance` against `LocalWeights` rather than a `hasattr` probe — see that
    Protocol for the reason, which is this tree's own rule three times over. An
    endpoint-backed embedder has nothing to load and is therefore always ready.

    Three answers and not two, because this row runs in a command that lives for
    one call: "not loaded" is true of every such process, and what a person
    asking needs to know is whether the next call will have to download.
    """
    if not isinstance(provider, LocalWeights):
        return "nothing to load"
    if provider.ready():
        return "loaded"
    return "downloaded, loads on first use" if provider.cached() else "not downloaded"


def _weights(provider: Embedder) -> str:
    """Where `provider`'s weights land, as a person is told it. Asked of the
    Protocol, for the reason `_model` is: an endpoint has none."""
    return provider.cache_folder if isinstance(provider, LocalWeights) else "the library default"


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


WRITER_LOCK = "writer.lock"
"""The lock file in each index directory that orders its writers (`writing`)."""

_LOCKED = "the text index"


@dataclass(slots=True)
class TextIndexSeam:
    """The service published as `ctx.text_index`.

    Holds the one embedder, and a copy of each workspace's index it has opened.
    Every change goes through `writing`. Its lock is not defensive tidiness: two
    calls that both loaded, both mutated and both saved would leave the sidecar
    describing one of them, whether the two are in this seam or in another
    sharing the directory.
    """

    ctx: Context
    config: Config
    provider: Embedder | None = None
    provider_by: Running | None = None
    _indexes: dict[Path, TextIndex] = field(default_factory=dict)
    """Per index directory, the copy this seam last loaded or wrote."""
    _guards: dict[Path, anyio.Lock] = field(default_factory=dict)
    """Per index directory, what keeps one use of the copy from another's.

    In a host a run and a search share one copy: the run adds a document's
    passages in a worker thread while a search scores against the same index in
    another. Held around each such step and never across an embedding, so a
    search waits for one document's bookkeeping and not for a model."""
    runs: BackgroundRuns = field(default_factory=BackgroundRuns)
    """The indexing runs, by index home, and how far each has got."""
    density: float = BYTES_PER_TOKEN
    """How many bytes of this corpus make a token, as far as is known: the
    careful guess, lowered by any refusal that said how many tokens its input
    was (`calibrate`)."""
    known: Limit | None = None
    """The embedder's limit, once known: asked of it, or learned from a refusal.
    Read as it is by whoever must not ask — a run reporting the limit it ended
    with."""
    _limit_asked: bool = False
    """Whether the embedder has been asked. Asked late — of a local model that
    means loading it — and so only by a run that has something to embed."""

    def register(self, provider: Embedder, *, scope: Context | None = None) -> Disposer:
        """Claim the embedder slot. One embedder: one index is one vector space."""
        return claim_slot(
            self.ctx.running_for(scope), self, "provider", provider, label="text_index_embedder"
        )

    @property
    def embedder(self) -> Embedder:
        if self.provider is None:
            raise HarnessError(
                "no embedder is registered on ctx.text_index; mount "
                "`text-index-local-engine` or a row of your own",
                TEXT_INDEX_FAILED,
            )
        return self.provider

    def root(self, workspace: Path) -> Path:
        """Where this embedder's index of one workspace lives.

        **Keyed by the embedder**, so switching models does not walk into the
        `IndexMismatch` it would otherwise cause — the two indexes simply do not
        collide, and switching back finds the old one intact.

        **And never shared between two trees**, as `CodeGraphSeam.store_for`
        keeps its own apart, because what an index holds belongs to one tree: its
        documents are named relative to it, one version-control token says how
        far it was indexed, and a run's sweep drops what that tree no longer has.
        Two workspaces in one directory took turns deleting each other's
        documents and re-embedding their own. So it is where the caller said
        (`ph.toolcli.state_dir`) — in a session, a directory of the calling
        agent's own scratch — or, with no caller saying, a directory per
        workspace under `$PH_CACHE/text-index`, which `writing` sweeps once its
        workspace is gone.

        Digests rather than names, because a model id and a path have slashes in
        them."""
        return self.home(workspace) / digest(self.embedder.name)

    async def index(self, workspace: Path) -> TextIndex:
        """One workspace's index, for a search.

        Loaded again when its files changed since this copy read or wrote them,
        but only if the writer lock is free this instant. A writer holding it is
        part-way through a change, and its files may be part-way through a save,
        so the search answers from the copy there is. Only a seam with no copy
        at all waits for the lock."""
        root = self.root(workspace)
        store = self._indexes.get(root)
        if store is not None and not store.stale():
            return store
        release = try_file_lock(root / WRITER_LOCK, what=_LOCKED)
        if release is None:
            if store is not None:
                return store
            release = await wait_for_file_lock(root / WRITER_LOCK, what=_LOCKED)
        try:
            return await self._fresh(root)
        finally:
            release()

    def home(self, workspace: Path) -> Path:
        """The directory this workspace's indexes are under, one per embedder
        (`ph.toolcli.state_home`). Where a run's `progress.json` goes, since a
        home has one run at a time."""
        return state_home(index_base(), workspace)

    def _store(self, root: Path) -> TextIndex:
        """The index at `root`, not yet opened."""
        return TextIndex(
            root=root,
            model=self.embedder.name,
            bit_width=self.config.bit_width,
            calibrate=self.config.calibrate,
        )

    def guard(self, workspace: Path) -> anyio.Lock:
        """The lock around each use of this workspace's copy. See `_guards`."""
        return self._guards.setdefault(self.root(workspace), anyio.Lock())

    def indexing(self, workspace: Path) -> Progress | None:
        """The run building this workspace's index right now, or `None`
        (`BackgroundRuns.working`): a run holds the writer lock while it works."""
        return self.runs.working(
            self.home(workspace), self.root(workspace) / WRITER_LOCK, what=_LOCKED
        )

    @asynccontextmanager
    async def writing(self, workspace: Path) -> AsyncIterator[TextIndex]:
        """One workspace's index, held for one change against every other writer.

        The lock is a file in the index directory, so it orders every writer of
        it: another root in the same daemon, a `phern -p`, a second daemon, and
        this seam's own calls too, since `flock` refuses a second holder even in
        one process. Each writer used to hold its own copy and save it whole, so
        the sidecar described whichever saved last, the other's vectors lost
        their meaning, and both allocated the same ids. Under the lock, a copy
        another writer has changed is loaded again, so every change starts from
        what is on disk.

        Waited for, because indexing is slow and a second writer is expected to
        queue behind the first.
        """
        # Before the lock: the record and the sweep touch nothing a writer holds.
        await anyio.to_thread.run_sync(claim_home, index_base(), workspace)
        root = self.root(workspace)
        async with waited_file_lock(root / WRITER_LOCK, what=_LOCKED):
            yield await self._fresh(root)

    async def _fresh(self, root: Path) -> TextIndex:
        """The copy of the index at `root`, loaded again if its files changed.
        Called holding the writer lock, so no writer is part-way through them."""
        store = self._indexes.get(root)
        if store is None or store.stale():
            store = self._store(root)
            await anyio.to_thread.run_sync(store.open)
            self._indexes[root] = store
        return store

    def configured_limit(self) -> int | None:
        """The limit a profile set on the embedder, if it set one. No model is
        loaded and no server asked to learn it."""
        embedder = self.embedder
        return embedder.max_tokens if isinstance(embedder, TokenBounded) else None

    async def limit(self) -> Limit | None:
        """How long an input the embedder reads, or `None` when nothing says."""
        embedder = self.embedder
        if self._limit_asked or not isinstance(embedder, TokenBounded):
            return self.known
        try:
            found = await anyio.to_thread.run_sync(embedder.limit)
        except Exception:
            # Not kept: a model that would not load says so, in its own words, on
            # the embedding that follows.
            log.debug("ph_text_index: the embedder's limit was not learned", exc_info=True)
            return self.known
        self._limit_asked = True
        if found is not None:
            self.learn(found)
        return self.known

    def learn(self, limit: Limit) -> None:
        """Take `limit` as the limit — what the embedder says when asked, or
        what it said in refusing an input as too long.

        What an embedder reports when asked is not always what it takes. A
        llama.cpp reports its window, and refuses an embedding longer than its
        *batch*, which `/props` does not mention — measured: a window of 2,048
        and a refusal past 512. Without this every document after the first
        would be packed to the reported size and refused; with it the rest of
        the run is packed to what the refusal said. Only ever downward, and only
        from a refusal that names a number: a guess from the size that happened
        to be taken would shrink every later document to fit the densest one."""
        if self.known is None or limit.tokens < self.known.tokens:
            self.known = limit

    def calibrate(self, size: int, tokens: int) -> None:
        """Note that `size` bytes of this corpus were `tokens` tokens, as a
        refusal said. Technical prose is denser than the guess — paths, code,
        tables — and one refusal that says by how much saves the next dense
        document from being packed too large and refused in its turn. Only ever
        toward denser, and never past a token a byte."""
        if tokens > 0:
            self.density = max(1.0, min(self.density, size / tokens))

    async def room(self, limit: Limit | None, *, query: bool = False) -> int | None:
        """How many tokens of its own text an input may be: `limit`, less what
        the embedder puts around it. `None` when nothing says how long an input
        it reads.

        **Less what is around it, and not the limit.** Every passage is sent
        behind a prefix (`search_document: `) and between the model's own start
        and end markers, and those are in what the embedder counts. Packed to the
        limit itself, a full passage measured a few tokens over it, and nearly
        every document was packed again and reported as needing smaller
        passages — on a live run, 14 of 30, where one was. A question has a
        prefix of its own, which for some models is a sentence."""
        if limit is None:
            return None
        # What an input costs before its text is the measure of an empty one.
        overhead = max(await self.measure([""], query=query), default=0)
        return max(MIN_PASSAGE_TOKENS, limit.tokens - overhead)

    async def measure(self, texts: list[str], *, query: bool = False) -> list[int]:
        """How many tokens each text is to the embedder as it is sent — counted
        where it has a tokenizer, estimated at this corpus's density where it has
        not; nothing for one that cannot say."""
        embedder = self.embedder
        if not isinstance(embedder, TokenBounded):
            return []
        return await anyio.to_thread.run_sync(
            partial(embedder.measure, texts, self.density, query=query)
        )

    def weigh(self, texts: Sequence[str]) -> Sequence[float]:
        """How many tokens each text adds to a passage it is part of
        (`_chunk.Weight.of`): the embedder's own count, or the text's bytes at
        this corpus's density. Bytes too for an embedder that cannot say how
        long an input it reads, so that a passage it refuses can still be cut
        by something. **Blocking**: a tokenizer's."""
        embedder = self.embedder
        if isinstance(embedder, TokenBounded):
            return embedder.weigh(texts, self.density)
        return weigh_by_bytes(texts, self.density)

    async def heaviest(self, texts: list[str]) -> float:
        """What the heaviest of `texts` weighs (`weigh`), off the event loop."""
        return max(await anyio.to_thread.run_sync(self.weigh, texts), default=0.0)

    async def embed(self, texts: list[str], *, query: bool) -> Vectors:
        embedder = self.embedder
        try:
            return await anyio.to_thread.run_sync(
                lambda: embedder.encode(texts, query=query), abandon_on_cancel=True
            )
        except Exception as error:
            if not isinstance(embedder, LocalWeights) or embedder.cached():
                raise
            # A model that is not on disk is fetched by the first call that needs
            # it, and a call made for a confined agent often cannot: its network
            # is the deployment's allowlist. The upstream sentence is kept, and
            # the way out is the one a person has.
            raise HarnessError(
                f"the embedding model is not downloaded, and fetching it failed: {error}. "
                f"Run `/{ROW} install` (or `ph-text-index install` from a shell) — a "
                "person's command, run outside the sandbox — and call again.",
                TEXT_INDEX_MODEL_MISSING,
            ) from error

    def doctor(self) -> list[tuple[str, str]]:
        """`phern doctor`'s section: what is indexed, and by what."""
        if self.provider is None:
            return [("embedder", "none registered — text_index is not offered")]
        root = self.root(self.ctx.require(FS).root_for(None))
        rows = [
            ("embedder", self.provider.name),
            ("model", _model(self.provider)),
            ("weights", _weights(self.provider)),
            ("path", str(root)),
        ]
        store = self._indexes.get(root)
        if store is None:
            if not root.is_dir():
                rows.append(("limit", self._limit_said(None)))
                rows.append(("state", "not built — run text_index"))
                return rows
            # A command asked for its status has opened nothing yet. Read without
            # the writer lock: these are counts for a person, and a writer
            # part-way through a save is a reason to say so, not to wait.
            store = self._store(root)
            try:
                store.open()
            except Exception as error:
                rows.append(("state", f"could not be read: {error}"))
                return rows
        stats = store.stats()
        rows.extend(
            [
                ("limit", self._limit_said(store.limit)),
                ("documents", str(stats["documents"])),
                ("chunks", str(stats["chunks"])),
                ("vectors", f"{stats['dim']} dim at {stats['bit_width']} bit"),
                ("calibration", str(stats["calibration"])),
            ]
        )
        return rows

    def _limit_said(self, stored: str | None) -> str:
        """How long an input the embedder reads, as far as can be said without
        asking it: what this process learned, else what the last run to embed
        into the index wrote beside it (`stored`), else the setting. Asking
        would load a local model to answer a question about status."""
        if self.known is not None:
            return self.known.said()
        if stored is not None:
            return f"{stored}, as the last run to embed found it"
        configured = self.configured_limit()
        if configured is not None:
            return Limit(configured, "configured").said()
        return "not known until a run embeds"


# ------------------------------------------------------------------ schemas ----


class IndexArgs(ToolModel):
    paths: list[str] | None = Field(
        None,
        description=(
            "Files or directories to index, relative to the workspace root. Defaults "
            "to the row's setting, which is the whole workspace unless a profile "
            "pointed it elsewhere."
        ),
    )
    glob: str | None = Field(
        None, description="Which files to take from a directory. Defaults to the row's setting."
    )
    exclude: list[str] | None = Field(
        None,
        description=(
            "Glob patterns for files to leave out, read against the workspace root "
            "(`**/drafts/**`). Added to the row's own exclusions."
        ),
    )
    forget: bool = Field(False, description="Remove these documents from the index instead.")


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
    """Documents this run has been through, and how many it will go through."""
    documents: list[str] = Field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    unchanged: int = 0
    """Documents proved unchanged — by the version control, or by their own
    digest — and so never re-embedded."""
    limit: str = ""
    """The embedder's limit, as a phrase, when it ended a passage of this run
    sooner than the row's `max_chars` would have; empty when it did not."""
    passage_tokens: int | None = None
    """How many tokens of its own text a passage was held to, when `limit` is
    said: the limit, less the prefix and the markers the embedder adds."""
    cut_smaller: list[str] = Field(default_factory=list)
    """Documents that were packed a second time, smaller, to fit: what was sent
    measured over the limit, or the embedder refused a passage as too long."""
    skipped: list[SkippedValue] = Field(default_factory=list)
    total_documents: int
    total_chunks: int


def _scoped(fs: FsService, paths: list[str] | None, run: ToolRunContext) -> list[str] | None:
    """The `paths` filter, in the spelling the records are stored under (X2).

    `index_tool` stores documents as `fs.collect` returns them, which is
    `fs.named()` — workspace-relative posix, because an absolute path in a
    stored record puts the machine and the run into the conversation. The filter
    was matched raw, exactly as the model typed it, so `["."]`, `["./docs"]` and
    any absolute path matched nothing at all and the reply said "among 0 indexed
    passages" — a wrong answer that reads like an empty index rather than like a
    filter that missed.

    **`.` and `""` mean the workspace, which is no filter.** `named(".")` is
    `"."`, and no document is stored under that name, so the obvious way to say
    "search everything" excluded everything. Dropping them leaves the unfiltered
    search, which is what the model meant and what the count then describes.
    """
    if paths is None:
        return None
    named = [fs.named(one, agent=run.agent) for one in paths]
    kept = [one for one in named if one not in ("", ".")]
    return kept or None


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


class IndexingValue(ToolModel):
    done: int
    total: int


class SearchValue(ToolModel):
    query: str
    hits: list[HitValue]
    searched: int
    """Chunks the query was scored against — the whole index, or the filtered subset."""
    indexing: IndexingValue | None = None
    """Set while a run is still building the index: the hits are from what it has
    indexed so far."""
    query_chars: int | None = None
    """How many characters of the question were searched, when it is longer than
    the embedder reads and only its head was; `None` when all of it was."""


# ------------------------------------------------------------------- render ----


def _render_index(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("state") == "indexing":
        return text_content(
            f"Indexing in the background: {value['done']} of "
            f"{count_of(value['total'], 'document')} so far. `text_search` answers now, "
            "from what is indexed already, and says while the index is still being "
            "built. You will be told when it finishes; call `text_index` again to see "
            "how far it has got.\n"
            f"The index holds {count_of(value['total_chunks'], 'passage')} "
            f"from {count_of(value['total_documents'], 'document')} so far."
        )
    return text_content("\n".join(_index_lines(as_bool(args.get("forget")), value)))


def _index_lines(forget: bool, value: Any, *, brief: bool = False) -> list[str]:  # noqa: ANN401
    """What a finished run did, in lines. `brief` for the note about a run nobody
    waited for (`Progress.note`): its documents are not listed, and what it
    skipped is counted past the first few."""
    verb = "Removed" if forget else "Indexed"
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
    if value.get("limit"):
        lines.append(
            f"Passages are held to {value['passage_tokens']} tokens of text, to fit "
            f"the embedder's limit of {value['limit']}."
        )
    if value.get("cut_smaller"):
        cut = value["cut_smaller"]
        lines.append(
            f"{count_of(len(cut), 'document')} needed smaller passages still to fit: "
            f"{', '.join(cut[:BRIEFLY])}{' …' if len(cut) > BRIEFLY else ''}"
        )
    if not brief:
        for path in value["documents"][:40]:
            lines.append(f"  {path}")
        if len(value["documents"]) > 40:
            lines.append(f"  [and {len(value['documents']) - 40} more]")
    skipped = value["skipped"]
    for one in skipped[:BRIEFLY] if brief else skipped:
        lines.append(f"  skipped {one['path']}: {one['reason']}")
    if brief and len(skipped) > BRIEFLY:
        lines.append(f"  [and {len(skipped) - BRIEFLY} more skipped]")
    lines.append(
        f"The index now holds {count_of(value['total_chunks'], 'passage')} "
        f"from {count_of(value['total_documents'], 'document')}."
    )
    return lines


def _unfinished(value: Any) -> str:  # noqa: ANN401
    """The warning a search carries while its index is still being built."""
    indexing = value.get("indexing")
    if not indexing:
        return ""
    return (
        f"Not finished indexing: {indexing['done']} of "
        f"{count_of(indexing['total'], 'document')}. These results cover what is "
        "indexed so far; search again once it finishes."
    )


def _cut_short(value: Any) -> str:  # noqa: ANN401
    """The warning a search carries when only the head of its question was
    searched."""
    searched = value.get("query_chars")
    if searched is None:
        return ""
    return (
        f"The question is longer than the embedder reads: only its first "
        f"{searched} characters were searched. Ask something shorter to search by all of it."
    )


def _echoed(query: str) -> str:
    """The question as a result repeats it, on one line: all of an ordinary one,
    and the start of one that is a page — the model has it already, in the call."""
    return repr(one_line(query, ECHOED_CHARS))


def _render_search(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    unfinished = _unfinished(value)
    cut_short = _cut_short(value)
    if not value["hits"]:
        return text_content(
            f"Nothing matched {_echoed(value['query'])} among "
            f"{count_of(value['searched'], 'indexed passage')}. "
            + (unfinished or "Index the documents first if this corpus is empty.")
            + (f" {cut_short}" if cut_short else "")
        )
    lines = [f"{count_of(len(value['hits']), 'passage')} for {_echoed(value['query'])}:"]
    if cut_short:
        lines.append(cut_short)
    if unfinished:
        lines.append(unfinished)
    for hit in value["hits"]:
        lines.append(
            f"\n{hit['path']}:{hit['start_line']}-{hit['end_line']}  (score {hit['score']:.3f})"
        )
        lines.append(hit["text"])
    return text_content("\n".join(lines))


# --------------------------------------------------------------------- body ----


@plugin("text-index-engine", affects="environment", inject=[TOOLS, FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam and register both tools."""
    if importlib.util.find_spec("turbovec") is None:
        raise MountRefusal(MISSING_TURBOVEC)

    seam = TextIndexSeam(ctx=ctx, config=config, runs=BackgroundRuns(grace=config.wait_s))
    ctx.provide(TEXT_INDEX, seam)

    def _forget_missing(store: TextIndex, run: ToolRunContext) -> int:
        """Drop every indexed document that is no longer on disk. One thread hop."""
        fs = ctx.require(FS)
        dropped = 0
        for path in store.documents():
            if not fs.resolve(path, agent=run.agent).exists():
                dropped += store.forget(path)
        return dropped

    async def index_run(
        args: IndexArgs, run: ToolRunContext, progress: RunProgress
    ) -> dict[str, Any]:
        """One indexing run, to its end. Its caller's cancellation is asked at
        each `progress.tick`, for as long as the run is that caller's own."""
        wanted = config.index.narrowed(paths=args.paths, glob=args.glob, exclude=args.exclude)
        documents = await ctx.require(FS).collect(
            wanted.paths,
            wanted.include,
            scope=run.scope,
            limit=config.max_files,
            agent=run.agent,
            exclude=wanted.exclude,
        )
        added = removed = unchanged = 0
        indexed: list[str] = []
        skipped: list[dict[str, str]] = []
        cut_smaller: list[str] = []
        held = False
        workspace = ctx.require(FS).root_for(run.agent)
        guard = seam.guard(workspace)
        saved = anyio.current_time()
        dirty = False

        async with seam.writing(workspace) as store:

            async def drop(path: str) -> None:
                """Take `path`'s passages out, under the guard a search shares."""
                nonlocal removed, dirty
                async with guard:
                    removed += await anyio.to_thread.run_sync(store.forget, path)
                dirty = True

            # Under the lock, so the report is this run's and not one waiting.
            progress.begin(len(documents))
            try:
                # **Ask the version control before reading anything.** Which
                # backend answers is the workspace provider's to say
                # (`ph.seams.changes`), and a tree with none answers an empty
                # state — then the digest below is what says "unchanged", at the
                # cost of the read. Measured on `docs/`: 20 s of MiniLM for 539
                # passages, or 185 s under nomic, is what either saves.
                #
                # Asked under the lock: the token it diffs from is the one the
                # last writer stored, whichever root or process that was.
                state = TreeState()
                if not args.forget:
                    state = await tree_state(ctx, workspace, since=store.token)
                for count, path in enumerate(documents):
                    progress.tick(count)
                    if dirty and anyio.current_time() - saved >= config.save_every_s:
                        async with guard:
                            await anyio.to_thread.run_sync(store.save)
                        saved, dirty = anyio.current_time(), False
                    if store.holds(path) and state.vouches_for(path, store.vcs_id(path)):
                        unchanged += 1
                        continue
                    if args.forget:
                        await drop(path)
                        indexed.append(path)
                        continue
                    text, reason, text_digest = await _read(ctx, seam, path, run)
                    if reason is None and store.holds(path) and store.digest(path) == text_digest:
                        # The same text as last time, said by the text itself: a
                        # tree with no version control, or a run that was
                        # stopped before it stored its token. Nothing is
                        # embedded, and the backend's id is noted so the next
                        # run need not read the document to learn this.
                        if state.id_for(path) != store.vcs_id(path):
                            store.vouch(path, state.id_for(path))
                            dirty = True
                        unchanged += 1
                        continue
                    paragraphs: Sequence[Paragraph] = ()
                    if reason is None:
                        # **Asked after the digest, because it is a question
                        # about the text** (X6): the same text gets the same
                        # answer, so an unchanged document is not asked.
                        # `skip_reason` was the cheap door and deliberately did
                        # not read; this is the one that can tell a minified
                        # bundle from a document. One split and one
                        # classification answer both questions — whether to
                        # index the document, and which of its blocks — and the
                        # packer is handed only what survived, so a README with
                        # an embedded image still indexes every paragraph around
                        # it.
                        kept = triage(text)
                        paragraphs = kept.kept
                        reason = kept.reason or (None if paragraphs else "is empty")
                    if reason is None:
                        # Here and not sooner, because it asks the embedder how
                        # long an input it reads: of a local model that means
                        # loading it, and a run that finds every document
                        # unchanged has no use for one. Outside the guard: this
                        # is the slow part, and a search in the same host must
                        # not wait for a model.
                        cut = await _embedded(seam, paragraphs)
                        if isinstance(cut, _Cut):
                            held = held or cut.held
                            if cut.again:
                                cut_smaller.append(path)
                            if seam.known is not None:
                                store.limit = seam.known.said()
                            async with guard:
                                removed += await anyio.to_thread.run_sync(store.forget, path)
                                added += await anyio.to_thread.run_sync(
                                    store.add,
                                    path,
                                    cut.chunks,
                                    cut.vectors,
                                    state.id_for(path),
                                    text_digest,
                                )
                            indexed.append(path)
                            dirty = True
                            continue
                        reason = cut
                    await drop(path)
                    skipped.append({"path": path, "reason": reason})
                if not args.forget:
                    # **Gone from disk is gone from the index** (X3). A deleted
                    # document kept its passages until somebody ran `forget` by
                    # hand, so search went on answering with text that is not
                    # there — worse here than for the code graph, because a
                    # passage is *quoted back to the model as evidence*.
                    #
                    # Asked of the filesystem rather than of the walk, for the
                    # reason the sibling sweep gives: `collect` stops at
                    # `max_files` and the policy screen prunes directories, so
                    # "absent from `documents`" is not "deleted" — it is also
                    # "past the cap" and "hidden by a rule".
                    #
                    # One thread hop for the whole sweep, not one per document.
                    async with guard:
                        dropped = await anyio.to_thread.run_sync(_forget_missing, store, run)
                    removed += dropped
                    dirty = dirty or bool(dropped)
                # After the writes: a token stored first would, on a crash
                # between the two, vouch for documents this run never indexed.
                moved = bool(state.token) and state.token != store.token
                if state.token:
                    store.remember(state.token)
                # Nothing written, nothing to persist. `save` serializes every
                # record — 7.1 MB and 16 ms at 6 000 chunks — plus a turbovec
                # fsync, and a call that proved the whole corpus unchanged is
                # the case this filter exists to make free.
                if dirty or moved:
                    async with guard:
                        await anyio.to_thread.run_sync(store.save)
                    dirty = False
            except BaseException:
                # Stopped or failed part-way. What it embedded is whole — each
                # document's passages and vectors went in together — so it is
                # kept, without the token: the next run finds these documents by
                # their digests and starts where this one ended. The failure is
                # written by `BackgroundRuns`.
                if dirty:
                    with anyio.CancelScope(shield=True):
                        async with guard:
                            await anyio.to_thread.run_sync(store.save)
                raise
        stats = store.stats()
        # As the run left it: a limit learned from a refusal is lower than the
        # one it began with. Said when it shaped a passage, and not otherwise.
        known = seam.known
        limit = known.said() if known is not None and (held or cut_smaller) else ""
        value = IndexValue(
            documents=indexed,
            done=len(documents),
            total=len(documents),
            chunks_added=added,
            chunks_removed=removed,
            unchanged=unchanged,
            limit=limit,
            passage_tokens=await seam.room(known) if limit else None,
            cut_smaller=cut_smaller,
            skipped=[SkippedValue(**one) for one in skipped],
            total_documents=stats["documents"],
            total_chunks=stats["chunks"],
        ).model_dump()
        return value

    async def index_tool(args: IndexArgs, run: ToolRunContext) -> dict[str, Any]:
        # In a host the run outlives this call, which answers with the run's
        # counts if it ends within the grace and with how far it has got if not.
        workspace = ctx.require(FS).root_for(run.agent)
        value = await seam.runs.run(
            ctx.get(TOOL_HOST),
            seam.home(workspace),
            args,
            partial(index_run, args, run),
            what="documents",
            # For whoever is told of a run nobody waited for: what the value
            # says to a caller that did.
            said=lambda value: "\n".join(_index_lines(args.forget, value, brief=True)),
            raise_if_canceled=run.raise_if_canceled,
        )
        if value is not None:
            return value
        progress = seam.indexing(workspace)
        stats = (await seam.index(workspace)).stats()
        return IndexValue(
            state="indexing",
            done=0 if progress is None else progress.done,
            total=0 if progress is None else progress.total,
            total_documents=stats["documents"],
            total_chunks=stats["chunks"],
        ).model_dump()

    async def search_tool(args: SearchArgs, run: ToolRunContext) -> dict[str, Any]:
        workspace = ctx.require(FS).root_for(run.agent)
        store = await seam.index(workspace)
        vector, searched_chars = await _asked(seam, args.query)
        run.raise_if_canceled()
        progress = seam.indexing(workspace)
        guard = seam.guard(workspace)
        # The id set is built **once** and handed to `search`, which used to
        # rebuild it internally while this computed it for the count — two
        # O(chunks) passes per filtered query (7.5 ms against a 0.22 ms
        # unfiltered search on a 10 000-chunk index).
        allowed: list[int] | None = None
        scope = _scoped(ctx.require(FS), args.paths, run)
        # Under the guard: in a host a run adds to this same copy as it goes.
        async with guard:
            if scope is not None:
                allowed = await anyio.to_thread.run_sync(store.ids_under, scope)
                searched = len(allowed)
            else:
                searched = int(store.stats()["chunks"])
            hits: list[tuple[float, Record]] = await anyio.to_thread.run_sync(
                lambda: store.search(vector, args.k, allowed=allowed)
            )
        return {
            "query": args.query,
            "searched": searched,
            "indexing": (
                None if progress is None else {"done": progress.done, "total": progress.total}
            ),
            "query_chars": searched_chars,
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

    ctx.require(TOOLS).register(
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
        )
    )
    ctx.require(TOOLS).register(
        define_tool(
            "text_search",
            SEARCH_DESCRIPTION,
            parameters=SearchArgs,
            output=ToolOutput(schema=SearchValue, render=_render_search),
            execute=search_tool,
            is_concurrency_safe=True,
            self_limits=True,
            **simple_views("search", "Search text", "query"),
        )
    )

    async def model(verb: Provisioning, _invocation: CommandContext) -> str:
        """`/text-index install` — fetch and load the model, now, on purpose.

        A **command** and not a tool, per the seam's own rule: this is a thing
        the *person* asks the harness to do, and routing it through a model turn
        would put the model in the log as having decided it. It costs no turn.
        """
        if seam.provider is None:
            return (
                "No embedder is registered, so there is nothing to install — "
                "mount `text-index-local-engine` or a provider row of your own."
            )
        # Asked of the *provider*, not of this row's config: where weights land
        # is the embedder's business, and an endpoint has none.
        weights = _weights(seam.provider)
        if verb == "status":
            state = _model(seam.provider)
            hint = " — `install` fetches it" if state == "not downloaded" else ""
            return f"{seam.provider.name}: {state}{hint}. Weights under {weights}."
        ok, sentence = await provision(seam.provider)
        if not ok:
            # Raised, so the command exits failing: a session that asked for the
            # model at its own mount (`preload`) refuses on it.
            raise HarnessError(
                f"{sentence}\n(weights would go under {weights})", TEXT_INDEX_MODEL_FAILED
            )
        return sentence

    command = install_or_status(
        ROW,
        "Download and load the embedding model, or report whether it is ready.",
        model,
    )
    contribute_item(
        ctx,
        COMMANDS,
        command,
        label="text-index command",
    )

    contribute(
        ctx,
        Diagnostic(id=ROW, title="Text index", order=61, read=seam.doctor),
    )


@dataclass(frozen=True, slots=True)
class _Cut:
    """One document as it was embedded: its passages and their vectors."""

    chunks: list[Chunk]
    vectors: Vectors
    held: bool
    """Whether the embedder's limit, and not `max_chars`, ended any of them."""
    again: bool
    """Whether the document was packed more than once to get here."""


async def _packed(
    seam: TextIndexSeam, paragraphs: Sequence[Paragraph], room: int | None
) -> tuple[list[Chunk], bool]:
    """`paragraphs` as passages of at most the row's `max_chars`, and of at most
    `room` tokens of the embedder's where there is a room to keep to — and
    whether that is what ended any of them. In a worker thread: weighing is a
    tokenizer's work."""
    config = seam.config
    weight = None if room is None else Weight(of=seam.weigh, most=room)
    chunks = await anyio.to_thread.run_sync(
        partial(
            chunk_paragraphs,
            paragraphs,
            max_chars=config.max_chars,
            overlap_chars=config.overlap_chars,
            weight=weight,
        )
    )
    return chunks, weight is not None and weight.bound


def _lighter(heaviest: float, room: int | None, share: float) -> int:
    """Less room than `room`, for texts that were too long packed to it: `share`
    of what the heaviest of them weighed.

    Of what it weighed and not of `room`, because room is not always what ended
    a passage: where the embedder reads 8,192 tokens and `max_chars` ends a
    passage at 300, half the room is still thirteen times the passage. Always
    under `room`, so asking again gets somewhere, and never under
    `MIN_PASSAGE_TOKENS`."""
    aimed = int(heaviest * share)
    if room is not None:
        aimed = min(aimed, room - 1)
    return max(MIN_PASSAGE_TOKENS, aimed)


async def _refused(
    seam: TextIndexSeam, error: HarnessError, texts: list[str], room: int | None
) -> int | None:
    """The room to pack to after the embedder refused `texts` with `error`, or
    `None` when they were already as small as a passage gets. Raises `error`
    when it is not about length: the one failure a smaller passage fixes is the
    code an adapter classifies a provider's refusal as, whichever embedder said
    it.

    **What the refusal says is read first.** A llama.cpp names both numbers —
    how long the input was and how long a one it takes. The second is the limit
    from here on (`TextIndexSeam.learn`) and the first is how dense this corpus
    is (`calibrate`). Weighed again at that density, texts heavier than that
    limit leaves room for are simply packed to it: the refusal said exactly how
    much fits.

    Otherwise nothing new says how much does — the refusal named no number, or
    by this seam's own weighing the texts fit already — and the next packing is
    aimed in proportion where it named two, a little under, and at half the size
    where it named none."""
    if error.code != CONTEXT_WINDOW_EXCEEDED:
        raise error
    sent, reads = too_long_facts(str(error))
    density = seam.density
    if reads is not None:
        seam.learn(Limit(reads, "refused"))
    if sent is not None:
        seam.calibrate(max(len(text.encode("utf-8")) for text in texts), sent)
    if room is not None and room <= MIN_PASSAGE_TOKENS:
        return None
    named = await seam.room(seam.known)
    heaviest = await seam.heaviest(texts)
    # Only where packing to it again is a different packing: less room, or the
    # same room and the texts now weighed heavier.
    if (
        named is not None
        and named < heaviest
        and (room is None or named < room or (named == room and seam.density < density))
    ):
        return named
    share = reads / sent * 0.9 if sent and reads and sent > reads else 0.5
    return _lighter(heaviest, room, share)


async def _embedded(seam: TextIndexSeam, paragraphs: Sequence[Paragraph]) -> _Cut | str:
    """Cut one document into passages its embedder reads whole, and embed them.

    Packed by what the embedder counts (`_packed`), so a passage is as long as
    the limit and `max_chars` both allow, wherever in the document it is. Then
    held to the limit two more ways, because a packing adds up the weights of a
    passage's parts and an embedder counts the whole. **Measured** as it is
    sent: a document with a passage over the limit is packed again with less
    room. And **refused**, where the measure was an estimate and wrong, or
    nothing said how long an input the embedder reads: a passage it turns away
    as too long is not the end of the run, the document is packed with less room
    (`_refused`) and sent again. Re-packed rather than split in place, so every
    passage still starts and ends on the lines it says it does.

    A document that does not fit even at `MIN_PASSAGE_TOKENS` is given up on,
    with the embedder's own words — what comes back in place of a cut is why —
    and the run goes on to the next.
    """
    room = await seam.room(await seam.limit())
    again = False
    while True:
        chunks, held = await _packed(seam, paragraphs, room)
        if not chunks:
            return "is empty"
        texts = [chunk.text for chunk in chunks]
        # As it stands now: a refusal on the way here may have lowered it.
        limit = seam.known
        if limit is not None and room is not None and room > MIN_PASSAGE_TOKENS:
            longest = max(await seam.measure(texts), default=0)
            if longest > limit.tokens:
                # In proportion, and a little under: aimed at the limit rather
                # than walked down to it.
                share = limit.tokens / longest * 0.9
                room = _lighter(await seam.heaviest(texts), room, share)
                again = True
                continue
        try:
            vectors = await seam.embed(texts, query=False)
        except HarnessError as error:
            smaller = await _refused(seam, error, texts, room)
            if smaller is None:
                return (
                    f"has text the embedder refuses as too long even in passages of "
                    f"{room} tokens: {error}"
                )
            room, again = smaller, True
            continue
        return _Cut(chunks, vectors, held, again)


async def _asked(seam: TextIndexSeam, query: str) -> tuple[Vectors, int | None]:
    """The question's vector, and how many of its characters were embedded when
    that is not all of them.

    A question is an input like any other: past what the embedder reads, a local
    model dropped its tail without a word and a server refused the search. So
    one that measures over the limit is cut to its head — as much of its start
    as fits (`_chunk.head_of`) — and the search says it was. Its head, and not
    an average of its parts: what a model that truncates would have searched by,
    now said. One refused all the same is cut smaller and asked again, as a
    passage is (`_refused`), until it is as small as a passage gets; then the
    refusal is the answer.

    Sent as it was typed when it fits, which is nearly always.
    """
    limit = await seam.limit()
    room: int | None = None
    if limit is not None:
        longest = max(await seam.measure([query], query=True), default=0)
        if longest > limit.tokens:
            room = await seam.room(limit, query=True)
    while True:
        text = (
            query
            if room is None
            else await anyio.to_thread.run_sync(head_of, query, Weight(of=seam.weigh, most=room))
        )
        try:
            vectors = await seam.embed([text], query=True)
        except HarnessError as error:
            smaller = await _refused(seam, error, [text], room)
            if smaller is None:
                raise
            room = smaller
            continue
        return vectors, None if text == query else len(text)


async def _read(
    ctx: Context, seam: TextIndexSeam, path: str, run: ToolRunContext
) -> tuple[str, str | None, str]:
    """One document's text or the reason it was skipped, and a digest of its
    text as it is cut.

    The digest covers the cut as well as the text — the two sizes that decide
    where a passage ends, and a limit a profile set on the embedder — so a
    profile that changes them gets its documents cut again rather than told they
    are unchanged. Only the *setting*: what a model or a server reports is
    learned by asking, and a run that would embed nothing must not load a model
    to compare digests.
    """
    refused = ctx.require(FS).skip_reason(path, max_bytes=seam.config.max_bytes, agent=run.agent)
    if refused:
        return "", refused, ""
    slice_ = await ctx.require(FS).read(
        path,
        # The whole file, which is what `limit=None` means here — a passage index
        # over a truncated document would answer confidently about the first two
        # thousand lines and silently about the rest.
        limit=None,
        # And `skip_reason` above is this walk's bound: a skip that reports what
        # it passed over, where `read`'s own default would raise.
        max_bytes=None,
        scope=run.scope,
        agent=run.agent,
        session=run.session,
    )
    config = seam.config
    cut = f"{config.max_chars}:{config.overlap_chars}:{seam.configured_limit() or ''}\n"
    text_digest = hashlib.sha256((cut + slice_.text).encode("utf-8")).hexdigest()
    return slice_.text, None, text_digest


@plugin(f"{LOCAL_ROW}-engine", affects="environment", inject=[TEXT_INDEX], config=LocalConfig)
async def local(ctx: Context, config: LocalConfig) -> None:
    """Register a local `sentence-transformers` model as the embedder."""
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
        max_tokens=config.max_tokens,
        # `$PH_CACHE/models` unless the operator said otherwise. See the field.
        cache_folder=str(weights_cache(config)),
        trust_remote_code=config.trust_remote_code,
    )
    # `preload` is not read here: a session asks this command to `install` once,
    # at its own mount, and a call loads the model when it embeds. Read here, it
    # loaded the model to answer `status`, and on every call a person's config
    # file asked for it.
    contribute_item(ctx, TEXT_INDEX, embedder, label=f"embedder({config.model})")


@plugin(f"{ENDPOINT_ROW}-engine", affects="environment", inject=[TEXT_INDEX], config=EndpointConfig)
async def endpoint(ctx: Context, config: EndpointConfig) -> None:
    """Register an OpenAI-compatible embeddings endpoint as the embedder.

    Nothing is asked of the endpoint at mount: a server that is down, or a key
    that is not set, is a sentence on the call that needed a vector, and not a
    command that will not print its manual. The key's variable is named for the
    manual (`ph.toolcli.ToolEnv`); in a session the row's `pass_env` hands it."""
    if config.api_key_env:
        contribute_item(ctx, TOOL_ENV, config.api_key_env, label="endpoint key variable")
    contribute_item(
        ctx,
        TEXT_INDEX,
        EndpointEmbedder(
            base_url=config.base_url,
            model=config.model,
            api_key_env=config.api_key_env,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            timeout_s=config.timeout_s,
            max_tokens=config.max_tokens,
        ),
        label=f"embedder({config.model or config.base_url})",
    )


@plugin(f"{ADAPTER_ROW}-engine", affects="environment", inject=[TEXT_INDEX], config=AdapterConfig)
async def adapter(ctx: Context, config: AdapterConfig) -> None:
    """Register the session this command is run for as the embedder.

    Refused where there is no session to ask: this embedder is a model route of
    a running pH session, and a command typed at a shell has none."""
    where = os.environ.get(EMBEDDER_ENV, "").strip()
    if not where:
        raise MountRefusal(
            f"the {ADAPTER_ROW} row embeds through a running pH session's model routes, "
            f"and no session told this command where to ask (${EMBEDDER_ENV} is not set). "
            f"From a shell, configure `{LOCAL_ROW}` or `{ENDPOINT_ROW}` instead."
        )
    contribute_item(
        ctx,
        TEXT_INDEX,
        SessionEmbedder(
            where=where,
            route=f"{config.provider}/{config.model}",
            agent=os.environ.get(AGENT_ENV, "").strip() or None,
            query_prefix=config.query_prefix,
            document_prefix=config.document_prefix,
            dimensions=config.dimensions,
            batch_size=config.batch_size,
            max_tokens=config.max_tokens,
        ),
        label=f"embedder({config.provider}/{config.model})",
    )
