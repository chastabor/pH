"""`text-index`: semantic retrieval over the agent's own documents.

This module has three rows and two tools. `text_index` reads documents through
`ctx.fs` and cuts them into passages that keep their line numbers. It embeds the
passages and puts them in a turbovec index on disk. `text_search` answers a
question with the passages themselves, and with the `path:start-end` that each
passage came from. A hit is therefore both an answer and a pointer that the
agent can `read` for the rest.

## Where these rows are mounted

These rows are mounted in the process of the `ph-text-index` command, not in the
process of a session. A session mounts the `text-index` row and one embedder
row, `text-index-local` or `text-index-endpoint` (`ph_text_index.row`). Those
rows register the UTCP manual of this package. The commands of the manual run
`ph-text-index`. That command mounts *these* rows and runs a tool through the
registry (`ph.toolcli`). The `ctx.fs` below therefore belongs to the command. It
is rooted at the working directory that the sandbox of the calling agent gave
the command, and `run.agent` is always `None`.

## The seam and the embedder

`text-index-engine` publishes the seam and the tools. One embedder row claims
the seam: `text-index-local-engine` or `text-index-endpoint-engine`. The command
always mounts exactly one embedder row (`ph.toolcli.ToolCli.providers`). An
embedder row that cannot mount refuses the mount of the command, so in this
process there is no seam without an embedder. The rule that a profile with no
embedder offers no tools belongs to the session. That rule is in the rows that a
session mounts (`ph_text_index.row`).

## Reading through `ctx.fs`

Every byte that is indexed arrives through `ctx.fs.read`. The ignore rules of
the walk, the size limit, and the skip reasons belong to that seam, and are
stated one time, there. In a session, the `ctx.fs` of the command is not the
`ctx.fs` of the session. `permissions-fs` governs calls through the `ctx.fs` of
a session, and a command is not such a call. In a session, the sandbox of the
calling agent therefore limits these reads, as it limits the reads of `bash`.

## One run at a time, and a search never waits

`text_index` is one command, and it finishes before it answers. A run holds the
writer lock while it works. A second run waits up to `wait_s` for its turn.
After that it answers `state: indexing`, which means that another run is
building the index, and it does not continue to wait. `text_search` takes no
lock. It reads what is saved, and says when a run is still building the index.
A run keeps a digest of each document, so an unchanged document is never
embedded twice. A run also saves as it goes. A run that is stopped therefore
keeps what it embedded, and a search during a run finds what the run has saved.
With `--progress`, the command reports its progress on stderr
(`ph.toolcli.reported`). A session uses that to show a run that it started in
the background.

## A passage fits its embedder

A model reads a fixed number of tokens and no more. Past that number, a local
model drops the rest without any notice, so the tail of the passage is in the
index and can never be found. A server refuses the request. Passages are
therefore packed by what the embedder counts as well as by `max_chars`
(`_chunk.Weight`, `_embed.TokenBounded`). The count is the embedder's own tokens
where it has a tokenizer. Where it has none, the count is the bytes of the text
at the density of the corpus. Each passage is as long as both limits allow. A
table is therefore cut small and the prose around it is not, whatever script
the table or the prose is in. The text that is sent is checked. A passage that
is still refused as too long is cut smaller and sent again, and the run does not
fail. The result reports when any of this happened. A question has the same
limit. A question that is longer than the embedder reads is searched by its
head, and the answer says so.

## Whose index

The index is where the caller said (`ph.toolcli.state_dir`). A session names a
directory in the scratch directory of the calling agent. Each agent therefore
has its own index of its own tree. That index is writable in a read-only
session, and the agent finds it again after a restart. When no caller names a
directory, there is one index directory for each workspace under
`$PH_CACHE/text-index`. In both cases there is one directory for each embedder
below that location. A change is made under a file lock in that directory
(`TextIndexSeam.writing`), because two processes can share one directory.

@module ph_text_index.engine
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.cordis import Context, Disposer, MountRefusal, Running, ServiceKey, plugin
from ph.indexable import Paragraph, triage
from ph.json import JsonObject, as_bool, as_int
from ph.keys import COMMANDS, FS, TOOLS
from ph.llm.adapter import refused_share, too_long_facts
from ph.llm.tokens import BYTES_PER_TOKEN
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED, ContentBlock
from ph.locks import KeyedLocks, file_lock_held
from ph.seams._registry import claim_slot, contribute_item
from ph.seams.changes import TreeState, tree_state
from ph.seams.commands import CommandContext, Provisioning, install_or_status
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.fs import FsService, skipped_lines
from ph.session import now_ms
from ph.text import BRIEFLY, count_of, first_lines, one_line
from ph.toolcli import RunProgress, claim_home, reported, run_lock, state_home
from ph.tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ph.tools.errors import HarnessError
from ph.tools.presentation import simple_views
from ph.workspace_caches import digest

from ._chunk import Chunk, Weight, chunk_paragraphs, head_of
from ._config import (
    ENDPOINT_ROW,
    LOCAL_ROW,
    MISSING_MODEL,
    ROW,
    Config,
    EmbedderRecord,
    EndpointConfig,
    Learned,
    Limit,
    LocalConfig,
    index_base,
    weights_cache,
)
from ._embed import (
    Embedder,
    EndpointEmbedder,
    LocalWeights,
    Metered,
    SentenceTransformerEmbedder,
    Served,
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
"""The smallest number of tokens of its own text that a passage is limited to so
that it fits an embedder. Below this number a "passage" is a phrase. A document
that the embedder still refuses at this size is skipped, and the result says so."""

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
    """The state of the model of `provider`.

    The check is `isinstance` against `LocalWeights`, not a `hasattr` probe.
    That Protocol gives the reason, which is a rule that this tree states three
    times. An endpoint-backed embedder has nothing to load, so it is always
    ready, and its answer is "nothing to load".

    A local model has three answers and not two, because this row runs in a
    command that lives for one call. "Not loaded" is true of every such process.
    A person who asks needs to know whether the next call will have to download.
    """
    if not isinstance(provider, LocalWeights):
        return "nothing to load"
    if provider.ready():
        return "loaded"
    return "downloaded, loads on first use" if provider.cached() else "not downloaded"


def _weights(provider: Embedder) -> str:
    """Where the weights of `provider` go, in the words that a person is shown.
    The check is against the Protocol, for the same reason as in `_model`: an
    endpoint has no weights."""
    return provider.cache_folder if isinstance(provider, LocalWeights) else "the library default"


async def provision(provider: Embedder) -> tuple[bool, str]:
    """Load the model now. Returns `(ok, sentence)` and never raises.

    A person triggered this and is waiting for an answer in words, so a failure
    is a sentence and not a traceback. The *upstream* sentence is kept verbatim,
    because the one failure that this function exists to catch says exactly what
    to do: "requires the following packages that were not found in your
    environment: einops". A reworded sentence would replace the useful half.
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

    It holds the one embedder, and a copy of the index of each workspace that it
    has opened. Every change goes through `writing`. The lock in `writing` is
    necessary. Two calls that both loaded, both changed, and both saved would
    leave a sidecar that describes only one of them. That is true whether the
    two calls are in this seam or in another seam that shares the directory.
    """

    ctx: Context
    config: Config
    provider: Embedder | None = None
    provider_by: Running | None = None
    _indexes: dict[Path, TextIndex] = field(default_factory=dict)
    """For each index directory, the copy that this seam last loaded or wrote."""
    _guards: KeyedLocks[Path] = field(default_factory=KeyedLocks)
    """For each index directory, the lock that keeps one use of the copy apart
    from another. A lock exists only while a caller holds it or waits for it.
    Without that, the table kept one lock for each worktree that was searched.

    Where one mount both indexes and searches, a run and a search share one
    copy. The run adds the passages of a document in a worker thread while a
    search scores against the same index in another thread. The lock is held
    around each such step and never across an embedding. A search therefore
    waits for the bookkeeping of one document and not for a model."""
    density: float = BYTES_PER_TOKEN
    """The number of bytes of this corpus that make one token, as far as is
    known. It starts as the careful guess. A refusal that said how many tokens
    its input was lowers it (`calibrate`)."""
    known: Limit | None = None
    """The limit of the embedder, when it is known: the embedder was asked, or a
    refusal gave it. A caller that must not ask reads this field as it is. That
    caller is a run that reports the limit that it ended with."""
    _kept: EmbedderRecord | None = None
    """What the embedder said about itself: its last answer when it was asked, or
    the entry in the log of the index (`TextIndex.embedder`). It is set when the
    limit is settled for this process. That is late, because asking a local
    model means loading it, so only a call that embeds sets it."""

    def register(self, provider: Embedder, *, scope: Context | None = None) -> Disposer:
        """Claim the embedder slot. There is one embedder, because one index is
        one vector space."""
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
        """Where the index of one workspace lives, for this embedder.

        **The location is keyed by the embedder.** Switching models therefore
        does not cause the `IndexMismatch` that it otherwise would. The two
        indexes do not collide, and switching back finds the old index intact.

        **Two trees never share an index**, as `CodeGraphSeam.store_for` keeps
        its own indexes apart. What an index holds belongs to one tree. Its
        documents are named relative to that tree. One version-control token
        says how far the tree was indexed. The sweep of a run drops what that
        tree no longer has. Two workspaces in one directory took turns to delete
        the documents of the other and embed their own again. The index is
        therefore where the caller said (`ph.toolcli.state_dir`), which in a
        session is a directory in the scratch directory of the calling agent.
        When no caller names a directory, it is one directory for each workspace
        under `$PH_CACHE/text-index`. `writing` sweeps that directory after its
        workspace is gone.

        The directory names are digests and not plain names, because a model id
        and a path contain slashes."""
        return self.home(workspace) / digest(self.embedder.name)

    async def index(self, workspace: Path) -> TextIndex:
        """The index of one workspace, for a search. The index is read, and the
        search never waits for it.

        No lock is taken. The open of a reader changes no file
        (`TextIndex.open`), and every file that a writer saves is replaced
        whole. A search during an indexing run therefore answers from what the
        run has saved. It does not wait behind the run for as long as the run
        embeds. The copy is loaded again when the files changed after this copy
        read or wrote them."""
        root = self.root(workspace)
        store = self._indexes.get(root)
        if store is None or store.stale():
            store = self._store(root)
            await anyio.to_thread.run_sync(partial(store.open, readonly=True))
            self._indexes[root] = store
        return store

    def home(self, workspace: Path) -> Path:
        """The directory that holds the indexes of this workspace, one for each
        embedder (`ph.toolcli.state_home`)."""
        return state_home(index_base(), workspace)

    def _store(self, root: Path) -> TextIndex:
        """The index at `root`, not yet opened."""
        return TextIndex(
            root=root,
            model=self.embedder.name,
            bit_width=self.config.bit_width,
            calibrate=self.config.calibrate,
        )

    def guard(self, workspace: Path) -> AbstractAsyncContextManager[None]:
        """The lock around each use of this workspace's copy. See `_guards`."""
        return self._guards.hold(self.root(workspace))

    def indexing(self, workspace: Path) -> bool:
        """Whether a run is building the index of this workspace now. A run holds
        the writer lock for as long as it works, and only a writer takes it."""
        return file_lock_held(self.root(workspace) / WRITER_LOCK)

    @asynccontextmanager
    async def writing(self, workspace: Path) -> AsyncIterator[TextIndex | None]:
        """Hold the index of one workspace for one change, against every other
        writer of it. Yields the index, or `None` when another writer still held
        the lock after `wait_s` (`ph.toolcli.run_lock`).

        The lock is a file in the index directory, so it orders every writer of
        that directory: another root in the same daemon, a `phern -p`, a second
        daemon, and also the calls of this seam, because `flock` refuses a
        second holder even inside one process. Each writer used to hold its own
        copy and save it whole. The sidecar then described the writer that saved
        last, the vectors of the other writer lost their meaning, and both
        writers allocated the same ids. Under the lock, a copy that another
        writer has changed is loaded again, so every change starts from what is
        on disk.

        A second writer waits, because it is expected to queue behind the first.
        It does not wait without limit, because the caller is a call.
        """
        root = self.root(workspace)
        async with run_lock(root / WRITER_LOCK, what=_LOCKED, wait_s=self.config.wait_s) as held:
            if not held:
                yield None
                return
            # This is under the lock, so a call that answers "indexing" sweeps
            # nothing.
            await anyio.to_thread.run_sync(claim_home, index_base(), workspace)
            yield await self._fresh(root)

    async def _fresh(self, root: Path) -> TextIndex:
        """The copy of the index at `root`, loaded again if its files changed.
        The caller holds the writer lock, so no writer is part-way through
        writing those files."""
        store = self._indexes.get(root)
        # Do not write from the copy of a reader. Its open reconciled nothing.
        if store is None or store.stale() or store.readonly:
            store = self._store(root)
            await anyio.to_thread.run_sync(store.open)
            self._indexes[root] = store
        return store

    def configured_limit(self) -> int | None:
        """The limit that a profile set on the embedder, or `None` when it set
        none. Reading it loads no model and asks no server."""
        embedder = self.embedder
        return embedder.max_tokens if isinstance(embedder, TokenBounded) else None

    async def limit(self, store: TextIndex, *, ask: bool = False) -> Limit | None:
        """The longest input that the embedder reads, or `None` when nothing
        gives it.

        The limit is settled one time in each process. For a caller that does
        not `ask`, the answer is the newest entry of the embedder log of `store`
        (`TextIndex.embedder`). Such a caller is a search. A search is one
        command, and it would otherwise send a question to the server every
        time. An index run asks. Any caller also asks when the entry came from
        another address. When the answer differs from the entry, the run adds it
        to the log (`record`), and what refusals had taught under the old entry
        does not carry over."""
        embedder = self.embedder
        if self._kept is not None or not isinstance(embedder, TokenBounded):
            return self.known
        address = embedder.asked_of() if isinstance(embedder, Served) else None
        kept = store.embedder()
        if kept is not None and kept.asked_of == address and not ask:
            self._take(kept)
            return self.known
        try:
            found = await anyio.to_thread.run_sync(embedder.limit)
        except Exception:
            # The failure is not kept. A model that does not load reports that in
            # its own words on the embedding that follows.
            log.debug("ph_text_index: the embedder's limit was not learned", exc_info=True)
            return self.known
        reported = embedder.reported() if isinstance(embedder, Served) else {}
        if kept is not None and (kept.asked_of, kept.limit, kept.reported) == (
            address,
            found,
            reported,
        ):
            self._take(kept)
        else:
            self._take(
                EmbedderRecord(
                    asked_of=address, asked_at_ms=now_ms(), reported=reported, limit=found
                )
            )
        return self.known

    def _take(self, record: EmbedderRecord) -> None:
        """Make `record` the answer, with what refusals had taught under it."""
        self._kept = record
        self.known = record.effective()
        if record.learned.density is not None:
            self.density = min(self.density, record.learned.density)

    def record(self) -> EmbedderRecord | None:
        """What is known about the embedder now, for the log of the index: what it
        said when asked, and what its refusals have taught since then. The next
        command then starts from this record, and does not spend a request or a
        refusal to learn it again. `None` before the limit is settled."""
        if self._kept is None:
            return None
        known = self.known
        refused = known is not None and known.source == "refused"
        learned = Learned(
            tokens=known.tokens if known is not None and refused else None,
            density=self.density if self.density < BYTES_PER_TOKEN else None,
        )
        return self._kept.model_copy(update={"learned": learned})

    def learn(self, limit: Limit) -> None:
        """Take `limit` as the limit: what the embedder says when asked, or what
        it said when it refused an input as too long.

        What an embedder reports when asked is not always what it accepts. A
        llama.cpp reports its window, and refuses an embedding that is longer
        than its *batch*, which `/props` does not mention. Measured: a window of
        2,048 and a refusal past 512. Without this method, every document after
        the first would be packed to the reported size and refused. With it, the
        rest of the run is packed to what the refusal said. The limit only moves
        down, and only from a refusal that names a number. A guess from the size
        that happened to be accepted would shrink every later document to fit
        the densest one."""
        if self.known is None or limit.tokens < self.known.tokens:
            self.known = limit

    def calibrate(self, size: int, tokens: int) -> None:
        """Record that `size` bytes of this corpus were `tokens` tokens, as a
        refusal said. Technical prose is denser than the guess, because it has
        paths, code, and tables. One refusal that says by how much keeps the
        next dense document from being packed too large and refused in turn. The
        density only moves toward denser, and never past one token for each
        byte."""
        if tokens > 0:
            self.density = max(1.0, min(self.density, size / tokens))

    async def room(self, limit: Limit | None, *, query: bool = False) -> int | None:
        """The number of tokens of its own text that an input may have: `limit`,
        less what the embedder puts around the text. `None` when nothing says
        how long an input the embedder reads.

        **The room is the limit less what is around the text, not the limit.**
        Every passage is sent after a prefix (`search_document: `) and between
        the start and end markers of the model. The embedder counts all of
        those. When passages were packed to the limit itself, a full passage
        measured a few tokens over it. Nearly every document was then packed
        again and reported as needing smaller passages: on a live run, 14 of 30,
        where one was. A question has its own prefix, which for some models is a
        sentence."""
        if limit is None:
            return None
        # The cost of an input before its text is the measure of an empty input.
        overhead = max(await self.measure([""], query=query), default=0)
        return max(MIN_PASSAGE_TOKENS, limit.tokens - overhead)

    async def measure(self, texts: list[str], *, query: bool = False) -> list[int]:
        """The number of tokens that the embedder counts for each text as it is
        sent. The number is counted where the embedder has a tokenizer, and
        estimated at the density of this corpus where it has none. The list is
        empty for an embedder that cannot say."""
        embedder = self.embedder
        if not isinstance(embedder, TokenBounded):
            return []
        return await anyio.to_thread.run_sync(
            partial(embedder.measure, texts, self.density, query=query)
        )

    def weigh(self, texts: Sequence[str]) -> Sequence[float]:
        """The number of tokens that each text adds to a passage that it is part
        of (`_chunk.Weight.of`). This is the count of the embedder, or the bytes
        of the text at the density of this corpus. Bytes are also used for an
        embedder that cannot say how long an input it reads, so that a passage
        that it refuses can still be cut by some measure. **Blocking**: this is
        the work of a tokenizer."""
        embedder = self.embedder
        if isinstance(embedder, TokenBounded):
            return embedder.weigh(texts, self.density)
        return weigh_by_bytes(texts, self.density)

    async def heaviest(self, texts: list[str]) -> float:
        """The weight of the heaviest of `texts` (`weigh`), computed off the event
        loop."""
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
            # The first call that needs a model that is not on disk fetches it. A
            # call made for a confined agent often cannot fetch it, because its
            # network is the allowlist of the deployment. The upstream sentence
            # is kept, and the message names the fix that a person can run.
            raise HarnessError(
                f"the embedding model is not downloaded, and fetching it failed: {error}. "
                f"Run `/{ROW} install` (or `ph-text-index install` from a shell) — a "
                "person's command, run outside the sandbox — and call again.",
                TEXT_INDEX_MODEL_MISSING,
            ) from error

    def doctor(self) -> list[tuple[str, str]]:
        """The section for `phern doctor`: what is indexed, and by what."""
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
            # A command that is asked for its status has opened nothing yet. The
            # index is read without the writer lock, and so as a reader, which
            # changes no file. These are counts for a person. A writer that is
            # part-way through a save is a reason to say so, not to wait.
            store = self._store(root)
            try:
                store.open(readonly=True)
            except Exception as error:
                rows.append(("state", f"could not be read: {error}"))
                return rows
        stats = store.stats()
        rows.extend(
            [
                ("limit", self._limit_said(store)),
                ("documents", str(stats["documents"])),
                ("chunks", str(stats["chunks"])),
                ("vectors", f"{stats['dim']} dim at {stats['bit_width']} bit"),
                ("calibration", str(stats["calibration"])),
            ]
        )
        return rows

    def _limit_said(self, store: TextIndex | None) -> str:
        """The longest input that the embedder reads, as far as it is known
        without asking the embedder. The sources, in order: what this process
        settled, the newest entry of the embedder log of the index
        (`TextIndex.embedder`), and the setting. Asking would load a local model
        to answer a question about status."""
        if self.known is not None:
            return self.known.said()
        kept = None if store is None else store.embedder()
        limit = None if kept is None else kept.effective()
        if kept is not None and limit is not None:
            where = "" if kept.asked_of is None else f" of {kept.asked_of}"
            return f"{limit.said()}, as the last run to embed asked{where}"
        configured = self.configured_limit()
        if configured is not None:
            return Limit(tokens=configured, source="configured").said()
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
    """`indexing`: another run holds the index, and it was still working after
    this call had waited `wait_s`. This call did nothing below. The totals are
    what the index holds so far."""
    documents: list[str] = Field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    unchanged: int = 0
    """The documents that were proved unchanged, by the version control or by
    their own digest, and so were not embedded again."""
    limit: str = ""
    """The limit of the embedder, as a phrase, when it ended a passage of this
    run sooner than the `max_chars` of the row would have. Empty when it did
    not."""
    embedder_changed: str | None = None
    """A sentence, when what the embedder says about itself differs from what it
    said the last time an index run asked: a server moved, or was given another
    window. What was learned from its refusals is dropped with the old answer."""
    spent: dict[str, int] | None = None
    """What this run spent on embedding, by unit (`Metered`): the `requests` and
    `texts` that the embedder was sent, and the `tokens` that its server
    reported. Absent when the run spent nothing that the embedder counts."""
    passage_tokens: int | None = None
    """The number of tokens of its own text that a passage was limited to, when
    `limit` is reported: the limit, less the prefix and the markers that the
    embedder adds."""
    cut_smaller: list[str] = Field(default_factory=list)
    """The documents that were packed a second time, smaller, to fit. Either the
    text that was sent measured over the limit, or the embedder refused a
    passage as too long."""
    skipped: list[SkippedValue] = Field(default_factory=list)
    total_documents: int
    total_chunks: int


def _scoped(fs: FsService, paths: list[str] | None, run: ToolRunContext) -> list[str] | None:
    """The `paths` filter, in the spelling that the records are stored under (X2).

    `index_tool` stores documents as `fs.collect` returns them, which is
    `fs.named()`: workspace-relative posix. An absolute path in a stored record
    puts the machine and the run into the conversation. The filter used to be
    matched raw, exactly as the model typed it. `["."]`, `["./docs"]`, and any
    absolute path then matched nothing at all, and the reply said "among 0
    indexed passages". That is a wrong answer that reads like an empty index and
    not like a filter that missed.

    **`.` and `""` mean the workspace, which is no filter.** `named(".")` is
    `"."`, and no document is stored under that name. The obvious way to say
    "search everything" therefore excluded everything. Dropping them leaves the
    unfiltered search, which is what the model meant and what the count then
    describes.
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


class SearchValue(ToolModel):
    query: str
    hits: list[HitValue]
    searched: int
    """The chunks that the query was scored against: the whole index, or the
    filtered subset."""
    indexing: bool = False
    """A run is still building the index. The hits come from what the run has
    saved so far."""
    query_chars: int | None = None
    """The number of characters of the question that were searched, when the
    question is longer than the embedder reads and only its head was searched.
    `None` when all of it was searched."""


# ------------------------------------------------------------------- render ----


def _render_index(args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("state") == "indexing":
        return text_content(
            "An indexing run is already building this index, so this call started "
            "none. `text_search` answers now, from what is indexed already, and "
            "says while the index is still being built. Call `text_index` again once "
            "that run has finished.\n"
            f"The index holds {count_of(value['total_chunks'], 'passage')} "
            f"from {count_of(value['total_documents'], 'document')} so far."
        )
    return text_content("\n".join(_index_lines(as_bool(args.get("forget")), value)))


def _changed(value: Mapping[str, Any]) -> bool:
    """Whether a run changed the index: a passage added, or one removed."""
    return as_int(value["chunks_added"]) + as_int(value["chunks_removed"]) > 0


def _busy(value: Mapping[str, Any]) -> bool:
    """Whether another run held the index, so this one did nothing."""
    return value.get("state") == "indexing"


def _noted(seam: TextIndexSeam, store: TextIndex) -> bool:
    """Add what is known about the embedder to the log of `store`
    (`TextIndex.note_embedder`). Returns whether it was added, so that the
    caller saves."""
    record = seam.record()
    return record is not None and store.note_embedder(record)


def _counts(embedder: Embedder) -> Counter[str]:
    """What `embedder` has spent in this process (`Metered`). Empty for an
    embedder that does not count."""
    return Counter(embedder.spent()) if isinstance(embedder, Metered) else Counter()


def _spent_since(embedder: Embedder, before: Counter[str]) -> Counter[str]:
    """What `embedder` has spent since `before`, which is the share of one run. A
    unit that it spent none of is left out."""
    return _counts(embedder) - before


def _spend_said(spent: Mapping[str, int]) -> str:
    """`{"requests": 1, "tokens": 66774}` as "1 request, 66,774 tokens"."""
    return ", ".join(
        f"{count:,} {unit.removesuffix('s') if count == 1 else unit}"
        for unit, count in spent.items()
    )


def _changed_said(before: EmbedderRecord | None, after: EmbedderRecord | None) -> str | None:
    """A sentence, when the embedder says that it reads a different length than
    when it was last asked, or when it is asked at another address. `None` when
    it says the same, or when there was nothing to compare."""
    if before is None or after is None:
        return None
    if (before.asked_of, before.limit) == (after.asked_of, after.limit):
        return None
    was = "no limit" if before.limit is None else before.limit.said()
    now = "no limit" if after.limit is None else after.limit.said()
    if before.asked_of != after.asked_of:
        return (
            f"The embedder is now asked at {after.asked_of} and reads {now}; it was "
            f"asked at {before.asked_of} and read {was}."
        )
    return f"The embedder now reads {now}; when last asked it read {was}."


def _index_lines(forget: bool, value: Any, *, brief: bool = False) -> list[str]:  # noqa: ANN401
    """What a finished run did, as lines. `brief` is for the note about a run that
    nobody waited for (`Progress.note`). That note does not list the documents,
    and it only counts what the run skipped after the first few."""
    verb = "Removed" if forget else "Indexed"
    lines = [
        f"{verb} {count_of(len(value['documents']), 'document')}: "
        f"{count_of(value['chunks_added'], 'passage')} added, "
        f"{value['chunks_removed']} removed"
        # This is added only when it happened. It says "not re-embedded" because
        # that is the cost that was avoided. The read is the cheap half.
        + (
            f"; {value['unchanged']} unchanged and not re-embedded."
            if value.get("unchanged")
            else "."
        )
    ]
    if value.get("embedder_changed"):
        lines.append(value["embedder_changed"])
    if value.get("spent"):
        lines.append(f"Spent embedding: {_spend_said(value['spent'])}.")
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
        lines += first_lines(value["documents"], 40, str)
    skipped = value["skipped"]
    lines += skipped_lines(skipped, BRIEFLY if brief else len(skipped))
    lines.append(
        f"The index now holds {count_of(value['total_chunks'], 'passage')} "
        f"from {count_of(value['total_documents'], 'document')}."
    )
    return lines


def _unfinished(value: Any) -> str:  # noqa: ANN401
    """The warning a search carries while its index is still being built."""
    if not value.get("indexing"):
        return ""
    return (
        "Not finished indexing. These results cover what is indexed so far; "
        "search again once it finishes."
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
    """The question as a result repeats it, on one line. The result repeats all of
    an ordinary question, and only the start of one that is a page long. The
    model already has the question, in the call."""
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

    seam = TextIndexSeam(ctx=ctx, config=config)
    ctx.provide(TEXT_INDEX, seam)

    def _forget_missing(store: TextIndex, run: ToolRunContext) -> int:
        """Drop every indexed document that is no longer on disk. This takes one
        thread hop."""
        fs = ctx.require(FS)
        dropped = 0
        for path in store.documents():
            if not fs.resolve(path, agent=run.agent).exists():
                dropped += store.forget(path)
        return dropped

    async def index_run(
        args: IndexArgs, run: ToolRunContext, progress: RunProgress
    ) -> dict[str, Any]:
        """One indexing run, to its end, or the answer that another run holds the
        index. The cancellation of the caller is checked at each document."""
        added = removed = unchanged = 0
        indexed: list[str] = []
        skipped: list[dict[str, str]] = []
        cut_smaller: list[str] = []
        held = False
        workspace = ctx.require(FS).root_for(run.agent)
        guard = seam.guard(workspace)
        saved = anyio.current_time()
        dirty = moved = False

        async with seam.writing(workspace) as store:
            if store is None:
                stats = (await seam.index(workspace)).stats()
                return IndexValue(
                    state="indexing",
                    total_documents=stats["documents"],
                    total_chunks=stats["chunks"],
                ).model_dump()
            # What the embedder said when it was last asked. The end of the run
            # uses this to report whether the embedder says something else now.
            before = store.embedder()
            # The embedder counts for its whole process. The run reports its own
            # share, which is asked for when a line is written. The share is
            # asked of the embedder and not of the seam, because the last line
            # of a failed run is written after the mount has gone.
            embedder = seam.embedder
            spent_before = _counts(embedder)
            progress.spent = partial(_spent_since, embedder, spent_before)
            # This is under the lock, so a call that will answer "indexing" walks
            # nothing.
            wanted = config.index.narrowed(paths=args.paths, glob=args.glob, exclude=args.exclude)
            documents = await ctx.require(FS).collect(
                wanted.paths,
                wanted.include,
                scope=run.scope,
                limit=config.max_files,
                agent=run.agent,
                exclude=wanted.exclude,
            )

            async def drop(path: str) -> None:
                """Remove the passages of `path`, under the guard that a search
                shares."""
                nonlocal removed, dirty
                async with guard:
                    removed += await anyio.to_thread.run_sync(store.forget, path)
                dirty = True

            # This is under the lock, so the report belongs to this run and not
            # to a run that is waiting.
            progress.begin(len(documents))
            try:
                # **Ask the version control before reading anything.** The
                # workspace provider decides which backend answers
                # (`ph.seams.changes`). A tree with no backend answers with an
                # empty state. The digest below then says "unchanged", at the
                # cost of the read. Measured on `docs/`: either check saves
                # 20 s of MiniLM for 539 passages, or 185 s under nomic.
                #
                # The question is asked under the lock. The token that it diffs
                # from is the one that the last writer stored, whichever root
                # or process that was.
                state = TreeState()
                if not args.forget:
                    state = await tree_state(ctx, workspace, since=store.token)
                for count, path in enumerate(documents):
                    run.raise_if_canceled()
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
                        # The text is the same as last time, and the text itself
                        # shows it. This is a tree with no version control, or a
                        # run that was stopped before it stored its token.
                        # Nothing is embedded. The id from the backend is
                        # recorded, so that the next run does not need to read
                        # the document to learn this.
                        if state.id_for(path) != store.vcs_id(path):
                            store.vouch(path, state.id_for(path))
                            dirty = True
                        unchanged += 1
                        continue
                    paragraphs: Sequence[Paragraph] = ()
                    if reason is None:
                        # **This is asked after the digest, because it is a
                        # question about the text** (X6). The same text gets the
                        # same answer, so an unchanged document is not asked.
                        # `skip_reason` was the cheap check, and deliberately
                        # did not read the file. This check can tell a minified
                        # bundle from a document. One split and one
                        # classification answer both questions: whether to
                        # index the document, and which of its blocks to index.
                        # The packer gets only the blocks that were kept, so a
                        # README with an embedded image still indexes every
                        # paragraph around the image.
                        kept = triage(text)
                        paragraphs = kept.kept
                        reason = kept.reason or (None if paragraphs else "is empty")
                    if reason is None:
                        # This is done here and not sooner, because it asks the
                        # embedder how long an input it reads. For a local model
                        # that means loading it, and a run that finds every
                        # document unchanged does not need a model. It is done
                        # outside the guard. This is the slow part, and a search
                        # in the same mount must not wait for a model.
                        cut = await _embedded(seam, paragraphs, store)
                        if isinstance(cut, _Cut):
                            held = held or cut.held
                            if cut.again:
                                cut_smaller.append(path)
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
                    # **A document that is gone from disk is removed from the
                    # index** (X3). A deleted document used to keep its passages
                    # until someone ran `forget` by hand. Search then continued
                    # to answer with text that no longer exists. That is worse
                    # here than for the code graph, because a passage is
                    # *quoted back to the model as evidence*.
                    #
                    # The question goes to the filesystem, not to the walk, for
                    # the reason that the sweep in the sibling package gives.
                    # `collect` stops at `max_files`, and the policy screen
                    # prunes directories. "Absent from `documents`" therefore
                    # does not mean "deleted". It can also mean "past the cap"
                    # or "hidden by a rule".
                    #
                    # The whole sweep takes one thread hop, not one for each
                    # document.
                    async with guard:
                        dropped = await anyio.to_thread.run_sync(_forget_missing, store, run)
                    removed += dropped
                    dirty = dirty or bool(dropped)
                # The token is stored after the writes. If the token were stored
                # first, a crash between the two would leave a token that
                # vouches for documents that this run never indexed.
                moved = bool(state.token) and state.token != store.token
                if state.token:
                    store.remember(state.token)
            finally:
                # This runs when the run is done, stopped, or failed. What the
                # run embedded is whole, because the passages and vectors of
                # each document went in together, so it is kept. A run that was
                # stopped part-way keeps it without the token. The next run
                # finds these documents by their digests and starts where this
                # run ended. The same save, under the lock of this run, also
                # keeps what the embedder said about itself and what its
                # refusals taught. When nothing was written, nothing is saved.
                # `save` serializes every record (7.1 MB and 16 ms at 6 000
                # chunks) and adds a turbovec fsync. A call that proved the
                # whole corpus unchanged is the case that this filter exists to
                # make free.
                with anyio.CancelScope(shield=True):
                    if _noted(seam, store) or dirty or moved:
                        async with guard:
                            await anyio.to_thread.run_sync(store.save)
        stats = store.stats()
        # The limit as the run left it. A limit that was learned from a refusal
        # is lower than the one that the run began with. It is reported only
        # when it shaped a passage.
        known = seam.known
        limit = known.said() if known is not None and (held or cut_smaller) else ""
        value = IndexValue(
            documents=indexed,
            chunks_added=added,
            chunks_removed=removed,
            unchanged=unchanged,
            limit=limit,
            embedder_changed=_changed_said(before, seam.record()),
            spent=dict(_spent_since(embedder, spent_before)) or None,
            passage_tokens=await seam.room(known) if limit else None,
            cut_smaller=cut_smaller,
            skipped=[SkippedValue(**one) for one in skipped],
            total_documents=stats["documents"],
            total_chunks=stats["chunks"],
        ).model_dump()
        return value

    async def index_tool(args: IndexArgs, run: ToolRunContext) -> dict[str, Any]:
        return await reported(
            partial(index_run, args, run),
            what="documents",
            # For the reader of the note about a run that nobody waited for:
            # what the value says to a caller that did wait.
            said=lambda value: "\n".join(_index_lines(args.forget, value, brief=True)),
            changed=_changed,
            busy=_busy,
        )

    async def search_tool(args: SearchArgs, run: ToolRunContext) -> dict[str, Any]:
        workspace = ctx.require(FS).root_for(run.agent)
        store = await seam.index(workspace)
        indexing = await anyio.to_thread.run_sync(seam.indexing, workspace)
        guard = seam.guard(workspace)
        async with guard:
            empty = not store.stats()["chunks"]
        if empty:
            # There is nothing to score against. The search answers without
            # loading a model to embed the question. For a local model, loading
            # costs seconds and the memory of the model.
            return {
                "query": args.query,
                "searched": 0,
                "indexing": indexing,
                "query_chars": None,
                "hits": [],
            }
        # What a refusal of the question teaches stays in this search. Only an
        # index run writes the embedder log, under its lock.
        vector, searched_chars = await _asked(seam, args.query, store)
        run.raise_if_canceled()
        # The id set is built **one time** and passed to `search`. `search` used
        # to rebuild it internally while this function computed it for the
        # count. That was two O(chunks) passes for each filtered query (7.5 ms
        # against a 0.22 ms unfiltered search on a 10 000-chunk index).
        allowed: list[int] | None = None
        scope = _scoped(ctx.require(FS), args.paths, run)
        # This is under the guard, because a run in this mount adds to this same
        # copy as it goes.
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
            "indexing": indexing,
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
            # Two of these calls in one batch would both load, both change, and
            # both save. See `TextIndexSeam`. The lock makes the outcome
            # correct, and this setting keeps the scheduler from queueing the
            # two calls against each other (B6).
            is_concurrency_safe=False,
            self_limits=True,
            # The passages are a payload that this call delivered to the index.
            # The model does not refer back to them.
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
        """`/text-index install|status`: fetch and load the model now,
        deliberately, or report whether it is ready.

        This is a **command** and not a tool, by the rule of the seam. The
        *user* asks the harness to do this. If the request went through a model
        turn, the log would show that the model decided it. It costs no turn.
        """
        if seam.provider is None:
            return (
                "No embedder is registered, so there is nothing to install — "
                "mount `text-index-local-engine` or a provider row of your own."
            )
        # This asks the *provider*, not the config of this row. Where the
        # weights go is decided by the embedder, and an endpoint has none.
        weights = _weights(seam.provider)
        if verb == "status":
            state = _model(seam.provider)
            hint = " — `install` fetches it" if state == "not downloaded" else ""
            return f"{seam.provider.name}: {state}{hint}. Weights under {weights}."
        ok, sentence = await provision(seam.provider)
        if not ok:
            # This is raised, so that the command exits with a failure. A session
            # that asked for the model at its own mount (`preload`) then refuses
            # that mount.
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
    """Whether the document was packed more than one time to get here."""


async def _packed(
    seam: TextIndexSeam, paragraphs: Sequence[Paragraph], room: int | None
) -> tuple[list[Chunk], bool]:
    """Pack `paragraphs` into passages of at most the `max_chars` of the row.
    Where `room` is given, each passage also has at most `room` tokens, as the
    embedder counts them. Also returns whether `room` ended any passage. The
    packing runs in a worker thread, because weighing is the work of a
    tokenizer."""
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
    """A smaller room than `room`, for texts that were too long when packed to
    `room`: `share` of the weight of the heaviest text.

    The share is of that weight and not of `room`, because the room is not
    always what ended a passage. Where the embedder reads 8,192 tokens and
    `max_chars` ends a passage at 300, half the room is still thirteen times the
    passage. The result is always under `room`, so that the next attempt makes
    progress, and never under `MIN_PASSAGE_TOKENS`."""
    aimed = int(heaviest * share)
    if room is not None:
        aimed = min(aimed, room - 1)
    return max(MIN_PASSAGE_TOKENS, aimed)


async def _refused(
    seam: TextIndexSeam, error: HarnessError, texts: list[str], room: int | None
) -> int | None:
    """The room to pack to after the embedder refused `texts` with `error`, or
    `None` when the texts were already as small as a passage gets. Raises
    `error` when it is not about length. A smaller passage fixes only one
    failure: the code that an adapter classifies the refusal of a provider as,
    whichever embedder reported it.

    **What the refusal says is read first.** A llama.cpp names both numbers: how
    long the input was, and how long an input it accepts. The second number is
    the limit from now on (`TextIndexSeam.learn`). The first number gives the
    density of this corpus (`calibrate`). The texts are weighed again at that
    density. Texts that are heavier than that limit leaves room for are packed
    to that room, because the refusal said exactly how much fits.

    Otherwise nothing new says how much fits: the refusal named no number, or
    the texts already fit by the weighing of this seam. The next packing is then
    aimed in proportion and a little under where the refusal named two numbers,
    and at half the size where it named none."""
    if error.code != CONTEXT_WINDOW_EXCEEDED:
        raise error
    sent, reads = too_long_facts(str(error))
    density = seam.density
    if reads is not None:
        seam.learn(Limit(tokens=reads, source="refused"))
    if sent is not None:
        seam.calibrate(max(len(text.encode("utf-8")) for text in texts), sent)
    if room is not None and room <= MIN_PASSAGE_TOKENS:
        return None
    named = await seam.room(seam.known)
    heaviest = await seam.heaviest(texts)
    # This applies only where packing to that room again gives a different
    # packing: there is less room, or the room is the same and the texts now
    # weigh more.
    if (
        named is not None
        and named < heaviest
        and (room is None or named < room or (named == room and seam.density < density))
    ):
        return named
    refused = refused_share(sent, reads)
    share = 0.5 if refused is None else refused * 0.9
    return _lighter(heaviest, room, share)


async def _embedded(
    seam: TextIndexSeam, paragraphs: Sequence[Paragraph], store: TextIndex
) -> _Cut | str:
    """Cut one document into passages that its embedder reads whole, and embed
    them.

    The document is packed by what the embedder counts (`_packed`). A passage is
    therefore as long as the limit and `max_chars` both allow, wherever it is in
    the document. Two more checks then hold the passages to the limit, because a
    packing adds up the weights of the parts of a passage and an embedder counts
    the whole:

    * **Measured**, as it is sent. A document with a passage over the limit is
      packed again with less room.
    * **Refused**, where the measure was an estimate and was wrong, or where
      nothing said how long an input the embedder reads. A passage that the
      embedder refuses as too long does not end the run. The document is packed
      with less room (`_refused`) and sent again.

    The document is packed again and not split in place, so every passage still
    starts and ends on the lines that it reports.

    A document that does not fit even at `MIN_PASSAGE_TOKENS` is abandoned. The
    function then returns the reason, in the words of the embedder, in place of
    a cut. The run continues with the next document.
    """
    # The limit is asked for, and not only read. An index run is where a server
    # that was moved or given another window is noticed.
    room = await seam.room(await seam.limit(store, ask=True))
    again = False
    while True:
        chunks, held = await _packed(seam, paragraphs, room)
        if not chunks:
            return "is empty"
        texts = [chunk.text for chunk in chunks]
        # The limit as it is now. A refusal before this point may have lowered it.
        limit = seam.known
        if limit is not None and room is not None and room > MIN_PASSAGE_TOKENS:
            longest = max(await seam.measure(texts), default=0)
            if longest > limit.tokens:
                # The new room is in proportion, and a little under. It is aimed
                # at the limit, and does not approach it in small steps.
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


async def _asked(seam: TextIndexSeam, query: str, store: TextIndex) -> tuple[Vectors, int | None]:
    """The vector of the question, and the number of its characters that were
    embedded when that is not all of them.

    A question is an input like any other. Past what the embedder reads, a local
    model dropped the tail of the question without any notice, and a server
    refused the search. A question that measures over the limit is therefore cut
    to its head, which is as much of its start as fits (`_chunk.head_of`), and
    the search reports that. The head is used, and not an average of the parts
    of the question. The head is what a model that truncates would have searched
    by, and the search now reports it. A question that is still refused is cut
    smaller and asked again, as a passage is (`_refused`), until it is as small
    as a passage gets. After that, the refusal is the answer.

    The question is sent as it was typed when it fits, which is nearly always.
    """
    limit = await seam.limit(store)
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
    """The text of one document or the reason that it was skipped, and a digest
    of the text as it is cut.

    The digest covers the cut as well as the text. The cut is the two sizes that
    decide where a passage ends, and a limit that a profile set on the embedder.
    When a profile changes them, its documents are cut again and are not
    reported as unchanged. Only the *setting* is covered. What a model or a
    server reports is learned by asking, and a run that would embed nothing must
    not load a model to compare digests.
    """
    refused = ctx.require(FS).skip_reason(path, max_bytes=seam.config.max_bytes, agent=run.agent)
    if refused:
        return "", refused, ""
    slice_ = await ctx.require(FS).read(
        path,
        # Read the whole file, which is what `limit=None` means here. A passage
        # index over a truncated document would answer with confidence about
        # the first two thousand lines and say nothing about the rest.
        limit=None,
        # `skip_reason` above is the limit for this walk. It skips and reports
        # what it passed over, where the default of `read` would raise.
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
    # This checks for the package, not the weights. See
    # `SentenceTransformerEmbedder`. A refusal here says "this deployment cannot
    # embed", which an operator can fix. Without it, a tool would fail on its
    # first call in a session that had already committed to using it.
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
    # `preload` is not read here. A session asks this command to `install` one
    # time, at its own mount, and a call loads the model when it embeds. When
    # `preload` was read here, it loaded the model to answer `status`, and on
    # every call when the config file of a person asked for it.
    contribute_item(ctx, TEXT_INDEX, embedder, label=f"embedder({config.model})")


@plugin(f"{ENDPOINT_ROW}-engine", affects="environment", inject=[TEXT_INDEX], config=EndpointConfig)
async def endpoint(ctx: Context, config: EndpointConfig) -> None:
    """Register an OpenAI-compatible embeddings endpoint as the embedder.

    The mount asks nothing of the endpoint. A server that is down, or a key that
    is not set, gives a sentence on the call that needed a vector. It does not
    give a command that cannot print its manual."""
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
