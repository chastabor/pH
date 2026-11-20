"""`rlm-context-loader`: a queryable corpus, off by default (P3-17, Q4).

The companion plan calls this "prompt-as-a-variable": a large body of text that
the agent should be able to consult without any of it entering the prompt. Prime
Agent does not implement it. Four decisions make this the version of pH and not
that sketch. The first two are the reason that the row is small.

**Access is a binding, not a namespace variable** (C2/C5, Q4). A bare
`context.search(...)` over a variable in the kernel produces no `call` frame.
Its results reach the model as merged cell stdout, capped by `max_log_bytes` and
never offloaded, with no provenance for each query. Registered *tools* give both
offloading and provenance. A cell reaches them as
`await tools.context_search(...)`, because Code Mode builds the `tools` namespace
from the registry. Each query settles as its own `tool/code-dispatch` pair. So
`tools/post-execute` can replace one oversized result with a preview and a spill
locator, while the other results pass through untouched. That is the pattern of
§6a one level up, and it is the whole argument for the binding form.

**The prompt gets metadata, never content.** It gets names, line and character
counts, and how to query. A corpus that is large enough to be worth loading is
large enough that putting any of it in the prompt defeats the purpose.

**The corpus is a recipe, not a snapshot** (D17, Q4). It is built from *declared*
sources, so the durable part is `{loader, sources, digest}`, which is recorded on
`context/loaded`. A later session resolves the sources again. A matching digest
is silent. A changed digest tells the model that the corpus was rebuilt. A
missing source is reported. The plan files this under
`kernel/snapshot {kind: "recipe"}`. That is the right place for a *kernel
variable* that the harness built. This corpus never enters the kernel, precisely
because access is a binding. To record it as a namespace variable would describe
a namespace that does not contain it. The mechanism is the same, and the event
is accurate.

**Below a threshold the row stands down.** `min_chars` exists because a corpus
that is small enough to read with `tools.read` should be read with `tools.read`.
C3 gave the model governed, logged, individually offloadable file access. That
demoted this row from a headline feature to a specialist feature, for corpora
that are not files and for repeated queries over one fixed set.

@module ph_rlm.context_loader
"""

from __future__ import annotations

import hashlib
import logging
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from functools import cached_property
from glob import iglob
from math import ceil
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.characters import pieces
from ph.cordis import Context, plugin
from ph.json import JsonObject, as_str
from ph.keys import FS, SYSTEM_PROMPT, TOOLS
from ph.llm.types import ContentBlock
from ph.session import Session, SessionTable
from ph.session.writers import log_writer
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ph.tools import ToolModel, define_tool, text_content
from ph.wire import WireModel

from .keys import CONTEXT_CORPUS

_LOG = log_writer(__name__)

__all__ = [
    "LOADED",
    "Config",
    "ContextService",
    "ContextSource",
    "Corpus",
    "Document",
    "apply",
    "render_manifest",
]

log = logging.getLogger("ph_rlm.context_loader")

LOADED = "context/loaded"
"""The recipe: what was loaded, from where, and whether it changed."""

SEARCH_TOOL = "context_search"
CHUNKS_TOOL = "context_chunks"
HEAD_TOOL = "context_head"

ORDER_CONTEXT = ORDER_TOOL_GUIDANCE + 40
"""After the doctrine: the corpus is a tool to use, and the section tells the
model how to reach it."""

MAX_MATCHES = 200
"""The ceiling on the `limit` of one query. The offload seam can spill a large
result, but nothing should build an unbounded result first."""


class ContextSource(WireModel):
    """One declared source. A `path` source may be a glob. A `text` source is
    literal text."""

    kind: Literal["path", "text"] = "path"
    value: str
    name: str = ""
    """The name of a `text` source in the manifest. A `path` source ignores it and
    is named by its own path, relative to the workspace root when the file is
    under that root."""


class Config(WireModel):
    """Row config for `rlm-context-loader`."""

    corpus: str = "context"
    sources: list[ContextSource] = Field(default_factory=list)
    min_chars: int = 0
    """The size in characters below which the row stands down entirely, with no
    tools and no prompt section. `0` loads whatever is configured. The
    `rlm-stable` profile sets 200 000, the threshold that Q4 settled on, so that a
    small corpus behaves conventionally."""
    max_matches: int = MAX_MATCHES
    """The most matches one `context_search` call may return, however large the
    model sets `limit`."""


@dataclass(frozen=True, slots=True)
class Document:
    """One document, with its line starts computed in advance.

    The class keeps offsets and not a tuple of lines. A corpus that is worth this
    row is large, and a split would hold every line as its own string for the
    life of the process. A line is a slice of `text`.
    """

    name: str
    text: str
    starts: tuple[int, ...]

    @staticmethod
    def of(name: str, text: str) -> Document:
        # `str.find` scans in C. To enumerate half a gigabyte of characters in
        # the interpreter blocked mount readiness for tens of seconds.
        starts = [0]
        position = text.find("\n")
        while position != -1:
            starts.append(position + 1)
            position = text.find("\n", position + 1)
        if len(starts) > 1 and starts[-1] == len(text):
            # A trailing newline ends the last line. It does not begin another.
            starts.pop()
        return Document(name=name, text=text, starts=tuple(starts))

    @property
    def line_count(self) -> int:
        return len(self.starts)

    def _end_of(self, number: int) -> int:
        """Where 1-based line `number` ends, excluding its terminator."""
        if number < self.line_count:
            return self.starts[number] - 1
        return len(self.text) - 1 if self.text.endswith("\n") else len(self.text)

    def line(self, number: int) -> str:
        """One 1-based line, without its terminator."""
        return self.text[self.starts[number - 1] : self._end_of(number)]

    def lines(self, first: int, last: int) -> str:
        """A 1-based inclusive line range, clamped. A match on line 1 asks for 0."""
        first = max(1, first)
        last = min(self.line_count, last)
        if first > last:
            return ""
        return self.text[self.starts[first - 1] : self._end_of(last)]


class Match(WireModel):
    """One hit, with enough around it to be worth reading."""

    document: str
    line: int
    text: str


class SearchValue(WireModel):
    matches: list[Match] = Field(default_factory=list)
    truncated: bool = False
    """More lines matched than `limit` allowed back. The model can narrow the
    query."""


class ChunkValue(WireModel):
    index: int
    chunks: int
    text: str


class HeadValue(WireModel):
    document: str = ""
    text: str = ""
    manifest: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Corpus:
    """A resolved corpus: the documents, and how it was reached.

    The class is frozen but has no slots, so that `cached_property` has a
    `__dict__` to store into. There is one corpus per mount, and the joined view
    below is the whole cost.
    """

    name: str
    documents: tuple[Document, ...]
    sources: tuple[str, ...]
    digest: str
    missing: tuple[str, ...] = ()

    @property
    def chars(self) -> int:
        return sum(len(document.text) for document in self.documents)

    @property
    def line_count(self) -> int:
        return sum(document.line_count for document in self.documents)

    def document(self, name: str) -> Document | None:
        return next((one for one in self.documents if one.name == name), None)

    def search(self, query: str, *, limit: int, context_lines: int, regex: bool) -> SearchValue:
        """The lines that match `query`, with `context_lines` on either side.

        The regex engine does the scan. `finditer` runs over the whole text of
        each document, with `re.MULTILINE` so that `^` and `$` keep their
        per-line meaning. A hit maps back to its line by bisecting the offsets.
        The work at Python level is therefore per *match*, not per line of a
        corpus that may be enormous. At most `limit` matches are ever built. One
        more matching line sets `truncated` and stops the scan.
        """
        try:
            pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE | re.MULTILINE)
        except re.error as error:
            raise ValueError(f"{query!r} is not a valid regular expression: {error}") from error
        matches: list[Match] = []
        for document in self.documents:
            last_line = 0
            for hit in pattern.finditer(document.text):
                number = bisect_right(document.starts, hit.start())
                if number == last_line:
                    continue  # a second hit on a line already reported
                last_line = number
                if len(matches) >= limit:
                    return SearchValue(matches=matches, truncated=True)
                matches.append(
                    Match(
                        document=document.name,
                        line=number,
                        text=document.lines(number - context_lines, number + context_lines),
                    )
                )
        return SearchValue(matches=matches, truncated=False)

    def _pages(self, size: int) -> list[str]:
        """The joined corpus in pages of at most `size` bytes, cut between
        characters (`ph.characters.pieces`). The pages are computed one time for
        each size that a model asks for, so a walk of the pages is one lookup for
        each page. When the text was sliced by index, "bytes" counted code
        points, and a page could end inside a character."""
        found = self._paged.get(size)
        if found is None:
            found = self._paged[size] = pieces(self._joined.text, size, in_bytes=True)
        return found

    @cached_property
    def _paged(self) -> dict[int, list[str]]:
        """`_pages` by size, filled as sizes are asked for."""
        return {}

    @cached_property
    def _joined(self) -> Document:
        """Every document as one pageable text, each headed by its name.

        The text is built one time, lazily. `chunk` slices it through the offsets
        that `Document` already keeps. To page a large corpus is then arithmetic
        for each call. The alternative is a fresh join and a new split of the
        whole corpus on every page of a walk that exists to visit all of it.
        """
        return Document.of(
            self.name,
            "\n\n".join(f"# {document.name}\n{document.text}" for document in self.documents),
        )

    def chunk(self, *, by: Literal["lines", "bytes"], size: int, index: int) -> ChunkValue:
        """One chunk of the whole corpus, with the documents in declared order.

        The corpus is paged and not returned in one big read. The model asks for
        chunk `n` of `total`, so a walk of the corpus is a sequence of governed
        dispatches that the model can stop. An index past the end is clamped, so
        paging cannot end in an error.
        """
        joined = self._joined
        if by == "bytes":
            paged = self._pages(size)
            total = max(1, len(paged))
            bounded = max(0, min(index, total - 1))
            text = paged[bounded] if paged else ""
        else:
            total = max(1, ceil(joined.line_count / size))
            bounded = max(0, min(index, total - 1))
            text = joined.lines(bounded * size + 1, (bounded + 1) * size)
        return ChunkValue(index=bounded, chunks=total, text=text)


def _resolve(root: Path, name: str, sources: list[ContextSource]) -> Corpus:
    """Read every declared source, and digest what was actually found. This call
    blocks.

    The digest covers the *resolved* set: the path, size and mtime for a file,
    and the content hash for a `text` source. A source that was edited between
    sessions therefore changes the digest, and the model is told that the corpus
    was rebuilt (Q4). Otherwise the model would quietly query something else.
    """
    documents: list[Document] = []
    missing: list[str] = []
    fingerprint = hashlib.sha256()
    for source in sources:
        if source.kind == "text":
            document_name = source.name or "pasted"
            documents.append(Document.of(document_name, source.value))
            fingerprint.update(
                f"text:{document_name}:{hashlib.sha256(source.value.encode()).hexdigest()}\n".encode()
            )
            continue
        # A relative source is resolved against `root`, and never against the
        # working directory of the process. A daemon does not run in the
        # workspace, so a file of the same name beside the process was read in
        # place of the file in the workspace.
        candidate = Path(source.value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            found = [candidate]
        else:
            # `Path.glob` refuses an absolute pattern, and a configured source is
            # as likely to be `/srv/corpus/*.md` as `docs/**/*.md`. The stdlib
            # `glob` module takes both.
            found = sorted(
                path
                for raw in iglob(str(candidate), recursive=True)
                if (path := Path(raw)).is_file()
            )
        if not found:
            missing.append(source.value)
            fingerprint.update(f"missing:{source.value}\n".encode())
            continue
        for path in found:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                stat = path.stat()
            except OSError:
                missing.append(str(path))
                continue
            document_name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            documents.append(Document.of(document_name, text))
            fingerprint.update(f"path:{document_name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return Corpus(
        name=name,
        documents=tuple(documents),
        sources=tuple(source.value for source in sources),
        digest=fingerprint.hexdigest(),
        missing=tuple(missing),
    )


def render_manifest(corpus: Corpus, *, note: str = "") -> str:
    """The prompt section: what is loaded and how to reach it. Never content."""
    lines = [
        "# Loaded context",
        "",
        f"A corpus named `{corpus.name}` is loaded and queryable — "
        f"{len(corpus.documents)} document(s), {corpus.line_count} lines, "
        f"{corpus.chars} characters. **None of it is in this prompt.**",
        "",
    ]
    lines.extend(
        f"- `{document.name}` — {document.line_count} lines, {len(document.text)} characters"
        for document in corpus.documents
    )
    lines.extend(
        [
            "",
            "Reach it from a cell:",
            "",
            f"- `await tools.{SEARCH_TOOL}(query=...)` — matching lines with surrounding context.",
            f"- `await tools.{CHUNKS_TOOL}(by='lines', size=200, index=0)` — page through it.",
            f"- `await tools.{HEAD_TOOL}(document=...)` — the start of one document, or the list.",
            "",
            "Each call is a separate governed dispatch, so ask narrow questions rather than "
            "pulling the corpus into the conversation.",
        ]
    )
    if corpus.missing:
        lines.extend(["", f"Sources that could not be read: {', '.join(corpus.missing)}."])
    if note:
        lines.extend(["", note])
    return "\n".join(lines)


@dataclass(slots=True)
class ContextService:
    """The service published as `ctx.context_corpus`."""

    corpus: Corpus
    _notes: SessionTable[str] = field(default_factory=SessionTable)
    """For each session: the note that the first announce settled. The log stays
    the durable answer, and a resume reads it again through `session.latest`.
    This table only stops every prompt assembly from asking again a question
    whose answer cannot change while this mount lives. An entry goes with its
    session."""
    _rendered: dict[str, str] = field(default_factory=dict)
    """For each note: the rendered manifest, which would otherwise be rebuilt on
    every step."""

    def manifest(self, session: Session | None) -> str:
        """The text of the prompt section: the manifest, plus anything that is new."""
        note = self.announce(session)
        cached = self._rendered.get(note)
        if cached is None:
            cached = self._rendered[note] = render_manifest(self.corpus, note=note)
        return cached

    def announce(self, session: Session | None) -> str:
        """Record the recipe for this session, one time, and return the note.

        The record is appended on the first prompt assembly and not at mount.
        That is when the corpus becomes visible to the model, and I3 says that
        model-visible means logged. The check for an earlier record is against
        the log itself and not against a flag, so a resumed session announces
        again only when the digest actually changed.
        """
        if session is None:
            return ""
        cached = self._notes.get(session)
        if cached is not None:
            return cached
        previous = session.latest(LOADED)
        if previous is not None and as_str(previous.data.get("digest")) == self.corpus.digest:
            return self._notes.keep(session, as_str(previous.data.get("note")))

        note = ""
        if previous is not None:
            # Rebuilt: the sources changed under a session that had already been
            # told about them. Any note is better than a model that finds the
            # corpus answering differently and reads that as its own mistake.
            note = (
                f"`{self.corpus.name}` was rebuilt from changed sources; earlier answers in "
                "this conversation may have come from different content."
            )
        elif self.corpus.missing:
            note = (
                f"Some of `{self.corpus.name}`'s sources are unavailable — "
                f"{', '.join(self.corpus.missing)} could not be read."
            )
        # Only the recipe is recorded: `{loader, sources, digest}`, plus the note
        # about it. The counts in the manifest stay derivable and off the wire.
        _LOG.append(
            session,
            LOADED,
            {
                "corpus": self.corpus.name,
                "loader": "rlm-context-loader",
                "sources": list(self.corpus.sources),
                "digest": self.corpus.digest,
                "note": note,
            },
        )
        return self._notes.keep(session, note)


class SearchArgs(ToolModel):
    """`tools.context_search(...)`."""

    query: str
    limit: int = 20
    context_lines: int = 2
    regex: bool = False


class ChunkArgs(ToolModel):
    """`tools.context_chunks(...)`."""

    by: Literal["lines", "bytes"] = "lines"
    size: int = 200
    index: int = 0


class HeadArgs(ToolModel):
    """`tools.context_head(...)`. No document names the manifest instead."""

    document: str | None = None
    lines: int = 40


def _render_matches(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    matches = value.get("matches") or []
    if not matches:
        return text_content("no matches")
    body = "\n\n".join(f"{match['document']}:{match['line']}\n{match['text']}" for match in matches)
    if value.get("truncated"):
        body = f"{body}\n\n(more matched than were returned; narrow the query)"
    return text_content(body)


def _render_chunk(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    return text_content(f"chunk {value['index'] + 1} of {value['chunks']}\n\n{value['text']}")


def _render_head(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("document"):
        return text_content(f"{value['document']}\n\n{value['text']}")
    manifest = value.get("manifest") or []
    return text_content("\n".join(f"- {row}" for row in manifest) or "the corpus is empty")


@plugin("rlm-context-loader", affects="environment", config=Config, inject=[TOOLS, SYSTEM_PROMPT])
async def apply(ctx: Context, config: Config) -> None:
    """Resolve the corpus, register the three queries, describe it in the prompt.

    The corpus is resolved at mount, in a worker thread because the read blocks.
    The row is opt-in, and the `min_chars` decision (whether to offer the tools
    at all) cannot be made without knowing what the sources hold.
    """
    tools = ctx.require(TOOLS)
    if not config.sources:
        log.debug("ph_rlm.context_loader: no sources configured; standing down")
        return
    # This is the workspace root that the fs seam resolved, so relative sources
    # and document names agree with the paths that `tools.read`/`grep`/`glob`
    # accept. Otherwise a deployment that configures the root of `fs-local` would
    # get a corpus resolved against the process cwd of pH, while every other file
    # tool looks elsewhere.
    root = Path(getattr(ctx.get(FS), "root", Path.cwd()))
    corpus = await anyio.to_thread.run_sync(_resolve, root, config.corpus, config.sources)
    if corpus.chars < config.min_chars:
        # Below the threshold, `tools.read` and `tools.grep` serve the model
        # better. They are governed, logged, offloadable, and already mounted.
        log.info(
            "ph_rlm.context_loader: %s characters is under min_chars=%s; standing down",
            corpus.chars,
            config.min_chars,
        )
        return

    service = ContextService(corpus=corpus)
    ctx.provide(CONTEXT_CORPUS, service)

    def search(args: SearchArgs, _run: object) -> dict[str, Any]:
        return corpus.search(
            args.query,
            limit=min(max(1, args.limit), config.max_matches),
            context_lines=max(0, args.context_lines),
            regex=args.regex,
        ).to_wire()

    def chunks(args: ChunkArgs, _run: object) -> dict[str, Any]:
        return corpus.chunk(by=args.by, size=max(1, args.size), index=args.index).to_wire()

    def head(args: HeadArgs, _run: object) -> dict[str, Any]:
        if args.document is None:
            return HeadValue(
                manifest=[
                    f"{document.name} ({document.line_count} lines)"
                    for document in corpus.documents
                ]
            ).to_wire()
        document = corpus.document(args.document)
        if document is None:
            raise ValueError(f'no document named "{args.document}" is in this corpus')
        return HeadValue(
            document=document.name, text=document.lines(1, max(1, args.lines))
        ).to_wire()

    tools.register(
        define_tool(
            SEARCH_TOOL,
            f"Search the loaded `{corpus.name}` corpus. Returns matching lines with "
            "surrounding context, never the whole corpus.",
            parameters=SearchArgs,
            output=SearchValue,
            render=_render_matches,
            execute=search,
            is_concurrency_safe=True,
        )
    )
    tools.register(
        define_tool(
            CHUNKS_TOOL,
            f"Page through the loaded `{corpus.name}` corpus one chunk at a time.",
            parameters=ChunkArgs,
            output=ChunkValue,
            render=_render_chunk,
            execute=chunks,
            is_concurrency_safe=True,
        )
    )
    tools.register(
        define_tool(
            HEAD_TOOL,
            f"The start of one document in `{corpus.name}`, or the list of documents.",
            parameters=HeadArgs,
            output=HeadValue,
            render=_render_head,
            execute=head,
            is_concurrency_safe=True,
        )
    )

    def section(request: AssembleContext) -> str:
        """Metadata only. The recipe is recorded the first time that the section
        is assembled (I3).

        This is a cached `section` and not a `context()`. The corpus is resolved
        one time at mount, so this text is fixed for the life of the session and
        belongs in the stable prefix (A12).
        """
        return service.manifest(request.session)

    ctx.require(SYSTEM_PROMPT).section(
        PromptSection(name="rlm:context", order=ORDER_CONTEXT, text=section)
    )
