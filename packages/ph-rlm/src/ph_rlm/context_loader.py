"""`rlm-context-loader` — a queryable corpus, off by default (P3-17, Q4).

The companion plan calls this "prompt-as-a-variable": a large body of text the
agent should be able to consult without any of it entering the prompt. Prime
Agent does not implement it. Four decisions make this pH's version rather than
that sketch, and the first two are why the row is small.

**Access is a binding, not a namespace variable** (C2/C5, Q4). A bare
`context.search(...)` over a variable in the kernel produces no `call` frame:
results reach the model as merged cell stdout, capped by `max_log_bytes` and
never offloaded, with no per-query provenance. Registered *tools* — reached as
`await tools.context_search(...)`, because Code Mode builds the `tools` namespace
from the registry — give both: each query settles as its own
`tool/code-dispatch` pair, so `tools/post-execute` can replace one oversized
result with a preview and a spill locator while its siblings pass through
untouched. That is §6a's pattern one level up, and it is the whole argument for
the binding form.

**The prompt gets metadata, never content.** Names, line and byte counts, and
how to query. A corpus large enough to be worth loading is large enough that
putting any of it in the prompt defeats the purpose.

**The corpus is a recipe, not a snapshot** (D17, Q4). It is built from *declared*
sources, so what is durable is `{loader, sources, digest}` — recorded on
`context/loaded` — and a later session re-resolves it. A matching digest is
silent; a changed one tells the model the corpus was rebuilt; a missing source
says so. The plan files this under `kernel/snapshot {kind: "recipe"}`, and that
is the right home for a *kernel variable* the harness built; this corpus never
enters the kernel, precisely because access is a binding — so recording it as a
namespace variable would describe a namespace that does not contain it. Same
mechanism, honest event.

**Below a threshold it stands down.** `min_chars` exists because a corpus small
enough to read with `tools.read` should be read with `tools.read`: C3 gave the
model governed, logged, individually offloadable file access, which is what
demoted this row from a headline feature to a specialist one for non-file
corpora and repeated queries over one fixed set.

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

from ph.cordis import Context, plugin
from ph.session import Session
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, PromptSection
from ph.tools import ToolModel, define_tool, text_content
from ph.wire import WireModel

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
"""The ceiling on one query's `limit`. The offload seam can spill a large
result, but nothing should build an unbounded one first."""


class ContextSource(WireModel):
    """One declared source. `path` may be a glob; `text` is literal."""

    kind: Literal["path", "text"] = "path"
    value: str
    name: str = ""
    """What to call a `text` source in the manifest. Ignored for paths, which are
    named by their own relative path."""


class Config(WireModel):
    """Row config for `rlm-context-loader`."""

    corpus: str = "context"
    sources: list[ContextSource] = Field(default_factory=list)
    min_chars: int = 0
    """Below this the row stands down entirely — no tools, no prompt section.

    `0` loads whatever is configured; the `rlm-stable` profile sets 200 000, the
    threshold Q4 settled on, so a small corpus behaves conventionally."""
    max_matches: int = MAX_MATCHES
    """The most matches one `context_search` call may return, however large the
    model sets `limit`."""


@dataclass(frozen=True, slots=True)
class Document:
    """One document, with its line starts precomputed.

    Offsets rather than a tuple of lines: a corpus worth this row is large, and
    splitting it would hold every line as its own string for the life of the
    process. A line is a slice of `text`.
    """

    name: str
    text: str
    starts: tuple[int, ...]

    @staticmethod
    def of(name: str, text: str) -> Document:
        # `str.find` scans in C; enumerating half a gigabyte of characters in
        # the interpreter blocked mount readiness for tens of seconds.
        starts = [0]
        position = text.find("\n")
        while position != -1:
            starts.append(position + 1)
            position = text.find("\n", position + 1)
        if len(starts) > 1 and starts[-1] == len(text):
            # A trailing newline ends the last line; it does not begin another.
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
        """1-based inclusive line range, clamped — a match on line 1 asks for 0."""
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
    """More matched than `limit` allowed back — the model can narrow the query."""


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

    Frozen but not slotted, so `cached_property` has a `__dict__` to land in —
    there is one corpus per mount, and the joined view below is the whole cost.
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
        """Lines matching `query`, with `context_lines` either side.

        The scan is the regex engine's — `finditer` over each document's whole
        text, `re.MULTILINE` so `^`/`$` keep their per-line meaning — and a hit
        maps back to its line by bisecting the offsets, so Python-level work is
        per *match*, not per line of a possibly enormous corpus. At most `limit`
        matches are ever built; one more matching line sets `truncated` and
        stops.
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

    @cached_property
    def _joined(self) -> Document:
        """Every document as one pageable text, headed by its name.

        Built once, lazily: `chunk` slices it through the offsets `Document`
        already keeps, so paging a large corpus is arithmetic per call rather
        than a fresh join — the first version rebuilt (and, for lines, re-split)
        the whole corpus on every page of a walk that exists to visit all of it.
        """
        return Document.of(
            self.name,
            "\n\n".join(f"# {document.name}\n{document.text}" for document in self.documents),
        )

    def chunk(self, *, by: Literal["lines", "bytes"], size: int, index: int) -> ChunkValue:
        """One chunk of the whole corpus, documents in declared order.

        Paging rather than one big read: the model asks for chunk `n` of `total`,
        so a walk of the corpus is a sequence of governed dispatches it can stop.
        An index past the end is clamped, so paging cannot fail into an error.
        """
        joined = self._joined
        if by == "bytes":
            total = max(1, ceil(len(joined.text) / size))
            bounded = max(0, min(index, total - 1))
            text = joined.text[bounded * size : (bounded + 1) * size]
        else:
            total = max(1, ceil(joined.line_count / size))
            bounded = max(0, min(index, total - 1))
            text = joined.lines(bounded * size + 1, (bounded + 1) * size)
        return ChunkValue(index=bounded, chunks=total, text=text)


def _resolve(root: Path, name: str, sources: list[ContextSource]) -> Corpus:
    """Blocking: read every declared source, and digest what was actually found.

    The digest covers the *resolved* set — path, size and mtime for a file, the
    content hash for a blob — so a source edited between sessions changes it and
    the model is told the corpus was rebuilt (Q4) rather than quietly querying
    something else.
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
        candidate = Path(source.value).expanduser()
        if candidate.is_file():
            found = [candidate]
        else:
            # `Path.glob` refuses an absolute pattern, and a configured source is
            # as likely to be `/srv/corpus/*.md` as `docs/**/*.md`. The stdlib
            # module takes both; relative patterns resolve against the cwd.
            pattern = str(candidate) if candidate.is_absolute() else str(root / source.value)
            found = sorted(
                path for raw in iglob(pattern, recursive=True) if (path := Path(raw)).is_file()
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
    _notes: dict[str, str] = field(default_factory=dict)
    """Per session id: the note settled at first announce. The log stays the
    durable answer — a resume re-reads it through `session.latest` — this only
    stops every prompt assembly from re-asking a question whose answer cannot
    change while this mount lives."""
    _rendered: dict[str, str] = field(default_factory=dict)
    """Per note: the rendered manifest, which is otherwise rebuilt every step."""

    def manifest(self, session: Session | None) -> str:
        """The prompt section's text: the manifest, plus whatever is news."""
        note = self.announce(session)
        cached = self._rendered.get(note)
        if cached is None:
            cached = self._rendered[note] = render_manifest(self.corpus, note=note)
        return cached

    def announce(self, session: Session | None) -> str:
        """Record the recipe for this session, once, and return the note.

        Appended on first prompt assembly rather than at mount, because that is
        when the corpus becomes model-visible — and I3 says model-visible means
        logged. Idempotent against the log itself, not a flag, so a resumed
        session re-announces only when the digest actually moved.
        """
        if session is None:
            return ""
        cached = self._notes.get(session.id)
        if cached is not None:
            return cached
        previous = session.latest(LOADED)
        if previous is not None and str(previous.data.get("digest")) == self.corpus.digest:
            note = str(previous.data.get("note") or "")
            self._notes[session.id] = note
            return note

        note = ""
        if previous is not None:
            # Rebuilt: the sources moved under a session that had already been
            # told about them. Anything is better than the model finding the
            # corpus answering differently and reading it as its own mistake.
            note = (
                f"`{self.corpus.name}` was rebuilt from changed sources; earlier answers in "
                "this conversation may have come from different content."
            )
        elif self.corpus.missing:
            note = (
                f"Some of `{self.corpus.name}`'s sources are unavailable — "
                f"{', '.join(self.corpus.missing)} could not be read."
            )
        # The recipe alone: `{loader, sources, digest}` plus what was said about
        # it. The manifest's counts stay derivable and off the wire.
        session.append(
            LOADED,
            {
                "corpus": self.corpus.name,
                "loader": "rlm-context-loader",
                "sources": list(self.corpus.sources),
                "digest": self.corpus.digest,
                "note": note,
            },
        )
        self._notes[session.id] = note
        return note


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


def _render_matches(_args: Any, value: Any) -> Any:
    matches = value.get("matches") or []
    if not matches:
        return text_content("no matches")
    body = "\n\n".join(f"{match['document']}:{match['line']}\n{match['text']}" for match in matches)
    if value.get("truncated"):
        body = f"{body}\n\n(more matched than were returned; narrow the query)"
    return text_content(body)


def _render_chunk(_args: Any, value: Any) -> Any:
    return text_content(f"chunk {value['index'] + 1} of {value['chunks']}\n\n{value['text']}")


def _render_head(_args: Any, value: Any) -> Any:
    if value.get("document"):
        return text_content(f"{value['document']}\n\n{value['text']}")
    manifest = value.get("manifest") or []
    return text_content("\n".join(f"- {row}" for row in manifest) or "the corpus is empty")


@plugin("rlm-context-loader", config=Config, inject=["tools", "system_prompt"])
async def apply(ctx: Context, config: Config) -> None:
    """Resolve the corpus, register the three queries, describe it in the prompt.

    Resolved at mount, blocking in a worker thread: the row is opt-in, and the
    `min_chars` decision — whether to offer the tools at all — cannot be made
    without knowing what the sources hold.
    """
    if not config.sources:
        log.debug("ph_rlm.context_loader: no sources configured; standing down")
        return
    # The workspace root the fs seam resolved, so relative sources and document
    # names agree with the paths `tools.read`/`grep`/`glob` accept — a deployment
    # that configures `fs-local`'s root would otherwise get a corpus resolved
    # against pH's process cwd while every other file tool looks elsewhere.
    root = Path(getattr(ctx.get("fs"), "root", Path.cwd()))
    corpus = await anyio.to_thread.run_sync(_resolve, root, config.corpus, config.sources)
    if corpus.chars < config.min_chars:
        # Below the threshold the model is better served by `tools.read` and
        # `tools.grep`: governed, logged, offloadable, and already mounted.
        log.info(
            "ph_rlm.context_loader: %s characters is under min_chars=%s; standing down",
            corpus.chars,
            config.min_chars,
        )
        return

    service = ContextService(corpus=corpus)
    ctx.provide("context_corpus", service)

    def search(args: SearchArgs, _run: Any) -> Any:
        return corpus.search(
            args.query,
            limit=min(max(1, args.limit), config.max_matches),
            context_lines=max(0, args.context_lines),
            regex=args.regex,
        ).to_wire()

    def chunks(args: ChunkArgs, _run: Any) -> Any:
        return corpus.chunk(by=args.by, size=max(1, args.size), index=args.index).to_wire()

    def head(args: HeadArgs, _run: Any) -> Any:
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

    ctx.tools.register(
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
    ctx.tools.register(
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
    ctx.tools.register(
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

    def section(request: Any) -> str:
        """Metadata only, and the recipe recorded on first sight (I3).

        A cached `section` rather than a `context()`: the corpus is resolved once
        at mount, so this text is fixed for the life of the session and belongs
        in the stable prefix (A12).
        """
        return service.manifest(getattr(request.agent, "session", None))

    ctx.system_prompt.section(PromptSection(name="rlm:context", order=ORDER_CONTEXT, text=section))
