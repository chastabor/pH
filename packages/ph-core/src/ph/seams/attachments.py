"""`ctx.attachments` — media the log points at but cannot reconstruct.

The log is lossless JSON (A1), so an image never enters it. A `MediaBlock`
carries an `AttachmentRef` and the bytes live here, named by their own SHA-256.

**Why this is not `ctx.spill_store`, which has the same mechanics.** The
lifecycles differ in the one way that matters. A spilled tool result is a
*forwarding address*: the log already holds the preview the model saw, and
losing the file costs a reader the way back to the original. An attachment is
content the log only **points at** — lose it and the conversation is missing a
piece nothing can rebuild. Someone will eventually clear a cache directory to
reclaim space, and that must not be able to delete conversation. So: its own
seam, its own directory, and no participation in F7's sweep.

**Addressed globally, with no owner directory.** A digest is a digest: two
sessions attaching the same photo share one file, and a fork references exactly
the digests its parent does rather than a directory the parent owns — which is
what stops deleting a parent session from breaking its children. The cost is
that collection needs a fold over *every* session, so nothing here collects
automatically; that is an explicit operation (P7-01's `ph attachments gc`),
deliberately unlike the spill store's per-owner sweep.

**Reading a path is the caller's business, not this store's** (I-9). The store
takes bytes. Who is allowed to turn a path into bytes is a security question with
two different answers — a person may attach anything they can read, a *model*
must go through `ctx.fs` so `permissions-fs` and the workspace tier bound it —
and a store that read paths itself would answer it once, wrongly, for both.
`save_path` exists for the human door and says so.

@module ph.seams.attachments
"""

from __future__ import annotations

import base64
import hashlib
import logging
import mimetypes
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..llm.dimensions import IMAGE_MIMES, image_dimensions
from ..llm.types import AttachmentRef
from ..paths import default_home_path
from ..wire import WireModel

__all__ = [
    "ENCODED_CACHE_BYTES",
    "EXTENSIONS",
    "LISTING_LIMIT",
    "MIN_AGE",
    "OCTET_STREAM",
    "AttachmentStore",
    "AttachmentSurvey",
    "Blob",
    "apply",
    "collect_attachments",
    "digest_of",
    "mime_for",
    "mime_of",
    "read_for_attach",
    "referenced_digests",
    "survey_attachments",
]

log = logging.getLogger("ph.seams.attachments")

OCTET_STREAM = "application/octet-stream"
"""What a file pH cannot classify is called."""

EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "video/mp4": ".mp4",
    "text/plain": ".txt",
}
"""MIME → suffix, so a person browsing the directory sees openable files.

A table rather than `mimetypes.guess_extension`, which answers `.jpe` for JPEG
and varies by platform — a stored file's name is part of what someone sees when
they go looking, and it should not depend on which machine wrote it. An unknown
type simply gets no suffix; the digest is the identity either way.
"""


def digest_of(content: bytes) -> str:
    """The attachment id for some bytes: `sha256:<hex>`."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class Config(WireModel):
    """Row config for the local attachment store."""

    root: str | None = None


ENCODED_CACHE_BYTES = 32 * 1024 * 1024
"""How much base64 the store keeps in memory before evicting the oldest.

A cap rather than an unbounded dict: base64 is a third larger than the bytes it
encodes, so a session with a handful of PDFs would otherwise hold tens of
megabytes of `str` for as long as the process lives.
"""


def mime_of(name: str) -> str:
    """What pH calls a file of this name, going only by the name.

    The guess half of the ladder. It decides more than a label: `path_for`
    derives the stored file's *extension* from the mime, and `IMAGE_MIMES`
    decides whether a block reaches the model as an image or as a document — so
    a second copy of it is how the same PNG becomes a document in one front end
    and an image in another.
    """
    return mimetypes.guess_type(name)[0] or OCTET_STREAM


def mime_for(declared: str | None, name: str) -> str:
    """What pH calls a file somebody has already named a type for.

    **A declared type wins, except when it is the one that says nothing.** A
    browser sends `Content-Type` with every dropped file and falls back to
    `application/octet-stream` for any extension *it* does not recognise — so a
    literal "declared wins" stores a `.png` from such a browser as a document,
    gives it the wrong extension out of `EXTENSIONS`, and misses `IMAGE_MIMES`.
    That is exactly the failure `mime_of` exists to prevent, arriving through the
    one door that has an opinion to override it with.

    Three callers, which is why the combination is a function rather than an
    `or` at each of them: a person's `--attach` (declared by the caller, guessed
    from the path), a browser upload (declared by the browser), and
    `attachment/put`, where a client that sends no `mime` at all still sent a
    name.
    """
    if declared and declared != OCTET_STREAM:
        return declared
    return mime_of(name)


async def read_for_attach(source: Path | str) -> tuple[str, str, bytes]:
    """`(name, mime, content)` for a file a *person* is attaching.

    The human door (I-9), as one function: reads with the caller's own
    permissions, no policy check, mime by `mime_of`. Exported because two front
    ends read a file this way — the in-process one through `save_path`, the
    socket one before `attachment/put`.
    """
    path = Path(source).expanduser()
    content = await anyio.to_thread.run_sync(path.read_bytes)
    return path.name, mime_of(path.name), content


@dataclass(slots=True)
class AttachmentStore:
    """The service published as `ctx.attachments`."""

    ctx: Context
    root: Path
    _encoded: OrderedDict[str, str] = field(default_factory=OrderedDict)
    _encoded_bytes: int = 0

    def path_for(self, ref: AttachmentRef) -> Path:
        """Where a reference resolves — derived, never looked up.

        The digest *is* the name, so this is a pure function of the ref and needs
        no directory scan. The suffix comes from the ref's own MIME, which means
        a ref that reaches here from a stored log resolves to the same path the
        write produced, on any machine.
        """
        digest = ref.attachment_id.partition(":")[2] or ref.attachment_id
        return self.root / f"{digest}{EXTENSIONS.get(ref.mime, '')}"

    async def save_bytes(
        self,
        *,
        content: bytes,
        mime: str,
        name: str | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_ms: int | None = None,
        pages: int | None = None,
    ) -> AttachmentRef:
        """Store `content` and return the reference a `MediaBlock` carries.

        Idempotent by construction: identical bytes produce one file.

        **Pixel dimensions are measured here when the caller did not supply
        them** (P7-03). Every other measurement argument stays a fact a caller
        already knows — a page count a PDF reader reported, a duration a
        container parsed — but an image's size is four integers at a fixed offset
        in its own header, which `ph.llm.dimensions` reads with no dependency.
        The argument still wins when given: an ingester that has already decoded
        the image knows at least as much as its header does.

        Not enforced (§5 rule 6): this fills in an attachment as it is *stored*,
        so a reference already in a log written before P7-03 keeps `width: None`
        for ever, and neither pixel ceiling can fire for it. Backfilling would
        mean rewriting stored references, which A1 does not allow, or measuring
        at read time, which puts a file read on the request path.
        """
        if width is None and height is None and mime in IMAGE_MIMES:
            measured = image_dimensions(content)
            if measured is not None:
                width, height = measured
        ref = AttachmentRef(
            attachment_id=digest_of(content),
            mime=mime,
            bytes=len(content),
            name=name,
            width=width,
            height=height,
            duration_ms=duration_ms,
            pages=pages,
        )
        path = self.path_for(ref)
        await anyio.to_thread.run_sync(_write, self.root, path, content)
        return ref

    async def save_path(self, source: Path | str, *, mime: str | None = None) -> AttachmentRef:
        """Read a file directly and store it — **the human door only** (I-9).

        Reads with the harness's own permissions and no policy check, which is
        correct for a person attaching a file they can already open and wrong for
        anything the model asked for: a tool that reached this would be an
        exfiltration primitive, attaching a private key as a "document" for a
        provider to OCR. A model-initiated attach reads through
        `ctx.fs.read_bytes` — which is the same `fs/read-intent` gate every other
        model-driven read passes — and calls `save_bytes` with what comes back.
        `tool-attach` is that caller, and it is the only one this seam expects.
        """
        name, _, content = await read_for_attach(source)
        return await self.save_bytes(content=content, mime=mime_for(mime, name), name=name)

    async def load_bytes(self, ref: AttachmentRef) -> bytes:
        return await anyio.to_thread.run_sync(self.path_for(ref).read_bytes)

    async def load_b64(self, ref: AttachmentRef) -> str:
        """The attachment as base64, encoded once per process.

        **The cache cannot go stale, and that is a property of the naming rather
        than a promise this has to keep.** `attachment_id` is the SHA-256 of the
        content, so two refs with one id have one body by definition — there is
        no invalidation to get wrong.

        Worth having because the alternative is paid on *every model step*: a media
        block stays in derived history for the life of the session, so an attachment
        added at the first turn would be re-read from disk and re-encoded for every
        request after it.
        """
        cached = self._encoded.get(ref.attachment_id)
        if cached is not None:
            self._encoded.move_to_end(ref.attachment_id)
            return cached
        encoded = base64.b64encode(await self.load_bytes(ref)).decode("ascii")
        self._encoded[ref.attachment_id] = encoded
        self._encoded_bytes += len(encoded)
        while self._encoded_bytes > ENCODED_CACHE_BYTES and len(self._encoded) > 1:
            _, evicted = self._encoded.popitem(last=False)
            self._encoded_bytes -= len(evicted)
        return encoded

    def exists(self, ref: AttachmentRef) -> bool:
        """Whether the bytes are still there.

        Worth asking before a request is built: a reference whose blob is gone
        must degrade to a pointer the model can read rather than a wire error it
        cannot act on.
        """
        return self.path_for(ref).is_file()


def _write(directory: Path, path: Path, payload: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Content-addressed, so an existing file with this name has these bytes.
        # Rewriting it would be one more chance to truncate something a live
        # session is reading for no gain.
        return
    path.write_bytes(payload)


@plugin("attachments-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local attachment store."""
    root = default_home_path(config.root, "attachments")
    ctx.provide("attachments", AttachmentStore(ctx=ctx, root=root))


# ------------------------------------------------------------- collection --
#
# `ph attachments gc` (P7-01). Here rather than in the command for
# `stored_survivors`' reason: the fold is the part with rules, the command is a
# way to ask for it, and a second asker — a diagnostic, a daemon sweep somebody
# later wants — must not re-derive which blobs are safe to remove.

DIGEST_TEXT = re.compile(r"sha256:[0-9a-f]{64}")
"""How an attachment id is recognised **in a log, by its shape**.

Deliberately not a list of the keys that carry one. Three producers already spell
it differently — `attachment/uploaded` writes `attachmentId`, the media notices
write `attachmentIds` as a list, and a `user/message` carries it nested inside a
content block's `attachment` — and every one of those was written at a different
time by a different row. A key list here would be a fourth spelling of the same
fact, maintained in a module none of them import, and the failure when it fell
behind is the one failure this command may not have: a blob quietly collected
because the row that referenced it used a name nobody added here.

The value's shape is what all of them share, because `digest_of` produces it. The
cost of matching too much is that an unrelated string that happens to be a
SHA-256 digest keeps a blob alive for ever, which is a wasted file; the cost of
matching too little is a session that will not open again.
"""

MIN_AGE = 86_400.0
"""How long a blob must have been on disk before it may be collected, in seconds.

**Not a retention policy, and the difference is the whole design of this
command.** Age never *authorises* collection here — a blob a stored log
references is kept however old it is, which is the constraint P7-01 wrote down
before anything was implemented. What this covers is the window between a blob
being written and the log that mentions it being written: a person can drop a
file on the composer and leave it staged for as long as they are composing, and
during that time the bytes are on disk with no `user/message` referencing them
anywhere. Collecting it would delete the file out from under the prompt they are
still typing.

A day, because that is the scale of the thing being waited for — a person's
attention — and because the cost of waiting is one unreferenced file on disk for
one more day.
"""

LISTING_LIMIT = 100_000
"""How many stored sessions the fold will list.

Every other listing in the harness is a *page* — 50 for the picker, 500 for the
lineage survey — because they answer "show me the recent ones". This one answers
"is there anyone at all who still needs this blob", and a bounded answer to that
question is not a smaller answer, it is a wrong one: the session that fell below
the cut is exactly the old conversation nobody has opened lately, which is the
one whose media a person would be most upset to lose.

So it is set high enough not to be reached in practice **and checked**: a survey
that hits it reports `truncated`, and a truncated survey collects nothing. A cap
at all, rather than none, because a store with more logs than this has something
else wrong with it and walking it silently for an hour is not the better failure.
"""


@dataclass(frozen=True, slots=True)
class Blob:
    """One stored file, and what the fold needs to decide about it."""

    path: Path
    digest: str
    bytes: int
    age: float
    """Seconds since it was last written, against the fold's own clock."""


@dataclass(frozen=True, slots=True)
class AttachmentSurvey:
    """What is stored, what still points at it, and what may go.

    Every count is reported rather than summarised into a verdict, because the
    person reading it is deciding whether to pass `--remove` and the number that
    decides it — how much of the store was actually read — is the one a summary
    would drop.
    """

    sessions: int
    """Stored logs whose references were read."""
    collect: tuple[Blob, ...]
    """Unreferenced, old enough, and safe to remove — if `safe`."""
    kept: tuple[Blob, ...]
    """Referenced by some stored log. Never collected, at any age."""
    recent: tuple[Blob, ...]
    """Unreferenced but younger than `MIN_AGE` — a staged file, most likely."""
    uploads: tuple[Path, ...]
    """Handle-cache entries for digests nothing references — `ctx.uploads`' answer.

    Asked of that seam rather than derived here, because the layout is its own:
    a collector that rebuilt `<provider>/<digest>.json` from the outside would
    report zero, silently and with nothing failing, the day the scheme changed.

    Swept under the same reference set as the blobs but **not for the same
    reason** — losing an entry costs one upload and losing a blob costs the
    conversation. That is why these are not aged: an unreferenced digest's cache
    entry is dead the moment the blob is, and there is no staged-file window to
    wait out because nothing uploads a file before a session mentions it."""
    unreadable: tuple[str, ...]
    """Logs that would not read. Their references are unknown, so nothing is safe."""
    truncated: bool
    """The listing hit `LISTING_LIMIT`, so "nothing references this" is unproven."""

    @property
    def safe(self) -> bool:
        """Whether the fold saw enough of the store to remove anything.

        **Fail closed, and the asymmetry is the reason.** A log this build cannot
        parse may reference any digest in the store, and a listing that was cut
        short hides whole sessions — in both cases "no session mentions this
        blob" is a statement the fold is not entitled to make. Being wrong in the
        cautious direction costs disk; being wrong in the other direction makes a
        conversation permanently unopenable, because after P7-03 the local blob
        is the last copy — a provider's is behind a handle that expires.
        """
        return not self.unreadable and not self.truncated

    @property
    def collectable_bytes(self) -> int:
        return sum(blob.bytes for blob in self.collect)


def referenced_digests(events: Iterable[Any]) -> set[str]:
    """Every attachment id mentioned anywhere in these events' payloads.

    Over the *payloads* rather than over a set of event types, for `DIGEST_TEXT`'s
    reason and one more: the surface layer means a stored log can carry an event
    type this build does not know (A3), and an unknown event that referenced a
    blob would otherwise read as no reference at all.
    """
    found: set[str] = set()
    for event in events:
        _walk(event.data, found)
    return found


def _walk(value: Any, found: set[str]) -> None:
    """Collect digest-shaped strings from a lossless-JSON payload.

    `Mapping` and `Sequence` rather than `dict` and `list`: the log freezes its
    payloads into `MappingProxyType` and tuples, which are neither — the same trap
    P4-05 hit reading a frozen argument tree, and here it would have found nothing
    at all in a stored log while passing every test written against a live one.
    """
    if isinstance(value, str):
        found.update(DIGEST_TEXT.findall(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            _walk(item, found)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _walk(item, found)


def _blobs(root: Path, now: float) -> list[Blob]:
    """Every file in the store, as the digest it is named for.

    The name *is* the identity — `path_for` derives it and never looks it up — so
    this is the inverse of that function rather than a scan for metadata. A file
    whose name is not a digest is not one of ours and is left alone: someone's
    note in the directory is not this command's to delete.
    """
    if not root.is_dir():
        return []
    found: list[Blob] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        digest = f"sha256:{path.name.partition('.')[0]}"
        if not DIGEST_TEXT.fullmatch(digest):
            continue
        try:
            stat = path.stat()
        except OSError:  # pragma: no cover - raced deletion
            continue
        found.append(
            Blob(path=path, digest=digest, bytes=stat.st_size, age=max(0.0, now - stat.st_mtime))
        )
    return found


def survey_attachments(
    store: Any,
    persistence: Any,
    *,
    uploads: Any = None,
    min_age: float = MIN_AGE,
    limit: int = LISTING_LIMIT,
    now: float | None = None,
) -> AttachmentSurvey:
    """Fold every stored session for the digests it still points at.

    Duck-typed on both stores like `stored_survivors`, so the fold has no reason
    to import a backend and a test can hand it a stub.

    **Unchained reads, one log at a time.** `read_own` rather than `read` is not
    an optimisation, it is the correct primitive for this question twice over: the
    union of references over every stored log is the same set as the union over
    every materialised chain, so following parents would re-read an ancestor once
    per descendant to learn nothing new — and a chained read *fails* when an
    ancestor is missing, which would turn one damaged log into an unreadable
    subtree and refuse a collection that is perfectly safe to make.

    The cost is a listing plus one read per log, and for the JSONL backend a
    directory search per log on top, since `StoredSession` does not carry the
    family a direct path would need. That is why nothing calls this on a timer.
    """
    moment = time() if now is None else now
    unreadable: list[str] = []
    referenced: set[str] = set()
    try:
        listed = list(persistence.stored(limit=limit))
    except Exception:
        log.warning("ph.seams.attachments: could not list stored sessions", exc_info=True)
        # An empty listing and `truncated` — the same refusal a cut-short one
        # gets, because "no sessions" and "could not ask" are indistinguishable
        # from here and only one of them makes collection safe.
        return AttachmentSurvey(
            sessions=0, collect=(), kept=(), recent=(), uploads=(), unreadable=(), truncated=True
        )
    for entry in listed:
        try:
            _header, events = persistence.read_own(entry.session_id)
        except Exception:
            log.warning(
                "ph.seams.attachments: could not read session %s", entry.session_id, exc_info=True
            )
            unreadable.append(entry.session_id)
            continue
        referenced |= referenced_digests(events)
    kept, recent, collect = [], [], []
    for blob in _blobs(store.root, moment):
        if blob.digest in referenced:
            kept.append(blob)
        elif blob.age < min_age:
            recent.append(blob)
        else:
            collect.append(blob)
    return AttachmentSurvey(
        sessions=len(listed) - len(unreadable),
        collect=tuple(collect),
        kept=tuple(kept),
        recent=tuple(recent),
        uploads=() if uploads is None else tuple(uploads.stale(referenced)),
        unreadable=tuple(unreadable),
        truncated=len(listed) >= limit,
    )


def _stale_uploads(uploads: Any, referenced: set[str]) -> tuple[Path, ...]:
    """Handle-cache entries for blobs nothing points at any more (P7-03).

    The other half of the same question, and the reason `ctx.uploads` says a sweep
    belongs here: an entry is written per `(provider, digest)` ever uploaded and
    stays after the provider has forgotten the file, so that directory grows with
    every distinct attachment a deployment has ever sent.

    Swept under the same reference set but **not for the same reason** — losing an
    entry costs one upload, losing a blob costs the conversation. That is why
    these are not aged: an unreferenced digest's cache entry is dead the moment
    the blob is, and there is no staged-file window to wait out because nothing
    uploads a file before a session mentions it.
    """
    root = getattr(uploads, "root", None)
    if not isinstance(root, Path) or not root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(root.rglob("*.json"))
        if f"sha256:{path.stem}" not in referenced and DIGEST_TEXT.fullmatch(f"sha256:{path.stem}")
    )


def collect_attachments(survey: AttachmentSurvey, uploads: Any = None) -> tuple[int, int]:
    """Remove what the survey cleared, and report `(blobs, upload entries)`.

    Refuses everything unless the survey is `safe`, rather than leaving that check
    to each caller: this is the one function in the harness that deletes
    conversation, and a guard a caller can forget is not a guard.

    The cache half goes back through `ctx.uploads`, which pairs the unlink with
    forgetting the entry — unlinking from here would leave that seam's memo
    answering with a handle whose file this command had just removed.
    """
    if not survey.safe:
        return (0, 0)
    blobs = 0
    for blob in survey.collect:
        try:
            blob.path.unlink()
        except OSError:  # pragma: no cover - raced deletion
            continue
        blobs += 1
    return (blobs, 0 if uploads is None else uploads.prune(survey.uploads))
