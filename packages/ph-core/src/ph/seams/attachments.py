"""`ctx.attachments`: media that the log points at but cannot rebuild.

The log is lossless JSON (A1), so an image never enters it. A `MediaBlock`
carries an `AttachmentRef`, and the bytes are stored here under their own
SHA-256.

**Why this is not `ctx.spill_store`, which has the same mechanics.** The two
lifecycles differ in one way, and it is the way that matters. A spilled tool
result is a *forwarding address*. The log already holds the preview that the
model saw, so a lost file costs a reader the way back to the original. An
attachment is content that the log only **points at**. If it is lost, the
conversation is missing a piece that nothing can rebuild. Someone will
eventually clear a cache directory to reclaim space, and that must not be able
to delete conversation. So attachments have their own seam and their own
directory, and the sweep of F7 does not touch them.

**A blob is addressed globally and has no owner directory.** A digest names the
same bytes everywhere. Two sessions that attach the same photo share one file. A
fork references exactly the digests that its parent references, and not a
directory that the parent owns. That is why deleting a parent session does not
break its children. The cost is that collection needs a fold over *every*
session, so nothing here collects automatically. Collection is an explicit
operation (`phern attachments gc`, P7-01). This is deliberately unlike the spill
store, which sweeps each owner.

**Reading a path is the business of the caller, not of this store** (I-9). The
store takes bytes. Who may turn a path into bytes is a security question with
two different answers. A user may attach anything that they can read. A *model*
must go through `ctx.fs`, so that `permissions-fs` and the workspace tier bound
it. A store that read paths itself would answer the question one time, wrongly,
for both. `save_path` exists for the user, and its docstring says so.

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
from typing import TYPE_CHECKING

import anyio

from ..cordis import Context, plugin
from ..json import JsonValue
from ..keys import ATTACHMENTS
from ..llm.dimensions import IMAGE_MIMES, image_dimensions
from ..llm.types import AttachmentRef
from ..paths import default_home_path, write_atomic
from ..wire import WireModel

if TYPE_CHECKING:
    # All three imports are for annotations only. This seam is a plugin entry
    # point, so an unguarded import put `ph.persistence` into every process
    # that touches an attachment. `ph/persistence/__init__.py` imports the
    # JSONL *backend*. So the fold, which exists so that this module imports no
    # backend, was importing one.
    from ..persistence.protocol import SessionArchive
    from ..session import SessionEvent
    from .uploads import UploadRegistry

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
"""The MIME type that pH gives a file that it cannot classify."""

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
"""A map from MIME type to suffix, so that a person who browses the directory
sees files that open.

This is a table and not `mimetypes.guess_extension`, which answers `.jpe` for
JPEG and varies by platform. The name of a stored file is part of what someone
sees when they look in the directory, and it must not depend on which machine
wrote the file. An unknown type gets no suffix. The digest is the identity in
both cases.
"""


def digest_of(content: bytes) -> str:
    """The attachment id for some bytes: `sha256:<hex>`."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


class Config(WireModel):
    """Row config for the local attachment store."""

    root: str | None = None


ENCODED_CACHE_BYTES = 32 * 1024 * 1024
"""How much base64 the store keeps in memory before it evicts the oldest entry.

The cache has a cap and is not an unbounded dict. Base64 is a third larger than
the bytes that it encodes. Without a cap, a session with a handful of PDFs
would hold tens of megabytes of `str` for as long as the process lives.
"""


def mime_of(name: str) -> str:
    """The MIME type that pH gives a file of this name, from the name alone.

    `mime_for` falls back to this guess. The guess decides more than a label.
    `path_for` derives the *extension* of the stored file from the MIME type,
    and `IMAGE_MIMES` decides whether a block reaches the model as an image or
    as a document. So a second copy of this function is how the same PNG becomes
    a document in one front end and an image in another.
    """
    return mimetypes.guess_type(name)[0] or OCTET_STREAM


def mime_for(declared: str | None, name: str) -> str:
    """The MIME type that pH gives a file for which somebody has already declared
    a type.

    **A declared type wins, except when it is the type that says nothing.** A
    browser sends `Content-Type` with every dropped file, and it falls back to
    `application/octet-stream` for any extension that *it* does not recognize.
    A literal "declared wins" would store a `.png` from such a browser as a
    document, give it the wrong extension from `EXTENSIONS`, and miss
    `IMAGE_MIMES`. That is exactly the failure that `mime_of` exists to prevent.
    It would arrive through the one path that has a declared type to override
    the guess with.

    There are three callers, which is why the rule is a function and not an
    `or` at each of them:

    - The `--attach` of a user, through `save_path`. The caller of `save_path`
      declares the type, or else the type is guessed from the path.
    - A browser upload, where the browser declares the type.
    - `attachment/put`, where a client that sends no `mime` at all still sent a
      name.
    """
    if declared and declared != OCTET_STREAM:
        return declared
    return mime_of(name)


async def read_for_attach(source: Path | str) -> tuple[str, str, bytes]:
    """`(name, mime, content)` for a file that a *user* is attaching.

    This one function is the user's way to attach (I-9). It reads with the
    permissions of the caller, makes no policy check, and takes the MIME type
    from `mime_of`. It is exported because two front ends read a file this way:
    the in-process front end through `save_path`, and the socket front end
    before `attachment/put`.
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
        """The path that a reference resolves to. It is derived and never looked up.

        The digest *is* the name, so the path is a pure function of the
        reference and needs no directory scan. The suffix comes from the MIME
        type of the reference itself. So a reference that arrives here from a
        stored log resolves to the same path that the write produced, on any
        machine.
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
        """Store `content` and return the reference that a `MediaBlock` carries.

        The call is idempotent by construction: identical bytes produce one file.

        **Pixel dimensions are measured here when the caller did not supply
        them** (P7-03). Every other measurement argument stays a fact that a
        caller already knows, such as a page count that a PDF reader reported or
        a duration that was parsed from a container. The size of an image is
        different. The header of the image holds it as plain integers, and
        `ph.llm.dimensions` reads them with no dependency. The argument still
        wins when it is given. An ingester that has already decoded the image
        knows at least as much as the header does.

        Not enforced (§5 rule 6): this fills in an attachment when it is
        *stored*. A reference that is already in a log written before P7-03
        keeps `width: None` forever, and neither pixel limit can apply to it. A
        backfill would mean rewriting stored references, which A1 does not
        allow, or measuring at read time, which puts a file read on the request
        path.
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
        await anyio.to_thread.run_sync(_write, path, content)
        return ref

    async def save_path(self, source: Path | str, *, mime: str | None = None) -> AttachmentRef:
        """Read a file directly and store it. **For the user only** (I-9).

        The read uses the permissions of the harness and makes no policy check.
        That is correct for a user who attaches a file that they can already
        open. It is wrong for anything that the model asked for. A tool that
        reached this method would be an exfiltration primitive: it could attach
        a private key as a "document" for a provider to OCR. An attach that the
        model starts reads through `ctx.fs.read_bytes`, which is the same
        `fs/read-intent` gate that every other model-driven read passes. It then
        calls `save_bytes` with the result. `tool-attach` is that caller, and it
        is the only one that this seam expects.
        """
        name, _, content = await read_for_attach(source)
        return await self.save_bytes(content=content, mime=mime_for(mime, name), name=name)

    async def load_bytes(self, ref: AttachmentRef) -> bytes:
        return await anyio.to_thread.run_sync(self.path_for(ref).read_bytes)

    async def load_b64(self, ref: AttachmentRef) -> str:
        """The attachment as base64, encoded one time for each process.

        **The cache cannot go stale. That follows from the naming, and is not a
        promise that this method has to keep.** `attachment_id` is the SHA-256
        of the content, so two references with one id have one body by
        definition. There is no invalidation to get wrong.

        The cache is worth having because the alternative cost is paid on *every
        model step*. A media block stays in derived history for the life of the
        session. So an attachment that was added at the first turn would be read
        from disk and encoded again for every request after it.
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

        A caller asks this before a request is built. A reference whose blob is
        gone must degrade to a pointer that the model can read, and not to a
        wire error that the model cannot act on.
        """
        return self.path_for(ref).is_file()


def _write(path: Path, payload: bytes) -> None:
    """Put one blob at its content-addressed name, atomically (K5).

    The name is the sha256 of these bytes, and that makes both halves correct.
    `skip_if_present` is correct because a file that is already at this name
    already holds the bytes. The atomic write is correct because a truncated
    file would be believed forever, for the same reason.
    """
    write_atomic(path, payload, skip_if_present=True)


@plugin("attachments-local", affects="deployment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local attachment store."""
    root = default_home_path(config.root, "attachments")
    ctx.provide(ATTACHMENTS, AttachmentStore(ctx=ctx, root=root))


# ------------------------------------------------------------- collection --
#
# `phern attachments gc` (P7-01). The fold is here and not in the command, for
# the same reason as `stored_survivors`. The fold is the part that has rules,
# and the command is one way to ask for it. A second asker, such as a
# diagnostic or a daemon sweep that somebody wants later, must not derive again
# which blobs are safe to remove.

DIGEST_TEXT = re.compile(r"sha256:[0-9a-f]{64}")
"""How an attachment id is recognized **in a log, by its shape**.

This is deliberately not a list of the keys that carry an id. Three producers
already spell the key differently:

- The upload records (`attachment/uploading` and `attachment/uploaded`) write
  `attachmentId`.
- The media notices write `attachmentIds` as a list.
- A `user/message` carries the id nested inside the `attachment` of a content
  block.

Each of them was written at a different time by a different row. A key list
here would be a fourth spelling of the same fact, maintained in a module that
none of them import. When that list fell behind, the result would be the one
failure that this command must not have: a blob is quietly collected, because
the row that referenced it used a name that nobody added here.

All of them share the shape of the value, because `digest_of` produces it.
Matching too much costs a wasted file: an unrelated string that happens to be a
SHA-256 digest keeps a blob alive forever. Matching too little costs a session
that will not open again.
"""

MIN_AGE = 86_400.0
"""How long a blob must have been on disk before it may be collected, in seconds.

**This is not a retention policy, and that difference is the design of this
command.** Age never *authorizes* collection here. A blob that a stored log
references is kept however old it is. P7-01 wrote that constraint down before
anything was implemented. This constant covers the window between the write of
a blob and the write of the log that mentions it. A user can drop a file on the
composer and leave it staged for as long as they are composing. During that
time the bytes are on disk, and no `user/message` anywhere references them. To
collect the blob would delete the file from under the prompt that the user is
still typing.

The value is a day. That is the scale of what is waited for, which is the
attention of a person. The cost of waiting is one unreferenced file on disk for
one more day.
"""

LISTING_LIMIT = 100_000
"""How many stored sessions the fold will list.

Every other listing in the harness is a *page*: 50 for the picker, 500 for the
lineage survey. Those listings answer "show me the recent ones". This listing
answers "does anyone at all still need this blob". A bounded answer to that
question is not a smaller answer. It is a wrong answer. The session that fell
below the cut is exactly the old conversation that nobody has opened lately,
and that is the conversation whose media a person would be most upset to lose.

So the limit is set high enough that it is not reached in practice, **and it is
checked**. A survey that reaches the limit reports `truncated`, and a truncated
survey collects nothing. There is a limit at all because a store with more logs
than this has something else wrong with it. To walk such a store silently for
an hour is not the better failure.
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
    """What is stored, what still points at it, and what may be removed.

    Every count is reported, and none is summarized into a verdict. The user who
    reads the survey is deciding whether to pass `--remove`. The number that
    decides it is how much of the store was actually read, and a summary would
    drop that number.
    """

    sessions: int
    """Stored logs whose references were read."""
    collect: tuple[Blob, ...]
    """Unreferenced and old enough. Safe to remove only if `safe` is true."""
    kept: tuple[Blob, ...]
    """Referenced by some stored log. Never collected, at any age."""
    recent: tuple[Blob, ...]
    """Unreferenced but younger than the minimum age of the survey (`MIN_AGE` by
    default). Most likely a staged file."""
    uploads: tuple[Path, ...]
    """Entries of the handle cache for digests that nothing references.

    `ctx.uploads` gives this answer. The survey does not derive it here, because
    the layout belongs to that seam. A collector that rebuilt
    `<provider>/<digest>.json` from the outside would report zero, silently and
    with nothing failing, on the day that the scheme changed.

    These entries are swept under the same reference set as the blobs, but **not
    for the same reason**. Losing an entry costs one upload, and losing a blob
    costs the conversation. That is why these entries are not aged. The cache
    entry of an unreferenced digest is dead as soon as the blob is. There is no
    staged-file window to wait out, because nothing uploads a file before a
    session mentions it."""
    unreadable: tuple[str, ...]
    """Logs that would not read. Their references are unknown, so nothing is safe."""
    truncated: bool
    """The listing hit its limit (`LISTING_LIMIT` by default) or could not be read,
    so "nothing references this" is unproven."""

    @property
    def safe(self) -> bool:
        """Whether the fold saw enough of the store to remove anything.

        **The check fails closed, because the two errors do not cost the same.**
        A log that this build cannot parse may reference any digest in the
        store, and a listing that was cut short hides whole sessions. In both
        cases the fold cannot state that "no session mentions this blob". An
        error in the cautious direction costs disk space. An error in the other
        direction makes a conversation permanently impossible to open. After
        P7-03 the local blob is the last copy, because the copy of a provider
        is behind a handle that expires.
        """
        return not self.unreadable and not self.truncated

    @property
    def collectable_bytes(self) -> int:
        return sum(blob.bytes for blob in self.collect)


def referenced_digests(events: Iterable[SessionEvent]) -> set[str]:
    """Every attachment id that is mentioned anywhere in the payloads of these events.

    The search is over the *payloads* and not over a set of event types. One
    reason is the reason that `DIGEST_TEXT` gives. The other is the surface
    layer: a stored log can carry an event type that this build does not know
    (A3), and an unknown event that referenced a blob would otherwise read as
    no reference at all.
    """
    found: set[str] = set()
    for event in events:
        _walk(event.data, found)
    return found


def _walk(value: JsonValue, found: set[str]) -> None:
    """Collect digest-shaped strings from a lossless-JSON payload.

    The checks are against `Mapping` and `Sequence`, not `dict` and `list`. The
    log freezes its payloads into `MappingProxyType` and tuples, which are
    neither. P4-05 hit the same trap when it read a frozen argument tree. Here
    the function would have found nothing at all in a stored log, and it would
    have passed every test that was written against a live log.
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
    """Every file in the store, as the digest that it is named for.

    The name *is* the identity: `path_for` derives it and never looks it up. So
    this function is the inverse of `path_for` and not a scan for metadata. A
    file whose name is not a digest does not belong to this store and is left
    alone. A note that someone put in the directory is not for this command to
    delete.
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
    store: AttachmentStore,
    persistence: SessionArchive,
    *,
    uploads: UploadRegistry | None = None,
    min_age: float = MIN_AGE,
    limit: int = LISTING_LIMIT,
    now: float | None = None,
) -> AttachmentSurvey:
    """Fold every stored session for the digests that it still points at.

    The parameters have named types and are not duck-typed. That keeps what the
    duck typing was for: no backend reaches this module at run time, and a test
    may still pass a stub. It also means that what the fold asks of each
    argument is checked. `SessionArchive` is the read-only half of
    `SessionPersistence`, so a store satisfies it by being a
    `SessionPersistence`.

    **The reads are unchained, one log at a time.** The fold uses `read_own`
    and not `read`. That is not an optimization. `read_own` is the correct
    primitive for this question, for two reasons. First, the union of
    references over every stored log is the same set as the union over every
    materialized chain. To follow parents would read an ancestor again for each
    descendant and learn nothing new. Second, a chained read *fails* when an
    ancestor is missing. One damaged log would become an unreadable subtree,
    and the fold would refuse a collection that is safe to make.

    The cost is a listing plus one read for each log, which is why nothing
    calls this function on a timer. It is no longer a *search* for each log:
    `StoredSession` carries the family, so each read is a path.
    """
    moment = time() if now is None else now
    unreadable: list[str] = []
    referenced: set[str] = set()
    try:
        listed = list(persistence.stored(limit=limit))
    except Exception:
        log.warning("ph.seams.attachments: could not list stored sessions", exc_info=True)
        # The result is an empty listing with `truncated` set. That is the same
        # refusal that a listing cut short gets. "No sessions" and "could not
        # ask" cannot be told apart from here, and only one of them makes
        # collection safe.
        return AttachmentSurvey(
            sessions=0, collect=(), kept=(), recent=(), uploads=(), unreadable=(), truncated=True
        )
    for entry in listed:
        try:
            # `family` comes from the listing row, so a log is a path and not a
            # directory search. Without it, every session that is not a root (a
            # fork, a segment, the log of a subagent) scanned the store again to
            # find itself. In a busy store those are most of the sessions. The
            # cost then grew with the size of the store and not with the number
            # of logs read.
            _header, events = persistence.read_own(entry.session_id, family=entry.family or None)
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


def collect_attachments(
    survey: AttachmentSurvey,
    uploads: UploadRegistry | None = None,
) -> tuple[int, int]:
    """Remove what the survey cleared, and report `(blobs, upload entries)`.

    The function removes nothing unless the survey is `safe`. The check is here
    and is not left to each caller, because this is the one function in the
    harness that deletes conversation. A guard that a caller can forget is not
    a guard.

    The cache half goes back through `ctx.uploads`, which pairs the unlink with
    forgetting the entry. An unlink from here would leave the memo of that seam
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
