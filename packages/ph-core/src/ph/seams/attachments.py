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
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..llm.dimensions import IMAGE_MIMES, image_dimensions
from ..llm.types import AttachmentRef
from ..paths import default_home_path
from ..wire import WireModel

__all__ = [
    "ENCODED_CACHE_BYTES",
    "EXTENSIONS",
    "AttachmentStore",
    "apply",
    "digest_of",
    "read_for_attach",
]

log = logging.getLogger("ph.seams.attachments")

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


async def read_for_attach(source: Path | str) -> tuple[str, str, bytes]:
    """`(name, mime, content)` for a file a *person* is attaching.

    The human door (I-9), as one function: reads with the caller's own
    permissions, no policy check, mime by name with `application/octet-stream`
    as the honest fallback. Exported because two front ends read a file this way
    — the in-process one through `save_path`, the socket one before
    `attachment/put` — and a rule about how a person's file is classified must
    have one author, or the same PNG is a document in one terminal and an image
    in another.
    """
    path = Path(source).expanduser()
    content = await anyio.to_thread.run_sync(path.read_bytes)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return path.name, mime, content


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
        provider to OCR. A model-initiated attach reads through `ctx.fs` and calls
        `save_bytes` — noting that `ctx.fs` has no binary read today, which is a
        gap that lands with P7-01's tool producer rather than being papered over
        here.
        """
        name, guessed, content = await read_for_attach(source)
        return await self.save_bytes(content=content, mime=mime or guessed, name=name)

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
