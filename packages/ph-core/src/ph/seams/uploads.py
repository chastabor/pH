"""`ctx.uploads` — a provider's copy of an attachment, and the handle for it (P7-03).

Some things cannot be sent inline. A provider's file API takes the bytes once and
gives back an id to reference on every later request — which video effectively
requires everywhere, and which large documents are worth on any route that offers
it, because a 4 MB PDF re-encoded as base64 on every step of a fifty-step session
is the same 5.5 MB uploaded fifty times.

That makes a `(provider, digest) → handle` map the one piece of genuinely
non-derivable state in this phase, and **where it lives is this row's real
question.** "State lives in the log" is the harness's default and departing from
it needs the argument written down, so:

* **The handle is not in the log, and the upload is.** A handle is a *prediction*
  — "this id will work until roughly T" — and predictions go stale on their own
  schedule; an append-only log (A1) that recorded one would be asserting
  something false the moment a provider expired it early, with no way to take it
  back. That bytes left this machine for a named provider is a *fact*, it is
  privacy-relevant, and it is exactly what a person auditing a session needs, so
  it is appended as `attachment/uploaded`.
* **It is keyed on the digest, not on the session.** Two sessions attaching one
  photo share one blob already; making them share one upload follows from the
  same naming. A per-session record would make one session's log the authority
  for another session's uploads, which is a coupling neither session asked for.
* **Losing it costs an upload, not information.** Every entry is reconstructible
  from bytes the store still holds, which is the definition of a cache — so it
  belongs beside the other derived state under `$PH_CACHE`, where a person can
  delete the directory and get nothing worse than one slow turn.

**One file per entry, not an index.** The daemon runs many roots in one process
and several processes may share a `$PH_CACHE`; a single index is a write
contended by all of them and a corrupt read that loses every handle at once. A
content-addressed file per `(provider, digest)` has the same failure mode as the
blob store it mirrors — one entry, recoverable by re-uploading.

**Expiry is checked twice, and the second time is the one that matters.** A
handle carries what the provider said, so an entry past its own `expires_at` is
re-uploaded before a request is built. But providers expire files early, delete
them from another session, or forget them; that shows up as a failure *mid-turn*,
which `invalidate_handle` plus the retry waterfall turns into one more attempt
rather than a lost turn.

Not enforced (§5 rule 6): **nothing prunes this directory.** An entry is written
per `(provider, digest)` ever uploaded and stays after the provider has forgotten
the file, so the cache grows with the number of distinct attachments a deployment
has ever sent. Each entry is a few hundred bytes and the directory is safe to
delete wholesale, but a sweep belongs with `ph attachments gc` (P7-01's loose
end), which has the same shape of question about the blobs themselves.

@module ph.seams.uploads
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import anyio

from ..cordis import Context, Disposer, Running, plugin, running
from ..llm.types import AttachmentRef
from ..paths import resolve_roots
from ..session import now_ms
from ..wire import WireModel
from ._registry import claim_key

__all__ = ["FileHandle", "UploadRegistry", "Uploader", "apply", "record_uploaded"]

log = logging.getLogger("ph.seams.uploads")


class FileHandle(WireModel):
    """What a provider gave back for one attachment's bytes."""

    provider: str
    attachment_id: str
    handle: str
    """The provider's own id, opaque here and passed back verbatim on the wire."""
    uploaded_at: int
    expires_at: int | None = None
    """When the provider said it stops working, or `None` when it did not say.

    `None` is "no expiry announced" rather than "never expires" — the difference
    matters because the second would be a promise this file cannot keep for any
    provider. Either way a mid-turn failure invalidates the entry, which is the
    check that does not depend on anyone having told the truth."""


class Uploader(Protocol):
    """A route's file API, as the seam needs it."""

    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle: ...


@dataclass(frozen=True, slots=True)
class _Registered:
    """An uploader and who registered it (P6-29).

    The same shape `AdapterHandle` carries and for the same reason: `upload` is
    row code this seam invokes *later*, so it has to run bound to the row that
    contributed it rather than to whatever happened to be running when a request
    needed a file.
    """

    uploader: Uploader
    by: Running


def record_uploaded(session: Any, handle: FileHandle, ref: AttachmentRef) -> None:
    """Append the fact that bytes left this machine.

    The half of this row that *is* the log's business: not the handle, which
    expires, but that a named provider was given this file. A person auditing
    where their data went reads this; nothing reads it back to find a handle.
    """
    session.append(
        "attachment/uploaded",
        {
            "provider": handle.provider,
            "attachmentId": ref.attachment_id,
            "mime": ref.mime,
            "name": ref.name,
            "bytes": ref.bytes,
            "expiresAt": handle.expires_at,
        },
    )


class Config(WireModel):
    """Row config for the handle cache."""

    root: str | None = None
    """Where the entries live, else `$PH_CACHE/uploads`.

    A settable path like every other storage row, so a deployment can put this on
    a different volume without moving `$PH_CACHE` wholesale. Spelled out rather
    than through `default_home_path`, which every sibling uses: that helper roots
    at `$PH_HOME`, and this is the first row whose default belongs under the
    *cache* — which is the whole argument at the top of this file, so borrowing a
    helper that says `home` would quietly contradict it."""


@dataclass(slots=True)
class UploadRegistry:
    """The service published as `ctx.uploads`."""

    ctx: Context
    root: Path
    uploaders: dict[str, _Registered] = field(default_factory=dict)
    _memo: dict[tuple[str, str], FileHandle] = field(default_factory=dict)
    """Entries read this process, so a handle is read from disk once.

    `load_b64`'s reason exactly: an attachment stays in derived history for the
    life of the session, so without this the same small JSON file is opened and
    parsed on every step of every turn. Cleared by `invalidate_handle`, which is
    the only thing that makes an entry wrong."""

    def register_uploader(self, provider: str, uploader: Uploader) -> Disposer:
        """Claim the file API for one provider.

        Per provider rather than one slot, because a deployment routing to two of
        them is ordinary and their file APIs have nothing in common.
        """
        by = self.ctx.running_for()
        return claim_key(
            by.owner, self.uploaders, provider, _Registered(uploader, by), label="uploader"
        )

    def path_for(self, provider: str, attachment_id: str) -> Path:
        digest = attachment_id.partition(":")[2] or attachment_id
        return self.root / provider / f"{digest}.json"

    def cached(self, provider: str, attachment_id: str) -> FileHandle | None:
        """The stored handle, if there is one and it has not announced its own end.

        A cache read that cannot raise: a truncated or hand-edited entry is
        treated as absent, because the remedy for both is the same upload and
        failing a turn over an unreadable *cache* would be absurd.
        """
        handle = self._memo.get((provider, attachment_id))
        if handle is None:
            path = self.path_for(provider, attachment_id)
            try:
                handle = FileHandle.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return None
            self._memo[(provider, attachment_id)] = handle
        if handle.expires_at is not None and handle.expires_at <= now_ms():
            return None
        return handle

    def invalidate_handle(self, provider: str, handle: str) -> None:
        """Forget the entry holding this provider id, wherever it is.

        The adapter that meets a dead handle mid-request knows the *file id*, not
        the digest it was made from — the request carries one and not the other.
        The memo answers it without touching the disk in the ordinary case; the
        scan is the cold path, after a restart, and it stops at the match.

        Called for the one handle the provider *named*, never for every handle a
        request referenced: a twenty-file request that threw away nineteen live
        uploads to replace one dead one would re-upload nineteen files to fix a
        problem it did not have.
        """
        for key, stored in list(self._memo.items()):
            if key[0] == provider and stored.handle == handle:
                del self._memo[key]
                self.path_for(*key).unlink(missing_ok=True)
                return
        directory = self.root / provider
        if not directory.is_dir():
            return
        for path in directory.glob("*.json"):
            try:
                stored = FileHandle.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            if stored.handle == handle:
                path.unlink(missing_ok=True)
                return

    async def handle_for(
        self, ref: AttachmentRef, *, provider: str, session: Any = None
    ) -> FileHandle | None:
        """The handle to reference these bytes by, uploading them if needed.

        `None` when this provider has no uploader — not an error, because the
        caller's alternative is to send the bytes inline, which is what every
        route did before this row existed.

        `session` is where the *fact* goes when an upload actually happens.
        Recorded here rather than by the caller because this is the one place
        that knows the difference between a cache hit and bytes leaving the
        machine, and it is that difference a person is owed.
        """
        cached = self.cached(provider, ref.attachment_id)
        if cached is not None:
            return cached
        entry = self.uploaders.get(provider)
        store = self.ctx.get("attachments")
        if entry is None or store is None:
            return None
        content = await store.load_bytes(ref)
        with running(entry.by):
            handle = await entry.uploader.upload(ref, content)
        await self._store(handle)
        if session is not None:
            record_uploaded(session, handle, ref)
        return handle

    async def _store(self, handle: FileHandle) -> None:
        self._memo[(handle.provider, handle.attachment_id)] = handle
        path = self.path_for(handle.provider, handle.attachment_id)
        payload = json.dumps(handle.to_wire())
        await anyio.to_thread.run_sync(_write, path, payload)


def _write(path: Path, payload: str) -> None:
    """Write one entry atomically.

    Temp-and-rename because a reader in another process must see either the old
    entry or the new one — a half-written handle read as JSON is the corruption
    `cached` has to treat as absent, and there is no reason to manufacture it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


@plugin("uploads-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the handle cache. Uploaders come from the adapter rows."""
    root = Path(config.root).expanduser() if config.root else resolve_roots().cache / "uploads"
    ctx.provide("uploads", UploadRegistry(ctx=ctx, root=root))
