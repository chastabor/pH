"""`ctx.uploads`: the copy of an attachment that a provider holds, and the handle
for it (P7-03).

Some things cannot be sent inline. The file API of a provider takes the bytes
one time and gives back an id to reference on every later request. Video
effectively requires this everywhere. Large documents are worth it on any route
that offers it, because a 4 MB PDF that is encoded again as base64 on every step
of a fifty-step session is the same 5.5 MB uploaded fifty times.

So a map from `(provider, digest)` to a handle is the one piece of state in this
phase that genuinely cannot be derived, and **where it lives is the real
question of this row.** "State lives in the log" is the default of the harness,
and a departure from it needs the argument written down:

* **The handle is not in the log, and the upload is.** A handle is a
  *prediction*: "this id will work until roughly T". A prediction goes stale on
  its own schedule. An append-only log (A1) that recorded one would assert
  something false as soon as a provider expired it early, with no way to take
  it back. That bytes left this machine for a named provider is a *fact*. It is
  privacy-relevant, and it is exactly what a person who audits a session needs.
  So the log records it: `attachment/uploading` before the bytes leave, and
  `attachment/uploaded` when the upload settles.
* **It is keyed on the digest, not on the session.** Two sessions that attach
  one photo already share one blob. That they share one upload follows from the
  same naming. A record for each session would make the log of one session the
  authority for the uploads of another session. Neither session asked for that
  coupling.
* **Losing it costs an upload, not information.** Every entry can be rebuilt
  from bytes that the store still holds, which is the definition of a cache. So
  it belongs beside the other derived state under `$PH_CACHE`, where a user can
  delete the directory and get nothing worse than one slow turn.

**One file for each entry, not an index.** The daemon runs many roots in one
process, and several processes may share a `$PH_CACHE`. A single index is a
write that all of them contend for, and a corrupt read of it loses every handle
at the same time. A content-addressed file for each `(provider, digest)` has the
same failure mode as the blob store that it mirrors: one entry is lost, and a
new upload recovers it.

**Expiry is checked twice, and the second check is the one that matters.** A
handle carries what the provider said, so an entry that is past its own
`expires_at` is uploaded again before a request is built. But providers expire
files early, delete them from another session, or forget them. That shows up as
a failure *mid-turn*. `invalidate_handle` and the retry waterfall turn that
failure into one more attempt and not a lost turn.

**`phern attachments gc` prunes the entries, from the same reference set as the
blobs.** An entry is written for each `(provider, digest)` that was ever
uploaded, and it stays after the provider has forgotten the file. Without a
sweep, the directory grows with the number of distinct attachments that a
deployment has ever sent. The sweep is in that command and not here because the
question is the same one: *does any stored session still point at this digest*.
To answer it twice would be two folds that can disagree. The reason is **not**
shared. Losing an entry costs one upload, and losing a blob costs the
conversation. That is why the blobs are aged and guarded by a completeness
check, and these entries are not.

Still not enforced (§5 rule 6): **nothing prunes the cache automatically.** The
command is manual, like the collection of the blobs that the cache mirrors. A
deployment that never runs it keeps every entry that it has ever written. Each
entry is a few hundred bytes, and the whole directory is safe to delete.

@module ph.seams.uploads
"""

from __future__ import annotations

import json
import logging
from collections.abc import Container, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import anyio

from ..cordis import Context, Disposer, Running, plugin, running
from ..json import JsonValue
from ..keys import ATTACHMENTS, SESSIONS, UPLOADS
from ..llm.types import AttachmentRef
from ..paths import default_cache_path, write_atomic
from ..session import IntentNotDurable, Session, intents_of, now_ms, session_written
from ..session.kinds import UPLOAD, upload_settled
from ..wire import WireModel
from ._registry import claim_key

__all__ = ["FileHandle", "UploadRegistry", "Uploader", "apply"]

log = logging.getLogger("ph.seams.uploads")


class FileHandle(WireModel):
    """What a provider gave back for the bytes of one attachment."""

    provider: str
    attachment_id: str
    handle: str
    """The id that the provider chose. It is opaque here, and it is passed back
    unchanged on the wire."""
    uploaded_at: int
    expires_at: int | None = None
    """When the provider said that the handle stops working, or `None` when it did
    not say.

    `None` means "no expiry announced" and not "never expires". The difference
    matters, because "never expires" would be a promise that this file cannot
    keep for any provider. In both cases a failure mid-turn invalidates the
    entry, and that check does not depend on anyone having told the truth."""


class Uploader(Protocol):
    """The file API of a route, as the seam needs it."""

    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle: ...


@dataclass(frozen=True, slots=True)
class _Registered:
    """An uploader and who registered it (P6-29).

    This is the shape that `AdapterHandle` carries, for the same reason.
    `upload` is row code that this seam invokes *later*. So it must run bound
    to the row that contributed it, and not to whatever was running when a
    request needed a file.
    """

    uploader: Uploader
    by: Running


def _uploading(provider: str, ref: AttachmentRef) -> dict[str, JsonValue]:
    """The `attachment/uploading` record: which file is about to go, and to whom.

    The record does not hold the handle, which expires. It holds the fact that
    a named provider is being given this file, which is the part of this row
    that *does* belong in the log. A person who audits where their data went
    reads the pair of records. Nothing reads the record back to find a handle.
    """
    return {
        "provider": provider,
        "attachmentId": ref.attachment_id,
        "mime": ref.mime,
        "name": ref.name,
        "bytes": ref.bytes,
    }


class Config(WireModel):
    """Row config for the handle cache."""

    root: str | None = None
    """Where the entries live. The default is `$PH_CACHE/uploads`.

    The path is a setting, like the path of every other storage row, so a
    deployment can put this cache on a different volume without moving all of
    `$PH_CACHE`. The default goes through `default_cache_path` and not through
    `default_home_path`, which the storage rows under `$PH_HOME` use. This
    default belongs under the *cache*. That is the argument at the top of this
    file, so a helper that says `home` would quietly contradict it."""


@dataclass(slots=True)
class UploadRegistry:
    """The service published as `ctx.uploads`."""

    ctx: Context
    root: Path
    uploaders: dict[str, _Registered] = field(default_factory=dict)
    _memo: dict[tuple[str, str], FileHandle] = field(default_factory=dict)
    """The entries that this process has read, so that a handle is read from disk
    one time.

    The reason is the same as for `load_b64`. An attachment stays in derived
    history for the life of the session. Without this memo, the same small JSON
    file would be opened and parsed on every step of every turn.
    `invalidate_handle` removes an entry, because a dead handle is the only
    thing that makes an entry wrong. `prune` removes the entries that it
    deletes from disk."""

    def register_uploader(self, provider: str, uploader: Uploader) -> Disposer:
        """Claim the file API for one provider.

        There is a slot for each provider and not one slot, because a deployment
        that routes to two providers is ordinary, and their file APIs have
        nothing in common.
        """
        by = self.ctx.running_for()
        return claim_key(
            by.owner, self.uploaders, provider, _Registered(uploader, by), label="uploader"
        )

    def path_for(self, provider: str, attachment_id: str) -> Path:
        digest = attachment_id.partition(":")[2] or attachment_id
        return self.root / provider / f"{digest}.json"

    def cached(self, provider: str, attachment_id: str) -> FileHandle | None:
        """The stored handle, if there is one and its announced expiry has not passed.

        This is a cache read that cannot raise. A truncated or hand-edited entry
        is treated as absent, because the remedy for both is the same upload. To
        fail a turn because a *cache* is unreadable would be absurd.
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
        """Forget the entry that holds this provider id, wherever it is.

        The adapter that meets a dead handle mid-request knows the *file id* and
        not the digest that it was made from, because the request carries the
        id and not the digest. In the ordinary case the memo answers without
        touching the disk. The scan is the cold path, after a restart, and it
        stops at the match.

        This method is called for the one handle that the provider *named*, and
        never for every handle that a request referenced. A request with twenty
        files that discarded nineteen live uploads to replace one dead upload
        would upload nineteen files again to fix a problem that it did not have.
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
        self, ref: AttachmentRef, *, provider: str, session_id: str | None = None
    ) -> FileHandle | None:
        """The handle to reference these bytes by. The bytes are uploaded if needed.

        The result is `None` when this provider has no uploader or no attachment
        store is mounted. That is not an error, because the caller has an
        alternative. It sends the bytes inline, which is what every route did
        before this row existed. For a file over the inline limit of the route
        it sends a pointer (`load_media` in `ph_app.adapters._media`).

        `session_id` names the session that records the *fact* when an upload
        actually happens. The fact is recorded here and not by the caller,
        because this is the one place that knows the difference between a cache
        hit and bytes that leave the machine. The user is owed exactly that
        difference.

        **The argument is an id, and this method resolves it.** The first
        version took the `Session` itself and made every adapter look one up.
        Three adapters had written the same five lines by the time the third
        wire landed. That shape invites an omission that is silent in the worst
        way. An adapter that forgets to pass the session loses the
        `attachment/uploading` and `attachment/uploaded` records, the only
        record that bytes left the machine, while every request still succeeds.
        This method has the `ctx` that those lookups needed, so the rule is
        written one time, beside the append that it guards.
        """
        cached = self.cached(provider, ref.attachment_id)
        if cached is not None:
            return cached
        entry = self.uploaders.get(provider)
        store = self.ctx.get(ATTACHMENTS)
        if entry is None or store is None:
            return None
        content = await store.load_bytes(ref)
        session = self._session(session_id)
        if session is None:
            with running(entry.by):
                handle = await entry.uploader.upload(ref, content)
        else:
            recorded = await self._recorded(session, provider, entry, ref, content)
            # **The settle is durable before the cache is** (F10). The cache is
            # what makes every later request skip this branch. When the cache
            # was written first, a crash before the next barrier kept the handle
            # and lost the fact. The loss was permanent, because nothing ever
            # uploads that file for that provider again, or records it. The
            # handle is not cached when the record cannot be written. The next
            # request then uploads again and records again. That is the
            # direction in which a person who audits their data needs this to
            # fail.
            if recorded is None or not await session_written(self.ctx, session):
                return recorded
            handle = recorded
        await self._store(handle)
        return handle

    async def _recorded(
        self,
        session: Session,
        provider: str,
        entry: _Registered,
        ref: AttachmentRef,
        content: bytes,
    ) -> FileHandle | None:
        """Upload under `UPLOAD`: **the record is on disk before the bytes leave** (S9).

        The record used to follow the upload. A crash between the two left the
        provider with a file that this log never said it was given. When the
        upload raises, `claim` settles it as `outcome-unknown`, because the
        bytes may have gone. An upload that the log cannot record is not made.
        The result is then `None`, and the caller sends the bytes inline, in the
        request that its own barrier has recorded. A file over the inline limit
        of the route goes as a pointer (`handle_for`).
        """
        journal = intents_of(self.ctx)
        try:
            async with journal.claim(session, UPLOAD, _uploading(provider, ref)) as held:
                with running(entry.by):
                    handle = await entry.uploader.upload(ref, content)
                journal.settle(
                    session, held, upload_settled(held.opened, expires_at=handle.expires_at)
                )
        except IntentNotDurable:
            return None
        return handle

    def _session(self, session_id: str | None) -> Session | None:
        sessions = self.ctx.get(SESSIONS)
        return None if sessions is None or session_id is None else sessions.get(session_id)

    def stale(self, referenced: Container[str]) -> tuple[Path, ...]:
        """The cache entries for digests that nothing points at any more (P7-01).

        This is the half of `phern attachments gc` that belongs to this seam. It
        is here because the layout is here: `path_for` decides that an entry is
        `<provider>/<digest>.json`. A collector that rebuilt that rule from the
        outside would keep reporting zero, silently and with nothing failing,
        from the day that the scheme changed. The attachment store still
        computes the *reference set*. This seam decides which files answer to
        it.

        **The entries are not aged, although the blobs are.** The cache entry of
        an unreferenced digest is dead as soon as its blob is. There is no
        staged-file window to wait out, because nothing uploads a file before a
        session mentions it. The two halves are swept under one reference set
        but not for the same reason: losing an entry costs one upload, and
        losing a blob costs the conversation.
        """
        if not self.root.is_dir():
            return ()
        return tuple(
            path
            for path in sorted(self.root.rglob("*.json"))
            if f"sha256:{path.stem}" not in referenced
        )

    def prune(self, paths: Iterable[Path]) -> int:
        """Remove these entries and forget what they said.

        The removal goes through the seam and is not an unlink from the
        collector, because an outside deleter cannot see the memo. `_memo` would
        go on answering `cached()` with a handle whose file this command had
        just removed, for the life of the process. `invalidate_handle` already
        pairs the two steps for one entry. This method is the same pairing for a
        sweep.
        """
        removed = 0
        for path in paths:
            try:
                path.unlink()
            except OSError:  # pragma: no cover - raced deletion
                continue
            self._memo.pop((path.parent.name, f"sha256:{path.stem}"), None)
            removed += 1
        return removed

    async def _store(self, handle: FileHandle) -> None:
        self._memo[(handle.provider, handle.attachment_id)] = handle
        path = self.path_for(handle.provider, handle.attachment_id)
        payload = json.dumps(handle.to_wire())
        # The write is atomic. A reader in another process must see either the
        # old entry or the new one. A half-written handle that is read as JSON
        # is the corruption that `cached` has to treat as absent.
        await anyio.to_thread.run_sync(write_atomic, path, payload)


@plugin("uploads-local", affects="deployment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the handle cache. Uploaders come from the adapter rows."""
    root = default_cache_path(config.root, "uploads")
    ctx.provide(UPLOADS, UploadRegistry(ctx=ctx, root=root))
