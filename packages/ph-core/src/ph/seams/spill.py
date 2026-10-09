"""`ctx.spill_store` — oversized content out of context, with a way back.

An offloaded tool result is not deleted, it is *relocated*: the model gets a
preview and a locator, and the locator resolves to the full text. That is what
makes G2/G3 offloading (Phase 4) an optimization rather than a lie — the
harness never tells the model something is gone when it is on disk.

`retrieval_hint` exists so the preview can say how to get the rest in the
model's own vocabulary (`read` this path, offset N), rather than making it guess.

@module ph.seams.spill
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, Disposer, plugin
from ..keys import SPILL_STORE
from ..paths import (
    default_home_path,
    holds,
    make_directories,
    replace_durably,
    sync_directory,
    write_atomic,
)
from ..session import Session
from ..wire import WireModel
from ._registry import claim_entry

__all__ = ["PlannedBlob", "SpillClaim", "SpillRef", "SpillStore", "apply", "handed_paths_of"]

log = logging.getLogger("ph.seams.spill")

STAGING = ".staging"
"""Where `reserve` puts a blob until the log names it. See `_staging_for`."""


class SpillRef(WireModel):
    """Where spilled content went, and how to ask for it back."""

    locator: str
    bytes: int
    retrieval_hint: str


@dataclass(frozen=True, slots=True)
class PlannedBlob:
    """Bytes, and the locator the store's naming rule gives them — derived together
    (`SpillStore.plan`), for a caller that must name a blob before storing it.

    One value rather than a locator and the bytes apart, for two reasons. The digest
    is taken **once**: `locator_for` hashes the whole payload, and a caller that
    derived the locator for its record and then reserved the bytes hashed them
    again — a second sha256 of a multi-megabyte tool result on the event loop. And
    the name cannot come apart from what it names: `reserve` stages *these* bytes at
    *this* locator, with no second derivation to disagree.
    """

    locator: Path
    content: bytes


def _plain_locator(data: Mapping[str, Any]) -> str | None:
    """`data["locator"]` when it is a string — the seam's own convention.

    A spill that failed open records its event with `locator: None`, so a
    non-string is skipped rather than coerced: `"None"` in the reference set
    keeps a file that does not exist and reads as a producer doing its job.
    """
    locator = data.get("locator")
    return locator if isinstance(locator, str) else None


@dataclass(frozen=True, slots=True)
class SpillClaim:
    """One producer's blobs: where they live, and which event names each (F7).

    Contributed rather than known here. The sweep began as one producer's own
    `session/created` listener; every producer added afterwards wrote blobs
    nothing collected, and a crash between a blob write and the append naming it
    leaked a file permanently. A per-producer sweep is how that happens twice, so
    there is one sweep and producers contribute to its fold.

    `owners` is unconditional — read even from a log with no spill events — so
    the crash case (blob written, event never appended) is still visited. Both
    readers are per event, which is what lets the seam fold every claim in **one
    pass** over the log: a producer whose owner is templated from the event
    (`kernel/<namespace>`) reads it there rather than scanning the log itself.
    """

    label: str
    event_type: str
    owners: Callable[[Session], Iterable[str]] = lambda _session: ()
    locator: Callable[[Mapping[str, Any]], str | None] = _plain_locator
    owner: Callable[[Mapping[str, Any]], str | None] = lambda _data: None
    hands_paths: bool = False
    """Whether this producer's locator is put in front of the model to follow.

    `tool-result-offload` and `input-offload` replace content with a preview and
    the path it was written to, so a read of that path is the model following
    something the harness handed it — see `handed_paths_of`, which folds this.
    A kernel snapshot is *not*: nothing shows the model where it went, and a
    producer that has to say so is the extension point the seam already has.
    """

    @classmethod
    def under_session(cls, label: str, event_type: str, *, hands_paths: bool = False) -> SpillClaim:
        """A producer writing under `session.id` whose events carry `locator`."""
        return cls(
            label=label,
            event_type=event_type,
            owners=lambda session: {session.id},
            hands_paths=hands_paths,
        )


@dataclass(slots=True)
class SpillStore:
    """The service published as `ctx.spill_store`."""

    ctx: Context
    root: Path
    _claims: list[SpillClaim] = field(default_factory=list)

    @property
    def claims(self) -> tuple[SpillClaim, ...]:
        """What producers have contributed, for a caller that folds them.

        A tuple rather than the list: the sweep folds on a worker thread while the
        event loop goes on, and a claim registered meanwhile must not change the
        fold under it."""
        return tuple(self._claims)

    def owner_root(self, owner: str) -> Path:
        """Where one owner's blobs live. The other half of the naming rule.

        `locator_for` derives a whole path; a caller that has to answer "is this
        path one of yours" needs the directory, and reconstructing `root / owner`
        outside this class is how the two spellings drift. `permissions-fs` is
        that caller.
        """
        return self.root / owner

    def locator_for(self, *, owner: str, suggested_name: str, content: bytes) -> Path:
        """Where `content` will be written — derived, not written.

        The one home of the naming rule (digest + sanitized name), so a caller
        that must record a blob's locator *before* writing it (write-ahead
        ordering, §4.9) derives the same path the write will use rather than
        mirroring the rule and hoping a test keeps the two in step.
        """
        digest = hashlib.sha256(content).hexdigest()[:16]
        safe = "".join(char if char.isalnum() or char in "-._" else "_" for char in suggested_name)
        return self.owner_root(owner) / f"{digest}-{safe}"

    async def save_bytes(
        self, *, owner: str, source: str, suggested_name: str, content: bytes
    ) -> SpillRef:
        """Write binary `content` and return its reference.

        Named by content digest, so re-spilling identical output costs one file
        rather than one file per occurrence. Text spills through here too, as
        UTF-8, so the naming rule has one implementation.
        """
        path = self.locator_for(owner=owner, suggested_name=suggested_name, content=content)
        await anyio.to_thread.run_sync(_write, path, content)
        return SpillRef(
            locator=str(path),
            bytes=len(content),
            retrieval_hint=f'read the file at "{path}" for the full {source}',
        )

    async def save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef:
        """Write `content` as UTF-8 and return its reference.

        The unordered spelling, for a caller with no log entry to keep in step
        with the write — a test planting a blob, or a producer that appends
        nothing. Anything that records a locator wants `reserve_text` and
        `commit` instead, in that order; `reserve` says why.
        """
        return await self.save_bytes(
            owner=owner,
            source=source,
            suggested_name=suggested_name,
            content=content.encode("utf-8"),
        )

    def _staging_for(self, locator: str) -> Path:
        """Where a reserved blob waits. Derived, so nothing has to remember it.

        Under the owner's own directory so the rename that publishes it cannot
        cross a filesystem, and in a *subdirectory* so no locator reaches it before
        `commit`: the sweep lists it separately, finishing what the log names and
        collecting what it does not.
        """
        final = Path(locator)
        return final.parent / STAGING / final.name

    def plan(self, *, owner: str, suggested_name: str, content: bytes) -> PlannedBlob:
        """Where `content` will go, held with it — `reserve` stages what this names."""
        return PlannedBlob(
            self.locator_for(owner=owner, suggested_name=suggested_name, content=content),
            content,
        )

    async def reserve(self, planned: PlannedBlob, *, source: str) -> SpillRef:
        """Stage `planned` and return the reference it will have once committed.

        **The write-ahead half of an ordering** (§4.9): the bytes are written somewhere
        a locator does not reach, so the producer can append the locator *before* the
        file exists at it, and the blob is referenced from the moment it appears. A
        run that dies between the append and `commit` leaves the bytes one directory
        down under the name the log already gives them, and the next read finishes the
        rename (`_complete_staged`); one that dies before the append leaves a stage
        nothing names, which the next read collects. It was written for a sweep that
        ran beside producers, which no longer happens (`sweep_session`).

        Two calls rather than one because the failure has to stay on this side of
        the append. Writing is what can fail, so it happens first and a caller
        that cannot proceed without durability learns it before it has logged
        anything; `commit` is a rename on the same filesystem, which is atomic
        and, having got this far, all but certain.

        **Bytes already published at the locator are linked, not rewritten** (`_stage`):
        the name carries their digest, so what is there is what would be written.
        """
        path = planned.locator
        await anyio.to_thread.run_sync(_stage, path, self._staging_for(str(path)), planned.content)
        return SpillRef(
            locator=str(path),
            bytes=len(planned.content),
            retrieval_hint=f'read the file at "{path}" for the full {source}',
        )

    async def reserve_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef:
        """`reserve`, for text nobody planned, as UTF-8."""
        planned = self.plan(
            owner=owner, suggested_name=suggested_name, content=content.encode("utf-8")
        )
        return await self.reserve(planned, source=source)

    async def try_reserve(self, planned: PlannedBlob, *, source: str) -> SpillRef | None:
        """`reserve`, or `None` when the store could not take it.

        The fail-open spelling: a producer that cannot store a blob must not be the
        reason the model loses what it held — an offload keeps the text inline, a
        kernel snapshot records a `clear`. Because the write happens here rather than
        at `commit`, that fallback is still available: the caller has logged nothing.
        """
        return await _fail_open(
            self.reserve(planned, source=source),
            owner=planned.locator.parent.name,
            name=planned.locator.name,
        )

    async def try_reserve_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef | None:
        """`try_reserve`, for text as UTF-8 — a text that cannot be encoded is one the
        store could not take, too."""
        return await _fail_open(
            self.reserve_text(
                owner=owner, source=source, suggested_name=suggested_name, content=content
            ),
            owner=owner,
            name=suggested_name,
        )

    async def commit(self, ref: SpillRef) -> bool:
        """Publish a reserved blob at its locator. Call it *after* the append.

        A rename within one directory, so the blob appears whole or not at all
        and never appears unreferenced — and a durable one (`replace_durably`),
        since the log naming the locator is `fsync`ed at the next barrier and the
        rename must not be the half a power cut loses (S6). `False` rather than a
        raise: by now the log names the locator, so the recoverable answer is a
        reader reporting a blob it cannot find, not a turn that fails after the fact.
        """
        staged = self._staging_for(ref.locator)
        try:
            await anyio.to_thread.run_sync(_publish, staged, Path(ref.locator), ref.bytes)
        except OSError:
            log.warning("ph.seams.spill: could not publish %s", ref.locator, exc_info=True)
            return False
        return True

    async def load_text(self, locator: str) -> str:
        return await anyio.to_thread.run_sync(lambda: Path(locator).read_text(encoding="utf-8"))

    async def load_bytes(self, locator: str) -> bytes:
        return await anyio.to_thread.run_sync(lambda: Path(locator).read_bytes())

    def claim(self, claim: SpillClaim, *, scope: Context | None = None) -> Disposer:
        """Contribute one producer's owners and references to the open-time sweep."""
        return claim_entry(
            self.ctx.owner_for(scope), self._claims, claim, label=f"spill.claim({claim.label})"
        )

    async def sweep_session(self, session: Session) -> list[str]:
        """Drop every blob this session's log no longer names (F7, P6-15).

        **Union first, then visit each owner once.** Three producers write under
        `session.id`, so sweeping each claim against only *its own* references
        would have each delete the others' files, every one behaving correctly.

        **A claim that raises aborts the sweep.** A reference set assembled from
        some of the claims is *smaller* than the truth, and a small reference set
        does not skip work — it deletes live blobs.

        One pass over the log and one thread hop for the whole thing: this runs
        whenever a stored log is read, before the session is handed out, and a
        fold of a long log belongs off the event loop.

        **It also finishes what a dead run started, and says what it cannot.**
        The fold holds both halves of the comparison, so having used one of them
        to find files nothing names, it uses the other for the two states a
        crash can leave. A blob the log names that is still staged is *completed*
        — the bytes are there and the log already says where they belong, so the
        rename the dead run never reached is the repair. A blob the log names
        that is nowhere is reported, because the alternative is the model being
        handed a path that fails when it follows it.

        **Only where nothing writes the session.** Its one caller is the reader's
        `session/loaded`, before anyone holds the session — a test pins that it stays
        the only one — and that is what lets it collect everything the log does not name —
        a dead run's leftovers included: a stage that never reached its append, a
        `write_atomic` temp a kill interrupted. Beside a producer it would delete what
        that producer had written and not yet named, which is what it did while it
        rode `session/created` (S7, D17), and why the leftovers used to be kept.
        """
        claims = self.claims
        seed = session.header.first_own_seq

        def run() -> list[str]:
            owners: set[str] = set()
            referenced: set[str] = set()
            by_type: dict[str, list[SpillClaim]] = {}
            for claim in claims:
                try:
                    owners.update(claim.owners(session))
                except Exception:
                    return _abort(claim, session)
                by_type.setdefault(claim.event_type, []).append(claim)

            for event in session.events:
                # **The seam's own convention names a blob whoever wrote it** (S7): a
                # producer this profile does not mount has no claim, so its blobs were
                # unreferenced to a sweep of the directory it shares with one that is —
                # and deleted, while the log still pointed the model at them.
                plain = _plain_locator(event.data)
                if plain is not None:
                    referenced.add(plain)
                for claim in by_type.get(event.type, ()):
                    try:
                        locator = claim.locator(event.data)
                        owner = claim.owner(event.data)
                    except Exception:
                        return _abort(claim, session)
                    if locator is not None:
                        referenced.add(locator)
                    # **Only the session's own records own a directory** (S7): a fork's
                    # seeded prefix names its parent's, where the parent went on writing
                    # blobs this log never saw.
                    if owner is not None and event.seq >= seed:
                        owners.add(owner)
            removed: list[str] = []
            for owner in sorted(owners):
                directory = self.owner_root(owner)
                staged = _collectable(directory / STAGING)
                for completed in _complete_staged(directory, staged, referenced):
                    log.info("ph.seams.spill: completed an interrupted write of %s", completed)
                # A stage the log does not name is a reservation that never reached its
                # append: every one that did is named, and was just finished above.
                abandoned = [one for one in staged if str(directory / one.name) not in referenced]
                listed = [*_collectable(directory), *abandoned]
                removed.extend(_remove_unreferenced(listed, referenced))
            for absent in sorted(one for one in referenced if not Path(one).exists()):
                log.warning(
                    "ph.seams.spill: session %s names a blob that is not there: %s",
                    session.id,
                    absent,
                )
            return removed

        return await anyio.to_thread.run_sync(run)


def _abort(claim: SpillClaim, session: Session) -> list[str]:
    log.warning(
        "ph.seams.spill: %s could not report its blobs; skipping the sweep for session %s "
        "rather than deleting against a partial reference set",
        claim.label,
        session.id,
        exc_info=True,
    )
    return []


def _complete_staged(directory: Path, staged: Iterable[Path], referenced: set[str]) -> list[str]:
    """Publish the `staged` blobs the log already names. Returns what it finished.

    The recovery half of write-ahead ordering. A run that died between appending
    a locator and renaming the bytes into place leaves the log naming a blob that
    is not at its locator — and the bytes sitting one directory down, under the
    name they were always going to take. Finishing that rename is the whole
    repair, and it needs no record of what was in flight: the log is the record.
    """
    completed: list[str] = []
    for path in staged:
        final = directory / path.name
        if str(final) not in referenced:
            continue
        # Through `commit`'s own rule, so a stage whose blob is already published —
        # a link a dead run never dropped, or a second reservation of the same
        # bytes — is dropped rather than kept to pin the bytes past the blob.
        existed = final.exists()
        _publish(path, final, path.stat().st_size)
        if not existed:
            completed.append(str(final))
    return completed


def _collectable(directory: Path) -> list[Path]:
    """The files in one directory a sweep may collect, if nothing names them.

    Files only, so an owner directory's `.staging` is passed over here and listed in
    its own right (`sweep_session`).
    """
    if not directory.is_dir():
        return []
    return [path for path in sorted(directory.iterdir()) if path.is_file()]


def _remove_unreferenced(files: Iterable[Path], referenced: set[str]) -> list[str]:
    """Delete the listed files that nothing references."""
    gone: list[str] = []
    for path in files:
        if str(path) not in referenced:
            path.unlink(missing_ok=True)
            gone.append(str(path))
    return gone


async def _fail_open(reserving: Awaitable[SpillRef], *, owner: str, name: str) -> SpillRef | None:
    """`reserving`'s ref, or `None` having said why — the `try_reserve_*` rule, once."""
    try:
        return await reserving
    except Exception:
        log.warning("ph.seams.spill: could not spill %s for %s", name, owner, exc_info=True)
        return None


def _stage(final: Path, staged: Path, payload: bytes) -> None:
    """Put `payload` under `.staging`, for `commit` to publish at `final`.

    **Bytes already published at `final` are hard-linked, not rewritten.** The name
    carries their digest, so the file there *is* the write this would make — a
    kernel variable returning to earlier bytes, a tool repeating its output — and a
    link stages it for a directory entry instead of the whole blob and its `fsync`.

    Any failure to link — a filesystem without hard links, a stage already there,
    `final` gone since — falls through to the write, which is always correct.
    """
    if holds(final, len(payload)):
        try:
            make_directories(staged.parent)
            os.link(final, staged)
        except OSError:
            pass
        else:
            sync_directory(staged.parent)
            return
    _write(staged, payload)


def _publish(staged: Path, final: Path, size: int) -> None:
    """Rename a stage into place — or find its bytes there already. `commit`'s rule,
    and the open-time sweep's (`_complete_staged`).

    **Already there, nothing staged**: a second reservation of the same bytes, whose
    one staged name the first commit took. The digest in the name says `final` holds
    these bytes, so that is success rather than a missing stage.

    **A stage linked to `final`** (`_stage`) is the same file, and `rename` between
    two links of one file does nothing — so the staging name is dropped instead, with
    no rename and no sync of `final`'s directory. Kept, it would pin the blob's bytes
    after the sweep collects `final`.
    """
    if not staged.exists():
        if holds(final, size):
            return
    elif final.exists() and os.path.samefile(staged, final):
        staged.unlink()
        sync_directory(staged.parent)
        return
    replace_durably(staged, final)


def _write(path: Path, payload: bytes) -> None:
    """One spilled blob, at its locator or under `.staging` (L7).

    K5's identical twin, and it had the same defect: the name carries the sha256
    of the bytes, and `write_bytes` truncates before it writes, so a crash
    mid-write leaves a prefix under a name that says it is complete. Nothing
    rewrites it, because the digest already matches what the caller asked for —
    which is what makes the lie permanent and what makes `skip_if_present` safe.

    **Both callers, including the staging one.** `_staging_for` keeps the
    locator's name, so the digest promises the contents there too. It is the
    path that needed this most: `_complete_staged` republishes a staged file whose
    locator the log names, so a torn reservation used to be renamed into the
    locator as a complete blob. Through the temp there is no torn file to
    publish — only, after a kill the `except` cannot reach, a `<name>.<hex>.tmp`
    that no locator names, which the next sweep collects.
    """
    write_atomic(path, payload, skip_if_present=True)


class Config(WireModel):
    """Row config for the local spill store."""

    root: str | None = None


def handed_paths_of(ctx: Context, *, session: Session | None) -> tuple[Path, ...]:
    """Paths this session was handed and may follow, asked of a seam that may not
    be mounted.

    The twin of `ph.seams.sandbox.allowed_paths_of`, and deliberately a different
    set: that one is "where the backend binds writes", this one is "what the
    harness put in front of the model to read". `tool-result-offload` replaces an
    oversized result with a preview and the file it was written to, so a rule
    refusing reads outside the workspace would hand the model a path and then
    refuse it when it followed it — the harness telling it something is on disk
    and then denying it.

    **Folded from the claims, not assumed.** The first cut answered
    `root/<session id>` for every mounted deployment, which made the set a
    property of the layout instead of of what any row actually does. Three
    producers put a locator in front of the model — both offloads and the
    compaction history — and each says so where it already declares everything
    else. A producer whose owner is *not* the session id would be picked up too,
    though none is today: `ph_rlm.snapshot`'s owner is per-event
    (`kernel/<namespace>`), so it contributes nothing here whatever it declares,
    and it is right not to — nothing shows the model where a snapshot went.

    Still this session's own directories and not the store: a locator is
    `root/<owner>/…`, so answering with `root` would let any agent read every
    other session's spilled results.

    Empty without a session, and empty without a store: a caller with neither was
    handed nothing.
    """
    store = ctx.get(SPILL_STORE)
    if store is None or session is None:
        return ()
    owners = {
        owner for claim in store.claims if claim.hands_paths for owner in claim.owners(session)
    }
    return tuple(store.owner_root(one) for one in sorted(owners))


@plugin("spill-local", affects="deployment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local spill store."""
    root = default_home_path(config.root, "spill")
    store = SpillStore(ctx=ctx, root=root)
    ctx.provide(SPILL_STORE, store)

    async def sweep_on_open(session: Session) -> None:
        """The one open-time sweep, owned by the store rather than by a producer.

        **On `session/loaded`, which the reader awaits** (`SessionStore.loaded` says
        why not `session/created`): the sweep is also the repair, and a resumed model
        may follow a locator in its first turn. A fresh session or a fork, with no
        crash to repair and no directory of its own to collect, is not asked.

        **No catch-up**, unlike `workspace-reconcile`. When this row activates no
        producer has contributed a claim yet — they inject the store this provides,
        and come up after it — so a sweep then would visit nothing; a session already
        live was swept when it was read; and a sweep beside a live session's producers
        is the one thing `sweep_session` must never be.
        """
        removed = await store.sweep_session(session)
        if removed:
            log.info("ph.seams.spill: swept %d unreferenced blob(s)", len(removed))

    ctx.on("session/loaded", sweep_on_open)
