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
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, Disposer, plugin
from ..keys import SPILL_STORE
from ..paths import default_home_path, is_atomic_temp, write_atomic
from ..session import Session
from ..wire import WireModel
from ._registry import claim_entry

__all__ = ["SpillClaim", "SpillRef", "SpillStore", "apply", "handed_paths_of"]

log = logging.getLogger("ph.seams.spill")

STAGING = ".staging"
"""Where `reserve_bytes` puts a blob until the log names it. See `_staging_for`."""


class SpillRef(WireModel):
    """Where spilled content went, and how to ask for it back."""

    locator: str
    bytes: int
    retrieval_hint: str


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

        A tuple rather than the list, for `sweep_session`'s reason one method
        down: a claim registered while a fold is running must not change the
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
        `commit` instead, in that order; `reserve_bytes` says why.
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
        cross a filesystem, and in a *subdirectory* so the sweep walks past it:
        `_remove_unreferenced` collects files, and this is a directory.
        """
        final = Path(locator)
        return final.parent / STAGING / final.name

    async def reserve_bytes(
        self, *, owner: str, source: str, suggested_name: str, content: bytes
    ) -> SpillRef:
        """Stage `content` and return the reference it will have once committed.

        **The write-ahead half of the ordering the sweep depends on** (§4.9). A
        blob is garbage exactly when the log does not name it, so a producer that
        writes first and appends second leaves a window in which its own blob is
        indistinguishable from garbage — and the open-time sweep, which folds the
        log on another task, deletes it. Reserving writes the bytes somewhere the
        sweep does not look, so the producer can append the locator *before* the
        file exists at it: from then on the blob is referenced from the moment it
        appears.

        Two calls rather than one because the failure has to stay on this side of
        the append. Writing is what can fail, so it happens first and a caller
        that cannot proceed without durability learns it before it has logged
        anything; `commit` is a rename on the same filesystem, which is atomic
        and, having got this far, all but certain.
        """
        path = self.locator_for(owner=owner, suggested_name=suggested_name, content=content)
        staged = self._staging_for(str(path))
        await anyio.to_thread.run_sync(_write, staged, content)
        return SpillRef(
            locator=str(path),
            bytes=len(content),
            retrieval_hint=f'read the file at "{path}" for the full {source}',
        )

    async def reserve_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef:
        """`reserve_bytes`, as UTF-8."""
        return await self.reserve_bytes(
            owner=owner,
            source=source,
            suggested_name=suggested_name,
            content=content.encode("utf-8"),
        )

    async def try_reserve_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef | None:
        """`reserve_text`, or `None` when the store could not take it.

        The fail-open spelling, for `try_save_text`'s reason and audience: an
        offload that cannot store the text must not be the reason the model loses
        it. Because the write happens here rather than at `commit`, that fallback
        is still available — the caller has logged nothing yet.
        """
        try:
            return await self.reserve_text(
                owner=owner, source=source, suggested_name=suggested_name, content=content
            )
        except Exception:
            log.warning(
                "ph.seams.spill: could not spill %s for %s", suggested_name, owner, exc_info=True
            )
            return None

    async def commit(self, ref: SpillRef) -> bool:
        """Publish a reserved blob at its locator. Call it *after* the append.

        A rename within one directory, so the blob appears whole or not at all
        and never appears unreferenced. `False` rather than a raise for the same
        reason `_write_blob` in `ph_rlm.snapshot` accepts this shape: by now the
        log names the locator, so the recoverable answer is a reader reporting a
        blob it cannot find, not a turn that fails after the fact.
        """
        staged = self._staging_for(ref.locator)
        try:
            await anyio.to_thread.run_sync(os.replace, staged, Path(ref.locator))
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
        on every session open, resume and fork, and a fold of a long log belongs
        off the event loop.

        **It also finishes what a dead run started, and says what it cannot.**
        The fold holds both halves of the comparison, so having used one of them
        to find files nothing names, it uses the other for the two states a
        crash can leave. A blob the log names that is still staged is *completed*
        — the bytes are there and the log already says where they belong, so the
        rename the dead run never reached is the repair. A blob the log names
        that is nowhere is reported, because the alternative is the model being
        handed a path that fails when it follows it.

        Nothing is ever deleted from `.staging`. A reservation in flight is
        indistinguishable from an abandoned one — neither is referenced yet, that
        being the whole point of write-ahead — so collecting the second would
        race the first, and the bookkeeping that told them apart bought less than
        it cost. What is left behind instead is the leak this module's
        `SpillClaim` already describes, one file per run that died mid-write.
        """
        claims = self.claims

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
                for claim in by_type.get(event.type, ()):
                    try:
                        locator = claim.locator(event.data)
                        owner = claim.owner(event.data)
                    except Exception:
                        return _abort(claim, session)
                    if locator is not None:
                        referenced.add(locator)
                    if owner is not None:
                        owners.add(owner)
            removed: list[str] = []
            for owner in sorted(owners):
                directory = self.owner_root(owner)
                for completed in _complete_staged(directory, referenced):
                    log.info("ph.seams.spill: completed an interrupted write of %s", completed)
                removed.extend(_remove_unreferenced(directory, referenced))
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


def _complete_staged(directory: Path, referenced: set[str]) -> list[str]:
    """Publish staged blobs the log already names. Returns what it finished.

    The recovery half of write-ahead ordering. A run that died between appending
    a locator and renaming the bytes into place leaves the log naming a blob that
    is not at its locator — and the bytes sitting one directory down, under the
    name they were always going to take. Finishing that rename is the whole
    repair, and it needs no record of what was in flight: the log is the record.
    """
    waiting = directory / STAGING
    if not waiting.is_dir():
        return []
    completed: list[str] = []
    for path in sorted(waiting.iterdir()):
        final = directory / path.name
        if path.is_file() and str(final) in referenced and not final.exists():
            os.replace(path, final)
            completed.append(str(final))
    return completed


def _remove_unreferenced(directory: Path, referenced: set[str]) -> list[str]:
    """Delete the files in one owner directory that no claim references.

    `.staging` is passed over because it is a directory and this collects files.
    Nothing in it is ever deleted; `sweep_session` says why.

    **Nor is a `write_atomic` temp** (D17), for `.staging`'s reason one level up.
    `save_bytes` writes its temp beside the locator, and the sweep runs on every
    session open — off-thread, concurrently with whatever else is writing — so
    collecting it deleted the temp of a write in flight and failed its rename. A
    kernel snapshot's blob is written that way *after* its event is durable, so
    the loss was a variable that would not restore.
    """
    if not directory.is_dir():
        return []
    gone: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and str(path) not in referenced and not is_atomic_temp(path):
            path.unlink(missing_ok=True)
            gone.append(str(path))
    return gone


def _write(path: Path, payload: bytes) -> None:
    """One spilled blob, at its locator or under `.staging` (L7).

    K5's identical twin, and it had the same defect: the name carries the sha256
    of the bytes, and `write_bytes` truncates before it writes, so a crash
    mid-write leaves a prefix under a name that says it is complete. Nothing
    rewrites it, because the digest already matches what the caller asked for —
    which is what makes the lie permanent and what makes `skip_if_present` safe.

    **Both callers, including the staging one.** `_staging_for` keeps the
    locator's name, so the digest promises the contents there too. It is the
    path that needed this most: `_finish_staged` republishes a staged file whose
    locator the log names, so a torn reservation used to be renamed into the
    locator as a complete blob. Through the temp there is no torn file to
    publish — only, after a kill the `except` cannot reach, a `<name>.<hex>.tmp`
    that the recovery sweep passes over because no locator matches it.
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


@plugin("spill-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local spill store."""
    root = default_home_path(config.root, "spill")
    store = SpillStore(ctx=ctx, root=root)
    ctx.provide(SPILL_STORE, store)

    async def sweep_on_open(session: Session) -> None:
        """The one open-time sweep, owned by the store rather than by a producer."""
        removed = await store.sweep_session(session)
        if removed:
            log.info("ph.seams.spill: swept %d unreferenced blob(s)", len(removed))

    ctx.on("session/created", sweep_on_open)
