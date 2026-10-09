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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, Disposer, plugin
from ..keys import SPILL_STORE
from ..paths import default_home_path, write_atomic, write_atomic_all
from ..session import Session
from ..wire import WireModel
from ._registry import claim_entry

__all__ = ["PlannedBlob", "SpillClaim", "SpillRef", "SpillStore", "apply", "handed_paths_of"]

log = logging.getLogger("ph.seams.spill")


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
    is taken **once**: `plan` hashes the whole payload, and a caller that
    derived the locator for its record and then stored the bytes hashed them
    again — a second sha256 of a multi-megabyte tool result on the event loop. And
    the name cannot come apart from what it names: `save` writes *these* bytes at
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

        `plan` derives a whole path; a caller that has to answer "is this
        path one of yours" needs the directory, and reconstructing `root / owner`
        outside this class is how the two spellings drift. `permissions-fs` is
        that caller.
        """
        return self.root / owner

    def plan(self, *, owner: str, suggested_name: str, content: bytes) -> PlannedBlob:
        """Where `content` will go, held with it — `save` writes what this names.

        The one home of the naming rule (digest + sanitized name), so a caller
        that must name a blob's locator before storing it — in the wording that
        replaces it, in the record that points at it — derives the same path the
        write will use rather than mirroring the rule and hoping a test keeps the
        two in step.
        """
        digest = hashlib.sha256(content).hexdigest()[:16]
        safe = "".join(char if char.isalnum() or char in "-._" else "_" for char in suggested_name)
        return PlannedBlob(self.owner_root(owner) / f"{digest}-{safe}", content)

    async def save(self, planned: PlannedBlob, *, source: str) -> SpillRef:
        """Write `planned` at its locator, durably, and return its reference.

        **Write first, then append the record naming it** (§4.9). The blob is on disk
        before any record names it, so the log never names bytes that are not there,
        and a write that fails does so while the producer has logged nothing — the
        fallback `try_save` keeps open. A run that dies between the two leaves a file
        nothing names, which the next read of the log collects (`sweep_session`).

        It used to be staged out of the locator's reach, named, and only then renamed
        into place (`reserve`, `commit`), because the sweep ran beside producers and
        collected a blob caught between its write and its append. The sweep runs only
        where nothing writes the session now, so that window is no hazard and the
        bytes go straight to where the record will say they are.

        Named by content digest, so re-spilling identical output costs one file rather
        than one per occurrence, and bytes already there are not written again
        (`_write`).
        """
        await anyio.to_thread.run_sync(_write, planned.locator, planned.content)
        return _ref(planned, source)

    async def save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef:
        """`save`, for text nobody planned, as UTF-8 — through the one naming rule."""
        planned = self.plan(
            owner=owner, suggested_name=suggested_name, content=content.encode("utf-8")
        )
        return await self.save(planned, source=source)

    async def try_save(self, planned: PlannedBlob, *, source: str) -> SpillRef | None:
        """`save`, or `None` when the store could not take it.

        The fail-open spelling: a producer that cannot store a blob must not be the
        reason the model loses what it held — an offload keeps the text inline, a
        kernel snapshot records a `clear`. The write comes before the append, so that
        fallback is still available: the caller has logged nothing.
        """
        return await _fail_open(self.save(planned, source=source), what=str(planned.locator))

    async def try_save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef | None:
        """`try_save`, for text as UTF-8 — a text that cannot be encoded is one the
        store could not take, too."""
        return await _fail_open(
            self.save_text(
                owner=owner, source=source, suggested_name=suggested_name, content=content
            ),
            what=f"{owner}/{suggested_name}",
        )

    async def try_save_all(self, blobs: Sequence[tuple[PlannedBlob, str]]) -> list[SpillRef | None]:
        """`try_save` for several blobs one batch of records will name, in order.

        For a producer that names more than one blob at once — a kernel cell's
        variables, which all live under one owner. One thread hop writes them all,
        each file synced as `save` syncs it, and their directory is synced **once**
        rather than once per blob (`write_atomic_all`), so a cell that spills K
        variables costs one directory sync rather than K. The records come after
        this returns, so every blob is durable before anything names it.

        A blob that could not be written answers `None`, for the producer to record as
        it would a refused `try_save`.
        """
        failures = await anyio.to_thread.run_sync(
            partial(
                write_atomic_all,
                [(planned.locator, planned.content) for planned, _ in blobs],
                skip_if_present=True,
            )
        )
        refs: list[SpillRef | None] = []
        for (planned, source), failure in zip(blobs, failures, strict=True):
            if failure is not None:
                _refused(str(planned.locator), failure)
            refs.append(_ref(planned, source) if failure is None else None)
        return refs

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

        **And it says what it cannot mend.** The fold holds both halves of the
        comparison, so having used one of them to find files nothing names, it uses
        the other for a log naming a blob that is nowhere — reported, because the
        alternative is the model being handed a path that fails when it follows it.

        **Only where nothing writes the session.** Its one caller is the reader's
        `session/loaded`, before anyone holds the session — a test pins that it stays
        the only one — and that is what lets it collect everything the log does not
        name, a dead run's leftovers included: a blob whose record never landed, a
        `write_atomic` temp a kill interrupted. Beside a producer it would delete what
        that producer had written and not yet named (`save` writes first), which is
        what it did while it rode `session/created` (S7, D17).
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
                removed.extend(
                    _remove_unreferenced(_collectable(self.owner_root(owner)), referenced)
                )
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


def _collectable(directory: Path) -> list[Path]:
    """The files in one owner directory a sweep may collect, if nothing names them."""
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


async def _fail_open(saving: Awaitable[SpillRef], *, what: str) -> SpillRef | None:
    """`saving`'s ref, or `None` having said why — the `try_save*` rule, once."""
    try:
        return await saving
    except Exception as error:
        _refused(what, error)
        return None


def _refused(what: str, error: BaseException) -> None:
    """Say why a blob was not spilled."""
    log.warning("ph.seams.spill: could not spill %s", what, exc_info=error)


def _ref(planned: PlannedBlob, source: str) -> SpillRef:
    """The reference a written blob is named by in a record."""
    return SpillRef(
        locator=str(planned.locator),
        bytes=len(planned.content),
        retrieval_hint=f'read the file at "{planned.locator}" for the full {source}',
    )


def _write(path: Path, payload: bytes) -> None:
    """One spilled blob, at its locator (L7).

    K5's identical twin, and it had the same defect: the name carries the sha256
    of the bytes, and `write_bytes` truncates before it writes, so a crash
    mid-write leaves a prefix under a name that says it is complete. Nothing
    rewrites it, because the digest already matches what the caller asked for —
    which is what makes the lie permanent and what makes `skip_if_present` safe.
    Through the temp there is no torn file — only, after a kill the `except` cannot
    reach, a `<name>.<hex>.tmp` that no locator names, which the next sweep collects.
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
