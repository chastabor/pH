"""`ctx.spill_store`: oversized content moved out of context, with a way back.

An offloaded tool result is not deleted. It is *relocated*: the model gets a
preview and a locator, and the locator resolves to the full text. That is what
makes G2/G3 offloading (Phase 4) an optimization and not a lie. The harness
never tells the model that something is gone when it is on disk.

`retrieval_hint` exists so that the preview can say how to get the rest in the
vocabulary of the model (`read` this path, offset N). The model does not have
to guess.

@module ph.seams.spill
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, Disposer, plugin
from ..keys import SPILL_STORE
from ..paths import default_home_path, write_atomic_all
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
    """Bytes, together with the locator that the naming rule of the store gives
    them (`SpillStore.plan`). It is for a caller that must name a blob before it
    stores the blob.

    The locator and the bytes are one value and not two, for two reasons. First,
    the digest is taken **one time**. `plan` hashes the whole payload. A caller
    that derived the locator for its record and then stored the bytes hashed
    them again, which was a second sha256 of a multi-megabyte tool result on
    the event loop. Second, the name cannot come apart from what it names.
    `try_save` writes *these* bytes at *this* locator, with no second derivation
    that could disagree.
    """

    locator: Path
    content: bytes


def _plain_locator(data: Mapping[str, Any]) -> str | None:
    """`data["locator"]` when it is a string, which is the convention of the seam.

    A spill that failed open records its event with `locator: None`. So a value
    that is not a string is skipped and not coerced. A `"None"` in the reference
    set would keep a file that does not exist, and it would look like a producer
    that does its job.
    """
    locator = data.get("locator")
    return locator if isinstance(locator, str) else None


@dataclass(frozen=True, slots=True)
class SpillClaim:
    """The blobs of one producer: where they live, and which event names each (F7).

    A claim is contributed by its producer and is not known here. The sweep
    began as the `session/created` listener of one producer. Every producer that
    was added afterwards wrote blobs that nothing collected, and a crash between
    a blob write and the append that names the blob leaked a file permanently. A
    sweep for each producer is how that happens a second time. So there is one
    sweep, and producers contribute to its fold.

    `owners` is unconditional: it is read even from a log with no spill events.
    So the crash case (blob written, event never appended) is still visited. The
    two readers, `locator` and `owner`, each take one event. That lets the seam
    fold every claim in **one pass** over the log. A producer whose owner is
    templated from the event (`kernel/<namespace>`) reads the owner there and
    does not scan the log itself.
    """

    label: str
    event_type: str
    owners: Callable[[Session], Iterable[str]] = lambda _session: ()
    locator: Callable[[Mapping[str, Any]], str | None] = _plain_locator
    owner: Callable[[Mapping[str, Any]], str | None] = lambda _data: None
    hands_paths: bool = False
    """Whether the locator of this producer is put in front of the model to follow.

    `tool-result-offload` and `input-offload` replace content with a preview and
    the path that the content was written to. `compaction-summarize` tells the
    model where the history was saved. A read of such a path is the model
    following something that the harness handed it. See `handed_paths_of`,
    which folds this flag. A kernel snapshot is *not* handed, because nothing
    shows the model where it went. A producer has to say which case it is, and
    the claim is the extension point that the seam already has.
    """

    @classmethod
    def under_session(cls, label: str, event_type: str, *, hands_paths: bool = False) -> SpillClaim:
        """A producer that writes under `session.id` and whose events carry `locator`."""
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
        """What producers have contributed, for a caller that folds the claims.

        The value is a tuple and not the list. The sweep folds on a worker
        thread while the event loop continues, and a claim that is registered in
        the meantime must not change the fold."""
        return tuple(self._claims)

    def owner_root(self, owner: str) -> Path:
        """Where the blobs of one owner live. This is the other half of the naming rule.

        `plan` derives a whole path. A caller that must answer "is this path one
        of yours" needs the directory. When `root / owner` is built again
        outside this class, the two spellings drift apart. `handed_paths_of` is
        that caller, on behalf of `permissions-fs`.
        """
        return self.root / owner

    def plan(self, *, owner: str, suggested_name: str, content: bytes) -> PlannedBlob:
        """Where `content` will go, held together with it. `try_save` writes what
        this names.

        This method is the one place that holds the naming rule (digest plus
        sanitized name). A caller sometimes must name the locator of a blob
        before it stores the blob: in the wording that replaces the content, or
        in the record that points at the blob. Such a caller derives the same
        path that the write will use. It does not mirror the rule and hope that
        a test keeps the two in step.
        """
        digest = hashlib.sha256(content).hexdigest()[:16]
        safe = "".join(char if char.isalnum() or char in "-._" else "_" for char in suggested_name)
        return PlannedBlob(self.owner_root(owner) / f"{digest}-{safe}", content)

    async def try_save(self, planned: PlannedBlob, *, source: str) -> SpillRef | None:
        """Write `planned` at its locator, durably, and return its reference. The
        result is `None` when the store could not take the blob.

        **Write first, then append the record that names the blob** (§4.9). The
        blob is on disk before any record names it. So the log never names bytes
        that are not there, and a write that fails does so while the producer
        has logged nothing. That keeps the fail-open answer open. A producer
        that cannot store a blob must not be the reason that the model loses
        what it held: an offload keeps the text inline, and a kernel snapshot
        records a `clear`. A run that dies between the write and the append
        leaves a file that nothing names. The next read of the log collects it
        (`sweep_session`).

        Earlier, the blob was staged out of reach of the locator, named, and
        only then renamed into place (`reserve`, `commit`). The reason was that
        the sweep ran beside producers, and it collected a blob that was caught
        between its write and its append. The sweep now runs only where nothing
        writes the session. So that window is no hazard, and the bytes go
        straight to where the record will say they are.

        This is a batch of one: `try_save_all` is the one writer.
        """
        [ref] = await self.try_save_all([(planned, source)])
        return ref

    def plan_text(self, *, owner: str, suggested_name: str, text: str) -> PlannedBlob | None:
        """`plan`, for text as UTF-8. The result is `None` when the text has no
        UTF-8 form.

        A lone surrogate cannot be encoded, and a `\\ud800` escape in a JSON tool
        result decodes to one. A text that cannot be encoded is a text that the
        store could not take. Its producer keeps it inline, as it would after a
        refused write. This is the one place here where text becomes bytes, so
        every text producer refuses in the same way.
        """
        try:
            content = text.encode("utf-8")
        except UnicodeEncodeError as error:
            _refused(f"{owner}/{suggested_name}", error)
            return None
        return self.plan(owner=owner, suggested_name=suggested_name, content=content)

    async def try_save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef | None:
        """`try_save`, for text that nobody planned (`plan_text`)."""
        planned = self.plan_text(owner=owner, suggested_name=suggested_name, text=content)
        return None if planned is None else await self.try_save(planned, source=source)

    async def try_save_all(self, blobs: Sequence[tuple[PlannedBlob, str]]) -> list[SpillRef | None]:
        """Write planned blobs, each with the `source` that its reference names, and
        return their references in order. This is the one writer of the store.

        It is for a producer that names more than one blob at the same time,
        such as the variables of a kernel cell, which all live under one owner.
        The blobs are written together. Each file is synced, and their directory
        is synced **one time** and not one time for each blob
        (`write_atomic_all`, whose docstring says how). So a cell that spills K
        variables costs one directory sync and not K. The records come after
        this method returns, so every blob is durable before anything names it.

        **The write goes through a temp file and a rename (L7)**, because the
        name carries the sha256 of the bytes. A write that truncated first and
        was cut off left a prefix under a name that says it is complete. Nothing
        rewrote it, because the digest already matched. The same digest is what
        makes it safe to skip a blob that is already there, so spilling
        identical output again costs one file and not one file for each
        occurrence. A kill that the cleanup of the temp file cannot reach leaves
        a `<name>.<hex>.tmp` that no locator names, and the next sweep collects
        it.

        A blob that could not be written gets `None`, and the producer records
        that as it would a refused `try_save`. The other blobs are still
        written.
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
        """Contribute the owners and references of one producer to the open-time sweep."""
        return claim_entry(
            self.ctx.owner_for(scope), self._claims, claim, label=f"spill.claim({claim.label})"
        )

    async def sweep_session(self, session: Session) -> list[str]:
        """Drop every blob that the log of this session no longer names (F7, P6-15).

        **Union first, then visit each owner one time.** Three producers write
        under `session.id`. If the sweep checked each claim against only *its
        own* references, each claim would delete the files of the others, and
        every one of them would be behaving correctly.

        **A claim that raises aborts the sweep.** A reference set that is
        assembled from some of the claims is *smaller* than the truth. A small
        reference set does not skip work. It deletes live blobs.

        The whole sweep is one pass over the log and one thread hop. It runs
        whenever a stored log is read, before the session is handed out, and a
        fold of a long log belongs off the event loop.

        **It also reports what it cannot mend.** The fold holds both halves of
        the comparison. It uses one half to find files that nothing names. It
        uses the other half to find a log that names a blob that is nowhere.
        That is reported, because the alternative is that the model is handed a
        path that fails when the model follows it.

        **It runs only where nothing writes the session.** Its one caller is the
        `session/loaded` listener of this row, which the reader runs before
        anyone holds the session. A test checks that it stays the only caller.
        That is what lets the sweep collect everything that the log does not
        name, including what a dead run left: a blob whose record never landed,
        and a `write_atomic` temp file that a kill interrupted. Beside a
        producer, the sweep would delete what that producer had written and not
        yet named (`try_save` writes first). It did exactly that while it was a
        `session/created` listener (S7, D17).
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
                # **The convention of the seam names a blob, whoever wrote it**
                # (S7). A producer that this profile does not mount has no claim.
                # Its blobs were therefore unreferenced to a sweep of the
                # directory that it shares with a mounted producer. They were
                # deleted while the log still pointed the model at them.
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
                    # **Only the own records of the session own a directory**
                    # (S7). The seeded prefix of a fork names the directory of
                    # its parent, where the parent went on writing blobs that
                    # this log never saw.
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
    """The files in one owner directory that a sweep may collect, if nothing names them."""
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


def _refused(what: str, error: BaseException) -> None:
    """Log why a blob was not spilled."""
    log.warning("ph.seams.spill: could not spill %s", what, exc_info=error)


def _ref(planned: PlannedBlob, source: str) -> SpillRef:
    """The reference by which a record names a written blob."""
    return SpillRef(
        locator=str(planned.locator),
        bytes=len(planned.content),
        retrieval_hint=f'read the file at "{planned.locator}" for the full {source}',
    )


class Config(WireModel):
    """Row config for the local spill store."""

    root: str | None = None


def handed_paths_of(ctx: Context, *, session: Session | None) -> tuple[Path, ...]:
    """The paths that this session was handed and may follow, asked of a seam that
    may not be mounted.

    This function is the twin of `ph.seams.sandbox.allowed_paths_of`, and it is
    deliberately a different set. That function answers "where the backend
    binds writes". This function answers "what the harness put in front of the
    model to read". `tool-result-offload` replaces an oversized result with a
    preview and the file that the result was written to. A rule that refuses
    reads outside the workspace would therefore hand the model a path and then
    refuse the path when the model followed it. The harness would tell the model
    that something is on disk and then deny it.

    **The set is folded from the claims and is not assumed.** The first version
    answered `root/<session id>` for every mounted deployment. That made the set
    a property of the layout, and not of what any row actually does. Three
    producers put a locator in front of the model: both offloads and the
    compaction history. Each says so in its claim, where it already declares
    everything else. A producer whose owner is *not* the session id would be
    picked up too, though none exists today. The owner of `ph_rlm.snapshot` is
    per event (`kernel/<namespace>`), so that producer contributes nothing here
    whatever it declares. That is correct, because nothing shows the model where
    a snapshot went.

    The answer is still the directories of this session and not the store. A
    locator is `root/<owner>/…`, so an answer of `root` would let any agent read
    the spilled results of every other session.

    The answer is empty without a session and empty without a store. A caller
    with neither was handed nothing.
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
        """The one sweep at open time. The store owns it, and no producer does.

        **It runs on `session/loaded`, which the reader awaits.**
        `SessionStore.loaded` says why the event is not `session/created`. The
        sweep is also the repair, and a resumed model may follow a locator in
        its first turn. A fresh session or a fork is not asked. It has no crash
        to repair and no directory of its own to collect.

        **There is no catch-up**, unlike `workspace-reconcile`, for three
        reasons. When this row activates, no producer has contributed a claim
        yet, because producers inject the store that this row provides and come
        up after it. A sweep at that time would visit nothing. A session that is
        already live was swept when it was read. And a sweep beside the
        producers of a live session is the one thing that `sweep_session` must
        never be.
        """
        removed = await store.sweep_session(session)
        if removed:
            log.info("ph.seams.spill: swept %d unreferenced blob(s)", len(removed))

    ctx.on("session/loaded", sweep_on_open)
