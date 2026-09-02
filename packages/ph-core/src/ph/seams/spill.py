"""`ctx.spill_store` — oversized content out of context, with a way back.

An offloaded tool result is not deleted, it is *relocated*: the model gets a
preview and a locator, and the locator resolves to the full text. That is what
makes G2/G3 offloading (Phase 4) an optimisation rather than a lie — the
harness never tells the model something is gone when it is on disk.

`retrieval_hint` exists so the preview can say how to get the rest in the
model's own vocabulary (`read` this path, offset N), rather than making it guess.

@module ph.seams.spill
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, Disposer, plugin
from ..paths import default_home_path
from ..session import Session
from ..wire import WireModel
from ._registry import claim_entry

__all__ = ["SpillClaim", "SpillRef", "SpillStore", "apply"]

log = logging.getLogger("ph.seams.spill")


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

    @classmethod
    def under_session(cls, label: str, event_type: str) -> SpillClaim:
        """A producer writing under `session.id` whose events carry `locator`."""
        return cls(label=label, event_type=event_type, owners=lambda session: {session.id})


@dataclass(slots=True)
class SpillStore:
    """The service published as `ctx.spill_store`."""

    ctx: Context
    root: Path
    _claims: list[SpillClaim] = field(default_factory=list)

    def locator_for(self, *, owner: str, suggested_name: str, content: bytes) -> Path:
        """Where `content` will be written — derived, not written.

        The one home of the naming rule (digest + sanitized name), so a caller
        that must record a blob's locator *before* writing it (write-ahead
        ordering, §4.9) derives the same path the write will use rather than
        mirroring the rule and hoping a test keeps the two in step.
        """
        digest = hashlib.sha256(content).hexdigest()[:16]
        safe = "".join(char if char.isalnum() or char in "-._" else "_" for char in suggested_name)
        return self.root / owner / f"{digest}-{safe}"

    async def save_bytes(
        self, *, owner: str, source: str, suggested_name: str, content: bytes
    ) -> SpillRef:
        """Write binary `content` and return its reference.

        Named by content digest, so re-spilling identical output costs one file
        rather than one file per occurrence. Text spills through here too, as
        UTF-8, so the naming rule has one implementation.
        """
        path = self.locator_for(owner=owner, suggested_name=suggested_name, content=content)
        await anyio.to_thread.run_sync(_write, path.parent, path, content)
        return SpillRef(
            locator=str(path),
            bytes=len(content),
            retrieval_hint=f'read the file at "{path}" for the full {source}',
        )

    async def save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef:
        """Write `content` as UTF-8 and return its reference."""
        return await self.save_bytes(
            owner=owner,
            source=source,
            suggested_name=suggested_name,
            content=content.encode("utf-8"),
        )

    async def try_save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef | None:
        """`save_text`, or `None` when the store could not take it.

        The **fail-open** spelling, for the callers whose content is an
        optimisation rather than an obligation: an offload that cannot store the
        text must not be the reason the model loses it. Written here because
        three callers were each remembering the rule in their own `try`, and had
        already drifted on which exception counts.

        `| None` rather than a raised-and-caught exception, so the failure
        branch is type-checked at every call site instead of remembered.
        `save_text` stays for a caller that must not proceed without durability.
        """
        try:
            return await self.save_text(
                owner=owner, source=source, suggested_name=suggested_name, content=content
            )
        except Exception:
            log.warning(
                "ph.seams.spill: could not spill %s for %s", suggested_name, owner, exc_info=True
            )
            return None

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
        """
        claims = tuple(self._claims)

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
                removed.extend(_remove_unreferenced(self.root / owner, referenced))
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


def _remove_unreferenced(directory: Path, referenced: set[str]) -> list[str]:
    """Delete the files in one owner directory that no claim references."""
    if not directory.is_dir():
        return []
    gone: list[str] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and str(path) not in referenced:
            path.unlink(missing_ok=True)
            gone.append(str(path))
    return gone


def _write(directory: Path, path: Path, payload: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class Config(WireModel):
    """Row config for the local spill store."""

    root: str | None = None


@plugin("spill-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local spill store."""
    root = default_home_path(config.root, "spill")
    store = SpillStore(ctx=ctx, root=root)
    ctx.provide("spill_store", store)

    async def sweep_on_open(session: Session) -> None:
        """The one open-time sweep, owned by the store rather than by a producer."""
        removed = await store.sweep_session(session)
        if removed:
            log.info("ph.seams.spill: swept %d unreferenced blob(s)", len(removed))

    ctx.on("session/created", sweep_on_open)
