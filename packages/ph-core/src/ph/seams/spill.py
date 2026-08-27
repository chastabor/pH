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
from dataclasses import dataclass
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path
from ..wire import WireModel

__all__ = ["SpillRef", "SpillStore", "apply"]


class SpillRef(WireModel):
    """Where spilled content went, and how to ask for it back."""

    locator: str
    bytes: int
    retrieval_hint: str


@dataclass(slots=True)
class SpillStore:
    """The service published as `ctx.spill_store`."""

    ctx: Context
    root: Path

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

    async def load_text(self, locator: str) -> str:
        return await anyio.to_thread.run_sync(lambda: Path(locator).read_text(encoding="utf-8"))

    async def load_bytes(self, locator: str) -> bytes:
        return await anyio.to_thread.run_sync(lambda: Path(locator).read_bytes())

    async def sweep(self, *, owner: str, referenced: set[str]) -> list[str]:
        """Delete this owner's blobs that no event references (F7).

        Called at session open, because a blob whose event never landed — a crash
        between the append and the write — is otherwise never reconciled by
        anything. Returns what it removed, so the caller can say so.
        """

        def remove() -> list[str]:
            directory = self.root / owner
            if not directory.is_dir():
                return []
            gone: list[str] = []
            for path in sorted(directory.iterdir()):
                if path.is_file() and str(path) not in referenced:
                    path.unlink(missing_ok=True)
                    gone.append(str(path))
            return gone

        return await anyio.to_thread.run_sync(remove)


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
    ctx.provide("spill_store", SpillStore(ctx=ctx, root=root))
