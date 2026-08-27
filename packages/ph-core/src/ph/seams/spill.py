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

    async def save_text(
        self, *, owner: str, source: str, suggested_name: str, content: str
    ) -> SpillRef:
        """Write `content` and return its reference.

        Named by content digest, so re-spilling identical output costs one file
        rather than one file per occurrence.
        """
        payload = content.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()[:16]
        safe = "".join(char if char.isalnum() or char in "-._" else "_" for char in suggested_name)
        directory = self.root / owner
        path = directory / f"{digest}-{safe}"
        await anyio.to_thread.run_sync(_write, directory, path, payload)
        return SpillRef(
            locator=str(path),
            bytes=len(payload),
            retrieval_hint=f'read the file at "{path}" for the full {source}',
        )

    async def load_text(self, locator: str) -> str:
        return await anyio.to_thread.run_sync(lambda: Path(locator).read_text(encoding="utf-8"))

    def referenced_locators(self) -> set[str]:  # pragma: no cover - Phase 3 blob GC
        return set()


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
