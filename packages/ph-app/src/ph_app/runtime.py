"""Composing a run: rows in, a mounted root context out.

One place that knows how a pH process starts, shared by every mode (print, json,
transcript, rpc, and from Phase 2 the TUI) so a mode cannot drift from the
profile semantics.

@module ph_app.runtime
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ph.cordis import Context, Loader

__all__ = ["MountedRun", "mounted"]


@dataclass(slots=True)
class MountedRun:
    """A live root context and the loader that composed it."""

    ctx: Context
    loader: Loader


@asynccontextmanager
async def mounted(documents: Sequence[Path]) -> AsyncIterator[MountedRun]:
    """Compose a profile, mount it, and unwind the whole tree on exit."""
    loader = Loader.from_paths(list(documents))
    ctx = Context()
    try:
        await loader.mount(ctx)
        yield MountedRun(ctx=ctx, loader=loader)
    finally:
        # Disposal is structural: every registration and every acquired
        # artifact unwinds with its scope, children first (invariant I2).
        await ctx.drain()
        await ctx.dispose()
