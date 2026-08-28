"""Composing a run: rows in, a mounted root context out.

One place that knows how a pH process starts, shared by every mode (print, json,
transcript, rpc, and from Phase 2 the TUI) so a mode cannot drift from the
profile semantics.

@module ph_app.runtime
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from ph.agent.types import AgentOptions
from ph.cordis import Context, Loader
from ph.session import Session

from .attach import ingest, prompt_message

__all__ = ["MountedRun", "mounted", "prompted"]


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


@asynccontextmanager
async def prompted(
    documents: Sequence[Path],
    prompt: str,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
    before: Callable[[Context, Session], None] | None = None,
) -> AsyncIterator[tuple[Context, Session]]:
    """Mount, create a session, drive one prompt to idle, flush — then yield.

    The sequence every one-shot mode shares. `before` runs after the session
    exists and before the prompt, for a mode that needs to attach a listener.

    The turn is opened with a message this builds rather than with
    `agent.prompt`, uniformly: with nothing attached the two are identical, so
    the alternative would be a branch whose two halves have to stay in step.
    """
    async with mounted(documents) as run:
        ctx = run.ctx
        session = ctx.sessions.create(session_id)
        if before is not None:
            before(ctx, session)
        # Before the agent exists: a file that cannot be read should fail the
        # command, not a turn — nothing is logged and there is nothing to unwind.
        refs = await ingest(ctx, attachments)
        agent = ctx.agents.create(session, AgentOptions(provider=provider, model=model))
        agent.followup(prompt_message(prompt, refs))
        await agent.run()
        await ctx.sessions.flush(session)
        yield ctx, session
