"""Composing a run: rows in, a mounted root context out.

One place that knows how a pH process starts, shared by every mode (print, json,
transcript, rpc, and from Phase 2 the TUI) so a mode cannot drift from the
profile semantics.

@module ph_app.runtime
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from ph.agent.types import AgentOptions
from ph.cordis import Context, Profile
from ph.keys import AGENTS, SESSIONS
from ph.persistence import open_session
from ph.session import Session

from .attach import ingest, prompt_message

__all__ = ["mounted", "prompted"]


@asynccontextmanager
async def mounted(profile: Profile, *, project: Path | None = None) -> AsyncIterator[Context]:
    """Mount a composed profile, and unwind the whole tree on exit.

    A `Profile`, composed once at `profile_or_exit`: nothing is read or composed
    here, and repeated mounts of one profile never share a `Context`. What this
    one became is `ctx.mount`.

    `project` is **where this mount works** — the directory a session's own
    header names — and it is handed to `Profile.mount`, which provides it beside
    `ctx.mount`. `ph.cordis.loader.PROJECT_ROOT` carries the argument for why it
    is a per-mount fact rather than a profile setting; this function only has to
    pass it on. It used to be provided *here*, before the row loop, which made
    the ordering an unwritten protocol every other caller of `mount` had to know
    and none of them did.
    """
    ctx = Context()
    try:
        await profile.mount(ctx, project=project)
        yield ctx
    finally:
        # Disposal is structural: every registration and every acquired
        # artifact unwinds with its scope, children first (invariant I2). Each
        # call shields itself and carries its own budget, which is what makes
        # this `finally` a promise rather than an intention — a cancellation
        # landing in the first used to take the second with it, and with it
        # every lease, worktree and kernel the mount was holding.
        await ctx.drain()
        await ctx.dispose()


@asynccontextmanager
async def prompted(
    profile: Profile,
    prompt: str,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
    before: Callable[[Context, Session], None] | None = None,
) -> AsyncIterator[tuple[Context, Session]]:
    """Mount, open a session, drive one prompt to idle, flush — then yield.

    The sequence every one-shot mode shares. `before` runs after the session
    exists and before the prompt, for a mode that needs to attach a listener.

    *Open*, not create: a `session_id` already on disk is resumed, so a second
    `phern -p --session x` is one more turn of one conversation rather than a second
    log written over the first (P5-03), and one another process holds is refused.

    The turn is opened with a message this builds rather than with
    `agent.prompt`, uniformly: with nothing attached the two are identical, so
    the alternative would be a branch whose two halves have to stay in step.
    """
    async with mounted(profile) as ctx:
        session = await open_session(ctx, session_id)
        if before is not None:
            before(ctx, session)
        # Before the agent exists: a file that cannot be read should fail the
        # command, not a turn — nothing is logged and there is nothing to unwind.
        refs = await ingest(ctx, attachments)
        agent = ctx.require(AGENTS).create(session, AgentOptions(provider=provider, model=model))
        agent.followup(prompt_message(prompt, refs))
        await agent.run()
        await ctx.require(SESSIONS).flush(session)
        yield ctx, session
