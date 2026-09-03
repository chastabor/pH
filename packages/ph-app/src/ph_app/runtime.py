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
    header names — and it is provided before the first row for the reason
    `Profile.mount` provides `ctx.mount` there: it is a fact about *this* mount
    that a row needs while applying. `fs-local` reads it as its root, and
    `workspace-lifecycle` then branches its worktrees from it and discovers that
    project's provisioning.

    A per-mount value and not a profile setting, because one daemon mounts one
    composition many times, once per session, and those sessions are in
    different repositories (P5-14). Composing a profile per root instead would
    re-import every plugin to change one path.
    """
    ctx = Context()
    try:
        if project is not None:
            ctx.provide("project_root", project)
        await profile.mount(ctx)
        yield ctx
    finally:
        # Disposal is structural: every registration and every acquired
        # artifact unwinds with its scope, children first (invariant I2).
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
    """Mount, create a session, drive one prompt to idle, flush — then yield.

    The sequence every one-shot mode shares. `before` runs after the session
    exists and before the prompt, for a mode that needs to attach a listener.

    The turn is opened with a message this builds rather than with
    `agent.prompt`, uniformly: with nothing attached the two are identical, so
    the alternative would be a branch whose two halves have to stay in step.
    """
    async with mounted(profile) as ctx:
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
