"""Composing a run: rows in, a mounted root context out.

One place that knows how a pH process starts, shared by every mode (print, json,
transcript, rpc, and from Phase 2 the TUI) so a mode cannot drift from the
profile semantics.

@module ph_app.runtime
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from ph.agent.types import AgentOptions
from ph.cordis import Context, Profile
from ph.persistence import ClaimingStore, resume_session
from ph.session import Session, new_session_id

from .attach import ingest, prompt_message

__all__ = ["mounted", "open_session", "prompted"]

log = logging.getLogger("ph_app.runtime")


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


async def open_session(
    ctx: Context, session_id: str | None = None, *, cwd: str | None = None
) -> Session:
    """Claim a session against every other writer, then resume it or create it (I-5).

    **The one door every host opens a session through** — the daemon's roots, the
    one-shot modes, rpc mode. Before it each host wrote its own, and the one-shot
    path did not resume at all: `ph -p --session x` on an id already on disk
    *created* a second session over the first one's file — the store saw the file,
    skipped the header and appended `seq` from zero — so two plain `ph -p` runs on
    one id, with no daemon and no race, left a log the trajectory reader refuses
    outright (P5-03). Resuming was the daemon's behaviour and is now everyone's.

    The claim comes first and comes from the store (`ClaimingStore`): the writer
    owns the lock, so a daemon, an rpc peer and a print run refuse each other with
    one code, `session_already_active`. A backend that cannot claim is said out
    loud rather than skipped — a silent skip is the shape that hides two daemons
    on one session.

    `cwd` reaches the *header* of a session created here, which is storage
    metadata beside the log rather than an event in it — so it describes where
    this conversation happened without becoming something the model reads or a
    replay has to re-apply. `SessionHeader` validates that it is absolute, so a
    relative path is refused rather than resolved against this process's own
    working directory, which is not the caller's.
    """
    resolved = session_id or new_session_id()
    store = ctx.get("session_persistence")
    if isinstance(store, ClaimingStore):
        await store.claim(resolved, scope=ctx)
    elif store is not None:
        log.warning(
            "ph_app.runtime: %s cannot claim a session; I-5 is not enforced for %s",
            type(store).__name__,
            resolved,
        )
    if store is not None and store.exists(resolved):
        session: Session = await resume_session(ctx, resolved)
        return session
    created: Session = ctx.sessions.create(resolved, meta={"cwd": cwd} if cwd else None)
    return created


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
    `ph -p --session x` is one more turn of one conversation rather than a second
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
        agent = ctx.agents.create(session, AgentOptions(provider=provider, model=model))
        agent.followup(prompt_message(prompt, refs))
        await agent.run()
        await ctx.sessions.flush(session)
        yield ctx, session
