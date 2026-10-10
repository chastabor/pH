"""Composing a run: rows in, a mounted root context out.

One place that knows how a pH process starts, shared by every mode (print, json,
transcript, rpc, and from Phase 2 the TUI) so a mode cannot drift from the
profile semantics.

@module ph_app.runtime
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
from pydantic import ValidationError

from ph.cordis import Context, LoaderError, MountRefusal, Profile
from ph.keys import AGENTS, NAMED_PROFILES, SESSIONS
from ph.paths import resolve_roots
from ph.persistence import open_session, stored_session
from ph.seams.models import ModelChoice, choose, start_on
from ph.session import Session, SessionForkError
from ph.session_profile import (
    logged_environment,
    withdraw_adoption,
    withdrawn_note,
)
from ph.wire import validation_summary

from .attach import ingest, prompt_message
from .console import err
from .daemon.recovery import resume_children
from .profiles import NAMED, StartingProfile, host_rows, kept_note, session_profile
from .sessions import RecordedStart, not_a_root, recorded_start

__all__ = ["mount_session", "mounted", "open_root", "prompted", "read_start"]

log = logging.getLogger(__name__)

_REFUSED = (LoaderError, MountRefusal, ValidationError)
"""What a profile's own refusal to mount is: a row that does not resolve, one that
declines on purpose, a config its model rejects. Anything else — a disk that filled,
a bug in a row — is a start failing, not the profile."""


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
        # Where named profiles live is this host's to know, and a child a parent
        # assigns one to is narrowed by it (S7b): provided before the rows, beside
        # what `Profile.mount` provides itself.
        ctx.provide(NAMED_PROFILES, NAMED)
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


async def read_start(session_id: str | None) -> RecordedStart:
    """What a session's log says before anything is mounted for it (`recorded_start`),
    read off the event loop, and nothing for a session about to be made.

    Off the loop because it is a search of the store, and for a fork or a segment a
    read of its whole lineage: on the daemon, every root it serves would wait behind
    one root's start. Read once by each host and handed on: to `mount_session` for the
    environment and the owner, and to `open_root` for the family, so the resume reads
    the log by path instead of searching for it a second time.
    """
    if not session_id:
        return RecordedStart()
    return await anyio.to_thread.run_sync(
        recorded_start, resolve_roots().sessions_dir(), session_id
    )


async def open_root(
    ctx: Context, start: RecordedStart, *, meta: Mapping[str, Any] | None = None
) -> Session:
    """`open_session` for the session `start` is the start of: its id and the family
    its log was found in, from one reading, so no host opens a stored root by id alone
    or pairs a family with another session's id. A session about to be made has
    neither, and `open_session` mints its id."""
    return await open_session(ctx, start.session_id, family=start.family, meta=meta)


async def mount_session(
    exits: AsyncExitStack,
    start: RecordedStart,
    requested: Profile,
    *,
    project: Path | None = None,
) -> tuple[Context, StartingProfile]:
    """A session mounted on `exits` in its own log's environment (`session_profile`),
    and **the one rule for a version the loader refuses**, for every host that starts
    one — the daemon's roots, `phern -p`, rpc.

    Every base change after the first is a pending adoption the next start applies
    (`/profile use`, a "yes" to a moved profile, `phern profiles adopt`), so a version
    that does not mount would be mounted again at every start, and the session would
    never open again. It is withdrawn instead — `profile/withdrawn`, under the
    session's lease on a mount of the host's rows alone — and the session mounted on
    the version it had; `StartingProfile.withdrawn` says so, for a host with somebody
    to tell. Only the profile's own refusals withdraw (`_REFUSED`): anything else fails
    the start as any start fails, and takes nothing back the next might mount.

    **A sub-agent's log is refused before anything is mounted** (P11-08): its one
    writer is the mount of the root that spawned it, and it is that child's only
    record. The daemon refuses first, with its own code (`NotARoot`); this is the same
    refusal for `phern -p --session <child>` and an rpc peer, which open through here.

    `start` is the session's start as its host read it (`read_start`): its id, the
    environment it mounts from, and the owner it is refused on. Composing the profile
    reads the named profiles' YAML, so it runs off the loop as the read did.
    """
    session_id = start.session_id
    if session_id and start.owner is not None:
        raise SessionForkError(not_a_root(session_id, start.owner), "SESSION_IS_SUBAGENT")
    starting = await anyio.to_thread.run_sync(session_profile, requested, start.environment)
    try:
        ctx = await exits.enter_async_context(mounted(starting.profile, project=project))
    except _REFUSED as error:
        adopting = starting.adopting
        if not session_id or adopting is None:
            raise
        reason = _refusal(error)
        log.warning(
            "ph_app: session %s: the adopted version of %s did not start (%s); "
            "withdrawing it and starting on the version it had",
            session_id,
            adopting.name,
            reason,
        )
        async with mounted(host_rows(requested)) as host:
            session = await stored_session(host, session_id, family=start.family)
            await withdraw_adoption(host, session, reason=reason)
            # Read from the session just written, which now says the version was
            # taken back — one store for the write and the read.
            environment = logged_environment(session)
        starting = replace(
            await anyio.to_thread.run_sync(session_profile, requested, environment),
            withdrawn=withdrawn_note(adopting.name, reason),
        )
        ctx = await exits.enter_async_context(mounted(starting.profile, project=project))
    return ctx, starting


def _refusal(error: LoaderError | MountRefusal | ValidationError) -> str:
    """Why a profile would not mount, in one line for its log and a person: pydantic's
    own message runs to a paragraph per field, with a link each."""
    if isinstance(error, ValidationError):
        return validation_summary(error, root="config")
    return str(error)


@asynccontextmanager
async def prompted(
    profile: Profile,
    prompt: str,
    *,
    choice: ModelChoice = ModelChoice(),
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

    **A resumed session mounts in its own log's environment** (S5): `profile` is what
    a new one is created on, and one that already has a base comes back as its log
    says (`mount_session`), whatever `--profile` this run was given. A named profile
    that moved since is kept, and said on stderr (S6): nobody is here to ask. So is an
    adopted version the loader refused, which this start took back.
    """
    async with AsyncExitStack() as exits:
        start = await read_start(session_id)
        ctx, starting = await mount_session(exits, start, profile)
        if starting.withdrawn:
            err.print(starting.withdrawn, style="yellow", markup=False)
        elif starting.change is not None and session_id:
            err.print(kept_note(starting.change, session_id), style="yellow", markup=False)
        # Resolved inside the mount, which is the only place that knows which
        # providers an adapter serves — and before the session opens, so a route
        # nothing can run leaves the command as its refusal with nothing on disk.
        choose(ctx, choice)
        session = await open_root(ctx, start)
        if before is not None:
            before(ctx, session)
        # This start's choice is an override of the session's model where it
        # differs (S4); the agent then runs on the session's own default, which a
        # `/model` logged before is part of.
        entry = await start_on(ctx, session, choice, source="cli", command=choice.flags)
        # Before the agent exists: a file that cannot be read should fail the
        # command, not a turn — nothing is logged and there is nothing to unwind.
        refs = await ingest(ctx, attachments)
        agent = ctx.require(AGENTS).create(session, entry.options())
        # Settled before the parent's turn can ask after them. A session this run
        # minted has none, and would only pay a scan of its family to find that.
        if session_id is not None:
            await resume_children(ctx, agent)
        agent.followup(prompt_message(prompt, refs))
        await agent.run()
        await ctx.require(SESSIONS).flush(session)
        yield ctx, session
