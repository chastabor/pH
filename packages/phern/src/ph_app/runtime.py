"""Composing a run: rows go in, and a mounted root context comes out.

This is the one place that knows how a pH process starts. Every host that mounts
a profile shares it: the one-shot modes (text, json, transcript), rpc, and the
daemon, whose roots the TUI attaches to. So no host can drift from the profile
semantics.

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
"""What counts as a refusal by the profile itself to mount: a row that does not
resolve, a row that declines on purpose, or a config that its model rejects.
Anything else, such as a full disk or a bug in a row, is a start that failed and
not a refusal by the profile."""


@asynccontextmanager
async def mounted(profile: Profile, *, project: Path | None = None) -> AsyncIterator[Context]:
    """Mount a composed profile, and unwind the whole tree on exit.

    `profile` is a `Profile` that the caller composed, for example at
    `profile_or_exit`. Nothing is read or composed here, and repeated mounts of
    one profile never share a `Context`. `ctx.mount` says what this mount became.

    `project` is **where this mount works**: the directory that the header of a
    session names. It is handed to `Profile.mount`, which provides it beside
    `ctx.mount`. `ph.cordis.loader.PROJECT_ROOT` gives the argument for why it
    is a fact of each mount and not a profile setting. This function only passes
    it on. It used to be provided *here*, before the row loop. That made the
    ordering an unwritten protocol that every other caller of `mount` had to
    know, and none of them did.
    """
    ctx = Context()
    try:
        # This host knows where named profiles live, and a child that a parent
        # assigns a named profile to is narrowed by that profile (S7b). So the
        # store is provided before the rows, beside what `Profile.mount`
        # provides itself.
        ctx.provide(NAMED_PROFILES, NAMED)
        await profile.mount(ctx, project=project)
        yield ctx
    finally:
        # Disposal is structural: every registration and every acquired
        # artifact unwinds with its scope, children first (invariant I2). Each
        # call shields itself and carries its own budget, so this `finally` is
        # a guarantee and not an intention. A cancellation that landed in the
        # first call used to take the second call with it, and with that every
        # lease, worktree and kernel that the mount was holding.
        await ctx.drain()
        await ctx.dispose()


async def read_start(session_id: str | None) -> RecordedStart:
    """Return what the log of a session says before anything is mounted for it
    (`recorded_start`). The read runs off the event loop. A session that is about
    to be made has nothing recorded.

    The read is off the loop because it is a search of the store, and for a fork
    or a segment it is a read of the whole lineage. On the daemon, every root
    that it serves would wait behind the start of one root. Each host reads the
    start one time and hands it on: to `mount_session` for the environment and
    the owner, and to `open_root` for the family. So the resume reads the log by
    path and does not search for it a second time.
    """
    if not session_id:
        return RecordedStart()
    return await anyio.to_thread.run_sync(
        recorded_start, resolve_roots().sessions_dir(), session_id
    )


async def open_root(
    ctx: Context, start: RecordedStart, *, meta: Mapping[str, Any] | None = None
) -> Session:
    """Call `open_session` for the session that `start` is the start of. The id
    and the family that the log was found in come from one reading. So no host
    opens a stored root by id alone, or pairs a family with the id of another
    session. A session that is about to be made has neither, and `open_session`
    mints its id."""
    return await open_session(ctx, start.session_id, family=start.family, meta=meta)


async def mount_session(
    exits: AsyncExitStack,
    start: RecordedStart,
    requested: Profile,
    *,
    project: Path | None = None,
) -> tuple[Context, StartingProfile]:
    """Mount a session on `exits` in the environment of its own log
    (`session_profile`). This function also holds **the one rule for a version
    that the loader refuses**, for every host that starts a session: the roots of
    the daemon, `phern -p`, and rpc.

    Every base change after the first is a pending adoption that the next start
    applies (`/profile use`, a "yes" to a moved profile, `phern profiles adopt`).
    So a version that does not mount would be mounted again at every start, and
    the session would never open again. The version is withdrawn instead:
    `profile/withdrawn` is written under the lease of the session, on a mount of
    the rows of the host alone. The session is then mounted on the version that
    it had. `StartingProfile.withdrawn` says so, for a host that has somebody to
    tell. Only the refusals of the profile itself withdraw (`_REFUSED`). Anything
    else fails the start as any start fails, and takes back nothing that the
    next start might mount.

    **The log of a sub-agent is refused before anything is mounted** (P11-08).
    Its one writer is the mount of the root that spawned it, and it is the only
    record of that child. The daemon refuses first, with its own code
    (`NotARoot`). This is the same refusal for `phern -p --session <child>` and
    for an rpc peer, which open through here.

    `start` is the start of the session as its host read it (`read_start`): its
    id, the environment that it mounts from, and the owner that it is refused on.
    Composing the profile reads the YAML of the named profiles, so it runs off
    the loop as the read did.
    """
    session_id = start.session_id
    if session_id and start.owner is not None:
        raise SessionForkError(not_a_root(session_id, start.owner.root), "SESSION_IS_SUBAGENT")
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
            # Read from the session that was just written, which now says that
            # the version was taken back. The write and the read use one store.
            environment = logged_environment(session)
        starting = replace(
            await anyio.to_thread.run_sync(session_profile, requested, environment),
            withdrawn=withdrawn_note(adopting.name, reason),
        )
        ctx = await exits.enter_async_context(mounted(starting.profile, project=project))
    return ctx, starting


def _refusal(error: LoaderError | MountRefusal | ValidationError) -> str:
    """Why a profile would not mount, in one line for its log and for the user.
    The message of pydantic itself runs to a paragraph for each field, with a
    link each."""
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
    """Mount, open a session, drive one prompt to idle, flush, and then yield.

    This is the sequence that every one-shot mode shares. `before` runs after the
    session exists and before the prompt, for a mode that needs to attach a
    listener.

    The session is *opened*, not created. A `session_id` that is already on disk
    is resumed, so a second `phern -p --session x` is one more turn of one
    conversation and not a second log written over the first (P5-03). A session
    that another process holds is refused.

    The turn is always opened with a message that this function builds, and not
    with `agent.prompt`. With nothing attached the two are identical, so the
    alternative would be a branch whose two halves must stay in step.

    **A resumed session mounts in the environment of its own log** (S5).
    `profile` is what a new session is created on. A session that already has a
    base comes back as its log says (`mount_session`), whatever `--profile` this
    run was given. A named profile that moved since then is kept, and stderr
    says so (S6), because nobody is here to ask. The same applies to an adopted
    version that the loader refused and that this start took back.
    """
    async with AsyncExitStack() as exits:
        start = await read_start(session_id)
        ctx, starting = await mount_session(exits, start, profile)
        if starting.withdrawn is not None:
            err.print(starting.withdrawn, style="yellow", markup=False)
        elif starting.change is not None and session_id:
            err.print(kept_note(starting.change, session_id), style="yellow", markup=False)
        # The route is resolved inside the mount, which is the only place that
        # knows which providers an adapter serves. It is resolved before the
        # session opens, so a route that nothing can run ends the command with
        # its refusal and leaves nothing on disk.
        choose(ctx, choice)
        session = await open_root(ctx, start)
        if before is not None:
            before(ctx, session)
        # The choice of this start is an override of the model of the session
        # where it differs (S4). The agent then runs on the default of the
        # session itself, which includes a `/model` that was logged before.
        entry = await start_on(ctx, session, choice, source="cli", command=choice.flags)
        # This runs before the agent exists. A file that cannot be read must fail
        # the command and not a turn: nothing is logged, and there is nothing to
        # unwind.
        refs = await ingest(ctx, attachments)
        agent = ctx.require(AGENTS).create(session, entry.options())
        # The children are settled before the turn of the parent can ask about
        # them. A session that this run minted has none, and would only pay for
        # a scan of its family to learn that.
        if session_id is not None:
            await resume_children(ctx, agent)
        agent.followup(prompt_message(prompt, refs))
        await agent.run()
        await ctx.require(SESSIONS).flush(session)
        yield ctx, session
