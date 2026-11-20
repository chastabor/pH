"""Opening a session: claim it, then resume it or create it, in one place (I-5).

@module ph.persistence.opening
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial
from typing import Any

import anyio

from ..cordis import Context
from ..keys import SESSION_PERSISTENCE, SESSIONS, SUBAGENTS
from ..seams.telemetry import ops_record
from ..session import (
    Session,
    SessionEvent,
    SessionForkError,
    SessionHeader,
    new_session_id,
    valid_session_id,
)
from ..session_profile import opened
from .lease import SessionBusy
from .protocol import ClaimingStore, SessionPersistence, read_if_stored
from .resume import resume_session

__all__ = ["open_session", "stored_session"]

log = logging.getLogger("ph.persistence.opening")


async def open_session(
    ctx: Context,
    session_id: str | None = None,
    *,
    family: str | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Session:
    """Claim a session against every other writer, then resume it or create it (I-5).

    **Every session is opened through this one function**: the roots of the
    daemon, the one-shot modes, rpc mode, and the own log of every sub-agent.
    Before it, each host wrote its own, and the one-shot path did not resume at
    all. `phern -p --session x` on an id already on disk *created* a second
    session over the file of the first one. The store saw the file, skipped the
    header and appended `seq` from zero. So two plain `phern -p` runs on one id,
    with no daemon and no race, left a log that the trajectory reader refuses
    outright (P5-03). Resuming was the behavior of the daemon, and now every
    host resumes.

    The claim comes first, and it comes from the store (`ClaimingStore`). The
    writer owns the lock, so a daemon, an rpc peer and a print run refuse each
    other with one code, `session_already_active`. A backend that cannot claim
    is reported with a warning and is not skipped silently. A silent skip is
    the shape that hides two daemons on one session.

    **The lease lives as long as the mount** (`ctx.root`), whichever scope of
    the mount `ctx` is. The lease is given back when the mount unwinds, after
    `write_on_unwind` has written every live log. `protocol.write_on_unwind`
    says why that has to be the last act of the mount. So a row that opens a
    log, such as the log of a sub-agent, cannot give the lease a shorter life by
    passing its own scope.

    **This function is in ph-core since the L2 fix.** It lived in
    `ph_app.runtime`, which ph-rlm sits below and cannot import. So the log of a
    sub-agent was resumed or created with no claim at all. That was the one
    writer that I-5 did not see, and a mass restart is when a second process is
    most likely to open such a log.

    `meta` reaches the *header* of a session created here. The header is storage
    metadata beside the log and not an event in it. So `meta` describes where
    and why this conversation began, and it does not become something that the
    model reads or that a replay has to apply again. `SessionHeader` validates
    that `cwd` is absolute. So a relative path is refused, and it is not
    resolved against the working directory of this process, which is not the
    working directory of the caller.

    `family` says where the session is filed, when the caller knows it. The
    session is looked for there and created there. It is exact, as `read` takes
    it: one `stat`, where a lookup by id alone searched every family in the
    store. The family of a child is the family of its parent
    (`open_child_log`). The family of a root is not in its id, because a fork is
    filed in the family of its source. So it comes from a host that has already
    found the log. Without a family, the log is searched for off the loop, one
    time: the read that finds it is the read that the resume uses
    (`read_if_stored`).
    """
    resolved = session_id or new_session_id()
    store = await _claimed(ctx, resolved)
    # An id minted here is new: the timestamp of one second and 24 random bits.
    # So it is not looked for. That search would list the store for a log that
    # cannot be there. A clash in the same second between two live processes is
    # for the claim to refuse.
    session = (
        await _resumed(ctx, store, resolved, family) if store is not None and session_id else None
    )
    resumed = session is not None
    if session is None:
        fields = dict(meta or {})
        if family is not None:
            fields["family"] = family
        session = ctx.require(SESSIONS).create(resolved, meta=fields or None)
    # The guard is held from here to the return (`SessionStore.opening`), as
    # `resume_session` held it up to here. A session whose children could not be
    # read, or whose start was refused, is let go and not left in the store for a
    # peer to run on.
    async with ctx.require(SESSIONS).opening(session):
        if resumed:
            # Read its children, from their own logs (Phase 11). The children of a
            # resumed session are on disk, not in its log, so they are read as the
            # session opens. That happens on every host, and not only where the
            # sweep of a daemon would have read them.
            subagents = ctx.get(SUBAGENTS)
            if subagents is not None:
                await subagents.load_children(session.id, session.header.family)
        # Record the environment that the session starts in, before anything runs
        # in it (S3). The record is made here because every root is opened through
        # this function. A root that is older than the record gets its first one
        # as it is opened. Then the options of this start are logged where they
        # differ, and the mount is brought to what the log says (S4). `opened`
        # does all of this, in that order. A start that `opened` refuses lets the
        # session go, and with it the records that the start refused to follow.
        await opened(ctx, session)
    return session


async def stored_session(ctx: Context, session_id: str, *, family: str | None = None) -> Session:
    """A stored session, claimed and loaded but **not resumed**. It is for adding a
    record to a session while nothing runs it. `phern profiles adopt` uses it
    (session profiles, S6).

    The session is claimed as `open_session` claims it, for as long as the mount
    of `ctx`. So a daemon or a print run that starts it meanwhile is refused and
    not interleaved. It is not resumed, because nothing runs: no tail is
    repaired, no call is reconciled, and no `session/resumed` is recorded. A
    crashed tail stays for the next start to close, when the tools that can
    answer for it are mounted. What is appended is written by the caller, or
    when the mount unwinds.

    What a log read off disk owes is still settled before the caller holds it
    (`SessionStore.loaded`).

    `family` is where the log is filed, when the caller holds it. The log is
    then read by that path (`SessionArchive.read`) and is not searched for. A
    parent that writes to its own child holds the family. It is exact, not a
    hint: a log that is not there is not looked for elsewhere.

    :raises SessionBusy: when another process holds it.
    :raises LookupError: when there is no stored session by that id.
    """
    store = ctx.require(SESSION_PERSISTENCE)
    await _claimed(ctx, session_id)
    found = await _read(store, session_id, family)
    if found is None:
        raise LookupError(f"no stored session {session_id!r}")
    header, events = found
    stored = Session(session_id, seed=events, header=header, durable=len(events))
    sessions = ctx.require(SESSIONS)
    async with sessions.opening(sessions.adopt(stored)) as session:
        await sessions.loaded(session)
    return session


async def _resumed(
    ctx: Context, store: SessionPersistence, session_id: str, family: str | None
) -> Session | None:
    """The stored session, resumed, or `None` when nothing is stored under its id.

    Only this frame holds the log as read. So that copy is released when the
    resume returns, and it does not stay beside the session that it seeded for
    the rest of the open.
    """
    found = await _read(store, session_id, family)
    return None if found is None else await resume_session(ctx, session_id, *found)


async def _read(
    store: SessionPersistence, session_id: str, family: str | None
) -> tuple[SessionHeader, list[SessionEvent]] | None:
    """`read_if_stored`, off the loop. Both opening functions start with this read.
    A long log is a long read, and a search for a log whose family is not given
    lists the store. The loop serves every root."""
    return await anyio.to_thread.run_sync(partial(read_if_stored, store, session_id, family=family))


async def _claimed(ctx: Context, session_id: str) -> SessionPersistence | None:
    """Refuse an id that no path may be built from, then claim it from the store of
    `ctx` (I-5).

    This is the half of opening a session that `open_session` and
    `stored_session` share. **The check runs before anything builds a path from
    the id** (K9). `SessionStore.create` and `.adopt` check the id too, and by
    then it is too late. `claim` creates `<root>/.leases/<id>.lock`, and a read
    locates `<root>/<id>/<id>.jsonl` and *opens and parses* it. All of that uses
    the raw id, and all of it happens before a `Session` object exists for the
    store to refuse.

    The lease lasts as long as the mount of `ctx`. A store that cannot claim is
    reported with a warning and is not skipped silently. A refusal is an `ops`
    fact and not a session fact, because the session that it concerns is the
    one that this process was just refused.

    :raises SessionForkError: when the id is not usable as a path component.
    :raises SessionBusy: when another process holds it.
    """
    if not valid_session_id(session_id):
        raise SessionForkError(
            f'session id "{session_id}" is not usable as a path component', "SESSION_ID_INVALID"
        )
    store = ctx.get(SESSION_PERSISTENCE)
    if isinstance(store, ClaimingStore):
        try:
            await store.claim(session_id, scope=ctx.root)
        except SessionBusy:
            await ops_record(
                ctx,
                "session refused: already active in another process",
                severity="warn",
                session_id=session_id,
            )
            raise
    elif store is not None:
        log.warning(
            "ph.persistence: %s cannot claim a session; I-5 is not enforced for %s",
            type(store).__name__,
            session_id,
        )
        await ops_record(
            ctx,
            "I-5 is not enforced: this session store cannot claim a session",
            severity="warn",
            session_id=session_id,
            store=type(store).__name__,
        )
    return store
