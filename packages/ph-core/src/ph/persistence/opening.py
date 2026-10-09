"""Opening a session: claim it, then resume it or create it — the one door (I-5).

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

    **The one door every session is opened through** — the daemon's roots, the
    one-shot modes, rpc mode, and every sub-agent's own log. Before it each host
    wrote its own, and the one-shot path did not resume at all: `phern -p --session x`
    on an id already on disk *created* a second session over the first one's file —
    the store saw the file, skipped the header and appended `seq` from zero — so two
    plain `phern -p` runs on one id, with no daemon and no race, left a log the
    trajectory reader refuses outright (P5-03). Resuming was the daemon's behavior
    and is now everyone's.

    The claim comes first and comes from the store (`ClaimingStore`): the writer
    owns the lock, so a daemon, an rpc peer and a print run refuse each other with
    one code, `session_already_active`. A backend that cannot claim is said out
    loud rather than skipped — a silent skip is the shape that hides two daemons
    on one session.

    **The lease lives as long as the mount** (`ctx.root`), whichever scope of it
    `ctx` is: it is given back when the mount unwinds, after `write_on_unwind` has
    written every live log (`protocol.write_on_unwind` says why that has to be the
    mount's last act). So a row opening a log, a sub-agent's, cannot hand the lease
    a shorter life by passing its own scope.

    **In ph-core since the L2 fix.** It lived in `ph_app.runtime`, which ph-rlm sits
    below and cannot import, so a sub-agent's log was resumed or created with no
    claim at all — the one writer I-5 did not see, and a mass restart is when a
    second process is likeliest to open one.

    `meta` reaches the *header* of a session created here, which is storage
    metadata beside the log rather than an event in it — so it describes
    where and why this conversation began without becoming something the model
    reads or a replay has to re-apply. `SessionHeader` validates that `cwd` is
    absolute, so a relative path is refused rather than resolved against this
    process's own working directory, which is not the caller's.

    `family`, when the caller knows where the session is filed — a child's is its
    parent's (`open_child_log`) — is where it is looked for and where it is created.
    Exact, as `read` takes it: a child a restart reopens is one `stat` in its
    parent's directory, where by id alone it was searched for across every family in
    the store, once per child. A root has none to give, since a fork's family is not
    in its id, so its log is searched for — off the loop, and **once**: the read
    that finds it is the resume's (`read_if_stored`), where asking `exists` and
    then reading looked it up twice.
    """
    resolved = session_id or new_session_id()
    store = await _claimed(ctx, resolved)
    # An id minted here is new — a second's timestamp and 24 random bits — so it is
    # not looked for: that search lists the store for a log that cannot be there,
    # and a clash in the same second between two live processes is the claim's to
    # refuse.
    session = (
        await _resumed(ctx, store, resolved, family) if store is not None and session_id else None
    )
    resumed = session is not None
    if session is None:
        fields = dict(meta or {})
        if family is not None:
            fields["family"] = family
        session = ctx.require(SESSIONS).create(resolved, meta=fields or None)
    # Held from here to its return (`SessionStore.opening`), as `resume_session` held
    # it to here: a session whose children could not be read, or whose start was
    # refused, is let go rather than left in the store for a peer to run on.
    async with ctx.require(SESSIONS).opening(session):
        if resumed:
            # Its children, from their own logs (Phase 11): a resumed session's family
            # is on disk, not in its log, so it is read as the session opens — on
            # every host, not only where a daemon's sweep would have read it.
            subagents = ctx.get(SUBAGENTS)
            if subagents is not None:
                await subagents.load_children(session.id, session.header.family)
        # The environment it starts in, before anything runs in it (S3): here, because
        # this is the door every root comes through, and a root from before the record
        # gets its first on the way in. Then this start's own options, logged where
        # they differ, and the mount brought to what the log says (S4) — `opened` is
        # the whole of it, in that order. A start it refuses lets the session go, and
        # with it the records it refused to follow.
        await opened(ctx, session)
    return session


async def stored_session(ctx: Context, session_id: str, *, family: str | None = None) -> Session:
    """A stored session, claimed and loaded but **not resumed**: for adding a record
    to while nothing runs it — `phern profiles adopt` (session profiles, S6).

    Claimed as `open_session` claims, for as long as `ctx`'s mount, so a daemon or
    a print run starting it meanwhile is refused rather than interleaved. Not
    resumed, because nothing runs: no tail is repaired, no call reconciled, and no
    `session/resumed` recorded — a crashed tail stays the next start's to close,
    with the tools that can answer for it mounted. What is appended is written by
    the caller, or when the mount unwinds.

    What a log read off disk owes is still settled before the caller holds it
    (`SessionStore.loaded`).

    `family` is where its log is filed, when the caller holds it, and then it is read
    by that path (`SessionArchive.read`) rather than searched for — a parent writing to
    its own child holds it. Exact, not a hint: a log not there is not looked for.

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

    The log as read is this frame's alone, so it goes when the resume returns rather
    than staying beside the session it seeded for the rest of the open.
    """
    found = await _read(store, session_id, family)
    return None if found is None else await resume_session(ctx, session_id, *found)


async def _read(
    store: SessionPersistence, session_id: str, family: str | None
) -> tuple[SessionHeader, list[SessionEvent]] | None:
    """`read_if_stored`, off the loop — the read both doors open with: a long log is
    a long read, a search for one whose family is not given lists the store, and the
    loop is every root's."""
    return await anyio.to_thread.run_sync(partial(read_if_stored, store, session_id, family=family))


async def _claimed(ctx: Context, session_id: str) -> SessionPersistence | None:
    """Refuse an id no path may be built from, then claim it from `ctx`'s store (I-5).

    The half of opening a session that `open_session` and `stored_session` share.
    **Before anything builds a path from it** (K9). `SessionStore.create` and
    `.adopt` check the id too, and by then it is too late: `claim` creates
    `<root>/.leases/<id>.lock`, and a read locates `<root>/<id>/<id>.jsonl` and
    *opens and parses* it — all from the raw id, and all before a
    `Session` object exists for the store to refuse.

    The lease lasts as long as `ctx`'s mount. A store that cannot claim is said out
    loud rather than skipped; a refusal is an `ops` fact, not a session one, since
    the session it concerns is the one this process was just refused.

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
