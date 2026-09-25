"""Opening a session: claim it, then resume it or create it — the one door (I-5).

@module ph.persistence.opening
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from ..cordis import Context
from ..keys import SESSION_PERSISTENCE, SESSIONS
from ..seams.telemetry import ops_record
from ..session import Session, SessionForkError, new_session_id, valid_session_id
from .jsonl import resume_session
from .lease import SessionBusy
from .protocol import ClaimingStore

__all__ = ["open_session"]

log = logging.getLogger("ph.persistence.opening")


async def open_session(
    ctx: Context,
    session_id: str | None = None,
    *,
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
    """
    resolved = session_id or new_session_id()
    # **Before anything builds a path from it** (K9). `SessionStore.create` and
    # `.adopt` check this too, and by then it is too late here: `claim` below
    # creates `<root>/.leases/<id>.lock`, `exists` locates `<root>/<id>/<id>.jsonl`,
    # and `resume_session` *opens and parses* that file — all from the raw id, and
    # all before a `Session` object exists for the store to refuse. Every session
    # is opened through here, so it is where the id stops being arbitrary.
    if not valid_session_id(resolved):
        raise SessionForkError(
            f'session id "{resolved}" is not usable as a path component', "SESSION_ID_INVALID"
        )
    store = ctx.get(SESSION_PERSISTENCE)
    if isinstance(store, ClaimingStore):
        try:
            await store.claim(resolved, scope=ctx.root)
        except SessionBusy:
            # An `ops` fact, not a session one: the session it concerns is the
            # one this process was just refused, so its log is not ours to write.
            await ops_record(
                ctx,
                "session refused: already active in another process",
                severity="warn",
                session_id=resolved,
            )
            raise
    elif store is not None:
        log.warning(
            "ph.persistence: %s cannot claim a session; I-5 is not enforced for %s",
            type(store).__name__,
            resolved,
        )
        await ops_record(
            ctx,
            "I-5 is not enforced: this session store cannot claim a session",
            severity="warn",
            session_id=resolved,
            store=type(store).__name__,
        )
    if store is not None and store.exists(resolved):
        return await resume_session(ctx, resolved)
    return ctx.require(SESSIONS).create(resolved, meta=dict(meta) if meta else None)
