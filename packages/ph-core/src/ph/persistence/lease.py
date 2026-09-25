"""One session log, one writer — the I-5 lease, taken by the store (P5-03).

Two processes appending to one log put `seq` backwards mid-file, which breaks
A1 and makes every fold double-count — and `_readmit` then refuses the whole
log ("seed must be contiguous from 0"), so the session is not merely confused
but unopenable by anything. The lease is what refuses the second writer first.

**The store claims, not the host.** The daemon leased its roots and nothing
else did, so `phern -p --session x` against a log a daemon held — or against a
log another `phern -p` had just finished — appended anyway, and two one-shot runs
on one id were enough to lose both. The writer is the store, so the claim is
the store's: every session is opened through `ph.persistence.open_session`, which
asks the store — a daemon's roots, the one-shot modes, and every sub-agent's log
(which, opened below `ph_app` where that door used to live, were claimed by
nothing until L2) — and a host that forgets is a missing call rather than a
missing mechanism. `ClaimingStore` is optional (`protocol.py`) because a backend with
no per-session file has nothing to lock and must say so rather than lock a
path that protects nothing.

`timeout=0`: "somebody else holds it" is known immediately and is a refusal,
not something to wait out — blocking here would stall the event loop for every
other root a daemon runs.

Through `scope.effect`, the repo's one mechanism for an acquired external
artifact, so the release is structural rather than remembered (§4.9, I2): a
passivated root gives its lease back by unwinding, and a crashed process gives
it back by dying, since `flock` is held by the open descriptor.

**`thread_local=False` is load-bearing, and its absence is silent.** filelock
keeps its re-entrancy counter in a thread-local by default, so a lease acquired
on a worker thread and released from the event loop finds a counter of zero
and returns *having released nothing* — no error, no warning, and a lock file
held until the process dies. The lease belongs to the process, not to
whichever thread took it.

@module ph.persistence.lease
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ..locks import LockBusy, acquire_file_lock

if TYPE_CHECKING:
    from ..cordis import Context

__all__ = ["LEASES", "SessionBusy", "claim_session"]


class SessionBusy(Exception):
    """Another process is writing this session's log.

    `code` is what `respond` puts in `data.reason`, so a client branches on the
    name rather than on prose — the same string whichever host refused, which
    is the point of the store owning it: the daemon, rpc mode and a one-shot run
    cannot spell one fact three ways.
    """

    code = "session_already_active"


LEASES = ".leases"
"""The directory holding one lock file per session, under the sessions root.

Dotted, and its own directory, so a lease is never mistaken for a lineage by
anything listing what is stored — see `families.family_dirs`, which states the
exclusion. `claim_session` has the argument for why a lease lives here at all
rather than beside the log it guards.
"""


def lease_path(root: Path, session_id: str) -> Path:
    """Where this session's lease lives. Derived from the id and nothing else."""
    return root / LEASES / f"{session_id}.lock"


async def claim_session(scope: Context, root: Path, session_id: str) -> None:
    """Hold this session's lease for as long as `scope` lives, or raise `SessionBusy`.

    **Keyed by the id, not by the log's path**, and that distinction is the
    defect this replaces. The lock used to be `<log>.lock`, derived through the
    store's `_path_for` — which answers with the *family* a log is filed under,
    and a session's family is not known until it is created: a root claimed
    before creation locked `<id>/<id>.jsonl.lock`, while `create` then filed the
    log under `<cwd-tag>-<id>`. A second process, finding the log on disk, locked
    the path beside it and was granted a lease the first process was not holding.
    Two writers on one log, which is the one thing I-5 exists to refuse.

    A store is free to move its files; this is the fixed point.
    """

    def acquire() -> Callable[[], None]:
        try:
            return acquire_file_lock(
                lease_path(root, session_id), timeout=0, what=f'session "{session_id}"'
            )
        except LockBusy as busy:
            raise SessionBusy(
                f'session "{session_id}" is already active in another process'
            ) from busy

    await scope.effect(acquire, label=f"session-lease({session_id})")
