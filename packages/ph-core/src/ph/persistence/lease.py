"""One session log, one writer — the I-5 lease, taken by the store (P5-03).

Two processes appending to one log put `seq` backwards mid-file, which breaks
A1 and makes every fold double-count — and `_readmit` then refuses the whole
log ("seed must be contiguous from 0"), so the session is not merely confused
but unopenable by anything. The lease is what refuses the second writer first.

**The store claims, not the host.** The daemon leased its roots and nothing
else did, so `ph -p --session x` against a log a daemon held — or against a
log another `ph -p` had just finished — appended anyway, and two one-shot runs
on one id were enough to lose both. The writer is the store, so the claim is
the store's: every host opens a session through `open_session`, which asks the
store, and a host that forgets is a missing call rather than a missing
mechanism. `ClaimingStore` is optional (`protocol.py`) because a backend with
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

from filelock import FileLock, Timeout

if TYPE_CHECKING:
    from ..cordis import Context

__all__ = ["SessionBusy", "claim_file"]


class SessionBusy(Exception):
    """Another process is writing this session's log.

    `code` is what `respond` puts in `data.reason`, so a client branches on the
    name rather than on prose — the same string whichever host refused, which
    is the point of the store owning it: the daemon, rpc mode and a one-shot run
    cannot spell one fact three ways.
    """

    code = "session_already_active"


async def claim_file(scope: Context, path: Path, session_id: str) -> None:
    """Hold `<path>.lock` for as long as `scope` lives, or raise `SessionBusy`."""

    def acquire() -> Callable[[], None]:
        # No mkdir: filelock's own `ensure_directory_exists` is the same
        # `parents=True, exist_ok=True` call on the same directory.
        lock = FileLock(f"{path}.lock", timeout=0, thread_local=False)
        try:
            lock.acquire()
        except Timeout as error:
            raise SessionBusy(
                f'session "{session_id}" is already active in another process'
            ) from error
        return lock.release

    await scope.effect(acquire, label=f"session-lease({session_id})")
