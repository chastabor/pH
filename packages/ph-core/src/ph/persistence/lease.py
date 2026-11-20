"""One session log has one writer: the I-5 lease, which the store takes (P5-03).

Two processes that append to one log put `seq` backwards in the middle of the
file. That breaks A1, and every fold then counts events twice. `_readmit` then
refuses the whole log ("seed must be contiguous from 0"). So the session is
not only confused: nothing can open it. The lease refuses the second writer
before any of that happens.

**The store claims, and not the host.** Earlier, the daemon leased its roots
and nothing else took a lease. So `phern -p --session x` appended anyway
against a log that a daemon held, or against a log that another `phern -p` had
just finished. Two one-shot runs on one id were enough to lose both. The store
is the writer, so the claim belongs to the store. Every session is opened
through `ph.persistence.open_session`, which asks the store. That covers the
roots of a daemon, the one-shot modes, and the log of every sub-agent. The
sub-agent logs were opened below `ph_app`, where `open_session` used to be,
and nothing claimed them until L2. A host that forgets is a missing call and
not a missing mechanism. `ClaimingStore` is optional (`protocol.py`). A backend
with no per-session file has nothing to lock. It must say so, and it must not
lock a path that protects nothing.

`timeout=0`: "somebody else holds it" is known immediately, and it is a
refusal. It is not something to wait for. Blocking here would stall the event
loop for every other root that a daemon runs.

The lease is taken through `scope.effect`, which is the one mechanism of the
repository for an acquired external artifact. So the release is structural and
does not depend on a caller remembering it (§4.9, I2). A passivated root gives
its lease back when it unwinds. A crashed process gives it back when it dies,
because the open descriptor holds the `flock`.

**`thread_local=False` is required, and nothing reports its absence.** By
default, filelock keeps its re-entrancy counter in a thread-local. A lease
that is acquired on a worker thread and released from the event loop then
finds a counter of zero and returns *having released nothing*. There is no
error and no warning, and the lock file is held until the process dies. The
lease belongs to the process, and not to the thread that took it.

@module ph.persistence.lease
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from ..locks import LockBusy, acquire_file_lock

if TYPE_CHECKING:
    from ..cordis import Context

__all__ = ["LEASES", "SessionBusy", "claim_session", "lease_directory", "leased", "take_lease"]


class SessionBusy(Exception):
    """Another process is writing the log of this session.

    `code` is what `respond` puts in `data.reason`, so a client branches on the
    name and not on prose. The string is the same whichever host refused. That
    is the purpose of the store owning it: the daemon, rpc mode and a one-shot
    run cannot spell one fact three ways.
    """

    code = "session_already_active"


LEASES = ".leases"
"""The directory that holds one lock file for each session, under the sessions root.

It is a directory of its own, and its name starts with a dot. So nothing that
lists what is stored takes a lease for a lineage. See `families.family_dirs`,
which states the exclusion. `claim_session` explains why a lease is here at all
and not beside the log that it guards.
"""


_HELD: WeakKeyDictionary[Context, set[str]] = WeakKeyDictionary()
"""The ids that each scope already holds a lease on, so that a second claim from
the scope does nothing.

The keys are weak references, so the entry of a scope that has unwound goes
away with the scope."""


def lease_directory(root: Path) -> Path:
    """Where every lease under a sessions root lives."""
    return root / LEASES


def lease_path(root: Path, session_id: str) -> Path:
    """Where this session's lease lives. Derived from the id and nothing else."""
    return lease_directory(root) / f"{session_id}.lock"


def take_lease(root: Path, session_id: str) -> Callable[[], None]:
    """Take the lease of this session now, or raise `SessionBusy`. Returns the release.

    This is the one place that states the path of the lease, its zero wait and
    its refusal. `claim_session` holds the lease for the life of a scope.
    `leased` holds it for a block. The block is the repair of a damaged log
    (`truncate_damaged`), which must not cut a file while another process
    appends to it.
    """
    try:
        return acquire_file_lock(
            lease_path(root, session_id), timeout=0, what=f'session "{session_id}"'
        )
    except LockBusy as busy:
        raise SessionBusy(f'session "{session_id}" is already active in another process') from busy


@contextmanager
def leased(root: Path, session_id: str) -> Iterator[None]:
    """Hold this session's lease for the block, or raise `SessionBusy`."""
    release = take_lease(root, session_id)
    try:
        yield
    finally:
        release()


async def claim_session(scope: Context, root: Path, session_id: str) -> None:
    """Hold this session's lease for as long as `scope` lives, or raise `SessionBusy`.

    **The lease is keyed by the id, and not by the path of the log.** That
    difference is the defect that this function replaces. The lock used to be
    `<log>.lock`, derived through `_path_for` of the store. `_path_for` answers
    with the *family* that a log is filed under, and the family of a session is
    not known until the session is created. A root that was claimed before
    creation locked `<id>/<id>.jsonl.lock`, and `create` then filed the log
    under `<cwd-tag>-<id>`. A second process found the log on disk and locked
    the path beside it. It was granted a lease that the first process did not
    hold. That gave two writers on one log, which is the one thing that I-5
    exists to refuse.

    A store is free to move its files. The id is the fixed point.

    **A second claim from the same scope holds what the scope already holds.**
    The lease refuses another *process*. Within one mount, the store refuses a
    second live `Session` for an id (`SessionStore.adopt`), so there is no
    second writer to keep out. One mount does open a log twice. The resume
    sweep writes the ending of a sub-agent, or its credential hold, into the
    log of that sub-agent. It then releases the log, and may later readmit the
    sub-agent (Phase 11). `flock` belongs to the open descriptor, so a second
    lock file handle from this process would refuse its own holder.
    """
    held = _HELD.setdefault(scope, set())
    if session_id in held:
        return
    # Taken before the await, so two claims racing in one scope do not both lock.
    held.add(session_id)

    def acquire() -> Callable[[], None]:
        unlock = take_lease(root, session_id)

        def release() -> None:
            held.discard(session_id)
            unlock()

        return release

    try:
        await scope.effect(acquire, label=f"session-lease({session_id})")
    except BaseException:
        held.discard(session_id)
        raise
