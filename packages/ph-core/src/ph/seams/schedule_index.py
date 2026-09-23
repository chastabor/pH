"""Which sessions hold a live appointment, and when each is next due (P6-23).

**A schedule outlives its process; the thing that reads it does not.** P5-06 is
right about the log — `schedule/created` is still there on Wednesday — and a
fresh daemon still fires nothing, because `Supervisor.tick` iterates the roots it
has mounted and a boot has none. The appointment survives and is never kept,
which fails in the worst available shape: **silence**. Nothing errors, nothing
logs, and it is found by somebody noticing a run that did not happen.

**The obvious fix is the wrong one, and this file is the shape of the right one.**
"Mount every stored session at boot" fails three ways. `StoredSession` carries no
schedule, so "does this log hold an appointment" is answerable only by reading and
folding the whole log — 500 reads before the daemon answers a connection, to find
the three that matter. Mounting is not cheap: a root is a whole profile, a
workspace, possibly a kernel, which is the cost P5-05 exists to *release*. And
`Supervisor.start` takes P5-03's **lease**, so auto-mounting everything claims
every session on the machine and the next `phern -p` over any of them is refused with
`session_already_active` — a strictly worse failure, because it is loud,
immediate, and hits sessions with no schedule at all.

So this is an **index**: the seam knows the moment an appointment is created,
canceled or fired, so it records which sessions have one and when each is next
due. A daemon reads one small file and mounts only what is actually due.

**It is a cache, and the logs stay authoritative (I-6).** Every value here is
derived from a log that still holds it, so a missing, stale or corrupt index costs
a late run and never a wrong one: a reader that finds nothing falls back to the
behavior that shipped before this existed. The projection is never the source of
truth, which is what keeps it from becoming a second answer to "what is
scheduled" (A11).

**Reconciliation needs no scan, because opening a session is the rebuild.** The
seam re-derives an entry from the log on every `session/created`, so an index that
is missing, stale or written by a build that had none corrects itself the moment
anything touches that session — which is exactly the condition the old behavior
required to fire a schedule at all. A wholesale rebuild would have to read every
stored log, which is the scan this file exists to avoid.

@module ph.seams.schedule_index
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import anyio

from ..locks import LockBusy, file_lock
from ..paths import write_atomic

if TYPE_CHECKING:
    from ..cordis import Context

__all__ = ["INDEX_NAME", "Appointment", "IndexRecorder", "IndexWriter", "ScheduleIndex"]

log = logging.getLogger("ph.seams.schedule_index")

INDEX_NAME = "schedules.json"
"""One file per `$PH_HOME`, beside the sessions it indexes.

A single document rather than a file per session: it is read whole on every
daemon tick and written only when an appointment changes, which is the opposite
of the access pattern a directory of files is good at.
"""

_VERSION = 1

_LOCK_TIMEOUT = 5.0
"""How long a write waits for the file lock before giving up (K10).

**Patient, because the wait is a worker thread's.** It was cut to a quarter
second while `record` ran on the event loop — every caller reaches it from the
loop, so the wait was time the daemon served nothing, including every other
root's schedules. `IndexWriter` moved the write off the loop, and a wait there
costs a thread and delays only the writes queued behind it, so waiting out
another host's write is worth more than dropping this one. Giving up is still
the answer past it: an index that cannot be written costs a late run.
"""


@dataclass(frozen=True, slots=True)
class Appointment:
    """One session's next due moment, and when that was last established.

    `updated` is what makes waking bounded. A daily cron refreshes it on every
    firing, so an index a daemon has been serving stays fresh; one nobody has
    served since March goes stale, and a daemon started today can decline to
    resurrect a root its owner abandoned. That question is the caller's — the
    index reports the age and does not enforce a policy with it.
    """

    session_id: str
    next_at: int
    updated: int


@dataclass(slots=True)
class ScheduleIndex:
    """The index as a value, so a caller states where it lives and nothing guesses."""

    root: Path

    @property
    def path(self) -> Path:
        return self.root / INDEX_NAME

    def read(self) -> dict[str, Appointment]:
        """Every appointment on record. Empty when there is nothing to say.

        **Every failure reads as empty**, deliberately: a missing file is the
        ordinary state of a `$PH_HOME` nobody has scheduled in, and a corrupt one
        is a cache that lost its contents. Both mean "this file cannot tell you
        what is due", and the honest response is the behavior that shipped
        before the index existed — a late run — rather than an exception on the
        daemon's boot path.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            log.warning("ph.seams.schedule_index: could not read %s", self.path, exc_info=True)
            return {}
        if not isinstance(raw, dict) or raw.get("version") != _VERSION:
            return {}
        found: dict[str, Appointment] = {}
        for session_id, entry in (raw.get("sessions") or {}).items():
            if not isinstance(entry, dict):
                continue
            next_at, updated = entry.get("nextAt"), entry.get("updated")
            if (
                isinstance(session_id, str)
                and isinstance(next_at, int)
                and isinstance(updated, int)
            ):
                found[session_id] = Appointment(session_id, next_at, updated)
        return found

    def record(self, session_id: str, *, next_at: int | None, now: int) -> None:
        """Set or clear one session's appointment. Blocking — see `IndexWriter`.

        `next_at=None` removes the entry, which is what a cancellation and a
        `once` that has fired both mean: nothing further is owed. Removing rather
        than tombstoning keeps the file the size of the work outstanding instead
        of the size of every schedule ever made.

        Read-modify-write under a lock, because two sessions in two processes
        legitimately schedule at the same moment and each knows only its own
        entry. Best-effort throughout — an index that cannot be written costs a
        late run, and taking a session's own `create` down to protect a cache
        would be the projection outranking the log.

        **A no-op when nothing moved**, which is what makes this callable on every
        session open: reconciling an entry that is already right must not cost a
        lock and a rewrite, and every fork and every subagent opens a session.
        """
        self.record_all({session_id: (next_at, now)})

    def record_all(self, changes: Mapping[str, tuple[int | None, int]]) -> None:
        """`record` for several sessions at once: one read, and at most one rewrite.

        `changes` maps a session to its `(next_at, now)`. What `IndexWriter`
        hands over in one thread hop, so a fan-out opening N sessions costs one
        read of the file rather than N, and a rewrite only if something moved.
        """
        # **The no-op is decided before the lock, not inside it** (K10). This is
        # called on every session open — every fork, every subagent — precisely
        # because reconciling an entry that is already right should cost
        # nothing, and it was paying a lock acquisition to discover that. Reading
        # first is safe: `read` treats every failure as empty and `_write` is an
        # atomic rename, so there is no torn state to observe, and a racing
        # writer only means the re-check under the lock finds nothing to do.
        if not _moves(self.read(), changes):
            return
        try:
            with file_lock(f"{self.path}.lock", timeout=_LOCK_TIMEOUT, what="the schedule index"):
                found = self.read()
                if not _moves(found, changes):
                    return
                for session_id, (next_at, now) in changes.items():
                    if next_at is None:
                        found.pop(session_id, None)
                    else:
                        current = found.get(session_id)
                        if current is None or current.next_at != next_at:
                            found[session_id] = Appointment(session_id, next_at, now)
                self._write(found)
        except LockBusy:
            log.warning("ph.seams.schedule_index: %s is locked; not recording", self.path)
        except OSError:
            log.warning("ph.seams.schedule_index: could not write %s", self.path, exc_info=True)

    def _write(self, found: dict[str, Appointment]) -> None:
        """Serialize and rename, so a reader never sees a half-written index."""
        document: dict[str, Any] = {
            "version": _VERSION,
            "sessions": {
                one.session_id: {"nextAt": one.next_at, "updated": one.updated}
                for one in found.values()
            },
        }
        write_atomic(self.path, json.dumps(document, indent=2))


def _moves(found: Mapping[str, Appointment], changes: Mapping[str, tuple[int | None, int]]) -> bool:
    """Whether writing `changes` over `found` would change anything."""
    for session_id, (next_at, _now) in changes.items():
        current = found.get(session_id)
        if next_at is None and current is not None:
            return True
        if next_at is not None and (current is None or current.next_at != next_at):
            return True
    return False


class IndexRecorder(Protocol):
    """What `ScheduleService` writes through: `ScheduleIndex` itself, or its writer."""

    def record(self, session_id: str, *, next_at: int | None, now: int) -> None: ...


@dataclass(slots=True)
class IndexWriter:
    """`ScheduleIndex.record` without the wait on the event loop (K10).

    Every caller — `create`, `cancel`, `claim`, the `session/created` reconcile —
    is synchronous and on the loop, and `record` is a file lock, a read and an
    atomic rewrite. So `record` here only notes the change, and a writer hands
    everything noted so far to `record_all` in one worker-thread hop.

    **Latest wins, per session.** The index holds one entry per session, so only
    the newest change for each matters: a `create` then a `cancel` noted before
    the writer ran is one write of the cancel, not two writes in order. A change
    noted *while* a batch is being written waits for the next one, so it still
    lands last.

    **Started on demand and finished when nothing is noted**, not a task that
    lives as long as the row: `Context.drain` awaits detached work, so a writer
    that never ended would hold every drain to its deadline — while one that ends
    makes `drain` the flush, which is what a host shutting down wants.
    """

    index: ScheduleIndex
    ctx: Context
    _pending: dict[str, tuple[int | None, int]] = field(default_factory=dict)
    _writing: bool = False

    def record(self, session_id: str, *, next_at: int | None, now: int) -> None:
        """Note one change; start the writer if none is running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop, so no loop to stall — and `detach` would close the writer
            # unstarted, leaving `_writing` set and every later change noted for
            # a writer that never ran.
            self.index.record(session_id, next_at=next_at, now=now)
            return
        self._pending[session_id] = (next_at, now)
        if not self._writing:
            self._writing = True
            self.ctx.detach(self._drain(), label="schedule index writer")

    async def _drain(self) -> None:
        # No await between the emptiness check and clearing the flag, so a change
        # noted during the last batch is either taken by the `while` or finds the
        # flag clear and starts a writer of its own.
        try:
            while self._pending:
                batch, self._pending = self._pending, {}
                await anyio.to_thread.run_sync(self.index.record_all, batch)
        finally:
            self._writing = False
