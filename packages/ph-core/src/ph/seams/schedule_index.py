"""Which sessions hold a live appointment, and when each is next due (P6-23).

**A schedule outlives its process, and the reader of the schedule does not.**
P5-06 is right about the log: `schedule/created` is still there on Wednesday.
But a fresh daemon still fires nothing, because `Supervisor.tick` iterates the
roots that it has mounted, and a daemon has none at boot. The appointment
survives and is never kept. That failure has the worst form: **silence**.
Nothing raises an error and nothing is logged. Somebody finds the failure by
noticing a run that did not happen.

**The obvious fix is wrong, and this file has the form of the right fix.** "Mount
every stored session at boot" fails in three ways.

- `StoredSession` carries no schedule. So "does this log hold an appointment"
  can be answered only by reading and folding the whole log. That is 500 reads
  before the daemon answers a connection, to find the three that matter.
- Mounting is not cheap. A root is a whole profile, a workspace and possibly a
  kernel. P5-05 exists to *release* that cost.
- `Supervisor.start` takes the **lease** of P5-03. To mount everything
  automatically claims every session on the machine. The next `phern -p` over
  any of them is then refused with `session_already_active`. That failure is
  strictly worse, because it is loud and immediate, and it hits sessions with no
  schedule at all.

So this is an **index**. The seam knows the moment at which an appointment is
created, canceled or fired, so it records which sessions have one and when each
is next due. A daemon reads one small file and mounts only what is due.

**The index is a cache, and the logs stay authoritative (I-6).** Every value
here is derived from a log that still holds it. A missing, stale or corrupt
index therefore costs a late run and never a wrong run. A reader that finds
nothing falls back to the behavior that shipped before this index existed. The
projection is never the source of truth, so it does not become a second answer
to "what is scheduled" (A11).

**Opening a session reconciles its entry.** The seam derives the entry from the
log again on every `session/created`. An entry that went stale is therefore
corrected the moment anything opens that session.

**An index that cannot vouch for itself is rebuilt from every log (S18).**
"Until someone opens that session" is silence again for a session that nobody
opens. The scan that this file exists to avoid is paid only then. Two signs say
when:

- **The index says that it is complete.** Only a rebuild writes
  `complete: true`, and a writer keeps the flag that it found. So a file is
  incomplete when it is missing, corrupt or of another version, or when a writer
  that read it as any of those wrote it again.
- **A new appointment leaves a mark until it is written** (`Claim`). The write
  runs after the append to the log, off the event loop. A host that dies or
  fails in between would otherwise lose the appointment without a trace. A mark
  that no live writer holds names that loss.

`Survey` asks both, and a daemon rebuilds when either says so.

@module ph.seams.schedule_index
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import anyio

from ..paths import make_directories, replace_durably
from ._json_index import JsonIndex

if TYPE_CHECKING:
    from ..cordis import Context

__all__ = [
    "CLAIMS_NAME",
    "INDEX_NAME",
    "Appointment",
    "Claim",
    "IndexRecorder",
    "IndexWriter",
    "ScheduleIndex",
    "Survey",
]

log = logging.getLogger("ph.seams.schedule_index")

INDEX_NAME = "schedules.json"
"""One file for each `$PH_HOME`, beside the sessions that it indexes.

The index is a single document and not one file for each session. It is read
whole on every scheduler pass of the daemon and written only when an appointment
changes. A directory of files is good at the opposite access pattern.
"""

CLAIMS_NAME = "schedules.claims"
"""The directory of `Claim`s, beside the index that they vouch for."""


@dataclass(frozen=True, slots=True)
class Appointment:
    """The next due moment of one session, and when that moment was last
    established.

    `updated` is what bounds waking. A daily cron refreshes it on every firing,
    so an index that a daemon has been serving stays fresh. An index that nobody
    has served since March goes stale, and a daemon that starts today can
    decline to bring back a root that its owner abandoned. That decision belongs
    to the caller. The index reports the age and does not enforce a policy with
    it.
    """

    session_id: str
    next_at: int
    updated: int


@dataclass(frozen=True, slots=True)
class Survey:
    """What the index holds, and whether it can vouch for it (S18)."""

    appointments: dict[str, Appointment]
    complete: bool
    """Whether a rebuild wrote this file and every writer since kept it whole."""
    abandoned: tuple[Path, ...]
    """The claims that no live writer holds. Each is a new appointment that
    possibly never reached the file."""

    @property
    def trusted(self) -> bool:
        return self.complete and not self.abandoned


@dataclass(slots=True)
class Claim:
    """A new appointment on its way to the index, marked on disk until it is
    written.

    **The mark is held with an exclusive `flock`** for as long as the change is
    on its way, so the mark says whether its writer is still alive. `close`
    removes the mark when the change is written. Otherwise `close` releases the
    lock and leaves the file. A host that dies leaves the file too, because the
    kernel releases the lock. In both cases a `Survey` finds a claim that nobody
    holds, and a rebuild answers it.
    """

    path: Path
    fd: int

    def close(self, *, written: bool) -> None:
        """Release the lock, and remove the mark if the change is in the index."""
        if written:
            with suppress(FileNotFoundError):
                self.path.unlink()
        os.close(self.fd)


@dataclass(slots=True)
class ScheduleIndex(JsonIndex[Appointment]):
    """The index as a value, so that a caller states where it lives and nothing
    guesses.

    `JsonIndex` does the reading, the locking, the rewriting and the rebuild
    merge of the file, and the live-roots index shares it. This class adds the
    entry of this index and its claims.
    """

    NAME: ClassVar[str] = INDEX_NAME
    VERSION: ClassVar[int] = 2
    """Version 2 added `complete`. A version-1 file reads as incomplete, so it is
    rebuilt one time."""
    WHAT: ClassVar[str] = "the schedule index"

    @property
    def claims(self) -> Path:
        return self.root / CLAIMS_NAME

    def survey(self) -> Survey:
        """The appointments, and whether this file can vouch for them (S18)."""
        appointments, complete = self._load()
        return Survey(appointments, complete, self._abandoned())

    def _decode(self, key: str, raw: Mapping[str, Any]) -> Appointment | None:
        next_at, updated = raw.get("nextAt"), raw.get("updated")
        if isinstance(next_at, int) and isinstance(updated, int):
            return Appointment(key, next_at, updated)
        return None

    def _encode(self, entry: Appointment) -> dict[str, Any]:
        return {"nextAt": entry.next_at, "updated": entry.updated}

    def _abandoned(self) -> tuple[Path, ...]:
        """Every claim whose writer is gone, which is every claim that this
        process can lock."""
        try:
            entries = list(os.scandir(self.claims))
        except FileNotFoundError:
            return ()
        except OSError:
            log.warning("ph.seams.schedule_index: could not list %s", self.claims, exc_info=True)
            return ()
        found: list[Path] = []
        for entry in entries:
            # A claim is locked under a dot name and then renamed into view. A
            # claim that is being made is therefore never taken for one that
            # was left behind (`claim`).
            if entry.name.startswith("."):
                continue
            try:
                fd = os.open(entry.path, os.O_RDONLY)
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue  # held, so its writer is alive — or a lock this cannot judge
            finally:
                os.close(fd)
            found.append(Path(entry.path))
        return tuple(found)

    def claim(self) -> Claim | None:
        """Mark a new appointment on disk before it is handed to a writer.

        **The call blocks on the event loop, because it must come first.** The
        flush of the log follows the append that this mark answers. A mark that
        was written after the flush would leave open the window that the mark
        exists to close. The cost is a create, a lock, a rename and one
        directory sync, and only a new appointment pays it.

        Returns `None` when the mark cannot be made. The write is still
        attempted, and only the trace of its loss is missing.
        """
        name = uuid.uuid4().hex
        hidden, path = self.claims / f".{name}", self.claims / name
        fd: int | None = None
        try:
            make_directories(self.claims)
            fd = os.open(hidden, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            replace_durably(hidden, path)
        except OSError:
            if fd is not None:
                os.close(fd)
                with suppress(OSError):
                    hidden.unlink()
            log.warning(
                "ph.seams.schedule_index: could not claim in %s", self.claims, exc_info=True
            )
            return None
        return Claim(path, fd)

    def record(self, session_id: str, *, next_at: int | None, now: int, new: bool = False) -> None:
        """Set or clear the appointment of one session. The call blocks: see
        `IndexWriter`.

        `next_at=None` removes the entry. A cancellation and a `once` that has
        fired both mean that nothing further is owed. The entry is removed and
        not replaced by a tombstone, so the file stays the size of the
        outstanding work and not the size of every schedule ever made.

        The call is a read-modify-write under a lock, because two sessions in
        two processes can legitimately schedule at the same moment, and each
        knows only its own entry. Every step is best-effort. An index that
        cannot be written costs a late run. To fail the `create` of a session
        to protect a cache would put the projection above the log.

        **The call does nothing when nothing moved**, so it can run on every
        session open. To reconcile an entry that is already right must not cost
        a lock and a rewrite, and every fork and every subagent opens a session.

        With `new` (`ScheduleService.reindex`), the appointment is claimed first
        (`claim`), and the claim is kept if the write fails.
        """
        claim = self.claim() if new else None
        written = self.record_all({session_id: (next_at, now)})
        if claim is not None:
            claim.close(written=written)

    def record_all(self, changes: Mapping[str, tuple[int | None, int]]) -> bool:
        """`record` for several sessions at once: one read, and at most one rewrite.

        `changes` maps a session to its `(next_at, now)`. `IndexWriter` hands
        over one such mapping in one thread hop. A fan-out that opens N sessions
        therefore costs one read of the file and not N, and a rewrite only if
        something moved.

        Returns whether the file now holds `changes`.
        """

        def apply(found: dict[str, Appointment]) -> None:
            for session_id, (next_at, now) in changes.items():
                if next_at is None:
                    found.pop(session_id, None)
                else:
                    current = found.get(session_id)
                    if current is None or current.next_at != next_at:
                        found[session_id] = Appointment(session_id, next_at, now)

        return self._change(lambda found: _moves(found, changes), apply)

    def replace(
        self, read: Mapping[str, Appointment | None], *, since: Survey, vouch: bool = True
    ) -> bool:
        """Write the answer of a rebuild, mark the file complete, and clear the
        claims that the rebuild answered. Returns whether it was written.

        `vouch` says whether the rebuild read the store in which the deployment
        keeps its logs. A rebuild that read somewhere else, such as a directory
        that was guessed with no store at hand, still writes what it found. It
        leaves the file as complete as it was and leaves the claims where they
        are, so a rebuild that can vouch still comes.

        `read` holds each session whose log the rebuild read: its appointment,
        or `None` for none. **The rebuild keeps what it did not read.** A log
        that it could not read or never found is no evidence that the
        appointment is gone. An extra entry costs one wake, and the reconcile
        on open then removes the entry. **The rebuild keeps what moved while it
        read, as moved.** `since` is the file as the rebuild began. An entry
        that differs from `since` was written after that by a writer that knew
        better.
        """
        if not self._merge(read, before=since.appointments, vouch=vouch):
            return False
        if not vouch:
            return True
        for path in since.abandoned:
            with suppress(FileNotFoundError):
                path.unlink()
        return True


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

    def record(
        self, session_id: str, *, next_at: int | None, now: int, new: bool = False
    ) -> None: ...


@dataclass(slots=True)
class IndexWriter:
    """`ScheduleIndex.record` without the wait on the event loop (K10).

    Every caller is synchronous and on the event loop: `create`, `cancel`, the
    `reindex` of `fire_due`, and the reconcile on `session/created`.
    `ScheduleIndex.record` is a file lock, a read and an atomic rewrite. So
    `record` here only notes the change, and a writer hands everything noted so
    far to `record_all` in one hop to a worker thread.

    **The latest change wins, for each session.** The index holds one entry for
    each session, so only the newest change for each matters. A `create` and
    then a `cancel` that were noted before the writer ran are one write of the
    cancel, and not two writes in order. A change that is noted *while* a batch
    is being written waits for the next batch, so it still lands last.

    **The writer starts on demand and ends when nothing is noted.** It is not a
    task that lives as long as the row. `Context.drain` awaits detached work, so
    a writer that never ended would hold every drain to its deadline. A writer
    that ends makes `drain` the flush, which is what a host that shuts down
    wants. The dispose of the row is the flush behind it. A batch that is still
    running when the share of the drain ran out, or that was noted after it, is
    waited for there and not left to the process.

    **A claim travels with the batch that it covers** (S18). It is taken with
    the first new appointment that is noted, and the write of that batch closes
    it.
    """

    index: ScheduleIndex
    ctx: Context
    _pending: dict[str, tuple[int | None, int]] = field(default_factory=dict)
    _writing: bool = False
    _claim: Claim | None = None
    """The claim that covers the new appointments in `_pending`, if `_pending`
    holds one."""

    def record(self, session_id: str, *, next_at: int | None, now: int, new: bool = False) -> None:
        """Note one change, and start the writer if none is running."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # There is no loop, so no loop can stall. `detach` would also close
            # the writer before it started. That would leave `_writing` set, and
            # every later change noted for a writer that never ran.
            self.index.record(session_id, next_at=next_at, now=now, new=new)
            return
        if new and self._claim is None:
            self._claim = self.index.claim()
        self._pending[session_id] = (next_at, now)
        if not self._writing:
            self._writing = True
            # `finish`: a change that is noted here is a write already promised.
            # The dispose of the row therefore waits for it, where a cancel
            # would drop the batch.
            self.ctx.detach(self._drain(), label="schedule index writer", at_dispose="finish")

    async def _drain(self) -> None:
        # There is no await between the emptiness check and the clearing of the
        # flag. A change that is noted during the last batch is therefore either
        # taken by the `while`, or it finds the flag clear and starts a writer
        # of its own.
        try:
            while self._pending:
                batch, self._pending = self._pending, {}
                claim, self._claim = self._claim, None
                written = False
                try:
                    written = await anyio.to_thread.run_sync(self.index.record_all, batch)
                finally:
                    # A batch that was cut off in mid-write possibly did not
                    # land. An extra mark costs a rebuild, and a missing mark
                    # costs an appointment.
                    if claim is not None:
                        claim.close(written=written)
        finally:
            self._writing = False
