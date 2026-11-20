"""`session-persistence-turso`: the second backend behind `SessionPersistence` (P5-08).

Turso is reached through `pyturso`. It is SQLite-compatible and DB-API 2.0. It
has a native asyncio surface and an optional cross-process WAL, which the
stdlib `sqlite3` does not offer.

**One database for each session**, and not one database that holds many. A
session is the unit that pH creates, resumes, leases and eventually discards,
so the storage should be addressable by that unit. To delete a session is to
delete a file, as with JSONL. It also lets `locate()` answer with a real path,
which is what a one-shot run reports as the log of its session. The I-5 lease
of P5-03 once locked that path, and a `None` there silently disabled the lease.
The lease is now keyed by the session id (`lease.claim_session`).

**`seq INTEGER PRIMARY KEY` is already the ordering.** In SQLite an integer
primary key *is* the rowid. So the table is clustered by it, and a scan comes
back in key order: the log is read back in the order that it was written. The
key also makes A1 structural. A1 says `seq == len(log)`, and the key cannot
hold two events at one number.

**There is no full-text search, deliberately.** JSONL cannot search either, so
parity never required it. Search over sessions is built on top of a backend.
It is not a hidden feature of one backend.

**The write path is deliberately the same as the JSONL one.** A `flush` reads
the events off the log past a cursor, and writes them off the event-loop
thread. The reason is that A1 is about `append` being free of I/O, and that is
a property of the *seam* and not of the storage.

**JSONL stays the default.** `pyturso` is pre-1.0, it is classified alpha, and
it ships no Windows wheels. The ordering of D5 ("JSONL first") is necessary and
not incidental. This is a row that a profile opts into.

@module ph.persistence.turso
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import closing, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from ..cordis import Context, plugin
from ..json import dumps
from ..keys import SESSIONS
from ..paths import make_directories, resolve_roots, sync_directory
from ..seams.diagnostics import Diagnostic, contribute
from ..session import Session, SessionEvent, SessionHeader
from ..wire import WireModel
from .families import children_under, locate_under, logs_under, path_under, stored_under
from .lease import claim_session
from .lineage import materialize
from .protocol import (
    NoStoredSession,
    SessionPersistence,
    StoredSession,
    attach,
    descendants_among,
    gated,
    stored_row,
    write_on_unwind,
)

if TYPE_CHECKING:
    from turso import Connection

__all__ = ["TursoSessionStore", "apply"]

log = logging.getLogger("ph.persistence.turso")

SUFFIX = ".db"

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS header (id TEXT PRIMARY KEY, wire TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, wire TEXT NOT NULL)",
)
"""Two tables for one session. `seq INTEGER PRIMARY KEY` clusters the log by its
own sequence number. So reading the log back in order needs no sort, and two
events cannot be held at one number."""

DURABILITY = ("PRAGMA journal_mode=WAL", "PRAGMA synchronous=FULL")
"""What a commit promises, stated here and not inherited (S19). The write-ahead
log is synced on every commit. So a flush that returned is on disk, as a JSONL
flush is after its `fsync`. These are the defaults of the driver today. They
are set here so that a changed default could not weaken a flush with nothing
saying so."""


def session_db(root: Path, session_id: str, family: str) -> Path:
    """Where a session's database is written: `<root>/<family>/<id><SUFFIX>`."""
    return path_under(root, family, session_id, SUFFIX)


def session_dbs(root: Path) -> list[tuple[Path, os.stat_result]]:
    """Every stored database, newest first."""
    return logs_under(root, SUFFIX)


def locate_db(root: Path, session_id: str) -> Path | None:
    """The database for one id, wherever it is. JSONL uses the same rule."""
    return locate_under(root, session_id, SUFFIX)


@dataclass(slots=True)
class _Progress:
    """How far this store has written the log of one session. It is not a copy of
    the log."""

    cursor: int = 0
    """The first seq that this database does not yet hold. The log is the queue.
    `_Progress.cursor` in `jsonl.py` gives the reason, which is the same for
    both backends. A list that a listener fills misses whatever is appended
    after the listener unwinds, and teardown is when that happens."""
    header_written: bool = False
    writing: anyio.Lock = field(default_factory=anyio.Lock)
    """One flush of this session at a time. A second flush waits, and then writes
    only what arrived since. It does not write a second transaction of the same
    rows."""
    family: str = ""
    """The directory that holds the database of this session.

    It is here and not in a dict of its own, because `forget` already clears
    these records. A parallel map keyed by session id was cleared by nothing,
    and it grew for the life of the process. JSONL keeps the same fact on
    `_Progress.path`.
    """
    connection: Connection | None = None
    """The handle of the writer. The first flush opens it. It is used only under
    `writing`, and it is closed when the session is forgotten. Nothing else uses
    it: a read opens a handle of its own (`_reading`, S19(d))."""

    def close(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:  # pragma: no cover - a closed handle is fine
                log.debug("ph.persistence.turso: a writer's handle did not close cleanly")


@dataclass(slots=True)
class TursoSessionStore:
    """The service published as `ctx.session_persistence`."""

    ctx: Context
    root: Path
    _progress: dict[str, _Progress] = field(default_factory=dict)

    # ------------------------------------------------------------- writing --

    def track(self, session: Session) -> None:
        """Start persisting a session. What it holds past `durable_length` is owed
        a write.

        This has exactly the shape of the JSONL `track`, and it must have: an
        early return, then a cursor that is set one time. The starting point is
        `durable_length`, which is a boundary that the *caller* declares. It is
        never an index into the log that is computed from what this store
        happens to hold.

        Nothing here opens the database. This is a synchronous `session/created`
        listener, so a query would run **on the event-loop thread**. And
        `INSERT OR REPLACE` means that nothing needs to ask whether a header is
        owed.
        """
        if session.id in self._progress:
            return
        # The family is on the header in hand. Remembering it here keeps the
        # path of the writer a function of what this store already knows, so
        # that a write needs no search.
        #
        # The cursor starts from `durable_length`, and not from zero. What is
        # below that line is already durable *somewhere*: in this database for
        # a resume, in the database of the parent for a reference fork. Writing
        # it anyway is not a harmless rewrite. It writes the child a full copy
        # of the prefix. The first event of that copy is at seq 0, and
        # `materialize` reads that as "this file is complete". Reference
        # forking then becomes a silent no-op on this backend, and nothing
        # anywhere fails. This line once queued `session.events`, while the
        # docstring above claimed that it had exactly the shape of JSONL.
        self._progress[session.id] = _Progress(
            cursor=session.durable_length, family=session.header.family
        )

    async def flush(self, session: Session) -> None:
        """Write what the log holds past the cursor.

        What is owed is read off the log at flush time. So a write that does
        not happen owes exactly what it owed before. `JsonlSessionStore.flush`
        explains that pairing, and it is reached the same way here:
        `anyio.to_thread.run_sync` checkpoints before it queues the work. A
        cancellation that arrives with passivation or teardown therefore raises
        before anything is written, and the cursor has not moved.

        **Being sure costs less here than in the JSONL backend.** `_write`
        compares what it owes with what the database holds at those seqs. It
        skips a row that would be written again unchanged, so nothing measures
        the database first. The lock is for cost and not for correctness:
        without it, two overlapping flushes each commit the same batch.
        """
        buffer = self._progress.get(session.id)
        if buffer is None:
            log.warning("ph.persistence.turso: session %s was untracked; tracking now", session.id)
            self.track(session)
            buffer = self._progress[session.id]
        async with buffer.writing:
            try:
                owed = session.events_from(buffer.cursor)
                if buffer.header_written and not owed:
                    return
                await anyio.to_thread.run_sync(self._write, session, owed, buffer)
                buffer.header_written = True
                if owed:
                    buffer.cursor = owed[-1].seq + 1
            finally:
                # The session was forgotten while this flush held the handle.
                # `forget` left the handle to this flush, and did not close it
                # under a write in flight.
                if self._progress.get(session.id) is not buffer:
                    buffer.close()

    def _write(self, session: Session, events: Sequence[SessionEvent], buffer: _Progress) -> None:
        """One flush in one transaction: all of it is committed, or none of it.

        **A write that fails is rolled back** (S19). When a failed write was
        left open, its rows stayed on the cached handle. Every read that
        borrowed the handle saw them as if they were written, and the commit of
        the next flush made them durable after all.

        **A seq is written one time.** A row that is owed again unchanged is
        skipped. That happens because a second store for a live session is told
        the boundary that was true when the session was built. A row that is
        owed with different content is refused, as a JSONL log with two events
        at one seq is refused on read. `INSERT OR REPLACE` used to put the new
        row over the old row, and nothing reported it.
        """
        if buffer.connection is None:
            buffer.connection = _open(session_db(self.root, session.id, buffer.family))
        connection = buffer.connection
        cursor = connection.cursor()
        try:
            # The header row is written unconditionally. Writing it on every
            # flush costs one statement. The flag that used to skip it was
            # answered by opening the database on the event loop.
            cursor.execute(
                "INSERT OR REPLACE INTO header VALUES (?, ?)",
                (session.id, dumps(session.header.to_wire())),
            )
            # **The whole envelope is stored.** Storing only `data` dropped
            # `surfaceOp`, `ignorable` and `sourceEventSeqs`. A session that
            # held any real `user/message` then came back unmarked, and
            # `Session(seed=…)` refused it. That is every resumable session.
            # `executemany` is used because the per-row loop took twice as long
            # as the batched write when measured.
            rows = [(event.seq, dumps(event.to_wire(thaw=False))) for event in events]
            if rows:
                cursor.execute("SELECT seq, wire FROM events WHERE seq >= ?", (rows[0][0],))
                held = dict(cursor.fetchall())
                for seq, wire in rows:
                    if seq in held and held[seq] != wire:
                        raise ValueError(
                            f"session {session.id}: the database holds a different "
                            f"event at seq {seq}; not written over"
                        )
                cursor.executemany(
                    "INSERT INTO events VALUES (?, ?)",
                    [(seq, wire) for seq, wire in rows if seq not in held],
                )
            connection.commit()
        except BaseException:
            with suppress(Exception):
                connection.rollback()
            raise

    def forget(self, session_id: str) -> None:
        """Drop the buffer of one session and close the handle of its writer.

        The exception is a handle that a flush holds. That flush closes the
        handle when it is done (`flush`). A handle that was closed from the
        loop went out from under a transaction on a worker thread (S19(d)).
        """
        buffer = self._progress.pop(session_id, None)
        if buffer is not None and not buffer.writing.locked():
            buffer.close()

    # ------------------------------------------------------------- reading --

    def exists(self, session_id: str, *, family: str | None = None) -> bool:
        return stored_under(self.root, session_id, SUFFIX, family)

    def read(
        self, session_id: str, *, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """The full log of the session, following its lineage when it stores a reference.

        This is the same walk that JSONL uses, over a different read of one
        database. That is the purpose of `materialize` taking a callable: the
        two backends disagree about everything below this line and about
        nothing above it.
        """
        # There is no release loop here any more. `read_own` closes what it
        # opened, and that is where the rule belongs: this walk is not the only
        # caller that reads a database that it does not own. See `read_own`.
        return materialize(self.read_own, session_id, family=family)

    def read_own(
        self,
        session_id: str,
        upto: int | None = None,
        family: str | None = None,
        *,
        types: frozenset[str] | None = None,
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This database and nothing else, up to `upto` if one is given.

        It reads through a connection of its own (`_reading`), which is closed
        when the call returns.
        """
        path = self._found(session_id, family)
        if path is None:
            raise NoStoredSession(f"no stored session {session_id!r}")
        with _reading(path) as connection:
            cursor = connection.cursor()
            rows = cursor.execute("SELECT wire FROM header WHERE id = ?", (session_id,)).fetchall()
            if not rows:
                raise NoStoredSession(f"session {session_id!r} has no header")
            header = SessionHeader.model_validate(json.loads(rows[0][0]))
            # `seq` is the clustered key, so `ORDER BY seq` returns the log in
            # the order that it was written. An append-only file gives that
            # property for free. For the same reason the bound is a range scan
            # that stops, and not a filter over rows that were already fetched.
            rows = (
                cursor.execute("SELECT wire FROM events ORDER BY seq")
                if upto is None
                else cursor.execute("SELECT wire FROM events WHERE seq < ? ORDER BY seq", (upto,))
            ).fetchall()
            # A flush is one transaction here, so no batch is ever torn. `types`
            # only saves the validate-and-freeze of the records that nobody
            # asked for.
            records = (json.loads(wire) for (wire,) in rows)
            return header, [
                SessionEvent.from_wire(record)
                for record in records
                if types is None or record.get("type") in types
            ]

    def _found(self, session_id: str, family: str | None = None) -> Path | None:
        """The database of this session, from what is known before what is on disk.
        `None` when a search finds it nowhere.

        A family in hand is a path. The family comes from the caller, or from
        the buffer that this store already keeps. Anything else is searched
        for. A read stops at `None` and does not try the fallback of `locate`,
        which is a path that the search just tried.
        """
        if family is None:
            buffer = self._progress.get(session_id)
            family = buffer.family if buffer is not None else None
        if family:
            return session_db(self.root, session_id, family)
        return locate_db(self.root, session_id)

    def directory(self) -> Path | None:
        return self.root

    def locate(self, session_id: str) -> Path | None:
        """One database for each session, so there is always a path to point at.

        That was once what made the lease of P5-03 work here at all. A shared
        database has no path to lock, and a `locate` that returned `None`
        silently disabled I-5. The lease is now keyed by the session id
        (`lease.claim_session`). So this function answers the ordinary
        question: where is the storage of this session. A session that is
        neither tracked nor on disk is a root that is about to be written, and
        the family of such a root is its own id.
        """
        return self._found(session_id) or session_db(self.root, session_id, session_id)

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this database against every other writer for the life of `scope` (I-5).

        The locking of the database itself serializes *statements*. It does
        not stop a second process that appends a second log's worth of `seq`
        to one session, and that is the hazard. So the lease is the same file
        lock that JSONL takes. As there, every live log is written before the
        lease is released (`write_on_unwind`).
        """
        await claim_session(scope, self.root, session_id)
        write_on_unwind(scope, self)

    def holding(
        self, types: frozenset[str], *, gate: str
    ) -> Iterator[tuple[str, list[SessionEvent]]]:
        """One database at a time. A database whose rows never name `gate` is not
        read, as JSONL searches the bytes of a log first. Most databases hold
        nothing that names it."""
        needle = f'"{gate}"'
        for path, _stat in session_dbs(self.root):
            session_id = path.name[: -len(SUFFIX)]
            try:
                with _reading(path) as connection:
                    named = connection.execute(
                        "SELECT 1 FROM events WHERE instr(wire, ?) > 0 LIMIT 1", (needle,)
                    ).fetchone()
                if named is None:
                    yield session_id, []
                    continue
                _header, events = self.read_own(session_id, family=path.parent.name, types=types)
            except Exception:
                log.warning("ph.persistence.turso: could not read %s", path, exc_info=True)
                continue
            yield session_id, gated(events, gate)

    def stored(self, *, limit: int | None = 50) -> list[StoredSession]:
        """What is on record, most recently touched first.

        This is a directory scan with the same shape as the JSONL one. That is
        the purpose of one file for each session: both backends answer the
        listing question in the same way, from the filesystem. Neither answers
        it from a query whose cost grows with total history.
        """
        found = session_dbs(self.root)
        listed: list[StoredSession] = []
        for path, stat in found[:limit]:
            session_id = path.name[: -len(SUFFIX)]
            listed.append(stored_row(session_id, _peek_header(session_id, path), stat.st_mtime))
        return listed

    def descendants_of(self, parent_id: str, family: str) -> tuple[StoredSession, ...]:
        """Every stored session beneath `parent_id`: one `scandir` of its family, then
        one header `SELECT` per database whose name has the parent's prefix.

        **One connection per candidate, not one query.** No table here holds more
        than one header, since each session is its own database. A single query
        would have to `ATTACH` every candidate, which opens each file anyway and
        adds the attach limit on top. So this peeks the way `stored` does, through
        `_reading`. The family and the prefix are what keep the count down: only
        the parent's descendants, and any database named after it, are opened.
        """
        candidates: list[StoredSession] = []
        for path, stat in children_under(self.root, family, parent_id, SUFFIX):
            session_id = path.name[: -len(SUFFIX)]
            candidates.append(stored_row(session_id, _peek_header(session_id, path), stat.st_mtime))
        return descendants_among(parent_id, candidates)


def _open(path: Path) -> Connection:
    """A writer's connection to `path`, making the database if it is new."""
    # The native driver is loaded on the first connection, so importing this
    # backend opens nothing.
    import turso  # noqa: PLC0415

    # Every new name is made durable in its parent, as the first write of a
    # JSONL log makes its own names durable (S20): the family directory in the
    # sessions root, and the database in its family.
    fresh = not path.exists()
    make_directories(path.parent)
    connection = turso.connect(str(path))
    cursor = connection.cursor()
    for statement in (*DURABILITY, *SCHEMA):
        cursor.execute(statement)
    connection.commit()
    if fresh:
        sync_directory(path.parent)
    return connection


def _reading(path: Path) -> closing[Connection]:
    """A connection of its own for one read, closed when the read is done.

    **It is never the connection of the writer** (S19(d)), so it is never shared
    between threads. The handle of the writer runs a flush on whichever worker
    thread it got. A reader that borrowed the handle during a flush saw rows
    that the flush might still roll back. Under WAL, a connection that the
    reader owns sees the last commit.

    **It is closed every time, and that keeps a survey from exhausting
    handles.** A reader once went through the cache of the writer. Every
    database that it peeked stayed open, with its `-wal` and `-shm` sidecars,
    for the life of the process: 500 of them for a listing of 500 sessions.
    Opening one costs tens of microseconds.

    **A reader writes nothing**: no schema, no pragma, and no file for a path
    that is not there. A database whose schema never committed is a first
    write that was cut off between the file and its tables. It reads as
    damage, with the error of the driver itself.
    """
    if not path.is_file():
        raise NoStoredSession(f"no stored session at {path}")
    import turso  # noqa: PLC0415

    return closing(turso.connect(str(path)))


def _peek_header(session_id: str, path: Path) -> SessionHeader | None:
    """The header of one database, or `None` when the database cannot give one.
    The listing still shows a row for it."""
    try:
        with _reading(path) as connection:
            rows = (
                connection.cursor()
                .execute("SELECT wire FROM header WHERE id = ?", (session_id,))
                .fetchall()
            )
        return SessionHeader.model_validate(json.loads(rows[0][0])) if rows else None
    except Exception:
        return None


class Config(WireModel):
    """Row config for `session-persistence-turso`."""

    root: str | None = None
    """The directory that holds the session databases. When this is unset, the
    row uses the sessions directory of the host: `$PH_HOME/sessions`, or the
    `paths.sessions` setting of `daemon.yaml`."""


@plugin(
    "session-persistence-turso",
    affects="deployment",
    inject=[SESSIONS],
    config=Config,
    log_layer="storage",
)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the Turso backend and wire it to the session firehose."""
    root = Path(config.root) if config.root else resolve_roots().sessions_dir()
    # The annotation makes mypy check this backend against the Protocol *with
    # signatures*. The runtime `isinstance` gate cannot do that, because a
    # `runtime_checkable` Protocol compares names only.
    store: SessionPersistence = TursoSessionStore(ctx=ctx, root=root)
    attach(ctx, store)

    contribute(
        ctx,
        Diagnostic(
            id="session-store",
            title="Session store",
            order=60,
            read=lambda: [("backend", "turso"), ("directory", str(root))],
        ),
    )
