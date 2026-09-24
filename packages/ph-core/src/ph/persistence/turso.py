"""`session-persistence-turso` — the second backend behind `SessionPersistence` (P5-08).

Turso through `pyturso`: SQLite-compatible and DB-API 2.0, with a native asyncio
surface and an optional cross-process WAL that stdlib `sqlite3` does not offer.

**One database per session**, not one database holding many. A session is the unit
pH creates, resumes, leases and eventually discards, so it is the unit the storage
should be addressable by: deleting a session is deleting a file, the same as JSONL.
It is also what lets `locate()` answer with a real path — which is what P5-03's
I-5 lease locks and what the session picker lists, and a `None` there silently
disables the lease.

**`seq INTEGER PRIMARY KEY`, which is already the ordering.** In SQLite an integer
primary key *is* the rowid, so the table is clustered by it and a scan comes back
in key order — a log read back in the order it was written. The key is also A1 made
structural: `seq == len(log)` cannot hold two events at one number.

**No full-text search**, deliberately: JSONL cannot search either, so it was never
parity. Search over sessions is a thing built on top of a backend rather than a
thing one backend secretly has.

**The write path is the JSONL one's, deliberately.** Read off the log past a
cursor on `flush`, off the event-loop thread — because A1 is about `append` being
I/O-free, and that is a property of the *seam*, not of the storage.

**JSONL stays the default.** `pyturso` is pre-1.0, classified alpha, and ships no
Windows wheels; D5's ordering — "JSONL first" — is load-bearing rather than
incidental, and this is a row a profile opts into.

@module ph.persistence.turso
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..json import dumps
from ..keys import SESSIONS
from ..paths import resolve_roots
from ..seams.diagnostics import Diagnostic, contribute
from ..session import Session, SessionEvent, SessionHeader
from ..wire import WireModel
from .families import locate_under, logs_under, path_under
from .lease import claim_session
from .lineage import materialize
from .protocol import SessionPersistence, StoredSession, attach, stored_row, write_on_unwind

__all__ = ["TursoSessionStore", "apply"]

log = logging.getLogger("ph.persistence.turso")

SUFFIX = ".db"

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS header (id TEXT PRIMARY KEY, wire TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, wire TEXT NOT NULL)",
)
"""Two tables, one session. `seq INTEGER PRIMARY KEY` clusters the log by its
own sequence number, so reading it back in order needs no sort and holding two
events at one number is impossible."""


def session_db(root: Path, session_id: str, family: str) -> Path:
    """Where a session's database is written: `<root>/<family>/<id><SUFFIX>`."""
    return path_under(root, family, session_id, SUFFIX)


def session_dbs(root: Path) -> list[tuple[Path, os.stat_result]]:
    """Every stored database, newest first."""
    return logs_under(root, SUFFIX)


def locate_db(root: Path, session_id: str) -> Path | None:
    """The database for one id, wherever it sits. JSONL's rule, shared."""
    return locate_under(root, session_id, SUFFIX)


@dataclass(slots=True)
class _Progress:
    """How far this store has written one session's log — not a copy of it."""

    cursor: int = 0
    """The first seq this database does not yet hold. The log is the queue —
    `JsonlSessionStore`'s `_Progress.cursor` gives the reason, which is the same
    for both backends: a list filled by a listener misses whatever is appended
    after the listener unwinds, and teardown is when that happens."""
    header_written: bool = False
    writing: anyio.Lock = field(default_factory=anyio.Lock)
    """One flush of this session at a time, so a second one waits and writes only
    what arrived since rather than a second transaction of the same rows."""
    family: str = ""
    """Which directory this session's database lives in.

    Here rather than in a dict of its own, because `forget` already clears
    these and a parallel map keyed by session id was cleared by nothing — it grew
    for the life of the process, which is the exact leak `_release` exists to
    argue against one method down. JSONL keeps the same fact on `_Progress.path`.
    """


@dataclass(slots=True)
class TursoSessionStore:
    """The service published as `ctx.session_persistence`."""

    ctx: Context
    root: Path
    _progress: dict[str, _Progress] = field(default_factory=dict)
    _connections: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- writing --

    def track(self, session: Session) -> None:
        """Start persisting a session; its existing seed is owed a write.

        JSONL's shape exactly, and it has to be: an early return, then a cursor set
        once. The starting point is `durable_length` — a boundary the *caller* declares —
        never an index into the log computed from what this store happens to hold.

        Nothing here opens the database. This is a synchronous `session/created` listener,
        so a query would run **on the event-loop thread**, and `INSERT OR REPLACE` means
        nothing needs to ask whether a header is owed.
        """
        if session.id in self._progress:
            return
        # The family is on the header in hand; remembering it here is what keeps
        # `_connect` a pure function of what this store already knows, rather
        # than a search on every write.
        #
        # From `durable_length`, not from zero. What is below that line is
        # already durable *somewhere* — in this database for a resume, in the
        # parent's for a reference-fork — and writing it anyway is not a
        # harmless rewrite: it writes the child a full copy of the prefix, whose
        # first event is then at seq 0, which `materialize` reads as "this file
        # is complete". Reference-forking becomes a silent no-op on this backend
        # and nothing anywhere fails. This line once queued `session.events`
        # while the docstring above claimed "JSONL's shape exactly".
        self._progress[session.id] = _Progress(
            cursor=session.durable_length, family=session.header.family
        )

    async def flush(self, session: Session) -> None:
        """Write what the log holds past the cursor.

        Read off the log at flush time, so a write that does not happen owes
        exactly what it owed before — the pairing `JsonlSessionStore.flush`
        explains, reachable the same way: `anyio.to_thread.run_sync` checkpoints
        before it queues the work, so a cancellation arriving with passivation or
        teardown raises before anything is written, and the cursor has not moved.

        **Cheaper to be sure of here than in the JSONL backend**, because
        `_write` is `INSERT OR REPLACE` keyed by `seq`: a row written twice is
        harmless, so nothing measures what the database already holds. The lock
        is for cost, not correctness — without it two overlapping flushes each
        commit the same batch.
        """
        buffer = self._progress.get(session.id)
        if buffer is None:
            log.warning("ph.persistence.turso: session %s was untracked; tracking now", session.id)
            self.track(session)
            buffer = self._progress[session.id]
        async with buffer.writing:
            owed = session.events_from(buffer.cursor)
            if buffer.header_written and not owed:
                return
            await anyio.to_thread.run_sync(self._write, session, owed)
            buffer.header_written = True
            if owed:
                buffer.cursor = owed[-1].seq + 1

    def _write(self, session: Session, events: Sequence[SessionEvent]) -> None:
        cursor = self._connect(session.id).cursor()
        # Unconditionally: `INSERT OR REPLACE` is idempotent, so writing the
        # header row every flush costs one statement — and the flag that skipped
        # it was answered by opening the database on the event loop.
        cursor.execute(
            "INSERT OR REPLACE INTO header VALUES (?, ?)",
            (session.id, dumps(session.header.to_wire())),
        )
        # **The whole envelope.** Storing only `data` dropped `surfaceOp`,
        # `ignorable` and `sourceEventSeqs`, so a session holding any real
        # `user/message` came back unmarked and `Session(seed=…)` refused it —
        # which is every resumable session. `executemany` because the per-row
        # loop measured twice the batched write.
        cursor.executemany(
            "INSERT OR REPLACE INTO events VALUES (?, ?)",
            [(event.seq, dumps(event.to_wire(thaw=False))) for event in events],
        )
        self._connections[session.id].commit()

    def forget(self, session_id: str) -> None:
        self._progress.pop(session_id, None)
        self._release(session_id)

    def _release(self, session_id: str) -> None:
        """Close one cached handle. **Not** the session's buffered work.

        Split out of `forget` because the read paths need exactly this half and
        emphatically not the other: an ancestor opened to satisfy a chained read,
        or a database peeked for a listing, may also be a *live* session with
        unflushed events, and `forget` would drop them on the floor with nothing
        raised. One database per session means one handle per session, and a
        daemon that ran for weeks would otherwise hold every one it ever opened.
        """
        connection = self._connections.pop(session_id, None)
        if connection is not None:
            try:
                connection.close()
            except Exception:  # pragma: no cover - a closed handle is fine
                log.debug("ph.persistence.turso: %s did not close cleanly", session_id)

    # ------------------------------------------------------------- reading --

    def exists(self, session_id: str) -> bool:
        return locate_db(self.root, session_id) is not None

    def read(self, session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
        """The session's full log, following its lineage when it stores a reference.

        The same walk JSONL uses, over a different one-database read — which is
        the point of `materialize` taking a callable: the two backends disagree
        about everything below this line and about nothing above it.
        """
        # No release loop here any more: `read_own` closes what it opened, which
        # is where the rule belongs — this walk is not the only caller that reads
        # a database it does not own. See `read_own`.
        return materialize(self.read_own, session_id)

    def read_own(
        self, session_id: str, upto: int | None = None, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This database and nothing else, up to `upto` if one is given.

        Reads through `_borrow`, so the handle it opens does not outlive the
        call — see that method for the rule and why it is there rather than here.
        """
        # One resolution, not two: the `exists` gate here searched the store and
        # then `_connect` searched it again for the same id, so a chained read
        # paid two full scans per ancestor.
        path = self._path_for(session_id, family)
        if not path.is_file():
            raise FileNotFoundError(f"no stored session {session_id!r}")
        with self._borrow(session_id, path) as connection:
            cursor = connection.cursor()
            rows = cursor.execute("SELECT wire FROM header WHERE id = ?", (session_id,)).fetchall()
            if not rows:
                raise FileNotFoundError(f"session {session_id!r} has no header")
            header = SessionHeader.model_validate(json.loads(rows[0][0]))
            # `ORDER BY seq` is the clustered key, so this is the log in the order
            # it was written — the property an append-only file gives for free.
            # `seq` is the clustered key, so the bound is a range scan that stops —
            # not a filter over rows already fetched.
            rows = (
                cursor.execute("SELECT wire FROM events ORDER BY seq")
                if upto is None
                else cursor.execute("SELECT wire FROM events WHERE seq < ? ORDER BY seq", (upto,))
            ).fetchall()
            return header, [SessionEvent.from_wire(json.loads(wire)) for (wire,) in rows]

    def _path_for(self, session_id: str, family: str | None = None) -> Path:
        """This session's database, by what is known before what is on disk.

        A family in hand — from the caller, or from the buffer this store is
        already keeping — is a path. Anything else is searched for, and a session
        that is neither tracked nor on disk is a root about to be written, whose
        family is its own id.
        """
        if family is None:
            buffer = self._progress.get(session_id)
            family = buffer.family if buffer is not None else None
        if family:
            return session_db(self.root, session_id, family)
        return locate_db(self.root, session_id) or session_db(self.root, session_id, session_id)

    def directory(self) -> Path | None:
        return self.root

    def locate(self, session_id: str) -> Path | None:
        """One database per session, so there is always a path to point at.

        That was once what made P5-03's lease work here at all — a shared
        database has no path to lock, and `locate` returning `None` silently
        disabled I-5. The lease is keyed by the session id now
        (`lease.claim_session`), so this answers the ordinary question instead:
        where does this session's storage live.
        """
        return self._path_for(session_id)

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this database against every other writer for `scope`'s life (I-5).

        The database's own locking serializes *statements*; it does not stop a
        second process appending a second log's worth of `seq` to one session,
        which is the hazard, so the lease is the same file lock JSONL takes —
        and, as there, every live log is written before it is let go
        (`write_on_unwind`).
        """
        await claim_session(scope, self.root, session_id)
        write_on_unwind(scope, self)

    def stored(self, *, limit: int = 50) -> list[StoredSession]:
        """What is on record, most recently touched first.

        A directory scan, the same shape as JSONL's — which is the point of one
        file per session: both backends answer the listing question the same
        way, from the filesystem, rather than one of them from a query whose
        cost grew with total history.
        """
        found = session_dbs(self.root)
        listed: list[StoredSession] = []
        for path, stat in found[:limit]:
            # The path the scan already found, not another search for it: this
            # loop threw it away and made `_peek_header` re-locate every row —
            # 199 full store scans to list 200 sessions.
            session_id = path.name[: -len(SUFFIX)]
            listed.append(
                stored_row(session_id, self._peek_header(session_id, path), stat.st_mtime)
            )
        return listed

    # ----------------------------------------------------------- internals --

    def _peek_header(self, session_id: str, path: Path | None = None) -> SessionHeader | None:
        try:
            with self._borrow(session_id, path) as connection:
                rows = (
                    connection.cursor()
                    .execute("SELECT wire FROM header WHERE id = ?", (session_id,))
                    .fetchall()
                )
            return SessionHeader.model_validate(json.loads(rows[0][0])) if rows else None
        except Exception:
            return None

    @contextmanager
    def _borrow(self, session_id: str, path: Path | None = None) -> Iterator[Any]:
        """A connection for one read, closed again unless a writer owns it.

        **The rule lives here because `_connect` is the only thing that opens a
        handle**, and `_connect` caches while `forget` runs only for sessions this
        store *buffers* — so any database read by something that does not write it
        stays open, with its `-wal` and `-shm` sidecars, for the life of the
        process. That leak was guarded twice and differently: `read` wrapped its
        chained walk, `stored` wrapped its header peeks, and the third reader to
        arrive — a fold over *every* stored session (`phern attachments gc`) at a
        limit of 100 000 — inherited neither. Two hand-rolled guards is how the
        third caller comes to have none, so both were replaced by this.

        The two also **disagreed**, which is the other reason not to keep them: the
        set-diff version released any handle opened during its block, including a
        session that is tracked but not yet connected — a live session's handle,
        which `_release`'s own docstring says this half must never touch.

        A buffered session is therefore exempt: its handle is the writer's, and
        closing it mid-session makes the next flush pay a reconnect and the schema
        DDL again.

        A context manager rather than a line at each return, because the release
        has to survive the raise: `read_own` refuses a database with no header
        between opening and returning, and a post-condition on the success path
        alone leaks exactly the handle a failing read opened.
        """
        try:
            yield self._connect(session_id, path)
        finally:
            if session_id not in self._progress:
                self._release(session_id)

    def _connect(self, session_id: str, path: Path | None = None) -> Any:  # noqa: ANN401
        connection = self._connections.get(session_id)
        if connection is None:
            import turso

            # the family directory is created with the path in `_path_for`
            path = path if path is not None else self._path_for(session_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = turso.connect(str(path))
            cursor = connection.cursor()
            for statement in SCHEMA:
                cursor.execute(statement)
            connection.commit()
            self._connections[session_id] = connection
        return connection


class Config(WireModel):
    """Row config for `session-persistence-turso`."""

    root: str | None = None
    """Where the logs live; `$PH_HOME/sessions` when unset."""


@plugin("session-persistence-turso", inject=[SESSIONS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the Turso backend and wire it to the session firehose."""
    root = Path(config.root) if config.root else resolve_roots().sessions_dir()
    # Annotated, so mypy checks this backend against the Protocol *with
    # signatures* — which the runtime `isinstance` gate cannot, a
    # `runtime_checkable` Protocol comparing names only.
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
