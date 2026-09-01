"""`session-persistence-turso` — the second backend behind `SessionPersistence` (P5-08).

Turso through `pyturso`: SQLite-compatible and DB-API 2.0, so this reads like
the `sqlite3` backend D5 originally named, with a native asyncio surface and an
optional cross-process WAL that stdlib `sqlite3` does not offer.

**One database per session**, not one database holding many. A session is the
unit pH creates, resumes, leases and eventually discards, so it is the unit the
storage should be addressable by: deleting a session is deleting a file, the
same as JSONL, rather than a `DELETE` that leaves pages behind. It is also what
lets `locate()` answer with a real path — which is what P5-03's I-5 lease locks
and what the session picker lists. A single shared database made `locate` return
`None`, and a `None` there silently disables the lease.

**`seq INTEGER PRIMARY KEY`, which is already the ordering.** In SQLite an
integer primary key *is* the rowid, so the table is clustered by it and a scan
comes back in key order — a log read back in the order it was written, which is
the one property an append-only file gives for free. (`WITHOUT ROWID` would say
the same thing explicitly and is gated behind an experimental flag on this
build; it earns its keep for *composite* keys, which one database per session
removes the need for.) The key is also A1 made structural: `seq == len(log)`
cannot hold two events at one number.

**No full-text search.** It was here and is deliberately gone: JSONL cannot
search either, so it was never parity — it was a feature riding along, and it
cost an experimental flag, an index whose incremental maintenance made writes
**quadratic** (a ten-event flush going from 1.0 ms to 227 ms once a session
passed six hundred events), and a ranking that does not work through a bound
parameter anyway. Search over sessions, if it is wanted, is a thing built on top
of a backend rather than a thing one backend secretly has.

**The write path is the JSONL one's, deliberately.** Buffered on `record`,
drained on `flush`, off the event-loop thread — because A1 is about `append`
being I/O-free, and that is a property of the *seam*, not of the storage.

**JSONL stays the default.** `pyturso` is pre-1.0, classified alpha, and ships
no Windows wheels; D5's ordering — "JSONL first" — is load-bearing rather than
incidental, and this is a row a profile opts into.

@module ph.persistence.turso
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..paths import resolve_roots
from ..seams.diagnostics import Diagnostic, contribute
from ..session import Session, SessionEvent, SessionHeader
from ..session.json import dumps
from .families import locate_under, logs_under, path_under
from .lineage import materialise
from .protocol import SessionPersistence, StoredSession, attach, stored_row

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
class _Buffer:
    pending: list[SessionEvent] = field(default_factory=list)
    header_written: bool = False
    family: str = ""
    """Which directory this session's database lives in.

    On the buffer rather than in a dict of its own, because `forget` already
    clears buffers and a parallel map keyed by session id was cleared by nothing
    — it grew for the life of the process, which is the exact leak `_release`
    exists to argue against one method down. JSONL keeps the same fact on
    `_Buffer.path`.
    """


@dataclass(slots=True)
class TursoSessionStore:
    """The service published as `ctx.session_persistence`."""

    ctx: Context
    root: Path
    _buffers: dict[str, _Buffer] = field(default_factory=dict)
    _connections: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- writing --

    def track(self, session: Session) -> None:
        """Start buffering a session; its existing seed is owed a write.

        JSONL's shape exactly, and it has to be: an early return, then the seed
        queued once. The first version re-entered on every call and seeded from
        an index into the *log* computed from the length of the *buffer*, which
        stop agreeing after the first flush — so a resumed session re-queued its
        whole log and rewrote it. It also opened the database here to ask
        whether a header was owed, **on the event-loop thread**, because this is
        a synchronous `session/created` listener; JSONL's equivalent is a
        `stat`, and `INSERT OR REPLACE` means nothing needed the answer.
        """
        if session.id in self._buffers:
            return
        # The family is on the header in hand; remembering it here is what keeps
        # `_connect` a pure function of what this store already knows, rather
        # than a search on every write.
        buffer = _Buffer(family=session.header.family)
        # `events[durable_length:]`, not the whole log. What is below that line
        # is already durable *somewhere* — in this database for a resume, in the
        # parent's for a reference-fork — and queueing it anyway is not a
        # harmless rewrite: it writes the child a full copy of the prefix, whose
        # first event is then at seq 0, which `materialise` reads as "this file
        # is complete". Reference-forking becomes a silent no-op on this backend
        # and nothing anywhere fails. This line said `session.events` while the
        # docstring above claimed "JSONL's shape exactly".
        buffer.pending.extend(session.events[session.durable_length :])
        self._buffers[session.id] = buffer

    def record(self, session: Session, event: SessionEvent) -> None:
        buffer = self._buffers.get(session.id)
        if buffer is None:
            log.warning("ph.persistence.turso: session %s was untracked; tracking now", session.id)
            self.track(session)
            return
        buffer.pending.append(event)

    async def flush(self, session: Session) -> None:
        buffer = self._buffers.get(session.id)
        if buffer is None or (buffer.header_written and not buffer.pending):
            return
        pending, buffer.pending = buffer.pending, []
        buffer.header_written = True
        await anyio.to_thread.run_sync(self._write, session, pending)

    def _write(self, session: Session, events: list[SessionEvent]) -> None:
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
        self._buffers.pop(session_id, None)
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
        the point of `materialise` taking a callable: the two backends disagree
        about everything below this line and about nothing above it.
        """
        held = set(self._connections)
        try:
            return materialise(self.read_own, session_id)
        finally:
            # `_connect` caches, and `forget` only ever runs for sessions this
            # store *buffers* — so every ancestor the walk touches would leave an
            # open database behind for the life of the process. That is precisely
            # the exhaustion `forget`'s own comment guards against, reintroduced
            # through a read. Ancestors are immutable and read once; nothing is
            # gained by keeping them.
            for ancestor in set(self._connections) - held:
                self._release(ancestor)

    def read_own(
        self, session_id: str, upto: int | None = None, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This database and nothing else, up to `upto` if one is given."""
        # One resolution, not two: the `exists` gate here searched the store and
        # then `_connect` searched it again for the same id, so a chained read
        # paid two full scans per ancestor.
        path = self._path_for(session_id, family)
        if not path.is_file():
            raise FileNotFoundError(f"no stored session {session_id!r}")
        cursor = self._connect(session_id, path).cursor()
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
        events = [SessionEvent.from_wire(json.loads(wire)) for (wire,) in rows]
        return header, events

    def _path_for(self, session_id: str, family: str | None = None) -> Path:
        """This session's database, by what is known before what is on disk.

        A family in hand — from the caller, or from the buffer this store is
        already keeping — is a path. Anything else is searched for, and a session
        that is neither tracked nor on disk is a root about to be written, whose
        family is its own id.
        """
        if family is None:
            buffer = self._buffers.get(session_id)
            family = buffer.family if buffer is not None else None
        if family:
            return session_db(self.root, session_id, family)
        return locate_db(self.root, session_id) or session_db(self.root, session_id, session_id)

    def locate(self, session_id: str) -> Path | None:
        """One database per session, so there is always a path to point at.

        Which is what makes P5-03's lease work here at all: a shared database
        had none, and `locate` returning `None` silently disabled I-5.
        """
        return self._path_for(session_id)

    def stored(self, *, limit: int = 50) -> list[StoredSession]:
        """What is on record, most recently touched first.

        A directory scan, the same shape as JSONL's — which is the point of one
        file per session: both backends answer the listing question the same
        way, from the filesystem, rather than one of them from a query whose
        cost grew with total history.
        """
        # Every `_peek_header` below opens a database, runs the schema DDL and
        # caches the handle — 1.53 ms each, against 0.021 ms for the header
        # `SELECT` it was opened for, and never reused. Left cached, a 500-session
        # survey holds 500 open databases and 500 `-wal`/`-shm` sidecars for the
        # life of the process. `read` already guards its chained reads this way;
        # a listing peeks far more.
        held = set(self._connections)
        try:
            return self._stored(limit=limit)
        finally:
            for peeked in set(self._connections) - held:
                self._release(peeked)

    def _stored(self, *, limit: int) -> list[StoredSession]:
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
            cursor = self._connect(session_id, path).cursor()
            rows = cursor.execute("SELECT wire FROM header WHERE id = ?", (session_id,)).fetchall()
            return SessionHeader.model_validate(json.loads(rows[0][0])) if rows else None
        except Exception:
            return None

    def _connect(self, session_id: str, path: Path | None = None) -> Any:
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


@plugin("session-persistence-turso", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the Turso backend and wire it to the session firehose."""
    setting = config.get("root") if isinstance(config, dict) else None
    root = Path(setting) if setting else resolve_roots().sessions_dir()
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
