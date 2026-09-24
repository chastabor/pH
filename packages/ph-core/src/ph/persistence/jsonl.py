"""`session-persistence-jsonl` — the log on disk, one JSON object per line.

The append hot path must never block on I/O (A1), so this provider writes only on
`session/flush`, and what it writes is read off the log itself — everything past
a cursor of what the file already holds. `session/flush` is a `parallel` event, so
a caller awaiting it has awaited every backend, not just the first one to answer.

The file format is deliberately dsh's: a header line, then one event per line,
camelCase throughout (Q2). A pH session is therefore a session dsh tooling
reads, and `ph session import` in the other direction needs no second parser.

Writes are whole lines, `fsync`ed at each barrier, and **a write is all or
nothing**: one that fails part-way takes back the bytes it managed before the
failure is reported (`_append_and_sync`), so the retry `flush` owes appends to a
clean end. What that cannot cover is a process that dies mid-write, which leaves
a torn final line. That is the one damage an append-only log can repair without
guessing, because nothing ever reported those bytes written: the reader drops
them and the writer's first flush trims them (`_settle_tail`). A malformed line
anywhere *before* the last is still refused. Encoding happens in the worker
thread beside the I/O: the checkpoint policy awaits a flush before every model
request, so nothing about a flush should hold the event loop.

@module ph.persistence.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

import anyio
from pydantic import ValidationError

from ..cordis import DEPLOYMENT, Context, plugin
from ..json import as_str, dumps
from ..keys import SESSION_PERSISTENCE, SESSIONS, TOOLS
from ..paths import resolve_roots
from ..session import BatchRef, Session, SessionEvent, SessionHeader
from ..session.writers import log_writer
from ..wire import WireModel
from .families import locate_under, logs_under, path_under
from .lease import claim_session
from .lineage import materialize
from .protocol import SessionPersistence, StoredSession, attach, stored_row, write_on_unwind
from .repair import CallOutcome, interrupted_turn_closers, unresolved_calls

_LOG = log_writer(__name__)

__all__ = [
    "JsonlSessionStore",
    "apply",
    "family_log",
    "locate_session",
    "read_records",
    "read_session",
    "resumption_of",
    "session_logs",
    "session_path",
]


def resumption_of(session: Session) -> dict[str, Any] | None:
    """What this session's last resume recorded, or `None` if it never was.

    Read from the log rather than returned from `resume_session`, so a front end
    that did not perform the resume — a TUI attaching to a daemon root, a
    trajectory reader opening a file — learns it the same way as the process
    that did.
    """
    event = session.latest("session/resumed")
    return dict(event.data) if event is not None else None


log = logging.getLogger("ph.persistence.jsonl")

HEADER_LINE_TYPE = "session/header"

SUFFIX = ".jsonl"
"""One statement of the extension, so a rename is one edit."""


def family_log(family_dir: Path, session_id: str) -> Path:
    """A log inside one family directory. Members are flat within a family."""
    return family_dir / f"{session_id}{SUFFIX}"


def session_path(root: Path, session_id: str, family: str) -> Path:
    """Where a session's log is written: `<root>/<family>/<id>.jsonl`."""
    return path_under(root, family, session_id, SUFFIX)


def session_logs(root: Path, *, tag: str = "") -> list[tuple[Path, os.stat_result]]:
    """Every stored log under `root`, newest first; `tag` narrows it to one cwd."""
    return logs_under(root, SUFFIX, tag=tag)


def locate_session(root: Path, session_id: str) -> Path | None:
    """The log for one id, wherever it sits. `None` if there is none.

    A caller holding the family should build the path instead — see
    `locate_under` for what this costs.
    """
    return locate_under(root, session_id, SUFFIX)


@dataclass(slots=True)
class _Progress:
    """How far this store has written one session's log — not a copy of it."""

    path: Path
    cursor: int = 0
    """The first seq this file does not yet hold — **the log is the queue** (F2).

    What is owed is `session.events_from(cursor)`, read at flush time, rather
    than a second list filled by the `session/event` listener. That list was only
    as complete as the listener's life: teardown is exactly when events are
    appended (`workspace/disposed`, tombstones, a canceled turn's closers) and
    exactly when this row's listeners have already unwound, so every one of them
    was appended to the session and never reached the file. A cursor over the
    log cannot miss an event, because it never needed to be told about one."""
    header_written: bool = False
    writing: anyio.Lock = field(default_factory=anyio.Lock)
    """One flush of this log at a time. See `JsonlSessionStore.flush`."""
    measured: bool = False
    """Whether the cursor and `header_written` have been checked against the file (B7).

    Resolved on the first flush rather than at `track`, because the answer costs
    reads of the file and `track` is a synchronous `session/created` listener —
    the thing `TursoSessionStore.track` refuses in its own docstring. A session
    that is tracked and never flushed never pays it. Cleared after a failed write,
    which may have left a fragment only a fresh measurement settles."""


_TAIL_CHUNK = 64 * 1024
"""How far one backwards read reaches when looking for a line boundary."""


@dataclass(frozen=True, slots=True)
class _Tail:
    """How a log ends: where its last finished line stops, and what is after it."""

    complete: int
    """Bytes through the last newline — every line a write finished."""
    torn: bytes
    """What follows that newline. Empty in a log every write finished; otherwise
    an unterminated final line, which only a write that did not finish leaves."""
    last_seq: int | None
    """The seq of the last finished record, or `None` when there is none."""
    open_batch: BatchRef | None = None
    """The batch that last record belongs to, when it is not the batch's last
    member — what a torn write can leave, and `_settle_batch` takes back."""


def _open_batch_of(record: dict[str, Any] | None) -> BatchRef | None:
    """The batch `record` is a member of and does not finish, if any."""
    raw = record.get("batch") if record is not None else None
    seq = _seq_of(record)
    if raw is None or seq is None:
        return None
    try:
        ref = BatchRef.model_validate(raw)
    except ValidationError:
        return None
    return ref if seq < ref.last else None


def _read_tail(path: Path) -> _Tail | None:
    """How this log ends, or `None` when there is no file to ask (B7).

    **The record's own seq, not a line count.** Counting lines and adding an
    offset needs to know which index the file *starts* at, and the two numbers
    available — `durable_length` and `header.seed_length` — disagree for a
    reference fork, so the arithmetic silently dropped a fork's first events.
    A seq is absolute: `Session._append` assigns `seq == len(log)` (A1) and a
    seed preserves it, so `events[i].seq == i` for every log, forked or not.

    Read from the end rather than by scanning: only the last line is wanted, so
    this is a few seeks however long the log is. **Backwards in chunks until a
    boundary is found**, not one fixed window: a single 64 KiB read split a
    final record longer than the window, parsed the fragment, and answered
    `None` — so a large tool result at the end of a log made a re-activated
    store re-queue everything the file already held.
    """
    try:
        handle = path.open("rb")
    except OSError:
        # No file yet — a fresh session, or a fork that has not written — or one
        # this process cannot read. The declared boundary then answers alone.
        return None
    with handle:
        size = handle.seek(0, os.SEEK_END)
        complete = _line_start(handle, size)
        handle.seek(complete)
        torn = handle.read(size - complete)
        end = complete
        while end > 0:
            start = _line_start(handle, end - 1)
            handle.seek(start)
            line = handle.read(end - start)
            if line.strip():
                record = _record(line)
                return _Tail(
                    complete=complete,
                    torn=torn,
                    last_seq=_seq_of(record),
                    open_batch=_open_batch_of(record),
                )
            end = start
    return _Tail(complete=complete, torn=torn, last_seq=None)


def _line_start(handle: BinaryIO, end: int) -> int:
    """The offset just past the last newline before `end`, or 0 if there is none."""
    position = end
    while position > 0:
        start = max(0, position - _TAIL_CHUNK)
        handle.seek(start)
        found = handle.read(position - start).rfind(b"\n")
        if found >= 0:
            return start + found + 1
        position = start
    return 0


def _record(line: bytes) -> dict[str, Any] | None:
    """The JSON object a line holds, or `None` when it holds none.

    Also the test for an unterminated final line: a record is a JSON **object**,
    and no proper prefix of an object's encoding parses — the closing brace is its
    last byte — so a fragment that parses as one is the entire record, cut after
    it and before the `\\n` that follows.
    """
    try:
        record = json.loads(line)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _seq_of(record: dict[str, Any] | None) -> int | None:
    seq = record.get("seq") if record is not None else None
    return seq if isinstance(seq, int) else None


def _settle_tail(path: Path) -> _Tail | None:
    """Measure how this log ends, and finish or take back an unfinished write (F6).

    Only a process that died mid-write leaves anything after the last newline —
    a write that *failed* takes its own bytes back (`_append_and_sync`) — and the
    next append would be glued onto it, turning one lost tail into a line nobody
    can parse in the middle of the log. So before this store appends anything:

    * a fragment that is a **whole record** gets the newline it lost, and counts
      as written, which is how `read_session` reads the same bytes;
    * anything else is **removed**. Nothing reported it written — `flush` had
      not returned — so no reader was told it exists, and `read_session` drops
      it for the same reason.

    The two rules are the reader's rules, applied by the one party allowed to
    change the file: the lease (I-5) makes this store its only writer.
    """
    tail = _read_tail(path)
    if tail is None:
        return tail
    if not tail.torn:
        return _settle_batch(path, tail)
    with path.open("r+b") as handle:
        fragment = _record(tail.torn)
        if fragment is not None:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            seq = _seq_of(fragment)
            settled = _Tail(
                complete=tail.complete + len(tail.torn) + 1,
                torn=b"",
                last_seq=tail.last_seq if seq is None else seq,
                open_batch=tail.open_batch if seq is None else _open_batch_of(fragment),
            )
        else:
            handle.truncate(tail.complete)
            settled = replace(tail, torn=b"")
            log.warning(
                "ph.persistence.jsonl: %s ended in %d byte(s) of a write that did not "
                "finish; removed them before appending",
                path,
                len(tail.torn),
            )
        handle.flush()
        os.fsync(handle.fileno())
    return _settle_batch(path, settled)


def _settle_batch(path: Path, tail: _Tail) -> _Tail:
    """Take back a batch a torn write cut short, as `read_session` drops it (P10-15).

    After the torn line is settled, so the last record is a whole one. If it
    belongs to a batch it does not finish, walk back line by line to the member
    whose seq is the batch's `first` and cut the file there — the reader keeps
    none of an unfinished batch, and the writer's first append must not land
    behind half of one.
    """
    batch = tail.open_batch
    if batch is None:
        return tail
    with path.open("r+b") as handle:
        end = tail.complete
        while True:
            if end == 0:
                # The batch's first member is not in this file — a reference fork's
                # prefix holds it — so there is nothing here to cut back to, and the
                # seed will refuse the log instead.
                return tail
            start = _line_start(handle, end - 1)
            handle.seek(start)
            if _seq_of(_record(handle.read(end - start))) == batch.first:
                break
            end = start
        handle.truncate(start)
        handle.flush()
        os.fsync(handle.fileno())
    log.warning(
        "ph.persistence.jsonl: %s ended in %d event(s) of a batch a write did not finish; "
        "removed them before appending",
        path,
        (tail.last_seq or batch.first) - batch.first + 1,
    )
    return _Tail(complete=start, torn=b"", last_seq=batch.first - 1 if batch.first > 0 else None)


@dataclass(slots=True)
class JsonlSessionStore:
    """The service published as `ctx.session_persistence`."""

    ctx: Context
    root: Path
    _progress: dict[str, _Progress] = field(default_factory=dict)

    def track(self, session: Session) -> None:
        """Start persisting a session; whatever it holds that we do not is owed.

        **What is owed starts at `durable_length`, not "everything if the file is
        new".** This backend appends, so it must write each event exactly once —
        and the question is not whether the *file* exists but how much of *this
        log* is in it. `Session.durable_length` is that number, stated by
        whoever seeded from storage.

        The earlier gate — `if not path.exists(): queue everything` — encoded a
        premise that is true of two cases and false of the third. A fresh
        session has an empty log, so it queues nothing either way. A fork writes
        a new file, so its whole seed is owed. But a **resume** re-opens an
        existing file with a log that already contains the repair closers and
        `session/end-seed` — present at `track` time, never written — so the
        gate discarded exactly the events that were owed. The result was a gap in
        the seq space, `_readmit` refusing the next seed, and a session that
        could be resumed once. `TursoSessionStore` was unaffected because it
        upserts by `seq` and so queues its whole log unconditionally; the two
        backends disagreed about a Protocol-level guarantee, and this one was
        wrong.

        One rule now covers all three: write what the log has and the store does
        not. `header_written` stays, narrowed to the one thing it was ever about
        — whether the header line is owed.

        **And the file is asked, not only the caller** (B7) — on the first
        flush, not here: `durable_length` is declared once at construction and
        never advances, so a second store instance for a live session is told
        the boundary that was true when the session was built. See `flush`.

                `TursoSessionStore` needs none of this — its `_write` is
        `INSERT OR REPLACE` keyed by seq, so writing a row twice is harmless there.
        """
        if session.id in self._progress:
            return
        path = session_path(self.root, session.id, session.header.family)
        # The **family directory**, not just the root: a log now lives one level
        # down, and creating only the root left every flush raising into
        # `session/flush`'s listener set.
        path.parent.mkdir(parents=True, exist_ok=True)
        # No `header_written=path.exists()`: the first flush measures the file
        # anyway, and a second statement of "is this log new" is one that can
        # disagree with the first — see `_append_and_sync`.
        self._progress[session.id] = _Progress(path=path, cursor=session.durable_length)

    async def flush(self, session: Session) -> None:
        """Write what this log has and the file does not.

        **What is owed is read off the log**, from the cursor, rather than
        drained from a queue — so a flush that does not happen owes exactly what
        it owed before, with nothing to restore. That is the case the old
        take-then-restore existed for, and it is the ordinary case:
        `anyio.to_thread.run_sync` begins with a checkpoint, so a cancellation
        delivered as this flush enters the thread pool raises **before** the
        work is queued — and passivation and teardown are exactly when
        cancellation arrives. An `OSError` (a full disk, a read-only mount) has
        the same shape. A queue emptied before a write that never happened left
        a hole in the seq space, and the next resume died in `_readmit`: *"seed
        must be contiguous from 0"* — a session that could never be opened again.

        **Serialized per log**, so two flushes that overlap — `checkpoint_policy`
        flushing on an event while the supervisor flushes on passivation or
        shutdown — write in turn, and the second finds the cursor already past
        what the first wrote and appends only what arrived since. A waiter
        canceled here has written nothing and owes nothing new.

        The cursor advances only once the write has returned, which is the one
        ordering the rest depends on.
        """
        buffer = self._progress.get(session.id)
        if buffer is None:
            # Every live session is tracked at creation or at activation, so an
            # untracked one is a lifecycle gap worth hearing about — and still
            # written, since what it owes is on the log.
            log.warning("ph.persistence.jsonl: session %s was untracked; tracking now", session.id)
            self.track(session)
            buffer = self._progress[session.id]
        async with buffer.writing:
            if not buffer.measured:
                # **What this file already holds, asked once** (B7).
                # `durable_length` is declared at construction and never
                # advances, so a store built later in a session's life — which
                # is every store after the persistence row re-activates — is
                # told a boundary that was true before anything was flushed and
                # would write the difference again. This backend appends, so that
                # is duplicate events in the file.
                #
                # Asked of the last record's seq, which is absolute
                # (`events[i].seq == i`), so nothing has to be mutated and no
                # offset has to be guessed. The declared value stays a floor: a
                # file *behind* it means events are missing, and re-writing them
                # repairs a hole rather than duplicating anything.
                #
                # Here rather than in `track` because `track` is a synchronous
                # listener, and because a session that never flushes never needs
                # the answer.
                #
                # And settled, not only measured (F6): a torn tail left by a
                # process that died mid-write is finished or removed before this
                # store appends behind it — see `_settle_tail`.
                tail = await anyio.to_thread.run_sync(_settle_tail, buffer.path)
                buffer.measured = True
                # No file, or one whose only line was torn: the header is owed.
                buffer.header_written = tail is not None and tail.complete > 0
                if tail is not None and tail.last_seq is not None:
                    buffer.cursor = max(buffer.cursor, tail.last_seq + 1)
            owed = session.events_from(buffer.cursor)
            header_owed = not buffer.header_written
            records: list[dict[str, Any]] = []
            if header_owed:
                records.append({"type": HEADER_LINE_TYPE, "header": session.header.to_wire()})
            records.extend(event.to_wire(thaw=False) for event in owed)
            if not records:
                return
            try:
                await anyio.to_thread.run_sync(
                    partial(_append_and_sync, buffer.path, records, fresh=header_owed)
                )
            except BaseException:
                # Asked again next time rather than assumed: a write that failed
                # part-way normally takes its bytes back, but if that took-back
                # failed too the file now ends in a fragment, and only a fresh
                # measurement settles it before the retry appends behind it.
                buffer.measured = False
                raise
            buffer.header_written = True
            if owed:
                buffer.cursor = owed[-1].seq + 1

    # ------------------------------------------------------------- reading --
    #
    # The four questions a consumer would otherwise answer by reaching for
    # `self.root` and rebuilding a filename. A backend with no per-session file
    # answers all four; a backend with one answers them from the filesystem.

    def exists(self, session_id: str) -> bool:
        return locate_session(self.root, session_id) is not None

    def read(self, session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
        """The session's full log, following its lineage when it stores a reference.

        `read_own` is this backend's one-file read; `materialize` decides whether
        a chain is owed by looking at the first event's seq. A log that starts at
        0 is complete and is returned unchanged, which is every log written so
        far — so this is a no-op until something writes a reference-fork.
        """
        return materialize(self.read_own, session_id)

    def read_own(
        self, session_id: str, upto: int | None = None, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This file and nothing else — the unchained read `materialize` walks with."""
        path = (
            session_path(self.root, session_id, family)
            if family is not None
            else locate_session(self.root, session_id)
        )
        if path is None or not path.is_file():
            raise FileNotFoundError(f"no stored session {session_id!r}")
        return read_session(path, upto=upto)

    def directory(self) -> Path | None:
        return self.root

    def locate(self, session_id: str) -> Path | None:
        """This backend writes files, so it can always say where."""
        return self._path_for(session_id)

    def _path_for(self, session_id: str) -> Path:
        """This session's log, by what is known before what is on disk.

        A tracked session's path is already decided; anything else is searched
        for; and a session that is neither is answered with where a root *would*
        go, whose family is its own id.

        **That last one is a guess, and it used to be load-bearing**: the lease
        was `<this path>.lock`, so a root claimed before creation locked the
        guess while `create` filed the log somewhere else. The lease is keyed by
        the session id now (`lease.claim_session`), which leaves this answering
        only for `locate`, whose caller is asking where a log is rather than
        holding one against another process.
        """
        buffer = self._progress.get(session_id)
        if buffer is not None:
            return buffer.path
        return locate_session(self.root, session_id) or session_path(
            self.root, session_id, session_id
        )

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this log against every other writer for `scope`'s life (I-5), and
        write every live log before letting go — see `write_on_unwind`."""
        await claim_session(scope, self.root, session_id)
        write_on_unwind(scope, self)

    def stored(self, *, limit: int = 50) -> list[StoredSession]:
        """What is on record, most recently touched first.

        One `stat` per entry — the directory scan's own — and one short read for
        the header, which is the first line. **No title**: deriving one means
        scanning forward for a `user/message` and joining its content blocks,
        and the join (`text_of_wire`) lives in the front end that wants it. The
        TUI's picker keeps its richer summary; this is the part every backend
        can answer, which is what the Protocol is for.
        """
        return [
            stored_row(path.stem, _peek_header(path), stat.st_mtime)
            for path, stat in session_logs(self.root)[:limit]
        ]

    def forget(self, session_id: str) -> None:
        self._progress.pop(session_id, None)


def _append_and_sync(path: Path, records: list[dict[str, Any]], *, fresh: bool) -> None:
    """Append these records and make them durable, file *and* directory.

    The file's own `fsync` is what the barrier promises. It is not enough for the
    first write to a new log: the bytes are durable and the directory entry
    naming them may not be, so an unclean shutdown can leave a session that was
    flushed and is not there. Syncing the directory is the cheap half of the
    promise and only matters once per log.

    `fresh` is told rather than sensed. `flush` already holds it — a header is
    owed on exactly the write whose directory entry is new — and a `path.exists()`
    here would pay a stat on every flush of every session forever to answer "yes"
    once, while making a second statement of "is this log new" that can disagree
    with `header_written`.

    **All or nothing** (F3). A write that fails part-way — a disk that fills
    mid-payload, an `EIO` — has already put some of these bytes in the file, and
    `flush` answers the failure by owing the same records again. Left there, the
    retry appends a full copy behind a half line, and `read_session` refuses the
    log at that line for good. So the file is cut back to the length it had
    before this write, and only then is the failure reported. Through the raw
    descriptor with `O_APPEND` rather than a text handle, because the length has
    to be known exactly and a buffered writer may hold bytes of its own.
    """
    payload = "".join(f"{dumps(record)}\n" for record in records).encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        start = os.fstat(fd).st_size
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        except BaseException:
            _take_back(fd, start, path)
            raise
    finally:
        os.close(fd)
    if fresh:
        # Best effort: a filesystem that refuses a directory handle (some
        # networked ones do) has already given us the file's own durability.
        with suppress(OSError):
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)


def _write_all(fd: int, payload: bytes) -> None:
    """Write every byte, however many calls that takes. A short write is not an error."""
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view) :]


def _take_back(fd: int, length: int, path: Path) -> None:
    """Cut the file back to `length` after a failed write, or say it could not be.

    Never raises: it runs while another exception is on its way out, and that one
    is the account of what happened. A take-back that fails leaves a fragment the
    next flush's measurement settles (`_settle_tail`), because `flush` stops
    trusting its measurement the moment a write fails.
    """
    try:
        os.ftruncate(fd, length)
        os.fsync(fd)
    except OSError:
        log.warning(
            "ph.persistence.jsonl: could not take back a partial write to %s; "
            "the next flush trims it",
            path,
            exc_info=True,
        )


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Every JSON object in an **append-only** log, tolerating a torn tail.

    The tolerance is the whole point, and it is one rule rather than a
    convenience: a log that is only ever appended to has exactly one way to be
    malformed — a process died between the write and the flush — and that
    damage is confined to the last line. Everything before it is sound, so a
    reader that refused the file would discard good records to protect against
    the one bad one. `ph_rlm`'s orphan journal and the Continual Harness's
    global log are both this shape, and both said so in their own comments
    before they said it here.

    Contrast `read_session`, which is deliberately **strict** about every line
    but the last: a session is a conversation, and silently skipping an
    unreadable line in it would hand the model a history that is missing its
    middle. Nothing about JSONL decides which rule applies — the *log's*
    contract does.

    Streamed, and a missing file is an empty log: both callers grow without
    bound, and both had already reached for `read_text()`.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _peek_header(path: Path) -> SessionHeader | None:
    """The header line alone, without reading the log behind it.

    A listing of fifty sessions must not parse fifty whole logs; the header is
    the first line by construction. A header pH cannot validate is not one, so
    this answers `None` rather than guessing.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
    except OSError:
        return None
    try:
        record = json.loads(first)
        return SessionHeader.model_validate(record.get("header"))
    except (json.JSONDecodeError, ValidationError, AttributeError):
        return None


def read_session(
    path: Path, *, upto: int | None = None
) -> tuple[SessionHeader, list[SessionEvent]]:
    """Read a stored session back, validating every envelope.

    Returns the raw header and events; acceptance — the known-types refusal,
    the header-id match, the surface rules — happens when they seed a
    `Session`, on the one path every seed takes.

    `upto` stops at the first event whose seq reaches it. The log is append-only
    and `seq` is its index, so everything after the first such line is at or
    above it too — there is nothing below the boundary further down to miss.

    **Strict about every line but an unterminated last one** (F6). A process
    that dies mid-write leaves a fragment after the final newline, and a reader
    that refused it made one crash cost the whole session — permanently, since
    no later write could get past it. Dropping it is not a guess: no flush
    returned for those bytes, so nothing was ever told they were written, and
    the writer's first flush removes them the same way (`_settle_tail`). An
    unterminated line that *parses* is a whole record that lost only its
    newline, and is kept by both. A malformed line with a newline after it is
    damage of some other kind, and still refuses the log.
    """
    header: SessionHeader | None = None
    events: list[SessionEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                if not line.endswith("\n"):
                    # Only the last line can lack its newline.
                    log.warning(
                        "ph.persistence.jsonl: %s:%d is a write that did not finish; "
                        "reading the log without it",
                        path,
                        number,
                    )
                    break
                raise ValueError(f"{path}:{number}: {error}") from error
            if record.get("type") == HEADER_LINE_TYPE:
                header = SessionHeader.model_validate(record["header"])
                continue
            if header is not None and upto is not None and record.get("seq", -1) >= upto:
                # Read off the **raw** record, before `from_wire`: skipping the
                # validate-and-freeze of the tail is the entire saving, and a
                # bound applied after parsing would save nothing at all. Guarded
                # on the header so a file that puts it after an event still
                # yields one rather than raising "no session header line".
                break
            events.append(SessionEvent.from_wire(record))
    if header is None:
        raise ValueError(f"{path}: no session header line")
    if upto is None:
        dropped = _unfinished_batch(events)
        if dropped:
            log.warning(
                "ph.persistence.jsonl: %s ends in %d event(s) of a batch a write did not "
                "finish; reading the log without them",
                path,
                dropped,
            )
            del events[len(events) - dropped :]
    return header, events


def _unfinished_batch(events: Sequence[SessionEvent]) -> int:
    """How many trailing events are a batch a torn write cut short (P10-15).

    The batch rule beside the torn-line rule, and for its reason: one flush wrote
    the whole batch, a process that died mid-write can cut between two of its
    lines, and nothing was told those bytes were written — so the reader keeps
    none of them rather than half. Only at the **end**: a batch cut short anywhere
    else is damage of another kind, and the seed refuses the log for it.

    Not applied to a bounded read (`upto`): that is a reference fork reading a
    prefix it cites, which never ends inside a batch, and dropping members there
    would silently shorten what a child depends on — the seed refuses instead.
    """
    if not events:
        return 0
    ref = events[-1].batch
    if ref is None or events[-1].seq == ref.last:
        return 0
    return events[-1].seq - ref.first + 1


async def resume_session(ctx: Context, session_id: str) -> Session:
    """Read a stored session, repair a crashed tail, and publish it.

    The repair runs on the seed rather than after publication, so a resumed
    session is provider-valid the first time anything reads it — an open turn
    that reached `derive_messages()` would be rejected by the provider before
    anyone noticed it was unclosed (A5).

    `interrupted` says whether the tail had to be closed, which is the honest
    signal for "this crashed" as against "this was reopened": a clean stop
    synthesizes no closers.
    """
    # Through the Protocol, not through this backend's filename: a store that
    # keeps sessions in a database has no path to build, and `resume_session` is
    # the one function every host calls to pick work back up.
    header, events = ctx.require(SESSION_PERSISTENCE).read(session_id)
    closers = interrupted_turn_closers(events, await _reconciled(ctx, session_id, header, events))
    revived = Session(session_id, seed=[*events, *closers], header=header, durable=len(events))
    # `durable=len(events)`: **what the store already holds is `events`, and
    # nothing else.** The closers
    # are synthesized here and the constructor appends `session/end-seed` on top;
    # both are in the log and neither has been written. A backend that inferred
    # durability from "the file exists" dropped them and left a gap in the seq
    # space, which `_readmit` refuses — so the session resumed once and never
    # again. Said here because this is the only place that knows the difference.
    session = ctx.require(SESSIONS).adopt(revived)
    # Recorded, not just returned. A resume is a fact about *provenance* — this
    # process picked up work somebody else started — and it is not derivable
    # from anything else in the log: a session that was reopened and one that
    # ran straight through look identical afterwards. It matters most where
    # nobody is watching, which is the daemon and a cron-started agent, and it
    # is what lets `phern doctor`, a trajectory reader or a person scrolling back
    # find the seam. One event per reopen, not per turn.
    _LOG.append(
        session,
        "session/resumed",
        {
            "events": len(events),
            "interrupted": bool(closers),
            "closed": len(closers),
        },
    )
    return session


async def _reconciled(
    ctx: Context, session_id: str, header: SessionHeader | None, events: list[SessionEvent]
) -> dict[str, CallOutcome]:
    """Ask each tool about its own started, unresolved call (P10-13).

    Here, mounted, and not in repair — which stays a pure fold a stored log can
    be put through with nothing mounted; only the answers are handed to it. At
    `DEPLOYMENT` scope, since no agent exists yet: an agent-scoped tool is not
    seen, and keeps `TOOL_OUTCOME_UNKNOWN`, as `ToolDefinition.reconcile` says.
    Through `ToolRuntime.reconciled`, the pipeline's own question, so a raise or
    an `Unknown` is no answer here either and a `Done` is rendered as the row that
    registered the tool.

    The session the tool is shown is a read-only copy of the stored log, built
    only when some call has a tool that can answer — the path is a crash with a
    call in flight, and the copy is the price of letting a tool read its own
    context (a workspace root, say) the way it would read a live one.
    """
    tools = ctx.get(TOOLS)
    if tools is None:
        return {}
    answers: dict[str, CallOutcome] = {}
    view: Session | None = None
    for call in unresolved_calls(events):
        definition = tools.get(as_str(call.data.get("name")), scope=DEPLOYMENT)
        if definition is None or definition.reconcile is None:
            continue
        view = view or Session(session_id, seed=events, header=header)
        said = await tools.reconciled(call, view, scope=DEPLOYMENT)
        if said is None:
            continue
        answers[as_str(call.data.get("callId"))] = (
            CallOutcome(done=True, content=tuple(block.to_wire() for block in said))
            if isinstance(said, tuple)
            else CallOutcome(done=False)
        )
    return answers


class Config(WireModel):
    """Row config for `session-persistence-jsonl`."""

    root: str | None = None
    """Where the logs live; `$PH_HOME/sessions` when unset."""


@plugin("session-persistence-jsonl", inject=[SESSIONS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the JSONL backend and wire it to the session firehose."""
    root = Path(config.root) if config.root else resolve_roots().sessions_dir()
    # Annotated, so mypy checks this backend against the Protocol *with
    # signatures* — which the runtime `isinstance` gate cannot: a
    # `runtime_checkable` Protocol compares names only.
    store: SessionPersistence = JsonlSessionStore(ctx=ctx, root=root)
    attach(ctx, store)
