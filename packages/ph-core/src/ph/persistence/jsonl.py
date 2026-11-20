"""`session-persistence-jsonl`: the log on disk, with one JSON object on each line.

The append hot path must never block on I/O (A1). So this provider writes only
on `session/flush`. What it writes is read off the log itself: everything past
a cursor that marks what the file already holds. `session/flush` is a
`parallel` event, so a caller that awaits it has awaited every backend, and not
only the first one to answer.

The file format is deliberately the format of dsh: a header line, then one
event on each line, with camelCase names throughout (Q2). So dsh tooling reads
a pH session. In the other direction, every reader accepts both snake_case and
camelCase names (`ph.wire`), so a JSONL that was written with snake_case names
needs no second parser.

Writes are whole lines, and each barrier `fsync`s them. **A write is all or
nothing.** A write that fails part-way takes back the bytes that it wrote,
before the failure is reported (`_append_and_sync`). So the retry that `flush`
owes appends to a clean end. This cannot cover a process that dies during a
write, which leaves a torn final line. That is the one kind of damage that an
append-only log can repair without guessing, because nothing ever reported
those bytes as written. The reader drops them, and the first flush of the
writer trims them (`_settle_tail`). A malformed line anywhere *before* the last
line is still refused. Encoding happens in the worker thread, beside the I/O.
The checkpoint policy awaits a flush before every model request, so nothing
about a flush should hold the event loop.

@module ph.persistence.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

import anyio
from pydantic import ValidationError

from ..cordis import Context, plugin
from ..json import JsonObject, dumps
from ..keys import SESSIONS
from ..paths import (
    free_sibling,
    make_directories,
    resolve_roots,
    sync_directory,
    write_all,
    write_atomic,
)
from ..session import (
    BatchRef,
    Session,
    SessionEvent,
    SessionHeader,
)
from ..wire import WireModel, validation_summary
from .families import children_under, locate_under, logs_under, path_under, stored_under
from .lease import claim_session, leased
from .lineage import materialize
from .protocol import (
    NoStoredSession,
    SessionPersistence,
    StoredSession,
    attach,
    descendants_among,
    stored_row,
    write_on_unwind,
)

__all__ = [
    "JsonlArchive",
    "JsonlSessionStore",
    "apply",
    "family_log",
    "locate_session",
    "logs_holding",
    "read_records",
    "read_session",
    "read_stored",
    "records_in",
    "session_logs",
    "session_path",
    "skim_session",
    "skim_stored",
]


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
    """Every stored log under `root`, newest first. `tag` narrows the list to one cwd."""
    return logs_under(root, SUFFIX, tag=tag)


def read_stored(
    root: Path,
    session_id: str,
    upto: int | None = None,
    family: str | None = None,
    *,
    types: frozenset[str] | None = None,
) -> tuple[SessionHeader, list[SessionEvent]]:
    """The file of one stored log under `root`, and nothing else. This is a `ReadOne`
    for `materialize` that has no store and no mount behind it. It is for a reader
    that needs a log before either exists.

    The session is absent when the search finds nothing, or when the open finds
    nothing. There is no `stat` first: for a path that the search had just
    found, a `stat` asked the same question a second time.
    """
    path = (
        session_path(root, session_id, family)
        if family is not None
        else locate_session(root, session_id)
    )
    if path is not None:
        with suppress(FileNotFoundError):
            return read_session(path, upto=upto, types=types)
    raise NoStoredSession(f"no stored session {session_id!r}")


def skim_stored(
    root: Path,
    session_id: str,
    upto: int | None = None,
    family: str | None = None,
    *,
    kinds: str,
) -> tuple[SessionHeader, int | None, list[SessionEvent]]:
    """The file of one stored log, keeping only the events whose type begins with
    `kinds`, together with the seq of the first event of the file. This is a
    `ReadSome` for `materialize_some`, as `read_stored` is a `ReadOne`. A host uses
    it for what it folds out of a lineage before anything is mounted (its profile
    records), with no store behind it.

    The read is a skim. The lines of other types are passed unparsed, as
    `skim_session` passes them, and the file is read only to `upto`. So a damaged
    line among them is not seen. That is correct for a fold that a strict read
    follows (the `materialize` of the open).
    """
    path = (
        session_path(root, session_id, family)
        if family is not None
        else locate_session(root, session_id)
    )
    if path is not None:
        with suppress(FileNotFoundError):
            header, first, events = _read(path, upto, None, (f"{_TYPED}{kinds}".encode(),))
            return header, first, [event for event in events if event.type.startswith(kinds)]
    raise NoStoredSession(f"no stored session {session_id!r}")


def logs_holding(
    root: Path, types: frozenset[str], *, gate: str
) -> Iterator[tuple[str, list[SessionEvent]]]:
    """Every stored log under `root`, by id, with its records of `types`. This is
    for a reader that has no store and no mount behind it, such as a daemon that
    rebuilds its schedule index (S18).

    **A log whose bytes never name `gate` is not parsed**, and it yields no
    records. Most logs hold nothing that such a reader wants, and one search of
    the bytes is the whole cost of finding that out. `gate` is the record without
    which the other records mean nothing: the creation of a schedule, for its
    ticks and its cancel. A log that cannot be read is logged and left out, so a
    reader can tell "holds none" from "could not say".
    """
    needle = gate.encode()
    for path, _stat in session_logs(root):
        try:
            if needle not in path.read_bytes():
                yield path.stem, []
                continue
            # This is a skim. A log that holds a schedule is read for its records
            # of these types. A log that is damaged elsewhere is for the resume
            # to refuse.
            _header, events = skim_session(path, types)
        except (OSError, ValueError):
            log.warning("ph.persistence.jsonl: could not read %s", path, exc_info=True)
            continue
        yield path.stem, events


def locate_session(root: Path, session_id: str) -> Path | None:
    """The log for one id, wherever it is. `None` if there is none.

    A caller that holds the family should build the path and not call this. See
    `locate_under` for the cost of the search.
    """
    return locate_under(root, session_id, SUFFIX)


@dataclass(slots=True)
class _Progress:
    """How far this store has written the log of one session. It is not a copy of
    the log."""

    path: Path
    cursor: int = 0
    """The first seq that this file does not yet hold. **The log is the queue** (F2).

    What is owed is `session.events_from(cursor)`, read at flush time. It is not
    a second list that the `session/event` listener fills. That list was only as
    complete as the life of the listener. Teardown is exactly when events are
    appended (`workspace/disposed`, tombstones, the closers of a canceled turn),
    and exactly when the listeners of this row have already unwound. So every
    one of those events was appended to the session and never reached the file.
    A cursor over the log cannot miss an event, because nothing ever had to tell
    it about one."""
    header_written: bool = False
    writing: anyio.Lock = field(default_factory=anyio.Lock)
    """One flush of this log at a time. See `JsonlSessionStore.flush`."""
    measured: bool = False
    """Whether the cursor and `header_written` have been checked against the file (B7).

    The check runs on the first flush and not at `track`. The answer costs reads
    of the file, and `track` is a synchronous `session/created` listener. The
    docstring of `TursoSessionStore.track` refuses the same thing. A session that
    is tracked and never flushed never pays for the check. The flag is cleared
    after a failed write, which may have left a fragment that only a new
    measurement settles."""


_TAIL_CHUNK = 64 * 1024
"""How far one backwards read reaches when looking for a line boundary."""


@dataclass(frozen=True, slots=True)
class _Tail:
    """How a log ends: where its last finished line stops, and what is after it."""

    complete: int
    """The bytes through the last newline: every line that a write finished."""
    torn: bytes
    """What follows that newline. It is empty in a log where every write finished.
    Otherwise it is an unterminated final line, which only a write that did not
    finish leaves."""
    last_seq: int | None
    """The seq of the last finished record, or `None` when there is none."""
    open_batch: BatchRef | None = None
    """The batch that the last record belongs to, when that record is not the last
    member of the batch. A torn write can leave this, and `_settle_batch` takes
    it back."""


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

    **The answer uses the seq of the record itself, and not a line count.**
    Counting lines and adding an offset needs the index that the file *starts*
    at. The two numbers available, `durable_length` and `header.seed_length`,
    disagree for a reference fork, so the arithmetic silently dropped the first
    events of a fork. A seq is absolute. `Session._append` assigns
    `seq == len(log)` (A1) and a seed preserves it, so `events[i].seq == i` for
    every log, forked or not.

    The file is read from the end and not scanned. Only the last line is
    wanted, so this takes a few seeks however long the log is. **It reads
    backwards in chunks until it finds a boundary**, and not in one fixed
    window. A single 64 KiB read split a final record that was longer than the
    window, parsed the fragment, and answered `None`. So a large tool result at
    the end of a log made a re-activated store queue again everything that the
    file already held.
    """
    try:
        handle = path.open("rb")
    except OSError:
        # There is no file yet (a new session, or a fork that has not written),
        # or this process cannot read the file. The declared boundary then
        # answers alone.
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
    """The JSON object that a line holds, or `None` when it holds none.

    This is also the test for an unterminated final line. A record is a JSON
    **object**, and no proper prefix of the encoding of an object parses,
    because the closing brace is its last byte. So a fragment that parses as an
    object is the entire record, cut after the record and before the `\\n` that
    follows.
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

    Only a process that died during a write leaves anything after the last
    newline. A write that *failed* takes its own bytes back
    (`_append_and_sync`). The next append would be joined onto the leftover
    bytes. That would turn one lost tail into a line in the middle of the log
    that nobody can parse. So before this store appends anything:

    * a fragment that is a **whole record** gets the newline that it lost, and
      counts as written. `read_session` reads the same bytes the same way.
    * anything else is **removed**. Nothing reported it as written, because
      `flush` had not returned. So no reader was told that it exists, and
      `read_session` drops it for the same reason.

    These two rules are the rules of the reader. They are applied by the one
    party that is allowed to change the file: the lease (I-5) makes this store
    the only writer of the file.
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
    """Take back a batch that a torn write cut short, as `read_session` drops it
    (P10-15).

    This runs after the torn line is settled, so the last record is a whole
    record. If that record belongs to a batch that it does not finish, the
    function walks back line by line to the member whose seq is the `first` of
    the batch, and cuts the file there. The reader keeps none of an unfinished
    batch, and the first append of the writer must not land behind half of one.
    """
    batch = tail.open_batch
    if batch is None:
        return tail
    with path.open("r+b") as handle:
        end = tail.complete
        while True:
            if end == 0:
                # The first member of the batch is not in this file, because the
                # prefix of a reference fork holds it. So there is nothing here
                # to cut back to, and the seed will refuse the log instead.
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
class JsonlArchive:
    """The read half of the JSONL backend: a `SessionArchive` over one directory.

    It is a class of its own because two readers need the reads of the store
    and have nothing to write. One is a daemon that rebuilds an index at boot,
    before any root has mounted a store (`Supervisor.owed`). The other is a test
    that reads what a run left behind. Both used to build a `JsonlSessionStore`
    with `ctx=None` and a type-ignore. That misstated a field which nothing on
    the read side uses. `JsonlSessionStore` adds the writer over this class.
    """

    root: Path

    def exists(self, session_id: str, *, family: str | None = None) -> bool:
        return stored_under(self.root, session_id, SUFFIX, family)

    def read(
        self, session_id: str, *, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """The full log of the session, following its lineage when it stores a reference.

        `read_own` is the one-file read of this backend. `materialize` decides
        whether a chain is owed by looking at the seq of the first event. A log
        that starts at 0 is complete and is returned unchanged. A log that
        starts later stores a reference, and `materialize` reads the rest from
        its ancestors.
        """
        return materialize(self.read_own, session_id, family=family)

    def read_own(
        self,
        session_id: str,
        upto: int | None = None,
        family: str | None = None,
        *,
        types: frozenset[str] | None = None,
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This file and nothing else: the unchained read that `materialize` walks with."""
        return read_stored(self.root, session_id, upto, family, types=types)

    def directory(self) -> Path | None:
        return self.root

    def holding(
        self, types: frozenset[str], *, gate: str
    ) -> Iterator[tuple[str, list[SessionEvent]]]:
        return logs_holding(self.root, types, gate=gate)

    def stored(self, *, limit: int | None = 50) -> list[StoredSession]:
        """What is on record, most recently touched first.

        This costs one `stat` for each entry, which the directory scan already
        made, and one short read for the header, which is the first line.
        **There is no title.** Deriving one means scanning forward for a
        `user/message` and joining its content blocks, and the join
        (`text_of_wire`) is in the front end that wants it. The picker of the
        TUI keeps its richer summary. This is the part that every backend can
        answer, and the Protocol exists for that.
        """
        return [
            stored_row(path.stem, _peek_header(path), stat.st_mtime)
            for path, stat in session_logs(self.root)[:limit]
        ]

    def descendants_of(self, parent_id: str, family: str) -> tuple[StoredSession, ...]:
        """Every stored session beneath `parent_id`: one `scandir` of its family, then
        one header line for each log whose name has the prefix of the parent.

        This is the same peek that `stored` makes for each row, so the row of a
        child here is the row that the listing would have shown for it. Only the
        candidates are read. They are the descendants of the parent and any log
        that is named after it.
        """
        return descendants_among(
            parent_id,
            (
                stored_row(path.stem, _peek_header(path), stat.st_mtime)
                for path, stat in children_under(self.root, family, parent_id, SUFFIX)
            ),
        )


@dataclass(slots=True)
class JsonlSessionStore(JsonlArchive):
    """The service published as `ctx.session_persistence`: `JsonlArchive`, and the
    writer over it."""

    ctx: Context
    _progress: dict[str, _Progress] = field(default_factory=dict)

    def track(self, session: Session) -> None:
        """Start persisting a session. Whatever it holds that the store does not is owed.

        **What is owed starts at `durable_length`, and not at "everything if the
        file is new".** This backend appends, so it must write each event
        exactly one time. The question is not whether the *file* exists. The
        question is how much of *this log* is in it. `Session.durable_length`
        is that number, stated by whoever seeded the session from storage. It
        holds until the first `flush` measures the file, and from then the tail
        of the file decides.

        The earlier gate was `if not path.exists(): queue everything`. It
        encoded a premise that was true of two cases and false of the third:

        - A new session has an empty log, so it queues nothing either way.
        - A fork then copied its seed into a new file, so its whole seed was
          owed.
        - A **resume** opens an existing file again, with a log that already
          contains the repair closers. They are present at `track` time and
          were never written, so the gate discarded exactly the events that
          were owed.

        The result was a gap in the seq space, `_readmit` refusing the next
        seed, and a session that could be resumed one time. `TursoSessionStore`
        was unaffected at that time, because it upserted by `seq` and queued
        its whole log unconditionally. The two backends disagreed about a
        guarantee at the level of the Protocol, and this one was wrong.

        One rule now covers all three cases: write what the log has and the
        store does not. `header_written` stays. It is narrowed to the one thing
        that it was always about, which is whether the header line is owed.

        **The file is asked too, and not only the caller** (B7). That happens on
        the first flush and not here. `durable_length` is declared one time at
        construction and never advances, so a second store instance for a live
        session is told the boundary that was true when the session was built.
        See `flush`.

        `TursoSessionStore` needs none of this. Its `_write` skips a row that
        the database already holds unchanged, so writing a row twice is harmless
        there.
        """
        if session.id in self._progress:
            return
        path = session_path(self.root, session.id, session.header.family)
        # The write that creates the log makes its directories, off the loop
        # and durably (`_append_and_sync`). There is no
        # `header_written=path.exists()` here. The first flush measures the file
        # anyway, and a second statement of "is this log new" can disagree with
        # the first. See `_append_and_sync`.
        self._progress[session.id] = _Progress(path=path, cursor=session.durable_length)

    async def flush(self, session: Session) -> None:
        """Write what this log has and the file does not.

        **What is owed is read off the log**, from the cursor. It is not
        drained from a queue. So a flush that does not happen owes exactly what
        it owed before, and there is nothing to restore. The old
        take-then-restore existed for that case, and it is the ordinary case.
        `anyio.to_thread.run_sync` begins with a checkpoint. A cancellation
        that is delivered as this flush enters the thread pool therefore raises
        **before** the work is queued, and passivation and teardown are exactly
        when cancellation arrives. An `OSError` (a full disk, a read-only
        mount) has the same shape. A queue that was emptied before a write that
        never happened left a hole in the seq space. The next resume then
        failed in `_readmit` with *"seed must be contiguous from 0"*, and the
        session could never be opened again.

        **Flushes are serialized for each log.** Two flushes can overlap:
        `checkpoint_policy` flushes on an event while the supervisor flushes on
        passivation or shutdown. They write in turn. The second finds the
        cursor already past what the first wrote, and appends only what arrived
        since. A waiter that is canceled here has written nothing and owes
        nothing new.

        The cursor advances only after the write has returned. Everything else
        depends on that one ordering.
        """
        buffer = self._progress.get(session.id)
        if buffer is None:
            # Every live session is tracked at creation or at activation. An
            # untracked session is therefore a gap in the lifecycle that is
            # worth a warning. It is still written, because what it owes is on
            # the log.
            log.warning("ph.persistence.jsonl: session %s was untracked; tracking now", session.id)
            self.track(session)
            buffer = self._progress[session.id]
        async with buffer.writing:
            if not buffer.measured:
                # **What this file already holds is asked one time** (B7).
                # `durable_length` is declared at construction and never
                # advances. A store that is built later in the life of a session
                # is every store after the persistence row re-activates. Such a
                # store is told a boundary that was true before anything was
                # flushed, and it would write the difference again. This backend
                # appends, so that would put duplicate events in the file.
                #
                # The question is asked of the seq of the last record, which is
                # absolute (`events[i].seq == i`). So nothing has to be mutated
                # and no offset has to be guessed. **What the file holds
                # decides, in both directions.** When the file is ahead of the
                # declared value, what it holds is not written again. When the
                # file is *behind* the declared value, events are missing, and
                # writing them repairs a hole that the next resume would refuse
                # ("seed must be contiguous from 0"). The declared value stands
                # only for a file that holds no event yet. That is the file of a
                # fork, whose prefix belongs to its parent.
                #
                # This is here and not in `track`, because `track` is a
                # synchronous listener, and because a session that never
                # flushes never needs the answer.
                #
                # The tail is also settled, and not only measured (F6). A torn
                # tail that a process left when it died during a write is
                # finished or removed before this store appends behind it. See
                # `_settle_tail`.
                tail = await anyio.to_thread.run_sync(_settle_tail, buffer.path)
                buffer.measured = True
                # No file, or one whose only line was torn: the header is owed.
                buffer.header_written = tail is not None and tail.complete > 0
                if tail is not None and tail.last_seq is not None:
                    buffer.cursor = tail.last_seq + 1
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
                # The file is measured again next time, and nothing is assumed.
                # A write that failed part-way normally takes its bytes back.
                # If the take-back failed too, the file now ends in a fragment,
                # and only a new measurement settles it before the retry appends
                # behind it.
                buffer.measured = False
                raise
            buffer.header_written = True
            if owed:
                buffer.cursor = owed[-1].seq + 1

    # ------------------------------------------------------------- reading --
    #
    # The four questions that a consumer would otherwise answer by reaching for
    # `self.root` and rebuilding a filename. A backend with no per-session file
    # answers all four. A backend with one answers them from the filesystem.
    # Most of the answers are in `JsonlArchive`, which this class extends.

    def locate(self, session_id: str) -> Path | None:
        """This backend writes files, so it can always say where."""
        return self._path_for(session_id)

    def _path_for(self, session_id: str) -> Path:
        """The log of this session, from what is known before what is on disk.

        The path of a tracked session is already decided. Any other session is
        searched for. A session that is neither tracked nor found is answered
        with where a root *would* go, and the family of a root is its own id.

        **That last answer is a guess, and it used to matter.** The lease was
        `<this path>.lock`, so a root that was claimed before creation locked
        the guess, and `create` then filed the log somewhere else. The lease is
        now keyed by the session id (`lease.claim_session`). So this function
        answers only for `locate`, whose caller asks where a log is and does
        not hold the log against another process.
        """
        buffer = self._progress.get(session_id)
        if buffer is not None:
            return buffer.path
        return locate_session(self.root, session_id) or session_path(
            self.root, session_id, session_id
        )

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this log against every other writer for the life of `scope` (I-5), and
        write every live log before the lease is released. See `write_on_unwind`."""
        await claim_session(scope, self.root, session_id)
        write_on_unwind(scope, self)

    def forget(self, session_id: str) -> None:
        self._progress.pop(session_id, None)


def _append_and_sync(path: Path, records: list[dict[str, Any]], *, fresh: bool) -> None:
    """Append these records and make them durable: the file *and* the directory.

    The `fsync` of the file is what the barrier promises. It is not enough for
    the first write to a new log. The bytes are durable, but the directory
    entry that names them may not be. An unclean shutdown can then leave a
    session that was flushed and is not there. Syncing the directory is the
    cheap half of the promise, and it matters only one time for each log. **The
    first write also makes the directories of the log**, with each new name
    synced in its parent (`make_directories`, S20). A family directory that the
    sessions root never recorded took the log with it.

    `fresh` is passed in and not sensed. `flush` already holds it: a header is
    owed on exactly the write whose directory entry is new. A `path.exists()`
    here would pay a stat on every flush of every session, forever, to answer
    "yes" one time. It would also be a second statement of "is this log new"
    that can disagree with `header_written`.

    **All or nothing** (F3). A write that fails part-way (a disk that fills in
    the middle of the payload, an `EIO`) has already put some of these bytes in
    the file, and `flush` answers the failure by owing the same records again.
    If the bytes stayed, the retry would append a full copy behind half a line,
    and `read_session` would refuse the log at that line permanently. So the
    file is cut back to the length that it had before this write, and only then
    is the failure reported. The write goes through the raw descriptor with
    `O_APPEND` and not through a text handle. The length must be known exactly,
    and a buffered writer may hold bytes of its own.
    """
    payload = "".join(f"{dumps(record)}\n" for record in records).encode("utf-8")
    if fresh:
        make_directories(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        start = os.fstat(fd).st_size
        try:
            write_all(fd, payload)
            os.fsync(fd)
        except BaseException:
            _take_back(fd, start, path)
            raise
    finally:
        os.close(fd)
    if fresh:
        sync_directory(path.parent)


def _take_back(fd: int, length: int, path: Path) -> None:
    """Cut the file back to `length` after a failed write, or log that it could not.

    This never raises. It runs while another exception is propagating, and that
    exception is the record of what happened. A take-back that fails leaves a
    fragment. The measurement of the next flush settles it (`_settle_tail`),
    because `flush` stops trusting its measurement as soon as a write fails.
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


def append_records(path: Path, records: Sequence[JsonObject]) -> None:
    """Append records to an **append-only** log and make them durable. This is the
    writer that pairs with `read_records`, for a log outside the session store (L4).

    It gives the guarantees of `_append_and_sync`: whole lines, `fsync`ed. A
    write that fails part-way is taken back, so the next write does not land
    behind half a line. The directory is synced on the write that creates the
    file. A log whose records other sessions act on, such as the global log of
    the Continual Harness, needs all of these. Blocking. Call it through
    `anyio.to_thread.run_sync` from async code.
    """
    _append_and_sync(path, [dict(record) for record in records], fresh=not path.exists())


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Every JSON object in an **append-only** log, tolerating a torn tail.

    The tolerance is the purpose of this function, and it is one rule and not a
    convenience. A log that is only ever appended to can be malformed in
    exactly one way: a process died between the write and the flush. That
    damage is confined to the last line. Everything before it is sound, so a
    reader that refused the file would discard good records to protect against
    the one bad record. The orphan journal (`ph.orphans`) and the global log of
    the Continual Harness both have this shape. Both said so in their own
    comments before this function said it.

    Contrast `read_session`, which is deliberately **strict** about every line
    except the last. A session is a conversation, and silently skipping an
    unreadable line in it would give the model a history with its middle
    missing. Nothing about JSONL decides which rule applies. The contract of
    the *log* decides.

    The records are streamed, and a missing file is an empty log. Both logs
    grow without bound, and the reader of each had already used `read_text()`.
    The orphan journal now reads its own bytes and applies this rule through
    `records_in`.
    """
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        yield from records_in(handle)


def records_in(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """The JSON objects among `lines`. A line that is not one is skipped. This is
    the rule of `read_records`, for a caller that already holds the lines."""
    for line in lines:
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

    A listing of fifty sessions must not parse fifty whole logs, and the header
    is the first line by construction. A header that pH cannot validate is not
    a header, so this answers `None` and does not guess.
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


_TYPED = '{"type":"'
"""How every line that this backend writes begins. `to_wire` puts `type` first
and `dumps` is compact, so the type of a line can be read off its text
(`skim_session`)."""

_TYPED_LINE = _TYPED.encode()
"""`_TYPED` as the reader meets it: a line in bytes, before it is decoded."""


def read_session(
    path: Path, *, upto: int | None = None, types: frozenset[str] | None = None
) -> tuple[SessionHeader, list[SessionEvent]]:
    """Read a stored session back, validating every envelope.

    Returns the raw header and events. Acceptance happens when they seed a
    `Session`, on the one path that every seed takes: the known-types refusal,
    the header-id match, the surface rules.

    `upto` stops at the first event whose seq reaches it. The log is
    append-only and `seq` is its index, so everything after the first such line
    is at or above it too. Nothing below the boundary can be further down and
    be missed.

    **Strict about every line except an unterminated last line** (F6). A
    process that dies during a write leaves a fragment after the final newline.
    A reader that refused the fragment made one crash cost the whole session,
    permanently, because no later write could get past it. Dropping the
    fragment is not a guess. No flush returned for those bytes, so nothing was
    ever told that they were written, and the first flush of the writer removes
    them in the same way (`_settle_tail`). An unterminated line that *parses*
    is a whole record that lost only its newline, and both the reader and the
    writer keep it. A malformed line with a newline after it is damage of some
    other kind, and it still refuses the log.

    `types` keeps only events of those types. The others are skipped on the
    **raw** record before `from_wire`, for the reason that `upto` has: the
    validate-and-freeze is the cost. The last record of any type is kept aside,
    so the torn-batch rule (`_unfinished_batch`) still judges the tail by what
    was written and not by what was kept.
    """
    header, _first, events = _read(path, upto, types, None)
    return header, events


def skim_session(path: Path, types: frozenset[str]) -> tuple[SessionHeader, list[SessionEvent]]:
    """`read_session(path, types=types)`, without parsing the lines that it discards.

    A line whose own text names another type (`_TYPED`) is passed unparsed.
    `json.loads` of the discarded lines was most of a typed read: 17 ms against
    5 on a 9 800-event log. The skim gives up one refusal, so it is for a reader
    that can lose that refusal: a damaged line among the passed lines is not
    seen, where `read_session` refuses the log. It is not for a reader that
    must fail closed, because a damaged type name reads as another type. The
    last line is parsed whatever it names, so the torn-tail and torn-batch
    rules judge the tail as a strict read does.
    """
    prefixes = tuple(f'{_TYPED}{one}"'.encode() for one in types)
    header, _first, events = _read(path, None, types, prefixes)
    return header, events


@dataclass(frozen=True, slots=True)
class LogDamage:
    """Where a stored log stops reading: the first line that `read_session` refuses,
    the reason, and how much before it a repair keeps (P13-09)."""

    path: Path
    line: int
    """The number of the refused line, counted from 1."""
    reason: str
    kept: int
    """The bytes before the refused line. This is the longest prefix that the reader
    accepts, and a repair truncates to it. It is zero when no header line reads
    before the refused line, because a prefix with no header is not a log. Then
    there is no repair."""


class LogDamaged(ValueError):
    """The refusal of a stored log by `read_session`, which says where (`damage`).

    It is a `ValueError`, as every refusal of the content of a log was before
    the refusal had a type of its own. So a caller that catches `ValueError`
    still catches it. A caller that must tell a log that will not read from any
    other failure catches this type. Examples: a sub-agent sweep that isolates
    one bad child (`ph.seams.subagents.stored_tree`), and `phern doctor sessions`.
    """

    def __init__(self, damage: LogDamage) -> None:
        super().__init__(f"{damage.path}:{damage.line}: {damage.reason}")
        self.damage = damage


def _read(
    path: Path,
    upto: int | None,
    types: frozenset[str] | None,
    wanted: tuple[bytes, ...] | None,
) -> tuple[SessionHeader, int | None, list[SessionEvent]]:
    """The loop of `read_session`. It is also the loop of `skim_session`, when
    `wanted` holds the line prefixes of the types to keep. It also returns the
    seq of the first event of the file, which says whether the file holds its
    own history (`lineage`).

    **This is the one statement of what a log must be to be read.** Every
    refusal is a `LogDamaged` that names the line and the bytes before it: a
    line that is not a record, an envelope that this build cannot validate, no
    header. So `damage_in` asks this loop a question, and is not a second copy
    of its rules. The file is read in bytes for that count, and each line is
    decoded on its own.

    **A skim stops at `upto` without parsing up to it.** The lines that a skim
    passes are not parsed, so their seqs are not read. But the seqs of a log
    run one for each line from its first, so the line that should reach `upto`
    is known. From that line, each line is parsed until one reaches `upto`. A
    file whose seqs do not run that way is only read further than it needed to
    be, and never short.
    """
    header: SessionHeader | None = None
    first: int | None = None
    lines = 0  # event lines so far: the next one's seq is `first + lines` as written
    events: list[SessionEvent] = []
    tail: dict[str, Any] | None = None
    tail_number = tail_start = 0
    passed: tuple[int, int, bytes] | None = None
    number = start = offset = 0
    with path.open("rb") as handle:
        try:
            for number, raw in enumerate(handle, start=1):
                start, offset = offset, offset + len(raw)
                if (
                    wanted is not None
                    and header is not None
                    and first is not None
                    and (upto is None or first + lines < upto)
                    and raw.startswith(_TYPED_LINE)
                    and not raw.startswith(wanted)
                    and raw.endswith(b"\n")
                ):
                    # The tail is the last passed line, when no line is parsed
                    # after it.
                    passed = (number, start, raw)
                    lines += 1
                    continue
                if not raw.strip():
                    continue
                record = _parse(path, number, raw)
                if record is None:
                    break
                if record.get("type") == HEADER_LINE_TYPE:
                    header = SessionHeader.model_validate(record.get("header"))
                    continue
                if first is None and isinstance(seq := record.get("seq"), int):
                    first = seq
                lines += 1
                if header is not None and upto is not None and record.get("seq", -1) >= upto:
                    # The bound is read off the **raw** record, before
                    # `from_wire`. Skipping the validate-and-freeze of the tail
                    # is the entire saving, and a bound applied after parsing
                    # would save nothing at all. The check is guarded on the
                    # header, so that a file that puts the header after an event
                    # still yields a header and does not raise "no session
                    # header line".
                    break
                tail, passed = record, None
                tail_number, tail_start = number, start
                if types is not None and record.get("type") not in types:
                    continue
                events.append(SessionEvent.from_wire(record))
        except ValueError as error:
            raise _damaged(path, number, start, error, header=header is not None) from error
    if header is None:
        raise LogDamaged(LogDamage(path, 1, "no session header line", 0))
    try:
        if passed is not None:
            tail_number, tail_start, raw = passed
            tail = _parse(path, tail_number, raw)
        last = SessionEvent.from_wire(tail) if upto is None and tail is not None else None
    except ValueError as error:
        raise _damaged(path, tail_number, tail_start, error, header=True) from error
    if last is not None:
        dropped = _unfinished_batch([last])
        if dropped:
            log.warning(
                "ph.persistence.jsonl: %s ends in %d event(s) of a batch a write did not "
                "finish; reading the log without them",
                path,
                dropped,
            )
            events = [event for event in events if event.seq <= last.seq - dropped]
    return header, first, events


def _damaged(path: Path, number: int, start: int, error: ValueError, *, header: bool) -> LogDamaged:
    """The refusal of line `number`, which began `start` bytes in."""
    reason = validation_summary(error) if isinstance(error, ValidationError) else str(error)
    return LogDamaged(LogDamage(path, number, reason, start if header else 0))


def _parse(path: Path, number: int, raw: bytes) -> dict[str, Any] | None:
    """The record of one line, or `None` for an unterminated last line that does not
    parse. Such a line is a write that did not finish (F6). It can be cut anywhere,
    including inside a multi-byte character. Anything else that is not a record
    raises `ValueError` with the reason."""
    if not raw.endswith(b"\n"):
        # Only the last line can lack its newline.
        try:
            record = json.loads(raw.decode("utf-8"))
        except ValueError:
            log.warning(
                "ph.persistence.jsonl: %s:%d is a write that did not finish; "
                "reading the log without it",
                path,
                number,
            )
            return None
    else:
        record = json.loads(raw.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("not a record")
    return record


def damage_in(path: Path) -> LogDamage | None:
    """Where a stored log stops reading, or `None` for a log that `read_session`
    takes whole.

    This asks the strict read itself (`_read`). So whatever a resume refuses,
    this finds, at the line that the resume would name. A torn last line is not
    damage: the reader drops it, and the first flush of the writer settles it
    (F6).
    """
    try:
        _read(path, None, None, None)
    except LogDamaged as damaged:
        return damaged.damage
    return None


def damaged_logs(root: Path) -> tuple[int, list[LogDamage]]:
    """Every stored log under `root` that the strict reader refuses, and how many
    logs it read.

    It reads every log, those of sub-agents as well as those of roots, through
    the listing that this backend lists by (`session_logs`). So it scans what a
    daemon on this `$PH_HOME` would find. `phern doctor sessions` uses it.
    """
    logs = session_logs(root)
    return len(logs), [found for path, _ in logs if (found := damage_in(path)) is not None]


def truncate_damaged(root: Path, damage: LogDamage) -> Path:
    """Cut a damaged log back to the prefix that its reader accepts. The whole file
    is first kept beside it. Both happen under the lease of the session. Returns
    the copy.

    **Under the lease** (`leased`), which is held for the cut and then released.
    A log that a daemon or a one-shot run is appending to raises `SessionBusy`.
    It is not truncated under its writer, which would put `seq` backwards. The
    lease is taken here and not at the caller, so that a cut is safe however it
    is reached.

    The copy is `<log>.damaged`, or a numbered name when an earlier copy exists
    (`free_sibling`). The listing does not read that suffix. The copy is written
    durably (`write_atomic`) before the cut, so a repair loses nothing that a
    person could not get back. Then the log is truncated and synced. A log with
    nothing to keep is refused with a `ValueError`.
    """
    if damage.kept == 0:
        raise ValueError(f"{damage.path}: nothing before line {damage.line} reads; not repaired")
    with leased(root, damage.path.stem):
        copy = free_sibling(damage.path, ".damaged")
        write_atomic(copy, damage.path.read_bytes())
        fd = os.open(damage.path, os.O_WRONLY)
        try:
            os.ftruncate(fd, damage.kept)
            os.fsync(fd)
        finally:
            os.close(fd)
    return copy


def _unfinished_batch(events: Sequence[SessionEvent]) -> int:
    """How many trailing events are a batch that a torn write cut short (P10-15).

    This is the batch rule beside the torn-line rule, and it has the same
    reason. One flush wrote the whole batch. A process that died during the
    write can cut between two of its lines, and nothing was told that those
    bytes were written. So the reader keeps none of them, and not half. The
    rule applies only at the **end**. A batch that is cut short anywhere else
    is damage of another kind, and the seed refuses the log for it.

    The rule is not applied to a bounded read (`upto`). That is a reference
    fork reading a prefix that it cites, which never ends inside a batch.
    Dropping members there would silently shorten what a child depends on, so
    the seed refuses instead.
    """
    if not events:
        return 0
    ref = events[-1].batch
    if ref is None or events[-1].seq == ref.last:
        return 0
    return events[-1].seq - ref.first + 1


class Config(WireModel):
    """Row config for `session-persistence-jsonl`."""

    root: str | None = None
    """The directory that holds the logs. When this is unset, the row uses the
    sessions directory of the host: `$PH_HOME/sessions`, or the `paths.sessions`
    setting of `daemon.yaml`."""


@plugin(
    "session-persistence-jsonl",
    affects="deployment",
    inject=[SESSIONS],
    config=Config,
    log_layer="storage",
)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the JSONL backend and wire it to the session firehose."""
    root = Path(config.root) if config.root else resolve_roots().sessions_dir()
    # The annotation makes mypy check this backend against the Protocol *with
    # signatures*. The runtime `isinstance` gate cannot do that, because a
    # `runtime_checkable` Protocol compares names only.
    store: SessionPersistence = JsonlSessionStore(ctx=ctx, root=root)
    attach(ctx, store)
