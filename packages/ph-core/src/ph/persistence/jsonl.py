"""`session-persistence-jsonl` — the log on disk, one JSON object per line.

The append hot path must never block on I/O (A1), so this provider buffers: it
subscribes to `session/event`, queues the event, and drains on `session/flush`.
`session/flush` is a `parallel` event, so a caller awaiting it has awaited every
backend, not just the first one to answer.

The file format is deliberately dsh's: a header line, then one event per line,
camelCase throughout (Q2). A pH session is therefore a session dsh tooling
reads, and `ph session import` in the other direction needs no second parser.

Writes are atomic-ish by construction — appends of whole lines, `flush()` +
`fsync()` at each barrier — because a torn last line is the one corruption a
JSONL reader cannot repair without guessing. Encoding happens in the worker
thread beside the I/O: the checkpoint policy awaits a flush before every model
request, so nothing about a flush should hold the event loop.

@module ph.persistence.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import anyio
from pydantic import ValidationError

from ..cordis import Context, plugin
from ..json import dumps
from ..keys import SESSION_PERSISTENCE, SESSIONS
from ..paths import resolve_roots
from ..session import Session, SessionEvent, SessionHeader
from ..wire import WireModel
from .families import locate_under, logs_under, path_under
from .lease import claim_session
from .lineage import materialize
from .protocol import SessionPersistence, StoredSession, attach, stored_row

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
class _Buffer:
    path: Path
    pending: list[SessionEvent] = field(default_factory=list)
    header_written: bool = False
    writing: anyio.Lock = field(default_factory=anyio.Lock)
    """One flush of this log at a time. See `JsonlSessionStore.flush`."""
    measured: bool = False
    """Whether this buffer has reconciled its queue against the file (B7).

    Resolved on the first flush rather than at `track`, because the answer costs
    a read of the whole log and `track` is a synchronous `session/created`
    listener — the thing `TursoSessionStore.track` refuses in its own docstring.
    A session that is tracked and never flushed never pays it."""


def _last_seq(path: Path) -> int | None:
    """The seq of the last complete record in this log, or `None` (B7).

    **The record's own seq, not a line count.** Counting lines and adding an
    offset needs to know which index the file *starts* at, and the two numbers
    available — `durable_length` and `header.seed_length` — disagree for a
    reference fork, so the arithmetic silently dropped a fork's first events.
    A seq is absolute: `Session.append` assigns `seq == len(log)` (A1) and a
    seed preserves it, so `events[i].seq == i` for every log, forked or not.

    Read from the tail rather than by scanning: only the last line is wanted,
    so this is a seek and one small read however long the log is. An
    unterminated final line is ignored, which is what a crash mid-write leaves
    and exactly what should not count as written.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            window = min(size, 64 * 1024)
            handle.seek(size - window)
            tail = handle.read(window)
    except OSError:
        # No file yet — a fresh session, or a fork that has not written — or one
        # this process cannot read. The declared boundary then answers alone.
        return None
    for line in reversed(tail.split(b"\n")[:-1]):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            return None
        seq = record.get("seq")
        return seq if isinstance(seq, int) else None
    return None


@dataclass(slots=True)
class JsonlSessionStore:
    """The service published as `ctx.session_persistence`."""

    ctx: Context
    root: Path
    _buffers: dict[str, _Buffer] = field(default_factory=dict)

    def track(self, session: Session) -> None:
        """Start buffering a session; whatever it holds that we do not is owed.

        **The queue is `events[durable_length:]`, not "everything if the file is
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
        `INSERT OR REPLACE` keyed by seq, so re-queueing is idempotent there.
        """
        if session.id in self._buffers:
            return
        path = session_path(self.root, session.id, session.header.family)
        # The **family directory**, not just the root: a log now lives one level
        # down, and creating only the root left every flush raising into
        # `session/flush`'s listener set.
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = _Buffer(path=path, header_written=path.exists())
        buffer.pending.extend(session.events[session.durable_length :])
        self._buffers[session.id] = buffer

    def record(self, session: Session, event: SessionEvent) -> None:
        buffer = self._buffers.get(session.id)
        if buffer is None:
            # Every live session is tracked at creation or at activation, so an
            # untracked one is a lifecycle gap worth hearing about. `track`
            # captures the whole log, this event included.
            log.warning("ph.persistence.jsonl: session %s was untracked; tracking now", session.id)
            self.track(session)
            return
        buffer.pending.append(event)

    async def flush(self, session: Session) -> None:
        """Write what this log has and the file does not, or still owe it.

        **The queue is emptied before the write and restored if the write does
        not happen**, and both halves are load-bearing for different reasons.

        *Before*, because two flushes can overlap — `checkpoint_policy` flushes
        on an event while the supervisor flushes on passivation or shutdown —
        and a second one that found the same events still queued would append
        them twice. Clearing first makes the concurrent flush a no-op.

        **Serialized per log, because take-then-restore is only safe alone.**
        Overlapping flushes were the case the clearing was *for*, and the restore
        is what it could not survive: the earlier flush, canceled at the thread
        hop, put its events back at the front of a queue the later one had
        already taken from and written. The file then held higher seqs before
        lower ones, and `header_written` was restored by whichever failed last —
        so a log could end up with no header line at all, which `read_session`
        refuses outright. A waiter canceled here has taken nothing, so the lock
        costs a concurrent flush exactly what the clearing already cost it: it
        finds the queue empty and does nothing.

        *Restored*, because clearing first is otherwise a way to lose them.
        `anyio.to_thread.run_sync` begins with a checkpoint, so a cancellation
        delivered as this flush enters the thread pool raises **before** the
        work is queued — and passivation and teardown are exactly when
        cancellation arrives. An `OSError` (a full disk, a read-only mount) has
        the same shape. Either way the events were dropped from `pending` and
        never written, so the file gains a hole in its seq space and the next
        resume dies in `_readmit`: *"seed must be contiguous from 0"*. That is
        not a lost flush, it is a session that can never be opened again —
        which is the failure `track`'s docstring above describes a previous
        incarnation of, from a different cause.

        Prepended rather than appended on the way back, because `pending` is
        ordered by seq and anything recorded while the write was in flight
        belongs after what this flush was carrying.
        """
        buffer = self._buffers.get(session.id)
        if buffer is None:
            return
        async with buffer.writing:
            if not buffer.measured:
                # **What this file already holds, asked once** (B7).
                # `durable_length` is declared at construction and never
                # advances, so a store built later in a session's life — which
                # is every store after the persistence row re-activates — is
                # told a boundary that was true before anything was flushed and
                # re-queues the difference. This backend appends, so that is
                # duplicate events in the file.
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
                buffer.measured = True
                last = await anyio.to_thread.run_sync(_last_seq, buffer.path)
                if last is not None:
                    already = max(0, last + 1 - session.durable_length)
                    del buffer.pending[:already]
            records: list[dict[str, Any]] = []
            header_owed = not buffer.header_written
            if header_owed:
                records.append({"type": HEADER_LINE_TYPE, "header": session.header.to_wire()})
            owed = list(buffer.pending)
            records.extend(event.to_wire(thaw=False) for event in owed)
            if not records:
                return
            buffer.pending.clear()
            buffer.header_written = True
            try:
                await anyio.to_thread.run_sync(
                    partial(_append_and_sync, buffer.path, records, fresh=header_owed)
                )
            except BaseException:
                buffer.pending[:0] = owed
                buffer.header_written = not header_owed
                raise

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
        buffer = self._buffers.get(session_id)
        if buffer is not None:
            return buffer.path
        return locate_session(self.root, session_id) or session_path(
            self.root, session_id, session_id
        )

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this log against every other writer for `scope`'s life (I-5)."""
        await claim_session(scope, self.root, session_id)

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
        self._buffers.pop(session_id, None)


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
    """
    payload = "".join(f"{dumps(record)}\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if fresh:
        # Best effort: a filesystem that refuses a directory handle (some
        # networked ones do) has already given us the file's own durability.
        with suppress(OSError):
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


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

    Contrast `read_session`, which is deliberately **strict**: a session is a
    conversation, and silently truncating one at the first unreadable line would
    hand the model a history that is missing its middle. Nothing about JSONL
    decides which rule applies — the *log's* contract does.

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
    return header, events


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
    from ..session import Session
    from .repair import interrupted_turn_closers

    # Through the Protocol, not through this backend's filename: a store that
    # keeps sessions in a database has no path to build, and `resume_session` is
    # the one function every host calls to pick work back up.
    header, events = ctx.require(SESSION_PERSISTENCE).read(session_id)
    closers = interrupted_turn_closers(events)
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
    session.append(
        "session/resumed",
        {
            "events": len(events),
            "interrupted": bool(closers),
            "closed": len(closers),
        },
    )
    return session


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
