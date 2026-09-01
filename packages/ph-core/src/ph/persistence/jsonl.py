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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from pydantic import ValidationError

from ..cordis import Context, plugin
from ..paths import resolve_roots
from ..session import Session, SessionEvent, SessionHeader
from ..session.json import dumps
from .lineage import materialise
from .protocol import SessionPersistence, StoredSession, attach

__all__ = [
    "JsonlSessionStore",
    "apply",
    "read_records",
    "read_session",
    "resumption_of",
    "session_path",
]


def resumption_of(session: Any) -> dict[str, Any] | None:
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


def session_path(root: Path, session_id: str) -> Path:
    return root / f"{session_id}.jsonl"


@dataclass(slots=True)
class _Buffer:
    path: Path
    pending: list[SessionEvent] = field(default_factory=list)
    header_written: bool = False


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
        """
        if session.id in self._buffers:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = session_path(self.root, session.id)
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
        buffer = self._buffers.get(session.id)
        if buffer is None:
            return
        records: list[dict[str, Any]] = []
        if not buffer.header_written:
            records.append({"type": HEADER_LINE_TYPE, "header": session.header.to_wire()})
        records.extend(event.to_wire(thaw=False) for event in buffer.pending)
        if not records:
            return
        buffer.pending.clear()
        buffer.header_written = True
        await anyio.to_thread.run_sync(_append_and_sync, buffer.path, records)

    # ------------------------------------------------------------- reading --
    #
    # The four questions a consumer used to answer by reaching for `self.root`
    # and rebuilding a filename. A backend with no per-session file answers all
    # four; a backend with one answers them from the filesystem, as here.

    def exists(self, session_id: str) -> bool:
        return session_path(self.root, session_id).is_file()

    def read(self, session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
        """The session's full log, following its lineage when it stores a reference.

        `read_own` is this backend's one-file read; `materialise` decides whether
        a chain is owed by looking at the first event's seq. A log that starts at
        0 is complete and is returned unchanged, which is every log written so
        far — so this is a no-op until something writes a reference-fork.
        """
        return materialise(self.read_own, session_id)

    def read_own(self, session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
        """This file and nothing else — the unchained read `materialise` walks with."""
        return read_session(session_path(self.root, session_id))

    def locate(self, session_id: str) -> Path | None:
        """This backend writes files, so it can always say where."""
        return session_path(self.root, session_id)

    def stored(self, *, limit: int = 50) -> list[StoredSession]:
        """What is on record, most recently touched first.

        One `stat` per entry — the directory scan's own — and one short read for
        the header, which is the first line. **No title**: deriving one means
        scanning forward for a `user/message` and joining its content blocks,
        and the join (`text_of_wire`) lives in the front end that wants it. The
        TUI's picker keeps its richer summary; this is the part every backend
        can answer, which is what the Protocol is for.
        """
        try:
            with os.scandir(self.root) as entries:
                found = [
                    (entry, entry.stat())
                    for entry in entries
                    if entry.name.endswith(".jsonl") and entry.is_file()
                ]
        except OSError:
            return []
        found.sort(key=lambda pair: pair[1].st_mtime, reverse=True)
        listed: list[StoredSession] = []
        for entry, stat in found[:limit]:
            header = _peek_header(Path(entry.path))
            listed.append(
                StoredSession(
                    session_id=Path(entry.path).stem,
                    modified=stat.st_mtime,
                    cwd=(header.cwd or "") if header is not None else "",
                    parent=header.parent_session if header is not None else None,
                )
            )
        return listed

    def forget(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)


def _append_and_sync(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(f"{dumps(record)}\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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


def read_session(path: Path) -> tuple[SessionHeader, list[SessionEvent]]:
    """Read a stored session back, validating every envelope.

    Returns the raw header and events; acceptance — the known-types refusal,
    the header-id match, the surface rules — happens when they seed a
    `Session`, on the one path every seed takes.
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
            events.append(SessionEvent.from_wire(record))
    if header is None:
        raise ValueError(f"{path}: no session header line")
    return header, events


async def resume_session(ctx: Any, session_id: str) -> Any:
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
    header, events = ctx.session_persistence.read(session_id)
    closers = interrupted_turn_closers(events)
    revived = Session(session_id, seed=[*events, *closers], header=header, durable=len(events))
    # `durable=len(events)`: **what the store already holds is `events`, and
    # nothing else.** The closers
    # are synthesized here and the constructor appends `session/end-seed` on top;
    # both are in the log and neither has been written. A backend that inferred
    # durability from "the file exists" dropped them and left a gap in the seq
    # space, which `_readmit` refuses — so the session resumed once and never
    # again. Said here because this is the only place that knows the difference.
    session = ctx.sessions.adopt(revived)
    # Recorded, not just returned. A resume is a fact about *provenance* — this
    # process picked up work somebody else started — and it is not derivable
    # from anything else in the log: a session that was reopened and one that
    # ran straight through look identical afterwards. It matters most where
    # nobody is watching, which is the daemon and a cron-started agent, and it
    # is what lets `ph doctor`, a trajectory reader or a person scrolling back
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


@plugin("session-persistence-jsonl", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the JSONL backend and wire it to the session firehose."""
    root_setting = config.get("root") if isinstance(config, dict) else None
    root = Path(root_setting) if root_setting else resolve_roots().sessions_dir()
    # Annotated, so mypy checks this backend against the Protocol *with
    # signatures* — which the runtime `isinstance` gate cannot: a
    # `runtime_checkable` Protocol compares names only.
    store: SessionPersistence = JsonlSessionStore(ctx=ctx, root=root)
    attach(ctx, store)
