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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..paths import resolve_roots
from ..session import Session, SessionEvent, SessionHeader
from ..session.json import dumps

__all__ = ["JsonlSessionStore", "apply", "read_session", "session_path"]

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
        """Start buffering a session; its existing log is queued for the first flush.

        The seed of a forked or resumed session is already in the log, so it is
        written on the first flush like any other event.
        """
        if session.id in self._buffers:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        path = session_path(self.root, session.id)
        buffer = _Buffer(path=path, header_written=path.exists())
        if not buffer.header_written:
            buffer.pending.extend(session.events)
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

    def forget(self, session_id: str) -> None:
        self._buffers.pop(session_id, None)


def _append_and_sync(path: Path, records: list[dict[str, Any]]) -> None:
    payload = "".join(f"{dumps(record)}\n" for record in records)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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


@plugin("session-persistence-jsonl", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the JSONL backend and wire it to the session firehose."""
    root_setting = config.get("root") if isinstance(config, dict) else None
    root = Path(root_setting) if root_setting else resolve_roots().sessions_dir()
    store = JsonlSessionStore(ctx=ctx, root=root)
    ctx.provide("session_persistence", store)

    # Catch up: a row (re)activated after sessions already exist owes them the
    # same buffering a freshly created one gets.
    for session in ctx.sessions.list():
        store.track(session)
    ctx.on("session/created", store.track)
    ctx.on("session/event", store.record)
    ctx.on("session/flush", store.flush)
    ctx.on("session/disposed", lambda session: store.forget(session.id))
