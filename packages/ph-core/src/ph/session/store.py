"""`ctx.sessions` — the live session store, and forking.

Publishes four events: `session/created`, `session/disposed`, `session/event`
(the post-commit append feed) and `session/flush` (the awaited durability
checkpoint, a `parallel` dispatch so every backend runs and the caller waits
for all of them).

Forking is the branching mechanism (D2): pH does not model branching as a
message tree, it models it as `fork(source, boundary)` plus `seed_length`. The
boundary must be a **closed turn** — a fork that ended inside an open turn
would inherit a half-executed step whose tool results never arrive.

@module ph.session.store
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..cordis import Context, events, plugin
from .events import SessionEvent, now_ms
from .session import Session, SessionHeader

__all__ = ["SessionForkError", "SessionStore", "apply", "new_session_id"]

log = logging.getLogger("ph.session")

events.declare(
    "session/created", "emit", owner="ph.session", doc="A session was published into the store."
)
events.declare("session/disposed", "emit", owner="ph.session", doc="A session left the store.")
events.declare(
    "session/event",
    "emit",
    owner="ph.session",
    doc="Post-commit append feed; listener failures are contained per listener.",
)
events.declare(
    "session/flush",
    "parallel",
    owner="ph.session",
    doc="Awaited durability checkpoint: every backend drains and the caller waits.",
)

ForkRejection = Literal[
    "SESSION_NOT_FOUND", "SESSION_ALREADY_EXISTS", "INVALID_BOUNDARY", "OPEN_TURN"
]


class SessionForkError(Exception):
    """A fork was refused, with the reason as a stable code."""

    def __init__(self, message: str, code: ForkRejection) -> None:
        super().__init__(message)
        self.code = code


def new_session_id() -> str:
    """A sortable, human-legible session id."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


@dataclass(slots=True)
class _Entry:
    session: Session
    unobserve: Any


@dataclass(slots=True)
class SessionStore:
    """The service published as `ctx.sessions`."""

    ctx: Context
    _entries: dict[str, _Entry] = field(default_factory=dict)

    # -------------------------------------------------------------- lifecycle --

    def create(
        self,
        session_id: str | None = None,
        *,
        seed: list[SessionEvent] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Session:
        """Publish a new live session, optionally seeded with existing events."""
        resolved = session_id or new_session_id()
        if resolved in self._entries:
            raise SessionForkError(f'session "{resolved}" already exists', "SESSION_ALREADY_EXISTS")
        fields: dict[str, Any] = {"id": resolved, "createdAt": now_ms()}
        fields.update(meta or {})
        header = SessionHeader.model_validate(fields)
        return self._publish(Session(resolved, seed=seed, header=header))

    def adopt(self, session: Session) -> Session:
        """Publish an already-constructed session (the resume path)."""
        if session.id in self._entries:
            raise SessionForkError(
                f'session "{session.id}" already exists', "SESSION_ALREADY_EXISTS"
            )
        return self._publish(session)

    def _publish(self, session: Session) -> Session:
        def observer(source: Session, event: SessionEvent) -> None:
            # Contained: one failing listener neither aborts the dispatch nor
            # un-appends a committed event (A1).
            self.ctx.emit("session/event", source, event, contained=True)

        self._entries[session.id] = _Entry(session=session, unobserve=session.observe(observer))
        self.ctx.emit("session/created", session)
        return session

    def get(self, session_id: str) -> Session | None:
        entry = self._entries.get(session_id)
        return entry.session if entry is not None else None

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise SessionForkError(f'session "{session_id}" not found', "SESSION_NOT_FOUND")
        return session

    def list(self) -> list[Session]:
        return [entry.session for entry in self._entries.values()]

    def dispose(self, session_id: str) -> None:
        entry = self._entries.pop(session_id, None)
        if entry is None:
            return
        entry.unobserve()
        self.ctx.emit("session/disposed", entry.session)

    async def flush(self, session: Session) -> None:
        """Await every persistence backend's drain for one session."""
        await self.ctx.parallel("session/flush", session)

    # ------------------------------------------------------------------ fork --

    def fork(
        self,
        source: Session | str,
        boundary: int | None = None,
        child_session_id: str | None = None,
    ) -> Session:
        """Create a live child session from a stable prefix of a live source.

        `boundary` is an inclusive source event seq; omitted means the source's
        current last event. The slice may end on a between-turn event but must
        not end inside an open turn.
        """
        if child_session_id is not None and child_session_id in self._entries:
            raise SessionForkError(
                f'session "{child_session_id}" already exists', "SESSION_ALREADY_EXISTS"
            )
        live = self._resolve_source(source)
        seed = self._fork_seed(live, boundary)
        meta: dict[str, Any] = {"parentSession": live.id, "seedLength": len(seed)}
        if live.header.cwd is not None:
            meta["cwd"] = live.header.cwd
        return self.create(child_session_id, seed=list(seed), meta=meta)

    def _resolve_source(self, source: Session | str) -> Session:
        if isinstance(source, str):
            return self.require(source)
        live = self.get(source.id)
        if live is None:
            raise SessionForkError(f'session "{source.id}" not found', "SESSION_NOT_FOUND")
        if live is not source:
            raise SessionForkError(
                f'session "{source.id}" is not the live store instance', "SESSION_NOT_FOUND"
            )
        return source

    @staticmethod
    def _fork_seed(session: Session, requested_boundary: int | None) -> tuple[SessionEvent, ...]:
        log = session.events
        if requested_boundary is None:
            if not log:
                return ()
            boundary = log[-1].seq
        else:
            boundary = requested_boundary
        if boundary < 0:
            raise SessionForkError(
                f'fork boundary for session "{session.id}" must be a non-negative '
                f"integer, got {boundary}",
                "INVALID_BOUNDARY",
            )
        if boundary >= len(log):
            last = log[-1].seq if log else None
            raise SessionForkError(
                f'fork boundary {boundary} does not exist in session "{session.id}" '
                f"(last seq: {last if last is not None else 'none'})",
                "INVALID_BOUNDARY",
            )
        if log[boundary].seq != boundary:
            raise SessionForkError(
                f"fork boundary {boundary} does not match a contiguous event seq in "
                f'session "{session.id}"',
                "INVALID_BOUNDARY",
            )
        last_turn_boundary: SessionEvent | None = None
        for event in log[: boundary + 1]:
            if event.type in ("turn/start", "turn/end"):
                last_turn_boundary = event
        if last_turn_boundary is not None and last_turn_boundary.type == "turn/start":
            turn = last_turn_boundary.data.get("turn")
            raise SessionForkError(
                f'fork boundary {boundary} in session "{session.id}" ends inside open turn {turn}',
                "OPEN_TURN",
            )
        return tuple(log[: boundary + 1])


@plugin("session")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the session store."""
    ctx.provide("sessions", SessionStore(ctx=ctx))
