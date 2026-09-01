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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..cordis import Context, Disposer, events, plugin
from .events import SessionEvent, now_ms
from .session import Session, SessionHeader, SessionKind

__all__ = [
    "SessionForkError",
    "SessionStore",
    "apply",
    "fork_boundaries",
    "is_fork_boundary",
    "new_session_id",
    "open_turn_at",
]

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
    unobserve: Disposer
    """The observer's teardown, as the type it already is.

    It was `Any`, which is the one annotation that names nothing — so
    `test_registration_ownership.py`'s walk over row-supplied callables could
    not see it and nothing forced anyone to say whether it runs inside a
    binding. `Session.observe` returns `Callable[[], None]`; naming it puts it
    in front of the gate, which files it as teardown."""


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
        inherited: int = 0,
    ) -> Session:
        """Publish a new live session, optionally seeded with existing events.

        `inherited` is how many leading events are **already durable in some
        other log** — a fork's prefix, which lives in the parent's file. It is
        the storage half of a fork and it is not `seed_length`: that one is the
        *provenance* boundary, the line five folds read to tell parent history
        from this session's own work, and it stays the fork boundary whatever a
        store happens to hold.

        Passed to the constructor rather than assigned after it, because
        `session/created` is what makes a backend queue the log: a store that saw
        the whole seed would write out the copy reference-forking exists to
        avoid. As a keyword that ordering cannot be got wrong.
        """
        resolved = session_id or new_session_id()
        if resolved in self._entries:
            raise SessionForkError(f'session "{resolved}" already exists', "SESSION_ALREADY_EXISTS")
        fields: dict[str, Any] = {"id": resolved, "createdAt": now_ms()}
        fields.update(meta or {})
        # **Inheritance at the one construction gate**, so a child shares its
        # parent's directory whatever made it. `_branch` was enforcing this and
        # the other caller that creates a child — the subagent spawn — was not,
        # so every subagent got a family of its own and the layout's central
        # claim, one conversation and everything it spawned in one directory,
        # was false for the thing that spawns most. A root needs no line here:
        # `SessionHeader` defaults an absent family to the session's own id.
        parent_id = fields.get("parentSession")
        parent = self.get(str(parent_id)) if parent_id else None
        if parent is not None:
            fields.setdefault("family", parent.header.family)
        header = SessionHeader.model_validate(fields)
        return self._publish(Session(resolved, seed=seed, header=header, durable=inherited))

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
        """Await every backend's drain for this session — **ancestors first**.

        A reference-forked child's log names a parent and a count, and is
        readable only once that parent's log actually holds those events. A child
        made durable first and then orphaned by a crash is a log nothing can
        read, while the reverse order is harmless at every point it can be
        interrupted — and this is the common case, not a corner: a fork's
        boundary is usually the parent's live tip, exactly the part not yet
        written.

        **Ordered here rather than in `attach`**, where it started. This is the
        sole dispatcher of `session/flush` in the repo, it owns `_entries` — the
        live graph the walk needs — and it owns `fork` and `roll`, the calls that
        create the debt being settled. Ordering inside one backend's listener
        instead meant the ancestors drained *outside* the dispatch, so a second
        listener (a mirror, an exporter) saw only the child and had to
        re-implement the walk; it ran once per attached backend; and the
        guarantee belonged to having gone through `attach` rather than to the
        Protocol, which is why the backend parity suite could not reach it.

        The honest trade: one `flush(child)` is now N dispatches rather than one.
        Both backends return early on an empty buffer, so the ancestors cost a
        dict lookup each.
        """
        for ancestor in self._lineage(session):
            await self.ctx.parallel("session/flush", ancestor)

    def _lineage(self, session: Session) -> tuple[Session, ...]:
        """`session` and every ancestor **live in this process**, oldest first.

        An ancestor this store does not hold is one nothing here is writing, so
        its log is already whatever it is going to be. Keyed by id so the dict
        is both the order and the cycle bound — a hand-edited header naming its
        own descendant costs a lookup rather than a hang.
        """
        # A tuple, and `chain` keyed by id: this class has its own `list` method,
        # which shadows the builtin inside the class body — `list(...)` here
        # resolves to it and mypy says so in a sentence nobody enjoys reading.
        chain: dict[str, Session] = {}
        current: Session | None = session
        while current is not None and current.id not in chain:
            chain[current.id] = current
            parent = current.header.parent_session
            current = self.get(parent) if parent is not None else None
        return tuple(reversed(chain.values()))

    # ------------------------------------------------------------------ fork --

    def fork(
        self,
        source: Session | str,
        boundary: int | None = None,
        child_session_id: str | None = None,
    ) -> Session:
        """Create a live child session from a stable prefix of a live source.

        `boundary` is an inclusive source event seq; omitted means the source's current
        last event. The slice may end on a between-turn event but must not end inside an
        open turn.

        **The child holds the prefix in memory and does not re-store it.** Every reader —
        the transcript fold, the surface fold, `derive_messages` — sees a whole session
        exactly as before. What changes is the disk: the child's file begins at
        `session/end-seed`, at seq `seed_length`, and its header says which log the rest
        came from.

        **The prefix a child depends on is immutable**, because the log is append-only, so
        the parent may run on forever without invalidating a descendant. This needs no
        lock, no copy-on-write and no invalidation.

        A fork is cheap on disk and **not** free in memory: `_readmit` re-freezes every
        seeded event. The seed is still handed over in full because every reader
        downstream wants a whole session, and a trusted-seed path would be a change to
        the session model rather than to `fork`.
        """
        if child_session_id is not None and child_session_id in self._entries:
            raise SessionForkError(
                f'session "{child_session_id}" already exists', "SESSION_ALREADY_EXISTS"
            )
        return self._branch(source, boundary, child_session_id, kind="fork")

    def _branch(
        self,
        source: Session | str,
        boundary: int | None,
        child_session_id: str | None,
        *,
        kind: SessionKind,
    ) -> Session:
        """The shared body of `fork` and `roll`; `kind` is the only difference.

        Private because `kind` is not a choice a caller makes — it is which of the two
        operations they called. Exposing it on `fork` would invite a third answer, and
        there are only two.
        """
        live = self._resolve_source(source)
        seed = self._fork_seed(live, boundary)
        meta: dict[str, Any] = {
            "parentSession": live.id,
            "seedLength": len(seed),
            "kind": kind,
        }
        if live.header.cwd is not None:
            meta["cwd"] = live.header.cwd
        return self.create(child_session_id, seed=list(seed), meta=meta, inherited=len(seed))

    def roll(self, source: Session | str, child_session_id: str | None = None) -> Session:
        """Continue a session in a fresh log — segmentation (§7 step 6).

        **A fork at the tip with no divergence**, and that is the whole of it:
        `fork(source, None)` already means "at the last event", so segmenting
        costs a marker and a name rather than a second code path. It inherits
        `_fork_seed`'s refusal too, which is the rule you want — a segment
        boundary inside an open turn would start a log mid-step.

        **The session id changes**, which was the decision taken and is the one
        cost. Worth stating what it does *not* break: `/revert` resolves a
        checkpoint by the tree hash carried in the `workspace/checkpoint` event,
        not by looking a ref up under the current id, and the ref that keeps that
        tree alive still stands under the old one. So a revert reaches back
        across a roll. What does move is anything keyed by id — a client holding
        the old one is holding a log that has stopped.

        `session/segmented` is appended to the **parent**, after the fork, so it
        is the parent's terminal record rather than something the child inherits.
        That gives the link both directions from state that already exists: the
        parent names its continuation, the child's header names its origin.
        Without it a segment and a branch are indistinguishable, because
        structurally they are the same thing.

        **The parent is left live**, deliberately. Disposing it would unhook the
        persistence observer while the `Session` object stayed perfectly usable,
        so an append through a stale reference would vanish with nothing raised.
        Leaving it published means an in-flight append still lands — and a caller
        that goes on writing to it has made a *branch*, which is a legitimate
        thing to have done and shows up as one, its events sitting after a marker
        that says where the other side went.

        Synchronous, like `fork`. Durability arrives with the next flush, which
        `attach` orders ancestors-first, so the parent's marker cannot reach disk
        after the child that references it.
        """
        live = self._resolve_source(source)
        child = self._branch(live, None, child_session_id, kind="segment")
        live.append("session/segmented", {"continues": child.id})
        return child

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
        """The prefix a fork copies, or the reason it may not (A6)."""
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
        open_turn = open_turn_at(log, boundary)
        if open_turn is not None:
            raise SessionForkError(
                f'fork boundary {boundary} in session "{session.id}" ends inside '
                f"open turn {open_turn.data.get('turn')}",
                "OPEN_TURN",
            )
        return tuple(log[: boundary + 1])


def open_turn_at(log: Sequence[SessionEvent], boundary: int) -> SessionEvent | None:
    """The `turn/start` a boundary would cut into, or `None` if it is safe (A6).

    The rule, in one place: a fork may end on any between-turn event, and may
    not end inside a turn that has not closed. Exported because a *reader* needs
    the same answer before it offers the action — the trajectory view marks
    which records a fork may aim at, and a second statement of this rule there
    said `turn/end` only, refusing three legal boundaries out of four and citing
    A6 while doing it.
    """
    if boundary < 0 or boundary >= len(log):
        return None
    last: SessionEvent | None = None
    for event in log[: boundary + 1]:
        if event.type in ("turn/start", "turn/end"):
            last = event
    return last if last is not None and last.type == "turn/start" else None


def is_fork_boundary(log: Sequence[SessionEvent], boundary: int) -> bool:
    """Whether `ctx.sessions.fork` would accept this boundary."""
    return 0 <= boundary < len(log) and open_turn_at(log, boundary) is None


def fork_boundaries(log: Sequence[SessionEvent]) -> set[int]:
    """Every seq `fork` would accept, from one pass over the log (A6).

    The same rule as `open_turn_at`, answered for the whole log at once, because a
    reader that marks *which records* a fork may aim at needs it for every one of them
    — and asking per record rescans the prefix each time, which is quadratic and
    freezes a UI that opens the trajectory from a running chat (P4-17).

    Keyed by seq and admitted only where the seq is the event's own index, which is the
    contiguity `_fork_seed` requires anyway — so this marks nothing the fork would then
    refuse.
    """
    boundaries: set[int] = set()
    open_turn = False
    for index, event in enumerate(log):
        if event.type == "turn/start":
            open_turn = True
        elif event.type == "turn/end":
            open_turn = False
        if not open_turn and event.seq == index:
            boundaries.add(event.seq)
    return boundaries


@plugin("session")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the session store."""
    ctx.provide("sessions", SessionStore(ctx=ctx))
