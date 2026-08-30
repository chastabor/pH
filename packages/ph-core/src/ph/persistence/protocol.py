"""`SessionPersistence` — what a backend owes, without saying where it writes.

Named throughout the plans since D5 and, until P5-08, declared nowhere: there
was one implementation, so every consumer reached for *its* shape instead. Four
of them read `store.root` and rebuilt a filename with `session_path` — the print
mode's `log_path`, the TUI's session picker, the daemon's resume check and its
I-5 lease — which is a JSONL fact four callers deep, and the reason a second
backend could not be added without breaking all four.

**The split that makes a non-file backend possible** is between "where are the
bytes" and "what can you tell me". A store that keeps sessions in a database has
no per-session path and no directory to list, but it can still answer *does this
exist*, *read it back*, *what is stored*, and *where would a person look* — the
last one honestly returning `None`. So the write side stays as it was (buffered
appends, drained by `session/flush`) and the read side is stated here rather
than inferred from a filename.

**`locate` is allowed to say no**, and a caller must handle that rather than
assume a path. Both shipped backends keep one file per session and answer with
it — which is what lets P5-03's lease work under either — but the Protocol does
not require it: a backend that kept sessions elsewhere would answer `None`, a
caller that wants to *show* a path shows nothing, and one that wants to *lock*
one declines loudly instead of inventing a path that protects nothing.

@module ph.persistence.protocol
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..session import Session, SessionEvent, SessionHeader

__all__ = ["SessionPersistence", "StoredSession", "attach"]


@dataclass(frozen=True, slots=True)
class StoredSession:
    """Enough about a stored session to choose it from a list.

    Deliberately not a `Path`: a picker wants an id, a time and a size, and the
    JSONL backend's answers happen to come from a `stat` while another's come
    from a row. `cwd` is what scopes a listing to one repo (P5-14).
    """

    session_id: str
    modified: float
    cwd: str = ""
    parent: str | None = None

    # **No `size` and no `title`**, deliberately. `size` meant two different
    # things — bytes on disk from JSONL, an event count from Turso — into one
    # field a picker renders with `filesize.decimal`, and 93% of the Turso
    # listing's cost went to producing the half nobody could interpret. `title`
    # was set by neither backend: a field that is always empty is an affordance
    # that lies, and it would have let a picker migration fall through to hex
    # ids with nothing failing. Both come back when a consumer needs them and
    # can say what they mean.


@runtime_checkable
class SessionPersistence(Protocol):
    """A place session logs go, and come back from.

    Typed rather than duck-typed, for the reason the seams give for their
    provider Protocols: a backend whose method drifted would otherwise fail at
    runtime inside a caller's `except` and be reported as "no stored sessions".
    """

    def track(self, session: Session) -> None:
        """Start persisting this session; its existing seed is owed a write."""
        ...

    def record(self, session: Session, event: SessionEvent) -> None:
        """Buffer one event. Never blocks — A1 keeps `append` I/O-free."""
        ...

    async def flush(self, session: Session) -> None:
        """Drain whatever is buffered for this session."""
        ...

    def forget(self, session_id: str) -> None:
        """Drop what this backend holds in memory for one session."""
        ...

    def exists(self, session_id: str) -> bool:
        """Whether this backend has a stored log under that id."""
        ...

    def read(self, session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
        """The stored header and events. Raises if there is nothing to read."""
        ...

    def stored(self, *, limit: int = 50) -> list[StoredSession]:
        """What is on record, most recently touched first."""
        ...

    def locate(self, session_id: str) -> Path | None:
        """Where a person would find this log, or `None` if it is not a file."""
        ...


def attach(ctx: Any, store: SessionPersistence) -> None:
    """Wire a store to the session firehose. One subscription list, not two.

    Both backends' `apply` had their own copy of this — the `provide`, the
    catch-up loop and all four `ctx.on` lines — so a new session event or a
    changed catch-up rule was two edits with nothing to fail if only one landed,
    and a backend that missed a subscription fails silently as "no stored
    sessions". That is the failure mode `SessionPersistence`'s own docstring
    gives as the reason for typing the Protocol.
    """
    ctx.provide("session_persistence", store)
    # Catch-up: a row (re)activated after sessions already exist owes them the
    # same buffering a freshly created one gets.
    for session in ctx.sessions.list():
        store.track(session)
    ctx.on("session/created", store.track)
    ctx.on("session/event", store.record)
    ctx.on("session/flush", store.flush)
    ctx.on("session/disposed", lambda session: store.forget(session.id))
