"""`SessionPersistence` — what a backend owes, without saying where it writes.

**The split that makes a non-file backend possible** is between "where are the
bytes" and "what can you tell me". A store that keeps sessions in a database has
no per-session path and no directory to list, but it can still answer *does this
exist*, *read it back*, *what is stored*, and *where would a person look* — the
last one honestly returning `None`. So the write side stays as it was (buffered
appends, drained by `session/flush`) and the read side is stated here rather than
inferred from a filename.

**`locate` is allowed to say no**, and a caller must handle that rather than
assume a path. Both shipped backends keep one file per session and answer with it
— which is what lets P5-03's lease work under either — but the Protocol does not
require it: a backend that kept sessions elsewhere answers `None`, a caller that
wants to *show* a path shows nothing, and one that wants to *lock* one declines
loudly instead of inventing a path that protects nothing.

@module ph.persistence.protocol
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..seams.diagnostics import Diagnostic, contribute
from ..session import Session, SessionEvent, SessionHeader
from .lineage import lineage_faults

SURVEY_LIMIT = 500
"""How many stored sessions the lineage check surveys.

Larger than the picker's 50 because this is asking a question about the
*store*, not showing a person a page: a broken chain that fell below the
cut is exactly the one nobody has looked at lately. Bounded all the same,
since the point of answering from the listing is not to walk a store
without limit.
"""

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

    # **No `kind`**, for the reason above. It was added here, filled by one
    # backend, missed by the other, and read by nobody: the picker resolves a
    # segment from `SessionSummary`, off its own header peek. It comes back when
    # a listing consumer needs it — and then through `stored_row`, so it cannot
    # reach one backend and miss the other again.

    # **No `size` and no `title`**, deliberately. `size` meant two different
    # things — bytes on disk from JSONL, an event count from Turso — into one
    # field a picker renders with `filesize.decimal`, and 93% of the Turso
    # listing's cost went to producing the half nobody could interpret. `title`
    # was set by neither backend: a field that is always empty is an affordance
    # that lies, and it would have let a picker migration fall through to hex
    # ids with nothing failing. Both come back when a consumer needs them and
    # can say what they mean.


def stored_row(session_id: str, header: SessionHeader | None, modified: float) -> StoredSession:
    """One listing row from one header peek. **The only place rows are built.**

    Both backends produced this by hand from the same two inputs, and the moment
    a field was added it reached one of them and not the other — silently, since
    the parity suite pinned `session_id` and `cwd` only. A header this build
    cannot parse still gets a row: losing a session from a listing is worse than
    showing it without its details.
    """
    return StoredSession(
        session_id=session_id,
        modified=modified,
        cwd=(header.cwd or "") if header is not None else "",
        parent=header.parent_session if header is not None else None,
    )


@runtime_checkable
class SessionPersistence(Protocol):
    """A place session logs go, and come back from.

    Typed rather than duck-typed, for the reason the seams give for their
    provider Protocols: a backend whose method drifted would otherwise fail at
    runtime inside a caller's `except` and be reported as "no stored sessions".
    """

    def track(self, session: Session) -> None:
        """Start persisting this session. **Queue `events[durable_length:]`.**

        Stated as a slice rather than as "its seed is owed a write", because the
        two stopped meaning the same thing and one backend kept the old reading:
        Turso queued the whole log, so a reference-forked child was written a
        full copy of its prefix, `materialise` saw a first seq of 0 and called
        the file complete, and forking silently stopped being O(1) there with
        every test still green. `durable_length` is what the log holds and this
        store does not — set at construction from a resume's stored length or a
        fork's inherited prefix.
        """
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
        """The full log, **materialised**: dense from seq 0, chain followed.

        A backend whose file stores only its own run must walk `parent_session`
        to assemble the rest — `materialise(self.read_own, session_id)` is that
        walk, and both backends' `read` is exactly that one line.
        """
        ...

    def read_own(
        self, session_id: str, upto: int | None = None, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This one stored log, unchained — the primitive `read` composes.

        `upto` is a hint: events at or above it are not wanted, and returning them anyway
        is slower but not wrong.

        `family` is **not** a hint. Every member of a lineage shares one family directory,
        so the walk knows where an ancestor lives and passing it turns a directory search
        into a path — without it a chained read paid one scan per generation, which scales
        with the size of the store rather than the length of the log.

        Declared here rather than left to convention because it is the half a backend
        actually implements. Without it a third backend can satisfy this Protocol with a
        `read` that returns one file's events, pass mypy, pass `runtime_checkable`, mount
        through `attach` and serve *segments* as whole sessions — surfacing as
        `_readmit`'s "contiguous from 0" refusal three layers from the cause.
        """
        ...

    def stored(self, *, limit: int = 50) -> list[StoredSession]:
        """What is on record, most recently touched first."""
        ...

    def locate(self, session_id: str) -> Path | None:
        """Where a person would find this log, or `None` if it is not a file."""
        ...


def lineage_faults_of(
    store: SessionPersistence, *, limit: int = SURVEY_LIMIT
) -> list[tuple[str, str]]:
    """Which of this store's logs will not materialise, and why.

    A module function rather than a closure inside `attach`, because the answer
    has more than one asker. `stored_survivors` already has a shortfall it
    cannot explain — an unreadable ancestor silently drops a whole subtree from
    its count — and a resume, or a future collector, wants the same question
    before it acts. Trapped inside a `Diagnostic`, the survey was reachable only
    by running `ph doctor`; here the diagnostic is one presentation of it.

    Backend-neutral on purpose, and so not a `SessionPersistence` method: it
    needs only the listing and `exists`, both already on the Protocol, and every
    backend implementing its own walk is what this whole module argues against.
    """
    listed = store.stored(limit=limit)
    return lineage_faults(((one.session_id, one.parent) for one in listed), store.exists)


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

    # **Silent while the store is healthy**, which is what `Diagnostic.read`'s
    # empty-list contract is for: a section on every run is a section nobody
    # reads. Registered here for `attach`'s own reason — one wiring, both
    # backends — and it needs no profile beyond the one that mounted the store,
    # because it answers entirely from `stored()`.
    contribute(
        ctx,
        Diagnostic(
            id="session-lineage",
            title="Session lineage",
            read=partial(lineage_faults_of, store),
            order=20,
        ),
    )
