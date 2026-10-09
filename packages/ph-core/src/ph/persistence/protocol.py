"""`SessionPersistence` — what a backend owes, without saying where it writes.

**The split that makes a non-file backend possible** is between "where are the
bytes" and "what can you tell me". A store that keeps sessions in a database has
no per-session path and no directory to list, but it can still answer *does this
exist*, *read it back*, *what is stored*, and *where would a person look* — the
last one honestly returning `None`. So the write side stays as it was (the log
itself is the queue, written on `session/flush`) and the read side is stated here
rather than inferred from a filename.

**`locate` is allowed to say no**, and a caller must handle that rather than
assume a path. Both shipped backends keep one file per session and answer with it
— which is what lets P5-03's lease work under either — but the Protocol does not
require it: a backend that kept sessions elsewhere answers `None`, a caller that
wants to *show* a path shows nothing, and one that wants to *lock* one declines
loudly instead of inventing a path that protects nothing.

@module ph.persistence.protocol
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from ..keys import SESSION_PERSISTENCE, SESSIONS
from ..seams.diagnostics import Diagnostic, contribute
from ..session import Session, SessionEvent, SessionHeader
from .families import descendants
from .lineage import lineage_faults

if TYPE_CHECKING:
    from ..cordis import Context

log = logging.getLogger("ph.persistence")

_OWED: WeakKeyDictionary[Context, set[int]] = WeakKeyDictionary()
"""The stores, by `id`, whose last write each scope still owes (`write_on_unwind`)."""

SURVEY_LIMIT = 500
"""How many stored sessions the lineage check surveys.

Larger than the picker's 50 because this is asking a question about the
*store*, not showing a person a page: a broken chain that fell below the
cut is exactly the one nobody has looked at lately. Bounded all the same,
since the point of answering from the listing is not to walk a store
without limit.
"""

__all__ = [
    "ClaimingStore",
    "NoStoredSession",
    "SessionArchive",
    "SessionPersistence",
    "StoredSession",
    "attach",
    "gated",
    "read_if_stored",
    "write_on_unwind",
]


class NoStoredSession(FileNotFoundError):
    """Nothing is stored under an id: how `read_own`, and so `read`, says so.

    Its own type because whether a session is resumed or created turns on it
    (`open_session`, through `read_if_stored`). Any other error reading a log that
    is there, a file vanishing mid-read among them, is damage to report rather than
    an absence to create a session over.
    """


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
    """The log this one continues or was spawned from — `SessionHeader.parent_session`,
    what a lineage check follows (`lineage_faults`)."""
    delegating_parent: str | None = None
    """The agent that spawned this one, or `None` for a session a person started —
    `SessionHeader.delegating_parent`. Not `parent`: a fork names the log it was cut
    from there, and is a session of its own, so a family walk over a listing
    (`workspace.stored_survivors`) follows this link and not that one."""
    family: str = ""
    """The directory this session's log lives in — its lineage's root id.

    A listing consumer finally needed it, which is the bar the notes below set
    for adding a field. `read_own` takes `family` and says it is *not* a hint:
    with it a log is a path, without it a directory search. A fold over every
    stored session — `phern attachments gc` — read one log per row and paid that
    search on each, so listing 4 000 sessions across 1 000 families cost a scan
    per session on top of the listing's own.

    Empty when the header would not parse, and that is the safe direction: the
    reader falls back to searching, which is what it did for every row before."""

    # **No `kind`**, for the reason above. It was added here, filled by one
    # backend, missed by the other, and read by nobody: the picker resolves a
    # segment from `SessionSummary`, off its own header peek. It comes back when
    # a listing consumer needs it — and then through `stored_row`, so it cannot
    # reach one backend and miss the other again.

    # **No `size` and no `title`**, deliberately. `size` meant two different
    # things — bytes on disk from JSONL, an event count from Turso — in one field
    # a picker renders with `filesize.decimal`, and producing the half nobody
    # could interpret was most of the Turso listing's cost. `title`
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
        delegating_parent=header.delegating_parent if header is not None else None,
        # No `or ""` guard, unlike `cwd` above: `SessionHeader.family` is a
        # `str` with `min_length=1` whose docstring says it is never absent, so
        # the idiom copied from the nullable line next to it would suggest this
        # field can be blank when a header parsed. The `else` arm is the one real
        # source of `""`, and `StoredSession.family` says so.
        family=header.family if header is not None else "",
    )


def descendants_among(parent_id: str, rows: Iterable[StoredSession]) -> tuple[StoredSession, ...]:
    """The rows beneath `parent_id` — its children, theirs, and so on — ordered by id.
    **The header decides.**

    Each backend narrows its candidates by the family directory and the id prefix
    (`families.children_under`), and neither is proof: a fork can be named after its
    source and filed in its family. So a row is beneath `parent_id` only when its
    `delegating_parent`, followed up through the other candidates, reaches it. That
    is the header's `origin: "subagent"` link, and a fork does not have one: its
    `parent` names the log it was cut from. Each row's own `delegating_parent` says
    which level it is.

    Stated here and not once per backend for `stored_row`'s reason: two backends
    that each wrote their own filter would drift, and nothing would fail. Ordered by
    id so both backends give the same answer.

    A row whose header would not parse has no `delegating_parent`, so it is not
    listed, and neither is anything beneath it: a log that cannot say whose child it
    is cannot be resumed as one. A chain that loops never reaches `parent_id`.
    """
    candidates = {row.session_id: row for row in rows}
    lineage = [(one.session_id, one.delegating_parent) for one in candidates.values()]
    beneath = descendants([(parent_id, None), *lineage], parent_id)[1:]
    return tuple(sorted((candidates[one] for one in beneath), key=lambda row: row.session_id))


@runtime_checkable
class SessionArchive(Protocol):
    """A session store's listing and its two reads — what a fold needs, and no more.

    Split out the way `AgentHandle` is split from `AgentDriver`, and for the same
    reason: a fold that only *reads* has no business holding `track`, `flush` or
    `forget`, and a test standing in for one should not implement seven methods it
    never calls. `SessionPersistence` extends this, so each signature is written
    once and a real backend satisfies both.

    `read` and `read_own` are not interchangeable, and the folds pick deliberately:
    `phern attachments gc` wants `read_own`, because a chained read fails when an
    ancestor is missing and would refuse a collection that is safe to make;
    `stored_survivors` wants `read`, because a tree is only accounted for by the
    whole lineage that built it.

    **Every method blocks**, on a file read or a search of the store: call it off
    the event loop.
    """

    def read_own(
        self,
        session_id: str,
        upto: int | None = None,
        family: str | None = None,
        *,
        types: frozenset[str] | None = None,
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This one stored log, unchained — the primitive `read` composes.

        `upto` is a hint: events at or above it are not wanted, and returning them anyway
        is slower but not wrong.

        `types` keeps only the events of those types, for a reader that folds a few
        records out of a long log — a child's state out of a log that is mostly
        `assistant/chunk` (`ph.seams.subagents`). A hint too, and the saving is the
        parse: the others are dropped before they are validated. A torn batch at the
        tail is still dropped whole, judged on every record, not on the kept ones.

        `family` is **not** a hint. Every member of a lineage shares one family directory,
        so the walk knows where an ancestor lives and passing it turns a directory search
        into a path — without it a chained read paid one scan per generation, which scales
        with the size of the store rather than the length of the log.

        Raises `NoStoredSession` when, and only when, nothing is stored under that id
        (in `family`, when given).

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

    def holding(
        self, types: frozenset[str], *, gate: str
    ) -> Iterator[tuple[str, list[SessionEvent]]]:
        """Every stored log, by id, with its own records of `types` — for a reader that
        needs a few records out of every log, such as a daemon rebuilding its schedule
        index (S18).

        A log with no `gate` record yields none: `gate` is the record the rest mean
        nothing without, a schedule's creation for its ticks and its cancel. One that
        cannot be read is logged and left out, so a reader can tell "holds none" from
        "could not say". **On the Protocol, not a walk over `read_own`**, because
        every log is read and each backend has a cheaper way than parsing them all:
        JSONL searches the bytes for `gate` first, Turso reads each database alone.
        """
        ...

    def descendants_of(self, parent_id: str, family: str) -> tuple[StoredSession, ...]:
        """Every stored session beneath `parent_id` — its children, theirs, and so on —
        ordered by id. Each row's `delegating_parent` says which level it is.

        **The whole tree, in one read.** Every descendant is filed in the root's
        family and named under its id, so the scan that finds a parent's children
        finds all of theirs with them; listing one level at a time would read the
        same directory once per level and leave a restart blind below the first.

        **Every one, not a page.** `stored` checks every log in the store and
        then cuts at `limit`, so a parent's children fall below the cut once the
        store is big enough, even at `SURVEY_LIMIT`. A picker can live with
        that. A resume ladder, a budget or a spawn cap that missed a child
        would be wrong. So this stays inside the parent's family, where every
        child is filed, and checks only the names that start with the parent's
        id (`families.children_under`). Its cost follows the family, not the store.

        `family` is required for the reason `read_own` gives for its own: it is
        what makes this one directory rather than all of them. A live parent
        has it on `session.header.family`, a stored one on `StoredSession.family`.
        Each row carries it too, so reading a child after this is
        `read_own(row.session_id, family=row.family)`, a path and not a search.

        Only logs that reached disk are listed. A child's header is written at
        its first flush, so one that is live and has never flushed is known
        only to the process running it. The header decides who is beneath the
        parent (`descendants_among`), so a fork is left out.
        """
        ...

    def read(
        self, session_id: str, *, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """The full log, **materialized**: dense from seq 0, chain followed.

        A backend whose file stores only its own run must walk `parent_session`
        to assemble the rest — `materialize(self.read_own, session_id, family=…)`
        is that walk, and both backends' `read` is exactly that one line.
        `family` locates the first file, as `read_own`'s does; a listing row carries it,
        and without it the store is searched. An ancestor that cannot be read is a
        `LineageError`, never `NoStoredSession`: only this id's own log is an absence.
        """
        ...


@runtime_checkable
class SessionPersistence(SessionArchive, Protocol):
    """A place session logs go, and come back from.

    Typed rather than duck-typed, for the reason the seams give for their
    provider Protocols: a backend whose method drifted would otherwise fail at
    runtime inside a caller's `except` and be reported as "no stored sessions".
    """

    def track(self, session: Session) -> None:
        """Start persisting this session. **Owe what you do not already hold.**

        `session.durable_length` is a *floor* the caller declares at
        construction — a resume's stored length, a fork's inherited prefix — and
        it never advances, so a store built later in a session's life is told a
        boundary that was true when the session was built (B7). A backend that
        can measure what its own medium holds must take the larger of the two;
        one whose write is idempotent by seq may ignore the question. See
        `JsonlSessionStore.flush` and `TursoSessionStore.track`.

        Stated as what is owed rather than as "its seed is owed a write",
        because the two stopped meaning the same thing and one backend kept the old reading:
        Turso queued the whole log, so a reference-forked child was written a
        full copy of its prefix, `materialize` saw a first seq of 0 and called
        the file complete, and forking silently stopped being O(1) there with
        every test still green. `durable_length` is what the log holds and this
        store does not — set at construction from a resume's stored length or a
        fork's inherited prefix.
        """
        ...

    async def flush(self, session: Session) -> None:
        """Write whatever this log holds that the backend does not.

        **Read off the log, never drained from a queue** — which is why there is
        no per-event hook here. A backend that buffered from the `session/event`
        firehose would miss whatever is appended after its listener unwinds, and
        teardown is when that happens (F2). A session it was never told about is
        tracked here and written.

        Callable after the row that mounted the backend has unwound, which is
        how a mount's last write reaches the events its own teardown appended —
        see `write_on_unwind`.
        """
        ...

    def forget(self, session_id: str) -> None:
        """Drop what this backend holds in memory for one session."""
        ...

    def exists(self, session_id: str) -> bool:
        """Whether this backend has a stored log under that id.

        A search of the store, since an id alone does not say where a log is filed:
        blocking, so off the event loop. To read the log when there is one,
        `read_if_stored` searches once.
        """
        ...

    def locate(self, session_id: str) -> Path | None:
        """Where a person would find this log, or `None` if it is not a file.

        A search of the store, as `exists` is: off the event loop.
        """
        ...

    def directory(self) -> Path | None:
        """Where this store keeps every session, or `None` if not on a filesystem.

        The picker's question, and a store-level one: a log lives at
        `<directory>/<family>/<id>.jsonl`, and a picker that only knew one log's
        path had to walk up two levels to find the rest — which was a rule about
        this backend's layout written into a front end.
        """
        ...


@runtime_checkable
class ClaimingStore(Protocol):
    """A backend that can hold one session against every other writer (I-5).

    **Optional, and its own Protocol rather than a `locate() is not None` probe.**
    The lease used to be the daemon's: it asked the store for a path and locked
    beside it, so only daemons were refused and `phern -p --session x` appended to
    a log a daemon held — or to one another `phern -p` had just written, which was
    enough on its own to make the session unopenable. The writer is the store,
    so the claim is the store's, and every host reaches it through one
    `open_session`.

    A backend with no per-session file does not implement this, and a host
    finding no `ClaimingStore` says so out loud rather than locking a path that
    protects nothing. Both shipped backends keep one file per session and
    implement it through `lease.claim_session`.

    `scope` is **required**: it is the lifetime that holds the lock, and a lease
    with a defaulted owner is one nobody remembers to release.
    """

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this session for `scope`'s life, or raise `SessionBusy` — and
        write every live log before letting go (`write_on_unwind`)."""
        ...


def gated(events: list[SessionEvent], gate: str) -> list[SessionEvent]:
    """`events`, or none of them when none is a `gate` record — the rule
    `SessionArchive.holding` states, kept in one place for every backend that
    filters after reading."""
    return events if any(event.type == gate for event in events) else []


def read_if_stored(
    archive: SessionArchive, session_id: str, *, family: str | None = None
) -> tuple[SessionHeader, list[SessionEvent]] | None:
    """`archive.read`, or `None` when nothing is stored under that id (`NoStoredSession`).

    One search of the store, where asking `exists` and then reading searched it
    twice. A broken chain is never taken for an absence: an ancestor that cannot be
    read is a `LineageError`. Blocking, as the read is.
    """
    try:
        return archive.read(session_id, family=family)
    except NoStoredSession:
        return None


def lineage_faults_of(
    store: SessionPersistence, *, limit: int = SURVEY_LIMIT
) -> list[tuple[str, str]]:
    """Which of this store's logs will not materialize, and why.

    A module function rather than a closure inside `attach`, because the answer
    has more than one asker. `stored_survivors` already has a shortfall it
    cannot explain — an unreadable ancestor silently drops a whole subtree from
    its count — and a resume, or a future collector, wants the same question
    before it acts. Trapped inside a `Diagnostic`, the survey was reachable only
    by running `phern doctor`; here the diagnostic is one presentation of it.

    Backend-neutral on purpose, and so not a `SessionPersistence` method: it
    needs only the listing and `exists`, both already on the Protocol, and every
    backend implementing its own walk is what this whole module argues against.
    """
    listed = store.stored(limit=limit)
    return lineage_faults(((one.session_id, one.parent) for one in listed), store.exists)


def attach(ctx: Context, store: SessionPersistence) -> None:
    """Wire a store to the session firehose. One subscription list, not two.

    Both backends' `apply` had their own copy of this — the `provide`, the
    catch-up loop and all four `ctx.on` lines — so a new session event or a
    changed catch-up rule was two edits with nothing to fail if only one landed,
    and a backend that missed a subscription fails silently as "no stored
    sessions". That is the failure mode `SessionPersistence`'s own docstring
    gives as the reason for typing the Protocol.
    """
    ctx.provide(SESSION_PERSISTENCE, store)
    # Catch-up: a row (re)activated after sessions already exist owes them what
    # a freshly created one is owed.
    #
    # A store built here is told a construction-time `durable_length` (B7);
    # asking the medium what it holds is each backend's job. See
    # `JsonlSessionStore.flush`.
    for session in ctx.require(SESSIONS).list():
        store.track(session)
    ctx.on("session/created", store.track)
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


def write_on_unwind(scope: Context, store: SessionPersistence) -> None:
    """Make writing every live session the last thing `scope` does before it lets go (F2).

    **Teardown appends.** An agent's workspace is released with `workspace/disposed`,
    a child's parent-scope effect writes its tombstone, a canceled turn closes
    itself. All of it happens while `scope` unwinds its children — every row, and
    beneath the `agent` row every agent scope — which is after each host's own
    flush, and for the agent scopes after the persistence row itself has gone:
    rows unwind in reverse, and persistence mounts after `agent`. So a clean stop
    wrote a log that read as a crash, and `workspace-reconcile` reclaimed trees a
    clean exit had already released.

    An effect of the **mount's own scope** is the one thing that runs after all of
    that: a scope unwinds its children first, then its own effects, LIFO. Each
    backend's `claim` registers this right after the lease (I-5) is taken on the
    same scope, so it runs before the lease is given back — the log is written
    while it is still this process's to write, and no host has to remember it.

    Straight to the backend rather than through `SessionStore.flush`, which
    dispatches `session/flush` to listeners that have unwound by now. Parents
    before children, through the store's own `lineage` and for its reason: a
    reference-forked child is unreadable without its parent's prefix, so a write
    cut short must leave the parent's done.

    A scope with no session store has nothing to write. **One registration per
    claim, one walk per scope.** Every claim registers, because only the newest
    registration sits above the newest lease; registering once, at the first
    claim, would give each later lease back before its log was written. The first
    to run is that newest one, and it writes every live log. The rest find nothing
    owed and return, where each used to walk every session again: a root with 200
    children spent 472ms unwinding, against 260ms with one walk.
    """
    sessions = scope.get(SESSIONS)
    if sessions is None:
        return
    _OWED.setdefault(scope, set()).add(id(store))

    async def last_write() -> None:
        owed = _OWED.get(scope, set())
        if id(store) not in owed:
            return
        owed.discard(id(store))
        written: set[str] = set()
        for live in sessions.list():
            for session in sessions.lineage(live):
                if session.id in written:
                    continue
                written.add(session.id)
                try:
                    await store.flush(session)
                except Exception:
                    # One unwritable log must not keep the next from being written.
                    log.warning(
                        "ph.persistence: session %s could not be written on unwind",
                        session.id,
                        exc_info=True,
                    )

    scope.add_disposer(last_write, label="session-last-write")
