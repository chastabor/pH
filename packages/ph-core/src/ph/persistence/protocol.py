"""`SessionPersistence`: what a backend must provide, with no statement of where it writes.

**The split that makes a non-file backend possible** is between "where are the
bytes" and "what can you tell me". A store that keeps sessions in a database has
no per-session path and no directory to list. It can still answer four
questions: *does this exist*, *read it back*, *what is stored*, and *where would
a person look*. To the last question it correctly answers `None`. The write side
therefore needs nothing new: the log itself is the queue, and a backend writes
it on `session/flush`. The read side is stated here and is not inferred from a
filename.

**`locate` may answer `None`**, and a caller must handle that answer and not
assume a path. Both shipped backends keep one file per session and answer with
that file. That is why the lease of P5-03 works under either backend. But the
Protocol does not require a file. A backend that keeps sessions elsewhere
answers `None`. A caller that wants to *show* a path then shows nothing. A
caller that wants to *lock* a path refuses with an error, and does not invent a
path that protects nothing.

@module ph.persistence.protocol
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from ..cancel import reason_of
from ..keys import SESSION_PERSISTENCE, SESSIONS
from ..seams.diagnostics import Diagnostic, contribute
from ..session import Session, SessionEvent, SessionHeader
from .families import descendants
from .lineage import lineage_faults

if TYPE_CHECKING:
    from ..cordis import Context

log = logging.getLogger("ph.persistence")

_OWED: WeakKeyDictionary[Context, set[int]] = WeakKeyDictionary()
"""For each scope, the stores (by `id`) to which the scope still owes a last write
(`write_on_unwind`)."""

_UNWRITTEN: WeakKeyDictionary[Context, dict[str, str]] = WeakKeyDictionary()
"""Each scope's record for `unwritten_at_unwind`."""

SURVEY_LIMIT = 500
"""The number of stored sessions that the lineage check surveys.

The limit is larger than the 50 of the picker, because this check asks a
question about the *store* and does not show a page to a person. A broken chain
that fell below the cut is exactly the chain that nobody has looked at lately.
The survey is still bounded, because the purpose of an answer from the listing
is to avoid a walk of the store without limit.
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
    "unwritten_at_unwind",
    "write_on_unwind",
]


class NoStoredSession(FileNotFoundError):
    """Nothing is stored under an id. `read_own` raises this, and so `read` does too.

    The error has its own type because it decides whether a session is resumed
    or created (`open_session`, through `read_if_stored`). Any other error from
    the read of a log that exists is damage to report. A file that vanishes
    during the read is one such error. It is not an absence that permits a new
    session.
    """


@dataclass(frozen=True, slots=True)
class StoredSession:
    """Enough about a stored session to choose it from a list.

    This is deliberately not a `Path`. A picker wants an id and a time. The
    JSONL backend gets its answers from a `stat`, and another backend gets them
    from a row. `cwd` scopes a listing to one repo (P5-14).
    """

    session_id: str
    modified: float
    cwd: str = ""
    parent: str | None = None
    """The log that this one continues or was spawned from:
    `SessionHeader.parent_session`. A lineage check follows this link
    (`lineage_faults`)."""
    delegating_parent: str | None = None
    """The agent that spawned this one, or `None` for a session that a person
    started: `SessionHeader.delegating_parent`. This is not `parent`. A fork names
    in `parent` the log that it was cut from, and a fork is a session of its own.
    So a family walk over a listing (`workspace.stored_survivors`) follows this
    link and not `parent`."""
    family: str = ""
    """The directory that holds the log of this session: the family of its
    lineage (`SessionHeader.family`).

    A listing consumer needed this field. That is the condition that the notes
    below set for a new field. `read_own` takes `family` and says that it is
    *not* a hint: with it a log is a path, and without it a log is a directory
    search. A fold over every stored session (`phern attachments gc`) read one
    log per row and paid that search for each row. So a listing of 4 000
    sessions across 1 000 families cost one scan per session in addition to the
    scan of the listing itself.

    The field is empty when the header does not parse. That is the safe
    direction. The reader falls back to a search, which is what it did for every
    row before this field existed."""

    # **No `kind`**, for the reason above. The field was added here. One backend
    # filled it, the other backend missed it, and nothing read it. The picker
    # resolves a segment from `SessionSummary`, from its own header peek. The
    # field comes back when a listing consumer needs it. It then goes through
    # `stored_row`, so that it cannot reach one backend and miss the other again.

    # **No `size` and no `title`**, deliberately. `size` meant two different
    # things in one field: bytes on disk from JSONL, and an event count from
    # Turso. A picker renders that field with `filesize.decimal`. To produce the
    # half that nobody could interpret was most of the cost of the Turso
    # listing. Neither backend set `title`. A field that is always empty
    # promises something that it does not deliver. It would also have let a
    # picker migration fall through to hex ids with no failure. Both fields come
    # back when a consumer needs them and can say what they mean.


def stored_row(session_id: str, header: SessionHeader | None, modified: float) -> StoredSession:
    """One listing row from one header peek. **The only place rows are built.**

    Both backends built this row by hand from the same two inputs. When a field
    was added, it reached one backend and not the other. Nothing failed, because
    the parity suite checked `session_id` and `cwd` only. A header that this
    build cannot parse still gets a row. To lose a session from a listing is
    worse than to show it without its details.
    """
    return StoredSession(
        session_id=session_id,
        modified=modified,
        cwd=(header.cwd or "") if header is not None else "",
        parent=header.parent_session if header is not None else None,
        delegating_parent=header.delegating_parent if header is not None else None,
        # There is no `or ""` guard here, unlike `cwd` above. `SessionHeader.family`
        # is a `str` with `min_length=1`, and its docstring says that it is never
        # absent. The idiom of the nullable line above would suggest that this
        # field can be blank when a header parsed. The `else` arm is the one real
        # source of `""`, and `StoredSession.family` says so.
        family=header.family if header is not None else "",
    )


def descendants_among(parent_id: str, rows: Iterable[StoredSession]) -> tuple[StoredSession, ...]:
    """The rows beneath `parent_id` (its children, their children, and so on), ordered
    by id. **The header decides.**

    Each backend narrows its candidates by the family directory and the id prefix
    (`families.children_under`). Neither is proof, because a fork can be named
    after its source and filed in the family of its source. So a row is beneath
    `parent_id` only when its `delegating_parent`, followed up through the other
    candidates, reaches `parent_id`. That link is the `origin: "subagent"` link
    of the header. A fork does not have one: its `parent` names the log that it
    was cut from. The `delegating_parent` of each row says which level the row
    is on.

    The rule is stated here and not one time per backend, for the reason that
    `stored_row` gives. Two backends that each wrote their own filter would
    drift, and nothing would fail. The rows are ordered by id so that both
    backends give the same answer.

    A row whose header does not parse has no `delegating_parent`. So it is not
    listed, and neither is anything beneath it. A log that cannot say whose
    child it is cannot be resumed as a child. A chain that loops never reaches
    `parent_id`.
    """
    candidates = {row.session_id: row for row in rows}
    lineage = [(one.session_id, one.delegating_parent) for one in candidates.values()]
    beneath = descendants([(parent_id, None), *lineage], parent_id)[1:]
    return tuple(sorted((candidates[one] for one in beneath), key=lambda row: row.session_id))


@runtime_checkable
class SessionArchive(Protocol):
    """The listing of a session store and its two reads. A fold needs these and no more.

    This Protocol is split out as `AgentHandle` is split from `AgentDriver`, and
    for the same reason. A fold that only *reads* must not hold `track`, `flush`
    or `forget`. A test that stands in for a store must not have to implement
    seven methods that it never calls. `SessionPersistence` extends this
    Protocol, so each signature is written one time and a real backend satisfies
    both.

    `read` and `read_own` are not interchangeable, and the folds choose between
    them deliberately:

    - `phern attachments gc` uses `read_own`. A chained read fails when an
      ancestor is missing, and it would refuse a collection that is safe to
      make.
    - `stored_survivors` uses `read`. Only the whole lineage that built a tree
      accounts for that tree.

    **Every method blocks**, on a file read or a search of the store. Call it
    off the event loop.
    """

    def read_own(
        self,
        session_id: str,
        upto: int | None = None,
        family: str | None = None,
        *,
        types: frozenset[str] | None = None,
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """This one stored log, without its chain. `read` is built from this primitive.

        `upto` is a hint. The caller does not want events at or above it. To
        return them anyway is slower but not wrong.

        `types` keeps only the events of those types. It is for a reader that
        folds a few records out of a long log, such as the state of a child out
        of a log that is mostly `assistant/chunk` (`ph.seams.subagents`). It is
        also a hint, and it saves the parse: the other events are dropped before
        they are validated. A torn batch at the tail is still dropped whole. The
        backend judges it on every record, not only on the kept ones.

        `family` is **not** a hint. Every member of a lineage shares one family
        directory, so the walk knows where an ancestor is. With `family`, a
        directory search becomes a path. Without it, a chained read paid one
        scan per generation. That cost grows with the size of the store and not
        with the length of the log.

        Raises `NoStoredSession` when, and only when, nothing is stored under
        that id (in `family`, when it is given).

        The method is declared here, and not left to convention, because it is
        the half that a backend really implements. Without the declaration, a
        third backend can satisfy this Protocol with a `read` that returns the
        events of one file. That backend passes mypy, passes
        `runtime_checkable`, mounts through `attach`, and serves *segments* as
        whole sessions. The fault then shows as the "contiguous from 0" refusal
        of `_readmit`, three layers from the cause.
        """
        ...

    def stored(self, *, limit: int | None = 50) -> list[StoredSession]:
        """What is on record, most recently touched first. The result has at
        most `limit` rows, or every row for `None`. A rebuild that must find
        every root asks for all of them."""
        ...

    def holding(
        self, types: frozenset[str], *, gate: str
    ) -> Iterator[tuple[str, list[SessionEvent]]]:
        """Every stored log, by id, with its own records of `types`. It is for a
        reader that needs a few records out of every log, such as a daemon that
        rebuilds its schedule index (S18).

        A log with no `gate` record yields no records. `gate` is the record
        without which the other records mean nothing. For example, the creation
        of a schedule is the gate for its ticks and its cancel. A log that
        cannot be read is logged and left out, so a reader can tell "holds none"
        from "could not say". **This is a Protocol method, and not a walk over
        `read_own`**, because every log is read and each backend has a cheaper
        way than a parse of all of them. JSONL searches the bytes for `gate`
        first. Turso reads each database alone.
        """
        ...

    def descendants_of(self, parent_id: str, family: str) -> tuple[StoredSession, ...]:
        """Every stored session beneath `parent_id` (its children, their
        children, and so on), ordered by id. The `delegating_parent` of each row
        says which level the row is on.

        **The whole tree, in one read.** Every descendant is filed in the family
        of the root and named under the id of its parent. So the scan that finds
        the children of a parent finds all of their descendants with them. A
        listing of one level at a time would read the same directory one time
        per level, and a restart would see nothing below the first level.

        **Every one, not a page.** `stored` checks every log in the store and
        then cuts at `limit`. So the children of a parent fall below the cut
        when the store is big enough, even at `SURVEY_LIMIT`. A picker can
        accept that. A resume ladder, a budget or a spawn cap that missed a
        child would be wrong. So this method stays inside the family of the
        parent, where every child is filed, and checks only the names that start
        with the id of the parent (`families.children_under`). Its cost follows
        the family, not the store.

        `family` is required, for the reason that `read_own` gives for its own
        `family`: it makes this read one directory and not all of them. A live
        parent has it on `session.header.family`, and a stored parent has it on
        `StoredSession.family`. Each row carries it too. So a read of a child
        after this call is `read_own(row.session_id, family=row.family)`, which
        is a path and not a search.

        Only logs that reached disk are listed. The header of a child is written
        at its first flush. So a child that is live and has never flushed is
        known only to the process that runs it. The header decides who is
        beneath the parent (`descendants_among`), so a fork is left out.
        """
        ...

    def read(
        self, session_id: str, *, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """The full log, **materialized**: dense from seq 0, with the chain followed.

        A backend whose file stores only its own run must walk `parent_session`
        to assemble the rest. `materialize(self.read_own, session_id, family=…)`
        is that walk, and the `read` of both backends is exactly that one line.
        `family` locates the first file, as the `family` of `read_own` does. A
        listing row carries it, and without it the store is searched. An
        ancestor that cannot be read is a `LineageError`, never
        `NoStoredSession`. Only the log of this id itself counts as an absence.
        """
        ...


@runtime_checkable
class SessionPersistence(SessionArchive, Protocol):
    """A place that session logs go to and come back from.

    The Protocol is typed and not duck-typed, for the reason that the seams give
    for their provider Protocols. Without the type, a backend whose method
    drifted would fail at runtime inside the `except` of a caller, and the
    failure would be reported as "no stored sessions".
    """

    def track(self, session: Session) -> None:
        """Start to persist this session. **The store owes what it does not already hold.**

        `session.durable_length` is a *floor* that the caller declares at
        construction: the stored length of a resume, or the inherited prefix of
        a fork. It never advances. So a store that is built later in the life of
        a session is told a boundary that was true when the session was built
        (B7). A backend that can measure what its own medium holds must take the
        larger of the two values. A backend whose write is idempotent by seq may
        ignore the question. See `JsonlSessionStore.flush` and
        `TursoSessionStore.track`.

        The rule is stated as what is owed, and not as "its seed is owed a
        write", because the two stopped meaning the same thing and one backend
        kept the old reading. Turso queued the whole log, so a reference-forked
        child was written a full copy of its prefix. `materialize` then saw a
        first seq of 0 and treated the file as complete. Forking on Turso was no
        longer O(1), with no error and with every test still passing.
        `durable_length` is what the log holds and this store does not hold. It
        is set at construction from the stored length of a resume or the
        inherited prefix of a fork.
        """
        ...

    async def flush(self, session: Session) -> None:
        """Write whatever this log holds that the backend does not.

        **The backend reads the log. It never drains a queue.** That is why
        there is no per-event hook here. A backend that buffered from the
        `session/event` firehose would miss whatever is appended after its
        listener unwinds, and teardown is when that happens (F2). A session that
        the backend was never told about is tracked here and written.

        This method can be called after the row that mounted the backend has
        unwound. That is how the last write of a mount reaches the events that
        the teardown of the mount itself appended. See `write_on_unwind`.
        """
        ...

    def forget(self, session_id: str) -> None:
        """Drop what this backend holds in memory for one session."""
        ...

    def exists(self, session_id: str, *, family: str | None = None) -> bool:
        """Whether this backend has a stored log under that id.

        `family`, as `read_own` takes it, is where the log is filed. With it,
        the answer is for that family alone, and it costs one `stat`. Without
        it, the store is searched, because an id alone does not say where a log
        is filed. The search blocks, so call it off the event loop. To read the
        log when there is one, use `read_if_stored`, which searches one time.
        """
        ...

    def locate(self, session_id: str) -> Path | None:
        """Where a person would find this log, or `None` if it is not a file.

        This is a search of the store, as `exists` is. Call it off the event
        loop.
        """
        ...

    def directory(self) -> Path | None:
        """Where this store keeps every session, or `None` if it is not on a filesystem.

        The picker asks this question, and it is a question about the store. A
        log is at `<directory>/<family>/<id>.jsonl`. A picker that knew only the
        path of one log had to walk up two levels to find the rest. That was a
        rule about the layout of this backend, written into a front end.
        """
        ...


@runtime_checkable
class ClaimingStore(Protocol):
    """A backend that can hold one session against every other writer (I-5).

    **This is optional, and it is its own Protocol and not a
    `locate() is not None` probe.** The lease formerly belonged to the daemon.
    The daemon asked the store for a path and locked beside it. So only daemons
    were refused, and `phern -p --session x` appended to a log that a daemon
    held. It also appended to a log that another `phern -p` had just written,
    and that alone was enough to make the session impossible to open. The store
    is the writer, so the claim belongs to the store, and every host reaches it
    through one `open_session`.

    A backend with no per-session file does not implement this Protocol. A host
    that finds no `ClaimingStore` reports that, and does not lock a path that
    protects nothing. Both shipped backends keep one file per session and
    implement the Protocol through `lease.claim_session`.

    `scope` is **required**. It is the lifetime that holds the lock, and nobody
    remembers to release a lease whose owner is a default.
    """

    async def claim(self, session_id: str, *, scope: Context) -> None:
        """Hold this session for the life of `scope`, or raise `SessionBusy`.
        Write every live log before the scope lets go (`write_on_unwind`)."""
        ...


def gated(events: list[SessionEvent], gate: str) -> list[SessionEvent]:
    """`events`, or none of them when none is a `gate` record. This is the rule
    that `SessionArchive.holding` states. It is kept in one place for every
    backend that filters after it reads."""
    return events if any(event.type == gate for event in events) else []


def read_if_stored(
    archive: SessionArchive, session_id: str, *, family: str | None = None
) -> tuple[SessionHeader, list[SessionEvent]] | None:
    """`archive.read`, or `None` when nothing is stored under that id (`NoStoredSession`).

    This is one search of the store. A call to `exists` and then a read searched
    it two times. A broken chain is never treated as an absence: an ancestor
    that cannot be read is a `LineageError`. This call blocks, as the read does.
    """
    try:
        return archive.read(session_id, family=family)
    except NoStoredSession:
        return None


def lineage_faults_of(
    store: SessionPersistence, *, limit: int = SURVEY_LIMIT
) -> list[tuple[str, str]]:
    """Which logs of this store do not materialize, and why.

    This is a module function and not a closure inside `attach`, because more
    than one caller asks the question. `stored_survivors` already has a
    shortfall that it cannot explain: an unreadable ancestor drops a whole
    subtree from its count without a report. A resume, or a future collector,
    wants the same answer before it acts. Inside a `Diagnostic`, the survey was
    reachable only through `phern doctor`. Here, the diagnostic is one
    presentation of the survey.

    The function is backend-neutral on purpose, and so it is not a
    `SessionPersistence` method. It needs only the listing and `exists`, which
    are both already on the Protocol. This whole module argues against a
    separate walk in every backend.

    An ancestor outside the listing is looked for in the family of the session
    whose chain reached it. Every member of a lineage shares that family. So
    each lookup is one `stat`, where a lookup by id alone was a search of the
    store.
    """
    listed = store.stored(limit=limit)
    families = {one.session_id: one.family for one in listed}
    return lineage_faults(
        ((one.session_id, one.parent) for one in listed),
        lambda ancestor, descendant: store.exists(ancestor, family=families[descendant]),
    )


def attach(ctx: Context, store: SessionPersistence) -> None:
    """Wire a store to the session firehose. There is one subscription list, not two.

    The `apply` of each backend had its own copy of this wiring: the `provide`,
    the catch-up loop and every `ctx.on` line. So a new session event or a
    changed catch-up rule was two edits, and nothing failed if only one was
    made. A backend that misses a subscription fails without an error, as "no
    stored sessions". The docstring of `SessionPersistence` gives that failure
    as the reason to type the Protocol.
    """
    ctx.provide(SESSION_PERSISTENCE, store)
    # Catch-up: a row that is activated or reactivated after sessions already
    # exist owes them what it owes a newly created session.
    #
    # A store that is built here is told a `durable_length` from construction
    # time (B7). Each backend must ask its medium what it holds. See
    # `JsonlSessionStore.flush`.
    for session in ctx.require(SESSIONS).list():
        store.track(session)
    ctx.on("session/created", store.track)
    ctx.on("session/flush", store.flush)
    ctx.on("session/disposed", lambda session: store.forget(session.id))

    # **Silent while the store is healthy.** The empty-list contract of
    # `Diagnostic.read` exists for this: nobody reads a section that appears on
    # every run. The diagnostic is registered here for the reason that `attach`
    # itself has: one wiring for both backends. It needs no profile beyond the
    # one that mounted the store, because it answers entirely from `stored()`.
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
    """Make the write of every live session the last thing that `scope` does
    before it lets go (F2).

    **Teardown appends.** The workspace of an agent is released with
    `workspace/disposed`. The parent-scope effect of a child writes its
    tombstone. A canceled turn closes itself. All of this happens while `scope`
    unwinds its children: every row, and beneath the `agent` row every agent
    scope. That is after the flush of each host. When persistence mounted after
    `agent`, it had already unwound by then. So a clean stop wrote a log that
    read as a crash, and `workspace-reconcile` reclaimed trees that a clean exit
    had already released.

    **This is now the backstop.** `ph-base` mounts persistence in the log layer
    beneath `agent`. The last barrier of the checkpoint policy writes the tail
    of the agents through the store as it unwinds. So in that profile this
    function finds nothing owed. It stays for a profile that composes the rows
    differently, or that drops the barrier.

    An effect of the **scope of the mount itself** is the one thing that runs
    after all of that. A scope unwinds its children first and then its own
    effects, in LIFO order. The `claim` of each backend registers this effect
    immediately after it takes the lease (I-5) on the same scope. So the effect
    runs before the lease is given back. The log is written while this process
    still has the right to write it, and no host has to remember to do so.

    The write goes straight to the backend and not through
    `SessionStore.flush`. `SessionStore.flush` dispatches `session/flush` to
    listeners that have unwound by now. Telemetry has unwound with them, so a
    failure here is kept for the host, on a record that the first claim on
    `scope` starts (`unwritten_at_unwind`). Parents are written before children,
    through the `lineage` of the session store and for its reason: a
    reference-forked child is unreadable without the prefix of its parent. So a
    write that is cut short must leave the write of the parent done.

    A scope with no session store has nothing to write. **One registration per
    claim, one walk per scope.** Every claim registers, because only the newest
    registration is above the newest lease. One registration at the first claim
    would give each later lease back before its log was written. The newest
    registration runs first, and it writes every live log. The others find
    nothing owed and return. Formerly each one walked every session again: a
    root with 200 children spent 472ms on its unwind, against 260ms with one
    walk.
    """
    sessions = scope.get(SESSIONS)
    if sessions is None:
        return
    record = _UNWRITTEN.get(scope)
    if record is None:
        record = _UNWRITTEN[scope] = _keep_unwritten(scope)
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
                except Exception as error:
                    # One unwritable log must not keep the next from being written.
                    log.warning(
                        "ph.persistence: session %s could not be written on unwind",
                        session.id,
                        exc_info=True,
                    )
                    record[session.id] = reason_of(error)
                else:
                    record.pop(session.id, None)

    scope.add_disposer(last_write, label="session-last-write")


def _keep_unwritten(scope: Context) -> dict[str, str]:
    """A record for `scope` of the logs that the store could not write.

    The record keeps each write that the store reports as raised, and removes
    the entry when a later write of the same log returns. The last write does
    not always try such a log again, because a sweep writes a stored child one
    time and then lets it go.
    """
    record: dict[str, str] = {}

    def unwritten(session: Session, error: Exception) -> None:
        record[session.id] = reason_of(error)

    def durable(session: Session, _through: int) -> None:
        record.pop(session.id, None)

    scope.on("session/unwritten", unwritten)
    scope.on("session/durable", durable)
    return record


def unwritten_at_unwind(scope: Context) -> Mapping[str, str]:
    """The logs that the mount of `scope` left short, by session id, with the reason.

    The mapping holds every write that raised, through the store or through the
    last write of the mount, and that no later write of the same log reversed
    (`write_on_unwind`).

    It is for the host that unwound `scope`, after the unwind is complete. The
    last write runs after every row has unwound, and telemetry and every
    listener have unwound with them, so nothing hears the failure as it happens.
    The mapping is empty when every log was written.
    """
    return _UNWRITTEN.get(scope, {})
