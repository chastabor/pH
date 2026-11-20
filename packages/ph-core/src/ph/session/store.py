"""`ctx.sessions`: the live session store, and forking.

The store publishes seven events:

* `session/created`
* `session/disposed`
* `session/event`: the post-commit append feed.
* `session/flush`: the awaited durability checkpoint. It is a `parallel`
  dispatch, so every backend runs and the caller waits for all of them.
* `session/durable`: what that checkpoint made durable, for a reader that must
  not run ahead of the log.
* `session/unwritten`: a checkpoint that raised, and why.
* `session/loaded`: what a log that was read from disk owes before anything
  runs in it. The function that read the log awaits it.

Forking is the branching mechanism (D2). pH does not model branching as a
message tree. It models branching as `fork(source, boundary)` plus
`seed_length`. The boundary must be a **closed turn**. A fork that ended inside
an open turn would inherit a half-executed step whose tool results never
arrive.

@module ph.session.store
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from ..cordis import Context, Disposer, events, plugin
from ..keys import INTENTS, SESSIONS
from ..seams.invariants import contribute_fold_cache
from .events import BatchRef, SessionEvent, now_ms
from .journal import IntentJournal
from .session import Session, SessionHeader, SessionKind
from .writers import log_writer

_LOG = log_writer(__name__)

__all__ = [
    "SessionForkError",
    "SessionStore",
    "apply",
    "child_session_id",
    "fork_boundaries",
    "is_child_id",
    "is_fork_boundary",
    "new_session_id",
    "open_turn_at",
    "session_written",
    "valid_session_id",
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
    doc=(
        "Post-commit append feed, once per event of every session, each streamed chunk "
        "included. Filter synchronously and return an awaitable only for what you act "
        "on: an async listener is a task per event."
    ),
)
events.declare(
    "session/flush",
    "parallel",
    owner="ph.session",
    doc="Awaited durability checkpoint: every backend drains and the caller waits.",
)
events.declare(
    "session/durable",
    "emit",
    owner="ph.session",
    doc=(
        "A flush every backend finished: the log holds the session's events below the "
        "count. Said when the count moved, and after a flush of the session failed."
    ),
)
events.declare(
    "session/unwritten",
    "emit",
    owner="ph.session",
    doc=(
        "A flush of the session raised, with the error, so the log holds less than the "
        "session does. The next flush of it that returns says `session/durable`."
    ),
)
events.declare(
    "session/loaded",
    "parallel",
    owner="ph.session",
    doc="A stored log was read into the store; the door that read it awaits every listener.",
)

ForkRejection = Literal[
    "SESSION_NOT_FOUND",
    "SESSION_ALREADY_EXISTS",
    "SESSION_ID_INVALID",
    "INVALID_BOUNDARY",
    "OPEN_TURN",
    "OPEN_BATCH",
    "SESSION_IS_SUBAGENT",
]
"""Why a session could not be created, adopted, forked or opened.

The set is closed because a client branches on the code: a daemon reply carries
it, and `phern` maps it to an exit status. A new refusal is therefore a
deliberate addition to a vocabulary, and not a string that someone invents at a
raise site. `SESSION_ID_INVALID` belongs to K9: an id that cannot be a path
component is refused before it names a directory in three stores.
`SESSION_IS_SUBAGENT` belongs to P11-08: the log of a sub-agent was opened as a
root, and only the mount of its root writes that log.
"""


class SessionForkError(Exception):
    """A fork was refused, with the reason as a stable code."""

    def __init__(self, message: str, code: ForkRejection) -> None:
        super().__init__(message)
        self.code = code


def new_session_id() -> str:
    """A sortable, human-legible session id."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def child_session_id(parent_id: str, run_id: str) -> str:
    """The session id of a sub-agent: the id of its parent, then the id of its run.

    **The prefix is how a parent finds its children on disk** (Phase 11). The
    store narrows a family directory to the names that begin with the id of the
    parent before it reads a header (`SessionArchive.descendants_of`). A child
    with any other name is therefore one that no restart finds. This function
    is the one spelling, for every provider that names a child and every reader
    that narrows by the name."""
    return f"{parent_id}-{run_id}"


def is_child_id(parent_id: str, session_id: str) -> bool:
    """Whether `session_id` is named as a child of `parent_id`, or as a child of one
    of its children. The id of a grandchild also begins with the id of its
    grandparent. The header decides which of the two it is."""
    return session_id.startswith(child_session_id(parent_id, ""))


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
"""What a session id may be (K9).

**A session id becomes a directory name in several stores**, and none of them
checks it again:

* `SpillStore.plan` joins it onto the spill root, and `sweep_session` unlinks
  unreferenced files under whatever that path resolves to.
* The session archive names a file after it.
* The workspace scratch root is `<scratch>/<session>/<agent>`.

Each of them sanitizes the *other* half of the name and trusts this half,
because `new_session_id` minted every id for the whole life of the codebase.

That is not always true. `create` takes a `session_id` that the caller supplies.
Such an id arrives from `phern -p --session`, from the `session/attach` of a
daemon client, and through `adopt` on the resume path. An id of `../../..` would
make the sweep delete outside the store that it was pointed at.

The id is checked one time here and is not sanitized at each store.
`ph.paths.canonical` states the reason, for spellings: a sanitizer at four call
sites is four chances to disagree about what `..` becomes. The id is also a
*key* (in `_entries`, in `parentSession`, in the roster), so a mapping of two
ids onto one name would be a defect of its own. A leading dot is refused with
the rest, so an id cannot be a hidden file. A bare `..` fails the first
character class.
"""


def valid_session_id(session_id: str) -> bool:
    """Whether `session_id` is safe to use as a path component and as a key."""
    return bool(_ID.match(session_id))


def _require_valid_id(session_id: str) -> None:
    """The refusal, in one sentence and one place. See `_ID` for why."""
    if not valid_session_id(session_id):
        raise SessionForkError(
            f'session id "{session_id}" is not usable as a path component: it must be '
            "alphanumerics, dots, dashes and underscores, starting with a letter or "
            "a digit",
            "SESSION_ID_INVALID",
        )


@dataclass(slots=True)
class _Entry:
    session: Session
    unobserve: Disposer
    """The teardown of the observer, with the type that it already has.

    The annotation was `Any`, which names nothing. The walk of
    `test_registration_ownership.py` over row-supplied callables therefore could
    not see this field, and nothing made anyone say whether it runs inside a
    binding. `Session.observe` returns `Callable[[], None]`. With the type
    named, the gate sees the field and files it as teardown."""
    announced: int
    """How much of the log `session/durable` has reported as held. The count starts
    from what a store held when the session was made (`durable_length`)."""
    unwritten: bool = False
    """Whether a flush of the session raised since the last flush that returned, as
    `session/unwritten` reported. The next flush that returns announces whatever
    it wrote."""


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

        `inherited` is the number of leading events that are **already durable
        in some other log**: the prefix of a fork, which lives in the file of
        the parent. It is the storage half of a fork, and it is not
        `seed_length`. `seed_length` is the *provenance* boundary. Five folds
        read it to tell the history of the parent from the work of this session,
        and it stays the fork boundary whatever a store holds.

        `inherited` is passed to the constructor and is not assigned after it,
        because `session/created` makes a backend queue the log. A store that
        saw the whole seed would write out the copy that reference-forking
        exists to avoid. A keyword argument makes that order impossible to get
        wrong.
        """
        resolved = session_id or new_session_id()
        _require_valid_id(resolved)
        if resolved in self._entries:
            raise SessionForkError(f'session "{resolved}" already exists', "SESSION_ALREADY_EXISTS")
        fields: dict[str, Any] = {"id": resolved, "createdAt": now_ms()}
        fields.update(meta or {})
        # **Inheritance happens at the one construction gate**, so a child shares
        # the directory of its parent whatever made it. `_branch` used to enforce
        # this, and the other caller that creates a child, the subagent spawn,
        # did not. Every subagent therefore got a family of its own. The central
        # claim of the layout is that one conversation and everything that it
        # spawned are in one directory, and that claim was false for the thing
        # that spawns most. A root needs no line here: `SessionHeader` defaults
        # an absent family to the id of the session.
        parent_id = fields.get("parentSession")
        parent = self.get(str(parent_id)) if parent_id else None
        if parent is not None:
            fields.setdefault("family", parent.header.family)
        header = SessionHeader.model_validate(fields)
        session = Session(resolved, seed=seed, header=header, durable=inherited)
        if seed is not None:
            # **Where the work of a branch begins**, at `seed_length`: the first
            # event that a fork or a segment writes itself. `create` is the one
            # function that makes a branch, so it lays the marker here. The
            # constructor does not lay it, because every resume, view and
            # replay goes through the constructor too. There the marker marked
            # nothing, and a viewer showed an event that the log on disk did not
            # have. The marker is appended before `_publish`. It is therefore
            # in the log when a backend first tracks the log, and no listener
            # treats it as a new event.
            _LOG.append(session, "session/end-seed", {})
        return self._publish(session)

    def adopt(self, session: Session) -> Session:
        """Publish an already-constructed session (the resume path).

        The id is checked here too (K9). `adopt` is how a log comes off disk, and
        a `Session` that a caller built never passed through `create`.
        """
        _require_valid_id(session.id)
        if session.id in self._entries:
            raise SessionForkError(
                f'session "{session.id}" already exists', "SESSION_ALREADY_EXISTS"
            )
        return self._publish(session)

    def _publish(self, session: Session) -> Session:
        def observer(source: Session, event: SessionEvent) -> None:
            # An `emit`, so one failing listener neither aborts the dispatch nor
            # un-appends a committed event (A1).
            self.ctx.emit("session/event", source, event)

        self._entries[session.id] = _Entry(
            session=session,
            unobserve=session.observe(observer),
            announced=session.durable_length,
        )
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
        # This runs after the event. A listener that reads the cached folds of
        # the session during the disposal (`SubagentService._let_go`) therefore
        # still finds them.
        for release in entry.session.mark_disposed():
            try:
                release(entry.session)
            except Exception:
                log.exception("ph.session: a release of session %s failed", session_id)

    async def flush(self, session: Session) -> None:
        """Await the drain of every backend for this session, **ancestors first**.

        The log of a reference-forked child names a parent and a count. It is
        readable only when the log of that parent holds those events. A child
        that is made durable first and then orphaned by a crash is a log that
        nothing can read. The reverse order is harmless at every point where it
        can be interrupted. This is the common case and not a rare one: the
        boundary of a fork is usually the live tip of the parent, which is
        exactly the part that is not yet written.

        **The order is set here and not in `attach`**, where it started. This
        method is the only dispatcher of `session/flush` in the repository. It
        owns `_entries`, the live graph that the walk needs. It also owns `fork`
        and `roll`, the calls that create the debt that a flush settles. When
        the order was set inside the listener of one backend, three things were
        wrong:

        * The ancestors drained *outside* the dispatch. A second listener (a
          mirror, an exporter) saw only the child and had to implement the walk
          again.
        * The walk ran one time for each attached backend.
        * The guarantee came from the use of `attach` and not from the Protocol.
          That is why the backend parity suite could not reach it.

        The cost is that one `flush(child)` is now N dispatches and not one.
        Both backends return early when nothing is owed, so each ancestor costs
        a dict lookup.

        **Then `session/durable` reports how much of the log is now held**, for
        a reader that must not run ahead of the log (the telemetry ledger). The
        count is taken before the dispatch. Each backend writes, under its own
        lock, everything that the log held when that write began, so the count
        is a floor. With no backend attached, every flush answers at once, as
        `written` does.

        **The event is emitted only when the count moved.** A flush that found
        nothing owed has nothing to report, and its listeners did work for each
        event. The daemon refolds the holds of a root over its children, so a
        shutdown that flushed every finished child of a root walked those
        children one time for each child. A session that this store does not
        hold has no count, and it is announced on every flush.

        **A flush that raised is reported** as `session/unwritten`, with the
        error. The event is emitted for the session that failed and for every
        session after it in the lineage, because those were not flushed either.
        It is emitted whichever caller started the flush. A reader such as the
        daemon, which marks the root unwritten, therefore hears every failure
        from one place. Only a `session/durable` announcement withdraws that
        report, so the next flush that returns makes one even if it wrote
        nothing. A cancellation is not a failure, as `written` says, and it is
        not reported.
        """
        lineage = self.lineage(session)
        for index, ancestor in enumerate(lineage):
            through = ancestor.seq
            try:
                await self.ctx.parallel("session/flush", ancestor)
            except Exception as error:
                for unsettled in lineage[index:]:
                    if (failed := self._entry_of(unsettled)) is not None:
                        failed.unwritten = True
                    self.ctx.emit("session/unwritten", unsettled, error)
                raise
            entry = self._entry_of(ancestor)
            if entry is not None:
                if through <= entry.announced and not entry.unwritten:
                    continue
                entry.announced, entry.unwritten = max(entry.announced, through), False
            self.ctx.emit("session/durable", ancestor, through)

    def _entry_of(self, session: Session) -> _Entry | None:
        """The entry holding this very session, and not another copy under its id."""
        entry = self._entries.get(session.id)
        return entry if entry is not None and entry.session is session else None

    async def written(self, session: Session) -> bool:
        """`flush`, but it answers whether the flush worked and does not raise.

        This method is for a caller that has to *decide* something about a
        record that it could not make durable: not to put a question to a person
        (`approval`, `user-questions`), or not to report a child done
        (`ph.seams.subagents`). The contract of such a caller is to answer and not
        to raise. There is one spelling, so the seams that need it do not each
        grow a `try` that logs differently. The module function
        `session_written(ctx, session)` does the same for a caller that holds
        only a `Context`. Cancellation still propagates, because it is not a
        failed write.
        """
        try:
            await self.flush(session)
        except Exception:
            log.warning("ph.session: could not write session %s", session.id, exc_info=True)
            return False
        return True

    async def loaded(self, session: Session) -> None:
        """Settle what a log that was read from disk owes, before anything runs in it.

        Both functions that bring a stored log into the store dispatch this
        event: `resume_session`, after `session/resumed` is recorded, and
        `stored_session`. Both await it, so the caller holds the session only
        after every listener is done. The dispatch is `parallel`, for the same
        reason as `session/flush`: every listener runs, and the caller waits for
        all of them. The event is only for a log that was read from disk. A
        fresh session or a fork has no crash of its own to settle.

        **Why not `session/created`, where both of its listeners started.**
        `session/created` is an `emit`, so the listeners ran detached. A resumed
        session was handed out, its agent started, and its model took a first
        turn. Meanwhile the leaked trees of its crash were still being
        reclaimed, and a blob that its log names was still staged. The records
        of the listeners landed wherever the listeners happened to finish. Only
        `drain()` bounded them.

        **A listener cannot refuse the read.** What a listener owes is the
        housekeeping of the last run: it reclaims the trees of that run and
        finishes its writes. A session that nobody can open because the
        housekeeping hit a bug would be the worse failure. What a listener left
        undone is found again at the next read. A failure is therefore logged
        here, for every listener at once, in the same way that `written` is the
        one spelling of a flush that may fail. Cancellation propagates, and the
        `opening` guard of the reader disposes the session.
        """
        try:
            await self.ctx.parallel("session/loaded", session)
        except Exception:
            log.exception("ph.session: work owed by %s was left unsettled", session.id)

    @asynccontextmanager
    async def opening(self, session: Session) -> AsyncIterator[Session]:
        """Dispose `session` if anything stops the span before the session is handed out.

        **Every reader of a stored log needs this guard, and this is its one
        spelling.** From the moment the store holds a session to the moment its
        caller holds it, a refusal, a failed read of its children or a
        cancellation must not leave it published. A session in the store that
        nothing finished opening is one that a peer that asked again would run
        on. `resume_session` and `stored_session` hold the guard from `adopt`
        through `loaded`, and `open_session` holds it from there to its return.
        """
        try:
            yield session
        except BaseException:
            self.dispose(session.id)
            raise

    def lineage(self, session: Session) -> tuple[Session, ...]:
        """`session` and every ancestor **whose prefix it references**, oldest first.

        Every write of a lineage takes this order: `flush`, and the last write of
        the mount (`ph.persistence.protocol.write_on_unwind`). The last write
        cannot go through `flush`, because the listeners of `flush` have unwound
        by then.

        **A parent is included only when the child inherits from it** (P10-04).
        The rule is for a reference fork or a segment. The file of such a child
        begins at `seed_length`, and it reads back only when the file of the
        parent holds the events that it names. A subagent child is created
        fresh, with a `parent_session` and no seed, so its file is readable on
        its own. Writing its parent first cost one parent fsync for each flush
        of every child. Under `rlm`, that is one for each child tool call that
        can reach past the tree. The parent is still written by its own flushes
        and by the last write, which walks every live session.

        An ancestor that this store does not hold is one that nothing here
        writes, so its log is already whatever it is going to be. The chain is
        keyed by id, so the dict is both the order and the cycle bound. A
        hand-edited header that names its own descendant costs a lookup and not
        a hang.
        """
        # The result is a tuple, and `chain` is keyed by id. This class has its
        # own `list` method, which shadows the builtin inside the class body.
        # `list(...)` here would resolve to that method, and mypy reports it
        # with a message that is hard to read.
        chain: dict[str, Session] = {}
        current: Session | None = session
        while current is not None and current.id not in chain:
            chain[current.id] = current
            parent = current.header.parent_session if current.header.seed_length else None
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

        `boundary` is an inclusive event seq of the source. When it is omitted,
        it is the current last event of the source. The slice may end on a
        between-turn event, but it must not end inside an open turn.

        **The child holds the prefix in memory and does not store it again.**
        Every reader (the transcript fold, the surface fold, `derive_messages`)
        sees a whole session exactly as before. What changes is the disk. The
        file of the child begins at `session/end-seed`, at seq `seed_length`,
        and its header says which log the rest came from.

        **The prefix that a child depends on is immutable**, because the log is
        append-only. The parent may therefore keep running without limit and
        never invalidate a descendant. This needs no lock, no copy-on-write and
        no invalidation.

        A fork is cheap on disk and **not** free in memory: `_readmit` freezes
        every seeded event again. The seed is still handed over in full,
        because every reader downstream wants a whole session. A trusted-seed
        path would be a change to the session model and not to `fork`.

        The narrower fix is to skip the *copy* and keep every check. It is
        measured and deliberately deferred. The module docstring of
        `ph.session.json` gives the reason. In short, a fast path through the
        gate that enforces A1 would be correct only as far as its tests reached.
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
        """The shared body of `fork` and `roll`. `kind` is the only difference.

        The method is private because `kind` is not a choice that a caller makes.
        It says which of the two operations the caller called. A `kind`
        parameter on `fork` would invite a third answer, and there are only two.
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
        """Continue a session in a fresh log: segmentation (§7 step 6).

        **A roll is a fork at the tip with no divergence, and nothing more.**
        `fork(source, None)` already means "at the last event", so segmenting
        costs a marker and a name and not a second code path. It also inherits
        the refusal of `_fork_seed`, which is the correct rule: a segment
        boundary inside an open turn would start a log in the middle of a step.

        **The session id changes.** That was the decision, and it is the one
        cost. Some things do *not* break. `/revert` resolves a checkpoint by the
        tree hash that the `workspace/checkpoint` event carries, and not by a
        lookup of a ref under the current id. The ref that keeps that tree alive
        still stands under the old id. A revert therefore reaches back across a
        roll. What does move is anything that is keyed by id: a client that
        holds the old id holds a log that has stopped.

        `session/segmented` is appended to the **parent**, after the fork. It is
        therefore the terminal record of the parent, and not something that the
        child inherits. The link then exists in both directions, from state
        that already exists: the parent names its continuation, and the header
        of the child names its origin. Without the record, a segment and a
        branch cannot be told apart, because they have the same structure.

        **The parent is left live**, deliberately. A disposal would unhook the
        persistence observer while the `Session` object stayed fully usable, so
        an append through a stale reference would vanish with nothing raised.
        Because the parent stays published, an in-flight append still lands. A
        caller that goes on writing to the parent has made a *branch*. That is
        legitimate, and it shows as a branch: its events sit after a marker that
        says where the other side went.

        The method is synchronous, like `fork`. Durability arrives with the next
        flush. `flush` orders ancestors first, so the marker of the parent
        cannot reach disk after the child that references it.
        """
        live = self._resolve_source(source)
        child = self._branch(live, None, child_session_id, kind="segment")
        _LOG.append(live, "session/segmented", {"continues": child.id})
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
        batch = _cut_batch_at(log[boundary])
        if batch is not None:
            # The prefix would hold half of a set of records that only mean
            # something together (P10-15). The seed would refuse that prefix
            # anyway, with a less readable error.
            raise SessionForkError(
                f'fork boundary {boundary} in session "{session.id}" ends inside the '
                f"batch at seq {batch.first}",
                "OPEN_BATCH",
            )
        return tuple(log[: boundary + 1])


def open_turn_at(log: Sequence[SessionEvent], boundary: int) -> SessionEvent | None:
    """The `turn/start` that a boundary would cut into, or `None` if the boundary is safe (A6).

    This is the one statement of the rule: a fork may end on any between-turn
    event, and it may not end inside a turn that has not closed. The function is
    exported because a *reader* needs the same answer before it offers the
    action. The trajectory view marks which records a fork may target. A second
    statement of the rule there accepted `turn/end` only. It refused three legal
    boundaries out of four, and it cited A6 for that.
    """
    if boundary < 0 or boundary >= len(log):
        return None
    last: SessionEvent | None = None
    for event in log[: boundary + 1]:
        if event.type in ("turn/start", "turn/end"):
            last = event
    return last if last is not None and last.type == "turn/start" else None


def _cut_batch_at(event: SessionEvent) -> BatchRef | None:
    """The batch that a boundary at `event` would cut into, or `None` if the
    boundary is safe (P10-15).

    A batch is a set of records that only mean something together. A prefix may
    end before the batch or on its last member, and not between two members.
    This is the one statement of that rule, for the same reason as
    `open_turn_at`: `_fork_seed` refuses with it, and the two functions below
    answer with it before a reader offers the action.
    """
    batch = event.batch
    return batch if batch is not None and event.seq != batch.last else None


def is_fork_boundary(log: Sequence[SessionEvent], boundary: int) -> bool:
    """Whether `ctx.sessions.fork` would accept this boundary."""
    return (
        0 <= boundary < len(log)
        and open_turn_at(log, boundary) is None
        and _cut_batch_at(log[boundary]) is None
    )


def fork_boundaries(log: Sequence[SessionEvent]) -> set[int]:
    """Every seq that `fork` would accept, from one pass over the log (A6).

    This is the same rule as `open_turn_at` and `_cut_batch_at`, answered for
    the whole log at one time. A reader that marks *which records* a fork may
    target needs the answer for every record. A separate question for each
    record scans the prefix again each time. That is quadratic, and it freezes
    a UI that opens the trajectory from a running chat (P4-17).

    The set is keyed by seq, and a seq is admitted only where it is the index of
    its event. `_fork_seed` requires that contiguity anyway, so this function
    marks nothing that the fork would then refuse.
    """
    boundaries: set[int] = set()
    open_turn = False
    for index, event in enumerate(log):
        if event.type == "turn/start":
            open_turn = True
        elif event.type == "turn/end":
            open_turn = False
        if not open_turn and event.seq == index and _cut_batch_at(event) is None:
            boundaries.add(event.seq)
    return boundaries


async def session_written(ctx: Context, session: Session) -> bool:
    """`SessionStore.written` for a caller that holds a `Context`.

    A deployment with no session store has nothing to write, and that is not a
    failed write. The function therefore answers `True`. The callers that must
    decide (to ask a person or not, to cache a handle or not) need no lookup of
    their own.
    """
    sessions = ctx.get(SESSIONS)
    return sessions is None or await sessions.written(session)


@plugin("session", affects="deployment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the session store, and the intent journal over it (decision 1).

    The journal is mounted here and not in a row of its own. Every profile with a
    log has intents in it, and the barrier of the journal is the flush of this
    store. A profile that dropped a separate journal row would leave every
    producer to test whether the journal is present.
    """
    sessions = SessionStore(ctx=ctx)
    ctx.provide(SESSIONS, sessions)
    journal = IntentJournal(sessions=sessions)
    ctx.provide(INTENTS, journal)
    contribute_fold_cache(ctx, id="intent-fold-cache", subject="intent index", stale=journal.stale)
