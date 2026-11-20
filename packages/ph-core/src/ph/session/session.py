"""The append-only session log and the model history that is derived from it.

Three invariants are kept in this file:

* **I3: model-visible means logged.** Anything that reaches a model request
  can be rebuilt from `Session.events` through `derive_messages()`.
* **I4: the log is append-only, and the surface is what changes.** Compaction
  and offload append a `replace`. Nothing rewrites history.
* **A1: `seq == len(log)`.** Every backend relies on this contiguity contract.

`_append` is the acceptance boundary. It is reached only through a writer
(`ph.session.writers`, T6). It validates and freezes the payload, then
validates the surface transition *before* the push, and only then publishes.
A failure therefore leaves the log and the surface untouched. A publication
failure cannot remove an event that is already committed.

Ported from dsh `packages/core/session/src/index.ts`.

@module ph.session.session
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import PurePath
from typing import Literal, TypeAlias, cast

from pydantic import Field, NonNegativeInt, field_validator

from ..json import JsonValue, as_str
from ..llm.types import Message
from ..selectors import matches_any, parse_all
from ..wire import WireModel
from .derive import derive_event_message, derive_transcript
from .events import SESSION_FORMAT_VERSION, BatchRef, SessionEvent, SurfaceIntent, now_ms
from .json import InvalidJsonValueError, freeze_json_value
from .known_event_types import UnknownEventTypeError, is_ignorable, is_known
from .request_header import (
    EpochHeader,
    RequestContext,
    fold_latest,
    parse_request_context,
    parse_request_header,
)
from .revision_index import RevisionIndex
from .surface import SurfaceManager, fold_surface

__all__ = ["Session", "SessionBatch", "SessionHeader", "SessionObserver", "cwd_tag", "family_for"]

CWD_TAG_LENGTH = 6
"""The number of hex characters of the cwd digest that tag a lineage directory.

The value is six (24 bits) because the tag is a **filter, not an identity**. It
lets a listing skip a directory without opening anything in it, and the `cwd`
of the header confirms a match. Two repos that collide cost one extra directory
scan and never give a wrong answer. More characters would add no benefit and
would make every path longer.
"""


def cwd_tag(cwd: str) -> str:
    """The short digest of a working directory, as it appears in a lineage directory name."""
    return sha256(cwd.encode("utf-8")).hexdigest()[:CWD_TAG_LENGTH]


def family_for(session_id: str, cwd: str | None) -> str:
    """The lineage directory for a new root: `<cwd-tag>-<id>`, or the bare `<id>`.

    The name is **bare when there is no cwd**. That is the correct answer, and
    a placeholder tag would not be. A search for one directory must not find a
    session that belongs to no directory. An unfiltered listing still sees it.
    """
    return f"{cwd_tag(cwd)}-{session_id}" if cwd else session_id


log = logging.getLogger("ph.session")

SessionObserver = Callable[["Session", SessionEvent], None]


SessionKind: TypeAlias = Literal["fork", "segment"]
"""Why a log has a parent: it is a **branch**, or it continues the same session.

The type has a name because six sites spell it: the header,
`SessionStore._branch`, two test builders and two listing rows. With a bare
`str` at any of them, a typo ("segement") becomes a value that validates and
never matches.
"""


class SessionHeader(WireModel):
    """Immutable storage metadata, kept *outside* the conversation log.

    The header is a storage concern and not conversation state that can be
    replayed. That is why it is stored beside the log and not in it. Every
    constraint is on a field, so a header is valid as soon as it exists. The
    one check across objects is `validated()`, which asks whether the header
    belongs to *this* session.
    """

    version: int = SESSION_FORMAT_VERSION
    id: str
    created_at: NonNegativeInt
    cwd: str | None = None
    parent_session: str | None = None
    seed_length: NonNegativeInt | None = None
    """How many leading events were inherited through a seed.

    Because this boundary is persisted, resume and replay can tell the history
    of the parent from the work of the child. Without it, a fork looks like a
    session that started with a long conversation.
    """
    family: str = Field(
        # `data.get` and not `data["id"]`: a listing can meet a stored header
        # with no `id` at all. A `KeyError` from a default factory is not a
        # `ValidationError`, so it escaped every caller that caught one, and
        # one hand-edited line made the whole `stored()` listing fail. With
        # `data.get`, a header with no `id` fails validation in the ordinary
        # way: the empty family fails `min_length`, or the missing `id` is
        # refused as a required field.
        default_factory=lambda data: family_for(as_str(data.get("id")), data.get("cwd")),
        min_length=1,
    )
    """The lineage that this log belongs to, and **where that lineage was worked**.

    It is the directory that holds the log: `sessions/<family>/<id>.jsonl`.
    Every fork, segment and subagent below a root inherits the value. One
    conversation and everything that it spawned is therefore one directory, so
    a listing of one directory answers "is anything in here orphaned".

    **The family of a root is `<cwd-tag>-<id>`** (since format 1). The tag is
    six hex characters of the digest of the working directory. The *directory
    names alone* therefore answer "which sessions belong to this repo". A
    listing skips a whole lineage without opening a file. Before, it read the
    header of every log in the store. The tag is a filter and not an identity:
    24 bits collide, and the `cwd` of the header confirms a match. A collision
    costs a directory scan and never gives a wrong row.

    The tag is fixed when the root is created, and it is inherited unchanged.
    No other behavior keeps a lineage in one place. A session never changes its
    working directory, and a fork of a session that was worked in `/a` belongs
    with that session even if the user has moved since.

    **Never absent.** `min_length=1` makes an explicit empty string a refusal,
    and not a value that is silently rewritten. The value is stored and not
    derived. To derive it, the code would walk `parent_session` to the root.
    The walk needs paths, and the family is what gives the path.
    """

    kind: SessionKind | None = None
    """Why this log has a parent: it is a **branch**, or it continues the same session.

    The two are identical in structure, because `roll` is `fork` at the tip.
    Nothing on disk could tell them apart, and every reader of `parent_session`
    had to guess "branch". `None` means a **root**. Every log with a
    `parent_session` has a kind, and the call that made the log sets it.

    The kind is on the header of the **child**, where the backward link already
    is, because every consumer reads that half from a one-line peek. The
    forward half must stay an event (`session/segmented`) for a mechanical
    reason. JSONL writes a header one time and never rewrites it. A
    `continued_by` on the *parent* would therefore be durable under Turso and
    silently lost under JSONL.
    """

    origin: Literal["subagent"] | None = None
    delegation_depth: NonNegativeInt | None = None

    @property
    def is_subagent(self) -> bool:
        """Whether this is the log of a sub-agent, which runs on the mount of its root.

        The answer comes from `origin` and not from `parent_session`. A fork and
        a segment name their parent too, and each is a session of its own: a
        person started it, and it has start options to log and a version of its
        profile to adopt. A child is neither (session profiles, S7)."""
        return self.origin == "subagent"

    @property
    def delegating_parent(self) -> str | None:
        """The agent that spawned this one, or `None` for a session that a person started.

        This is the family link that messaging and the roster read (C7). It is
        `parent_session` for the log of a sub-agent. It is `None` for a fork or
        a segment, whose `parent_session` is the log that it continues and not
        an agent that delegated to it. A forked root is a root. It is a sibling
        of the other roots, and not a child of the session that it was cut
        from."""
        return self.parent_session if self.is_subagent else None

    agent_preset: str | None = None

    @field_validator("version")
    @classmethod
    def _current_format_only(cls, value: int) -> int:
        # The version is checked on load and never migrated. While pH is
        # experimental, an incompatible log is rejected and not half-understood.
        if value != SESSION_FORMAT_VERSION:
            raise ValueError(
                f"session header version must be {SESSION_FORMAT_VERSION}, got {value}"
            )
        return value

    @field_validator("cwd")
    @classmethod
    def _absolute_cwd(cls, value: str | None) -> str | None:
        if value is not None and not PurePath(value).is_absolute():
            raise ValueError(f'session header cwd must be an absolute path, got "{value}"')
        return value

    @property
    def first_own_seq(self) -> int:
        """Where the own work of this log begins: after what it inherited, or 0.

        A fold asks this when the seed of a fork is the history of its source
        and not its own. A splice, a goal or a resume in that seed belongs to
        the source."""
        return self.seed_length or 0

    def validated(self, session_id: str) -> SessionHeader:
        if self.id != session_id:
            raise ValueError(
                f'session header id "{self.id}" does not match session id "{session_id}"'
            )
        return self


def _the_event(event: SessionEvent) -> SessionEvent:
    """The parser that `latest` folds with: it returns the event itself.

    The function is at module level on purpose. As an inline `lambda` it was a
    new object on every `latest()` call. In `Session.projection`, a new object
    is a new registry key, a new fold and a fresh walk of the log each time.
    """
    return event


type _Key = tuple[str, Callable[[SessionEvent], object]]
"""What identifies a fold: the event type that it watches and the parser that reads it.

Both halves are needed. With the type alone, two seams that read different
things from one event would collide. With the parser alone, two folds of the
same function over different types could not be told apart."""


class _LatestFold[T]:
    """An incrementally maintained "latest event of type X" over a growing log."""

    __slots__ = ("_event_type", "_parse", "_seen", "value")

    def __init__(self, event_type: str, parse: Callable[[SessionEvent], T | None]) -> None:
        self._event_type = event_type
        self._parse = parse
        self._seen = 0
        self.value: T | None = None

    def read(self, log: list[SessionEvent]) -> T | None:
        """The current projection. Only the events that arrived since the last
        read are parsed.

        This method is a cursor plus `fold_latest`, and not a second copy of
        that function. For a time it had its own backward walk, to avoid the
        `log[self._seen:]` copy that a slice makes for each read of each fold.
        That duplicated one loop to avoid one allocation, and the rule that the
        loop implements is documented in the other file. `since=` gives both
        benefits.
        """
        self.value = fold_latest(log, self._event_type, self._parse, self.value, since=self._seen)
        self._seen = len(log)
        return self.value


class Session:
    """An event-sourced session: an append-only log of `SessionEvent`s."""

    __slots__ = (
        "_batch",
        "_batch_open",
        "_derived",
        "_derived_generation",
        "_derived_nodes",
        "_events_snapshot",
        "_latest",
        "_log",
        "_observers",
        "_on_disposed",
        "_publishing",
        "_revisions",
        "_surface",
        "durable_length",
        "first_live_seq",
        "header",
    )

    def __init__(
        self,
        session_id: str,
        seed: Sequence[SessionEvent] | None = None,
        header: SessionHeader | None = None,
        *,
        durable: int = 0,
    ) -> None:
        self._log: list[SessionEvent] = []
        self._surface = SurfaceManager(self._log)
        self._events_snapshot: tuple[SessionEvent, ...] | None = None
        self._observers: tuple[SessionObserver, ...] = ()
        self._on_disposed: dict[Callable[[Session], None], None] | None = {}
        """The callbacks that wait for the store to let this session go
        (`when_disposed`), in the order that they asked. `None` after the store
        has let it go. It is a dict so that one callback registers one time."""
        self._publishing = False
        self._batch: SessionBatch | None = None
        """The batch whose block is open, if any. Only one batch is open at a
        time, and it is the only one whose `_append` still stamps."""
        self._batch_open: BatchRef | None = None
        self._derived: tuple[Message, ...] = ()
        self._derived_nodes = 0
        self._derived_generation = 0
        self._latest: dict[_Key, _LatestFold[object]] = {}
        self._revisions = RevisionIndex()

        if seed is not None:
            # Validate the seed to the SAME invariants that `_append` enforces.
            # A replay or a fork must not be able to construct a live log that
            # no backend could store. Otherwise a bad seed shows later as a
            # flush rejection or, worse, as a silent divergence from disk.
            #
            # The loop goes through `admit`, which is that rule with a name. A
            # seed event is already stamped, exactly like an event that arrives
            # over a wire. The rule was written inline here at one time. It was
            # then two statements a screen apart, and this loop was the copy
            # without the docstring. The publish inside `admit` does nothing
            # here by construction: `_observers` is empty until the constructor
            # returns, so nothing can watch a session that does not exist yet.
            for source in seed:
                try:
                    self.admit(source)
                except ValueError as error:
                    # Add the index, which the caller can act on. `_readmit`
                    # names it too, so only the errors of the surface gain it.
                    # Those errors say what is wrong with the event. This one
                    # says which event.
                    raise ValueError(
                        f"invalid seed event at index {len(self._log)}: {error}"
                    ) from error
            if self._batch_open is not None:
                # **A seed may not end inside a batch** (P10-15). A reader
                # drops a batch that a torn write cut short before the batch
                # gets here. An unfinished batch that reaches a seed is
                # therefore damage. To seed it would give every reader half of
                # something that has meaning only when it is whole. Only the
                # seed path makes this check: a replica that admits a batch
                # member by member is legitimately mid-batch between frames.
                opened = self._batch_open
                raise ValueError(
                    f"the seed ends inside the batch at seq {opened.first} "
                    f"({len(self._log) - opened.first} of {opened.count} events)"
                )

        self.durable_length = durable
        """How many leading events a **store already holds**. 0 unless declared.

        A backend reads it to tell what it still owes from what is already
        written. A store queues at least `events[durable_length:]`, and it may
        measure its own medium for more. See `SessionPersistence.track`.

        **It is a constructor argument, and not an attribute that is set
        afterwards.** A store queues a session when the session is published:
        `session/created` reaches `track` synchronously. A boundary that is
        assigned one line after construction is one line too late. The window
        is invisible, because the store only writes more than it needed to.

        Three callers declare it, and they mean two different numbers.
        `resume_session` and `stored_session` pass the length of the stored
        log. `SessionStore.create` passes the inherited prefix of a fork. It
        would be wrong to derive one number from the other: the
        `header.seed_length` of a resumed fork is its *original* fork boundary.

        **It is not `first_live_seq`.** A resume seeds the stored events *plus*
        the repair closers. The closers are in the log and have never been
        written. So only the caller knows what a store holds, and the number is
        never inferred from what is present at `track` time.

        It is a plain attribute and not a header field because it describes the
        relationship of *this process* to *one* store, and not the session.
        `seed_length` beside it is the durable fork boundary and travels with
        the log. This value is not durable and is not a property of the log.
        """
        self.first_live_seq = len(self._log)
        """The first seq that was appended IN THIS PROCESS.

        Events below it entered through construction (replay, fork, resume)
        and were never published on the `session/event` firehose. A consumer
        that replays the log as a substitute for publication starts here. It is
        distinct from `header.seed_length`, which is the durable *fork lineage*
        boundary. The constructor seed of a resumed session is its whole stored
        log, while its header still carries the original fork value.
        """

        base = header or SessionHeader(id=session_id, created_at=now_ms())
        self.header = base.validated(session_id)

    # -------------------------------------------------------------- identity --

    @property
    def id(self) -> str:
        return self.header.id

    def __repr__(self) -> str:
        return f"<Session {self.id} seq={len(self._log)}>"

    # ------------------------------------------------------------------- log --

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """An immutable snapshot of the log, reused until the next append."""
        if self._events_snapshot is None:
            self._events_snapshot = tuple(self._log)
        return self._events_snapshot

    def own_events(self) -> tuple[SessionEvent, ...]:
        """The events past the inherited seed (`SessionHeader.first_own_seq`)."""
        return self.events_from(self.header.first_own_seq)

    def events_from(self, index: int, limit: int | None = None) -> tuple[SessionEvent, ...]:
        """The events appended at or after `index`, at most `limit` of them.

        This is the accessor for an incremental fold. `events` caches one
        snapshot of the *whole* log and rebuilds it whenever the log grew. That
        is right for a reader that wants all of the log. It is quadratic for a
        reader that runs for each event and only looks at the tail. A
        `SessionFoldCache` extender reads through here.

        `limit` exists because a slice of the result copies the tail *first*. A
        pager that took 2048 events from a 50 000-event log built a
        50 000-element tuple and discarded 48 000 of it, one time per page.
        That is quadratic in the length of the session that it pages.
        """
        stop = len(self._log) if limit is None else index + limit
        return tuple(self._log[index:stop])

    @property
    def seq(self) -> int:
        """The sequence number of the next event. It is always the log length (A1)."""
        return len(self._log)

    def at(self, seq: int) -> SessionEvent | None:
        """The event with this sequence number, or `None` if there is none.

        A1 makes this a lookup and not a search: `seq` is assigned as the log
        length at append, so it *is* the index. The accessor exists because the
        two obvious spellings are both wrong for a caller that runs one time
        per event. `events[seq]` rebuilds the whole-log cache, and
        `events_from(seq)` copies the entire tail. Readers are also meant to
        follow the `source_event_seqs` link.
        """
        return self._log[seq] if 0 <= seq < len(self._log) else None

    @property
    def surface(self) -> SurfaceManager:
        return self._surface

    @property
    def disposed(self) -> bool:
        """Whether the store that held this session has let it go."""
        return self._on_disposed is None

    def when_disposed(self, callback: Callable[[Session], None]) -> bool:
        """Register `callback(self)` to run one time, when the store that holds
        this session lets it go (`SessionStore.dispose`). After the store has
        let it go, the answer is `False` and nothing is kept.

        The session is the scope of what a row keeps about it (`SessionTable`).
        The release is therefore registered on the session and not on the row.
        A table needs no context to follow, and a second request with the same
        callback is one registration, not two.
        """
        if self._on_disposed is None:
            return False
        self._on_disposed[callback] = None
        return True

    def mark_disposed(self) -> tuple[Callable[[Session], None], ...]:
        """Record that the store has let this session go. Return the callbacks
        that waited for that, for the store to run. Only `SessionStore.dispose`
        calls it."""
        waiting = tuple(self._on_disposed or ())
        self._on_disposed = None
        return waiting

    def observe(self, observer: SessionObserver) -> Callable[[], None]:
        """Subscribe to the post-commit append feed.

        Each observer is contained separately. A failing observer is logged. It
        cannot remove a committed event, and it cannot stop the observers after
        it from seeing the event.
        """
        self._observers = (*self._observers, observer)

        def off() -> None:
            self._observers = tuple(o for o in self._observers if o is not observer)

        return off

    def _append(
        self,
        event_type: str,
        data: Mapping[str, JsonValue],
        surface: SurfaceIntent | None = None,
    ) -> SessionEvent:
        """Append one event and notify observers synchronously. Call it **through
        a writer** (`ph.session.writers`), which is the only caller outside this
        package (T6).

        `data` is a `Mapping` of JSON values. The type states the producer's
        half of A1. A payload that carries a `Path`, a dataclass or a set is a
        type error at the call. Without the type, `freeze_json_value` would
        raise at runtime. The runtime gate stays, because a type is not a proof
        about a value that arrived as `Any`. But a producer that can be checked
        is checked.

        The hot path never blocks on I/O. Persistence reads what it owes from
        this log on `session/flush`.

        An event is `ignorable` when a *different* build that does not know its
        type may skip it. That is a property of the type. It is read from the
        vocabulary and not passed here, so no two call sites can disagree about
        one type.

        **The type must be one that this build reads** (F11): a type of ph-core,
        or one that a package declared with `declare_log_type`. The read path
        refuses an unknown required type on every seed. A write path that
        accepted one wrote a log that could not be resumed anywhere. The
        `Mapping` check in `_event` closes the same asymmetry for payloads.

        :raises UnknownEventTypeError: when `event_type` is in no vocabulary.
        :raises InvalidJsonValueError: when `data` is not losslessly JSON, or
            is not a JSON **object**.
        :raises SurfaceError: when the surface metadata is wrong for this type.
        :raises RuntimeError: when re-entered during publication.
        """
        return self._commit(self._event(event_type, data, surface, len(self._log)))

    @contextmanager
    def batch(self) -> Iterator[SessionBatch]:
        """Append several events as one: all of them land, or none of them (P10-14).

        ```python
        with session.batch() as batch:
            _LOG.append(batch, "compaction/summarized", accounting)
            _LOG.append(batch, "user/message", summary, SurfaceIntent(...))
        ```

        A batch is for records that have meaning only together, such as an
        accounting record and the replacement that it describes. If they are
        appended one at a time and the surface refuses a later member, the
        earlier members stay in the log and describe a replacement that never
        landed.

        **Nothing is committed until the block exits.** Each member is stamped
        when a writer appends it to the batch (`SessionBatch._append`). Its seq
        is its place in the batch, so a later member may cite an earlier one.
        On exit, all members are planned against the surface together
        (`SurfaceManager.validate_batch`). Only if all of them pass are they
        pushed and published, in order, under one reentrancy guard, so no
        observer can append between two members. An exception inside the block,
        a refused type or payload, or a refused plan pushes nothing. That is
        all of in-process failure: there is nothing to roll back, because
        nothing was committed.

        **Synchronous by contract.** An `await` inside the block lets another
        task append. The members were then stamped against a log that has
        moved. The whole batch is refused, and it does not land at seqs that
        belong to another writer. The rule that the block awaits nothing is not
        enforced. Only its consequence is caught.

        Batches do not nest: one batch at a time for each session.

        :raises RuntimeError: when a batch is already open, when the log moved
            while this one was open, or when re-entered during publication.
        :raises SurfaceError: when the surface transition of any member is
            refused.
        """
        if self._batch is not None:
            raise RuntimeError("a session batch cannot be opened inside another")
        batch = self._batch = SessionBatch(self, len(self._log))
        try:
            yield batch
        finally:
            self._batch = None
        self._commit_batch(batch)

    def _event(
        self,
        event_type: str,
        data: Mapping[str, JsonValue],
        surface: SurfaceIntent | None,
        seq: int,
    ) -> SessionEvent:
        """Build the event that `_append` would commit at `seq`, with the same refusals."""
        if not is_known(event_type):
            raise UnknownEventTypeError(
                f'"{event_type}" is not a session event type this build can read back; '
                "declare it with ph.session.declare_log_type"
            )
        if not isinstance(data, Mapping):
            # `_EventWire` refuses a non-object payload on the way *in* from
            # disk, and `SessionEvent.data` is declared a `JsonObject`. Without
            # this check, the write path and the read path disagree. A producer
            # that reached here as `Any` (a test tree outside mypy, or a plugin
            # built against an older signature) appends a list. The bytes reach
            # disk, and the session cannot be loaded at the next resume. A log
            # that cannot be reconstructed is the one failure that A1 exists to
            # prevent. So the write path refuses what the read path refuses, in
            # its own vocabulary.
            raise InvalidJsonValueError(
                "", f"an event payload must be a JSON object, not {type(data).__name__}"
            )
        return SessionEvent(
            type=event_type,
            seq=seq,
            time=now_ms(),
            data=freeze_json_value(data),
            source_event_seqs=None if surface is None else surface.source_event_seqs,
            surface_op=None if surface is None else surface.surface_op,
            ignorable=is_ignorable(event_type),
        )

    def admit(self, event: SessionEvent) -> SessionEvent:
        """Append an event that already carries its `seq` and `time`. This is the
        path of a replica.

        `_append` is for the process that *owns* a log: it assigns the seq and
        stamps the clock. A front end that mirrors the session of a daemon over
        the wire owns nothing. It receives events that the daemon already
        stamped, and it must keep them as they are. To stamp `time` again would
        put the clock of this client on the record of the daemon, and the
        timings of the trajectory read `event.time`. So the mirror admits and
        does not append. The mirror is a real `Session`, with the same surface
        fold, the same `stale()` check and the same `cursor_of` as the log that
        it copies. It is not a list of events that is rebuilt into a `Session`
        at every read.

        `admit` is held to the rules of the seed path, through the function of
        the seed path. `_readmit` refuses a seq that is not the next index: a
        replica that skipped a frame must stop, and must not admit a log with a
        hole in it. `_readmit` also refuses an unrecognized required type. Then
        the surface is validated and the event is published. `_append` uses the
        same tail, so an observer cannot tell which of the two methods an event
        came through.

        :raises ValueError: when `event.seq` is not the next index, when its type
            is unknown and not `ignorable`, or when it breaks an open batch
            (`_within_batch`).
        :raises SurfaceError: when the surface metadata is wrong for this type.
        :raises RuntimeError: when re-entered during publication.
        """
        admitted = _readmit(event, len(self._log))
        batch_open = _within_batch(self._batch_open, admitted)
        committed = self._commit(admitted)
        self._batch_open = batch_open
        return committed

    def _commit(self, event: SessionEvent) -> SessionEvent:
        """Validate against the surface, push, and publish. This is the one tail
        of `_append` and `admit`.

        Validation comes BEFORE the push: a rejected candidate must leave both
        the log and the surface exactly as they were.

        The re-entrancy guard is here and not on each caller, because
        `_publishing` is the flag of this method. A reentrant commit would land
        a seq inside the publication of another event, so observers would see
        the log grow under them. The guard is stated one time, so a third
        caller cannot arrive without it. The two callers also cannot disagree
        about the rule, which they did for a short time.
        """
        self._refuse_reentry()
        self._surface.validate_next(event)
        self._push((event,))
        return event

    def _commit_batch(self, batch: SessionBatch) -> None:
        """`_commit` for several events that are planned together. See `batch`."""
        self._refuse_reentry()
        if len(self._log) != batch.base:
            raise RuntimeError(
                f"session {self.id} moved from seq {batch.base} to {len(self._log)} while a "
                "batch was open; its members were stamped against a log that no longer "
                "exists, so none of them is appended"
            )
        events = batch.events
        if not events:
            return
        if len(events) > 1:
            # Every member says which batch it is in (P10-15). A reader can then
            # tell a batch that a torn write cut short from a whole one. A single
            # event is not stamped, because there is nothing to keep together.
            ref = BatchRef(first=batch.base, count=len(events))
            events = tuple(replace(event, batch=ref) for event in events)
        self._surface.validate_batch(events)
        self._push(events)

    def _refuse_reentry(self) -> None:
        if self._publishing:
            raise RuntimeError(
                "session append cannot reenter while another append is being published"
            )

    def _push(self, events: Sequence[SessionEvent]) -> None:
        """Push and publish validated events. An observer sees each event when
        the log ends at that event."""
        self._publishing = True
        try:
            for event in events:
                self._log.append(event)
                self._events_snapshot = None
                for observer in self._observers:
                    try:
                        observer(self, event)
                    except Exception:
                        log.exception(
                            "ph.session: observer failed for %s at seq %s", event.type, event.seq
                        )
        finally:
            self._publishing = False

    # --------------------------------------------------------------- folds --

    def request_header(self) -> EpochHeader | None:
        """The header that the NEXT request is compared against."""
        return self.projection("request/header", parse_request_header)

    def request_context(self) -> RequestContext | None:
        """The latest resolved route metadata, folded incrementally."""
        return self.projection("request/context", parse_request_context)

    # ------------------------------------------------------------- derivation --

    def stale(self) -> list[str]:
        """Every incremental projection of the log that no longer equals its fold (I6).

        There are three. Each is maintained lazily, and each can tell a reader
        about a log that no longer exists:

        * `events`, a snapshot that an append invalidates.
        * The surface (`SurfaceManager`), which is folded from where it stopped.
        * `derive_messages`, which is memoized per node. I3 compares every
          request against this projection, so `agent-loop-invariant` would pass
          a drift here.

        `fold_surface` is the canonical replay of the same rules over the whole
        log. Its docstring says that an external reconstructor must reach the
        same nodes. This method checks that claim and does not assume it.

        The check is here and not in the invariant row that declares it, for
        the reason that `ToolRuntime.stale_views` gives. These caches are
        private to this class, and a rename silently disables a check that is
        written against `_log` from outside. Every way to trip the check is a
        writer that bypassed `_append`. An append is what invalidates these
        caches, and only a check inside the class can detect a write that
        skipped it.

        Each call is O(events), because it runs the whole fold. So the check is
        polled, and never run on append.
        """
        found: list[str] = []
        if len(self.events) != len(self._log):
            found.append(
                f"session {self.id}: the events snapshot holds {len(self.events)} where the "
                f"log holds {len(self._log)}"
            )
        canonical = fold_surface(self._log).nodes
        if self._surface.nodes != canonical:
            found.append(
                f"session {self.id}: the surface projects {len(self._surface.nodes)} node(s) "
                f"where its fold gives {len(canonical)}"
            )
            # The derivation is memoized *over* the surface. When the surface
            # is wrong, the derivation is wrong for the same reason. To force
            # it would index the log at nodes that the log no longer has.
            return found
        derived = tuple(
            message
            for message in (derive_event_message(self._log[seq]) for seq in canonical)
            if message is not None
        )
        if self.derive_messages() != derived:
            found.append(
                f"session {self.id}: derive_messages holds {len(self.derive_messages())} "
                f"message(s) where a fresh derivation gives {len(derived)}"
            )
        return found

    def derive_messages(self) -> tuple[Message, ...]:
        """The LLM message history, derived from the ordered surface.

        The surface is the single source of derived history. An event with no
        `surfaceOp` (a chunk, a turn boundary) is correctly absent. A compaction
        `replace` removes the shadowed nodes from the derivation and leaves
        them in the log.

        The result is cached per node. A surface rewrite increases
        `replace_generation` and causes a rebuild. Ordinary appends cost O(new
        nodes). The result is a tuple, so the copy that a holder has cannot
        grow.
        """
        surface = self._surface
        generation = surface.replace_generation
        if generation != self._derived_generation:
            self._derived = ()
            self._derived_nodes = 0
            self._derived_generation = generation
        fresh = surface.nodes_from(self._derived_nodes)
        if fresh:
            projected = (derive_event_message(self._log[seq]) for seq in fresh)
            self._derived = (*self._derived, *(m for m in projected if m is not None))
            self._derived_nodes += len(fresh)
        return self._derived

    def transcript(self) -> tuple[Message, ...]:
        """The human transcript: every append-origin message, compaction or not."""
        return derive_transcript(self._log)

    # -------------------------------------------------------------- helpers --

    def latest(self, event_type: str) -> SessionEvent | None:
        """The most recent event of one type, folded incrementally.

        Every "current policy" question has this shape: approval policy,
        sandbox mode, permission preset. A check that runs on every call must
        not answer it with a scan of a log that is mostly `assistant/chunk`
        events.
        """
        return self.projection(event_type, _the_event)

    def records_of(self, seq: int) -> tuple[SessionEvent, ...]:
        """The records that explain the replacement at `seq`, in log order.

        The producer of the replacement wrote these records through the
        revision functions (`ph.session.revise`), which stamped them with the
        replacement. The result is empty for a node that no revision put there.
        It is also empty for a replacement that was written before the revision
        functions stamped their records.

        The index is built as the log grows, so a reader that asks about every
        node pays one time.
        """
        return self._revisions.read(self._log, seq)

    def history(self, seq: int) -> tuple[SessionEvent, ...]:
        """Every replacement that led to node `seq` as it is now, oldest first.

        The result holds the node itself when it is a replacement, and every
        replacement among the nodes that it stands for, followed through
        `shadowed_by`. Git keeps one line of history per file. This history
        merges: a summary of five revised nodes has their five revisions behind
        it, and its own.

        The result is kept per node after it is worked out, because what a node
        stands for is fixed when the node is written.
        """
        return self._revisions.history(self.at, seq)

    def projection[T](self, event_type: str, parse: Callable[[SessionEvent], T | None]) -> T | None:
        """Any "latest event of type X, read as Y" question, folded incrementally.

        **This is the registry that `latest` already had, opened to the seams.**
        Reported usage first arrived here as a method, a `__slots__` entry and a
        private parser. `Session` had to learn `TokenUsage` so that a *seam*
        could read it. That is the Consumer editing the Definition (I5), and
        every next metric would have repeated the shape. A seam now keeps its
        own parser and asks with it:

            usage = session.projection("assistant/message", reported_usage)

        **The parser is half the key, so nothing needs a name.** This method
        first took a `name`, and then a guard that checked that two callers
        with one name meant the same thing. The guard computed
        `(event_type, parse)`, the real identity, only to police a string that
        a human typed. The key is now that pair. That removes the guard, the
        collision error, the "prefix it with the row that owns it" convention
        and the constant that each caller kept for its own name. Two rows
        cannot collide, because two rows are not the same function.

        **So `parse` must be stable across calls**, and every caller already
        passes a stable one. A module-level function or a bound method is the
        same key each time. A frozen dataclass hashes by value, which is how
        `workspace.latest_checkpoint` gets one fold *per agent* from one parser
        class, where a name would have had to carry the agent id. An inline
        `lambda` is the one thing that does not work. It is a new key on each
        call, so it would build a fold on each call and parse the log again
        each time. `_the_event` exists for exactly that reason.

        When `parse` returns `None`, it means "this event carries nothing of
        that", and the fold keeps what it had. See `fold_latest`.
        """
        key = (event_type, parse)
        fold = self._latest.get(key)
        if fold is None:
            fold = self._latest[key] = _LatestFold[object](event_type, parse)
        # The cast is sound because the parser is in the key. The fold found
        # here was built from *this* parse, so it holds what this parse returns.
        return cast("T | None", fold.read(self._log))

    @property
    def last_event(self) -> SessionEvent | None:
        """The most recently appended event, or `None` for an empty log.

        This is the unfiltered form of `last_event_of`. It is the accessor for a
        "when did this session last do anything" question (the passivation
        sweeper of P5-05). `events[-1]` answers that too, but it materializes a
        snapshot of the entire log to read one element, and the sweeper asks
        the question of every root on every pass.
        """
        return self._log[-1] if self._log else None

    def select(self, *patterns: str) -> tuple[SessionEvent, ...]:
        """Every event under one or more namespace selectors, in log order (P6-33).

        `session.select("workspace")` is every `workspace/*` type.
        `session.select("workspace/acquired")` is that one type. The vocabulary
        is `log`, so a bare pattern needs no prefix. `bus:tools` is **refused**
        and not answered with an empty result, because an empty result would
        read as "there are none".

        `select` is namespace-aware, where `last_event_of` is exact: that method
        takes whole type names and answers with the newest event. Both stay,
        because "every workspace record" must not be spelled as a list of
        literals that a new type would silently be missing from.

        With no patterns, the result is the whole log.
        """
        selectors = parse_all(patterns, vocabulary="log")
        return tuple(event for event in self._log if matches_any(event.type, selectors))

    def last_event_of(self, *types: str) -> SessionEvent | None:
        """The most recent event of any of `types`. For one type, prefer `latest()`."""
        for event in reversed(self._log):
            if event.type in types:
                return event
        return None


class SessionBatch:
    """The events that one `Session.batch()` block will append. See that method.

    A batch holds stamped events and nothing else. It never touches the log, so
    a batch that is dropped (an exception out of the block) needs no cleanup.
    """

    __slots__ = ("_events", "_session", "base")

    def __init__(self, session: Session, base: int) -> None:
        self._session = session
        self._events: list[SessionEvent] = []
        self.base = base
        """The seq that the first member takes: the length of the log when the batch opened."""

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """The members as stamped. The log holds them as committed, with the
        batch membership on each."""
        return tuple(self._events)

    def _append(
        self,
        event_type: str,
        data: Mapping[str, JsonValue],
        surface: SurfaceIntent | None = None,
    ) -> SessionEvent:
        """Stamp an event for this batch. The event is refused as `Session._append`
        would refuse it. Call it through a writer, as for every write (T6).

        The event is returned so that a later member can cite its seq. It is
        **not yet in the log**, and it never enters the log if the batch is
        refused.
        """
        if self._session._batch is not self:
            # The block has exited, whether the batch landed or not. A member
            # stamped now would belong to no commit, so it is refused and not
            # silently dropped.
            raise RuntimeError("this session batch has closed; open another to append")
        event = self._session._event(event_type, data, surface, self.base + len(self._events))
        self._events.append(event)
        return event


def _within_batch(open_batch: BatchRef | None, event: SessionEvent) -> BatchRef | None:
    """Hold `event` to the batch that the log is inside, and return the batch that
    the log is inside after it.

    The members of a batch are contiguous, and each carries the same ref. The
    first member is at `first`. No unstamped event and no other batch comes
    before the last member. The check runs for every event that arrives already
    stamped (seed, fork, resume, a replica). The `batch()` of an owner stamps
    its members correctly by construction, so it needs no check.

    :raises ValueError: when the event breaks a batch or starts one out of place.
    """
    ref = event.batch
    if open_batch is not None and ref != open_batch:
        raise ValueError(
            f"seq {event.seq} interrupts the batch at seq {open_batch.first} "
            f"({event.seq - open_batch.first} of {open_batch.count} events)"
        )
    if ref is None:
        return None
    if open_batch is None and ref.first != event.seq:
        raise ValueError(
            f"seq {event.seq} claims the batch at seq {ref.first}, which it does not continue"
        )
    return None if event.seq == ref.last else ref


def _readmit(source: SessionEvent, index: int) -> SessionEvent:
    """Hold a seeded event to the acceptance rules that a live append meets.

    The refusal of unknown types is here, on the one path that every seed takes
    (fork, resume, replay, a replica), and not in one storage backend. An
    unrecognized *required* event may change how the rest of the log is read.
    To skip it would reconstruct a wrong session, not a partial one.
    """
    if source.seq != index:
        raise ValueError(
            f"seed event at index {index} has seq {source.seq} (expected {index}); "
            "seed must be contiguous from 0"
        )
    if not is_known(source.type) and not source.ignorable:
        raise ValueError(
            f'seed event at index {index} has unrecognized required type "{source.type}"; '
            "this log was written by a newer harness and reading it here would "
            "reconstruct a wrong session"
        )
    return source.readmitted()
