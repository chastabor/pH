"""The append-only session log and its derived model history.

Three invariants live in this file:

* **I3 — model-visible means logged.** Anything reaching a model request is
  reconstructable from `Session.events` through `derive_messages()`.
* **I4 — the log is append-only; the surface is what changes.** Compaction and
  offload append a `replace`; nothing rewrites history.
* **A1 — `seq == len(log)`.** The contiguity contract every backend relies on.

`append` is the acceptance boundary: it validates and freezes the payload,
validates the surface transition *before* the push, and only then publishes.
A failure therefore leaves the log and the surface untouched, and a publication
failure cannot un-append what is already committed.

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
from .surface import SurfaceManager, fold_surface

__all__ = ["Session", "SessionBatch", "SessionHeader", "SessionObserver", "cwd_tag", "family_for"]

CWD_TAG_LENGTH = 6
"""Hex characters of the cwd digest that tag a lineage directory.

Six — 24 bits — because this is a **filter, not an identity**: it lets a listing
skip a directory without opening anything in it, and the header's own `cwd` is
what confirms a match. Two repos that collide cost one extra directory scan and
no wrong answer, so buying more characters would buy nothing and make every path
longer.
"""


def cwd_tag(cwd: str) -> str:
    """The short digest of a working directory, as a lineage directory wears it."""
    return sha256(cwd.encode("utf-8")).hexdigest()[:CWD_TAG_LENGTH]


def family_for(session_id: str, cwd: str | None) -> str:
    """The lineage directory for a new root: `<cwd-tag>-<id>`, or bare `<id>`.

    **Bare when there is no cwd**, which is the honest answer rather than a
    placeholder tag: a session that belongs to no directory should not be found
    by a search for one, and an unfiltered listing still sees it.
    """
    return f"{cwd_tag(cwd)}-{session_id}" if cwd else session_id


log = logging.getLogger("ph.session")

SessionObserver = Callable[["Session", SessionEvent], None]


SessionKind: TypeAlias = Literal["fork", "segment"]
"""Why a log has a parent: a **branch**, or the same session continued.

Named because six sites spell it — the header, `SessionStore._branch`, two test
builders and two listing rows — and a bare `str` at any of them turns a typo
("segement") into a value that validates and never matches.
"""


class SessionHeader(WireModel):
    """Immutable storage metadata, kept *outside* the conversation log.

    A storage concern, not replayable conversation state — which is why it lives
    beside the log rather than in it. Every constraint is on the field, so a
    header is valid the moment it exists; the one cross-object check — does it
    belong to *this* session — is `validated()`.
    """

    version: int = SESSION_FORMAT_VERSION
    id: str
    created_at: NonNegativeInt
    cwd: str | None = None
    parent_session: str | None = None
    seed_length: NonNegativeInt | None = None
    """How many leading events were inherited through a seed.

    Persisting this boundary is what lets resume and replay tell parent history
    from child work — a fork otherwise looks like a session that simply started
    with a long conversation.
    """
    family: str = Field(
        # `data.get`, not `data["id"]`: a stored header with no `id` at all is
        # something a listing meets rather than something that cannot happen, and
        # a `KeyError` out of a default factory is not a `ValidationError` — so
        # it escaped every caller that was catching one, and one hand-edited line
        # took the whole `stored()` listing down with it. Empty fails
        # `min_length` instead, which is the refusal this field already has.
        default_factory=lambda data: family_for(as_str(data.get("id")), data.get("cwd")),
        min_length=1,
    )
    """Which lineage this log belongs to, and **where that lineage was worked**.

    The directory a log lives in: `sessions/<family>/<id>.jsonl`. Every fork, segment
    and subagent beneath a root inherits its value, so one conversation and everything
    it spawned is one directory — which is what lets "is anything in here orphaned" be
    answered by listing one directory.

    **A root's is `<cwd-tag>-<id>`** (format 1). The tag is six hex of the working
    directory's digest, so "which sessions belong to this repo" is answerable from
    the *directory names alone* — a listing skips a whole lineage without opening a
    file, where before it read the header of every log in the store. It is a filter
    and not an identity: 24 bits collide, and the header's own `cwd` is what confirms
    a match, so a collision costs a directory scan and never a wrong row.

    The tag is fixed when the root is created and inherited unchanged, which is the
    only behaviour that keeps a lineage in one place — a session `cd`s nowhere, and
    a fork of a session worked in `/a` belongs with it even if the person has since
    moved.

    **Never absent.** `min_length=1` makes an explicit empty string a refusal rather
    than something quietly rewritten. Stored rather than derived, because deriving it
    means walking `parent_session` to the root, and walking needs paths, and the path
    is what the family is for.
    """

    kind: SessionKind | None = None
    """Why this log has a parent: a **branch**, or the same session continued.

    Structurally the two are identical — `roll` is `fork` at the tip — so nothing on
    disk could tell them apart, and every reader of `parent_session` had to guess
    "branch". `None` means a **root**; every log with a `parent_session` has a kind,
    set by whichever call made it.

    On the **child's** header, where the backward link already lives, because that is
    the half every consumer reads from a one-line peek. The forward half has to stay
    an event (`session/segmented`) for a mechanical reason: JSONL writes a header once
    and never rewrites it, so a `continued_by` on the *parent* would be durable under
    Turso and silently lost under JSONL.
    """

    origin: Literal["subagent"] | None = None
    delegation_depth: NonNegativeInt | None = None
    agent_preset: str | None = None

    @field_validator("version")
    @classmethod
    def _current_format_only(cls, value: int) -> int:
        # Checked on load and never migrated: while pH is unreleased an
        # incompatible log is rejected rather than half-understood.
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

    def validated(self, session_id: str) -> SessionHeader:
        if self.id != session_id:
            raise ValueError(
                f'session header id "{self.id}" does not match session id "{session_id}"'
            )
        return self


def _the_event(event: SessionEvent) -> SessionEvent:
    """The parser `latest` folds with: the event itself, read as itself.

    Module level, and that is the whole point. As an inline `lambda` it was a
    new object on every `latest()` call — which is a new registry key, a new
    fold and a fresh walk of the log each time, per `Session.projection`.
    """
    return event


type _Key = tuple[str, Callable[[SessionEvent], object]]
"""What identifies a fold: the event type it watches and the parser that reads it.

Both halves are load-bearing. The type alone would collide the moment two seams
read different things off one event; the parser alone cannot distinguish two
folds of the same function over different types."""


class _LatestFold[T]:
    """An incrementally maintained "latest event of type X" over a growing log."""

    __slots__ = ("_event_type", "_parse", "_seen", "value")

    def __init__(self, event_type: str, parse: Callable[[SessionEvent], T | None]) -> None:
        self._event_type = event_type
        self._parse = parse
        self._seen = 0
        self.value: T | None = None

    def read(self, log: list[SessionEvent]) -> T | None:
        """The projection as of now, parsing only what has arrived since.

        A cursor and `fold_latest`, not a second copy of it: this held its own
        backwards walk for a while so it could avoid the `log[self._seen:]` copy
        a slice makes per read per fold — one loop's worth of duplication to
        dodge one allocation, with the rule it implements documented in the
        other file. `since=` gets both.
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
        "_publishing",
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
        self._publishing = False
        self._batch: SessionBatch | None = None
        """The batch whose block is open, if any — one at a time, and the only one
        whose `append` still stamps."""
        self._batch_open: BatchRef | None = None
        self._derived: tuple[Message, ...] = ()
        self._derived_nodes = 0
        self._derived_generation = 0
        self._latest: dict[_Key, _LatestFold[object]] = {}

        if seed is not None:
            # Validate the seed to the SAME invariants `append` enforces. A
            # replay or fork must not be able to construct a live log that no
            # backend could store — otherwise a bad seed surfaces later as a
            # flush rejection or, worse, as a silent divergence from disk.
            #
            # Through `admit`, which is that rule with a name on it: a seed event
            # is already stamped, exactly like one arriving over a wire. Written
            # inline here once, which made the rule two statements a screen apart
            # — and this loop the copy without the docstring. The publish inside
            # is a no-op here by construction: `_observers` is empty until the
            # constructor returns, so nothing can be watching a session that does
            # not exist yet.
            for source in seed:
                try:
                    self.admit(source)
                except ValueError as error:
                    # The index the caller can act on. `_readmit` names it too, so
                    # only the surface's errors gain it — theirs say what is wrong
                    # with the event, this says which one.
                    raise ValueError(
                        f"invalid seed event at index {len(self._log)}: {error}"
                    ) from error
            if self._batch_open is not None:
                # **A seed may not end inside a batch** (P10-15). A reader drops a
                # batch a torn write cut short before it gets here, so an
                # unfinished one reaching a seed is damage, and seeding it would
                # hand every reader half of something that only means anything
                # whole. The check is the seed path's alone: a replica admitting
                # a batch member by member is mid-batch between frames, legitimately.
                opened = self._batch_open
                raise ValueError(
                    f"the seed ends inside the batch at seq {opened.first} "
                    f"({len(self._log) - opened.first} of {opened.count} events)"
                )

        self.durable_length = durable
        """How many leading events a **store already holds**; 0 unless declared.

        Read by a backend to tell what it still owes from what is already
        written: a store queues at least `events[durable_length:]`, and may
        measure its own medium for more — see `SessionPersistence.track`.

        **A constructor argument, not an attribute set afterwards.** Publishing a session
        is what makes a store queue it — `session/created` reaches `track` synchronously
        — so a boundary assigned one line after construction is one line too late, and
        the window is invisible: the store simply writes more than it needed to.

        Two callers declare it and they mean different numbers: `resume_session` passes
        the stored log's length, `SessionStore.create` passes a fork's inherited prefix.
        Deriving one from the other would be wrong — a resumed fork's
        `header.seed_length` is its *original* fork boundary.

        **Not `first_live_seq`.** A resume seeds the stored events *plus* the repair
        closers and `session/end-seed`, which are in the log and have never been
        written — so what a store holds is a number only the caller knows, never one
        inferred from what happens to be present at `track` time.

        A plain attribute rather than a header field because it describes *this
        process's* relationship to *one* store, not the session. `seed_length` next door
        is the durable fork boundary and travels with the log; this is neither durable nor
        a property of the log.
        """
        self.first_live_seq = len(self._log)
        """The first seq appended IN THIS PROCESS.

        Events below it entered through construction — replay, fork, resume —
        and were never published on the `session/event` firehose, so a consumer
        replaying the log as a publication substitute starts here. Distinct from
        `header.seed_length`, which is the durable *fork lineage* boundary: a
        resumed session's constructor seed is its whole stored log, while its
        header still carries the original fork value.
        """

        base = header or SessionHeader(id=session_id, created_at=now_ms())
        self.header = base.validated(session_id)

        # Appended here so the marker is already in `events` when a backend
        # captures the creation seed: no load-time write. A seed already ending
        # in one is not re-marked, so repeatedly opening a cold session does not
        # grow its log per open.
        if seed is not None and (not self._log or self._log[-1].type != "session/end-seed"):
            self.append("session/end-seed", {})

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

    def events_from(self, index: int, limit: int | None = None) -> tuple[SessionEvent, ...]:
        """The events appended at or after `index`, at most `limit` of them.

        The accessor an incremental fold wants. `events` caches one snapshot of
        the *whole* log and rebuilds it whenever the log grew — right for a
        reader that wants all of it, and quadratic for one that runs per event
        and only ever looks at the tail. A `SessionFoldCache` extender reads
        through here.

        `limit` exists because slicing the result copies the tail *first*: a
        pager taking 2048 events from a 50 000-event log built a 50 000-element
        tuple to discard 48 000 of it, once per page, which is quadratic in the
        length of the session it is paging.
        """
        stop = len(self._log) if limit is None else index + limit
        return tuple(self._log[index:stop])

    @property
    def seq(self) -> int:
        """The next event's sequence number — always the log length (A1)."""
        return len(self._log)

    def at(self, seq: int) -> SessionEvent | None:
        """The event with this sequence number, or `None` if there is none.

        A1 is what makes this a lookup rather than a search: `seq` is assigned as
        the log length at append, so it *is* the index. The accessor exists
        because the two obvious spellings are both wrong for a caller running
        once per event — `events[seq]` rebuilds the whole-log cache, and
        `events_from(seq)` copies the entire tail — and `source_event_seqs` is a
        link readers are meant to follow.
        """
        return self._log[seq] if 0 <= seq < len(self._log) else None

    @property
    def surface(self) -> SurfaceManager:
        return self._surface

    def observe(self, observer: SessionObserver) -> Callable[[], None]:
        """Subscribe to the post-commit append feed.

        Observers are invoked with per-listener containment: a failing observer
        is logged and cannot un-append a committed event, nor stop the ones
        after it from seeing it.
        """
        self._observers = (*self._observers, observer)

        def off() -> None:
            self._observers = tuple(o for o in self._observers if o is not observer)

        return off

    def append(
        self,
        event_type: str,
        data: Mapping[str, JsonValue],
        surface: SurfaceIntent | None = None,
    ) -> SessionEvent:
        """Append one event and synchronously notify observers.

        `data` is a `Mapping` of JSON values, and that is the producer's half of
        A1 said in the type: a payload carrying a `Path`, a dataclass or a set is
        an error here, at the call, where `freeze_json_value` would have raised
        at runtime. The runtime gate stays — a type is not a proof about a value
        that arrived as `Any` — but a producer that can be checked is.

        The hot path never blocks on I/O — persistence reads what it owes off
        this log on `session/flush`.

        Whether the event is `ignorable` — skippable by a *different* build that
        does not know its type — is a property of the type, read from the
        vocabulary rather than passed here, so no two call sites can disagree
        about one type.

        **The type must be one this build reads** (F11): ph-core's own, or one a
        package declared with `declare_log_type`. The read door refuses an
        unknown required type on every seed, so a write door that accepted one
        wrote a log that resumed nowhere — the same asymmetry the `Mapping` check
        below closes for payloads, one field over.

        :raises UnknownEventTypeError: when `event_type` is in no vocabulary.
        :raises InvalidJsonValueError: when `data` is not losslessly JSON, or
            is not a JSON **object**.
        :raises SurfaceError: when the surface metadata is wrong for this type.
        :raises RuntimeError: when re-entered during publication.
        """
        return self._commit(self._event(event_type, data, surface, len(self._log)))

    @contextmanager
    def batch(self) -> Iterator[SessionBatch]:
        """Append several events as one: all of them land, or none of them do (P10-14).

        ```python
        with session.batch() as batch:
            batch.append("compaction/summarized", accounting)
            batch.append("user/message", summary, SurfaceIntent(...))
        ```

        For records that only mean something together — an accounting record and
        the replacement it describes. Appended one at a time, a later member the
        surface refuses leaves the earlier ones in the log, describing a
        replacement that never landed.

        **Nothing is committed until the block exits.** Each member is stamped
        when `batch.append` is called — its seq is its place in the batch, so a
        later member may cite an earlier one — and on exit every member is planned
        against the surface together (`SurfaceManager.validate_batch`); only if
        all of them pass are they pushed and published, in order, under one
        reentrancy guard, so no observer can append between two members. An
        exception inside the block, a refused type or payload, or a refused plan
        pushes nothing. That is the whole of in-process failure: there is nothing
        to roll back, because nothing was committed.

        **Synchronous by contract.** An `await` inside the block lets another
        task append, and the members were stamped against a log that has since
        moved; the batch is then refused whole rather than landed at seqs that
        belong to someone else. Not enforced: that the block awaits nothing —
        only its consequence is caught.

        Not nested: one batch at a time per session.

        :raises RuntimeError: when a batch is already open, when the log moved
            while this one was, or when re-entered during publication.
        :raises SurfaceError: when any member's surface transition is refused.
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
        """Build the event `append` would commit at `seq`, refusing what it refuses."""
        if not is_known(event_type):
            raise UnknownEventTypeError(
                f'"{event_type}" is not a session event type this build can read back; '
                "declare it with ph.session.declare_log_type"
            )
        if not isinstance(data, Mapping):
            # `_EventWire` refuses a non-object payload on the way *in* from
            # disk, and `SessionEvent.data` is declared a `JsonObject`. Without
            # this the two doors disagree: a producer that reached here as `Any`
            # — a test tree outside mypy, a plugin built against an older
            # signature — appends a list, the bytes reach disk, and the session
            # becomes unloadable at the next resume. A log that cannot be
            # reconstructed is the one failure A1 exists to prevent, so the
            # write door refuses what the read door refuses, in its vocabulary.
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
        """Append an event that already carries its `seq` and `time` — a replica's path.

        `append` is for the process that *owns* a log: it mints the seq and stamps
        the clock. A front end mirroring a daemon's session over the wire owns
        nothing; it receives events the daemon already stamped and must keep them
        as they are — re-stamping `time` would put this client's clock on the
        daemon's record, and the trajectory's timings read `event.time`. So the
        mirror admits rather than appends, and the mirror is a real `Session`: the
        same surface fold, the same `stale()` check, the same `cursor_of` as the
        log it copies, rather than a list of events rebuilt into a `Session` from
        scratch at every read.

        Held to the seed path's rules, through the seed path's function: `_readmit`
        refuses a seq that is not the next index — a replica that skipped a frame
        must stop rather than admit a log with a hole in it — and an unrecognized
        required type. Then the surface is validated and the event published, the
        same tail `append` uses, so an observer cannot tell which door an event
        came through.

        :raises ValueError: when `event.seq` is not `len(self)`, or its type is
            unknown and not `ignorable`.
        :raises SurfaceError: when the surface metadata is wrong for this type.
        :raises RuntimeError: when re-entered during publication.
        """
        admitted = _readmit(event, len(self._log))
        batch_open = _within_batch(self._batch_open, admitted)
        committed = self._commit(admitted)
        self._batch_open = batch_open
        return committed

    def _commit(self, event: SessionEvent) -> SessionEvent:
        """Validate against the surface, push, and publish — `append` and `admit`'s
        one tail.

        Validated BEFORE the push: a rejected candidate must leave both the log
        and the surface exactly as they were.

        The re-entrancy guard is here rather than on each door, because
        `_publishing` is this method's own flag: a reentrant commit would land a
        seq inside another event's publication, so observers would see the log
        grow underneath them. Stated once, so a third door cannot arrive without
        it — and so the two doors cannot disagree about the sentence, which they
        briefly did.
        """
        self._refuse_reentry()
        self._surface.validate_next(event)
        self._push((event,))
        return event

    def _commit_batch(self, batch: SessionBatch) -> None:
        """`_commit` for several events planned together; see `batch`."""
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
            # Every member says which batch it is in (P10-15), so a reader can tell
            # one a torn write cut short from one that is whole. One event is not
            # stamped: there is nothing to keep together.
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
        """Push and publish validated events, each seen with the log ending at it."""
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
        """The header the NEXT request will be compared against."""
        return self.projection("request/header", parse_request_header)

    def request_context(self) -> RequestContext | None:
        """The latest resolved route metadata, folded incrementally."""
        return self.projection("request/context", parse_request_context)

    # ------------------------------------------------------------- derivation --

    def stale(self) -> list[str]:
        """Every incremental projection of the log that no longer equals its fold (I6).

        Three of them, each maintained lazily and each a way for a reader to be
        told about a log that no longer exists: `events` (a snapshot invalidated on
        append), the surface (`SurfaceManager`, folded from where it left off), and
        `derive_messages` (memoized per node — the projection I3 compares every
        request against, so a drift here is one `agent-loop-invariant` would pass).
        `fold_surface` is the canonical replay of the same rules over the whole log;
        the manager's own docstring says an external reconstructor "must reach the
        same nodes", and this is where that is asked rather than assumed.

        Here rather than in the invariant row that declares it, for
        `ToolRuntime.stale_views`' reason: these caches are this class's own secret,
        and a check written against `_log` from outside is one a rename disables
        without anybody noticing. Every trip path is a writer that bypassed `append`
        — which is what these caches exist to be invalidated by, and what only a
        check inside the class can honestly detect.

        O(events) per call — the fold is the point — so polled, never run on append.
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
            # The derivation is memoized *over* the surface, so with the surface
            # wrong it is wrong for the surface's reason — and forcing it would
            # index the log at nodes the log no longer has.
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

        The surface is the single source of derived history: an event with no
        `surfaceOp` (a chunk, a turn boundary) is correctly absent, and a
        compaction `replace` removes the shadowed nodes from the derivation
        while leaving them in the log.

        Cached per node. A surface rewrite bumps `replace_generation` and
        rebuilds; ordinary appends cost O(new nodes). The result is a tuple, so
        a holder's copy structurally cannot grow under them.
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

        The shape every "current policy" question takes — approval policy,
        sandbox mode, permission preset — and one a per-call check must not
        answer by scanning a log that is mostly `assistant/chunk`s.
        """
        return self.projection(event_type, _the_event)

    def projection[T](self, event_type: str, parse: Callable[[SessionEvent], T | None]) -> T | None:
        """Any "latest event of type X, read as Y" question, folded incrementally.

        **The registry `latest` already was, opened to the seams.** Reported
        usage arrived here as a method plus a `__slots__` entry plus a private
        parser, which made `Session` learn `TokenUsage` so that a *seam* could
        read it — the Consumer editing the Definition (I5), and the shape every
        next metric would have repeated. A seam now keeps its own parser and
        asks with it:

            usage = session.projection("assistant/message", reported_usage)

        **The parser is half the key, so nothing has to be named.** This took a
        `name` first, and then a guard to check that two callers sharing a name
        meant the same thing — which computed `(event_type, parse)`, the real
        identity, purely to police a string a human typed. Keying on it directly
        deletes the guard, the collision error, the "prefix it with the row that
        owns it" convention and the constants each caller kept for its own name;
        two rows cannot collide because two rows are not the same function.

        **So `parse` must be stable across calls**, which is what every caller
        already passes: a module-level function or a bound method is the same
        key each time, and a frozen dataclass hashes by value — which is how
        `workspace.latest_checkpoint` gets one fold *per agent* out of one
        parser class, where a name would have had to carry the agent id. An
        inline `lambda` is the one thing that does not work: it is a new key per
        call, so it would build a fold per call and reparse the log each time.
        `_the_event` exists for exactly that reason.

        `parse` returning `None` means "this event carries nothing of that", and
        the fold keeps what it had — see `fold_latest`.
        """
        key = (event_type, parse)
        fold = self._latest.get(key)
        if fold is None:
            fold = self._latest[key] = _LatestFold[object](event_type, parse)
        # Sound because the parser is in the key: the fold found here was built
        # from *this* parse, so what it holds is what this parse returns.
        return cast("T | None", fold.read(self._log))

    @property
    def last_event(self) -> SessionEvent | None:
        """The most recently appended event, or `None` for an empty log.

        The unfiltered form of `last_event_of`, and the accessor a "when did this session
        last do anything" question wants (P5-05's passivation sweeper). `events[-1]`
        answers it too, at the cost of materializing a snapshot of the entire log to read
        one element — and the sweeper asks it of every root on every pass.
        """
        return self._log[-1] if self._log else None

    def select(self, *patterns: str) -> tuple[SessionEvent, ...]:
        """Every event under one or more namespace selectors, in log order (P6-33).

        `session.select("workspace")` is all five `workspace/*` types;
        `session.select("workspace/acquired")` is the one. The vocabulary is `log`, so a
        bare pattern needs no prefix and `bus:tools` is **refused** rather than answered
        emptily — an empty result would read as "there are none".

        Namespace-aware where `last_event_of` is exact: that one takes whole type names
        and answers with the newest. Both stay, because "every workspace record" should
        not be spelled as a list of five literals a new type would silently fall out of.

        No patterns returns the whole log.
        """
        selectors = parse_all(patterns, vocabulary="log")
        return tuple(event for event in self._log if matches_any(event.type, selectors))

    def last_event_of(self, *types: str) -> SessionEvent | None:
        """The most recent event of any of `types`. One type: prefer `latest()`."""
        for event in reversed(self._log):
            if event.type in types:
                return event
        return None


class SessionBatch:
    """The events one `Session.batch()` block will append; see there.

    Holds stamped events and nothing else — it never touches the log, so dropping
    one (an exception out of the block) needs no cleanup.
    """

    __slots__ = ("_events", "_session", "base")

    def __init__(self, session: Session, base: int) -> None:
        self._session = session
        self._events: list[SessionEvent] = []
        self.base = base
        """The seq the first member takes: the log's length when the batch opened."""

    @property
    def events(self) -> tuple[SessionEvent, ...]:
        """The members as stamped. The log holds them as committed, with the batch's
        membership on each."""
        return tuple(self._events)

    def append(
        self,
        event_type: str,
        data: Mapping[str, JsonValue],
        surface: SurfaceIntent | None = None,
    ) -> SessionEvent:
        """Stamp an event for this batch, refused as `Session.append` would refuse it.

        Returned so a later member can cite its seq; **not yet in the log**, and
        not in it at all if the batch is refused.
        """
        if self._session._batch is not self:
            # The block has exited, landed or not: a member stamped now would belong
            # to no commit, so it is refused rather than dropped.
            raise RuntimeError("this session batch has closed; open another to append")
        event = self._session._event(event_type, data, surface, self.base + len(self._events))
        self._events.append(event)
        return event


def _within_batch(open_batch: BatchRef | None, event: SessionEvent) -> BatchRef | None:
    """Hold `event` to the batch the log is inside, and say which it is inside after.

    A batch's members are contiguous and each carries the same ref: the first is at
    `first`, and nothing unstamped, and no other batch, falls before its last. Asked
    of every event that arrives already stamped — seed, fork, resume, a replica —
    since an owner's own `batch()` stamps its members correctly by construction.

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
    """Hold a seeded event to the acceptance rules a live append meets.

    The known-types refusal lives here, on the one path every seed takes —
    fork, resume, replay, import — rather than in one storage backend. An
    unrecognized *required* event may change how the rest of the log is read,
    so skipping it would reconstruct a wrong session, not a partial one.
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
