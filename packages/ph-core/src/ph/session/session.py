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
from collections.abc import Callable, Sequence
from pathlib import PurePath
from typing import Any, Literal, TypeAlias

from pydantic import Field, NonNegativeInt, field_validator

from ..llm.types import Message
from ..selectors import matches_any, parse_all
from ..wire import WireModel
from .derive import derive_event_message, derive_transcript
from .events import SESSION_FORMAT_VERSION, SessionEvent, SurfaceIntent, now_ms
from .json import freeze_json_value
from .known_event_types import IGNORABLE_SESSION_EVENT_TYPES, KNOWN_SESSION_EVENT_TYPES
from .request_header import (
    EpochHeader,
    RequestContext,
    fold_latest,
    parse_request_context,
    parse_request_header,
)
from .surface import SurfaceManager

__all__ = ["Session", "SessionHeader", "SessionObserver"]

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
    family: str = Field(default_factory=lambda data: data["id"], min_length=1)
    """Which lineage this log belongs to — the id of the root it descends from.

    The directory a log lives in: `sessions/<family>/<id>.jsonl`. Every fork, segment
    and subagent beneath a root inherits its value, so one conversation and everything
    it spawned is one directory — which is what lets "is anything in here orphaned" be
    answered by listing one directory.

    **Never absent.** A root heads its own lineage, so the default *is* the id — a
    `default_factory` reading the already-validated `id`, covering direct construction
    and `model_validate` alike. `min_length=1` makes an explicit empty string a
    refusal rather than something quietly rewritten. Stored rather than derived,
    because deriving it means walking `parent_session` to the root, and walking needs
    paths, and the path is what the family is for.
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


class _LatestFold[T]:
    """An incrementally maintained "latest event of type X" over a growing log."""

    __slots__ = ("_event_type", "_parse", "_seen", "value")

    def __init__(self, event_type: str, parse: Callable[[SessionEvent], T]) -> None:
        self._event_type = event_type
        self._parse = parse
        self._seen = 0
        self.value: T | None = None

    def read(self, log: list[SessionEvent]) -> T | None:
        if self._seen < len(log):
            self.value = fold_latest(log[self._seen :], self._event_type, self._parse, self.value)
            self._seen = len(log)
        return self.value


class Session:
    """An event-sourced session: an append-only log of `SessionEvent`s."""

    __slots__ = (
        "_context_fold",
        "_derived",
        "_derived_generation",
        "_derived_nodes",
        "_events_snapshot",
        "_header_fold",
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
        self._derived: tuple[Message, ...] = ()
        self._derived_nodes = 0
        self._derived_generation = 0
        self._header_fold = _LatestFold("request/header", parse_request_header)
        self._context_fold = _LatestFold("request/context", parse_request_context)
        self._latest: dict[str, _LatestFold[SessionEvent]] = {}

        if seed is not None:
            # Validate the seed to the SAME invariants `append` enforces. A
            # replay or fork must not be able to construct a live log that no
            # backend could store — otherwise a bad seed surfaces later as a
            # flush rejection or, worse, as a silent divergence from disk.
            for index, source in enumerate(seed):
                event = _readmit(source, index)
                try:
                    self._surface.validate_next(event)
                except ValueError as error:
                    raise ValueError(f"invalid seed event at index {index}: {error}") from error
                self._log.append(event)

        self.durable_length = durable
        """How many leading events a **store already holds**; 0 unless declared.

        Read by a backend to tell what it still owes from what is already written:
        `track` queues `events[durable_length:]`.

        **A constructor argument, not an attribute set afterwards.** Publishing a session
        is what makes a store queue it — `session/created` reaches `track` synchronously
        — so a boundary assigned one line after construction is one line too late, and
        the window is invisible: the store simply writes more than it needed to.

        Two callers declare it and they mean different numbers: `resume_session` passes
        the stored log's length, `SessionStore.create` passes a fork's inherited prefix.
        Deriving one from the other would be wrong — a resumed fork's
        `header.seed_length` is its *original* fork boundary.

        **Not `first_live_seq`**, and why that distinction is load-bearing:
        `tests/test_persistence_backends.py`.

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

    def events_from(self, index: int) -> tuple[SessionEvent, ...]:
        """The events appended at or after `index`.

        The accessor an incremental fold wants. `events` caches one snapshot of
        the *whole* log and rebuilds it whenever the log grew — right for a
        reader that wants all of it, and quadratic for one that runs per event
        and only ever looks at the tail. A `SessionFoldCache` extender reads
        through here.
        """
        return tuple(self._log[index:])

    @property
    def seq(self) -> int:
        """The next event's sequence number — always the log length (A1)."""
        return len(self._log)

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
        data: Any,
        surface: SurfaceIntent | None = None,
    ) -> SessionEvent:
        """Append one event and synchronously notify observers.

        The hot path never blocks on I/O — persistence buffers asynchronously
        and drains on `session/flush`.

        Whether the event is `ignorable` — skippable by a *different* build that
        does not know its type — is a property of the type, read from
        `IGNORABLE_SESSION_EVENT_TYPES` rather than passed here, so no two call
        sites can disagree about one type.

        :raises InvalidJsonValueError: when `data` is not losslessly JSON.
        :raises SurfaceError: when the surface metadata is wrong for this type.
        :raises RuntimeError: when re-entered during publication.
        """
        if self._publishing:
            # A reentrant append would assign a seq inside another event's
            # publication, so observers would see the log grow underneath them.
            raise RuntimeError(
                "session append cannot reenter while another append is being published"
            )
        event = SessionEvent(
            type=event_type,
            seq=len(self._log),
            time=now_ms(),
            data=freeze_json_value(data),
            source_event_seqs=None if surface is None else surface.source_event_seqs,
            surface_op=None if surface is None else surface.surface_op,
            ignorable=event_type in IGNORABLE_SESSION_EVENT_TYPES,
        )
        # Validated BEFORE the push: a rejected candidate must leave both the
        # log and the surface exactly as they were.
        self._surface.validate_next(event)

        self._publishing = True
        try:
            self._log.append(event)
            self._events_snapshot = None
            for observer in self._observers:
                try:
                    observer(self, event)
                except Exception:
                    log.exception(
                        "ph.session: observer failed for %s at seq %s", event.type, event.seq
                    )
            return event
        finally:
            self._publishing = False

    # --------------------------------------------------------------- folds --

    def request_header(self) -> EpochHeader | None:
        """The header the NEXT request will be compared against."""
        return self._header_fold.read(self._log)

    def request_context(self) -> RequestContext | None:
        """The latest resolved route metadata, folded incrementally."""
        return self._context_fold.read(self._log)

    # ------------------------------------------------------------- derivation --

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
        fold = self._latest.get(event_type)
        if fold is None:
            fold = self._latest[event_type] = _LatestFold(event_type, lambda event: event)
        return fold.read(self._log)

    @property
    def last_event(self) -> SessionEvent | None:
        """The most recently appended event, or `None` for an empty log.

        The unfiltered form of `last_event_of`, and the accessor a "when did
        this session last do anything" question wants (P5-05's passivation
        sweeper). `events[-1]` answers it too, at the cost of materialising a
        snapshot of the entire log — 4 MB and 4.7 ms at 500 000 events — to read
        one element, and the sweeper asks it of every root on every pass.
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
    if source.type not in KNOWN_SESSION_EVENT_TYPES and not source.ignorable:
        raise ValueError(
            f'seed event at index {index} has unrecognized required type "{source.type}"; '
            "this log was written by a newer harness and reading it here would "
            "reconstruct a wrong session"
        )
    return source.readmitted()
