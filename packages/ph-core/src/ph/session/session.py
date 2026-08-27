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
from typing import Any, Literal

from pydantic import NonNegativeInt, field_validator

from ..llm.types import Message
from ..wire import WireModel
from .derive import derive_event_message, derive_transcript
from .events import SESSION_FORMAT_VERSION, SessionEvent, SurfaceIntent, now_ms
from .json import freeze_json_value
from .known_event_types import KNOWN_SESSION_EVENT_TYPES
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
        "first_live_seq",
        "header",
    )

    def __init__(
        self,
        session_id: str,
        seed: Sequence[SessionEvent] | None = None,
        header: SessionHeader | None = None,
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
