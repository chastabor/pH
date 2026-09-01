"""The session event envelope, its wire form, and the surface vocabulary.

`SessionEvent` is dsh's envelope byte-for-byte: `{type, seq, time, data}` plus
the optional `ignorable`, `sourceEventSeqs` and `surfaceOp` fields (D2, Q2).
A frozen `dataclass(slots=True)` rather than a pydantic model, because it is the
append hot path (D4) — the payload inside it is already validated and frozen by
`ph.session.json`.

The **read** side is not hot, so it goes through `WireModel` like every other
boundary: `_EventWire` supplies unknown-field rejection, either-casing
acceptance and integer bounds from the one mechanism, and `from_wire` just
lifts the result into the dataclass.

@module ph.session.events
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, StrictInt, field_validator

from ..wire import WireDataclass, WireModel
from .json import freeze_json_value, thaw_json

__all__ = [
    "SESSION_FORMAT_VERSION",
    "SURFACE_EVENT_TYPES",
    "SessionEvent",
    "SurfaceIntent",
    "SurfaceOp",
    "SurfaceReplace",
    "is_surface_eligible_type",
    "now_ms",
]

SESSION_FORMAT_VERSION = 0
"""The on-disk format version, stamped into every header and checked on load.

A single monotonic integer with no major/minor split. Bump exactly when an older
runtime could no longer read a new log with full semantic correctness — the
header shape, the envelope, core event semantics, or the surface mechanism.
Adding an ordinary event type does not bump: `ignorable` covers vocabulary
growth. While pH is unreleased it is pinned at 0 and incompatible logs are
rejected rather than migrated.
"""

SURFACE_EVENT_TYPES: frozenset[str] = frozenset(
    {"user/message", "assistant/message", "tool/result"}
)
"""The only event types that produce model-visible messages.

Everything else — turn and step boundaries, raw chunks, request headers,
log-only records — is trace and replay data. Only these three may carry
`surfaceOp` and `sourceEventSeqs`.
"""

Seq: TypeAlias = Annotated[StrictInt, Field(ge=0)]
"""A non-negative event sequence number. Strict, so `True` is not a seq."""


def is_surface_eligible_type(event_type: str) -> bool:
    return event_type in SURFACE_EVENT_TYPES


def now_ms() -> int:
    """Unix epoch milliseconds, as a non-negative safe integer."""
    return int(_time.time() * 1000)


class SurfaceReplace(WireModel):
    """Replace the named surface nodes with this event.

    **An id-set, not a positional range** (§5.6), and the difference is a
    guarantee rather than a spelling. `start..end` meant "whatever currently
    occupies those positions", so the same operation applied to a surface that
    had moved on silently shadowed different messages. Naming the nodes makes it
    either exactly right or a loud refusal, which is the property tau's
    `replaces_entry_ids` has by construction and pH's range did not.

    Every named seq must be a surface node *now*; the replacement lands where the
    earliest of them was, and the rest are removed. One seq replaces one node.
    The event's `source_event_seqs` must still cite every name here — it may cite
    more, such as the chunks a message was built from, which is why the set lives
    on the op rather than being read back off the citation.
    """

    op: Literal["replace"] = "replace"
    replaces: tuple[Seq, ...]

    @field_validator("replaces")
    @classmethod
    def _named_and_distinct(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        # A replacement that names nothing would shadow nothing and still take a
        # surface slot; a repeat means the writer built the set by accident.
        if not value:
            raise ValueError("surfaceOp replace must name at least one surface node")
        if len(set(value)) != len(value):
            raise ValueError("surfaceOp replace must not name a surface node twice")
        return value


SurfaceOp: TypeAlias = 'Literal["append"] | SurfaceReplace'


@dataclass(frozen=True, slots=True)
class SurfaceIntent:
    """Surface placement supplied at `Session.append`.

    Required on the three message-producing types and forbidden on every other:
    a message-producing event must declare how it joins the surface, since the
    surface is the sole source of derived model history.
    """

    surface_op: SurfaceOp = "append"
    source_event_seqs: tuple[int, ...] | None = None


class _EventWire(WireModel):
    """The envelope as read from JSON. Validation lives in the field types."""

    type: Annotated[str, Field(min_length=1)]
    seq: Seq
    time: Seq
    data: Any
    ignorable: Literal[True] | None = None
    """Absent means required. A writer sets the marker only on purely
    informational records, so an explicit `false` is a writer that misunderstood
    the field rather than one being thorough — and is refused."""
    source_event_seqs: list[Seq] | None = None
    surface_op: Literal["append"] | SurfaceReplace | None = None


@dataclass(frozen=True, slots=True)
class SessionEvent(WireDataclass):
    """One immutable entry in the session log."""

    type: str
    seq: int
    """Monotonic sequence number within the session; always its log index."""
    time: int
    """Unix epoch milliseconds."""
    data: Any
    """The frozen, lossless-JSON payload that entered the log."""
    ignorable: bool = False
    """Whether a reader that does not recognize `type` may safely skip it.

    Absent means required: a reader meeting an unrecognized type without this
    marker must refuse to reconstruct the session rather than silently drop the
    event, because an unrecognized required event may change how the rest of
    the log is interpreted. Defaulting to required means a forgotten marker
    over-refuses — an inconvenience — instead of quietly resuming a gutted
    session.
    """
    source_event_seqs: tuple[int, ...] | None = None
    """Seqs of earlier events this one cites: the chunks that built a message,
    or the surface nodes a replacement shadows."""
    surface_op: SurfaceOp | None = None
    """How this event entered the surface; `None` for a non-surface event."""

    def to_wire(self, *, thaw: bool = True) -> dict[str, Any]:
        """The camelCase JSON object, with absent optional fields omitted.

        `thaw=False` shares the frozen payload instead of copying it into plain
        containers — right for serialization (`JsonEncoder` handles the frozen
        forms), wrong for dict equality against a plain-`list` fixture.
        """
        # Explicit base call: `slots=True` rebuilds the class, so zero-arg
        # `super()` would bind to the pre-slots class and fail.
        wire = WireDataclass.to_wire(self)
        if not self.ignorable:
            del wire["ignorable"]
        if thaw:
            wire["data"] = thaw_json(self.data)
        return wire

    @classmethod
    def from_wire(cls, wire: Any) -> SessionEvent:
        """Rebuild an event from its JSON form, validating the envelope."""
        parsed = _EventWire.model_validate(wire)
        return cls(
            type=parsed.type,
            seq=parsed.seq,
            time=parsed.time,
            data=freeze_json_value(parsed.data),
            ignorable=parsed.ignorable is True,
            source_event_seqs=(
                None if parsed.source_event_seqs is None else tuple(parsed.source_event_seqs)
            ),
            surface_op=parsed.surface_op,
        )

    def readmitted(self) -> SessionEvent:
        """A copy whose payload has been re-validated and re-frozen.

        A seed crosses a persistence or replay boundary, so it is held to the
        same rules as a live append — including a hand-built event whose `data`
        is still a plain `dict`.
        """
        return replace(self, data=freeze_json_value(self.data, frozen_input=True))
