"""The session event envelope, its wire form, and the surface vocabulary.

`SessionEvent` is the envelope of dsh byte-for-byte: `{type, seq, time, data}`
plus the optional `ignorable`, `sourceEventSeqs` and `surfaceOp` fields (D2,
Q2). Since format 2 the envelope also carries the optional `batch` field
(P10-15). The class is a frozen `dataclass(slots=True)` and not a pydantic
model, because it is on the append hot path (D4). `ph.session.json` has already
validated and frozen the payload inside it.

The **read** side is not hot, so it goes through `WireModel` like every other
boundary. `_EventWire` supplies unknown-field rejection, acceptance of either
casing, and integer bounds, all from the one mechanism. `from_wire` only lifts
the result into the dataclass.

@module ph.session.events
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import Field, StrictInt, field_validator

from ..json import JsonObject, JsonValue, thaw_json
from ..wire import WireDataclass, WireModel
from .json import freeze_json_value

__all__ = [
    "SESSION_FORMAT_VERSION",
    "SURFACE_EVENT_TYPES",
    "BatchRef",
    "SessionEvent",
    "SurfaceIntent",
    "SurfaceOp",
    "SurfaceReplace",
    "is_surface_eligible_type",
    "now_ms",
]

SESSION_FORMAT_VERSION = 4
"""The on-disk format version. Every header carries it, and a load checks it.

The version is a single monotonic integer with no major/minor split. Increase
it exactly when an older runtime could no longer read a new log with full
semantic correctness. That covers the header shape, the envelope, core event
semantics, and the surface mechanism. A new ordinary event type does not
increase it, because `ignorable` covers vocabulary growth. While pH is
experimental, an incompatible log is rejected and not migrated.

**1 (0.2.0): the lineage directory carries the working directory.** The
`family` of a root *is* the directory that holds its log. It is now
`<cwd-tag>-<id>` and not `<id>`, so a listing can skip all the sessions of a
repo without opening one of them (`SessionHeader.family`). Nothing about the
envelope or the event semantics changed. The version increased because the
*layout* changed: a filtered listing would no longer find a 0.1.x log that is
in an undecorated directory. The refusal reports that.

**2 (0.4.0): the envelope carries batch membership** (P10-15). Each event that
`Session.batch()` appends together with others says which batch it belongs to:
`batch: {first, count}`. A reader can then tell a batch that a torn write cut
short from a complete batch, and drop the torn batch whole. It can do so
anywhere in a log and not only at the tail. A format-1 reader would refuse the
field as unknown, and that is exactly why the version increased. This header
check refuses a format-1 log and does not migrate it.

**3 (0.6.0): each session owns its log** (Phase 11). The `subagent/*` records of
a sub-agent are in the log of the sub-agent itself, and the log of a parent
holds none. `subagent/usage-attributed` is gone, because the answers of a child
carry their own usage. A format-2 reader would read the `subagent/admitted` of
a child as a record that the child admitted a child of its own. In a format-2
root, the children are in records that this build no longer reads. So a
format-2 log is refused and not migrated.

**4 (0.9.0): the sampling of a request header is one value.** `LlmCallConfig`
carries the six sampling settings as `sampling: {temperature, topP, ...}` and
not as six fields beside `model`. A route now sets them in the same form
(`Sampling`). Each side refuses the header of the other side, because that
header has fields that it does not know. So a format-3 log is refused here by
name, and not at its first `request/header`.
"""

SURFACE_EVENT_TYPES: frozenset[str] = frozenset(
    {"user/message", "assistant/message", "tool/result"}
)
"""The only event types that produce model-visible messages.

Everything else is trace and replay data: turn and step boundaries, raw chunks,
request headers, and log-only records. Only these three types may carry
`surfaceOp` and `sourceEventSeqs`.
"""

# Why this set is a *whitelist* and not a flag: surface membership is a property
# of the **type**. That makes a type like `shell/*` (the shell command of the
# user, logged in full) invisible to the model by construction, and not by a
# filter that every writer must remember. `_surface_op_of` enforces both
# directions: an eligible type must carry a `surfaceOp`, and an ineligible type
# may not. So "log it as a `tool/result` and filter it out of the context"
# cannot be expressed. It would be wrong anyway. A tool result with no
# `tool_use` block to pair with is an orphan that several providers reject.
# That is why the `!` of tau splices a *user message* that carries the output
# as text.

Seq: TypeAlias = Annotated[StrictInt, Field(ge=0)]
"""A non-negative event sequence number. Strict, so `True` is not a seq."""


def is_surface_eligible_type(event_type: str) -> bool:
    return event_type in SURFACE_EVENT_TYPES


def now_ms() -> int:
    """Unix epoch milliseconds, as a non-negative safe integer."""
    return int(_time.time() * 1000)


class SurfaceReplace(WireModel):
    """Replace the named surface nodes with this event.

    **The op names an id set, not a positional range** (§5.6). The difference is
    a guarantee and not a spelling. `start..end` meant "whatever currently
    occupies those positions". So the same operation, applied to a surface that
    had changed, shadowed different messages without an error. With named nodes
    the operation is either exactly right or refused with an error. The
    `replaces_entry_ids` of tau has that property by construction, and the range
    of pH did not.

    Every named seq must be a surface node *now*. The replacement lands where
    the earliest of them was, and the others are removed. One seq replaces one
    node. The `source_event_seqs` of the event must still cite every seq named
    here. It may cite more, such as the chunks that a message was built from.
    That is why the set is on the op and is not read back from the citation.
    """

    op: Literal["replace"] = "replace"
    replaces: tuple[Seq, ...]

    @field_validator("replaces")
    @classmethod
    def _named_and_distinct(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        # A replacement that names nothing would shadow nothing and still take a
        # surface slot. A repeated seq means that the writer built the set by
        # accident.
        if not value:
            raise ValueError("surfaceOp replace must name at least one surface node")
        if len(set(value)) != len(value):
            raise ValueError("surfaceOp replace must not name a surface node twice")
        return value


SurfaceOp: TypeAlias = 'Literal["append"] | SurfaceReplace'


class BatchRef(WireModel):
    """Which batch an event was appended in: the seq of its first member, and the count.

    **The ref is on every member.** So the question "is this batch whole?" has
    an exact answer at any point of a log. A whole batch is the members from
    `first` to `first + count - 1`, each with the same ref and with nothing else
    between them. A batch of one is never stamped, because there is nothing to
    keep together.
    """

    first: Seq
    count: Annotated[StrictInt, Field(ge=2)]

    @property
    def last(self) -> int:
        """The seq of the batch's last member."""
        return self.first + self.count - 1


@dataclass(frozen=True, slots=True)
class SurfaceIntent:
    """Surface placement supplied at `Session._append`.

    It is required on the three message-producing types and forbidden on every
    other type. A message-producing event must declare how it joins the surface,
    because the surface is the sole source of derived model history.
    """

    surface_op: SurfaceOp = "append"
    source_event_seqs: tuple[int, ...] | None = None


class _EventWire(WireModel):
    """The envelope as read from JSON. Validation lives in the field types."""

    type: Annotated[str, Field(min_length=1)]
    seq: Seq
    time: Seq
    data: dict[str, Any]
    """An object. Anything else is refused. Every payload that this harness
    writes is an object, because `Session._append` takes a `Mapping`. Every log
    that this repository carries has only objects, the dsh replay fixtures
    included: 5 172 events across 283 files, checked. Because the envelope
    refuses the exception, `SessionEvent.data` can be declared a `JsonObject`
    and not asserted to be one. So a reader does not narrow the top level of
    every payload that it opens."""
    ignorable: Literal[True] | None = None
    """Absent means required. A writer sets the marker only on purely
    informational records. So an explicit `false` comes from a writer that
    misunderstood the field, not from a thorough writer, and it is refused."""
    source_event_seqs: list[Seq] | None = None
    surface_op: Literal["append"] | SurfaceReplace | None = None
    batch: BatchRef | None = None


@dataclass(frozen=True, slots=True)
class SessionEvent(WireDataclass):
    """One immutable entry in the session log."""

    type: str
    seq: int
    """Monotonic sequence number within the session. It is always the log index."""
    time: int
    """Unix epoch milliseconds."""
    data: JsonObject
    """The frozen, lossless-JSON payload that entered the log.

    It is always an object, because `_EventWire` refuses anything else, and it
    is typed as one. So `event.data["turn"]` is a `JsonValue` that a reader
    narrows, and not an `Any` whose type the reader assumes. It is frozen in
    memory (a `MappingProxyType` over tuples) and plain after `to_wire`.
    `JsonValue` covers both forms deliberately."""
    ignorable: bool = False
    """Whether a reader that does not recognize `type` may safely skip it.

    Absent means required. A reader that meets an unrecognized type without this
    marker must refuse to reconstruct the session, and must not drop the event
    without an error. An unrecognized required event may change how the rest of
    the log is interpreted. Because the default is required, a forgotten marker
    refuses too much, which is an inconvenience. The other default would resume
    a session with events missing and give no warning.
    """
    source_event_seqs: tuple[int, ...] | None = None
    """The seqs of earlier events that this one cites: the chunks that built a
    message, or the surface nodes that a replacement shadows."""
    surface_op: SurfaceOp | None = None
    """How this event entered the surface. `None` for a non-surface event."""
    batch: BatchRef | None = None
    """The batch that this event was appended in (P10-15). `None` for an event
    that was appended alone. `Session.batch()` stamps it, and every seed and
    admit checks it."""

    def to_wire(self, *, thaw: bool = True) -> dict[str, JsonValue]:
        """The camelCase JSON object, with absent optional fields omitted.

        `thaw=False` shares the frozen payload and does not copy it into plain
        containers. That is right for serialization, because `JsonEncoder`
        handles the frozen forms. It is wrong for dict equality against a
        fixture that holds a plain `list`.
        """
        # The base call is explicit. `slots=True` rebuilds the class, so a
        # zero-argument `super()` would bind to the pre-slots class and fail.
        wire = WireDataclass.to_wire(self)
        if not self.ignorable:
            del wire["ignorable"]
        if thaw:
            wire["data"] = thaw_json(self.data)
        return wire

    @classmethod
    def from_wire(cls, wire: object) -> SessionEvent:
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
            batch=parsed.batch,
        )

    def readmitted(self) -> SessionEvent:
        """A copy whose payload is validated and frozen again.

        A seed crosses a persistence or replay boundary, so it is held to the
        same rules as a live append. That includes a hand-built event whose
        `data` is still a plain `dict`.
        """
        return replace(self, data=freeze_json_value(self.data, frozen_input=True))
