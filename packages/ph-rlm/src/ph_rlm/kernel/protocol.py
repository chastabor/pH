"""The fd-3 frame vocabulary, host side.

This module is the twin of `ph_runtime.protocol`. That module says which parts
are shared and which are written twice. `test_protocol_mirror.py` keeps the
parts that are written twice equal.

This module is split in two on purpose:

* **Outbound** frames (host → guest) are `WireModel`s. Their camelCase field
  names therefore come from the one alias function of pH and not from string
  literals, and their field sets are *derived* from the models. The host cannot
  drift from its own outbound schema.
* **Inbound** frames (guest → host) are `TypedDict`s. `INBOUND`, the `FieldSpec`
  table that the codec rebuilds from, is *derived* from them (P8-07), in the
  same way that the outbound field sets are derived from the models.

The inbound frames are `TypedDict`s and not `WireModel`s on purpose. The guest
is a hostile peer (C10). The rule of the codec is that junk becomes `None` and a
handler never raises, but the `extra="forbid"` of a model would *raise* on a
forged field. A `TypedDict` is only a type. The dict that `decode` rebuilds is
exactly the dict that the type describes: every key that the type names and no
other key, each with its declared shape. The type name is therefore correct at
the one point where it is applied, and `manager.py` reads `frame["id"]` from a
`DoneFrame` and not from a `dict[str, Any]`.

`FRAME_FIELDS` is the union of the two halves, and the mirror test compares it
with the `FRAME_FIELDS` of the guest.

@module ph_rlm.kernel.protocol
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Final,
    Literal,
    NotRequired,
    TypeAlias,
    TypedDict,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from ph.text import truncation_marker
from ph.wire import WireModel

# **Imported, not declared again.** `ph-rlm` already depends on
# `ph-runtime-guest`, so the host can read the declarations of the guest, and
# the two sides cannot disagree at all. The names are re-exported through
# `__all__` below. This module therefore stays the complete host-side view, and
# `venv` and `manager` keep importing the names from here. `ph_runtime.protocol`
# holds the reasoning for each constant and says what else is shared.
from ph_runtime.protocol import (
    FD_ENV,
    FRAME_BYTES_ENV,
    NAMESPACE_ENV,
    PROTOCOL_FD,
    PROTOCOL_VERSION,
)

__all__ = [
    "FD_ENV",
    "FRAME_BYTES_ENV",
    "FRAME_FIELDS",
    "GUEST_FRAMES",
    "HOST_FRAMES",
    "INBOUND",
    "NAMESPACE_ENV",
    "PROTOCOL_FD",
    "PROTOCOL_VERSION",
    "BootAckFrame",
    "BootFrame",
    "CallFrame",
    "CancelFrame",
    "DisplayFrame",
    "DoneFrame",
    "FaultFrame",
    "FieldKind",
    "FieldSpec",
    "InboundFrame",
    "LogFrame",
    "PingFrame",
    "PongFrame",
    "ReplyFrame",
    "RestoreFrame",
    "RunFrame",
    "ShutdownFrame",
    "SnapshotFrame",
    "truncation_marker",
]

HOST_FRAMES: Final = frozenset({"boot", "run", "reply", "restore", "cancel", "shutdown", "ping"})
GUEST_FRAMES: Final = frozenset(
    {"boot-ack", "call", "log", "display", "snapshot", "done", "fault", "pong"}
)


# ------------------------------------------------------------ host → guest --


class BootFrame(WireModel):
    """Limits and bindings, sent one time, before anything runs.

    Every limit is required, because the host is the only owner of a default.
    If the guest had its own fallbacks, there would be two answers to "what is
    the log cap", and the answer that applied would depend on which side was
    older.
    """

    type: Literal["boot"] = "boot"
    protocol: int = PROTOCOL_VERSION
    cpu_seconds: int
    idle_cpu_seconds: int
    address_space_bytes: int
    max_log_bytes: int
    max_value_bytes: int
    max_snapshot_bytes: int
    namespaces: list[dict[str, Any]]
    namespace_id: str | None = None
    skills: list[str]
    """The Python skills to import and bind as callables before any cell runs (P3-18).

    These are import names and not requirement specs. What to install is the
    concern of the venv builder and happens one time. This field is what the
    *namespace* must hold."""


class RunFrame(WireModel):
    type: Literal["run"] = "run"
    id: int
    program: str


class ReplyFrame(WireModel):
    """The answer to one `call`.

    `fatal` is C3 on the wire. A denial or a budget settles the whole run. The
    proxy therefore raises an exception that the program is not meant to catch.
    """

    type: Literal["reply"] = "reply"
    id: int
    ok: bool
    value: Any = None
    message: str | None = None
    name: str | None = None
    """The `name` of the call, echoed back so that a failure can say what failed.

    The guest raises `ToolFailed(name, message)`. It used to read the name from
    a field that the host never sent. Every tool failure inside a cell therefore
    reported the tool as the literal string `"the call"`, from the day the RLM
    landed. The field is optional because only a failure needs it. A successful
    reply is the hot path and carries the value instead."""
    fatal: bool | None = None


class RestoreFrame(WireModel):
    type: Literal["restore"] = "restore"
    id: int
    variables: list[dict[str, Any]]
    more: bool = False
    """Whether another batch follows (O3). The guest accumulates, and answers on
    the frame where `more` is false. One `restore` is therefore still one `done`."""


class CancelFrame(WireModel):
    type: Literal["cancel"] = "cancel"
    id: int | None = None


class ShutdownFrame(WireModel):
    type: Literal["shutdown"] = "shutdown"


class PingFrame(WireModel):
    """A probe that the reader task of the guest answers from its event loop (M2).

    **It is a gauge first.** The round trip is the only measurement that the
    host has of how far behind the event loop of the guest is. A cell that burns
    CPU starves the reader task. A late or missing answer is therefore read as
    *load*, which `phern doctor` reports, and never as a reason to kill
    anything. The CPU budget of the guest (`ph_runtime.limits.arm_cpu_budget`)
    limits a busy cell.
    """

    type: Literal["ping"] = "ping"
    id: int
    """The number of the probe itself. `_ActiveRun.answered` compares it, so
    that a late or unsolicited answer is told apart from the current answer and
    times nothing."""
    run: int
    """The run that the host is waiting on, so that the guest can settle it if
    nothing else will.

    The question travels outward. The state of the guest does not travel back.
    A guest that is not running this run and does not owe it a `done` sends one
    immediately. The host learns through an ordinary terminal frame, and the
    namespace survives. The other direction was to publish `_owed` and let the
    host conclude. That left the host able only to conclude, and the remedy for
    a conclusion is a kill."""


_OUTBOUND: Final[tuple[type[BaseModel], ...]] = (
    BootFrame,
    RunFrame,
    ReplyFrame,
    RestoreFrame,
    CancelFrame,
    ShutdownFrame,
    PingFrame,
)


# ------------------------------------------------------------ guest → host --

FieldKind: TypeAlias = Literal["int", "str", "bool", "obj", "list", "any"]

_SCALARS: Final[dict[Any, FieldKind]] = {bool: "bool", int: "int", str: "str"}
"""The scalar annotations that `_kind_of` can name, and the name that the codec
uses for each."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field that the codec rebuilds, and the only shape that it accepts."""

    name: str
    kind: FieldKind
    required: bool = True


# The eight frames that a guest may send, each with the shape that it has
# *after* `decode` has rebuilt it. `NotRequired` is the `required=False` of the
# codec. A `dict[str, Any]` is its `"obj"`, a `list[Any]` is its `"list"`, and
# `Any` is its `"any"`. `_kind_of` is the mapping. A field with a shape that
# `_kind_of` cannot name is an error at import time, and not a frame that the
# codec quietly accepts.


class BootAckFrame(TypedDict):
    type: Literal["boot-ack"]
    protocol: int
    python: str
    limits: dict[str, Any]


# Only this frame uses the functional form. The wire key is `global`, which a
# class body cannot spell. `global` is the dsh name, and the module docstring of
# `ph_runtime.protocol` says why it stays.
CallFrame = TypedDict(
    "CallFrame",
    {
        "type": Literal["call"],
        "id": int,
        "run": int,
        "global": str,
        "name": str,
        "args": dict[str, Any],
    },
)
"""`run` says which program issued this call, and it is required (F4).

A cell can leave a task behind: one that `asyncio.create_task` started and that
was never awaited. That task can call a binding after its run has settled. The
call then arrived while the *next* program was open, and it was served against
the bindings, the budget and the parent dispatch id of that program. The work of
one cell was recorded as the work of another, and the approval and limit
decisions were taken for the wrong turn. The guest now stamps each call with the
run that it belongs to, and the host refuses a stamp that is not the open run."""


class LogFrame(TypedDict):
    type: Literal["log"]
    stream: str
    text: str
    truncated: NotRequired[bool]


class DisplayFrame(TypedDict):
    type: Literal["display"]
    mime: str
    data: str
    meta: NotRequired[dict[str, Any]]


class SnapshotFrame(TypedDict):
    type: Literal["snapshot"]
    id: int
    variables: list[Any]


class DoneFrame(TypedDict):
    type: Literal["done"]
    id: int
    value: NotRequired[Any]
    error: NotRequired[dict[str, Any]]
    truncated: NotRequired[bool]
    dropped: NotRequired[int]


class FaultFrame(TypedDict):
    type: Literal["fault"]
    message: str


class PongFrame(TypedDict):
    """The answer of the guest to a `ping`: the clock, and nothing else (M2).

    The frame carries no verdict, because the guest acts on the question and
    does not report about it. See `PingFrame.run`. What is left is the round
    trip, which measures how far behind the event loop of the guest is. That is
    a *load* reading: a slow answer is a busy cell, and no answer is a starved
    cell. The CPU budget of the guest (`ph_runtime.limits.arm_cpu_budget`)
    limits both.
    """

    type: Literal["pong"]
    id: int


InboundFrame: TypeAlias = (
    BootAckFrame
    | CallFrame
    | LogFrame
    | DisplayFrame
    | SnapshotFrame
    | DoneFrame
    | FaultFrame
    | PongFrame
)
"""What `decode` returns. It is a tagged union on `type`, so
`if frame["type"] == "done":` narrows it. Compare the tag *directly*: mypy does
not carry the narrowing through an intermediate `kind = frame["type"]`."""

_INBOUND: Final[tuple[Any, ...]] = get_args(InboundFrame)
"""The members of the union, read from the union itself. A frame that is added
to `InboundFrame` therefore cannot be missing from `INBOUND`. With a separate
list it could be missing. The symptom would be that `decode` returns `None` for
every instance of the new frame: the "junk becomes `None`" rule of the codec
would apply to a frame that is not junk."""


def _kind_of(annotation: object) -> FieldKind:
    """The `FieldKind` of the codec for one declared field type. This function is
    the whole mapping."""
    if annotation is Any:
        return "any"
    if get_origin(annotation) is Literal:
        return "str"  # every tag is a string, and `type` is the only Literal
    # This is an identity test and not `isinstance`. An annotation is a class
    # and not a value, so the order of `bool` and `int` does not matter here.
    # That hazard is in `codec._coerce`, which has the comment about it.
    if annotation in _SCALARS:
        return _SCALARS[annotation]
    origin = get_origin(annotation) or annotation
    if origin is dict:
        return "obj"
    if origin is list:
        return "list"
    raise TypeError(f"no codec shape for an inbound field typed {annotation!r}")


def _inbound_specs() -> dict[str, tuple[FieldSpec, ...]]:
    """Build `INBOUND` from the frame types, so that the declaration has one source.

    **Use `get_type_hints`, never `__required_keys__`.** Under
    `from __future__ import annotations`, the annotations of a class body are
    strings when `TypedDict` builds the class, and `TypedDict` cannot see
    `NotRequired` inside a string. On this interpreter every key is then
    reported as required. That is the exact drift that the `TypedDict`s of the
    guest had before they were deleted (`test_protocol_mirror` tells the story).
    `get_type_hints(..., include_extras=True)` resolves the strings and keeps
    the qualifier.
    """
    specs: dict[str, tuple[FieldSpec, ...]] = {}
    for frame in _INBOUND:
        hints = get_type_hints(frame, include_extras=True)
        fields: list[FieldSpec] = []
        for name, hint in hints.items():
            optional = get_origin(hint) is NotRequired
            inner = get_args(hint)[0] if optional else hint
            fields.append(FieldSpec(name, _kind_of(inner), required=not optional))
        (tag,) = get_args(hints["type"])
        specs[str(tag)] = tuple(fields)
    return specs


INBOUND: Final[dict[str, tuple[FieldSpec, ...]]] = _inbound_specs()


def _outbound_fields() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """Build the field sets from the models, so that the declaration has one source."""
    fields: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for model in _OUTBOUND:
        required: set[str] = set()
        optional: set[str] = set()
        for name, info in model.model_fields.items():
            alias = info.alias or name
            # The rule follows the dump. `encode` uses `exclude_none=True`, so a
            # field is omitted from the wire exactly when its value is `None`. A
            # field with a default that is not `None` (`type`, `protocol`) is
            # therefore always *sent*. The guest sees it as required, even
            # though a caller does not have to pass it.
            if info.is_required() or info.default is not None:
                required.add(alias)
            else:
                optional.add(alias)
        name_literal = model.model_fields["type"].default
        fields[str(name_literal)] = (frozenset(required), frozenset(optional))
    return fields


FRAME_FIELDS: Final[dict[str, tuple[frozenset[str], frozenset[str]]]] = {
    **_outbound_fields(),
    **{
        frame: (
            frozenset(spec.name for spec in specs if spec.required),
            frozenset(spec.name for spec in specs if not spec.required),
        )
        for frame, specs in INBOUND.items()
    },
}


# Re-exported, not declared again. The sentence moved to `ph.text` when `!!` and
# `tool-bash` became the third and fourth things that discard output against a
# cap (P7-13). The mirror test still compares `host.truncation_marker` with the
# deliberate copy in the guest, and that is the assertion that matters.
truncation_marker = truncation_marker
