"""The fd-3 frame vocabulary, host side.

The twin of `ph_runtime.protocol`, written separately on purpose — see that
module for why there is no shared definition, and `test_protocol_mirror.py` for
what keeps the two honest.

The split inside this module is deliberate:

* **Outbound** frames (host → guest) are `WireModel`s, so their camelCase field
  names come from pH's one alias function rather than from string literals, and
  their field sets are *derived* from the models. The host cannot drift from its
  own outbound schema.
* **Inbound** frames (guest → host) are `TypedDict`s, and the `FieldSpec` table
  the codec rebuilds from — `INBOUND` — is *derived* from them (P8-07), the way
  the outbound field sets are derived from the models. `TypedDict` rather than
  `WireModel` on purpose: the guest is a hostile peer (C10) and the codec's rule
  is that junk becomes `None` and a handler never raises, where a model's
  `extra="forbid"` would *raise* on a forged field. A `TypedDict` is only a
  type, and the dict `decode` rebuilds is exactly the dict the type describes —
  every key it names and no other, each of its declared shape — so the name is
  honest at the one point it is applied, and `manager.py` reads `frame["id"]`
  off a `DoneFrame` rather than off a `dict[str, Any]`.

`FRAME_FIELDS` is the union, and the mirror test compares it to the guest's.

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

__all__ = [
    "FD_ENV",
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
    "ReplyFrame",
    "RestoreFrame",
    "RunFrame",
    "ShutdownFrame",
    "SnapshotFrame",
    "truncation_marker",
]

PROTOCOL_VERSION: Final = 1
PROTOCOL_FD: Final = 3
FD_ENV: Final = "PH_RUNTIME_FD"
NAMESPACE_ENV: Final = "PH_NAMESPACE_ID"

HOST_FRAMES: Final = frozenset({"boot", "run", "reply", "restore", "cancel", "shutdown"})
GUEST_FRAMES: Final = frozenset({"boot-ack", "call", "log", "display", "snapshot", "done", "fault"})


# ------------------------------------------------------------ host → guest --


class BootFrame(WireModel):
    """Limits and bindings, once, before anything runs.

    Every limit is required, because the host is the only owner of a default.
    A guest with its own fallbacks would mean two answers to "what is the log
    cap", and the one that applied would depend on which side was older.
    """

    type: Literal["boot"] = "boot"
    protocol: int = PROTOCOL_VERSION
    cpu_seconds: int
    address_space_bytes: int
    max_log_bytes: int
    max_value_bytes: int
    max_snapshot_bytes: int
    namespaces: list[dict[str, Any]]
    namespace_id: str | None = None
    skills: list[str]
    """Python skills to import and bind callable before any cell runs (P3-18).

    Import names, not requirement specs: what to install is the venv builder's
    business and happens once, while this is what the *namespace* should hold."""


class RunFrame(WireModel):
    type: Literal["run"] = "run"
    id: int
    program: str


class ReplyFrame(WireModel):
    """The answer to one `call`.

    `fatal` is C3 on the wire: a denial or a budget settles the whole run, so the
    proxy raises something the program is not offered a chance to catch.
    """

    type: Literal["reply"] = "reply"
    id: int
    ok: bool
    value: Any = None
    message: str | None = None
    name: str | None = None
    """The call's own `name`, echoed back so a failure can say what failed.

    The guest raises `ToolFailed(name, message)`, and read it off a field the
    host never sent: every tool failure inside a cell reported the tool as the
    literal string `"the call"`, from the day the RLM landed. Optional because
    only a failure needs it — a successful reply is the hot path and carries the
    value instead."""
    fatal: bool | None = None


class RestoreFrame(WireModel):
    type: Literal["restore"] = "restore"
    id: int
    variables: list[dict[str, Any]]


class CancelFrame(WireModel):
    type: Literal["cancel"] = "cancel"
    id: int | None = None


class ShutdownFrame(WireModel):
    type: Literal["shutdown"] = "shutdown"


_OUTBOUND: Final[tuple[type[BaseModel], ...]] = (
    BootFrame,
    RunFrame,
    ReplyFrame,
    RestoreFrame,
    CancelFrame,
    ShutdownFrame,
)


# ------------------------------------------------------------ guest → host --

FieldKind: TypeAlias = Literal["int", "str", "bool", "obj", "list", "any"]

_SCALARS: Final[dict[Any, FieldKind]] = {bool: "bool", int: "int", str: "str"}
"""The scalar annotations `_kind_of` can name, and what the codec calls each."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field the codec will rebuild, and the only shape it will accept."""

    name: str
    kind: FieldKind
    required: bool = True


# The seven frames a guest may send, as the shape each has *after* `decode` has
# rebuilt it. `NotRequired` is the codec's `required=False`; a `dict[str, Any]`
# is its `"obj"`, a `list[Any]` its `"list"`, `Any` its `"any"` — `_kind_of` is
# the mapping, and a field of a shape it cannot name is an import-time error
# rather than a frame the codec quietly accepts.


class BootAckFrame(TypedDict):
    type: Literal["boot-ack"]
    protocol: int
    python: str
    limits: dict[str, Any]


# Functional form for this one only: the wire key is `global` — dsh's name, and
# the module docstring says why it stays — which a class body cannot spell.
CallFrame = TypedDict(
    "CallFrame",
    {"type": Literal["call"], "id": int, "global": str, "name": str, "args": dict[str, Any]},
)


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


class FaultFrame(TypedDict):
    type: Literal["fault"]
    message: str


InboundFrame: TypeAlias = (
    BootAckFrame | CallFrame | LogFrame | DisplayFrame | SnapshotFrame | DoneFrame | FaultFrame
)
"""What `decode` returns. A tagged union on `type`, so `if frame["type"] == "done":`
narrows it — compared *directly*: mypy does not carry the narrowing through an
intermediate `kind = frame["type"]`."""

_INBOUND: Final[tuple[Any, ...]] = get_args(InboundFrame)
"""The union's members, read off the union — so a frame added to `InboundFrame`
cannot be missing from `INBOUND`. Listed separately it could be, and the symptom
would be `decode` returning `None` for every instance of the new frame: the
codec's "junk becomes `None`" rule firing on a frame that is not junk."""


def _kind_of(annotation: Any) -> FieldKind:
    """The codec's `FieldKind` for one declared field type — the whole mapping."""
    if annotation is Any:
        return "any"
    if get_origin(annotation) is Literal:
        return "str"  # every tag is a string, and `type` is the only Literal
    # Identity, not `isinstance`: an annotation is a class, not a value, so
    # `bool`/`int` need no ordering care here — `codec._coerce` is where that
    # hazard lives, and it carries the comment.
    if annotation in _SCALARS:
        return _SCALARS[annotation]
    origin = get_origin(annotation) or annotation
    if origin is dict:
        return "obj"
    if origin is list:
        return "list"
    raise TypeError(f"no codec shape for an inbound field typed {annotation!r}")


def _inbound_specs() -> dict[str, tuple[FieldSpec, ...]]:
    """`INBOUND`, read off the frame types, so the declaration has one source.

    **Through `get_type_hints`, never `__required_keys__`.** Under
    `from __future__ import annotations` a class body's annotations are strings
    when `TypedDict` builds the class, and it cannot see `NotRequired` inside a
    string: on this interpreter every key then reports as required. That is the
    exact drift the guest's own `TypedDict`s had before they were deleted
    (`test_protocol_mirror` tells the story). `get_type_hints(...,
    include_extras=True)` resolves the strings and keeps the qualifier.
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
    """Field sets read off the models, so the declaration has one source."""
    fields: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for model in _OUTBOUND:
        required: set[str] = set()
        optional: set[str] = set()
        for name, info in model.model_fields.items():
            alias = info.alias or name
            # The rule follows the dump: `encode` uses `exclude_none=True`, so a
            # field is omitted from the wire exactly when its value is `None`.
            # A field with a non-`None` default — `type`, `protocol` — is
            # therefore always *sent*, and so is required as the guest sees it,
            # even though a caller need not pass it.
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


# Re-exported, not re-spelled: the sentence moved to `ph.text` when `!!` and
# `tool-bash` became the third and fourth things that discard output against a
# cap (P7-13). The mirror test still compares `host.truncation_marker` with the
# guest's deliberate copy, which is the assertion that matters.
truncation_marker = truncation_marker
