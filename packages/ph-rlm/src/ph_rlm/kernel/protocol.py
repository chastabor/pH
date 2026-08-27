"""The fd-3 frame vocabulary, host side.

The twin of `ph_runtime.protocol`, written separately on purpose — see that
module for why there is no shared definition, and `test_protocol_mirror.py` for
what keeps the two honest.

The split inside this module is deliberate:

* **Outbound** frames (host → guest) are `WireModel`s, so their camelCase field
  names come from pH's one alias function rather than from string literals, and
  their field sets are *derived* from the models. The host cannot drift from its
  own outbound schema.
* **Inbound** frames (guest → host) are declared as an explicit `FieldSpec`
  table, because the guest is a hostile peer (C10) and the codec has to rebuild
  every frame from a spec rather than parse it into a model. `extra="forbid"`
  would *raise* on a forged field; the rule is that junk becomes `None` and the
  handler never raises.

`FRAME_FIELDS` is the union, and the mirror test compares it to the guest's.

@module ph_rlm.kernel.protocol
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel

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
    "BootFrame",
    "CancelFrame",
    "FieldKind",
    "FieldSpec",
    "ReplyFrame",
    "RestoreFrame",
    "RunFrame",
    "ShutdownFrame",
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


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One field the codec will rebuild, and the only shape it will accept."""

    name: str
    kind: FieldKind
    required: bool = True


INBOUND: Final[dict[str, tuple[FieldSpec, ...]]] = {
    "boot-ack": (
        FieldSpec("type", "str"),
        FieldSpec("protocol", "int"),
        FieldSpec("python", "str"),
        FieldSpec("limits", "obj"),
    ),
    "call": (
        FieldSpec("type", "str"),
        FieldSpec("id", "int"),
        FieldSpec("global", "str"),
        FieldSpec("name", "str"),
        FieldSpec("args", "obj"),
    ),
    "log": (
        FieldSpec("type", "str"),
        FieldSpec("stream", "str"),
        FieldSpec("text", "str"),
        FieldSpec("truncated", "bool", required=False),
    ),
    "display": (
        FieldSpec("type", "str"),
        FieldSpec("mime", "str"),
        FieldSpec("data", "str"),
        FieldSpec("meta", "obj", required=False),
    ),
    "snapshot": (
        FieldSpec("type", "str"),
        FieldSpec("id", "int"),
        FieldSpec("variables", "list"),
    ),
    "done": (
        FieldSpec("type", "str"),
        FieldSpec("id", "int"),
        FieldSpec("value", "any", required=False),
        FieldSpec("error", "obj", required=False),
        FieldSpec("truncated", "bool", required=False),
    ),
    "fault": (FieldSpec("type", "str"), FieldSpec("message", "str")),
}


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


def truncation_marker(dropped: int, cap: int) -> str:
    """The text that stands in for output a cap discarded (D4).

    Duplicated from the guest by design and asserted equal by the mirror test:
    a reader comparing a transcript to a log must not find two different
    sentences for the same event.
    """
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
