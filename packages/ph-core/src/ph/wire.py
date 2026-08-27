"""One rule for every JSON boundary: declare aliases, never derive them (Q2).

pH is snake_case in Python and camelCase on the wire — the session JSONL, the
`json`/`rpc` output, and (from Phase 3) the fd-3 runtime frames. That is not a
cosmetic choice: dsh's envelope is already camelCase, so a pH log is a log dsh
tooling reads directly (D2).

The mechanism is one `ConfigDict` on a shared base. Aliases are fixed at class
definition, so a field name is never reconstructed from a wire string —
`to_camel` → `to_snake` happens to round-trip for every field in this plan, but
relying on that would be fragile at acronyms and digits.

`populate_by_name=True` makes every reader tolerant of both forms, which is what
lets `ph session import` ingest a foreign JSONL without a second parser.

Hot-path envelopes (`SessionEvent`, the stream chunks) are frozen dataclasses
rather than pydantic models (D4). `WireDataclass` gives them the same wire rule
through the same alias function, so there is one casing convention and not two.

@module ph.wire
"""

from __future__ import annotations

import dataclasses
from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

__all__ = ["WireDataclass", "WireModel", "wire_alias"]


def wire_alias(field_name: str) -> str:
    """The one alias function. Exported so tests can pin mappings against it."""
    return to_camel(field_name)


class WireModel(BaseModel):
    """Base for every model that crosses a JSON boundary.

    Dumps by alias (camelCase), validates either form, and is frozen: a model
    that reached the wire is a value, not a mutable buffer.
    """

    model_config = ConfigDict(
        alias_generator=wire_alias,
        populate_by_name=True,
        frozen=True,
        extra="forbid",
    )

    def to_wire(self) -> dict[str, Any]:
        """The camelCase JSON form, with absent optional fields omitted."""
        return self.model_dump(by_alias=True, exclude_none=True)


class WireDataclass:
    """Mixin giving a frozen dataclass the `WireModel.to_wire()` contract.

    A `type` field, when present, is emitted first so a discriminated reader
    sees it before anything else; `None` fields are omitted; nested values that
    know how to serialize themselves are asked to.
    """

    __slots__ = ()

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {}
        fields = dataclasses.fields(self)  # type: ignore[arg-type]
        ordered = sorted(fields, key=lambda field: field.name != "type")
        for field in ordered:
            value = getattr(self, field.name)
            if value is None:
                continue
            wire[wire_alias(field.name)] = _wire_value(value)
        return wire


def _wire_value(value: Any) -> Any:
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return to_wire()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    return value
