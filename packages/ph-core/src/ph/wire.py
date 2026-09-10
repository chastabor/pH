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

import collections.abc
import dataclasses
import typing
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic.alias_generators import to_camel

__all__ = [
    "WireDataclass",
    "WireModel",
    "declarable",
    "declarable_fields",
    "literal_lookup",
    "validation_errors",
    "wire_alias",
]

_Member = typing.TypeVar("_Member", bound=str)


def wire_alias(field_name: str) -> str:
    """The one alias function. Exported so tests can pin mappings against it."""
    return to_camel(field_name)


def literal_lookup(alias: object) -> collections.abc.Mapping[str, _Member]:
    """Every member of a `Literal` by its own spelling — the read side of Q2.

    A `Literal` constrains writers and dies at the payload boundary, so a reader
    holding a `str` has to check. `raw in get_args(X)` is that check and does
    **not** narrow, which is why every such site carried a `cast` beside it
    asserting what the test had just established. Keyed by its members instead,
    the lookup *is* the check and it narrows: `MODES.get(raw)` is the member or
    `None`, with no cast to go stale, and a member added to the alias appears
    here for free.

    Six sites hand-rolled `{one: one for one in get_args(X)}` before this
    existed, each carrying a copy of the paragraph above. The declaration is
    still the single source — this only stops it being restated.

    **The caller annotates, and must.** The member type comes from the
    annotation, not the argument: `get_args` returns `tuple[Any, ...]`, and a
    `Literal` alias is not a `type[X]`, so there is no way to read it off the
    parameter without PEP 747's `TypeForm`. The bound is `str` so the
    annotation cannot be something a `.get` would narrow wrongly, and omitting
    it is a hard `error: Need type annotation` rather than a silent `Any` —
    which is exactly the property the *narrowing* form of this helper
    (`as_literal(raw, alias) -> X | None`) fails to have, and why that one is
    not written here. Pairing the annotation with the wrong alias is not
    policed, and was not policed in the hand-rolled form either.
    """
    return {one: one for one in typing.get_args(alias)}


def validation_errors(error: ValidationError, *, root: str = "<root>") -> list[str]:
    """Pydantic's error list as sentences, one per violation, in validation order.

    Here because two places render it and they were rendering it the same way by
    coincidence: `validate_json_schema_value` for a tool's arguments, and
    `ph_app.protocol.parse_params` for an RPC method's. What a person reads when
    a call is refused should not depend on which door the call came through, and
    the two copies would have drifted the first time either wanted the offending
    input or the error's `type` in the text.

    `root` is what to call a violation with no path — the whole value was wrong.
    The two callers differ there and only there: a schema names it `<root>`, and
    a method names it `params`, because that is what the caller sent.
    """
    return [
        f"{'.'.join(str(part) for part in one['loc']) or root}: {one['msg']}"
        for one in error.errors()
    ]


class WireModel(BaseModel):
    """Base for every model that crosses a JSON boundary.

    Dumps by alias (camelCase), validates either form, and is frozen: a model
    that reached the wire is a value, not a mutable buffer.

    It also accepts **frozen log data** directly. The session log freezes
    payloads into `MappingProxyType`/`tuple`, and pydantic coerces those into
    declared `dict`/`list` fields — but a field typed `Any` keeps what it was
    given, so a JSON array read back from the log would stay a tuple and compare
    unequal to an identical list. That broke `header_equals` once, silently, and
    with it prefix caching. Thawing here, at the one layer every model shares,
    means no call site has to remember to.
    """

    model_config = ConfigDict(
        alias_generator=wire_alias,
        populate_by_name=True,
        frozen=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _thaw_frozen_input(cls, data: Any) -> Any:
        if isinstance(data, MappingProxyType):
            # Lazy import: `ph.session.json` is import-safe on its own, but
            # importing it at module load would run `ph.session.__init__`, which
            # imports `ph.llm.types`, which imports this module.
            from .session.json import thaw_json

            return thaw_json(data)
        return data

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


def declarable_fields(dataclass_type: type) -> tuple[str, ...]:
    """The fields of a definition that can travel: everything not a callable.

    Derived from the annotations rather than kept in a list, because a list is
    the thing that drifts: the row this serves (P7-11) began with a hand-kept
    table of "the one field that cannot travel" per definition, and a table is
    exactly what silently stops being true when a second callable is added.
    `get_type_hints` is needed — the seams use `from __future__ import
    annotations`, so a bare `field.type` is a string.
    """
    hints = typing.get_type_hints(dataclass_type)
    return tuple(
        field.name
        for field in dataclasses.fields(dataclass_type)
        if typing.get_origin(hints.get(field.name)) is not collections.abc.Callable
    )


def declarable(instance: Any) -> dict[str, Any]:
    """The travelling half of a definition, as the kwargs of its schema.

    A `schema()` built as `Schema.model_validate(declarable(self))` names no
    field, so a field added to the definition is carried without an edit — and,
    because `WireModel` forbids extras, one added to the definition and *not* to
    the schema fails at the first call rather than quietly reaching the wire as
    the schema's default.
    """
    return {name: getattr(instance, name) for name in declarable_fields(type(instance))}


def _wire_value(value: Any) -> Any:
    to_wire = getattr(value, "to_wire", None)
    if callable(to_wire):
        return to_wire()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    return value
