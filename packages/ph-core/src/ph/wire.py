"""One rule for every JSON boundary: declare aliases, never derive them (Q2).

pH uses snake_case in Python and camelCase on the wire. The wire is the session
JSONL, the `json` and `rpc` output, and the fd-3 runtime frames.
The choice is not cosmetic. The envelope of dsh is already camelCase, so dsh
tooling reads a pH log directly (D2).

The mechanism is one `ConfigDict` on a shared base. Aliases are fixed when a
class is defined, so a field name is never reconstructed from a wire string.
`to_camel` followed by `to_snake` happens to round-trip for every field that
exists today, but code that relied on that would be fragile at acronyms and
digits.

`populate_by_name=True` makes every reader accept both forms. For that reason,
a JSONL that was written with snake_case names needs no second parser.

Hot-path envelopes (`SessionEvent`, the stream chunks) are frozen dataclasses
and not pydantic models (D4). `WireDataclass` gives them the same wire rule
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

from .json import JsonValue, thaw_json

__all__ = [
    "EnvName",
    "WireDataclass",
    "WireForm",
    "WireModel",
    "declarable",
    "declarable_fields",
    "literal_lookup",
    "validation_errors",
    "validation_summary",
    "wire_alias",
]

_Member = typing.TypeVar("_Member", bound=str)


@dataclasses.dataclass(frozen=True, slots=True)
class EnvName:
    """Marks a config field whose value is the **name** of an environment
    variable that the command of a row reads:
    `api_key_env: Annotated[str, EnvName()]`.

    The mark is declared one time, on the field. Every place that describes the
    command reads it (`ph.toolcli.env_names`). The manual of a tool command
    lists the variable, because a UTCP client gives a command only the variables
    that its template names. The row of a session passes the variable to the
    command by name. The mark holds a name, never the value of the variable. The
    class is here, beside the models that it marks, so that a config module
    needs nothing else."""


def wire_alias(field_name: str) -> str:
    """The one alias function. It is exported so that tests can check mappings against it."""
    return to_camel(field_name)


def literal_lookup(alias: object) -> collections.abc.Mapping[str, _Member]:
    """Every member of a `Literal`, keyed by its own spelling (the read side of Q2).

    A `Literal` constrains writers, but its guarantee ends at the payload
    boundary. A reader that holds a `str` must therefore check it. `raw in
    get_args(X)` is that check, and it does **not** narrow the type. For that
    reason, every such site had a `cast` beside it that asserted what the test
    had just established. A lookup keyed by the members *is* the check, and it
    narrows: `MODES.get(raw)` is the member or `None`. No cast can become stale,
    and a member that is added to the alias appears here with no other change.

    Before this function existed, six sites hand-wrote
    `{one: one for one in get_args(X)}`, and each had a copy of the paragraph
    above. The declaration is still the single source. This function only stops
    the repetition.

    **The caller must annotate the result.** The member type comes from the
    annotation, not from the argument. `get_args` returns `tuple[Any, ...]`, and
    a `Literal` alias is not a `type[X]`, so the function cannot read the member
    type from the parameter without the `TypeForm` of PEP 747. The bound is
    `str`, so the annotation cannot be a type that a `.get` would narrow
    wrongly. A missing annotation is a hard `error: Need type annotation` and
    not a silent `Any`. The *narrowing* form of this helper
    (`as_literal(raw, alias) -> X | None`) does not have that property, and for
    that reason it is not written here. Nothing checks that the annotation is
    paired with the correct alias. The hand-written form did not check that
    either.
    """
    return {one: one for one in typing.get_args(alias)}


def validation_errors(error: ValidationError, *, root: str = "<root>") -> list[str]:
    """Pydantic's error list as sentences, one per violation, in validation order.

    The function is here because more than one place renders the list. The first
    two were `validate_json_schema_value`, for the arguments of a tool, and
    `ph_app.protocol.parse_params`, for the parameters of an RPC method. They
    rendered the list the same way only by coincidence. The text that a person
    reads when a call is refused must not depend on the route that the call
    took. The two copies would also have drifted the first time that either
    wanted the offending input or the `type` of the error in the text.

    `root` is the name for a violation with no path, where the whole value was
    wrong. The callers differ only there. For example, a schema names it
    `<root>`, and a method names it `params`, because that is what the caller
    sent.
    """
    return [
        f"{'.'.join(str(part) for part in one['loc']) or root}: {one['msg']}"
        for one in error.errors()
    ]


def validation_summary(error: ValidationError, *, root: str = "<root>") -> str:
    """`validation_errors` in one line: the reason that a refusal gives, with one
    clause per violation. Before this function existed, five refusals joined the
    errors by hand, each with its own `root`."""
    return "; ".join(validation_errors(error, root=root))


class WireForm:
    """Declares that a value serializes itself to wire JSON, and nothing else.

    It is separate from `WireDataclass` because that base means a second thing:
    *keys derived from `wire_alias`*. `SubagentRun` serializes itself but cannot
    take the derived body. It holds a `Grant`, a `Context`, a `Disposer` and a
    callable, and the alias body emits every non-`None` field. If `SubagentRun`
    inherited `WireDataclass` to satisfy one `isinstance`, it would declare
    conformance to a rule that the class contradicts.

    **The metaclass is plain on purpose.** `_wire_value` tests for this class
    for each field of every `to_wire`, so the common case is a scalar that
    matches nothing. Against `WireModel` directly, that miss runs
    `ModelMetaclass.__instancecheck__` of pydantic, which is one Python frame
    per miss. It measured 40 ns against 19 ns here, and 69 ns when the two bases
    were tested as a tuple.
    """

    __slots__ = ()

    def to_wire(self) -> dict[str, JsonValue]:
        """The camelCase JSON form of this value."""
        raise NotImplementedError


class WireModel(WireForm, BaseModel):
    """The base of every model that crosses a JSON boundary.

    A model dumps by alias (camelCase), validates either form, and is frozen. A
    model that reached the wire is a value, not a mutable buffer.

    A model also accepts **frozen log data** directly. The session log freezes
    payloads into `MappingProxyType` and `tuple`, and pydantic coerces those
    into declared `dict` and `list` fields. But a field typed `Any` keeps what
    it was given, so a JSON array that was read back from the log would stay a
    tuple and compare unequal to an identical list. That broke `header_equals`
    one time, silently, and prefix caching with it. The input is thawed here, at
    the one layer that every model shares, so that no call site must remember to
    do it.
    """

    model_config = ConfigDict(
        alias_generator=wire_alias,
        populate_by_name=True,
        frozen=True,
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def _thaw_frozen_input(cls, data: object) -> object:
        if isinstance(data, MappingProxyType):
            return thaw_json(data)
        return data

    def to_wire(self) -> dict[str, JsonValue]:
        """The camelCase JSON form, with absent optional fields omitted.

        `JsonValue` is a claim about the *fields*. `model_dump` without
        `mode="json"` returns whatever Python objects the fields hold, so a
        `datetime` or enum field would break the claim. `test_wire_forms`
        enforces that no such field exists, so the claim does not depend on
        this docstring.
        """
        return self.model_dump(by_alias=True, exclude_none=True)


class WireDataclass(WireForm):
    """A mixin that gives a frozen dataclass the `WireModel.to_wire()` contract.

    A `type` field, when present, is emitted first, so that a discriminated
    reader sees it before anything else. `None` fields are omitted. A nested
    value that can serialize itself is asked to do so.
    """

    __slots__ = ()

    def to_wire(self) -> dict[str, JsonValue]:
        wire: dict[str, JsonValue] = {}
        fields = dataclasses.fields(self)  # type: ignore[arg-type]
        ordered = sorted(fields, key=lambda field: field.name != "type")
        for field in ordered:
            # This is the one real type erasure in this walk: `getattr` over
            # `dataclasses.fields` is untyped by construction. The value is
            # narrowed here so that `_wire_value` and the returned mapping are
            # checked and not asserted.
            value: JsonValue | WireForm = getattr(self, field.name)
            if value is None:
                continue
            wire[wire_alias(field.name)] = _wire_value(value)
        return wire


def declarable_fields(dataclass_type: type) -> tuple[str, ...]:
    """The fields of a definition that can travel: every field that is not a callable.

    The fields are derived from the annotations and not kept in a list, because
    a list drifts. The plan row that this serves (P7-11) began with a hand-kept
    table of "the one field that cannot travel" for each definition. Such a
    table silently stops being true when a second callable is added.
    `get_type_hints` is needed because the seams use
    `from __future__ import annotations`, so a bare `field.type` is a string.
    """
    hints = typing.get_type_hints(dataclass_type)
    return tuple(
        field.name
        for field in dataclasses.fields(dataclass_type)
        if typing.get_origin(hints.get(field.name)) is not collections.abc.Callable
    )


def declarable(instance: object) -> dict[str, Any]:
    """The part of a definition that travels, as the kwargs of its schema.

    A `schema()` that is built as `Schema.model_validate(declarable(self))`
    names no field, so a field that is added to the definition is carried
    without an edit. `WireModel` forbids extras. A field that is added to the
    definition and *not* to the schema therefore fails at the first call, and
    does not quietly reach the wire as the default of the schema.
    """
    return {name: getattr(instance, name) for name in declarable_fields(type(instance))}


def _wire_value(value: JsonValue | WireForm) -> JsonValue:
    """One nested field, as the wire carries it.

    The test for "can this serialize itself" is nominal: a type that is meant to
    be nested declares `WireForm`. A `runtime_checkable` Protocol would express
    the same test structurally, and it is deliberately not used. This function
    runs for each field of every `to_wire`, so the common case is a scalar that
    matches nothing. The `isinstance` of a Protocol measured about 60x slower on
    exactly that miss path.
    """
    if isinstance(value, WireForm):
        return value.to_wire()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    return value
