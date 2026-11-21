"""The JSON Schema subset that pH validates, and where a schema comes from.

A tool declares its arguments and its canonical output. Both are JSON Schema on
the wire, because a provider accepts that form and an MCP server supplies it.
Inside pH a declaration may instead be a **pydantic model**. That is strictly
better when the tool is written in Python: one definition produces the schema
*and* the validator, and the body receives a typed object.

The raw-dict path exists for schemas that pH did not author (MCP, the declared
shape of a subagent). It validates a deliberate subset: `type`, `enum`, `const`,
`required`, `properties`, `items`, `additionalProperties`, `anyOf`, `pattern`,
and the numeric, length and item-count bounds. The module **reports the limits
of that subset**. A keyword outside the subset is ignored, but it is not
silently treated as satisfied: `unsupported_keywords` reports what was skipped,
so a caller can refuse a schema that pH cannot fully enforce.

@module ph.tools.json_schema
"""

from __future__ import annotations

import json
import math
import re
from functools import cache
from typing import Any

from pydantic import BaseModel, ValidationError

from ..json import JsonValue, thaw_json
from ..wire import validation_errors

__all__ = [
    "CONSTRAINING_KEYWORDS",
    "SUPPORTED_KEYWORDS",
    "checked_object_schema",
    "object_schema_problems",
    "schema_of",
    "unsupported_keywords",
    "validate_json_schema_value",
]

SUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "enum",
        "const",
        "required",
        "properties",
        "items",
        "additionalProperties",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "anyOf",
        # Annotations: carried to the model, never validated.
        "title",
        "description",
        "default",
        "examples",
        "$defs",
        "$ref",
        "definitions",
    }
)

CONSTRAINING_KEYWORDS: frozenset[str] = frozenset(
    {"type", "anyOf", "oneOf", "allOf", "$ref", "enum", "const"}
)
"""The keywords that make a `properties` entry say something about its value.

This set is beside `SUPPORTED_KEYWORDS` because both are the vocabulary of this
module. It is **deliberately not a subset of it**. `SUPPORTED_KEYWORDS` lists
what *the validator of pH* enforces. This set lists what a **grammar built from
the schema** would constrain. `oneOf` and `allOf` are in this set and not in
`SUPPORTED_KEYWORDS`. A route that compiles the schema honors them.
`validate_json_schema_value` does not, and `unsupported_keywords` correctly
reports that. `anyOf` is in both sets. Pydantic writes `str | None` as an
`anyOf`, so the schema of a pydantic model is one that pH can hold a value to.
The two questions are answered in one place, so that the next reader sees the
difference and does not have to discover it.

`llm.structured.structural_warning` reads this set. It used to test
`"type" in entry` alone. Every schema with an optional required field then
warned that its grammar would not constrain the reply, because pydantic renders
`str | None` as an `anyOf` and a `Literal` as an `enum` (C11).
"""

_TYPE_CHECKS: dict[str, Any] = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def schema_of(declaration: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    """The JSON Schema for a declaration, whichever form it took."""
    if isinstance(declaration, type) and issubclass(declaration, BaseModel):
        return _model_schema(declaration)
    return declaration


@cache
def _model_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The schema of one model, built one time.

    Pydantic does not memoize `model_json_schema()`. Every call was measured at
    903 µs for `RefinementProposal` and 226 µs for `ReviewVerdict`, and the
    schema of a class is a constant. This matters because `ask_for_shape` asks
    for one schema for each model request. There it was the largest single cost
    that the structured path introduced.

    The cache is keyed on the class, which lives as long as the process. So the
    number of declarations bounds the cache, and nothing that a caller supplies
    does."""
    return model.model_json_schema()


def unsupported_keywords(schema: object) -> set[str]:
    """Every keyword in `schema` that this validator does not enforce.

    The keywords are reported, and are not assumed to hold. A schema that pH
    cannot fully check should be refused by its author, and not silently
    half-validated.

    `pattern` is reported when Python cannot compile it. A JSON Schema pattern
    is an ECMA 262 expression, and `re` does not accept all of them (`\\p{L}` is
    one). The validator skips such a pattern, so the keyword is not enforced
    there.
    """
    found: set[str] = set()
    if isinstance(schema, dict):
        found.update(key for key in schema if key not in SUPPORTED_KEYWORDS)
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and not _compiles(pattern):
            found.add("pattern")
        for key in ("properties", "$defs", "definitions"):
            nested = schema.get(key)
            if isinstance(nested, dict):
                for child in nested.values():
                    found |= unsupported_keywords(child)
        for key in ("items", "additionalProperties"):
            nested = schema.get(key)
            if isinstance(nested, dict):
                found |= unsupported_keywords(nested)
        branches = schema.get("anyOf")
        if isinstance(branches, list):
            for branch in branches:
                found |= unsupported_keywords(branch)
    return found


def _compiles(pattern: str) -> bool:
    """Whether `re` accepts `pattern`."""
    try:
        re.compile(pattern)
    except re.error:
        return False
    return True


def object_schema_problems(schema: object) -> list[str]:
    """Why `schema` cannot be the declared shape of a value that pH holds a caller to.

    The possible problems are: it is not a mapping, its root is not
    `type: object`, or it uses a keyword that this validator does not enforce.
    The list is empty when the schema can be that shape.

    This is for a schema that pH did not author: the `output` of a skill or of a
    spawn. A schema that uses a keyword that pH would skip is refused, and not
    half-checked (`unsupported_keywords`). The root must be an object, because
    the value is also the arguments of a tool.
    """
    schema = thaw_json(schema)
    if not isinstance(schema, dict):
        return ["it is not a mapping"]
    problems: list[str] = []
    if schema.get("type") != "object":
        problems.append("its root is not `type: object`")
    unsupported = unsupported_keywords(schema)
    if unsupported:
        problems.append(f"it uses keywords pH does not check: {', '.join(sorted(unsupported))}")
    return problems


def checked_object_schema(schema: object) -> dict[str, Any]:
    """`schema` as the plain dict that a value is held to. Raises `ValueError`
    with the reason when it cannot be one (`object_schema_problems`). This is
    the one check for every caller that takes a declared shape."""
    plain = thaw_json(schema)
    problems = object_schema_problems(plain)
    if problems or not isinstance(plain, dict):
        raise ValueError("; ".join(problems))
    return plain


def validate_json_schema_value(
    schema: type[BaseModel] | dict[str, Any], value: object
) -> list[str]:
    """The violations of `value` against `schema`, in validation order.

    An empty list means that the value is valid. A pydantic declaration
    delegates to pydantic, so the model of the tool is the validator.

    Both JSON forms are checked: frozen (`MappingProxyType` and `tuple`, which
    is how the arguments of a tool and a logged schema arrive) and plain. The
    function thaws them here, one time, because the checks below are `dict` and
    `list` checks. A frozen schema that was not thawed would pass everything
    below its root unchecked.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            schema.model_validate(value)
        except ValidationError as error:
            return validation_errors(error)
        return []
    plain = thaw_json(schema)
    violations: list[str] = []
    if isinstance(plain, dict):
        _validate(plain, thaw_json(value), "", plain, violations)
    return violations


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local `$ref` one hop at a time. A foreign ref stays unresolved."""
    seen: set[str] = set()
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        ref: str = schema["$ref"]
        if ref in seen or not ref.startswith("#/"):
            return schema
        seen.add(ref)
        target: Any = root
        for part in ref.removeprefix("#/").split("/"):
            if not isinstance(target, dict) or part not in target:
                return schema
            target = target[part]
        if not isinstance(target, dict):
            return schema
        schema = target
    return schema


# (keyword, comparison) pairs. They are written out and not built from lambdas,
# so that the direction of each comparison is readable where it is declared.
_NUMERIC_BOUNDS: tuple[tuple[str, str], ...] = (
    ("minimum", ">="),
    ("maximum", "<="),
    ("exclusiveMinimum", ">"),
    ("exclusiveMaximum", "<"),
)
_LENGTH_BOUNDS: tuple[tuple[str, str], ...] = (("minLength", ">="), ("maxLength", "<="))
_ITEM_BOUNDS: tuple[tuple[str, str], ...] = (("minItems", ">="), ("maxItems", "<="))


def _compare(left: float, operator: str, right: float) -> bool:
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left < right


def _check_bounds(
    schema: dict[str, Any],
    where: str,
    out: list[str],
    measured: float,
    bounds: tuple[tuple[str, str], ...],
) -> None:
    for keyword, operator in bounds:
        bound = schema.get(keyword)
        if (
            isinstance(bound, (int, float))
            and not isinstance(bound, bool)
            and not _compare(measured, operator, bound)
        ):
            out.append(f"{where}: fails {keyword} {bound}")


def _validate(
    schema: object,
    value: object,
    path: str,
    root: dict[str, Any],
    out: list[str],
) -> None:
    if not isinstance(schema, dict):
        return
    schema = _resolve(schema, root)
    where = path or "<root>"

    branches = schema.get("anyOf")
    if isinstance(branches, list) and branches:
        failed: list[list[str]] = []
        for branch in branches:
            found: list[str] = []
            _validate(branch, value, path, root, found)
            if not found:
                break
            failed.append(found)
        else:
            tried = " | ".join("; ".join(one) for one in failed)
            out.append(f"{where}: matches none of its alternatives ({tried})")
            return

    declared = schema.get("type")
    types = declared if isinstance(declared, list) else [declared] if declared else []
    if types:
        checks = [_TYPE_CHECKS[name] for name in types if name in _TYPE_CHECKS]
        if checks and not any(check(value) for check in checks):
            out.append(f"{where}: expected {' or '.join(str(t) for t in types)}")
            return

    if "const" in schema and value != schema["const"]:
        out.append(f"{where}: must equal {schema['const']!r}")
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        out.append(f"{where}: must be one of {schema['enum']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            out.append(f"{where}: must be a finite number")
        _check_bounds(schema, where, out, value, _NUMERIC_BOUNDS)

    if isinstance(value, str):
        _check_bounds(schema, where, out, len(value), _LENGTH_BOUNDS)
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    out.append(f"{where}: does not match {pattern!r}")
            except re.error:
                # A pattern that pH cannot compile is a pattern that it does not
                # enforce. `unsupported_keywords` reports it.
                pass

    if isinstance(value, list):
        _check_bounds(schema, where, out, len(value), _ITEM_BOUNDS)
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate(items, item, f"{path}[{index}]", root, out)

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    out.append(f"{where}: missing required property {key!r}")
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for key, item in value.items():
            child = properties.get(key)
            if isinstance(child, dict):
                _validate(child, item, f"{path}.{key}" if path else str(key), root, out)
            elif schema.get("additionalProperties") is False:
                out.append(f"{where}: unexpected property {key!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(
                    schema["additionalProperties"],
                    item,
                    f"{path}.{key}" if path else str(key),
                    root,
                    out,
                )


def parse_arguments(raw: str) -> JsonValue:
    """Parse the arguments from the model, and keep invalid JSON as text.

    A malformed argument string is a problem for the *tool* to report, and not
    one for the loop to crash on. The tool sees the raw text and fails with a
    message that the model can act on. `JsonValue` states exactly that at no
    cost, because a `str` is a `JsonValue`. The raw text is also read
    downstream: the destructive gate of `ph_stabilize` scans it for a call whose
    JSON never parsed.

    The function is here, in the leaf module of the package, because
    `presentation.py` needs it too, and `batch` is three imports above
    `presentation.py`: `presentation` → `batch` → `definition` → `presentation`
    is a cycle. It was never a concern of the batch. It is `json.loads` with one
    documented fallback.
    """
    if not raw:
        return {}
    try:
        parsed: JsonValue = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return parsed
