"""The JSON-Schema subset pH validates, and where it comes from.

A tool declares its arguments and its canonical output. Both are JSON Schema on
the wire, because that is what a provider accepts and what an MCP server hands
over. Inside pH a declaration may instead be a **pydantic model**, which is
strictly better when the tool is written in Python: one definition produces the
schema *and* the validator, and the body receives a typed object.

The raw-dict path exists for schemas pH did not author (MCP, a subagent's
declared shape). It validates a deliberate subset — `type`, `enum`, `const`,
`required`, `properties`, `items`, `additionalProperties`, and the numeric and
string bounds — and **says so**: a keyword outside the subset is ignored rather
than silently treated as satisfied, and `unsupported_keywords` reports what was
skipped so a caller can refuse a schema pH cannot fully enforce.

@module ph.tools.json_schema
"""

from __future__ import annotations

import json
import math
import re
from functools import cache
from typing import Any

from pydantic import BaseModel

__all__ = [
    "SUPPORTED_KEYWORDS",
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
    """One model's schema, built once.

    Pydantic does not memoize `model_json_schema()` — measured at 903 µs for
    `RefinementProposal` and 226 µs for `ReviewVerdict`, every call — and a
    class's schema is a constant. It matters because `ask_for_shape` asks for one
    per model request, where it was the largest single cost the structured path
    introduced.

    Keyed on the class, which is a process-lifetime object, so the cache is
    bounded by the number of declarations rather than by anything a caller
    supplies."""
    return model.model_json_schema()


def unsupported_keywords(schema: Any) -> set[str]:
    """Every keyword in `schema` that this validator does not enforce.

    Reported rather than assumed: a schema pH cannot fully check should be
    refused by its author, not quietly half-validated.
    """
    found: set[str] = set()
    if isinstance(schema, dict):
        found.update(key for key in schema if key not in SUPPORTED_KEYWORDS)
        for key in ("properties", "$defs", "definitions"):
            nested = schema.get(key)
            if isinstance(nested, dict):
                for child in nested.values():
                    found |= unsupported_keywords(child)
        for key in ("items", "additionalProperties"):
            nested = schema.get(key)
            if isinstance(nested, dict):
                found |= unsupported_keywords(nested)
    return found


def validate_json_schema_value(schema: type[BaseModel] | dict[str, Any], value: Any) -> list[str]:
    """Violations of `value` against `schema`, in validation order.

    An empty list means valid. A pydantic declaration delegates to pydantic, so
    the tool's own model is the validator.
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        from pydantic import ValidationError

        try:
            schema.model_validate(value)
        except ValidationError as error:
            return [
                f"{'.'.join(str(part) for part in item['loc']) or '<root>'}: {item['msg']}"
                for item in error.errors()
            ]
        return []
    violations: list[str] = []
    _validate(schema, value, "", schema, violations)
    return violations


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Follow a local `$ref` one hop at a time; a foreign ref stays unresolved."""
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


# (keyword, comparison) pairs. Spelled out rather than built from lambdas so the
# comparison direction is readable where it is declared.
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


def _validate(schema: Any, value: Any, path: str, root: dict[str, Any], out: list[str]) -> None:
    if not isinstance(schema, dict):
        return
    schema = _resolve(schema, root)
    where = path or "<root>"

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
                # A schema pH cannot compile is a schema it does not enforce;
                # `unsupported_keywords` is where that gets reported.
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


def parse_arguments(raw: str) -> Any:
    """Parse model arguments, preserving invalid JSON as text.

    A malformed argument string is the *tool's* problem to report, not the
    loop's to crash on: the tool sees the raw text and fails with a message the
    model can act on.

    Here, in the package's leaf module, because `presentation.py` needs it too
    and `batch` is three imports above it — `presentation` → `batch` →
    `definition` → `presentation` is a cycle. It was never a batch concern: it
    is `json.loads` with one documented fallback.
    """
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
