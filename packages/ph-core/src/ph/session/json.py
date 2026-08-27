"""Lossless-JSON validation, detached snapshots, and immutability.

The session log is the durable source of truth, so a bad payload must fail at
the append site rather than later during a backend flush. `freeze_json_value`
validates, detaches and freezes in **one pass** — a value that is not losslessly
JSON-serializable is rejected, and what enters the log is a read-only copy the
caller cannot reach back into.

What "lossless" excludes, and why each one matters:

| rejected | why |
|---|---|
| `NaN` / `±Infinity` | not JSON; `json.dumps` emits bare `NaN`, which no other parser reads |
| `-0.0` | round-trips to `0`, silently changing the value |
| `int` outside ±(2**53-1) | survives Python's JSON but not a JavaScript
  reader's — and dsh tooling reads pH logs (Q2) |
| `tuple`, `set` | a tuple would come back a list: a type change nobody declared |
| `dict` subclasses, class instances | JSON keeps the fields and drops the type |
| non-`str` keys | `json.dumps` coerces `1` to `"1"` |
| cycles | unrepresentable |

Python has no `Object.freeze`, so the frozen form is `MappingProxyType` for
objects and `tuple` for arrays (D4). Re-admitting an already-frozen tree (a
seed taken from a live session) is the one case where a tuple *is* an array,
and `frozen_input=True` says so explicitly rather than weakening the rule.

@module ph.session.json
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, TypeAlias

__all__ = [
    "JSON_MAX_SAFE_INTEGER",
    "InvalidJsonValueError",
    "JsonEncoder",
    "JsonValue",
    "dumps",
    "freeze_json_value",
    "is_json_value",
    "snapshot_json_value",
    "thaw_json",
]

JsonValue: TypeAlias = "bool | int | float | str | list[Any] | dict[str, Any] | None"

JSON_MAX_SAFE_INTEGER = 2**53 - 1
"""`Number.MAX_SAFE_INTEGER`. Beyond it a JavaScript reader loses precision."""

_INFINITIES = (math.inf, -math.inf)


class InvalidJsonValueError(ValueError):
    """A value cannot round-trip through JSON losslessly."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path or '<root>'}: {reason}")


class _Walker:
    """One traversal: validate, detach, and (optionally) freeze.

    The path is kept as a stack of keys and only formatted on failure, so the
    success path allocates nothing beyond the copy itself.
    """

    __slots__ = ("_ancestors", "_freeze", "_frozen_input", "_trail")

    def __init__(self, *, freeze: bool, frozen_input: bool) -> None:
        self._freeze = freeze
        self._frozen_input = frozen_input
        self._trail: list[str | int] = []
        self._ancestors: set[int] = set()

    def _fail(self, reason: str) -> InvalidJsonValueError:
        parts: list[str] = []
        for key in self._trail:
            parts.append(f"[{key}]" if isinstance(key, int) else (f".{key}" if parts else key))
        return InvalidJsonValueError("".join(parts), reason)

    def walk(self, value: Any) -> Any:
        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, int):
            if abs(value) > JSON_MAX_SAFE_INTEGER:
                raise self._fail(f"integer {value} exceeds the JSON safe-integer range")
            return value
        if isinstance(value, float):
            if value != value or value in _INFINITIES:
                raise self._fail("NaN and infinities are not JSON")
            if value == 0.0 and math.copysign(1.0, value) < 0:
                raise self._fail("negative zero does not round-trip")
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, MappingProxyType)):
            if type(value) not in (dict, MappingProxyType):
                raise self._fail(f"{type(value).__name__} is a dict subclass, not a JSON object")
            return self._object(value)
        if isinstance(value, (list, tuple)):
            if type(value) is tuple and not self._frozen_input:
                raise self._fail("tuple would come back as a list")
            if type(value) not in (list, tuple):
                raise self._fail(f"{type(value).__name__} is a list subclass, not a JSON array")
            return self._array(value)
        raise self._fail(f"{type(value).__name__} is not JSON")

    def _enter(self, value: Any) -> None:
        identity = id(value)
        if identity in self._ancestors:
            raise self._fail("circular reference")
        self._ancestors.add(identity)

    def _object(self, value: Mapping[Any, Any]) -> Any:
        self._enter(value)
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise self._fail(f"object key {key!r} is not a string")
            self._trail.append(key)
            result[key] = self.walk(item)
            self._trail.pop()
        self._ancestors.discard(id(value))
        return MappingProxyType(result) if self._freeze else result

    def _array(self, value: Sequence[Any]) -> Any:
        self._enter(value)
        result: list[Any] = []
        for index, item in enumerate(value):
            self._trail.append(index)
            result.append(self.walk(item))
            self._trail.pop()
        self._ancestors.discard(id(value))
        return tuple(result) if self._freeze else result


def freeze_json_value(value: Any, *, frozen_input: bool = False) -> Any:
    """Validate, detach and freeze in one pass — the append path's entry point.

    :param frozen_input: accept `MappingProxyType`/`tuple` containers as the
        object/array forms, for re-admitting a tree this module already froze.
    :raises InvalidJsonValueError: when the value cannot round-trip losslessly.
    """
    return _Walker(freeze=True, frozen_input=frozen_input).walk(value)


def snapshot_json_value(value: Any) -> Any:
    """Validate and detach to a plain mutable copy, without freezing."""
    return _Walker(freeze=False, frozen_input=False).walk(value)


def is_json_value(value: Any) -> bool:
    """Test the same lossless boundary without keeping the copy."""
    try:
        _Walker(freeze=False, frozen_input=False).walk(value)
    except InvalidJsonValueError:
        return False
    return True


def thaw_json(value: Any) -> Any:
    """A plain mutable copy of a frozen tree, for callers that need `dict`/`list`."""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


class JsonEncoder(json.JSONEncoder):
    """Serializes the frozen views this module produces, without thawing.

    Tuples encode natively; `MappingProxyType` is handled in `default`.
    `allow_nan=False` is deliberate: a non-finite number should already have
    been refused at append, and a second refusal here is cheaper than emitting
    a token no other parser reads.
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("allow_nan", False)
        kwargs.setdefault("separators", (",", ":"))
        kwargs.setdefault("ensure_ascii", False)
        super().__init__(**kwargs)

    def default(self, o: Any) -> Any:
        if isinstance(o, MappingProxyType):
            return dict(o)
        return super().default(o)


_ENCODER = JsonEncoder()


def dumps(value: Any) -> str:
    """Canonical compact JSON for one log line or wire frame."""
    return _ENCODER.encode(value)
