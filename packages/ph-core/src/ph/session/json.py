"""Lossless-JSON validation, detached snapshots, and immutability — the A1 gate.

**Only the gate.** The vocabulary this validates against (`JsonValue`,
`JsonObject`, `PlainJsonValue`), the narrowings a reader applies, and the
conversions between the two in-memory shapes all live in `ph.json`, a
stdlib-only leaf — see its docstring for what that separation costs and buys.
What is here is what A1 needs: the walk that decides whether a payload may enter
a log, and it is here because `Session.append` is the append site.

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

## Re-freezing costs real time, and the obvious fix is refused

Every container is rebuilt on every pass — a fresh `MappingProxyType` per object,
a fresh `tuple` per array — even when the input is already exactly the frozen form
this would produce. `frozen_input=True` accepts that shape; it does not skip the
copy. So a tree that has already been through here pays for a second structural
copy each time it is re-admitted, and the cost scales with **node count**, not
bytes, because strings pass through by identity.

Measured (2026-09-07): re-freezing a streamed chunk payload is **0.95 µs**; a
500-node tool result is **232 µs**. Four paths pay it — `Session.admit` per wire
event on a remote front end, `Session(seed=…)`, `resume_session` on every daemon
rehydrate, and `SessionStore.fork`, whose own docstring already says "a fork is
cheap on disk and not free in memory".

The obvious optimisation is a validation-only pre-walk: check the tree against
every rule above and, when it is already frozen, return the *input* rather than a
copy. It measured at roughly half the cost. **It is deliberately not built**, and
the reason is not that it looked hard:

> This function is where invariant A1 is enforced — it is the gate that decides
> what may enter a log. A fast path adds a second route through that gate, and if
> the "already frozen?" predicate is wrong anywhere, the failure is a value
> entering the log *unvalidated*. Nothing in the type system distinguishes a
> `MappingProxyType` wrapping a frozen tree from one wrapping a live mutable dict,
> or a `tuple` holding frozen children from one holding a `list`. The predicate
> would therefore be correct only insofar as it was tested — and a test suite is
> the wrong kind of guarantee for a gate, because it enumerates the hostile shapes
> somebody thought of.

So this waits for a design whose correctness is **structural** rather than tested:
a frozen tree that carries its own proof (a distinct wrapper type the walker can
recognise by identity, say, so "already frozen" is a type question rather than an
inspection), or a freeze that is idempotent by construction. Until then the copy
stays, and the cost above is the price of a gate that cannot be walked around.
See DESIGN.md §8 and plan row P6-44.

@module ph.session.json
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, overload

from ..json import JSON_MAX_SAFE_INTEGER, JsonObject, JsonValue

__all__ = [
    "InvalidJsonValueError",
    "freeze_json_value",
]

_INFINITIES = (math.inf, -math.inf)


class InvalidJsonValueError(ValueError):
    """A value cannot round-trip through JSON losslessly."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path or '<root>'}: {reason}")


class _Walker:
    """One traversal: validate, detach, freeze.

    **One route, and only one.** It took a `freeze` flag while `is_json_value`
    wanted the validation without the copy — which is the second route through
    this gate that the module docstring spends a paragraph refusing, standing
    ready and covered by no test. That caller is gone and so is the flag: what
    this returns is frozen because everything that passes through here is.

    The path is kept as a stack of keys and only formatted on failure, so the
    success path allocates nothing beyond the copy itself.
    """

    __slots__ = ("_ancestors", "_frozen_input", "_trail")

    def __init__(self, *, frozen_input: bool) -> None:
        self._frozen_input = frozen_input
        self._trail: list[str | int] = []
        self._ancestors: set[int] = set()

    def _fail(self, reason: str) -> InvalidJsonValueError:
        parts: list[str] = []
        for key in self._trail:
            parts.append(f"[{key}]" if isinstance(key, int) else (f".{key}" if parts else key))
        return InvalidJsonValueError("".join(parts), reason)

    def walk(self, value: object) -> JsonValue:
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

    def _enter(self, value: object) -> None:
        identity = id(value)
        if identity in self._ancestors:
            raise self._fail("circular reference")
        self._ancestors.add(identity)

    def _object(self, value: Mapping[Any, Any]) -> Mapping[str, JsonValue]:
        self._enter(value)
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise self._fail(f"object key {key!r} is not a string")
            self._trail.append(key)
            result[key] = self.walk(item)
            self._trail.pop()
        self._ancestors.discard(id(value))
        return MappingProxyType(result)

    def _array(self, value: Sequence[Any]) -> Sequence[JsonValue]:
        self._enter(value)
        result: list[JsonValue] = []
        for index, item in enumerate(value):
            self._trail.append(index)
            result.append(self.walk(item))
            self._trail.pop()
        self._ancestors.discard(id(value))
        return tuple(result)


@overload
def freeze_json_value(value: Mapping[str, object], *, frozen_input: bool = False) -> JsonObject: ...
@overload
def freeze_json_value(value: object, *, frozen_input: bool = False) -> JsonValue: ...
def freeze_json_value(value: object, *, frozen_input: bool = False) -> JsonValue:
    """Validate, detach and freeze in one pass — the append path's entry point.

    Overloaded on the input's shape so that an object in is an object out: the
    walker preserves shape, and `SessionEvent.data` is declared a `JsonObject`
    because the envelope only admits one — a caller freezing a payload should
    not have to narrow what it already knows.

    :param frozen_input: accept `MappingProxyType`/`tuple` containers as the
        object/array forms, for re-admitting a tree this module already froze.
    :raises InvalidJsonValueError: when the value cannot round-trip losslessly.
    """
    return _Walker(frozen_input=frozen_input).walk(value)
