"""Lossless-JSON validation, detached snapshots, and immutability: the A1 gate.

**Only the gate is here.** `ph.json` is a stdlib-only leaf. It holds the
vocabulary that this module validates against (`JsonValue`, `JsonObject`,
`PlainJsonValue`), the narrowings that a reader applies, and the conversions
between the two in-memory shapes. See its docstring for the cost and the benefit
of that separation. This module holds what A1 needs: the walk that decides
whether a payload may enter a log. The walk is here because `Session._append` is
the append site.

The session log is the durable source of truth. So a bad payload must fail at
the append site, and not later during a backend flush. `freeze_json_value`
validates, detaches and freezes in **one pass**. It rejects a value that is not
losslessly JSON-serializable. What enters the log is a read-only copy that the
caller cannot change afterward.

What "lossless" excludes, and why each one matters:

| rejected | why |
|---|---|
| `NaN` / `±Infinity` | not JSON: `json.dumps` emits a bare `NaN`, which no other parser reads |
| `-0.0` | round-trips to `0`, which changes the value without an error |
| `int` outside ±(2**53-1) | a JavaScript reader does not keep it (see below) |
| `tuple`, `set` | a tuple would come back as a list: a type change that nobody declared |
| `dict` subclasses, class instances | JSON keeps the fields and drops the type |
| non-`str` keys | `json.dumps` coerces `1` to `"1"` |
| cycles | cannot be represented |

An integer outside that range survives the JSON of Python but not the JSON of a
JavaScript reader, and dsh tooling reads pH logs (Q2).

Python has no `Object.freeze`. So the frozen form is `MappingProxyType` for
objects and `tuple` for arrays (D4). The re-admission of an already-frozen tree
(a seed taken from a live session) is the one case where a tuple *is* an array.
`frozen_input=True` states that case explicitly, so the rule is not weakened.

## Re-freezing costs real time, and the obvious fix is refused

Every pass rebuilds every container: a new `MappingProxyType` per object and a
new `tuple` per array. It does so even when the input is already exactly the
frozen form that the pass would produce. `frozen_input=True` accepts that shape.
It does not skip the copy. So a tree that has already passed through this module
pays for a second structural copy each time it is re-admitted. The cost grows
with the **node count** and not with the bytes, because strings pass through by
identity.

Measured (2026-09-07): re-freezing a streamed chunk payload takes **0.95 µs**,
and re-freezing a 500-node tool result takes **232 µs**. Four paths pay it:

- `Session.admit`, per wire event on a remote front end
- `Session(seed=…)`
- `resume_session`, on every daemon rehydrate
- `SessionStore.fork`, whose docstring already says that a fork is cheap on disk
  and not free in memory

The obvious optimization is a validation-only pre-walk. It checks the tree
against every rule above and, when the tree is already frozen, returns the
*input* and not a copy. It measured at roughly half the cost. **It is
deliberately not built**, and the reason is not that it looked hard:

> This function is where invariant A1 is enforced. It is the gate that decides
> what may enter a log. A fast path adds a second route through that gate. If
> the "already frozen?" predicate is wrong anywhere, the failure is a value that
> enters the log *unvalidated*. Nothing in the type system distinguishes a
> `MappingProxyType` that wraps a frozen tree from one that wraps a live mutable
> dict, or a `tuple` that holds frozen children from one that holds a `list`.
> So the predicate would be correct only as far as it was tested. A test suite
> is the wrong kind of guarantee for a gate, because it lists only the hostile
> shapes that somebody thought of.

So this optimization waits for a design whose correctness is **structural** and
not tested. One such design is a frozen tree that carries its own proof: for
example, a distinct wrapper type that the walker can recognize by identity, so
that "already frozen" is a type question and not an inspection. Another is a
freeze that is idempotent by construction. Until then the copy stays, and the
cost above is the price of a gate that has no second route. See DESIGN.md §8 and
plan row P6-44.

@module ph.session.json
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import overload

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

    **One route, and only one.** The walker formerly took a `freeze` flag,
    because `is_json_value` wanted the validation without the copy. That was the
    second route through this gate that the module docstring refuses. It was
    ready for use and no test covered it. That caller is gone, and so is the
    flag. What the walker returns is frozen, because everything that passes
    through it is frozen.

    The path is kept as a stack of keys and is formatted only on failure. So the
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

    def _object(self, value: Mapping[object, object]) -> Mapping[str, JsonValue]:
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

    def _array(self, value: Sequence[object]) -> Sequence[JsonValue]:
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
    """Validate, detach and freeze in one pass. This is the entry point of the append path.

    The function is overloaded on the shape of the input, so that an object in
    gives an object out. The walker preserves shape, and `SessionEvent.data` is
    declared a `JsonObject` because the envelope admits only an object. A caller
    that freezes a payload must not have to narrow what it already knows.

    :param frozen_input: accept `MappingProxyType`/`tuple` containers as the
        object/array forms, for the re-admission of a tree that this module
        already froze.
    :raises InvalidJsonValueError: when the value cannot round-trip losslessly.
    """
    return _Walker(frozen_input=frozen_input).walk(value)
