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

import json
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, TypeAlias, overload

__all__ = [
    "JSON_MAX_SAFE_INTEGER",
    "InvalidJsonValueError",
    "JsonEncoder",
    "JsonObject",
    "JsonValue",
    "PlainJsonValue",
    "as_bool",
    "as_int",
    "as_obj",
    "as_seq",
    "as_str",
    "dumps",
    "freeze_json_value",
    "is_json_value",
    "thaw_json",
]

JsonValue: TypeAlias = (
    "bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue] | None"
)
"""A lossless-JSON tree, in either of the two shapes this module produces.

**Recursive, and over the abstract containers on purpose.** `list[Any] |
dict[str, Any]` — what this was — said nothing past the first level and could
not describe the frozen form at all, since a `MappingProxyType` is not a `dict`
and a `tuple` is not a `list`; a reader typed against it was untyped one index
down. `Sequence`/`Mapping` are true of both shapes — `tuple`/`MappingProxyType`
in memory, `list`/`dict` on disk and after `thaw_json` — and they are covariant,
so a producer holding `list[dict[str, Any]]` may hand it to `Session.append`
where `list[JsonValue]`, being invariant, would have refused it.

What the type does *not* say is "frozen". Nothing in the type system
distinguishes the two shapes (the module docstring's P6-44 paragraph); a
wrapper that carries its own proof is the deferred decision, and this alias is
written so that wrapper can subtype it later without moving a reader.

The wart is `str`: it is a `Sequence[str]`, and `str` is a `JsonValue`, so a
reader narrowing with `isinstance(x, Sequence)` meets it. `as_seq` below is
the one place that exclusion is spelled."""

JsonObject: TypeAlias = "Mapping[str, JsonValue]"
"""A JSON object — every event payload, and the shape `Session.append` takes."""

PlainJsonValue: TypeAlias = (
    "bool | int | float | str | list[PlainJsonValue] | dict[str, PlainJsonValue] | None"
)
"""The plain shape only: `list`/`dict` at every level, as `thaw_json` builds it
and as a log reads back from disk.

A second alias, because the first cannot say this. `JsonValue` is over the
abstract containers so that it is true of the frozen tree too — and abstract
containers cannot be assigned into. `compaction-summarize` thaws a payload
precisely in order to rewrite one block of it, and a return typed `Sequence`
would refuse the assignment the thaw exists to permit. A `PlainJsonValue` is a
`JsonValue` (list is a Sequence, dict a Mapping), so a thawed tree still flows
into `Session.append` without a cast; the reverse is not true, which is the
point."""

JSON_MAX_SAFE_INTEGER = 2**53 - 1
"""`Number.MAX_SAFE_INTEGER`. Beyond it a JavaScript reader loses precision."""

_INFINITIES = (math.inf, -math.inf)

_EMPTY_OBJECT: JsonObject = MappingProxyType({})
"""`as_obj`'s answer to a missing field. Hoisted because it is 8% of that call, and
read-only because a shared mutable empty is an aliasing hazard the moment a
caller writes to a narrowed result."""


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
        return MappingProxyType(result) if self._freeze else result

    def _array(self, value: Sequence[Any]) -> Sequence[JsonValue]:
        self._enter(value)
        result: list[JsonValue] = []
        for index, item in enumerate(value):
            self._trail.append(index)
            result.append(self.walk(item))
            self._trail.pop()
        self._ancestors.discard(id(value))
        return tuple(result) if self._freeze else result


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
    return _Walker(freeze=True, frozen_input=frozen_input).walk(value)


def is_json_value(value: object) -> bool:
    """Test the same lossless boundary without keeping the copy."""
    try:
        _Walker(freeze=False, frozen_input=False).walk(value)
    except InvalidJsonValueError:
        return False
    return True


@overload
def thaw_json(value: Mapping[str, object]) -> dict[str, PlainJsonValue]: ...
@overload
def thaw_json(value: list[object] | tuple[object, ...]) -> list[PlainJsonValue]: ...
@overload
def thaw_json(value: object) -> PlainJsonValue: ...
def thaw_json(value: object) -> PlainJsonValue:
    """A plain mutable copy of a frozen tree, for callers that need `dict`/`list`.

    Overloaded on the input's shape, as `freeze_json_value` is, so an object
    thaws to a `dict` *in the type* — which is what a caller that thaws in order
    to **mutate** needs: `compaction-summarize` rewrites one block of a thawed
    `assistant/message` and re-appends it.

    The array overload is selected by a `tuple` — the frozen array shape, which
    is covariant — and **not** by a `list[str]` or any other concretely
    parameterised list, because `list` is invariant and `list[str]` is not a
    `list[object]`. Those fall through to the third overload and get the union.
    No production caller passes a statically typed list, so this is a documented
    limit rather than a gap to close.
    """
    if isinstance(value, Mapping):
        # No `str(key)`: `freeze_json_value` refuses a non-string key at the
        # gate, so a tree from this module cannot have one, and the coercion
        # measured at 11% of a 5,000-event thaw — which `to_wire` pays per
        # event, per attached front end.
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value  # type: ignore[return-value]  # a scalar this module admitted


def as_int(value: object, default: int = 0) -> int:
    """A JSON number as an `int`, or `default` — the reader's narrowing for one.

    Twenty readers wrote `int(event.data.get("turn", 0))`, and with `data` typed
    that is an `int()` of a union a `Mapping` belongs to, which the checker
    refuses. This is the third of the family, and it now answers the way the
    other two do.

    **It used to raise, and that was the odd one out.** `as_obj` and `as_seq`
    answer a mis-shaped field with the empty container — "a missing one must
    cost a row rather than the transcript" — while this raised a `TypeError` on
    anything that was not a number. On `persistence.repair`,
    `agent_loop.driver._last_turn_of` and `llm.replay.recorded_steps` that raise
    is not contained: one mistyped numeric field in a log some other build wrote
    turned a single unreadable row into a **failed resume**. That is the policy
    the family shares: a mis-shaped field costs a row, not a raise.

    **What this one does *not* share is the strictness.** `as_str` and `as_bool`
    narrow — not the type, so nothing — while this keeps every coercion `int()`
    made. The difference is what each builtin got wrong: `str()` fabricates and
    `bool()` inverts, so their coercions had to go; `int()`'s coercions are
    fine and only its *raising* was the defect. The visible edge is that
    `as_int(True)` is `1` while `as_bool(1)` is the default — deliberate, pinned
    on both sides, and the reason `as_bool` is not simply this function's shape
    with a different type.

    **What the default is for.** `freeze_json_value` is a JSON-*ness* gate, not
    a schema gate: it walks a payload for values JSON can round-trip, and a
    `str`, a `None` or a list is one. `Session.append({"turn": "3"})` succeeds.
    So the population reaching the default is not only a foreign build or a
    hand-edited log — it includes a producer of ours writing the wrong type into
    a numeric field — and a field that is simply absent, which is why `None`
    takes it too and why the call sites pass `as_int(data.get("turn"))` rather
    than defaulting twice.

    That the default is *quiet* is right for a cosmetic number and a real cost
    for a load-bearing one: a junk `turn` reaching `driver._last_turn_of` gives
    a resumed run a duplicate turn number, which is worse than the failed resume
    it replaced because nothing says so. Refusing it belongs at the row — a
    typed payload for `turn/start` and `step/start` — which is issue 74's work
    and not this helper's.

    The coercions `int()` made are unchanged: a `float` truncates and a numeric
    `str` parses. What changed is that a container, `None`, a non-numeric string
    and a non-finite float now read as `default` instead of raising.
    """
    if type(value) is int:
        # The whole point of the helper, and ~every call: a JSON integer,
        # already an `int`, returned without the `isinstance` walk behind it
        # (22 ns against 110). `type(...) is int` excludes `bool` deliberately —
        # it is a subclass, and it belongs on the coercing branch below.
        return value
    if isinstance(value, (bool, float, str)):
        # Exactly what `int()` accepted. `NaN` and `±Infinity` are the reason
        # this catches two exceptions rather than one: `json.loads` parses both
        # from a log that spells them bare, and they raised straight through
        # here — `ValueError` and `OverflowError` respectively — which left the
        # hole this row set out to close while claiming one policy for the
        # family. A try that never fires costs nothing measurable.
        try:
            return int(value)
        except (ValueError, OverflowError):
            return default
    return default


def as_bool(value: object, default: bool = False) -> bool:
    """A JSON boolean, or `default` — and pointedly **not** `bool()`.

    The fifth of the family and the one whose builtin is most dangerous, because
    `bool()` never raises and never looks wrong: it answers *truthiness*, which
    is defined for every JSON value and is the wrong question for a boolean
    field. **`bool("false")` is `True`.** A log that spells a flag as a string —
    a foreign build, a hand-edited file, a producer of ours writing the wrong
    type, all of which `freeze_json_value` admits because it is a JSON-*ness*
    gate and not a schema gate — reads as the opposite of what it says. On
    `isError` that draws an error card for a tool call that succeeded.

    **This is where the family's rule divides, and the division is the point.**
    `as_int` keeps every coercion `int()` made — a float truncates, a numeric
    string parses — and only replaces the *raise*, because raising was its
    defect. `bool()` has no raise to replace; its coercion **is** the defect. So
    this narrows instead: a `bool` is a `bool`, and everything else is `default`.

    `1` therefore reads as `default`, not `True`. That is deliberate and it is
    the same judgement: JSON has a boolean type, a producer that wrote `1` into
    a boolean field did not write a boolean, and guessing which one they meant is
    the fabrication this family exists to refuse.

    **Where `bool()` is still right**, and why three callers keep it: a
    *presence* check. `bool(self._state["next-turn"] or self._state["next-step"])`
    asks "is anything queued" of a Python list, and
    `bool(os.environ.get(name))` asks "is this variable set and non-empty".
    Neither is reading a JSON boolean, and truthiness is exactly the question.
    """
    return value if isinstance(value, bool) else default


def as_str(value: object, default: str = "") -> str:
    """A JSON string, or `default` — the fourth of the family, and the last.

    One policy, the one `as_int`'s docstring argues for: a mis-shaped field
    answers with the empty value rather than raising, because a reader of a log
    some other build wrote must lose a row and not a session.

    **Not `str(value)`**, which is what a reader writes without this and is worse
    than useless: `str(None)` is `"None"` and `str(3)` is `"3"`, so a field that
    is absent or of the wrong type comes back as a plausible-looking answer that
    no assertion catches. Narrowing says "this was not a string" by giving back
    nothing, which is the same thing `as_obj` and `as_seq` say.
    """
    return value if isinstance(value, str) else default


def as_obj(value: object) -> JsonObject:
    """A JSON object, or an empty one — the reader's narrowing for a payload field.

    Absence is normal: the log is JSON, every field is optional to a reader, and
    a missing one must cost a row rather than the transcript. Here rather than in
    `ph_app.wire`, where it was born, because the question it answers is about the
    tree and not the app: a core reader chaining `data.get("message").get(...)`
    needs exactly this narrowing, and had no typed way to say so.

    **The test names the two shapes rather than asking the ABC.** `isinstance`
    against `Mapping` goes through `ABCMeta.__instancecheck__`, and a
    `MappingProxyType` — the in-memory form, so the common case — is its worst
    input at 220 ns against 56 ns for the concrete pair. This runs per field per
    event on the crash-repair scan and the TUI fold; measured over 8,499 events
    the ABC form cost 3.2x the concrete one. Naming `dict` and `MappingProxyType`
    is not a weakening either, because the D4 paragraph above says those are the
    only two object shapes this module produces or reads.
    """
    return value if isinstance(value, (dict, MappingProxyType)) else _EMPTY_OBJECT


def as_seq(value: object) -> Sequence[JsonValue]:
    """A JSON array, or an empty one. A tuple in memory, a list on disk.

    The frozen/plain duality is why this is a function and not an
    `isinstance(value, list)` — a reader that tested for `list` worked on resume
    and silently saw nothing live. Naming both shapes also makes the `str`
    exclusion **structural** rather than a special case: `str` is a `Sequence`,
    so an ABC test admits it and a reader that forgot iterated a word's letters
    as rows; `str` is neither a `list` nor a `tuple`, so it simply falls out.
    Same measurement as `as_obj`: 244 ns against 56 ns, per field per event.
    """
    return value if isinstance(value, (list, tuple)) else ()


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


def dumps(value: object) -> str:
    """Canonical compact JSON for one log line or wire frame."""
    return _ENCODER.encode(value)
