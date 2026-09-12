"""JSON as a *representation*: what a value may be, how to narrow it, how to
carry it.

Four groups, and they are one subject: the vocabulary (`JsonValue`,
`JsonObject`, `PlainJsonValue`), the five narrowings a reader applies to an
untrusted field, the conversion between the two in-memory shapes (`thaw_json`),
and the conversion to text (`JsonEncoder`, `dumps`) — plus
`JSON_MAX_SAFE_INTEGER`, the bound both the gate and the RLM codec check against.
None of them knows what a session is.

**What is not here, and why that is the whole split.** `ph.session.json` keeps
`freeze_json_value` — invariant A1's gate, the thing that decides what may enter
a log — and its module docstring makes that argument. A gate belongs beside the
thing it guards; a vocabulary belongs where anyone can say it.

**The cost this was separated from.** Reaching these names through
`ph.session.json` cost **about 120 ms and 347 modules**, because importing a submodule
runs its parent package and `ph.session` reaches `ph.cordis` — pydantic, anyio,
asyncio. From here it is **about 5 ms and 68**. Of that, `typing` is about 4 ms
(`TypeAlias` and `@overload` both pin it) and stdlib `json` about 1 ms; `json`
could leave with `JsonEncoder`/`dumps`, but a fourth module to save 1 ms on a
leaf 95 files import is not a trade worth making.

**Two callers get the whole win**, and naming them matters because the static
census misleads: twenty-odd modules that narrow JSON never mention a session,
but almost all are reached *through* a package whose own `__init__` loads the
session closure anyway, so they save nothing.

| | before | after |
|---|---|---|
| `ph_app.daemon.framing` | 127 ms / 351 | **43 ms / 205** |
| `ph_app.protocol` | 130 ms / 348 | **71 ms / 225** |

Framing is the larger of the two and only the second half of the split reached
it — it imported `ph.session` for `dumps` alone. Everyone else pays one extra
module, 133 µs, and the module is cheap enough that that is the right trade.

@module ph.json
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, TypeAlias, overload

__all__ = [
    "JSON_MAX_SAFE_INTEGER",
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
    "thaw_json",
]


# ------------------------------------------------------------ vocabulary --

JsonValue: TypeAlias = (
    "bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue] | None"
)
"""A lossless-JSON tree, in either of the two shapes a log is held in.

**Recursive, and over the abstract containers on purpose.** `list[Any] |
dict[str, Any]` — what this was — said nothing past the first level and could
not describe the frozen form at all, since a `MappingProxyType` is not a `dict`
and a `tuple` is not a `list`; a reader typed against it was untyped one index
down. `Sequence`/`Mapping` are true of both shapes — `tuple`/`MappingProxyType`
in memory, `list`/`dict` on disk and after `thaw_json` — and they are covariant,
so a producer holding `list[dict[str, Any]]` may hand it to `Session.append`
where `list[JsonValue]`, being invariant, would have refused it.

What the type does *not* say is "frozen". Nothing in the type system
distinguishes the two shapes (`ph.session.json`'s P6-44 paragraph); a
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

_EMPTY_OBJECT: JsonObject = MappingProxyType({})
"""`as_obj`'s answer to a missing field. Hoisted because it is 8% of that call, and
read-only because a shared mutable empty is an aliasing hazard the moment a
caller writes to a narrowed result."""


# ----------------------------------------------------------- narrowings --


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
    """A JSON string, or `default` — the fourth of the family, and the copied one.

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
    is not a weakening either, because the D4 paragraph in `ph.session.json`
    says those are the only two object shapes this tree is built in or read from.
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


JSON_MAX_SAFE_INTEGER = 2**53 - 1
"""`Number.MAX_SAFE_INTEGER`. Beyond it a JavaScript reader loses precision."""


# ---------------------------------------------------------------- thaw --


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


# ------------------------------------------------------------ to text --


class JsonEncoder(json.JSONEncoder):
    """Serializes the frozen views `ph.session.json` produces, without thawing.

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
