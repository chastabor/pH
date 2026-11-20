"""JSON as a *representation*: what a value may be, how to narrow it, and how
to carry it.

This module has four groups, and they are one subject:

* The vocabulary: `JsonValue`, `JsonObject` and `PlainJsonValue`.
* The five narrowings that a reader applies to an untrusted field.
* The conversion between the two in-memory shapes: `thaw_json`.
* The conversion to and from text: `JsonEncoder`, `dumps`, `loads` and
  `first_json_value`.

It also holds `JSON_MAX_SAFE_INTEGER`, the bound that both the gate and the RLM
codec check against. None of these names knows what a session is.

**What is not here.** `ph.session.json` keeps `freeze_json_value`. That function
is the gate of invariant A1: it decides what may enter a log. The module
docstring of `ph.session.json` gives the argument. The split follows one rule. A
gate belongs beside the thing that it guards, and a vocabulary belongs where any
module can use it.

**The cost that the split removed.** Reaching these names through
`ph.session.json` cost **about 120 ms and 347 modules**. Importing a submodule
runs its parent package, and `ph.session` reaches `ph.cordis`, which loads
pydantic, anyio and asyncio. From this module the cost is **about 5 ms and 68
modules**. Of that, `typing` is about 4 ms, and both `TypeAlias` and `@overload`
need it. The stdlib `json` is about 1 ms. `json` could move out together with
`JsonEncoder` and `dumps`. But a fourth module that saves 1 ms on a leaf that 95
files import is not worth it.

**Two callers get the full benefit.** The table names them, because a static
count of the importers is misleading. A little over twenty modules that narrow
JSON never mention a session. But almost all of them are reached *through* a
package whose own `__init__` loads the session closure anyway, so they save
nothing.

| | before | after |
|---|---|---|
| `ph_app.daemon.framing` | 127 ms / 351 | **43 ms / 205** |
| `ph_app.protocol` | 130 ms / 348 | **71 ms / 225** |

Framing gets the larger benefit of the two. Only the second half of the split
reached it, because it imported `ph.session` for `dumps` alone. Every other
caller pays one extra module, 133 µs. The module is cheap enough that this is
the right trade.

@module ph.json
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final, TypeAlias, overload

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
    "first_json_value",
    "loads",
    "thaw_json",
]


# ------------------------------------------------------------ vocabulary --

JsonValue: TypeAlias = (
    "bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue] | None"
)
"""A lossless-JSON tree, in either of the two shapes that a log is held in.

**The type is recursive, and it is over the abstract containers on purpose.**
The earlier type was `list[Any] | dict[str, Any]`. It said nothing below the
first level, so a reader typed against it was untyped one index down. It also
could not describe the frozen form at all, because a `MappingProxyType` is not a
`dict` and a `tuple` is not a `list`. `Sequence` and `Mapping` are true of both
shapes: `tuple` and `MappingProxyType` in memory, and `list` and `dict` on disk
and after `thaw_json`. They are also covariant. A producer that holds
`list[dict[str, Any]]` may therefore pass it to `Session._append`. A parameter
typed `list[JsonValue]` is invariant and would have refused it.

The type does *not* say "frozen". Nothing in the type system distinguishes the
two shapes (see the P6-44 paragraph of `ph.session.json`). A wrapper that
carries its own proof is the deferred decision. This alias is written so that
the wrapper can subtype it later, and no reader has to move.

The flaw is `str`. A `str` is a `Sequence[str]`, and `str` is a `JsonValue`, so
a reader that narrows with `isinstance(x, Sequence)` meets it. `as_seq` below is
the one place that writes the exclusion.

**This alias is not usable as a pydantic field annotation, and the alternative
is worse.** The schema generator of pydantic expands this alias eagerly, and not
as a recursive reference. A `WireModel` field typed `JsonValue` or `JsonObject`
therefore never finishes building its validator. It recurses until the stack
ends, at class definition. `pydantic.JsonValue` does build, because pydantic
constructs it as a self-reference. But it then *validates* recursively. A
200-block `assistant/message` notice measured **77.4 µs against 0.7 µs** for
`dict[str, Any]`, on the frame that the daemon sends one time per appended event
per watcher. `ph_app.payloads._CarriesJson` exists because of a measurement of
that cost, and it exists to avoid the cost. So the fields that it names stay
`dict[str, Any]`. That is the `Any` that this type exists to retire. It is kept
deliberately at the one boundary where the retirement costs too much. Everywhere
that the tree is *not* a pydantic field, this alias is the type to use."""

JsonObject: TypeAlias = "Mapping[str, JsonValue]"
"""A JSON object: every event payload, and the shape that `Session._append` takes."""

PlainJsonValue: TypeAlias = (
    "bool | int | float | str | list[PlainJsonValue] | dict[str, PlainJsonValue] | None"
)
"""The plain shape only: `list` and `dict` at every level, as `thaw_json` builds
it and as a log reads back from disk.

This is a second alias, because the first cannot say this. `JsonValue` is over
the abstract containers so that it is true of the frozen tree too, and abstract
containers do not accept an assignment into them. `compaction-summarize` thaws a
payload in order to rewrite one block of it. A return typed `Sequence` would
refuse the assignment that the thaw exists to permit. A `PlainJsonValue` is a
`JsonValue`, because a list is a Sequence and a dict is a Mapping. A thawed tree
therefore still passes into `Session._append` without a cast. The reverse is not
true, and that is the purpose of the second alias."""

_EMPTY_OBJECT: JsonObject = MappingProxyType({})
"""The answer of `as_obj` to a missing field. It is built one time at module
level, because building it is 8% of that call. It is read-only, because a shared
mutable empty object is an aliasing hazard as soon as a caller writes to a
narrowed result."""


# ----------------------------------------------------------- narrowings --


def as_int(value: object, default: int = 0) -> int:
    """A JSON number as an `int`, or `default`. This is the narrowing that a
    reader applies to a numeric field.

    Twenty readers wrote `int(event.data.get("turn", 0))`. With `data` typed,
    that is an `int()` of a union that a `Mapping` belongs to, and the type
    checker refuses it. This function is the third of the family, and it now
    answers in the same way as the other two.

    **It used to raise, and it was the only narrowing that did.** `as_obj` and
    `as_seq` answer a mis-shaped field with the empty container: "a missing
    field must cost one row and not the transcript". This function raised a
    `TypeError` on anything that was not a number. Nothing contains that raise
    in `persistence.repair`, `agent_loop.driver._last_turn_of` and
    `llm.replay.recorded_steps`. One mistyped numeric field, in a log that some
    other build wrote, turned a single unreadable row into a **failed resume**.
    The family shares one policy: a mis-shaped field costs a row and does not
    raise.

    **This function does *not* share the strictness of the others.** `as_str`
    and `as_bool` narrow: a value that is not of the type gives nothing but the
    default. This function keeps every coercion that `int()` made. The
    difference comes from what each builtin got wrong. `str()` fabricates and
    `bool()` inverts, so their coercions had to go. The coercions of `int()`
    are correct, and only its *raising* was the defect. The visible result is
    that `as_int(True)` is `1` while `as_bool(1)` is the default. This is
    deliberate, and tests check both sides. It is also the reason that `as_bool`
    is not this function with a different type.

    **The purpose of the default.** `freeze_json_value` is a gate for
    JSON-*ness*, not a schema gate. It walks a payload for values that JSON can
    round-trip, and a `str`, a `None` or a list is such a value.
    `Session.append({"turn": "3"})` succeeds. So the values that reach the
    default do not come only from a foreign build or a hand-edited log. They
    also come from one of our own producers that writes the wrong type into a
    numeric field, and from a field that is absent. That is why `None` takes the
    default too, and why the call sites pass `as_int(data.get("turn"))` and do
    not give a default twice.

    A *quiet* default is right for a cosmetic number. It is a real cost for a
    number that the program depends on. A junk `turn` that reaches
    `driver._last_turn_of` gives a resumed run a duplicate turn number. That is
    worse than the failed resume that it replaced, because nothing reports it.
    The refusal belongs at the row, as a typed payload for `turn/start` and
    `step/start`. That is the work of issue 74 and not of this helper.

    The coercions that `int()` made are unchanged: a `float` truncates and a
    numeric `str` parses. The change is that a container, `None`, a non-numeric
    string and a non-finite float now read as `default` and do not raise.
    """
    if type(value) is int:
        # This is the purpose of the helper, and nearly every call: a JSON
        # integer that is already an `int`. It is returned without the
        # `isinstance` walk below (22 ns against 110). `type(...) is int`
        # excludes `bool` deliberately. `bool` is a subclass of `int`, and it
        # belongs on the coercing branch below.
        return value
    if isinstance(value, (bool, float, str)):
        # These are exactly the types that `int()` accepted. `NaN` and
        # `±Infinity` are the reason that this catches two exceptions and not
        # one. `json.loads` parses both from a log that writes them bare, and
        # they used to raise through this function: `ValueError` and
        # `OverflowError` respectively. That left open the gap that this row
        # set out to close, while the docstring claimed one policy for the
        # family. A `try` that never fires has no measurable cost.
        try:
            return int(value)
        except (ValueError, OverflowError):
            return default
    return default


def as_bool(value: object, default: bool = False) -> bool:
    """A JSON boolean, or `default`. This is deliberately **not** `bool()`.

    This is the fifth of the family, and its builtin is the most dangerous one.
    `bool()` never raises and never looks wrong. It answers *truthiness*, which
    is defined for every JSON value and is the wrong question for a boolean
    field. **`bool("false")` is `True`.** A log can write a flag as a string. The
    log can come from a foreign build, a hand-edited file, or one of our own
    producers that writes the wrong type. `freeze_json_value` admits all of
    these, because it is a gate for JSON-*ness* and not a schema gate. Such a
    flag reads as the opposite of what it says. On `isError`, that draws an
    error card for a tool call that succeeded.

    **The rule of the family divides here, and the division is deliberate.**
    `as_int` keeps every coercion that `int()` made: a float truncates, and a
    numeric string parses. It replaces only the *raise*, because raising was
    the defect of `int()`. `bool()` has no raise to replace. Its coercion **is**
    the defect. So this function narrows: a `bool` is a `bool`, and everything
    else is `default`.

    `1` therefore reads as `default`, not `True`. That is deliberate, and it is
    the same judgment. JSON has a boolean type. A producer that wrote `1` into a
    boolean field did not write a boolean. To guess which value the producer
    meant is the fabrication that this family exists to refuse.

    **`bool()` is still right for a *presence* check**, and callers keep it for
    that reason. `bool(self._state["next-turn"] or self._state["next-step"])`
    asks "is anything queued" of a Python list. That does not read a JSON
    boolean, and truthiness is exactly the question.
    """
    return value if isinstance(value, bool) else default


def as_str(value: object, default: str = "") -> str:
    """A JSON string, or `default`. This is the fourth of the family, and the
    one that `ph_runtime` copies.

    It follows one policy, the policy that the docstring of `as_int` argues for.
    A mis-shaped field answers with the empty value and does not raise, because
    a reader of a log that some other build wrote must lose a row and not a
    session.

    **This is not `str(value)`.** A reader without this function writes
    `str(value)`, and that is worse than useless. `str(None)` is `"None"` and
    `str(3)` is `"3"`. A field that is absent or of the wrong type then comes
    back as an answer that looks plausible, and no assertion catches it. A
    narrowing reports "this was not a string" by returning nothing. `as_obj` and
    `as_seq` report the same thing in the same way.
    """
    return value if isinstance(value, str) else default


def as_obj(value: object) -> JsonObject:
    """A JSON object, or an empty one. This is the narrowing that a reader
    applies to a payload field.

    Absence is normal. The log is JSON, and every field is optional to a reader.
    A missing field must cost one row and not the transcript. The function is
    here and not in `ph_app.wire`, where it started, because the question that
    it answers is about the tree and not about the app. A core reader that
    chains `data.get("message").get(...)` needs exactly this narrowing, and had
    no typed way to say so.

    **The test names the two shapes and does not ask the ABC.** `isinstance`
    against `Mapping` goes through `ABCMeta.__instancecheck__`. A
    `MappingProxyType` is the in-memory form, so it is the common case, and it
    is the worst input for that check: 220 ns against 56 ns for the concrete
    pair. This function runs for each field of each event on the crash-repair
    scan and the TUI fold. Measured over 8,499 events, the ABC form cost 3.2x
    the concrete one. Naming `dict` and `MappingProxyType` does not weaken the
    test either. The D4 paragraph in `ph.session.json` says that those are the
    only two object shapes that this tree is built in or read from.
    """
    return value if isinstance(value, (dict, MappingProxyType)) else _EMPTY_OBJECT


def as_seq(value: object) -> Sequence[JsonValue]:
    """A JSON array, or an empty one. It is a tuple in memory and a list on disk.

    The two shapes, frozen and plain, are the reason that this is a function
    and not an `isinstance(value, list)`. A reader that tested for `list` worked
    on resume and silently saw nothing live. Naming both shapes also makes the
    exclusion of `str` **structural**, and not a special case. `str` is a
    `Sequence`, so an ABC test admits it, and a reader that forgot this iterated
    the letters of a word as rows. `str` is neither a `list` nor a `tuple`, so
    this test excludes it with no extra rule. The measurement is the same as for
    `as_obj`: 244 ns against 56 ns, for each field of each event.
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
    """A plain mutable copy of a frozen tree, for callers that need `dict` and
    `list`.

    The function is overloaded on the shape of the input, as `freeze_json_value`
    is. An object therefore thaws to a `dict` *in the type*. A caller that thaws
    in order to **mutate** needs that: `compaction-summarize` rewrites one block
    of a thawed `assistant/message` and appends it again.

    A `tuple` selects the array overload. A `tuple` is the frozen array shape,
    and it is covariant. A `list[str]`, or any other concretely parameterized
    list, does **not** select it, because `list` is invariant and `list[str]` is
    not a `list[object]`. Such a list falls through to the third overload and
    gets the union. No production caller passes a statically typed list, so this
    is a documented limit and not a gap to close.
    """
    if isinstance(value, Mapping):
        # There is no `str(key)`. `freeze_json_value` refuses a non-string key
        # at the gate, so a tree from this module cannot have one. The coercion
        # measured at 11% of a 5,000-event thaw, and `to_wire` pays that for
        # each event, for each attached front end.
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value  # type: ignore[return-value]  # a scalar this module admitted


# ------------------------------------------------------------ to text --


class JsonEncoder(json.JSONEncoder):
    """Serializes the frozen views that `ph.session.json` produces, without
    thawing.

    Tuples encode natively. `default` handles `MappingProxyType`.
    `allow_nan=False` is deliberate. A non-finite number should already have
    been refused at append. A second refusal here is cheaper than emitting a
    token that no other parser reads.
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        kwargs.setdefault("allow_nan", False)
        kwargs.setdefault("separators", (",", ":"))
        kwargs.setdefault("ensure_ascii", False)
        super().__init__(**kwargs)

    def default(self, o: object) -> object:
        if isinstance(o, MappingProxyType):
            return dict(o)
        return super().default(o)


_ENCODER = JsonEncoder()


def dumps(value: object) -> str:
    """Canonical compact JSON for one log line or wire frame."""
    return _ENCODER.encode(value)


# ---------------------------------------------------------- from text --


_DECODER: Final = json.JSONDecoder()
"""Built one time, and **without hooks**.

*Built one time*: `json.loads` dispatches again on every call, through a wrapper
frame and the test for str or bytes. That measured **0.406 µs against 0.334 µs**
for this object on an 84-byte RLM `log` frame. That frame is the most numerous
one on that stdout path, where decoding runs on every frame that the guest
sends. *Without hooks*: a `parse_int` guard here used to reject whole frames
over a bound that only some fields must keep. The bound is therefore checked
where a field is read (`ph_rlm.kernel.codec._coerce`).
"""


def first_json_value(text: str, start: int = 0) -> str | None:
    """The first complete JSON value in `text` that begins at `start`, as a
    slice.

    `raw_decode` is the stdlib function that parses one value and reports where
    it ended. A reader that tolerates prose around a document therefore does not
    need a brace scanner of its own. A hand-written scanner must get string
    literals and escapes right, and that part looks finished long before it is.

    The function returns a *slice* and not the decoded value. The caller that
    wants this is choosing between candidate spans, and it passes the chosen
    span to its own validation. Returning the value would make this function
    the parser, and the caller would have to encode the value again to report
    what it read.
    """
    try:
        _, ended = _DECODER.raw_decode(text, start)
    except ValueError:
        return None
    return text[start:ended]


def loads(raw: str | bytes) -> JsonValue:
    """One JSON document as the tree that `JsonValue` names. It is the inverse
    of `dumps`.

    The function is here and not at each reader because `json.loads` is typed
    `Any`, and every caller of it must undo that. Three sites had each written
    their own argument for why the result is a JSON tree, and every later site
    would have written a fourth.

    Bytes are decoded to `str` and not passed to the decoder. That measured
    0.49 µs against 0.65 µs on an 84-byte frame, because the bytes path detects
    the encoding before it does exactly this.
    """
    value: JsonValue = _DECODER.decode(raw if isinstance(raw, str) else raw.decode("utf-8"))
    return value
