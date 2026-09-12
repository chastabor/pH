"""P8-06 — the readers' side of a typed payload: `ph.json`.

`SessionEvent.data` is a `JsonObject`: a recursive `Mapping[str, JsonValue]`
over the abstract containers, true of both shapes the log takes — `tuple` and
`MappingProxyType` in memory, `list` and `dict` on disk. That type is honest,
which is the point, and honesty has a cost at the read site: `int(data.get(
"turn", 0))` is now an `int()` of a union a `Mapping` belongs to, and a chained
`data.get("message").get("content")` is a `.get` on something that may be a
string. `as_int`, `as_obj`, `as_seq`, `as_str` and `as_bool` are the five narrowings the
~167 readers needed, and this file pins what each promises.

The overloads on `freeze_json_value` and `thaw_json` are claims to the checker
— an object in is an object out — that `mypy` verifies against every production
caller and that tests, being outside `mypy`'s reach, cannot. What they *can*
pin is the runtime shape those overloads describe, so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from ph.json import as_bool, as_int, as_obj, as_seq, as_str, thaw_json
from ph.session.json import freeze_json_value

# ---------------------------------------------------------------- as_int --


@pytest.mark.parametrize(
    ("value", "expected"),
    [(7, 7), (7.9, 7), ("42", 42), (True, 1), (False, 0), (0, 0)],
)
def test_as_int_keeps_every_coercion_int_made(value: object, expected: int) -> None:
    """Twenty readers wrote `int(...)` of a payload field. The replacement must
    accept exactly what those sites accepted — a float truncates, a numeric
    string parses, a bool is its integer — or a log that read yesterday would
    fail today."""
    assert as_int(value) == expected
    assert as_int(value, -1) == expected, "a real value is never the default"


@pytest.mark.parametrize(
    "junk",
    [
        None,
        "many",
        [1],
        (1,),
        {"n": 1},
        MappingProxyType({"n": 1}),
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=[
        "none",
        "non-numeric",
        "list",
        "tuple",
        "dict",
        "mappingproxy",
        "nan",
        "inf",
        "-inf",
    ],
)
def test_a_field_that_is_not_a_number_reads_as_the_default(junk: object) -> None:
    """One family, one policy — and this is the half that changed (49b).

    It used to raise: `TypeError` for a container, `ValueError` for a
    non-numeric string. `as_obj` and `as_seq` answer a mis-shaped field with the
    empty container instead, because "a missing one must cost a row rather than
    the transcript" — and on `persistence.repair`, `driver._last_turn_of` and
    `replay.recorded_steps` that raise was **not** contained: one mistyped
    numeric field in a log another build wrote turned an unreadable row into a
    failed resume.

    `nan` and the two infinities are here because they are the shapes that
    survived the first cut: `int()` raises `ValueError` for one and
    `OverflowError` for the other, and only `ValueError` was caught, so a log
    spelling `NaN` bare — which `json.loads` parses — still failed a resume
    while the helper claimed one policy for the family.

    `freeze_json_value` does **not** keep our own logs out of here. It is a
    JSON-ness gate, not a schema gate: `Session.append({"turn": "3"})`
    succeeds, so a producer of ours writing the wrong type into a numeric
    field lands on the default too.
    """
    assert as_int(junk) == 0
    assert as_int(junk, -1) == -1, "and the caller's own default is what it answers with"


# ---------------------------------------------------------------- as_bool --


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (None, False),
        ("false", False),
        ("true", False),
        (1, False),
        (0, False),
        ([], False),
        ([True], False),
    ],
    ids=[
        "true",
        "false",
        "missing",
        "the-string-false",
        "the-string-true",
        "one",
        "zero",
        "empty-list",
        "truthy-list",
    ],
)
def test_as_bool_reads_a_boolean_and_never_guesses_at_one(value: object, expected: bool) -> None:
    """**`bool("false")` is `True`** — the reason this exists.

    A flag spelled as a string reads as the *opposite* of what the log says, and
    `bool()` never raises or looks wrong while doing it. Every row here that
    expects `False` against a truthy value is a row `bool()` would have got
    backwards or invented: `"false"`, `"true"`, `1` and `[True]` are not
    booleans, and which one the producer meant is not this helper's to guess.

    The `1` row is also where the family's rule divides, and it divides in both
    directions: `as_int(True)` is `1` (a bool *is* a number), while `as_bool(1)`
    is the default (a number is *not* a bool). `as_int` keeps what `int()`
    coerced because *raising* was its defect; `bool()` has no raise, and its
    coercion is the defect.
    """
    assert as_bool(value) is expected


def test_as_bool_takes_a_default_for_a_flag_that_is_on_unless_said_otherwise() -> None:
    """`show_thinking` and its three siblings in `tui.json`, which default on.

    The default is what a *missing* field reads as — and, for this family, what a
    mis-typed one reads as too. A hand-edited `"show_tools": "no"` therefore
    leaves the panel shown rather than silently flipping it, which is the
    conservative direction for a setting a person did not successfully express.
    """
    assert as_bool(None, True) is True
    assert as_bool("no", True) is True
    assert as_bool(False, True) is False


# ---------------------------------------------------------------- as_str --


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("text", "text"),
        ("", ""),
        (None, ""),
        (3, ""),
        (True, ""),
        (["text"], ""),
        ({"text": "text"}, ""),
    ],
    ids=["a-string", "empty", "missing", "a-number", "a-bool", "a-list", "an-object"],
)
def test_as_str_answers_the_empty_string_for_anything_that_is_not_one(
    value: object, expected: str
) -> None:
    """One policy with its siblings: a mis-shaped field costs a row, not a raise.

    **The point is what it does *not* do.** `str(value)` — which is what a reader
    writes without this — turns `None` into `"None"` and `3` into `"3"`, so a
    field that is absent or of the wrong type comes back looking like an answer.
    Every row below that expects `""` is a row `str()` would have passed.
    """
    assert as_str(value) == expected


def test_as_str_returns_the_string_it_was_given_by_identity() -> None:
    """Not a copy, for `as_obj`'s reason: a narrowing is a claim about a value,
    not a new value."""
    value = "the same object"
    assert as_str(value) is value


def test_as_str_takes_a_default_for_a_reader_that_has_a_better_empty() -> None:
    """`as_int`'s arrangement: the family answers with nothing, and a caller that
    knows a better nothing says so. A card with no name reads `?`, not blank."""
    assert as_str(None, "?") == "?"
    assert as_str("named", "?") == "named"


# ------------------------------------------------------------- obj / seq --


def test_obj_returns_a_mapping_by_identity_in_either_shape() -> None:
    """Not a copy: a reader narrowing a frozen payload must not thaw it by
    accident, and a plain dict read off disk must stay the dict it was."""
    frozen = MappingProxyType({"a": 1})
    plain = {"a": 1}
    assert as_obj(frozen) is frozen
    assert as_obj(plain) is plain


@pytest.mark.parametrize("value", [None, "text", 3, 2.5, True, [1, 2], (1, 2)])
def test_obj_answers_absence_with_an_empty_object(value: object) -> None:
    """A missing or mis-shaped field costs the reader a row, not the transcript;
    the empty object lets `as_obj(x).get(...)` chain without a branch."""
    assert as_obj(value) == {}
    assert isinstance(as_obj(value), Mapping)


def test_seq_returns_an_array_by_identity_in_either_shape() -> None:
    """A tuple in memory, a list on disk — a reader that tested `isinstance(x,
    list)` worked on resume and silently saw nothing live. Both pass."""
    frozen = (1, 2)
    plain = [1, 2]
    assert as_seq(frozen) is frozen
    assert as_seq(plain) is plain


@pytest.mark.parametrize("value", [None, "abc", 3, {"a": 1}, MappingProxyType({"a": 1})])
def test_seq_answers_absence_and_strings_with_an_empty_array(value: object) -> None:
    """`str` is a `Sequence[str]` and a `JsonValue`, so a reader narrowing with
    `isinstance(x, Sequence)` meets it and iterates a word's letters as rows.
    `as_seq` is the one place that exclusion is spelled."""
    assert as_seq(value) == ()
    assert isinstance(as_seq(value), Sequence)


# -------------------------------------------------------- shape preservation --

_TREE = {"message": {"content": [{"type": "text", "text": "hi"}, 1, None]}, "usage": {"in": 3}}


def test_freeze_preserves_shape_at_every_level() -> None:
    """The overload says an object in is an object out. At runtime the walker
    rebuilds every container in the frozen form, and does so all the way down —
    a `MappingProxyType` wrapping a live `dict` would be the shape that lets a
    caller reach back into a log."""
    frozen = freeze_json_value(_TREE)
    assert isinstance(frozen, MappingProxyType)
    message = frozen["message"]
    assert isinstance(message, MappingProxyType)
    content = message["content"]
    assert isinstance(content, tuple)
    assert isinstance(content[0], MappingProxyType)
    assert isinstance(frozen["usage"], MappingProxyType)


def test_thaw_preserves_shape_at_every_level_and_is_mutable() -> None:
    """`thaw_json` exists so a caller can *assign* — `compaction-summarize`
    rewrites one block of a thawed `assistant/message` and re-appends it. The
    overloads promise `dict` for an object and `list` for an array; the plain
    shape must hold at every depth or the first nested assignment fails."""
    thawed = thaw_json(freeze_json_value(_TREE))
    assert type(thawed) is dict
    message = thawed["message"]
    assert type(message) is dict
    content = message["content"]
    assert type(content) is list
    assert type(content[0]) is dict
    content[0] = {**content[0], "text": "rewritten"}
    thawed.pop("usage")
    assert thawed == {"message": {"content": [{"type": "text", "text": "rewritten"}, 1, None]}}


def test_thaw_of_an_array_is_a_list_and_of_a_scalar_is_itself() -> None:
    assert thaw_json((1, (2, 3))) == [1, [2, 3]]
    assert type(thaw_json((1, 2))) is list
    assert thaw_json("text") == "text"
    assert thaw_json(None) is None


def test_a_thawed_tree_re_enters_the_log_unchanged() -> None:
    """A `PlainJsonValue` is a `JsonValue` — list is a Sequence, dict a Mapping —
    so the tree `compaction-summarize` rewrites flows back into `Session.append`
    without a cast, and freezing it again is the identity on content."""
    frozen = freeze_json_value(_TREE)
    assert freeze_json_value(thaw_json(frozen)) == frozen
    assert thaw_json(frozen) == _TREE
