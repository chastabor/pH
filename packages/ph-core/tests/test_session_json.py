"""P8-06 — the readers' side of a typed payload.

`SessionEvent.data` is a `JsonObject`: a recursive `Mapping[str, JsonValue]`
over the abstract containers, true of both shapes the log takes — `tuple` and
`MappingProxyType` in memory, `list` and `dict` on disk. That type is honest,
which is the point, and honesty has a cost at the read site: `int(data.get(
"turn", 0))` is now an `int()` of a union a `Mapping` belongs to, and a chained
`data.get("message").get("content")` is a `.get` on something that may be a
string. `as_int`, `as_obj` and `as_seq` are the three narrowings the ~167 readers
needed, and this file pins what each promises.

The overloads on `freeze_json_value` and `thaw_json` are claims to the checker
— an object in is an object out — that `mypy` verifies against every production
caller and that tests, being outside `mypy`'s reach, cannot. What they *can*
pin is the runtime shape those overloads describe, so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pytest

from ph.session.json import as_int, as_obj, as_seq, freeze_json_value, thaw_json

# ---------------------------------------------------------------- as_int --


@pytest.mark.parametrize(
    ("value", "expected"),
    [(7, 7), (7.9, 7), ("42", 42), (True, 1), (False, 0)],
)
def test_as_int_keeps_every_coercion_int_made(value: object, expected: int) -> None:
    """Twenty readers wrote `int(...)` of a payload field. The replacement must
    accept exactly what those sites accepted — a float truncates, a numeric
    string parses, a bool is its integer — or a log that read yesterday would
    fail today."""
    assert as_int(value) == expected


def test_as_int_still_raises_value_error_on_a_non_numeric_string() -> None:
    with pytest.raises(ValueError):
        as_int("many")


@pytest.mark.parametrize(
    "container",
    [None, [1], (1,), {"n": 1}, MappingProxyType({"n": 1})],
    ids=["none", "list", "tuple", "dict", "mappingproxy"],
)
def test_as_int_refuses_a_container_naming_its_shape(container: object) -> None:
    """The one behavioural change from bare `int()`: a container fails with the
    field's actual type in the message rather than "int() argument must be a
    string, a bytes-like object or a real number" from three frames down."""
    with pytest.raises(TypeError, match=type(container).__name__):
        as_int(container)


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
