"""`ph.text` — the prose helpers that must not be retyped wrongly the fifth time.

The module exists for formatting that appears in more than one package and has a
wrong answer. `brief_value` is the clearest case of both halves: two readers in
two packages rendered payload values with `str()`, which is right for a scalar
and a Python literal for anything else, and neither had a test that could see it.
"""

from __future__ import annotations

from ph.json import JsonObject
from ph.session.json import freeze_json_value
from ph.text import brief_value, count_of


def test_a_container_is_named_rather_than_dumped() -> None:
    """`str()` on a nested value is the bug this replaced.

    A list is counted because a list is never one-line material; a mapping is
    expanded one level, because that is where the readable facts usually are.
    """
    assert brief_value([1, 2, 3]) == "[3 items]"
    assert brief_value([1]) == "[1 item]", "and it agrees with itself on one"
    assert brief_value({"turns": 9}) == "{turns=9}"
    assert brief_value({"b": {"c": 1, "d": 2}}) == "{b={2 fields}}", "counted below one level"
    assert brief_value("plain") == "plain"
    assert brief_value(7) == "7"


def test_the_frozen_shapes_render_the_same_as_the_plain_ones() -> None:
    """**The half the first version got wrong, and the reason this file exists.**

    The log freezes payloads — `MappingProxyType` over `tuple` — so a renderer
    that tested `list` and `Mapping` worked on decoded wire frames and did
    nothing at all on the live path, where it fell through to `str()` and emitted
    `(mappingproxy({...}),)`. That is worse than the bug it was fixing, and it
    shipped because every test payload was a plain literal.

    Asserted as *equality between the two shapes* rather than against expected
    strings: the property is that a reader cannot tell which form it was handed,
    and a second table of expected output would be one more thing to keep in step.
    """
    payloads: list[JsonObject] = [
        {"todos": [{"content": "fix it", "status": "pending"}], "n": 2},
        {"limit": "turns", "spent": {"turns": 9}, "cap": 8},
        {"a": {"b": {"c": 1, "d": 2}}},
        {"empty": [], "nothing": {}},
    ]
    for payload in payloads:
        frozen = freeze_json_value(payload)
        assert brief_value(frozen) == brief_value(payload), payload
        assert "mappingproxy" not in brief_value(frozen)


def test_no_python_literal_ever_reaches_a_reader() -> None:
    """The falsifiable form of the whole point, over both shapes.

    Sabotage: drop either concrete-type test in `brief_value` and the markers
    below appear for the shape that stopped matching.
    """
    payload: JsonObject = {
        "inserted": [{"id": "m1", "content": [{"type": "text"}]}],
        "spent": {"turns": 9},
    }
    for value in (payload, freeze_json_value(payload)):
        line = brief_value(value)
        assert "{'" not in line and "[{" not in line and "': " not in line, line
        assert "mappingproxy" not in line and "(" not in line, line


def test_count_of_agrees_with_itself() -> None:
    assert count_of(1, "item") == "1 item"
    assert count_of(2, "item") == "2 items"
    assert count_of(2, "entry", "entries") == "2 entries"
