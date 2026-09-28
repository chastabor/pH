"""How two rows' limits on a child combine (`ChildLimit.join`, session profiles S7b).

Each row an assigned profile keeps says what it holds its child to; the child is held
to all of them at once, so each field joins as the tighter of the two.
"""

from __future__ import annotations

import pytest

from ph.cordis import ChildLimit, NarrowingRefused


def test_two_limits_hold_a_child_to_the_tighter_of_each() -> None:
    """Sabotage: keep one side's withheld skills or paths, and the child holds what
    the other row kept back."""
    one = ChildLimit(withheld_skills=frozenset({"audit"}), writable_paths=("/srv/a", "/srv/b"))
    other = ChildLimit(
        model_key="classify",
        read_only=True,
        withheld_skills=frozenset({"deploy"}),
        writable_paths=("/srv/b",),
    )

    joined = one.join(other)

    assert joined == ChildLimit(
        model_key="classify",
        read_only=True,
        withheld_skills=frozenset({"audit", "deploy"}),
        writable_paths=("/srv/b",),
    )
    assert one.join(ChildLimit()) == one, "a row that keeps nothing back changes nothing"


def test_two_rows_naming_different_models_are_refused() -> None:
    with pytest.raises(NarrowingRefused, match="two models for its child, main and classify"):
        ChildLimit(model_key="main").join(ChildLimit(model_key="classify"))
