"""P7-11 — a seam's wire form is derived from the seam, and covers it.

`tools_of` never spelled a field because `ToolSchema` is a `WireModel`. Three
other seams reached a front end as dicts written by hand at the daemon edge, and
that failure is the quiet kind: add a field to `CommandDefinition` and the
terminal shows it, the browser does not, and nothing fails. The projection tests
could not catch it either — they re-spelled the same keys on the expected side.

So each seam now describes itself (`to_wire()`, or a `.schema()` split from its
callable the way `ToolDefinition.schema()` is), and **this file is the assertion
those tests could not make**: the wire form's fields are exactly the definition's
fields minus the ones that cannot travel. A field left out of a schema fails here
rather than in front of a person.
"""

from __future__ import annotations

import dataclasses

import pytest

from ph.seams.commands import CommandDefinition, CommandSchema
from ph.seams.tui_screens import ScreenDefinition, ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.wire import declarable_fields, wire_alias


@pytest.mark.parametrize(
    ("definition", "schema"),
    [(CommandDefinition, CommandSchema), (ScreenDefinition, ScreenSchema)],
    ids=["command", "screen"],
)
def test_a_schema_carries_every_field_of_its_definition_but_the_body(
    definition: type, schema: type
) -> None:
    """Coverage by construction: the sets are compared, not a sample of them.

    Sabotage: add `danger: bool = False` to `CommandDefinition` and not to
    `CommandSchema` — this fails naming the field, where before the browser's
    palette simply never showed it.
    """
    assert set(schema.model_fields) == set(declarable_fields(definition)), (
        f"{definition.__name__} and {schema.__name__} disagree about what travels"
    )


def test_what_cannot_travel_is_derived_not_listed() -> None:
    """The callable is found by its annotation, so a second one cannot slip past.

    The first version of this gate kept a table naming `run` and `build` by hand
    — which is the hand-kept-list failure P7-11 exists to end, one layer down.
    """
    assert declarable_fields(CommandDefinition) == ("name", "summary", "argument_hint")
    assert declarable_fields(ScreenDefinition) == ("id", "label", "order", "key")


def test_a_reading_is_its_own_wire_form() -> None:
    """Pure data, so the mixin is the whole change — and `level` survives.

    The projection that this replaced was one line, and the one-line sabotage
    that motivated the row was dropping `level` from it: every reading rendered
    `normal`, and a warning stopped looking like one.
    """
    reading = StatusReading(text="context 86%", level="warn")

    assert reading.to_wire() == {"text": "context 86%", "level": "warn"}
    assert set(reading.to_wire()) == {
        wire_alias(field.name) for field in dataclasses.fields(StatusReading)
    }


def test_the_schema_is_what_the_definition_says_it_is() -> None:
    """`schema()` copies values, not just field names, and aliases as the wire does."""
    command = CommandDefinition(
        name="compact", summary="Fold the transcript.", run=lambda a, c: None, argument_hint="<n>"
    )
    screen = ScreenDefinition(id="trajectory", label="Trajectory", build=lambda s: None, key="t")

    assert command.schema().to_wire() == {
        "name": "compact",
        "summary": "Fold the transcript.",
        "argumentHint": "<n>",
    }
    assert screen.schema().to_wire() == {
        "id": "trajectory",
        "label": "Trajectory",
        "order": 100,
        "key": "t",
    }
