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

import importlib
import pkgutil
from typing import Any, Final, get_args, get_origin

import pytest
from pydantic import BaseModel

from ph.seams.commands import CommandDefinition, CommandSchema
from ph.seams.tui_screens import ScreenDefinition, ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.wire import WireModel, declarable_fields, wire_alias


@pytest.mark.parametrize(
    ("definition", "schema"),
    [(CommandDefinition, CommandSchema), (ScreenDefinition, ScreenSchema)],
    ids=["command", "screen"],
)
def test_a_schema_carries_every_field_of_its_definition_but_the_body(
    definition: type, schema: type[BaseModel]
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
    """Pure data, so `WireModel` is the whole change — and `level` survives both ways.

    The projection that this replaced was one line, and the one-line sabotage
    that motivated the row was dropping `level` from it: every reading rendered
    `normal`, and a warning stopped looking like one. A `WireModel` rather than a
    dataclass so the *reader* is derived too: a remote front end rebuilds a
    reading with `model_validate`, and a field added here reaches it with no edit
    at either edge. It also validates, which the dataclass did not — this test
    used to construct a reading with `level="warn"`, a level the seam has never
    had, and nothing said so.
    """
    reading = StatusReading(id="context", text="context 86%", level="warning")

    assert reading.to_wire() == {
        "id": "context",
        "slot": "line",
        "text": "context 86%",
        "level": "warning",
    }
    assert StatusReading.model_validate(reading.to_wire()) == reading
    assert set(reading.to_wire()) == {wire_alias(name) for name in StatusReading.model_fields}


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


# ------------------------------------------------- what may ride the wire --

_JSON_SCALARS: Final = (str, int, float, bool, type(None))


def _json_safe(annotation: object) -> bool:
    """Whether a declared field type is something `model_dump` leaves JSON-shaped.

    `Any` passes: it is the declared escape for a field holding somebody else's
    wire JSON (`_CarriesJson` documents why those exist and are dumped by
    reference). This gate is for a *concrete* type that cannot serialize, which
    is the failure nothing else catches — an `Any` field was already unchecked
    and says so.
    """
    if annotation is Any or annotation in _JSON_SCALARS or annotation is None:
        return True
    origin = get_origin(annotation)
    if origin is not None:
        return all(_json_safe(arg) for arg in get_args(annotation) if arg is not Ellipsis)
    if isinstance(annotation, type):
        # A nested wire model dumps to a dict of its own checked fields.
        if issubclass(annotation, BaseModel):
            return True
        return annotation in _JSON_SCALARS
    # A `Literal[...]` member, a `TypeVar`, a forward ref that resolved to a value.
    return not isinstance(annotation, type)


def _wire_models() -> set[type[WireModel]]:
    """Every `WireModel` reachable from the shipped packages, read off each module.

    Off the modules rather than `WireModel.__subclasses__()`, for the reason
    `test_payloads._notice_classes` gives: the global registry makes the assertion
    depend on what else the run imported.
    """
    found: set[type[WireModel]] = set()
    for package in ("ph", "ph_app", "ph_rlm", "ph_stabilize", "ph_text_index", "ph_code_graph"):
        try:
            root = importlib.import_module(package)
        except ImportError:  # pragma: no cover - a package this deployment lacks
            continue
        for info in pkgutil.walk_packages(root.__path__, f"{package}."):
            try:
                module = importlib.import_module(info.name)
            except Exception:  # pragma: no cover - optional extras
                continue
            found |= {
                one
                for one in vars(module).values()
                if isinstance(one, type) and issubclass(one, WireModel) and one is not WireModel
            }
    return found


def test_no_wire_model_declares_a_field_model_dump_cannot_serialize() -> None:
    """`WireModel.to_wire()` claims `dict[str, JsonValue]`, and this is what makes it true.

    `model_dump()` without `mode="json"` hands back whatever the fields hold, so a
    `datetime`, `Path` or enum field would land in a `dict[str, JsonValue]` with
    nothing static firing — and then raise from `dumps` at the daemon framing edge,
    once per watcher per frame, naming the encoder rather than the field.
    `mode="json"` would make it true by construction but rebuilds the tree, which
    is the cost `_CarriesJson` exists to avoid.

    Sabotage: add `created_at: datetime` to any `WireModel` — this names it.
    """
    offenders = [
        f"{model.__module__}.{model.__qualname__}.{name}: {field.annotation}"
        for model in _wire_models()
        for name, field in model.model_fields.items()
        if not _json_safe(field.annotation)
    ]
    assert not offenders, "fields `model_dump` would not leave JSON-shaped:\n" + "\n".join(
        sorted(offenders)
    )
