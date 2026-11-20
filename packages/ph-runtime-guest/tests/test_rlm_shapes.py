"""A model class as the `output` of a child, and an instance back (typed child results, B4).

The cell defines the shape in the Python way, as a pydantic model, and gets an
instance back. `rlm.run(..., output=Sort)` sends the JSON Schema of the class to
the host, and the child is held to that schema. `await rlm.wait(handle)` returns
`Sort(...)`. These tests cover the guest half, with a dispatch that answers as
the host would. The binding tests of `ph-rlm` run it across the real process
boundary.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from ph_runtime.channel import jsonable
from ph_runtime.proxies import RlmNamespace

pytestmark = pytest.mark.anyio


class Sort(BaseModel):
    """What a sorting child returns."""

    kind: Literal["bug", "feature"]
    note: str | None = None


class _Host:
    """A dispatch that answers as the host does: a handle for `run`, and a settled
    child for `wait`. It keeps every frame that it was sent."""

    def __init__(self, settled: dict[str, Any]) -> None:
        self.settled = settled
        self.frames: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, _namespace: str, name: str, arguments: dict[str, Any]) -> Any:  # noqa: ANN401
        self.frames.append((name, arguments))
        return {"childId": "child-1"} if name == "run" else self.settled


def _rlm(host: _Host) -> RlmNamespace:
    bindings = {"run": ["prompt", "name", "output"], "wait": ["child_id"]}
    return RlmNamespace("rlm", bindings, host)


async def test_a_model_class_is_remembered_and_comes_back_an_instance() -> None:
    """The guest half of the gate. Sabotage: forget the class, or return the dict."""
    host = _Host({"childId": "child-1", "status": "done", "value": {"kind": "bug"}})
    rlm = _rlm(host)

    handle = await rlm.run(prompt="sort this", output=Sort)
    value = await rlm.wait(handle)
    await rlm("sort this too", output=Sort)

    assert host.frames[1] == ("wait", {"child_id": "child-1"}), "a handle is read as its id"
    assert isinstance(value, Sort) and value.kind == "bug"
    assert isinstance(await rlm.wait("child-1"), Sort), "the shorthand takes an output too"


def test_the_wire_spells_a_model_class_as_its_schema_and_an_instance_as_its_json() -> None:
    """A class crosses to the host as the shape that it declares, wherever it is
    in a frame. The child is held to that shape. An instance crosses as its JSON,
    as the value of a cell does. Sabotage: leave both to `repr`."""
    frame = {"arguments": {"output": Sort}, "value": Sort(kind="feature")}

    assert json.loads(json.dumps(frame, default=jsonable)) == {
        "arguments": {"output": Sort.model_json_schema()},
        "value": {"kind": "feature", "note": None},
    }


@pytest.mark.parametrize(
    ("output", "settled", "expected"),
    [
        ({"type": "object"}, {"status": "done", "value": {"kind": "bug"}}, {"kind": "bug"}),
        (None, {"status": "done", "answer": "it is a bug"}, "it is a bug"),
    ],
    ids=["a-schema-not-a-class", "no-output"],
)
async def test_a_child_asked_without_a_class_hands_back_what_it_returned(
    output: dict[str, Any] | None, settled: dict[str, Any], expected: object
) -> None:
    """A schema that is given as a dict comes back as the dict. A child that was
    asked for no shape returns its answer. Sabotage: validate against a class
    that was never given."""
    rlm = _rlm(_Host(settled))

    await rlm.run(prompt="sort this", output=output)

    assert await rlm.wait("child-1") == expected
