"""P0-07 — the `SessionEvent` envelope and its wire form.

Gate: *the pin test; the wire form equals dsh's envelope byte-for-byte on a
fixture.*

Byte-compatibility is not nostalgia: D2/Q2 make a pH log a log dsh tooling
reads directly, which is why no `--format` renderer is needed anywhere.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from ph.session.events import SessionEvent, SurfaceReplace, now_ms
from ph.wire import wire_alias


def test_every_envelope_field_maps_to_to_camel_of_its_name() -> None:
    # The envelope is a dataclass, not a pydantic model, so its mapping is
    # hand-written — and therefore pinned here rather than trusted.
    mapping = {
        "type": "type",
        "seq": "seq",
        "time": "time",
        "data": "data",
        "ignorable": "ignorable",
        "source_event_seqs": "sourceEventSeqs",
        "surface_op": "surfaceOp",
    }
    names = {field.name for field in dataclasses.fields(SessionEvent)}
    assert names == set(mapping)
    for name, expected in mapping.items():
        assert wire_alias(name) == expected


def test_wire_form_matches_the_dsh_envelope() -> None:
    event = SessionEvent(
        type="assistant/message",
        seq=12,
        time=1_700_000_000_000,
        data={"turn": 1, "step": 1, "message": {"id": "m", "role": "assistant"}},
        source_event_seqs=(7, 8, 9),
        surface_op="append",
    )
    assert event.to_wire() == {
        "type": "assistant/message",
        "seq": 12,
        "time": 1_700_000_000_000,
        "data": {"turn": 1, "step": 1, "message": {"id": "m", "role": "assistant"}},
        "sourceEventSeqs": [7, 8, 9],
        "surfaceOp": "append",
    }


def test_absent_optional_fields_are_omitted() -> None:
    event = SessionEvent(type="turn/start", seq=0, time=1, data={"turn": 1})
    assert set(event.to_wire()) == {"type", "seq", "time", "data"}


def test_replace_op_round_trips() -> None:
    event = SessionEvent(
        type="user/message",
        seq=9,
        time=1,
        data={"id": "m", "role": "user", "content": [], "source": {"kind": "user"}},
        source_event_seqs=(3, 4),
        surface_op=SurfaceReplace(start=3, end=4),
    )
    wire = event.to_wire()
    assert wire["surfaceOp"] == {"op": "replace", "start": 3, "end": 4}
    restored = SessionEvent.from_wire(json.loads(json.dumps(wire)))
    assert restored.surface_op == SurfaceReplace(start=3, end=4)
    assert restored.source_event_seqs == (3, 4)
    assert restored.to_wire() == wire


def test_from_wire_accepts_either_casing() -> None:
    snake = {
        "type": "user/message",
        "seq": 0,
        "time": 1,
        "data": {},
        "source_event_seqs": [],
        "surface_op": "append",
    }
    assert SessionEvent.from_wire(snake).surface_op == "append"


@pytest.mark.parametrize(
    "wire",
    [
        {"type": "", "seq": 0, "time": 1, "data": {}},
        {"type": "turn/start", "seq": -1, "time": 1, "data": {}},
        {"type": "turn/start", "seq": 0, "time": -1, "data": {}},
        {"type": "turn/start", "seq": True, "time": 1, "data": {}},
        {"type": "turn/start", "seq": 0, "time": 1, "data": {}, "extra": 1},
        {"type": "turn/start", "seq": 0, "time": 1, "data": {}, "ignorable": False},
        {"type": "user/message", "seq": 0, "time": 1, "data": {}, "surfaceOp": {"op": "x"}},
    ],
)
def test_malformed_envelopes_are_refused(wire: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SessionEvent.from_wire(wire)


def test_now_ms_is_a_non_negative_safe_integer() -> None:
    stamp = now_ms()
    assert isinstance(stamp, int)
    assert 0 < stamp < 2**53 - 1
