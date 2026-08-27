"""The program is a hostile peer (C10).

Model code holds fd 3 and can write whatever it likes onto it. Every assertion
here is about the host surviving that: no forged field reaches a handler, no
non-numeric id is ever echoed back, and nothing the child can send makes the
decoder raise — because a decoder that raises is a decoder the child can crash
the host with, on demand.
"""

from __future__ import annotations

import json
import random

import pytest

from ph.session.json import JSON_MAX_SAFE_INTEGER
from ph_rlm.kernel.codec import decode, encode, has_unsafe_integer
from ph_rlm.kernel.protocol import ReplyFrame

HOSTILE = [
    "",
    "   ",
    "not json at all",
    "null",
    "[]",
    '"a string"',
    "123",
    "{}",
    '{"type": 7}',
    '{"type": "run"}',  # a *host* frame; the host never accepts one inbound
    '{"type": "unknown-frame", "id": 1}',
    '{"type": "done"}',  # missing the required id
    '{"type": "done", "id": "1"}',  # a string id
    '{"type": "done", "id": true}',  # bool is an int in Python; must not pass
    '{"type": "done", "id": 1.5}',
    '{"type": "done", "id": null}',
    '{"type": "call", "id": 1, "global": "tools", "name": "read"}',  # no args
    '{"type": "call", "id": 1, "global": "tools", "name": "read", "args": []}',
    '{"type": "log", "stream": "stdout", "text": 42}',
    '{"type": "snapshot", "id": 1, "variables": {}}',
    '{"nested": {"deeply": {"type": "done", "id": 1}}}',
    '{"type": "done", "id": 1' + "," * 500 + "}",
    "\x00\x01\x02",
]


@pytest.mark.parametrize("raw", HOSTILE)
def test_hostile_input_never_raises(raw: str) -> None:
    assert decode(raw) is None


def test_a_valid_frame_is_rebuilt_and_stripped() -> None:
    """Declared fields survive; everything else is dropped, not carried."""
    frame = decode(
        json.dumps(
            {
                "type": "done",
                "id": 3,
                "value": {"ok": True},
                "forged": "surprise",
                "__class__": "evil",
            }
        )
    )
    assert frame == {"type": "done", "id": 3, "value": {"ok": True}}


def test_an_id_that_is_not_a_number_is_never_echoed() -> None:
    """The frame is dropped before there *is* an id to reply to.

    That ordering is the point: a codec that coerced `"1"` to `1` would let the
    child choose which pending call a reply lands on.
    """
    for bad in ('"1"', "1.0", "true", "null", "[1]"):
        frame = f'{{"type": "call", "id": {bad}, "global": "t", "name": "n", "args": {{}}}}'
        assert decode(frame) is None


def test_an_integer_too_large_for_a_js_reader_is_refused() -> None:
    """pH's log is JSON that dsh's TypeScript tooling reads (Q2).

    Past 2^53 a JS reader loses precision silently, so the frame is refused here
    rather than written into a log that cannot be read back faithfully.
    """
    assert has_unsafe_integer(f'{{"type": "done", "id": {JSON_MAX_SAFE_INTEGER + 1}}}')
    assert decode(f'{{"type": "done", "id": {JSON_MAX_SAFE_INTEGER + 1}}}') is None
    assert decode(f'{{"type": "done", "id": {JSON_MAX_SAFE_INTEGER}}}') == {
        "type": "done",
        "id": JSON_MAX_SAFE_INTEGER,
    }
    # Digits inside a string are not a number, which a text scan would confuse.
    assert not has_unsafe_integer('{"type": "fault", "message": "99999999999999999999"}')


def test_fuzzed_frames_neither_raise_nor_forge() -> None:
    random.seed(20260827)
    alphabet = '{}[]",:0123456789abcdefgtruenulls -\\\n\x00'
    for _ in range(4000):
        raw = "".join(random.choice(alphabet) for _ in range(random.randint(0, 60)))
        frame = decode(raw)
        if frame is None:
            continue
        assert frame["type"] in {
            "boot-ack",
            "call",
            "log",
            "display",
            "snapshot",
            "done",
            "fault",
        }
        if "id" in frame:
            assert type(frame["id"]) is int


def test_outbound_frames_encode_as_one_camelcase_line() -> None:
    line = encode(ReplyFrame(id=4, ok=True, value={"n": 1}))
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert json.loads(line) == {"type": "reply", "id": 4, "ok": True, "value": {"n": 1}}
