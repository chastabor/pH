"""The program is a hostile peer (C10).

Model code holds fd 3 and can write anything to it. Every assertion here is
about the host surviving that:

* No forged field reaches a handler.
* No non-numeric id is ever echoed back.
* Nothing that the child can send makes the decoder raise. A decoder that raises
  is a decoder that the child can use to crash the host on demand.

## Why `ph.json._DECODER` is built one time, with no hooks

`decode` runs on every frame that the guest sends. `json.loads` pays a dispatch
on every call: a wrapper frame plus a str/bytes sniff. That cost is worth
removing: **0.334 µs against 0.406 µs** on a `log` frame, the most numerous kind
on the stdout path. The `parse_int` hook that the decoder carried before is
gone. That is a separate rule and not a performance rule. The integer bound
belongs to the `int` fields that the spec declares (`_coerce`), and not to a
parse hook that vetoed a whole frame over a field that the spec never asked to
see.
"""

from __future__ import annotations

import json
import random

import pytest

from ph.json import JSON_MAX_SAFE_INTEGER
from ph_rlm.kernel.codec import decode, encode
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
    '{"type": "call", "id": 1, "run": 1, "global": "tools", "name": "read"}',  # no args
    '{"type": "call", "id": 1, "run": 1, "global": "tools", "name": "read", "args": []}',
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
    """Declared fields survive. Everything else is dropped and not carried."""
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

    That order is the purpose. A codec that coerced `"1"` to `1` would let the
    child choose which pending call a reply lands on.
    """
    for bad in ('"1"', "1.0", "true", "null", "[1]"):
        frame = (
            f'{{"type": "call", "id": {bad}, "run": 1, "global": "t", "name": "n", "args": {{}}}}'
        )
        assert decode(frame) is None


def test_an_integer_too_large_for_a_js_reader_is_refused_where_the_host_reads_one() -> None:
    """The log of pH is JSON that the TypeScript tooling of dsh reads (Q2), and an
    `id` that the host echoes back is exactly the number that must survive that
    reader. Past 2^53, an `int` field therefore fails coercion like any other
    wrong shape, and a required one drops the frame."""
    assert decode(f'{{"type": "done", "id": {JSON_MAX_SAFE_INTEGER + 1}}}') is None
    assert decode(f'{{"type": "done", "id": {-JSON_MAX_SAFE_INTEGER - 1}}}') is None
    assert decode(f'{{"type": "done", "id": {JSON_MAX_SAFE_INTEGER}}}') == {
        "type": "done",
        "id": JSON_MAX_SAFE_INTEGER,
    }
    # Digits inside a string are not a number. A text scan would confuse the two.
    assert decode('{"type": "fault", "message": "99999999999999999999"}') is not None


def test_a_large_integer_inside_a_payload_does_not_veto_the_frame() -> None:
    """The bound applies to the fields that the spec declares `int`, and to nothing
    that the spec declines to inspect. The bound was a `parse_int` hook before,
    and then any big number anywhere in the line dropped the whole frame. A
    `boot-ack` whose `limits` said `RLIM_INFINITY` never arrived, and a cell that
    ended in `2**60` never settled (2026-09-07). The walker of the log itself
    judges what reaches the log, with a path in its message."""
    huge = 2**63 - 1
    ack = decode(f'{{"type": "boot-ack", "protocol": 2, "python": "3", "limits": {{"a": {huge}}}}}')
    assert ack is not None and ack["type"] == "boot-ack"
    assert ack["limits"] == {"a": huge}
    done = decode(f'{{"type": "done", "id": 1, "value": {2**60}}}')
    assert done is not None and done["type"] == "done"
    assert done["value"] == 2**60
    call = decode(
        '{"type": "call", "id": 1, "run": 1, "global": "g", "name": "n", '
        f'"args": {{"n": {huge}}}}}'
    )
    assert call is not None and call["type"] == "call"
    assert call["args"] == {"n": huge}


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
        # The check narrows on the tag and not on `"id" in frame`. A membership
        # test tells mypy nothing about a `TypedDict` union, which is the same
        # lesson that `_WORKSPACE_KINDS` records for `Literal`s (issue 49c).
        # Naming the three that carry an id also made the set *checkable*.
        # `fault` was in the first draft of this line, and `FaultFrame` has no
        # `id`. The `in frame` form could never have reported that.
        if frame["type"] in {"call", "done", "snapshot"}:
            assert type(frame["id"]) is int


def test_outbound_frames_encode_as_one_camelcase_line() -> None:
    line = encode(ReplyFrame(id=4, ok=True, value={"n": 1}))
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert json.loads(line) == {"type": "reply", "id": 4, "ok": True, "value": {"n": 1}}
