"""P1-25 — the output modes.

Gate: *an RPC round trip in the dsh Python SDK's client shape.*

The modes share one property worth testing rather than assuming: `json` and
`rpc` emit **the log's own envelopes**, camelCase, not a per-mode rendering
(I-7). A wrapper consuming a stream and a tool reading the stored file then
parse one format — and dsh's tooling reads both.

## Why the envelope is one module and not one per transport

P5-01 shipped the two transports as two: the daemon grew `root/*` methods and
**dropped the `"jsonrpc": "2.0"` field** the RPC mode sends. The divergence was
invisible because **each transport only ever tested itself** — which is why the
envelope, the version and the capability block now live in `ph_app.protocol` and
each server owns only its method table.

## Why the daemon composes the profile once and mounts it many times

The daemon mounts one `Context` per root and the YAML never changes between them,
so re-reading it per root was **~74% of the cost of starting one**. Every other
mode composes exactly once.

## Why the snapshot page is a count and not a byte budget

The first draft measured each event with its own `dumps` to fill a 512 KiB page,
which cost **8.2 ms per page — 2.2x the encode it existed to bound** — and all of
it was discarded. A count needs no measuring pass, and the transport's `MAX_LINE`
is the real protection against an oversized frame.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ph.cordis import Profile
from ph_app.modes import render_transcript, run_json, run_rpc, run_transcript
from ph_app.profiles import compose_profile

pytestmark = pytest.mark.anyio

ENVELOPE_FIELDS = {"type", "seq", "time", "data", "ignorable", "sourceEventSeqs", "surfaceOp"}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Profile:
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    return compose_profile("headless")


async def test_json_mode_streams_the_logs_own_envelopes(profile: Profile) -> None:
    out = io.StringIO()
    result = await run_json(
        profile, "hello", provider="fake", model="fake-1", session_id="demo", out=out
    )
    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert lines[0]["type"] == "session/header"

    events = lines[1:]
    assert [event["seq"] for event in events] == list(range(len(events)))
    for event in events:
        assert set(event) <= ENVELOPE_FIELDS
    assert result.session_id == "demo"
    # Same count as the log, because it is the log.
    assert result.events == len(events)


async def test_transcript_mode_reads_what_a_person_saw(profile: Profile) -> None:
    result = await run_transcript(
        profile, "what is a session log?", provider="fake", model="fake-1"
    )
    assert "you: what is a session log?" in result.text
    assert "pH: ok" in result.text


def test_the_transcript_renderer_labels_every_block_kind() -> None:
    from ph.llm.types import (
        create_assistant_message,
        create_tool_result_message,
        create_user_message,
    )

    messages = (
        create_user_message(content=[{"type": "text", "text": "do it"}], source={"kind": "user"}),
        create_user_message(
            content=[{"type": "text", "text": "cwd: /x"}],
            source={"kind": "plugin", "plugin": "workspace", "form": "snapshot", "sections": []},
        ),
        create_assistant_message(
            content=[
                {"type": "reasoning", "text": "considering"},
                {"type": "text", "text": "on it"},
                {"type": "tool-call", "id": "c1", "name": "read", "arguments": '{"path":"a"}'},
            ],
            provider="fake",
            model="m",
        ),
        create_tool_result_message(
            call_id="c1", content=[{"type": "text", "text": "file body"}], is_error=False
        ),
    )
    rendered = render_transcript(messages).splitlines()
    assert rendered[0] == "you: do it"
    # Injected context is labelled as context, not as the user talking.
    assert rendered[1] == "context: cwd: /x"
    assert rendered[2] == "pH (thinking): considering"
    assert rendered[3] == "pH: on it"
    assert rendered[4].startswith("pH → read(")
    assert rendered[5] == "← file body"


async def test_an_rpc_round_trip_in_the_sdk_shape(profile: Profile) -> None:
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"sessionId": "rpc-1"}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {"sessionId": "rpc-1", "prompt": "hello"},
        },
        {"jsonrpc": "2.0", "id": 4, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": {}},
    ]
    stdin = io.StringIO("".join(f"{json.dumps(request)}\n" for request in requests))
    out = io.StringIO()
    await run_rpc(profile, provider="fake", model="fake-1", stdin=stdin, out=out)

    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    replies = {frame["id"]: frame for frame in frames if "id" in frame}
    notifications = [frame for frame in frames if "method" in frame]

    assert replies[1]["result"]["protocolVersion"] == 1
    assert replies[1]["result"]["capabilities"]["streaming"] is True
    assert replies[2]["result"]["sessionId"] == "rpc-1"
    assert replies[3]["result"]["events"] > 0
    assert {schema["name"] for schema in replies[4]["result"]["tools"]} >= {"read", "edit", "bash"}
    assert replies[5]["result"] == {"ok": True}

    # Streaming notifications carry the log's envelopes, not a rendering.
    events = [frame for frame in notifications if frame["method"] == "session.event"]
    assert events, "no session.event notifications were sent"
    assert set(events[0]["params"]["event"]) <= ENVELOPE_FIELDS
    statuses = [
        frame["params"]["status"] for frame in notifications if frame["method"] == "session.status"
    ]
    assert statuses == ["running", "idle"]


async def test_an_unknown_rpc_method_is_an_error_not_a_crash(
    profile: Profile,
) -> None:
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "nonsense", "params": {}}) + "\n"
    )
    out = io.StringIO()
    await run_rpc(profile, provider="fake", model="fake-1", stdin=stdin, out=out)
    (frame,) = [json.loads(line) for line in out.getvalue().splitlines()]
    assert frame["error"]["code"] == -32000
    assert "nonsense" in frame["error"]["message"]


async def test_a_malformed_rpc_line_is_ignored(profile: Profile) -> None:
    stdin = io.StringIO(
        "{not json\n\n" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
    )
    out = io.StringIO()
    await run_rpc(profile, provider="fake", model="fake-1", stdin=stdin, out=out)
    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    # A peer sending garbage must not take the endpoint down.
    assert len(frames) == 1
    assert frames[0]["id"] == 1
