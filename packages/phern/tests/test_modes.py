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
import os
import signal
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from rlm_fixtures import HOST_INTERPRETER

from ph.cordis import Profile, ProfileDocument
from ph.cordis import context as cordis_context
from ph.keys import SESSION_TELEMETRY, SUBAGENTS
from ph.llm.fake import FakeAdapter
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    StreamChunk,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
    text_of,
)
from ph.persistence import SessionBusy, open_session, read_session
from ph.resources import Stopped, until_signaled
from ph.seams.subagents import (
    STATUS,
    SUSPENDED_DETAIL,
    UNRECOVERABLE_DETAIL,
    SubagentService,
    child_state_of,
)
from ph.session import child_session_id
from ph.testing import (
    admitted_child,
    hold_session,
    log_event,
    noted,
    raising,
    searches_into,
    stored_log,
)
from ph_app.daemon.recovery import CHILD_RETRY_LIMIT
from ph_app.modes import render_transcript, run_json, run_print, run_rpc, run_transcript
from ph_app.profiles import compose_profile
from ph_app.protocol import PROTOCOL_VERSION, request, result_of
from ph_app.runtime import mounted
from ph_rlm.presentation import IPYTHON

pytestmark = pytest.mark.anyio

ENVELOPE_FIELDS = {
    "type",
    "seq",
    "time",
    "data",
    "ignorable",
    "sourceEventSeqs",
    "surfaceOp",
    # Batch membership (log format 2, P10-15): a prompt's claim, its step and its
    # message are one batch (S3), so every run carries it.
    "batch",
}


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
        profile,
        "hello",
        session_id="demo",
        out=out,
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


async def test_a_resumed_print_run_prints_only_the_new_answer(profile: Profile) -> None:
    """`-p` prints what this run produced, not the conversation so far (G6).

    A `--session` that already exists is resumed — that is P5-03's fix, and it is
    right — but the text was read off `session.transcript()`, which is every
    assistant message the session ever held. So the second run printed two
    answers, the third printed three, and a script capturing stdout got a reply
    that grew by a paragraph each time it asked a follow-up.

    The whole log is still written and still readable; only what this invocation
    puts on stdout is narrowed. `--mode transcript` is the mode that prints the
    conversation, and it is unchanged.
    """
    first = await run_print(profile, "one", session_id="resumed")
    second = await run_print(profile, "two", session_id="resumed")

    assert first.text == "ok"
    assert second.text == "ok", "the first run's answer was printed again"
    # And the session really did continue rather than starting over: the second
    # run's log holds both turns.
    assert second.events > first.events
    assert second.ended == "completed"


async def test_transcript_mode_reads_what_a_person_saw(profile: Profile) -> None:
    result = await run_transcript(profile, "what is a session log?")
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
    # Injected context is labeled as context, not as the user talking.
    assert rendered[1] == "context: cwd: /x"
    assert rendered[2] == "pH (thinking): considering"
    assert rendered[3] == "pH: on it"
    assert rendered[4].startswith("pH → read(")
    assert rendered[5] == "← file body"


def test_a_transcript_names_media_rather_than_dropping_it() -> None:
    """The rule `ph_app.tui.trajectory._text` states: an auditor wants to see that
    an image was there. Every other renderer of blocks already follows it; this
    mode dropped instead, so a transcript of a turn carrying a screenshot did not
    say one existed. Nothing was lost from the log — only the reader was misled.
    """
    from ph.llm.types import create_user_message

    messages = (
        create_user_message(
            content=[
                {"type": "text", "text": "look at this"},
                {
                    "type": "media",
                    "attachment": {
                        "attachmentId": "sha256:abc",
                        "mime": "image/png",
                        "bytes": 12,
                    },
                },
            ],
            source={"kind": "user"},
        ),
    )
    assert render_transcript(messages).splitlines() == ["you: look at this", "you: [media]"]


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
    await run_rpc(profile, stdin=stdin, out=out)

    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    replies = {frame["id"]: frame for frame in frames if "id" in frame}
    notifications = [frame for frame in frames if "method" in frame]

    # The constant, not a literal — `--mode rpc` and the daemon answer with one
    # number, and a copy here made a protocol bump a search for the copies.
    assert replies[1]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert replies[1]["result"]["capabilities"]["streaming"] is True
    assert replies[2]["result"]["sessionId"] == "rpc-1"
    assert replies[3]["result"]["events"] > 0
    assert {schema["name"] for schema in replies[4]["result"]["tools"]} >= {"read", "edit", "bash"}
    # `shutdown` is a `Notify`: no reply model, so no result. A peer that sends
    # it *with* an id — as this round trip does, to prove the body still runs —
    # gets an acknowledged frame carrying nothing, where it used to carry an
    # `{"ok": true}` nothing had asked for.
    assert replies[5]["result"] is None

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
    await run_rpc(profile, stdin=stdin, out=out)
    (frame,) = [json.loads(line) for line in out.getvalue().splitlines()]
    assert frame["error"]["code"] == -32000
    assert "nonsense" in frame["error"]["message"]


async def test_a_malformed_rpc_line_is_ignored(profile: Profile) -> None:
    stdin = io.StringIO(
        "{not json\n\n" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}) + "\n"
    )
    out = io.StringIO()
    await run_rpc(profile, stdin=stdin, out=out)
    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    # A peer sending garbage must not take the endpoint down.
    assert len(frames) == 1
    assert frames[0]["id"] == 1


async def _rpc_prompt(profile: Profile, session_id: str) -> None:
    prompt = request(1, "session/prompt", {"sessionId": session_id, "prompt": "hello"})
    await run_rpc(profile, stdin=io.StringIO(f"{json.dumps(prompt)}\n"), out=io.StringIO())


# `print` stands for the one-shot modes: `--mode json` and transcript make their
# agent through the same `runtime.prompted`.
_HOSTS: dict[str, Callable[[Profile, str], Awaitable[object]]] = {
    "rpc": _rpc_prompt,
    "print": lambda profile, sid: run_print(profile, "hello", session_id=sid),
}


@pytest.mark.parametrize("host", sorted(_HOSTS))
async def test_every_host_sweeps_the_children_a_stopped_run_left_working(
    host: str, profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every host that makes a root's agent sweeps its children, not only the daemon.

    Before, a child that a stopped or crashed run left `running` still read as
    working under `--mode rpc` and the one-shot modes: in its own log, in the
    model's list of its children, and to the `task` crash check, which told the
    model it "is started again with its parent" — which those hosts never made
    true. The sweep runs where the session's agent is made, because a readmitted
    child hangs off its parent's agent: at rpc's first prompt, and in `prompted`
    before the one-shot's turn.

    Asserted on a child whose owner nothing here mounts, the one decision a host can
    be checked by without running a provider: nothing can readmit it, so the sweep
    ends it in its own log. The bound is the daemon's, and only a host can be
    checked for passing it, since the seam states none.

    Sabotage: drop `resume_children` from `RpcServer._prompt` or from
    `runtime.prompted`, and the child is still `running`.
    """
    bounds: list[int] = []
    original = SubagentService.resume_children

    async def spy(self: Any, parent: Any, *, retry_limit: int) -> Any:  # noqa: ANN401
        bounds.append(retry_limit)
        return await original(self, parent, retry_limit=retry_limit)

    monkeypatch.setattr(SubagentService, "resume_children", spy)
    async with mounted(profile) as ctx:
        parent = await open_session(ctx, "kids")
        child = admitted_child(ctx, parent, "r1", {"prompt": "look"})
        log_event(child, STATUS, {"status": "running"})

    await _HOSTS[host](profile, "kids")

    child_id = child_session_id("kids", "r1")
    header, events = read_session(stored_log(tmp_path / "sessions", child_id, family="kids"))
    state = child_state_of(child_id, header, events)
    assert state.status == "error", f"the child was left {state.status}"
    assert state.detail == UNRECOVERABLE_DETAIL
    assert bounds == [CHILD_RETRY_LIMIT], "the daemon's own ladder bound, not another"


async def test_a_signal_stops_an_rpc_server_waiting_on_its_peer(profile: Profile) -> None:
    """A peer that closes `--mode rpc` with `SIGTERM` is the common case.

    The server spends its life in a worker thread blocked on the peer's next line,
    and a thread call does not let a cancellation through by default. So the stop a
    signal asks for waited for a line the peer was never going to send, and the
    hard stop ended the process with its sessions unwound by nobody.

    Sabotage: drop `abandon_on_cancel` from `run_rpc`'s read, and the stop waits for
    the line this test writes after five seconds.
    """
    read_end, write_end = os.pipe()
    stdin = os.fdopen(read_end, "r")
    # A line after five seconds, so a server still waiting for one is reported as
    # slow rather than hanging the suite.
    late = anyio.Event()

    async def serving() -> None:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(partial(run_rpc, profile, stdin=stdin, out=io.StringIO()))
            await anyio.sleep(0.2)
            os.kill(os.getpid(), signal.SIGTERM)
            await late.wait()

    async def unblock() -> None:
        await anyio.sleep(5)
        os.write(write_end, b"\n")

    started = time.monotonic()
    try:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(unblock)
            stopped = await until_signaled(serving)
            elapsed = time.monotonic() - started
            tasks.cancel_scope.cancel()
    finally:
        os.close(write_end)
        await anyio.sleep(0.05)
        stdin.close()

    assert stopped == Stopped(signal.SIGTERM)
    assert elapsed < 3, f"the stop waited {elapsed:.1f}s for the peer's next line"


async def test_a_signal_mid_turn_hands_a_print_runs_session_back(
    profile: Profile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`phern -p` under `SIGTERM` ended at once: the default action, which is a crash
    with a politer name. Now the run is canceled and its mount unwinds.

    Asserted by what only a finished unwind leaves: the lease given back, so the
    next run on the session is not refused, and a log that reads whole, holding
    both turns.

    Sabotage: drop the `body.cancel()` in `until_signaled`'s stop, and the run waits
    on its model call until the bound below fails it.
    """

    async def stuck(self: object, options: object) -> AsyncIterator[Any]:
        os.kill(os.getpid(), signal.SIGTERM)
        await anyio.sleep_forever()
        yield  # pragma: no cover

    with monkeypatch.context() as patched:
        patched.setattr(FakeAdapter, "stream", stuck)
        with anyio.fail_after(30):
            stopped = await until_signaled(partial(run_print, profile, "hello", session_id="cut"))
    assert stopped == Stopped(signal.SIGTERM)

    await run_json(profile, "and again", session_id="cut", out=io.StringIO())

    _header, events = read_session(stored_log(tmp_path / "sessions", "cut"))
    assert [event.seq for event in events] == list(range(len(events)))
    assert sum(event.type == "turn/start" for event in events) == 2


CHILD_TASK = "look into it"


def _from_the_child(options: GenerateOptions) -> bool:
    """Whether a request is the child's. Its task reached it as a user message; the
    parent holds the same words only inside its own tool call."""
    return any(
        message.role == "user" and CHILD_TASK in text_of(message.content)
        for message in options.messages
    )


def _cell_answered(options: GenerateOptions) -> bool:
    """Whether the parent's request already carries its cell's result."""
    return any(
        block.type == "tool-result" for message in options.messages for block in message.content
    )


async def _delegating_cell() -> AsyncIterator[StreamChunk]:
    """The parent's first step: one cell that spawns a child and does not wait for it."""
    arguments = json.dumps(
        {"program": f"h = await rlm.run(prompt={CHILD_TASK!r}, name='scout')\nh['name']"}
    )
    yield BlockStart(index=0, block_type="tool-call")
    yield ToolCallDelta(index=0, id="cell-1", name=IPYTHON, arguments_delta=arguments)
    yield BlockEnd(index=0, block=ToolCallBlock(id="cell-1", name=IPYTHON, arguments=arguments))
    yield UsageChunk(usage=TokenUsage(input_tokens=10, output_tokens=10))
    yield Finish(reason=FinishReason(kind="tool-calls"))


async def test_a_print_run_suspends_a_child_still_working_when_its_turn_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, guest_coverage: None
) -> None:
    """What `phern -p` does with an `rlm.run` child its turn ended without, pinned.

    `rlm.run` hands back an admission and does not wait, and the parent's run ends
    when its own inbox is empty, so a one-shot can finish its turn while a child is
    still working. Its mount then unwinds: the drain waits for the child's drive
    for at most `DRAIN_SECONDS`, and the provider suspends what is left (`queued`,
    `SUSPENDED_DETAIL`), keeping it resumable and spending no restart attempt. The
    parent is not driven again, so the printed answer is the one its turn gave, and
    the child's work waits for a host that readmits it (DESIGN.md §8, "A one-shot
    run does not sweep children").

    Held at its model call, the child is mid-turn when the parent's turn ends, and
    the drain's window is shortened so the test does not spend the real five
    seconds waiting out a child that never finishes.

    Sabotage: drop the `suspend` disposer from `ph_rlm.subagents.apply`, and the
    parent's teardown revokes the child instead: it ends `canceled`, and no later
    start resumes its work.
    """
    drain = 0.5
    monkeypatch.setattr(cordis_context, "DRAIN_SECONDS", drain)
    original = FakeAdapter.stream
    child_working = anyio.Event()
    asked: list[str] = []

    async def scripted(self: FakeAdapter, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        if _from_the_child(options):
            asked.append("child")
            child_working.set()
            await anyio.sleep_forever()
        if _cell_answered(options):
            # Only once the child is mid-turn, so the parent's turn ends on a child
            # that is working rather than one that has not started.
            await child_working.wait()
            asked.append("parent answers")
            async for chunk in original(self, options):
                yield chunk
            return
        asked.append("parent delegates")
        async for chunk in _delegating_cell():
            yield chunk

    monkeypatch.setattr(FakeAdapter, "stream", scripted)
    profile = compose_profile(
        "rlm",
        then=[ProfileDocument("test", [{"id": "code-runtime-python", "config": HOST_INTERPRETER}])],
    )

    started = time.monotonic()
    with anyio.fail_after(60):
        result = await run_print(profile, "hello", session_id="delegator")
    elapsed = time.monotonic() - started

    assert result.text == "ok", "the answer the parent's own turn gave"
    assert asked == ["parent delegates", "child", "parent answers"], (
        "the parent was driven again after its turn"
    )
    assert elapsed >= drain, "the run left without waiting for the child's drive"

    parent, _events = read_session(stored_log(tmp_path / "sessions", "delegator"))
    async with mounted(compose_profile("headless")) as ctx:
        children = await ctx.require(SUBAGENTS).load_children("delegator", parent.family)
    (child,) = children.values()
    assert child.starts == 1, "the child never started"
    assert child.status == "queued", f"the child was left {child.status}"
    assert child.detail == SUSPENDED_DETAIL
    assert not child.deleted, "a stop is not a revocation"


async def test_a_second_one_shot_run_on_one_session_resumes_it(
    profile: Profile, tmp_path: Path
) -> None:
    """`--session x` twice is one conversation, not two logs in one file (P5-03).

    Before `open_session` the second run *created* a session over the first one's
    file — one header, `seq` restarting at zero — and the trajectory reader then
    refused the whole log, so both turns were lost with no daemon and no race
    involved. Asserted the way that reader sees it: contiguous from zero, and
    holding both turns.
    """
    for prompt in ("hello", "and again"):
        await run_json(
            profile,
            prompt,
            session_id="demo",
            out=io.StringIO(),
        )

    header, events = read_session(stored_log(tmp_path / "sessions", "demo"))
    assert header.id == "demo"
    assert [event.seq for event in events] == list(range(len(events)))
    assert sum(event.type == "turn/start" for event in events) == 2
    assert any(event.type == "session/resumed" for event in events), "resumed, not recreated"


async def test_a_one_shot_run_reads_a_stored_session_once(
    profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed one-shot run reads its session's start once, before mounting: the
    environment and the owner for the mount, and the family the log was found in for
    the open. It read the start twice — once for the owner, once inside
    `session_profile` — and the open searched the store a third time.

    Sabotage: drop `family` from `prompted`'s open, and the store is searched twice.
    """
    from ph.persistence import jsonl
    from ph_app import runtime, sessions

    await run_json(profile, "hello", session_id="demo", out=io.StringIO())
    starts: list[str] = []
    searched: list[str] = []
    real_start = sessions.recorded_start
    monkeypatch.setattr(
        runtime,
        "recorded_start",
        lambda directory, session_id: noted(starts, session_id, real_start(directory, session_id)),
    )
    monkeypatch.setattr(jsonl, "locate_under", searches_into(searched))

    await run_json(profile, "and again", session_id="demo", out=io.StringIO())

    assert starts == ["demo"]
    assert searched.count("demo") == 1, searched


async def test_an_rpc_session_made_without_an_id_is_not_searched_for(
    profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`session/new` with no id leaves the id to `open_session`, which knows an id it
    makes is new and does not look for it. rpc minted its own and passed it in, so
    every new session searched the whole store for a log made a moment before. A
    prompt naming the id it got back is served on the same mount.

    Sabotage: mint the id in `RpcServer` and pass it to `open_session`, and the store
    is searched.
    """
    from ph.persistence import jsonl
    from ph_app.modes.rpc_mode import RpcServer

    monkeypatch.setattr(
        jsonl, "locate_under", raising(AssertionError("searched for a session rpc had just made"))
    )
    out = io.StringIO()
    async with AsyncExitStack() as exits:
        server = RpcServer(profile=profile, exits=exits, out=out)
        await server.handle({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}})
        made = result_of(_rpc_replies(out)[1])["sessionId"]
        prompt = {"sessionId": made, "prompt": "hello"}
        await server.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": prompt}
        )

        assert result_of(_rpc_replies(out)[2])["sessionId"] == made
        assert list(server._served) == [made]


async def test_an_rpc_session_named_by_a_stored_id_is_searched_for_once(
    profile: Profile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer naming a stored session: rpc reads its start before the mount and opens
    it where that read found the log — one search of the store, not one for the start
    and another for the resume.

    Sabotage: open by id alone in `RpcServer._open`, and the store is searched twice.
    """
    from ph.persistence import jsonl
    from ph_app.modes.rpc_mode import RpcServer

    await _rpc_prompt(profile, "stored")
    searched: list[str] = []
    monkeypatch.setattr(jsonl, "locate_under", searches_into(searched))
    out = io.StringIO()
    async with AsyncExitStack() as exits:
        server = RpcServer(profile=profile, exits=exits, out=out)
        new = {"sessionId": "stored"}
        await server.handle({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": new})

        assert result_of(_rpc_replies(out)[1])["sessionId"] == "stored"
    assert searched.count("stored") == 1, searched


def _rpc_replies(out: io.StringIO) -> dict[int, Any]:
    """The replies an rpc server has written so far, by request id."""
    frames = [json.loads(line) for line in out.getvalue().splitlines()]
    return {frame["id"]: frame for frame in frames if "id" in frame}


async def test_a_one_shot_run_is_refused_a_session_another_process_holds(
    profile: Profile, tmp_path: Path
) -> None:
    """The half the lease used to miss: a print run against a held log is refused.

    The holder is `hold_session`, which takes the lease on the path the store
    would claim — what a daemon, or another `phern -p`, looks like from here —
    and the refusal is the store's own, by name, so the CLI and the daemon
    protocol say one thing.
    Nothing is written: a refused run leaves no partial turn to explain.
    """
    with (
        hold_session(tmp_path / "sessions", "held") as log_path,
        pytest.raises(SessionBusy) as refused,
    ):
        await run_json(
            profile,
            "hello",
            session_id="held",
            out=io.StringIO(),
        )
    assert refused.value.code == "session_already_active"
    assert not log_path.exists()


async def test_a_refused_open_leaves_an_ops_record(profile: Profile, tmp_path: Path) -> None:
    """P5-09's first producer: a fact about the harness, not about a conversation.

    The session this concerns is the one this process was refused, so its log is
    not ours to write — which is exactly what the `ops` channel is for, and why
    it had no producer until something had a fact of that shape to record.
    """
    from ph.persistence import open_session
    from ph_app.runtime import mounted

    seen: list[Any] = []
    with hold_session(tmp_path / "sessions", "held"):
        async with mounted(profile) as ctx:
            ctx.require(SESSION_TELEMETRY).add_sink(seen.append)
            with pytest.raises(SessionBusy):
                await open_session(ctx, "held")

    ops = [record for record in seen if record.channel == "ops"]
    assert [record.severity for record in ops] == ["warn"]
    assert ops[0].attributes["session_id"] == "held"


def test_the_transcript_never_shows_the_person_encrypted_reasoning() -> None:
    """G8's other half, which did not land with the first fix.

    `redacted_thinking` carries ciphertext only Anthropic can read, and it is
    still a `ReasoningBlock` — so the wire half was fixed (it is re-sent as
    itself) while every renderer that shows `block.text` went on putting a wall
    of base64 in front of the person as something the assistant had said. The
    field existed and had no neutral reader.

    Sabotage: render `block.text` unconditionally and the blob is in the output.
    """
    from ph.llm.types import create_assistant_message
    from ph.text import redacted_marker

    blob = "AAAAB3NzaC1yc2EAAAADAQAB" * 4
    messages = (
        create_assistant_message(
            content=[
                {"type": "reasoning", "text": blob, "redacted": True},
                {"type": "reasoning", "text": "considering"},
                {"type": "text", "text": "on it"},
            ],
            provider="anthropic",
            model="m",
        ),
    )

    rendered = render_transcript(messages)

    assert blob not in rendered, "the person was shown ciphertext"
    assert redacted_marker() in rendered, "and told nothing was there"
    # Ordinary reasoning is untouched — this narrows one case, it does not
    # switch the thinking rows off.
    assert "considering" in rendered
    assert "on it" in rendered
