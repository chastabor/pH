"""P0-15 — JSONL persistence and the flush barrier.

Gates: *flush drains; `append` never awaits I/O.*

The second is a property of the whole design, not a micro-optimisation: if
`append` could block, every listener on the post-commit feed would be running
behind disk latency, and the checkpoint policy would have nothing left to
decide.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from ph.persistence.jsonl import read_session
from ph.session import Session, SessionEvent, SurfaceIntent
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import user_payload

pytestmark = pytest.mark.anyio


def _root(tmp_path: Path) -> dict[str, Any]:
    return {"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}}


def test_append_is_synchronous_and_io_free() -> None:
    # A coroutine here would put disk latency in front of every observer.
    assert not inspect.iscoroutinefunction(Session.append)
    assert "await" not in inspect.getsource(Session.append)


async def test_flush_writes_a_header_line_and_one_line_per_event(
    mount: Any, tmp_path: Path
) -> None:
    ctx = await mount(_root(tmp_path))
    session = ctx.sessions.create("s")
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    path = tmp_path / "sessions" / "s.jsonl"
    assert not path.exists(), "append must not touch the disk"

    await ctx.sessions.flush(session)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith('{"type":"session/header"')


async def test_a_stored_session_reads_back_identically(mount: Any, tmp_path: Path) -> None:
    ctx = await mount(_root(tmp_path))
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, FAKE).prompt("hello")
    await ctx.sessions.flush(session)

    header, events = read_session(tmp_path / "sessions" / "s.jsonl")
    assert header.id == "s"
    assert [event.to_wire() for event in events] == [event.to_wire() for event in session.events]
    # And it re-derives to the same messages, which is what "resume" means.
    assert Session("s2", seed=events).derive_messages() == session.derive_messages()


async def test_flush_is_idempotent_and_appends_only_new_events(mount: Any, tmp_path: Path) -> None:
    ctx = await mount(_root(tmp_path))
    session = ctx.sessions.create("s")
    session.append("turn/start", {"turn": 1})
    await ctx.sessions.flush(session)
    await ctx.sessions.flush(session)
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await ctx.sessions.flush(session)

    assert len((tmp_path / "sessions" / "s.jsonl").read_text().splitlines()) == 3


async def test_a_forked_session_stores_its_seed(mount: Any, tmp_path: Path) -> None:
    ctx = await mount(_root(tmp_path))
    parent = ctx.sessions.create("parent")
    parent.append("turn/start", {"turn": 1})
    parent.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await ctx.sessions.flush(parent)

    child = ctx.sessions.fork(parent, None, "child")
    await ctx.sessions.flush(child)
    header, events = read_session(tmp_path / "sessions" / "child.jsonl")
    assert header.parent_session == "parent"
    assert header.seed_length == 2
    assert [event.type for event in events] == ["turn/start", "turn/end", "session/end-seed"]


def test_read_session_hands_acceptance_to_the_session(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    path.write_text(
        '{"type":"session/header","header":{"version":0,"id":"f","createdAt":1}}\n'
        '{"type":"quantum/entangle","seq":0,"time":1,"data":{}}\n',
        encoding="utf-8",
    )
    # The reader validates envelopes and returns; the known-types refusal is the
    # Session's, so every seed path — not just this backend — applies it.
    header, events = read_session(path)
    assert [event.type for event in events] == ["quantum/entangle"]
    with pytest.raises(ValueError, match="unrecognized required type"):
        Session("f", seed=events, header=header)


def test_a_log_with_no_header_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "headerless.jsonl"
    path.write_text('{"type":"turn/start","seq":0,"time":1,"data":{"turn":1}}\n')
    with pytest.raises(ValueError, match="no session header"):
        read_session(path)


def test_a_wrong_format_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v9.jsonl"
    path.write_text('{"type":"session/header","header":{"version":9,"id":"f","createdAt":1}}\n')
    with pytest.raises(ValueError, match="version must be 0"):
        read_session(path)


async def test_the_checkpoint_policy_flushes_once_before_each_request(
    mount: Any, tmp_path: Path
) -> None:
    """Barrier 1 (A4). One fsync per step, not two: the "step end" barrier on
    the request path *is* this one, since the next request's flush covers
    everything the previous step committed."""
    ctx = await mount(_root(tmp_path))
    session = ctx.sessions.create("s")
    written: list[int] = []
    # `session/flush` is a parallel dispatch, so an extra listener observes the
    # barriers without displacing the backend that actually writes.
    ctx.on("session/flush", lambda target: written.append(len(target.events)))

    await ctx.agents.create(session, FAKE).prompt("hello")
    assert len(written) == 1, f"expected exactly one barrier on a tool-less step, saw {written}"
    # By the time the model request goes out, the message that motivated it and
    # the header it was built under are both durable.
    durable = [event.type for event in session.events[: written[0]]]
    assert "user/message" in durable
    assert "request/header" in durable


async def test_a_rejected_step_still_reaches_disk(mount: Any, tmp_path: Path) -> None:
    """The one step end barrier 1 never reaches: no request follows a reject."""
    from ph.agent.types import PreStepDecision

    ctx = await mount(_root(tmp_path))
    session = ctx.sessions.create("s")
    written: list[int] = []
    ctx.on("session/flush", lambda target: written.append(len(target.events)))
    ctx.on("agent/pre-step", lambda request, next_: PreStepDecision(kind="reject"))

    await ctx.agents.create(session, FAKE).prompt("hello")
    assert written, "a rejected step was never flushed"


async def test_a_top_level_tool_body_is_preceded_by_a_barrier(mount: Any, tmp_path: Path) -> None:
    """Barrier 2: the `tool/call` is durable before the side effect happens."""
    from ph.testing import simple_tool

    ctx = await mount(_root(tmp_path))
    flushed_before_body: list[bool] = []

    def body(_args: Any, run: Any) -> str:
        durable = ctx.session_persistence._buffers[run.session.id].pending
        flushed_before_body.append(not durable)
        return "ok"

    ctx.tools.register(simple_tool("touch", body))
    session = ctx.sessions.create("s")
    run = ctx.tools.create_execution(
        __import__("ph.tools", fromlist=["ToolExecutionInput"]).ToolExecutionInput(
            call_id="c", name="touch", arguments={}, scope=ctx, session=session
        )
    )
    session.append("turn/start", {"turn": 1})
    await ctx.tools.dispatch(run)
    # Nothing pending when the body ran: the barrier drained the buffer first.
    assert flushed_before_body == [True]


def test_events_survive_a_wire_round_trip() -> None:
    session = Session("s")
    session.append("user/message", user_payload("hi"), SurfaceIntent("append"))
    for event in session.events:
        assert SessionEvent.from_wire(event.to_wire()).to_wire() == event.to_wire()
