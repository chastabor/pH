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
from ph.testing import stored_log, user_payload, write_reference_fork

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

    path = stored_log(tmp_path / "sessions", "s")
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

    header, events = read_session(stored_log(tmp_path / "sessions", "s"))
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

    assert len(stored_log(tmp_path / "sessions", "s").read_text().splitlines()) == 3


async def test_a_forked_session_stores_a_reference_not_a_copy(mount: Any, tmp_path: Path) -> None:
    """**Step 4.** The prefix stays in the parent's file; the child stores a pointer.

    On disk the child begins at `session/end-seed`, stamped at seq
    `seed_length` — which both signals that the file owes a prefix and measures
    how much. What a *reader* gets back is byte-identical to what the copy
    produced, and that is the whole trade: one copy of the events, the same log.

    `seed_length` keeps its old meaning. It is the **provenance** boundary that
    five folds read as "where the parent's history ends and this session's own
    work starts" — goals spend, schedule, both inboxes, daemon recovery — and
    repurposing it as a storage offset would have quietly changed all five.
    """
    ctx = await mount(_root(tmp_path))
    parent = ctx.sessions.create("parent")
    parent.append("turn/start", {"turn": 1})
    parent.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await ctx.sessions.flush(parent)

    child = ctx.sessions.fork(parent, None, "child")
    await ctx.sessions.flush(child)

    header, own = read_session(stored_log(tmp_path / "sessions", "child", family="parent"))
    assert header.parent_session == "parent"
    assert header.seed_length == 2
    assert [event.type for event in own] == ["session/end-seed"], "the seed was not re-written"
    assert own[0].seq == 2, "and the first seq says how much it owes"

    assert [event.type for event in child.events] == [
        "turn/start",
        "turn/end",
        "session/end-seed",
    ], "in memory the child is a whole session, sharing the parent's immutable events"

    _, whole = ctx.session_persistence.read("child")
    assert [event.type for event in whole] == [event.type for event in child.events]
    assert [event.seq for event in whole] == [0, 1, 2]


async def test_a_child_is_never_durable_before_the_prefix_it_references(
    mount: Any, tmp_path: Path
) -> None:
    """**Write ordering is the one thing copying used to give for free.**

    A copied child was self-sufficient the moment it hit disk. A child that
    stores a *reference* is only readable once the log it points at holds the
    events it names — so a crash between the child's flush and the parent's next
    one would leave an unreadable child, and the fork boundary is very often the
    parent's live tip, which is exactly the part not yet written.

    Nothing in the caller can order this: `fork` is synchronous and the flushes
    are independent. So the rule lives where both backends are wired, and it is
    the plain one — a log is flushed after everything it references.
    """
    ctx = await mount(_root(tmp_path))
    parent = ctx.sessions.create("parent")
    parent.append("turn/start", {"turn": 1})
    parent.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    # The parent has never been flushed. Its file does not exist.
    child = ctx.sessions.fork(parent, None, "child")
    await ctx.sessions.flush(child)

    assert stored_log(tmp_path / "sessions", "parent").exists(), "the ancestor went first"
    _, whole = ctx.session_persistence.read("child")
    assert [event.seq for event in whole] == [0, 1, 2]


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


async def test_a_broken_lineage_is_reported_by_ph_doctor(mount: Any, tmp_path: Path) -> None:
    """**Step 5, where it can actually be acted on.**

    The plan's guard was "refuse to remove a session that has descendants", but
    nothing in pH removes a session log — so there is no removal to refuse, and
    writing the guard anyway would be a check with no caller. What exists is a
    person with `rm`, and what they get today is a `LineageError` at resume, one
    session at a time, long after the fact.

    So the store answers the question the other way round, and says it where a
    person goes to ask what is wrong. The section is absent while the store is
    healthy: `Diagnostic.read`'s empty-list contract, for the reason it gives —
    a report that shows every section every time is one where the section that
    matters cannot be found.
    """
    ctx = await mount(_root(tmp_path))
    registry = ctx.get("diagnostics")
    assert registry is not None, "the base profile mounts it"
    assert "Session lineage" not in dict(registry.report()), "nothing to say yet"

    root = tmp_path / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    write_reference_fork(root, "orphan", "deleted-parent", boundary=4)

    assert dict(registry.report())["Session lineage"] == [
        ("orphan", "ancestor deleted-parent is missing")
    ]


async def test_segments_each_hold_only_their_own_run(mount: Any, tmp_path: Path) -> None:
    """**Segmentation on disk: three files, one contiguous log (§7 step 6).**

    Each file's events are disjoint from its neighbours' and they tile exactly,
    which is the same property forking already relies on — a segment *is* a fork
    at the tip, so nothing here is a second mechanism. Reading the newest one
    walks the chain and hands back the whole run.
    """
    ctx = await mount(_root(tmp_path))
    first = ctx.sessions.create("s0")
    for turn in (1, 2):
        first.append("turn/start", {"turn": turn})
        first.append("turn/end", {"turn": turn, "reason": {"kind": "completed"}})

    second = ctx.sessions.roll(first, "s1")
    for turn in (3, 4):
        second.append("turn/start", {"turn": turn})
        second.append("turn/end", {"turn": turn, "reason": {"kind": "completed"}})

    third = ctx.sessions.roll(second, "s2")
    third.append("turn/start", {"turn": 5})
    third.append("turn/end", {"turn": 5, "reason": {"kind": "completed"}})
    await ctx.sessions.flush(third)

    root = tmp_path / "sessions"
    held = {
        name: [event.seq for event in read_session(stored_log(root, name, family="s0"))[1]]
        for name in ("s0", "s1", "s2")
    }
    assert held == {"s0": [0, 1, 2, 3, 4], "s1": [4, 5, 6, 7, 8, 9], "s2": [9, 10, 11]}, (
        "the shared seqs are the parent's marker against the child's end-seed: "
        "different events in different lineages, never both in one materialised log"
    )

    _, whole = ctx.session_persistence.read("s2")
    assert [event.seq for event in whole] == list(range(12))
    assert [event.type for event in whole].count("session/segmented") == 0, (
        "a marker belongs to the log that stopped, not to the one that carried on"
    )
