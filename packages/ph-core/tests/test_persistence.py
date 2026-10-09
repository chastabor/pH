"""P0-15 — JSONL persistence and the flush barrier.

Gates: *flush drains; `append` never awaits I/O.*

The second is a property of the whole design, not a micro-optimization: if
`append` could block, every listener on the post-commit feed would be running
behind disk latency, and the checkpoint policy would have nothing left to
decide.

## Why `SessionPersistence` had to be declared before a second backend existed

It was named throughout the plans since D5 and declared nowhere, so every consumer
reached for the JSONL implementation's shape instead. **Four of them read
`store.root` and rebuilt a filename with `session_path`** — the print mode's
`log_path`, the TUI's session picker, the daemon's resume check and its I-5 lease
— which is a JSONL fact four callers deep, and the reason a second backend could
not be added without breaking all four.

That is what `locate` exists to replace, and why it is allowed to answer `None`.

## Why the passivation sweeper uses `last_event` and not `events[-1]`

`events[-1]` materializes a snapshot of the entire log to read one element —
**4 MB and 4.7 ms at 500 000 events** — and the sweeper asks it of every root on
every pass.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

import pytest

from ph.keys import AGENTS, SESSION_PERSISTENCE, SESSIONS, TOOLS
from ph.locks import LockBusy, acquire_file_lock
from ph.persistence.jsonl import (
    _TYPED,
    JsonlSessionStore,
    append_records,
    read_records,
    read_session,
    skim_session,
)
from ph.persistence.lease import lease_path
from ph.session import (
    SESSION_FORMAT_VERSION,
    BatchRef,
    Session,
    SessionEvent,
    SessionHeader,
    SurfaceIntent,
)
from ph.session.writers import LogWriter
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import (
    MountProfile,
    disk_fills_midway,
    log_event,
    stored_log,
    user_payload,
    write_reference_fork,
)
from ph.tools import ToolRunContext

pytestmark = pytest.mark.anyio


def _root(tmp_path: Path) -> dict[str, Any]:
    return {"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}}


def _caught_up(ctx: Any, session: Session | None) -> bool:  # noqa: ANN401
    """Whether the mounted JSONL store holds everything up to the log's end.

    The progress table is that backend's own, not part of the `SessionPersistence`
    Protocol, so the narrowing says which backend the test mounted.
    """
    store = ctx.require(SESSION_PERSISTENCE)
    assert isinstance(store, JsonlSessionStore)
    assert session is not None
    return store._progress[session.id].cursor == session.seq


def test_append_is_synchronous_and_io_free() -> None:
    # A coroutine here would put disk latency in front of every observer.
    assert not inspect.iscoroutinefunction(Session._append)
    assert "await" not in inspect.getsource(Session._append)
    assert not inspect.iscoroutinefunction(LogWriter.append), "nor the door to it (T6)"


async def test_flush_writes_a_header_line_and_one_line_per_event(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    path = stored_log(tmp_path / "sessions", "s")
    assert not path.exists(), "append must not touch the disk"

    await ctx.require(SESSIONS).flush(session)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith('{"type":"session/header"')
    # `skim_session` reads a type off a line's text, and turns into a strict read
    # without a word if this ever stops holding.
    assert all(line.startswith(_TYPED) for line in lines)


async def test_a_stored_session_reads_back_identically(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")
    await ctx.require(SESSIONS).flush(session)

    header, events = read_session(stored_log(tmp_path / "sessions", "s"))
    assert header.id == "s"
    assert [event.to_wire() for event in events] == [event.to_wire() for event in session.events]
    # And it re-derives to the same messages, which is what "resume" means.
    assert Session("s2", seed=events).derive_messages() == session.derive_messages()


async def test_flush_is_idempotent_and_appends_only_new_events(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    log_event(session, "turn/start", {"turn": 1})
    await ctx.require(SESSIONS).flush(session)
    await ctx.require(SESSIONS).flush(session)
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await ctx.require(SESSIONS).flush(session)

    assert len(stored_log(tmp_path / "sessions", "s").read_text().splitlines()) == 3


async def test_a_forked_session_stores_a_reference_not_a_copy(
    mount: MountProfile, tmp_path: Path
) -> None:
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
    parent = ctx.require(SESSIONS).create("parent")
    log_event(parent, "turn/start", {"turn": 1})
    log_event(parent, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await ctx.require(SESSIONS).flush(parent)

    child = ctx.require(SESSIONS).fork(parent, None, "child")
    await ctx.require(SESSIONS).flush(child)

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

    _, whole = ctx.require(SESSION_PERSISTENCE).read("child")
    assert [event.type for event in whole] == [event.type for event in child.events]
    assert [event.seq for event in whole] == [0, 1, 2]


async def test_a_child_is_never_durable_before_the_prefix_it_references(
    mount: MountProfile, tmp_path: Path
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
    parent = ctx.require(SESSIONS).create("parent")
    log_event(parent, "turn/start", {"turn": 1})
    log_event(parent, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    # The parent has never been flushed. Its file does not exist.
    child = ctx.require(SESSIONS).fork(parent, None, "child")
    await ctx.require(SESSIONS).flush(child)

    assert stored_log(tmp_path / "sessions", "parent").exists(), "the ancestor went first"
    _, whole = ctx.require(SESSION_PERSISTENCE).read("child")
    assert [event.seq for event in whole] == [0, 1, 2]


async def test_flushing_a_subagent_child_does_not_write_its_parent(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P10-04. The ordering rule above is for a child that *references* a prefix.

    A subagent child names its parent and inherits nothing, so its file reads back
    on its own — and writing the parent first on every child flush was one parent
    fsync per child tool call that reaches past the tree. The parent is still
    written: by its own flushes, and by the mount's last write.
    """
    ctx = await mount(_root(tmp_path))
    sessions = ctx.require(SESSIONS)
    parent = sessions.create("parent")
    log_event(parent, "turn/start", {"turn": 1})
    child = sessions.create("child", meta={"parentSession": "parent", "origin": "subagent"})
    log_event(child, "turn/start", {"turn": 1})

    assert sessions.lineage(child) == (child,)
    await sessions.flush(child)

    assert not stored_log(tmp_path / "sessions", "parent").exists()
    header, whole = ctx.require(SESSION_PERSISTENCE).read("child")
    assert header is not None and header.parent_session == "parent"
    assert [event.type for event in whole] == ["turn/start"]


async def test_a_segment_still_writes_its_parent_first(mount: MountProfile, tmp_path: Path) -> None:
    """The other child that references a prefix: `roll` is a fork at the tip."""
    ctx = await mount(_root(tmp_path))
    sessions = ctx.require(SESSIONS)
    parent = sessions.create("parent")
    log_event(parent, "turn/start", {"turn": 1})
    log_event(parent, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})

    child = sessions.roll(parent, "child")
    assert sessions.lineage(child) == (parent, child)
    await sessions.flush(child)

    assert stored_log(tmp_path / "sessions", "parent").exists(), "the ancestor went first"
    _, whole = ctx.require(SESSION_PERSISTENCE).read("child")
    assert [event.type for event in whole][:2] == ["turn/start", "turn/end"]


def test_read_session_hands_acceptance_to_the_session(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    path.write_text(
        f'{{"type":"session/header","header":{{"version":{SESSION_FORMAT_VERSION},'
        '"id":"f","createdAt":1}}\n'
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


@pytest.mark.parametrize("version", [1, 9], ids=["format-1", "a-later-one"])
def test_a_wrong_format_version_is_refused(tmp_path: Path, version: int) -> None:
    """Format 1 by name (P10-15): its envelope has no batch membership, and a log
    of it is refused rather than half-understood."""
    path = tmp_path / f"v{version}.jsonl"
    path.write_text(
        f'{{"type":"session/header","header":{{"version":{version},"id":"f","createdAt":1}}}}\n'
    )
    with pytest.raises(ValueError, match=f"version must be {SESSION_FORMAT_VERSION}"):
        read_session(path)


async def test_the_checkpoint_policy_flushes_once_before_each_request(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Barrier 1 (A4). One fsync per step, not two: the "step end" barrier on
    the request path *is* this one, since the next request's flush covers
    everything the previous step committed."""
    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    written: list[int] = []
    # `session/flush` is a parallel dispatch, so an extra listener observes the
    # barriers without displacing the backend that actually writes.
    ctx.on("session/flush", lambda target: written.append(len(target.events)))

    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")
    assert len(written) == 1, f"expected exactly one barrier on a tool-less step, saw {written}"
    # By the time the model request goes out, the message that motivated it and
    # the header it was built under are both durable.
    durable = [event.type for event in session.events[: written[0]]]
    assert "user/message" in durable
    assert "request/header" in durable


@pytest.mark.parametrize("outside", [False, True], ids=["inside", "outside"])
async def test_a_rejected_step_still_reaches_disk(
    mount: MountProfile, tmp_path: Path, outside: bool
) -> None:
    """The one step end barrier 1 never reaches: no request follows a reject.

    Flushed once the decision is final (`agent/step-rejected`), whichever row made
    it. Flushed from inside `agent/pre-step`, a row outside the barrier that rejected
    without calling on — limits, mounted in another order — was never seen, and the
    batch the reject took was not flushed either.

    Sabotage: flush from inside `agent/pre-step` again, and `outside` flushes nothing.
    """
    from ph.agent.types import PreStepDecision

    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    written: list[int] = []
    ctx.on("session/flush", lambda target: written.append(len(target.events)))
    ctx.on("agent/pre-step", lambda request, next_: PreStepDecision(kind="reject"), prepend=outside)

    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")

    assert written, "a rejected step was never flushed"
    assert "agent/inbox/spliced" in [event.type for event in session.events[: written[0]]]


async def test_what_a_stream_listener_appends_goes_out_flushed(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Barrier 1 is innermost (A4). Around `llm/stream` it was only as deep as mount
    order put it: a stream listener registered after it wrapped inside it, and what
    that listener appended — media-degrade's notices, on every request — went out
    with the request unflushed. On `llm/send` it is after every one of them.

    Sabotage: flush around `llm/stream` again, and the notice is not in the flush.
    """
    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    written: list[int] = []
    ctx.on("session/flush", lambda target: written.append(len(target.events)))

    async def notice(request: Any, next_: Any) -> Any:  # noqa: ANN401
        log_event(
            session,
            "attachment/degraded",
            {"provider": "fake", "attachmentIds": ["x"], "attachments": []},
        )
        return await next_()

    ctx.on("llm/stream", notice)

    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")

    assert "attachment/degraded" in [event.type for event in session.events[: written[0]]]


@pytest.mark.parametrize("wrapped", [False, True])
async def test_a_top_level_tool_body_is_preceded_by_a_barrier(
    mount: MountProfile, tmp_path: Path, wrapped: bool
) -> None:
    """Barrier 2: the `tool/call` is durable before the side effect happens.

    And so is whatever a `tools/execute` wrapper appended, however late it was
    registered: around `tools/execute` the barrier was only as deep as mount order
    put it, and a wrapper registered after it ran inside it. On `tools/body` it is
    after every one of them.

    Sabotage: flush around `tools/execute` again, and the wrapped case finds the
    wrapper's record unwritten when the body runs.
    """
    from ph.testing import simple_tool

    ctx = await mount(_root(tmp_path))
    flushed_before_body: list[bool] = []

    def body(_args: object, run: ToolRunContext) -> str:
        flushed_before_body.append(_caught_up(ctx, run.session))
        return "ok"

    ctx.require(TOOLS).register(simple_tool("touch", body))
    session = ctx.require(SESSIONS).create("s")
    if wrapped:

        async def notice(execution: Any, next_: Any) -> Any:  # noqa: ANN401
            log_event(
                session,
                "attachment/degraded",
                {"provider": "fake", "attachmentIds": ["x"], "attachments": []},
            )
            return await next_()

        ctx.on("tools/execute", notice)
    run = ctx.require(TOOLS).create_execution(
        __import__("ph.tools", fromlist=["ToolExecutionInput"]).ToolExecutionInput(
            call_id="c", name="touch", arguments={}, scope=ctx, session=session
        )
    )
    log_event(session, "turn/start", {"turn": 1})
    await ctx.require(TOOLS).dispatch(run)
    # Nothing owed when the body ran: the barrier wrote it first.
    assert flushed_before_body == [True]


async def test_a_call_canceled_at_the_barrier_never_starts(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The last check before a body, made by the registry for every call: a
    cancellation that lands while the barrier flushes, or in any slow wrapper,
    ends the call unstarted — not only a call the barrier flushed for.

    Sabotage: drop the registry's check after `tools/body`, and the body runs.
    """
    from ph.cancel import CancelToken
    from ph.testing import simple_tool
    from ph.tools import ToolExecutionInput

    ctx = await mount(_root(tmp_path))
    ran: list[str] = []

    def body(_args: object, _run: ToolRunContext) -> str:
        ran.append("ran")
        return "ok"

    ctx.require(TOOLS).register(simple_tool("touch", body))
    session = ctx.require(SESSIONS).create("s")
    cancel = CancelToken()
    ctx.on("tools/body", lambda _execution: cancel.cancel())
    run = ctx.require(TOOLS).create_execution(
        ToolExecutionInput(
            call_id="c", name="touch", arguments={}, scope=ctx, session=session, cancel=cancel
        )
    )
    log_event(session, "turn/start", {"turn": 1})

    prepared = await ctx.require(TOOLS).dispatch(run)

    assert ran == []
    assert prepared.result is not None and prepared.result.error is not None
    assert prepared.result.error.kind == "aborted"


@pytest.mark.parametrize("restore_point", [True, False])
async def test_a_nested_dispatch_that_reaches_past_the_tree_is_preceded_by_a_barrier(
    mount: MountProfile, tmp_path: Path, restore_point: bool
) -> None:
    """Barrier 2 for a Code Mode dispatch (F4): durable before it can escape.

    One barrier per cell used to cover every dispatch inside it, and under the
    `rlm` profile every tool the model calls is one. A crash mid-cell left the
    outer `tool/call` and none of the `tool/code-dispatch-start` records, so
    `/revert`'s list of what a restore does not undo came back empty — failing
    in the one direction that list exists to prevent — and the limits folds
    under-counted the work.

    Asked of the rule `/revert` lists by (`ToolRuntime.restore_covers`): `note`
    says its effects stay in the workspace and runs unflushed, which keeps a cell
    of reads and edits at one fsync; `publish` says nothing, so it is not covered.

    **Only when the run took a restore point** (S4). Without one — no tier that can
    take it, or a capture that failed — nothing would take `note`'s edits back
    either, so it is flushed like anything else.

    Sabotage: restore the `execution.parent is not None` early return and
    `publish` runs with its record still in memory; drop `has_restore_point` from
    the skip and `note` runs unflushed with no restore point to cover it. `look`
    changes nothing (`effect_free`), so it never needs one: drop that branch, and a
    read pays a barrier on every profile whose workspace takes no restore points.
    """
    from collections.abc import Callable, Mapping

    from ph.keys import CODE_RUNTIME_STUB
    from ph.seams.workspace import CHECKPOINT
    from ph.testing import code_mode_stub, run_tool, simple_tool
    from ph.tools.registry import RUN_CODE

    ctx = await mount(_root(tmp_path), code_mode_stub())
    durable_at_body: dict[str, bool] = {}

    def body(name: str) -> Callable[[object, ToolRunContext], str]:
        def run(_args: object, run: ToolRunContext) -> str:
            durable_at_body[name] = _caught_up(ctx, run.session)
            return name

        return run

    tools = ctx.require(TOOLS)
    tools.register(simple_tool("note", body("note"), effects_confined_to_workspace=True))
    tools.register(simple_tool("look", body("look"), effect_free=True))
    tools.register(simple_tool("publish", body("publish")))

    async def program(ns: Mapping[str, object], _emit: Callable[[str], None]) -> str:
        await ns["tools"].note()  # type: ignore[attr-defined]
        await ns["tools"].look()  # type: ignore[attr-defined]
        await ns["tools"].publish()  # type: ignore[attr-defined]
        return "done"

    ctx.require(CODE_RUNTIME_STUB).register_program("cell", program)
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    if restore_point:
        # What `workspace-checkpoint` records for the run it is about to let start.
        log_event(session, CHECKPOINT, {"agentId": agent.id, "tree": "t", "callId": "call-1"})
    result = await run_tool(ctx, RUN_CODE, {"program": "cell"}, agent=agent, session=session)
    assert result.is_error is False
    assert durable_at_body == {"note": not restore_point, "look": False, "publish": True}


async def test_a_dispatch_is_asked_once_whether_a_restore_covers_it(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S4. A dispatch's record and the barrier before it both need to know whether a
    restore takes the call back — and the answer follows links, which a sibling
    dispatch can move between two asks. Asked twice, the record could say "not
    covered" of a call the barrier let run unflushed as covered. `prepare` asks once
    (`ToolExecution.restore_covered`), and both read that.

    Sabotage: ask `restore_covers` again in the barrier, and the dispatch is asked
    twice.
    """
    from collections.abc import Callable, Mapping

    from ph.keys import CODE_RUNTIME_STUB
    from ph.seams.workspace import CHECKPOINT
    from ph.testing import code_mode_stub, run_tool, simple_tool
    from ph.tools.registry import RUN_CODE, ToolRuntime

    ctx = await mount(_root(tmp_path), code_mode_stub())
    ctx.require(TOOLS).register(simple_tool("note", effects_confined_to_workspace=True))
    asked: list[str] = []
    real = ToolRuntime.restore_covers

    def counted(self: ToolRuntime, name: str, *args: Any, **kwargs: Any) -> bool:  # noqa: ANN401
        asked.append(name)
        return real(self, name, *args, **kwargs)

    monkeypatch.setattr(ToolRuntime, "restore_covers", counted)

    async def program(ns: Mapping[str, object], _emit: Callable[[str], None]) -> str:
        await ns["tools"].note()  # type: ignore[attr-defined]
        return "done"

    ctx.require(CODE_RUNTIME_STUB).register_program("cell", program)
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    log_event(session, CHECKPOINT, {"agentId": agent.id, "tree": "t", "callId": "call-1"})
    result = await run_tool(ctx, RUN_CODE, {"program": "cell"}, agent=agent, session=session)

    assert result.is_error is False
    assert asked == ["note"]
    start = session.latest("tool/code-dispatch-start")
    assert start is not None and start.data["restoreCovered"] is True


def test_events_survive_a_wire_round_trip() -> None:
    session = Session("s")
    log_event(session, "user/message", user_payload("hi"), SurfaceIntent("append"))
    for event in session.events:
        assert SessionEvent.from_wire(event.to_wire()).to_wire() == event.to_wire()


async def test_a_broken_lineage_is_reported_by_ph_doctor(
    mount: MountProfile, tmp_path: Path
) -> None:
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


async def test_segments_each_hold_only_their_own_run(mount: MountProfile, tmp_path: Path) -> None:
    """**Segmentation on disk: three files, one contiguous log (§7 step 6).**

    Each file's events are disjoint from its neighbors' and they tile exactly,
    which is the same property forking already relies on — a segment *is* a fork
    at the tip, so nothing here is a second mechanism. Reading the newest one
    walks the chain and hands back the whole run.
    """
    ctx = await mount(_root(tmp_path))
    first = ctx.require(SESSIONS).create("s0")
    for turn in (1, 2):
        log_event(first, "turn/start", {"turn": turn})
        log_event(first, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})

    second = ctx.require(SESSIONS).roll(first, "s1")
    for turn in (3, 4):
        log_event(second, "turn/start", {"turn": turn})
        log_event(second, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})

    third = ctx.require(SESSIONS).roll(second, "s2")
    log_event(third, "turn/start", {"turn": 5})
    log_event(third, "turn/end", {"turn": 5, "reason": {"kind": "completed"}})
    await ctx.require(SESSIONS).flush(third)

    root = tmp_path / "sessions"
    held = {
        name: [event.seq for event in read_session(stored_log(root, name, family="s0"))[1]]
        for name in ("s0", "s1", "s2")
    }
    assert held == {"s0": [0, 1, 2, 3, 4], "s1": [4, 5, 6, 7, 8, 9], "s2": [9, 10, 11]}, (
        "the shared seqs are the parent's marker against the child's end-seed: "
        "different events in different lineages, never both in one materialized log"
    )

    _, whole = ctx.require(SESSION_PERSISTENCE).read("s2")
    assert [event.seq for event in whole] == list(range(12))
    assert [event.type for event in whole].count("session/segmented") == 0, (
        "a marker belongs to the log that stopped, not to the one that carried on"
    )


# ------------------------------------------------------------ unfinished writes --
#
# A JSONL log has two ways to be left holding bytes no flush finished: a write
# that *fails* part-way (a disk that fills mid-payload), and a process that *dies*
# mid-write. Before these, both ended the same way — the retry, or the next
# process's first flush, appended whole lines behind a half one, and `read_session`
# refused the log at that line for good. One crash, or one full disk, and a session
# could never be opened again.


def _tracked(tmp_path: Path, session: Session | None = None) -> tuple[JsonlSessionStore, Session]:
    """A bare store tracking `session` (a fresh one by default), as `attach` would."""
    store = JsonlSessionStore(ctx=None, root=tmp_path)  # type: ignore[arg-type]
    tracked = session if session is not None else Session("s")
    store.track(tracked)
    return store, tracked


async def test_a_write_that_fails_part_way_takes_its_bytes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3. The failed write leaves the file exactly as long as it was.

    `flush` owes the records again on any failure, so the half it had written
    would otherwise sit in front of the retry's full copy. Measured before the
    fix: `read_session` → `…:3: Extra data`, a log no later write could repair.

    Sabotage: drop `_take_back` from `_append_and_sync` and the length assertion
    fails — the retry still reads back, but only because the re-measure below
    catches it, which is the second line and not the first.
    """
    from ph.persistence import jsonl

    store, session = _tracked(tmp_path)
    log_event(session, "turn/start", {"turn": 1})
    await store.flush(session)
    path = stored_log(tmp_path, "s")
    before = path.stat().st_size

    log_event(session, "step/start", {"turn": 1, "step": 1})
    log_event(session, "step/end", {"turn": 1, "step": 1})
    with monkeypatch.context() as patch:
        patch.setattr(jsonl, "write_all", disk_fills_midway)
        with pytest.raises(OSError, match="No space left"):
            await store.flush(session)
    assert path.stat().st_size == before, "the failed write left its partial bytes behind"

    await store.flush(session)
    _header, events = read_session(path)
    assert [event.seq for event in events] == [0, 1, 2]


async def test_a_new_family_directory_is_on_disk_with_its_first_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S20. A log's first flush synced the log and the family directory's entry for
    it, and never the sessions root's entry for the family directory, which a `mkdir`
    had only just made. A power cut could take the whole directory back, with a log
    flushed as durable inside it. Each directory the first write makes is synced in
    its parent now, and a second write makes none.

    Sabotage: make the directories with a plain `mkdir(parents=True)`, and neither the
    root nor its parent is synced.
    """
    from ph import paths
    from ph.persistence import jsonl

    synced: list[Path] = []
    real = paths.sync_directory

    def record(directory: Path) -> None:
        synced.append(directory)
        real(directory)

    monkeypatch.setattr(paths, "sync_directory", record)
    monkeypatch.setattr(jsonl, "sync_directory", record)
    root = tmp_path / "sessions"
    store, session = _tracked(root)
    log_event(session, "turn/start", {"turn": 1})

    await store.flush(session)
    family = stored_log(root, "s").parent
    assert synced == [tmp_path, root, family]

    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)
    assert synced == [tmp_path, root, family], "a log already made syncs no directory"


def test_a_record_log_outside_the_store_is_synced_and_takes_a_failed_write_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L4. `append_records` is `read_records`' durable writer, for a log outside the
    session store.

    The Continual Harness's global log went through a plain append: nothing synced,
    so a record every future session reads could be gone after a power loss, and a
    write that failed part-way left half a line for the next one to land behind.

    Sabotage: append through `write_text_under` instead, and nothing is synced and
    the failed write's half line stays in the file.
    """
    from ph.persistence import jsonl

    path = tmp_path / "harness" / "events.jsonl"
    real_fsync = os.fsync
    synced: list[int] = []

    def fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    append_records(path, [{"n": 1}])
    assert synced, "the record was not synced"

    before = path.stat().st_size
    with monkeypatch.context() as patch:
        patch.setattr(jsonl, "write_all", disk_fills_midway)
        with pytest.raises(OSError, match="No space left"):
            append_records(path, [{"n": 2}])
    assert path.stat().st_size == before, "the failed write left its partial bytes behind"

    append_records(path, [{"n": 3}])
    assert list(read_records(path)) == [{"n": 1}, {"n": 3}]


async def test_a_retry_behind_a_write_nobody_took_back_still_appends_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F3, the second line: a take-back that itself failed.

    `flush` stops trusting its measurement when a write fails, so the retry
    settles the fragment (`_settle_tail`) before appending behind it.

    Sabotage: remove `buffer.measured = False` from `flush`'s failure path and the
    retry glues its first record onto the half line.
    """
    from ph.persistence import jsonl

    store, session = _tracked(tmp_path)
    log_event(session, "turn/start", {"turn": 1})
    await store.flush(session)
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    with monkeypatch.context() as patch:
        patch.setattr(jsonl, "write_all", disk_fills_midway)
        patch.setattr(jsonl, "_take_back", lambda *_args: None)
        with pytest.raises(OSError):
            await store.flush(session)

    await store.flush(session)
    _header, events = read_session(stored_log(tmp_path, "s"))
    assert [event.type for event in events] == ["turn/start", "turn/end"]


def _raw_log(tmp_path: Path, session_id: str, events: list[SessionEvent], tail: str) -> Path:
    """A stored log as a death mid-write leaves it: whole lines, then `tail` as given.

    Written with the store's own encoders at the store's own path, so only `tail`
    — a torn fragment, a half-written batch — is spelled by hand."""
    from ph.json import dumps
    from ph.persistence.jsonl import HEADER_LINE_TYPE

    header = SessionHeader(id=session_id, created_at=1).to_wire()
    records = [{"type": HEADER_LINE_TYPE, "header": header}, *(e.to_wire() for e in events)]
    path = stored_log(tmp_path, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{dumps(record)}\n" for record in records) + tail, encoding="utf-8")
    return path


def _torn(tmp_path: Path, tail: str) -> Path:
    """A two-event log, then `tail` with no newline after it."""
    turn = SessionEvent(type="turn/start", seq=0, time=1, data={"turn": 1})
    step = SessionEvent(type="step/start", seq=1, time=1, data={"turn": 1, "step": 1})
    return _raw_log(tmp_path, "torn", [turn, step], tail)


def test_a_torn_final_line_is_read_without_it(tmp_path: Path) -> None:
    """F6. The fragment a crash mid-write leaves is not a reason to lose the session.

    No flush returned for those bytes, so nothing was told they exist — dropping
    them reads the log as it was the last time anything was promised about it.
    Before: `ValueError`, on every open, forever.
    """
    path = _torn(tmp_path, '{"type":"step/end","seq":2,"ti')
    _header, events = read_session(path)
    assert [event.seq for event in events] == [0, 1]


def test_a_record_that_lost_only_its_newline_is_kept(tmp_path: Path) -> None:
    """An unterminated line that parses is a whole record: an object's encoding
    ends in its closing brace, so no proper prefix of one parses. Reader and
    writer (`_settle_tail`) keep it alike, or a resume would declare it durable
    while the writer cut it from the file."""
    path = _torn(tmp_path, '{"type":"step/end","seq":2,"time":1,"data":{"turn":1,"step":1}}')
    _header, events = read_session(path)
    assert [event.seq for event in events] == [0, 1, 2]


def test_a_malformed_line_before_the_last_still_refuses(tmp_path: Path) -> None:
    """The tolerance is for the one damage an append-only log can suffer. A bad line
    with a newline after it is some other damage, and skipping it would hand the
    model a history missing its middle."""
    path = _torn(tmp_path, "")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines.insert(2, "{not a record\n")
    path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match=r"torn\.jsonl:3"):
        read_session(path)


@pytest.mark.parametrize(
    ("tail", "kept"),
    [
        ('{"type":"step/end","seq":2,"ti', [0, 1]),
        ('{"type":"step/end","seq":2,"time":1,"data":{"turn":1,"step":1}}', [0, 1, 2]),
    ],
    ids=["fragment", "finished-record"],
)
async def test_a_resumed_log_appends_behind_its_torn_tail_cleanly(
    tmp_path: Path, tail: str, kept: list[int]
) -> None:
    """F6, the writer's half: the first flush after a resume settles the tail —
    finishes a record that lost its newline, removes a fragment — before appending.

    Sabotage: drop `_settle_tail` for the bare `_read_tail` and the resumed log's
    first appended record lands on the fragment's line, which `read_session` then
    refuses as mid-file damage.
    """
    path = _torn(tmp_path, tail)
    header, events = read_session(path)
    assert [event.seq for event in events] == kept
    store, session = _tracked(
        tmp_path, Session("torn", seed=events, header=header, durable=len(events))
    )
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)

    _header, reread = read_session(path)
    assert [event.seq for event in reread] == list(range(len(session.events)))
    assert path.read_bytes().endswith(b"\n"), "a record was left without its newline"


async def test_a_log_behind_its_declared_length_is_filled_in(tmp_path: Path) -> None:
    """The measured tail decides in both directions. A file *behind* the length its
    session declares durable is missing events; the cursor stayed at the declared
    value, so the flush appended past the gap and the next resume refused the log
    ("seed must be contiguous from 0"). Now the missing events are written.

    Sabotage: take the larger of the declared value and the file's tail again, and
    the file skips seqs.
    """
    path = _torn(tmp_path, "")
    header, events = read_session(path)
    ahead = [
        *events,
        SessionEvent(type="step/end", seq=2, time=1, data={"turn": 1, "step": 1}),
        SessionEvent(
            type="turn/end", seq=3, time=1, data={"turn": 1, "reason": {"kind": "completed"}}
        ),
    ]
    store, session = _tracked(
        tmp_path, Session("torn", seed=ahead, header=header, durable=len(ahead))
    )

    await store.flush(session)

    _header, reread = read_session(path)
    assert [event.seq for event in reread] == list(range(len(session.events)))


def _batched(tmp_path: Path, members: list[str], after: str = "") -> Path:
    """A turn, then a two-member batch written line by line — `members` are the
    lines as a torn write may leave them — then `after`."""
    turn = SessionEvent(type="turn/start", seq=0, time=1, data={"turn": 1})
    return _raw_log(tmp_path, "batched", [turn], "".join(members) + after)


def _member(seq: int, *, newline: bool = True) -> str:
    from ph.json import dumps

    ref = BatchRef(first=1, count=2)
    line = dumps(
        SessionEvent(
            type="compaction/args-truncated", seq=seq, time=1, data={"n": seq}, batch=ref
        ).to_wire()
    )
    return line + ("\n" if newline else "")


@pytest.mark.parametrize(
    "members",
    [[_member(1)], [_member(1), _member(2, newline=False)[:20]]],
    ids=["second-never-written", "second-torn"],
)
def test_a_batch_cut_by_a_torn_write_is_dropped_whole(tmp_path: Path, members: list[str]) -> None:
    """P10-15. One flush wrote the whole batch; a death mid-write cut it; nothing
    was told those bytes were written — so the reader keeps none of the batch,
    rather than an accounting record whose replacement never landed.

    Sabotage: drop `_unfinished_batch` from `read_session` and the first member
    is kept alone, which the seed then refuses.
    """
    _header, events = read_session(_batched(tmp_path, members))
    assert [event.seq for event in events] == [0]


def _line(event: SessionEvent) -> str:
    from ph.json import dumps

    return dumps(event.to_wire()) + "\n"


def test_a_skimmed_read_judges_the_tail_by_the_last_line_written(tmp_path: Path) -> None:
    """`skim_session` passes a line unparsed, and the last line is still the tail: a
    batch whose final member was passed is whole, and its first member stays.

    Sabotage: leave the skimmed tail unparsed, so the tail is the first member, and
    the batch reads as cut short and loses it.
    """
    ref = BatchRef(first=1, count=2)
    kept = SessionEvent(type="turn/end", seq=1, time=1, data={"turn": 1}, batch=ref)
    passed = SessionEvent(type="step/start", seq=2, time=1, data={"turn": 1}, batch=ref)
    turn = SessionEvent(type="turn/start", seq=0, time=1, data={"turn": 1})
    path = _raw_log(tmp_path, "skim", [turn, kept, passed], "")

    _header, events = skim_session(path, frozenset({"turn/end"}))

    assert [event.seq for event in events] == [1]


def test_a_skimmed_read_passes_damage_it_has_no_use_for(tmp_path: Path) -> None:
    """The trade `skim_session` makes, and its limit: a damaged line whose own text names a
    type the reader does not want goes unseen, where a strict read refuses the log;
    a damaged line it would have kept still refuses.

    Sabotage: pass every line that begins as a written one does, and the wanted
    damage is passed too.
    """
    turn = SessionEvent(type="turn/start", seq=0, time=1, data={"turn": 1})
    end = SessionEvent(type="turn/end", seq=2, time=1, data={"turn": 1})
    unwanted = '{"type":"assistant/chunk","seq":1,"tim\n'
    path = _raw_log(tmp_path, "unwanted", [turn], unwanted + _line(end))
    wanted = frozenset({"turn/end"})

    with pytest.raises(ValueError, match="unwanted"):
        read_session(path, types=wanted)
    assert [event.seq for event in skim_session(path, wanted)[1]] == [2]

    damaged = _raw_log(tmp_path, "wanted", [turn], '{"type":"turn/end","seq":1,"tim\n')
    with pytest.raises(ValueError, match="wanted"):
        skim_session(damaged, wanted)


def test_a_complete_batch_reads_back_intact(tmp_path: Path) -> None:
    header, events = read_session(_batched(tmp_path, [_member(1), _member(2)]))
    assert [event.seq for event in events] == [0, 1, 2]
    assert {event.batch for event in events[1:]} == {BatchRef(first=1, count=2)}
    Session("batched", seed=events, header=header)


async def test_a_resumed_log_appends_behind_a_dropped_batch_cleanly(tmp_path: Path) -> None:
    """The writer's half, as for a torn line: the first flush cuts the file back
    to the batch's first member before appending, so the log it leaves reads
    back contiguous — and not with a new event behind half a batch.

    Sabotage: drop `_settle_batch` from `_settle_tail` and the new event lands
    after the dangling member, which the reread refuses.
    """
    path = _batched(tmp_path, [_member(1)])
    header, events = read_session(path)
    store, session = _tracked(
        tmp_path, Session("batched", seed=events, header=header, durable=len(events))
    )
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)

    _header, reread = read_session(path)
    assert [event.type for event in reread] == ["turn/start", "turn/end"]
    assert [event.seq for event in reread] == list(range(len(session.events)))


def test_an_unfinished_batch_before_the_end_refuses_the_log(tmp_path: Path) -> None:
    """Only a *trailing* batch can be a torn write. One cut short with more after
    it is damage of another kind, and seeding it would hand every reader half of
    something that only means anything whole."""
    after = SessionEvent(type="turn/end", seq=2, time=1, data={"turn": 1}).to_wire()
    from ph.json import dumps

    header, events = read_session(_batched(tmp_path, [_member(1)], after=dumps(after) + "\n"))
    with pytest.raises(ValueError, match="seq 2 interrupts the batch at seq 1"):
        Session("batched", seed=events, header=header)


async def test_a_final_record_longer_than_one_read_is_still_measured(tmp_path: Path) -> None:
    """B7 over a large last record. The measurement read one fixed 64 KiB window,
    so a final record longer than that was split, its fragment failed to parse,
    and the answer was "no records" — and a re-activated store wrote the whole
    log again behind itself. A large tool result is exactly such a record."""
    first, session = _tracked(tmp_path)
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "tool/call", {"callId": "c", "name": "read", "arguments": "x" * 200_000})
    await first.flush(session)

    second, _ = _tracked(tmp_path, session)  # the row re-activates: a store new to it
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await second.flush(session)

    _header, events = read_session(stored_log(tmp_path, "s"))
    assert [event.seq for event in events] == [0, 1, 2], "the log was rewritten behind itself"


# ------------------------------------------------------------ the last write --


async def test_what_a_teardown_appends_is_written_by_the_mounts_last_act(
    mount: MountProfile, tmp_path: Path
) -> None:
    """F2. A record appended while the tree unwinds reaches the file.

    Every host flushes *before* it unwinds, and unwinding is exactly when records
    are appended — the workspace seam's release closure writes
    `workspace/disposed` as an agent scope lets go. Rows unwind in reverse and the
    agent scopes hang off the `agent` row, which mounts before persistence, so
    those appends landed after the persistence row's listeners were gone: in
    memory, never on disk. A clean stop read as a crash, and `workspace-reconcile`
    reclaimed trees the clean exit had already released.

    Held by the claim, which is the contract every host already goes through: a
    store that holds a session writes every live log before letting it go
    (`write_on_unwind`). Probe: `reviews/probes/probe_teardown_flush.py`.

    Sabotage: drop `write_on_unwind` from `JsonlSessionStore.claim` and the last
    record is `turn/start`.
    """
    from ph.seams.workspace import DISPOSED
    from ph.testing import workspace_disposed

    ctx = await mount(_root(tmp_path))
    store = ctx.require(SESSION_PERSISTENCE)
    assert isinstance(store, JsonlSessionStore)
    await store.claim("s", scope=ctx)
    sessions = ctx.require(SESSIONS)
    session = sessions.create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    agent.ctx.add_disposer(
        lambda: log_event(session, *workspace_disposed(agent.id)), label="release"
    )
    log_event(session, "turn/start", {"turn": 1})
    await sessions.flush(session)  # what every host does, and all it did
    await ctx.dispose()

    _header, events = read_session(stored_log(tmp_path / "sessions", "s"))
    assert [event.type for event in events][-2:] == ["turn/start", DISPOSED]


async def test_a_scope_holding_many_leases_writes_each_log_once_while_it_is_held(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F2 for a scope that claims more than one session, as a root claims each of
    its sub-agents' logs (L2).

    Every claim registers the last write, since only the newest registration sits
    above the newest lease. So each log has to be written before any lease is given
    back, and once: each registration used to walk every live session again, which
    grew with the square of a root's children.

    Sabotage: drop the `owed` check in `write_on_unwind`, and each log is written
    once per claim. Register only at a scope's first claim, and the second log is
    written after its lease is given back.
    """
    ctx = await mount(_root(tmp_path))
    store = ctx.require(SESSION_PERSISTENCE)
    assert isinstance(store, JsonlSessionStore)
    sessions = ctx.require(SESSIONS)
    for name in ("first", "second"):
        await store.claim(name, scope=ctx)
        sessions.create(name)
    written: list[tuple[str, bool]] = []
    flush = JsonlSessionStore.flush

    async def watched(self: JsonlSessionStore, session: Session) -> None:
        try:
            acquire_file_lock(lease_path(self.root, session.id), timeout=0, what="a probe")()
        except LockBusy:
            written.append((session.id, True))
        else:
            written.append((session.id, False))
        await flush(self, session)

    monkeypatch.setattr(JsonlSessionStore, "flush", watched)
    await ctx.dispose()

    assert sorted(written) == [("first", True), ("second", True)], (
        "each log written once, while this process still held it"
    )


async def test_disposing_an_agent_writes_what_its_teardown_appended(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The mid-run half of F2: an agent let go while the mount lives on — a
    settled subagent is the common one — records its workspace's release as the
    scope unwinds, and `AgentRegistry.dispose` writes that before returning
    rather than leaving it for whenever the whole mount unwinds.

    Sabotage: drop the `session_written` call from `AgentRegistry.dispose` and
    the store does not hold the record.
    """
    from ph.seams.workspace import DISPOSED
    from ph.testing import stored_types, workspace_disposed

    ctx = await mount(_root(tmp_path))
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    agent.ctx.add_disposer(
        lambda: log_event(session, *workspace_disposed(agent.id)), label="release"
    )
    await ctx.require(AGENTS).dispose(agent.id)

    assert stored_types(ctx, "s")[-1] == DISPOSED


def test_a_log_is_found_in_its_lineage_before_every_other_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read by id alone looks in the directories whose names carry the id's root
    first. Only a bare root was found without a scan, and since format 1 a root with
    a cwd is filed as `<tag>-<id>`, so nearly every lookup `stat`ed every family.

    Sabotage: look in the families in listing order, and this probes all of them.
    """
    from ph.persistence.families import locate_under
    from ph.session import family_for

    for index in range(40):
        (tmp_path / family_for(f"other{index}", "/elsewhere")).mkdir()
    lineage = tmp_path / family_for("lead", "/work")
    lineage.mkdir()
    child = lineage / "lead-child-ab.jsonl"
    child.write_text("", encoding="utf-8")
    probed: list[Path] = []
    real = Path.is_file

    def counted(path: Path) -> bool:
        probed.append(path)
        return real(path)

    monkeypatch.setattr(Path, "is_file", counted)

    assert locate_under(tmp_path, "lead-child-ab", ".jsonl") == child
    assert len(probed) <= 2, probed


async def test_a_reopened_child_is_looked_for_where_its_parent_files_it(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child is filed in its parent's family, and `open_child_log` knows which, so
    the log a restart reopens is a `stat` there. Opened by id alone it was searched
    for, and a parent with a cwd is filed under a tagged name the child's id does not
    carry: every family directory in the store, listed once per child readmitted.

    Sabotage: drop `family` from `open_session`'s lookup, and the store is listed.
    """
    from ph.persistence import families
    from ph.seams.subagents import open_child_log
    from ph.session import child_session_id

    ctx = await mount(_root(tmp_path))
    sessions = ctx.require(SESSIONS)
    parent = sessions.create("lead", meta={"cwd": "/work"})
    child = sessions.create(child_session_id("lead", "r1"), meta={"parentSession": "lead"})
    log_event(child, "turn/start", {"turn": 0})
    assert await sessions.written(child)
    sessions.dispose(child.id)

    def listed(*_args: object) -> None:
        raise AssertionError("listed the store for a child whose family was known")

    monkeypatch.setattr(families, "family_dirs", listed)
    reopened = await open_child_log(ctx, parent, "r1")

    assert reopened.header.family == parent.header.family
    assert [event.type for event in reopened.events][: len(child.events)] == [
        event.type for event in child.events
    ]
