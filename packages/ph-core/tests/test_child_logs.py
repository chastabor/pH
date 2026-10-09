"""Phase 11 — each session owns its log: the seam reads and writes a child's own log.

A sub-agent's admission, its starts, its ending and its tombstone are in the
sub-agent's log, and its parent's log holds none of them. So everything the seam
decides about a child — whether to let it run, whether to put it back after a
restart, which provider owns it, whether it may be revoked — is read from and
written to that one log. These pin the seam's half; ph-rlm's tests pin the
provider's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.json import JsonValue, as_obj, as_seq, as_str
from ph.keys import AGENTS, SESSION_PERSISTENCE, SESSIONS, SUBAGENTS
from ph.seams.subagents import (
    ADMITTED,
    DELETED,
    PARENT_TEARDOWN,
    STATUS,
    ChildNotice,
    SubagentRequest,
    SubagentRun,
    SubagentSpawnError,
    exhausted_detail,
    record_deleted,
    record_ended,
)
from ph.session import Session, SessionEvent, SessionHeader, SurfaceIntent, session_written
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    StubSubagentProvider,
    admitted_child,
    assistant_payload,
    log_event,
    not_none,
    raising,
    stored_events,
    stored_types,
    write_skill,
)

pytestmark = pytest.mark.anyio


def _parent(ctx: Context) -> AgentDriver:
    return ctx.require(AGENTS).create(ctx.require(SESSIONS).create("lead"), FAKE_OPTIONS)


@dataclass(slots=True)
class _Readmitting:
    """A stub that says which children it was asked to take back, and declines them."""

    inner: StubSubagentProvider
    asked: list[str] = field(default_factory=list)

    async def start(self, request: SubagentRequest) -> SubagentRun:
        return await self.inner.start(request)

    async def readmit(
        self, request: SubagentRequest, *, run_id: str, session_id: str, restarts: int = 0
    ) -> SubagentRun | None:
        self.asked.append(run_id)
        return None


async def _from_an_earlier_process(
    ctx: Context, parent: Session, *records: tuple[str, dict[str, JsonValue]], owner: str = "stub"
) -> str:
    """A child admitted by `owner`, its log on disk and nothing about it in memory —
    what the next process finds. Returns its session id."""
    child = admitted_child(
        ctx,
        parent,
        "r1",
        {"name": "scout", "owner": owner, "prompt": "look", "modelProvider": "fake"},
    )
    for kind, data in records:
        log_event(child, kind, data)
    await _as_an_earlier_process_left_them(ctx, parent, child)
    return child.id


async def _as_an_earlier_process_left_them(ctx: Context, parent: Session, *logs: Session) -> None:
    """Each of `logs` on disk and nothing about them in memory — what the next process
    finds."""
    for log in logs:
        assert await session_written(ctx, log)
        ctx.require(SESSIONS).dispose(log.id)
    ctx.require(SUBAGENTS).forget_session(parent.id)


async def test_a_child_is_on_its_own_disk_before_its_gate_opens(mount: MountProfile) -> None:
    """S2, in the child's own log. The admission is the record a restart finds a child
    by, and it is on the child's disk — with the provider that owns it and the call
    that asked for it — by the time the child may take a step; its parent's log holds
    nothing about it.

    Sabotage: flush the parent rather than the child in `_admit`, and the child's store
    is empty when its gate opens.
    """
    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("stub", StubSubagentProvider(root=ctx))
    parent = _parent(ctx)
    assert parent.session is not None

    run = await ctx.require(SUBAGENTS).start(
        "stub", SubagentRequest(prompt="look", parent=parent, call_id="c1")
    )

    assert run.ready.is_set()
    assert stored_types(ctx, run.session_id) == [ADMITTED]
    admitted = stored_events(ctx, run.session_id)[0].data
    assert (admitted["owner"], admitted["callId"], admitted["prompt"]) == ("stub", "c1", "look")
    assert "sessionId" not in admitted and "parentId" not in admitted
    assert not [one for one in parent.session.events if one.type.startswith("subagent/")]


@dataclass(slots=True)
class _Logless:
    """A provider that opens no log for its child, which `_admit` refuses."""

    inner: StubSubagentProvider

    async def start(self, request: SubagentRequest) -> SubagentRun:
        run = await self.inner.start(request)
        run.session_id = "nowhere"
        return run


async def test_a_provider_that_opened_no_log_for_its_child_is_refused(
    mount: MountProfile,
) -> None:
    """A child's own log is its only record, so a child without one is one nothing
    could bring back after a restart — refused, as a child whose admission could not
    be written is.

    Sabotage: skip the check in `_admit`, and the spawn succeeds with no record.
    """
    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("logless", _Logless(StubSubagentProvider(root=ctx)))

    with pytest.raises(SubagentSpawnError, match="opened no log"):
        await ctx.require(SUBAGENTS).start(
            "logless", SubagentRequest(prompt="look", parent=_parent(ctx))
        )


async def test_a_refused_spawn_records_no_skill_it_would_have_briefed(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A `skill/read` says a body reached an agent. The brief was read, and recorded,
    as the grant was built — before the provider started the child or `_admit` could
    refuse it — so a refused spawn left a read of a prompt no child ever had.

    Sabotage: record the brief as it is read again, and the refused spawn leaves one.
    """
    write_skill(tmp_path, "sort", body="Sort by kind.")
    ctx = await mount({"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}})
    ctx.require(SUBAGENTS).register_provider("logless", _Logless(StubSubagentProvider(root=ctx)))
    parent = _parent(ctx)

    with pytest.raises(SubagentSpawnError, match="opened no log"):
        await ctx.require(SUBAGENTS).start(
            "logless", SubagentRequest(prompt="look", parent=parent, skills=("sort",))
        )

    assert list(not_none(parent.session).select("skill/read")) == []


async def test_a_spent_child_is_ended_in_its_own_log(mount: MountProfile) -> None:
    """The ladder, read and written in the child's own log. A child restarted
    `retry_limit` times without an answer between is ended — in its log, on its
    disk — rather than started again, and nothing is written to its parent's.

    Sabotage: skip `_end_child` for a spent child, and it is readmitted forever.
    """
    ctx = await mount()
    provider = _Readmitting(StubSubagentProvider(root=ctx))
    ctx.require(SUBAGENTS).register_provider("stub", provider)
    parent = _parent(ctx)
    assert parent.session is not None
    resumed: dict[str, JsonValue] = {"status": "running", "cause": "resumed"}
    child_id = await _from_an_earlier_process(ctx, parent.session, *[(STATUS, resumed)] * 3)

    revived = await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=3)

    assert (revived, provider.asked) == ([], [])
    state = ctx.require(SUBAGENTS).children(parent.session.id)["r1"]
    assert (state.status, state.detail) == ("error", exhausted_detail(3))
    last = stored_events(ctx, child_id)[-1]
    assert (last.type, last.data["status"]) == (STATUS, "error")
    assert not [one for one in parent.session.events if one.type.startswith("subagent/")]


async def test_readmission_finds_the_provider_that_admitted_the_child(
    mount: MountProfile,
) -> None:
    """Defect 1 of the plan. A child's provider was never recorded, so a readmission
    resolved "whichever one is mounted" and, with two, none. The admission names its
    `owner` now, and the sweep asks that one.

    Sabotage: resolve the readmitter without the recorded owner, and neither is asked.
    """
    ctx = await mount()
    first = _Readmitting(StubSubagentProvider(root=ctx))
    second = _Readmitting(StubSubagentProvider(root=ctx))
    ctx.require(SUBAGENTS).register_provider("first", first)
    ctx.require(SUBAGENTS).register_provider("second", second)
    parent = _parent(ctx)
    assert parent.session is not None
    await _from_an_earlier_process(ctx, parent.session, owner="second")

    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=3)

    assert (first.asked, second.asked) == ([], ["r1"])


async def test_a_child_settled_by_an_earlier_process_can_be_deleted(mount: MountProfile) -> None:
    """Defect 5 of the plan. A revocation reached only children this process held in
    memory, so a child settled before a restart could never be revoked. The seam
    tombstones it in its own stored log — and, since it had ended, does not cancel it.

    Sabotage: answer `False` for a child no provider holds, and nothing is written.
    """
    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("stub", StubSubagentProvider(root=ctx))
    parent = _parent(ctx)
    assert parent.session is not None
    child_id = await _from_an_earlier_process(
        ctx, parent.session, (STATUS, {"status": "done", "answerPreview": "42"})
    )

    deleted = await ctx.require(SUBAGENTS).delete(parent.session, "r1", reason="user")

    assert deleted
    assert stored_types(ctx, child_id)[-1] == DELETED
    state = ctx.require(SUBAGENTS).children(parent.session.id)["r1"]
    assert (state.deleted, state.status) == (True, "done")
    assert not await ctx.require(SUBAGENTS).delete(parent.session, "r1", reason="user")


async def test_a_stored_child_is_read_once_per_mount(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child this process is not running changes only when this process opens it,
    so its stored log is read once per mount — not per prompt, per cap check, per
    panel refresh.

    Sabotage: drop the `_stored` check in `load_children`, and it is read twice.
    """
    ctx = await mount()
    parent = _parent(ctx)
    assert parent.session is not None
    child_id = await _from_an_earlier_process(ctx, parent.session)
    store = ctx.require(SESSION_PERSISTENCE)
    reads: list[str] = []
    read_own = type(store).read_own

    def counted(
        self: object,
        session_id: str,
        upto: int | None = None,
        family: str | None = None,
        *,
        types: frozenset[str] | None = None,
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        reads.append(session_id)
        return read_own(self, session_id, upto, family, types=types)  # type: ignore[arg-type]

    monkeypatch.setattr(type(store), "read_own", counted)
    subagents = ctx.require(SUBAGENTS)

    first = await subagents.load_children(parent.session.id, parent.session.header.family)
    second = await subagents.load_children(parent.session.id, parent.session.header.family)

    assert reads == [child_id]
    assert first == second and list(first) == ["r1"]


async def test_a_childs_own_children_are_read_with_its_root(mount: MountProfile) -> None:
    """**The whole tree, in one read.** A restart used to read a parent's children one
    level at a time, and only beneath children it readmitted — so a grandchild beneath
    a child that had ended was never read: its spend reached no goal, and nothing
    listed it. Every descendant is filed in its root's family under its root's id, so
    the one read that finds the children finds all of theirs.

    Sabotage: file only the parent's direct children in `_stored_tree`, and the
    grandchild — and what it spent — is missing.
    """
    ctx = await mount()
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1")
    log_event(child, STATUS, {"status": "done"})
    grandchild = admitted_child(ctx, child, "r2")
    log_event(
        grandchild,
        "assistant/message",
        {**assistant_payload("found it", "m1"), "usage": {"inputTokens": 5, "outputTokens": 2}},
        SurfaceIntent("append", ()),
    )
    log_event(grandchild, STATUS, {"status": "done"})
    await _as_an_earlier_process_left_them(ctx, parent.session, child, grandchild)
    subagents = ctx.require(SUBAGENTS)

    await subagents.load_children(parent.session.id, parent.session.header.family)

    assert list(subagents.children(child.id)) == ["r2"], "the grandchild was not read"
    assert subagents.delegated_tokens(parent.session.id) == 7


@pytest.mark.parametrize("ending", ["ended-before-the-crash", "given-up-by-the-sweep"])
async def test_what_an_ended_child_left_unfinished_is_revoked(
    mount: MountProfile, ending: str
) -> None:
    """**An ended child's own children end with it** — in their own logs, on a restart.

    A child's children are artifacts of its scope, revoked (`PARENT_TEARDOWN`) as it
    unwinds. A crash between a child's ending and theirs left a grandchild `running`
    on disk beneath a child that had ended, and nothing would ever readmit it: its
    root read as working for good. The sweep finishes that teardown — for a child that
    ended before the crash, and for one it gives up on itself.

    Sabotage: skip `_revoke_beneath`, and the grandchild is still `running`.
    """
    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("stub", _Readmitting(StubSubagentProvider(root=ctx)))
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1", {"prompt": "look"})
    if ending == "ended-before-the-crash":
        log_event(child, STATUS, {"status": "done"})
    else:
        for _ in range(3):
            log_event(child, STATUS, {"status": "running", "cause": "resumed"})
    grandchild = admitted_child(ctx, child, "r2", {"prompt": "dig"})
    log_event(grandchild, STATUS, {"status": "running"})
    await _as_an_earlier_process_left_them(ctx, parent.session, child, grandchild)

    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=3)

    revoked = ctx.require(SUBAGENTS).state(grandchild.id)
    assert revoked is not None
    assert (revoked.status, revoked.deleted, revoked.deleted_reason) == (
        "canceled",
        True,
        PARENT_TEARDOWN,
    )
    assert stored_types(ctx, grandchild.id)[-2:] == [STATUS, DELETED]


@pytest.mark.parametrize("ended_children", [1, 2], ids=["one-ended-child", "two-ended-children"])
async def test_ended_childrens_unfinished_children_are_revoked_together(
    mount: MountProfile, ended_children: int
) -> None:
    """Each descendant's log is opened through `stored_session`, which settles what
    reading it owes before the tombstone is written — a spill sweep, and a `git`
    reconcile per leaked tree. Walked one log at a time, or one ended child's subtree
    at a time, a restart paid that back to back; nothing orders one tombstone against
    another, so every subtree's are written together.

    Asserted as a rendezvous: each load waits for the others to begin. Sabotage: walk
    the logs one at a time — or revoke each ended child's family as the sweep reaches
    it — and the first load never meets the second.
    """
    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("stub", _Readmitting(StubSubagentProvider(root=ctx)))
    parent = _parent(ctx)
    assert parent.session is not None
    ended = [
        admitted_child(ctx, parent.session, f"r{index}", {"prompt": "look"})
        for index in range(ended_children)
    ]
    for child in ended:
        log_event(child, STATUS, {"status": "done"})
    beneath = [
        admitted_child(ctx, ended[index % ended_children], f"d{index}", {"prompt": "dig"})
        for index in range(2)
    ]
    for one in beneath:
        log_event(one, STATUS, {"status": "running"})
    await _as_an_earlier_process_left_them(ctx, parent.session, *ended, *beneath)
    waiting = {one.id for one in beneath}
    entered: set[str] = set()
    everyone = anyio.Event()
    met: list[bool] = []

    async def rendezvous(session: Session) -> None:
        if session.id not in waiting:
            return
        entered.add(session.id)
        if entered == waiting:
            everyone.set()
        with anyio.move_on_after(5):
            await everyone.wait()
        met.append(everyone.is_set())

    ctx.on("session/loaded", rendezvous)

    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=3)

    assert met == [True, True], "the descendants were revoked one log at a time"
    for one in beneath:
        assert not_none(ctx.require(SUBAGENTS).state(one.id)).deleted_reason == PARENT_TEARDOWN


async def test_a_stored_child_is_written_where_its_header_files_it(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child the sweep read from the store carries the family its own header files
    it in (`ChildState.family`), so the write that ends or revokes it opens that path.
    Looked up by id, it listed every family directory in the store — on the event
    loop, once per child the sweep wrote to.

    Sabotage: drop `family` from `_write_child`'s `stored_session`, and the store is
    searched.
    """
    from ph.persistence import jsonl

    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("stub", _Readmitting(StubSubagentProvider(root=ctx)))
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1", {"prompt": "look"})
    log_event(child, STATUS, {"status": "done"})
    grandchild = admitted_child(ctx, child, "r2", {"prompt": "dig"})
    log_event(grandchild, STATUS, {"status": "running"})
    await _as_an_earlier_process_left_them(ctx, parent.session, child, grandchild)

    searched = raising(
        AssertionError("searched the store for a child whose family its state holds")
    )
    monkeypatch.setattr(jsonl, "locate_under", searched)

    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=3)

    revoked = not_none(ctx.require(SUBAGENTS).state(grandchild.id))
    assert revoked.deleted_reason == PARENT_TEARDOWN


@pytest.mark.parametrize("door", ["ended", "deleted"])
async def test_a_child_that_ends_takes_what_it_left_unfinished_with_it(
    mount: MountProfile, door: str
) -> None:
    """**In this process too, not only on a restart**, and whichever door ended it.

    Its running children are its provider's, revoked as its scope unwinds. What
    nothing else ended was a child this process is not running — here one waiting
    beneath it, and that one's own child — which read as working for good: it held
    its root out of passivation, and a credential it waited for would have readmitted
    it under a parent that had ended. A child that finished stays finished.

    Sabotage: drop `_end_beneath` from the door, and the waiting grandchild is still
    `queued`.
    """
    ctx = await mount()
    subagents = ctx.require(SUBAGENTS)
    subagents.register_provider("stub", StubSubagentProvider(root=ctx))
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1")
    waiting = admitted_child(ctx, child, "r2")
    beneath = admitted_child(ctx, waiting, "r3")
    log_event(beneath, STATUS, {"status": "running"})
    finished = admitted_child(ctx, child, "r4")
    log_event(finished, STATUS, {"status": "done"})
    child_agent = ctx.require(AGENTS).create(child, FAKE_OPTIONS)
    running = await subagents.start("stub", SubagentRequest(prompt="go", parent=child_agent))

    if door == "ended":
        await record_ended(ctx, child, "done")
    else:
        await record_deleted(ctx, child, "not needed")

    for revoked in (waiting, beneath):
        state = subagents.state(revoked.id)
        assert state is not None
        assert (state.status, state.deleted_reason) == ("canceled", PARENT_TEARDOWN)
    left = [subagents.state(one) for one in (finished.id, running.session_id)]
    assert [(one.status, one.deleted) for one in left if one is not None] == [
        ("done", False),
        ("queued", False),
    ], "a finished child and a running one are not the cascade's"


def _notices(parent: Session, notice: ChildNotice) -> int:
    """How many times `notice` reached `parent`'s inbox, as its log records it."""
    delivered = [
        as_str(as_obj(message).get("id"))
        for event in parent.events
        if event.type == "agent/inbox/spliced"
        for message in as_seq(event.data.get("inserted"))
    ]
    return delivered.count(notice.id)


async def test_an_ending_carries_its_notice_to_the_parent(mount: MountProfile) -> None:
    """What a child's ending tells its parent is on the child's own record, on its
    disk, and then in the parent's inbox. It was an inject beside the ending, so a
    crash after the child's ending reached its disk and before the parent's did lost
    it for good: the sweep skips a child that ended.

    Sabotage: leave the notice off the ending record, and the stored child says
    nothing of it.
    """
    ctx = await mount()
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1")
    notice = ChildNotice(text="[the child finished]", summary="finished")

    await record_ended(ctx, child, "done", notice=notice)

    assert _notices(parent.session, notice) == 1
    await _as_an_earlier_process_left_them(ctx, parent.session, child)
    stored = await ctx.require(SUBAGENTS).load_children(
        parent.session.id, parent.session.header.family
    )
    assert stored["r1"].notice == notice, "on the child's disk, where a restart reads it"


async def test_a_notice_a_crash_kept_from_the_parent_reaches_it_once(mount: MountProfile) -> None:
    """A crash between the child's ending and the parent's next write left a notice
    the child's log holds and the parent's does not. The sweep that brings the parent
    back delivers it — and a second sweep, finding it in the parent's log, does not
    deliver it again.

    Sabotage: skip the notice in the sweep, and it never arrives; deliver it without
    asking the parent's log, and it arrives twice.
    """
    ctx = await mount()
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1")
    notice = ChildNotice(text="[the child finished]", summary="finished")
    log_event(child, STATUS, {"status": "done", "notice": notice.to_wire()})
    await _as_an_earlier_process_left_them(ctx, parent.session, child)
    subagents = ctx.require(SUBAGENTS)

    await subagents.resume_children(parent, retry_limit=3)
    subagents.forget_session(parent.session.id)
    await subagents.resume_children(parent, retry_limit=3)

    assert _notices(parent.session, notice) == 1


async def test_a_stored_child_that_is_deleted_takes_its_children_with_it(
    mount: MountProfile,
) -> None:
    """A delete that reaches a child on disk writes its tombstone there, and what it
    left unfinished beneath it goes too. It used to tombstone the child alone.

    Sabotage: as above, and the grandchild is still `running`.
    """
    ctx = await mount()
    parent = _parent(ctx)
    assert parent.session is not None
    child = admitted_child(ctx, parent.session, "r1", {"prompt": "look"})
    grandchild = admitted_child(ctx, child, "r2", {"prompt": "dig"})
    log_event(grandchild, STATUS, {"status": "running"})
    await _as_an_earlier_process_left_them(ctx, parent.session, child, grandchild)

    assert await ctx.require(SUBAGENTS).delete(parent.session, "r1", reason="not needed")

    assert stored_types(ctx, grandchild.id)[-2:] == [STATUS, DELETED]
    revoked = ctx.require(SUBAGENTS).state(grandchild.id)
    assert revoked is not None and revoked.deleted_reason == PARENT_TEARDOWN
