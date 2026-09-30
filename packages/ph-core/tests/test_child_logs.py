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

import pytest

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.json import JsonValue
from ph.keys import AGENTS, SESSION_PERSISTENCE, SESSIONS, SUBAGENTS
from ph.seams.subagents import (
    ADMITTED,
    DELETED,
    STATUS,
    SubagentRequest,
    SubagentRun,
    SubagentSpawnError,
    exhausted_detail,
)
from ph.session import Session, SessionEvent, SessionHeader, session_written
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    StubSubagentProvider,
    admitted_child,
    log_event,
    stored_events,
    stored_types,
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
    assert await session_written(ctx, child)
    ctx.require(SESSIONS).dispose(child.id)
    ctx.require(SUBAGENTS).forget_session(parent.id)
    return child.id


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


async def test_a_provider_that_opened_no_log_for_its_child_is_refused(
    mount: MountProfile,
) -> None:
    """A child's own log is its only record, so a child without one is one nothing
    could bring back after a restart — refused, as a child whose admission could not
    be written is.

    Sabotage: skip the check in `_admit`, and the spawn succeeds with no record.
    """

    @dataclass(slots=True)
    class _Logless:
        inner: StubSubagentProvider

        async def start(self, request: SubagentRequest) -> SubagentRun:
            run = await self.inner.start(request)
            run.session_id = "nowhere"
            return run

    ctx = await mount()
    ctx.require(SUBAGENTS).register_provider("logless", _Logless(StubSubagentProvider(root=ctx)))

    with pytest.raises(SubagentSpawnError, match="opened no log"):
        await ctx.require(SUBAGENTS).start(
            "logless", SubagentRequest(prompt="look", parent=_parent(ctx))
        )


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
