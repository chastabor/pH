"""A spawn on its way: counted by the caps, and holding its child's name.

A spawn's guards run before its provider builds the child, and the child is in
`children` only once its admission is written. So two spawns from one step (the
driver runs a step's tool calls side by side) each saw the other missing: both could
pass a cap one of them crossed, and both could take one name, since the rlm provider
named a child from the admitted ones. The seam now names every child right after the
guards, and a spawn is on its parent's list from there until its admission lands or
it is refused. `SubagentService.child_counts` counts that list with the children.
"""

from __future__ import annotations

import anyio
import pytest

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.keys import AGENTS, SESSIONS, SUBAGENTS
from ph.seams import subagents as subagents_seam
from ph.seams.subagents import (
    MAX_NAME_CHARS,
    ChildCounts,
    SubagentRequest,
    SubagentSpawnError,
)
from ph.session import Session, session_written
from ph.testing import (
    FAKE_OPTIONS,
    HoldingProvider,
    MountProfile,
    RefusingProvider,
    StubSubagentProvider,
)

pytestmark = pytest.mark.anyio


def _parent(ctx: Context) -> tuple[Session, AgentDriver]:
    session = ctx.require(SESSIONS).create("lead")
    return session, ctx.require(AGENTS).create(session, FAKE_OPTIONS)


async def test_a_spawn_on_its_way_counts_as_a_child(mount: MountProfile) -> None:
    """While the first child is being built, it is already on its parent's count,
    in this turn and in all.

    Sabotage: leave the spawns on their way out of `child_counts`, and it reads none.
    """
    ctx = await mount()
    session, parent = _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    held = HoldingProvider(StubSubagentProvider(root=ctx))
    subagents.register_provider("stub", held)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(subagents.start, "stub", SubagentRequest(prompt="one", parent=parent))
        await held.building.wait()
        assert subagents.child_counts(session) == ChildCounts(turn=1, session=1)
        held.go.set()

    assert subagents.child_counts(session) == ChildCounts(turn=1, session=1)


async def test_a_child_is_counted_once_while_its_admission_is_written(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spawn comes off its parent's list in the step that writes its admission,
    which is the step it starts counting as a child. Counted both ways while that
    admission is flushed, a spawn arriving then would be refused a place that was free.

    Sabotage: take a spawn off its list only once `start` returns, and the count reads
    two.
    """
    ctx = await mount()
    session, parent = _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    subagents.register_provider("stub", StubSubagentProvider(root=ctx))
    writing, go = anyio.Event(), anyio.Event()

    async def held_write(ctx: Context, child: Session) -> bool:
        writing.set()
        await go.wait()
        return await session_written(ctx, child)

    monkeypatch.setattr(subagents_seam, "session_written", held_write)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(subagents.start, "stub", SubagentRequest(prompt="one", parent=parent))
        await writing.wait()
        assert subagents.child_counts(session) == ChildCounts(turn=1, session=1)
        go.set()


async def test_two_spawns_at_once_cannot_take_one_name(mount: MountProfile) -> None:
    """While the first "scout" is being built, a second spawn asking for the name is
    refused, and its provider is never asked.

    Sabotage: leave the spawns on their way out of the names `_named` counts as
    taken, and a second "scout" is made.
    """
    ctx = await mount()
    session, parent = _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    held = HoldingProvider(StubSubagentProvider(root=ctx))
    subagents.register_provider("stub", held)

    async with anyio.create_task_group() as tasks:
        first = SubagentRequest(prompt="one", parent=parent, name="scout")
        tasks.start_soon(subagents.start, "stub", first)
        await held.building.wait()
        with pytest.raises(SubagentSpawnError, match='already named "scout"'):
            await subagents.start(
                "stub", SubagentRequest(prompt="two", parent=parent, name="scout")
            )
        held.go.set()

    assert [child.name for child in subagents.children(session.id).values()] == ["scout"]
    assert [one.prompt for one in held.inner.requests] == ["one"]


async def test_a_made_name_is_held_while_its_child_is_built(mount: MountProfile) -> None:
    """A spawn that leaves its name out gets one made from its task, and holds it
    the same way: a sibling asking for that name meanwhile is refused.

    Sabotage: as above, and the second child takes the first's made name.
    """
    ctx = await mount()
    session, parent = _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    held = HoldingProvider(StubSubagentProvider(root=ctx))
    subagents.register_provider("stub", held)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(subagents.start, "stub", SubagentRequest(prompt="survey", parent=parent))
        await held.building.wait()
        assert held.first is not None and held.first.name is not None
        made = held.first.name
        assert made.startswith("subagent-survey-")
        with pytest.raises(SubagentSpawnError, match="already named"):
            await subagents.start("stub", SubagentRequest(prompt="two", parent=parent, name=made))
        held.go.set()

    assert [child.name for child in subagents.children(session.id).values()] == [made]


async def test_a_refused_spawn_gives_its_place_and_name_back(mount: MountProfile) -> None:
    """A spawn its provider refuses after the guards passed leaves no child, so it
    leaves the count as it was and its name free. Kept, it would spend a place in the
    cap on a child that never existed, for the rest of the mount.

    Sabotage: take a spawn off its list only when its admission lands, and the count
    still holds it and the second "scout" is refused.
    """
    ctx = await mount()
    session, parent = _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    subagents.register_provider("refusing", RefusingProvider())
    subagents.register_provider("stub", StubSubagentProvider(root=ctx))

    with pytest.raises(SubagentSpawnError, match="no room"):
        await subagents.start(
            "refusing", SubagentRequest(prompt="one", parent=parent, name="scout")
        )
    assert subagents.child_counts(session) == ChildCounts()
    run = await subagents.start("stub", SubagentRequest(prompt="two", parent=parent, name="scout"))

    assert run.name == "scout"
    assert subagents.child_counts(session) == ChildCounts(turn=1, session=1)


@pytest.mark.parametrize("name", ["", "   ", "x" * (MAX_NAME_CHARS + 1)])
async def test_a_name_a_spawn_cannot_have_is_refused_before_its_provider(
    mount: MountProfile, name: str
) -> None:
    ctx = await mount()
    _session, parent = _parent(ctx)
    subagents = ctx.require(SUBAGENTS)
    provider = StubSubagentProvider(root=ctx)
    subagents.register_provider("stub", provider)

    with pytest.raises(SubagentSpawnError, match=f"1..{MAX_NAME_CHARS} characters"):
        await subagents.start("stub", SubagentRequest(prompt="one", parent=parent, name=name))

    assert provider.requests == []
