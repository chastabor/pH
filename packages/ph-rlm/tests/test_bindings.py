"""The `rlm` namespace: delegation as a governed dispatch (P3-10, C2/C3/C4).

Prime Agent's `rlm(...)` was a comm-channel RPC — invisible to `pre-execute`, to
approval, to the call limits. These are the tests that ours is not: a spawn from
inside a cell produces a durable dispatch pair, a policy row can deny it, and the
spawn budget counts it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from conftest import BINDINGS_ROW, PROVIDER_ROW
from runtime_helpers import run_cell

from ph.system_prompt.assembly import AssembleContext, render_prompt
from ph.tools import Deny
from ph.tools.registry import ToolRestriction
from ph_rlm.bindings import DELETE_TOOL, LIST_TOOL, NAMESPACE, RUN_TOOL
from ph_rlm.presentation import IPYTHON

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]


@pytest.fixture
def delegating_runtime(mounted_runtime: Mounted) -> Callable[..., Any]:
    """The real kernel plus the delegation rows, so a cell can spawn for real."""

    async def build(**kwargs: Any) -> tuple[Any, Any, Any]:
        return await mounted_runtime(
            session_id="parent",
            presentation=True,
            extra_rows=[PROVIDER_ROW, BINDINGS_ROW, *kwargs.pop("extra_rows", [])],
            **kwargs,
        )

    return build


async def _cell(ctx: Any, program: str, *, agent: Any, session: Any, call_id: str = "c1") -> Any:
    return await run_cell(ctx, program, agent=agent, session=session, call_id=call_id, name=IPYTHON)


# ------------------------------------------------------------------- surface --


async def test_the_namespace_appears_in_the_sdk_block(delegating_runtime: Mounted) -> None:
    """One SDK route per capability: the namespaced form, not the tool name."""
    ctx, _session, agent = await delegating_runtime()
    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    text = render_prompt(assembly)

    assert f"{NAMESPACE}.run" in text
    assert f"{NAMESPACE}.list_subagents" in text
    assert f"{NAMESPACE}.delete_subagent" in text
    # The governed tool names are not what a cell writes, and the block says so
    # by rendering only the namespaced form.
    assert f"tools.{RUN_TOOL}" not in text


async def test_a_restricted_tool_leaves_the_sdk_block(delegating_runtime: Mounted) -> None:
    """The prompt asks the same factory the run does, so it cannot advertise what
    a cell could not call — the claim the previous test's docstring used to make
    without checking."""
    ctx, _session, agent = await delegating_runtime()
    ctx.tools.restrict(ToolRestriction(deny=frozenset({DELETE_TOOL})), scope=agent.ctx)

    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    text = render_prompt(assembly)
    assert f"{NAMESPACE}.run" in text
    assert f"{NAMESPACE}.delete_subagent" not in text


async def test_the_tools_are_not_directly_callable_under_code_mode(
    delegating_runtime: Mounted,
) -> None:
    """C6: the model reaches them through the transport or not at all."""
    ctx, session, agent = await delegating_runtime()
    from ph.testing import run_tool

    result = await run_tool(ctx, RUN_TOOL, {"prompt": "x"}, agent=agent, session=session)
    assert result.is_error is True
    assert IPYTHON in repr(result.content)


# -------------------------------------------------------------- the dispatch --


async def test_a_spawn_from_a_cell_is_a_durable_dispatch(delegating_runtime: Mounted) -> None:
    """C2: one governed dispatch pair per spawn, not one blob per cell."""
    ctx, session, agent = await delegating_runtime()
    result = await _cell(
        ctx,
        "handle = await rlm.run(prompt='find the bug', name='scout')\nhandle['name']",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == "scout"
    assert result.value["dispatches"] == 1

    starts = [e for e in session.events if e.type == "tool/code-dispatch-start"]
    settles = [e for e in session.events if e.type == "tool/code-dispatch"]
    assert [e.data["name"] for e in starts] == [RUN_TOOL]
    assert len(settles) == 1
    # And the admission the dispatch caused is in the same log.
    admitted = [e for e in session.events if e.type == "subagent/admitted"]
    assert len(admitted) == 1
    assert admitted[0].data["name"] == "scout"


async def test_the_handle_is_not_the_answer(delegating_runtime: Mounted) -> None:
    """A cell gets admission facts, so the model knows to keep working."""
    ctx, session, agent = await delegating_runtime()
    result = await _cell(
        ctx,
        "h = await rlm.run(prompt='think about it')\nsorted(h.keys())",
        agent=agent,
        session=session,
    )
    keys = result.value["value"]
    assert "childId" in keys and "sessionId" in keys and "grantedAccess" in keys
    assert "answer" not in keys and "result" not in keys


async def test_access_defaults_to_read_from_a_cell(delegating_runtime: Mounted) -> None:
    """E4 all the way through: the binding's default is the seam's default.

    What comes back is the *tier's* answer, and the shipped tier is `advisory`,
    where nothing can make a repository read-only — so a `read` request is
    honestly reported as a writable shared checkout rather than flattered with
    the word it asked for (§4.8's "not read-only" row).
    """
    ctx, session, agent = await delegating_runtime()
    result = await _cell(
        ctx,
        "h = await rlm.run(prompt='just read')\n(h['requestedAccess'], h['grantedAccess'])",
        agent=agent,
        session=session,
    )
    assert result.value["value"] == ["read", "write"]


async def test_nothing_narrowed_means_no_note_to_read(delegating_runtime: Mounted) -> None:
    """The note is for a *narrowing*, and there is none at this tier.

    A note on every spawn is one a model learns to skip; this one appears when a
    child asked for the repository and a tier refused it, which is the moment
    the parent has to re-plan.
    """
    ctx, session, agent = await delegating_runtime()
    result = await _cell(
        ctx,
        "h = await rlm.run(prompt='implement it', access='write')\n"
        "(h['grantedAccess'], h.get('note'))",
        agent=agent,
        session=session,
    )
    assert result.value["value"] == ["write", None]


# ---------------------------------------------------------------- governance --


async def test_a_policy_row_can_deny_spawning(delegating_runtime: Mounted) -> None:
    """C3: the denial ends the run, and the program cannot catch past it.

    This is the property prime-agent's comm channel could not have: spawning is
    a tool call, so it is deniable by exactly the mechanism that denies a write.
    """
    ctx, session, agent = await delegating_runtime()

    async def refuse(execution: Any, next_: Any) -> Any:
        if execution.name == RUN_TOOL:
            return Deny(reason="this deployment does not allow subagents")
        return await next_()

    ctx.on("tools/pre-execute", refuse)
    result = await _cell(
        ctx,
        "try:\n    await rlm.run(prompt='sneak')\nexcept BaseException:\n    pass\n'survived'",
        agent=agent,
        session=session,
    )
    assert result.is_error is True
    assert "does not allow subagents" in repr(result.content)
    # Nothing was admitted, and the cell did not get to report success.
    assert [e for e in session.events if e.type == "subagent/admitted"] == []


async def test_the_spawn_budget_bounds_one_cell(delegating_runtime: Mounted) -> None:
    """C4: one approved cell cannot fan out without limit on that one approval."""
    ctx, session, agent = await delegating_runtime(
        extra_rows=[{"id": "tools-code-mode", "config": {"maxSubagentSpawnsPerRun": 2}}]
    )
    result = await _cell(
        ctx,
        "for index in range(5):\n    await rlm.run(prompt=f'task {index}')\n'unreached'",
        agent=agent,
        session=session,
    )
    assert result.is_error is True
    assert "max_subagent_spawns_per_run=2" in repr(result.content)
    assert len([e for e in session.events if e.type == "subagent/admitted"]) == 2


async def test_a_refused_spawn_is_the_programs_to_handle(delegating_runtime: Mounted) -> None:
    """A *failed* spawn is not a denial: the model may re-plan around it."""
    ctx, session, agent = await delegating_runtime()
    result = await _cell(
        ctx,
        "await rlm.run(prompt='first', name='dup')\n"
        "try:\n"
        "    await rlm.run(prompt='second', name='dup')\n"
        "    outcome = 'no error'\n"
        "except Exception as error:\n"
        "    outcome = 'caught'\n"
        "outcome",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == "caught"


# --------------------------------------------------------- roster and delete --


async def test_the_roster_a_cell_reads_is_the_fold(delegating_runtime: Mounted) -> None:
    ctx, session, agent = await delegating_runtime()
    await _cell(
        ctx,
        "await rlm.run(prompt='one', name='alpha')\nawait rlm.run(prompt='two', name='beta')",
        agent=agent,
        session=session,
        call_id="c1",
    )
    await ctx.drain()
    result = await _cell(
        ctx,
        "roster = await rlm.list_subagents()\nsorted(c['name'] for c in roster['children'])",
        agent=agent,
        session=session,
        call_id="c2",
    )
    assert result.value["value"] == ["alpha", "beta"]


async def test_deleting_from_a_cell_tombstones_the_child(delegating_runtime: Mounted) -> None:
    ctx, session, agent = await delegating_runtime()
    spawn = await _cell(
        ctx, "h = await rlm.run(prompt='doomed')\nh['childId']", agent=agent, session=session
    )
    child_id = spawn.value["value"]
    result = await _cell(
        ctx,
        f"outcome = await rlm.delete_subagent(child_id={child_id!r})\noutcome['deleted']",
        agent=agent,
        session=session,
        call_id="c2",
    )
    assert result.value["value"] is True

    tombstones = [e for e in session.events if e.type == "subagent/deleted"]
    assert [e.data["runId"] for e in tombstones] == [child_id]
    # The dispatch is recorded under the governed tool name, not the cell's.
    names = [e.data["name"] for e in session.events if e.type == "tool/code-dispatch-start"]
    assert DELETE_TOOL in names
    assert LIST_TOOL not in names
