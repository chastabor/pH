"""The RLM doctrine: what it says, what it withholds, and where it lives (P3-14).

Two properties are worth more than the wording. **The delegation section is
conditional on the surface actually being there**, so the prompt cannot advertise
a call the agent would be denied. And **the volatile facts are a `context()`, not
a cached section**, so a turn that changes the depth or the family does not
re-bill the whole prefix (A12).

## Why the doctrine carries no call listing

Prime Agent's doctrine carried an "RLM-native call contract" paragraph, and a
first draft of `rlm-prompt` carried a bullet list of the five delegation calls.
Both existed because prime-agent had no generated listing. `tools:sdk` *is* that
listing, built from the registry, so prose beside it is a second description of
one surface — with the hand-written copy going stale first.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from conftest import BINDINGS_ROW, DOCTRINE_ROW, PROVIDER_ROW

from ph.seams.subagents import SubagentRequest
from ph.system_prompt import join_context_sections, render_context_sections, render_prompt
from ph_rlm.presentation import IPYTHON
from ph_rlm.prompt import CHILD_DOCTRINE, DELEGATION, DOCTRINE, WORKSPACE_LINE
from ph_rlm.subagents import PROVIDER_NAME, RLM_MAX_DEPTH

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]


@pytest.fixture
def prompted(mounted_runtime: Mounted) -> Callable[..., Any]:
    """The real transport plus the doctrine, optionally plus delegation.

    `extra_rows` is *additive*, so a test that wants a config patch passes only
    the patch and cannot silently drop the default rows.
    """

    async def build(
        *,
        delegation: bool = True,
        extra_rows: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, Any]:
        rows = [
            *([PROVIDER_ROW, BINDINGS_ROW] if delegation else []),
            DOCTRINE_ROW,
            *(extra_rows or []),
        ]
        return await mounted_runtime(
            session_id="parent", presentation=True, extra_rows=rows, **kwargs
        )

    return build


async def _assemble(ctx: Any, agent: Any) -> Any:
    return await ctx.system_prompt.assemble(agent.ctx, agent=agent)


async def _prompt(ctx: Any, agent: Any) -> str:
    return render_prompt(await _assemble(ctx, agent))


async def _snapshot(ctx: Any, agent: Any) -> str:
    """The snapshot exactly as the driver builds it — same joiner, same text."""
    return join_context_sections(render_context_sections(await _assemble(ctx, agent)))


# ---------------------------------------------------------------- doctrine --


async def test_the_doctrine_names_the_one_callable_and_the_kernel(prompted: Mounted) -> None:
    ctx, _session, agent = await prompted()
    text = await _prompt(ctx, agent)

    assert DOCTRINE.strip() in text
    # The two facts the constant would be wrong without: the transport's
    # *presented* name, and the shell-cell rule.
    assert IPYTHON in DOCTRINE
    # Named, and refused. The doctrine used to promise `%%bash` shell cells —
    # ported from prime-agent, where they existed — while pH's guest raises a
    # `SyntaxError` for one (P3-22). It still names the case the model was
    # trained on, because that is the thing it would otherwise try.
    assert "%%bash" in DOCTRINE
    assert "no IPython magics" in DOCTRINE


async def test_the_doctrine_does_not_re_describe_the_sdk_surface(prompted: Mounted) -> None:
    """Prime Agent's "RLM-native call contract" paragraph is dropped on purpose.

    It existed because prime-agent had no generated listing; `tools:sdk` *is*
    one, so keeping the prose would be two descriptions of one surface with the
    hand-written one going stale first.
    """
    ctx, _session, agent = await prompted()
    text = await _prompt(ctx, agent)

    assert "call_skill" not in text
    assert "RLM-native call contract" not in text
    # And nothing re-lists the calls the SDK block already renders.
    assert "async def tools." in text
    for call in ("rlm.list_subagents()", "agent_observe.get(", "rlm.delete_subagent("):
        assert call not in DELEGATION, f"{call} is described twice"


async def test_the_doctrine_comes_after_the_sdk_block(prompted: Mounted) -> None:
    """It refers to "the SDK block above", so the order is load-bearing."""
    ctx, _session, agent = await prompted()
    text = await _prompt(ctx, agent)
    assert text.index("async def tools.") < text.index("# Writing code to solve tasks")


# -------------------------------------------------------------- delegation --


async def test_the_delegation_section_states_the_non_blocking_rule(prompted: Mounted) -> None:
    """`rlm.run` returns a handle, so a model that waits waits forever."""
    ctx, _session, agent = await prompted()
    text = await _prompt(ctx, agent)

    assert DELEGATION.strip() in text
    # The rules the generated listing cannot state, checked against the constant
    # so a reflow of the prose cannot break a test about semantics.
    assert "returns immediately" in DELEGATION
    assert "time.sleep()" in DELEGATION
    assert "access" in DELEGATION and '"read"' in DELEGATION


async def test_no_delegation_section_without_the_namespace(prompted: Mounted) -> None:
    """The prompt must not advertise a call the agent would be denied."""
    ctx, _session, agent = await prompted(delegation=False)
    text = await _prompt(ctx, agent)

    assert DOCTRINE.strip() in text, "the doctrine still applies"
    assert "# Delegating to child agents" not in text
    assert "rlm.run" not in text
    # And it is *absent* from the assembly rather than present-and-empty, so a
    # consumer enumerating section names does not see a phantom.
    names = [name for name, _body in (await _assemble(ctx, agent)).sections]
    assert "rlm:delegation" not in names
    assert "rlm:doctrine" in names


async def test_a_child_is_told_it_is_a_child(prompted: Mounted) -> None:
    """Depth > 0 gets the reply instruction; a root does not."""
    ctx, _session, parent = await prompted()
    assert CHILD_DOCTRINE.strip() not in await _prompt(ctx, parent)

    run = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="look into it", parent=parent, name="scout")
    )
    child = ctx.agents.get(run.session_id)
    assert child is not None
    text = await _prompt(ctx, child)
    assert CHILD_DOCTRINE.strip() in text
    assert 'receiver_role="parent"' in text


# ------------------------------------------------------- the facts snapshot --


async def test_the_volatile_facts_are_a_context_not_a_section(prompted: Mounted) -> None:
    """A12: depth, cwd and family move between turns, so they must not sit in the
    cached prefix."""
    ctx, session, agent = await prompted()
    prompt = await _prompt(ctx, agent)
    snapshot = await _snapshot(ctx, agent)

    assert "# Session" in snapshot
    assert "# Session" not in prompt, "volatile facts leaked into the cached prefix"
    assert f"Recursive agent depth: 0 of {RLM_MAX_DEPTH}" in snapshot
    assert f"Conversation log: {session.id}" in snapshot


async def test_the_snapshot_says_no_workspace_has_been_acquired(prompted: Mounted) -> None:
    """A child told nothing about its workspace attempts writes and reads the
    failures as its own bug — which is the failure this line exists to prevent,
    so it ships saying what is true now: the seam is mounted (P4-07) but nothing
    acquires until the agent lifecycle does (P4-08)."""
    ctx, _session, agent = await prompted()
    snapshot = await _snapshot(ctx, agent)

    assert WORKSPACE_LINE in snapshot
    assert "recorded but not granted" in snapshot


async def test_an_acquired_workspace_replaces_the_none_acquired_line(prompted: Mounted) -> None:
    """The line is reached by *asking the seam for this agent*, so acquiring one
    changes the answer.

    An asserted absence would keep telling every agent nothing was acquired after
    something was — and this test would have defended the falsehood. It asks
    `ctx.workspace.of(agent.id)` rather than reading `ctx.workspace` as if the
    provision were itself a workspace, which is what the first draft did and what
    would have started describing the seam object the day it was mounted.
    """
    ctx, _session, agent = await prompted()
    await ctx.workspace.acquire(
        session_id="s1", agent_id=agent.id, base=Path("/repo"), session=agent.session
    )
    snapshot = await _snapshot(ctx, agent)

    assert WORKSPACE_LINE not in snapshot
    assert "Workspace: /repo (writable, shared)" in snapshot
    # The scratch line is not decoration: it is the only place a child whose repo
    # is read-only is told where it *may* write.
    assert "Writable scratch: " in snapshot


async def test_the_snapshot_lists_the_family_and_the_children(prompted: Mounted) -> None:
    ctx, _session, parent = await prompted()
    run = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="find it", parent=parent, name="scout")
    )
    snapshot = await _snapshot(ctx, parent)

    assert "child scout" in snapshot
    assert "Your children: scout" in snapshot

    child = ctx.agents.get(run.session_id)
    assert child is not None
    child_snapshot = await _snapshot(ctx, child)
    assert "Reachable agents: parent " in child_snapshot
    assert f"Recursive agent depth: 1 of {RLM_MAX_DEPTH}" in child_snapshot


async def test_a_child_at_the_depth_limit_is_told_so(prompted: Mounted) -> None:
    """Reading a depth without the limit is how a model spends a turn on a denial."""
    ctx, _session, parent = await prompted(
        extra_rows=[{"id": "rlm-subagent-provider", "config": {"maxDepth": 1}}]
    )
    run = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="one level", parent=parent, name="scout")
    )
    child = ctx.agents.get(run.session_id)
    assert child is not None
    assert "you may not delegate further" in await _snapshot(ctx, child)
