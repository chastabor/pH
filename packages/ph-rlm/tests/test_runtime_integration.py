"""The runtime inside the harness: C1 and C2 against the real pipeline.

`test_kernel.py` drives the runtime directly, with handler functions standing in
for bindings. This file uses neither stand-in: it mounts the profile, registers a
real tool, and calls `run_code` through `ctx.tools.execute`, so every binding
call goes out over fd 3, comes back through `tools/pre-execute` → guards →
approval → `tools/execute` → `tools/post-execute`, and lands in the session log.

That is the claim the whole containment argument rests on: **one cell is one tool
call, but forty writes are forty governance evaluations.** Under prime-agent's
`ipython` they were one, which is what the feature map records and what C1-C3
exist to undo.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from runtime_helpers import run_cell

from ph.system_prompt.assembly import AssembleContext
from ph.testing import FAKE_OPTIONS, run_tool, simple_tool
from ph.tools.registry import RUN_CODE

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]


async def test_the_runtime_registers_as_the_code_runtime_provider(
    mounted_runtime: Mounted,
) -> None:
    ctx, _session, _agent = await mounted_runtime(snapshots=False)
    provider = ctx.code_runtime.require()
    assert provider.language == "python"
    assert provider.isolation == "process"
    assert provider.persistence == "namespace"
    # The promise the seam checked at registration (D6).
    assert provider.declares_kernel_snapshots is True


async def test_a_cell_runs_through_the_transport(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(snapshots=False)
    result = await run_cell(ctx, "6 * 7", agent=agent, session=session)
    assert result.is_error is False
    assert result.value["value"] == 42


async def test_each_binding_call_is_its_own_governed_dispatch(mounted_runtime: Mounted) -> None:
    """C2: three calls, three durable pairs, and `ToolCallLimit` sees three.

    The gate's own wording. Under one bespoke `ipython` tool these three writes
    would have been one `tool/call` and one `tool/result`.
    """
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.tools.register(simple_tool("ping", lambda _args, _run: "pong"))

    result = await run_cell(
        ctx,
        "import asyncio\nanswers = [await tools.ping() for _ in range(3)]\nanswers",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["dispatches"] == 3

    starts = [event for event in session.events if event.type == "tool/code-dispatch-start"]
    settles = [event for event in session.events if event.type == "tool/code-dispatch"]
    assert len(starts) == 3
    assert len(settles) == 3
    assert [event.data["name"] for event in starts] == ["ping", "ping", "ping"]
    # Ordered and pairable without a timestamp: the sub-call id carries the index.
    assert [event.data["subCallId"] for event in starts] == [
        "call-1:code:0",
        "call-1:code:1",
        "call-1:code:2",
    ]
    assert {event.data["parentCallId"] for event in starts} == {"call-1"}


async def test_dispatch_records_are_log_only(mounted_runtime: Mounted) -> None:
    """C2's other half: sub-calls never re-enter model context.

    Forty dispatch records in the log must not become forty messages in the next
    request, or the compaction they would force is worse than the blob they
    replaced.
    """
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.tools.register(simple_tool("ping", lambda _args, _run: "pong"))
    await run_cell(ctx, "await tools.ping()", agent=agent, session=session)

    rendered = repr([message.to_wire() for message in session.derive_messages()])
    assert "code-dispatch" not in rendered
    assert any(event.type == "tool/code-dispatch" for event in session.events)


async def test_a_tools_result_reaches_the_program(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.tools.register(simple_tool("echo", lambda args, _run: f"heard {args.get('what')}"))
    result = await run_cell(
        ctx, "reply = await tools.echo(what='you')\nreply", agent=agent, session=session
    )
    assert result.value["value"] == "heard you"


async def test_the_namespace_is_the_agent_and_persists_between_calls(
    mounted_runtime: Mounted,
) -> None:
    """The namespace key is the agent id, so a kernel is scoped like its tools."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    await run_cell(ctx, "remembered = 'from the first call'", agent=agent, session=session)
    result = await run_cell(ctx, "remembered", agent=agent, session=session, call_id="call-2")
    assert result.value["value"] == "from the first call"


async def test_two_agents_do_not_share_a_namespace(mounted_runtime: Mounted) -> None:
    ctx, first_session, first = await mounted_runtime(session_id="agent-one", snapshots=False)
    second_session = ctx.sessions.create("agent-two")
    second = ctx.agents.create(second_session, FAKE_OPTIONS)

    await run_cell(ctx, "mine = 'first'", agent=first, session=first_session)
    result = await run_cell(ctx, "'mine' in dir()", agent=second, session=second_session)
    assert result.value["value"] is False


async def test_the_sdk_section_lists_the_bindings(mounted_runtime: Mounted) -> None:
    """The generated block replaces prime-agent's hand-written call contract.

    Mechanically generated from the registry, so it cannot describe a surface
    that no longer exists.
    """
    ctx, _session, agent = await mounted_runtime(snapshots=False)
    ctx.tools.register(simple_tool("ping", lambda _args, _run: "pong"))
    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    text = "\n".join(body for _name, body in assembly.sections)
    assert "tools.ping" in text
    # And the rule that only the transport may be called directly (C6's prompt half).
    assert RUN_CODE in text


async def test_a_native_call_under_code_mode_is_refused(mounted_runtime: Mounted) -> None:
    """C6, with the runtime mounted: the denial names the route back."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.tools.register(simple_tool("ping", lambda _args, _run: "pong"))
    result = await run_tool(ctx, "ping", agent=agent, session=session, call_id="native-1")
    assert result.is_error is True
    assert RUN_CODE in repr(result.content) or "run_code" in repr(result.error)


async def test_the_kernel_closes_when_the_agent_is_disposed(mounted_runtime: Mounted) -> None:
    """The namespace is the agent id, so the child unwinds with the agent (F1)."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    await run_cell(ctx, "1", agent=agent, session=session)
    runtime = ctx.get("python_runtime")
    assert agent.id in runtime._kernels  # the kernel table is the subject

    # Disposal unwinds the agent's scope, and the kernel is an effect of it.
    await ctx.agents.dispose(agent.id)
    assert agent.id not in runtime._kernels
