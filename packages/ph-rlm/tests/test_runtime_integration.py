"""The runtime inside the harness: C1 and C2 against the real pipeline.

`test_kernel.py` drives the runtime directly, with handler functions that stand
in for bindings. This file uses neither stand-in. It mounts the profile,
registers a real tool, and calls `run_code` through `ctx.tools.execute`. Every
binding call therefore goes out over fd 3, comes back through
`tools/pre-execute` → guards → approval → `tools/execute` →
`tools/post-execute`, and lands in the session log.

The whole containment argument depends on this claim: **one cell is one tool
call, but forty writes are forty governance evaluations.** Under the `ipython`
of prime-agent they were one. The feature map records that, and C1-C3 exist to
undo it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rlm_fixtures import MountedRuntime
from runtime_helpers import run_cell

from ph.keys import AGENTS, CODE_RUNTIME, FS, SESSIONS, SYSTEM_PROMPT, TOOLS, WORKSPACE
from ph.seams.subprocess import DEFAULT_SCRUB
from ph.testing import FAKE_OPTIONS, StubWorkspaceProvider, not_none, run_tool, simple_tool
from ph.tools.registry import RUN_CODE

pytestmark = pytest.mark.anyio


async def test_the_runtime_registers_as_the_code_runtime_provider(
    mounted_runtime: MountedRuntime,
) -> None:
    ctx, _session, _agent = await mounted_runtime(snapshots=False)
    provider = ctx.require(CODE_RUNTIME).require()
    assert provider.language == "python"
    assert provider.isolation == "process"
    assert provider.persistence == "namespace"
    # The promise that the seam checked at registration (D6).
    assert getattr(provider, "declares_kernel_snapshots", False) is True


async def test_a_cell_runs_through_the_transport(mounted_runtime: MountedRuntime) -> None:
    ctx, session, agent = await mounted_runtime(snapshots=False)
    result = await run_cell(ctx, "6 * 7", agent=agent, session=session)
    assert result.is_error is False
    assert result.value["value"] == 42


async def test_each_binding_call_is_its_own_governed_dispatch(
    mounted_runtime: MountedRuntime,
) -> None:
    """C2: three calls, three durable pairs, and `ToolCallLimit` sees three.

    This is the wording of the gate itself. Under one bespoke `ipython` tool,
    these three calls would have been one `tool/call` and one `tool/result`.
    """
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.require(TOOLS).register(simple_tool("ping", lambda _args, _run: "pong"))

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
    # The records can be ordered and paired without a timestamp, because the
    # sub-call id carries the index.
    assert [event.data["subCallId"] for event in starts] == [
        "call-1:code:0",
        "call-1:code:1",
        "call-1:code:2",
    ]
    assert {event.data["parentCallId"] for event in starts} == {"call-1"}


async def test_dispatch_records_are_log_only(mounted_runtime: MountedRuntime) -> None:
    """The other half of C2: sub-calls never enter model context again.

    Forty dispatch records in the log must not become forty messages in the next
    request. The compaction that they would force is worse than the blob that
    they replaced.
    """
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.require(TOOLS).register(simple_tool("ping", lambda _args, _run: "pong"))
    await run_cell(ctx, "await tools.ping()", agent=agent, session=session)

    rendered = repr([message.to_wire() for message in session.derive_messages()])
    assert "code-dispatch" not in rendered
    assert any(event.type == "tool/code-dispatch" for event in session.events)


async def test_a_tools_result_reaches_the_program(mounted_runtime: MountedRuntime) -> None:
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.require(TOOLS).register(simple_tool("echo", lambda args, _run: f"heard {args.get('what')}"))
    result = await run_cell(
        ctx, "reply = await tools.echo(what='you')\nreply", agent=agent, session=session
    )
    assert result.value["value"] == "heard you"


async def test_the_namespace_is_the_agent_and_persists_between_calls(
    mounted_runtime: MountedRuntime,
) -> None:
    """The namespace key is the agent id, so a kernel is scoped like the tools of the agent."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    await run_cell(ctx, "remembered = 'from the first call'", agent=agent, session=session)
    result = await run_cell(ctx, "remembered", agent=agent, session=session, call_id="call-2")
    assert result.value["value"] == "from the first call"


async def test_two_agents_do_not_share_a_namespace(mounted_runtime: MountedRuntime) -> None:
    ctx, first_session, first = await mounted_runtime(session_id="agent-one", snapshots=False)
    second_session = ctx.require(SESSIONS).create("agent-two")
    second = ctx.require(AGENTS).create(second_session, FAKE_OPTIONS)

    await run_cell(ctx, "mine = 'first'", agent=first, session=first_session)
    result = await run_cell(ctx, "'mine' in dir()", agent=second, session=second_session)
    assert result.value["value"] is False


async def test_the_sdk_section_lists_the_bindings(mounted_runtime: MountedRuntime) -> None:
    """The generated block replaces the hand-written call contract of prime-agent.

    The block is generated mechanically from the registry, so it cannot describe
    a surface that no longer exists.
    """
    ctx, _session, agent = await mounted_runtime(snapshots=False)
    ctx.require(TOOLS).register(simple_tool("ping", lambda _args, _run: "pong"))
    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    text = "\n".join(body for _name, body in assembly.sections)
    assert "tools.ping" in text
    # The text also has the rule that only the transport may be called directly
    # (the prompt half of C6).
    assert RUN_CODE in text


async def test_a_native_call_under_code_mode_is_refused(mounted_runtime: MountedRuntime) -> None:
    """C6, with the runtime mounted: the denial names the route back."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    ctx.require(TOOLS).register(simple_tool("ping", lambda _args, _run: "pong"))
    result = await run_tool(ctx, "ping", agent=agent, session=session, call_id="native-1")
    assert result.is_error is True
    assert RUN_CODE in repr(result.content) or "run_code" in repr(result.error)


async def test_the_kernel_closes_when_the_agent_is_disposed(
    mounted_runtime: MountedRuntime,
) -> None:
    """The namespace is the agent id, so the child unwinds with the agent (F1)."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    await run_cell(ctx, "1", agent=agent, session=session)
    runtime = ctx.get("python_runtime")
    assert agent.id in runtime._kernels  # the kernel table is the subject

    # Disposal unwinds the scope of the agent, and the kernel is an effect of
    # that scope.
    await ctx.require(AGENTS).dispose(agent.id)
    assert agent.id not in runtime._kernels


# ------------------------------------------------------------- containment --


async def test_a_cell_runs_inside_the_agents_workspace(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """D21, and the sentence that the whole `worktree` tier depends on.

    The `open("notes.txt", "w")` of a cell reaches no waterfall. A deny-list
    needs a registered name, and the model is authoring the tool (§4.8). But the
    write *does* resolve against the cwd of the child, so the tier limits
    authored code and does not only observe it. This test asserts that the kernel
    is actually started there. If a runtime inherited the directory of the host,
    that whole argument would be false, and every relative write from a cell
    would land in the shared tree.

    An absolute path still escapes. For that reason, §4.8 calls this collision
    isolation and reserves confinement for `sandbox` (E13).
    """
    ctx, session, agent = await mounted_runtime(snapshots=False)
    scratch = tmp_path / "scratch"
    ctx.require(WORKSPACE).register_provider(
        StubWorkspaceProvider(
            root=tmp_path / "trees", env={"PYTHONPYCACHEPREFIX": str(scratch / "pycache")}
        )
    )
    await ctx.require(WORKSPACE).acquire(
        session_id=session.id, agent_id=agent.id, base=ctx.require(FS).root, session=session
    )
    root = not_none(ctx.require(WORKSPACE).of(agent.id), "the agent workspace").root

    result = await run_cell(
        ctx,
        "import os\n"
        "open('authored.txt', 'w').write('a cell wrote this')\n"
        "(os.getcwd(), os.environ['PYTHONPYCACHEPREFIX'])",
        agent=agent,
        session=session,
    )

    where, pycache = result.value["value"]
    assert where == str(root)
    # A policy row could never have stopped this write. It landed in the tree of
    # the agent itself, which is the whole benefit of the tier.
    assert (root / "authored.txt").read_text(encoding="utf-8") == "a cell wrote this"
    # The redirection environment also reached the child, so a cell that runs the
    # tests of the project does not leave caches in that tree (E12).
    assert pycache == str(scratch / "pycache")


async def test_the_scrub_of_the_deployment_reaches_the_kernel(
    mounted_runtime: MountedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel runs the code that the model wrote, so the `scrub` and `keep`
    of `subprocess-local` have to reach it.

    The kernel does not spawn through `ctx.subprocess`, and it used to build its
    environment from the default scrub. A deployment that added a mark still
    passed the marked variable to every cell, and a name in `keep` never
    arrived.

    Sabotage: build the environment in `Kernel.start` from `scrub_env` and the
    cell reads `["configured", None]`.
    """
    monkeypatch.setenv("PH_TEST_SHIBBOLETH", "configured")
    monkeypatch.setenv("PH_TEST_KEPT_API_KEY", "kept")
    ctx, session, agent = await mounted_runtime(
        snapshots=False,
        extra_rows=[
            {
                "id": "subprocess",
                "config": {
                    "scrub": [*DEFAULT_SCRUB, "SHIBBOLETH"],
                    "keep": ["PH_TEST_KEPT_API_KEY"],
                },
            }
        ],
    )

    result = await run_cell(
        ctx,
        "import os\n"
        "[os.environ.get(name) for name in ('PH_TEST_SHIBBOLETH', 'PH_TEST_KEPT_API_KEY')]",
        agent=agent,
        session=session,
    )

    assert result.is_error is False
    assert result.value["value"] == [None, "kept"]
