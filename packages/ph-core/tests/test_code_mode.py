"""P1-04 and P1-05 — Code Mode governance (C1-C4, C6).

These are the tests the whole containment argument rests on. A `run_code` cell
is one tool call; the claim is that it is nonetheless **not** one governance
evaluation. So:

* every `await tools.<name>(...)` re-enters the full pipeline as a sub-call (C1);
* three binding calls produce three durable dispatch pairs, not one blob (C2);
* a denial fails the **run**, and the program cannot catch its way past it (C3);
* budgets bound one approved cell (C4);
* a model-direct native call under `mode: code` is refused before policy (C6).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ph.cordis import DEPLOYMENT, Context
from ph.seams.code_runtime import CodeBindingNamespace
from ph.system_prompt.assembly import render_prompt
from ph.testing import FAKE_OPTIONS, raising, run_tool, simple_tool
from ph.tools import Deny, ToolExecutionInput, text_content
from ph.tools.code_mode import ToolCallError, governed_binding
from ph.tools.definition import ToolOutput, TransportPresentation
from ph.tools.registry import RUN_CODE

pytestmark = pytest.mark.anyio

CODE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "code-runtime-stub", "name": "code-runtime-stub"},
    {"id": "tools-code-mode", "name": "tools-code-mode"},
)


async def _code_ctx(mount: Any, **overrides: Any) -> Context:
    rows: list[dict[str, Any]] = [
        {"insert": [dict(row) for row in CODE_ROWS]},
    ]
    if overrides:
        rows.append({"id": "tools-code-mode", "config": overrides})
    return await mount(*rows)


def _recorder(name: str, calls: list[str], *, safe: bool = True) -> Any:
    def body(args: Any, _run: Any) -> Any:
        calls.append(f"{name}:{(args or {}).get('n')}")
        return f"{name} ok"

    return simple_tool(name, body, safe=safe)


async def _run(ctx: Context, program_name: str, program: Any) -> Any:
    ctx.code_runtime_stub.register_program(program_name, program)
    session = ctx.sessions.create(f"s-{program_name}")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    result = await ctx.tools.execute(
        ToolExecutionInput(
            call_id="root-1",
            name=RUN_CODE,
            arguments={"program": program_name},
            scope=agent.ctx,
            session=session,
            agent=agent,
        )
    )
    return result, session


async def test_a_binding_call_re_enters_the_whole_pipeline(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.tools.register(_recorder("touch", calls))
    seen: list[str] = []
    ctx.on("tools/pre-execute", lambda execution, next_: seen.append(execution.name) or next_())

    async def program(ns: Any, emit: Any) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    result, _session = await _run(ctx, "one-call", program)
    assert not result.is_error
    # The transport and the sub-call are both policed; the sub-call did not
    # bypass anything by virtue of being inside a cell (C1).
    assert seen == [RUN_CODE, "touch"]
    assert calls == ["touch:1"]


async def test_three_binding_calls_produce_three_durable_dispatch_pairs(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.tools.register(_recorder("touch", calls))

    async def program(ns: Any, emit: Any) -> str:
        for index in range(3):
            await ns["tools"].touch(n=index)
        return "done"

    _result, session = await _run(ctx, "three-calls", program)
    starts = [e for e in session.events if e.type == "tool/code-dispatch-start"]
    settles = [e for e in session.events if e.type == "tool/code-dispatch"]
    # Forty writes must not be one stdout blob (C2).
    assert len(starts) == len(settles) == 3
    assert [e.data["subCallId"] for e in starts] == [
        "root-1:code:0",
        "root-1:code:1",
        "root-1:code:2",
    ]
    assert all(e.data["parentCallId"] == "root-1" for e in settles)


async def test_dispatch_records_stay_out_of_model_context(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.register(_recorder("touch", []))

    async def program(ns: Any, emit: Any) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    _result, session = await _run(ctx, "log-only", program)
    assert any(event.type == "tool/code-dispatch" for event in session.events)
    # Log-only: a sub-call is a durable record, not a message. Deriving them
    # would put the same work in context twice.
    assert session.derive_messages() == ()


async def test_a_denied_binding_call_fails_the_whole_run(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.tools.register(_recorder("touch", calls))
    ctx.tools.register(_recorder("forbidden", calls))
    ctx.tools.guard(
        lambda execution: "policy forbids this" if execution.name == "forbidden" else None
    )
    reached: list[str] = []

    async def program(ns: Any, emit: Any) -> str:
        await ns["tools"].touch(n=1)
        try:
            await ns["tools"].forbidden(n=2)
        except Exception:
            # A program that could route around a refusal would make the
            # refusal advisory. It cannot: CodeRunFailure is not catchable in
            # any way that continues the run.
            reached.append("caught")
            await ns["tools"].touch(n=3)
        return "done"

    result, _session = await _run(ctx, "denied", program)
    assert result.is_error
    assert "was refused" in result.error.message
    # Everything after the refusal is abandoned, so partial state is bounded to
    # this one cell (C3).
    assert calls == ["touch:1"]


async def test_a_failed_binding_call_is_the_programs_to_handle(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.register(simple_tool("flaky", raising(RuntimeError("transient"))))
    handled: list[str] = []

    async def program(ns: Any, emit: Any) -> str:
        try:
            await ns["tools"].flaky()
        except ToolCallError as error:
            handled.append(str(error))
        return "recovered"

    result, _session = await _run(ctx, "failed", program)
    # A failure is different from a refusal: the model may retry it, so the
    # program keeps dsh's ToolCallError semantics.
    assert not result.is_error
    assert handled and "transient" in handled[0]


async def test_a_runaway_program_fails_at_its_budget(mount: Any) -> None:
    ctx = await _code_ctx(mount, maxDispatchesPerRun=4)
    calls: list[str] = []
    ctx.tools.register(_recorder("touch", calls))

    async def program(ns: Any, emit: Any) -> str:
        for index in range(100):
            await ns["tools"].touch(n=index)
        return "done"

    result, _session = await _run(ctx, "runaway", program)
    assert result.is_error
    # The budget is named, so the model can split the work rather than guess.
    assert "max_dispatches_per_run=4" in result.error.message
    assert len(calls) == 4


async def test_a_model_direct_native_call_is_refused_under_code_mode(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.register(_recorder("touch", []))
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    ctx.tools.present_as("code", scope=agent.ctx)
    seen: list[str] = []
    ctx.on("tools/pre-execute", lambda execution, next_: seen.append(execution.name) or next_())

    result = await ctx.tools.execute(
        ToolExecutionInput(call_id="c", name="touch", arguments={}, scope=agent.ctx, agent=agent)
    )
    assert result.is_error
    assert "await tools.touch(...)" in result.error.message
    # Resolved before policy, so a permissive row cannot allow a name the prompt
    # never offered (C6).
    assert seen == []


async def test_the_sdk_section_lists_the_bindings_and_the_code_only_rule(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.register(_recorder("touch", []))
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    ctx.tools.present_as("code", scope=agent.ctx)

    from ph.system_prompt.assembly import render_prompt

    assembly = await ctx.system_prompt.assemble(agent.ctx, agent=agent)
    prompt = render_prompt(assembly)
    assert "async def tools.touch(" in prompt
    assert f"only\n`{RUN_CODE}` is callable" in prompt or RUN_CODE in prompt
    # Under Code Mode the model is offered one callable, not a schema list.
    assert ctx.tools.schemas(scope=agent.ctx) == []


async def test_the_transport_name_cannot_be_shadowed(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    assert ctx.tools.get(RUN_CODE, scope=DEPLOYMENT) is not None
    with pytest.raises(ValueError, match="reserved"):
        ctx.tools.register(_recorder(RUN_CODE, []))


async def test_an_oversized_dispatch_can_be_reshaped_before_it_is_logged(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.register(_recorder("touch", []))

    async def shrink(record: Any, content: Any, next_: Any) -> Any:
        return text_content("[spilled]")

    ctx.on("tools/code-dispatch-log", shrink)

    async def program(ns: Any, emit: Any) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    _result, session = await _run(ctx, "spill", program)
    (settle,) = [e for e in session.events if e.type == "tool/code-dispatch"]
    # Each dispatch is offloadable individually (C5), so one large read does not
    # melt its siblings.
    assert settle.data["content"][0]["text"] == "[spilled]"


async def test_a_spawning_binding_is_held_to_the_spawn_budget(mount: Any) -> None:
    """Counted by a property the binding declares, not by guessing from its name."""
    from ph.cancel import CancelToken
    from ph.seams.code_runtime import CodeBinding
    from ph.tools.code_mode import CodeRunFailure, DispatchBridge

    ctx = await _code_ctx(mount)
    ctx.tools.register(_recorder("spawn", []))
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    run = ctx.tools.create_execution(
        ToolExecutionInput(
            call_id="root", name=RUN_CODE, arguments={"program": "x"}, scope=agent.ctx, agent=agent
        )
    )
    bridge = DispatchBridge(
        tools=ctx.tools,
        ctx=ctx,
        execution=run.execution,
        session=session,
        token=CancelToken(),
        max_spawns=1,
    )
    spawner = CodeBinding(name="spawn", description="", parameters={}, counts_as_spawn=True)
    await bridge.call(spawner, {"n": 1})
    with pytest.raises(CodeRunFailure) as caught:
        await bridge.call(spawner, {"n": 2})
    assert caught.value.kind == "budget"
    assert "max_subagent_spawns_per_run=1" in str(caught.value)


async def test_a_pre_execute_denial_of_a_sub_call_also_fails_the_run(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.register(_recorder("touch", []))
    ctx.on(
        "tools/pre-execute",
        lambda execution, next_: (
            Deny(reason="not now") if execution.parent is not None else next_()
        ),
    )

    async def program(ns: Any, emit: Any) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    result, _session = await _run(ctx, "pre-denied", program)
    assert result.is_error
    assert "not now" in result.error.message


# ------------------------------------------------- transport presentation --
#
# P3-09. The transport name is reserved so nothing can occupy it and misdirect a
# model told to call it (C6) — but a profile still has to be able to present it
# under its own name. These are the tests that renaming it moves *every* place
# the name is load-bearing, not just the schema.


def _as_ipython() -> TransportPresentation:
    return TransportPresentation(
        name="ipython",
        description="Python cells.",
        output=ToolOutput(schema={"type": "object"}, render=lambda _a, v: text_content(repr(v))),
    )


async def test_a_profile_can_present_the_transport_under_its_own_name(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    ctx.tools.present_transport(_as_ipython())

    view = ctx.tools.view(DEPLOYMENT)
    assert view.transport_name == "ipython"
    # Renamed, not duplicated: two callables would be exactly the ambiguity the
    # reservation exists to prevent.
    ipython = ctx.tools.get("ipython", scope=DEPLOYMENT)
    assert ipython is not None
    assert ctx.tools.get(RUN_CODE, scope=DEPLOYMENT) is None
    assert ipython.description == "Python cells."


async def test_the_presented_name_is_what_the_model_may_call(mount: Any) -> None:
    """The C6 refusal follows the rename, or the model is told to call a name
    that no longer resolves."""
    ctx = await _code_ctx(mount)
    ctx.tools.present_transport(_as_ipython())
    ctx.tools.register(_recorder("touch", []))
    ctx.code_runtime_stub.register_program("p", lambda _b: "ok")
    session = ctx.sessions.create("s-alias")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    settled = await run_tool(ctx, "ipython", {"program": "p"}, agent=agent, session=session)
    assert settled.is_error is False

    # And the old name is gone from the model's surface — with the denial naming
    # the presented transport, so the model can correct itself (C6).
    refused = await run_tool(
        ctx, RUN_CODE, {"program": "p"}, agent=agent, session=session, call_id="c2"
    )
    assert refused.is_error is True
    assert 'presented as "ipython"' in repr(refused.content)


async def test_the_route_back_names_the_presented_transport(mount: Any) -> None:
    """A native call is refused with the SDK path; under a rename that path has
    to be the name the model was actually offered."""
    ctx = await _code_ctx(mount)
    ctx.tools.present_transport(_as_ipython())
    ctx.tools.register(_recorder("touch", []))
    session = ctx.sessions.create("s-route")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    ctx.tools.present_as("code", scope=agent.ctx)

    result = await run_tool(ctx, "touch", {}, agent=agent, session=session)
    assert result.is_error is True
    assert "from inside ipython" in repr(result.content)


async def test_the_presented_name_is_not_bound_into_its_own_namespace(mount: Any) -> None:
    """`tools.ipython` inside a cell would hand the program a way to re-enter
    the transport, which the `RUN_CODE` skip existed to prevent."""
    ctx = await _code_ctx(mount)
    ctx.tools.present_transport(_as_ipython())
    ctx.tools.register(_recorder("touch", []))
    session = ctx.sessions.create("s-bindings")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    assembly = await ctx.system_prompt.assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)
    assert "tools.touch" in text
    assert "tools.ipython" not in text
    # The code-only rule names the presented transport, and the reserved name —
    # which this profile's model has never seen — appears nowhere at all.
    assert "ipython" in text
    assert RUN_CODE not in text


async def test_the_presented_name_cannot_be_occupied_either(mount: Any) -> None:
    """Both orders fail: the name is unshadowable however the race runs."""
    ctx = await _code_ctx(mount)
    ctx.tools.present_transport(_as_ipython())
    with pytest.raises(ValueError, match="presents the Code Mode transport"):
        ctx.tools.register(_recorder("ipython", []))

    other = await _code_ctx(mount)
    other.tools.register(_recorder("ipython", []))
    with pytest.raises(ValueError, match="unshadowable"):
        other.tools.present_transport(_as_ipython())


async def test_a_scoped_presentation_cannot_be_occupied_from_a_parent_scope(mount: Any) -> None:
    """The claim-time checks are scope-local snapshots; the view build is the
    backstop that turns the ordering hole into a loud failure instead of a
    silent clobber of whichever side resolved second."""
    ctx = await _code_ctx(mount)
    session = ctx.sessions.create("s-backstop")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    ctx.tools.present_transport(_as_ipython(), scope=agent.ctx)
    # Registered globally, where no presentation is in sight — the claim-time
    # check passes, and only the agent's view knows there is a contradiction.
    ctx.tools.register(_recorder("ipython", []))
    with pytest.raises(ValueError, match="unshadowable"):
        ctx.tools.view(agent.ctx)


async def test_disposing_the_presentation_restores_the_reserved_name(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    dispose = ctx.tools.present_transport(_as_ipython())
    assert ctx.tools.view(DEPLOYMENT).transport_name == "ipython"
    dispose()
    assert ctx.tools.view(DEPLOYMENT).transport_name == RUN_CODE
    assert ctx.tools.get(RUN_CODE, scope=DEPLOYMENT) is not None


# ------------------------------------------------------- binding namespaces --
#
# P3-10's extension point. `tools` is built by the row itself and is not
# optional; every other namespace is a `register_code_namespace` claim, and the
# SDK prompt asks the same factories the run does.


def _extra_namespace(request: Any) -> CodeBindingNamespace:
    """A contributed namespace in the shape P3-10 actually needs.

    The binding the program writes (`rlm.run`) is not the tool it dispatches to
    (`spawn_child`), because a namespace cannot claim a global tool name — and it
    goes through the *bridge*, so the budgets and the dispatch records are the
    bridge's, exactly as for the `tools` namespace.
    """
    definition = SimpleNamespace(
        name="spawn_child",
        description="Start a child.",
        parameters={"type": "object", "properties": {"n": {"type": "string"}}},
    )
    binding = governed_binding(request, "run", definition, counts_as_spawn=True)
    return CodeBindingNamespace(name="rlm", description="children", bindings=(binding,))


async def test_a_row_can_contribute_a_binding_namespace(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.tools.register(_recorder("spawn_child", calls))
    ctx.tools.register_code_namespace("rlm", _extra_namespace)

    async def program(bindings: Any, _emit: Any) -> Any:
        return await bindings["rlm"].run(n="research")

    result, session = await _run(ctx, "spawn", program)
    assert result.is_error is False
    assert result.value["value"] == "spawn_child ok"
    assert calls == ["spawn_child:research"]
    # C2 holds for a contributed namespace too: the durable record names the
    # governed tool, not the binding the program wrote.
    starts = [e for e in session.events if e.type == "tool/code-dispatch-start"]
    assert [e.data["name"] for e in starts] == ["spawn_child"]


async def test_the_sdk_block_describes_the_contributed_namespace(mount: Any) -> None:
    """The prompt asks the same waterfall the run does, so the block cannot list
    a namespace the program could not reach — or omit one it can."""
    ctx = await _code_ctx(mount)
    ctx.tools.register_code_namespace("rlm", _extra_namespace)
    ctx.tools.register(_recorder("touch", []))
    session = ctx.sessions.create("s-sdk")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    assembly = await ctx.system_prompt.assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)
    assert "tools.touch" in text
    assert "rlm.run" in text


async def test_two_rows_cannot_claim_one_namespace_name(mount: Any) -> None:
    """A name conflict is a mount-time configuration fact, and it fails there —
    not per cell in a deployment that booted green."""
    ctx = await _code_ctx(mount)
    ctx.tools.register_code_namespace("rlm", _extra_namespace)
    with pytest.raises(ValueError, match="already registered"):
        ctx.tools.register_code_namespace("rlm", _extra_namespace)
    # And the one namespace the row itself contributes is not claimable.
    with pytest.raises(ValueError, match="contributed by Code Mode itself"):
        ctx.tools.register_code_namespace("tools", _extra_namespace)


async def test_a_contributed_binding_counts_against_the_spawn_budget(mount: Any) -> None:
    """C4 is enforced by the bridge, so a contributed namespace is held to the
    same budget as the one the row built itself."""
    ctx = await _code_ctx(mount, maxSubagentSpawnsPerRun=1)
    ctx.tools.register(_recorder("spawn_child", []))
    ctx.tools.register_code_namespace("rlm", _extra_namespace)

    async def program(bindings: Any, _emit: Any) -> Any:
        await bindings["rlm"].run(n="one")
        await bindings["rlm"].run(n="two")
        return "unreached"

    result, _session = await _run(ctx, "greedy", program)
    assert result.is_error is True
    assert "max_subagent_spawns_per_run=1" in repr(result.content)


async def test_a_namespace_owns_the_tools_it_presents(mount: Any) -> None:
    """One SDK route per capability.

    A tool a namespace presents stays dispatchable and stays addressable by a
    policy row — it just stops appearing in `tools` as well, because two routes
    to one capability invites the model to take the one the prompt did not
    explain. Declared on the *binding* (`CodeBinding.presents`), so the
    suppression list is the binding list and the two cannot drift.
    """
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.tools.register(_recorder("spawn_child", calls))
    ctx.tools.register(_recorder("touch", []))
    ctx.tools.register_code_namespace("rlm", _extra_namespace)
    session = ctx.sessions.create("s-owns")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    assembly = await ctx.system_prompt.assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)
    assert "rlm.run" in text
    assert "tools.spawn_child" not in text
    # An unowned tool is unaffected, and the owned one is still *callable*.
    assert "tools.touch" in text
    assert ctx.tools.get("spawn_child", scope=agent.ctx) is not None

    async def program(bindings: Any, _emit: Any) -> Any:
        return await bindings["rlm"].run(n="still works")

    result, _session = await _run(ctx, "owned", program)
    assert result.is_error is False
    assert calls == ["spawn_child:still works"]
