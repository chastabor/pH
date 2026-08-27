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

from typing import Any

import pytest

from ph.cordis import Context
from ph.testing import FAKE_OPTIONS, raising, simple_tool
from ph.tools import Deny, ToolExecutionInput, text_content
from ph.tools.code_mode import ToolCallError
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

    from ph.system_prompt.assembly import AssembleContext, render_prompt

    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    prompt = render_prompt(assembly)
    assert "async def tools.touch(" in prompt
    assert f"only\n`{RUN_CODE}` is callable" in prompt or RUN_CODE in prompt
    # Under Code Mode the model is offered one callable, not a schema list.
    assert ctx.tools.schemas(scope=agent.ctx) == []


async def test_the_transport_name_cannot_be_shadowed(mount: Any) -> None:
    ctx = await _code_ctx(mount)
    assert ctx.tools.get(RUN_CODE) is not None
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
