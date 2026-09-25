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

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import anyio
import pytest

from ph.cordis import DEPLOYMENT, Context
from ph.json import as_obj, as_seq
from ph.keys import AGENTS, CODE_RUNTIME_STUB, SESSIONS, SYSTEM_PROMPT, TOOLS
from ph.llm.types import ToolSchema
from ph.persistence import interrupted_turn_closers
from ph.seams.code_runtime import CodeBindingNamespace
from ph.session import Session, unsettled_why
from ph.session.kinds import DISPATCH_INTERRUPTED, DISPATCH_REF_KEYS
from ph.system_prompt.assembly import render_prompt
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    log_event,
    not_none,
    noting,
    parked_gate,
    raising,
    run_tool,
    simple_tool,
)
from ph.tools import Deny, ToolExecutionInput, ToolExecutionResult, text_content
from ph.tools.code_mode import (
    CodeDispatchRef,
    ToolCallError,
    governed_binding,
)
from ph.tools.definition import ToolOutput, TransportPresentation
from ph.tools.registry import RUN_CODE, ToolRuntime

pytestmark = pytest.mark.anyio

CODE_ROWS: tuple[dict[str, Any], ...] = (
    {"id": "code-runtime-stub", "name": "code-runtime-stub"},
    {"id": "tools-code-mode", "name": "tools-code-mode"},
)


async def _code_ctx(mount: MountProfile, **overrides: object) -> Context:
    rows: list[dict[str, Any]] = [
        {"insert": [dict(row) for row in CODE_ROWS]},
    ]
    if overrides:
        rows.append({"id": "tools-code-mode", "config": overrides})
    return await mount(*rows)


def _recorder(name: str, calls: list[str], *, safe: bool = True) -> Any:  # noqa: ANN401
    def body(args: Any, _run: Any) -> Any:  # noqa: ANN401
        calls.append(f"{name}:{(args or {}).get('n')}")
        return f"{name} ok"

    return simple_tool(name, body, safe=safe)


async def _run(
    ctx: Context, program_name: str, program: Callable[..., Any]
) -> tuple[ToolExecutionResult, Session]:
    ctx.require(CODE_RUNTIME_STUB).register_program(program_name, program)
    session = ctx.require(SESSIONS).create(f"s-{program_name}")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    result = await ctx.require(TOOLS).execute(
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


async def test_a_binding_call_re_enters_the_whole_pipeline(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))
    seen: list[str] = []
    ctx.on("tools/pre-execute", lambda execution, next_: noting(seen, execution.name, next_))

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    result, _session = await _run(ctx, "one-call", program)
    assert not result.is_error
    # The transport and the sub-call are both policed; the sub-call did not
    # bypass anything by virtue of being inside a cell (C1).
    assert seen == [RUN_CODE, "touch"]
    assert calls == ["touch:1"]


async def test_three_binding_calls_produce_three_durable_dispatch_pairs(
    mount: MountProfile,
) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
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


async def test_dispatch_records_stay_out_of_model_context(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(_recorder("touch", []))

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    _result, session = await _run(ctx, "log-only", program)
    assert any(event.type == "tool/code-dispatch" for event in session.events)
    # Log-only: a sub-call is a durable record, not a message. Deriving them
    # would put the same work in context twice.
    assert session.derive_messages() == ()


async def test_a_denied_binding_call_fails_the_whole_run(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))
    ctx.require(TOOLS).register(_recorder("forbidden", calls))
    ctx.require(TOOLS).guard(
        lambda execution: "policy forbids this" if execution.name == "forbidden" else None
    )
    reached: list[str] = []

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
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
    assert "was refused" in not_none(result.error).message
    # Everything after the refusal is abandoned, so partial state is bounded to
    # this one cell (C3).
    assert calls == ["touch:1"]


async def test_a_ceiling_that_ends_the_turn_is_not_the_programs_to_handle(
    mount: MountProfile,
) -> None:
    """A spent budget ends the run too, and it is not a denial (D7).

    The rule was keyed on `kind == "denied"` alone, so a refusal that ends the
    *turn* while reading as `failed` — which is what a spent tool-call ceiling
    is — arrived as an ordinary `ToolCallError` the program could `except`. It
    would then spend the rest of the turn on calls that could only fail, and the
    posture meant to surface a breach most loudly was the one a generated
    program could swallow.

    Read off `concludes_turn` rather than a second kind, because that is the
    flag the loop itself stops on: whatever ends the turn out there ends the run
    in here, without this needing a list of the rows that can do it.

    Sabotage: drop the `result.concludes_turn` branch in `DispatchBridge` and
    `touch:3` runs.
    """
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))
    ctx.require(TOOLS).register(_recorder("metered", calls))
    ctx.on(
        "tools/pre-execute",
        lambda execution, next_: (
            Deny(reason="call budget spent", concludes_turn=True, failure_kind="failed")
            if execution.name == "metered"
            else next_(execution)
        ),
    )

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        try:
            await ns["tools"].metered(n=2)
        except Exception:
            await ns["tools"].touch(n=3)
        return "done"

    result, _session = await _run(ctx, "budget", program)
    assert result.is_error
    # A budget, not policy: nothing judged the call, and the model reads that.
    assert not_none(result.error).kind == "failed"
    assert "stopped the turn" in not_none(result.error).message
    assert calls == ["touch:1"], "the program continued past a budget it had spent"


async def test_a_ceiling_inside_a_program_ends_the_outer_turn_too(
    mount: MountProfile,
) -> None:
    """C13 — the program stopped and the turn did not (D7's promise, half kept).

    `run_code` answers with an ordinary *successful* cell value, so a ceiling
    that ended the turn inside a program left `concludes_turn` false on the one
    result the loop actually reads. The turn ran on, the model spent a step, and
    the gate denied its next call — D7's "a spent budget ends the turn", off by
    one model call, in the execution mode where most calls happen.

    `ToolRunContext.conclude_turn` is the loop's own way to say a successful
    result is terminal, and this is its first production caller.

    Sabotage: drop the `bridge.concluded_turn` arm in `run_code` and the result
    comes back with `concludes_turn` unset.
    """
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("metered", calls))
    ctx.on(
        "tools/pre-execute",
        lambda execution, next_: (
            Deny(reason="call budget spent", concludes_turn=True, failure_kind="failed")
            if execution.name == "metered"
            else next_(execution)
        ),
    )

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].metered(n=1)
        return "unreachable"

    result, _session = await _run(ctx, "ceiling", program)

    assert result.concludes_turn, "the turn ran on after a ceiling ended it inside the cell"
    assert calls == []


async def test_a_failed_binding_call_is_the_programs_to_handle(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(simple_tool("flaky", raising(RuntimeError("transient"))))
    handled: list[str] = []

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
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


async def test_a_runaway_program_fails_at_its_budget(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount, maxDispatchesPerRun=4)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        for index in range(100):
            await ns["tools"].touch(n=index)
        return "done"

    result, _session = await _run(ctx, "runaway", program)
    assert result.is_error
    # The budget is named, so the model can split the work rather than guess.
    assert "max_dispatches_per_run=4" in not_none(result.error).message
    assert len(calls) == 4


async def test_a_model_direct_native_call_is_refused_under_code_mode(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(_recorder("touch", []))
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    ctx.require(TOOLS).present_as("code", scope=agent.ctx)
    seen: list[str] = []
    ctx.on("tools/pre-execute", lambda execution, next_: noting(seen, execution.name, next_))

    result = await ctx.require(TOOLS).execute(
        ToolExecutionInput(call_id="c", name="touch", arguments={}, scope=agent.ctx, agent=agent)
    )
    assert result.is_error
    assert "await tools.touch(...)" in not_none(result.error).message
    # Resolved before policy, so a permissive row cannot allow a name the prompt
    # never offered (C6).
    assert seen == []


async def test_the_sdk_section_lists_the_bindings_and_the_code_only_rule(
    mount: MountProfile,
) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(_recorder("touch", []))
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    ctx.require(TOOLS).present_as("code", scope=agent.ctx)

    from ph.system_prompt.assembly import render_prompt

    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    prompt = render_prompt(assembly)
    assert "async def tools.touch(" in prompt
    assert f"only\n`{RUN_CODE}` is callable" in prompt or RUN_CODE in prompt
    # Under Code Mode the model is offered one callable, not a schema list.
    assert ctx.require(TOOLS).schemas(scope=agent.ctx) == []


async def test_the_transport_name_cannot_be_shadowed(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    assert ctx.require(TOOLS).get(RUN_CODE, scope=DEPLOYMENT) is not None
    with pytest.raises(ValueError, match="reserved"):
        ctx.require(TOOLS).register(_recorder(RUN_CODE, []))


async def test_an_oversized_dispatch_can_be_reshaped_before_it_is_logged(
    mount: MountProfile,
) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(_recorder("touch", []))

    async def shrink(record: object, content: object, next_: object) -> Any:  # noqa: ANN401
        return text_content("[spilled]")

    ctx.on("tools/code-dispatch-log", shrink)

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    _result, session = await _run(ctx, "spill", program)
    (settle,) = [e for e in session.events if e.type == "tool/code-dispatch"]
    # Each dispatch is offloadable individually (C5), so one large read does not
    # melt its siblings.
    assert as_obj(as_seq(settle.data["content"])[0])["text"] == "[spilled]"


async def test_a_spawning_binding_is_held_to_the_spawn_budget(mount: MountProfile) -> None:
    """Counted by a property the binding declares, not by guessing from its name."""
    from ph.cancel import CancelToken
    from ph.seams.code_runtime import CodeBinding
    from ph.tools.code_mode import CodeRunFailure, DispatchBridge

    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(_recorder("spawn", []))
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    run = ctx.require(TOOLS).create_execution(
        ToolExecutionInput(
            call_id="root", name=RUN_CODE, arguments={"program": "x"}, scope=agent.ctx, agent=agent
        )
    )
    bridge = DispatchBridge(
        tools=ctx.require(TOOLS),
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


async def test_a_pre_execute_denial_of_a_sub_call_also_fails_the_run(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register(_recorder("touch", []))
    ctx.on(
        "tools/pre-execute",
        lambda execution, next_: (
            Deny(reason="not now") if execution.parent is not None else next_()
        ),
    )

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    result, session = await _run(ctx, "pre-denied", program)
    assert result.is_error
    assert "not now" in not_none(result.error).message
    # A refusal is still an opened-and-settled pair (P7-15): the pipeline writes the
    # start for every call its gate decides. Repair's count for a kind it lacks is
    # exact only because nothing writes a settle with no opening.
    types = [e.type for e in session.events if e.type.startswith("tool/code-dispatch")]
    assert types == ["tool/code-dispatch-start", "tool/code-dispatch"]


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


async def test_a_profile_can_present_the_transport_under_its_own_name(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).present_transport(_as_ipython())

    view = ctx.require(TOOLS).view(DEPLOYMENT)
    assert view.transport_name == "ipython"
    # Renamed, not duplicated: two callables would be exactly the ambiguity the
    # reservation exists to prevent.
    ipython = ctx.require(TOOLS).get("ipython", scope=DEPLOYMENT)
    assert ipython is not None
    assert ctx.require(TOOLS).get(RUN_CODE, scope=DEPLOYMENT) is None
    assert ipython.description == "Python cells."


async def test_renaming_the_transport_keeps_what_it_declares_about_itself(
    mount: MountProfile,
) -> None:
    """**P6-16's motivating case, pinned where it would silently break.**

    The transport declares `is_irreversible` — a cell is model-authored raw
    Python and may do anything — and `hitl` reads that declaration to decide
    whether a call is worth asking a person about. It reads it *through the
    view*, under whatever name the profile presents, so the whole capability
    gate for the surface that most needs one rides on the rename carrying the
    field.

    `rename` uses `dataclasses.replace` and so carries it by construction, which
    is precisely why this is worth a test: the safe behavior here is the
    *absence* of an explicit field list, and the natural way to break it is for
    someone to make `rename` construct a `ToolDefinition` by hand. That would
    drop the declaration, raise nothing, and turn the gate off.
    """
    ctx = await _code_ctx(mount)
    before = ctx.require(TOOLS).get(RUN_CODE, scope=DEPLOYMENT)
    assert before is not None and before.irreversible({"program": "anything"})

    ctx.require(TOOLS).present_transport(_as_ipython())

    after = ctx.require(TOOLS).get("ipython", scope=DEPLOYMENT)
    assert after is not None
    assert after.irreversible({"program": "anything"}), (
        "the declaration must travel with the tool, not with the name it was registered under"
    )


async def test_the_presented_name_is_what_the_model_may_call(mount: MountProfile) -> None:
    """The C6 refusal follows the rename, or the model is told to call a name
    that no longer resolves."""
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).present_transport(_as_ipython())
    ctx.require(TOOLS).register(_recorder("touch", []))
    ctx.require(CODE_RUNTIME_STUB).register_program("p", lambda _b: "ok")
    session = ctx.require(SESSIONS).create("s-alias")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    settled = await run_tool(ctx, "ipython", {"program": "p"}, agent=agent, session=session)
    assert settled.is_error is False

    # And the old name is gone from the model's surface — with the denial naming
    # the presented transport, so the model can correct itself (C6).
    refused = await run_tool(
        ctx, RUN_CODE, {"program": "p"}, agent=agent, session=session, call_id="c2"
    )
    assert refused.is_error is True
    assert 'presented as "ipython"' in repr(refused.content)


async def test_the_route_back_names_the_presented_transport(mount: MountProfile) -> None:
    """A native call is refused with the SDK path; under a rename that path has
    to be the name the model was actually offered."""
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).present_transport(_as_ipython())
    ctx.require(TOOLS).register(_recorder("touch", []))
    session = ctx.require(SESSIONS).create("s-route")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    ctx.require(TOOLS).present_as("code", scope=agent.ctx)

    result = await run_tool(ctx, "touch", {}, agent=agent, session=session)
    assert result.is_error is True
    assert "from inside ipython" in repr(result.content)


async def test_the_presented_name_is_not_bound_into_its_own_namespace(mount: MountProfile) -> None:
    """`tools.ipython` inside a cell would hand the program a way to re-enter
    the transport, which the `RUN_CODE` skip existed to prevent."""
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).present_transport(_as_ipython())
    ctx.require(TOOLS).register(_recorder("touch", []))
    session = ctx.require(SESSIONS).create("s-bindings")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)
    assert "tools.touch" in text
    assert "tools.ipython" not in text
    # The code-only rule names the presented transport, and the reserved name —
    # which this profile's model has never seen — appears nowhere at all.
    assert "ipython" in text
    assert RUN_CODE not in text


async def test_the_presented_name_cannot_be_occupied_either(mount: MountProfile) -> None:
    """Both orders fail: the name is unshadowable however the race runs."""
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).present_transport(_as_ipython())
    with pytest.raises(ValueError, match="presents the Code Mode transport"):
        ctx.require(TOOLS).register(_recorder("ipython", []))

    other = await _code_ctx(mount)
    other.require(TOOLS).register(_recorder("ipython", []))
    with pytest.raises(ValueError, match="unshadowable"):
        other.require(TOOLS).present_transport(_as_ipython())


async def test_a_scoped_presentation_cannot_be_occupied_from_a_parent_scope(
    mount: MountProfile,
) -> None:
    """The claim-time checks are scope-local snapshots; the view build is the
    backstop that turns the ordering hole into a loud failure instead of a
    silent clobber of whichever side resolved second."""
    ctx = await _code_ctx(mount)
    session = ctx.require(SESSIONS).create("s-backstop")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    ctx.require(TOOLS).present_transport(_as_ipython(), scope=agent.ctx)
    # Registered globally, where no presentation is in sight — the claim-time
    # check passes, and only the agent's view knows there is a contradiction.
    ctx.require(TOOLS).register(_recorder("ipython", []))
    with pytest.raises(ValueError, match="unshadowable"):
        ctx.require(TOOLS).view(agent.ctx)


async def test_disposing_the_presentation_restores_the_reserved_name(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    dispose = ctx.require(TOOLS).present_transport(_as_ipython())
    assert ctx.require(TOOLS).view(DEPLOYMENT).transport_name == "ipython"
    dispose()
    assert ctx.require(TOOLS).view(DEPLOYMENT).transport_name == RUN_CODE
    assert ctx.require(TOOLS).get(RUN_CODE, scope=DEPLOYMENT) is not None


# ------------------------------------------------------- binding namespaces --
#
# P3-10's extension point. `tools` is built by the row itself and is not
# optional; every other namespace is a `register_code_namespace` claim, and the
# SDK prompt asks the same factories the run does.


def _extra_namespace(request: Any) -> CodeBindingNamespace:  # noqa: ANN401
    """A contributed namespace in the shape P3-10 actually needs.

    The binding the program writes (`rlm.run`) is not the tool it dispatches to
    (`spawn_child`), because a namespace cannot claim a global tool name — and it
    goes through the *bridge*, so the budgets and the dispatch records are the
    bridge's, exactly as for the `tools` namespace.
    """
    definition = ToolSchema(
        name="spawn_child",
        description="Start a child.",
        parameters={"type": "object", "properties": {"n": {"type": "string"}}},
    )
    binding = governed_binding(request, "run", definition, counts_as_spawn=True)
    return CodeBindingNamespace(name="rlm", description="children", bindings=(binding,))


async def test_a_row_can_contribute_a_binding_namespace(mount: MountProfile) -> None:
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("spawn_child", calls))
    ctx.require(TOOLS).register_code_namespace("rlm", _extra_namespace)

    async def program(bindings: Mapping[str, Any], _emit: Callable[[str], None]) -> Any:  # noqa: ANN401
        return await bindings["rlm"].run(n="research")

    result, session = await _run(ctx, "spawn", program)
    assert result.is_error is False
    assert result.value["value"] == "spawn_child ok"
    assert calls == ["spawn_child:research"]
    # C2 holds for a contributed namespace too: the durable record names the
    # governed tool, not the binding the program wrote.
    starts = [e for e in session.events if e.type == "tool/code-dispatch-start"]
    assert [e.data["name"] for e in starts] == ["spawn_child"]


async def test_the_sdk_block_describes_the_contributed_namespace(mount: MountProfile) -> None:
    """The prompt asks the same waterfall the run does, so the block cannot list
    a namespace the program could not reach — or omit one it can."""
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register_code_namespace("rlm", _extra_namespace)
    ctx.require(TOOLS).register(_recorder("touch", []))
    session = ctx.require(SESSIONS).create("s-sdk")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)
    assert "tools.touch" in text
    assert "rlm.run" in text


async def test_two_rows_cannot_claim_one_namespace_name(mount: MountProfile) -> None:
    """A name conflict is a mount-time configuration fact, and it fails there —
    not per cell in a deployment that booted green."""
    ctx = await _code_ctx(mount)
    ctx.require(TOOLS).register_code_namespace("rlm", _extra_namespace)
    with pytest.raises(ValueError, match="already registered"):
        ctx.require(TOOLS).register_code_namespace("rlm", _extra_namespace)
    # And the one namespace the row itself contributes is not claimable.
    with pytest.raises(ValueError, match="contributed by Code Mode itself"):
        ctx.require(TOOLS).register_code_namespace("tools", _extra_namespace)


async def test_a_contributed_binding_counts_against_the_spawn_budget(mount: MountProfile) -> None:
    """C4 is enforced by the bridge, so a contributed namespace is held to the
    same budget as the one the row built itself."""
    ctx = await _code_ctx(mount, maxSubagentSpawnsPerRun=1)
    ctx.require(TOOLS).register(_recorder("spawn_child", []))
    ctx.require(TOOLS).register_code_namespace("rlm", _extra_namespace)

    async def program(bindings: Mapping[str, Any], _emit: Callable[[str], None]) -> str:
        await bindings["rlm"].run(n="one")
        await bindings["rlm"].run(n="two")
        return "unreached"

    result, _session = await _run(ctx, "greedy", program)
    assert result.is_error is True
    assert "max_subagent_spawns_per_run=1" in repr(result.content)


async def test_a_namespace_owns_the_tools_it_presents(mount: MountProfile) -> None:
    """One SDK route per capability.

    A tool a namespace presents stays dispatchable and stays addressable by a
    policy row — it just stops appearing in `tools` as well, because two routes
    to one capability invites the model to take the one the prompt did not
    explain. Declared on the *binding* (`CodeBinding.presents`), so the
    suppression list is the binding list and the two cannot drift.
    """
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("spawn_child", calls))
    ctx.require(TOOLS).register(_recorder("touch", []))
    ctx.require(TOOLS).register_code_namespace("rlm", _extra_namespace)
    session = ctx.require(SESSIONS).create("s-owns")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)
    assert "rlm.run" in text
    assert "tools.spawn_child" not in text
    # An unowned tool is unaffected, and the owned one is still *callable*.
    assert "tools.touch" in text
    assert ctx.require(TOOLS).get("spawn_child", scope=agent.ctx) is not None

    async def program(bindings: Mapping[str, Any], _emit: Callable[[str], None]) -> Any:  # noqa: ANN401
        return await bindings["rlm"].run(n="still works")

    result, _session = await _run(ctx, "owned", program)
    assert result.is_error is False
    assert calls == ["spawn_child:still works"]


async def test_a_dispatch_and_its_settle_are_paired_by_one_declaration(mount: MountProfile) -> None:
    """Both records carry the same identity, spelled by the same type.

    Readers pair them by these four fields and nothing else: the TUI adapter
    finds the parent card by `parentCallId` and the dispatch card by
    `subCallId`, and `/revert` reads `parentCallId` and `name` off the start to
    list what a run did outside the tree it restored.

    The start record used to be a hand-written camelCase dict while the settle
    went through `to_wire()` — four fields spelled twice. Renaming one, or
    changing `wire_alias`, would have moved the settle record and left the start
    record's literals behind: every reader unpaired at once, and nothing failing
    to say so. `CodeDispatchRef` is the one declaration both now go through.

    Sabotage: hand-write any of the four keys in `_log_start` and the key sets
    stop matching.
    """
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    _result, session = await _run(ctx, "paired", program)
    (start,) = [e for e in session.events if e.type == "tool/code-dispatch-start"]
    (settle,) = [e for e in session.events if e.type == "tool/code-dispatch"]
    identity = {info.alias or name for name, info in CodeDispatchRef.model_fields.items()}

    assert set(start.data) == identity | {"arguments"}
    assert set(settle.data) == identity | {"isError", "content"}
    # And they agree field for field, which is what a reader pairs on.
    assert {key: start.data[key] for key in identity} == {key: settle.data[key] for key in identity}


def test_the_leaf_spells_the_dispatch_identity_as_the_model_does() -> None:
    """T4. The settle repair writes pairs by `DISPATCH_REF_KEYS`, which `ph.session.kinds`
    spells because a leaf cannot import this model — so the two are held equal here.

    Sabotage: rename a `CodeDispatchRef` field, and repair's settle of a crashed
    dispatch would stop pairing with its start while every live test kept passing.
    """
    model = tuple(info.alias or name for name, info in CodeDispatchRef.model_fields.items())
    assert model == DISPATCH_REF_KEYS


async def test_a_dispatch_parked_on_its_gate_has_no_start_record(mount: MountProfile) -> None:
    """The second instance of P7-15: `tool/code-dispatch-start` is a write-ahead too.

    A binding re-enters the pipeline (C1), so a dispatch can park on a human
    exactly as a model-direct call can — and `/revert` reads the start records as
    the list of what the program *did*. A start written before the gate said the
    program did something while a person was still deciding whether it could.
    """
    ctx = await _code_ctx(mount)
    calls: list[str] = []
    ctx.require(TOOLS).register(_recorder("touch", calls))
    reached, release = parked_gate(ctx, only="touch")

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    # `_run` creates its own session; this needs the handle mid-flight.
    ctx.require(CODE_RUNTIME_STUB).register_program("parked", program)
    session = ctx.require(SESSIONS).create("s-parked")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(
            partial(run_tool, ctx, RUN_CODE, {"program": "parked"}, agent=agent, session=session)
        )
        await reached.wait()
        starts = [e for e in session.events if e.type == "tool/code-dispatch-start"]
        assert starts == [], "parked: the program has not done this yet"
        release.set()

    kinds = [e.type for e in session.events]
    assert kinds.index("tool/code-dispatch-start") < kinds.index("tool/code-dispatch")
    assert calls == ["touch:1"]


async def test_an_orphaned_dispatch_is_settled_by_repair(mount: MountProfile) -> None:
    """P10-11. The log a crash mid-dispatch leaves — the start record written, the
    binding running, no settle — is the log the tool body sees, so it is taken
    there. Repair settles the dispatch by its own kind's closer: the start's
    identity, an error, and a body a card can draw, so no dispatch card runs
    forever after a crash."""
    ctx = await _code_ctx(mount)
    crashed: list[Any] = []

    def body(args: Any, run: Any) -> Any:  # noqa: ANN401
        crashed.extend(run.session.events)
        return "ok"

    ctx.require(TOOLS).register(simple_tool("touch", body, safe=True))

    async def program(ns: Mapping[str, Any], emit: Callable[[str], None]) -> str:
        await ns["tools"].touch(n=1)
        return "done"

    await _run(ctx, "orphaned", program)
    (start,) = [event for event in crashed if event.type == "tool/code-dispatch-start"]
    assert not [event for event in crashed if event.type == "tool/code-dispatch"]

    (settle,) = interrupted_turn_closers(crashed)
    assert settle.type == "tool/code-dispatch"
    identity = ("rootCallId", "parentCallId", "subCallId", "name")
    assert [settle.data[key] for key in identity] == [start.data[key] for key in identity]
    assert settle.data["isError"] is True and unsettled_why(settle.data) == "outcome-unknown"
    assert as_obj(as_seq(settle.data["content"])[0])["text"] == DISPATCH_INTERRUPTED


async def test_revert_still_lists_an_orphaned_dispatch_as_not_undone() -> None:
    """`/revert` reads the start records, so a dispatch repair settled is still one
    the run did — and one a tree restore does not take back."""
    from ph.commands.revert import _not_undone

    root = Context()
    tools = ToolRuntime(ctx=root)
    root.provide(TOOLS, tools)
    session = Session("s")
    log_event(
        session,
        "tool/code-dispatch-start",
        {
            "rootCallId": "c1",
            "parentCallId": "c1",
            "subCallId": "c1:code:0",
            "name": "publish",
            "arguments": {"to": "prod"},
        },
    )
    for closer in interrupted_turn_closers(session.events):
        session.admit(closer)

    assert session.latest("tool/code-dispatch") is not None, "repair settled it"
    listed = "\n".join(_not_undone(root, root, session, "c1"))
    assert "publish(" in listed
