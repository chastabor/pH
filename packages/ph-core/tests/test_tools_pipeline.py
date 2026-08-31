"""P1-02 — the pipeline, in order, with every gate.

Gates: *dsh's ordering invariants; a guard denial survives a later listener; a
crashing body leaves its `tool/call` and yields `is_error`.*

The order is `tools/pre-execute` → approval on `ask` → **guards** →
`tools/execute` (around) → body → `tools/post-execute` → normalize →
`finalize_content` → `tools/result`. Guards run *last* and are deny-only, which
is what "monotonic" means: policy that must not be reorderable stays a guard
rather than a listener, and no amount of listener ordering — or a human's
approval — turns its denial back into permission.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.llm.types import create_user_message
from ph.testing import StubAgent, boundary_for, raising, simple_tool, tool_runtime
from ph.tools import (
    Accept,
    Allow,
    Ask,
    Block,
    Deny,
    Respond,
    ToolExecutionInput,
    ToolNotFoundError,
    ToolOutput,
    define_tool,
    text_content,
)
from ph.tools.definition import ToolExecutionResult

pytestmark = pytest.mark.anyio


def _echo(**kwargs: Any) -> Any:
    return simple_tool("echo", lambda args, _run: (args or {}).get("text", ""), **kwargs)


def _call(name: str = "echo", *, agent: Any = None, **arguments: Any) -> ToolExecutionInput:
    return ToolExecutionInput(
        call_id="call-1",
        name=name,
        arguments=arguments,
        agent=agent,
        # One resolver, shared with `run_tool` — a second spelling here drifted
        # once already (a truthiness guard where the helper checks the type).
        scope=boundary_for(None, agent),
    )


async def test_the_happy_path_renders_the_declared_value() -> None:
    _root, tools = tool_runtime()
    tools.register(_echo())
    result = await tools.execute(_call(text="hello"))
    assert not result.is_error
    assert result.value == "hello"
    assert result.content[0].text == "hello"


async def test_stages_run_in_the_documented_order() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())
    order: list[str] = []

    async def pre(execution: Any, next_: Any) -> Any:
        order.append("pre")
        return await next_()

    def guard(execution: Any) -> None:
        order.append("guard")
        return None

    async def around(execution: Any, next_: Any) -> Any:
        order.append("execute:before")
        result = await next_()
        order.append("execute:after")
        return result

    async def post(execution: Any, result: Any, next_: Any) -> Any:
        order.append("post")
        return await next_()

    root.on("tools/pre-execute", pre)
    tools.guard(guard)
    root.on("tools/execute", around)
    root.on("tools/post-execute", post)
    root.on("tools/result", lambda execution, result: order.append("result"))

    await tools.execute(_call(text="x"))
    assert order == ["pre", "guard", "execute:before", "execute:after", "post", "result"]


async def test_a_guard_denial_cannot_be_re_permitted() -> None:
    root, tools = tool_runtime()
    ran: list[str] = []
    tools.register(_echo())
    tools.guard(lambda execution: "policy forbids this")

    async def permissive(execution: Any, next_: Any) -> Any:
        # Registered after the guard and returning `allow` anyway.
        ran.append("listener")
        return Allow()

    root.on("tools/pre-execute", permissive)
    root.on("tools/execute", lambda execution, next_: ran.append("body") or next_())

    result = await tools.execute(_call(text="x"))
    assert result.is_error
    assert result.error is not None
    assert "policy forbids this" in result.error.message
    # The guard has no allow result, so ordering cannot undo it — and the body
    # never ran.
    assert "body" not in ran


async def test_a_denial_is_a_denial_and_a_crash_is_a_failure() -> None:
    """`ToolFailure.kind` is the fact Code Mode branches on (C3); it must be set
    by the producer, not guessed downstream."""
    _root, tools = tool_runtime()
    tools.register(_echo())
    tools.register(simple_tool("boom", raising(RuntimeError("broke"))))
    tools.guard(lambda execution: "no" if execution.name == "echo" else None)

    denied = await tools.execute(_call(text="x"))
    failed = await tools.execute(_call("boom"))
    assert denied.error is not None and denied.error.kind == "denied"
    assert failed.error is not None and failed.error.kind == "failed"
    # An unknown tool is policy refusing too: the name was never offered.
    unknown = await tools.execute(_call("nonexistent"))
    assert unknown.error is not None and unknown.error.kind == "denied"


async def test_guards_run_after_approval_so_they_have_the_last_word() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())
    seen: list[str] = []

    class Approver:
        async def request(self, **kwargs: Any) -> str:
            seen.append("approved")
            return "allowed-once"

    root.provide("approval", Approver())
    tools.guard(lambda execution: seen.append("guard") or "still no")
    root.on("tools/pre-execute", lambda execution, next_: Ask(reason="dangerous"))

    result = await tools.execute(_call(text="x", agent=StubAgent(root)))
    # The human said yes and the guard still refused: that is the point of
    # having a monotonic guard rather than another listener.
    assert seen == ["approved", "guard"]
    assert result.is_error
    assert result.error is not None and "still no" in result.error.message


async def test_a_pre_execute_denial_skips_the_guards() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())
    guarded: list[str] = []
    tools.guard(lambda execution: guarded.append("guard") or None)
    root.on("tools/pre-execute", lambda execution, next_: Deny(reason="no"))
    result = await tools.execute(_call(text="x"))
    assert result.is_error
    # Nothing left to decide: asking a guard to confirm a denial would only
    # create a chance to re-permit it.
    assert guarded == []


async def test_a_pre_execute_row_can_answer_in_the_tools_own_voice() -> None:
    """`Respond` (P4-05) is the pipeline's own decision, not approval's.

    The human answering "you don't need that, the port is 8080" is the case that
    motivated it, but the capability is *short-circuit this call with a result*
    — which a cache row, a replay row, and a "disabled here, do this instead"
    row all want. Wired to one consumer it would have been reinvented by the
    next; and it is a **successful** result, because the model asked a question
    and got one.
    """
    root, tools = tool_runtime()
    ran: list[str] = []
    tools.register(simple_tool("echo", lambda _args, _run: ran.append("body") or "ran"))
    root.on("tools/pre-execute", lambda execution, next_: Respond(message="8080"))

    result = await tools.execute(_call("echo", text="x"))
    assert not result.is_error, "an answer is not a failure"
    assert result.content[0].text == "8080"
    assert ran == [], "the body ran anyway"


async def test_a_pre_execute_row_can_substitute_the_arguments() -> None:
    """`Allow(arguments=…)` — the correction path, applied by the pipeline.

    The substitution lands on the decision rather than on `execution.arguments`
    directly, because the run belongs to the pipeline: a listener assigning to it
    would be rewriting the one field every listener before it has already been
    handed. `tool/call` is untouched either way — it recorded what the *model*
    asked for, and attributing a human's arguments to the model is the falsehood
    this codebase refuses everywhere.
    """
    root, tools = tool_runtime()
    seen: list[Any] = []
    tools.register(simple_tool("echo", lambda args, _run: seen.append(dict(args)) or "ran"))
    root.on(
        "tools/pre-execute",
        lambda execution, next_: Allow(arguments={"text": "corrected"}, has_arguments=True),
    )

    result = await tools.execute(_call("echo", text="original"))
    assert not result.is_error
    assert seen == [{"text": "corrected"}]


async def test_ask_without_an_approval_service_denies() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())
    root.on("tools/pre-execute", lambda execution, next_: Ask())
    result = await tools.execute(_call(text="x", agent=StubAgent(root)))
    assert result.is_error
    # Absence is not consent.
    assert result.error is not None and "requires approval" in result.error.message


async def test_ask_with_no_agent_denies_for_lack_of_anywhere_to_ask() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())

    class Approver:
        async def request(self, **kwargs: Any) -> str:  # pragma: no cover - never reached
            return "allowed-once"

    root.provide("approval", Approver())
    root.on("tools/pre-execute", lambda execution, next_: Ask())
    result = await tools.execute(_call(text="x"))
    # No agent means no session to audit to and no UI to route to, so there is
    # nothing to ask — which denies rather than proceeding.
    assert result.is_error
    assert result.error is not None and "no agent to route it through" in result.error.message


@pytest.mark.parametrize(
    ("outcome", "fragment"),
    [
        ("rejected", "the user rejected"),
        ("cancelled", "was cancelled"),
        ("unavailable", "no approval channel"),
    ],
)
async def test_every_non_grant_denies_with_its_own_reason(outcome: str, fragment: str) -> None:
    root, tools = tool_runtime()
    tools.register(_echo())

    class Approver:
        async def request(self, **kwargs: Any) -> str:
            return outcome

    root.provide("approval", Approver())
    root.on("tools/pre-execute", lambda execution, next_: Ask())
    result = await tools.execute(_call(text="x", agent=StubAgent(root)))
    assert result.is_error
    # Distinct reasons: a model must be able to tell a human's "no" from a
    # missing channel, because only one of them is worth re-planning around.
    assert result.error is not None and fragment in result.error.message


async def test_a_raising_body_becomes_a_structured_error() -> None:
    _root, tools = tool_runtime()
    tools.register(simple_tool("boom", raising(RuntimeError("the tool broke"))))
    result = await tools.execute(_call("boom"))
    assert result.is_error
    assert result.error is not None and result.error.message == "the tool broke"
    # Content carries the Native envelope the model expects, so a failure reads
    # the same as every other result.
    assert result.content[0].text == "Error: the tool broke"


async def test_an_unknown_tool_is_refused_before_any_listener() -> None:
    root, tools = tool_runtime()
    seen: list[str] = []
    root.on("tools/pre-execute", lambda execution, next_: seen.append("pre") or next_())
    result = await tools.execute(_call("nonexistent"))
    assert result.is_error
    assert result.error is not None and "unknown tool" in result.error.message
    # A name the prompt never offered cannot be observed by policy, let alone
    # allowed by a permissive row (C6).
    assert seen == []


async def test_a_native_call_under_code_mode_names_the_route_back() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())
    agent = StubAgent(root.scope("agent:a"))
    tools.present_as("code", scope=agent.ctx)

    result = await tools.execute(_call(agent=agent))
    assert result.is_error
    assert result.error is not None and "await tools.echo(...)" in result.error.message


async def test_a_value_violating_the_output_schema_fails_the_call() -> None:
    _root, tools = tool_runtime()
    tools.register(
        define_tool(
            "liar",
            "returns the wrong type",
            parameters={"type": "object", "properties": {}},
            output=ToolOutput(schema={"type": "string"}, render=lambda _a, v: text_content(str(v))),
            execute=lambda _args, _run: {"not": "a string"},
        )
    )
    result = await tools.execute(_call("liar"))
    assert result.is_error
    assert result.error is not None
    assert result.error.info == {"name": "ToolOutputError", "code": "INVALID_TOOL_OUTPUT"}


async def test_post_execute_can_replace_content_or_block() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())

    async def replace(execution: Any, result: Any, next_: Any) -> Any:
        return Accept(content=text_content("rewritten"))

    disposer = root.on("tools/post-execute", replace)
    result = await tools.execute(_call(text="original"))
    assert result.content[0].text == "rewritten"
    assert not result.is_error
    disposer()

    root.on(
        "tools/post-execute",
        lambda execution, result, next_: Block(feedback=text_content("try again")),
    )
    blocked = await tools.execute(_call(text="original"))
    assert blocked.is_error
    assert blocked.content[0].text == "try again"


async def test_post_execute_runs_on_a_denied_call() -> None:
    root, tools = tool_runtime()
    tools.register(_echo())
    seen: list[str] = []
    root.on("tools/pre-execute", lambda execution, next_: Deny(reason="nope"))
    root.on("tools/post-execute", lambda execution, result, next_: seen.append("post") or next_())
    await tools.execute(_call(text="x"))
    # A denial is still a result, and policy may still shape what the model reads.
    assert seen == ["post"]


async def test_finalize_content_runs_even_for_a_failure() -> None:
    _root, tools = tool_runtime()
    seen: list[bool] = []

    def finalize(execution: Any, result: ToolExecutionResult) -> Any:
        seen.append(result.is_error)
        return text_content("finalized")

    tools.register(simple_tool("boom", raising(RuntimeError("x")), finalize_content=finalize))
    result = await tools.execute(_call("boom"))
    # Invoked exactly once, for the failure that bypassed post-execute: a tool
    # whose content needs a last-mile transform must not have to trust policy.
    assert seen == [True]
    assert result.content[0].text == "finalized"


def _notice() -> Any:
    return create_user_message(
        content=[{"type": "text", "text": "a notice"}], source={"kind": "plugin", "plugin": "test"}
    )


async def test_deferred_context_rides_the_result() -> None:
    _root, tools = tool_runtime()

    def body(_args: Any, run: Any) -> Any:
        run.defer_context(_notice())
        run.conclude_turn()
        return "done"

    tools.register(simple_tool("defer", body))
    result = await tools.execute(_call("defer"))
    assert len(result.additional_contexts) == 1
    assert result.concludes_turn is True


async def test_a_block_discards_context_the_body_deferred() -> None:
    root, tools = tool_runtime()

    def body(_args: Any, run: Any) -> Any:
        run.defer_context(_notice())
        return "done"

    tools.register(simple_tool("defer", body))
    root.on(
        "tools/post-execute",
        lambda execution, result, next_: Block(feedback=text_content("blocked")),
    )
    result = await tools.execute(_call("defer"))
    # The context belonged to an outcome that no longer stands.
    assert result.additional_contexts == ()


async def test_arguments_reaching_the_body_are_frozen() -> None:
    _root, tools = tool_runtime()
    captured: list[Any] = []
    tools.register(simple_tool("capture", lambda args, _run: captured.append(args) or "ok"))
    await tools.execute(_call("capture", nested={"a": 1}))
    with pytest.raises(TypeError):
        captured[0]["nested"] = 2


async def test_a_timeout_budget_is_enforced_by_its_row(mount: Any) -> None:
    """`timeout_ms` is a promise the `tools-timeout` row keeps; a declared
    budget with nothing behind it would be the type telling a lie."""
    import anyio

    ctx = await mount()

    async def slow(_args: Any, _run: Any) -> str:
        await anyio.sleep(1.0)
        return "late"

    ctx.tools.register(simple_tool("slow", slow, timeout_ms=50))
    result = await ctx.tools.execute(_call("slow"))
    assert result.is_error
    assert result.error is not None
    assert result.error.info == {"name": "Timeout", "code": "TIMEOUT"}
    assert "50 ms" in result.error.message


async def test_unknown_tool_error_is_routable() -> None:
    error = ToolNotFoundError("edit", "call it through run_code")
    assert error.code == "UNKNOWN_TOOL"
    assert error.failure_kind == "denied"
    assert "call it through run_code" in str(error)
