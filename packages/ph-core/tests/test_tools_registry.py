"""P1-01 — the registry: registration, shadowing, restrictions, visibility.

Gate: *a restricted-away global is absent to that scope and present to others.*

The asymmetry is the feature (B7). One agent seeing a different tool set than
another is how presets, subagent surfaces and per-agent policy work at all; a
registry that could only answer one global question would force every one of
those to be a fork.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.testing import raising, simple_tool, tool_runtime
from ph.tools import RUN_CODE, ToolExecutionInput, ToolRestriction, define_tool
from ph.tools.definition import ToolOutput, text_content

pytestmark = pytest.mark.anyio


async def test_a_global_tool_is_visible_everywhere() -> None:
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    agent = root.scope("agent:a")
    assert tools.names() == ["read"]
    assert tools.names(scope=agent) == ["read"]


async def test_a_scoped_registration_shadows_the_global_by_name() -> None:
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    agent = root.scope("agent:a")
    other = root.scope("agent:b")
    scoped = simple_tool("read")
    tools.register(scoped, scope=agent)

    assert tools.get("read", scope=agent) is scoped
    assert tools.get("read", scope=other) is not scoped
    # The global is untouched: shadowing is one-directional.
    assert tools.get("read") is not scoped


async def test_registering_the_same_name_twice_in_one_scope_is_refused() -> None:
    _root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    with pytest.raises(ValueError, match="already registered globally"):
        tools.register(simple_tool("read"))


async def test_a_restriction_hides_a_global_from_one_scope_only() -> None:
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    tools.register(simple_tool("write"))
    agent = root.scope("agent:a")
    other = root.scope("agent:b")
    tools.restrict(ToolRestriction(deny=frozenset({"write"})), scope=agent)

    assert tools.names(scope=agent) == ["read"]
    assert tools.names(scope=other) == ["read", "write"]
    assert tools.names() == ["read", "write"]


async def test_restrictions_intersect() -> None:
    root, tools = tool_runtime()
    for name in ("read", "write", "bash"):
        tools.register(simple_tool(name))
    agent = root.scope("agent:a")
    tools.restrict(ToolRestriction(allow=frozenset({"read", "write"})), scope=agent)
    tools.restrict(ToolRestriction(deny=frozenset({"write"})), scope=agent)
    # Both apply; neither loosens the other.
    assert tools.names(scope=agent) == ["read"]


async def test_a_restriction_cannot_mask_a_scoped_registration() -> None:
    root, tools = tool_runtime()
    agent = root.scope("agent:a")
    tools.register(simple_tool("secret"), scope=agent)
    tools.restrict(ToolRestriction(deny=frozenset({"secret"})), scope=agent)
    # A restriction filters GLOBAL names, so an agent's own tool stays its own.
    assert "secret" in tools.names(scope=agent)


async def test_disposing_a_registration_removes_the_tool() -> None:
    _root, tools = tool_runtime()
    release = tools.register(simple_tool("read"))
    assert tools.names() == ["read"]
    release()
    assert tools.names() == []


async def test_the_view_is_cached_until_the_registry_changes() -> None:
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    agent = root.scope("agent:a")
    first = tools.view(agent)
    assert tools.view(agent) is first
    # Any mutation invalidates: a stale view would show a tool that no longer
    # exists, or hide one that does.
    tools.register(simple_tool("write"))
    assert tools.view(agent) is not first
    assert tools.names(scope=agent) == ["read", "write"]


async def test_every_mutation_announces_itself() -> None:
    root, tools = tool_runtime()
    changes: list[int] = []
    root.on("tools/change", lambda: changes.append(1))
    release = tools.register(simple_tool("read"))
    tools.restrict(ToolRestriction(deny=frozenset({"x"})), scope=root.scope("a"))
    release()
    # Consumers re-read `schemas()` on this event; a silent mutation would leave
    # them offering a stale tool set.
    assert len(changes) == 3


async def test_schemas_expose_only_the_model_facing_fields() -> None:
    _root, tools = tool_runtime()
    tools.register(
        define_tool(
            "read",
            "Read a file.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            output=ToolOutput(schema={"type": "string"}, render=lambda _a, v: text_content(v)),
            execute=lambda _args, _run: "x",
            timeout_ms=5_000,
            is_concurrency_safe=True,
        )
    )
    (schema,) = tools.schemas()
    # A timeout budget and a concurrency classifier are internal; sending either
    # to the model would be leaking harness policy into the prompt.
    assert set(schema.to_wire()) == {"name", "description", "parameters"}


async def test_the_transport_name_is_reserved() -> None:
    _root, tools = tool_runtime()
    with pytest.raises(ValueError, match="reserved Code Mode transport"):
        tools.register(simple_tool(RUN_CODE))
    # And the one door that claims it is only for it.
    with pytest.raises(ValueError, match="register_transport is for"):
        tools.register_transport(simple_tool("read"))


async def test_register_transport_takes_the_ordinary_path() -> None:
    root, tools = tool_runtime()
    changes: list[int] = []
    root.on("tools/change", lambda: changes.append(1))
    release = tools.register_transport(simple_tool(RUN_CODE))
    assert RUN_CODE in tools.names()
    # Same registration path as every tool: schema consumers hear about it, and
    # disposal removes it.
    assert changes == [1]
    release()
    assert RUN_CODE not in tools.names()


async def test_presentation_is_per_agent_and_shadows_the_default() -> None:
    root, tools = tool_runtime()
    agent = root.scope("agent:a")
    other = root.scope("agent:b")
    assert tools.mode_for(agent) == "native"
    tools.present_as("code", scope=agent)
    assert tools.mode_for(agent) == "code"
    assert tools.mode_for(other) == "native"


async def test_code_mode_offers_no_native_schemas() -> None:
    root, tools = tool_runtime()
    tools.register(simple_tool("read"))
    agent = root.scope("agent:a")
    tools.present_as("code", scope=agent)
    # Under Code Mode the model is offered one callable and reaches the rest
    # through the generated SDK.
    assert tools.schemas(scope=agent) == []
    assert tools.names(scope=agent) == ["read"]


async def test_concurrency_classification_defaults_to_exclusive() -> None:
    _root, tools = tool_runtime()
    tools.register(simple_tool("write"))
    tools.register(simple_tool("read", safe=True))
    tools.register(simple_tool("broken", safe=raising(RuntimeError("classifier"))))

    def mode(name: str) -> str:
        return tools.execution_mode(ToolExecutionInput(call_id="c", name=name, arguments={})).kind

    assert mode("read") == "parallel"
    # Omission and a raising classifier are both exclusive: a tool that has not
    # thought about overlap must not be assumed to tolerate it.
    assert mode("write") == "exclusive"
    assert mode("broken") == "exclusive"
    assert mode("absent") == "exclusive"


def _unused(value: Any) -> Any:  # pragma: no cover
    return value
