"""P0-02 — Context: services, effects, scopes, disposal.

Gate: *disposal unwinds every effect in reverse; a disposed scope's services
are gone.* Invariant I2 is the reason: cleanup has to be structural, not
remembered, or every new plugin is a new chance to leak a subprocess.
"""

from __future__ import annotations

import pytest

from ph.cordis import (
    Context,
    InactiveScopeError,
    ServiceConflictError,
    ServiceNotFoundError,
    plugin,
)

pytestmark = pytest.mark.anyio


async def test_services_resolve_most_specific_first() -> None:
    root = Context()
    root.provide("tools", "global-tools")
    agent = root.scope("agent:a")
    assert agent.tools == "global-tools"

    agent.provide("tools", "agent-tools")
    assert agent.tools == "agent-tools"
    # Shadowing is one-directional: the agent sees its own, the root still sees
    # the global one. That asymmetry is what makes a per-agent tool set safe.
    assert root.tools == "global-tools"

    await root.dispose()


async def test_missing_service_raises_attribute_error() -> None:
    root = Context()
    with pytest.raises(ServiceNotFoundError):
        _ = root.nothing
    # AttributeError subclassing keeps getattr/hasattr behaving normally.
    assert getattr(root, "nothing", "fallback") == "fallback"
    assert not root.has("nothing")


async def test_second_provider_in_one_realm_conflicts() -> None:
    root = Context()
    root.provide("llm", object())
    with pytest.raises(ServiceConflictError):
        root.provide("llm", object())


async def test_effects_unwind_in_reverse_and_children_first() -> None:
    root = Context()
    order: list[str] = []
    root.add_disposer(lambda: order.append("root-1"))
    child = root.scope("child")
    child.add_disposer(lambda: order.append("child-1"))
    child.add_disposer(lambda: order.append("child-2"))
    root.add_disposer(lambda: order.append("root-2"))

    await root.dispose()
    # Children before parents; within a scope, last registered is first released.
    assert order == ["child-2", "child-1", "root-2", "root-1"]


async def test_async_effect_acquires_and_releases() -> None:
    root = Context()
    released: list[str] = []

    async def acquire() -> object:
        def release() -> None:
            released.append("worktree")

        return release

    await root.effect(acquire, label="worktree")
    assert released == []
    await root.dispose()
    assert released == ["worktree"]


async def test_disposed_scope_loses_services_and_refuses_registration() -> None:
    root = Context()
    scope = root.scope("agent")
    scope.provide("thing", 1)
    await scope.dispose()
    assert not scope.active
    assert not scope.has("thing")
    with pytest.raises(InactiveScopeError):
        scope.provide("other", 2)


async def test_disposing_a_provider_removes_the_service() -> None:
    root = Context()

    @plugin("provider")
    async def provider(ctx: Context, config: object) -> None:
        ctx.provide("thing", "value")

    fork = root.plugin(provider)
    await root.reconcile()
    assert root.thing == "value"

    await fork.dispose()
    assert not root.has("thing")


async def test_plugin_waits_for_its_injected_services() -> None:
    root = Context()
    applied: list[str] = []

    @plugin("dependent", inject=["base"])
    async def dependent(ctx: Context, config: object) -> None:
        applied.append("dependent")

    root.plugin(dependent)
    await root.reconcile()
    # File order did not start it: the load order is expressed by `inject`.
    assert applied == []

    disposer = root.provide("base", object())
    await root.reconcile()
    assert applied == ["dependent"]

    # Removing the service deactivates the dependent; restoring it reactivates.
    disposer()
    await root.reconcile()
    root.provide("base", object())
    await root.reconcile()
    assert applied == ["dependent", "dependent"]


async def test_plugin_provides_into_the_realm_it_was_mounted_in() -> None:
    root = Context()

    @plugin("provider")
    async def provider(ctx: Context, config: object) -> None:
        ctx.provide("shared", "yes")

    @plugin("consumer", inject=["shared"])
    async def consumer(ctx: Context, config: object) -> None:
        ctx.provide("saw", ctx.shared)

    root.plugin(provider)
    root.plugin(consumer)
    await root.reconcile()
    # A row's service is visible to every sibling row, not trapped in the fork.
    assert root.saw == "yes"


async def test_failed_activation_unwinds_its_own_scope() -> None:
    root = Context()

    @plugin("broken")
    async def broken(ctx: Context, config: object) -> None:
        ctx.provide("half", 1)
        raise RuntimeError("boom")

    root.plugin(broken)
    with pytest.raises(RuntimeError, match="boom"):
        await root.reconcile()
    assert not root.has("half")


async def test_activation_scopes_are_transparent_and_agent_scopes_isolate() -> None:
    root = Context()

    @plugin("row")
    async def row(ctx: Context, config: object) -> None:
        ctx.provide("row_scope", ctx)

    root.plugin(row)
    await root.reconcile()
    activation: Context = root.row_scope
    agent = root.scope("agent")
    other = root.scope("other")
    # A row reaches every agent; an agent reaches only itself.
    assert activation.reaches(agent) and activation.reaches(other)
    assert agent.reaches(agent) and not agent.reaches(other)
    assert not other.reaches(agent)
