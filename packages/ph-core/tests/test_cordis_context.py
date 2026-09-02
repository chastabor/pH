"""P0-02 — Context: services, effects, scopes, disposal.

Gate: *disposal unwinds every effect in reverse; a disposed scope's services
are gone.* Invariant I2 is the reason: cleanup has to be structural, not
remembered, or every new plugin is a new chance to leak a subprocess.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from ph.cordis import (
    Context,
    InactiveScopeError,
    ServiceConflictError,
    ServiceNotFoundError,
    plugin,
)
from ph.seams.scope_invariant import violations as scope_violations
from ph.testing import raising

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


# ------------------------------------------------- cancellation mid-dispose --


async def test_a_cancelled_dispose_still_leaves_the_tree(caplog: pytest.LogCaptureFixture) -> None:
    """I2's structural half, held at the one path a live process can break it by.

    `CancelledError` is a `BaseException`, so the effect loop's `except Exception`
    deliberately does not catch it — and before the `finally`, a cancellation
    partway through left the scope still in its parent's `_children`, holding its
    services, and unretryable. See `Context.dispose` for why that state is a leak
    of everything beneath it.

    Raising `CancelledError` from a disposer rather than racing a real timeout:
    the property under test is what `dispose` guarantees when an `await` in it
    does not return, and a `move_on_after` around a sleep would test the same
    thing with a clock in the way.
    """
    root = Context()
    child = root.scope("child")
    child.provide("thing", "value")
    ran: list[str] = []

    child.add_disposer(lambda: ran.append("stranded"), label="stranded")
    child.add_disposer(raising(asyncio.CancelledError()), label="cancelled-here")
    child.add_disposer(lambda: ran.append("first"), label="first")

    with (
        caplog.at_level(logging.WARNING, logger="ph.cordis"),
        pytest.raises(asyncio.CancelledError),
    ):
        await child.dispose()

    # The cancellation still propagates — swallowing it would leave the caller
    # believing a teardown it asked to stop had finished.
    assert ran == ["first"], "the loop continued past the cancellation"
    # ...the scope is gone from the tree regardless...
    assert child not in root.children
    assert not child.active and not child.has("thing")
    # ...and what it could not finish is named rather than silently dropped.
    assert "was cut short while unwinding" in caplog.text
    assert "stranded" in caplog.text


async def test_an_ordinary_dispose_reports_nothing_abandoned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning has to stay rare enough to be worth reading.

    On the ordinary path both lists are already empty by the time the scope
    leaves the tree, so a clean unwind is silent. A warning on every disposal
    would be the same as none.
    """
    root = Context()
    child = root.scope("child")
    child.add_disposer(lambda: None, label="ordinary")
    grandchild = child.scope("grandchild")
    grandchild.add_disposer(lambda: None, label="also-ordinary")

    with caplog.at_level(logging.WARNING, logger="ph.cordis"):
        await child.dispose()

    assert caplog.text == ""
    assert child not in root.children and grandchild not in child.children


async def test_a_cancelled_child_does_not_strand_its_parent() -> None:
    """The cascade, which is where the leak is largest.

    `dispose` unwinds children before its own effects, so a child cancelled
    partway through propagates out of the *parent's* loop too. Both must leave
    the tree: a parent that stayed linked would keep the whole subtree — its
    services and everything they close over — reachable for as long as the root
    lives, which for a deployment scope is the process.

    Asserted through `scope_invariant.violations`, the P6-01 poll that watches for
    exactly this state, so the guarantee and its checker cannot drift apart. It
    covers the half a membership check misses: a child whose `parent` no longer
    lists it.
    """
    root = Context()
    parent = root.scope("parent")
    child = parent.scope("child")
    child.add_disposer(raising(asyncio.CancelledError()), label="cancelled-here")

    with pytest.raises(asyncio.CancelledError):
        await parent.dispose()

    assert scope_violations(root) == [], "a disposed scope was left reachable from the root"
    assert parent not in root.children and child not in parent.children
    assert not parent.active and not child.active
