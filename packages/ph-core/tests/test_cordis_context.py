"""P0-02 — Context: services, effects, scopes, disposal.

Gate: *disposal unwinds every effect in reverse; a disposed scope's services
are gone.* Invariant I2 is the reason: cleanup has to be structural, not
remembered, or every new plugin is a new chance to leak a subprocess.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import weakref
from functools import partial
from unittest.mock import patch

import anyio
import pytest

from ph.cordis import (
    ABANDONED_LEDGER,
    Context,
    Disposer,
    InactiveScopeError,
    ServiceConflictError,
    ServiceNotFoundError,
    plugin,
)
from ph.cordis import context as context_module
from ph.seams.scope_invariant import violations as scope_violations
from ph.testing import raising

pytestmark = pytest.mark.anyio


async def test_services_resolve_most_specific_first() -> None:
    root = Context()
    root.provide("tools", "global-tools")
    agent = root.scope("agent:a")
    assert agent.require("tools") == "global-tools"

    agent.provide("tools", "agent-tools")
    assert agent.require("tools") == "agent-tools"
    # Shadowing is one-directional: the agent sees its own, the root still sees
    # the global one. That asymmetry is what makes a per-agent tool set safe.
    assert root.require("tools") == "global-tools"

    await root.dispose()


async def test_missing_service_raises_attribute_error() -> None:
    root = Context()
    with pytest.raises(ServiceNotFoundError):
        _ = root.require("nothing")
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

    async def acquire() -> Disposer:
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
    async def provider(ctx: Context, config: None) -> None:
        ctx.provide("thing", "value")

    fork = root.plugin(provider)
    await root.reconcile()
    assert root.require("thing") == "value"

    await fork.dispose()
    assert not root.has("thing")


async def test_plugin_waits_for_its_injected_services() -> None:
    root = Context()
    applied: list[str] = []

    @plugin("dependent", inject=["base"])
    async def dependent(ctx: Context, config: None) -> None:
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
    async def provider(ctx: Context, config: None) -> None:
        ctx.provide("shared", "yes")

    @plugin("consumer", inject=["shared"])
    async def consumer(ctx: Context, config: None) -> None:
        ctx.provide("saw", ctx.require("shared"))

    root.plugin(provider)
    root.plugin(consumer)
    await root.reconcile()
    # A row's service is visible to every sibling row, not trapped in the fork.
    assert root.require("saw") == "yes"


async def test_failed_activation_unwinds_its_own_scope() -> None:
    root = Context()

    @plugin("broken")
    async def broken(ctx: Context, config: None) -> None:
        ctx.provide("half", 1)
        raise RuntimeError("boom")

    root.plugin(broken)
    with pytest.raises(RuntimeError, match="boom"):
        await root.reconcile()
    assert not root.has("half")


async def test_activation_scopes_are_transparent_and_agent_scopes_isolate() -> None:
    root = Context()

    @plugin("row")
    async def row(ctx: Context, config: None) -> None:
        ctx.provide("row_scope", ctx)

    root.plugin(row)
    await root.reconcile()
    activation: Context = root.require("row_scope")
    agent = root.scope("agent")
    other = root.scope("other")
    # A row reaches every agent; an agent reaches only itself.
    assert activation.reaches(agent) and activation.reaches(other)
    assert agent.reaches(agent) and not agent.reaches(other)
    assert not other.reaches(agent)


# ------------------------------------------------- cancellation mid-dispose --


async def test_a_canceled_dispose_still_leaves_the_tree(caplog: pytest.LogCaptureFixture) -> None:
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
    child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")
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
    # The ledger's half of the same claim: silent means nothing recorded either.
    assert root.abandoned == () and root.abandoned_dropped == 0


async def test_an_outside_cancellation_no_longer_strands_the_teardown() -> None:
    """The shield, which is what `dispose` gained and the ledger was standing in for.

    A Ctrl-C or a closing task group used to abandon every effect behind the one
    in flight: the worktree unlinked but the child unreaped, the lease held. The
    cancellation still ends the *caller's* wait — the `move_on_after` below
    returns as soon as it fires — but the unwind it interrupted finishes.

    Three shutdown paths already wrapped their own `ctx.dispose()` in exactly
    this, which is the tell that it belonged to the primitive; four other callers
    had nothing.
    """
    root = Context()
    child = root.scope("child")
    ran: list[str] = []

    async def slow() -> None:
        await anyio.sleep(0.05)
        ran.append("slow")

    # Popped LIFO, so `slow` runs first and the cancellation lands inside it —
    # `after` is the effect that used to be stranded by it.
    child.add_disposer(lambda: ran.append("after"), label="after")
    child.add_disposer(slow, label="slow")

    with anyio.move_on_after(0.01):
        await child.dispose()

    assert ran == ["slow", "after"], "the interrupted unwind still finished"
    assert root.abandoned == (), "so there is nothing to record"


async def test_the_budget_still_bounds_a_disposer_that_hangs() -> None:
    """A shield without a deadline is a process that will not exit.

    `daemon/server.py` states the rule this keeps: a root that will not unwind
    must not stop the process from going away. So the shield is a `move_on_at`,
    and a disposer that outlasts the budget still loses the effects behind it —
    which is the case the abandonment ledger genuinely exists for now.
    """
    root = Context()
    child = root.scope("child")
    ran: list[str] = []
    child.add_disposer(lambda: ran.append("behind"), label="behind")
    child.add_disposer(partial(anyio.sleep, 30), label="hangs")

    with patch.object(context_module, "GRACE_SECONDS", 0.02):
        await child.dispose()

    assert ran == [], "the budget was spent on the one that hung"
    (entry,) = root.abandoned
    assert entry.outstanding == ("behind",), "still runnable by whoever holds it"
    # The in-flight one is the point: the effect a deadline cuts off mid-unmount
    # is the one most likely to have left something half-done *and* the one
    # nobody can retry. Popping it before the await used to make it invisible.
    assert entry.unreclaimable == ("hangs",), "and nothing can finish what it started"


async def test_the_budget_is_per_tree_rather_than_per_scope() -> None:
    """Budgets must not multiply across the scopes one unwind visits.

    **Siblings, not depth** — the first version of this test used nested scopes
    and could not fail: `dispose` recurses immediately, so every level starts its
    clock at the same instant and a per-scope budget expires at the same moment a
    shared one would. Siblings are where it bites, because the second child's
    unwind begins only after the first has spent the whole grace.

    `daemon/server.py` names the cost when it insists on one shared number: three
    roots that will not unwind must not be three consecutive grace periods
    between a person pressing Ctrl-C and the process going away.

    The starved sibling is not silently lost — `_leave_tree` records the parent
    as cut short with it still listed, which is the ledger's job.
    """
    root = Context()
    for name in ("first", "second"):
        child = root.scope(name)
        child.add_disposer(partial(anyio.sleep, 30), label=f"hangs-{name}")

    start = anyio.current_time()
    with patch.object(context_module, "GRACE_SECONDS", 0.05):
        await root.dispose()
    spent = anyio.current_time() - start

    assert spent < 0.05 * 1.8, f"each sibling took its own grace: {spent:.3f}s"
    assert root.abandoned, "and what the budget cost is recorded rather than lost"


async def test_one_budget_can_be_spent_across_several_trees() -> None:
    """The case `deadline` alone cannot express, and the daemon's actual shape.

    Each root is its own `Context` with its own runtime, and `dispose` shields
    itself — so a caller that disposes ten of them gets ten `GRACE_SECONDS`
    behind ten shields, from a line that says ten seconds once. And it cannot
    pass a deadline down, because a root unwinds through an `AsyncExitStack`
    with no argument to thread. `unwind_by` seeds the budget instead.

    Two independent trees, each with a disposer that outlasts the budget: shared,
    the second is already out of time when it starts.
    """
    trees = [Context(), Context()]
    for tree in trees:
        tree.add_disposer(partial(anyio.sleep, 30), label="hangs")

    with patch.object(context_module, "GRACE_SECONDS", 0.05):
        until = anyio.current_time() + 0.05
        start = anyio.current_time()
        for tree in trees:
            tree.unwind_by(until)
            await tree.dispose()
        spent = anyio.current_time() - start

    assert spent < 0.05 * 1.8, f"each tree took a budget of its own: {spent:.3f}s"
    assert all(tree.abandoned for tree in trees), "and what the bound cost is recorded"


async def test_a_tree_handed_no_budget_takes_its_own() -> None:
    """The ordinary case stays ordinary: no caller, no seeding, full grace."""
    root = Context()
    root.add_disposer(partial(anyio.sleep, 30), label="hangs")

    start = anyio.current_time()
    with patch.object(context_module, "GRACE_SECONDS", 0.05):
        await root.dispose()
    spent = anyio.current_time() - start

    assert 0.04 < spent < 0.2, f"it spent its own grace, not somebody else's: {spent:.3f}s"


async def test_a_cut_short_unwind_is_recorded_where_something_can_read_it() -> None:
    """The ledger, and the reason it cannot live on the scope it describes.

    By the last line of `_leave_tree` the scope that could not finish is
    unreachable from anything — unlinked from its parent, its services dropped —
    which is exactly why a stranded lease used to be visible only in a
    `log.warning` no shipped entry point handled. The record goes on the shared
    runtime instead, so any scope in the tree can answer for it.
    """
    root = Context()
    child = root.scope("child")
    child.add_disposer(lambda: None, label="stranded-lease")
    child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")

    with pytest.raises(asyncio.CancelledError):
        await child.dispose()

    (entry,) = root.abandoned
    assert entry.path == "root/child"
    # The two kinds, split by whether anybody can still act. `stranded-lease` was
    # never claimed, so its holder can run it; `canceled-here` was claimed and
    # never finished, which `release` refuses on the `done` guard.
    assert entry.outstanding == ("stranded-lease",)
    assert entry.unreclaimable == ("canceled-here",)
    assert child not in root.children, "the scope itself is still gone from the tree"


async def test_an_effect_released_afterwards_stops_being_reported() -> None:
    """`outstanding` is derived from the effects, never a snapshot of their names.

    The closure `add_disposer` hands out stays callable after the unwind is
    abandoned — `dispose()` cannot be the one to call it, since its early return
    makes a second call a no-op, so those closures are the only way back. A
    ledger holding names would keep accusing a deployment of a lease somebody had
    already returned.
    """
    root = Context()
    child = root.scope("child")
    ran: list[str] = []
    release = child.add_disposer(lambda: ran.append("late"), label="stranded-lease")
    child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")

    with pytest.raises(asyncio.CancelledError):
        await child.dispose()

    (entry,) = root.abandoned
    assert "stranded-lease" in entry.outstanding

    release()

    assert ran == ["late"], "the disposer really did run"
    assert "stranded-lease" not in entry.outstanding, "so the ledger stops reporting it"
    assert root.abandoned, "the entry stays; what changed is what it says is owed"


async def test_the_ledger_does_not_pin_what_it_reports_on() -> None:
    """A diagnostic about a leak must not be one.

    The entry watches its effects weakly. A strong reference would pin far more
    than the disposer: `provide`'s `unprovide` closure captures the `Context` and
    the service, so any scope that ever provided anything kept its whole subtree
    and every service in it alive for the life of the tree — none of which
    `outstanding` ever mentioned.

    Asserted through the collector rather than by inspecting the entry: the claim
    is about reachability, and only the collector can answer that.
    """
    root = Context()
    child = root.scope("child")
    child.provide("thing", object())
    child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")
    watch = weakref.ref(child)

    with pytest.raises(asyncio.CancelledError):
        await child.dispose()

    assert root.abandoned, "it was recorded"
    del child
    gc.collect()

    assert watch() is None, "and the ledger is not what keeps the scope alive"


async def test_an_effect_nothing_can_release_is_reported_as_such() -> None:
    """Two kinds of owed teardown, because a reader needs different things.

    `outstanding` means somebody still holds the closure `add_disposer` returned,
    so the resource can be released and naming it is actionable. Once that holder
    is gone nothing can ever call the disposer again — the resource is stranded
    until the process ends, and telling an operator to go and act on it would be
    telling them to do something impossible.
    """
    root = Context()
    child = root.scope("child")
    release = child.add_disposer(lambda: None, label="a-worktree")
    child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")

    with pytest.raises(asyncio.CancelledError):
        await child.dispose()

    (entry,) = root.abandoned
    assert "a-worktree" in entry.outstanding

    # The scope goes; the closure a row kept is what still makes it actionable.
    del child
    gc.collect()
    assert "a-worktree" in entry.outstanding, "the holder can still release it"

    del release
    gc.collect()

    assert "a-worktree" not in entry.outstanding, "nobody can release it any more"
    assert "a-worktree" in entry.unreclaimable, "reported as stranded, not actionable"


async def test_the_ledger_is_bounded_and_says_what_it_dropped() -> None:
    """It holds the abandoned disposers alive, so it cannot be unbounded.

    Keeping them is the point — each stands for a resource nobody has freed and
    is still callable — but an unbounded list of them in a process that keeps
    abandoning would be a leak about a leak. What falls off is counted, so the
    report never quietly under-states.
    """
    root = Context()
    for index in range(ABANDONED_LEDGER + 3):
        child = root.scope(f"child-{index}")
        child.add_disposer(lambda: None, label="stranded")
        child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")
        with pytest.raises(asyncio.CancelledError):
            await child.dispose()

    assert len(root.abandoned) == ABANDONED_LEDGER
    assert root.abandoned_dropped == 3
    # Oldest first: the newest abandonment is the one somebody can still act on.
    assert root.abandoned[-1].path.endswith(f"child-{ABANDONED_LEDGER + 2}")


async def test_a_canceled_child_does_not_strand_its_parent() -> None:
    """The cascade, which is where the leak is largest.

    `dispose` unwinds children before its own effects, so a child canceled
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
    child.add_disposer(raising(asyncio.CancelledError()), label="canceled-here")

    with pytest.raises(asyncio.CancelledError):
        await parent.dispose()

    assert scope_violations(root) == [], "a disposed scope was left reachable from the root"
    assert parent not in root.children and child not in parent.children
    assert not parent.active and not child.active


async def test_drain_ignores_a_canceled_detached_task() -> None:
    """A task canceled on its own account is not the drainer's cancellation.

    `CancelledError` is a `BaseException`, so the `suppress(Exception)` this
    replaces let it through and `drain` raised into a caller nobody had canceled.
    That is only a lost teardown where something drains *before* it unwinds,
    which is exactly `ph_app.runtime.mounted` — so the cost of getting it wrong
    was a whole mounted profile left live because one listener had been stopped.

    The second listener is the half that says the loop continues: a drain that
    swallowed the first and then returned early would pass an assertion about
    the exception alone.
    """
    root = Context()
    settled: list[str] = []

    async def stopped_by_somebody_else() -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        await anyio.sleep(30)

    async def ordinary() -> None:
        settled.append("ordinary")

    root.detach(stopped_by_somebody_else(), label="stopped")
    root.detach(ordinary(), label="ordinary")

    await root.drain()

    assert settled == ["ordinary"], "the drain carried on to the task that had not stopped"


async def _acquired_into_a_dying_scope(disposer: Disposer) -> None:
    """Take an artifact through `effect` while the scope is disposed under it.

    The window `effect` has to answer for, in one place: `enter()` is awaited, a
    `dispose()` on another task lands inside that await, and the disposer arrives
    for a scope that is already down. Both tests below turn on what happens next
    and differ only in what the disposer does, so the race belongs here rather
    than spelled twice with one line changed.
    """
    scope = Context().scope("agent")
    acquiring, finish = anyio.Event(), anyio.Event()

    async def acquire() -> Disposer:
        acquiring.set()
        await finish.wait()
        return disposer

    async def take() -> None:
        with pytest.raises(InactiveScopeError, match="was disposed while"):
            await scope.effect(acquire, label="worktree")

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(take)
        await acquiring.wait()
        await scope.dispose()
        finish.set()


async def test_an_effect_acquired_into_a_dead_scope_is_released() -> None:
    """I2's one window: the artifact exists and its scope is already gone.

    Registering the disposer anyway would be worse than refusing — the effect
    list has already drained, so the entry would sit there with nothing left to
    run it. Releasing at once is the only answer that keeps "everything unwinds"
    from depending on timing. A worktree stands in for the artifact because that
    is the case that cost something real: the directory is on disk, and nothing
    that could have removed it knows it is there.
    """
    released: list[str] = []

    await _acquired_into_a_dying_scope(partial(released.append, "worktree"))

    assert released == ["worktree"], "the artifact outlived the scope that took it"


async def test_a_row_that_failed_to_apply_does_not_block_the_rest() -> None:
    """A failed `apply` is not retried, and does not stand in the queue.

    See `_Dependent.failure` for why the retry happened at all. What this holds
    is the consequence a reader cares about: one broken row used to mount nothing
    behind it, because the next `reconcile` reached the same `apply` again before
    any row still waiting its turn. A daemon that retries a mount is where that
    was paid for.
    """
    root = Context()
    applied: list[str] = []

    @plugin("broken")
    async def broken(ctx: Context, config: None) -> None:
        applied.append("broken")
        raise RuntimeError("boom")

    @plugin("sound")
    async def sound(ctx: Context, config: None) -> None:
        applied.append("sound")
        ctx.provide("sound", True)

    bad, good = root.plugin(broken), root.plugin(sound)
    with pytest.raises(RuntimeError, match="boom"):
        await root.reconcile()

    await root.reconcile()
    await root.reconcile()

    assert applied == ["broken", "sound"], "the failure ran once and stopped blocking the queue"
    assert good.state == "active" and root.has("sound")
    # `failed` rather than `waiting`: the one state that does not resolve itself,
    # and the reason beside it so a report does not send its reader to the logs.
    assert bad.state == "failed"
    assert bad.failure == "RuntimeError: boom"


async def test_two_siblings_disposing_at_once_are_both_shielded() -> None:
    """The shield has to be per unwind, not per tree.

    The flag this replaces lived on the runtime every scope in a tree shares. It
    answered the *nested* case correctly — see `_UNWINDING` for what a shield per
    layer costs — but it could not tell a nested call from a sibling on another
    task, which then ran its whole unwind bare and had its effects stranded by
    the first one's cancellation.

    A passivation overlapping a shutdown is the daemon's spelling of this, and
    the stranded effect is whatever the second root was holding: a lease, a
    worktree, a kernel.
    """
    root = Context()
    ran: list[str] = []

    async def slow(name: str) -> None:
        await anyio.sleep(0.05)
        ran.append(f"{name}-slow")

    for name in ("first", "second"):
        child = root.scope(name)
        child.add_disposer(partial(ran.append, f"{name}-after"), label=f"{name}-after")
        child.add_disposer(partial(slow, name), label=f"{name}-slow")

    with anyio.move_on_after(0.01):
        async with anyio.create_task_group() as tasks:
            for child in root.children:
                tasks.start_soon(child.dispose)

    assert sorted(ran) == ["first-after", "first-slow", "second-after", "second-slow"]
    assert root.abandoned == (), "neither unwind was cut short"


async def test_a_release_that_fails_after_the_scope_died_still_refuses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One situation, one exception, whatever the disposer does.

    `_unwind` already answers a raising disposer this way — it ran, what it left
    behind is its own business, and the traceback is the account of it — and a
    caller here can act on neither failure. Letting the release's exception
    through instead would make the error a caller sees depend on whether an
    artifact it never registered came apart cleanly.
    """
    with caplog.at_level(logging.ERROR, logger="ph.cordis"):
        await _acquired_into_a_dying_scope(raising(RuntimeError("the worktree was already gone")))

    assert "could not be released" in caplog.text
    assert "the worktree was already gone" in caplog.text


async def test_a_seeded_budget_covers_the_drain_as_well_as_the_unwind() -> None:
    """One declared total, not one per half and one per tree.

    A shield is opaque to a deadline outside it, which is why `dispose` takes the
    instant rather than being wrapped in it — and the drain ahead of it has the
    same property, so answering only to `DRAIN_SECONDS` would spend that much
    *per tree* inside a caller's total. `daemon/server.py` closes ten roots under
    one ten-second bound; ten drains of five seconds is fifty.

    `unwind_by` is the one declaration, so this asserts against it rather than
    against a wall-clock guess: the listener below never settles, and the whole
    teardown still ends when the seeded instant says it does.
    """
    root = Context()
    released: list[str] = []
    await root.effect(lambda: partial(released.append, "lease"), label="lease")

    async def never() -> None:
        await anyio.sleep(30)

    root.detach(never(), label="a listener that will not settle")
    root.unwind_by(anyio.current_time() + 0.02)

    started = anyio.current_time()
    # Patched like every budget test above, so a regression costs milliseconds
    # here rather than a real `DRAIN_SECONDS` of suite. The seeded 0.02 still
    # wins the `min`, which is the thing under test.
    with patch.object(context_module, "DRAIN_SECONDS", 0.05):
        await root.drain()
        await root.dispose()
    spent = anyio.current_time() - started

    assert released == ["lease"], "the unwind still ran"
    assert spent < 0.05, f"the drain spent its own share inside a smaller budget: {spent:.3f}s"
