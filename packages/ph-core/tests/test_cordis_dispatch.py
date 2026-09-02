"""P0-03 — the five dispatch modes.

Gate: *waterfall veto stops built-in behaviour; `next()` ordering; parallel
aggregates rejections.* The waterfall veto is the load-bearing one — it is how
every policy plugin in Phase 4 (limits, HITL, permissions) replaces built-in
behaviour without the built-in knowing it was replaced.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.cordis import Context, events, plugin

pytestmark = pytest.mark.anyio

events.declare("test/emit", "emit", owner="tests")
events.declare("test/serial", "serial", owner="tests")
events.declare("test/parallel", "parallel", owner="tests")
events.declare("test/waterfall", "waterfall", owner="tests")


async def test_emit_runs_listeners_in_registration_order() -> None:
    root = Context()
    seen: list[str] = []
    root.on("test/emit", lambda value: seen.append(f"a{value}"))
    root.on("test/emit", lambda value: seen.append(f"b{value}"))
    root.on("test/emit", lambda value: seen.append(f"z{value}"), prepend=True)
    root.emit("test/emit", 1)
    assert seen == ["z1", "a1", "b1"]


async def test_contained_emit_logs_and_continues() -> None:
    root = Context()
    heard: list[int] = []

    def boom(value: int) -> None:
        raise RuntimeError("bad listener")

    root.on("test/emit", boom)
    root.on("test/emit", lambda value: heard.append(value))
    with pytest.raises(RuntimeError):
        root.emit("test/emit", 1)
    # Contained: the record of something that already happened reaches every
    # listener even when one of them is broken.
    root.emit("test/emit", 2, contained=True)
    assert heard == [2]


async def test_waterfall_listeners_wrap_the_built_in() -> None:
    root = Context()
    trace: list[str] = []

    async def outer(payload: dict[str, Any], next_: Any) -> str:
        trace.append("outer-in")
        result = await next_()
        trace.append("outer-out")
        return f"[{result}]"

    async def inner_listener(payload: dict[str, Any], next_: Any) -> str:
        trace.append("inner-in")
        return await next_()

    async def built_in(payload: dict[str, Any]) -> str:
        trace.append("built-in")
        return "value"

    root.on("test/waterfall", outer)
    root.on("test/waterfall", inner_listener)
    result = await root.waterfall("test/waterfall", {}, inner=built_in)
    assert result == "[value]"
    # Outermost first, and the built-in runs last — the shape every policy
    # listener relies on.
    assert trace == ["outer-in", "inner-in", "built-in", "outer-out"]


async def test_waterfall_veto_stops_the_built_in() -> None:
    root = Context()
    ran: list[str] = []

    async def veto(payload: dict[str, Any], next_: Any) -> str:
        return "denied"

    async def downstream(payload: dict[str, Any], next_: Any) -> str:
        ran.append("downstream")
        return await next_()

    async def built_in(payload: dict[str, Any]) -> str:
        ran.append("built-in")
        return "allowed"

    root.on("test/waterfall", veto)
    root.on("test/waterfall", downstream)
    assert await root.waterfall("test/waterfall", {}, inner=built_in) == "denied"
    # Not calling next() vetoes everything downstream, the built-in included.
    assert ran == []


async def test_waterfall_next_can_hand_down_replaced_arguments() -> None:
    root = Context()

    async def rewrite(value: str, next_: Any) -> str:
        return await next_("rewritten")

    async def observe(value: str, next_: Any) -> str:
        return await next_()

    async def built_in(value: str) -> str:
        return value

    root.on("test/waterfall", rewrite)
    root.on("test/waterfall", observe)
    # Cordis expects a listener to mutate a shared payload; pH's payloads are
    # frozen values, so a rewrite is explicit and a reader can see it.
    assert await root.waterfall("test/waterfall", "original", inner=built_in) == "rewritten"


async def test_serial_stops_on_the_first_bail() -> None:
    root = Context()
    seen: list[int] = []

    async def first(value: int) -> None:
        seen.append(1)

    async def second(value: int) -> str:
        seen.append(2)
        return "stop"

    async def third(value: int) -> None:
        seen.append(3)

    root.on("test/serial", first)
    root.on("test/serial", second)
    root.on("test/serial", third)
    assert await root.serial("test/serial", 0) == "stop"
    assert seen == [1, 2]


async def test_serial_treats_zero_as_an_answer() -> None:
    """`is_bailed`'s one rule, held where it is still used.

    This was pinned on `bail`, the synchronous twin of `serial`, until that mode
    was dropped: it had no declared event in any package and no counterpart in
    dsh — pH's own addition, unused across six phases. The rule outlives the
    mode: `0` bails, because a decision object is never falsy by accident, and
    treating a legitimate zero as "no answer" is the bug this exists to prevent.
    """
    root = Context()
    root.on("test/serial", lambda: None)
    root.on("test/serial", lambda: 0)
    root.on("test/serial", lambda: "later")
    assert await root.serial("test/serial") == 0


async def test_parallel_runs_every_listener_and_aggregates_failures() -> None:
    root = Context()
    ran: list[str] = []

    async def ok(value: int) -> None:
        ran.append("ok")

    async def boom(value: int) -> None:
        ran.append("boom")
        raise ValueError("first")

    async def bang(value: int) -> None:
        ran.append("bang")
        raise KeyError("second")

    root.on("test/parallel", ok)
    root.on("test/parallel", boom)
    root.on("test/parallel", bang)
    with pytest.raises(ExceptionGroup) as caught:
        await root.parallel("test/parallel", 1)
    assert sorted(ran) == ["bang", "boom", "ok"]
    assert len(caught.value.exceptions) == 2


async def test_scope_filtering_isolates_agent_listeners() -> None:
    root = Context()
    heard: list[str] = []

    plugin_scope = Context(root, label="plugin-fork")
    plugin_scope.on("test/emit", lambda value: heard.append(f"plugin:{value}"))

    agent_a = root.scope("agent:a")
    agent_b = root.scope("agent:b")
    agent_a.on("test/emit", lambda value: heard.append(f"a:{value}"))
    agent_b.on("test/emit", lambda value: heard.append(f"b:{value}"))

    agent_a.emit("test/emit", 1)
    # A plugin row hears every agent; an agent hears only itself.
    assert heard == ["plugin:1", "a:1"]

    heard.clear()
    agent_b.emit("test/emit", 2)
    assert heard == ["plugin:2", "b:2"]


async def test_global_listeners_opt_out_of_filtering() -> None:
    root = Context()
    heard: list[int] = []
    agent_a = root.scope("agent:a")
    agent_b = root.scope("agent:b")
    agent_a.on("test/emit", lambda value: heard.append(value), global_=True)
    agent_b.emit("test/emit", 7)
    assert heard == [7]


async def test_disposal_removes_listeners() -> None:
    root = Context()
    heard: list[int] = []

    @plugin("listener")
    async def listener(ctx: Context, config: object) -> None:
        ctx.on("test/emit", lambda value: heard.append(value))

    fork = root.plugin(listener)
    await root.reconcile()
    root.emit("test/emit", 1)
    await fork.dispose()
    root.emit("test/emit", 2)
    # A registration is an effect: unloading the plugin unregisters it, with
    # nothing to remember (invariant I2).
    assert heard == [1]
