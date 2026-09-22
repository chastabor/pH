"""P0-03 — the five dispatch modes.

Gate: *waterfall veto stops built-in behavior; `next()` ordering; parallel
aggregates rejections.* The waterfall veto is the load-bearing one — it is how
every policy plugin in Phase 4 (limits, HITL, permissions) replaces built-in
behavior without the built-in knowing it was replaced.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from ph.cordis import Context, events, plugin, settled

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

    async def outer(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
        trace.append("outer-in")
        result = await next_()
        trace.append("outer-out")
        return f"[{result}]"

    async def inner_listener(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
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


async def test_a_listener_that_answers_the_wrong_type_names_the_event() -> None:
    """`waterfall` hands back what it was given; the producer is what checks it.

    The round trip is the assertion: `waterfall` returns the `42` unexamined —
    its last step is a `cast` — and `settled` is where that becomes a refusal.
    """

    async def confused(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
        return 42  # type: ignore[return-value]

    async def built_in(payload: dict[str, Any]) -> str:
        return "value"

    root = Context()
    root.on("test/waterfall", confused)
    answered = await root.waterfall("test/waterfall", {}, inner=built_in)
    # Declared `str` by the chain, an `int` at runtime — the gap `settled` closes.
    # The `object` rebind is the assertion: mypy rejects comparing the declared
    # type to `42`, which is the same blind spot the cast leaves at the producer.
    returned: object = answered
    assert returned == 42, "the chain does not check; that is what `settled` is for"

    with pytest.raises(TypeError, match=r"test/waterfall must resolve to a str, not 42"):
        settled("test/waterfall", answered, str)


async def test_a_producer_names_the_refusal_its_seam_owns() -> None:
    """`ctx.fs` raises `FsDenied`, not `TypeError`, for a listener that misanswers.

    A veto has to reach a consumer as a *denial*: `FsDenied` is a `HarnessError`
    so a Code Mode program cannot `except` a policy refusal and route around it.
    A listener answering the wrong shape is still a refusal from that gate, so it
    must not arrive wearing the other kind.
    """

    class Refused(Exception):
        pass

    with pytest.raises(Refused, match=r"must resolve to a str, not 7"):
        settled("fs/write-intent", 7, str, refusal=Refused)


async def test_waterfall_veto_stops_the_built_in() -> None:
    root = Context()
    ran: list[str] = []

    async def veto(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
        return "denied"

    async def downstream(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
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

    async def rewrite(value: str, next_: Callable[..., Awaitable[str]]) -> str:
        return await next_("rewritten")

    async def observe(value: str, next_: Callable[..., Awaitable[str]]) -> str:
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
    async def listener(ctx: Context, config: None) -> None:
        ctx.on("test/emit", lambda value: heard.append(value))

    fork = root.plugin(listener)
    await root.reconcile()
    root.emit("test/emit", 1)
    await fork.dispose()
    root.emit("test/emit", 2)
    # A registration is an effect: unloading the plugin unregisters it, with
    # nothing to remember (invariant I2).
    assert heard == [1]


async def test_a_listener_that_retries_runs_the_rest_of_the_chain_again() -> None:
    """A6 — `Context.waterfall`'s per-frame cursor, which says why.

    What this pins is the *ordering*: the listeners between the retry and the
    built-in run on both attempts. With a shared index the second call resumed
    past all of them, so a retry got `inner` alone.
    """
    root = Context()
    ran: list[str] = []

    async def retrying(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
        first = await next_()
        return f"{first}+{await next_()}"

    async def counted(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
        ran.append("inner-listener")
        return await next_()

    async def built_in(payload: dict[str, Any]) -> str:
        ran.append("built-in")
        return "answer"

    root.on("test/waterfall", retrying)
    root.on("test/waterfall", counted)

    assert await root.waterfall("test/waterfall", {}, inner=built_in) == "answer+answer"
    assert ran == ["inner-listener", "built-in", "inner-listener", "built-in"], (
        "the second attempt skipped the listeners between the retry and the built-in"
    )


@pytest.mark.parametrize(
    ("answering", "expected"),
    [("outer", "outer"), ("inner", "inner"), ("both", "outer"), ("nobody", "")],
)
async def test_the_answer_is_attributed_to_the_plugin_that_produced_it(
    answering: str, expected: str
) -> None:
    """G13 — `waterfall_attributed` names who answered, so a row need not say.

    The producer is the outermost listener whose return is not what its `next_`
    handed it. A listener that inspects the answer and passes it back unchanged —
    a policy row that only *might* veto — is not the producer; `inner`'s own
    answer, or one from outside any plugin, is `""`.

    When both replace it, the outer one's answer is the one returned, so it is
    the producer. Sabotage: attribute the outermost listener that ran, or the
    innermost that produced anything, and one case below names the wrong plugin.
    """
    root = Context()

    def listener(name: str) -> Any:  # noqa: ANN401
        @plugin(name)
        async def apply(ctx: Context, config: None) -> None:
            async def decide(payload: dict[str, Any], next_: Callable[..., Awaitable[str]]) -> str:
                answer = await next_()
                return f"{name}'s" if answering in (name, "both") else answer

            ctx.on("test/waterfall", decide)

        return apply

    root.plugin(listener("outer"))
    root.plugin(listener("inner"))
    await root.reconcile()

    async def built_in(payload: dict[str, Any]) -> str:
        return "default"

    _answer, by = await root.waterfall_attributed("test/waterfall", {}, inner=built_in)

    assert by == expected


async def test_a_replacement_is_scoped_to_the_call_that_passed_it() -> None:
    """The other half of A6: `state` was a list every frame mutated.

    `next_(*replacement)` rewrote the shared list, so a rewrite one attempt made
    was still in force on the next — a retry that narrowed its arguments once
    could not widen them again, and a sibling listener read arguments it never
    saw passed.
    """
    root = Context()
    seen: list[str] = []

    async def rewriting(value: str, next_: Callable[..., Awaitable[str]]) -> str:
        await next_("narrowed")
        return await next_()

    async def built_in(value: str) -> str:
        seen.append(value)
        return value

    root.on("test/waterfall", rewriting)

    assert await root.waterfall("test/waterfall", "original", inner=built_in) == "original"
    assert seen == ["narrowed", "original"], "a replacement outlived the call that passed it"


def test_disposing_one_registration_leaves_its_twin_where_it_was() -> None:
    """A8 — `Hook` was an ordinary dataclass, so `list.remove` matched by field.

    Two registrations of one callback on one scope produce hooks that compare
    equal — same ctx, same callback, same flags — and `remove` takes the *first*
    equal element. Disposing the second therefore deleted the first.

    The damage is **ordering**, which is why the two hooks being interchangeable
    is not a defence: with a third listener between them the survivor moves from
    in front of it to behind it, so a listener registered first starts running
    last. Order is the one thing `on`/`prepend` promises, and every policy chain
    is built on it.

    Two registrations of one callback is not exotic — a row mounted per agent, a
    wrapper registered around itself — and nothing about the symptom points at
    equality.
    """
    root = Context()
    heard: list[str] = []

    def twice(value: int) -> None:
        heard.append("twice")

    def between(value: int) -> None:
        heard.append("between")

    root.on("test/emit", twice)
    root.on("test/emit", between)
    second = root.on("test/emit", twice)

    second()
    root.emit("test/emit", 1)

    assert heard == ["twice", "between"], "the disposer took the wrong registration"
