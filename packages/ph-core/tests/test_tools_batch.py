"""P1-03 — the batch scheduler: overlap without reordering.

Gates: *an exclusive call never overlaps; results commit in model order
regardless of completion order.*

The second is the one that would be easy to get wrong and hard to notice: if a
fast call committed before a slow one that the model asked for first, the
transcript the model reads next would not match the order it requested, and
every later turn would reason about a conversation that never happened.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from ph.cancel import CancelToken
from ph.llm.types import ToolCallBlock, create_user_message
from ph.session import Session
from ph.testing import StubAgent, parked_gate, raising, simple_tool, tool_runtime
from ph.tools import TOOL_ABORTED_BEFORE_DISPATCH, Deny, ToolRuntime
from ph.tools.batch import execute_tool_calls, parse_arguments

pytestmark = pytest.mark.anyio


def _setup() -> tuple[Any, ToolRuntime, StubAgent, list[str]]:
    root, tools = tool_runtime()
    agent = StubAgent(root, Session("s"))
    return root, tools, agent, []


def _slow(name: str, trace: list[str], delay: float, *, safe: bool) -> Any:
    async def body(_args: Any, _run: Any) -> str:
        trace.append(f"{name}:start")
        await anyio.sleep(delay)
        trace.append(f"{name}:end")
        return name

    return simple_tool(name, body, safe=safe)


def _blocks(*names: str) -> list[ToolCallBlock]:
    return [
        ToolCallBlock(id=f"call-{i}", name=name, arguments="{}") for i, name in enumerate(names)
    ]


async def _run(root: Any, agent: StubAgent, *names: str, **kwargs: Any) -> Any:
    token = kwargs.pop("token", CancelToken())
    accept = kwargs.pop("accept", lambda _c: None)
    return await execute_tool_calls(root, agent, 1, 1, _blocks(*names), token, accept, **kwargs)


async def test_parallel_calls_overlap() -> None:
    root, tools, agent, trace = _setup()
    tools.register(_slow("a", trace, 0.05, safe=True))
    tools.register(_slow("b", trace, 0.01, safe=True))
    await _run(root, agent, "a", "b")
    # b finishes while a is still running, which is the whole point.
    assert trace == ["a:start", "b:start", "b:end", "a:end"]


async def test_results_commit_in_model_order_not_completion_order() -> None:
    root, tools, agent, trace = _setup()
    tools.register(_slow("slow", trace, 0.05, safe=True))
    tools.register(_slow("fast", trace, 0.01, safe=True))
    await _run(root, agent, "slow", "fast")
    results = [
        event.data["message"]["content"][0]["content"][0]["text"]
        for event in agent.session.events
        if event.type == "tool/result"
    ]
    assert results == ["slow", "fast"]


async def test_an_exclusive_call_never_overlaps() -> None:
    root, tools, agent, trace = _setup()
    tools.register(_slow("safe1", trace, 0.02, safe=True))
    tools.register(_slow("lock", trace, 0.02, safe=False))
    tools.register(_slow("safe2", trace, 0.02, safe=True))
    await _run(root, agent, "safe1", "lock", "safe2")
    # The exclusive call is a barrier in both directions: nothing overlaps it.
    assert trace == [
        "safe1:start",
        "safe1:end",
        "lock:start",
        "lock:end",
        "safe2:start",
        "safe2:end",
    ]


async def test_the_pool_is_bounded() -> None:
    root, tools, agent, trace = _setup()
    for name in ("a", "b", "c", "d"):
        tools.register(_slow(name, trace, 0.02, safe=True))
    await _run(root, agent, "a", "b", "c", "d", max_parallel=2)
    running = peak = 0
    for entry in trace:
        running += 1 if entry.endswith(":start") else -1
        peak = max(peak, running)
    assert peak == 2


async def test_every_call_is_logged_before_it_executes() -> None:
    root, tools, agent, trace = _setup()
    tools.register(_slow("a", trace, 0.0, safe=True))
    await _run(root, agent, "a")
    assert [event.type for event in agent.session.events] == ["tool/call", "tool/result"]
    # The result cites its call, so a reader can pair them without guessing.
    assert agent.session.events[1].source_event_seqs == (0,)


async def test_a_parked_call_has_no_record_until_the_gate_decides() -> None:
    """**P7-15's gate.** Between asking and being answered, nothing has happened.

    A database rolls an unfinished transaction back because it *owns* the effect;
    a tool call's effect escapes, so its record has to precede execution (B4).
    But it does not have to precede the *wait*: a call parked on a human has done
    nothing yet, and a `tool/call` written there says an act was attempted when
    it was only asked about. So the record is written after the gate decides —
    whichever way — and before the body runs.

    Observed mid-flight rather than by comparing two events' order, because the
    obvious form passes for the old ordering too. Sabotage: move `_append_call`
    back into `fill_pool`, and the log holds a call while the gate is still open.
    """
    root, tools, agent, trace = _setup()
    tools.register(_slow("a", trace, 0.0, safe=True))
    reached, release = parked_gate(root)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_run, root, agent, "a")
        await reached.wait()
        assert [e.type for e in agent.session.events] == [], "parked: nothing has happened"
        release.set()

    assert [e.type for e in agent.session.events] == ["tool/call", "tool/result"]
    assert agent.session.events[1].source_event_seqs == (0,), "and the pair still pairs"


async def test_a_denied_call_still_logs_its_pair() -> None:
    """The record is written for a "no" too, or the pair stops being a pair.

    `_append_skipped` exists to keep call/result paired; `ph-stabilize`'s limits
    fold counts `tool/call`; `cards.py` finds a result's arguments through the
    call it cites. Sabotage: write the record only on allow — a model that spams
    denied calls never reaches the tool-call limit that exists to bound exactly
    that, and every denial card loses its arguments.
    """
    root, tools, agent, trace = _setup()
    tools.register(_slow("a", trace, 0.0, safe=True))
    root.on("tools/pre-execute", lambda _execution, _next: Deny(reason="not today"))

    await _run(root, agent, "a")

    assert [e.type for e in agent.session.events] == ["tool/call", "tool/result"]
    assert agent.session.events[1].data["failureKind"] == "denied"
    assert agent.session.events[1].source_event_seqs == (0,)
    assert "a:start" not in trace


async def test_results_keep_model_order_when_the_gate_settles_out_of_order() -> None:
    """What the model reads is unchanged by where the record moved.

    Slot 0 is parked on its gate while slot 1 sails through, so slot 1's
    `tool/call` lands first — the log records what became runnable in the order
    it did, and every reader of `tool/call` keys by `callId` or `seq`, never by
    position. What must not move is the *surface*: results commit in model order
    through `commit_ready`, each citing its own call, so `derive_messages()` and
    the provider's cached prefix (A12) are byte-identical to a run where slot 0
    was never parked.
    """
    root, tools, agent, trace = _setup()
    tools.register(_slow("slow", trace, 0.0, safe=True))
    _reached, release = parked_gate(root, only="slow")

    async def fast_body(_args: Any, _run: Any) -> str:
        release.set()  # slot 1 has run before slot 0 was even allowed to
        return "fast"

    tools.register(simple_tool("fast", fast_body, safe=True))

    await _run(root, agent, "slow", "fast")

    events = agent.session.events
    calls = [e for e in events if e.type == "tool/call"]
    results = [e for e in events if e.type == "tool/result"]
    assert [c.data["name"] for c in calls] == ["fast", "slow"], "recorded as they became runnable"
    assert [r.data["message"]["source"]["callId"] for r in results] == ["call-0", "call-1"], (
        "committed as the model asked"
    )
    by_id = {c.data["callId"]: c.seq for c in calls}
    for result in results:
        assert result.source_event_seqs == (by_id[result.data["message"]["source"]["callId"]],)


async def test_a_crashing_body_still_leaves_its_call_and_a_result() -> None:
    root, tools, agent, _trace = _setup()
    tools.register(simple_tool("boom", raising(RuntimeError("nope"))))
    await _run(root, agent, "boom")
    result = agent.session.events[1]
    assert [event.type for event in agent.session.events] == ["tool/call", "tool/result"]
    assert result.data["message"]["content"][0]["isError"] is True
    # The kind travels into the log, so a card can colour a failure differently
    # from a refusal without re-deriving it.
    assert result.data["failureKind"] == "failed"


async def test_cancellation_records_a_result_for_every_skipped_call() -> None:
    root, tools, agent, trace = _setup()
    token = CancelToken()

    async def cancelling(_args: Any, _run: Any) -> str:
        token.cancel("user")
        return "first"

    tools.register(simple_tool("first", cancelling))
    tools.register(_slow("second", trace, 0.0, safe=False))

    outcome = await _run(root, agent, "first", "second", token=token)
    assert outcome.aborted
    # Replay must see a result for every call, or the pairing a provider
    # requires is broken.
    calls = [e for e in agent.session.events if e.type == "tool/call"]
    results = [e for e in agent.session.events if e.type == "tool/result"]
    assert len(calls) == len(results) == 2
    assert results[1].data["error"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    assert results[1].data["failureKind"] == "aborted"
    assert "second:start" not in trace


async def test_deferred_context_reaches_the_acceptor_after_the_result() -> None:
    root, tools, agent, _trace = _setup()

    def body(_args: Any, run: Any) -> str:
        run.defer_context(
            create_user_message(
                content=[{"type": "text", "text": "notice"}],
                source={"kind": "plugin", "plugin": "t"},
            )
        )
        return "ok"

    tools.register(simple_tool("defer", body))
    seen_at: list[int] = []
    await _run(root, agent, "defer", accept=lambda _c: seen_at.append(len(agent.session.events)))
    assert len(seen_at) == 1
    # Handed over only after the result is durable, so call/result adjacency
    # survives — a context spliced between them breaks the pairing.
    assert agent.session.events[seen_at[0] - 1].type == "tool/result"


async def test_conclude_turn_propagates_to_the_batch_outcome() -> None:
    root, tools, agent, _trace = _setup()
    tools.register(simple_tool("finish", lambda _a, run: run.conclude_turn() or "done"))
    outcome = await _run(root, agent, "finish")
    assert outcome.concluded


def test_malformed_arguments_survive_as_text() -> None:
    assert parse_arguments("") == {}
    assert parse_arguments('{"a": 1}') == {"a": 1}
    # A broken argument string is the tool's problem to report, not the loop's
    # to crash on.
    assert parse_arguments("{not json") == "{not json"
