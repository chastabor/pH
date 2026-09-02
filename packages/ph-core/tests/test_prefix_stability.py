"""P1-24 — replay, and the prefix-cache assertion it makes possible (A12).

Prefix stability is the difference between a harness that is cheap to run and
one that silently pays full price every turn, and it is invisible without a
test: a cache miss looks exactly like a cache hit, only on the invoice.

The property is structural, so it can be checked offline over a recording:

* the system prompt is byte-identical across consecutive requests, and
* each request's message list **extends** the previous one — every earlier
  message is unchanged and in the same position.

`context()` snapshots are what make this non-trivial. They are dynamic content,
and they preserve the property only because they materialize *after* retained
history and only when their text changed. A plugin that put changing text in a
static `section` instead would break this test, which is exactly what it is for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.llm.types import GenerateOptions
from ph.system_prompt.assembly import PromptContext, PromptSection
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import REPLAY_ROW, ReplayAdapter, recorded_steps, shared_prefix
from ph.tools import ToolOutput, define_tool, text_content

pytestmark = pytest.mark.anyio


def _shape(message: Any) -> dict[str, Any]:
    """What the model would read, with per-run facts removed.

    Two things legitimately differ between a recording and its replay, and
    neither is content:

    * the message **id**, minted when the message is created;
    * an assistant message's **provider**, which records the route that actually
      produced it. A replay must not claim the original provider ran — the log
      would then assert something that did not happen.
    """
    wire = message.to_wire()
    wire.pop("id", None)
    source = wire.get("source")
    if isinstance(source, dict) and source.get("kind") == "model":
        source.pop("provider", None)
    return wire


def _assert_prefix_stable(requests: list[GenerateOptions]) -> None:
    """Every request extends the last in full — `shared_prefix` is the whole predecessor.

    `shared_prefix` is A12's definition of a hit, shared with the P6-03
    benchmark that prices it, so the structural test and the priced one cannot
    come to disagree about what a hit is.
    """
    for index in range(1, len(requests)):
        previous, current = requests[index - 1], requests[index]
        assert current.system == previous.system, (
            f"request {index} changed the system prompt, so every cached prefix "
            "before it is invalidated"
        )
        assert len(current.messages) >= len(previous.messages)
        assert shared_prefix(previous, current) == len(previous.messages), (
            f"request {index} rewrote history: a cached prefix only survives if earlier "
            "messages stay put"
        )


async def _record(ctx: Any, prompts: list[str]) -> Any:
    session = ctx.sessions.create("recorded")
    agent = ctx.agents.create(session, FAKE)
    for prompt in prompts:
        await agent.prompt(prompt)
    return session


async def test_consecutive_requests_share_their_prefix(mount: Any) -> None:
    ctx = await mount()
    ctx.system_prompt.section(PromptSection(name="identity", text="You are pH.", order=-100))
    await _record(ctx, ["first", "second", "third"])
    requests = ctx.llm_fake.requests
    assert len(requests) == 3
    _assert_prefix_stable(requests)


async def test_a_changing_context_snapshot_keeps_the_prefix_stable(mount: Any) -> None:
    ctx = await mount()
    clock = {"value": "09:00"}
    ctx.system_prompt.section(PromptSection(name="identity", text="You are pH.", order=-100))
    ctx.system_prompt.context(PromptContext(name="time", text=lambda _c: clock["value"]))

    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    await agent.prompt("first")
    clock["value"] = "10:00"
    await agent.prompt("second")
    clock["value"] = "11:00"
    await agent.prompt("third")

    # The changing text is appended as new history, never folded into the
    # cached prefix — which is the entire reason `context()` is not a `section`.
    _assert_prefix_stable(ctx.llm_fake.requests)
    assert all(request.system == "You are pH." for request in ctx.llm_fake.requests)


async def test_a_memory_edit_does_not_move_the_prefix(mount: Any, tmp_path: Path) -> None:
    """G8's gate, and the reason `memory-agents-md` is a `context()` (P4-13).

    `AGENTS.md` exists to be edited. Registered as a static section — which is
    where Phase 1 put it — every edit would rewrite the system prompt and
    invalidate every cached token before it, on the one file a user is most
    likely to touch mid-session. The edit still has to *arrive*, so this asserts
    both halves: the prompt is byte-identical across the edit, and the new text
    reached the model anyway.
    """
    ctx = await mount()
    ctx.system_prompt.section(PromptSection(name="identity", text="You are pH.", order=-100))
    memory = tmp_path / "AGENTS.md"
    memory.write_text("Prefer tabs.", encoding="utf-8")

    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    await agent.prompt("first")
    memory.write_text("Prefer spaces, and say why.", encoding="utf-8")
    await agent.prompt("second")

    requests = ctx.llm_fake.requests
    _assert_prefix_stable(requests)
    assert all(request.system == "You are pH." for request in requests)
    delivered = json.dumps([_shape(message) for message in requests[-1].messages])
    assert "Prefer spaces, and say why." in delivered, "the edit never reached the model"


async def test_a_tool_registration_is_the_kind_of_change_that_does_move_the_prefix(
    mount: Any,
) -> None:
    """Stated explicitly: the test asserts stability, not that nothing ever changes.

    Registering a tool mid-conversation genuinely alters what the model was
    told, so a new `request/header` is correct. The assertion protects against
    *accidental* churn, not against real changes.
    """
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    await agent.prompt("first")
    ctx.tools.register(
        define_tool(
            "ping",
            "returns pong",
            parameters={"type": "object", "properties": {}},
            output=ToolOutput(schema={"type": "string"}, render=lambda _a, v: text_content(v)),
            execute=lambda _args, _run: "pong",
            is_concurrency_safe=True,
        )
    )
    await agent.prompt("second")
    headers = [event for event in session.events if event.type == "request/header"]
    assert [event.data["reason"] for event in headers] == ["initial", "change"]
    # History still extends rather than being rewritten.
    _assert_prefix_stable(ctx.llm_fake.requests)


async def test_replay_reproduces_the_recorded_derivation(mount: Any) -> None:
    ctx = await mount()
    ctx.system_prompt.section(PromptSection(name="identity", text="You are pH.", order=-100))
    recorded = await _record(ctx, ["first", "second"])
    expected = [_shape(message) for message in recorded.derive_messages()]

    # A second process replays the recording with no provider at all.
    replay_ctx = await mount(REPLAY_ROW)
    replay_ctx.system_prompt.section(PromptSection(name="identity", text="You are pH.", order=-100))
    adapter: ReplayAdapter = replay_ctx.llm_replay
    adapter.steps = recorded_steps(recorded.events)

    session = replay_ctx.sessions.create("replayed")
    agent = replay_ctx.agents.create(session, AgentOptions(provider="replay", model="fake-1"))
    await agent.prompt("first")
    await agent.prompt("second")

    assert [_shape(message) for message in session.derive_messages()] == expected
    # And the provenance is honest about who actually answered.
    replayed_sources = [
        message.source.provider
        for message in session.derive_messages()
        if message.role == "assistant"
    ]
    assert set(replayed_sources) == {"replay"}
    assert adapter.exhausted
    _assert_prefix_stable(adapter.requests)


def test_recorded_steps_group_by_turn_and_step() -> None:
    from ph.session import Session

    session = Session("s")
    for turn, step in ((1, 1), (1, 2), (2, 1)):
        for text in ("a", "b"):
            session.append(
                "assistant/chunk",
                {
                    "turn": turn,
                    "step": step,
                    "chunk": {"type": "text-delta", "index": 0, "text": text},
                },
            )
    steps = recorded_steps(session.events)
    assert [(step.turn, step.step) for step in steps] == [(1, 1), (1, 2), (2, 1)]
    assert all(len(step.chunks) == 2 for step in steps)


async def test_replay_refuses_to_invent_a_step(mount: Any) -> None:
    ctx = await mount(REPLAY_ROW)
    ctx.llm_replay.steps = []
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, AgentOptions(provider="replay", model="m"))
    await agent.prompt("hello")
    # The turn fails rather than the replay fabricating output.
    reason = session.events[-1].data["reason"]
    assert reason["kind"] == "error"
    assert reason["error"]["code"] == "REPLAY_EXHAUSTED"
