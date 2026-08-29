"""P0-13 and P0-14 — the loop lifecycle, and the invariant that guards it.

Gates: *the lifecycle events appear in order on a fake run*; *the invariant
fires on a deliberately bypassed request.*

The second gate is the one that earns its keep. "Model-visible means logged"
(I3) is a claim about every request the harness will ever make; a test that only
checks the happy path proves nothing about the plugin someone writes next year.
So the check runs at runtime, on the request the adapter is about to receive.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ph.agent.types import (
    AgentCancelCause,
    AgentOptions,
    PreStepDecision,
    PreStepRequest,
    RequestErrorAction,
    RequestFailure,
    RequestProposal,
)
from ph.agent_loop.invariant import ModelVisibleNotLoggedError
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    LlmCallConfig,
    LlmFailure,
    TextBlock,
    TextDelta,
    create_user_message,
)
from ph.session import SurfaceIntent
from ph.system_prompt.assembly import PromptContext, PromptSection
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import user_payload

pytestmark = pytest.mark.anyio


def _plugin_snapshots(session: Any) -> list[Any]:
    return [
        e
        for e in session.events
        if e.type == "user/message" and e.data["source"]["kind"] == "plugin"
    ]


async def test_lifecycle_events_appear_in_order(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, FAKE).prompt("hello")

    types = [event.type for event in session.events]
    assert types[:7] == [
        "agent/inbox/spliced",
        "turn/start",
        "agent/inbox/spliced",
        # Before the first step, because the agent's cwd has to exist before
        # anything it does resolves against one (P4-08). Once per agent, not
        # once per turn: the seam already holds it on the second pass.
        "workspace/acquired",
        "step/start",
        "user/message",
        "request/header",
    ]
    assert types[-3:] == ["assistant/message", "step/end", "turn/end"]
    assert "assistant/chunk" in types
    assert session.events[-1].data["reason"]["kind"] == "completed"


async def test_every_request_is_exactly_derive_messages(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, FAKE).prompt("hello")
    assert ctx.llm_fake.requests
    derived = session.derive_messages()
    for request in ctx.llm_fake.requests:
        assert request.is_loop_request
        assert [m.id for m in request.messages] == [m.id for m in derived[: len(request.messages)]]


async def test_the_invariant_fires_on_a_bypassed_request(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    caught: list[BaseException] = []

    # A plugin that smuggles content past the log — the failure I3 exists to
    # make impossible. It must not reach the adapter. Note it does not have to
    # do anything to stay a loop request: session-bound and no other purpose.
    async def bypass(request: GenerateOptions, next_: Any) -> Any:
        forged = create_user_message(
            content=[{"type": "text", "text": "never logged"}],
            source={"kind": "plugin", "plugin": "smuggler"},
        )
        tampered = GenerateOptions(
            provider=request.provider,
            model=request.model,
            messages=(*request.messages, forged),
            session_id=request.session_id,
        )
        try:
            return await next_(tampered)
        except ModelVisibleNotLoggedError as error:
            caught.append(error)
            raise

    # Prepended so it sits OUTSIDE the invariant, letting the tampered request
    # reach it.
    ctx.on("llm/stream", bypass, prepend=True)

    await ctx.agents.create(session, FAKE).prompt("hello")
    assert caught, "the invariant did not fire on a bypassed request"
    # The turn ends in error rather than quietly succeeding on smuggled input.
    assert session.events[-1].data["reason"]["kind"] == "error"


async def test_pre_step_reject_blocks_the_turn(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")

    async def deny(request: PreStepRequest, next_: Any) -> PreStepDecision:
        return PreStepDecision(kind="reject", reason="over budget")

    ctx.on("agent/pre-step", deny)
    await ctx.agents.create(session, FAKE).prompt("hello")

    assert "step/start" not in [event.type for event in session.events]
    assert session.events[-1].data["reason"]["kind"] == "blocked"


async def test_agent_request_waterfall_can_reroute(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")

    async def reroute(proposal: RequestProposal, next_: Any) -> LlmCallConfig:
        config = await next_()
        return LlmCallConfig(provider=config.provider, model="rerouted", temperature=0.1)

    ctx.on("agent/request", reroute)
    await ctx.agents.create(session, FAKE).prompt("hello")

    header = session.request_header()
    assert header is not None
    assert header.config.model == "rerouted"
    assert header.config.temperature == 0.1


async def test_the_request_derives_its_messages_after_the_waterfall(mount: Any) -> None:
    """A listener may append, and the request it is proposing will carry it.

    Pinned because `ph-stabilize`'s `input-offload` leans on exactly this
    ordering: it appends a surface `replace` from `agent/request` and returns
    the config untouched, and the loop's own `derive_messages()` is what applies
    the substitution. Nothing in the waterfall's declared contract — "the call
    config the loop proposes" — says derivation happens afterwards, so the row
    depended on an ordering no test held. It does now.
    """
    ctx = await mount()
    session = ctx.sessions.create("s")

    async def inject(proposal: RequestProposal, next_: Any) -> LlmCallConfig:
        if not any(event.type == "assistant/message" for event in session.events):
            session.append(
                "user/message",
                user_payload("appended from agent/request", "injected"),
                SurfaceIntent("append"),
            )
        return await next_()

    ctx.on("agent/request", inject)
    await ctx.agents.create(session, FAKE).prompt("hello")

    # Read off the adapter, which is the only place the *sent* messages exist.
    sent = ctx.llm_fake.requests
    assert sent, "no request reached the adapter"
    text = [
        block.text
        for message in sent[0].messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "appended from agent/request" in text, (
        "the request was derived before the waterfall could add to the log"
    )


async def test_request_header_is_logged_only_when_it_changes(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    await agent.prompt("first")
    await agent.prompt("second")
    headers = [e for e in session.events if e.type == "request/header"]
    # Re-logging an unchanged header on every step is what breaks prefix
    # caching, so it is logged once and its reason recorded.
    assert len(headers) == 1
    assert headers[0].data["reason"] == "initial"


async def test_prompt_sections_are_static_and_context_is_snapshotted(mount: Any) -> None:
    ctx = await mount()
    clock = {"value": "09:00"}
    ctx.system_prompt.section(PromptSection(name="identity", text="You are pH.", order=-100))
    ctx.system_prompt.context(PromptContext(name="time", text=lambda _c: clock["value"]))

    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    await agent.prompt("first")

    header = session.request_header()
    assert header is not None and header.system == "You are pH."
    assert len(_plugin_snapshots(session)) == 1

    # Unchanged context is not re-sent: that is what keeps the cached prefix
    # stable across turns (A12).
    await agent.prompt("second")
    assert len(_plugin_snapshots(session)) == 1

    clock["value"] = "10:00"
    await agent.prompt("third")
    assert len(_plugin_snapshots(session)) == 2


async def test_adapter_failures_become_a_terminal_finish_and_end_the_turn(mount: Any) -> None:
    ctx = await mount()

    class Exploding:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            raise RuntimeError("provider is down")
            yield  # pragma: no cover

    ctx.llm.register_adapter(["boom"], Exploding())
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, AgentOptions(provider="boom", model="m")).prompt("hello")

    reason = session.events[-1].data["reason"]
    assert reason["kind"] == "error"
    assert reason["error"]["message"] == "provider is down"


async def test_request_error_waterfall_can_retry(mount: Any) -> None:
    ctx = await mount()
    attempts = {"count": 0}

    class Flaky:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                yield Finish(
                    reason=FinishReason(
                        kind="error", failure=LlmFailure(message="transient", code="TRANSIENT")
                    )
                )
                return
            yield BlockStart(index=0, block_type="text")
            yield TextDelta(index=0, text="recovered")
            yield BlockEnd(index=0, block=TextBlock(text="recovered"))
            yield Finish(reason=FinishReason(kind="stop"))

    ctx.llm.register_adapter(["flaky"], Flaky())

    async def retry_once(failure: RequestFailure, next_: Any) -> Any:
        if failure.failure.code == "TRANSIENT":
            return RequestErrorAction(kind="retry")
        return await next_()

    ctx.on("agent/request-error", retry_once)
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, AgentOptions(provider="flaky", model="m")).prompt("hello")

    assert attempts["count"] == 2
    assert session.events[-1].data["reason"]["kind"] == "completed"
    assert session.derive_messages()[-1].content[0].text == "recovered"


async def test_turn_stopping_listener_can_keep_the_turn_alive(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    seen = {"count": 0}

    def object_once(agent_handle: Any, turn: int) -> None:
        seen["count"] += 1
        if seen["count"] == 1:
            agent_handle.steer(
                create_user_message(
                    content=[{"type": "text", "text": "keep going"}], source={"kind": "user"}
                )
            )

    ctx.on("agent/turn-stopping", object_once)
    await agent.prompt("hello")

    # One turn, two steps: the listener objected by steering rather than by
    # reaching into loop state.
    types = [e.type for e in session.events]
    assert types.count("turn/start") == 1
    assert types.count("step/start") == 2


async def test_cancelling_ends_the_turn_as_aborted(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)

    async def cancel_at_pre_step(request: PreStepRequest, next_: Any) -> Any:
        agent.cancel(AgentCancelCause(kind="user"), keep_inbox=True)
        return await next_()

    ctx.on("agent/pre-step", cancel_at_pre_step)
    await agent.prompt("hello")
    reason = session.events[-1].data["reason"]
    assert reason["kind"] == "aborted"
    assert reason["reason"]["kind"] == "user"


async def test_agent_scoped_listeners_hear_only_their_agent(mount: Any) -> None:
    ctx = await mount()
    a = ctx.agents.create(ctx.sessions.create("a"), FAKE)
    b = ctx.agents.create(ctx.sessions.create("b"), FAKE)
    heard: list[str] = []
    a.ctx.on("agent/status", lambda agent, status: heard.append(f"{agent.id}:{status}"))
    await b.prompt("hello")
    assert heard == []
    await a.prompt("hello")
    assert heard == ["a:running", "a:idle"]
