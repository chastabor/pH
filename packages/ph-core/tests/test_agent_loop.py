"""P0-13 and P0-14 — the loop lifecycle, and the invariant that guards it.

Gates: *the lifecycle events appear in order on a fake run*; *the invariant
fires on a deliberately bypassed request.*

The second gate is the one that earns its keep. "Model-visible means logged"
(I3) is a claim about every request the harness will ever make; a test that only
checks the happy path proves nothing about the plugin someone writes next year.
So the check runs at runtime, on the request the adapter is about to receive.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from ph.agent.inbox import InboxSplice
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
from ph.json import as_obj, as_str, thaw_json
from ph.keys import AGENTS, LLM, LLM_FAKE, SESSIONS, SYSTEM_PROMPT, TOOLS
from ph.llm.replay import RecordedStep, ReplayAdapter, text_chunks, tool_call_chunks
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
from ph.testing import MountProfile, block_text, simple_tool, user_payload

pytestmark = pytest.mark.anyio


def _plugin_snapshots(session: Any) -> list[Any]:  # noqa: ANN401
    return [
        e
        for e in session.events
        if e.type == "user/message" and e.data["source"]["kind"] == "plugin"
    ]


async def test_lifecycle_events_appear_in_order(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")

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
    assert as_obj(session.events[-1].data["reason"])["kind"] == "completed"


async def test_every_request_is_exactly_derive_messages(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")
    assert ctx.require(LLM_FAKE).requests
    derived = session.derive_messages()
    for request in ctx.require(LLM_FAKE).requests:
        assert request.is_loop_request
        assert [m.id for m in request.messages] == [m.id for m in derived[: len(request.messages)]]


async def test_the_invariant_fires_on_a_bypassed_request(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    caught: list[BaseException] = []

    # A plugin that smuggles content past the log — the failure I3 exists to
    # make impossible. It must not reach the adapter. Note it does not have to
    # do anything to stay a loop request: session-bound and no other purpose.
    async def bypass(request: GenerateOptions, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
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

    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")
    assert caught, "the invariant did not fire on a bypassed request"
    # The turn ends in error rather than quietly succeeding on smuggled input.
    assert as_obj(session.events[-1].data["reason"])["kind"] == "error"


async def test_pre_step_reject_blocks_the_turn(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")

    async def deny(request: PreStepRequest, next_: object) -> PreStepDecision:
        return PreStepDecision(kind="reject", reason="over budget")

    ctx.on("agent/pre-step", deny)
    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")

    assert "step/start" not in [event.type for event in session.events]
    assert as_obj(session.events[-1].data["reason"])["kind"] == "blocked"


async def test_agent_request_waterfall_can_reroute(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")

    async def reroute(
        proposal: RequestProposal, next_: Callable[..., Awaitable[Any]]
    ) -> LlmCallConfig:
        config = await next_()
        return LlmCallConfig(provider=config.provider, model="rerouted", temperature=0.1)

    ctx.on("agent/request", reroute)
    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")

    header = session.request_header()
    assert header is not None
    assert header.config.model == "rerouted"
    assert header.config.temperature == 0.1


async def test_the_request_derives_its_messages_after_the_waterfall(mount: MountProfile) -> None:
    """A listener may append, and the request it is proposing will carry it.

    Pinned because `ph-stabilize`'s `input-offload` leans on exactly this
    ordering: it appends a surface `replace` from `agent/request` and returns
    the config untouched, and the loop's own `derive_messages()` is what applies
    the substitution. Nothing in the waterfall's declared contract — "the call
    config the loop proposes" — says derivation happens afterwards, so the row
    depended on an ordering no test held. It does now.
    """
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")

    async def inject(
        proposal: RequestProposal, next_: Callable[..., Awaitable[LlmCallConfig]]
    ) -> LlmCallConfig:
        if not any(event.type == "assistant/message" for event in session.events):
            session.append(
                "user/message",
                user_payload("appended from agent/request", "injected"),
                SurfaceIntent("append"),
            )
        return await next_()

    ctx.on("agent/request", inject)
    await ctx.require(AGENTS).create(session, FAKE).prompt("hello")

    # Read off the adapter, which is the only place the *sent* messages exist.
    sent = ctx.require(LLM_FAKE).requests
    assert sent, "no request reached the adapter"
    text = [
        block_text(block)
        for message in sent[0].messages
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    assert "appended from agent/request" in text, (
        "the request was derived before the waterfall could add to the log"
    )


async def test_request_header_is_logged_only_when_it_changes(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    await agent.prompt("first")
    await agent.prompt("second")
    headers = [e for e in session.events if e.type == "request/header"]
    # Re-logging an unchanged header on every step is what breaks prefix
    # caching, so it is logged once and its reason recorded.
    assert len(headers) == 1
    assert headers[0].data["reason"] == "initial"


async def test_prompt_sections_are_static_and_context_is_snapshotted(mount: MountProfile) -> None:
    ctx = await mount()
    clock = {"value": "09:00"}
    ctx.require(SYSTEM_PROMPT).section(
        PromptSection(name="identity", text="You are pH.", order=-100)
    )
    ctx.require(SYSTEM_PROMPT).context(PromptContext(name="time", text=lambda _c: clock["value"]))

    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
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


async def test_adapter_failures_become_a_terminal_finish_and_end_the_turn(
    mount: MountProfile,
) -> None:
    ctx = await mount()

    class Exploding:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            raise RuntimeError("provider is down")
            yield  # pragma: no cover

    ctx.require(LLM).register_adapter(["boom"], Exploding())
    session = ctx.require(SESSIONS).create("s")
    await (
        ctx.require(AGENTS)
        .create(session, AgentOptions(provider="boom", model="m"))
        .prompt("hello")
    )

    reason = session.events[-1].data["reason"]
    assert as_obj(reason)["kind"] == "error"
    assert as_obj(as_obj(reason)["error"])["message"] == "provider is down"


async def test_request_error_waterfall_can_retry(mount: MountProfile) -> None:
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

    ctx.require(LLM).register_adapter(["flaky"], Flaky())

    async def retry_once(failure: RequestFailure, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        if failure.failure.code == "TRANSIENT":
            return RequestErrorAction(kind="retry")
        return await next_()

    ctx.on("agent/request-error", retry_once)
    session = ctx.require(SESSIONS).create("s")
    await (
        ctx.require(AGENTS)
        .create(session, AgentOptions(provider="flaky", model="m"))
        .prompt("hello")
    )

    assert attempts["count"] == 2
    assert as_obj(session.events[-1].data["reason"])["kind"] == "completed"
    assert block_text(session.derive_messages()[-1].content[0]) == "recovered"


async def test_turn_stopping_listener_can_keep_the_turn_alive(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    seen = {"count": 0}

    def object_once(agent_handle: Any, turn: int) -> None:  # noqa: ANN401
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


async def test_canceling_ends_the_turn_as_aborted(mount: MountProfile) -> None:
    ctx = await mount()
    session = ctx.require(SESSIONS).create("s")
    agent = ctx.require(AGENTS).create(session, FAKE)

    async def cancel_at_pre_step(
        request: PreStepRequest, next_: Callable[..., Awaitable[Any]]
    ) -> Any:  # noqa: ANN401
        agent.cancel(AgentCancelCause(kind="user"), keep_inbox=True)
        return await next_()

    ctx.on("agent/pre-step", cancel_at_pre_step)
    await agent.prompt("hello")
    reason = session.events[-1].data["reason"]
    assert as_obj(reason)["kind"] == "aborted"
    assert as_obj(as_obj(reason)["reason"])["kind"] == "user"


async def test_agent_scoped_listeners_hear_only_their_agent(mount: MountProfile) -> None:
    ctx = await mount()
    a = ctx.require(AGENTS).create(ctx.require(SESSIONS).create("a"), FAKE)
    b = ctx.require(AGENTS).create(ctx.require(SESSIONS).create("b"), FAKE)
    heard: list[str] = []
    a.ctx.on("agent/status", lambda agent, status: heard.append(f"{agent.id}:{status}"))
    await b.prompt("hello")
    assert heard == []
    await a.prompt("hello")
    assert heard == ["a:running", "a:idle"]


@pytest.mark.parametrize(
    ("splice", "field"),
    [
        ({"inserted": [{"role": "user"}]}, "inserted.0.id"),
        ({"inserted": ["nope"]}, "inserted.0"),
        ({"inserted": "nope"}, "inserted"),
        ({"target": "elsewhere", "inserted": []}, "target"),
        ({"start": "x", "inserted": []}, "start"),
    ],
)
async def test_a_corrupt_inbox_splice_names_the_field_that_is_wrong(
    mount: MountProfile, splice: dict[str, Any], field: str
) -> None:
    """Replay refuses a log it cannot read, and says which event and which field.

    The field is the assertion. Before `InboxSplice` these were hand-parsed, and
    two of the five reached `Inbox.__init__` as `KeyError`/`TypeError` — past the
    `except ValueError` written to name the seq — while two malformed items in one
    splice both read as id `""` and tripped the duplicate rule, reporting a
    message collision that was not the fault.
    """
    ctx = await mount()
    session = ctx.require(SESSIONS).create("corrupt")
    session.append("agent/inbox/spliced", {"target": "next-turn", "start": 0, **splice})

    with pytest.raises(ValueError, match="invalid persisted inbox splice") as caught:
        ctx.require(AGENTS).create(session, FAKE)
    assert field in str(caught.value.__cause__)


TEXT_A: dict[str, Any] = {"type": "text", "text": "a"}
TEXT_B: dict[str, Any] = {"type": "text", "text": "b"}


async def test_an_inbox_splice_written_today_is_read_back_unchanged(
    mount: MountProfile,
) -> None:
    """The durable form is the model's form, for every shape a write can take.

    `agent/inbox/spliced` is on disk in logs this build did not write, so the keys
    and their absences are a compatibility surface: `removedCount` and `outcome`
    are omitted rather than defaulted, and a key a later build adds is skipped
    rather than refused.
    """
    ctx = await mount()
    session = ctx.require(SESSIONS).create("rt")
    agent = ctx.require(AGENTS).create(session, FAKE)
    inbox = agent.inbox
    said = {"kind": "user"}

    inbox.append("next-turn", create_user_message(content=[TEXT_A], source=said))
    inbox.append("next-step", create_user_message(content=[TEXT_B], source=said))
    inbox.claim("next-step", 1)
    inbox.clear()

    # `thaw_json`, because the log freezes payloads into `MappingProxyType`/tuple
    # and `to_wire()` answers plain dicts and lists — a difference of container,
    # not of content, and not what this test is about.
    written = [thaw_json(one.data) for one in session.events if one.type == "agent/inbox/spliced"]
    assert written, "no splices were recorded"
    for payload in written:
        assert set(payload) <= {"target", "start", "inserted", "removedCount", "outcome"}
        assert payload.get("removedCount") != 0, "an absent count must not be written as 0"
        assert InboxSplice.model_validate(payload).to_wire() == payload

    payload = {**written[0], "futureKey": 1}
    assert "futureKey" not in InboxSplice.model_validate(payload).to_wire()


async def test_a_tool_step_after_max_tokens_gets_a_model_call(mount: MountProfile) -> None:
    """A cap on one step must not end a turn that has more to do.

    `max-tokens` is sticky, and rightly: a turn whose answer was cut off has to
    say so however tidily it finishes afterwards. But the sticky value was also
    the deciding one — one variable answered both "what do we report" and "are we
    done" — so once a step had been capped, the `None` that means *tools ran,
    keep going* was discarded. The tool then ran, its result was logged, and the
    turn ended with a `tool/result` the model was never shown.

    A steer re-opens the turn here, which is the ordinary way a capped turn
    continues. The control is the same script ending in `stop`: what the two must
    share is the number of model calls, and what they must not share is the
    reason.
    """

    async def drive(first: FinishReason) -> tuple[int, str]:
        ctx = await mount()
        adapter = ReplayAdapter(
            steps=[
                RecordedStep(
                    turn=1, step=1, chunks=(*text_chunks("partial")[:-1], Finish(reason=first))
                ),
                RecordedStep(turn=1, step=2, chunks=tool_call_chunks("c1", "ping", "{}")),
                RecordedStep(turn=2, step=1, chunks=text_chunks("done")),
            ]
        )
        ctx.require(LLM).register_adapter(["scripted"], adapter)
        ctx.require(TOOLS).register(simple_tool("ping"))

        async def keep_alive(agent: Any, turn: int) -> None:  # noqa: ANN401
            # **Exactly one steer, and only after the capped call.** This hook
            # fires again when the turn really does stop, and steering there too
            # would open a fourth call; steering on every firing re-opens a turn
            # per step, which is what the bug did — so the test would pass either
            # way. The adapter's own count says which firing this is.
            if len(adapter.requests) == 1:
                agent.steer(
                    create_user_message(
                        content=[{"type": "text", "text": "continue"}], source={"kind": "user"}
                    )
                )

        ctx.on("agent/turn-stopping", keep_alive)
        session = ctx.require(SESSIONS).create("s")
        await (
            ctx.require(AGENTS)
            .create(session, AgentOptions(provider="scripted", model="m"))
            .prompt("hello")
        )
        return len(adapter.requests), as_str(as_obj(session.events[-1].data["reason"])["kind"])

    capped_calls, capped_reason = await drive(FinishReason(kind="max-tokens"))
    plain_calls, plain_reason = await drive(FinishReason(kind="stop"))

    assert capped_calls == plain_calls == 3, "the cap swallowed the tool continuation"
    assert plain_reason == "completed"
    assert capped_reason == "max-tokens", "and the cap is still what the turn reports"
