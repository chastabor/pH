"""The transcript is a fold over the log — the P2-01 gate.

The load-bearing claim: a resumed session and a live one show the same
conversation, because both are built from `session.events` and nothing else. If
the adapter ever grew a live-only shortcut — a chunk it kept that the replay
could not reconstruct — these two sequences would drift, and the person who
resumed would silently see a different transcript from the one they left.

The second claim is narrower and just as easy to lose: the adapter reads the
log's plain JSON, not models, so every field is optional and a malformed one is
a missing row rather than a crash.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.llm.adapter import ResolvedModel
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    UsageChunk,
)
from ph.session import Session, SessionEvent, SurfaceIntent, SurfaceReplace
from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES
from ph.testing import simple_tool, user_payload
from ph_app.tui.adapter import HANDLERS, RECORDLESS, TuiEventAdapter
from ph_app.tui.state import TuiState

pytestmark = pytest.mark.anyio

SCRIPTED = AgentOptions(provider="scripted", model="s1")


class _CallsThenAnswers:
    """Calls `ping` on the first request, then answers.

    The gate needs a tool call in it. A transcript that only ever holds text
    would pass while the card projection — built from `tool/call` and
    `tool/result` — diverged between live and replay, which is exactly where a
    live-only shortcut would be tempting to write.
    """

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, options: GenerateOptions) -> Any:
        self.requests += 1
        if self.requests == 1:
            yield BlockStart(index=0, block_type="tool-call")
            yield BlockEnd(index=0, block=ToolCallBlock(id="c1", name="ping", arguments="{}"))
            yield Finish(reason=FinishReason(kind="tool-calls"))
            return
        yield BlockStart(index=0, block_type="text")
        yield TextDelta(index=0, text="done")
        yield BlockEnd(index=0, block=TextBlock(text="done"))
        yield UsageChunk(usage=TokenUsage(input_tokens=120, output_tokens=4))
        yield Finish(reason=FinishReason(kind="stop"))

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(context_window=8192)


def _shape(state: TuiState) -> list[tuple[str, str]]:
    return [(item.role, item.text) for item in state.visible_items()]


def _replay(session: Session) -> TuiState:
    return TuiEventAdapter().replay(session)


async def _drive(mount: Any, *, prompt: str = "hello there") -> tuple[TuiState, Session]:
    """Run one prompt with a tool registered, collecting the live transcript."""
    ctx: Context = await mount()
    live = TuiEventAdapter(tools=ctx.get("tools"))

    def observe(_source: Session, event: SessionEvent) -> None:
        live.apply(event)

    ctx.on("session/event", observe)
    ctx.tools.register(simple_tool("ping", lambda _args, _run: "pong"))
    ctx.llm.register_adapter(("scripted",), _CallsThenAnswers())
    session = ctx.sessions.create("tui-gate")
    agent = ctx.agents.create(session, SCRIPTED)
    await agent.prompt(prompt)
    return live.state, session


async def test_replay_and_live_agree(mount: Any) -> None:
    live_state, session = await _drive(mount)
    # Non-trivial: the prompt, the tool card, and the answer.
    assert [role for role, _ in _shape(live_state)] == ["user", "tool", "assistant"]
    assert _shape(_replay(session)) == _shape(live_state)


async def test_the_tool_card_settles_the_same_way_on_replay(mount: Any) -> None:
    live_state, session = await _drive(mount)
    replayed = _replay(session)
    live_card = next(item.tool for item in live_state.items if item.tool is not None)
    replayed_card = next(item.tool for item in replayed.items if item.tool is not None)
    assert live_card.name == replayed_card.name == "ping"
    assert live_card.settled and replayed_card.settled
    assert live_card.is_error is replayed_card.is_error is False
    assert live_card.body == replayed_card.body


async def test_the_user_prompt_reaches_the_transcript(mount: Any) -> None:
    live_state, _ = await _drive(mount, prompt="hello there")
    assert ("user", "hello there") in _shape(live_state)


async def test_bracketed_text_is_carried_verbatim(mount: Any) -> None:
    """Markup is never parsed on the way in — the widgets do that check too."""
    typed = "run foo[0] and [bold]not bold[/bold]"
    live_state, session = await _drive(mount, prompt=typed)
    assert ("user", typed) in _shape(live_state)
    assert ("user", typed) in _shape(_replay(session))


async def test_a_malformed_event_costs_one_row_not_the_transcript(mount: Any) -> None:
    ctx: Context = await mount()
    session = ctx.sessions.create("tui-malformed")
    adapter = TuiEventAdapter()
    session.append("user/message", {"content": "not a block list"}, SurfaceIntent("append"))
    session.append("tool/result", {"message": None}, SurfaceIntent("append"))
    for event in session.events:
        adapter.apply(event)
    # No exception, and the transcript is still a list of rows.
    assert isinstance(adapter.state.items, list)


async def test_an_unknown_event_type_is_ignored(mount: Any) -> None:
    adapter = TuiEventAdapter()
    adapter.apply(
        SessionEvent.from_wire(
            {"type": "something/newer", "seq": 1, "time": 1, "data": {}, "ignorable": True}
        )
    )
    assert adapter.state.items == []


async def test_compaction_marks_what_it_replaced_and_keeps_it(mount: Any) -> None:
    """The gate's other half: a compacted range is dimmed, never dropped.

    Rebuilding from `derive_messages()` would delete these rows, because that is
    the model's view and the summary shadows them there.
    """
    ctx: Context = await mount()
    session = ctx.sessions.create("tui-compaction")
    first = session.append(
        "user/message", user_payload("the original question", "m1"), SurfaceIntent("append")
    )
    session.append(
        "user/message",
        user_payload("(summary of earlier conversation)", "m2"),
        SurfaceIntent(SurfaceReplace(start=0, end=0), (first.seq,)),
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, live=False)

    roles = [(item.role, item.shadowed) for item in adapter.state.visible_items()]
    assert ("user", True) in roles, "the replaced row must survive, marked"
    assert ("compaction", False) in roles, "the summary is what the model sees now"
    original = next(item for item in adapter.state.items if item.role == "user")
    assert original.text == "the original question"
    assert original.is_visible_to_model is False


async def test_a_plugins_replacement_is_not_called_a_compaction(mount: Any) -> None:
    """A surface `replace` is a mechanism, not a cause (P4-02).

    `input-offload` is the first row to substitute on the surface for a reason
    other than compaction, and until this test the adapter keyed on the
    replacement alone: an offloaded paste rendered as "(history compacted)",
    telling the reader their conversation had been summarized when a blob had
    been relocated. The attribution the log already carries is the discriminator
    — and the shadowing, which *is* the mechanism, applies either way.
    """
    ctx: Context = await mount()
    session = ctx.sessions.create("tui-offload")
    pasted = session.append(
        "user/message", user_payload("a two megabyte paste", "m1"), SurfaceIntent("append")
    )
    session.append(
        "user/message",
        {
            "id": "m2",
            "role": "user",
            "content": [{"type": "text", "text": "Message content too large…"}],
            "source": {"kind": "plugin", "plugin": "input-offload", "form": "notice"},
        },
        SurfaceIntent(SurfaceReplace(start=0, end=0), (pasted.seq,)),
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, live=False)

    roles = [(item.role, item.shadowed) for item in adapter.state.visible_items()]
    assert ("context", False) in roles, "a plugin's notice is that plugin's row"
    assert "compaction" not in [role for role, _ in roles]
    assert ("user", True) in roles, "the paste is still there, dimmed — the mechanism holds"


async def test_usage_feeds_the_context_gauge(mount: Any) -> None:
    """The gauge reads the provider's count from `assistant/message.usage`."""
    live_state, session = await _drive(mount)
    assert live_state.tokens > 0
    assert live_state.context_window == 8192
    assert live_state.pressure is not None
    assert _replay(session).tokens == live_state.tokens


async def test_a_refinement_says_what_changed(mount: Any) -> None:
    """A refinement changes the model's own prompt, so it is a row, not a record.

    `/refine` is only one way here — the planner refines at turn end with no
    command to show for it — so a user who could not see this would have no way
    to know why the next turn behaves differently.
    """
    ctx: Context = await mount()
    session = ctx.sessions.create("tui-harness")
    session.append(
        "harness/refined",
        {
            "refineId": "refine-1",
            "scope": "local",
            "summary": "learned to prefer uv",
            "appliedEdits": [{"action": "create", "kind": "note", "id": "prefer-uv"}],
            "rejected": ['skill "imaginary" does not resolve'],
        },
    )
    session.append(
        "harness/refined",
        {
            "refineId": "refine-2",
            "scope": "local",
            "summary": "rollback of refine-1",
            "appliedEdits": [{"action": "delete", "kind": "note", "id": "prefer-uv"}],
            "rollbackOf": "refine-1",
        },
    )
    applied, rolled = (item.text for item in _replay(session).items)
    assert "learned to prefer uv" in applied and "1 edit(s)" in applied
    assert "1 edit(s) refused" in applied, "the refused half is the interesting one"
    assert "Rolled back refine-1" in rolled


async def test_a_corpus_is_a_row_only_when_it_changed(mount: Any) -> None:
    """The ordinary load is already described in the system prompt; what is news
    is the corpus having changed under a conversation that was told about it."""
    ctx: Context = await mount()
    session = ctx.sessions.create("tui-context")
    session.append("context/loaded", {"corpus": "notes", "digest": "a", "note": ""})
    session.append(
        "context/loaded",
        {"corpus": "notes", "digest": "b", "note": "`notes` was rebuilt from changed sources"},
    )

    (item,) = _replay(session).items
    assert "was rebuilt from changed sources" in item.text


def test_every_known_event_type_is_rendered_or_classified() -> None:
    """The adapter's vocabulary equals the log's — no silent omissions.

    A new event type has to land in `HANDLERS` or be named in `RECORDLESS`.
    Strict equality: `todo/write` was carried as a declared forward reference
    from Phase 2 until its producer landed (P4-01), and the subtraction that
    allowed it went with it — the next type rendered ahead of the vocabulary
    should fail here loudly and be argued for, not slip through a standing
    exemption.
    """
    assert set(HANDLERS) & RECORDLESS == set()
    assert set(HANDLERS) | RECORDLESS == KNOWN_SESSION_EVENT_TYPES
