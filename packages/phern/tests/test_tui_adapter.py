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

from ph.agent.inbox import Inbox, InboxNotifications
from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.keys import AGENTS, LLM, SESSIONS, TOOLS
from ph.llm.adapter import ResolvedModel
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    PluginSource,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    UsageChunk,
    create_user_message,
)
from ph.session import Session, SessionEvent, SurfaceIntent, SurfaceReplace
from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES
from ph.testing import MountProfile, assistant_payload, plugin_payload, simple_tool, user_payload
from ph_app.tui.adapter import HANDLERS, RECORDLESS, REPLAY, RULES, TuiEventAdapter
from ph_app.tui.state import Surface, TuiState

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

    async def stream(self, options: GenerateOptions) -> Any:  # noqa: ANN401
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


async def _drive(mount: MountProfile, *, prompt: str = "hello there") -> tuple[TuiState, Session]:
    """Run one prompt with a tool registered, collecting the live transcript."""
    ctx: Context = await mount()
    live = TuiEventAdapter(tools=ctx.get("tools"))

    def observe(_source: Session, event: SessionEvent) -> None:
        live.apply(event)

    ctx.on("session/event", observe)
    ctx.require(TOOLS).register(simple_tool("ping", lambda _args, _run: "pong"))
    ctx.require(LLM).register_adapter(("scripted",), _CallsThenAnswers())
    session = ctx.require(SESSIONS).create("tui-gate")
    agent = ctx.require(AGENTS).create(session, SCRIPTED)
    await agent.prompt(prompt)
    return live.state, session


async def test_replay_and_live_agree(mount: MountProfile) -> None:
    live_state, session = await _drive(mount)
    # Non-trivial: the prompt, the tool card, and the answer.
    assert [role for role, _ in _shape(live_state)] == ["user", "tool", "assistant"]
    assert _shape(_replay(session)) == _shape(live_state)


async def test_the_tool_card_settles_the_same_way_on_replay(mount: MountProfile) -> None:
    live_state, session = await _drive(mount)
    replayed = _replay(session)
    live_card = next(item.tool for item in live_state.items if item.tool is not None)
    replayed_card = next(item.tool for item in replayed.items if item.tool is not None)
    assert live_card.name == replayed_card.name == "ping"
    assert live_card.settled and replayed_card.settled
    assert live_card.is_error is replayed_card.is_error is False
    assert live_card.body == replayed_card.body


async def test_the_user_prompt_reaches_the_transcript(mount: MountProfile) -> None:
    live_state, _ = await _drive(mount, prompt="hello there")
    assert ("user", "hello there") in _shape(live_state)


async def test_bracketed_text_is_carried_verbatim(mount: MountProfile) -> None:
    """Markup is never parsed on the way in — the widgets do that check too."""
    typed = "run foo[0] and [bold]not bold[/bold]"
    live_state, session = await _drive(mount, prompt=typed)
    assert ("user", typed) in _shape(live_state)
    assert ("user", typed) in _shape(_replay(session))


async def test_a_malformed_event_costs_one_row_not_the_transcript(mount: MountProfile) -> None:
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-malformed")
    adapter = TuiEventAdapter()
    session.append("user/message", {"content": "not a block list"}, SurfaceIntent("append"))
    session.append("tool/result", {"message": None}, SurfaceIntent("append"))
    for event in session.events:
        adapter.apply(event)
    # No exception, and the transcript is still a list of rows.
    assert isinstance(adapter.state.items, list)


async def test_a_chunk_moves_the_transcript_and_nothing_else(mount: MountProfile) -> None:
    """The entry the surface table exists for.

    An `assistant/chunk` arrives faster than the draw's coalescing window for
    the whole of a streaming turn, so before the table every one of them redrew
    the session panel, the todo list and the subagent fold — none of which a
    chunk can move. The draw asks `take_touched` what actually changed.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-surfaces")
    adapter = TuiEventAdapter()
    session.append("assistant/chunk", {"turn": 1, "step": 1, "chunk": {"type": "text-delta"}})
    for event in session.events:
        adapter.apply(event)

    assert adapter.take_touched() == Surface.TRANSCRIPT
    assert adapter.take_touched() == Surface.NOTHING, "and asking again finds nothing new"


async def test_a_panel_event_does_not_redraw_the_transcript(mount: MountProfile) -> None:
    """The other direction, so the table is not just "chunks are cheap".

    A todo write and a child's token counter move the sidebar and nothing else;
    re-syncing a long transcript for either is the cost this avoids.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-panels")
    adapter = TuiEventAdapter()
    session.append("todo/write", {"todos": []})
    for event in session.events:
        adapter.apply(event)

    assert adapter.take_touched() == Surface.SIDEBAR


async def test_a_rule_written_without_surfaces_redraws_everything(mount: MountProfile) -> None:
    """`ALL` is the default, and that is the safety property.

    A surface left stale shows yesterday's answer; a redraw nobody needed costs
    a frame. So an event type earns a narrow entry rather than losing one, and
    a rule written without surfaces is correct by default.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-default")
    adapter = TuiEventAdapter()
    plain = next(name for name, rule in RULES.items() if rule.surfaces == Surface.ALL)
    session.append(plain, {})
    for event in session.events:
        adapter.apply(event)

    assert adapter.take_touched() == Surface.ALL


def test_a_handler_that_draws_a_row_declares_the_transcript() -> None:
    """The half of a rule that can be derived, held against what it says.

    The handler and its surfaces are one `EventRule` now, so they cannot be
    edited apart — but a rule can still *say* the wrong thing, and one did:
    `subagent/admitted` draws "Delegated to X" and was declared `SIDEBAR`
    alone, so the row would have waited in the fold until something unrelated
    marked the transcript — a stale pane, which is the exact failure the
    per-surface draw was introduced to avoid.

    **One direction only.** A handler that draws a row must declare the
    transcript; the converse is false and must stay false —
    `assistant/chunk` streams *into* a row that already exists rather than
    adding one, so it declares `TRANSCRIPT` and calls neither `_row` nor
    `_card_row`. `FOOTER` and `SIDEBAR` cannot be derived at all: a handler
    assigning `self.state.todos` is not distinguishable by shape from one
    assigning `queued`.

    Reading the handler's source rather than running it, because the question
    is what it *would* draw.
    """
    import ast
    import inspect
    import textwrap

    for event_type, rule in RULES.items():
        if rule.handler is None:
            continue
        surfaces = rule.surfaces
        tree = ast.parse(textwrap.dedent(inspect.getsource(rule.handler)))
        draws = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"_row", "_card_row"}
            for node in ast.walk(tree)
        )
        assert not draws or Surface.TRANSCRIPT in surfaces, (
            f"{event_type} draws a row and does not declare TRANSCRIPT"
        )
    assert TuiEventAdapter is not None


async def test_an_unknown_event_type_is_ignored(mount: MountProfile) -> None:
    adapter = TuiEventAdapter()
    adapter.apply(
        SessionEvent.from_wire(
            {"type": "something/newer", "seq": 1, "time": 1, "data": {}, "ignorable": True}
        )
    )
    assert adapter.state.items == []


async def test_compaction_marks_what_it_replaced_and_keeps_it(mount: MountProfile) -> None:
    """The gate's other half: a compacted range is dimmed, never dropped.

    Rebuilding from `derive_messages()` would delete these rows, because that is
    the model's view and the summary shadows them there.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-compaction")
    first = session.append(
        "user/message", user_payload("the original question", "m1"), SurfaceIntent("append")
    )
    session.append(
        "user/message",
        # Attributed the way `compaction-summarize` attributes it (P4-03): a
        # plugin's text, declaring `form: compaction`. Built with
        # `user_payload` this test passed while asserting against a message no
        # producer writes — a person's own words never shadow anything.
        plugin_payload(
            "(summary of earlier conversation)",
            "m2",
            plugin="compaction-summarize",
            form="compaction",
            summary="1 message summarized",
        ),
        SurfaceIntent(SurfaceReplace(replaces=(first.seq,)), (first.seq,)),
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, REPLAY)

    roles = [(item.role, item.shadowed) for item in adapter.state.visible_items()]
    assert ("user", True) in roles, "the replaced row must survive, marked"
    assert ("compaction", False) in roles, "the summary is what the model sees now"
    original = next(item for item in adapter.state.items if item.role == "user")
    assert original.text == "the original question"
    assert original.is_visible_to_model is False


async def test_an_argument_truncation_does_not_add_a_second_assistant_row(
    mount: MountProfile,
) -> None:
    """Argument truncation (P4-03) rewrites an old assistant message in place.

    Two things must not happen. Its text is the model's own and is already on
    screen, so rendering the replacement would show the assistant saying the
    same thing twice; and the rows it stands for must *not* be dimmed, because
    the message is still exactly what the model sees — only a tool-call argument
    was elided.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-truncated")
    original = session.append(
        "assistant/message",
        {
            **assistant_payload("here is the file", "a1"),
            "usage": {"inputTokens": 400, "outputTokens": 10},
        },
        SurfaceIntent("append"),
    )
    session.append(
        "assistant/message",
        assistant_payload("here is the file", "a1"),
        SurfaceIntent(SurfaceReplace(replaces=(original.seq,)), (original.seq,)),
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, REPLAY)

    rows = [(item.role, item.shadowed) for item in adapter.state.visible_items()]
    assert rows == [("assistant", False)], "the replacement drew a second row"


async def test_a_truncation_replacement_does_not_reset_the_token_footer(
    mount: MountProfile,
) -> None:
    """The other half of the same hazard, and the one a reader would not guess.

    The footer shows the last reported usage it saw. A replacement is appended
    at the end of the log, so falling through would set the footer from whatever
    turn the *rewritten* message belonged to — the same stale-baseline bug the
    engine avoids by dropping `usage` from the replacement.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-truncated-usage")
    old = session.append(
        "assistant/message",
        {**assistant_payload("first", "a1"), "usage": {"inputTokens": 100, "outputTokens": 0}},
        SurfaceIntent("append"),
    )
    session.append(
        "assistant/message",
        {**assistant_payload("second", "a2"), "usage": {"inputTokens": 900, "outputTokens": 0}},
        SurfaceIntent("append"),
    )
    session.append(
        "assistant/message",
        assistant_payload("first", "a1"),
        SurfaceIntent(SurfaceReplace(replaces=(old.seq,)), (old.seq,)),
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, REPLAY)

    assert adapter.state.tokens == 900


async def test_truncated_arguments_are_announced(mount: MountProfile) -> None:
    """The tool cards above still show the arguments as sent, so this notice is
    the only place the transcript can say the model is no longer shown them."""
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-truncation-notice")
    session.append(
        "compaction/args-truncated",
        {"trigger": "pressure", "seqs": [3, 7], "savedChars": 41_000},
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, REPLAY)

    (row,) = [item for item in adapter.state.visible_items() if item.role == "notice"]
    assert "2 messages" in row.text
    assert "41000" in row.text


async def test_a_declined_compaction_is_a_notice(mount: MountProfile) -> None:
    """A compaction that did not happen leaves no row of its own.

    Its successful sibling does — the summary the replacement carries — which is
    why `compaction/summarized` is record-less here and this one is not. The
    reader is at the limit the compaction would have relieved, and the next
    thing they may see is a turn ending in a provider refusal.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-declined")
    session.append(
        "compaction/declined",
        {"trigger": "overflow", "code": "summary", "reason": "the summarize call failed"},
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, REPLAY)

    (row,) = [item for item in adapter.state.visible_items() if item.role == "notice"]
    assert "the summarize call failed" in row.text


async def test_a_surfaced_shell_command_is_marked_apart_from_a_quiet_one(
    mount: MountProfile,
) -> None:
    """`!` and `!!` must not draw the same card.

    The two differ in exactly one consequence — whether the agent reads the
    output — and in none of the three things a shell card otherwise shows: the
    command, what it printed, the exit code. A person who cannot tell them apart
    on screen cannot tell whether they just handed the model 4 KiB of build log.

    Read off the event, so a resumed session marks it the same way it was drawn
    live — the P2-01 claim this file exists for, applied to the one field that
    says where a command's output went.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-shell-mark")
    session.append("shell/command", {"command": "make", "surface": True})
    session.append("shell/command", {"command": "make", "surface": False})

    titles = [item.tool.title for item in _replay(session).visible_items() if item.tool]
    assert titles == ["Shell → agent", "Shell"], "the loud one is named and the quiet one is not"


async def test_canceled_pending_input_leaves_a_row_and_not_a_falling_count(
    mount: MountProfile,
) -> None:
    """Esc throws the inbox away, and the transcript has to say so.

    `AgentDriver.cancel` clears every pending message unless the caller asks it
    not to — the intended behavior, and until now an entirely silent one: the
    footer's "1 queued" fell to nothing and a queued prompt, or the output of a
    `!` waiting for the next step, was gone with no account of it anywhere on
    screen. The account is the point, not a veto on the clearing.

    Driven through the real `Inbox` rather than a hand-written payload, because
    what is being pinned is that the adapter reads the `outcome` key `_mutate`
    actually writes — the one thing that still tells a dropped message from a
    claimed one once both are only a `removedCount` on the log.

    Sabotage: drop the `outcome` test in the handler and a *claimed* batch —
    which removes messages too — starts reporting itself as a cancellation.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-inbox-drop")
    quiet = InboxNotifications(
        inserted=lambda _message: None,
        discarded=lambda _message: None,
        claimed=lambda _message, _turn: None,
    )
    inbox = Inbox(session, quiet)
    # The shape a `!` actually splices, so the row under test is the row a
    # person would lose: a relayed plugin message, not something they typed.
    relay = PluginSource(plugin="ph-app.shell", form="relay")
    inbox.append("next-step", create_user_message(content=[TextBlock(text="$ make")], source=relay))

    state = _replay(session)
    assert state.queued == 1, "waiting, and the footer says so"
    assert not [item for item in state.visible_items() if item.role == "notice"]

    inbox.clear()

    state = _replay(session)
    assert state.queued == 0
    (row,) = [item for item in state.visible_items() if item.role == "notice"]
    assert row.text == "1 pending message canceled"

    # A claim removes messages too, and is not a loss: the model got them.
    inbox.append("next-step", create_user_message(content=[TextBlock(text="second")], source=relay))
    assert inbox.claim("next-step", 1), "consumed, not dropped"
    assert len([item for item in _replay(session).visible_items() if item.role == "notice"]) == 1


async def test_a_violated_invariant_is_a_notice_in_the_conversation(mount: MountProfile) -> None:
    """The person reading the transcript is this record's reader (I6).

    `supervisor/unreachable` draws a row because its reader arrives *afterwards*,
    when nothing could connect. This one is the inversion: what drifted is a
    projection of the log, so the reader who needs telling is the one looking at
    the transcript right now — the thing they are reading may no longer be what
    the model was sent.

    The invariant ids and not the details: an id names which promise broke and
    fits on a row, while "derive_messages holds 0 message(s) where a fresh
    derivation gives 1" is a sentence about node counts that belongs in the log
    the row points at.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-violated")
    session.append(
        "supervisor/violated",
        {
            "violations": [
                {"invariant": "session-log", "detail": "derive_messages holds 0 where 1"},
                {"invariant": "tools-view", "detail": "differs from a rebuild"},
            ],
            "count": 2,
            "pid": 123,
        },
    )

    (row,) = [item for item in _replay(session).visible_items() if item.role == "notice"]
    assert "session-log" in row.text and "tools-view" in row.text, "which promises broke"
    assert "2 invariants" in row.text, "how many"
    assert "123" not in row.text, "the pid is for correlating logs, not for the transcript"


async def test_a_cleared_invariant_is_good_news_and_reads_like_it(mount: MountProfile) -> None:
    """The same event type carries both transitions, and they are opposite facts.

    `verify_invariants` records the clearing too — a transcript that says
    "violated" and then goes quiet leaves a reader unable to tell a repaired
    cache from a daemon that stopped looking. But it writes it as the *same*
    type with an empty list, so a renderer that reads only the type says the
    alarming thing about the reassuring event. This one did: "stopped holding 1
    invariant (an unnamed invariant)", from two `or` fallbacks that looked like
    defensive dead code and were the only branch firing on half the traffic.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-cleared")
    session.append("supervisor/violated", {"violations": [], "pid": 123})

    (row,) = [item for item in _replay(session).visible_items() if item.role == "notice"]
    assert "holds its invariants again" in row.text
    assert "stopped holding" not in row.text, "the clearing must not read as a violation"


async def test_a_plugins_replacement_is_not_called_a_compaction(mount: MountProfile) -> None:
    """A surface `replace` is a mechanism, not a cause (P4-02).

    `input-offload` is the first row to substitute on the surface for a reason
    other than compaction, and until this test the adapter keyed on the
    replacement alone: an offloaded paste rendered as "(history compacted)",
    telling the reader their conversation had been summarized when a blob had
    been relocated. The attribution the log already carries is the discriminator
    — and the shadowing, which *is* the mechanism, applies either way.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-offload")
    pasted = session.append(
        "user/message", user_payload("a two megabyte paste", "m1"), SurfaceIntent("append")
    )
    session.append(
        "user/message",
        plugin_payload(
            "Message content too large…",
            "m2",
            plugin="input-offload",
            form="notice",
            summary="2 MB offloaded",
        ),
        SurfaceIntent(SurfaceReplace(replaces=(0,)), (pasted.seq,)),
    )
    adapter = TuiEventAdapter()
    for event in session.events:
        adapter.apply(event, REPLAY)

    roles = [(item.role, item.shadowed) for item in adapter.state.visible_items()]
    assert ("context", False) in roles, "a plugin's notice is that plugin's row"
    assert "compaction" not in [role for role, _ in roles]
    assert ("user", True) in roles, "the paste is still there, dimmed — the mechanism holds"


async def test_usage_feeds_the_context_gauge(mount: MountProfile) -> None:
    """The gauge reads the provider's count from `assistant/message.usage`."""
    live_state, session = await _drive(mount)
    assert live_state.tokens > 0
    assert live_state.context_window == 8192
    assert live_state.pressure is not None
    assert _replay(session).tokens == live_state.tokens


async def test_a_refinement_says_what_changed(mount: MountProfile) -> None:
    """A refinement changes the model's own prompt, so it is a row, not a record.

    `/refine` is only one way here — the planner refines at turn end with no
    command to show for it — so a user who could not see this would have no way
    to know why the next turn behaves differently.
    """
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-harness")
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


async def test_a_corpus_is_a_row_only_when_it_changed(mount: MountProfile) -> None:
    """The ordinary load is already described in the system prompt; what is news
    is the corpus having changed under a conversation that was told about it."""
    ctx: Context = await mount()
    session = ctx.require(SESSIONS).create("tui-context")
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


def test_every_declared_child_status_has_a_glyph() -> None:
    """The TUI's status table is complete against the seam's vocabulary.

    A second consumer-side enumeration of `SubagentStatus`, correct today and
    checked by nothing — so a status added to the seam would render as a blank
    cell in the subagent panel and nobody would find out from a test. This is
    the same gate `KNOWN_SESSION_EVENT_TYPES` gets from both front ends, applied
    to the other vocabulary that crosses the log as `Any`.
    """
    from typing import get_args

    from ph.seams.subagents import SubagentStatus
    from ph_app.tui.state import STATUS_GLYPHS

    declared = set(get_args(SubagentStatus))
    assert set(STATUS_GLYPHS) == declared, (
        f"unglyphed: {declared - set(STATUS_GLYPHS)}; unknown: {set(STATUS_GLYPHS) - declared}"
    )


def test_a_sandbox_refusal_is_a_notice_with_the_way_out() -> None:
    """P6-38: the record carries the seam's sentence, and the transcript shows it
    as a notice — the command's own result already said it failed."""
    from ph.seams.sandbox import Denial

    session = Session("s")
    session.append(
        "sandbox/denied", Denial(kind="network", via="proxy", host="h", port=443).record("a")
    )

    (item,) = _replay(session).items
    assert item.role == "notice"
    assert (
        item.text == "Sandbox blocked network access to h:443. Allow it with /sandbox allow host h."
    )
