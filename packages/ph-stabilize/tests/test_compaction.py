"""P4-03 — `compaction-summarize` and `/compact` (G4, G10, A3).

The row's gates, one test each: *0.85 / 170k triggers; never splits a pair; log
intact; variables survive.*

The one to read is the split, and it is A3 stated as an assertion: after a
compaction the model reads a summary while the log still holds every message the
conversation had. That is the same mechanism `input-offload` uses (P4-02) and the
reason compaction here is a *reading* of an append-only log rather than an edit
to it — a test that checked only the model's side would pass equally for a design
that deleted the history it summarized.

The two thresholds are asserted as **pairs**, for the reason the offload row's
boundary is: a threshold tested only from the far side passes for any limit at or
below the value tested. Both numbers are derived from the module's constants, so
a policy change moves the gate with it rather than leaving it asserting the old
one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, break_spill

from ph.agent.types import AgentOptions
from ph.cancel import Cancelled
from ph.cordis import Context
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    LlmFailure,
    ToolCallBlock,
    ToolResultBlock,
    text_of,
)
from ph.seams.compaction import CompactionNote
from ph.session import Session, SurfaceIntent, derive_event_message, thaw_json
from ph.session.known_event_types import (
    IGNORABLE_SESSION_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
)
from ph.testing import FAKE_OPTIONS, assistant_payload, tool_result_payload, user_payload
from ph_stabilize.compaction import (
    KEEP_FRACTION,
    MAX_ARG_LENGTH,
    REPLACEMENT_WITHOUT_PATH,
    SUMMARY_MAX_TOKENS,
    TRIGGER_FRACTION,
    TRIGGER_TOKENS,
    TRUNCATION_TEXT,
    Config,
    SummarizeEngine,
    balanced_cuts,
    render_for_summary,
    safe_cutoff,
    truncated_arguments,
)

pytestmark = pytest.mark.anyio

SUMMARY = "## SESSION INTENT\n\nthe scripted summary"


# ------------------------------------------------------------------ helpers --


def _route(ctx: Context) -> None:
    """Answer a compaction call with `SUMMARY` and everything else with a reply.

    Keyed on `purpose`, which is the field that says a request is *about* the
    conversation rather than part of it — so the test distinguishes the two the
    same way the harness does.
    """
    adapter = ctx.llm_fake
    adapter.respond = lambda request: SUMMARY if request.purpose == "compaction" else "answer"


def _summary_requests(ctx: Context) -> list[GenerateOptions]:
    return [one for one in ctx.llm_fake.requests if one.purpose == "compaction"]


def _loop_requests(ctx: Context) -> list[GenerateOptions]:
    return [one for one in ctx.llm_fake.requests if one.is_loop_request]


def _instruction(request: GenerateOptions) -> str:
    """The trailing instruction — where the extraction prompt lives on a replay.

    Under the replay shape the summarize prompt cannot go in `system`: anything
    put there changes the prefix and forfeits the cache the shape exists for. So
    a test that asks "did the model see this" asks the last message, not the
    system prompt.
    """
    return text_of(request.messages[-1].content)


def _model_text(session: Session) -> str:
    """What the model would be sent — the derived surface, not the raw log."""
    return "\n".join(text_of(message.content) for message in session.derive_messages())


def _human_text(session: Session) -> str:
    return "\n".join(text_of(message.content) for message in session.transcript())


def _derived_text(session: Session) -> str:
    """Everything the model would be sent, tool results included.

    `text_of` reads `TextBlock`s only, so a tool result — whose text is nested
    inside a `ToolResultBlock` — is invisible to it. The clip tests are entirely
    about tool-result content, and asserting against `_model_text` there passed
    for the wrong reason: an empty string contains no locator either.
    """
    return render_for_summary(session.derive_messages(), trimmed=False)


def _events(session: Session, event_type: str) -> list[Any]:
    return [event for event in session.events if event.type == event_type]


QUESTION_CHARS = 1_200
"""Big enough that the *shipped* retention actually cuts.

The fake adapter reports an 8 192-token window, so keeping `KEEP_FRACTION` of it
is a budget of 819 tokens — about 3 300 characters. A conversation of four
one-line prompts is under that, and `/compact` on it correctly does nothing, so
a test built from one would have asserted against a no-op. Eight turns of this
size clear the budget several times over while staying far below the 0.85
pressure trigger, which is what keeps these tests about the *manual* path.
"""


async def _conversation(ctx: Context, session_id: str, turns: int = 8) -> Any:
    """A session with `turns` question/answer pairs, long enough to compact."""
    session = ctx.sessions.create(session_id)
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    for index in range(turns):
        await agent.prompt(f"question {index} " + "detail " * (QUESTION_CHARS // 7))
    return agent


async def _reaching_for_a_tool() -> Any:
    """A reply that is one tool call and no prose — what a model under the
    session's own system prompt, holding the session's own tools, may well do
    when asked to summarize."""
    yield BlockStart(index=0, block_type="tool-call")
    yield BlockEnd(
        index=0, block=ToolCallBlock(id="t1", name="read", arguments='{"path": "notes.md"}')
    )
    yield Finish(reason=FinishReason(kind="tool-calls"))


async def _overflowing() -> Any:
    """A stream that ends the way a provider ends one it cannot accept."""
    yield Finish(
        reason=FinishReason(
            kind="error",
            failure=LlmFailure(message="too long", code=CONTEXT_WINDOW_EXCEEDED, status=400),
        )
    )


# ------------------------------------------------------------- the triggers --


def test_the_fraction_trigger_fires_at_0_85_of_a_known_window() -> None:
    """G4's first number, as a pair: 0.85 of the window triggers, just under does not.

    Driven through the real baseline rather than a stubbed one — reported usage
    on the newest `assistant/message` is what `ctx.token_meter` reads, so this
    also pins that the trigger sees *provider* numbers once any exist (D15).
    """
    window = 1_000
    at_the_line = int(window * TRIGGER_FRACTION)

    def engine_for(total: int) -> tuple[SummarizeEngine, Session]:
        ctx = _bare_context()
        session = Session(f"pressure-{total}")
        session.append(
            "request/context",
            {"provider": "fake", "model": "f", "contextWindow": window},
        )
        session.append(
            "assistant/message",
            {**assistant_payload("hi", "m1"), "usage": {"inputTokens": total, "outputTokens": 0}},
            SurfaceIntent("append"),
        )
        return SummarizeEngine(ctx=ctx, config=Config()), session

    under, under_session = engine_for(at_the_line - 1)
    over, over_session = engine_for(at_the_line)
    assert under._under_pressure(under.ctx.token_meter.baseline(under_session)) is False
    assert over._under_pressure(over.ctx.token_meter.baseline(over_session)) is True


def test_without_a_window_the_trigger_is_the_fixed_token_count() -> None:
    """G4's second number, the branch upstream takes when the model profile has
    no `max_input_tokens`: 170 000 tokens, and 169 999 is not it.

    pH reaches this branch on the first step of every session — `request/context`
    is appended inside `_build_request`, which runs *after* `agent/pre-step` —
    so it is not a rare fallback, it is the opening state of every conversation.
    """

    def engine_for(total: int) -> tuple[SummarizeEngine, Session]:
        session = Session(f"unwindowed-{total}")
        session.append(
            "assistant/message",
            {**assistant_payload("hi", "m1"), "usage": {"inputTokens": total, "outputTokens": 0}},
            SurfaceIntent("append"),
        )
        return SummarizeEngine(ctx=_bare_context(), config=Config()), session

    under, under_session = engine_for(TRIGGER_TOKENS - 1)
    over, over_session = engine_for(TRIGGER_TOKENS)
    assert under._under_pressure(under.ctx.token_meter.baseline(under_session)) is False
    assert over._under_pressure(over.ctx.token_meter.baseline(over_session)) is True


def _bare_context() -> Context:
    """A context with just the token meter — what `_under_pressure` reads.

    Built by hand rather than mounted: the two threshold tests are about
    arithmetic over a log, and standing a whole profile up to ask them would make
    the failure of an unrelated row look like a threshold regression.
    """
    from ph.seams.token_meter import TokenMeter

    ctx = Context()
    ctx.provide("token_meter", TokenMeter(ctx=ctx))
    return ctx


# ---------------------------------------------------------- the safe cutoff --


def _paired_session() -> Session:
    """user · assistant(tool-call) · tool/result · user · assistant."""
    session = Session("paired")
    session.append("user/message", user_payload("do it", "m1"), SurfaceIntent("append"))
    session.append(
        "assistant/message",
        assistant_payload(
            "",
            "m2",
            content=[{"type": "tool-call", "id": "c1", "name": "read", "arguments": "{}"}],
        ),
        SurfaceIntent("append"),
    )
    session.append(
        "tool/result", tool_result_payload("the file", "m3", "c1"), SurfaceIntent("append")
    )
    session.append("user/message", user_payload("thanks", "m4"), SurfaceIntent("append"))
    session.append("assistant/message", assistant_payload("done", "m5"), SurfaceIntent("append"))
    return session


def test_a_cut_between_a_call_and_its_result_is_unbalanced() -> None:
    """The fold itself: five nodes, six cuts, one of them straddling the pair."""
    assert balanced_cuts(_paired_session()) == (True, True, False, True, True, True)


def test_the_cut_never_separates_a_call_from_its_result() -> None:
    """The row's gate. Asked for a cutoff that lands *between* the assistant's
    tool call and the result answering it; the answer moves back to before the
    call, so the pair travels into the summary together.

    Backward, not forward: advancing would summarize the call and leave the
    model holding a `tool-result` for a call it can no longer see, which several
    providers reject outright.
    """
    session = _paired_session()
    assert safe_cutoff(_projected(session), 2) == 1, "the cut moved back past the tool call"
    assert safe_cutoff(_projected(session), 3) == 3, "a balanced target is left where it is"


def test_a_conversation_with_no_balanced_cut_is_not_compacted() -> None:
    """An outstanding call across the whole surface: `0`, meaning no safe range.

    Not an error and not a partial compaction — dsh states the same contract:
    a single oversized retained unit cannot be repaired by replacing a surface.
    """
    session = Session("unbalanced")
    session.append(
        "assistant/message",
        assistant_payload(
            "",
            "m1",
            content=[{"type": "tool-call", "id": "c1", "name": "read", "arguments": "{}"}],
        ),
        SurfaceIntent("append"),
    )
    session.append("user/message", user_payload("hello", "m2"), SurfaceIntent("append"))
    assert safe_cutoff(_projected(session), 2) == 0


# ------------------------------------------------------- the split, which is A3 --


async def test_the_model_reads_the_summary_and_the_log_keeps_the_conversation(
    mount: Any,
) -> None:
    """A3 in one assertion pair, and the whole reason compaction is a surface op.

    The model's history is replaced; the person's is not; and neither is the log.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "split")
    session = agent.session
    before = len(session.events)

    result = await ctx.commands.dispatch("/compact", session=session, agent=agent)

    assert SUMMARY in _model_text(session)
    assert "question 0" not in _model_text(session), "the model was sent the history anyway"
    assert "question 0" in _human_text(session), "the transcript lost what was said"
    assert len(session.events) > before, "the log shrank"
    assert "compacted" in (result or "")


async def test_the_replaced_conversation_is_written_to_conversation_history(
    mount: Any,
) -> None:
    """A3's other half: the originals are on disk, at the path the model was given.

    A relocation, not a deletion — the same promise `ctx.spill_store` makes for
    an offloaded tool result, which is why the replacement names the file.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "history")
    session = agent.session

    await ctx.commands.dispatch("/compact", session=session, agent=agent)

    (record,) = _events(session, "compaction/summarized")
    locator = record.data["locator"]
    assert "conversation_history" in locator
    assert locator in _model_text(session), "the model was not told where the history went"
    assert "question 0" in Path(locator).read_text(encoding="utf-8")


async def test_the_summary_is_the_plugins_text_and_says_it_is_a_compaction(
    mount: Any,
) -> None:
    """Attribution, and the discriminator every reader keys on.

    `input-offload` also appends a plugin-authored replacement; only this one is
    a claim that conversation has left the derivation, so only this one says
    `form: compaction`. The TUI reads exactly this field to decide whether to
    call a row "compacted".
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "attribution")
    session = agent.session

    await ctx.commands.dispatch("/compact", session=session, agent=agent)

    replacement = next(
        event
        for event in session.events
        if event.type == "user/message" and event.surface_op != "append"
    )
    source = replacement.data["source"]
    assert source["kind"] == "plugin"
    assert source["plugin"] == "compaction-summarize"
    assert source["form"] == "compaction"


async def test_the_replacement_cites_every_node_it_shadows(mount: Any) -> None:
    """What makes the substitution reversible reading rather than deletion.

    `Session.append` refuses a replace that shadows a node it does not cite, so
    this is really a test that the *plan* and the *op* agree — the range is
    chosen by surface position and the citation is built from the same list.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "provenance")
    session = agent.session

    await ctx.commands.dispatch("/compact", session=session, agent=agent)

    (record,) = _events(session, "compaction/summarized")
    replacement = session.events[record.seq + 1]
    assert replacement.type == "user/message", "the record must sit beside its replacement"
    # `list(...)` on both sides: an appended payload is deep-frozen, so the seqs
    # come back as a tuple and an `==` against the citation would compare shapes.
    assert list(replacement.source_event_seqs or ()) == list(record.data["shadowedSeqs"])


# --------------------------------------------------- argument truncation --
# §7.4 item 2. The cheap half of the row: bytes the model itself sent, elided
# from what it is shown, with the log keeping its exact words. No model call.


WINDOW = 1_000
"""A small window in a hand-built session, so the *shipped* fractions bite.

The truncation and clip passes are pure functions of a session, and building
one directly is how a test can put the surface in the exact shape each pass
exists for — an over-budget trailing tool batch, a long-argument call in old
history — which a scripted conversation reaches only by accident.
"""


def _windowed(session_id: str) -> Session:
    """A session whose baseline is already over 0.85 of a known window."""
    session = Session(session_id)
    session.append("request/context", {"provider": "fake", "model": "f", "contextWindow": WINDOW})
    return session


def _write_call(call_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "tool-call",
        "id": call_id,
        "name": "write",
        "arguments": json.dumps({"path": "notes.md", "content": content}),
    }


def _pressured(session: Session, used: int = 900) -> None:
    """Append the assistant message whose reported usage trips the trigger."""
    session.append(
        "assistant/message",
        {
            **assistant_payload("done", f"u{session.seq}"),
            "usage": {"inputTokens": used, "outputTokens": 0},
        },
        SurfaceIntent("append"),
    )


def _long_write_session(session_id: str, body: str) -> Session:
    """user · assistant(write with `body`) · tool/result · assistant(usage)."""
    session = _windowed(session_id)
    session.append("user/message", user_payload("save it", "m1"), SurfaceIntent("append"))
    session.append(
        "assistant/message",
        assistant_payload("", "m2", content=[_write_call("c1", body)]),
        SurfaceIntent("append"),
    )
    session.append(
        "tool/result", tool_result_payload("written", "m3", "c1"), SurfaceIntent("append")
    )
    _pressured(session)
    return session


def _engine(ctx: Context) -> SummarizeEngine:
    engine: SummarizeEngine = ctx.compaction.engine
    return engine


def _truncate(ctx: Context, session: Session, agent: Any = None) -> tuple[int, ...]:
    """The truncation pass with the baseline its caller now computes once.

    `agent` may be `None`: the tool lookup falls back to the root scope, which
    is where the fs tools register, so a hand-built session still asks the real
    registry whether a tool declares its arguments disposable.
    """
    return _engine(ctx).truncate_arguments(
        agent, session, "pressure", ctx.token_meter.baseline(session)
    )


def _projected(session: Session) -> list[Any]:
    """The surface as messages — what `safe_cutoff` now takes."""
    events = session.events
    return [derive_event_message(events[seq]) for seq in session.surface.nodes]


def _arguments_of(session: Session, name: str = "write") -> list[str]:
    """The call arguments the *model* now sees, from the derived surface."""
    return [
        block.arguments
        for message in session.derive_messages()
        for block in message.content
        if isinstance(block, ToolCallBlock) and block.name == name
    ]


async def test_only_a_tool_that_declares_it_has_arguments_elided(mount: Any) -> None:
    """`write` declares `arguments_disposable`; the registry is what says so.

    A deployment that renames the fs tools, or an MCP server that adds its own,
    keeps working — which a hardcoded name list in this package could not.
    """
    ctx = await mount(profile=PROFILE)
    assert ctx.tools.get("write").arguments_disposable
    assert not ctx.tools.get("read").arguments_disposable, "reading is not a payload"


async def test_a_long_call_argument_is_elided_from_what_the_model_sees(mount: Any) -> None:
    """The pass, and the split it preserves.

    The model stops being shown a file body it wrote and the log keeps the bytes
    it actually sent — the same log-and-surface split compaction uses, applied
    without spending a model call.
    """
    ctx = await mount(profile=PROFILE)
    body = "x" * (MAX_ARG_LENGTH + 1)
    session = _long_write_session("elided", body)

    rewritten = _truncate(ctx, session)

    assert rewritten, "nothing was truncated"
    (arguments,) = _arguments_of(session)
    assert TRUNCATION_TEXT in arguments
    assert body not in arguments, "the model is still being shown the body"
    assert body in json.dumps(thaw_json(session.events[rewritten[0]].data)), "the log lost it"


async def test_an_argument_at_the_limit_is_left_alone(mount: Any) -> None:
    """The boundary as a pair, for the reason every threshold here is: one
    tested only from the far side passes for any limit at or below it."""
    ctx = await mount(profile=PROFILE)

    at_limit = _long_write_session("at-limit", "x" * MAX_ARG_LENGTH)
    over = _long_write_session("over-limit", "x" * (MAX_ARG_LENGTH + 1))

    assert _truncate(ctx, at_limit) == ()
    assert _truncate(ctx, over) != ()


async def test_the_elision_keeps_the_call_id_so_the_pair_still_balances(
    mount: Any,
) -> None:
    """The rewrite must not change what the call/result pairing sees.

    A replacement that dropped or renamed the call would leave its result
    orphaned, which is the same failure `safe_cutoff` exists to prevent — and it
    would show up only later, as a compaction that could no longer find a cut.
    """
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("pairing", "x" * (MAX_ARG_LENGTH + 1))
    before = balanced_cuts(session)

    _truncate(ctx, session)

    assert balanced_cuts(session) == before
    assert json.loads(_arguments_of(session)[0])["path"] == "notes.md", "a short arg was touched"


async def test_the_replacement_carries_no_usage(mount: Any) -> None:
    """The hazard that makes this rewrite different from the others.

    `TokenMeter.last_usage` scans *backward* for the newest `assistant/message`
    with a usage block, and a replacement is appended at the end of the log. One
    that copied its original's usage would become the meter's baseline and tell
    the pressure trigger the session had shrunk — and the TUI footer, which
    reads the last usage it sees, would show the same stale number.
    """
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("usage", "x" * (MAX_ARG_LENGTH + 1))
    _pressured(session, used=950)
    baseline_before = ctx.token_meter.baseline(session).tokens

    _truncate(ctx, session)

    replacement = session.events[-2]
    assert replacement.type == "assistant/message"
    assert "usage" not in replacement.data
    assert ctx.token_meter.baseline(session).tokens == baseline_before


async def test_a_recent_message_is_not_truncated(mount: Any) -> None:
    """Retention, which is the whole reason this is not just "shorten everything":
    the model is still working with the tail, and an argument it is about to act
    on is exactly the one it must be able to read."""
    ctx = await mount(profile=PROFILE)
    session = _windowed("recent")
    _pressured(session)
    session.append(
        "assistant/message",
        assistant_payload("", "m2", content=[_write_call("c1", "x" * (MAX_ARG_LENGTH + 1))]),
        SurfaceIntent("append"),
    )

    assert _truncate(ctx, session) == ()


async def test_truncating_twice_changes_nothing_the_second_time(mount: Any) -> None:
    """Idempotence falls out of the length check — an elided argument is 40-odd
    characters — but it is asserted, because a pass that ran every step and
    appended a replacement every step would grow the log without bound."""
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("twice", "x" * (MAX_ARG_LENGTH + 1))

    assert _truncate(ctx, session) != ()
    length = len(session.events)
    assert _truncate(ctx, session) == ()
    assert len(session.events) == length


async def test_a_tool_that_does_not_declare_it_keeps_its_arguments(mount: Any) -> None:
    """The restriction is real, and it is asked of the *tool*.

    Upstream matches `{"write_file", "edit_file"}` — names pH does not register —
    so this asks `ToolDefinition.arguments_disposable` instead, the way the
    offload row asks `self_limits`. A long `run_code` cell in old history is the
    argument a model most often re-reads, and it declares nothing, so it stays.
    """
    ctx = await mount(profile=PROFILE)
    session = _windowed("other-tool")
    session.append("user/message", user_payload("run it", "m1"), SurfaceIntent("append"))
    cell = {
        "type": "tool-call",
        "id": "c1",
        "name": "run_code",
        "arguments": json.dumps({"program": "x" * (MAX_ARG_LENGTH + 1)}),
    }
    session.append(
        "assistant/message", assistant_payload("", "m2", content=[cell]), SurfaceIntent("append")
    )
    session.append("tool/result", tool_result_payload("ok", "m3", "c1"), SurfaceIntent("append"))
    _pressured(session)

    assert _truncate(ctx, session) == ()


def test_arguments_that_are_not_a_json_object_are_left_alone() -> None:
    """A malformed argument string is not something to rewrite behind the model's
    back — and re-serializing a non-object would change its shape, not just its
    size."""
    assert truncated_arguments("not json at all", 10) is None
    assert truncated_arguments(json.dumps(["a" * 100]), 10) is None
    assert truncated_arguments(json.dumps({"short": "ok"}), 10) is None


async def test_truncation_runs_inside_the_automatic_path(mount: Any) -> None:
    """The wiring: the free remedy is tried before the expensive one.

    Asked through `compact_if_needed`, which is what `agent/pre-step` calls — so
    this fails if the pass is correct but nothing invokes it.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = _long_write_session("automatic", "x" * (MAX_ARG_LENGTH + 1))
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await _engine(ctx).compact_if_needed(agent, "pressure")

    assert _events(session, "compaction/args-truncated"), "the cheap pass never ran"


# ------------------------------------------------------- the overflow clip --
# §7.4 item 7. The case summarization is least able to help with: a trailing
# tool-result batch larger than the retention budget leaves no balanced cut, so
# `safe_cutoff` correctly declines and nothing else would ever shrink it.


def _tool_batch_session(session_id: str, *sizes: int, opening: bool = True) -> Session:
    """A session ending in a batch of tool results of the given sizes.

    `opening=False` drops the leading user message, which is what makes the
    surface genuinely uncuttable: with it, the first node is a balanced cut all
    by itself and summarization has somewhere to go.
    """
    session = _windowed(session_id)
    if opening:
        session.append("user/message", user_payload("read them", "m1"), SurfaceIntent("append"))
    calls = [
        {"type": "tool-call", "id": f"c{index}", "name": "read", "arguments": "{}"}
        for index, _ in enumerate(sizes)
    ]
    session.append(
        "assistant/message", assistant_payload("", "m2", content=calls), SurfaceIntent("append")
    )
    for index, size in enumerate(sizes):
        session.append(
            "tool/result",
            tool_result_payload(f"{index}" * size, f"r{index}", f"c{index}"),
            SurfaceIntent("append"),
        )
    return session


async def test_an_oversized_trailing_batch_is_clipped_and_recoverable(mount: Any) -> None:
    """Each result is relocated, not dropped — the promise `ctx.spill_store`
    makes everywhere in this bundle."""
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("clip", 30_000, 30_000)
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    clipped = await _engine(ctx).clip_overflow_tail(agent)

    assert len(clipped) == 2
    spilled = _events(session, "offload/spilled")
    assert len(spilled) == 2
    # Keyed by call id, because the two results differ: an assertion that
    # compared both files to one expected body would pass for a clip that
    # spilled the same result twice.
    originals = {"c0": "0" * 30_000, "c1": "1" * 30_000}
    for record in spilled:
        locator = record.data["locator"]
        assert locator in _derived_text(session)
        assert Path(locator).read_text(encoding="utf-8") == originals[record.data["callId"]]
    for body in originals.values():
        assert body not in _derived_text(session)


async def test_the_clip_changes_only_the_result_content(mount: Any) -> None:
    """`Session.append` refuses a `tool/result` replacement that touches anything
    but content, so this is really a test that the payload is rebuilt from the
    original rather than synthesized — a fresh message id would be rejected, and
    a rejected append would take the whole overflow path down with it.
    """
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("content-only", 30_000)
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await _engine(ctx).clip_overflow_tail(agent)

    replacement = session.derive_messages()[-1]
    assert replacement.id == "r0", "the message identity moved"
    block = replacement.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_call_id == "c0"


async def test_a_small_result_is_not_replaced_by_a_larger_pointer(mount: Any) -> None:
    """The batch is what must shrink, and a member that would grow is not part
    of shrinking it. Upstream clips every message in an over-budget batch; a
    hundred-character result swapped for a nine-hundred-character pointer is a
    clip that made the request bigger."""
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("mixed", 30_000, 40)
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    clipped = await _engine(ctx).clip_overflow_tail(agent)

    assert len(clipped) == 1
    assert "1" * 40 in _derived_text(session), "the small result was replaced anyway"


async def test_a_conversation_that_does_not_end_in_tool_results_is_untouched(
    mount: Any,
) -> None:
    """The cheap check that keeps this pass free in the ordinary case."""
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("no-tail", "x" * 100)
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    assert await _engine(ctx).clip_overflow_tail(agent) == ()


async def test_the_clip_alone_is_enough_to_retry_the_request(mount: Any) -> None:
    """Why the clip is worth having beside summarization rather than inside it.

    This session has one enormous outstanding call/result pair and nothing else,
    so *every* balanced cut is the start of the conversation and summarization
    correctly declines. Before the clip the overflow path had no remedy at all
    and the turn ended in the provider's refusal; now the tail shrinks and the
    loop is asked to try again.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = _tool_batch_session("only-remedy", 30_000, opening=False)
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    engine = _engine(ctx)

    assert engine._plan(session) is None, "the fixture is not the no-safe-cut shape"
    assert await engine.clip_overflow_tail(agent) != ()


# --------------------------------------------------------------- pressure --


async def test_pressure_compacts_without_anyone_asking(mount: Any) -> None:
    """The automatic half of G4, end to end and at the shipped threshold.

    Nothing here calls the engine: a long enough conversation crosses 0.85 of
    the fake adapter's 8 192-token window on its own, and `agent/pre-step` is
    where that is noticed. The two threshold tests above prove the arithmetic;
    this one proves the arithmetic is wired to something.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = ctx.sessions.create("pressure")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    for index in range(10):
        await agent.prompt(f"question {index} " + "detail " * 600)

    (record,) = _events(session, "compaction/summarized")
    assert record.data["trigger"] == "pressure"
    assert SUMMARY in _model_text(session)
    # And the compaction relieved what triggered it, rather than firing on every
    # step from here on: one record, and the baseline is back under the line.
    assert ctx.token_meter.baseline(session).pressure < TRIGGER_FRACTION


async def test_the_automatic_triggers_can_be_turned_off_without_losing_the_command(
    mount: Any,
) -> None:
    """dsh's `auto` knob, and why it is separate from the row itself.

    A deployment can want a human to decide when history is replaced while still
    wanting the human to be *able* to — so `auto: false` disarms the two policy
    hooks and leaves `/compact` working.
    """
    ctx = await mount({"id": "compaction-summarize", "config": {"auto": False}}, profile=PROFILE)
    _route(ctx)
    session = ctx.sessions.create("manual-only")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    for index in range(10):
        await agent.prompt(f"question {index} " + "detail " * 600)
    assert not _events(session, "compaction/summarized"), "the automatic trigger stayed armed"

    await ctx.commands.dispatch("/compact", session=session, agent=agent)
    assert _events(session, "compaction/summarized"), "the command was disarmed too"


# ----------------------------------------------------------- the envelope --
# §7.4 item 5. The summarize call reuses the conversation's own system prompt
# and tools so the request is a strict *prefix* of the one the loop just made —
# which is what makes it nearly free on a provider that caches prefixes, and
# what stops the summarizer reading a tail of the range it is replacing.


async def test_the_summarize_request_is_a_prefix_of_the_conversations(
    mount: Any,
) -> None:
    """The cache property, asserted structurally rather than hoped for.

    A cache miss looks exactly like a hit, only on the invoice — the same reason
    `test_prefix_stability` exists — so the shape is checked instead: identical
    system, identical tools, and a message list that the conversation's own
    request begins with.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "prefix")

    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    conversation = _loop_requests(ctx)[-1]
    (summary,) = _summary_requests(ctx)
    assert summary.system == conversation.system, "the system prompt moved; the prefix is lost"
    assert summary.tools == conversation.tools, "the tool schemas moved; the prefix is lost"
    assert summary.tools, "the fixture has no tools, so the assertion above is vacuous"
    shared = len(summary.messages) - 1
    assert summary.messages[:shared] == conversation.messages[:shared]


async def test_the_summarizer_is_shown_the_whole_range(mount: Any) -> None:
    """No tail, no rendering: the shadowed messages are sent as themselves.

    Two things follow. The model reads real `tool-call`/`tool-result` blocks
    rather than this module's text rendering of them, and nothing in the range
    is silently withheld — a summary written from a tail cannot mention what it
    was never shown, and could not say so either.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "whole-range")

    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    (record,) = _events(agent.session, "compaction/summarized")
    (summary,) = _summary_requests(ctx)
    assert record.data["shape"] == "replay"
    # Every shadowed node, plus the one instruction appended after it.
    assert len(summary.messages) == len(record.data["shadowedSeqs"]) + 1
    assert "question 0" in text_of(summary.messages[0].content)


async def test_the_replay_does_not_inherit_a_reasoning_budget(mount: Any) -> None:
    """One of two deliberate departures from byte-identical.

    A reasoning model would otherwise spend a thinking budget on an extraction.
    Neither this nor the `max_tokens` override is part of the token prefix, so
    neither costs the cache hit — which is exactly why they are the two things
    worth departing on.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = ctx.sessions.create("thinking")
    options = AgentOptions(provider="fake", model="fake-1", reasoning_effort="high")
    agent = ctx.agents.create(session, options)
    for index in range(8):
        await agent.prompt(f"question {index} " + "detail " * (QUESTION_CHARS // 7))

    await ctx.commands.dispatch("/compact", session=session, agent=agent)

    (summary,) = _summary_requests(ctx)
    assert _loop_requests(ctx)[-1].reasoning_effort == "high", "the fixture proves nothing"
    assert summary.reasoning_effort is None
    assert summary.max_tokens == SUMMARY_MAX_TOKENS


async def test_a_reply_of_only_tool_calls_falls_back_without_them(mount: Any) -> None:
    """The cost of carrying the conversation's tools, and the answer to it.

    The replay hands the model the session's tool schemas under the session's own
    system prompt, so a model primed to *act* can answer with a tool call and no
    prose. That is not a failure to report — it is a request shape that did not
    suit, so the row asks again without tools and with the whole range. Cheap
    first, correct second; nothing is hidden by falling back, only the cache is
    given up.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "tool-happy")
    answered = {"once": False}

    async def call_a_tool_first(options: GenerateOptions, next_: Any) -> Any:
        if options.purpose == "compaction" and not answered["once"]:
            answered["once"] = True
            ctx.llm_fake.requests.append(options)
            return _reaching_for_a_tool()
        return await next_(options)

    ctx.on("llm/stream", call_a_tool_first)
    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    first, second = _summary_requests(ctx)
    assert first.tools, "the first attempt did not carry the conversation's tools"
    assert not second.tools, "the retry carried the tools that caused the problem"
    assert len(second.messages) == 1, "the retry did not fall back to the self-contained shape"
    (record,) = _events(agent.session, "compaction/summarized")
    assert record.data["shape"] == "direct-after-replay"
    assert SUMMARY in _model_text(agent.session), "the fallback did not produce a summary"


async def test_a_session_with_no_logged_request_still_compacts(mount: Any) -> None:
    """There is no envelope to replay before the first request is built, and a
    hand-built session may never have one — so the self-contained shape is the
    floor rather than a failure path."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = _tool_batch_session("headerless", 200, 200)
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    assert session.request_header() is None
    result = await _engine(ctx).compact_now(agent)

    assert result is not None
    (record,) = _events(session, "compaction/summarized")
    assert record.data["shape"] == "direct"


# -------------------------------------------------------------- the overflow --


async def test_a_context_overflow_compacts_and_retries_the_request(mount: Any) -> None:
    """The trigger `llm-retry` deliberately leaves alone (G4).

    A provider that says `CONTEXT_WINDOW_EXCEEDED` has already answered the
    question a pressure estimate can only guess at, and it is the one failure
    with a remedy — so this row compacts and asks the loop to try again, and the
    turn completes on the second attempt rather than ending in the error.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "overflow")
    session = agent.session
    refused = {"once": False}

    async def refuse_the_first_request(options: GenerateOptions, next_: Any) -> Any:
        # On `llm/stream`, which is the seam retry and replay already attach to,
        # rather than by swapping the adapter's method — `FakeAdapter` has slots,
        # and the waterfall is where a provider failure is *supposed* to come
        # from, so the request travels the same path a real refusal would.
        if options.is_loop_request and not refused["once"]:
            refused["once"] = True
            return _overflowing()
        return await next_(options)

    ctx.on("llm/stream", refuse_the_first_request)
    await agent.prompt("one more question")

    (record,) = _events(session, "compaction/summarized")
    # Never a replay: the envelope being replayed is the request the provider
    # just refused for being too large, and re-sending it with an instruction
    # appended refuses again.
    assert record.data["shape"] == "direct"
    assert SUMMARY in _model_text(session)
    assert text_of(session.derive_messages()[-1].content) == "answer", "the retry did not land"


# --------------------------------------------------------------- declining --


async def test_a_summarizer_that_fails_leaves_the_conversation_untouched(
    mount: Any,
) -> None:
    """A compaction that cannot summarize must change nothing.

    The failure mode this guards is the worst one available: shadowing a range
    with an empty or missing summary would delete the conversation from the
    model's view and put nothing in its place.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "broken-summarizer")
    session = agent.session
    before = _model_text(session)
    ctx.llm_fake.respond = lambda request: "" if request.purpose == "compaction" else "answer"

    reply = await ctx.commands.dispatch("/compact", session=session, agent=agent)

    assert not _events(session, "compaction/summarized")
    assert _model_text(session) == before
    assert "unchanged" in (reply or "")


async def test_an_automatic_decline_is_recorded_and_not_retried_at_once(
    mount: Any,
) -> None:
    """Automatic compaction never raises — it records why and stands aside.

    And the decline holds until the surface moves: without that, the overflow
    path would re-attempt inside the same step loop, spending one model call per
    attempt to reach the same answer.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "declined")
    session = agent.session
    engine: SummarizeEngine = ctx.compaction.engine
    ctx.llm_fake.respond = lambda request: "" if request.purpose == "compaction" else "answer"

    assert await engine.compact_if_needed(agent, "overflow") is None
    (declined,) = _events(session, "compaction/declined")
    assert declined.data["code"] == "summary"

    calls = len(_summary_requests(ctx))
    assert await engine.compact_if_needed(agent, "overflow") is None
    assert len(_summary_requests(ctx)) == calls, "the decline was retried with nothing changed"


async def test_a_manual_failure_is_recorded_in_the_log(mount: Any) -> None:
    """The command tells the person "the attempt is recorded in the session log",
    and until this test that sentence was false.

    Only the automatic path recorded a decline; `/compact` raised through to the
    command, which turned it into a message and left no compaction record at all
    — the log showed a `command/done` marked `ok` whose detail happened to
    describe a failure. dsh's own wording is true because its `compaction/end`
    carries the error; pH's is true because of this.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "manual-failure")
    ctx.llm_fake.respond = lambda request: "" if request.purpose == "compaction" else "answer"

    reply = await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    assert "recorded in the session log" in (reply or ""), "the claim under test moved"
    (declined,) = _events(agent.session, "compaction/declined")
    assert declined.data["trigger"] == "manual"
    assert declined.data["code"] == "summary"


async def test_a_refusal_before_any_attempt_is_not_recorded(mount: Any) -> None:
    """`busy` is not a failed compaction — nothing was attempted, so there is no
    attempt to account for. Recording one would put an event in the log for
    every mistyped command."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "busy-unrecorded")
    agent._set_phase("running")

    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    assert not _events(agent.session, "compaction/declined")


async def test_an_unexpected_failure_neither_escapes_nor_goes_unrecorded(
    mount: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug in this row must not end the person's turn.

    `agent/pre-step` propagates to `_turn`, where the driver contains the
    failure with one debug line — so an exception here is a stabilization row
    silently killing a conversation. The guard is `Exception`, not
    `CompactionError`, and it records what it swallowed.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "buggy")
    engine = _engine(ctx)

    async def explode(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("the disk went away")

    monkeypatch.setattr(type(engine), "_land", explode)

    assert await engine.compact_if_needed(agent, "overflow") is None
    (declined,) = _events(agent.session, "compaction/declined")
    assert declined.data["code"] == "error"
    assert "the disk went away" in declined.data["reason"]


async def test_cancellation_is_not_swallowed(mount: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The one thing the guard must let through.

    `Cancelled` derives from `Exception`, so a guard written to contain bugs
    would also contain a person pressing stop — and the turn would carry on as
    though nothing had been asked of it.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "cancelled")
    engine = _engine(ctx)

    async def stop(*_args: Any, **_kwargs: Any) -> Any:
        raise Cancelled("the user pressed stop")

    monkeypatch.setattr(type(engine), "_land", stop)

    with pytest.raises(Cancelled):
        await engine.compact_if_needed(agent, "overflow")


async def test_a_short_conversation_reports_nothing_to_compact(mount: Any) -> None:
    """`None`, not an error: "your conversation is short" and "your summarizer
    is broken" are different outcomes and the seam keeps them apart."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = ctx.sessions.create("short")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    reply = await ctx.commands.dispatch("/compact", session=session, agent=agent)

    assert reply == "no compactable history yet"
    assert not _summary_requests(ctx), "a summarize call was spent on an empty session"


# --------------------------------------------------------------- fail open --


async def test_a_history_that_cannot_be_written_still_compacts_and_claims_no_path(
    mount: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spill that failed must not leave the model reading a path that holds nothing.

    Upstream keeps both wordings for exactly this, and the without-path one is
    the load-bearing half: the summary is then all there is, and telling the
    model otherwise would be the harness inventing a file.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "no-disk")
    session = agent.session
    break_spill(monkeypatch)

    await ctx.commands.dispatch("/compact", session=session, agent=agent)

    (record,) = _events(session, "compaction/summarized")
    assert record.data["locator"] is None
    assert REPLACEMENT_WITHOUT_PATH.format(summary=SUMMARY) in _model_text(session)
    assert "conversation_history" not in _model_text(session)


# ------------------------------------------------------------- the command --


async def test_compact_refuses_while_the_agent_is_working(mount: Any) -> None:
    """A compaction landing mid-turn would move the surface underneath a request
    the loop had already derived."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "busy")

    agent._set_phase("running")
    reply = await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    assert "idle session" in (reply or "")
    assert not _events(agent.session, "compaction/summarized")


async def test_compact_carries_what_the_user_is_about_to_work_on(mount: Any) -> None:
    """`/compact [instructions]`, and the reason it takes any.

    dsh refuses arguments here and deepagents' compact tool takes none. But the
    moment someone compacts on purpose is usually the moment they are changing
    subject, and what they are about to do is the one thing the summarizer
    cannot read off the conversation. It reaches the prompt labelled as the
    person's — before `<messages>`, so it is read as an instruction rather than
    as one more message to compress — and lands on the event, because a summary
    weighted towards one thing should say who asked for that.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "focus")

    await ctx.commands.dispatch(
        "/compact next I am rewriting the retry policy", session=agent.session, agent=agent
    )

    (request,) = _summary_requests(ctx)
    assert "rewriting the retry policy" in _instruction(request)
    (record,) = _events(agent.session, "compaction/summarized")
    assert record.data["instructions"] == "next I am rewriting the retry policy"


async def test_a_bare_compact_asks_the_summarizer_to_focus_on_nothing(mount: Any) -> None:
    """No argument means no focus block — an empty section would be a heading
    the model has to decide what to do with."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "no-focus")

    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    (request,) = _summary_requests(ctx)
    assert "what_the_user_asked_you_to_focus_on" not in _instruction(request)
    (record,) = _events(agent.session, "compaction/summarized")
    assert record.data["instructions"] is None


async def test_compact_without_an_engine_says_so(mount: Any) -> None:
    """The command over a profile that layered the seam and no backend.

    Silence would read as "nothing to compact", which is a different and false
    statement — the automatic path is the one that is right to say nothing.
    """
    ctx = await mount({"id": "command-compact", "name": "command-compact"})
    session = ctx.sessions.create("engineless")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    reply = await ctx.commands.dispatch("/compact", session=session, agent=agent)

    assert "no compaction engine" in (reply or "")


# ----------------------------------------------------------------- the notes --


async def test_a_note_tells_the_summarizer_what_survives_the_cut(mount: Any) -> None:
    """G10's mechanism, without G10's producer.

    A summary replaces *conversation*; a note is how a plugin says "this other
    thing is still here". It rides the trailing instruction, after the range
    being summarized, so the model reads it as part of what it was asked to do
    rather than as one more piece of the history it is compressing.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    ctx.compaction.note(CompactionNote(name="test:state", text=lambda _s: "`df` is still loaded"))
    agent = await _conversation(ctx, "notes")

    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    (request,) = _summary_requests(ctx)
    assert "`df` is still loaded" in _instruction(request)


async def test_a_note_that_renders_nothing_costs_no_prompt(mount: Any) -> None:
    """Empty means absent, the rule `PromptContext` already uses — so a note
    about a namespace holding nothing does not spend a paragraph saying so."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    ctx.compaction.note(CompactionNote(name="test:empty", text=lambda _s: ""))
    agent = await _conversation(ctx, "empty-note")

    await ctx.commands.dispatch("/compact", session=agent.session, agent=agent)

    (request,) = _summary_requests(ctx)
    assert "state_that_survives_this_summary" not in _instruction(request)


# ------------------------------------------------------------ the vocabulary --


def test_the_event_types_are_in_the_vocabulary() -> None:
    """The proof a producer outside ph-core owes through its own bundle."""
    for event_type in (
        "compaction/summarized",
        "compaction/declined",
        "compaction/args-truncated",
    ):
        assert event_type in KNOWN_SESSION_EVENT_TYPES
        assert event_type in IGNORABLE_SESSION_EVENT_TYPES


def test_the_retention_fraction_is_smaller_than_the_trigger() -> None:
    """dsh validates this at plugin load and refuses otherwise, for a reason
    worth restating: retaining more than the threshold means every compaction
    leaves the session still over it, and the next step compacts again."""
    assert KEEP_FRACTION < TRIGGER_FRACTION
