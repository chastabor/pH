"""P4-03: `compaction-summarize` and `/compact` (G4, G10, A3).

Gates of the row, one test each:

* The triggers are 0.85 and 170k.
* A compaction never splits a pair.
* The log is intact.
* Variables survive.

The test to read first is the split. It is A3 stated as an assertion: after a
compaction the model reads a summary, and the log still holds every message that
the conversation had. `input-offload` uses the same mechanism (P4-02). It is the
reason why compaction here is a *reading* of an append-only log and not an edit
to it. A test that checked only the side of the model would pass equally for a
design that deleted the history that it summarized.

The two thresholds are asserted as **pairs**, for the same reason as the
boundary of the offload row: a threshold that is tested only from the far side
passes for any limit at or below the tested value. Both numbers are derived from
the constants of the module. A policy change therefore moves the gate with it,
and does not leave the gate asserting the old value.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, NoReturn

import pytest
from stabilize_helpers import PROFILE, break_spill

from ph.agent.types import AgentOptions
from ph.agent_loop.driver import ReactLoopAgent
from ph.cancel import Canceled
from ph.cordis import DEPLOYMENT, Context
from ph.json import as_int, as_obj, as_seq, thaw_json
from ph.keys import AGENTS, COMMANDS, COMPACTION, LLM_FAKE, SESSIONS, TOKEN_METER, TOOLS
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
from ph.seams.compaction import CompactionError, CompactionNote
from ph.session import Session, SurfaceError, SurfaceIntent, balanced_cuts
from ph.session.events import SurfaceReplace
from ph.session.known_event_types import IGNORABLE_SESSION_EVENT_TYPES
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    StubAgent,
    assistant_payload,
    log_event,
    models_overlay,
    not_none,
    plugin_payload,
    simple_tool,
    streamed,
    tool_call_chunks,
    tool_result_payload,
    user_payload,
)
from ph_stabilize import compaction
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
    render_for_summary,
    truncated_arguments,
)

pytestmark = pytest.mark.anyio

SUMMARY = "## SESSION INTENT\n\nthe scripted summary"


# ------------------------------------------------------------------ helpers --


def _route(ctx: Context) -> None:
    """Answer a compaction call with `SUMMARY` and everything else with a reply.

    The route is keyed on `purpose`, the field that says that a request is
    *about* the conversation and not part of it. The test therefore
    distinguishes the two in the same way that the harness does.
    """
    adapter = ctx.require(LLM_FAKE)
    adapter.respond = lambda request: SUMMARY if request.purpose == "compaction" else "answer"


def _summary_requests(ctx: Context) -> list[GenerateOptions]:
    return [one for one in ctx.require(LLM_FAKE).requests if one.purpose == "compaction"]


def _loop_requests(ctx: Context) -> list[GenerateOptions]:
    return [one for one in ctx.require(LLM_FAKE).requests if one.is_loop_request]


def _instruction(request: GenerateOptions) -> str:
    """The trailing instruction, which holds the extraction prompt on a replay.

    Under the replay shape the summarize prompt cannot go in `system`: anything
    put there changes the prefix and loses the cache that the shape exists for.
    A test that asks "did the model see this" therefore reads the last message,
    not the system prompt.
    """
    return text_of(request.messages[-1].content)


def _model_text(session: Session) -> str:
    """What the model would be sent: the derived surface, not the raw log."""
    return "\n".join(text_of(message.content) for message in session.derive_messages())


def _human_text(session: Session) -> str:
    return "\n".join(text_of(message.content) for message in session.transcript())


def _derived_text(session: Session) -> str:
    """Everything that the model would be sent, tool results included.

    `text_of` reads only `TextBlock`s, so it cannot see a tool result, whose
    text is nested inside a `ToolResultBlock`. The clip tests are entirely about
    tool-result content. An assertion against `_model_text` there passed for the
    wrong reason: an empty string contains no locator either.
    """
    return render_for_summary(session.derive_messages(), trimmed=False)


def _events(session: Session, event_type: str) -> list[Any]:
    return [event for event in session.events if event.type == event_type]


QUESTION_CHARS = 1_200
"""Big enough that the *shipped* retention actually cuts.

The fake adapter reports an 8 192-token window, so keeping `KEEP_FRACTION` of it
is a budget of 819 tokens, about 3 300 characters. A conversation of four
one-line prompts is under that budget, and `/compact` on it correctly does
nothing. A test built from such a conversation would have asserted against a
no-op. Eight turns of this size exceed the budget several times and stay far
below the 0.85 pressure trigger, which keeps these tests about the *manual*
path.
"""


async def _conversation(ctx: Context, session_id: str, turns: int = 8) -> ReactLoopAgent:
    """A session with `turns` question/answer pairs, long enough to compact."""
    session = ctx.require(SESSIONS).create(session_id)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    # The concrete driver, whose `session` is a `Session` and not the
    # `Session | None` of the handle. Every read below then narrows by itself.
    assert isinstance(agent, ReactLoopAgent)
    for index in range(turns):
        await agent.prompt(f"question {index} " + "detail " * (QUESTION_CHARS // 7))
    return agent


async def _reaching_for_a_tool() -> Any:  # noqa: ANN401
    """A reply that is one tool call and no prose. A model that runs under the
    system prompt of the session, and holds the tools of the session, may well
    answer this way when it is asked to summarize."""
    yield BlockStart(index=0, block_type="tool-call")
    yield BlockEnd(
        index=0, block=ToolCallBlock(id="t1", name="read", arguments='{"path": "notes.md"}')
    )
    yield Finish(reason=FinishReason(kind="tool-calls"))


async def _overflowing() -> Any:  # noqa: ANN401
    """A stream that ends the way a provider ends one it cannot accept."""
    yield Finish(
        reason=FinishReason(
            kind="error",
            failure=LlmFailure(message="too long", code=CONTEXT_WINDOW_EXCEEDED, status=400),
        )
    )


# ------------------------------------------------------------- the triggers --


def test_the_fraction_trigger_fires_at_0_85_of_a_known_window() -> None:
    """The first number of G4, as a pair: 0.85 of the window triggers, and just
    under 0.85 does not.

    The test goes through the real baseline and not a stubbed one.
    `ctx.token_meter` reads the reported usage on the newest
    `assistant/message`, so the test also checks that the trigger sees
    *provider* numbers when any exist (D15).
    """
    window = 1_000
    at_the_line = int(window * TRIGGER_FRACTION)

    def engine_for(total: int) -> tuple[SummarizeEngine, Session]:
        ctx = _bare_context()
        session = Session(f"pressure-{total}")
        log_event(
            session,
            "request/context",
            {"provider": "fake", "model": "f", "contextWindow": window},
        )
        log_event(
            session,
            "assistant/message",
            {**assistant_payload("hi", "m1"), "usage": {"inputTokens": total, "outputTokens": 0}},
            SurfaceIntent("append"),
        )
        return SummarizeEngine(ctx=ctx, config=Config()), session

    under, under_session = engine_for(at_the_line - 1)
    over, over_session = engine_for(at_the_line)
    assert under._under_pressure(under.ctx.require(TOKEN_METER).baseline(under_session)) is False
    assert over._under_pressure(over.ctx.require(TOKEN_METER).baseline(over_session)) is True


def test_without_a_window_the_trigger_is_the_fixed_token_count() -> None:
    """The second number of G4: the branch that upstream takes when the model
    profile has no `max_input_tokens`. 170 000 tokens trigger, and 169 999 do
    not.

    pH reaches this branch on the first step of every session, because
    `request/context` is appended inside `_build_request`, which runs *after*
    `agent/pre-step`. It is therefore not a rare fallback. It is the opening
    state of every conversation.
    """

    def engine_for(total: int) -> tuple[SummarizeEngine, Session]:
        session = Session(f"unwindowed-{total}")
        log_event(
            session,
            "assistant/message",
            {**assistant_payload("hi", "m1"), "usage": {"inputTokens": total, "outputTokens": 0}},
            SurfaceIntent("append"),
        )
        return SummarizeEngine(ctx=_bare_context(), config=Config()), session

    under, under_session = engine_for(TRIGGER_TOKENS - 1)
    over, over_session = engine_for(TRIGGER_TOKENS)
    assert under._under_pressure(under.ctx.require(TOKEN_METER).baseline(under_session)) is False
    assert over._under_pressure(over.ctx.require(TOKEN_METER).baseline(over_session)) is True


def _bare_context() -> Context:
    """A context with only the token meter, which is what `_under_pressure` reads.

    The context is built by hand and not mounted. The two threshold tests are
    about arithmetic over a log. A whole profile mounted for them would make the
    failure of an unrelated row look like a threshold regression.
    """
    from ph.seams.token_meter import TokenMeter

    ctx = Context()
    ctx.provide("token_meter", TokenMeter(ctx=ctx))
    return ctx


# ------------------------------------------------------- the split, which is A3 --


async def test_the_model_reads_the_summary_and_the_log_keeps_the_conversation(
    mount: MountProfile,
) -> None:
    """A3 in one pair of assertions, and the reason why compaction is a surface op.

    The history of the model is replaced. The history of the user is not, and
    neither is the log.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "split")
    session = agent.session
    before = len(session.events)

    result = await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    assert SUMMARY in _model_text(session)
    assert "question 0" not in _model_text(session), "the model was sent the history anyway"
    assert "question 0" in _human_text(session), "the transcript lost what was said"
    assert len(session.events) > before, "the log shrank"
    assert "compacted" in (result or "")


async def test_the_replaced_conversation_is_written_to_conversation_history(
    mount: MountProfile,
) -> None:
    """The other half of A3: the originals are on disk, at the path that the
    model was given.

    This is a relocation, not a deletion. `ctx.spill_store` makes the same
    promise for an offloaded tool result, which is why the replacement names the
    file.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "history")
    session = agent.session

    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    (record,) = _events(session, "compaction/summarized")
    locator = record.data["locator"]
    assert "conversation_history" in locator
    assert locator in _model_text(session), "the model was not told where the history went"
    assert "question 0" in Path(locator).read_text(encoding="utf-8")


async def test_the_summary_is_the_plugins_text_and_says_it_is_a_compaction(
    mount: MountProfile,
) -> None:
    """Attribution, and the discriminator that every reader keys on.

    `input-offload` also appends a plugin-authored replacement. Only this one
    claims that conversation has left the derivation, so only this one says
    `form: compaction`. The TUI reads exactly this field to decide whether to
    call a row "compacted".
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "attribution")
    session = agent.session

    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    replacement = next(
        event
        for event in session.events
        if event.type == "user/message" and event.surface_op != "append"
    )
    source = as_obj(replacement.data["source"])
    assert source["kind"] == "plugin"
    assert source["plugin"] == "compaction-summarize"
    assert source["form"] == "compaction"


async def test_the_replacement_cites_every_node_it_shadows(mount: MountProfile) -> None:
    """What makes the substitution a reversible reading and not a deletion.

    `Session.append` refuses a replace that shadows a node that it does not
    cite. This is therefore really a test that the *plan* and the *op* agree:
    the range is chosen by surface position, and the citation is built from the
    same list.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "provenance")
    session = agent.session

    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    (record,) = _events(session, "compaction/summarized")
    [replaced] = [as_int(seq) for seq in as_seq(record.data["revises"])]
    replacement = session.events[replaced]
    assert replacement.type == "user/message", "the record does not explain the summary"
    assert session.records_of(replaced) == (record,)
    size = as_obj(record.data["size"])
    assert (size["unit"], size["before"]) == ("tokens", record.data["shadowedTokens"])
    assert 0 < as_int(size["after"]) < as_int(size["before"])
    # `list(...)` on both sides: an appended payload is deep-frozen, so the seqs
    # come back as a tuple, and an `==` against the citation would compare
    # shapes.
    assert list(replacement.source_event_seqs or ()) == list(record.data["shadowedSeqs"])


# --------------------------------------------------- argument truncation --
# §7.4 item 2. The cheap half of the row. Bytes that the model itself sent are
# elided from what it is shown, and the log keeps its exact words. There is no
# model call.


WINDOW = 1_000
"""A small window in a hand-built session, so that the *shipped* fractions apply.

The truncation and clip passes are pure functions of a session. A test that
builds a session directly can put the surface in the exact shape that each pass
exists for: an over-budget trailing tool batch, or a long-argument call in old
history. A scripted conversation reaches those shapes only by accident.
"""


def _windowed(session_id: str) -> Session:
    """A session whose baseline is already over 0.85 of a known window."""
    session = Session(session_id)
    log_event(
        session, "request/context", {"provider": "fake", "model": "f", "contextWindow": WINDOW}
    )
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
    log_event(
        session,
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
    log_event(session, "user/message", user_payload("save it", "m1"), SurfaceIntent("append"))
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=[_write_call("c1", body)]),
        SurfaceIntent("append"),
    )
    log_event(
        session, "tool/result", tool_result_payload("written", "m3", "c1"), SurfaceIntent("append")
    )
    _pressured(session)
    return session


def _engine(ctx: Context) -> SummarizeEngine:
    # `CompactionSeam.engine` is `CompactionEngine | None`: absent until a row
    # registers one. Every caller here mounts the row that registers it, and
    # the assertion below states that.
    engine = ctx.require(COMPACTION).engine
    assert isinstance(engine, SummarizeEngine), "the summarize row is not mounted"
    return engine


async def test_the_engine_reports_its_trigger_and_none_when_it_does_not_compact(
    mount: MountProfile,
) -> None:
    """The value that the readouts of ph-clm warn ahead of: the share at which
    this engine compacts by itself, read through the core seam. The engine
    reports nothing when automatic compaction is off, because then no summary is
    coming to warn about."""
    armed = await mount(
        {"id": "compaction-summarize", "config": {"triggerFraction": 0.8}}, profile=PROFILE
    )
    disarmed = await mount(
        {"id": "compaction-summarize", "config": {"auto": False}}, profile=PROFILE
    )

    assert armed.require(COMPACTION).trigger_fraction() == 0.8
    assert disarmed.require(COMPACTION).trigger_fraction() is None


def _truncate(ctx: Context, session: Session) -> tuple[int, ...]:
    """The truncation pass, with the baseline that its caller now computes one
    time.

    The agent is a `StubAgent` on the mount, and not the `None` that this used
    to take and that no caller ever passed. The lookup needs a scope, and the
    mount is the root scope on which the fs tools register. A hand-built session
    therefore still asks the real registry whether a tool declares its arguments
    disposable. That `None` prevented `truncate_arguments` and
    `_elides_arguments` from having a non-optional agent.
    """
    return _engine(ctx).truncate_arguments(
        StubAgent(ctx=ctx, session=session),
        session,
        "pressure",
        ctx.require(TOKEN_METER).baseline(session),
    )


def _arguments_of(session: Session, name: str = "write") -> list[str]:
    """The call arguments the *model* now sees, from the derived surface."""
    return [
        block.arguments
        for message in session.derive_messages()
        for block in message.content
        if isinstance(block, ToolCallBlock) and block.name == name
    ]


async def test_only_a_tool_that_declares_it_has_arguments_elided(mount: MountProfile) -> None:
    """`write` declares `arguments_disposable`, and the registry gives that
    answer.

    A deployment that renames the fs tools, or an MCP server that adds its own,
    keeps working. A hardcoded name list in this package could not do that.
    """
    ctx = await mount(profile=PROFILE)
    assert not_none(ctx.require(TOOLS).get("write", scope=DEPLOYMENT)).arguments_disposable
    assert not not_none(ctx.require(TOOLS).get("read", scope=DEPLOYMENT)).arguments_disposable, (
        "reading is not a payload"
    )


async def test_a_long_call_argument_is_elided_from_what_the_model_sees(mount: MountProfile) -> None:
    """The pass, and the split that it preserves.

    The model is no longer shown a file body that it wrote, and the log keeps
    the bytes that it actually sent. This is the same split between log and
    surface that compaction uses, applied without a model call.
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


async def test_an_argument_at_the_limit_is_left_alone(mount: MountProfile) -> None:
    """The boundary as a pair, for the same reason as every threshold here: a
    threshold that is tested only from the far side passes for any limit at or
    below it."""
    ctx = await mount(profile=PROFILE)

    at_limit = _long_write_session("at-limit", "x" * MAX_ARG_LENGTH)
    over = _long_write_session("over-limit", "x" * (MAX_ARG_LENGTH + 1))

    assert _truncate(ctx, at_limit) == ()
    assert _truncate(ctx, over) != ()


async def test_the_elision_keeps_the_call_id_so_the_pair_still_balances(
    mount: MountProfile,
) -> None:
    """The rewrite must not change what the call/result pairing sees.

    A replacement that dropped or renamed the call would leave its result
    orphaned. That is the same failure that `safe_cutoff` exists to prevent, and
    it would show only later, as a compaction that could no longer find a cut.
    """
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("pairing", "x" * (MAX_ARG_LENGTH + 1))
    before = balanced_cuts(session)

    _truncate(ctx, session)

    assert balanced_cuts(session) == before
    assert json.loads(_arguments_of(session)[0])["path"] == "notes.md", "a short arg was touched"


async def test_the_replacement_carries_no_usage(mount: MountProfile) -> None:
    """The hazard that makes this rewrite different from the others.

    `TokenMeter.last_usage` folds to the newest `assistant/message` with a usage
    block, and a replacement is appended at the end of the log. A replacement
    that copied the usage of its original would become the last count of the
    meter: a request that was never made. The TUI footer, which reads the last
    usage that it sees, would show it.

    The baseline does move, and should. The elided argument left the surface, so
    it is estimated and taken off the count of the provider. That lets the
    elision avoid a summary this step.
    """
    ctx = await mount(profile=PROFILE)
    meter = ctx.require(TOKEN_METER)
    session = _long_write_session("usage", "x" * (MAX_ARG_LENGTH + 1))
    _pressured(session, used=950)
    usage_before = meter.last_usage(session)
    baseline_before = meter.baseline(session).tokens

    _truncate(ctx, session)

    replacement = session.events[-2]
    assert replacement.type == "assistant/message"
    assert "usage" not in replacement.data
    assert meter.last_usage(session) == usage_before, "the replacement became the count"
    assert meter.baseline(session).tokens < baseline_before, "the elision is not in the baseline"


async def test_a_recent_message_is_not_truncated(mount: MountProfile) -> None:
    """Retention, which is the reason why this pass does not just "shorten
    everything". The model is still working with the tail, and an argument that
    it is about to act on is exactly the one that it must be able to read."""
    ctx = await mount(profile=PROFILE)
    session = _windowed("recent")
    _pressured(session)
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=[_write_call("c1", "x" * (MAX_ARG_LENGTH + 1))]),
        SurfaceIntent("append"),
    )

    assert _truncate(ctx, session) == ()


async def test_truncating_twice_changes_nothing_the_second_time(mount: MountProfile) -> None:
    """Idempotence follows from the length check, because an elided argument is
    40-odd characters. It is still asserted, because a pass that ran every step
    and appended a replacement every step would grow the log without bound."""
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("twice", "x" * (MAX_ARG_LENGTH + 1))

    assert _truncate(ctx, session) != ()
    length = len(session.events)
    assert _truncate(ctx, session) == ()
    assert len(session.events) == length


async def test_a_truncation_pass_lands_whole_or_not_at_all(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P10-14. The rewrites and `compaction/args-truncated` are one batch.

    When they were appended one at a time, a rewrite that was refused partway
    left the earlier ones in the log, elided from what the model sees, with no
    record of the elision or of its reason. Here the second rewrite renames its
    message. The surface refuses that at the write, after the first rewrite has
    been stamped. Nothing of the pass lands, and the caller gets the refusal and
    not a half-applied pass.
    """
    ctx = await mount(profile=PROFILE)
    body = "x" * (MAX_ARG_LENGTH + 1)
    session = _windowed("whole")
    for turn in (1, 2):
        log_event(session, "user/message", user_payload("save it", f"u{turn}"), SurfaceIntent())
        log_event(
            session,
            "assistant/message",
            assistant_payload("", f"a{turn}", content=[_write_call(f"c{turn}", body)]),
            SurfaceIntent(),
        )
        log_event(
            session,
            "tool/result",
            tool_result_payload("written", f"r{turn}", f"c{turn}"),
            SurfaceIntent(),
        )
    _pressured(session)
    before = (session.seq, session.surface.nodes)

    real = compaction.truncated_assistant_message
    calls: list[int] = []

    def refused_second(event: Any, **options: Any) -> Any:  # noqa: ANN401
        calls.append(event.seq)
        found = real(event, **options)
        if len(calls) == 2 and found is not None:
            return {**found[0], "id": "renamed"}, found[1]
        return found

    monkeypatch.setattr(compaction, "truncated_assistant_message", refused_second)
    with pytest.raises(SurfaceError, match="must keep the message id"):
        _truncate(ctx, session)

    assert len(calls) == 2, "the pass never reached a second rewrite"
    assert (session.seq, session.surface.nodes) == before
    assert session.latest("compaction/args-truncated") is None

    monkeypatch.setattr(compaction, "truncated_assistant_message", real)
    assert len(_truncate(ctx, session)) == 2
    record = not_none(session.latest("compaction/args-truncated"))
    assert record.seq == session.seq - 1
    # One record for both rewrites, and each finds it.
    replacements = [as_int(seq) for seq in as_seq(record.data["revises"])]
    assert len(replacements) == 2
    assert all(session.records_of(seq) == (record,) for seq in replacements)
    size = as_obj(record.data["size"])
    assert size["unit"] == "tokens" and as_int(size["after"]) < as_int(size["before"])


async def test_a_tool_that_does_not_declare_it_keeps_its_arguments(mount: MountProfile) -> None:
    """The restriction is real, and the *tool* answers it.

    Upstream matches `{"write_file", "edit_file"}`, names that pH does not
    register. This row asks `ToolDefinition.arguments_disposable` instead, in
    the way that the offload row asks `self_limits`. A long `run_code` cell in
    old history is the argument that a model most often reads again. It declares
    nothing, so it stays.
    """
    ctx = await mount(profile=PROFILE)
    session = _windowed("other-tool")
    log_event(session, "user/message", user_payload("run it", "m1"), SurfaceIntent("append"))
    cell = {
        "type": "tool-call",
        "id": "c1",
        "name": "run_code",
        "arguments": json.dumps({"program": "x" * (MAX_ARG_LENGTH + 1)}),
    }
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=[cell]),
        SurfaceIntent("append"),
    )
    log_event(
        session, "tool/result", tool_result_payload("ok", "m3", "c1"), SurfaceIntent("append")
    )
    _pressured(session)

    assert _truncate(ctx, session) == ()


def test_arguments_that_are_not_a_json_object_are_left_alone() -> None:
    """A malformed argument string must not be rewritten without telling the
    model. And to serialize a non-object again would change its shape, not only
    its size."""
    assert truncated_arguments("not json at all", 10) is None
    assert truncated_arguments(json.dumps(["a" * 100]), 10) is None
    assert truncated_arguments(json.dumps({"short": "ok"}), 10) is None


async def test_truncation_runs_inside_the_automatic_path(mount: MountProfile) -> None:
    """The wiring: the free remedy is tried before the expensive one.

    The test goes through `compact_if_needed`, which is what `agent/pre-step`
    calls. It therefore fails if the pass is correct but nothing invokes it.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = _long_write_session("automatic", "x" * (MAX_ARG_LENGTH + 1))
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await _engine(ctx).compact_if_needed(agent, "pressure")

    assert _events(session, "compaction/args-truncated"), "the cheap pass never ran"


# ------------------------------------------------------- the overflow clip --
# §7.4 item 7. The case where summarization can help least. A trailing
# tool-result batch that is larger than the retention budget leaves no balanced
# cut, so `safe_cutoff` correctly declines, and nothing else would ever shrink
# the batch.


def _tool_batch_session(session_id: str, *sizes: int, opening: bool = True) -> Session:
    """A session that ends in a batch of tool results of the given sizes.

    `opening=False` drops the leading user message, which makes the surface
    really uncuttable. With the message, the first node is a balanced cut by
    itself, and summarization has a place to cut.
    """
    session = _windowed(session_id)
    if opening:
        log_event(session, "user/message", user_payload("read them", "m1"), SurfaceIntent("append"))
    calls = [
        {"type": "tool-call", "id": f"c{index}", "name": "read", "arguments": "{}"}
        for index, _ in enumerate(sizes)
    ]
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=calls),
        SurfaceIntent("append"),
    )
    for index, size in enumerate(sizes):
        log_event(
            session,
            "tool/result",
            tool_result_payload(f"{index}" * size, f"r{index}", f"c{index}"),
            SurfaceIntent("append"),
        )
    return session


async def test_an_oversized_trailing_batch_is_clipped_and_recoverable(mount: MountProfile) -> None:
    """Each result is relocated, not dropped. `ctx.spill_store` makes that
    promise everywhere in this bundle."""
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("clip", 30_000, 30_000)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    clipped = await _engine(ctx).clip_overflow_tail(agent)

    assert len(clipped) == 2
    spilled = _events(session, "offload/spilled")
    assert len(spilled) == 2
    # Keyed by call id, because the two results differ. An assertion that
    # compared both files to one expected body would pass for a clip that
    # spilled the same result twice.
    originals = {"c0": "0" * 30_000, "c1": "1" * 30_000}
    for record in spilled:
        locator = record.data["locator"]
        assert locator in _derived_text(session)
        assert Path(locator).read_text(encoding="utf-8") == originals[record.data["callId"]]
    for body in originals.values():
        assert body not in _derived_text(session)


async def test_a_result_the_clip_would_grow_is_not_spilled(mount: MountProfile) -> None:
    """S14: `_clip_one` used to spill, record and publish before it decided.

    A result that is shorter than its pointer is left inline. It used to be left
    inline *after* its blob was written and an `offload/spilled` record was
    appended: "full text at X" in the transcript, for text that never moved. The
    decision now comes first, so a declined clip leaves nothing behind. A clip
    that does happen lands its accounting and its replacement in one batch.

    Sabotage: spill before the length check again, and the short result has a record.
    """
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("clip-short", 30_000, 40)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    clipped = await _engine(ctx).clip_overflow_tail(agent)

    assert len(clipped) == 1
    (spilled,) = _events(session, "offload/spilled")
    assert spilled.data["callId"] == "c0", "the short result was spilled and left inline"
    [revised] = [as_int(seq) for seq in as_seq(spilled.data["revises"])]
    replaced = session.events[revised]
    assert replaced.type == "tool/result" and replaced.batch == spilled.batch
    assert spilled.batch is not None
    assert session.records_of(revised) == (spilled,)
    size = as_obj(spilled.data["size"])
    assert (size["unit"], size["before"]) == ("bytes", 30_000)
    assert as_int(size["after"]) < 30_000


async def test_the_clip_changes_only_the_result_content(mount: MountProfile) -> None:
    """`Session.append` refuses a `tool/result` replacement that touches anything
    but content. This is therefore really a test that the payload is rebuilt
    from the original and not synthesized. A fresh message id would be rejected,
    and a rejected append would make the whole overflow path fail.
    """
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("content-only", 30_000)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await _engine(ctx).clip_overflow_tail(agent)

    replacement = session.derive_messages()[-1]
    assert replacement.id == "r0", "the message identity moved"
    block = replacement.content[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_call_id == "c0"


async def test_a_small_result_is_not_replaced_by_a_larger_pointer(mount: MountProfile) -> None:
    """The batch is what must shrink, and a member that would grow is not part
    of shrinking it. Upstream clips every message in an over-budget batch. A
    hundred-character result swapped for a nine-hundred-character pointer is a
    clip that made the request bigger."""
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("mixed", 30_000, 40)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    clipped = await _engine(ctx).clip_overflow_tail(agent)

    assert len(clipped) == 1
    assert "1" * 40 in _derived_text(session), "the small result was replaced anyway"


async def test_a_conversation_that_does_not_end_in_tool_results_is_untouched(
    mount: MountProfile,
) -> None:
    """The cheap check that keeps this pass free in the ordinary case."""
    ctx = await mount(profile=PROFILE)
    session = _long_write_session("no-tail", "x" * 100)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    assert await _engine(ctx).clip_overflow_tail(agent) == ()


async def test_results_the_step_s_batch_came_after_are_still_clipped(mount: MountProfile) -> None:
    """A refusal comes after the batch of the step, and the batch can put a
    message after the results: the session facts of rlm, added again after a
    summary has shadowed them. This was found live: the clip found no results at
    the very end, and the turn ended in the refusal. Results that the model has
    answered since are not the tail, however large they are.
    Sabotage: look for results at the very end only, or past a reply too."""
    ctx = await mount(profile=PROFILE)
    session = _tool_batch_session("past-the-batch", 30_000)
    answered = _tool_batch_session("answered", 30_000)
    _pressured(answered)
    for one in (session, answered):
        log_event(one, "user/message", user_payload("# Session", "m9"), SurfaceIntent("append"))
    agents = ctx.require(AGENTS)

    assert len(await _engine(ctx).clip_overflow_tail(agents.create(session, FAKE_OPTIONS))) == 1
    assert "0" * 30_000 not in _derived_text(session)
    assert "# Session" in _model_text(session), "the batch was clipped too"
    assert await _engine(ctx).clip_overflow_tail(agents.create(answered, FAKE_OPTIONS)) == ()


async def test_the_clip_alone_is_enough_to_retry_the_request(mount: MountProfile) -> None:
    """Why the clip is worth having beside summarization and not inside it.

    This session has one enormous outstanding call/result pair and nothing else.
    *Every* balanced cut is therefore the start of the conversation, and
    summarization correctly declines. Before the clip existed, the overflow path
    had no remedy at all, and the turn ended in the refusal of the provider. Now
    the tail shrinks and the loop is asked to try again.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = _tool_batch_session("only-remedy", 30_000, opening=False)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    engine = _engine(ctx)

    assert engine._plan(session) is None, "the fixture is not the no-safe-cut shape"
    assert await engine.clip_overflow_tail(agent) != ()


# --------------------------------------------------------------- pressure --


async def test_pressure_compacts_without_anyone_asking(mount: MountProfile) -> None:
    """The automatic half of G4, end to end and at the shipped threshold.

    Nothing here calls the engine. A conversation that is long enough crosses
    0.85 of the 8 192-token window of the fake adapter by itself, and
    `agent/pre-step` is where that is noticed. The two threshold tests above
    prove the arithmetic. This test proves that the arithmetic is wired to
    something.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = ctx.require(SESSIONS).create("pressure")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    for index in range(10):
        await agent.prompt(f"question {index} " + "detail " * 600)

    (record,) = _events(session, "compaction/summarized")
    assert record.data["trigger"] == "pressure"
    assert SUMMARY in _model_text(session)
    # The compaction also relieved what triggered it, and does not fire on every
    # step from here on: there is one record, and the baseline is back under the
    # line.
    pressure = ctx.require(TOKEN_METER).baseline(session).pressure
    assert pressure is not None and pressure < TRIGGER_FRACTION


async def test_the_automatic_triggers_can_be_turned_off_without_losing_the_command(
    mount: MountProfile,
) -> None:
    """The `auto` setting of dsh, and why it is separate from the row itself.

    A deployment can want a human to decide when history is replaced, and still
    want the human to be *able* to do it. `auto: false` therefore turns off
    automatic compaction under all three triggers and leaves `/compact` working.
    """
    ctx = await mount({"id": "compaction-summarize", "config": {"auto": False}}, profile=PROFILE)
    _route(ctx)
    session = ctx.require(SESSIONS).create("manual-only")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    for index in range(10):
        await agent.prompt(f"question {index} " + "detail " * 600)
    assert not _events(session, "compaction/summarized"), "the automatic trigger stayed armed"

    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)
    assert _events(session, "compaction/summarized"), "the command was disarmed too"


# ----------------------------------------------------------- the envelope --
# §7.4 item 5. The summarize call reuses the system prompt and tools of the
# conversation, so that the request is a strict *prefix* of the request that the
# loop just made. That makes the call nearly free on a provider that caches
# prefixes, and it stops the summarizer from reading only a tail of the range
# that it replaces.


async def test_the_summarize_request_is_a_prefix_of_the_conversations(
    mount: MountProfile,
) -> None:
    """The cache property, asserted structurally and not assumed.

    A cache miss looks exactly like a hit, except on the invoice.
    `test_prefix_stability` exists for the same reason. The shape is therefore
    checked: an identical system, identical tools, and a message list that the
    request of the conversation begins with.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "prefix")

    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    conversation = _loop_requests(ctx)[-1]
    (summary,) = _summary_requests(ctx)
    assert summary.system == conversation.system, "the system prompt moved; the prefix is lost"
    assert summary.tools == conversation.tools, "the tool schemas moved; the prefix is lost"
    assert summary.tools, "the fixture has no tools, so the assertion above is vacuous"
    shared = len(summary.messages) - 1
    assert summary.messages[:shared] == conversation.messages[:shared]


async def test_the_summarizer_is_shown_the_whole_range(mount: MountProfile) -> None:
    """No tail and no rendering: the shadowed messages are sent as themselves.

    Two things follow. The model reads real `tool-call`/`tool-result` blocks and
    not the text rendering of them by `render_for_summary`. And nothing in the
    range is silently withheld. A summary that is written from a tail cannot
    mention what it was never shown, and could not say so either.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "whole-range")

    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    (record,) = _events(agent.session, "compaction/summarized")
    (summary,) = _summary_requests(ctx)
    assert record.data["shape"] == "replay"
    # Every shadowed node, plus the one instruction appended after it.
    assert len(summary.messages) == len(record.data["shadowedSeqs"]) + 1
    assert "question 0" in text_of(summary.messages[0].content)


async def test_the_replay_does_not_inherit_a_reasoning_budget(mount: MountProfile) -> None:
    """One of two deliberate departures from byte-identical.

    A reasoning model would otherwise spend a thinking budget on an extraction.
    Neither this departure nor the `max_tokens` override is part of the token
    prefix, so neither costs the cache hit. That is exactly why these are the
    two things worth departing on.
    """
    budget = {"thinking": {"type": "enabled", "budget_tokens": 4096}}
    route = {"provider": "fake", "model": "fake-1", "efforts": {"high": budget}}
    ctx = await mount(models_overlay(main=route), profile=PROFILE)
    _route(ctx)
    session = ctx.require(SESSIONS).create("thinking")
    options = AgentOptions(provider="fake", model="fake-1", reasoning_effort="high")
    agent = ctx.require(AGENTS).create(session, options)
    for index in range(8):
        await agent.prompt(f"question {index} " + "detail " * (QUESTION_CHARS // 7))

    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    (summary,) = _summary_requests(ctx)
    assert _loop_requests(ctx)[-1].effort_fields == budget, "the fixture proves nothing"
    assert summary.effort_fields is None
    assert summary.max_tokens == SUMMARY_MAX_TOKENS


async def test_a_reply_of_only_tool_calls_falls_back_without_them(mount: MountProfile) -> None:
    """The cost of carrying the tools of the conversation, and the answer to it.

    The replay gives the model the tool schemas of the session under the system
    prompt of the session, so a model that is primed to *act* can answer with a
    tool call and no prose. That is not a failure to report. It is a request
    shape that did not fit, so the row asks again without tools and with the
    whole range. Cheap first, correct second. The fallback hides nothing. It
    gives up only the cache.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "tool-happy")
    answered = {"once": False}

    async def call_a_tool_first(
        options: GenerateOptions, next_: Callable[..., Awaitable[Any]]
    ) -> Any:  # noqa: ANN401
        if options.purpose == "compaction" and not answered["once"]:
            answered["once"] = True
            ctx.require(LLM_FAKE).requests.append(options)
            return _reaching_for_a_tool()
        return await next_(options)

    ctx.on("llm/stream", call_a_tool_first)
    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    first, second = _summary_requests(ctx)
    assert first.tools, "the first attempt did not carry the conversation's tools"
    assert not second.tools, "the retry carried the tools that caused the problem"
    assert len(second.messages) == 1, "the retry did not fall back to the self-contained shape"
    (record,) = _events(agent.session, "compaction/summarized")
    assert record.data["shape"] == "direct-after-replay"
    assert SUMMARY in _model_text(agent.session), "the fallback did not produce a summary"


async def test_a_session_with_no_logged_request_still_compacts(mount: MountProfile) -> None:
    """There is no envelope to replay before the first request is built, and a
    hand-built session may never have one. The self-contained shape is therefore
    the floor and not a failure path."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = _tool_batch_session("headerless", 200, 200)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    assert session.request_header() is None
    result = await _engine(ctx).compact_now(agent)

    assert result is not None
    (record,) = _events(session, "compaction/summarized")
    assert record.data["shape"] == "direct"


# -------------------------------------------------------------- the overflow --


async def test_a_context_overflow_compacts_and_retries_the_request(mount: MountProfile) -> None:
    """The trigger that `llm-retry` deliberately leaves alone (G4).

    A provider that says `CONTEXT_WINDOW_EXCEEDED` has already answered the
    question that a pressure estimate can only guess at, and it is the one
    failure with a remedy. This row therefore compacts and asks the loop to try
    again, and the turn completes on the second attempt and does not end in the
    error.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "overflow")
    session = agent.session
    refused = {"once": False}

    async def refuse_the_first_request(
        options: GenerateOptions, next_: Callable[..., Awaitable[Any]]
    ) -> Any:  # noqa: ANN401
        # On `llm/stream`, the seam that retry and replay already attach to, and
        # not by a swap of the method of the adapter. `FakeAdapter` has slots,
        # and the waterfall is where a provider failure is *supposed* to come
        # from, so the request travels the same path that a real refusal would.
        if options.is_loop_request and not refused["once"]:
            refused["once"] = True
            return _overflowing()
        return await next_(options)

    ctx.on("llm/stream", refuse_the_first_request)
    await agent.prompt("one more question")

    (record,) = _events(session, "compaction/summarized")
    # Never a replay: the envelope that would be replayed is the request that
    # the provider just refused for being too large, and to send it again with
    # an instruction appended is refused again.
    assert record.data["shape"] == "direct"
    assert SUMMARY in _model_text(session)
    assert text_of(session.derive_messages()[-1].content) == "answer", "the retry did not land"


SLOT = 2_000
"""A window that one tool result can outgrow by itself, as a 64 KB cell outgrows
the 16 384-token slot of a local server."""


async def test_a_request_past_the_window_is_clipped_before_it_is_sent(
    mount: MountProfile,
) -> None:
    """The result of a step is larger than the whole window, on a server that
    refuses what does not fit. The clip runs at the request of the next step,
    before the request is sent, so nothing is refused.

    A summary leaves the latest turn whole, so the request used to go out
    anyway, be refused, and only then be clipped. That was one wasted request
    each time, and a line in the log of the server that a person reads as a
    fault. Sabotage: leave the clip to the refusal, and the server refuses once.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    ctx.require(LLM_FAKE).context_window = SLOT
    meter = ctx.require(TOKEN_METER)
    body = "word " * (SLOT * 2)
    ctx.require(TOOLS).register(simple_tool("dump", lambda _arguments, _run: body))
    refused: list[int] = []
    called = False

    async def server(options: GenerateOptions, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        nonlocal called
        if not options.is_loop_request:
            return await next_(options)
        size = meter.estimate_messages(options.messages)
        if size >= SLOT:
            refused.append(size)
            return _overflowing()
        if not called:
            called = True
            return streamed(tool_call_chunks("c1", "dump", "{}"))
        return await next_(options)

    ctx.on("llm/stream", server)
    session = ctx.require(SESSIONS).create("past-the-window")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await agent.prompt("dump it")

    assert refused == [], "a request past the window was sent"
    (spilled,) = _events(session, "offload/spilled")
    assert spilled.data["callId"] == "c1"
    assert Path(spilled.data["locator"]).read_text(encoding="utf-8") == body
    assert text_of(session.derive_messages()[-1].content) == "answer"


async def test_a_conversation_still_past_the_window_is_summarized_from_its_tail(
    mount: MountProfile,
) -> None:
    """Past the window with no results to clip, the check of the request itself
    summarizes. It sends the summarizer the tail of the conversation and not a
    replay of it, because the provider would refuse the replay as well. The
    pressure check of the step leaves the conversation to that check.
    Sabotage: choose the shape by the trigger alone, or summarize at the step's
    start as well."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "past-the-window-history")
    session = agent.session
    log_event(session, "request/context", {"provider": "fake", "model": "f", "contextWindow": SLOT})
    pressure = ctx.require(TOKEN_METER).baseline(session).pressure
    assert pressure is not None and pressure >= 1, "the fixture is not past the window"

    assert await _engine(ctx).compact_if_needed(agent, "pressure") is None
    assert await _engine(ctx).compact_if_needed(agent, "window") is not None

    (record,) = _events(session, "compaction/summarized")
    assert (record.data["trigger"], record.data["shape"]) == ("window", "direct")


# --------------------------------------------------------------- declining --


async def test_a_summarizer_that_fails_leaves_the_conversation_untouched(
    mount: MountProfile,
) -> None:
    """A compaction that cannot summarize must change nothing.

    This guards against the worst failure mode available: to shadow a range with
    an empty or missing summary would delete the conversation from the view of
    the model and put nothing in its place.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "broken-summarizer")
    session = agent.session
    before = _model_text(session)
    ctx.require(LLM_FAKE).respond = lambda request: (
        "" if request.purpose == "compaction" else "answer"
    )

    reply = await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    assert not _events(session, "compaction/summarized")
    assert _model_text(session) == before
    assert "unchanged" in (reply or "")


async def test_an_automatic_decline_is_recorded_and_not_retried_at_once(
    mount: MountProfile,
) -> None:
    """Automatic compaction never raises. It records the reason and returns.

    The decline also holds until the surface moves. Without that, the overflow
    path would attempt again inside the same step loop, and spend one model call
    per attempt to reach the same answer.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "declined")
    session = agent.session
    engine = ctx.require(COMPACTION).engine
    assert isinstance(engine, SummarizeEngine)
    ctx.require(LLM_FAKE).respond = lambda request: (
        "" if request.purpose == "compaction" else "answer"
    )

    assert await engine.compact_if_needed(agent, "overflow") is None
    (declined,) = _events(session, "compaction/declined")
    assert declined.data["code"] == "summary"

    calls = len(_summary_requests(ctx))
    assert await engine.compact_if_needed(agent, "overflow") is None
    assert len(_summary_requests(ctx)) == calls, "the decline was retried with nothing changed"


async def test_a_manual_failure_is_recorded_in_the_log(mount: MountProfile) -> None:
    """The command tells the user "the attempt is recorded in the session log",
    and until this test that sentence was false.

    Only the automatic path recorded a decline. `/compact` raised through to the
    command, which turned the error into a message and left no compaction record
    at all. The log showed a `command/done` marked `ok` whose detail happened to
    describe a failure. The wording of dsh is true because its `compaction/end`
    carries the error. The wording of pH is true because of this.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "manual-failure")
    ctx.require(LLM_FAKE).respond = lambda request: (
        "" if request.purpose == "compaction" else "answer"
    )

    reply = await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    assert "recorded in the session log" in (reply or ""), "the claim under test moved"
    (declined,) = _events(agent.session, "compaction/declined")
    assert declined.data["trigger"] == "manual"
    assert declined.data["code"] == "summary"


async def test_a_refusal_before_any_attempt_is_not_recorded(mount: MountProfile) -> None:
    """`busy` is not a failed compaction. Nothing was attempted, so there is no
    attempt to account for. A record of one would put an event in the log for
    every mistyped command."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "busy-unrecorded")
    agent._set_phase("running")

    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    assert not _events(agent.session, "compaction/declined")


async def test_a_stub_that_says_it_is_working_is_refused_like_a_driver(
    mount: MountProfile,
) -> None:
    """The refusal reads `AgentHandle.status`, so a stub can state it.

    It could not before. `status` was on `AgentDriver`, and the guard asked
    `getattr(agent, "status", "idle")`. Every `StubAgent` therefore answered
    *idle*, and no test could reach this branch unless it ran a whole loop. The
    sibling above drives a real loop through `_set_phase`. This test asks for
    the same refusal from the surface that the seams actually hold.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = ctx.require(SESSIONS).create("busy-stub")
    working = StubAgent(ctx=ctx, session=session, status="running")

    with pytest.raises(CompactionError) as refused:
        await _engine(ctx).compact_now(working)

    assert refused.value.code == "busy"
    assert "the agent is working" in str(refused.value)


async def test_an_unexpected_failure_neither_escapes_nor_goes_unrecorded(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug in this row must not end the turn of the user.

    `agent/pre-step` propagates to `_turn`, where the driver contains the
    failure with one debug line. An exception here is therefore a stabilization
    row that silently ends a conversation. The guard is `Exception`, not
    `CompactionError`, and it records what it swallowed.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "buggy")
    engine = _engine(ctx)

    async def explode(*_args: object, **_kwargs: object) -> NoReturn:
        raise OSError("the disk went away")

    monkeypatch.setattr(type(engine), "_land", explode)

    assert await engine.compact_if_needed(agent, "overflow") is None
    (declined,) = _events(agent.session, "compaction/declined")
    assert declined.data["code"] == "error"
    assert "the disk went away" in declined.data["reason"]


async def test_cancellation_is_not_swallowed(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one thing that the guard must let through.

    `Canceled` derives from `Exception`, so a guard that is written to contain
    bugs would also contain a stop request from the user. The turn would then
    continue as though nothing had been asked of it.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "canceled")
    engine = _engine(ctx)

    async def stop(*_args: object, **_kwargs: object) -> NoReturn:
        raise Canceled("the user pressed stop")

    monkeypatch.setattr(type(engine), "_land", stop)

    with pytest.raises(Canceled):
        await engine.compact_if_needed(agent, "overflow")


async def test_a_short_conversation_reports_nothing_to_compact(mount: MountProfile) -> None:
    """`None`, not an error. "Your conversation is short" and "your summarizer
    is broken" are different outcomes, and the seam keeps them apart."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    session = ctx.require(SESSIONS).create("short")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    reply = await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    assert reply == "no compactable history yet"
    assert not _summary_requests(ctx), "a summarize call was spent on an empty session"


# --------------------------------------------------------------- fail open --


async def test_a_history_that_cannot_be_written_still_compacts_and_claims_no_path(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spill that failed must not leave the model reading a path that holds nothing.

    Upstream keeps both wordings for exactly this case, and the wording without
    a path is the necessary half. The summary is then all that exists, and to
    tell the model otherwise would be the harness inventing a file.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "no-disk")
    session = agent.session
    break_spill(monkeypatch)

    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    (record,) = _events(session, "compaction/summarized")
    assert record.data["locator"] is None
    assert REPLACEMENT_WITHOUT_PATH.format(summary=SUMMARY) in _model_text(session)
    assert "conversation_history" not in _model_text(session)


# ------------------------------------------------------------- the command --


async def test_compact_refuses_while_the_agent_is_working(mount: MountProfile) -> None:
    """A compaction landing mid-turn would move the surface underneath a request
    the loop had already derived."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "busy")

    agent._set_phase("running")
    reply = await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    assert "idle session" in (reply or "")
    assert not _events(agent.session, "compaction/summarized")


async def test_compact_carries_what_the_user_is_about_to_work_on(mount: MountProfile) -> None:
    """`/compact [instructions]`, and the reason why it takes any.

    dsh refuses arguments here, and the compact tool of deepagents takes none.
    But the moment when someone compacts on purpose is usually the moment when
    they change subject, and what they are about to do is the one thing that the
    summarizer cannot read from the conversation. The text reaches the prompt
    labeled as the text of the user. It is placed with the instruction: before
    `<messages>` in the direct shape, and in the trailing instruction on a
    replay, which is what this test reads. It is therefore read as an
    instruction and not as one more message to compress. It is also recorded on
    the event, because a summary that is weighted towards one thing should say
    who asked for that.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "focus")

    await ctx.require(COMMANDS).dispatch(
        "/compact next I am rewriting the retry policy", session=agent.session, agent=agent
    )

    (request,) = _summary_requests(ctx)
    assert "rewriting the retry policy" in _instruction(request)
    (record,) = _events(agent.session, "compaction/summarized")
    assert record.data["instructions"] == "next I am rewriting the retry policy"


async def test_a_bare_compact_asks_the_summarizer_to_focus_on_nothing(mount: MountProfile) -> None:
    """No argument means no focus block. An empty section would be a heading
    that the model must decide what to do with."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    agent = await _conversation(ctx, "no-focus")

    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    (request,) = _summary_requests(ctx)
    assert "what_the_user_asked_you_to_focus_on" not in _instruction(request)
    (record,) = _events(agent.session, "compaction/summarized")
    assert record.data["instructions"] is None


async def test_compact_without_an_engine_says_so(mount: MountProfile) -> None:
    """The command over a profile that layered the seam and no backend.

    Silence would read as "nothing to compact", which is a different and false
    statement. The automatic path is the one that is right to say nothing.
    """
    ctx = await mount({"id": "command-compact", "name": "command-compact"})
    session = ctx.require(SESSIONS).create("engineless")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    reply = await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)

    assert "no compaction engine" in (reply or "")


# ----------------------------------------------------------------- the notes --


async def test_a_note_tells_the_summarizer_what_survives_the_cut(mount: MountProfile) -> None:
    """The mechanism of G10, without the producer of G10.

    A summary replaces *conversation*. A note is how a plugin says "this other
    thing is still here". The note is appended to the trailing instruction,
    after the range that is summarized. The model therefore reads it as part of
    what it was asked to do, and not as one more piece of the history that it
    compresses.
    """
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    ctx.require(COMPACTION).note(
        CompactionNote(name="test:state", text=lambda _s: "`df` is still loaded")
    )
    agent = await _conversation(ctx, "notes")

    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    (request,) = _summary_requests(ctx)
    assert "`df` is still loaded" in _instruction(request)


async def test_a_note_that_renders_nothing_costs_no_prompt(mount: MountProfile) -> None:
    """Empty means absent, the rule that `PromptContext` already uses. A note
    about a namespace that holds nothing therefore does not spend a paragraph to
    say so."""
    ctx = await mount(profile=PROFILE)
    _route(ctx)
    ctx.require(COMPACTION).note(CompactionNote(name="test:empty", text=lambda _s: ""))
    agent = await _conversation(ctx, "empty-note")

    await ctx.require(COMMANDS).dispatch("/compact", session=agent.session, agent=agent)

    (request,) = _summary_requests(ctx)
    assert "state_that_survives_this_summary" not in _instruction(request)


# ------------------------------------------------------------ the vocabulary --


def test_the_event_types_are_ignorable() -> None:
    """Accounting beside the replacement that it describes, and the replacement
    is what a reader reads. (That the types are *known* is checked in
    `test_log_writers.py`, for every package.)"""
    for event_type in (
        "compaction/summarized",
        "compaction/declined",
        "compaction/args-truncated",
    ):
        assert event_type in IGNORABLE_SESSION_EVENT_TYPES


def test_the_retention_fraction_is_smaller_than_the_trigger() -> None:
    """dsh validates this at plugin load and refuses otherwise. The reason is
    worth repeating: if a row retains more than the threshold, every compaction
    leaves the session still over the threshold, and the next step compacts
    again."""
    assert KEEP_FRACTION < TRIGGER_FRACTION


# ----------------------------------------------- compacting what was compacted --


def _summarized(session_id: str, tail: str, *, compacted: bool = True) -> Session:
    """A session with a tail that fills the retention budget by itself.

    `compacted` decides whether a previous summary is above the tail. That is
    the only difference between the two halves of the test below. They share one
    fixture so that the control cannot drift into a test of a different shape.

    The session is built by hand and not by two runs of `/compact`, because the
    shape is what matters and it is a narrow one: the surface is a previous
    summary followed by a live turn that is big enough that the retention budget
    asks for a cutoff of 1. The replacement is written with `plugin_payload`,
    which gives it the shape that `_land` writes. The plugin, the form and the
    `SurfaceReplace` are what the reader under test keys on.
    """
    session = _windowed(session_id)
    first = log_event(
        session, "user/message", user_payload("the question", "m1"), SurfaceIntent("append")
    )
    second = log_event(
        session, "assistant/message", assistant_payload("the answer", "m2"), SurfaceIntent("append")
    )
    if compacted:
        shadowed = (first.seq, second.seq)
        log_event(
            session,
            "user/message",
            plugin_payload(
                "Earlier conversation, summarized.",
                "m-sum",
                plugin="compaction-summarize",
                form="compaction",
                summary="2 messages summarized",
            ),
            SurfaceIntent(surface_op=SurfaceReplace(replaces=shadowed), source_event_seqs=shadowed),
        )
    log_event(session, "user/message", user_payload(tail, "m3"), SurfaceIntent("append"))
    return session


async def test_a_summary_is_not_summarized_again(mount: MountProfile) -> None:
    """D9: `_plan` gives the reason, and this is the shape that reaches it.

    A retained tail that is one pair over `keep_fraction` keeps pressure up. The
    pre-step trigger therefore fires again, and the only thing above the cutoff
    is the paragraph that the previous call wrote.

    The control is the same session without that paragraph. It still plans, so
    this test shows that `_plan` declines the re-compaction, and does not
    decline whenever the budget is tight.
    """
    ctx = await mount(profile=PROFILE)
    engine = _engine(ctx)
    tail = "the live turn " * 400

    assert engine._plan(_summarized("recompaction", tail)) is None
    assert engine._plan(_summarized("ordinary", tail, compacted=False)) is not None
