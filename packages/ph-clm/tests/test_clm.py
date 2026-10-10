"""ph-clm Phase 1: the section map, the edit door, and the five context tools.

The gates (`plans/Ph_Clm_Context_Editing_Plan.md`, Phase 1): each verb lands as a
surface `replace`; the model's view shows the replacement and the person's
transcript the original; a resumed session derives the same context; every refusal
leaves the log as it was; the record lands in the edit's own batch; and the
agent-loop invariant holds across an edit made mid-step.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.json import as_obj, as_seq
from ph.keys import AGENTS, LLM_FAKE, SESSIONS, TOKEN_METER
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    StreamChunk,
    ToolCallBlock,
    text_of,
)
from ph.session import (
    Session,
    SurfaceIntent,
    balanced_cuts,
    is_in_place_rewrite,
    is_replacement_surface_event,
)
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    assert_fold_laws,
    assistant_payload,
    log_event,
    run_tool,
    tool_result_payload,
    user_payload,
)
from ph.testing.builders import reconciled_call, resume_stored
from ph.tools.definition import NotDone, ToolExecutionResult
from ph_clm.edits import REVISED
from ph_clm.sections import Section, SectionMap, label, render_messages

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "clm-context", "name": "clm-context"}

OUTPUT = "parser.py line 1\n" + "noise from a long listing\n" * 60
"""A tool output long enough that removing it is worth something."""

_SEQ = itertools.count()


async def _mounted(mount: MountProfile, **config: object) -> Context:
    return await mount({**ROW, "config": config} if config else ROW)


def _agent(ctx: Context) -> ReactLoopAgent:
    agent = ctx.require(AGENTS).create(
        ctx.require(SESSIONS).create(f"clm-{next(_SEQ)}"), FAKE_OPTIONS
    )
    assert isinstance(agent, ReactLoopAgent)
    return agent


def _converse(session: Session) -> None:
    """task · step(read) · reply · user · step(grep) · reply."""
    log_event(
        session, "user/message", user_payload("fix the parser", "m1"), SurfaceIntent("append")
    )
    call = {"type": "tool-call", "id": "c1", "name": "read", "arguments": '{"path": "parser.py"}'}
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=[call]),
        SurfaceIntent("append"),
    )
    log_event(
        session, "tool/result", tool_result_payload(OUTPUT, "m3", "c1"), SurfaceIntent("append")
    )
    log_event(
        session,
        "assistant/message",
        assistant_payload("the bug is on line 1 of parser.py", "m4"),
        SurfaceIntent("append"),
    )
    log_event(
        session, "user/message", user_payload("now run the tests", "m5"), SurfaceIntent("append")
    )
    grep = {"type": "tool-call", "id": "c2", "name": "grep", "arguments": '{"pattern": "def"}'}
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m6", content=[grep]),
        SurfaceIntent("append"),
    )
    log_event(
        session,
        "tool/result",
        tool_result_payload("3 matches", "m7", "c2"),
        SurfaceIntent("append"),
    )
    log_event(
        session, "assistant/message", assistant_payload("all green", "m8"), SurfaceIntent("append")
    )


def _sections(ctx: Context, session: Session) -> tuple[Section, ...]:
    return SectionMap(ctx.require(TOKEN_METER), protect_task=True)(session)


def _model_text(session: Session) -> str:
    """What the model is sent, tool output included."""
    return render_messages(session.derive_messages())


def _human_text(session: Session) -> str:
    return render_messages(session.transcript())


async def _edit(
    ctx: Context, agent: ReactLoopAgent, name: str, arguments: dict[str, object]
) -> ToolExecutionResult:
    return await run_tool(ctx, name, arguments, agent=agent, session=agent.session)


# ------------------------------------------------------------------- the map --


async def test_the_map_cuts_where_no_call_is_outstanding(mount: MountProfile) -> None:
    """A call and its result are one section; every other message is its own."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)

    sections = _sections(ctx, agent.session)

    assert [one.kind for one in sections] == [
        "task",
        "step",
        "assistant",
        "user",
        "step",
        "assistant",
    ]
    assert len(sections[1].nodes) == 2, "the read and its result were split"
    assert sections[1].calls == ("read",)
    assert sections[0].protected is not None, "the task was editable"
    assert sections[-1].after == 0
    assert [one.after for one in sections] == sorted((one.after for one in sections), reverse=True)
    assert sections[1].after == sum(one.tokens for one in sections[2:])


async def test_a_section_keeps_its_id_as_the_log_grows(mount: MountProfile) -> None:
    """The ids are seqs (A1): appending moves nothing the model already read."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    before = [one.id for one in _sections(ctx, agent.session)]

    log_event(
        agent.session, "user/message", user_payload("one more", "m9"), SurfaceIntent("append")
    )

    assert [one.id for one in _sections(ctx, agent.session)][: len(before)] == before


async def test_the_section_map_is_a_fold_of_the_log(mount: MountProfile) -> None:
    """The per-event memo keeps `SessionFoldCache`'s contract, checked two ways over a
    log with revisions in it: the fold laws over every prefix, and a map kept across
    the edits agreeing with one built from nothing."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    kept = SectionMap(ctx.require(TOKEN_METER), protect_task=True)
    _converse(agent.session)
    kept(agent.session)
    step, reply = _sections(ctx, agent.session)[1:3]
    await _edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})
    await _edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )
    log_event(agent.session, "user/message", user_payload("more", "m9"), SurfaceIntent("append"))

    assert kept(agent.session) == _sections(ctx, agent.session)
    assert_fold_laws(agent.session, kept.fold, kept.extend)


async def test_the_step_in_flight_cannot_be_edited(mount: MountProfile) -> None:
    """A call still waiting for its result is the step making the edit."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    pending = {"type": "tool-call", "id": "c3", "name": "context_tombstone", "arguments": "{}"}
    log_event(
        agent.session,
        "assistant/message",
        assistant_payload("", "m9", content=[pending]),
        SurfaceIntent("append"),
    )

    last = _sections(ctx, agent.session)[-1]
    logged = len(agent.session.events)
    refused = await _edit(ctx, agent, "context_tombstone", {"sections": [last.name], "reason": "x"})

    assert last.protected == "it holds the step in flight"
    assert refused.is_error and "step in flight" in text_of(refused.content)
    assert len(agent.session.events) == logged


# ------------------------------------------------------------------ the verbs --


async def test_a_tombstone_shadows_the_run_and_keeps_the_log(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step, reply = _sections(ctx, agent.session)[1:3]
    logged = len(agent.session.events)

    result = await _edit(
        ctx,
        agent,
        "context_tombstone",
        {"sections": [f"{step.name}..{reply.name}"], "reason": "read the whole file for one line"},
    )

    assert not result.is_error, result.content
    assert "noise from a long listing" not in _model_text(agent.session)
    assert "read the whole file for one line" in _model_text(agent.session)
    assert "noise from a long listing" in _human_text(agent.session), "the transcript lost it"
    replacement, record = agent.session.events[logged:]
    assert is_replacement_surface_event(replacement)
    assert as_obj(replacement.data.get("source")).get("form") == "compaction"
    assert record.type == REVISED
    assert replacement.batch is not None and replacement.batch == record.batch, (
        "the record and the edit it describes landed apart"
    )
    assert tuple(as_seq(record.data.get("shadowed"))) == (*step.nodes, *reply.nodes)
    assert agent.session.stale() == []


async def test_a_replace_lands_the_text_the_model_wrote(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step, reply = _sections(ctx, agent.session)[1:3]

    result = await _edit(
        ctx,
        agent,
        "context_replace",
        {"sections": [step.name, reply.name], "text": "Read parser.py: the bug is line 1."},
    )

    assert not result.is_error, result.content
    model = _model_text(agent.session)
    assert "Read parser.py: the bug is line 1." in model
    assert f"standing for {step.name}..{reply.name}" in model
    assert "noise from a long listing" not in model
    revised = _sections(ctx, agent.session)[1]
    assert revised.kind == "revised"
    assert set(revised.stands_for) == {*step.nodes, *reply.nodes}


async def test_a_rewrite_changes_only_the_passage(mount: MountProfile) -> None:
    """In place: the call id and the pairing stay, so nothing about the step moves."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step = _sections(ctx, agent.session)[1]
    cuts = balanced_cuts(agent.session)
    logged = len(agent.session.events)

    result = await _edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": step.name, "old": "noise from a long listing\n" * 60, "new": "(listing cut)\n"},
    )

    assert not result.is_error, result.content
    rewritten, record = agent.session.events[logged:]
    assert rewritten.type == "tool/result"
    assert is_in_place_rewrite(rewritten)
    assert rewritten.batch is not None and rewritten.batch == record.batch
    assert balanced_cuts(agent.session) == cuts
    assert "(listing cut)" in _model_text(agent.session)
    assert "noise from a long listing" in _human_text(agent.session)
    assert [one.id for one in _sections(ctx, agent.session)][1] == step.id, (
        "an in-place rewrite moved the section"
    )


async def test_a_rewritten_reply_keeps_its_section_name(mount: MountProfile) -> None:
    """In place, through the door: the same message, so the same section — what the
    door keeps beyond that (the id, the usage left behind) is tested at the door."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    reply = _sections(ctx, agent.session)[2]
    logged = len(agent.session.events)

    result = await _edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )

    assert not result.is_error, result.content
    rewritten = agent.session.events[logged]
    assert rewritten.type == "assistant/message"
    assert is_in_place_rewrite(rewritten)
    assert _sections(ctx, agent.session)[2].id == reply.id, "the rewrite renamed the section"


async def test_adjacent_tombstones_become_one_marker(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step, reply = _sections(ctx, agent.session)[1:3]
    await _edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale read"})

    await _edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "old note"})

    sections = _sections(ctx, agent.session)
    markers = [one for one in sections if one.kind == "revised"]
    assert len(markers) == 1, [one.preview for one in sections]
    model = _model_text(agent.session)
    assert "stale read" in model and "old note" in model


# --------------------------------------------------------------- the refusals --


@pytest.mark.parametrize(
    ("tool", "arguments", "says"),
    [
        ("context_tombstone", {"sections": ["<task>"], "reason": "x"}, "it is the task"),
        ("context_tombstone", {"sections": ["<step>", "<user>"], "reason": "x"}, "unbroken run"),
        ("context_tombstone", {"sections": ["S99999"], "reason": "x"}, "not a section"),
        ("context_tombstone", {"sections": ["<inside>"], "reason": "x"}, "is inside"),
        ("context_replace", {"sections": ["<step>"], "text": "   "}, "needs text"),
        ("context_rewrite", {"section": "<step>", "old": "absent", "new": "y"}, "does not occur"),
        ("context_rewrite", {"section": "<step>", "old": "noise", "new": "y"}, "times in"),
    ],
)
async def test_a_refused_edit_leaves_the_log_as_it_was(
    mount: MountProfile, tool: str, arguments: dict[str, Any], says: str
) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    sections = _sections(ctx, agent.session)
    names = {
        "<task>": sections[0].name,
        "<step>": sections[1].name,
        "<user>": sections[3].name,
        "<inside>": label(sections[1].nodes[1]),
    }

    def fill(value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, list):
            return [fill(one) for one in value]
        return names.get(value, value) if isinstance(value, str) else value

    filled = {key: fill(value) for key, value in arguments.items()}
    logged = len(agent.session.events)

    result = await _edit(ctx, agent, tool, filled)

    assert result.is_error
    assert says in text_of(result.content)
    assert len(agent.session.events) == logged, "a refused edit wrote to the log"


async def test_a_revised_id_names_what_stands_for_it(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step = _sections(ctx, agent.session)[1]
    await _edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})

    result = await _edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "x"})

    assert result.is_error
    marker = _sections(ctx, agent.session)[1]
    assert f"{step.name} has been revised; {marker.name} stands for it now" in text_of(
        result.content
    )


async def test_a_tombstoned_message_gets_a_new_name(mount: MountProfile) -> None:
    """A substitution standing for one node is not an in-place rewrite: the marker is
    a new section, and the old name says what stands for it now."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    reply = _sections(ctx, agent.session)[2]
    await _edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "stale"})

    marker = _sections(ctx, agent.session)[2]
    again = await _edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "x"})

    assert marker.kind == "revised"
    assert marker.id != reply.id
    assert again.is_error and "has been revised" in text_of(again.content)


async def test_the_shrink_gate_refuses_growth(mount: MountProfile) -> None:
    ctx = await _mounted(mount, gate="shrink")
    agent = _agent(ctx)
    _converse(agent.session)
    reply = _sections(ctx, agent.session)[2]
    logged = len(agent.session.events)

    result = await _edit(
        ctx, agent, "context_replace", {"sections": [reply.name], "text": "longer " * 200}
    )

    assert result.is_error and "gate is `shrink`" in text_of(result.content)
    assert len(agent.session.events) == logged


async def test_with_no_gate_growth_is_taken_and_said(mount: MountProfile) -> None:
    """Decision 4: growth is the model's call, and the receipt names it."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    reply = _sections(ctx, agent.session)[2]

    result = await _edit(
        ctx, agent, "context_replace", {"sections": [reply.name], "text": "longer " * 200}
    )

    assert not result.is_error
    assert "GREW" in text_of(result.content)


# ------------------------------------------------------------ reading it back --


async def test_recall_reads_back_what_a_revision_stands_for(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step = _sections(ctx, agent.session)[1]
    await _edit(ctx, agent, "context_replace", {"sections": [step.name], "text": "read it"})
    marker = _sections(ctx, agent.session)[1]

    whole = await _edit(ctx, agent, "context_recall", {"section": marker.name, "max_tokens": 8000})
    cut = await _edit(ctx, agent, "context_recall", {"section": marker.name, "max_tokens": 128})

    assert "noise from a long listing" in text_of(whole.content)
    assert "[call read" in text_of(whole.content)
    assert "cut at the token bound" in text_of(cut.content)


async def test_a_diff_shows_both_versions(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    reply = _sections(ctx, agent.session)[2]
    await _edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )

    result = await _edit(ctx, agent, "context_diff", {"section": reply.name})

    diff = text_of(result.content)
    assert "-the bug is on line 1 of parser.py" in diff
    assert "+the bug is on parser.py:1" in diff


async def test_an_unrevised_section_has_nothing_to_recall(mount: MountProfile) -> None:
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    reply = _sections(ctx, agent.session)[2]

    result = await _edit(ctx, agent, "context_recall", {"section": reply.name})

    assert result.is_error and "has not been revised" in text_of(result.content)


# --------------------------------------------------------- durability, resume --


async def test_a_resumed_session_derives_the_same_context(mount: MountProfile) -> None:
    """The edits are core events: the store's fold rebuilds exactly this view."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step, reply = _sections(ctx, agent.session)[1:3]
    await _edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})
    await _edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )
    await ctx.require(SESSIONS).flush(agent.session)
    expected = agent.session.derive_messages()
    ctx.require(SESSIONS).dispose(agent.session.id)

    resumed = await resume_stored(ctx, agent.session.id)

    assert resumed.derive_messages() == expected


async def test_a_crash_after_an_edit_reconciles_as_done(mount: MountProfile) -> None:
    """The record is in the edit's batch, so finding it is finding the edit."""
    ctx = await _mounted(mount)
    agent = _agent(ctx)
    _converse(agent.session)
    step = _sections(ctx, agent.session)[1]
    arguments = {"sections": [step.name], "reason": "stale"}
    await run_tool(
        ctx, "context_tombstone", arguments, agent=agent, session=agent.session, call_id="call-1"
    )
    fresh = _agent(ctx)
    _converse(fresh.session)

    done = await reconciled_call(ctx, agent.session, "context_tombstone", arguments)
    not_done = await reconciled_call(ctx, fresh.session, "context_tombstone", arguments)

    assert isinstance(done, tuple) and "Removed" in text_of(list(done))
    assert isinstance(not_done, NotDone)


# --------------------------------------------------------------- in the loop --


def _calling(name: str, arguments: dict[str, Any]) -> AsyncIterator[StreamChunk]:
    async def stream() -> AsyncIterator[StreamChunk]:
        yield BlockStart(index=0, block_type="tool-call")
        yield BlockEnd(
            index=0, block=ToolCallBlock(id="t1", name=name, arguments=json.dumps(arguments))
        )
        yield Finish(reason=FinishReason(kind="tool-calls"))

    return stream()


async def test_an_edit_made_mid_step_reaches_the_next_request(mount: MountProfile) -> None:
    """The whole path: the model calls the tool, the edit lands while its step is
    open, and the next request is built from the edited surface — which the
    `agent-loop-invariant` row checks on every request."""
    ctx = await _mounted(mount)
    session = ctx.require(SESSIONS).create(f"clm-{next(_SEQ)}")
    _converse(session)
    step, reply = _sections(ctx, session)[1:3]
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    assert isinstance(agent, ReactLoopAgent)
    asked = {"once": False}

    async def edit_first(options: GenerateOptions, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        if options.is_loop_request and not asked["once"]:
            asked["once"] = True
            ctx.require(LLM_FAKE).requests.append(options)
            return _calling(
                "context_replace",
                {"sections": [f"{step.name}..{reply.name}"], "text": "Bug: parser.py line 1."},
            )
        return await next_(options)

    ctx.on("llm/stream", edit_first)
    await agent.prompt("tidy your context")

    first, second = [one for one in ctx.require(LLM_FAKE).requests if one.is_loop_request]
    before = "\n".join(text_of(message.content) for message in first.messages)
    after = "\n".join(text_of(message.content) for message in second.messages)
    assert "the bug is on line 1" in before
    assert "Bug: parser.py line 1." in after
    assert "the bug is on line 1" not in after
    assert session.select(REVISED), "the edit left no record"
