"""ph-clm Phase 1: the section map, the edit door, and the context tools.

The gates (`plans/Ph_Clm_Context_Editing_Plan.md`, Phase 1): each verb lands as a
surface `replace`; the model's view shows the replacement and the person's
transcript the original; a resumed session derives the same context; every refusal
leaves the log as it was; the record lands in the edit's own batch; and the
agent-loop invariant holds across an edit made mid-step.
"""

from __future__ import annotations

from typing import Any

import pytest
from clm_helpers import (
    agent_in,
    answer_first_request,
    converse,
    edit,
    filler,
    human_text,
    model_text,
    mounted,
    sections_of,
)

from ph.json import as_obj, as_seq
from ph.keys import LLM_FAKE, SESSIONS, TOKEN_METER
from ph.llm.types import PluginSource, TokenUsage, create_user_message, text_of
from ph.session import (
    SurfaceIntent,
    balanced_cuts,
    derive_event_message,
    is_in_place_rewrite,
    is_replacement_surface_event,
    is_stand_in,
    substitute,
)
from ph.testing import (
    MountProfile,
    assert_fold_laws,
    assistant_payload,
    log_event,
    run_tool,
    user_payload,
)
from ph.testing.builders import reconciled_call, resume_stored
from ph.text import thousands
from ph.tools.definition import NotDone
from ph_clm.keys import CLM
from ph_clm.kinds import REVISED
from ph_clm.sections import EditRefused, SectionMap, label

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------- the map --


async def test_the_map_cuts_where_no_call_is_outstanding(mount: MountProfile) -> None:
    """A call and its result are one section; every other message is its own."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)

    sections = sections_of(ctx, agent.session)

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
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    before = [one.id for one in sections_of(ctx, agent.session)]

    log_event(
        agent.session, "user/message", user_payload("one more", "m9"), SurfaceIntent("append")
    )

    assert [one.id for one in sections_of(ctx, agent.session)][: len(before)] == before


async def test_the_section_map_is_a_fold_of_the_log(mount: MountProfile) -> None:
    """The per-event memo keeps `SessionFoldCache`'s contract, checked two ways over a
    log with revisions in it: the fold laws over every prefix, and a map kept across
    the edits agreeing with one built from nothing."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    kept = SectionMap(ctx.require(TOKEN_METER), protect_task=True)
    converse(agent.session)
    kept(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})
    await edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )
    log_event(agent.session, "user/message", user_payload("more", "m9"), SurfaceIntent("append"))

    assert kept(agent.session) == sections_of(ctx, agent.session)
    assert_fold_laws(agent.session, kept.fold, kept.extend)


async def test_the_step_in_flight_cannot_be_edited(mount: MountProfile) -> None:
    """A call still waiting for its result is the step making the edit."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    pending = {"type": "tool-call", "id": "c3", "name": "context_tombstone", "arguments": "{}"}
    log_event(
        agent.session,
        "assistant/message",
        assistant_payload("", "m9", content=[pending]),
        SurfaceIntent("append"),
    )

    last = sections_of(ctx, agent.session)[-1]
    logged = len(agent.session.events)
    refused = await edit(ctx, agent, "context_tombstone", {"sections": [last.name], "reason": "x"})

    assert last.protected == "it holds the step in flight"
    assert refused.is_error and "step in flight" in text_of(refused.content)
    assert len(agent.session.events) == logged


# ------------------------------------------------------------------ the verbs --


async def test_a_tombstone_shadows_the_run_and_keeps_the_log(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]
    logged = len(agent.session.events)

    result = await edit(
        ctx,
        agent,
        "context_tombstone",
        {"sections": [f"{step.name}..{reply.name}"], "reason": "read the whole file for one line"},
    )

    assert not result.is_error, result.content
    assert "noise from a long listing" not in model_text(agent.session)
    assert "read the whole file for one line" in model_text(agent.session)
    assert "noise from a long listing" in human_text(agent.session), "the transcript lost it"
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
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]

    result = await edit(
        ctx,
        agent,
        "context_replace",
        {"sections": [step.name, reply.name], "text": "Read parser.py: the bug is line 1."},
    )

    assert not result.is_error, result.content
    model = model_text(agent.session)
    assert "Read parser.py: the bug is line 1." in model
    assert f"standing for {step.name}..{reply.name}" in model
    assert "noise from a long listing" not in model
    revised = sections_of(ctx, agent.session)[1]
    assert revised.kind == "revised"
    assert set(revised.stands_for) == {*step.nodes, *reply.nodes}


async def test_a_rewrite_changes_only_the_passage(mount: MountProfile) -> None:
    """In place: the call id and the pairing stay, so nothing about the step moves."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step = sections_of(ctx, agent.session)[1]
    cuts = balanced_cuts(agent.session)
    logged = len(agent.session.events)

    result = await edit(
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
    assert "(listing cut)" in model_text(agent.session)
    assert "noise from a long listing" in human_text(agent.session)
    assert [one.id for one in sections_of(ctx, agent.session)][1] == step.id, (
        "an in-place rewrite moved the section"
    )


async def test_a_rewritten_reply_keeps_its_section_name(mount: MountProfile) -> None:
    """In place, through the door: the same message, so the same section — what the
    door keeps beyond that (the id, the usage left behind) is tested at the door."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    logged = len(agent.session.events)

    result = await edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )

    assert not result.is_error, result.content
    rewritten = agent.session.events[logged]
    assert rewritten.type == "assistant/message"
    assert is_in_place_rewrite(rewritten)
    assert sections_of(ctx, agent.session)[2].id == reply.id, "the rewrite renamed the section"


async def test_adjacent_tombstones_become_one_marker(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale read"})

    await edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "old note"})

    sections = sections_of(ctx, agent.session)
    markers = [one for one in sections if one.kind == "revised"]
    assert len(markers) == 1, [one.preview for one in sections]
    model = model_text(agent.session)
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
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    sections = sections_of(ctx, agent.session)
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

    result = await edit(ctx, agent, tool, filled)

    assert result.is_error
    assert says in text_of(result.content)
    assert len(agent.session.events) == logged, "a refused edit wrote to the log"


async def test_a_revised_id_names_what_stands_for_it(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})

    result = await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "x"})

    assert result.is_error
    marker = sections_of(ctx, agent.session)[1]
    assert f"{step.name} has been revised; {marker.name} stands for it now" in text_of(
        result.content
    )


async def test_a_tombstoned_message_gets_a_new_name(mount: MountProfile) -> None:
    """A substitution standing for one node is not an in-place rewrite: the marker is
    a new section, and the old name says what stands for it now."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    await edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "stale"})

    marker = sections_of(ctx, agent.session)[2]
    again = await edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "x"})

    assert marker.kind == "revised"
    assert marker.id != reply.id
    assert again.is_error and "has been revised" in text_of(again.content)


async def test_the_shrink_gate_refuses_growth(mount: MountProfile) -> None:
    ctx = await mounted(mount, gate="shrink")
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    logged = len(agent.session.events)

    result = await edit(
        ctx, agent, "context_replace", {"sections": [reply.name], "text": "longer " * 200}
    )

    assert result.is_error and "gate is `shrink`" in text_of(result.content)
    assert len(agent.session.events) == logged


async def test_with_no_gate_growth_is_taken_and_said(mount: MountProfile) -> None:
    """Decision 4: growth is the model's call, and the receipt names it."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]

    result = await edit(
        ctx, agent, "context_replace", {"sections": [reply.name], "text": "longer " * 200}
    )

    assert not result.is_error
    assert "GREW" in text_of(result.content)


# ------------------------------------------------------------ reading it back --


async def test_recall_reads_back_what_a_revision_stands_for(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_replace", {"sections": [step.name], "text": "read it"})
    marker = sections_of(ctx, agent.session)[1]

    whole = await edit(ctx, agent, "context_recall", {"section": marker.name, "max_tokens": 8000})
    cut = await edit(ctx, agent, "context_recall", {"section": marker.name, "max_tokens": 128})

    assert "noise from a long listing" in text_of(whole.content)
    assert "[call read" in text_of(whole.content)
    assert "cut at the token bound" in text_of(cut.content)


async def test_a_diff_shows_both_versions(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    await edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )

    result = await edit(ctx, agent, "context_diff", {"section": reply.name})

    diff = text_of(result.content)
    assert "-the bug is on line 1 of parser.py" in diff
    assert "+the bug is on parser.py:1" in diff


async def test_an_unrevised_section_has_nothing_to_recall(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]

    result = await edit(ctx, agent, "context_recall", {"section": reply.name})

    assert result.is_error and "has not been revised" in text_of(result.content)


# ------------------------------------------------------------------- restore --


async def test_a_restored_rewrite_is_the_text_before_it_exactly(mount: MountProfile) -> None:
    """In place and exact: the same message comes back, so the section keeps its id.
    Sabotage: restore through a substitution instead, and the id moves."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    before = agent.session.derive_messages()
    await edit(
        ctx,
        agent,
        "context_rewrite",
        {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"},
    )

    result = await edit(ctx, agent, "context_restore", {"section": reply.name})

    assert not result.is_error, result.content
    assert agent.session.derive_messages() == before
    assert sections_of(ctx, agent.session)[2].id == reply.id
    said = text_of(result.content)
    assert f"Restored {reply.name} to its text before its last edit" in said
    assert "GREW" not in said, "a restore was warned about for growing, which is its point"
    record = agent.session.events[-1]
    assert (record.type, record.data["verb"]) == (REVISED, "restore")


async def test_a_restored_tombstone_brings_its_run_back_as_text(mount: MountProfile) -> None:
    """Flattened, under a label, and not a stand-in: it is the conversation back, so
    compaction may summarize it again. Sabotage: land it as `compaction`, and it reads
    as a summary compaction declines to touch."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]
    shown = human_text(agent.session)
    await edit(
        ctx, agent, "context_tombstone", {"sections": [step.name, reply.name], "reason": "done"}
    )
    marker = sections_of(ctx, agent.session)[1]

    result = await edit(ctx, agent, "context_restore", {"section": marker.name})

    assert not result.is_error, result.content
    model = model_text(agent.session)
    assert f"[restored context, standing for {step.name}..{reply.name}]" in model
    assert "noise from a long listing" in model and "[call read" in model
    assert "removed from context" not in model
    restored = sections_of(ctx, agent.session)[1]
    event = agent.session.events[restored.nodes[0]]
    assert not is_stand_in(event) and restored.kind == "context"
    assert f"now {restored.name}" in text_of(result.content)
    assert human_text(agent.session) == shown, "a restore changed the person's transcript"


async def test_restoring_a_restore_brings_the_marker_back_exactly(mount: MountProfile) -> None:
    """One user-role message stood there, so it comes back as itself. Sabotage: drop
    the copy, and the marker returns as text under a second label."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "done"})
    marker = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_restore", {"section": marker.name})

    result = await edit(
        ctx, agent, "context_restore", {"section": sections_of(ctx, agent.session)[1].name}
    )

    assert not result.is_error, result.content
    first = derive_event_message(agent.session.events[marker.nodes[0]])
    again = derive_event_message(agent.session.events[sections_of(ctx, agent.session)[1].nodes[0]])
    assert first is not None and again is not None
    assert (again.content, again.source) == (first.content, first.source)


async def test_restoring_a_restore_redoes_the_edit_it_undid(mount: MountProfile) -> None:
    """A restore is an edit, and the latest one: restoring again toggles, it does not
    step further back. Sabotage: pass over restores when picking the edit, and the
    second restore finds nothing to undo."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    for old, new in (("line 1", "line one"), ("parser.py", "the parser")):
        await edit(ctx, agent, "context_rewrite", {"section": reply.name, "old": old, "new": new})

    await edit(ctx, agent, "context_restore", {"section": reply.name})
    once = model_text(agent.session)
    await edit(ctx, agent, "context_restore", {"section": reply.name})

    assert "the bug is on line one of parser.py" in once
    assert "the bug is on line one of the parser" in model_text(agent.session)


async def test_only_a_context_edit_is_restored(mount: MountProfile) -> None:
    """A section as written has nothing to undo, and one the harness revised — a
    compaction summary here — is not the model's edit to take back. Sabotage: restore
    any replacement, and the summary is unpicked."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]
    summary = create_user_message(
        content=[{"type": "text", "text": "the parser was read"}],
        source=PluginSource(plugin="compaction-summarize", form="compaction", summary="read"),
    )
    substitute(agent.session, step.nodes, summary)
    summarized = sections_of(ctx, agent.session)[1]
    logged = len(agent.session.events)

    harness = await edit(ctx, agent, "context_restore", {"section": summarized.name})
    written = await edit(ctx, agent, "context_restore", {"section": reply.name})

    assert harness.is_error and "revised by the harness" in text_of(harness.content)
    assert written.is_error and "has not been revised" in text_of(written.content)
    assert len(agent.session.events) == logged


async def test_the_shrink_gate_refuses_a_restore(mount: MountProfile) -> None:
    """A restore brings text back, which is growth: `shrink` refuses it, as the plan
    says it would (decision 4)."""
    ctx = await mounted(mount, gate="shrink")
    agent = agent_in(ctx)
    converse(agent.session)
    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "done"})
    marker = sections_of(ctx, agent.session)[1]
    logged = len(agent.session.events)

    result = await edit(ctx, agent, "context_restore", {"section": marker.name})

    assert result.is_error and "gate is `shrink`" in text_of(result.content)
    assert len(agent.session.events) == logged


# --------------------------------------------------------- durability, resume --


async def test_a_resumed_session_derives_the_same_context(mount: MountProfile) -> None:
    """The edits are core events: the store's fold rebuilds exactly this view."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step, reply = sections_of(ctx, agent.session)[1:3]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})
    await edit(
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
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    step = sections_of(ctx, agent.session)[1]
    arguments = {"sections": [step.name], "reason": "stale"}
    await run_tool(
        ctx, "context_tombstone", arguments, agent=agent, session=agent.session, call_id="call-1"
    )
    fresh = agent_in(ctx)
    converse(fresh.session)

    done = await reconciled_call(ctx, agent.session, "context_tombstone", arguments)
    not_done = await reconciled_call(ctx, fresh.session, "context_tombstone", arguments)

    assert isinstance(done, tuple) and "Removed" in text_of(list(done))
    assert isinstance(not_done, NotDone)


async def test_a_reconciled_restore_says_what_it_did(mount: MountProfile) -> None:
    """The record does not say whether a restore kept its section's id; the edit it
    describes does. Sabotage: leave `in_place` unset when reconciling, and a restored
    rewrite reads as a section that came back under a new id."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    reply = sections_of(ctx, agent.session)[2]
    rewrite = {"section": reply.name, "old": "line 1 of parser.py", "new": "parser.py:1"}
    await run_tool(
        ctx, "context_rewrite", rewrite, agent=agent, session=agent.session, call_id="call-0"
    )
    arguments = {"section": reply.name}
    await run_tool(
        ctx, "context_restore", arguments, agent=agent, session=agent.session, call_id="call-1"
    )

    done = await reconciled_call(ctx, agent.session, "context_restore", arguments)

    assert isinstance(done, tuple)
    assert f"Restored {reply.name} to its text before its last edit" in text_of(list(done))


# --------------------------------------------------------------- in the loop --


async def test_an_edit_made_mid_step_reaches_the_next_request(mount: MountProfile) -> None:
    """The whole path: the model calls the tool, the edit lands while its step is
    open, and the next request is built from the edited surface — which the
    `agent-loop-invariant` row checks on every request."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    session = agent.session
    converse(session)
    step, reply = sections_of(ctx, session)[1:3]
    answer_first_request(
        ctx,
        "context_replace",
        lambda: {"sections": [f"{step.name}..{reply.name}"], "text": "Bug: parser.py line 1."},
    )
    await agent.prompt("tidy your context")

    first, second = [one for one in ctx.require(LLM_FAKE).requests if one.is_loop_request]
    before = "\n".join(text_of(message.content) for message in first.messages)
    after = "\n".join(text_of(message.content) for message in second.messages)
    assert "the bug is on line 1" in before
    assert "Bug: parser.py line 1." in after
    assert "the bug is on line 1" not in after
    assert session.select(REVISED), "the edit left no record"


async def test_a_receipt_counts_the_context_as_compaction_does(mount: MountProfile) -> None:
    """The receipt's "Context ~X" is the meter's baseline — the provider's last count,
    system prompt included — so a receipt and a readout never disagree about the size
    of one context."""
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    usage = TokenUsage(input_tokens=5_000, output_tokens=10)
    reply = assistant_payload("noted", "m20", usage=usage)
    log_event(agent.session, "assistant/message", reply, SurfaceIntent("append"))
    before = ctx.require(TOKEN_METER).baseline(agent.session).tokens
    step = sections_of(ctx, agent.session)[1]

    result = await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "x"})

    (record,) = agent.session.select(REVISED)
    assert record.data["contextBefore"] == before
    assert f"Context ~{thousands(before)} →" in text_of(result.content)


async def test_fit_gates_a_batch_as_one_change(mount: MountProfile) -> None:
    """Two replacements in one batch that each fit the window can together overflow it:
    the batch is gated as the one change its receipt reports.

    Sabotage: gate each draft on its own again, and the batch lands.
    """
    ctx = await mounted(mount, gate="fit")
    agent = agent_in(ctx)
    converse(agent.session)
    editor = ctx.require(CLM)
    meter = ctx.require(TOKEN_METER)
    room = 1_000
    window = meter.baseline(agent.session).tokens + room
    log_event(
        agent.session,
        "request/context",
        {"provider": "fake", "model": "m", "contextWindow": window},
    )
    text = filler(meter, room * 6 // 10)
    sections = editor.sections(agent.session)
    drafts = [editor.replacing(sections, position, position + 1, text) for position in (2, 3)]

    with pytest.raises(EditRefused, match="window"):
        editor.land(agent.session, drafts, call_id=None)
    assert not agent.session.select(REVISED), "part of a refused batch landed"
    assert editor.land(agent.session, drafts[:1], call_id=None), "one alone should fit"
