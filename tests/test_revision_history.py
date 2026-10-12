"""Revision history across the packages: sizes in `/diff` (H2), and the export (H4).

**A repo-level scenario**, because it crosses three packages: the producers are
ph-stabilize's and ph-clm's, and the readers are the app's `/diff` and `phern
revisions`. Each producer hands the revision door its record and its size, and the door
writes the record after the replacement, stamped with it, so both readers ask the log
one way for every producer.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context
from ph.keys import AGENTS, COMMANDS, COMPACTION, SESSIONS, TOKEN_METER
from ph.seams.token_meter import TokenBaseline
from ph.session import Session, SurfaceIntent, is_replacement_surface_event, session_written
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    assistant_payload,
    log_event,
    run_tool,
    store_root,
    stored_log,
    tool_result_payload,
    user_payload,
)
from ph_app.cli import app
from ph_app.revisions import revision_rows
from ph_app.tui.diffs import build_diffs
from ph_clm.sections import SectionMap
from ph_stabilize import BUNDLE as STABILIZE
from ph_stabilize.compaction import MAX_ARG_LENGTH, SummarizeEngine
from ph_stabilize.input_offload import HUMAN_TOKEN_LIMIT_BEFORE_EVICT
from ph_stabilize.offload import NUM_CHARS_PER_TOKEN

pytestmark = pytest.mark.anyio

CLM_ROW: dict[str, Any] = {"id": "clm-context", "name": "clm-context"}

PASTE = "x" * (NUM_CHARS_PER_TOKEN * HUMAN_TOKEN_LIMIT_BEFORE_EVICT + 1)
"""One character past input-offload's limit."""


async def _mounted(mount: MountProfile) -> Context:
    return await mount(CLM_ROW, profile=[BASE, HEADLESS, STABILIZE])


def _engine(ctx: Context) -> SummarizeEngine:
    engine = ctx.require(COMPACTION).engine
    assert isinstance(engine, SummarizeEngine)
    return engine


def _sizes(session: Session) -> list[tuple[str, bool]]:
    """Each entry's recorded unit, and whether it says the revision shrank."""
    return [
        (entry.recorded.unit, entry.recorded.after < entry.recorded.before)
        for group in build_diffs(session)
        for entry in group.entries
        if entry.recorded is not None
    ]


def _step(session: Session, call: dict[str, Any], output: str | None) -> None:
    """An assistant turn making `call`, and its result when `output` is given."""
    log_event(
        session,
        "assistant/message",
        assistant_payload("", f"a{session.seq}", content=[call]),
        SurfaceIntent("append"),
    )
    if output is not None:
        log_event(
            session,
            "tool/result",
            tool_result_payload(output, f"r{session.seq}", call["id"]),
            SurfaceIntent("append"),
        )


def _read(call_id: str) -> dict[str, Any]:
    return {"type": "tool-call", "id": call_id, "name": "read", "arguments": '{"path": "a.md"}'}


def _stepped(ctx: Context, name: str, output: str) -> Session:
    """task · assistant(call) · tool/result printing `output`."""
    session = ctx.require(SESSIONS).create(name)
    log_event(session, "user/message", user_payload("read it", "m1"), SurfaceIntent("append"))
    _step(session, _read("c1"), output)
    return session


async def _tombstone(ctx: Context, session: Session, node: int, call_id: str) -> None:
    """A model's own tombstone of the section holding `node`, through the tool, as the
    call `call_id`."""
    sections = SectionMap(ctx.require(TOKEN_METER), protect_task=True)(session)
    (section,) = [one for one in sections if node in one.nodes]
    arguments = {"sections": [section.name], "reason": "stale"}
    done = await run_tool(
        ctx,
        "context_tombstone",
        arguments,
        agent=ctx.require(AGENTS).create(session, FAKE_OPTIONS),
        session=session,
        call_id=call_id,
    )
    assert not done.is_error, done.content


async def test_every_producer_s_revision_shows_the_size_it_recorded(mount: MountProfile) -> None:
    """H2's gate: compaction's summary in tokens, offload's clip and moved paste in
    bytes, a ph-clm edit in tokens. Sabotage: drop any producer's size, and its entry
    shows none."""
    ctx = await _mounted(mount)
    sessions, agents = ctx.require(SESSIONS), ctx.require(AGENTS)

    pasted = sessions.create("sizes-paste")
    await agents.create(pasted, FAKE_OPTIONS).prompt(PASTE)

    summarized = sessions.create("sizes-summary")
    asking = agents.create(summarized, FAKE_OPTIONS)
    for index in range(8):
        await asking.prompt(f"question {index} " + "detail " * 200)
    await ctx.require(COMMANDS).dispatch("/compact", session=summarized, agent=asking)

    clipped = _stepped(ctx, "sizes-clip", "0" * 30_000)
    assert await _engine(ctx).clip_overflow_tail(agents.create(clipped, FAKE_OPTIONS))

    edited = _stepped(ctx, "sizes-clm", "noise from a long listing\n" * 60)
    await _tombstone(ctx, edited, edited.surface.nodes[1], "edit-1")

    assert _sizes(pasted) == [("bytes", True)]
    assert _sizes(summarized) == [("tokens", True)]
    assert _sizes(clipped) == [("bytes", True)]
    assert _sizes(edited) == [("tokens", True)]


async def test_the_export_has_a_row_for_every_change_with_its_record(
    mount: MountProfile,
) -> None:
    """H4's gate: one session revised by every producer — a moved paste, elided
    arguments, a summary, a clip and a model's own edit — exports one row per change
    from the log on disk, each with its producer's record, and the edit's row with the
    call that asked for it. Sabotage: skip the rows of a record for several, or drop
    the records."""
    ctx = await _mounted(mount)
    session = ctx.require(SESSIONS).create("every-producer")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    engine = _engine(ctx)

    await agent.prompt(PASTE)
    body = json.dumps({"path": "notes.md", "content": "y" * (MAX_ARG_LENGTH + 1)})
    _step(session, {"type": "tool-call", "id": "w1", "name": "write", "arguments": body}, "ok")
    for index in range(4):
        await agent.prompt(f"question {index} " + "detail " * 200)
    window = TokenBaseline(tokens=8192, source="estimate", context_window=8192)
    assert engine.truncate_arguments(agent, session, "manual", window)
    await ctx.require(COMMANDS).dispatch("/compact", session=session, agent=agent)
    read = session.seq
    _step(session, _read("c9"), "0" * 30_000)
    assert await engine.clip_overflow_tail(agent)
    edit = {"type": "tool-call", "id": "edit-1", "name": "context_tombstone", "arguments": "{}"}
    _step(session, edit, None)
    await _tombstone(ctx, session, read, "edit-1")
    assert await session_written(ctx, session)

    result = CliRunner().invoke(app, ["revisions", str(stored_log(store_root(ctx), session.id))])

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in result.output.splitlines()]
    assert rows == json.loads(json.dumps(list(revision_rows(session))))
    changes = [event.seq for event in session.events if is_replacement_surface_event(event)]
    assert [row["seq"] for row in rows] == changes
    assert all(row["records"] for row in rows), "a change was exported without its record"
    assert {record["type"] for row in rows for record in row["records"]} == {
        "offload/input-spilled",
        "compaction/args-truncated",
        "compaction/summarized",
        "offload/spilled",
        "clm/revised",
    }
    (edited,) = [row for row in rows if row["records"][0]["type"] == "clm/revised"]
    assert [call["name"] for call in edited["calls"]] == ["context_tombstone"]
