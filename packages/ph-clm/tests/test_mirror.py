"""ph-clm Phase 2: the context file, edited with ordinary tools.

The gates (`plans/Ph_Clm_Context_Editing_Plan.md`, Phase 2): a write to the file
lands as the edits it describes; a reorder, a damaged header, a section that is not
there, a protected section and a file written before a revision are each refused,
recorded and rewritten; sections appended since the file was written are left
alone; a dispatch inside a cell does not read back, the call that ran the cell
does; and the edit lands before the result that reports it — so a crash between
them leaves an edit that landed beside a call that did not finish.

The model's write is a tool here (`scribble`) rather than `edit` or `bash`: the
read-back keys on the file, not on which tool touched it, and a tool of the test's
own keeps the fs and sandbox rows out of a test about neither.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from clm_helpers import (
    CALL_ID,
    MIRROR_ROW,
    OUTPUT,
    agent_in,
    answer_first_request,
    converse,
    edit,
    model_text,
    mounted,
    sections_of,
)

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import DEPLOYMENT, Context
from ph.json import as_str
from ph.keys import LLM_FAKE, TOOLS
from ph.llm.types import (
    text_of,
)
from ph.session import (
    SurfaceIntent,
    balanced_cuts,
    editable_message,
    is_in_place_rewrite,
    settle_of,
)
from ph.testing import (
    MountProfile,
    assistant_payload,
    log_event,
    result_text,
    run_tool,
    simple_tool,
    user_payload,
)
from ph.tools import ToolExecutionInput
from ph_clm.edits import REVISED
from ph_clm.keys import CLM_MIRROR
from ph_clm.mirror import DECLINED, parse

pytestmark = pytest.mark.anyio


async def _ready(mount: MountProfile) -> tuple[Context, ReactLoopAgent, Path]:
    """Both rows, a conversation, the file rendered, and a tool that writes it."""
    ctx = await mounted(mount, MIRROR_ROW)
    agent = agent_in(ctx)
    converse(agent.session)
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)

    def scribble(args: Any, _run: Any) -> str:  # noqa: ANN401
        path.write_text(as_str(args.get("text")), encoding="utf-8")
        return "written"

    ctx.require(TOOLS).register(simple_tool("scribble", scribble))
    return ctx, agent, path


async def _write(ctx: Context, agent: ReactLoopAgent, text: str) -> str:
    """The model writes the file; what the call's result says, folded to lower case so a
    receipt reads the same whichever edit it names first."""
    result = await edit(ctx, agent, "scribble", {"text": text})
    assert not result.is_error, result.content
    return text_of(result.content).lower()


def _without(text: str, name: str) -> str:
    """The file with one section's block cut out."""
    return re.sub(rf"\n\[\[SECTION {name} .*?(?=\n\[\[SECTION |\Z)", "", text, flags=re.S)


# ---------------------------------------------------------------- the file --


async def test_the_file_round_trips_to_no_change(mount: MountProfile) -> None:
    """Rendered and read back untouched, the file is nothing — including a body line
    that looks like a header, which is escaped on the way out and back on the way in."""
    ctx, agent, path = await _ready(mount)
    log_event(
        agent.session,
        "user/message",
        user_payload("[[SECTION S1 fake]]\nhi", "m9"),
        SurfaceIntent("append"),
    )
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)
    text = path.read_text(encoding="utf-8")
    logged = len(agent.session.events)

    said = await _write(ctx, agent, text)

    assert said == "written", "a file left as written produced a receipt"
    assert len(agent.session.events) == logged
    assert list(parse(text).blocks) == ctx.require(CLM_MIRROR).blocks(agent.session)


async def test_the_file_names_every_section_by_its_id(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)

    text = path.read_text(encoding="utf-8")

    assert text.startswith(f"[[LIVE_CONTEXT session={agent.session.id} generation=0 ")
    for section in sections_of(ctx, agent.session):
        assert f"[[SECTION {section.name} {section.kind}" in text
    assert "[[RESULT" in text and "protected" in text


# ------------------------------------------------------------------ edits --


async def test_a_deleted_section_becomes_a_tombstone(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]

    said = await _write(ctx, agent, _without(path.read_text(encoding="utf-8"), step.name))

    assert f"removed {step.name}".lower() in said
    assert "noise from a long listing" not in model_text(agent.session)
    (record,) = agent.session.select(REVISED)
    assert record.data["via"] == "mirror" and record.data["verb"] == "tombstone"
    assert f"[[SECTION {step.name} " not in path.read_text(encoding="utf-8"), "not rewritten"


async def test_a_changed_result_is_rewritten_in_place(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)
    cuts = balanced_cuts(agent.session)
    text = path.read_text(encoding="utf-8")

    said = await _write(
        ctx, agent, text.replace(OUTPUT.strip("\n"), "parser.py line 1 (listing cut)")
    )

    assert "rewrote" in said
    (rewritten,) = [one for one in agent.session.select("tool/result") if is_in_place_rewrite(one)]
    assert "(listing cut)" in model_text(agent.session)
    assert balanced_cuts(agent.session) == cuts, "the rewrite moved the call/result pairing"
    assert editable_message(rewritten)["content"][0]["toolCallId"] == "c1"


async def test_a_reply_written_above_the_results_keeps_its_calls(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    text = path.read_text(encoding="utf-8")
    header = next(line for line in text.splitlines() if line.startswith(f"[[SECTION {step.name} "))

    said = await _write(ctx, agent, text.replace(header, f"{header}\nI read parser.py first."))

    assert "rewrote" in said
    (reply,) = [
        one for one in agent.session.select("assistant/message") if is_in_place_rewrite(one)
    ]
    kinds = [block["type"] for block in editable_message(reply)["content"]]
    assert kinds == ["text", "tool-call"]


async def test_removing_a_steps_result_lines_replaces_it_with_the_text(
    mount: MountProfile,
) -> None:
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    text = path.read_text(encoding="utf-8")
    result_line = next(line for line in text.splitlines() if line.startswith("[[RESULT"))

    said = await _write(ctx, agent, text.replace(result_line + "\n", "", 1))

    assert f"replaced {step.name}".lower() in said
    revised = sections_of(ctx, agent.session)[1]
    assert revised.kind == "revised"
    assert "parser.py line 1" in model_text(agent.session)


async def test_one_write_lands_all_its_edits_together(mount: MountProfile) -> None:
    """A write that asks for several edits lands them in one batch: all or none."""
    ctx, agent, path = await _ready(mount)
    sections = sections_of(ctx, agent.session)
    reply, grep_step = sections[2], sections[4]
    text = _without(path.read_text(encoding="utf-8"), reply.name)

    said = await _write(ctx, agent, text.replace("3 matches", "3 matches, all in parser.py"))

    assert f"removed {reply.name}".lower() in said
    assert f"rewrote a passage in {grep_step.name}".lower() in said
    records = agent.session.select(REVISED)
    assert len(records) == 2
    batches = {one.batch for one in records}
    assert len(batches) == 1 and None not in batches, "the edits landed apart"


async def test_deletions_either_side_of_a_marker_become_one_marker(mount: MountProfile) -> None:
    """Two runs with only an earlier marker between them are one run: one marker, which
    says what all three stood for, rather than two markers both claiming the first."""
    ctx, agent, path = await _ready(mount)
    reply = sections_of(ctx, agent.session)[2]
    await edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "old note"})
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)
    step, marker, user = sections_of(ctx, agent.session)[1:4]
    assert marker.kind == "revised"

    said = await _write(
        ctx, agent, _without(_without(path.read_text(encoding="utf-8"), step.name), user.name)
    )

    assert "removed" in said
    kinds = [one.kind for one in sections_of(ctx, agent.session)]
    assert kinds.count("revised") == 1, kinds
    assert "old note" in model_text(agent.session), "the absorbed marker's reason was lost"


async def test_a_file_with_no_sections_replaces_everything_editable(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)
    first_line = path.read_text(encoding="utf-8").splitlines()[0]

    said = await _write(ctx, agent, f"{first_line}\nFixed the parser; the tests pass.\n")

    assert "replaced" in said
    kinds = [one.kind for one in sections_of(ctx, agent.session)]
    assert kinds == ["task", "revised"], "the task was protected; everything after it went"


async def test_sections_appended_since_the_file_was_written_are_left_alone(
    mount: MountProfile,
) -> None:
    ctx, agent, path = await _ready(mount)
    written = path.read_text(encoding="utf-8")
    step = sections_of(ctx, agent.session)[1]
    log_event(
        agent.session, "user/message", user_payload("one more", "m9"), SurfaceIntent("append")
    )
    log_event(
        agent.session,
        "assistant/message",
        assistant_payload("done", "m10"),
        SurfaceIntent("append"),
    )

    said = await _write(ctx, agent, _without(written, step.name))

    assert f"removed {step.name}".lower() in said
    assert "one more" in model_text(agent.session) and "done" in model_text(agent.session)


# --------------------------------------------------------------- refusals --


def _renamed(text: str) -> str:
    return re.sub(r"\[\[SECTION S\d+ step", "[[SECTION S9999 step", text, count=1)


def _strayed(text: str) -> str:
    return text.replace("\n[[SECTION", "\nstray words\n[[SECTION", 1)


def _reordered(text: str) -> str:
    blocks = re.split(r"(?=\n\[\[SECTION )", text)
    blocks[2], blocks[3] = blocks[3], blocks[2]
    return "".join(blocks)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        (_reordered, "cannot be reordered"),
        (lambda text: text.replace("[[RESULT S", "[[RESULT oops S", 1), "damaged header"),
        (_renamed, "is not a section"),
        (lambda text: text.replace("fix the parser", "fix everything", 1), "cannot be edited"),
        (lambda text: "\n".join(text.splitlines()[1:]), "first line"),
        (_strayed, "belongs to no section"),
    ],
)
async def test_a_refused_file_is_recorded_and_rewritten(
    mount: MountProfile, change: Callable[[str], str], says: str
) -> None:
    ctx, agent, path = await _ready(mount)
    text = path.read_text(encoding="utf-8")

    said = await _write(ctx, agent, change(text))

    assert "not applied" in said and says in said, said
    assert not agent.session.select(REVISED)
    (declined,) = agent.session.select(DECLINED)
    assert says in as_str(declined.data.get("reason"))
    assert path.read_text(encoding="utf-8") == text, "the refused file was not rewritten"


async def test_a_file_written_before_a_revision_is_refused(mount: MountProfile) -> None:
    """The tools edited the context after the file was read; its ids may be retired."""
    ctx, agent, path = await _ready(mount)
    written = path.read_text(encoding="utf-8")
    step, reply = sections_of(ctx, agent.session)[1:3]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})

    said = await _write(ctx, agent, _without(written, reply.name))

    assert "not applied" in said and "revised after this file was written" in said
    assert len(agent.session.select(REVISED)) == 1, "only the tool's own edit landed"


# ----------------------------------------------------------- where and when --


async def test_a_dispatch_inside_a_cell_does_not_read_back(mount: MountProfile) -> None:
    """The cell's own call does: it ends after every write the cell made."""
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    path.write_text(_without(path.read_text(encoding="utf-8"), step.name), encoding="utf-8")
    ctx.require(TOOLS).register(simple_tool("noop"))

    nested = await ctx.require(TOOLS).execute(
        ToolExecutionInput(
            call_id="cell-1:code:0",
            name="noop",
            arguments={},
            scope=DEPLOYMENT,
            session=agent.session,
            agent=agent,
            parent=object(),
        )
    )
    assert "context file" not in text_of(nested.content)
    assert not agent.session.select(REVISED)

    top = await run_tool(ctx, "noop", {}, agent=agent, session=agent.session)
    assert f"removed {step.name}".lower() in text_of(top.content).lower()


async def test_the_edit_lands_before_the_result_that_reports_it(mount: MountProfile) -> None:
    """In the loop: rendered before the request, edited by the model's call, landed in
    that call's post-execute — before its `tool/result`, which carries the receipt.
    So a crash between them leaves the edit in the log and the call unfinished. And
    nothing but the revision and the call's own result is added to the context."""
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    answer_first_request(
        ctx,
        "scribble",
        lambda: {"text": _without(path.read_text(encoding="utf-8"), step.name)},
    )
    await agent.prompt("tidy your context")

    (record,) = agent.session.select(REVISED)
    (result,) = [
        one for one in agent.session.select("tool/result") if settle_of(one) == (CALL_ID, False)
    ]
    assert record.seq < result.seq, "the receipt's result was logged before the edit landed"
    assert f"removed {step.name}".lower() in result_text(agent.session, CALL_ID).lower()
    added = [one for one in agent.session.select("user/message") if one.seq > record.seq]
    assert not added, "a notice was added to the context beside the edit and its result"
    first = next(one for one in ctx.require(LLM_FAKE).requests if one.is_loop_request)
    assert str(path) in (first.system or ""), "the prompt does not name the file"
