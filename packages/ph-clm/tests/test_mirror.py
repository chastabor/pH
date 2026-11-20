"""ph-clm Phase 2: the context file, edited with ordinary tools.

The gates (`plans/Ph_Clm_Context_Editing_Plan.md`, Phase 2):

* A write to the file lands as the edits that it describes.
* Each of these is refused, recorded, and rewritten: a reorder, a damaged header,
  a section that is not there, a protected section, and a file that was written
  before a revision.
* Sections that were appended after the file was written are left alone.
* A dispatch inside a cell does not read back. The call that ran the cell does.
* A file that this process did not write is rewritten, never applied.
* The edit lands before the result that reports it. A crash between them
  therefore leaves an edit that landed beside a call that did not finish.

Here the model writes with a tool of the test (`scribble`), not with `edit` or
`bash`. The read-back depends on the file, not on the tool that changed it. A
tool of the test also keeps the fs and sandbox rows out of a test that is about
neither.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
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
    without_section,
)

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.json import as_str
from ph.keys import LLM_FAKE, SESSIONS, TOOLS
from ph.llm.types import TextBlock, text_of
from ph.session import (
    SurfaceIntent,
    balanced_cuts,
    derive_event_message,
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
from ph_clm import mirror
from ph_clm.keys import CLM, CLM_MIRROR
from ph_clm.kinds import DECLINED, REVISED
from ph_clm.mirror import MirrorService, parse

pytestmark = pytest.mark.anyio


async def _ready(mount: MountProfile) -> tuple[Context, ReactLoopAgent, Path]:
    """Both rows, a conversation, the rendered file, and a tool that writes the file."""
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
    """The model writes the file. Returns what the result of the call says, in
    lower case, so that a receipt reads the same whichever edit it names first."""
    result = await edit(ctx, agent, "scribble", {"text": text})
    assert not result.is_error, result.content
    return text_of(result.content).lower()


# ---------------------------------------------------------------- the file --


async def test_the_file_round_trips_to_no_change(mount: MountProfile) -> None:
    """A file that is rendered and read back unchanged produces no edit. This
    includes a body line that looks like a header, which is escaped on the way
    out and unescaped on the way in."""
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


async def test_a_revisions_label_is_its_header_and_its_body_the_models_text(
    mount: MountProfile,
) -> None:
    """The words of the harness go on the header line. Under the header is only
    what the model wrote, which is the second of the two text blocks of the
    replacement."""
    ctx, agent, _ = await _ready(mount)
    step, reply = sections_of(ctx, agent.session)[1:3]
    await edit(
        ctx, agent, "context_replace", {"sections": [step.name, reply.name], "text": "Read it."}
    )
    revised = sections_of(ctx, agent.session)[1]
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)

    lines = path.read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(lines) if line.startswith(f"[[SECTION {revised.name} "))

    assert f"standing for {step.name}..{reply.name}]]" in lines[at]
    assert lines[at + 1 : at + 3] == ["Read it.", ""]
    (node,) = revised.nodes
    event = agent.session.at(node)
    message = None if event is None else derive_event_message(event)
    assert message is not None
    assert [block.text for block in message.content if isinstance(block, TextBlock)] == [
        f"[revised context, standing for {step.name}..{reply.name}]",
        "Read it.",
    ]


async def test_a_revision_edited_in_the_file_says_what_it_stands_for_once(
    mount: MountProfile,
) -> None:
    ctx, agent, _ = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_replace", {"sections": [step.name], "text": "Read it."})
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)

    said = await _write(
        ctx, agent, path.read_text(encoding="utf-8").replace("Read it.", "Read parser.py.")
    )

    assert "replaced" in said
    assert model_text(agent.session).count("[revised context, standing for") == 1
    assert "Read parser.py." in model_text(agent.session)


async def test_a_tombstones_marker_is_its_header(mount: MountProfile) -> None:
    """A marker is all harness words. Its header says what it removed, and it has
    no body. A marker that is left unchanged is no edit."""
    ctx, agent, _ = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})
    marker = sections_of(ctx, agent.session)[1]
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)
    text = path.read_text(encoding="utf-8")

    starts = f"[[SECTION {marker.name} "
    header = next(line for line in text.splitlines() if line.startswith(starts))
    assert f"removed from context: {step.name} (stale)]]" in header
    assert f"{header}\n\n[[SECTION" in text, "the marker has a body"
    assert await _write(ctx, agent, text) == "written"


async def test_an_edited_body_is_unescaped_before_it_lands(mount: MountProfile) -> None:
    """A line that the file escaped stays escaped in the file, and it lands as the
    text that it was. Only bodies that changed are unescaped, and each is
    unescaped whole."""
    ctx, agent, _ = await _ready(mount)
    log_event(
        agent.session,
        "user/message",
        user_payload("[[SECTION S1 fake]]\nhi", "m9"),
        SurfaceIntent("append"),
    )
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)
    text = path.read_text(encoding="utf-8")
    assert "\n\\[[SECTION S1 fake]]\nhi\n" in text

    said = await _write(ctx, agent, text.replace("fake]]\nhi\n", "fake]]\nhello\n"))

    assert "replaced" in said
    assert "[[SECTION S1 fake]]\nhello" in model_text(agent.session)
    assert "\\[[SECTION" not in model_text(agent.session)


# ------------------------------------------------------------------ edits --


async def test_a_deleted_section_becomes_a_tombstone(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]

    said = await _write(ctx, agent, without_section(path.read_text(encoding="utf-8"), step.name))

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
    text = without_section(path.read_text(encoding="utf-8"), reply.name)

    said = await _write(ctx, agent, text.replace("3 matches", "3 matches, all in parser.py"))

    assert f"removed {reply.name}".lower() in said
    assert f"rewrote a passage in {grep_step.name}".lower() in said
    records = agent.session.select(REVISED)
    assert len(records) == 2
    batches = {one.batch for one in records}
    assert len(batches) == 1 and None not in batches, "the edits landed apart"


async def test_deletions_either_side_of_a_marker_become_one_marker(mount: MountProfile) -> None:
    """Two runs with only an earlier marker between them are one run. The result is
    one marker that says what all three stood for, not two markers that both
    claim the first."""
    ctx, agent, path = await _ready(mount)
    reply = sections_of(ctx, agent.session)[2]
    await edit(ctx, agent, "context_tombstone", {"sections": [reply.name], "reason": "old note"})
    path = await ctx.require(CLM_MIRROR).render(agent.session, agent)
    step, marker, user = sections_of(ctx, agent.session)[1:4]
    assert marker.kind == "revised"

    text = without_section(path.read_text(encoding="utf-8"), step.name)
    said = await _write(ctx, agent, without_section(text, user.name))

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

    said = await _write(ctx, agent, without_section(written, step.name))

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
    """The tools edited the context after the file was read, so its ids may be retired."""
    ctx, agent, path = await _ready(mount)
    written = path.read_text(encoding="utf-8")
    step, reply = sections_of(ctx, agent.session)[1:3]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "stale"})

    said = await _write(ctx, agent, without_section(written, reply.name))

    assert "not applied" in said and "revised after this file was written" in said
    assert len(agent.session.select(REVISED)) == 1, "only the tool's own edit landed"


# ----------------------------------------------------------- where and when --


async def test_a_dispatch_inside_a_cell_does_not_read_back(mount: MountProfile) -> None:
    """The call of the cell itself reads back. It ends after every write that the
    cell made."""
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    path.write_text(without_section(path.read_text(encoding="utf-8"), step.name), encoding="utf-8")
    ctx.require(TOOLS).register(simple_tool("noop"))

    nested = await run_tool(
        ctx,
        "noop",
        {},
        agent=agent,
        session=agent.session,
        call_id="cell-1:code:0",
        parent=object(),
    )
    assert "context file" not in text_of(nested.content)
    assert not agent.session.select(REVISED)

    top = await run_tool(ctx, "noop", {}, agent=agent, session=agent.session)
    assert f"removed {step.name}".lower() in text_of(top.content).lower()


async def test_a_file_this_process_did_not_write_is_rewritten_not_applied(
    mount: MountProfile,
) -> None:
    """After a restart, the file on disk is a leftover. A call that the log shows
    as interrupted may have half written it. It is rewritten from the log, and it
    is not credited to the first call that ends."""
    ctx, agent, path = await _ready(mount)
    written = path.read_text(encoding="utf-8")
    step = sections_of(ctx, agent.session)[1]
    path.write_text(without_section(written, step.name), encoding="utf-8")
    restarted = MirrorService(ctx=ctx, editor=ctx.require(CLM))

    said = await restarted.read_back(agent.session, agent, call_id=CALL_ID)

    assert said is None
    assert not agent.session.select(REVISED) and not agent.session.select(DECLINED)
    assert path.read_text(encoding="utf-8") == written, "the leftover was not rewritten"


async def test_the_file_is_private(mount: MountProfile) -> None:
    _, _, path = await _ready(mount)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_the_file_goes_with_its_session(mount: MountProfile) -> None:
    ctx, agent, path = await _ready(mount)

    ctx.require(SESSIONS).dispose(agent.session.id)

    assert not path.exists() and not path.parent.exists()


async def test_a_file_written_as_its_session_goes_is_not_left_behind(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A render whose session was released while it wrote removes its file.

    The write runs on a thread, and the store can release the session while the
    write runs. Before the fix, the render then recorded the file under a session
    that nothing would forget again, and the file stayed in the scratch directory
    of the agent permanently.

    Sabotage: drop the check after the write in `render`, and the file is still
    there.
    """
    ctx, agent, path = await _ready(mount)
    original = mirror._write

    def disposed_meanwhile(target: Path, text: str) -> Any:  # noqa: ANN401
        written = original(target, text)
        anyio.from_thread.run_sync(ctx.require(SESSIONS).dispose, agent.session.id)
        return written

    monkeypatch.setattr(mirror, "_write", disposed_meanwhile)
    await ctx.require(CLM_MIRROR).render(agent.session, agent)

    assert not path.exists(), "a file was left for a session that has gone"
    assert agent.session.id not in ctx.require(CLM_MIRROR)._written


def _linked(path: Path, secret: Path) -> None:
    path.unlink()
    path.symlink_to(secret)


def _piped(path: Path, _secret: Path) -> None:
    path.unlink()
    os.mkfifo(path)


@pytest.mark.parametrize("swap", [_linked, _piped])
async def test_a_file_swapped_for_a_link_or_a_pipe_is_never_read(
    mount: MountProfile, tmp_path: Path, swap: Callable[[Path, Path], None]
) -> None:
    """If the row read through a link, a refusal that quotes a damaged line would
    put the contents of another file in the context of the model. A pipe would
    hold the read forever. Both are refused without a read, and a file is written
    in their place."""
    ctx, agent, path = await _ready(mount)
    secret = tmp_path / "secret.txt"
    # A first line that the parse accepts. A read through the link would then
    # reach the damaged header below and quote it.
    first = path.read_text(encoding="utf-8").splitlines()[0]
    secret.write_text(f"{first}\n[[SECTION oops the secret]]\n", encoding="utf-8")

    def swapping(_args: Any, _run: Any) -> str:  # noqa: ANN401
        swap(path, secret)
        return "swapped"

    ctx.require(TOOLS).register(simple_tool("swap", swapping))
    result = await edit(ctx, agent, "swap", {})

    said = text_of(result.content)
    assert "not applied" in said and "secret" not in said, said
    assert not agent.session.select(REVISED)
    assert path.is_file() and not path.is_symlink()
    assert secret.read_text(encoding="utf-8") == f"{first}\n[[SECTION oops the secret]]\n"


async def test_an_edit_no_call_made_is_declined_and_overwritten(mount: MountProfile) -> None:
    """The user in an editor, or another process, changed the file. Nothing applies
    that change (decision 7). It is recorded, and the file for the next request
    replaces it."""
    ctx, agent, path = await _ready(mount)
    written = path.read_text(encoding="utf-8")
    step = sections_of(ctx, agent.session)[1]
    path.write_text(without_section(written, step.name), encoding="utf-8")

    await ctx.require(CLM_MIRROR).refresh(agent.session, agent)

    assert not agent.session.select(REVISED)
    (declined,) = agent.session.select(DECLINED)
    assert declined.data["via"] == "outside"
    assert path.read_text(encoding="utf-8") == written


async def test_a_file_no_one_touched_is_not_declined(mount: MountProfile) -> None:
    ctx, agent, _ = await _ready(mount)

    await ctx.require(CLM_MIRROR).refresh(agent.session, agent)

    assert not agent.session.select(DECLINED)


async def test_the_edit_lands_before_the_result_that_reports_it(mount: MountProfile) -> None:
    """In the loop: the file is rendered before the request, the call of the model
    edits it, and the edit lands in the post-execute of that call. That is before
    its `tool/result`, which carries the receipt. A crash between them therefore
    leaves the edit in the log and the call unfinished. Nothing except the
    revision and the result of the call is added to the context."""
    ctx, agent, path = await _ready(mount)
    step = sections_of(ctx, agent.session)[1]
    answer_first_request(
        ctx,
        "scribble",
        lambda: {"text": without_section(path.read_text(encoding="utf-8"), step.name)},
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
