"""ph-clm Phase 3's gate, live: a model on a local server edits its own context.

**Opt-in by `LLAMA_BASE_URL`**, like `test_local_server.py`, and run with it by
`./test.sh smoke`. What the fake adapter cannot show:

* **that a readout reaches a real model at its share of a real window.** The window
  is the slot's, probed from `/props`; the count is the provider's, system prompt and
  tool schema included (`TokenMeter.baseline`). A result that takes the context past
  half of it ends with the readout line, and nothing else is added to the context.
* **that a model edits its context when told how** — once through the context tools
  from a cell, once by editing the context file in a cell — each landing as a
  `clm/revised` before the call's result.

**The edits are model-following assertions**, the bargain
`test_a_tool_call_comes_back_parsed` strikes: the prompt names the tool, the section
and the file, and a model that will not take that instruction is not one to run an
agent on. The readout is not: once the model runs the cell, whether the line appears
is the harness's alone.

**Sized for a small slot** (16k on the box this was written against). History is
seeded by hand rather than earned turn by turn, sized from the window the route probed
and the count the first reply reported. The readout test has to fill half the window,
so on a slot past `LARGEST_WINDOW` it skips rather than prefill a hundred thousand
tokens; the edit tests use a fixed-size listing whatever the slot.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from clm_helpers import context_tokens, filler, model_text, readout_share
from local_server import LIVE, cells_after, live_agent, output_of

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.keys import LLM, TOKEN_METER
from ph.session import SurfaceIntent
from ph.testing import (
    MountProfile,
    assistant_payload,
    log_event,
    tool_result_payload,
    user_payload,
)
from ph_clm.keys import CLM, CLM_MIRROR
from ph_clm.kinds import DECLINED, REVISED
from ph_clm.sections import Section
from ph_rlm.presentation import PYTHON

pytestmark = [pytest.mark.anyio, LIVE]

PROSE = (
    "The review walked the parser module function by function. The tokenizer keeps "
    "its position in a cursor, the expression reader recurses on parentheses, and "
    "the error path reports the line where the stream ran out. "
)
LINE = "the quick brown fox jumps over the lazy dog and keeps on running\n"
"""Plain words, because a tokenizer that splits digits one by one would make a listing
of numbers several times the size its characters suggest."""

NOISE = "a directory listing nobody needs any more: build output, caches and logs\n"

LARGEST_WINDOW = 65_536
"""The readout test fills about 60% of the window; past this, it skips rather than
make the server prefill a hundred thousand tokens for one assertion."""


def _window(ctx: Context, route: str) -> int:
    window = ctx.require(LLM).resolve_model("llama", route).context_window
    assert window, "the llama route published no context window; nothing to take a share of"
    return window


def _seed(ctx: Context, agent: ReactLoopAgent, tokens: int) -> None:
    """Earlier exchanges of plain prose, about `tokens` long by the meter's estimate."""
    meter = ctx.require(TOKEN_METER)
    added, part = 0, 0
    while added < tokens:
        ask, said = f"What did part {part} of the review find?", f"Part {part}: {PROSE * 4}"
        log_event(
            agent.session, "user/message", user_payload(ask, f"seed-u{part}"), SurfaceIntent()
        )
        log_event(
            agent.session,
            "assistant/message",
            assistant_payload(said, f"seed-a{part}"),
            SurfaceIntent(),
        )
        added += meter.measure_text(ask) + meter.measure_text(said)
        part += 1


def _noisy(ctx: Context, agent: ReactLoopAgent) -> Section:
    """A task, then a cell that printed a listing nobody needs — the section to remove."""
    log_event(
        agent.session, "user/message", user_payload("Tidy up the parser.", "task"), SurfaceIntent()
    )
    program = '{"program": "import os\\nprint(os.listdir())"}'
    call = {"type": "tool-call", "id": "noise-1", "name": PYTHON, "arguments": program}
    log_event(
        agent.session,
        "assistant/message",
        assistant_payload("", "noise-a", content=[call]),
        SurfaceIntent(),
    )
    log_event(
        agent.session,
        "tool/result",
        tool_result_payload(NOISE * 40, "noise-r", "noise-1"),
        SurfaceIntent(),
    )
    log_event(
        agent.session,
        "assistant/message",
        assistant_payload("That listing was not useful.", "noise-b"),
        SurfaceIntent(),
    )
    return next(one for one in ctx.require(CLM).sections(agent.session) if one.kind == "step")


def _declined(agent: ReactLoopAgent) -> list[str]:
    return [str(event.data.get("reason")) for event in agent.session.select(DECLINED)]


async def test_a_result_past_half_the_window_ends_with_a_readout(
    mount: MountProfile, route: str, tmp_path: Path
) -> None:
    """One reply to learn what the provider counts, history to just under half the
    window, then a cell whose output takes the context past it: that result ends
    with the readout — naming the context file — and nothing else is added."""
    ctx, agent = await live_agent(mount, route, "rlm-clm", "clm-live-readout")
    window = _window(ctx, route)
    if window > LARGEST_WINDOW:
        pytest.skip(f"a {window:,}-token slot; this test prefills about 60% of it")
    await agent.prompt("Reply with the single word: ok")
    _seed(ctx, agent, int(window * 0.38) - context_tokens(ctx, agent.session))
    listing = filler(ctx.require(TOKEN_METER), int(window * 0.22), LINE)
    (tmp_path / "listing.txt").write_text(listing, encoding="utf-8")
    asked = len(agent.session.events)

    await agent.prompt(
        "Run one python cell: print(open('listing.txt').read()). "
        "Then reply with the single word: done."
    )

    cells = cells_after(agent.session, asked)
    # The first readout of the turn, on whichever cell crossed: a model may look for
    # the file in a cell of its own before it prints it.
    outputs = [(cell, output_of(agent.session, cell)) for cell in cells]
    readouts = [
        (cell, line)
        for cell, said in outputs
        for line in said.splitlines()
        if line.startswith("[context: ")
    ]
    assert readouts, (
        f"no readout on any of {len(cells)} results; the context reached "
        f"~{context_tokens(ctx, agent.session):,} of {window:,} tokens. Last result's "
        f"tail: {outputs[-1][1][-400:]!r}"
    )
    crossing, readout = readouts[0]
    assert 50 <= readout_share(readout) < 75, readout
    assert str(ctx.require(CLM_MIRROR).path(agent.session, agent)) in readout
    # The readout rides on the result: no message is *appended* after the cell. A
    # compaction later in the turn — the model reading on past the readout — lands
    # as a replacement, which is compaction's doing and not a notice.
    appended = [
        event
        for event in agent.session.select("user/message")
        if event.seq > cells[0].seq and event.surface_op == "append"
    ]
    assert not appended, "a notice was added to the context beside the readout"
    early = [e for e in agent.session.select("compaction/summarized") if e.seq < crossing.seq]
    assert not early, "compaction ran before the result that carried the readout"


async def test_the_model_removes_a_section_with_the_context_tools(
    mount: MountProfile, route: str
) -> None:
    """Told which tool, which section and why, from a cell: the edit lands as a
    tombstone through the tools, and the listing leaves what the model is sent."""
    ctx, agent = await live_agent(mount, route, "rlm-clm", "clm-live-tools")
    step = _noisy(ctx, agent)
    asked = len(agent.session.events)

    await agent.prompt(
        "In one python cell, run: print(await tools.context_tombstone("
        f"sections=['{step.name}'], reason='no longer needed')). "
        "Then reply with the single word: done."
    )

    cells_after(agent.session, asked)
    revisions = [one for one in agent.session.select(REVISED) if one.seq > asked]
    assert revisions, f"no edit landed; declined: {_declined(agent)}"
    assert revisions[0].data["via"] == "tool" and revisions[0].data["verb"] == "tombstone"
    assert revisions[0].data["first"] == step.id
    assert NOISE not in model_text(agent.session), "the listing is still in what the model sees"


async def test_the_model_removes_a_section_by_editing_the_context_file(
    mount: MountProfile, route: str
) -> None:
    """Told where the file is and which block to delete, in a cell: the cell's own call
    reads it back, the edit lands through the mirror, and that call's result says so."""
    ctx, agent = await live_agent(mount, route, "rlm-clm", "clm-live-file")
    step = _noisy(ctx, agent)
    path = ctx.require(CLM_MIRROR).path(agent.session, agent)
    asked = len(agent.session.events)

    await agent.prompt(
        f"Your context is written to the file {path}. In one python cell, read that "
        f"file, delete the block for section {step.name} — its line starting "
        f"'[[SECTION {step.name} ' and every line after it up to the next line starting "
        "'[[SECTION ' — and write the file back. Then reply with the single word: done."
    )

    cells = cells_after(agent.session, asked)
    revisions = [one for one in agent.session.select(REVISED) if one.seq > asked]
    assert revisions, f"no edit landed; declined: {_declined(agent)}"
    assert revisions[0].data["via"] == "mirror"
    applied = [cell for cell in cells if "[context file applied." in output_of(agent.session, cell)]
    assert applied, "no cell's result carried the receipt"
    assert NOISE not in model_text(agent.session), "the listing is still in what the model sees"
