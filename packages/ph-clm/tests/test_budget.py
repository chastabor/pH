"""ph-clm Phase 3: context readouts on tool results.

The gates (`plans/Ph_Clm_Context_Editing_Plan.md`, Phase 3): a result that takes the
context past a share of its window ends with one readout line, and nothing else is
added to the context; the next result past the same share says nothing, and once an
edit brings the context back down, crossing it again says so again; the last share
says it is the last; the readout names the context file when a row keeps one, and
measures after that file's edit has landed; with no window there is no readout.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from clm_helpers import MIRROR_ROW, agent_in, converse, edit, mounted, sections_of

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.json import as_str
from ph.keys import TOKEN_METER, TOOLS
from ph.llm.types import TokenUsage, text_of
from ph.session import SurfaceIntent
from ph.testing import MountProfile, assistant_payload, log_event, run_tool, simple_tool
from ph_clm.keys import CLM_MIRROR

pytestmark = pytest.mark.anyio


def _tokens(ctx: Context, agent: ReactLoopAgent) -> int:
    """The context as the readout measures it."""
    return ctx.require(TOKEN_METER).baseline(agent.session).tokens


async def _filled(
    mount: MountProfile, share: float, *extra: dict[str, Any], **config: object
) -> tuple[Context, ReactLoopAgent, int]:
    """A conversation filling `share` of its window, a tool that prints as many tokens
    as it is asked for, and the window."""
    ctx = await mounted(mount, *extra, **config)
    agent = agent_in(ctx)
    converse(agent.session)
    window = int(_tokens(ctx, agent) / share)
    log_event(
        agent.session,
        "request/context",
        {"provider": "fake", "model": "m", "contextWindow": window},
    )
    meter = ctx.require(TOKEN_METER)

    def emit(args: Any, _run: Any) -> str:  # noqa: ANN401
        wanted = int(as_str(args.get("tokens")))
        text = "word " * wanted
        while meter.measure_text(text) < wanted:
            text += "word " * 16
        return text

    ctx.require(TOOLS).register(simple_tool("emit", emit))
    return ctx, agent, window


async def _emit(ctx: Context, agent: ReactLoopAgent, tokens: int) -> str:
    result = await edit(ctx, agent, "emit", {"tokens": str(tokens)})
    assert not result.is_error, result.content
    return text_of(result.content)


def _readout(said: str) -> str | None:
    last = said.rstrip().splitlines()[-1]
    return last if last.startswith("[context: ") else None


async def test_a_result_that_crosses_a_share_ends_with_a_readout(mount: MountProfile) -> None:
    ctx, agent, window = await _filled(mount, 0.4)

    said = _readout(await _emit(ctx, agent, window // 6))

    assert said is not None
    share = re.search(r"tokens \((\d+)%\)", said)
    assert share is not None and 50 <= int(share.group(1)) < 75, said
    assert "next reminder comes at 75%" in said
    assert "context_tombstone" in said


async def test_a_result_below_every_share_says_nothing(mount: MountProfile) -> None:
    ctx, agent, _ = await _filled(mount, 0.2)

    assert _readout(await _emit(ctx, agent, 10)) is None


async def test_a_share_is_reported_once_until_an_edit_brings_the_context_back(
    mount: MountProfile,
) -> None:
    ctx, agent, window = await _filled(mount, 0.4)
    assert _readout(await _emit(ctx, agent, window // 6)) is not None

    assert _readout(await _emit(ctx, agent, window // 6)) is None, "told twice"

    step = sections_of(ctx, agent.session)[1]
    await edit(ctx, agent, "context_tombstone", {"sections": [step.name], "reason": "done"})
    needed = window // 2 - _tokens(ctx, agent) + window // 20
    assert _readout(await _emit(ctx, agent, needed)) is not None, "not re-armed by the edit"


async def test_the_last_share_says_it_is_the_last(mount: MountProfile) -> None:
    ctx, agent, window = await _filled(mount, 0.4)

    said = _readout(await _emit(ctx, agent, window * 2 // 5))

    assert said is not None and "last reminder" in said, said


async def test_with_no_window_there_is_no_readout(mount: MountProfile) -> None:
    ctx = await mounted(mount)
    agent = agent_in(ctx)
    converse(agent.session)
    ctx.require(TOOLS).register(simple_tool("emit", lambda _args, _run: "word " * 50_000))

    assert _readout(await _emit(ctx, agent, 0)) is None


async def test_no_shares_turns_readouts_off(mount: MountProfile) -> None:
    ctx, agent, window = await _filled(mount, 0.4, remindAt=[])

    assert _readout(await _emit(ctx, agent, window)) is None


async def test_the_readout_names_the_context_file(mount: MountProfile) -> None:
    ctx, agent, window = await _filled(mount, 0.4, MIRROR_ROW)
    path = ctx.require(CLM_MIRROR).path(agent.session, agent)

    said = _readout(await _emit(ctx, agent, window // 6))

    assert said is not None and f"edit {path}" in said, said


async def test_the_readout_measures_after_the_files_edit_lands(mount: MountProfile) -> None:
    """One call that removes the long step from the file and prints a little: measured
    before the edit, it would cross 50%; after, it does not."""
    ctx, agent, window = await _filled(mount, 0.45, MIRROR_ROW)
    service = ctx.require(CLM_MIRROR)
    path = await service.render(agent.session, agent)
    step = sections_of(ctx, agent.session)[1]
    text = path.read_text(encoding="utf-8")
    start = text.index(f"\n[[SECTION {step.name} ")
    end = text.index("\n[[SECTION ", start + 1)
    filler = "word " * (window // 10)

    def tidy(_args: Any, _run: Any) -> str:  # noqa: ANN401
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        return filler

    ctx.require(TOOLS).register(simple_tool("tidy", tidy))
    result = await edit(ctx, agent, "tidy", {})

    said = text_of(result.content)
    assert f"removed {step.name}".lower() in said.lower()
    assert _readout(said) is None, said


async def test_a_dispatch_inside_a_cell_carries_no_readout(mount: MountProfile) -> None:
    """The cell's own call does, once, rather than each dispatch it made."""
    ctx, agent, window = await _filled(mount, 0.4)

    nested = await run_tool(
        ctx,
        "emit",
        {"tokens": str(window // 6)},
        agent=agent,
        session=agent.session,
        call_id="cell-1:code:0",
        parent=object(),
    )

    assert _readout(text_of(nested.content)) is None


async def test_the_readout_counts_what_the_provider_counted(mount: MountProfile) -> None:
    """The system prompt and tool schemas are in the provider's count and in no section:
    a conversation at 30% of the window by its sections, in a request the provider
    counted at 60%, is past the first share — in the units compaction reads."""
    ctx, agent, window = await _filled(mount, 0.3)
    assert _readout(await _emit(ctx, agent, 10)) is None
    usage = TokenUsage(input_tokens=window * 3 // 5, output_tokens=10)
    reply = assistant_payload("noted", "m20", usage=usage)
    log_event(agent.session, "assistant/message", reply, SurfaceIntent("append"))

    said = _readout(await _emit(ctx, agent, 10))

    assert said is not None
    share = re.search(r"tokens \((\d+)%\)", said)
    assert share is not None and 55 <= int(share.group(1)) < 75, said
