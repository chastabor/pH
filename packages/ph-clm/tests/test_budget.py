"""ph-clm Phase 3: context readouts on tool results.

The gates (`plans/Ph_Clm_Context_Editing_Plan.md`, Phase 3): a result that takes the
context past a share of its window ends with one readout line, and nothing else is
added to the context; the next result past the same share says nothing, and once an
edit brings the context back down, crossing it again says so again; the last share
says it is the last; the readout names the context file when a row keeps one, and
measures after that file's edit has landed; with no window there is no readout.
"""

from __future__ import annotations

from typing import Any

import pytest
from clm_helpers import (
    MIRROR_ROW,
    agent_in,
    context_tokens,
    converse,
    edit,
    filler,
    mounted,
    readout_share,
    sections_of,
)

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.json import as_str
from ph.keys import CODE_RUNTIME_STUB, COMPACTION, TOKEN_METER, TOOLS
from ph.llm.types import TokenUsage, text_of
from ph.session import SurfaceIntent
from ph.testing import (
    MountProfile,
    StubCompactionEngine,
    assistant_payload,
    code_mode_stub,
    log_event,
    run_tool,
    simple_tool,
)
from ph.tools.registry import RUN_CODE
from ph_clm.budget import Shares
from ph_clm.keys import CLM_MIRROR

pytestmark = pytest.mark.anyio


async def _filled(
    mount: MountProfile, share: float, *extra: dict[str, Any], **config: object
) -> tuple[Context, ReactLoopAgent, int]:
    """A conversation filling `share` of its window, a tool that prints as many tokens
    as it is asked for, and the window."""
    ctx = await mounted(mount, *extra, **config)
    agent = agent_in(ctx)
    converse(agent.session)
    window = int(context_tokens(ctx, agent.session) / share)
    log_event(
        agent.session,
        "request/context",
        {"provider": "fake", "model": "m", "contextWindow": window},
    )
    meter = ctx.require(TOKEN_METER)

    def emit(args: Any, _run: Any) -> str:  # noqa: ANN401
        return filler(meter, int(as_str(args.get("tokens"))))

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
    assert 50 <= readout_share(said) < 75, said
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
    needed = window // 2 - context_tokens(ctx, agent.session) + window // 20
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
    assert 55 <= readout_share(said) < 75, said


def test_the_last_reminder_gives_way_below_compactions_trigger() -> None:
    """A margin below the engine's trigger, so the model has room to act before a summary;
    a share at or past that point gives way to it. Nothing moves without an engine, and
    no shares stays no readouts."""
    assert Shares.of([0.5, 0.75], 0.2, 0.85, 16_384).thresholds == (0.5, 0.65)
    assert Shares.of([0.5, 0.75], 0.2, None, 16_384).thresholds == (0.5, 0.75)
    assert Shares.of([], 0.2, 0.85, 16_384).thresholds == ()
    assert Shares.of([0.5], 0.9, 0.85, 16_384).thresholds == (0.5,)


async def test_the_last_reminder_comes_before_compaction_and_says_so(
    mount: MountProfile,
) -> None:
    """With an engine compacting at 85%, the last reminder is at 65% — and it says what
    happens if the model does nothing.

    Sabotage: ignore the engine's trigger, and 70% is no last reminder at all.
    """
    ctx, agent, window = await _filled(mount, 0.55)
    ctx.require(COMPACTION).register(StubCompactionEngine(0.85))

    said = _readout(await _emit(ctx, agent, window * 15 // 100))

    assert said is not None and "last reminder" in said, said
    assert "At 85% older context is summarized for you." in said


async def test_under_code_mode_the_readout_spells_the_call(mount: MountProfile) -> None:
    """The keywords and all, as a cell writes it: a live model passed the sections
    positionally and lost its one chance to edit to the error."""
    ctx, agent, window = await _filled(mount, 0.4, code_mode_stub())
    ctx.require(TOOLS).present_as("code", scope=agent.ctx)
    ctx.require(CODE_RUNTIME_STUB).register_program("big", lambda _b, _emit: "word " * window)

    result = await run_tool(ctx, RUN_CODE, {"program": "big"}, agent=agent, session=agent.session)

    said = _readout(text_of(result.content))
    assert said is not None, text_of(result.content)[-300:]
    assert "await tools.context_tombstone(sections=['S12'], reason='...')" in said, said
