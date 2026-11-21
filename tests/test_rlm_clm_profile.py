"""ph-clm Phase 3 — `rlm-clm`: the RLM, editing its own context.

**A repo-level scenario, not a package test**, for `test_rlm_stable_profile.py`'s
reason: it composes `phern`'s profile over `ph-rlm` and `ph-clm`, and ph-clm depends
on neither of the others.

What a composition can get wrong: that both of ph-clm's rows activate beside the RLM's
(an unmet `inject` key mounts a row that never runs), that the model is told about the
file, and that the file is written where the agent's own tools can reach it. The order
of the listeners is also a composition fact: `clm-mirror` writes the file and appends
its receipt after the rows of `ph-stabilize` that revise the surface and the result.
Those tests mount the rows themselves, in the order of `rlm-clm` and in the reverse
order, because the mirror must come out outside them in any mount order.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from clm_helpers import MIRROR_ROW, ROW, agent_in, without_section

from ph.agent.types import AgentHandle
from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import DEPLOYMENT, Context, Profile
from ph.keys import AGENTS, COMPACTION, LLM_FAKE, SESSIONS, TOOLS
from ph.llm.types import GenerateOptions, TextBlock
from ph.seams.compaction import CompactionResult, CompactionTrigger
from ph.session import Session
from ph.testing import FAKE_OPTIONS, MountProfile, run_tool, simple_tool
from ph_app.profiles import available_profiles, resolve_profile
from ph_clm import BUNDLE
from ph_clm.keys import CLM, CLM_MIRROR
from ph_clm.mirror import parse
from ph_stabilize.compaction import SummarizeEngine

pytestmark = pytest.mark.anyio

PROFILE = "rlm-clm"


def test_the_profile_is_offered_by_name() -> None:
    assert PROFILE in available_profiles()
    assert BUNDLE in resolve_profile(PROFILE)


def test_it_composes_the_rlm_and_the_context_rows() -> None:
    rows = {row.id for row in Profile.from_paths(resolve_profile(PROFILE)).rows}

    assert {"code-runtime-python", "clm-context", "clm-mirror"} <= rows


async def test_the_profile_boots_runs_a_turn_and_writes_the_file(mount: MountProfile) -> None:
    ctx = await mount(profile=resolve_profile(PROFILE))
    session = ctx.require(SESSIONS).create("clm")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    await agent.prompt("hello")

    assert ctx.get(CLM) is not None and ctx.get(CLM_MIRROR) is not None, "a row never activated"
    assert ctx.require(TOOLS).get("context_tombstone", scope=DEPLOYMENT) is not None
    path = ctx.require(CLM_MIRROR).path(session, agent)
    assert path.read_text(encoding="utf-8").startswith(f"[[LIVE_CONTEXT session={session.id} ")
    first = next(one for one in ctx.require(LLM_FAKE).requests if one.is_loop_request)
    assert str(path) in (first.system or ""), "the prompt does not name the file"


# ------------------------------------------------- the order of the listeners --


def _orders(row: dict[str, Any]) -> list[Any]:
    """`row` before the two rows of ph-clm, as `rlm-clm` mounts them, and after them."""
    return [
        pytest.param((row, ROW, MIRROR_ROW), id="as-rlm-clm"),
        pytest.param((ROW, MIRROR_ROW, row), id="reversed"),
    ]


async def _conversed(
    mount: MountProfile, rows: tuple[dict[str, Any], ...]
) -> tuple[Context, ReactLoopAgent]:
    """`rows`, mounted, and an agent with one finished turn: a task and a reply."""
    ctx = await mount(*rows)
    agent = agent_in(ctx)
    await agent.prompt("hello")
    return ctx, agent


def _unprotected(ctx: Context, session: Session) -> str:
    """The name of the first section that an edit may remove."""
    return next(one.name for one in ctx.require(CLM).sections(session) if not one.protected)


@pytest.mark.parametrize(
    "rows", _orders({"id": "compaction-summarize", "name": "compaction-summarize"})
)
async def test_a_compaction_at_the_request_is_in_the_file_the_model_reads(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch, rows: tuple[dict[str, Any], ...]
) -> None:
    """`compaction` revises the surface at the request itself (`window`), after the
    rest of its chain. `clm-mirror` writes the file after that revision. When the
    file was written first, the model read a file one generation old, and the
    edit that it made from that file was refused.

    The engine is replaced by one that removes a section at `window`, so that the
    test does not depend on the estimate that makes a real one act."""
    ctx, agent = await _conversed(mount, rows)
    session = agent.session
    engine = ctx.require(COMPACTION).require()
    assert isinstance(engine, SummarizeEngine)
    original = SummarizeEngine.compact_if_needed
    armed = True

    async def compact_if_needed(
        self: SummarizeEngine, handle: AgentHandle, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        nonlocal armed
        if trigger == "window" and armed:
            armed = False
            ctx.require(CLM).tombstone(
                session, [_unprotected(ctx, session)], "compacted", call_id=None
            )
            return None
        return await original(self, handle, trigger)

    monkeypatch.setattr(SummarizeEngine, "compact_if_needed", compact_if_needed)
    path = ctx.require(CLM_MIRROR).path(session, agent)
    seen: list[tuple[int, int]] = []

    async def read_at_the_request(
        options: GenerateOptions, next_: Callable[..., Awaitable[Any]]
    ) -> Any:  # noqa: ANN401
        if options.is_loop_request:
            header = parse(path.read_text(encoding="utf-8")).header
            seen.append((int(header["generation"]), session.surface.replace_generation))
        return await next_(options)

    ctx.on("llm/stream", read_at_the_request)

    await agent.prompt("go on")

    assert not armed, "the request-time compaction never ran"
    ((in_the_file, current),) = seen
    assert in_the_file == current, "the model read a file written before the compaction"


@pytest.mark.parametrize(
    "rows", _orders({"id": "tool-result-offload", "name": "tool-result-offload"})
)
async def test_the_receipt_of_an_edit_is_not_spilled_with_a_large_result(
    mount: MountProfile, rows: tuple[dict[str, Any], ...]
) -> None:
    """A call that edits the context file and prints more than `tool-result-offload`
    keeps inline. The offload spills what the tool printed. The receipt of the
    edit comes after the offload, so it stays a line of its own in the context,
    and not a numbered line of the preview of the spilled output.

    Without the Code Mode of `rlm-clm`: under it the model calls only `python`,
    and a test tool would have to run in a kernel to be the call that the mirror
    reads back for."""
    ctx, agent = await _conversed(mount, rows)
    session = agent.session
    path = await ctx.require(CLM_MIRROR).render(session, agent)
    tidied = without_section(path.read_text(encoding="utf-8"), _unprotected(ctx, session))

    def tidy_and_list(_args: Any, _run: Any) -> str:  # noqa: ANN401
        path.write_text(tidied, encoding="utf-8")
        return "a long listing\n" * 20_000

    ctx.require(TOOLS).register(simple_tool("tidy_and_list", tidy_and_list))

    result = await run_tool(ctx, "tidy_and_list", {}, agent=agent, session=session)

    assert not result.is_error, result.content
    *output, receipt = result.content
    printed = "".join(one.text for one in output if isinstance(one, TextBlock))
    assert isinstance(receipt, TextBlock), receipt
    assert receipt.text.startswith("[context file applied."), receipt.text
    assert "a long listing" in printed, "the output is gone"
    assert len(printed) < 20_000, "the output was not spilled"
