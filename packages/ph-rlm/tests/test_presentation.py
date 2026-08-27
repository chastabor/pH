"""`rlm-presentation` — one callable, and what a settled cell reads like (P3-09).

`test_code_mode.py` in ph-core tests the *mechanism* (a profile may present the
transport under its own name, and every place the name is load-bearing follows).
This file tests the RLM profile's use of it against the real kernel: the model
sees `ipython`, the text it gets back is prime-agent's layout, and the card
payload is derived from the durable result alone.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from runtime_helpers import run_cell

from ph.cancel import CancelToken
from ph.llm.types import ToolCallBlock
from ph.testing import simple_tool
from ph.tools.batch import execute_tool_calls
from ph.tools.registry import RUN_CODE
from ph_rlm.presentation import IPYTHON, IPYTHON_DESCRIPTION, cell_details, render_cell

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]


async def _cell(ctx: Any, program: str, *, agent: Any, session: Any, call_id: str = "c1") -> Any:
    return await run_cell(ctx, program, agent=agent, session=session, call_id=call_id, name=IPYTHON)


# ------------------------------------------------------------------ surface --


async def test_the_description_is_prime_agents_verbatim(mounted_runtime: Mounted) -> None:
    """The description is the contract the model was trained against.

    Asserted against the constant rather than a paraphrase so an edit to the
    wording has to be deliberate.
    """
    ctx, _session, agent = await mounted_runtime(presentation=True)
    definition = ctx.tools.get(IPYTHON, scope=agent.ctx)
    assert definition is not None
    assert definition.description == IPYTHON_DESCRIPTION
    assert "%%bash" in definition.description
    assert "the target project's own environment" in definition.description


async def test_a_cell_runs_under_the_presented_name(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(presentation=True)
    result = await _cell(ctx, "6 * 7", agent=agent, session=session)
    assert result.is_error is False
    assert result.value["value"] == 42


async def test_the_log_records_the_name_the_model_used(mounted_runtime: Mounted) -> None:
    """I3: model-visible means logged, and the model never saw `run_code`.

    Through the batch scheduler rather than `tools.execute`, because that is the
    path a model's call actually takes and the path that writes `tool/call`.
    """
    ctx, session, agent = await mounted_runtime(presentation=True)
    outcome = await execute_tool_calls(
        ctx,
        agent,
        turn=1,
        step=0,
        tool_calls=[
            ToolCallBlock(id="call-1", name=IPYTHON, arguments=json.dumps({"program": "6 * 7"}))
        ],
        token=CancelToken(),
        accept_context=lambda _message: None,
    )
    assert outcome.aborted is False
    calls = [event for event in session.events if event.type == "tool/call"]
    assert [event.data["name"] for event in calls] == [IPYTHON]
    results = [event for event in session.events if event.type == "tool/result"]
    assert len(results) == 1
    assert "[result] 42" in repr(results[0].data)


# ------------------------------------------------------------- result shape --


def test_the_result_text_is_logs_then_result_then_error() -> None:
    """Prime Agent's section order, with absent sections dropped."""
    [block] = render_cell(
        None, {"logs": "printed\n", "value": 42, "error": "Traceback: boom", "dispatches": 0}
    )
    assert block.text == "printed\n[result] 42\nTraceback: boom"

    [only_value] = render_cell(None, {"logs": "", "value": "x", "error": None})
    assert only_value.text == "[result] 'x'"

    [nothing] = render_cell(None, {"logs": "", "value": None, "error": None})
    assert nothing.text == "(no output)"


def test_a_falsy_value_is_still_a_result() -> None:
    """`0`, `False` and `""` are answers. `None` is the absence of one."""
    for value, expected in ((0, "[result] 0"), (False, "[result] False"), ("", "[result] ''")):
        [block] = render_cell(None, {"logs": "", "value": value, "error": None})
        assert block.text == expected


def test_the_details_payload_is_derived_from_the_durable_result() -> None:
    """Nothing in the card comes from live execution state, so a replayed cell
    draws the same card as a live one (A11)."""
    details = cell_details(
        None,
        {
            "logs": "out",
            "value": None,
            "error": "boom",
            "dispatches": 3,
            "truncated": True,
            "displays": [{"mime": "image/png"}, {"mime": "text/html"}],
        },
    )
    assert details == {
        "status": "error",
        "dispatches": 3,
        "truncated": True,
        "attachments": 2,
        "reset": False,
    }


def test_the_details_payload_reports_a_reset_namespace() -> None:
    details = cell_details(None, {"reset": True})
    assert details["reset"] is True
    assert details["status"] == "ok"


async def test_a_reset_kernel_reports_it_on_the_card(mounted_runtime: Mounted) -> None:
    """The card flag is the runtime's own boolean, not a sniff of the notice.

    Recovering it from the log prefix froze the notice's wording forever and let
    any program forge the flag by printing the marker as its first output.
    """
    ctx, session, agent = await mounted_runtime(presentation=True)
    await _cell(ctx, "import os\nos._exit(1)", agent=agent, session=session, call_id="c1")
    result = await _cell(ctx, "'alive'", agent=agent, session=session, call_id="c2")
    assert result.meta["reset"] is True
    assert result.meta["status"] == "ok"

    forged = await _cell(
        ctx, "print('<runtime_reset>')", agent=agent, session=session, call_id="c3"
    )
    assert forged.meta["reset"] is False


async def test_a_real_cell_produces_the_card_and_the_text(mounted_runtime: Mounted) -> None:
    """The two projections, computed by the pipeline rather than called directly."""
    ctx, session, agent = await mounted_runtime(presentation=True)
    ctx.tools.register(simple_tool("ping", lambda _args, _run: "pong"))
    result = await _cell(
        ctx,
        "print('hello')\nawait tools.ping()\n'done'",
        agent=agent,
        session=session,
    )
    assert result.is_error is False

    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    assert "hello" in text
    assert "[result] 'done'" in text
    assert result.meta == {
        "status": "ok",
        "dispatches": 1,
        "truncated": False,
        "attachments": 0,
        "reset": False,
    }


async def test_a_capped_cell_says_so_on_its_card(mounted_runtime: Mounted) -> None:
    """`truncated` reached the card only after `run_code` stopped dropping it.

    Both `truncated` and `displays` were on `CodeRunResult` and absent from the
    transport's return value, so a presentation reading them off the tool value
    saw a complete cell every time. Driven through a real capped stream rather
    than a hand-built dict, because a hand-built dict is exactly what missed it.
    """
    ctx, session, agent = await mounted_runtime(presentation=True)
    result = await _cell(
        ctx, "for _ in range(4000):\n    print('x' * 64)", agent=agent, session=session
    )
    assert result.is_error is False
    assert result.meta["truncated"] is True
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    assert "output truncated" in text


async def test_a_failing_cell_is_an_error_carrying_its_traceback(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(presentation=True)
    result = await _cell(ctx, "1 / 0", agent=agent, session=session)
    # The cell failed, but the *tool call* succeeded: a traceback is the model's
    # to read and act on, exactly as under prime-agent.
    assert result.is_error is False
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    assert "ZeroDivisionError" in text
    assert result.meta["status"] == "error"


# ---------------------------------------------------------------- integrity --


async def test_without_the_row_the_transport_keeps_its_reserved_name(
    mounted_runtime: Mounted,
) -> None:
    """The rename is the profile's, not the runtime's: a deployment that mounts
    the kernel without this row still gets `run_code`."""
    ctx, session, agent = await mounted_runtime(snapshots=False)
    assert ctx.tools.view(agent.ctx).transport_name == RUN_CODE
    result = await run_cell(ctx, "1 + 1", agent=agent, session=session)
    assert result.value["value"] == 2
