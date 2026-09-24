"""P10-12 — tools name their own effect, and a repeat of it is answered from the log.

DESIGN I2 asks outside state to be idempotent, because a session may repeat an
action; and after `TOOL_OUTCOME_UNKNOWN` it will, under a **new** call id, which is
why nothing keyed on the call can recognize the repeat. These hold the three
answers the pipeline gives a call whose tool named its effect — the recorded result,
a run with a note, a plain run — against a far side that counts what reached it.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.cordis import Context
from ph.json import JsonValue
from ph.keys import AGENTS, SESSIONS, TOOLS
from ph.llm.types import text_of
from ph.persistence import interrupted_turn_closers
from ph.session import Session
from ph.testing import FAKE_OPTIONS, MountProfile, external_tool, run_tool, simple_tool
from ph.tools import ToolExecutionResult
from ph.tools.registry import EFFECT_MAY_HAVE_HAPPENED

pytestmark = pytest.mark.anyio

SEND: dict[str, JsonValue] = {"id": "m1", "message": "the quarterly numbers"}


async def _call(
    ctx: Context, session: Session, call_id: str, arguments: JsonValue, name: str = "send"
) -> ToolExecutionResult:
    agent = ctx.require(AGENTS).get(session.id) or ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    return await run_tool(ctx, name, arguments, agent=agent, session=session, call_id=call_id)


async def _mounted(mount: MountProfile, **options: Any) -> tuple[Context, Session, Any]:  # noqa: ANN401
    ctx = await mount()
    tool, far = external_tool(**options)
    ctx.require(TOOLS).register(tool)
    return ctx, ctx.require(SESSIONS).create("effects"), far


async def test_a_repeated_effect_returns_the_recorded_result(mount: MountProfile) -> None:
    """The model re-issues a call with a new id; the effect is the same, so the far
    side hears it once and the second call is handed the first one's result.

    Sabotage: drop the dedupe from `_open_effect` and the far side hears it twice.
    """
    ctx, session, far = await _mounted(mount)

    first = await _call(ctx, session, "c1", SEND)
    again = await _call(ctx, session, "c2", SEND)

    assert far.deliveries == ["the quarterly numbers"]
    assert text_of(again.content) == text_of(first.content) == "delivered m1"
    assert again.meta is not None and dict(again.meta) == {"repeated": True, "repeatOf": "c1"}
    assert not again.is_error


async def test_a_repeat_after_an_unknown_outcome_is_not_silently_suppressed(
    mount: MountProfile,
) -> None:
    """The first attempt reached disk and nothing after it did — a crash — so
    repair settled it `unknown`. Suppressing the retry would claim the effect
    happened; running it silently would hide that it may have. It runs, and says so.
    """
    ctx, session, far = await _mounted(mount)
    _unknown_prior(session)
    assert session.latest("tool/effect-settled") is not None, "repair settled it"

    result = await _call(ctx, session, "c2", SEND)

    assert far.deliveries == ["the quarterly numbers"], "a retry of an unknown runs"
    assert EFFECT_MAY_HAVE_HAPPENED.format(call_id="c1") in text_of(result.content)


async def test_a_repeat_of_a_failed_effect_runs_again(mount: MountProfile) -> None:
    """The tool said it failed, and the tool is the one that knows its far side:
    the retry runs, with no note second-guessing it."""
    ctx, session, far = await _mounted(mount, fails=True)

    first = await _call(ctx, session, "c1", SEND)
    again = await _call(ctx, session, "c2", SEND)

    assert first.is_error and again.is_error
    assert len(far.deliveries) == 2
    assert "may already have happened" not in text_of(again.content)


async def test_a_tool_that_declares_no_key_runs_every_time(mount: MountProfile) -> None:
    ctx, session, far = await _mounted(mount, keyed=False)

    await _call(ctx, session, "c1", SEND)
    await _call(ctx, session, "c2", SEND)

    assert len(far.deliveries) == 2
    assert session.latest("tool/effect") is None, "an unkeyed call records no effect"


async def test_a_different_effect_is_a_different_key(mount: MountProfile) -> None:
    ctx, session, far = await _mounted(mount)

    await _call(ctx, session, "c1", SEND)
    await _call(ctx, session, "c2", {**SEND, "id": "m2"})

    assert len(far.deliveries) == 2


async def test_the_run_key_is_stable_across_a_tools_own_retries(mount: MountProfile) -> None:
    """`ToolRunContext.idempotency_key` is for the tool's own retries against a far
    side that takes a key — the same for every attempt inside one call, and
    different for the next call."""
    ctx = await mount()
    seen: list[str] = []

    def retrying(_args: object, run: Any) -> str:  # noqa: ANN401
        seen.extend(run.idempotency_key for _ in range(3))
        return "ok"

    ctx.require(TOOLS).register(simple_tool("post", retrying))
    session = ctx.require(SESSIONS).create("keys")
    await _call(ctx, session, "c1", {}, name="post")
    await _call(ctx, session, "c2", {}, name="post")

    assert seen == ["keys/c1"] * 3 + ["keys/c2"] * 3


def _unknown_prior(session: Session) -> None:
    """The first attempt reached disk and a crash left it for repair to settle."""
    session.append("tool/effect", {"key": "send:m1", "tool": "send", "callId": "c1"})
    for closer in interrupted_turn_closers(session.events):
        session.admit(closer)


async def test_a_tool_that_can_tell_is_asked_before_the_retry_runs(mount: MountProfile) -> None:
    """P10-13 inside the pipeline: the tool that can ask its far side is asked
    whether the unknown attempt happened. It did, so the retry is answered with the
    tool's own rendering of that — and the far side hears nothing twice."""
    ctx, session, far = await _mounted(mount, reconciles=True)
    far.ids.add("m1")
    far.deliveries.append("the quarterly numbers")
    _unknown_prior(session)

    result = await _call(ctx, session, "c2", SEND)

    assert far.deliveries == ["the quarterly numbers"]
    assert text_of(result.content) == "delivered m1"
    assert result.meta is not None and dict(result.meta) == {"reconciled": True, "repeatOf": "c1"}


async def test_a_tool_that_says_it_did_not_happen_runs_without_the_note(
    mount: MountProfile,
) -> None:
    ctx, session, far = await _mounted(mount, reconciles=True)
    _unknown_prior(session)

    result = await _call(ctx, session, "c2", SEND)

    assert far.deliveries == ["the quarterly numbers"]
    assert "may already have happened" not in text_of(result.content)
