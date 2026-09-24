"""P10-12 — tools name their own effect, and a repeat of it is answered from the log.

DESIGN I2 asks outside state to be idempotent, because a session may repeat an
action; and after `TOOL_OUTCOME_UNKNOWN` it will, under a **new** call id, which is
why nothing keyed on the call can recognize the repeat. These hold the three
answers the pipeline gives a call whose tool named its effect — the recorded result,
a run with a note, a plain run — against a far side that counts what reached it.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from ph.cordis import Context
from ph.json import JsonValue, as_str
from ph.keys import AGENTS, SESSIONS, TOOLS
from ph.llm.types import text_of
from ph.persistence import interrupted_turn_closers
from ph.session import Session, unsettled_why
from ph.testing import FAKE_OPTIONS, MountProfile, external_tool, log_event, run_tool, simple_tool
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
    log_event(session, "tool/effect", {"key": "send:m1", "tool": "send", "callId": "c1"})
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


def _held_tool(far: list[str], release: anyio.Event, entered: anyio.Event) -> Any:  # noqa: ANN401
    """A keyed tool whose delivery waits on `release` — a call still running."""

    async def deliver(args: Any, _run: Any) -> str:  # noqa: ANN401
        entered.set()
        await release.wait()
        far.append(as_str(args.get("message")))
        return f"delivered {as_str(args.get('id'))}"

    return simple_tool("send", deliver, idempotency_key=lambda args: as_str(args.get("id")))


async def test_two_concurrent_calls_with_one_effect_run_it_once(mount: MountProfile) -> None:
    """T0. The second call finds the first's intent open *in this process* — still
    running, not a crash's orphan. It is refused rather than run, and rather than
    opening a second intent underneath the first, which unseated the first call's
    settle and raised it out of the pipeline.

    Sabotage: drop the in-flight check and the far side hears it twice — and the
    first call's settle raises `IntentError`.
    """
    ctx = await mount()
    far: list[str] = []
    release, entered = anyio.Event(), anyio.Event()
    ctx.require(TOOLS).register(_held_tool(far, release, entered))
    session = ctx.require(SESSIONS).create("in-flight")
    results: dict[str, ToolExecutionResult] = {}

    async def call(call_id: str) -> None:
        results[call_id] = await _call(ctx, session, call_id, SEND)

    # Bounded, so a regression that runs the second call fails rather than hangs:
    # it would wait on the same `release` the test only sets after it returns.
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(call, "c1")
            await entered.wait()
            await call("c2")
            release.set()

    assert far == ["the quarterly numbers"], "one effect, however many asked for it"
    assert text_of(results["c1"].content) == "delivered m1"
    second = results["c2"]
    assert second.is_error and second.error is not None
    assert second.error.info == {"name": "ToolEffectInFlight", "code": "TOOL_EFFECT_IN_FLIGHT"}
    assert "call c1" in text_of(second.content)
    again = await _call(ctx, session, "c3", SEND)
    assert again.meta is not None and again.meta["repeatOf"] == "c1", "answered from the log now"


async def test_a_canceled_keyed_call_leaves_its_effect_unknown_not_running(
    mount: MountProfile,
) -> None:
    """A cancellation with the body entered may have let the effect happen. Left
    open, the intent would read as still running in this process and refuse every
    repeat until a restart; settled `unknown`, the repeat runs with the note."""
    ctx = await mount()
    far: list[str] = []
    release, entered = anyio.Event(), anyio.Event()
    ctx.require(TOOLS).register(_held_tool(far, release, entered))
    session = ctx.require(SESSIONS).create("canceled")

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_call, ctx, session, "c1", SEND)
            await entered.wait()
            tasks.cancel_scope.cancel()

    settle = session.latest("tool/effect-settled")
    assert settle is not None and unsettled_why(settle.data) == "outcome-unknown"
    release.set()
    with anyio.fail_after(5):
        result = await _call(ctx, session, "c2", SEND)
    assert far == ["the quarterly numbers"]
    assert EFFECT_MAY_HAVE_HAPPENED.format(call_id="c1") in text_of(result.content)
