"""The governance gate (P3-21): C1-C4 and C7, against the **shipped** profile.

Every claim here has unit tests elsewhere. Those unit tests mount hand-picked
rows. The claim that the containment argument depends on is about the profile
that a user gets: *one cell is one tool call, but forty writes are forty
governance evaluations.* If these tests fail, the fold did not land. That is the
wording of the plan, and it is the reason that this is a named module.

Every test therefore mounts `ph-base` + `headless` + the `rlm` bundle
(`ph_rlm/bundle.yaml`) through the loader (`shipped_profile`) and runs real
cells in a real kernel.

**Two claims name Phase 4 consumers.** `ToolCallLimit` in (b) and the spill
policy in (c) are Phase 4 rows. Phase 3 owes the *boundary* that they attach
to. So (b) uses a shipped counter that already counts that boundary, and (c)
mounts a listener on the seam that the real row uses. Both are marked at the
assertion.

**The gate records one Phase 4 caveat and does not test it.** `permissions-fs`
is the row that gives an `fs/write-intent` veto a real policy to enforce. It
ships in `ph-stabilize` and is not in the profile that this gate mounts. So (a)
asserts the interception and that the veto is a *denial*, which is what Phase 3
owes.

## Why a denial aborts the run and does not merely reply

A denial is recorded, answered, *and* aborted. Only the abort enforces C3.

The reply makes a well-behaved cell raise `RunStopped` and unwind. But a cell
does not have to behave well. **`except BaseException: pass` followed by
`Path(...).write_text(...)` completed the write.** The run then "failed"
afterwards: the tool call reported a refusal that the program had already
bypassed. No waterfall can reach raw Python, so the only thing that can stop it
is to end the turn of the process.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from rlm_fixtures import ModelGate, ShippedProfile
from runtime_helpers import dispatch_names, run_python_cell, settled_dispatches

from ph.keys import AGENTS, SESSIONS, SUBAGENTS
from ph.seams.subagents import SubagentRequest
from ph.testing import not_none, run_tool
from ph.tools import Accept, Allow, Deny, ToolExecution
from ph_rlm.messaging import OUT_OF_REACH
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------- (a) --


async def test_a_a_governed_write_is_intercepted_before_it_touches_disk(
    shipped_profile: ShippedProfile, tmp_path: Path
) -> None:
    """C1: the write of the cell goes through the seam, not around it.

    `fs/write-intent` fires *before* the bytes are written. That is the
    difference between a gate and a report. The `edit` skill of prime-agent
    could not offer it, because that skill emitted its diff after it wrote. The
    test asserts this with a *veto*: if the interception came after the write,
    the file would exist anyway.
    """
    ctx, session, agent = await shipped_profile()
    target = tmp_path / "gate-intercepted.txt"
    intents: list[str] = []

    async def veto(intent: Any, _next: Any) -> Any:  # noqa: ANN401
        intents.append(str(intent.path))
        return "the gate vetoed this write"

    ctx.on("fs/write-intent", veto)
    result = await run_python_cell(
        ctx,
        f"await tools.write(path={str(target)!r}, content='should not land')",
        agent=agent,
        session=session,
    )

    assert intents == [str(target)], "the write did not pass through fs/write-intent"
    assert not target.exists(), "the veto came after the bytes — a report, not a gate"
    # The write went through the pipeline as one governed dispatch. It was not a
    # raw side effect that the harness learned about afterwards.
    assert dispatch_names(session) == ["write"]
    # The veto of the seam reaches the run as a *denial*. C3 therefore applies to
    # it too, and a consumer can tell a policy refusal from a fault. `FsDenied`
    # had to become a `HarnessError` for this to hold.
    assert result.is_error is True
    assert result.error.kind == "denied"
    assert "the gate vetoed this write" in result.error.message


async def test_a_a_denied_dispatch_ends_the_run_uncatchably(
    shipped_profile: ShippedProfile, tmp_path: Path
) -> None:
    """C3, and the one deliberate difference from dsh.

    A program that can `except` a refusal can bypass it. A *denial* therefore
    settles the whole run. The `except BaseException` of the cell cannot report
    success, and a consumer receives the outcome as `denied`, which it can tell
    apart from any other failure.

    The `sleep` is the window that the abort ladder needs. The docstring of
    `Kernel.cancel_grace` states what that ladder can and cannot preempt.
    """
    # The grace is short so that the assertion tests the ladder. With the shipped
    # default, the test would race that default against the sleep of the cell.
    ctx, session, agent = await shipped_profile(
        {"code-runtime-python": {"cancelGraceSeconds": 0.5}}
    )
    target, routed = tmp_path / "gate-denied.txt", tmp_path / "gate-routed-around.txt"

    async def refuse(execution: ToolExecution, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        if execution.name == "write":
            return Deny(reason="this deployment does not allow writes")
        return await next_()

    ctx.on("tools/pre-execute", refuse)
    result = await run_python_cell(
        ctx,
        "import time\n"
        "from pathlib import Path\n"
        "try:\n"
        f"    await tools.write(path={str(target)!r}, content='x')\n"
        "except BaseException:\n"
        "    pass\n"
        "time.sleep(5)\n"
        f"Path({str(routed)!r}).write_text('routed around it')\n"
        "'the program continued'",
        agent=agent,
        session=session,
    )
    assert result.is_error is True
    # `denied`, not `failed`: a consumer must be able to tell a refused cell from
    # one that timed out. This gate found that the kind was collapsed to a bool.
    assert result.error.kind == "denied"
    assert "does not allow writes" in result.error.message
    assert result.value is None, "the cell reported a value past a refusal"
    assert not target.exists(), "a pre-execute denial reached the tool body"
    # The abort stopped the cell at the `sleep`, so the bypass never ran.
    assert not routed.exists(), "the program caught a refusal and routed around it"


# ------------------------------------------------------------------- (b) --


async def test_b_three_binding_calls_are_three_governed_evaluations(
    shipped_profile: ShippedProfile, tmp_path: Path
) -> None:
    """C2: the number that the single tool of prime-agent made invisible.

    The test uses a *shipped* counter and not a stand-in for `ToolCallLimit`.
    The dispatch budget already counts exactly the boundary that the Phase 4
    limit counts. The test sets the budget to 3 and issues a fourth dispatch.
    That proves that the boundary is per dispatch, and the test does not have to
    write the policy that it checks.
    """
    ctx, session, agent = await shipped_profile({"tools-code-mode": {"maxDispatchesPerRun": 3}})
    result = await run_python_cell(
        ctx,
        f"seen = [await tools.glob(pattern='*.nothing', path={str(tmp_path)!r})"
        " for _ in range(3)]\nlen(seen)",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == 3

    # Three governed evaluations and three durable pairs, not one of each.
    assert dispatch_names(session) == ["glob", "glob", "glob"]
    assert len(settled_dispatches(session)) == 3

    # And the counter counts *dispatches*: a fourth in one cell is refused.
    fourth = await run_python_cell(
        ctx,
        f"[await tools.glob(pattern='*.nothing', path={str(tmp_path)!r}) for _ in range(4)]",
        agent=agent,
        session=session,
        call_id="c2",
    )
    assert fourth.is_error is True
    assert "max_dispatches_per_run=3" in fourth.error.message


# ------------------------------------------------------------------- (c) --


async def test_c_one_oversized_dispatch_is_offloaded_without_its_siblings(
    shipped_profile: ShippedProfile, tmp_path: Path
) -> None:
    """C5: offload is per dispatch, so one big read does not affect the others.

    The listener is on `tools/post-execute`. The Phase 4 spill row attaches
    there, and it is the only seam that changes what the *program* receives.
    `tools/code-dispatch-log` reshapes only the durable record. A listener there
    would leave the full bytes in the cell and would prove nothing about the
    context of the model.

    The stand-in replaces the *value*, because the value is what `bridge.call`
    returns to the program. This docstring used to record a problem that a real
    row would have to solve. An `Accept(has_value=True)` re-renders content
    through the output schema of the tool, so a generic policy cannot invent a
    replacement value without knowledge of the tool. D10 settled the problem by
    dropping the premise: a spill policy never needed to invent a value. It
    spills the *render* and passes the value through, because the program
    already holds the object and only the render costs context. This test still
    asserts the Phase 3 half: the boundary is per dispatch, and one dispatch can
    be reshaped alone.
    """
    ctx, session, agent = await shipped_profile()
    big_path, small_path = tmp_path / "gate-big.txt", tmp_path / "gate-small.txt"
    big_path.write_text("x" * 4096)
    small_path.write_text("small")

    seen: list[str] = []

    async def offload(
        execution: ToolExecution,
        result: Any,  # noqa: ANN401
        next_: Callable[..., Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        seen.append(execution.name)
        text = str((result.value or {}).get("text", ""))
        if len(text) <= 1024:
            return await next_(execution, result)
        spilled = {**result.value, "text": "[spilled]", "truncated": True}
        return Accept(value=spilled, has_value=True)

    ctx.on("tools/post-execute", offload)

    result = await run_python_cell(
        ctx,
        f"big = await tools.read(path={str(big_path)!r})\n"
        f"small = await tools.read(path={str(small_path)!r})\n"
        "(big['text'], small['text'])",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    # What the *program* saw: the big read was replaced and its sibling was
    # intact. That is the fact that C5 is about: an oversized result must not
    # cost its neighbors.
    assert result.value["value"] == ["[spilled]", "small"]
    assert dispatch_names(session) == ["read", "read"]
    # The seam fired for each dispatch, and not only one time for the cell. That
    # is the boundary that the Phase 4 row needs.
    assert seen == ["read", "read", "python"]


# ------------------------------------------------------------------- (d) --


async def test_d_a_non_family_send_cannot_be_re_permitted(
    shipped_profile: ShippedProfile, gate: ModelGate
) -> None:
    """C7: the family boundary is a monotonic guard, not a policy listener.

    Guards run *last* and can only deny. No ordering therefore exists in which a
    permissive row lets a send out of the family. The listener that this test
    mounts allows everything, and the send is still refused.

    The target is a **grandchild**, which is exactly one generation too far. Two
    roots would be siblings under the rule, and a parent with no children
    refuses for the wrong reason. Neither case would exercise the boundary.

    **The children are held on their first model call** (`gate`), so the family
    still exists when the send is refused. A child whose fake model answers at
    once settles at an unpredictable await point. A child that settled while the
    grandchild was being admitted under it tore down the scope of the grandchild
    in the middle of the admission. A real child cannot do that, because it
    spawns from its own running turn.
    """
    ctx, session, parent = await shipped_profile()
    child_run = await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="delegate on", parent=parent, name="scout")
    )
    child = ctx.require(AGENTS).get(child_run.session_id)
    assert child is not None
    grandchild = await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="one level down", parent=child, name="recon")
    )
    assert ctx.require(SESSIONS).get(grandchild.session_id) is not None

    ctx.on("tools/pre-execute", lambda _execution, _next: Allow())

    # The send comes from a cell. Under this profile a cell is the only way that
    # the model can reach a tool (C6): a native call is refused before policy
    # runs.
    result = await run_python_cell(
        ctx,
        "await agent_message.send(message='hello', receiver_role='child', receiver_name='recon')",
        agent=parent,
        session=session,
    )
    assert result.is_error is True
    assert result.error.kind == "denied", "the boundary must deny, not merely fail"
    assert OUT_OF_REACH in result.error.message or "no child is named" in result.error.message


# ------------------------------------------------------------------- (e) --


async def test_e_a_runaway_cell_fails_at_its_budget(
    shipped_profile: ShippedProfile, tmp_path: Path
) -> None:
    """C4: one approved cell is still one decision.

    Without a cap for each cell, a single approval permits an unlimited number
    of governed calls. That is exactly the leverage that a single-tool surface
    gives a model.
    """
    ctx, session, agent = await shipped_profile({"tools-code-mode": {"maxDispatchesPerRun": 4}})
    result = await run_python_cell(
        ctx,
        "for _ in range(20):\n"
        f"    await tools.glob(pattern='*.nothing', path={str(tmp_path)!r})\n"
        "'unreached'",
        agent=agent,
        session=session,
    )
    assert result.is_error is True
    assert "max_dispatches_per_run=4" in result.error.message
    assert len(dispatch_names(session)) == 4, "the budget did not stop the cell where it said"


# ------------------------------------------- the profile the gate stands on --


async def test_the_shipped_profile_still_refuses_a_native_tool_call(
    shipped_profile: ShippedProfile,
) -> None:
    """C6, and the foundation of the gate itself, as one observable property.

    Every test above reaches tools *through* the transport. None of them would
    notice if the profile stopped being Code Mode. `tools.mode: code` is a
    config patch with no `name`. A change of that value to `native` removes C6
    and leaves all six behavioral tests green.

    The test asserts a property of the mounted profile, not a list of row names.
    This one call fails if the mode changed, if the transport was renamed, or if
    `rlm-presentation` left the bundle. It needs no inventory that can go stale
    when a row is renamed.
    """
    ctx, session, agent = await shipped_profile()
    result = await run_tool(
        ctx, "write", {"path": "/tmp/x", "content": "y"}, agent=agent, session=session
    )

    assert result.is_error is True
    failure = not_none(result.error)
    assert failure.kind == "denied"
    assert "python" in failure.message, "the route back does not name the transport"
