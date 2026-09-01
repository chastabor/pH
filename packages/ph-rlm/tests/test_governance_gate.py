"""The governance gate (P3-21): C1-C4 and C7, against the **shipped** profile.

Every claim here has unit tests elsewhere. This module exists because those mount
hand-picked rows, and the claim the containment argument rests on is about the
profile a user gets: *one cell is one tool call, but forty writes are forty
governance evaluations.* If these fail, the fold did not land — the plan's own
wording, and the reason this is a named module.

So everything mounts `ph-base` + `rlm/bundle.yaml` through the loader
(`shipped_profile`) and runs real cells in a real kernel.

**Two claims name Phase 4 consumers.** (b)'s `ToolCallLimit` and (c)'s spill
policy are Phase 4 rows. What Phase 3 owes is the *boundary* they attach to, so
(b) uses a shipped counter that already counts it and (c) mounts a listener on
the seam the real row will use. Both are marked at the assertion.

**One Phase 4 caveat the gate records rather than tests.** `permissions-fs` is
what makes an `fs/write-intent` veto load-bearing; until it ships, (a) asserts the
interception and that the veto is a *denial*, which is what Phase 3 owes.

## Why a denial aborts the run and does not merely reply

Recorded, answered, *and* aborted — only the third actually enforces C3.

The reply makes a well-behaved cell raise `RunStopped` and unwind. But a cell is
not obliged to behave: **`except BaseException: pass` followed by
`Path(...).write_text(...)` completed the write**, and the run then "failed"
afterwards — the tool call reported a refusal the program had already routed
around. Raw Python is not reachable by any waterfall, so the only thing that can
stop it is ending the process's turn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from runtime_helpers import dispatch_names, run_ipython_cell, settled_dispatches

from ph.seams.subagents import SubagentRequest
from ph.testing import run_tool
from ph.tools import Accept, Allow, Deny
from ph_rlm.messaging import OUT_OF_REACH
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------- (a) --


async def test_a_a_governed_write_is_intercepted_before_it_touches_disk(
    shipped_profile: Any, tmp_path: Path
) -> None:
    """C1: the cell's write goes through the seam, not around it.

    `fs/write-intent` fires *before* the bytes land — the difference between a
    gate and a report, and precisely what prime-agent's `edit` skill could not
    offer because it emitted its diff after writing. Asserted by *vetoing*: if
    the interception were after the write, the file would exist anyway.
    """
    ctx, session, agent = await shipped_profile()
    target = tmp_path / "gate-intercepted.txt"
    intents: list[str] = []

    async def veto(intent: Any, _next: Any) -> Any:
        intents.append(str(intent.path))
        return "the gate vetoed this write"

    ctx.on("fs/write-intent", veto)
    result = await run_ipython_cell(
        ctx,
        f"await tools.write(path={str(target)!r}, content='should not land')",
        agent=agent,
        session=session,
    )

    assert intents == [str(target)], "the write did not pass through fs/write-intent"
    assert not target.exists(), "the veto came after the bytes — a report, not a gate"
    # And it went through the pipeline as one governed dispatch, not as a raw side
    # effect the harness only heard about afterwards.
    assert dispatch_names(session) == ["write"]
    # The seam's veto reaches the run as a *denial*, so C3 applies to it too and a
    # consumer can tell policy from a fault. `FsDenied` had to become a
    # `HarnessError` for this to hold.
    assert result.is_error is True
    assert result.error.kind == "denied"
    assert "the gate vetoed this write" in result.error.message


async def test_a_a_denied_dispatch_ends_the_run_uncatchably(
    shipped_profile: Any, tmp_path: Path
) -> None:
    """C3, and the one deliberate divergence from dsh.

    A program that could `except` a refusal can route around it. So a *denial*
    settles the whole run: the cell's `except BaseException` does not get to
    report success, and the outcome reaches a consumer as `denied` rather than as
    an indistinguishable failure.

    The `sleep` is the window the abort ladder needs — `Kernel.cancel_grace` owns
    what that ladder can and cannot preempt.
    """
    # A short grace so the assertion is about the ladder rather than about racing
    # the shipped default with the cell's own sleep.
    ctx, session, agent = await shipped_profile(
        {"code-runtime-python": {"cancelGraceSeconds": 0.5}}
    )
    target, routed = tmp_path / "gate-denied.txt", tmp_path / "gate-routed-around.txt"

    async def refuse(execution: Any, next_: Any) -> Any:
        if execution.name == "write":
            return Deny(reason="this deployment does not allow writes")
        return await next_()

    ctx.on("tools/pre-execute", refuse)
    result = await run_ipython_cell(
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
    # `denied`, not `failed`: a refused cell must be distinguishable from one that
    # timed out. This gate is what found the kind being collapsed to a bool.
    assert result.error.kind == "denied"
    assert "does not allow writes" in result.error.message
    assert result.value is None, "the cell reported a value past a refusal"
    assert not target.exists(), "a pre-execute denial reached the tool body"
    # The abort caught the cell at the `sleep`, so the routing-around never ran.
    assert not routed.exists(), "the program caught a refusal and routed around it"


# ------------------------------------------------------------------- (b) --


async def test_b_three_binding_calls_are_three_governed_evaluations(
    shipped_profile: Any, tmp_path: Path
) -> None:
    """C2: the number prime-agent's single tool made invisible.

    A *shipped* counter rather than a stand-in for `ToolCallLimit`: the dispatch
    budget already counts exactly the boundary Phase 4 will count, so setting it
    to 3 and issuing a fourth proves the boundary is per dispatch without this
    test authoring the policy it checks.
    """
    ctx, session, agent = await shipped_profile({"tools-code-mode": {"maxDispatchesPerRun": 3}})
    result = await run_ipython_cell(
        ctx,
        f"seen = [await tools.glob(pattern='*.nothing', path={str(tmp_path)!r})"
        " for _ in range(3)]\nlen(seen)",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == 3

    # Three governed evaluations, three durable pairs — not one of each.
    assert dispatch_names(session) == ["glob", "glob", "glob"]
    assert len(settled_dispatches(session)) == 3

    # And the counter counts *dispatches*: a fourth in one cell is refused.
    fourth = await run_ipython_cell(
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
    shipped_profile: Any, tmp_path: Path
) -> None:
    """C5: offload is per dispatch, so one big read does not melt the others.

    On `tools/post-execute`, which is where the Phase 4 spill row attaches and the
    only seam that changes what the *program* receives — `tools/code-dispatch-log`
    reshapes the durable record alone, so a listener there would leave the cell
    holding the full bytes and prove nothing about the model's context.

    The stand-in replaces the *value*, because that is what `bridge.call` hands
    back to the program. Note the shape a real row will have to solve: an
    `Accept(has_value=True)` re-renders content through the tool's own output
    schema, so a generic policy cannot invent a replacement value without knowing
    the tool. That is P4-02's problem; what Phase 3 owes, and what this asserts,
    is that the boundary is per dispatch and one dispatch can be reshaped alone.
    """
    ctx, session, agent = await shipped_profile()
    big_path, small_path = tmp_path / "gate-big.txt", tmp_path / "gate-small.txt"
    big_path.write_text("x" * 4096)
    small_path.write_text("small")

    seen: list[str] = []

    async def offload(execution: Any, result: Any, next_: Any) -> Any:
        seen.append(execution.name)
        text = str((result.value or {}).get("text", ""))
        if len(text) <= 1024:
            return await next_(execution, result)
        spilled = {**result.value, "text": "[spilled]", "truncated": True}
        return Accept(value=spilled, has_value=True)

    ctx.on("tools/post-execute", offload)

    result = await run_ipython_cell(
        ctx,
        f"big = await tools.read(path={str(big_path)!r})\n"
        f"small = await tools.read(path={str(small_path)!r})\n"
        "(big['text'], small['text'])",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    # What the *program* saw: the big read replaced, its sibling intact. That is
    # the fact C5 is about — an oversized result must not cost its neighbours.
    assert result.value["value"] == ["[spilled]", "small"]
    assert dispatch_names(session) == ["read", "read"]
    # And the seam fired per dispatch rather than once for the cell, which is the
    # boundary the Phase 4 row needs to exist.
    assert seen == ["read", "read", "ipython"]


# ------------------------------------------------------------------- (d) --


async def test_d_a_non_family_send_cannot_be_re_permitted(shipped_profile: Any) -> None:
    """C7: the family boundary is a monotonic guard, not a policy listener.

    Guards run *last* and are deny-only, so there is no ordering in which a
    permissive row lets a send out of the family. The row mounted here allows
    everything and still cannot.

    The target is a **grandchild** — genuinely one generation too far. Two roots
    would be siblings under the rule, and an empty roster refuses for the wrong
    reason, so neither would exercise the boundary.
    """
    ctx, session, parent = await shipped_profile()
    child_run = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="delegate on", parent=parent, name="scout")
    )
    child = ctx.agents.get(child_run.session_id)
    assert child is not None
    grandchild = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="one level down", parent=child, name="recon")
    )
    assert ctx.sessions.get(grandchild.session_id) is not None

    ctx.on("tools/pre-execute", lambda _execution, _next: Allow())

    # From a cell, because under this profile that is the only way the model can
    # reach a tool at all (C6) — a native call is refused before policy runs.
    result = await run_ipython_cell(
        ctx,
        "await agent_message.send(message='hello', receiver_role='child', receiver_name='recon')",
        agent=parent,
        session=session,
    )
    assert result.is_error is True
    assert result.error.kind == "denied", "the boundary must deny, not merely fail"
    assert OUT_OF_REACH in result.error.message or "no child is named" in result.error.message


# ------------------------------------------------------------------- (e) --


async def test_e_a_runaway_cell_fails_at_its_budget(shipped_profile: Any, tmp_path: Path) -> None:
    """C4: one approved cell is still one decision.

    Without a per-cell cap, a single approval buys unbounded governed calls — the
    exact leverage a single-tool surface hands a model.
    """
    ctx, session, agent = await shipped_profile({"tools-code-mode": {"maxDispatchesPerRun": 4}})
    result = await run_ipython_cell(
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
    shipped_profile: Any,
) -> None:
    """C6, and the gate's own foundation, as one observable property.

    Every test above reaches tools *through* the transport, so none would notice
    if the profile stopped being Code Mode at all: `tools.mode: code` is a config
    patch with no `name`, and flipping it to `native` removes C6 while leaving all
    five behavioural tests green.

    Asserted as a property of the mounted profile rather than as a list of row
    names — this one call fails if the mode changed, if the transport was renamed,
    or if `rlm-presentation` left the bundle, and it needs no inventory to go
    stale when a row is renamed.
    """
    ctx, session, agent = await shipped_profile()
    result = await run_tool(
        ctx, "write", {"path": "/tmp/x", "content": "y"}, agent=agent, session=session
    )

    assert result.is_error is True
    assert result.error.kind == "denied"
    assert "ipython" in result.error.message, "the route back does not name the transport"
