"""P4-12 — `ph doctor`, and the seam that lets a row answer it (E1, E9, E10).

Two claims, and the second is the one the gate names.

**The report prints §4.8's table, not a paraphrase of it.** E1's failure is a
tier *name* overstating what the tier bounds, so the sentences are asserted
verbatim against `TIERS` — a doctor that summarised them would be free to drift
into exactly the overstatement the row exists to prevent, and it would look
tidier while doing it.

**Both rungs, when a deployment runs two.** The shipped posture is `advisory`
for the person's own agent and `worktree` for its children, so a report naming
one rung would be wrong about half the process.

The registry mechanics below mirror `tui_status`' (see `test_seams.py`): same
ordering, same drop-on-raise, same unwind-with-the-row. They are asserted again
rather than assumed, because "the same shape" is a claim about two files that
nothing else checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ph.cordis import Context
from ph.seams.containment import TIERS
from ph.seams.diagnostics import Diagnostic, DiagnosticsRegistry
from ph.testing import StubWorkspaceProvider, acquire_for_role, report_section

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------------ registry --


async def test_sections_are_ordered_then_named() -> None:
    registry = DiagnosticsRegistry(ctx=Context())
    registry.register(Diagnostic(id="second", read=lambda: [("a", "1")], order=10))
    registry.register(Diagnostic(id="first", read=lambda: [("b", "2")], order=1))

    assert [title for title, _ in registry.report()] == ["first", "second"]


async def test_a_section_with_nothing_to_say_is_omitted() -> None:
    """An empty section is a heading with no finding under it, which is how a
    report grows long enough that nobody reads the part that matters."""
    registry = DiagnosticsRegistry(ctx=Context())
    registry.register(Diagnostic(id="quiet", read=list))

    assert registry.report() == []


async def test_a_failing_section_does_not_take_the_report_down() -> None:
    """`ph doctor` is what a person runs *because* something is wrong, so the
    one section that cannot read is the worst possible moment to lose the rest —
    and it says so in place rather than vanishing."""
    registry = DiagnosticsRegistry(ctx=Context())

    def explode() -> list[tuple[str, str]]:
        raise RuntimeError("no")

    registry.register(Diagnostic(id="broken", read=explode))
    registry.register(Diagnostic(id="fine", read=lambda: [("still", "here")], order=1))

    report = dict(registry.report())
    assert report["fine"] == [("still", "here")]
    assert "failed" in report["broken"][0][0]


async def test_a_section_unwinds_with_the_row_that_registered_it() -> None:
    """I2 in the report: a row that unloads must not keep answering for a
    deployment it is no longer part of."""
    root = Context()
    registry = DiagnosticsRegistry(ctx=root)
    row = root.scope("row")
    registry.register(Diagnostic(id="owned", read=lambda: [("x", "y")]), scope=row)
    assert registry.report()

    await row.dispose()

    assert registry.report() == []


async def test_a_section_is_contributed_whatever_the_row_order(mount: Any, tmp_path: Path) -> None:
    """The reason `contribute` waits on the key instead of reading it at `apply`.

    `base.yaml` opens by promising that row order carries no load semantics, and
    a contributor that read `ctx.get("diagnostics")` at its own `apply` would
    have made this one pair the exception — silently, because the failure is a
    section that is simply missing from a report nobody can tell is incomplete.
    So the contributor is mounted **above** the seam here, which is the order a
    profile author is free to write and the one an apply-time read would lose.
    """
    profile = tmp_path / "reversed.yaml"
    profile.write_text(
        yaml.safe_dump(
            [
                {"id": "fs", "name": "fs-local", "config": {"root": str(tmp_path)}},
                {"id": "containment", "name": "containment", "config": {"tier": "advisory"}},
                {"id": "diagnostics", "name": "diagnostics"},
            ]
        ),
        encoding="utf-8",
    )

    ctx = await mount(profile=[profile])

    assert "Containment" in dict(ctx.diagnostics.report())


# ----------------------------------------------------------------- the table --


async def test_the_report_prints_the_tier_table_verbatim(mount: Any) -> None:
    """E1's gate, asserted against the one home the sentences have.

    Verbatim, because the defect is a description that claims more than the rung
    delivers — and a report free to reword is a second place for that claim to
    be made.
    """
    ctx = await mount({"id": "containment", "config": {"tier": "advisory"}})

    rows = report_section(ctx, "Containment")

    assert rows["tier (effective)"] == "advisory"
    assert rows["bounds"] == TIERS["advisory"].bounds
    assert rows["does NOT bound"] == TIERS["advisory"].does_not_bound
    assert rows["buys"] == TIERS["advisory"].buys
    assert rows["strict"] == "no"


async def test_both_rungs_are_described_when_a_deployment_runs_two(
    mount: Any, tmp_path: Path
) -> None:
    """The shipped `rlm` posture. A report that printed only the root agent's
    rung would say "bounds: nothing" about a process where the children — the
    ones doing the fan-out writing — are bounded."""
    ctx = await mount(
        {"id": "containment", "config": {"tier": "advisory", "childTier": "worktree"}}
    )
    ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))

    rows = report_section(ctx, "Containment")

    assert rows["tier (effective)"] == "advisory"
    assert rows["tier for children"] == "worktree"
    assert rows["advisory bounds"] == TIERS["advisory"].bounds
    assert rows["worktree does NOT bound"] == TIERS["worktree"].does_not_bound


async def test_the_effective_tier_is_reported_not_the_configured_one(mount: Any) -> None:
    """E10. An operator who set `worktree` and got nothing is owed that fact;
    reading their own setting back to them is how a person comes to believe in
    containment they do not have."""
    ctx = await mount({"id": "containment", "config": {"tier": "worktree"}})

    rows = report_section(ctx, "Containment")

    assert rows["tier (effective)"] == "advisory"
    assert rows["tier (configured)"] == "worktree — not in force here"


async def test_strict_reports_whether_it_is_satisfied(mount: Any) -> None:
    """`strict` that cannot be honoured refuses to start (E8), so a *running*
    process reporting `strict: yes` is one where it is satisfied — and saying so
    is what distinguishes "confined" from "asked to be"."""
    ctx = await mount({"id": "containment", "config": {"tier": "sandbox"}})
    ctx.containment.strict = True

    assert report_section(ctx, "Containment")["strict is satisfied"] == "no"


# ------------------------------------------------------------------ per agent --


async def test_workspaces_are_reported_per_agent(mount: Any, tmp_path: Path) -> None:
    """Since P4-11 there is no single answer, so the report has one row per
    agent that actually holds something."""
    ctx = await mount(
        {"id": "containment", "config": {"tier": "advisory", "childTier": "worktree"}}
    )
    ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    root = await acquire_for_role(ctx, tmp_path)
    child = await acquire_for_role(ctx, tmp_path, child=True)

    rows = report_section(ctx, "Workspaces")

    assert rows["provider"] == "worktree"
    # The two agents of the shipped posture, and the report says which is which:
    # the person's own agent is in their checkout, its child is not.
    assert root.kind == "shared" and root.kind in rows["agent root"]
    assert child.kind == "worktree" and child.kind in rows["agent child"]
    assert "writable" in rows["agent child"]


async def test_an_agent_that_holds_nothing_is_not_described(mount: Any) -> None:
    """`doctor` on an idle process has only the profile-level rows to show, and
    inventing one per configured agent would describe workspaces nobody holds."""
    ctx = await mount()

    assert not [label for label in report_section(ctx, "Workspaces") if label.startswith("agent ")]
