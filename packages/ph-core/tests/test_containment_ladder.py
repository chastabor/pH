"""P4-16 — the tier table's columns, asserted (E1, E13).

Every other workspace module tests a *mechanism*: `test_workspace_git.py` drives
real `git worktree`, `test_workspace_lifecycle.py` pins who acquires and when,
`test_containment.py` pins which rung a role gets, `test_diagnostics.py` pins
what `ph doctor` renders. This module tests the **claims** — §4.8's `bounds`,
`does NOT bound` and `buys` — because those are what a person reads before
deciding what they are protected from, and a claim is the one part of a design
that rots without any code changing.

So it holds only what no other module can say, one test per column. A first
draft also restated scratch, the redirection env and the fan-out, and was cut
back: a claims module that duplicates mechanism tests cannot tell the next
person which module their new assertion belongs in.

**The writes are `ctx.subprocess`, not `ctx.shell`.** That is the difference
between measuring the ladder and measuring a seam: `ShellService.run(agent=...)`
applies `workspace_policy(workspace)` whenever a sandbox backend is available,
so a shell-based escape test would pass or fail on *what the profile mounted*
and say nothing about the rung — and P6-06's "same write, refused" would then
pass because of that branch rather than because of `sandbox`. A bare spawn is
what the tier sentence is actually about: an `open()` that never consults a cwd.

**The escape is asserted on purpose.** `worktree` buys collision isolation and
revertibility, not confinement; the day someone "fixes" the escape at this rung
without moving to `sandbox`, the tier table becomes a lie. E13 exists to make
that day fail a test rather than fool a reader. The other half — the same write,
refused under `sandbox` — is P6-06.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from ph.seams.containment import TIERS
from ph.seams.subprocess import SubprocessSpawnSpec, scrub_env
from ph.testing import WORKTREE_ROWS, needs_git, report_section, worktree_agent

pytestmark = pytest.mark.anyio


async def _write(ctx: Any, workspace: Any, target: str) -> tuple[int, str]:
    """A raw `open()` from a process whose cwd is the agent's tree.

    No `agent=`, so nothing resolves the path on its behalf and no policy is
    applied — which is what anything that is not a pH tool does, and exactly the
    surface the table's first two columns are about.
    """
    outcome = await ctx.subprocess.run(
        SubprocessSpawnSpec(
            argv=(sys.executable, "-c", f"open({target!r}, 'w').write('written')"),
            cwd=workspace.root,
            env=scrub_env(extra=workspace.env),
        )
    )
    return outcome.exit_code, outcome.stdout + outcome.stderr


# ------------------------------------------------------------------- bounds --


@needs_git
async def test_a_relative_raw_write_is_bounded_by_the_tree(mount: Any, tmp_path: Path) -> None:
    """A relative path resolves against the process's cwd, and under `worktree`
    that cwd is the agent's own checkout — which is the whole of what the rung
    buys: eight children writing `notes.txt` write eight files instead of racing
    for one."""
    ctx, _session, _agent, workspace = await worktree_agent(mount, tmp_path)

    code, output = await _write(ctx, workspace, "notes.txt")

    assert code == 0, output
    assert (workspace.root / "notes.txt").read_text(encoding="utf-8") == "written"
    assert not (tmp_path / "repo" / "notes.txt").exists(), "the write reached the parent tree"


# ----------------------------------------------------------- does NOT bound --


@needs_git
async def test_an_absolute_raw_write_escapes_the_tree(mount: Any, tmp_path: Path) -> None:
    """E13, asserted so the table cannot regress.

    This is the test that keeps `ph doctor` honest. A raw write to an absolute
    path never consults a cwd, so it lands where it was told to. Asserting the
    escape — rather than leaving it undocumented, or asserting a bound that is
    not there — is what makes that column a checked claim instead of a hopeful
    sentence.

    If this ever fails because the write was *contained*, the fix is not to
    delete it: something now confines at this rung, and the table, `ph doctor`
    and E13 all have to say so.

    The target is under `tmp_path` and outside the worktree, which is as far
    outside as a test may honestly reach.
    """
    ctx, _session, _agent, workspace = await worktree_agent(mount, tmp_path)
    outside = tmp_path / "escaped.txt"

    code, output = await _write(ctx, workspace, str(outside))

    assert code == 0, output
    assert outside.read_text(encoding="utf-8") == "written", (
        "the worktree tier contained an absolute-path write — if that is now "
        "true, the tier table and ph doctor are wrong and E13 must be revised"
    )


# --------------------------------------------------------------------- buys --


@needs_git
async def test_an_ephemeral_tree_is_discarded_with_its_work(mount: Any, tmp_path: Path) -> None:
    """Revertibility, through the path a real agent takes.

    `test_workspace_git.py` asserts the same discard through
    `ctx.workspace.dispose(agent_id)`; this one goes through `ctx.agents.dispose`,
    which is the I2 unwinding a live deployment actually uses and the only route
    that shows the workspace is an *effect of the agent's scope*. And what the
    child was asked to produce survives, because scratch is not part of the
    thing being discarded (E5).
    """
    ctx, _session, parent, _workspace = await worktree_agent(mount, tmp_path)
    child_session = ctx.sessions.create("child")
    # `parent=`: a child's scope nests inside its parent's (P6-27), and this
    # test is about a child's *rung*, so it must be shaped like a real one.
    child = ctx.agents.create(child_session, parent.options, parent=parent)
    ephemeral = await ctx.workspace.acquire(
        session_id=child_session.id,
        agent_id=child.id,
        base=tmp_path / "repo",
        access="read",
        session=child_session,
        # The agent's own scope, as `workspace-lifecycle` passes it. Without it
        # the tree is an effect of the *seam* and outlives the agent — the trap
        # this test would otherwise walk into silently.
        scope=child.ctx,
    )
    assert ephemeral.kind == "worktree-ephemeral"
    code, output = await _write(ctx, ephemeral, "draft.txt")
    assert code == 0, output

    await ctx.agents.dispose(child.id)

    assert not ephemeral.root.exists(), "an ephemeral tree survived with work in it"
    assert ephemeral.scratch.exists(), "scratch went with the tree it was meant to outlive"


# ------------------------------------------------------------ the words used --


def test_the_table_says_what_the_two_writes_above_showed() -> None:
    """The prose and the behaviour, pinned to each other.

    `TIERS` is what `ph doctor` renders verbatim and what P6-06's docs test will
    check, so this is the tripwire for a reword that quietly promises more than
    the rung delivers. Positive claims only: a *negative* substring check would
    trip on a rewording that made the sentence more honest and pass one that
    made it less, which is E13's polarity backwards.
    """
    assert "absolute-path raw write" in TIERS["worktree"].does_not_bound
    assert "collision isolation" in TIERS["worktree"].buys
    assert "confinement" in TIERS["sandbox"].buys


async def test_doctor_reports_a_real_tier_as_the_rung_in_force(mount: Any) -> None:
    """E10 against the shipped provider rather than a stub — the one thing
    `test_diagnostics.py` cannot say, since it registers `StubWorkspaceProvider`.

    No repository and no acquire: `effective_tier` reads the registered provider
    and the containment choice, so "a tier that actually served" is not a state
    `describe()` can distinguish, and building one would be scenery. The row
    mounts a provider without running git, so this needs no git binary either.
    """
    ctx = await mount(*WORKTREE_ROWS)

    assert report_section(ctx, "Containment")["tier (effective)"] == "worktree"
