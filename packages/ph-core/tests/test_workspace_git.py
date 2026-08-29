"""P4-08 — the `worktree` tier against a real repository (D21, E2, E3, E12).

Real `git`, real checkouts, no mocks. A provider whose whole job is to drive
`git worktree` cannot be tested against a fake that agrees with whatever the
implementation does: the interesting failures — a branch that already exists, a
tree that is dirty only because `pytest` ran, a prune that never happened — are
all git's behaviour, not ours.

**What these pin is isolation and revertibility, never confinement.** An
absolute-path write escapes a worktree and is supposed to; only the `sandbox`
tier refuses it (E13). A test here asserting otherwise would be the tier-table
regression §12 Q10 exists to prevent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.seams.subprocess import SubprocessSpawnSpec, scrub_env
from ph.seams.workspace import redirection_env
from ph.seams.workspace_git import sanitize_ref
from ph.testing import git, git_repo, needs_git

pytestmark = [pytest.mark.anyio, needs_git]


TIER_ROW = {"insert": [{"id": "workspace-git-worktree", "name": "workspace-git-worktree"}]}
"""The row under test. Mounted rather than hand-assembled, so a typo in the entry
point fails here rather than in someone's profile — and so the worktree and
scratch roots come from `$PH_HOME`, which the `mount` fixture already points at
`tmp_path`, instead of being spelled a second time."""


async def _tiered(mount: Any, tmp_path: Path) -> tuple[Any, Path]:
    """A mounted profile with the tier on, and a repository to point it at.

    The repository goes under `tmp_path`, never `ctx.fs.root` — that is the
    *process's* directory, which for a test run is this checkout. A `base` taken
    from it would have every test in this file initialising a repository inside
    pH's own tree and sharing one branch namespace.
    """
    ctx = await mount(TIER_ROW)
    return ctx, await git_repo(ctx, tmp_path / "repo")


# ------------------------------------------------------------------ acquire --


async def test_a_write_agent_gets_its_own_checkout_on_its_own_branch(
    mount: Any, tmp_path: Path
) -> None:
    """E2, one half. The branch name is `ph/<session>/<agent>` because a person
    reading `git branch` after the fact has to be able to tell whose work it is."""
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert workspace.kind == "worktree"
    assert workspace.ref == "ph/s1/a1"
    assert workspace.root != base
    assert (workspace.root / "README.md").read_text(encoding="utf-8") == "base\n"
    # Writable, and honestly so: the checkout is the agent's to change.
    assert workspace.repo_writable is True


async def test_two_children_work_without_collision_and_merge_back(
    mount: Any, tmp_path: Path
) -> None:
    """E2, and the reason the tier exists.

    Eight children fanning out into one checkout is the hazard dsh's own
    `agent-team` lists as a known limitation. Here each child writes its own
    file in its own tree, and the parent merges both without a conflict —
    reviewing a diff instead of trusting sibling writes.
    """
    ctx, base = await _tiered(mount, tmp_path)

    one = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")
    two = await ctx.workspace.acquire(session_id="s1", agent_id="a2", base=base, access="write")

    # Concurrent, in the only sense that matters here: both trees are live at
    # once, and neither write is visible to the other.
    (one.root / "one.txt").write_text("from a1\n", encoding="utf-8")
    (two.root / "two.txt").write_text("from a2\n", encoding="utf-8")
    assert not (two.root / "one.txt").exists()

    for workspace, agent in ((one, "a1"), (two, "a2")):
        await git(ctx, workspace.root, "add", "-A")
        await git(ctx, workspace.root, "commit", "-m", f"work from {agent}")

    for ref in ("ph/s1/a1", "ph/s1/a2"):
        code, _, err = await git(ctx, base, "merge", "--no-edit", ref)
        assert code == 0, err

    assert (base / "one.txt").exists()
    assert (base / "two.txt").exists()


async def test_a_read_agent_gets_an_ephemeral_checkout_it_may_still_write(
    mount: Any, tmp_path: Path
) -> None:
    """E3. `worktree` cannot enforce read-only, so `access="read"` buys a
    different *kind* rather than a permission: writes happen and reach nobody.

    `repo_writable` stays `True`, which is the honest answer — a `False` here
    would describe a guarantee only the sandbox tier can make, and a caller
    would act on it.
    """
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read"
    )

    assert workspace.kind == "worktree-ephemeral"
    assert workspace.repo_writable is True
    (workspace.root / "notes.txt").write_text("scratch thinking\n", encoding="utf-8")


async def test_a_non_repository_declines_and_the_seam_falls_back(
    mount: Any, tmp_path: Path
) -> None:
    """E10. Half the directories a person runs pH in are not repositories, so
    declining must be a notice rather than a refusal to start — and the seam
    then reports the *effective* tier as advisory in everything but its config.
    """
    ctx = await mount(TIER_ROW)
    base = tmp_path / "plain"
    base.mkdir()

    workspace = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base)

    assert workspace.kind == "shared"
    assert workspace.root == base


# ------------------------------------------------------------------ disposal --


async def test_a_clean_worktree_is_removed_and_a_dirty_one_is_kept(
    mount: Any, tmp_path: Path
) -> None:
    """The policy, and both halves of it: an agent that changed nothing leaves
    nothing to clean up, and an agent that did work leaves it for the user to
    inspect and merge.

    `kept` rides `workspace/disposed` for exactly this reason — "nothing changed,
    so it was removed" and "these writes were thrown away" are different facts
    and a reader cannot derive either from the kind.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")

    clean = await ctx.workspace.acquire(
        session_id="s1", agent_id="clean", base=base, access="write", session=session
    )
    await ctx.workspace.dispose("clean")
    assert not clean.root.exists()

    dirty = await ctx.workspace.acquire(
        session_id="s1", agent_id="dirty", base=base, access="write", session=session
    )
    (dirty.root / "work.txt").write_text("real work\n", encoding="utf-8")
    await ctx.workspace.dispose("dirty")
    assert (dirty.root / "work.txt").exists()

    kept = {
        event.data["agentId"]: event.data["kept"]
        for event in session.events
        if event.type == "workspace/disposed"
    }
    assert kept == {"clean": False, "dirty": True}


async def test_an_ephemeral_worktree_is_discarded_even_when_dirty(
    mount: Any, tmp_path: Path
) -> None:
    """E3's second half, and the promise the kind is named for.

    A `worktree-ephemeral` child that wrote is *still* discarded — that is what
    "writes reach nobody" means. The parent's branch is untouched either way.
    """
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read"
    )
    (workspace.root / "discarded.txt").write_text("nobody reads this\n", encoding="utf-8")

    await ctx.workspace.dispose("a1")

    assert not workspace.root.exists()
    assert not (base / "discarded.txt").exists()
    _, out, _ = await git(ctx, base, "branch", "--list", "ph/s1/a1")
    assert out.strip() == "", "the ephemeral branch is deleted with its tree"


async def test_committed_work_survives_disposal_of_a_clean_worktree(
    mount: Any, tmp_path: Path
) -> None:
    """The keep-dirty rule reads `git status`, and a commit empties it.

    So an agent that finished its work properly presents as *clean* — and
    force-deleting its branch there would discard exactly what the rule exists to
    protect. The tree goes (it is reconstructible from the branch); the branch
    stays, because git refuses to drop an unmerged one, and `kept` says so.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (workspace.root / "work.txt").write_text("finished\n", encoding="utf-8")
    await git(ctx, workspace.root, "add", "-A")
    await git(ctx, workspace.root, "commit", "-m", "the child's work")

    await ctx.workspace.dispose("a1")

    _, branches, _ = await git(ctx, base, "branch", "--list", "ph/s1/a1")
    assert branches.strip().endswith("ph/s1/a1"), "the branch holding the work was deleted"
    (disposed,) = [event for event in session.events if event.type == "workspace/disposed"]
    assert disposed.data["kept"] is True


async def test_disposal_leaves_the_repository_able_to_re_acquire(
    mount: Any, tmp_path: Path
) -> None:
    """The state a resume finds has to be usable.

    Git refuses to add a worktree over a registration it still holds, so a
    provider that removed directories without deregistering them would decline
    for the rest of the repository's life — which the seam would report as
    `shared` and nobody would understand why.
    """
    ctx, base = await _tiered(mount, tmp_path)

    first = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="read")
    await ctx.workspace.dispose("a1")
    second = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="read")

    assert second.kind == "worktree-ephemeral"
    assert second.root == first.root


async def test_an_existing_worktree_is_reused_rather_than_recreated(
    mount: Any, tmp_path: Path
) -> None:
    """A resume finds this agent's own tree, and recreating it would discard the
    work the keep-dirty policy exists to preserve."""
    ctx, base = await _tiered(mount, tmp_path)

    first = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")
    (first.root / "in-progress.txt").write_text("half done\n", encoding="utf-8")

    second = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")

    assert second.root == first.root
    assert (second.root / "in-progress.txt").read_text(encoding="utf-8") == "half done\n"


# ---------------------------------------------------------------- the env --


async def test_the_redirection_env_keeps_a_test_run_out_of_the_tree(
    mount: Any, tmp_path: Path
) -> None:
    """E12, at this tier: `pytest` writes `.pytest_cache/` and `__pycache__/`
    into the tree it runs against.

    Without redirection every worktree an agent ran tests in would be dirty, so
    every one would be *kept* — and "remove a clean worktree" would quietly stop
    meaning anything. This asserts the policy survives a test run, which is the
    phase-4 form of the gate; the read-only repo it also enables arrives with the
    sandbox tier (E11).
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    # Not vacuous: a declined tier would hand back the shared checkout with an
    # empty `env`, and every assertion below would then pass for the wrong
    # reason. This test caught exactly that once.
    assert workspace.kind == "worktree"

    (workspace.root / "test_sample.py").write_text("def test_ok():\n    assert True\n", "utf-8")
    await git(ctx, workspace.root, "add", "-A")
    await git(ctx, workspace.root, "commit", "-m", "add a test")

    code, out, err = await ctx.subprocess.run(
        SubprocessSpawnSpec(
            argv=("python", "-m", "pytest", "-q", "test_sample.py"),
            cwd=workspace.root,
            env=scrub_env(extra=workspace.env),
        )
    )
    assert code == 0, out + err

    _, status, _ = await git(ctx, workspace.root, "status", "--porcelain", "-uall")
    assert status.strip() == "", f"the test run dirtied the worktree: {status}"
    assert not (workspace.root / ".pytest_cache").exists()


def test_the_redirection_env_points_everything_inside_scratch(tmp_path: Path) -> None:
    """Every entry, not most of them: one variable left pointing at the default
    is one build tool still writing beside the sources, and the failure looks
    like a flaky dirty-worktree rather than a missing setting."""
    scratch = tmp_path / "scratch"

    env = redirection_env(scratch)

    assert set(env) == {
        "TMPDIR",
        "PYTHONPYCACHEPREFIX",
        "PYTEST_ADDOPTS",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "GIT_CONFIG_GLOBAL",
    }
    for name, value in env.items():
        assert str(scratch) in value, f"{name} escapes scratch"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("s1", "s1"), ("a/b", "a-b"), ("..", "agent"), ("child-9f2c", "child-9f2c")],
)
def test_ref_components_are_an_allow_list(raw: str, expected: str) -> None:
    """Built from what git accepts rather than from what it rejects: a branch
    name that fails validation *after* the worktree exists is a half-made
    artifact somebody has to clean up by hand."""
    assert sanitize_ref(raw) == expected


async def test_mounting_the_row_claims_the_tier(mount: Any) -> None:
    """P4-08's mounted form: the row is the tier, so `ph doctor` reports
    `worktree` from the moment a profile layers it — and `advisory` when one
    does not, which is P4-07's gate and still holds."""
    ctx = await mount(TIER_ROW)

    assert ctx.workspace.tier == "worktree"


# -------------------------------------------------------------- provisioning --


PROVISION_ROW = {
    "id": "workspace-lifecycle",
    "config": {"provision": [{"source": ".env"}, {"source": "deps", "mode": "hardlink"}]},
}
"""Materials on the row `ph-base` already layers, patched rather than inserted.

Composition happens *inside* the row — the profile's list plus the project's
`.ph-workspace.yml` — not by mounting a second copy: `workspace-lifecycle` also
claims `fs.rebase`, and two answers to "where does this agent write" is the
contradiction `claim_slot` exists to refuse.
"""


async def _repo_with_materials(ctx: Any, path: Path) -> Path:
    base = await git_repo(ctx, path)
    (base / ".gitignore").write_text(".env\ndeps/\n", encoding="utf-8")
    (base / ".env").write_text("TOKEN=shhh\n", encoding="utf-8")
    (base / "deps").mkdir()
    (base / "deps" / "lib.py").write_text("VALUE = 1\n", encoding="utf-8")
    await git(ctx, base, "add", "-A")
    await git(ctx, base, "commit", "-m", "ignore the materials")
    return base


async def test_a_worktree_arrives_with_the_materials_a_checkout_lacks(
    mount: Any, tmp_path: Path
) -> None:
    """E14 against real git, which is the only way to show the problem exists:
    `git worktree add` genuinely does not carry a gitignored file, so the child
    starts without `.env` and without its dependencies."""
    ctx = await mount(TIER_ROW, PROVISION_ROW)
    base = await _repo_with_materials(ctx, tmp_path / "repo")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert workspace.kind == "worktree"
    assert (workspace.root / ".env").read_text(encoding="utf-8") == "TOKEN=shhh\n"
    assert (workspace.root / "deps" / "lib.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert workspace.provision_failures == ()


async def test_a_worktree_holding_only_its_materials_is_still_clean(
    mount: Any, tmp_path: Path
) -> None:
    """The property that keeps this row from defeating the tier it serves.

    The project here deliberately does *not* gitignore its materials, so a
    provisioned `.env` and `deps/` are untracked files and `git status` calls the
    tree dirty. If disposal read that at face value, every worktree an agent was
    given materials in would be kept, and "remove a clean worktree, keep a dirty
    one" would decay into "keep everything" — P4-08's `.pytest_cache` finding
    from the other direction.

    Asserted on the *disposal decision* rather than on `git status`, because that
    is the only consumer that should care: the person looking at the worktree
    should still see the files, which is why this is a pathspec at the check and
    not an `info/exclude` write (git has no per-worktree exclude — it resolves
    that path against the common directory).
    """
    ctx = await mount(TIER_ROW, PROVISION_ROW)
    base = await _repo_with_materials(ctx, tmp_path / "repo")
    (base / ".gitignore").write_text("", encoding="utf-8")
    await git(ctx, base, "commit", "-am", "stop ignoring them")
    session = ctx.sessions.create("s1")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    assert (workspace.root / ".env").exists()
    # Untracked as far as git is concerned — the materials really are there.
    _, status, _ = await git(ctx, workspace.root, "status", "--porcelain", "-uall")
    assert ".env" in status

    await ctx.workspace.dispose("a1")

    (disposed,) = [e for e in session.events if e.type == "workspace/disposed"]
    assert disposed.data["kept"] is False, "materials were mistaken for the agent's work"
    assert not workspace.root.exists()


async def test_work_beside_the_materials_still_keeps_the_worktree(
    mount: Any, tmp_path: Path
) -> None:
    """The other direction, and the one that matters more: subtracting the
    materials must not subtract the work sitting next to them."""
    ctx = await mount(TIER_ROW, PROVISION_ROW)
    base = await _repo_with_materials(ctx, tmp_path / "repo")
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (workspace.root / "real-work.txt").write_text("the agent did this\n", encoding="utf-8")

    await ctx.workspace.dispose("a1")

    (disposed,) = [e for e in session.events if e.type == "workspace/disposed"]
    assert disposed.data["kept"] is True
    assert (workspace.root / "real-work.txt").exists()


async def test_nothing_is_provisioned_into_a_shared_workspace(mount: Any, tmp_path: Path) -> None:
    """The root *is* the base, so every material is already in it — and copying
    `.env` onto itself is the one way this could destroy the file it exists to
    provide."""
    ctx = await mount(PROVISION_ROW)
    base = tmp_path / "plain"
    base.mkdir()
    (base / ".env").write_text("TOKEN=shhh\n", encoding="utf-8")

    workspace = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base)

    assert workspace.kind == "shared"
    assert (base / ".env").read_text(encoding="utf-8") == "TOKEN=shhh\n"
    assert workspace.provision_failures == ()


async def test_a_material_that_does_not_arrive_reaches_the_agent(
    mount: Any, tmp_path: Path
) -> None:
    """A failure in a log line reaches an operator tomorrow; this reaches the
    model on the step it matters, because the agent is the one about to wonder
    why the tests fail (E14)."""
    ctx = await mount(
        TIER_ROW,
        {"id": "workspace-lifecycle", "config": {"provision": [{"source": "../outside"}]}},
    )
    base = await git_repo(ctx, tmp_path / "repo")
    session = ctx.sessions.create("s1")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, session=session
    )

    assert len(workspace.provision_failures) == 1
    (event,) = [e for e in session.events if e.type == "workspace/provisioned"]
    assert event.data["agentId"] == "a1"


# ------------------------------------------------------------------ declines --


async def test_a_decline_says_which_one_it_was(mount: Any, tmp_path: Path) -> None:
    """E15. A fallback that cannot say *why* is indistinguishable from "no tier
    configured", which is the confusion `ph doctor` exists to remove: an operator
    who set `worktree` and got `shared` is owed the reason."""
    ctx = await mount(TIER_ROW)
    base = tmp_path / "plain"
    base.mkdir()
    session = ctx.sessions.create("s1")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, session=session
    )

    assert workspace.kind == "shared"
    (event,) = [e for e in session.events if e.type == "workspace/acquired"]
    assert event.data["declined"] == "not-a-repository"


async def test_no_tier_configured_is_not_a_decline(mount: Any, tmp_path: Path) -> None:
    """The distinction the field exists for: `advisory` because nobody asked for
    containment is a different fact from `advisory` because the tier could not
    serve, and a doctor that ran them together would be useless."""
    ctx = await mount()
    session = ctx.sessions.create("s1")

    await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=tmp_path, session=session)

    (event,) = [e for e in session.events if e.type == "workspace/acquired"]
    assert "declined" not in event.data
