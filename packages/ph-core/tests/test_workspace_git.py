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

## Why the decline code is read off git's state and not its stderr

The first version matched `"already used by worktree"` in the message. That string
is gettext-translated — `ctx.subprocess` passes `LANG`/`LC_ALL` through, since they
are not credential-shaped — so on a non-English host **every decline collapsed to
the generic code**, in the row whose entire purpose is telling an operator why.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ph.seams.subprocess import SubprocessSpawnSpec, scrub_env
from ph.seams.workspace import redirection_env, workspace_survivors
from ph.seams.workspace_git import sanitize_ref, tree_hash
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


async def test_a_dirty_worktree_is_committed_to_its_branch_and_then_removed(
    mount: Any, tmp_path: Path
) -> None:
    """The policy, and both halves of it: an agent that changed nothing leaves
    nothing behind, and an agent that did work leaves a **branch** to inspect and
    merge — never a directory.

    The directory is a resource the agent borrowed; the branch is the artifact. A
    checkout left on disk is an orphan: nothing enumerates it, nothing collects it,
    and its contents are invisible to `export`, `merge` and every other verb
    `/workspaces` offers, all of which name refs. It is also the thing a crash
    destroys — a branch is not.

    This is what the overlay tier had always done ("the worktree goes, the branch
    stays"); the worktree tier was the outlier, and disposal is now one policy at
    both tiers.

    `kept` rides `workspace/disposed` to separate "nothing changed, so it was
    removed" from "this branch holds work" — neither is derivable from the kind.
    Note that it needs no bookkeeping of its own: `git branch -d` refuses an
    unmerged branch, and *that refusal is the fact*.
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
    assert not dirty.root.exists(), "the checkout outlived the agent that borrowed it"

    # The work is not gone — it moved to where a crash cannot reach it.
    code, out, _ = await git(ctx, base, "show", f"{dirty.ref}:work.txt")
    assert code == 0 and out == "real work\n"

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
    """Disposal reads `git status` to decide whether to commit, and a commit empties it.

    So an agent that finished its work properly presents as *clean*, takes no
    commit of its own, and reaches the removal with everything already on the
    branch — where force-deleting would discard exactly what disposal exists to
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


async def test_a_worktree_is_rebuilt_from_its_branch_with_the_work_on_it(
    mount: Any, tmp_path: Path
) -> None:
    """The round trip the whole policy is for, and the reason a crash is survivable.

    Disposal takes the checkout back and leaves the branch, so re-acquiring has to
    rebuild the directory *with the work in it* — otherwise "the branch is the
    artifact" is a claim about storage rather than about recovery. `git worktree
    add` attaches an existing branch rather than resetting it (`-B` would silently
    drop it), which is what makes this hold.

    An agent that died mid-run reaches this same path: `reclaim` runs the same
    disposal, so the branch holds what the tree held when the process stopped, and
    starting the agent again hands it back its own work. What a crash costs is a
    directory.
    """
    ctx, base = await _tiered(mount, tmp_path)
    first = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")
    (first.root / "half-done.txt").write_text("interrupted here\n", encoding="utf-8")

    await ctx.workspace.dispose("a1")
    assert not first.root.exists()

    second = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")

    assert second.root == first.root
    assert (second.root / "half-done.txt").read_text(encoding="utf-8") == "interrupted here\n", (
        "the rebuilt worktree did not come back with the work its branch holds"
    )


async def test_an_existing_worktree_is_reused_rather_than_recreated(
    mount: Any, tmp_path: Path
) -> None:
    """A resume finds this agent's own tree, and recreating it would discard the
    uncommitted work disposal exists to put on the branch."""
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

    outcome = await ctx.subprocess.run(
        SubprocessSpawnSpec(
            argv=("python", "-m", "pytest", "-q", "test_sample.py"),
            cwd=workspace.root,
            env=scrub_env(extra=workspace.env),
        )
    )
    assert outcome.exit_code == 0, outcome.stdout + outcome.stderr

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

    assert ctx.workspace.effective_tier(child=False) == "worktree"


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
    assert not workspace.root.exists()
    code, out, _ = await git(ctx, base, "show", "ph/s1/a1:real-work.txt")
    assert code == 0 and out == "the agent did this\n", "the work beside the materials was lost"


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


async def test_the_tree_hash_changes_only_when_the_work_does(mount: Any, tmp_path: Path) -> None:
    """P5-07's fingerprint, and P4-09's restore point, are one derivation.

    A content address: two identical trees hash identically and any edit changes
    it. That property is what lets a quality gate be skipped — a gate that
    failed against this exact tree cannot have changed its mind — so it is
    asserted here, where the git tier lives, rather than trusted from the seam
    that consumes it.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    first = await tree_hash(ctx, workspace)
    assert first, "a worktree could not be fingerprinted"
    assert await tree_hash(ctx, workspace) == first, "the hash moved with nothing changed"

    (workspace.root / "new.txt").write_text("work\n", encoding="utf-8")
    changed = await tree_hash(ctx, workspace)
    assert changed and changed != first, "an edit did not change the fingerprint"

    (workspace.root / "new.txt").unlink()
    assert await tree_hash(ctx, workspace) == first, "undoing the edit did not restore the hash"


async def test_a_workspace_with_no_checkout_has_no_fingerprint(mount: Any, tmp_path: Path) -> None:
    """`shared` is the process's own directory, so there is nothing to hash.

    `None` rather than a made-up value: an empty fingerprint that compared equal
    to another empty one would let a gate answer from a run against somebody
    else's tree.
    """
    ctx = await mount()
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=tmp_path, access="write"
    )
    assert workspace.kind == "shared"
    assert await tree_hash(ctx, workspace) is None


# --- P6-28: a settled child's tree is the evidence ---------------------------


async def test_a_retained_ephemeral_tree_survives_the_kind_that_discards_it(
    mount: Any, tmp_path: Path
) -> None:
    """P6-28's mechanism, against the promise it makes an exception to.

    `worktree-ephemeral` discards even when dirty — the test directly above pins
    that, and it is why a child that *failed* leaves its parent diagnosing from a
    transcript. `Workspace.retained` is the exception, and it is an exception to
    `discard`, not to the whole policy: a retained tree takes the same
    commit-then-remove path an ordinary `worktree` does.

    **So retention buys a branch, not a directory** — and that is a strictly better
    thing to buy. The evidence a parent needs to diagnose a cancelled child now
    survives the crash of the process holding it, is reachable by `export` and
    `merge` like any other artifact, and costs a ref instead of a checkout. What
    would be wrong is retaining the *tree*: nothing enumerates an orphaned
    directory, and it is exactly what a crash destroys.

    A retained ephemeral branch is visible where an unretained one is deleted
    outright, and that is the exception working as intended rather than a leak in
    the kind's promise: nothing is merged, so the writes still reach nobody, and
    the retention is a logged act with a reason attached.

    The reason rides `workspace/disposed`, which lets a fold tell three outcomes
    apart: a deliberate keep says why, an ordinary keep is `kept` with no reason,
    and a leak has no closing event at all.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )
    (workspace.root / "evidence.txt").write_text("what the child was doing\n", encoding="utf-8")

    assert ctx.workspace.retain("a1", "error") is True
    await ctx.workspace.dispose("a1")

    assert not workspace.root.exists(), "retention kept the directory instead of the work"
    code, out, _ = await git(ctx, base, "show", "ph/s1/a1:evidence.txt")
    assert code == 0 and out == "what the child was doing\n", (
        "the evidence a parent needs to inspect was discarded with the checkout"
    )
    closed = [one for one in session.events if one.type == "workspace/disposed"]
    assert [one.data.get("retained") for one in closed] == ["error"]
    assert [one.data.get("kept") for one in closed] == [True]


async def test_retention_is_refused_once_the_scope_is_gone(mount: Any, tmp_path: Path) -> None:
    """The window, stated because the teardown path is outside it (§5 rule 6).

    `retain` marks a *held* workspace, and a workspace is held only while the
    scope that acquired it is alive. On `ph_rlm`'s `parent-teardown` path the
    child's scope is already gone by the time the provider learns the child was
    cancelled — `Context.dispose` unwinds `_children` before its own effects, so
    the tree is released *before* the settle handler runs.

    So a policy that retains the evidence of a cancelled child cannot call this
    from that handler; it has to mark while the child is live. `False` rather
    than a raise, because the ordinary caller is a settle path that does not know
    whether this tier hands out trees at all.
    """
    ctx, base = await _tiered(mount, tmp_path)
    await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="read")

    assert ctx.workspace.retain("a1", "error") is True, "a live agent can be marked"
    await ctx.workspace.dispose("a1")
    assert ctx.workspace.retain("a1", "too late") is False, "the window closes with the scope"
    assert ctx.workspace.retain("never-existed", "probe") is False


async def test_the_mark_is_written_the_moment_it_is_made(mount: Any, tmp_path: Path) -> None:
    """Why retention is its own event rather than only a field on the closing half.

    The decision is made *because* a run went wrong, and the most complete way
    for one to go wrong is for the process to die — which writes no `disposed` at
    all. A mark held only in memory would be lost by exactly the failure it
    exists to survive, and reconciliation would then discard the tree it had been
    told to keep.

    The withdrawal is the same event with an empty reason, because it is the same
    decision revisited: the shipped policy marks at acquire, so a clean settle is
    a caller saying "never mind" and that has to be as durable as the mark.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )

    ctx.workspace.retain("a1", "the child was cancelled")
    marks = [one.data.get("retained") for one in session.events if one.type == "workspace/retained"]
    assert marks == ["the child was cancelled"]
    (open_record,) = workspace_survivors(session)
    assert (open_record.outcome, open_record.closed) == ("retained", False), (
        "a crash here leaves a record that is both retained and unclosed"
    )

    ctx.workspace.retain("a1", "")
    assert workspace_survivors(session)[0].outcome == "leaked", "the withdrawal is durable too"


async def test_reconciliation_honours_a_retention_the_way_release_does(
    mount: Any, tmp_path: Path
) -> None:
    """One rule for one word, across both paths that can end a tree.

    This had a rule of its own for one round — a reason meant "touch nothing" —
    while an orderly release treated it as an exception to `discard` and so kept
    a retained tree only when it was **dirty**. Two rules for one word is how a
    tree gets deleted by whichever path happened to reach it first, so the
    special case is gone and a reason folds into `discard` here as it does there.

    The uncommitted work below is the evidence, and it reaches the branch on the
    crash path exactly as it does on a clean shutdown — which is the property that
    makes a crash cost a directory rather than a day's work. What a child
    committed itself needs no retention at all: it is already on the branch, which
    `-d` declines to delete.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )
    (workspace.root / "evidence.txt").write_text("what the child was doing\n", encoding="utf-8")
    ctx.workspace.retain("a1", "the child was cancelled")
    (record,) = workspace_survivors(session)

    assert await ctx.workspace.provider.reclaim(record) is True
    assert not workspace.root.exists(), "reconciliation left the checkout behind"
    code, out, _ = await git(ctx, base, "show", "ph/s1/a1:evidence.txt")
    assert code == 0 and out == "what the child was doing\n", (
        "the crash path discarded the evidence"
    )


async def test_disposal_leaves_the_collector_nothing_to_collect(mount: Any, tmp_path: Path) -> None:
    """What the age-bounded stay of execution is *for*, once nothing is retained on disk.

    P6-28's collector was written when a retained tree was a directory, and its job
    was disk pressure: an evidence checkout nobody came back for is a copy of the
    repository that outlives its reason. Disposal now commits and removes, so on the
    ordinary path there is no directory to age out and every verdict is `gone`.

    **That is a narrowing, not a hole, and what changed is what gets retained.** The
    branches below survive the collector, and should: a branch is the artifact, it
    costs a ref rather than a checkout, and a garbage collector that deleted the
    thing disposal just went to the trouble of saving would be the bug this row
    exists to prevent. What still reaches `collect` is a tree disposal *could not*
    remove — the case pinned at the end — which is now the only way a directory
    outlives its agent.

    Not enforced (§5 rule 6): retained branches have no age bound at all. Bounding
    them needs a record to know its repository, and it does not — a
    `WorkspaceRecord` locates a *tree*, and once the tree is gone nothing on it can
    find the ref's repo. `/workspaces remove` is the deliberate verb meanwhile.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    ephemeral = await ctx.workspace.acquire(
        session_id="s1", agent_id="gone", base=base, access="read", session=session
    )
    ordinary = await ctx.workspace.acquire(
        session_id="s1", agent_id="dirty", base=base, access="write", session=session
    )
    for agent_id, workspace in (("gone", ephemeral), ("dirty", ordinary)):
        (workspace.root / "work.txt").write_text("uncommitted\n", encoding="utf-8")
        ctx.workspace.retain(agent_id, "the child was cancelled")
        await ctx.workspace.dispose(agent_id)

    records = workspace_survivors(session)
    rows = ctx.workspace.collectable(records, older_than=1.0, now=1e9, touched={"s1": 0.0})
    assert sorted(one.verdict for one in rows) == ["gone", "gone"]
    assert await ctx.workspace.collect(rows) == []

    # Nothing was destroyed: both agents' work is on their branches, the ephemeral
    # one included — its retention is what bought it a branch rather than a `-D`.
    for agent_id in ("gone", "dirty"):
        code, out, _ = await git(ctx, base, "show", f"ph/s1/{agent_id}:work.txt")
        assert code == 0 and out == "uncommitted\n", f"{agent_id} lost its evidence"

    # The one live subject left: a checkout disposal could not remove. Retention
    # still buys it a stay of execution, and the stay still expires.
    stranded = ordinary.root
    stranded.mkdir(parents=True, exist_ok=True)
    (stranded / "left.txt").write_text("disposal could not remove this\n", encoding="utf-8")
    stuck = replace(records[0], agent_id="stuck", root=stranded, ref=None)
    (row,) = ctx.workspace.collectable([stuck], older_than=1.0, now=1e9, touched={"s1": 0.0})
    assert row.verdict == "collect"


async def test_a_provisioned_secret_is_not_committed_to_the_branch(
    mount: Any, tmp_path: Path
) -> None:
    """The risk disposal took on the day it started committing, pinned at its edge.

    A provisioned file the project does **not** gitignore is the sharp case. Under
    the old policy it was merely dirt — `git status` called the tree dirty, the tree
    was kept, and the worst outcome was a checkout nobody wanted. Now the same file
    is one `git add -A` away from a ref somebody merges, which turns a
    provisioning convenience into a credential in the project's history.

    So the exclusion is not only about `kept` reading correctly any more, and this
    asserts the stronger property: the material is *on disk* in the tree (the agent
    needs it) and *absent from* the commit, while the work beside it survives.

    `git show` on the ref rather than a `status` count, because what is being pinned
    is the content of the artifact — the one thing a person merges.
    """
    ctx = await mount(
        TIER_ROW,
        {"id": "workspace-lifecycle", "config": {"provision": [{"source": "secret.env"}]}},
    )
    base = await git_repo(ctx, tmp_path / "repo")
    (base / "secret.env").write_text("TOKEN=shhh\n", encoding="utf-8")
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    assert (workspace.root / "secret.env").exists(), "the agent never got the material"
    (workspace.root / "real-work.txt").write_text("the agent did this\n", encoding="utf-8")

    await ctx.workspace.dispose("a1")

    code, out, _ = await git(ctx, base, "show", "ph/s1/a1:real-work.txt")
    assert code == 0 and out == "the agent did this\n", "the work was not committed"
    code, _, _ = await git(ctx, base, "show", "ph/s1/a1:secret.env")
    assert code != 0, "a provisioned credential was committed to a branch somebody merges"
