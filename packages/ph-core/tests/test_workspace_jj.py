"""The `worktree` tier over Jujutsu, against a real repository (D21, E2, E3, E12).

Real `jj`, real workspaces, no mocks — `test_workspace_git.py`'s argument, and it
applies harder here. Every interesting thing this provider relies on is jj's own
behaviour and none of it is obvious from the documentation: that a workspace
forked from the working-copy commit is *rebased and staled* by the parent's next
keystroke, that a bookmark follows its commit through a snapshot without being
told to, that `git export` from a secondary workspace reaches the colocated repo.
Each of those was measured before it was coded, and a fake that agreed with the
implementation would have pinned the wrong one twice.

**What these pin is isolation and inheritance, never confinement.** An
absolute-path write escapes a jj workspace exactly as it escapes a git worktree,
and is supposed to (E13). A test here asserting otherwise would be the tier-table
regression §12 Q10 exists to prevent.

## Why the decline code is read off jj's state and not its stderr

`workspace_git`'s finding, adopted rather than re-learned: a message match is a
guess about wording nobody promised to keep, and it fails in the one direction
that matters — the row whose whole purpose is telling an operator *why* collapses
to the generic code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.keys import AGENTS, COMMANDS, SESSIONS, WORKSPACE
from ph.seams.workspace import CHECKPOINT, WorkspaceRecord
from ph.seams.workspace_jj import _needs_the_tree
from ph.testing import FAKE_OPTIONS, MountProfile
from ph.testing.git import git
from ph.testing.jj import jj, jj_agent, jj_repo

pytestmark = [pytest.mark.anyio, pytest.mark.needs_jj]


TIER_ROW = {"insert": [{"id": "workspace-jj", "name": "workspace-jj"}]}
"""The row under test. Mounted rather than hand-assembled, so a typo in the entry
point fails here rather than in someone's profile — and so the workspace and
scratch roots come from `$PH_HOME`, which the `mount` fixture already points at
`tmp_path`."""


async def _tiered(mount: MountProfile, tmp_path: Path, *extra: dict[str, Any]) -> tuple[Any, Path]:
    """A mounted profile with the tier on, and a colocated repository to point it at.

    The repository goes under `tmp_path`, never `ctx.fs.root` — that is the
    *process's* directory, which for a test run is this checkout. A `base` taken
    from it would have every test here initialising jj inside pH's own tree.
    """
    ctx = await mount(TIER_ROW, *extra)
    return ctx, await jj_repo(ctx, tmp_path / "repo")


async def _bookmarks(ctx: Context, base: Path) -> str:
    _, out, _ = await jj(ctx, base, "bookmark", "list")
    return out


# ------------------------------------------------------------------ acquire --


async def test_a_write_agent_gets_its_own_workspace_on_its_own_bookmark(
    mount: MountProfile, tmp_path: Path
) -> None:
    """E2, one half. The name is `ph/<session>/<agent>` — the git tier's shape, and
    deliberately so: `/workspaces` enumerates one prefix to find what disposal
    leaves, and two tiers naming their artifacts differently would make it find
    half of them."""
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert workspace.kind == "worktree"
    assert workspace.ref == "ph/s1/a1"
    assert workspace.root != base
    assert (workspace.root / "README.md").read_text(encoding="utf-8") == "base\n"
    assert workspace.repo_writable is True


async def test_a_child_starts_from_the_parents_work_in_progress(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The reason this tier exists**, and the one thing the git tier cannot do.

    The parent has edited a tracked file and added a new one, and has committed
    neither — which is the normal state of a live agent, since the git tier
    commits only at disposal. Under `workspace-git` a child spawned here gets the
    base commit and sees none of this. Under jj it gets the work.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / "README.md").write_text("parent edited this\n", encoding="utf-8")
    (base / "wip.txt").write_text("parent's work in progress\n", encoding="utf-8")

    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert (workspace.root / "wip.txt").read_text(encoding="utf-8") == "parent's work in progress\n"
    assert (workspace.root / "README.md").read_text(encoding="utf-8") == "parent edited this\n"


async def test_the_parent_may_keep_working_without_staling_its_children(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The trap `_fork_point` exists to avoid, pinned as behaviour rather than as a
    comment.

    Forking a child from the parent's working-copy commit is the obvious reading
    of "start from the parent's work", and it is wrong: jj rewrites that commit on
    every snapshot and auto-rebases its descendants, so the child's workspace goes
    **stale** — every `jj` command in it refusing until someone updates it, which
    would then pull the parent's newer work into a tree the child is editing.

    A stale child cannot be released, so this failure would surface as work
    silently not reaching a branch. What proves the fork point is frozen is that
    the child answers a question about itself *after* the parent has moved on.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / "wip.txt").write_text("first\n", encoding="utf-8")
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    # The parent works on, and snapshots — which any jj command does.
    (base / "wip.txt").write_text("second\n", encoding="utf-8")
    (base / "later.txt").write_text("after the spawn\n", encoding="utf-8")
    await jj(ctx, base, "status")

    code, out, err = await jj(ctx, workspace.root, "status")
    assert code == 0, err
    assert "stale" not in err
    assert out
    # And the child kept the state it was given, rather than being dragged forward.
    assert (workspace.root / "wip.txt").read_text(encoding="utf-8") == "first\n"
    assert not (workspace.root / "later.txt").exists()


async def test_two_children_fork_from_one_point_and_do_not_collide(
    mount: MountProfile, tmp_path: Path
) -> None:
    """E2, and the fan-out this tier is bought for.

    Both children see the parent's uncommitted work and neither sees the other's,
    which is the property "eight agents in one checkout" destroys. The parent's
    history gains **one** fork point for the pair, not one per child: the first
    spawn freezes the work and its sibling finds nothing new to freeze.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / "shared.txt").write_text("from the parent\n", encoding="utf-8")

    one = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    two = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a2", base=base, access="write"
    )

    for workspace in (one, two):
        assert (workspace.root / "shared.txt").read_text(encoding="utf-8") == "from the parent\n"
    (one.root / "one.txt").write_text("from a1\n", encoding="utf-8")
    (two.root / "two.txt").write_text("from a2\n", encoding="utf-8")
    assert not (two.root / "one.txt").exists()
    assert not (one.root / "two.txt").exists()

    forks = [
        (await jj(ctx, workspace.root, "log", "--no-graph", "-r", "@-", "-T", "commit_id"))[1]
        for workspace in (one, two)
    ]
    assert forks[0] and forks[0] == forks[1], "one fork point for the pair, not one per child"


async def test_a_read_agent_gets_an_ephemeral_workspace_it_may_still_write(
    mount: MountProfile, tmp_path: Path
) -> None:
    """E3. This tier cannot enforce read-only any more than the git tier can, so
    `access="read"` buys a different *kind* rather than a permission: the writes
    happen and reach nobody.

    `repo_writable` stays `True`, which is the honest answer — `False` would
    describe a guarantee only the sandbox tier can make, and a caller would act on
    it.
    """
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="read"
    )

    assert workspace.kind == "worktree-ephemeral"
    assert workspace.repo_writable is True
    (workspace.root / "notes.txt").write_text("scratch thinking\n", encoding="utf-8")


async def test_a_directory_jj_does_not_manage_declines_and_the_seam_falls_back(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The decline that keeps this row safe to layer anywhere.

    **A plain git repository is the case that matters**, not an empty directory:
    it is what most projects are, and converting one would mean writing `.jj/` into
    somebody's tree and changing what their own `git` commands report. So the tier
    declines and the person keeps a shared workspace, rather than acquiring a
    repository layout nobody asked for.
    """
    from ph.testing.git import git_repo

    ctx = await mount(TIER_ROW)
    base = await git_repo(ctx, tmp_path / "plain")

    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert workspace.kind == "shared"
    assert workspace.root == base
    assert not (base / ".jj").exists(), "the tier converted a repository it was only shown"


# ------------------------------------------------------------------ release --


async def test_release_puts_the_childs_work_on_a_git_branch_and_removes_the_tree(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The merge-back story, end to end, and it ends in **git**.

    The bookmark is jj's, the branch is git's, and colocation is what makes them
    the same name. A person who never installs jj still merges what a child
    produced with the `git` they already use — which is the whole claim that lets
    this be a provider swap rather than a migration.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.require(SESSIONS).create("s1")
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (workspace.root / "child.txt").write_text("work from a1\n", encoding="utf-8")

    await ctx.require(WORKSPACE).dispose("a1")

    assert not workspace.root.exists(), "the checkout is a resource, not an artifact"
    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is True
    code, out, err = await git(ctx, base, "show", "--stat", "ph/s1/a1")
    assert code == 0, err
    assert "child.txt" in out
    code, _, err = await git(ctx, base, "merge", "--no-edit", "ph/s1/a1")
    assert code == 0, err
    assert (base / "child.txt").read_text(encoding="utf-8") == "work from a1\n"


async def test_a_parent_that_spawned_a_child_still_keeps_its_own_work(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Data loss, found by building the tier on top of itself.

    Freezing a parent's work so a child can fork from it moves the parent's `@` to a
    fresh empty commit — the work is in `@-` now. A release that asked "is `@`
    non-empty" therefore saw a parent that had done nothing, **deleted its bookmark
    and removed its tree**, and every command on the way succeeded.

    The question a release has to ask is what this workspace has done *since it
    forked*, which is why `acquire` resolves the fork point to a real commit instead
    of leaving it as `@-`.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.require(SESSIONS).create("s1")
    parent = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="parent", base=base, access="write", session=session
    )
    (parent.root / "parent-work.txt").write_text("hours of it\n", encoding="utf-8")

    # Spawning is what empties the parent's tip.
    await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="child", base=parent.root, access="write", session=session
    )
    await ctx.require(WORKSPACE).dispose("child")
    await ctx.require(WORKSPACE).dispose("parent")

    (disposed,) = [
        one
        for one in session.events
        if one.type == "workspace/disposed" and one.data["agentId"] == "parent"
    ]
    assert disposed.data["kept"] is True, "the parent was reported as having done nothing"
    code, out, err = await git(ctx, base, "show", "ph/s1/parent:parent-work.txt")
    assert code == 0, err
    assert out == "hours of it\n"


async def test_a_child_that_did_nothing_leaves_nothing_behind(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The other half of the disposal policy: a bookmark on an empty commit is
    noise a person then has to work out how to clean up."""
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.require(SESSIONS).create("s1")
    await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )

    await ctx.require(WORKSPACE).dispose("a1")

    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is False
    assert "ph/s1/a1" not in await _bookmarks(ctx, base)


async def test_an_ephemeral_release_discards_the_work_and_the_ref(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`worktree-ephemeral`'s whole promise: the writes happened and reach nobody,
    **even though the tree was dirty**."""
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.require(SESSIONS).create("s1")
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )
    (workspace.root / "throwaway.txt").write_text("nobody sees this\n", encoding="utf-8")

    await ctx.require(WORKSPACE).dispose("a1")

    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is False
    assert "ph/s1/a1" not in await _bookmarks(ctx, base)
    code, _, _ = await git(ctx, base, "rev-parse", "--verify", "refs/heads/ph/s1/a1")
    assert code != 0


async def test_a_retained_ephemeral_workspace_keeps_its_work(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P6-28. Retention is an exception to `discard`, so a child whose run went
    wrong leaves its work on a bookmark to read rather than a directory to trip
    over."""
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.require(SESSIONS).create("s1")
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )
    (workspace.root / "evidence.txt").write_text("what went wrong\n", encoding="utf-8")
    ctx.require(WORKSPACE).retain("a1", "the run failed and this is why")

    await ctx.require(WORKSPACE).dispose("a1")

    assert not workspace.root.exists()
    code, out, err = await git(ctx, base, "show", "--stat", "ph/s1/a1")
    assert code == 0, err
    assert "evidence.txt" in out


async def test_disposal_leaves_the_repository_able_to_re_acquire(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The state a resume finds has to be usable.

    jj refuses to add a workspace under a name it still knows, so a provider that
    removed directories without deregistering them would decline for the rest of
    the repository's life — which the seam reports as `shared`, with nobody able
    to see why.
    """
    ctx, base = await _tiered(mount, tmp_path)

    first = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="read"
    )
    await ctx.require(WORKSPACE).dispose("a1")
    second = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="read"
    )

    assert second.kind == "worktree-ephemeral"
    assert second.root == first.root


async def test_a_workspace_already_at_the_path_is_reused(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The rehydrate path: an agent given its workspace back must find its own work
    in it, not a fresh fork over the top of it."""
    ctx, base = await _tiered(mount, tmp_path)
    first = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (first.root / "half-done.txt").write_text("mid-turn\n", encoding="utf-8")

    second = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert second.root == first.root
    assert (second.root / "half-done.txt").read_text(encoding="utf-8") == "mid-turn\n"


# ------------------------------------------------------------------ reclaim --


async def test_reclaim_recovers_a_crashed_childs_work_from_the_record_alone(
    mount: MountProfile, tmp_path: Path
) -> None:
    """F6. A crash must cost a directory, not a day.

    The reconciling process has none of the acquiring one's state — only what the
    log wrote — so this drives `reclaim` from a `WorkspaceRecord` built by hand.
    The workspace name comes back off `ref`, and every command runs in the tree
    the record names, which is why no repository path is needed or guessed at.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "unsaved.txt").write_text("the process died here\n", encoding="utf-8")
    record = WorkspaceRecord(
        agent_id="a1", kind="worktree", root=workspace.root, ref="ph/s1/a1", session_id="s1"
    )

    kept = await ctx.require(WORKSPACE).provider.reclaim(record)

    assert kept is True
    assert not workspace.root.exists()
    code, out, err = await git(ctx, base, "show", "--stat", "ph/s1/a1")
    assert code == 0, err
    assert "unsaved.txt" in out


async def test_reclaiming_a_tree_that_is_already_gone_says_so(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Both an orderly release and a reconciliation reach disposal, and now that the
    ordinary path removes the tree, the second arrival is the common case. It
    decides nothing — it reports what the first left, which is the bookmark."""
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "work.txt").write_text("done\n", encoding="utf-8")
    record = WorkspaceRecord(
        agent_id="a1", kind="worktree", root=workspace.root, ref="ph/s1/a1", session_id="s1"
    )
    assert await ctx.require(WORKSPACE).provider.reclaim(record) is True

    assert await ctx.require(WORKSPACE).provider.reclaim(record) is False


async def test_work_that_cannot_be_named_keeps_its_tree_instead_of_losing_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**A failure to name cancels the removal, not the work** — the git tier's rule
    at the same point in disposal, and the reason release names the work itself
    rather than trusting the bookmark acquire wrote.

    `kept` is a durable claim that a person can go and find this work. Reporting it
    on the strength of a call made minutes earlier, and then deleting the tree, is
    how a claim becomes a loss. An orphaned directory is worse than a clean
    disposal and far better than that.

    The unnameable ref is contrived — jj refuses a bookmark with a space in it —
    because the realistic causes are not reachable from a test: a repository that
    went read-only under a running agent, a jj that failed mid-write. The *policy*
    is what is pinned, and it is the policy that decides whether a real one costs a
    directory or a day.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "irreplaceable.txt").write_text("hours of work\n", encoding="utf-8")
    record = WorkspaceRecord(
        agent_id="a1", kind="worktree", root=workspace.root, ref="ph/no bookmark", session_id="s1"
    )

    kept = await ctx.require(WORKSPACE).provider.reclaim(record)

    assert kept is True
    assert (workspace.root / "irreplaceable.txt").exists(), "work that reached no ref was deleted"


async def test_export_names_the_bookmark_the_work_is_on(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`/workspaces` asks the seam one question rather than asking which tier
    answered."""
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    record = WorkspaceRecord(
        agent_id="a1",
        kind="worktree",
        root=workspace.root,
        ref=workspace.ref,
        session_id="s1",
    )

    assert await ctx.require(WORKSPACE).export(record) == "ph/s1/a1"


# --------------------------------------------------------------------- tier --


async def test_the_tier_advertises_exactly_what_it_has(mount: MountProfile, tmp_path: Path) -> None:
    """E1, in the one place a person looks to check — and in **both** directions.

    While this tier had no restore points, `buys` said so, and printing the rung's
    stock text would have advertised a mechanism it did not have. Now it has them,
    and the stale sentence would be the same failure reversed: a person asking
    whether `/revert` works here would be told no by the tool while the mechanism
    sat behind it.

    So the row keeps the rung's words *and* names the addition — and what pins it is
    `can_checkpoint` agreeing with the sentence, rather than the sentence alone.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    described = ctx.require(WORKSPACE).provider.describe_tier()

    assert "per-run checkpoints" in described.buys
    assert "no per-run checkpoints" not in described.buys
    assert "work in progress" in described.buys
    assert "absolute-path raw write" in described.does_not_bound
    assert ctx.require(WORKSPACE).can_checkpoint(workspace), (
        "the sentence promises what is not there"
    )


# ------------------------------------------------------------- restore points --


async def test_a_run_can_be_reverted_to_the_state_before_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P4-09 at this tier: capture, damage, put it back.

    A denial settles the whole run (Q9a), which bounds partial state to about one
    cell — this is what makes that cell recoverable. All three shapes of damage a
    cell does are here, because a restore that handled two of them would look right
    in a demo: an edit, a deletion, and a file the run created.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "tracked.txt").write_text("before the run\n", encoding="utf-8")

    token = await ctx.require(WORKSPACE).capture(workspace)
    assert token is not None
    (workspace.root / "tracked.txt").write_text("the run clobbered this\n", encoding="utf-8")
    (workspace.root / "added.txt").write_text("the run made this\n", encoding="utf-8")
    (workspace.root / "README.md").unlink()

    added = await ctx.require(WORKSPACE).restore(workspace, token)

    assert (workspace.root / "tracked.txt").read_text(encoding="utf-8") == "before the run\n"
    assert (workspace.root / "README.md").exists(), "a file the run deleted did not come back"
    assert not (workspace.root / "added.txt").exists()
    assert added == ("added.txt",), "the message would not have named what the run created"


async def test_the_operation_log_keeps_a_restore_point_reachable(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**What replaces the git tier's hidden ref**, and the reason this tier writes none.

    A git tree nothing references is eligible for `gc`, so the git tier has to pin
    each capture under `refs/.../pre-run/`. Here every jj command records an
    operation and a commit an operation refers to is not collected until that
    operation is — so the restore point survives by the same mechanism that lets a
    person undo anything jj did.

    Pinned as behaviour rather than as a comment: a lot of work happens between the
    capture and the revert, and each step is a jj command rewriting the working-copy
    commit. If reachability depended on being the current commit, the second capture
    alone would have taken the first out.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "keep.txt").write_text("the state to come back to\n", encoding="utf-8")
    token = await ctx.require(WORKSPACE).capture(workspace)
    assert token is not None

    for step in range(5):
        (workspace.root / f"cell-{step}.txt").write_text(f"{step}\n", encoding="utf-8")
        await ctx.require(WORKSPACE).capture(workspace)

    added = await ctx.require(WORKSPACE).restore(workspace, token)

    assert (workspace.root / "keep.txt").read_text(
        encoding="utf-8"
    ) == "the state to come back to\n"
    assert added == tuple(f"cell-{step}.txt" for step in range(5))


async def test_a_restore_leaves_ignored_files_alone(mount: MountProfile, tmp_path: Path) -> None:
    """A `/revert` that wiped a build cache would turn a recovery into a rebuild.

    Free here rather than arranged: an ignored path was never in the commit, so it
    is not considered in either direction — which is the same reason the git tier's
    `read-tree` never sees one.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / ".gitignore").write_text("build/\n", encoding="utf-8")
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "build").mkdir()
    (workspace.root / "build" / "cache").write_text("expensive\n", encoding="utf-8")
    token = await ctx.require(WORKSPACE).capture(workspace)
    assert token is not None
    (workspace.root / "build" / "cache").write_text("still expensive\n", encoding="utf-8")

    await ctx.require(WORKSPACE).restore(workspace, token)

    assert (workspace.root / "build" / "cache").read_text(encoding="utf-8") == "still expensive\n"


async def test_a_restore_touches_only_its_own_workspace(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Why this is `jj restore --from` and not `jj op restore`.

    The operation log is what keeps a restore point *reachable*, which makes
    `op restore` the obvious way to use it and the wrong one: its unit is the whole
    repository, so reverting one child's bad cell would take every sibling's work
    back with it — in the tier bought precisely so a fan-out does not collide.
    """
    ctx, base = await _tiered(mount, tmp_path)
    one = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    two = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a2", base=base, access="write"
    )
    token = await ctx.require(WORKSPACE).capture(one)
    assert token is not None
    (one.root / "mine.txt").write_text("a1's cell\n", encoding="utf-8")
    (two.root / "theirs.txt").write_text("a2 is still working\n", encoding="utf-8")

    await ctx.require(WORKSPACE).restore(one, token)

    assert not (one.root / "mine.txt").exists()
    assert (two.root / "theirs.txt").read_text(encoding="utf-8") == "a2 is still working\n"


async def test_a_token_names_one_state(mount: MountProfile, tmp_path: Path) -> None:
    """The whole of what a token must promise: **equal tokens mean equal content**.

    That is the direction `unchanged_failure` depends on — a gate that failed
    against this exact state need not run again — and it is the direction this tier
    keeps. The converse it does not: a jj commit id covers the committer timestamp
    as well as the tree, so content that returns to a state it held before gets a
    *new* id, and the cost is a gate re-run rather than a wrong answer.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "work.txt").write_text("one\n", encoding="utf-8")

    first = await ctx.require(WORKSPACE).capture(workspace)
    unchanged = await ctx.require(WORKSPACE).capture(workspace)
    (workspace.root / "work.txt").write_text("two\n", encoding="utf-8")
    changed = await ctx.require(WORKSPACE).capture(workspace)

    assert first is not None and first == unchanged, "the token moved with nothing changed"
    assert changed != first, "an edit did not change the token"


async def test_a_restore_point_that_is_gone_is_reported_rather_than_crashing(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`FileNotFoundError`, matching the git tier, because it is what `/revert`
    catches to say "no longer available" instead of showing a person a traceback for
    a restore point it had offered them.

    Not `"0" * 40`, which the git tier's version of this test uses: in jj that is
    the **root commit** and it resolves, so a restore to it would empty the
    workspace rather than refuse. Unreachable from `capture`, which only ever
    returns a real commit, and a reminder that the token is opaque to everything
    above the tier that made it.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    with pytest.raises(FileNotFoundError):
        await ctx.require(WORKSPACE).restore(workspace, "deadbeef" * 5)


async def test_revert_offers_restore_points_for_a_jj_workspace(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The sentence a person reads, end to end.

    P6-20's refusal is now about the *tier*, so this is the half that would regress
    silently: a jj workspace that could checkpoint but was refused would say "has no
    restore mechanism" and be believed.
    """
    ctx, base = await _tiered(
        mount, tmp_path, {"insert": [{"id": "workspace-revert", "name": "workspace-revert"}]}
    )
    session = ctx.require(SESSIONS).create("s1")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id=session.id, agent_id=agent.id, base=base, access="write", session=session
    )
    (workspace.root / "work.txt").write_text("before\n", encoding="utf-8")
    await ctx.require(WORKSPACE).checkpoint(
        workspace, session=session, agent_id=agent.id, call_id="call-1"
    )
    (workspace.root / "work.txt").write_text("after\n", encoding="utf-8")

    (point,) = [one for one in session.events if one.type == CHECKPOINT]
    shown = str(
        await ctx.require(COMMANDS).dispatch(f"/revert {point.seq}", session=session, agent=agent)
    )

    assert "restored the workspace" in shown, shown
    assert (workspace.root / "work.txt").read_text(encoding="utf-8") == "before\n"


async def test_a_live_jj_workspace_is_refused_by_workspaces_remove(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The refusal that protects an agent still working, which did not fire here.

    `/workspaces` learned `held` from `git worktree list`, and **a jj workspace is
    not a git worktree** — so every jj row read as unheld and
    `remove --with-branch --force-branch` deleted the branch of an agent that was
    mid-turn in it. The verb succeeded, said so, and the work had nowhere left to
    go.

    `held` comes from the seam now, matched on `ref`, which is the one name both
    tiers put on a `Workspace` and carry rather than derive.
    """
    ctx, base, session, agent = await jj_agent(mount)
    held = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (held.root / "mid-turn.txt").write_text("still working\n", encoding="utf-8")

    shown = str(
        await ctx.require(COMMANDS).dispatch(
            "/workspaces remove a1 --with-branch --force-branch", session=session, agent=agent
        )
    )

    assert "still holds this workspace" in shown, shown
    code, _, _ = await git(ctx, base, "rev-parse", "--verify", "refs/heads/ph/s1/a1")
    assert code == 0, "a live agent's branch was deleted"
    assert held.root.exists()


async def test_a_stray_jj_workspace_is_listed_with_its_path_and_can_be_removed(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The gap `ArtifactProvider` closes, from a person's side.

    `/workspaces` joined branches against `git worktree list`, and **a jj workspace
    is not a git worktree** — so a directory disposal could not take back had no
    path in the listing and no verb that could remove it. A person could see that
    the disk was gone and had nothing to do about it.

    The stray is made by asking the *provider* directly, which is what a crash
    leaves: a workspace jj knows about, on disk, with a bookmark, that the seam
    never held and so will never dispose.
    """
    ctx, base, session, agent = await jj_agent(mount)
    stray = await ctx.require(WORKSPACE).provider.acquire(
        session_id="s1", agent_id="a1", base=base, scratch=tmp_path / "scratch"
    )
    assert stray is not None
    (stray.root / "abandoned.txt").write_text("nobody came back\n", encoding="utf-8")

    shown = str(
        await ctx.require(COMMANDS).dispatch("/workspaces list", session=session, agent=agent)
    )
    assert str(stray.root) in shown, shown
    assert "stray" in shown, shown

    removed = str(
        await ctx.require(COMMANDS).dispatch("/workspaces remove a1", session=session, agent=agent)
    )

    assert removed == f"removed {stray.root}", removed
    assert not stray.root.exists()
    # The checkout is a resource; the bookmark is the artifact, and `remove` without
    # `--with-branch` must not touch it.
    code, _, _ = await git(ctx, base, "rev-parse", "--verify", "refs/heads/ph/s1/a1")
    assert code == 0, "remove took the branch as well as the directory"


async def test_the_persons_own_repository_is_never_a_stray(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The root filter, and it is the whole safety of this enumeration.

    `jj workspace list` reports **every** workspace of the repository, and the first
    of them is `default` — the person's own checkout, at the repository root. Without
    the filter it becomes a row like any other, and `remove` on that row would run
    `jj workspace forget default` and then delete the directory: the repository, its
    `.git`, and everything not yet pushed.

    A workspace jj knows about that lives outside this tier's root is somebody's own.
    The tier answers only for what it put where it puts things.
    """
    ctx, base = await _tiered(mount, tmp_path)
    ours = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    mine = tmp_path / "my-own-workspace"
    mine.parent.mkdir(parents=True, exist_ok=True)
    code, _, err = await jj(ctx, base, "workspace", "add", "--name", "mine", str(mine))
    assert code == 0, err

    found = await ctx.require(WORKSPACE).strays(base)

    assert {one.path for one in found.values()} == {ours.root}
    assert base not in {one.path for one in found.values()}, "the repository itself was listed"
    assert mine not in {one.path for one in found.values()}
    # And the verb refuses a path it did not enumerate, rather than trusting a caller.
    assert await ctx.require(WORKSPACE).discard(base) != ""
    assert base.exists() and (base / ".jj").exists()


async def test_a_live_jj_workspace_shows_where_the_disk_went(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`held` rows carry a path now, which is the other half of the same fix.

    An operator wondering where the disk went should see every checkout, and under
    this tier they saw none — every jj row was a bare branch with `-` for a path,
    whether an agent was working in it or not.
    """
    ctx, base, session, agent = await jj_agent(mount)
    held = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )

    shown = str(
        await ctx.require(COMMANDS).dispatch("/workspaces list", session=session, agent=agent)
    )

    assert "held" in shown, shown
    assert str(held.root) in shown, shown
    # **Once**, and a live child is exactly the case that would show it twice: its
    # bookmark tracks the working copy, so it has moved since the last export and
    # `jj bookmark list` renders it once for itself and once for `@git`.
    assert shown.count("ph/s1/a1") == 1, shown


async def test_workspaces_needs_no_git_binary_on_a_jj_host(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**jj embeds its own git, so a jj deployment need not have the binary.**

    `/workspaces` listed branches with `git branch --list ph/*`, and on such a host
    that returned nothing — so the command answered "no agent workspaces are left
    behind" with the bookmarks sitting right there. Silent, and the worst kind of
    wrong: a person is told their agents produced nothing.

    `git` is shadowed by a shim that exits 127 rather than removed from `PATH`, so
    the test can *prove* the binary is unusable — a `PATH` with no git at all would
    also break every other tool the harness needs, and a green test would then say
    nothing about which of them was being exercised.
    """
    ctx, base, session, agent = await jj_agent(mount)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (workspace.root / "work.txt").write_text("the agent did this\n", encoding="utf-8")
    await ctx.require(WORKSPACE).dispose("a1")

    shim = tmp_path / "no-git"
    shim.mkdir()
    (shim / "git").write_text("#!/bin/sh\nexit 127\n", encoding="utf-8")
    (shim / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}:{os.environ['PATH']}")
    broken, _, _ = await git(ctx, base, "--version")
    assert broken != 0, "the shim is not in force, so this test proves nothing"

    shown = str(
        await ctx.require(COMMANDS).dispatch("/workspaces list", session=session, agent=agent)
    )

    assert "ph/s1/a1" in shown, shown
    assert "no agent workspaces are left behind" not in shown
    # Once. A colocated repo has a `git` remote, so `jj bookmark list` reports every
    # bookmark twice and an unfiltered template would give the person two rows per
    # agent — the same work, listed as if it were two.
    assert shown.count("ph/s1/a1") == 1, shown
    # And the verbs work, not just the listing.
    merged = str(
        await ctx.require(COMMANDS).dispatch("/workspaces merge a1", session=session, agent=agent)
    )
    assert merged == "merged ph/s1/a1", merged
    assert (base / "work.txt").read_text(encoding="utf-8") == "the agent did this\n"


async def test_merging_reports_the_conflicts_jj_records_instead_of_claiming_success(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The trap that makes `merge` a tier verb rather than one exit code.

    git exits non-zero on a conflict and refuses. **jj exits zero**, records the
    conflict inside the commit and writes markers into the file — so a caller reading
    the exit code tells the person the merge was clean and lets them find out by
    opening the file. Measured, not assumed: `jj new` on a two-sided conflict returns
    0 with a warning on stderr.
    """
    ctx, base, session, agent = await jj_agent(mount)
    (base / "shared.txt").write_text("original\n", encoding="utf-8")
    child = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (child.root / "shared.txt").write_text("the child's version\n", encoding="utf-8")
    await ctx.require(WORKSPACE).dispose("a1")
    # The person moved the same line while the child was working.
    (base / "shared.txt").write_text("the person's version\n", encoding="utf-8")

    shown = str(
        await ctx.require(COMMANDS).dispatch("/workspaces merge a1", session=session, agent=agent)
    )

    assert "conflict" in shown, shown
    assert "shared.txt" in shown, shown
    assert shown != "merged ph/s1/a1", "a conflicted merge was reported as a clean one"


async def test_removing_a_bookmark_refuses_work_nothing_else_has(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`git branch -d`'s refusal, which jj has no equivalent of and so gets one.

    `jj bookmark delete` always succeeds — there is no `-d`/`-D` distinction — so the
    guard that stands between a person and the work disposal went to the trouble of
    saving had to be built rather than borrowed. The question git answers with "not
    fully merged" is a revset here, and it is the same question: by disposal
    everything the agent did is on that bookmark.
    """
    ctx, base, session, agent = await jj_agent(mount)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (workspace.root / "unmerged.txt").write_text("nobody has this\n", encoding="utf-8")
    await ctx.require(WORKSPACE).dispose("a1")

    refused = str(
        await ctx.require(COMMANDS).dispatch(
            "/workspaces remove a1 --with-branch", session=session, agent=agent
        )
    )

    assert "kept branch ph/s1/a1" in refused, refused
    assert "--force-branch" in refused
    assert "ph/s1/a1" in await ctx.require(WORKSPACE).refs(base)

    forced = str(
        await ctx.require(COMMANDS).dispatch(
            "/workspaces remove a1 --with-branch --force-branch", session=session, agent=agent
        )
    )

    assert forced.endswith("and branch ph/s1/a1"), forced
    assert "ph/s1/a1" not in await ctx.require(WORKSPACE).refs(base)


async def _at(ctx: Context, cwd: Path, template: str = "commit_id") -> str:
    """`@` rendered by `template`, asked *without* committing the working copy.

    A plain `jj log` would commit the tree on the way to answering, which is the very
    thing the tests below check nobody does — so the flag is what makes the
    observation possible, and it is spelled here once rather than at each assertion.
    """
    _, out, _ = await jj(
        ctx, cwd, "--ignore-working-copy", "log", "--no-graph", "-r", "@", "-T", template
    )
    return out.strip()


async def test_reading_does_not_snapshot_the_persons_own_checkout(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The hole `auto_track` could not close, and the reason `snapshot` has no default.**

    `_managed` runs `jj workspace root` in the person's own checkout — a *probe*, by
    name — and `_base_materials` correctly answers `()` there, because nothing was
    provisioned into their repo. So the exclusion had nothing to exclude, and the
    probe committed whatever they had left untracked. No fileset could have fixed
    that: the file was theirs.

    `--ignore-working-copy` is what fixes it, and it is stronger than the exclusion
    it replaces — `auto_track` stops a file *becoming tracked*, this stops the
    command touching the tree at all.

    Driven through `/workspaces list`, which is reads end to end: `bookmark list`
    for the refs and `workspace list` for the checkouts.
    """
    ctx, base, session, agent = await jj_agent(mount)
    (base / "my-own-note.txt").write_text("mid-thought\n", encoding="utf-8")
    before = await _at(ctx, base)

    await ctx.require(COMMANDS).dispatch("/workspaces list", session=session, agent=agent)

    assert await _at(ctx, base) == before, "a read committed the person's working copy"
    # And their file never entered a commit. Asked of `@`'s own diff rather than of
    # `jj status`, which reports *nothing* under `--ignore-working-copy` precisely
    # because it does not scan the tree — the state this test is about is only
    # observable without disturbing it.
    adopted = await _at(ctx, base, 'diff.files().map(|f| f.path()).join(",")')
    assert adopted == "", adopted
    assert (base / "my-own-note.txt").exists(), "the file itself must be untouched"


def test_a_read_of_a_verb_that_needs_the_tree_is_refused() -> None:
    """The veto, and it exists because the loudness it replaces was not real.

    `_read` was justified on the grounds that jj refuses `--ignore-working-copy` on
    anything that must write, *by name*, so a wrong choice would fail loudly.
    Measured against the installed binary, that is true of `workspace add` alone —
    and even there only *after* it registers the workspace, leaving an empty
    directory. `jj --ignore-working-copy new` exits 0 in silence and produces a fork
    point with **none of the parent's work in it**, which is the one property this
    provider exists to offer.

    So the loudness is pH's now. argv cannot *derive* the choice — `log` and
    `bookmark set` are each right on both sides — but it can veto the combination
    that never is.
    """
    assert _needs_the_tree(("new",))
    assert _needs_the_tree(("restore", "--from", "abc"))
    assert _needs_the_tree(("diff", "--summary"))
    assert _needs_the_tree(("workspace", "add", "--name", "x"))
    # The two that are correct on both sides must not be vetoed.
    assert not _needs_the_tree(("log", "--no-graph", "-r", "@"))
    assert not _needs_the_tree(("bookmark", "set", "ph/x", "-r", "@"))
    assert not _needs_the_tree(("workspace", "root"))
    assert not _needs_the_tree(("workspace", "forget", "x"))


async def test_the_calls_that_must_see_the_tree_still_do(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The other side of the same parameter, so the fix cannot be "read everything".

    Three verbs' answers *are* the tree, and each would be silently wrong if it
    stopped committing first: a fork point that misses the parent's newest work, a
    restore point that names a state the agent has moved past, and a release that
    decides the child did nothing.
    """
    ctx, base, session, _agent = await jj_agent(mount)
    (base / "wip.txt").write_text("the parent's work\n", encoding="utf-8")

    # A fork point must see it — this is the whole tier.
    child = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    assert (child.root / "wip.txt").read_text(encoding="utf-8") == "the parent's work\n"

    # A capture must see the child's newest edit.
    (child.root / "cell.txt").write_text("first\n", encoding="utf-8")
    first = await ctx.require(WORKSPACE).capture(child)
    (child.root / "cell.txt").write_text("second\n", encoding="utf-8")
    second = await ctx.require(WORKSPACE).capture(child)
    assert first is not None and second is not None and first != second

    # And a release must not conclude the child did nothing.
    await ctx.require(WORKSPACE).dispose("a1")
    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is True


# --------------------------------------------------- the seam's own materials --


async def test_a_provisioned_material_stays_out_of_what_an_agent_contributes(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P6-35's security property, arriving at this tier — and it needed work here.

    **jj snapshots everything the project does not gitignore**, so a material the
    seam copied in so the tests could run is in the working-copy commit by default.
    That puts it on the bookmark, and the bookmark is a git ref somebody merges.
    Every command succeeds while it happens, which is what makes it worth a test
    rather than a comment.

    **The source is gitignored and the destination is not**, which is what makes
    this a test of the exclusion rather than of `.gitignore`. It is also the
    realistic shape: the material is hidden where it lives and lands under the name
    the project's tooling expects — and jj would snapshot that name.

    Both places pH could put it are covered, and the second is the one easy to miss:
    the agent's own commit, and the **fork point** — freezing a parent's work
    snapshots the parent's tree, so a material sitting there would land in the
    commit every child forks from and be one `git log -p` away from a reader.
    """
    ctx = await mount(
        TIER_ROW,
        {
            "id": "workspace-lifecycle",
            "config": {"provision": [{"source": "secret.env", "dest": "config.env"}]},
        },
    )
    base = await jj_repo(ctx, tmp_path / "repo")
    (base / "secret.env").write_text("TOKEN=shhh\n", encoding="utf-8")
    (base / ".gitignore").write_text("secret.env\n", encoding="utf-8")
    session = ctx.require(SESSIONS).create("s1")
    parent = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="parent", base=base, access="write", session=session
    )
    assert (parent.root / "config.env").exists(), "the agent never got the material"
    (parent.root / "real-work.txt").write_text("the agent did this\n", encoding="utf-8")
    # A child forked from that tree — the fork point is the parent's work frozen,
    # and the material is sitting in it.
    child = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="child", base=parent.root, access="write", session=session
    )
    (child.root / "child-work.txt").write_text("the child did this\n", encoding="utf-8")

    await ctx.require(WORKSPACE).dispose("child")
    await ctx.require(WORKSPACE).dispose("parent")

    code, out, _ = await git(ctx, base, "show", "ph/s1/parent:real-work.txt")
    assert code == 0 and out == "the agent did this\n", "the work did not reach the bookmark"
    code, _, _ = await git(ctx, base, "show", "ph/s1/parent:config.env")
    assert code != 0, "a provisioned material reached a ref somebody merges"
    code, _, _ = await git(ctx, base, "show", "ph/s1/child:config.env")
    assert code != 0, "the fork point carried a provisioned material into every child"


async def test_a_file_the_base_already_tracks_reaches_children_and_that_is_the_tier(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The non-guarantee, stated as behaviour so nobody discovers it by merging.**

    A file sitting untracked in the base is in the base's *working-copy commit*,
    because that is what jj means by a working copy — jj put it there before pH
    looked. Children fork from that commit, so they inherit it, and it is reachable
    from their bookmarks.

    That is not a leak this tier can close; it **is** the tier. "A child starts from
    its parent's work in progress" and "an uncommitted file in the base does not
    reach a child's branch" are the same sentence with opposite signs, and the git
    tier is the one that keeps the second. `snapshot.auto-track` cannot help either:
    it governs what *becomes* tracked, and jj tracked this before pH ran.

    So the rule for a deployment is the ordinary one, and it is the person's own
    `jj status` that shows it: a file that should not travel belongs in
    `.gitignore`, where jj will not snapshot it and provisioning is how a workspace
    gets it instead.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / "in-progress.txt").write_text("the parent is mid-thought\n", encoding="utf-8")

    child = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert (child.root / "in-progress.txt").exists(), "this tier exists to do exactly this"
    code, out, _ = await git(ctx, base, "show", "ph/s1/a1:in-progress.txt")
    assert code == 0 and out == "the parent is mid-thought\n", (
        "the fork point is reachable from the child's bookmark, and carries it"
    )
