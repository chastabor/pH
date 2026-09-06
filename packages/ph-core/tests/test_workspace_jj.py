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

from pathlib import Path
from typing import Any

import pytest

from ph.seams.workspace import WorkspaceRecord
from ph.testing import git, jj, jj_repo, needs_jj

pytestmark = [pytest.mark.anyio, needs_jj]


TIER_ROW = {"insert": [{"id": "workspace-jj", "name": "workspace-jj"}]}
"""The row under test. Mounted rather than hand-assembled, so a typo in the entry
point fails here rather than in someone's profile — and so the workspace and
scratch roots come from `$PH_HOME`, which the `mount` fixture already points at
`tmp_path`."""


async def _tiered(mount: Any, tmp_path: Path) -> tuple[Any, Path]:
    """A mounted profile with the tier on, and a colocated repository to point it at.

    The repository goes under `tmp_path`, never `ctx.fs.root` — that is the
    *process's* directory, which for a test run is this checkout. A `base` taken
    from it would have every test here initialising jj inside pH's own tree.
    """
    ctx = await mount(TIER_ROW)
    return ctx, await jj_repo(ctx, tmp_path / "repo")


async def _bookmarks(ctx: Any, base: Path) -> str:
    _, out, _ = await jj(ctx, base, "bookmark", "list")
    return out


# ------------------------------------------------------------------ acquire --


async def test_a_write_agent_gets_its_own_workspace_on_its_own_bookmark(
    mount: Any, tmp_path: Path
) -> None:
    """E2, one half. The name is `ph/<session>/<agent>` — the git tier's shape, and
    deliberately so: `/workspaces` enumerates one prefix to find what disposal
    leaves, and two tiers naming their artifacts differently would make it find
    half of them."""
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert workspace.kind == "worktree"
    assert workspace.ref == "ph/s1/a1"
    assert workspace.root != base
    assert (workspace.root / "README.md").read_text(encoding="utf-8") == "base\n"
    assert workspace.repo_writable is True


async def test_a_child_starts_from_the_parents_work_in_progress(mount: Any, tmp_path: Path) -> None:
    """**The reason this tier exists**, and the one thing the git tier cannot do.

    The parent has edited a tracked file and added a new one, and has committed
    neither — which is the normal state of a live agent, since the git tier
    commits only at disposal. Under `workspace-git` a child spawned here gets the
    base commit and sees none of this. Under jj it gets the work.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / "README.md").write_text("parent edited this\n", encoding="utf-8")
    (base / "wip.txt").write_text("parent's work in progress\n", encoding="utf-8")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert (workspace.root / "wip.txt").read_text(encoding="utf-8") == "parent's work in progress\n"
    assert (workspace.root / "README.md").read_text(encoding="utf-8") == "parent edited this\n"


async def test_the_parent_may_keep_working_without_staling_its_children(
    mount: Any, tmp_path: Path
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
    workspace = await ctx.workspace.acquire(
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
    mount: Any, tmp_path: Path
) -> None:
    """E2, and the fan-out this tier is bought for.

    Both children see the parent's uncommitted work and neither sees the other's,
    which is the property "eight agents in one checkout" destroys. The parent's
    history gains **one** fork point for the pair, not one per child: the first
    spawn freezes the work and its sibling finds nothing new to freeze.
    """
    ctx, base = await _tiered(mount, tmp_path)
    (base / "shared.txt").write_text("from the parent\n", encoding="utf-8")

    one = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")
    two = await ctx.workspace.acquire(session_id="s1", agent_id="a2", base=base, access="write")

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
    mount: Any, tmp_path: Path
) -> None:
    """E3. This tier cannot enforce read-only any more than the git tier can, so
    `access="read"` buys a different *kind* rather than a permission: the writes
    happen and reach nobody.

    `repo_writable` stays `True`, which is the honest answer — `False` would
    describe a guarantee only the sandbox tier can make, and a caller would act on
    it.
    """
    ctx, base = await _tiered(mount, tmp_path)

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read"
    )

    assert workspace.kind == "worktree-ephemeral"
    assert workspace.repo_writable is True
    (workspace.root / "notes.txt").write_text("scratch thinking\n", encoding="utf-8")


async def test_a_directory_jj_does_not_manage_declines_and_the_seam_falls_back(
    mount: Any, tmp_path: Path
) -> None:
    """The decline that keeps this row safe to layer anywhere.

    **A plain git repository is the case that matters**, not an empty directory:
    it is what most projects are, and converting one would mean writing `.jj/` into
    somebody's tree and changing what their own `git` commands report. So the tier
    declines and the person keeps a shared workspace, rather than acquiring a
    repository layout nobody asked for.
    """
    from ph.testing import git_repo

    ctx = await mount(TIER_ROW)
    base = await git_repo(ctx, tmp_path / "plain")

    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )

    assert workspace.kind == "shared"
    assert workspace.root == base
    assert not (base / ".jj").exists(), "the tier converted a repository it was only shown"


# ------------------------------------------------------------------ release --


async def test_release_puts_the_childs_work_on_a_git_branch_and_removes_the_tree(
    mount: Any, tmp_path: Path
) -> None:
    """The merge-back story, end to end, and it ends in **git**.

    The bookmark is jj's, the branch is git's, and colocation is what makes them
    the same name. A person who never installs jj still merges what a child
    produced with the `git` they already use — which is the whole claim that lets
    this be a provider swap rather than a migration.
    """
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )
    (workspace.root / "child.txt").write_text("work from a1\n", encoding="utf-8")

    await ctx.workspace.dispose("a1")

    assert not workspace.root.exists(), "the checkout is a resource, not an artifact"
    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is True
    code, out, err = await git(ctx, base, "show", "--stat", "ph/s1/a1")
    assert code == 0, err
    assert "child.txt" in out
    code, _, err = await git(ctx, base, "merge", "--no-edit", "ph/s1/a1")
    assert code == 0, err
    assert (base / "child.txt").read_text(encoding="utf-8") == "work from a1\n"


async def test_a_child_that_did_nothing_leaves_nothing_behind(mount: Any, tmp_path: Path) -> None:
    """The other half of the disposal policy: a bookmark on an empty commit is
    noise a person then has to work out how to clean up."""
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write", session=session
    )

    await ctx.workspace.dispose("a1")

    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is False
    assert "ph/s1/a1" not in await _bookmarks(ctx, base)


async def test_an_ephemeral_release_discards_the_work_and_the_ref(
    mount: Any, tmp_path: Path
) -> None:
    """`worktree-ephemeral`'s whole promise: the writes happened and reach nobody,
    **even though the tree was dirty**."""
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )
    (workspace.root / "throwaway.txt").write_text("nobody sees this\n", encoding="utf-8")

    await ctx.workspace.dispose("a1")

    (disposed,) = [one for one in session.events if one.type == "workspace/disposed"]
    assert disposed.data["kept"] is False
    assert "ph/s1/a1" not in await _bookmarks(ctx, base)
    code, _, _ = await git(ctx, base, "rev-parse", "--verify", "refs/heads/ph/s1/a1")
    assert code != 0


async def test_a_retained_ephemeral_workspace_keeps_its_work(mount: Any, tmp_path: Path) -> None:
    """P6-28. Retention is an exception to `discard`, so a child whose run went
    wrong leaves its work on a bookmark to read rather than a directory to trip
    over."""
    ctx, base = await _tiered(mount, tmp_path)
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="read", session=session
    )
    (workspace.root / "evidence.txt").write_text("what went wrong\n", encoding="utf-8")
    ctx.workspace.retain("a1", "the run failed and this is why")

    await ctx.workspace.dispose("a1")

    assert not workspace.root.exists()
    code, out, err = await git(ctx, base, "show", "--stat", "ph/s1/a1")
    assert code == 0, err
    assert "evidence.txt" in out


async def test_disposal_leaves_the_repository_able_to_re_acquire(
    mount: Any, tmp_path: Path
) -> None:
    """The state a resume finds has to be usable.

    jj refuses to add a workspace under a name it still knows, so a provider that
    removed directories without deregistering them would decline for the rest of
    the repository's life — which the seam reports as `shared`, with nobody able
    to see why.
    """
    ctx, base = await _tiered(mount, tmp_path)

    first = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="read")
    await ctx.workspace.dispose("a1")
    second = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="read")

    assert second.kind == "worktree-ephemeral"
    assert second.root == first.root


async def test_a_workspace_already_at_the_path_is_reused(mount: Any, tmp_path: Path) -> None:
    """The rehydrate path: an agent given its workspace back must find its own work
    in it, not a fresh fork over the top of it."""
    ctx, base = await _tiered(mount, tmp_path)
    first = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")
    (first.root / "half-done.txt").write_text("mid-turn\n", encoding="utf-8")

    second = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="write")

    assert second.root == first.root
    assert (second.root / "half-done.txt").read_text(encoding="utf-8") == "mid-turn\n"


# ------------------------------------------------------------------ reclaim --


async def test_reclaim_recovers_a_crashed_childs_work_from_the_record_alone(
    mount: Any, tmp_path: Path
) -> None:
    """F6. A crash must cost a directory, not a day.

    The reconciling process has none of the acquiring one's state — only what the
    log wrote — so this drives `reclaim` from a `WorkspaceRecord` built by hand.
    The workspace name comes back off `ref`, and every command runs in the tree
    the record names, which is why no repository path is needed or guessed at.
    """
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "unsaved.txt").write_text("the process died here\n", encoding="utf-8")
    record = WorkspaceRecord(
        agent_id="a1", kind="worktree", root=workspace.root, ref="ph/s1/a1", session_id="s1"
    )

    kept = await ctx.workspace.provider.reclaim(record)

    assert kept is True
    assert not workspace.root.exists()
    code, out, err = await git(ctx, base, "show", "--stat", "ph/s1/a1")
    assert code == 0, err
    assert "unsaved.txt" in out


async def test_reclaiming_a_tree_that_is_already_gone_says_so(mount: Any, tmp_path: Path) -> None:
    """Both an orderly release and a reconciliation reach disposal, and now that the
    ordinary path removes the tree, the second arrival is the common case. It
    decides nothing — it reports what the first left, which is the bookmark."""
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "work.txt").write_text("done\n", encoding="utf-8")
    record = WorkspaceRecord(
        agent_id="a1", kind="worktree", root=workspace.root, ref="ph/s1/a1", session_id="s1"
    )
    assert await ctx.workspace.provider.reclaim(record) is True

    assert await ctx.workspace.provider.reclaim(record) is False


async def test_work_that_cannot_be_named_keeps_its_tree_instead_of_losing_it(
    mount: Any, tmp_path: Path
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
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    (workspace.root / "irreplaceable.txt").write_text("hours of work\n", encoding="utf-8")
    record = WorkspaceRecord(
        agent_id="a1", kind="worktree", root=workspace.root, ref="ph/no bookmark", session_id="s1"
    )

    kept = await ctx.workspace.provider.reclaim(record)

    assert kept is True
    assert (workspace.root / "irreplaceable.txt").exists(), "work that reached no ref was deleted"


async def test_export_names_the_bookmark_the_work_is_on(mount: Any, tmp_path: Path) -> None:
    """`/workspaces` asks the seam one question rather than asking which tier
    answered."""
    ctx, base = await _tiered(mount, tmp_path)
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, access="write"
    )
    record = WorkspaceRecord(
        agent_id="a1",
        kind="worktree",
        root=workspace.root,
        ref=workspace.ref,
        session_id="s1",
    )

    assert await ctx.workspace.export(record) == "ph/s1/a1"


# --------------------------------------------------------------------- tier --


async def test_the_tier_does_not_advertise_checkpoints_it_cannot_take(
    mount: Any, tmp_path: Path
) -> None:
    """E1's single failure, in the one place a person looks to check.

    `TIERS["worktree"]` sells "per-run checkpoints, /revert", and
    `workspace-checkpoint` is a **git** row: it hashes a tree with `git
    write-tree`, and a jj workspace is not a git checkout. Printing the rung's
    stock text here would advertise a mechanism the mounted tier does not have.
    """
    ctx, _ = await _tiered(mount, tmp_path)

    described = ctx.workspace.provider.describe_tier()

    assert "no per-run checkpoints and no /revert" in described.buys
    assert "absolute-path raw write" in described.does_not_bound
