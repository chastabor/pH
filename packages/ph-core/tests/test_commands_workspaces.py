"""P4-08b — `/workspaces`, the human half of the disposal policy (E15).

The `worktree` tier commits an agent's work to its own branch and takes the
checkout back, and that *creates* the accumulation `wtp`'s README opens with:
`ph/*` branches piling up with nothing in the harness to see or finish them.
This is the command that finishes them, and every interesting assertion here is
a **refusal**.

**Rows are branches, and that is the whole shape of this command.** It listed
checkouts for one round, which meant it could see exactly the two states that
are not the ordinary one — a tree a live agent holds, and a tree disposal failed
to remove — and reported every agent that finished cleanly as nothing at all.
The artifact is what should be enumerable; a directory is a resource.

Real `git` throughout, for the same reason the tier's own tests use it: what is
being pinned is git's behaviour — that `-d` declines an unmerged branch, that a
removed worktree deregisters — not our arithmetic about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.seams.workspace import WorkspaceRecord
from ph.testing import FAKE_OPTIONS, git, git_repo, needs_git

pytestmark = [pytest.mark.anyio, needs_git]

ROWS = (
    {"insert": [{"id": "workspace-git-worktree", "name": "workspace-git-worktree"}]},
    {"insert": [{"id": "workspace-commands", "name": "workspace-commands"}]},
)


async def _tiered(mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    """A mounted profile with the tier and the command, over a real repository.

    `ctx.fs.root` is the process's directory — this checkout — so the command,
    which reads it as the base, is pointed at a scratch repository instead. That
    is the same mistake `test_workspace_git.py` caught in its own fixtures once.
    """
    ctx = await mount(*ROWS)
    base = await git_repo(ctx, tmp_path / "repo")
    monkeypatch.setattr(ctx.fs, "root", base)
    return ctx, base


async def _left_behind(ctx: Any, base: Path, agent_id: str, *, work: bool) -> Path:
    """One disposed agent, left behind by the policy itself.

    Returns the (now removed) checkout path, because a few tests need to say that
    it is gone. With `work=True` disposal commits before removing, so what this
    leaves is a branch with the agent's work on it and no directory — which is why
    no test here has to `git commit` by hand any more.
    """
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id=agent_id, base=base, access="write"
    )
    if work:
        (workspace.root / "work.txt").write_text("real\n", encoding="utf-8")
    await ctx.workspace.dispose(agent_id)
    return workspace.root


async def _run(ctx: Any, argument: str = "") -> str:
    shown = await ctx.commands.dispatch(f"/workspaces {argument}".strip())
    assert shown is not None
    return str(shown)


# ---------------------------------------------------------------------- list --


async def test_listing_shows_what_the_policy_left_behind(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reason the command exists: an agent's work is committed to a
    branch deliberately, and nothing else in the harness would tell a person it
    was there."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    removed = await _left_behind(ctx, base, "dirty-one", work=True)

    shown = await _run(ctx, "list")

    assert "dirty-one" in shown
    assert "ph/s1/dirty-one" in shown
    assert "branch" in shown, "the row's state should say the artifact is a branch"
    assert str(removed) not in shown and not removed.exists()


async def test_listing_ignores_worktrees_ph_did_not_make(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository is full of branches and worktrees that are not pH's, and a
    management command that offered to delete those would be a different and much
    worse tool.

    `BRANCH_PREFIX` is the whole of the guard now that rows are branches — the
    `root` check that used to do this job only ever saw checkouts, and there are
    usually none. A person's own branch is not under `ph/`, so it never becomes a
    row and `remove` can never be aimed at it.
    """
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await git(ctx, base, "worktree", "add", "-b", "mine", str(tmp_path / "mine"), "HEAD")
    await git(ctx, base, "branch", "feature/x")

    shown = await _run(ctx, "list")

    assert "mine" not in shown and "feature/x" not in shown
    assert shown == "no agent workspaces are left behind"


# -------------------------------------------------------------------- refuse --


async def test_a_workspace_a_live_agent_holds_is_refused(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asked of the seam, not the filesystem: a worktree that is clean this
    instant belongs to an agent that may write to it in the next."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await ctx.workspace.acquire(session_id="s1", agent_id="live", base=base, access="write")

    shown = await _run(ctx, "remove live")

    assert "still holds" in shown
    assert ctx.workspace.of("live") is not None


async def test_an_unmerged_branch_is_kept_and_the_flag_is_named(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-d`, never `-D`. A clean worktree is not evidence that its branch was
    merged — an agent that committed its work leaves nothing in `git status` —
    so git refuses, the work survives, and the answer says which flag overrides
    it rather than making a person guess."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await _left_behind(ctx, base, "committed", work=True)

    shown = await _run(ctx, "remove committed --with-branch")

    assert "--force-branch" in shown
    _, branches, _ = await git(ctx, base, "branch", "--list", "ph/s1/committed")
    assert branches.strip(), "the branch holding unmerged work was deleted"


async def test_force_branch_alone_means_nothing(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flag that silently does nothing is worse than one that refuses."""
    ctx, _base = await _tiered(mount, tmp_path, monkeypatch)

    shown = await _run(ctx, "remove whatever --force-branch")

    assert "--with-branch" in shown


async def test_an_unknown_name_lists_what_there_is(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await _left_behind(ctx, base, "kept-one", work=True)

    shown = await _run(ctx, "remove typo")

    assert "kept-one" in shown


# -------------------------------------------------------------------- finish --


async def test_remove_with_branch_takes_both_when_the_branch_is_merged(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accumulation, actually finished. `wtp`'s README opens on exactly this
    — remove the worktree, forget the branch, orphans pile up — and one command
    doing both is its answer."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    root = await _left_behind(ctx, base, "done", work=True)
    await git(ctx, base, "merge", "--no-edit", "ph/s1/done")

    shown = await _run(ctx, "remove done --with-branch")

    assert "ph/s1/done" in shown
    assert not root.exists()
    _, branches, _ = await git(ctx, base, "branch", "--list", "ph/s1/done")
    assert branches.strip() == ""


async def test_merge_brings_the_work_back_to_the_tree_the_person_is_in(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parent reviews a diff rather than trusting sibling writes (E2) — this
    is where that diff gets taken."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await _left_behind(ctx, base, "child", work=True)

    shown = await _run(ctx, "merge child")

    assert "merged" in shown
    assert (base / "work.txt").read_text(encoding="utf-8") == "real\n"


# ---------------------------------------------------------------------- LIFO --


async def test_an_agents_subprocesses_unwind_before_its_workspace(mount: Any) -> None:
    """A worktree cannot be removed out from under a process still running in it.

    True today by construction — `ctx.effect` unwinds LIFO and a kernel is
    acquired after the workspace it runs in — and asserted nowhere, which is how
    an ordering property stops being true. `wtp` states the human form of the
    same rule by refusing to remove the worktree you are standing in.
    """
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    order: list[str] = []

    await agent.prompt("hello")
    assert ctx.workspace.of(agent.id) is not None
    # Registered *after* the workspace, the way a kernel started for this agent
    # is, so LIFO must tear it down first.
    await agent.ctx.effect(lambda: lambda: order.append("subprocess"), label="pretend-kernel")
    ctx.on("session/event", lambda _s, event: order.append(event.type), global_=True)

    await ctx.agents.dispose(agent.id)

    assert order.index("subprocess") < order.index("workspace/disposed")


# -------------------------------------------------------------------- export --


async def test_export_names_the_branch_a_worktree_is_already_on(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**One verb for both isolating tiers, and this is the trivial half.**

    A worktree agent has been committing to its branch all along, so exporting
    one is naming it. The verb exists because the overlay tier's answer is not
    trivial — its work sits in a delta until somebody assembles a commit — and a
    command that asked which provider was mounted would have to grow a branch per
    tier. `ExportingProvider` is what makes it one question.
    """
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await _left_behind(ctx, base, "a1", work=True)

    ref = await ctx.workspace.export(
        WorkspaceRecord(session_id="s1", agent_id="a1", kind="worktree", root=base, ref="ph/s1/a1")
    )

    assert ref == "ph/s1/a1"


async def test_merge_takes_a_branch_that_has_no_worktree(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**What makes `export`'s own closing sentence true.**

    `list` and `merge` enumerate with `git worktree list`, which an overlay never
    appears in — it leaves a branch and no checkout. So the name `export` hands a
    person back would have been refused by the very command it tells them to run.
    """
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await git(ctx, base, "branch", "ph/s1/exported")

    shown = await _run(ctx, "merge ph/s1/exported")

    assert "refusing" not in shown, shown
    assert "ph/s1/exported" in shown


# ------------------------------------------------------------------- strays --


async def test_removing_a_branch_only_row_asks_for_the_flag(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal that keeps `--with-branch` meaning what it says.

    Every successfully disposed agent is now a row with no checkout, so a bare
    `remove` has nothing to remove. Doing the branch anyway would make the flag
    decorative and delete an artifact a person did not ask about; reporting
    "removed" and touching nothing would be a lie. So it refuses, says where the
    work actually is, and names the flag — the shape every other refusal here
    takes.
    """
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await _left_behind(ctx, base, "finished", work=True)

    shown = await _run(ctx, "remove finished")

    assert "--with-branch" in shown and "ph/s1/finished" in shown
    _, branches, _ = await git(ctx, base, "branch", "--list", "ph/s1/finished")
    assert branches.strip(), "a bare remove deleted the branch it refused to remove"


async def test_a_checkout_disposal_could_not_remove_is_listed_as_a_stray(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one way a directory outlives its agent, and the reason `dirty` survives.

    Disposal removes the checkout unless git refuses — a commit that failed, a
    `worktree remove` that did. That leftover is the only thing here a person
    might want to delete *without* touching the branch, and whether it is safe to
    is exactly what `dirty` answers: clean means the branch already has
    everything, so the directory is pure waste.
    """
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    root = await _left_behind(ctx, base, "stuck", work=True)
    await git(ctx, base, "worktree", "add", str(root), "ph/s1/stuck")
    (root / "unsaved.txt").write_text("never reached the branch\n", encoding="utf-8")

    assert "stray-dirty" in await _run(ctx, "list")

    shown = await _run(ctx, "remove stuck")

    assert str(root) in shown and not root.exists()
    _, branches, _ = await git(ctx, base, "branch", "--list", "ph/s1/stuck")
    assert branches.strip(), "removing the stray checkout took the artifact with it"
