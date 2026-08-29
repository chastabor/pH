"""P4-08b — `/workspaces`, the human half of the keep-dirty policy (E15).

The `worktree` tier keeps a dirty worktree on disposal on purpose, and that
*creates* the accumulation `wtp`'s README opens with: trees and `ph/*` branches
piling up with nothing in the harness to see or finish them. This is the command
that finishes them, and every interesting assertion here is a **refusal**.

Real `git` throughout, for the same reason the tier's own tests use it: what is
being pinned is git's behaviour — that `-d` declines an unmerged branch, that a
removed worktree deregisters — not our arithmetic about it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

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
    """One disposed agent's worktree, kept or removed by the policy itself."""
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
    """The whole reason the command exists: a dirty tree is kept deliberately,
    and until now nothing could tell a person it was there."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await _left_behind(ctx, base, "dirty-one", work=True)

    shown = await _run(ctx, "list")

    assert "dirty-one" in shown
    assert "ph/s1/dirty-one" in shown
    assert "dirty" in shown


async def test_listing_ignores_worktrees_ph_did_not_make(
    mount: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`git worktree list` reports the user's own trees too, and a management
    command that offered to delete those would be a different and much worse
    tool."""
    ctx, base = await _tiered(mount, tmp_path, monkeypatch)
    await git(ctx, base, "worktree", "add", "-b", "mine", str(tmp_path / "mine"), "HEAD")

    shown = await _run(ctx, "list")

    assert "mine" not in shown
    assert shown == "no agent worktrees are left behind"


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
    root = await _left_behind(ctx, base, "committed", work=True)
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "the child's work")

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
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "work")
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
    root = await _left_behind(ctx, base, "child", work=True)
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "work")

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
