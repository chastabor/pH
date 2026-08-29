"""A real git repository for a test, built the same way in every package.

`git`, `_repo` and the `init` / `user.email` / `user.name` triple were written
out in two test modules and about to be a third. The setup is exactly the kind
that grows a line — `commit.gpgsign=false`, a `safe.directory` — in one copy and
not the other, and the copy that misses it fails only on a contributor's
machine. `StubWorkspaceProvider` next door carries the same argument for the
same reason.

The `worktree` tier's tests use real git deliberately: what they pin is git's
behaviour — that `-d` declines an unmerged branch, that a removed worktree
deregisters — not our arithmetic about it.

@module ph.testing.git
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ..seams.workspace_git import git

__all__ = ["WORKTREE_ROWS", "git", "git_repo", "needs_git", "worktree_agent"]

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="the worktree tier needs git")
"""Shared, because the third module to drive real git forgot it.

Two modules carried this marker and the next one did not, which on a machine
without git turns a clean skip into a dozen errors — exactly the drift this
module exists to stop."""

WORKTREE_ROWS: tuple[dict[str, Any], ...] = (
    {"insert": [{"id": "workspace-git-worktree", "name": "workspace-git-worktree"}]},
    {"insert": [{"id": "workspace-checkpoint", "name": "workspace-checkpoint"}]},
    {"insert": [{"id": "workspace-revert", "name": "workspace-revert"}]},
    {"insert": [{"id": "workspace-commands", "name": "workspace-commands"}]},
)
"""Every row of the `worktree` tier, for a test that wants the whole thing.

Layered nowhere by default — which profile pays for a checkout per agent is
P4-11's decision — so a test that wants the tier says so explicitly."""


async def worktree_agent(
    mount: Any, tmp_path: Path, *extra_rows: dict[str, Any]
) -> tuple[Any, Any, Any, Any]:
    """`(ctx, session, agent, workspace)` — one agent holding a real worktree.

    The whole tier is mounted, and the repository is built under `tmp_path` and
    never under `ctx.fs.root`: that is the *process's* directory, which for a
    test run is this checkout — a `base` taken from it has every test in the
    suite initialising git repositories inside pH's own tree and sharing one
    branch namespace. That mistake has been made once already.
    """
    from ..testing import FAKE_OPTIONS

    ctx = await mount(*WORKTREE_ROWS, *extra_rows)
    base = await git_repo(ctx, tmp_path / "repo")
    (base / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (base / "tracked.txt").write_text("original\n", encoding="utf-8")
    await git(ctx, base, "add", "-A")
    await git(ctx, base, "commit", "-m", "content")
    session = ctx.sessions.create("s1")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id=agent.id, base=base, access="write", session=session
    )
    assert workspace.kind == "worktree", "these tests need a real checkout"
    return ctx, session, agent, workspace


async def git_repo(ctx: Any, path: Path) -> Path:
    """A repository with one commit — the least a worktree can branch from.

    Identity is set on the repository rather than inherited, so a machine with
    no global git identity runs these tests as well as one that has it.
    """
    path.mkdir(parents=True, exist_ok=True)
    for args in (
        ("init", "-b", "main"),
        ("config", "user.email", "ph@example.invalid"),
        ("config", "user.name", "pH"),
        ("config", "commit.gpgsign", "false"),
    ):
        await git(ctx, path, *args)
    (path / "README.md").write_text("base\n", encoding="utf-8")
    await git(ctx, path, "add", "-A")
    await git(ctx, path, "commit", "-m", "base")
    return path
