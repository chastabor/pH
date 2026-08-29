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

from pathlib import Path
from typing import Any

from ..seams.workspace_git import git

__all__ = ["git", "git_repo"]


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
