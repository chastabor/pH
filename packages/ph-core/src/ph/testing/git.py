"""A real git repository for a test, built the same way in every package.

`git`, `_repo` and the three steps `init`, `user.email` and `user.name` were
written out in two test modules, and a third module was about to copy them.
This kind of setup gains a line (`commit.gpgsign=false`, a `safe.directory`) in
one copy and not in the other. The copy that lacks the line then fails only on
the machine of a contributor. `StubWorkspaceProvider` in
`ph.testing.stub_workspace` makes the same argument for the same reason.

The tests of the `worktree` tier use real git deliberately. They test the
behavior of git and not our arithmetic about it: that `-d` declines an unmerged
branch, and that a removed worktree deregisters.

@module ph.testing.git
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent.types import AgentDriver
from ..cordis import Context
from ..keys import AGENTS, SESSIONS, WORKSPACE
from ..seams.workspace import Workspace
from ..seams.workspace_git import git
from ..session import Session
from . import MountProfile
from .builders import FAKE_OPTIONS

__all__ = ["WORKTREE_ROWS", "git", "git_repo", "worktree_agent"]

WORKTREE_ROWS: tuple[dict[str, Any], ...] = (
    {"insert": [{"id": "workspace-git-worktree", "name": "workspace-git-worktree"}]},
    {"insert": [{"id": "workspace-checkpoint", "name": "workspace-checkpoint"}]},
    {"insert": [{"id": "workspace-revert", "name": "workspace-revert"}]},
    {"insert": [{"id": "workspace-commands", "name": "workspace-commands"}]},
)
"""Every row of the `worktree` tier, for a test that wants the whole tier.

The bundles that the `mount` fixture loads by default do not contain these
rows, because they cost a checkout per agent. Only the `ph-rlm` bundle layers
them (P4-11). So a test that wants the tier says so explicitly."""


async def worktree_agent(
    mount: MountProfile,
    tmp_path: Path,
    *extra_rows: dict[str, Any],
) -> tuple[Context, Session, AgentDriver, Workspace]:
    """Return `(ctx, session, agent, workspace)`: one agent that holds a real worktree.

    The whole tier is mounted. The repository is built at `tmp_path / "repo"`,
    and never in the directory of the *process*, which for a test run is this
    checkout. With a `base` taken from the checkout, every test in the suite
    would initialize git repositories inside the pH tree, and all of them would
    share one branch namespace. That mistake has already been made one time.
    """
    ctx = await mount(*WORKTREE_ROWS, *extra_rows)
    base = await git_repo(ctx, tmp_path / "repo")
    (base / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (base / "tracked.txt").write_text("original\n", encoding="utf-8")
    await git(ctx, base, "add", "-A")
    await git(ctx, base, "commit", "-m", "content")
    session = ctx.require(SESSIONS).create("s1")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id=agent.id, base=base, access="write", session=session
    )
    assert workspace.kind == "worktree", "these tests need a real checkout"
    return ctx, session, agent, workspace


async def git_repo(ctx: Context, path: Path) -> Path:
    """A repository with one commit, which is the least that a worktree can branch from.

    The identity is set on the repository and is not inherited. So a machine
    with no global git identity runs these tests as well as a machine that has
    one.
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
