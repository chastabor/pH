"""A real jj repository for a test, built the same way in every package.

This module is the counterpart of `ph.testing.git`, and it exists for the
reason that module states. This kind of setup gains a line in one copy and not
in the other, and the copy that lacks the line fails only on the machine of a
contributor.

The tests of the jj tier drive the real binary deliberately. They test the
behavior of **jj**, which is not arithmetic of ours to stub:

- A workspace forked from a frozen commit does not go stale while the parent
  continues to work.
- A bookmark follows its commit through a snapshot.
- `git export` from a secondary workspace reaches the colocated repo.

@module ph.testing.jj
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..agent.types import AgentDriver
from ..cordis import Context
from ..keys import AGENTS, FS, SESSIONS
from ..seams.workspace_jj import jj
from ..session import Session
from . import MountProfile
from .builders import FAKE_OPTIONS
from .git import git_repo

__all__ = ["JJ_ROWS", "jj", "jj_agent", "jj_repo"]

JJ_ROWS: tuple[dict[str, Any], ...] = (
    {"insert": [{"id": "workspace-jj", "name": "workspace-jj"}]},
    {"insert": [{"id": "workspace-commands", "name": "workspace-commands"}]},
)
"""The jj tier as profile rows. No bundle layers them by default, because P4-11
decides which provider a deployment pays for. So a test that wants the tier
says so."""


async def jj_repo(ctx: Context, path: Path) -> Path:
    """A colocated jj repository with one git commit, the least that a fork can start from.

    **The repository is colocated, because this tier is for that shape.** The
    bookmarks that the tier writes must come back as git branches that a person
    can `git show`. With a non-colocated repo, every test would pass while the
    property that the module advertises went untested.

    The repository is built through `ph.testing.git.git_repo` and not beside
    it. So the identity and `commit.gpgsign` settings, which make git work on a
    machine with no global config, are stated one time.
    """
    await git_repo(ctx, path)
    await jj(ctx, path, "git", "init", "--colocate")
    # The identity is set on the repository and is not inherited, as in
    # `git_repo`. So a machine with no jj identity configured runs these tests
    # as well as a machine that has one.
    for setting in (("user.name", "pH"), ("user.email", "ph@example.invalid")):
        await jj(ctx, path, "config", "set", "--repo", *setting)
    return path


async def jj_agent(
    mount: MountProfile,
    *extra_rows: dict[str, Any],
) -> tuple[Context, Path, Session, AgentDriver]:
    """Return `(ctx, base, session, agent)`: a mounted jj tier and an agent to query it.

    This is the counterpart of `worktree_agent`, and it exists for the same
    reason. Six tests had written the same four lines, and this kind of
    preamble gains a row in one copy and not in the others.

    The repository is built **at `ctx.fs.root`**, which the `mount` fixture
    points at `tmp_path`. The base of `/workspaces` is `ctx.fs.root`, so the
    command then looks at the same repository that the tier branches from. With
    a repo built beside it, every command in a test would report nothing, and
    every assertion would pass for the wrong reason.
    """
    ctx = await mount(*JJ_ROWS, *extra_rows)
    base = await jj_repo(ctx, ctx.require(FS).root)
    session = ctx.require(SESSIONS).create("s1")
    return ctx, base, session, ctx.require(AGENTS).create(session, FAKE_OPTIONS)
