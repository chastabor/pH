"""A real jj repository for a test, built the same way in every package.

`ph.testing.git`'s twin, and it exists for that module's stated reason: the setup
is exactly the kind that grows a line in one copy and not the other, and the copy
that misses it fails only on a contributor's machine.

The jj tier's tests drive the real binary deliberately. What they pin is **jj's**
behaviour — that a workspace forked from a frozen commit does not go stale when
the parent works on, that a bookmark follows its commit through a snapshot, that
`git export` from a secondary workspace reaches the colocated repo — none of
which is arithmetic of ours to stub.

@module ph.testing.jj
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ..seams.workspace_jj import jj

__all__ = ["JJ_ROWS", "jj", "jj_repo", "needs_jj"]

needs_jj = pytest.mark.skipif(shutil.which("jj") is None, reason="the jj tier needs jj")
"""Shared for `needs_git`'s reason: the third module to drive a real binary forgot
the marker, and on a machine without it a clean skip became a dozen errors."""

JJ_ROWS: tuple[dict[str, Any], ...] = (
    {"insert": [{"id": "workspace-jj", "name": "workspace-jj"}]},
)
"""The jj tier as a profile row. Layered nowhere by default — which provider a
deployment pays for is P4-11's decision — so a test that wants it says so."""


async def jj_repo(ctx: Any, path: Path) -> Path:
    """A colocated jj repository with one git commit — the least a fork can start from.

    **Colocated, because that is the shape this tier is for**: the bookmarks it
    writes are meant to come back as git branches a person can `git show`, and a
    non-colocated repo would let every test pass while the property the module
    advertises went untested.

    Built through `ph.testing.git.git_repo` rather than beside it, so the identity
    and `commit.gpgsign` settings that make git work on a machine with no global
    config are stated once.
    """
    from .git import git_repo

    await git_repo(ctx, path)
    await jj(ctx, path, "git", "init", "--colocate")
    # On the repository rather than inherited, `git_repo`'s rule: a machine with
    # no jj identity configured runs these tests as well as one that has it.
    for setting in (("user.name", "pH"), ("user.email", "ph@example.invalid")):
        await jj(ctx, path, "config", "set", "--repo", *setting)
    return path
