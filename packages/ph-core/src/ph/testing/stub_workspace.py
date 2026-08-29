"""A `ctx.workspace` tier for tests that need one without needing `git`.

Three test modules in two packages had written this fake, and the copies had
already drifted three ways — one returned `base`, one returned a per-agent
directory, one recorded what it was asked, one picked `kind` from `access`. Each
absorbed the provider protocol behind a `**_`, which is what makes the drift
invisible: a parameter added to `WorkspaceProvider` breaks all three and none of
them fail.

What it is *not* is a substitute for `test_workspace_git.py`, which drives real
`git worktree` against a real repository. The question this answers is which
`base` and which `access` reach a tier, and what the resolved `kind` makes of
everything downstream — none of which needs a checkout to pin.

@module ph.testing.stub_workspace
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..seams.workspace import ContainmentTier, Workspace, WorkspaceAccess

__all__ = ["StubWorkspaceProvider"]


@dataclass(slots=True)
class StubWorkspaceProvider:
    """A tier that answers the way `worktree` does, on any directory.

    `root=None` hands back `base` itself, for a test about *which* base reached
    the tier rather than about isolation.
    """

    root: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    tier: ContainmentTier = "worktree"
    bases: list[Path] = field(default_factory=list)
    """Every `base` this tier was asked about, in order — the assertion a spawn
    test makes, since branching from the *parent's* root is what puts a fan-out
    on sibling branches."""

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace:
        self.bases.append(base)
        tree = base if self.root is None else self.root / agent_id
        tree.mkdir(parents=True, exist_ok=True)
        return Workspace(
            root=tree,
            scratch=scratch,
            # The same resolution the git tier makes, because it is the tier's
            # answer — not the request — that everything downstream reads.
            kind="worktree-ephemeral" if access == "read" else "worktree",
            repo_writable=True,
            ref=f"ph/{session_id}/{agent_id}",
            env=self.env,
        )
