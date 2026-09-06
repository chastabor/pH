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
from hashlib import blake2b
from pathlib import Path
from typing import Any

from ..seams.workspace import ContainmentTier, Workspace, WorkspaceAccess, WorkspaceKind

__all__ = ["StubCheckpointingProvider", "StubWorkspaceProvider", "acquire_for_role"]


@dataclass(slots=True)
class StubWorkspaceProvider:
    """A tier that answers the way `worktree` does, on any directory.

    `root=None` hands back `base` itself, for a test about *which* base reached
    the tier rather than about isolation.
    """

    root: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    tier: ContainmentTier = "worktree"
    kinds: tuple[WorkspaceKind, WorkspaceKind] = ("worktree", "worktree-ephemeral")
    """What this tier answers for `write` and for `read`, in that order.

    A pair rather than a hardcoded resolution, so a test can drive the seam with
    a kind whose *provider* is not installed on the host — `("overlay",
    "overlay-ephemeral")` exercises retention, disposal and `/revert`'s decline
    for the overlay tier on a machine with no AgentFS, which is the CI shape
    P6-21's own gate names. The default is the git tier's answer, so every
    existing caller reads the same.
    """
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
            # The tier's answer — not the request — is what everything
            # downstream reads, which is the whole point of resolving here.
            kind=self.kinds[1] if access == "read" else self.kinds[0],
            repo_writable=True,
            ref=f"ph/{session_id}/{agent_id}",
            env=self.env,
        )


@dataclass(slots=True)
class StubCheckpointingProvider(StubWorkspaceProvider):
    """The same tier, plus restore points — a `CheckpointingProvider` for tests.

    **A separate class rather than a flag, because `isinstance` is the question.**
    The capability is a runtime-checkable Protocol, so what makes a tier able to
    checkpoint is having the methods; a `checkpoints: bool = False` on the one class
    would be invisible to the seam's own gate, and a test that set it would prove
    nothing.

    The snapshots are in memory and the whole point is the *seam*: which workspaces
    are offered a restore point, that `/revert` lists rather than refuses, that the
    token recorded is the token restored. What a real capture costs, and whether it
    disturbs an agent's index, is `test_workspace_checkpoint.py`'s question against
    real git.
    """

    saved: dict[str, dict[str, bytes]] = field(default_factory=dict)

    async def capture(self, workspace: Workspace) -> str | None:
        held = {
            str(one.relative_to(workspace.root)): one.read_bytes()
            for one in sorted(workspace.root.rglob("*"))
            if one.is_file()
        }
        digest = blake2b(digest_size=8)
        for name, content in held.items():
            digest.update(name.encode() + b"\0" + content + b"\0")
        token = digest.hexdigest()
        self.saved[token] = held
        return token

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]:
        held = self.saved.get(token)
        if held is None:
            raise FileNotFoundError(f"restore point {token} is gone")
        added: list[str] = []
        for one in sorted(workspace.root.rglob("*")):
            if one.is_file() and str(one.relative_to(workspace.root)) not in held:
                added.append(str(one.relative_to(workspace.root)))
                one.unlink()
        for name, content in held.items():
            target = workspace.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return tuple(sorted(added))


async def acquire_for_role(ctx: Any, base: Path, *, child: bool = False) -> Any:
    """One agent's workspace, acquired the way a real spawn asks for it.

    **A child is a session stamped `origin: "subagent"`, and nothing else.**
    That is the encoding `WorkspaceSeam.acquire` derives the rung from — P4-11
    deleted the `tier=` threading precisely so a caller could not forget — so a
    test that hand-rolled the session would be re-deriving the one fact under
    test. It was written out in two modules in this directory before it moved
    here, which is the same road `git_repo` and this file's own provider took.

    The provider is registered on first use and only then: two answers to "what
    makes a workspace" is a contradiction, and what differs per role is whether
    the caller asks for one.
    """
    if ctx.workspace.provider is None:
        ctx.workspace.register_provider(StubWorkspaceProvider(root=base / "trees"))
    session = ctx.sessions.create(
        "child-session" if child else "root-session",
        meta={"origin": "subagent"} if child else None,
    )
    return await ctx.workspace.acquire(
        session_id=session.id,
        agent_id="child" if child else "root",
        base=base,
        session=session,
    )
