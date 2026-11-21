"""A `ctx.workspace` tier for tests that need one and do not need `git`.

Three test modules in two packages had written this fake, and the copies had
already drifted: one returned `base`, one returned a per-agent
directory, one recorded what it was asked, and one picked `kind` from `access`.
Each copy absorbed the provider protocol behind a `**_`. That makes the drift
invisible: a parameter added to `WorkspaceProvider` breaks all three copies,
and none of them fails.

This module is *not* a substitute for `test_workspace_git.py`, which drives
real `git worktree` against a real repository. It answers which `base` and
which `access` reach a tier, and what the resolved `kind` does to everything
downstream. A test of those questions needs no checkout.

@module ph.testing.stub_workspace
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path

from ..cordis import Context
from ..keys import SESSIONS, WORKSPACE
from ..seams.workspace import ContainmentTier, Workspace, WorkspaceAccess, WorkspaceKind

__all__ = ["StubCheckpointingProvider", "StubWorkspaceProvider", "acquire_for_role"]


@dataclass(slots=True)
class StubWorkspaceProvider:
    """A tier that answers as `worktree` does, on any directory.

    `root=None` returns `base` itself, for a test about *which* base reached
    the tier and not about isolation.
    """

    root: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    tier: ContainmentTier = "worktree"
    kinds: tuple[WorkspaceKind, WorkspaceKind] = ("worktree", "worktree-ephemeral")
    """What this tier answers for `write` and for `read`, in that order.

    It is a pair and not a hardcoded resolution. So a test can drive the seam
    with a kind whose *provider* is not installed on the host. For example,
    `("overlay", "overlay-ephemeral")` exercises retention, disposal and the
    decline of `/revert` for the overlay tier on a machine with no AgentFS.
    That is the CI shape that the gate of P6-21 names. The default is the
    answer of the git tier, so every existing caller reads the same.
    """
    bases: list[Path] = field(default_factory=list)
    """Every `base` that this tier was asked about, in order. A spawn test
    asserts on it, because branching from the root of the *parent* is what puts
    a fan-out on sibling branches."""

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
        # The rule of the real tiers: a tree that is already at the path of this
        # agent belongs to this agent.
        reused = self.root is not None and tree.is_dir()
        tree.mkdir(parents=True, exist_ok=True)
        return Workspace(
            root=tree,
            scratch=scratch,
            reused=reused,
            # Everything downstream reads the answer of the tier, not the
            # request. That is the purpose of resolving here.
            kind=self.kinds[1] if access == "read" else self.kinds[0],
            repo_writable=True,
            ref=f"ph/{session_id}/{agent_id}",
            env=self.env,
        )


@dataclass(slots=True)
class StubCheckpointingProvider(StubWorkspaceProvider):
    """The same tier, plus restore points: a `CheckpointingProvider` for tests.

    **It is a separate class and not a flag, because `isinstance` is the
    question.** The capability is a runtime-checkable Protocol, so a tier can
    checkpoint when it has the methods. A `checkpoints: bool = False` on the
    one class would be invisible to the gate of the seam itself, and a test
    that set it would prove nothing.

    The snapshots are in memory, and the purpose is the *seam*: which
    workspaces are offered a restore point, that `/revert` lists and does not
    refuse, and that the token recorded is the token restored. What a real
    capture costs, and whether it disturbs the index of an agent, is the
    question of `test_workspace_checkpoint.py` against real git.
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


async def acquire_for_role(ctx: Context, base: Path, *, child: bool = False) -> Workspace:
    """The workspace of one agent, acquired the way a real spawn asks for it.

    **A child is a session stamped `origin: "subagent"`, and nothing else.**
    `WorkspaceSeam.acquire` derives the rung from that encoding. P4-11 deleted
    the threading of `tier=` so that a caller cannot forget it. So a test that
    built the session by hand would derive again the one fact under test. This
    helper was written out in two test modules of ph-core before it moved here,
    as `git_repo` and the provider in this file also moved here.

    The provider is registered on first use and only then. Two answers to "what
    makes a workspace" would be a contradiction, and what differs per role is
    whether the caller asks for one.
    """
    workspace_seam = ctx.require(WORKSPACE)
    if workspace_seam.provider is None:
        workspace_seam.register_provider(StubWorkspaceProvider(root=base / "trees"))
    session = ctx.require(SESSIONS).create(
        "child-session" if child else "root-session",
        meta={"origin": "subagent"} if child else None,
    )
    return await workspace_seam.acquire(
        session_id=session.id,
        agent_id="child" if child else "root",
        base=base,
        session=session,
    )
