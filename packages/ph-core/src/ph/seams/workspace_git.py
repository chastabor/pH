"""`workspace-git-worktree` — the `worktree` containment tier (D21, §4.8).

The middle rung of the ladder, and the one whose name most invites overreading.
It gives an agent its own checkout on `ph/<session>/<agent>`, sharing the
repository's object store so creation is cheap, and points `ctx.fs`'s root and
`ctx.subprocess`'s cwd at it. That bounds **every tool-mediated write and every
relative-path raw write**, because both resolve against the agent's cwd. It does
not bound `open("/etc/passwd", "w")`, which never consults a cwd — only the
`sandbox` tier refuses that, at the kernel. The property bought here is
**collision isolation and revertibility**, not confinement, and any sentence
here, in `ph doctor`, or in a config comment that blurs the two is a defect
(§12 Q10, E13).

What that buys is concrete: eight children fanning out no longer write one tree
concurrently, and the parent reviews a diff instead of trusting sibling writes.

**`access="read"` is a different kind, not a different permission.** A research
child should not be handed a checkout it might mutate — but "read-only" is an
enforcement claim and this tier cannot make one, so `read` yields
`worktree-ephemeral`: a full checkout the child may write, **discarded on
disposal and never merged**. `repo_writable` stays `True`, because the writes
happen; they simply reach nobody. Reporting `False` would be describing a
guarantee no tier made, which is the one thing `repo_writable` exists to prevent.

**Disposal is a policy, and the redirection env is what keeps it meaningful.**
An unchanged worktree is removed and a changed one is kept for the user to
inspect and merge — a rule that only says something if "changed" means the
agent's work. `pytest` writes `.pytest_cache/` and `__pycache__/` into the tree
it runs against, so without redirection every worktree would end up dirty and
every one would be kept, and the policy would decay into "keep everything". The
env therefore points the build tools' caches inside `scratch` (E12), which is
outside the worktree and survives disposal. Best-effort by construction: a
toolchain that insists on writing beside its sources still will, and the answer
to that is `access="write"` for that agent, not a weaker tier.

@module ph.seams.workspace_git
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..cordis import Context, plugin
from ..paths import default_home_path
from ..wire import WireModel
from .subprocess import SubprocessSpawnSpec
from .workspace import ContainmentTier, Workspace, WorkspaceAccess, redirection_env

__all__ = ["GitWorktreeProvider", "apply", "sanitize_ref"]

log = logging.getLogger("ph.seams.workspace_git")

_UNSAFE_REF = re.compile(r"[^A-Za-z0-9._-]+")
"""Everything git refuses in a ref component, plus `/`, which would nest.

Session and agent ids are pH's, not a user's, but a ref name is a filesystem
path under `.git/refs` on most setups — so this collapses rather than trusts.
"""


def sanitize_ref(component: str) -> str:
    """One ref path component, safe by construction.

    Git's own rules are a deny-list (`git check-ref-format`); this is the
    allow-list, because a branch name that fails validation *after* a worktree
    has been created is a half-made artifact to clean up.
    """
    cleaned = _UNSAFE_REF.sub("-", component).strip("-.")
    return cleaned or "agent"


@dataclass(slots=True)
class GitWorktreeProvider:
    """The `worktree` tier: one checkout per agent, on its own branch."""

    ctx: Context
    root: Path
    """Where worktrees are checked out — `$PH_HOME/worktrees/<session>/<agent>`,
    outside the repository, so a checkout is never itself a candidate for the
    walk the agent runs over its own tree."""
    tier: ContainmentTier = field(default="worktree", init=False)
    _toplevels: dict[Path, Path | None] = field(default_factory=dict, init=False)
    """`base` → its repository, asked once.

    Every sibling in a fan-out is handed the *same* `base` — the parent's root —
    so without this, `rev-parse` runs once per child to be told the same thing.
    Per process, and a `None` is cached too: a directory that is not a repository
    does not become one while pH is running, and re-asking per agent is how a
    profile that declines pays for the tier it is not using.
    """

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None:
        """A checkout for this agent, or `None` when `base` is not a repository.

        Declining is the normal answer for half the directories a person runs pH
        in, and the seam's fallback makes it a notice rather than a refusal to
        start.
        """
        toplevel = await self._toplevel(base)
        if toplevel is None:
            return None

        ref = f"ph/{sanitize_ref(session_id)}/{sanitize_ref(agent_id)}"
        path = self.root / sanitize_ref(session_id) / sanitize_ref(agent_id)
        if not await self._add(toplevel, path, ref):
            return None

        ephemeral = access == "read"
        return Workspace(
            root=path,
            scratch=scratch,
            kind="worktree-ephemeral" if ephemeral else "worktree",
            # True for both, and deliberately: an ephemeral child writes freely,
            # its writes simply reach nobody. `False` here would be a
            # confinement claim only the sandbox tier can make.
            repo_writable=True,
            ref=ref,
            env=redirection_env(scratch),
            release=lambda: self._release(toplevel, path, ref, discard=ephemeral),
        )

    async def _toplevel(self, base: Path) -> Path | None:
        if base not in self._toplevels:
            self._toplevels[base] = await self._ask_toplevel(base)
        return self._toplevels[base]

    async def _ask_toplevel(self, base: Path) -> Path | None:
        """The repository `base` is in, or `None`.

        Asked of git rather than by looking for a `.git` entry, because a
        worktree's `.git` is a file, a bare checkout has none, and a
        subdirectory is a perfectly good `base` — three cases a directory probe
        gets wrong in the direction of a false positive.
        """
        code, out, _ = await self._git(base, "rev-parse", "--show-toplevel")
        if code != 0:
            log.info(
                "ph.seams.workspace_git: %s is not a git repository; declining so the "
                "seam falls back to a shared workspace",
                base,
            )
            return None
        return Path(out.strip())

    async def _add(self, toplevel: Path, path: Path, ref: str) -> bool:
        """`git worktree add`, tolerating the two states a resume can find.

        A worktree already checked out at this path is *reused*: the agent id is
        the key, so finding one means finding this agent's own tree, and
        recreating it would discard the work the keep-dirty policy exists to
        preserve. A branch that exists without a worktree is attached rather
        than reset, for the same reason — `-B` would silently drop it.
        """
        if (path / ".git").exists():
            log.info("ph.seams.workspace_git: reusing the worktree already at %s", path)
            return True
        code, _, err = await self._git(toplevel, "worktree", "add", "-b", ref, str(path), "HEAD")
        if code != 0:
            # Both recoveries hang off the failure, which is what makes them
            # free: a stale registration from a crash is *why* `add` refuses, and
            # an existing branch is the other why. Pruning up front spent a
            # subprocess per acquire to prepare for a case that had not happened.
            await self._git(toplevel, "worktree", "prune")
            retry = (
                ("worktree", "add", str(path), ref)
                if await self._has_ref(toplevel, ref)
                else ("worktree", "add", "-b", ref, str(path), "HEAD")
            )
            code, _, err = await self._git(toplevel, *retry)
        if code != 0:
            log.warning(
                "ph.seams.workspace_git: could not create a worktree at %s (%s); declining",
                path,
                err.strip() or f"git exited {code}",
            )
        return code == 0

    async def _release(self, toplevel: Path, path: Path, ref: str, *, discard: bool) -> bool:
        """Remove or keep, and report which. The disposal policy, in one place.

        Returns whether anything was **kept**, which is what rides
        `workspace/disposed` — the difference between "nothing changed, so it was
        removed" and "these writes were thrown away by design" is not derivable
        from the kind, and a reader should not have to guess it. Every path
        reports what actually happened rather than what was intended: a removal
        that failed leaves the tree on disk, and saying `False` there would make
        the field a statement of policy instead of a record.
        """
        if not discard and await self._dirty(path):
            log.info(
                "ph.seams.workspace_git: keeping %s on %s — it has changes to review", path, ref
            )
            return True
        # `--force` even for the clean case: a tree measured clean a moment ago
        # can still hold an untracked file git would object to, and an ephemeral
        # tree is discarded *even if dirty*, which is the kind's whole promise.
        code, _, err = await self._git(toplevel, "worktree", "remove", "--force", str(path))
        if code != 0:
            log.warning(
                "ph.seams.workspace_git: could not remove the worktree at %s (%s)",
                path,
                err.strip() or f"git exited {code}",
            )
            return True
        if discard:
            await self._git(toplevel, "branch", "-D", ref)
            return False
        # `-d`, not `-D`: a clean worktree is not evidence that its branch was
        # merged — an agent that committed its work leaves nothing in
        # `git status`, and force-deleting there would discard exactly the work
        # the keep-dirty rule exists to protect. Git refuses instead, the branch
        # survives, and `kept` says so.
        code, _, _ = await self._git(toplevel, "branch", "-d", ref)
        return code != 0

    async def _dirty(self, path: Path) -> bool:
        """Whether this worktree holds anything worth keeping.

        `--porcelain` with untracked files included: a new file nobody staged is
        exactly the work a discarded worktree would lose, and it is the common
        shape of what an agent produces.
        """
        code, out, _ = await self._git(path, "status", "--porcelain", "--untracked-files=all")
        if code != 0:
            # Unreadable is treated as dirty: keeping a tree nobody wanted costs
            # disk, and removing one that held work costs the work.
            return True
        return bool(out.strip())

    async def _has_ref(self, toplevel: Path, ref: str) -> bool:
        code, _, _ = await self._git(
            toplevel, "show-ref", "--verify", "--quiet", f"refs/heads/{ref}"
        )
        return code == 0

    async def _git(self, cwd: Path, *args: str) -> tuple[int, str, str]:
        """One git invocation, through `ctx.subprocess` — never `os.system`.

        The seam scrubs the credential-shaped environment (F1) and reaps the
        child in a `finally` (F4); a bare `subprocess.run` here would opt this
        module out of both for no gain.
        """
        spec = SubprocessSpawnSpec(argv=("git", *args), cwd=cwd)
        result: tuple[int, str, str] = await self.ctx.subprocess.run(spec)
        return result


class Config(WireModel):
    """Row config for the worktree tier."""

    root: str | None = None
    """Where checkouts live. `$PH_HOME/worktrees` by default, and outside the
    repository on purpose: a worktree inside `base` would be walked by the
    agent's own `glob`, committed by its own `git add -A`, and nested one level
    deeper by every child."""


@plugin("workspace-git-worktree", inject=["workspace", "subprocess"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register the worktree tier as the workspace provider.

    Layered by a profile that wants isolation, never by `ph-base`: the tier costs
    a checkout per agent, and which profiles pay for it is P4-11's decision, not
    this module's.
    """
    provider = GitWorktreeProvider(ctx=ctx, root=default_home_path(config.root, "worktrees"))
    ctx.workspace.register_provider(provider, scope=ctx)
