"""`workspace-readonly-scratch`: the workspace kind of the `sandbox` rung (P6-05, E3, E11).

This is the one kind where the repository is truly unwritable, and the only
kind that a tier below `sandbox` cannot fake. The root of the agent is a
directory inside its own scratch directory. The repository is readable at its
own path and writable nowhere.

**The workspace produces the policy. It does not enforce it.**
`writable_roots` returns `(root, scratch)`, and `workspace_policy` turns that
pair into the `workspace_root` and the `writable_extra` of a `SandboxPolicy`.
So to root the workspace at the scratch directory is the whole mechanism: the
repository is unwritable because it is not in the writable set. `ctx.shell`
confines every command against that policy.

This row holds two invariants:

* **It claims the rung only when a backend is mounted and enforcing.**
  `repo_writable=False` is the one field that §12 Q10 says must never be a
  statement of intent. To hand it out without a kernel behind it is the
  failure that E1 describes, in the field that E1 is about.
* **`access` does not widen the tier.** A caller that asks for `write` gets
  this kind too, and `repo_writable=False` says that it did not get what it
  asked for.

@module ph.seams.workspace_scratch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..keys import SANDBOX, WORKSPACE
from .diagnostics import Diagnostic, contribute
from .sandbox import enforcement_of
from .workspace import ContainmentTier, Workspace, WorkspaceAccess, redirection_env

__all__ = ["ReadonlyScratchProvider", "apply"]

log = logging.getLogger("ph.seams.workspace_scratch")

WORK_DIR = "work"
"""The root of the agent. It is a subdirectory of the scratch directory, so that
the caches that `redirection_env` redirects there are siblings of the work tree
and not litter inside it."""


@dataclass(frozen=True, slots=True)
class ReadonlyScratchProvider:
    """The `sandbox` tier: a writable scratch, and a repository that is not."""

    tier: ContainmentTier = "sandbox"

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace:
        """A workspace rooted at scratch, whatever was asked for. Never declines.

        The enforcement question is settled at registration, so no per-acquire
        failure is left for this method.
        """
        root = scratch / WORK_DIR
        await anyio.to_thread.run_sync(lambda: root.mkdir(parents=True, exist_ok=True))
        return Workspace(
            root=root,
            scratch=scratch,
            kind="readonly-scratch",
            repo_writable=False,
            env=redirection_env(scratch),
        )


@plugin("workspace-readonly-scratch", affects="environment", inject=[WORKSPACE, SANDBOX])
async def apply(ctx: Context, config: None) -> None:
    """Claim the `sandbox` rung on `profile/mounted`, if a backend can enforce it."""
    contribute(
        ctx,
        Diagnostic(
            id="workspace-readonly-scratch",
            title="Read-only scratch workspaces",
            read=lambda: _describe(ctx),
            order=25,
        ),
    )

    def claim() -> None:
        because = _unenforceable(ctx)
        if because is not None:
            log.info("ph.seams.workspace_scratch: declining — %s", because)
            return
        ctx.require(WORKSPACE).register_provider(ReadonlyScratchProvider())

    ctx.on("profile/mounted", claim)


def _describe(ctx: Context) -> list[tuple[str, str]]:
    """What `phern doctor` prints. It is read when doctor asks, not at mount.

    It gives only the reason why the rung is *not* held. Whether the rung is
    held, and what the backend enforces, are already the two rows of the
    Containment section. A fact that is stated in two places can drift, and
    `enforcement_of` exists to prevent that drift.
    """
    because = _unenforceable(ctx)
    return [("declined", because)] if because is not None else []


def _unenforceable(ctx: Context) -> str | None:
    """Why this rung cannot be claimed, or `None` if it can.

    The result is a sentence and not a bool, because the diagnostic prints it.
    "Why am I not on the sandbox tier" is a question that a user asks of the
    tool and not of the source. The threshold is deliberately the same one that
    `ContainmentService.verify` applies: a partial boundary is a refusal and
    not a downgrade (E8). So the two never disagree about one backend.
    """
    enforcement = enforcement_of(ctx)
    if enforcement is None:
        return "no sandbox backend is mounted, so an unwritable repository cannot be enforced"
    if enforcement != "full":
        return (
            f'the mounted backend enforces "{enforcement}", and a partial boundary is a '
            "refusal rather than a downgrade"
        )
    return None
