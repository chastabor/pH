"""`workspace-readonly-scratch` — the `sandbox` rung's workspace kind (P6-05, E3, E11).

The one kind where the repository is genuinely unwritable, and the only one a tier
below `sandbox` cannot fake. The agent's root is a directory inside its own
scratch; the repository is readable at its own path and writable nowhere.

**The workspace produces the policy; it does not enforce it.** `writable_roots`
answers `(root, scratch)` and `workspace_policy` turns that into
`SandboxPolicy(mode="workspace-write", …)`, so rooting at scratch is the whole
mechanism — the repository is unwritable because it is not in the writable set.
`ctx.shell` confines every command against that policy.

Invariants this row holds:

* **It claims the rung only when a backend is mounted and enforcing.**
  `repo_writable=False` is the one field §12 Q10 says must never be a statement of
  intent, and without a kernel behind it, handing it out is E1's failure in the
  field E1 is about.
* **`access` does not widen the tier.** A caller asking `write` gets the kind too,
  with `repo_writable=False` saying it did not get what it asked for.

@module ph.seams.workspace_scratch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from .diagnostics import Diagnostic, contribute
from .sandbox import enforcement_of
from .workspace import ContainmentTier, Workspace, WorkspaceAccess, redirection_env

__all__ = ["ReadonlyScratchProvider", "apply"]

log = logging.getLogger("ph.seams.workspace_scratch")

WORK_DIR = "work"
"""The agent's root, a subdirectory of scratch so the redirected caches
`redirection_env` points there are siblings of the work tree rather than litter
inside it."""


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

        The enforcement question is settled at registration, so there is no
        per-acquire failure left for this to have.
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


@plugin("workspace-readonly-scratch", inject=["workspace", "sandbox"])
async def apply(ctx: Context, _config: object) -> None:
    """Claim the `sandbox` rung on `profile/mounted`, once a backend can enforce it."""
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
        ctx.workspace.register_provider(ReadonlyScratchProvider())

    ctx.on("profile/mounted", claim)


def _describe(ctx: Context) -> list[tuple[str, str]]:
    """What `ph doctor` prints, read when it is asked rather than at mount.

    Only the reason the rung is *not* held: whether it is, and what the backend
    enforces, are already the Containment section's two rows, and a fact stated
    twice is the drift `enforcement_of` exists to prevent.
    """
    because = _unenforceable(ctx)
    return [("declined", because)] if because is not None else []


def _unenforceable(ctx: Context) -> str | None:
    """Why this rung cannot be claimed, or `None` if it can.

    A sentence rather than a bool, because the diagnostic prints it: "why am I not
    on the sandbox tier" is a question asked of the tool rather than of the source.
    Deliberately the same threshold `ContainmentService.verify` applies — a partial
    boundary is a refusal rather than a downgrade (E8) — so the two never disagree
    about one backend.
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
