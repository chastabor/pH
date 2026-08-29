"""`ctx.shell` — bash over `ctx.subprocess`, confined when a backend exists.

A thin layer, deliberately: everything that makes a command safe (the scrubbed
environment, the explicit cwd, termination and reaping) already belongs to the
subprocess seam, and everything that makes it *bounded* belongs to the sandbox
seam. What is left here is turning a command string into an argv and asking for
confinement when the policy says to.

@module ph.seams.shell
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..cordis import Context, plugin
from .sandbox import SandboxPolicy
from .subprocess import SubprocessSpawnSpec, platform_shell, scrub_env
from .workspace import workspace_of

__all__ = ["ShellResult", "ShellService", "apply"]


@dataclass(frozen=True, slots=True)
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    argv: tuple[str, ...]
    confined_by: str | None = None
    """The backend that bounded this run; `None` means nothing did."""


@dataclass(slots=True)
class ShellService:
    """The service published as `ctx.shell`."""

    ctx: Context

    async def run(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        agent: Any = None,
        timeout_ms: int | None = None,
        policy: SandboxPolicy | None = None,
        scope: Context | None = None,
    ) -> ShellResult:
        """Run one command. `policy` requests confinement and fails if it cannot.

        **`agent` is how a command lands in the right place.** Given one, the
        seam resolves the working directory and the workspace's redirection
        environment (D21, E12) itself, rather than making each shell-shaped tool
        remember to. `tool-bash` was doing that derivation, and it is the same
        rule for every caller — a command run *for* an agent runs in that
        agent's workspace, or the tier bounds the tools and nothing else.

        `cwd` overrides, for a caller that means somewhere specific.
        """
        argv = (*platform_shell(), command)
        workspace = workspace_of(self.ctx, agent)
        if cwd is None:
            fs = self.ctx.get("fs")
            cwd = Path.cwd() if fs is None else fs.root_for(agent)
        confined_by: str | None = None
        if policy is not None:
            # Requesting confinement and getting none is an error, not a
            # fallback: see ph.seams.sandbox.
            confined = self.ctx.sandbox.confine(argv, policy)
            argv = confined.argv
            confined_by = confined.backend
        spec = SubprocessSpawnSpec(
            argv=argv,
            cwd=cwd,
            grace_ms=timeout_ms or 5_000,
            # Additive, not wholesale: the redirection variables are the only
            # thing being said here, and a command that inherited nothing else
            # would not find its own toolchain.
            env=scrub_env(extra=workspace.env) if workspace and workspace.env else None,
        )
        code, out, err = await self.ctx.subprocess.run(spec, scope=scope)
        return ShellResult(
            exit_code=code, stdout=out, stderr=err, argv=argv, confined_by=confined_by
        )


@plugin("shell-local", inject=["subprocess"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the local shell provider."""
    ctx.provide("shell", ShellService(ctx=ctx))
