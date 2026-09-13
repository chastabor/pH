"""`ctx.shell` — bash over `ctx.subprocess`, confined when a backend exists.

A thin layer, deliberately: everything that makes a command safe (the scrubbed
environment, the explicit cwd, termination and reaping) already belongs to the
subprocess seam, and everything that makes it *bounded* belongs to the sandbox
seam. What is left here is turning a command string into an argv and asking for
confinement when the policy says to.

@module ph.seams.shell
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..agent.types import AgentHandle
from ..cordis import Context, plugin
from ..keys import FS, SANDBOX, SHELL, SUBPROCESS
from .sandbox import ConfinedArgv, SandboxPolicy
from .subprocess import SubprocessSpawnSpec, platform_shell
from .workspace import workspace_of, workspace_policy

log = logging.getLogger("ph.seams.shell")

__all__ = ["ShellResult", "ShellService", "apply"]


@dataclass(frozen=True, slots=True)
class ShellResult:
    exit_code: int
    stdout: str
    stderr: str
    argv: tuple[str, ...]
    dropped: int = 0
    """Bytes the child printed past the seam's cap and nobody kept (P7-13)."""
    timed_out: bool = False
    """Whether `timeout_ms` ended it rather than the command finishing."""
    cap: int = 0
    """The per-stream ceiling that did the dropping, for the sentence that says so."""

    @property
    def truncated(self) -> bool:
        """Whether what this holds is a prefix of what the command printed.

        Here rather than only on `SubprocessResult`, because this is the record a
        renderer actually holds — the seam's own property had no reader at all."""
        return self.dropped > 0

    cwd: str = ""
    """Where the child actually ran.

    Reported rather than re-derived, because a caller that wants to *record*
    where a command ran had only one way to find out — repeat this seam's own
    `fs.root_for(agent)` derivation and hope the two never diverge. `run`
    honours a `cwd` override and a workspace redirection, so the copy could
    disagree with the fact it claimed to describe."""
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
        agent: AgentHandle | None = None,
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
        subprocess_service = self.ctx.require(SUBPROCESS)
        argv = (*platform_shell(), command)
        workspace = workspace_of(self.ctx, agent)
        if cwd is None:
            fs = self.ctx.get(FS)
            cwd = Path.cwd() if fs is None else fs.root_for(agent)
        if policy is None and workspace is not None:
            # The agent's own workspace as the writable root (E6), but *only*
            # where something can enforce it: `confine()` refuses rather than
            # passing through, so requesting confinement with no backend would
            # turn every shell command into a `SANDBOX_UNAVAILABLE` denial. The
            # policy is the same set `workspace-write-scope` prompts about, so
            # the two describe one boundary.
            sandbox = self.ctx.get(SANDBOX)
            if sandbox is not None and sandbox.available:
                policy = workspace_policy(workspace)
            else:
                # Said out loud, because `confined_by=None` cannot tell "the tier
                # wanted confinement and there was no backend" from "nobody
                # asked" — and collapsing those two is the passthrough this seam
                # refuses one layer down. P4-11's `containment.strict` is what
                # turns this from a notice into a refusal.
                log.debug(
                    "ph.seams.shell: %s has a workspace but no sandbox backend; running unconfined",
                    cwd,
                )
        agent_id = agent if isinstance(agent, str) else (agent.id if agent is not None else None)
        confined_by: str | None = None
        confined: ConfinedArgv | None = None
        if policy is not None:
            # Requesting confinement and getting none is an error, not a
            # fallback: see ph.seams.sandbox. The agent goes with it so a host
            # the proxy refuses is recorded in this agent's session.
            confined = self.ctx.require(SANDBOX).confine(argv, policy, agent=agent_id)
            argv = confined.argv
            confined_by = confined.backend
        spec = SubprocessSpawnSpec(
            argv=argv,
            cwd=cwd,
            timeout_ms=timeout_ms,
            # Additive, not wholesale: the redirection variables are the only
            # thing being said here, and a command that inherited nothing else
            # would not find its own toolchain.
            env=subprocess_service.env(extra=workspace.env)
            if workspace and workspace.env
            else None,
        )
        outcome = await subprocess_service.run(spec, scope=scope)
        if confined is not None and outcome.exit_code != 0:
            # stderr first: that is where these sentences come from, and passing the
            # streams separately avoids joining up to 16 MiB per confined command.
            # Only for a command that failed — see `report_denial`.
            self.ctx.require(SANDBOX).report_denial(
                confined, (outcome.stderr, outcome.stdout), agent_id
            )
        return ShellResult(
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            dropped=outcome.dropped,
            timed_out=outcome.timed_out,
            cap=subprocess_service.max_output,
            argv=argv,
            cwd=str(cwd),
            confined_by=confined_by,
        )


@plugin("shell-local", inject=[SUBPROCESS])
async def apply(ctx: Context, config: None) -> None:
    """Mount the local shell provider."""
    ctx.provide(SHELL, ShellService(ctx=ctx))
