"""`ctx.shell` — bash over `ctx.subprocess`, confined when a backend exists.

A thin layer, deliberately: everything that makes a command safe (the scrubbed
environment, the explicit cwd, termination and reaping) already belongs to the
subprocess seam, and everything that makes it *bounded* belongs to the sandbox
seam. What is left here is turning a command string into an argv and asking for
confinement when the policy says to.

**The seam owns the pair a person's command writes** (P10-08): `SHELL_COMMAND`,
`shell/command` opened before the child starts and `shell/result` settled after,
through `ctx.intents`. Declared here rather than beside `ph_app.shell`, its one
producer, because repair must settle the pair on any resume that has ph-core —
and ph-core cannot import the app.

@module ph.seams.shell
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..agent.types import AgentHandle
from ..cancel import Cancellation
from ..cordis import Context, plugin
from ..json import JsonObject
from ..keys import FS, SANDBOX, SHELL, SUBPROCESS
from ..session import IntentKind, SessionEvent, Unsettled, declare_intent
from .sandbox import ConfinedArgv, SandboxPolicy
from .subprocess import SubprocessSpawnSpec, platform_shell
from .workspace import workspace_of, workspace_policy

log = logging.getLogger("ph.seams.shell")

__all__ = ["SHELL_COMMAND", "ShellResult", "ShellService", "apply", "command_seq"]


def _opened_key(event: SessionEvent) -> str:
    """A command is keyed by its own seq: nothing else about it is unique."""
    return str(event.seq)


def command_seq(event: SessionEvent) -> str | None:
    """The command a `shell/result` settles — its `commandSeq`, as a key."""
    seq = event.data.get("commandSeq")
    return str(seq) if isinstance(seq, int) and not isinstance(seq, bool) else None


def _interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle for a command whose own result was never written.

    `interrupted` says which half is known: `outcome-unknown` when the record
    was on disk and the harness then stopped — the child may have run, may have
    finished — and `not-started` when the record could not be written, so the
    child was never spawned. `ok` is false either way: nothing here saw it
    succeed.
    """
    return {"commandSeq": opened.seq, "ok": False, "interrupted": why}


SHELL_COMMAND = declare_intent(
    IntentKind(
        opened="shell/command",
        settled="shell/result",
        opened_key=_opened_key,
        settled_key=command_seq,
        # The child may have run: a command that takes the daemon down with it is
        # the case this pair exists for, and repair cannot know how far it got.
        orphan="outcome-unknown",
        # On disk before the child starts (F9): a command whose record cannot be
        # written does not run.
        barrier="durable",
        closer=_interrupted,
        owner="ph.seams.shell",
    )
)
"""A person's `!` or `!!`: the command, then what it did."""


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
    canceled: bool = False
    """Whether the caller's `signal` ended it rather than the command finishing.

    Carried rather than folded into `timed_out`: see `SubprocessResult.canceled`.
    A caller that conflates them reports "the command timed out" to a person who
    pressed stop."""
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
    honors a `cwd` override and a workspace redirection, so the copy could
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
        signal: Cancellation | None = None,
    ) -> ShellResult:
        """Run one command. `policy` requests confinement and fails if it cannot.

        **`agent` is how a command lands in the right place.** Given one, the
        seam resolves the working directory and the workspace's redirection
        environment (D21, E12) itself, rather than making each shell-shaped tool
        remember to. `tool-bash` was doing that derivation, and it is the same
        rule for every caller — a command run *for* an agent runs in that
        agent's workspace, or the tier bounds the tools and nothing else.

        `cwd` overrides, for a caller that means somewhere specific.

        `signal` is how a *person* stops a command (C7). `timeout_ms` is a number
        chosen before the command ran; this is the decision made while watching
        it, and without it `bash sleep 3600` could not be interrupted at all.

        `Cancellation`, not `CancelToken`, for `AgentHandle.signal`'s reason: a
        seam may ask whether the work is still wanted and may not end it.
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
        outcome = await subprocess_service.run(spec, scope=scope, signal=signal)
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
            canceled=outcome.canceled,
            cap=subprocess_service.max_output,
            argv=argv,
            cwd=str(cwd),
            confined_by=confined_by,
        )


@plugin("shell-local", inject=[SUBPROCESS])
async def apply(ctx: Context, config: None) -> None:
    """Mount the local shell provider."""
    ctx.provide(SHELL, ShellService(ctx=ctx))
