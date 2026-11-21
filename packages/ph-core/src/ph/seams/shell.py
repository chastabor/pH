"""`ctx.shell`: bash over `ctx.subprocess`, confined when a backend exists.

This layer is thin on purpose. Everything that makes a command safe (the
scrubbed environment, the explicit cwd, termination and reaping) already
belongs to the subprocess seam. Everything that makes a command *bounded*
belongs to the sandbox seam. What is left here is to turn a command string into
an argv, and to ask for confinement when the policy says so.

A shell command outside a tool call writes a pair of records (P10-08):
`shell/command`, opened before the child starts, and `shell/result`, settled
after, both through `ctx.intents`. The pair is `SHELL_COMMAND`. It is declared
in `ph.session.kinds` with the other kinds of ph-core (T4).
`ShellService.run_recorded` fills it for its two callers: `ph_app.shell` for
the `!` or `!!` of a user, and `ph.commands.autonomous` for the gate of an
autonomous goal.

@module ph.seams.shell
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..agent.types import AgentHandle
from ..cancel import Cancellation
from ..characters import head_bytes, utf8_size
from ..cordis import Context, plugin
from ..json import JsonObject, JsonValue
from ..keys import FS, SANDBOX, SHELL, SUBPROCESS
from ..session import Session, SessionEvent, intents_of
from ..session.kinds import SHELL_COMMAND
from .sandbox import ConfinedArgv, SandboxPolicy, grants_apart
from .subprocess import SubprocessSpawnSpec, platform_shell
from .workspace import confined_env, workspace_of, workspace_policy

log = logging.getLogger("ph.seams.shell")

__all__ = ["SHELL_OUTPUT", "ShellResult", "ShellService", "apply"]

SHELL_OUTPUT = 64 * 1024
"""How much of one stream the `shell/result` of a recorded command keeps.

The amount is bounded because the log is durable and `!!find /` is one
keystroke. It is generous because a user runs a command to read what it
printed. The bound applies to each stream. Truncation is recorded on the event,
and is not left for a reader to infer from a suspiciously round length.

**This is a display clip on top of a bound that already held.** `ctx.subprocess`
caps what it *keeps* from a child (P7-13), so the string is already bounded
when this runs. This constant decides how much of the string a durable log
should carry. That is a much smaller number and a different question. Both
bounds can apply to one command, and the event records them separately:
`dropped` counts what the seam discarded, and `clipped` says that this clip
kept output back.
"""


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

    This is carried separately and not folded into `timed_out`. See
    `SubprocessResult.canceled`. A caller that conflates the two reports "the
    command timed out" to a user who pressed stop."""
    cap: int = 0
    """The per-stream ceiling that did the dropping, for the sentence that says so."""

    cwd: str = ""
    """Where the child actually ran.

    The directory is reported and not derived again. A caller that wanted to
    *record* where a command ran had only one way to find out: repeat the
    `fs.root_for(agent)` derivation of this seam and hope that the two never
    diverge. `run` honors a `cwd` override and a workspace redirection, so the
    copy could disagree with the fact that it claimed to describe."""
    confined_by: str | None = None
    """The backend that bounded this run. `None` means that nothing did."""


def _shell_settled(command_seq: int, result: ShellResult) -> JsonObject:
    """The `shell/result` that settles the command recorded at `command_seq`."""
    return {
        # The command that this settles. A fold can then pair the two records,
        # and a front end need not assume that only one command is ever in
        # flight. Two attached UIs can each be running one, and the log is what
        # tells them apart.
        "commandSeq": command_seq,
        "exitCode": result.exit_code,
        "ok": result.exit_code == 0,
        "cwd": result.cwd,
        "confinedBy": result.confined_by,
        "stdout": head_bytes(result.stdout, SHELL_OUTPUT),
        "stderr": head_bytes(result.stderr, SHELL_OUTPUT),
        # **Two bounds, two keys.** A fold that explains why a user sees a
        # prefix must tell "the seam threw 37 MB away" from "we kept 64 KiB of
        # what survived". One `truncated` bool lost that. The sizes are
        # compared on the originals, in the bytes that the cap names, *before*
        # anything is concatenated.
        "dropped": result.dropped,
        "cap": result.cap,
        "timedOut": result.timed_out,
        "clipped": utf8_size(result.stdout) > SHELL_OUTPUT
        or utf8_size(result.stderr) > SHELL_OUTPUT,
    }


@dataclass(slots=True)
class ShellService:
    """The service published as `ctx.shell`."""

    ctx: Context

    async def run_recorded(
        self,
        command: str,
        *,
        agent: AgentHandle,
        session: Session,
        surface: bool = False,
        gate: str | None = None,
    ) -> tuple[ShellResult, SessionEvent]:
        """Run `command` as `SHELL_COMMAND`: recorded, run, settled. Returns what ran
        and the settle.

        **The record is on disk before the child starts** (F9). That is the
        `durable` barrier of the kind. A command that hangs, or that takes the
        daemon down with it, still shows in the log what was started. A command
        whose record could not be written does not run (`IntentNotDurable`).
        When `run` raises, `claim` settles the pair `outcome-unknown` and does
        not leave it open.

        This is one recipe for the two callers that run a shell command outside
        a tool call (S16). One is the `!` or `!!` of a user, and `surface` says
        which. The other is the gate of an autonomous goal, and `gate` names
        the goal so that a front end can label the card. The gate used to run
        with nothing recorded first, so a crash in the middle of a gate left no
        trace that it had run.
        """
        intents = intents_of(self.ctx)
        # `surface` goes on the *command* event and not on the result. It is
        # what the user asked for, it is known before the child starts, and a
        # front end draws the card from this event. Without it, `!make` and
        # `!!make` render the same, and nobody can see which one is about to
        # put output in front of the model.
        #
        # The key is written on every command, `False` included, and not only
        # when it is true. With one shape for every `shell/command`, no reader
        # has to know that an absent key means the quiet one of the two.
        opened: dict[str, JsonValue] = {"command": command, "surface": surface}
        if gate is not None:
            opened["gate"] = gate
        async with intents.claim(session, SHELL_COMMAND, opened) as held:
            result = await self.run(command, agent=agent)
            settled = intents.settle(session, held, _shell_settled(held.opened.seq, result))
        return result, settled

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
        """Run one command. `policy` requests confinement, and the call fails if
        no backend can confine.

        **`agent` puts a command in the right place.** Given an agent, the seam
        itself resolves the working directory and the redirection environment
        of the workspace (D21, E12). Each shell-shaped tool does not have to
        remember to do so. `tool-bash` used to do that derivation, and the rule
        is the same for every caller: a command run *for* an agent runs in the
        workspace of that agent. Otherwise the tier bounds the tools and
        nothing else.

        `cwd` overrides the directory, for a caller that means a specific
        place.

        `signal` is how a *user* stops a command (C7). `timeout_ms` is a number
        chosen before the command ran. `signal` is the decision made while
        watching the command, and without it `bash sleep 3600` could not be
        interrupted at all.

        The type is `Cancellation` and not `CancelToken`, for the reason that
        `AgentHandle.signal` gives: a seam may ask whether the work is still
        wanted, and may not end it.
        """
        return await self.run_argv(
            (*platform_shell(), command),
            cwd=cwd,
            agent=agent,
            timeout_ms=timeout_ms,
            policy=policy,
            scope=scope,
            signal=signal,
        )

    async def run_argv(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path | None = None,
        agent: AgentHandle | None = None,
        timeout_ms: int | None = None,
        policy: SandboxPolicy | None = None,
        scope: Context | None = None,
        signal: Cancellation | None = None,
        keep_env: Collection[str] | None = None,
        extra_env: Mapping[str, str] | None = None,
        writable_extra: Sequence[str] = (),
        on_stderr_line: Callable[[str], None] | None = None,
    ) -> ShellResult:
        """Run one argv for an agent, with the placement and confinement of `run`.
        This is for a caller that has its own interpreter line.

        `run` is this method with the login shell of the platform in front. A
        caller whose *output is data* cannot use that shell. `bash -lc` reads
        the profile of the user, and whatever a profile prints lands in front
        of the result. So the caller names its own interpreter
        (`bash -c <script>`) and still gets the working directory of the agent,
        the redirection variables of the workspace and the sandbox. `ph-utcp`
        is that caller: the stdout of a UTCP tool is parsed as its answer.

        The environment starts as the scrubbed environment of the deployment,
        which the subprocess seam supplies. It is built in three layers:

        * `keep_env` keeps only these names from it, for a command that
          declares what it reads. It selects *from* the scrubbed environment.
          So naming a variable that the deployment dropped, such as a
          credential, does not fetch it.
        * The redirection variables of the workspace go over that. They are
          where the command runs, and not something that it opted into: a
          command that inherits `TMPDIR` gets the `TMPDIR` of the agent. A
          confined command gets a `TMPDIR` in the scratch directory of the
          agent whatever its workspace, because the sandbox leaves it no other
          writable temporary directory.
        * `extra_env` goes over everything. These are values that the caller
          supplies itself, so they win.

        `writable_extra` adds writable paths to the workspace policy that this
        method derives. It is for a command that keeps something outside the
        workspace and the scratch directory of the agent, such as a grammar
        cache under `$PH_CACHE`. A grant holds under every sandbox mode,
        `read-only` included. A grant that overlaps the workspace is dropped
        (`ph.seams.sandbox.grants_apart`): read-only bars writes to the tree,
        and no grant reopens it. `writable_extra` widens nothing that a
        `policy` passed in already says, and it means nothing where no backend
        confines the command.

        `on_stderr_line` receives each line of the stderr of the command as it
        is written, and that stream is then not in the result
        (`SubprocessSpawnSpec.on_stderr_line`). It is for a command that
        reports its progress while it runs.
        """
        subprocess_service = self.ctx.require(SUBPROCESS)
        workspace = workspace_of(self.ctx, agent)
        if cwd is None:
            fs = self.ctx.get(FS)
            cwd = Path.cwd() if fs is None else fs.root_for(agent)
        if policy is None and workspace is not None:
            # The workspace of the agent itself is the writable root (E6), but
            # *only* where something can enforce it. `confine()` refuses and
            # does not pass through, so a request for confinement with no
            # backend would turn every shell command into a
            # `SANDBOX_UNAVAILABLE` denial. The policy is the same set that the
            # `outside-workspace` rule of `permissions-fs` prompts about, so
            # the two describe one boundary.
            sandbox = self.ctx.get(SANDBOX)
            if sandbox is not None and sandbox.available:
                policy = workspace_policy(workspace)
                if writable_extra:
                    # A grant is honored under every mode, and refused where it
                    # overlaps the workspace. See `grants_apart` for both
                    # halves of the rule.
                    granted, refused = grants_apart(str(workspace.root), writable_extra)
                    if refused:
                        log.debug(
                            "ph.seams.shell: not granted, because it overlaps the workspace %s: %s",
                            workspace.root,
                            ", ".join(refused),
                        )
                    policy = policy.model_copy(
                        update={"writable_extra": [*(policy.writable_extra or []), *granted]}
                    )
            else:
                # This is logged because `confined_by=None` cannot tell "the
                # tier wanted confinement and there was no backend" from
                # "nobody asked". To collapse those two is the passthrough
                # that this seam refuses one layer down. `containment.strict`
                # (P4-11) is what turns this from a notice into a refusal.
                log.debug(
                    "ph.seams.shell: %s has a workspace but no sandbox backend; running unconfined",
                    cwd,
                )
        agent_id = agent if isinstance(agent, str) else (agent.id if agent is not None else None)
        confined_by: str | None = None
        confined: ConfinedArgv | None = None
        if policy is not None:
            # A request for confinement that gets none is an error and not a
            # fallback: see `ph.seams.sandbox`. The agent goes with the
            # request, so that a host that the proxy refuses is recorded in
            # the session of this agent.
            confined = self.ctx.require(SANDBOX).confine(argv, policy, agent=agent_id)
            argv = confined.argv
            confined_by = confined.backend
        redirection: Mapping[str, str] = {}
        if workspace is not None:
            # A confined command has no `/tmp` of its own (`confined_env`).
            redirection = confined_env(workspace) if confined is not None else workspace.env
        env: dict[str, str] | None = None
        if keep_env is not None or extra_env or redirection:
            # The environment is added to and not replaced. A command that
            # inherited nothing would not find its own toolchain. So this
            # starts from the environment of the seam, and a caller that wants
            # less says which names (`keep_env`).
            env = subprocess_service.env()
            if keep_env is not None:
                env = {name: env[name] for name in keep_env if name in env}
            if redirection:
                env.update(redirection)
            if extra_env:
                env.update(extra_env)
        spec = (
            SubprocessSpawnSpec(
                argv=argv, cwd=cwd, timeout_ms=timeout_ms, env=env, on_stderr_line=on_stderr_line
            )
            if confined is None
            else confined.spawn_spec(
                cwd, timeout_ms=timeout_ms, env=env, on_stderr_line=on_stderr_line
            )
        )
        outcome = await subprocess_service.run(spec, scope=scope, signal=signal)
        if confined is not None and outcome.exit_code != 0:
            # stderr goes first, because that is where these sentences come
            # from. Passing the streams separately avoids a join of up to
            # 16 MiB for each confined command at the default cap. This runs
            # only for a command that failed: see `report_denial`.
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


@plugin("shell-local", affects="environment", inject=[SUBPROCESS])
async def apply(ctx: Context, config: None) -> None:
    """Mount the local shell provider."""
    ctx.provide(SHELL, ShellService(ctx=ctx))
