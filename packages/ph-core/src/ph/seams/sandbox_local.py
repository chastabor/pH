"""`sandbox-local` — the first real confinement backend (P6-04).

`bwrap` on Linux, `sandbox-exec` on macOS. Both do the same job in the same shape:
take an argv and hand back an argv that the **kernel** bounds, which is what
separates the `sandbox` rung from every rung below it. A worktree bounds writes
that resolve against cwd and an overlay bounds the same set into a delta; only
this one refuses `open("/etc/passwd", "w")`.

**`confine` is pure argv construction, and that is deliberate.** It builds a
command line and runs nothing, so every rule this module encodes is assertable
without a kernel that will enforce it. The argv *is* the policy, so a test that
reads it is testing the thing — and it is what lets the Seatbelt half be reviewed
on a machine that cannot run it.

**The probe decides registration, and it is the only reason blind implementation
is safe.** A confinement backend that silently fails open is the worst object in
this codebase: every caller believes the kernel is holding the line and nothing
says otherwise. So registration is gated on a canary that writes to an absolute
path *outside* the workspace and checks the host copy is unchanged. A backend
whose rules are wrong in the permissive direction fails that and declines, rather
than claiming a tier it does not occupy.

**bwrap is verified against a real kernel; Seatbelt is not.** The prerequisite on
Ubuntu 23.10+ is an AppArmor profile at `/etc/apparmor.d/bwrap`:
`kernel.apparmor_restrict_unprivileged_userns=1` is the default there, and an
unprofiled `bwrap` is not setuid, so without one it dies with `setting up uid map:
Permission denied`. `sandbox-exec` is macOS-only and cannot be run here at all, so
its profile is written deny-by-default and the probe is what stands between
"unverified" and "claimed".

What was measured — of the confinement, and of what the wrapper costs:
`tests/test_sandbox_local.py`.

@module ph.seams.sandbox_local
"""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path
from ..wire import WireModel
from .diagnostics import Diagnostic, contribute
from .sandbox import ConfinedArgv, Enforcement, SandboxPolicy, writable_paths
from .subprocess import SubprocessSpawnSpec, first_line, scrub_env

__all__ = [
    "Bubblewrap",
    "Seatbelt",
    "apply",
    "local_backend",
    "probe_sandbox",
    "seatbelt_profile",
]

log = logging.getLogger("ph.seams.sandbox_local")


@dataclass(frozen=True, slots=True)
class Bubblewrap:
    """`bwrap` (Linux): a mount namespace where only named paths are writable."""

    enforcement: Enforcement = "full"

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv:
        """Wrap `argv` in a namespace that binds the filesystem read-only.

        `--ro-bind / /` first and writable binds after, because bwrap applies
        them in order and the last one covering a path wins — so the whole tree
        is read-only and the declared roots are punched back through.

        `--dev` and `--proc` rather than binds of the host's, **and
        `--unshare-pid` with them**: a fresh `/proc` mount alone still reports
        the PID namespace it is in, so the confined process could see and signal
        every process on the box. Filesystem confinement with the process table
        wide open is not what this rung claims.

        `danger-full-access` is still wrapped, with `/` bound writable. The mode
        means "no confinement" and the honest way to say that is a sandbox that
        permits everything, not an unwrapped argv — the seam's own contract is
        that a caller must never be able to mistake absence for confinement, and
        an argv that comes back unchanged is exactly that mistake.
        """
        parts = ["bwrap", "--die-with-parent"]
        if policy.mode == "danger-full-access":
            parts += ["--bind", "/", "/"]
        else:
            parts += ["--ro-bind", "/", "/"]
            for path in writable_paths(policy):
                parts += ["--bind", path, path]
        # `--unshare-pid` is what makes the `/proc` argument above true.
        # Measured: with `--proc /proc` alone the confined process saw **628**
        # PIDs against the host's 626 — the whole process table, signalable —
        # because a fresh `/proc` mount still reports the namespace it is in.
        # With the flag, 4. The docstring claimed process isolation the flags
        # did not deliver, which is the E1 shape one level down.
        parts += ["--dev", "/dev", "--proc", "/proc", "--unshare-pid"]
        if not policy.network:
            # The namespace is the enforcement: an unshared net namespace has no
            # interface but loopback, so this is not a filter that can be talked
            # past.
            parts.append("--unshare-net")
        return ConfinedArgv(
            argv=(*parts, "--", *argv), enforcement=self.enforcement, backend="bwrap"
        )


def seatbelt_profile(policy: SandboxPolicy) -> str:
    """The Seatbelt profile for one policy, as `sandbox-exec -p` takes it.

    **Deny by default and allow back**, which is the only direction that fails
    closed: a profile that allows by default and denies a list is one where every
    rule anybody forgot is permitted.

    Reads stay open. A confinement tier is about what an agent can *change* —
    pH's read boundary is `ctx.fs`'s, one layer up, and a Seatbelt profile that
    also denied reads would refuse the toolchain its own libraries and be
    switched off by the first person who met it.

    A separate function because it is the part worth reading in a test: the argv
    around it is three tokens and the profile is the policy.
    """
    if policy.mode == "danger-full-access":
        # Here rather than in `confine`, because this function is documented as
        # *the* profile for a policy and one policy's profile was being built
        # outside it — two places to look for one answer.
        return "(version 1)\n(allow default)"
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec process-fork signal)",
        "(allow sysctl-read)",
        "(allow file-read*)",
        '(allow file-write-data (literal "/dev/null") (literal "/dev/stdout")'
        ' (literal "/dev/stderr"))',
    ]
    for path in writable_paths(policy):
        lines.append(f'(allow file-write* (subpath "{path}"))')
    if policy.network:
        # Only the permitting arm is emitted: `(deny network*)` is a no-op under
        # `(deny default)` above, and a line that changes nothing is a line a
        # test can assert while proving nothing — which is what happened.
        lines.append("(allow network*)")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Seatbelt:
    """`sandbox-exec` (macOS): Apple's Seatbelt, driven by a generated profile."""

    enforcement: Enforcement = "full"

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv:
        """Wrap `argv` in `sandbox-exec -p <profile>`.

        Every mode goes through `seatbelt_profile`, `danger-full-access`
        included — it returns a permissive profile rather than this returning an
        unwrapped argv, for `Bubblewrap`'s reason: the caller must not be able to
        mistake absence for confinement.
        """
        return ConfinedArgv(
            argv=("sandbox-exec", "-p", seatbelt_profile(policy), *argv),
            enforcement=self.enforcement,
            backend="sandbox-exec",
        )


LocalBackend: TypeAlias = Bubblewrap | Seatbelt
"""The backends this row can mount.

Named because the union was spelled in two public signatures, and P6-04's own
summary still owes a third (Landlock) — which would otherwise edit both. Not
`SandboxProvider`: these are frozen dataclasses, and mypy rejects a frozen
attribute against that Protocol's settable `enforcement` member.
"""


def local_backend() -> tuple[LocalBackend | None, str]:
    """This platform's backend, or `None` and why there is not one.

    Platform first and binary second, so a Linux host without `bwrap` is told to
    install `bwrap` rather than that its platform is unsupported.
    """
    if sys.platform == "darwin":
        if shutil.which("sandbox-exec") is None:
            return None, "sandbox-exec is not installed"
        return Seatbelt(), "sandbox-exec"
    if sys.platform.startswith("linux"):
        if shutil.which("bwrap") is None:
            return None, "bwrap is not installed"
        return Bubblewrap(), "bwrap"
    return None, f"no confinement backend for {sys.platform}"


@dataclass(frozen=True, slots=True)
class SandboxProbe:
    """What one escape attempt learned about this host."""

    confines: bool
    because: str


async def probe_sandbox(ctx: Context, backend: LocalBackend, scratch: Path) -> SandboxProbe:
    """Try to escape the sandbox, and claim the tier only if the kernel stopped it.

    **Two checks in one process, and the second is what stops a useless backend
    passing.** A profile that denies everything refuses the escape and would look
    confining while being unusable — the first command an agent ran would fail
    and somebody would switch the tier off. So the probe writes *inside* the
    workspace, which must succeed, and *outside* it by absolute path, which must
    not.

    One spawn rather than two, because `sh` keeps going after a failed redirect
    so both writes are attempted and both outcomes are readable afterwards. The
    checks stay independent — "refused inside" and "escaped" remain
    distinguishable — while the probe stops paying twice for namespace setup:
    measured 14.5 ms as two spawns against 8.2 ms as one, on the serial startup
    path where nothing else can overlap it.

    Not the exit code, ever. That lesson is written twice next door: `agentfs run
    --experimental-sandbox` exits 0, prints nothing, and writes straight through
    to the host; and `agentfs run` prints its whole session banner when the
    sandbox failed to start and the command never ran. What the probe reads is
    the files.

    The backend's **own words** are the decline, when it has any. "the sandbox
    refused a write inside the workspace" is true and useless; `setting up uid
    map: Permission denied` is what tells an operator to add an AppArmor profile.

    Once, at mount, because it is a property of the host — and deliberately not
    cached across processes: `kernel.apparmor_restrict_unprivileged_userns` is a
    live sysctl, so a remembered "yes" is exactly the fail-open this row exists
    to prevent.
    """
    work = scratch / "sandbox-probe"
    workspace, outside = work / "work", work / "outside.txt"
    inside = workspace / "inside.txt"

    def prepare() -> None:
        shutil.rmtree(work, ignore_errors=True)
        workspace.mkdir(parents=True)
        outside.write_text("host", encoding="utf-8")

    def verdict() -> tuple[bool, str]:
        return inside.is_file(), outside.read_text(encoding="utf-8")

    await anyio.to_thread.run_sync(prepare)
    try:
        policy = SandboxPolicy(mode="workspace-write", workspace_root=str(workspace))
        confined = backend.confine(
            ("/bin/sh", "-c", f"echo landed > {inside}; echo escaped > {outside}"), policy
        )
        _, _, err = await ctx.subprocess.run(
            SubprocessSpawnSpec(argv=confined.argv, cwd=work, env=scrub_env())
        )
        landed, host = await anyio.to_thread.run_sync(verdict)
        if not landed:
            return SandboxProbe(
                False,
                first_line(err) or "the sandbox refused a write inside the workspace it was given",
            )
        if host != "host":
            return SandboxProbe(False, "an absolute-path write escaped the sandbox")
        return SandboxProbe(
            True, "writes are bounded to the workspace; an absolute path was refused"
        )
    except Exception as error:  # pragma: no cover - a host that fails in a new way
        return SandboxProbe(False, f"{type(error).__name__}: {error}")
    finally:
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(work, ignore_errors=True))


class Config(WireModel):
    """Row config for the local confinement backend."""

    root: str | None = None
    """Where the probe does its work. `$PH_HOME/sandbox` by default."""


@plugin("sandbox-local", inject=["sandbox", "subprocess"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Probe the host, and claim the sandbox slot only if the kernel enforced it.

    **Registration is what the probe gates**, for `workspace-agentfs`'s reason
    one module over — but the consequence here is sharper. `SandboxSeam.confine`
    raises when no backend is mounted, so a row that declines leaves callers with
    a refusal they can see. A row that *registered* and could not enforce would
    hand every caller a wrapped argv that bounds nothing, and `enforcement_of`
    would tell `containment.strict` the deployment is confined. Failing closed
    means not claiming the slot.

    The result reaches `ph doctor` either way: "why is strict refusing to start"
    is a question asked of the tool, and a row that declined in silence is
    indistinguishable from one nobody mounted.
    """
    backend, why = local_backend()
    result = (
        SandboxProbe(False, why)
        if backend is None
        else await probe_sandbox(ctx, backend, default_home_path(config.root, "sandbox"))
    )

    contribute(
        ctx,
        Diagnostic(
            id="sandbox-local",
            title="Local confinement",
            read=lambda: [
                ("backend", why),
                ("confines", "yes" if result.confines else "no"),
                ("because", result.because),
            ],
            order=15,
        ),
    )
    if backend is None or not result.confines:
        log.info("ph.seams.sandbox_local: declining — %s", result.because)
        return
    ctx.sandbox.register_provider(backend)
