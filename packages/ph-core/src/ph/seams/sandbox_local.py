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
on a machine that cannot run it. That stays true with the network door: when the
effective policy carries an `Egress`, the argv gains the proxy variables and a
`socat` shim wrapped around the command, and still nothing here runs.

**The probe decides registration, and it is the only reason blind implementation
is safe.** A confinement backend that silently fails open is the worst object in
this codebase: every caller believes the kernel is holding the line and nothing
says otherwise. So registration is gated on a canary that writes to an absolute
path *outside* the workspace and checks the host copy is unchanged. A backend
whose rules are wrong in the permissive direction fails that and declines, rather
than claiming a tier it does not occupy.

**The egress bridge is probed the same way, and claimed only when it works.** The
proxy is started, and a confined `socat` sends one `CONNECT` through the shim to
the host the proxy always refuses; the bridge is registered on the seam only if
the proxy's own 403 comes back. Without `socat`, or with a bridge that does not
answer, the row registers no bridge — and `allowlist` then means no network,
which `SandboxSeam.network_posture` says in so many words.

**bwrap is verified against a real kernel; Seatbelt is not.** The prerequisite on
Ubuntu 23.10+ is an AppArmor profile at `/etc/apparmor.d/bwrap`:
`kernel.apparmor_restrict_unprivileged_userns=1` is the default there, and an
unprofiled `bwrap` is not setuid, so without one it dies with `setting up uid map:
Permission denied`. `sandbox-exec` is macOS-only and cannot be run here at all, so
its profile is written deny-by-default and the probe is what stands between
"unverified" and "claimed" — the egress lines included.

@module ph.seams.sandbox_local
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import TypeAlias
from urllib.parse import quote

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path, resolve_roots
from ..resources import temporary_directory
from ..wire import WireModel
from .diagnostics import Diagnostic, contribute
from .sandbox import (
    ConfinedArgv,
    Denial,
    DenialKind,
    Egress,
    Enforcement,
    SandboxPolicy,
    writable_paths,
)
from .sandbox_egress import PROBE_HOST, EgressProxy
from .subprocess import SubprocessSpawnSpec, first_line

__all__ = [
    "BYPASS_VARIABLES",
    "DENIAL_SIGNATURES",
    "EGRESS_PORT",
    "PROXY_PASSWORD",
    "PROXY_VARIABLES",
    "Bubblewrap",
    "Seatbelt",
    "apply",
    "egress_shim",
    "egress_socket_path",
    "local_backend",
    "probe_egress",
    "probe_sandbox",
    "proxy_url",
    "seatbelt_profile",
]

log = logging.getLogger("ph.seams.sandbox_local")

EGRESS_PORT = 3128
"""Where the shim listens inside a sandbox.

One number for every sandbox, because each has its own network namespace and a
port in one is invisible from another — the only thing it can collide with is a
listener the confined command itself opens. Squid's port, so a person who sees
`HTTPS_PROXY=http://127.0.0.1:3128` inside knows what kind of thing is there."""

PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
"""Both cases, because the tools disagree: `curl` reads the lower-case ones and
refuses upper-case `HTTP_PROXY` on purpose (CGI hijacking), while Go and most
Python clients read either."""

BYPASS_VARIABLES = ("NO_PROXY", "no_proxy")
"""Unset inside, whatever the host had: a host listed here is one the client would
dial directly, and inside the namespace there is nothing to dial."""

SOCKET_PATH_MAX = 100
"""Headroom under `sun_path`'s 108 bytes. A unix socket path longer than that
cannot be bound at all."""

DENIAL_SIGNATURES: tuple[tuple[DenialKind, str], ...] = (
    ("filesystem", "Read-only file system"),
    ("network", "Network is unreachable"),
    ("network", "Could not resolve host"),
    ("network", "Temporary failure in name resolution"),
    ("network", "Name or service not known"),
)
"""What `bwrap`'s silent refusals look like from inside a command.

Here rather than on the seam because every one of these is a fact about **this
backend's platform** — Linux, glibc, `bwrap` — and Seatbelt refuses in words
nobody here has measured. A table of these sentences in the seam definition would
be one kernel's dialect asserted on every backend's behalf; `Bubblewrap` is a
`DenialReader` and `Seatbelt` is deliberately not.

Measured, not guessed: a redirect onto the read-only tree prints the first from
`sh`, and a raw `connect()` inside an unshared namespace raises `[Errno 101]
Network is unreachable`; the three resolver texts are what `curl`, glibc and
Python print when a command bypasses the proxy variables and asks for DNS a
namespace does not have. Substrings rather than patterns, and only ones no
ordinary command prints on success — which is also why `ShellService` only asks
about a command that actually failed.
"""


# ------------------------------------------------------------------ bwrap --


PROXY_PASSWORD = "ph"
"""The password beside the agent in the proxy URL — a constant, not a secret.

Present because `urllib` sends `Proxy-Authorization` only when *both* parts of the
user info are non-empty (`if user and password:`), so an agent-only URL reached the
proxy with no identity and the refusal was charged to nobody — measured, in the
end-to-end test. The proxy reads the user and ignores this."""


def proxy_url(egress: Egress) -> str:
    """`http://<agent>:ph@127.0.0.1:<port>` — the agent in the user part.

    Every client that honours the URL sends the user back as Basic credentials
    on each request, which is how the proxy knows whose session a refusal belongs
    in without a side channel. Percent-encoded, because an agent id is a session
    id and a session id is whatever the store made it.
    """
    user = f"{quote(egress.agent, safe='')}:{PROXY_PASSWORD}@" if egress.agent else ""
    return f"http://{user}127.0.0.1:{egress.port}"


def egress_shim(argv: tuple[str, ...], egress: Egress) -> tuple[str, ...]:
    """Wrap `argv` so a `socat` shim is listening on loopback before it runs.

    `sh -c` starts the shim in the background, waits until the port accepts a
    connection, and `exec`s the command — so the command's exit status and
    signals are its own, and the shim is a sibling that dies when the PID
    namespace does (`--unshare-pid`, measured: nothing survives the command).

    The wait is a bounded poll rather than a fixed sleep, because the shim is up in
    a few milliseconds and a command that reached for the network in its first
    would otherwise race it. A shim that never comes up is *said* on stderr and
    the command runs anyway — with no network, which is what it had before.

    **The interval is two-phase**, because the listener binds in 1-3 ms and a flat
    10 ms sleep therefore costs a whole tick on a probe that was always going to
    miss once. Measured here, median of 21 runs of `/bin/true`: `bwrap` alone
    **6.1 ms**, with a flat 10 ms poll **12.8 ms**, with 1 ms for the first 40 tries
    and 10 ms after **10.8 ms** — so **2 ms back on every confined command**, and
    the same ~2 s budget for a host where something is actually wrong. (A review
    pass measuring against the real proxy rather than a bare accepting socket saw a
    larger gap, 25.5 ms against 15.8; the direction is the same and the smaller
    number is the one this comment is willing to stand behind.) The spawns are not
    the cost — a socat-native `retry=` loop measured within a millisecond of this
    one — the interval is.
    """
    listen = f"TCP-LISTEN:{egress.port},bind=127.0.0.1,fork,reuseaddr"
    probe = f"TCP:127.0.0.1:{egress.port},connect-timeout=0.2"
    script = (
        f"socat {listen} UNIX-CONNECT:{shlex.quote(egress.socket)} & "
        f"i=0; s=0.001; until socat -u OPEN:/dev/null {probe} 2>/dev/null; do "
        'i=$((i+1)); if [ "$i" -ge 40 ]; then s=0.01; fi; '
        'if [ "$i" -ge 230 ]; then '
        'echo "ph sandbox: the egress shim did not come up; no network" >&2; break; fi; '
        "sleep $s; done; "
        'exec "$@"'
    )
    return ("/bin/sh", "-c", script, "ph-egress", *argv)


@dataclass(frozen=True, slots=True)
class Bubblewrap:
    """`bwrap` (Linux): a mount namespace where only named paths are writable."""

    enforcement: Enforcement = "full"
    backend: str = "bwrap"

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

        **Network is a namespace, then a door.** `--unshare-net` whenever the
        policy does not say `network=True` — `None` included, which is the closed
        reading a backend owes a policy that reached it without the seam. When
        the effective policy carries an `Egress`, the proxy variables are set
        inside and the command is wrapped in `egress_shim`; the namespace is still
        unshared, so what the shim does not carry is unreachable.

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
            if policy.egress is not None:
                url = proxy_url(policy.egress)
                for name in PROXY_VARIABLES:
                    parts += ["--setenv", name, url]
                for name in BYPASS_VARIABLES:
                    parts += ["--unsetenv", name]
                argv = egress_shim(argv, policy.egress)
        return ConfinedArgv(
            argv=(*parts, "--", *argv),
            enforcement=self.enforcement,
            backend=self.backend,
            # Measured: `SIGINT` to this argv kills `bwrap` itself (`rc=-2`), and
            # `--unshare-pid` means the namespace goes with it. A caller meaning to
            # interrupt would be demolishing instead.
            forwards_signals=False,
        )

    def read_denial(self, output: str, *, network: bool) -> Denial | None:
        """The first refusal this kernel's words show, or `None` — the `DenialReader`
        half of the backend (see `DENIAL_SIGNATURES`).

        `network=True` — the command had the host's network — skips the network
        signatures: an outage is not the sandbox's doing, and saying so would be the
        misattribution this record must not make.

        `find` over the whole buffer rather than a loop over `splitlines()`, because
        this runs on every confined command and the seam keeps up to 8 MiB *per
        stream*: at that size the line loop measured **51 ms** and ~27 MB of
        transient `str` objects to almost always find nothing, against **13 ms** and
        no allocation for five C-level scans. The earliest match wins, as the line
        loop's did, which is why every signature is scanned rather than the first
        hit returned.
        """
        found: tuple[int, DenialKind] | None = None
        for kind, mark in DENIAL_SIGNATURES:
            if kind == "network" and network:
                continue
            at = output.find(mark)
            if at >= 0 and (found is None or at < found[0]):
                found = (at, kind)
        if found is None:
            return None
        at, kind = found
        start = output.rfind("\n", 0, at) + 1
        end = output.find("\n", at)
        line = output[start:] if end < 0 else output[start:end]
        return Denial(kind=kind, via="output", evidence=line.strip()[:200])


# --------------------------------------------------------------- seatbelt --


def seatbelt_profile(policy: SandboxPolicy) -> str:
    """The Seatbelt profile for one policy, as `sandbox-exec -p` takes it.

    **Deny by default and allow back**, which is the only direction that fails
    closed: a profile that allows by default and denies a list is one where every
    rule anybody forgot is permitted.

    Reads stay open. A confinement tier is about what an agent can *change* —
    pH's read boundary is `ctx.fs`'s, one layer up, and a Seatbelt profile that
    also denied reads would refuse the toolchain its own libraries and be
    switched off by the first person who met it.

    The egress lines are **unverified**: no macOS host has run them. They allow
    exactly the four things the bridge needs — the proxy's socket outbound, and
    the shim's loopback port bound, accepted and dialled — and nothing else, so a
    line that is wrong fails the probe rather than opening anything.

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
    elif policy.egress is not None:
        egress = policy.egress
        lines += [
            f'(allow network-outbound (remote unix-socket (path-literal "{egress.socket}")))',
            f'(allow network-bind (local ip "localhost:{egress.port}"))',
            f'(allow network-inbound (local ip "localhost:{egress.port}"))',
            f'(allow network-outbound (remote ip "localhost:{egress.port}"))',
        ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Seatbelt:
    """`sandbox-exec` (macOS): Apple's Seatbelt, driven by a generated profile.

    Deliberately **not** a `DenialReader`: what Seatbelt prints when it refuses a
    write has not been measured on a real macOS, and guessing the sentence would
    make the seam record refusals that did not happen and miss the ones that did.
    Recording nothing is the honest answer until somebody measures it.
    """

    enforcement: Enforcement = "full"
    backend: str = "sandbox-exec"

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv:
        """Wrap `argv` in `sandbox-exec -p <profile>`.

        Every mode goes through `seatbelt_profile`, `danger-full-access`
        included — it returns a permissive profile rather than this returning an
        unwrapped argv, for `Bubblewrap`'s reason: the caller must not be able to
        mistake absence for confinement.

        `sandbox-exec` sets no environment, so the proxy variables ride an `env`
        prefix inside the profile, ahead of the shim.
        """
        wrapped = argv
        if policy.egress is not None and not policy.network:
            url = proxy_url(policy.egress)
            assignments = [f"{name}={url}" for name in PROXY_VARIABLES]
            unset = [flag for name in BYPASS_VARIABLES for flag in ("-u", name)]
            wrapped = ("/usr/bin/env", *unset, *assignments, *egress_shim(argv, policy.egress))
        return ConfinedArgv(
            argv=(self.backend, "-p", seatbelt_profile(policy), *wrapped),
            enforcement=self.enforcement,
            backend=self.backend,
            # `sandbox-exec` execs its target and unshares nothing, so a signal
            # lands on the command. Unverified like the rest of this backend, and
            # the default anyway — stated because its sibling above states the
            # opposite and silence would read as "nobody considered it".
            forwards_signals=True,
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


# ------------------------------------------------------------------ probes --


@dataclass(frozen=True, slots=True)
class SandboxProbe:
    """What one escape attempt learned about this host."""

    confines: bool
    because: str


async def probe_sandbox(ctx: Context, backend: LocalBackend, scratch: Path) -> SandboxProbe:
    """Try to escape the sandbox, and claim the tier only if the kernel stopped it.

    **Two checks, and the second is what stops a useless backend passing.** A profile
    that denies everything refuses the escape and would look confining while being
    unusable — the first command an agent ran would fail and somebody would switch
    the tier off. So the probe writes *inside* the workspace, which must succeed, and
    *outside* it by absolute path, which must not.

    One spawn rather than two, because `sh` keeps going after a failed redirect, so
    both writes are attempted and both outcomes readable. The checks stay independent
    — "refused inside" and "escaped" remain distinguishable — while the probe stops
    paying twice for namespace setup on the serial startup path.

    **Never the exit code.** What the probe reads is the files: a backend can exit 0
    having written straight through to the host, and can print a whole session banner
    when the sandbox failed to start and the command never ran.

    The backend's **own words** are the decline, when it has any. "the sandbox refused
    a write inside the workspace" is true and useless; `setting up uid map: Permission
    denied` is what tells an operator to add an AppArmor profile.

    Once, at mount, because it is a property of the host — and deliberately **not**
    cached across processes: `kernel.apparmor_restrict_unprivileged_userns` is a live
    sysctl, so a remembered "yes" is exactly the fail-open this row exists to prevent.
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
        probe = await ctx.subprocess.run(
            SubprocessSpawnSpec(argv=confined.argv, cwd=work, env=ctx.subprocess.env())
        )
        err = probe.stderr
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


async def probe_egress(
    ctx: Context, backend: LocalBackend, bridge: Egress, scratch: Path
) -> SandboxProbe:
    """Send one `CONNECT` through the bridge from inside, and expect the proxy's refusal.

    The whole path in one spawn: the shim has to come up, reach the socket through
    the read-only bind, and the proxy has to answer. `PROBE_HOST` is what it asks
    for — the host the proxy refuses whatever the allowances say — so a `403` is
    the only pass, and it is the proxy's own. A `502`, a connection refused or
    silence is the bridge not working, quoted.

    `socat` on the *host* side too, because the same binary is what runs inside:
    the read-only bind of `/` is the host's filesystem, so a missing binary here is
    a missing binary there.
    """
    if shutil.which("socat") is None:
        return SandboxProbe(
            False, "socat is not installed; it is the shim between the sandbox and the proxy"
        )
    work = scratch / "egress-probe"
    await anyio.to_thread.run_sync(lambda: work.mkdir(parents=True, exist_ok=True))
    request = f"CONNECT {PROBE_HOST}:443 HTTP/1.1\\r\\nHost: {PROBE_HOST}\\r\\n\\r\\n"
    script = f"printf '{request}' | socat -T2 - TCP:127.0.0.1:{bridge.port},connect-timeout=2"
    # Straight to the backend, so `network` is the resolved field at its closed
    # default and the door is the bridge being proved.
    policy = SandboxPolicy(mode="workspace-write", workspace_root=str(work), egress=bridge)
    try:
        confined = backend.confine(("/bin/sh", "-c", script), policy)
        probe = await ctx.subprocess.run(
            SubprocessSpawnSpec(
                argv=confined.argv, cwd=work, env=ctx.subprocess.env(), timeout_ms=10_000
            )
        )
        answer = first_line(probe.stdout)
        if answer.startswith("HTTP/1.1 403"):
            return SandboxProbe(True, f"reachable from inside; the proxy refused {PROBE_HOST}")
        because = first_line(probe.stderr) or answer or "the bridge did not answer"
        return SandboxProbe(False, because)
    except Exception as error:  # pragma: no cover - a host that fails in a new way
        return SandboxProbe(False, f"{type(error).__name__}: {error}")
    finally:
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(work, ignore_errors=True))


_SOCKETS = count(1)


async def egress_socket_path(ctx: Context) -> Path:
    """Where this mount's proxy listens: `$PH_RUNTIME/sandbox/egress-<pid>-<n>.sock`.

    Under `$PH_RUNTIME` for the reason the daemon socket is (`ph.paths`): per boot,
    per user, `0700`, and never carried anywhere by a sync. Per process and per
    mount, because one daemon mounts one profile once per root and each mount is
    its own proxy.

    **Falls back to a private temp directory when the path would not fit** —
    `sun_path` is 108 bytes and a test's `$PH_RUNTIME` under pytest's numbered tree
    can exceed it. Through `temporary_directory`, so the fallback is `0700` and
    unwinds with the row's scope rather than with garbage collection (§4.9).
    """
    roots = resolve_roots(create=True)
    directory = roots.runtime / "sandbox"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"egress-{os.getpid()}-{next(_SOCKETS)}.sock"
    if len(str(path).encode()) > SOCKET_PATH_MAX:
        path = await temporary_directory(ctx, prefix="ph-egress-") / "egress.sock"
    return path


# --------------------------------------------------------------------- row --


_BRIDGE_VERDICT: SandboxProbe | None = None
"""What `probe_egress` learned, remembered for the life of the process.

**Per process, never persisted, and deliberately unlike the filesystem probe.**
That one refuses to be cached at all because
`kernel.apparmor_restrict_unprivileged_userns` is a live sysctl and a remembered
"yes" would be a fail-open. Nothing this probe establishes can flip that way: it
shows `socat` is installed, that a confined shell reaches a unix socket under
`$PH_RUNTIME` through the read-only bind, and that the proxy answers. And if the
sysctl *did* flip, `probe_sandbox` — still uncached, still run at every mount —
declines first and this is never reached.

It matters because a daemon mounts the profile **once per root**, and the probe
costs 19.4 ms of serial startup each time: eight roots paid ~155 ms to re-learn
one fact about the host.
"""


@dataclass(slots=True)
class _Egress:
    """What this row did about network egress, for the diagnostic to read live."""

    proxy: EgressProxy | None = None
    verdict: SandboxProbe | None = None

    def rows(self) -> list[tuple[str, str]]:
        if self.verdict is None:
            return [("egress", "not attempted — no confining backend")]
        if self.proxy is None:
            return [("egress", f"declined — {self.verdict.because}")]
        return [
            ("egress", f"proxy at {self.proxy.path}; {self.verdict.because}"),
            ("egress refused", str(self.proxy.denied)),
        ]

    async def mount(self, ctx: Context, backend: LocalBackend, scratch: Path) -> None:
        """Start the proxy, prove the bridge, and only then register it.

        The proxy runs before the probe because the probe needs something to
        answer; `on_denied` is attached *after* the probe because the probe's own
        refusal is not anybody's. A bridge that fails its probe is closed again
        here rather than left listening for nothing.
        """
        path = await egress_socket_path(ctx)
        proxy = EgressProxy(path=path, permits=ctx.sandbox.permits)
        try:
            await proxy.start()
        except OSError as error:
            # A host that will not let this process listen — a sandbox of its own
            # refusing unix sockets, a runtime dir it cannot write — is a host
            # with no bridge, not a row that fails to mount.
            self.verdict = SandboxProbe(False, f"could not listen on {path}: {error}")
            log.info("ph.seams.sandbox_local: no egress bridge — %s", self.verdict.because)
            return
        ctx.add_disposer(proxy.aclose, label="sandbox-local(egress proxy)")
        global _BRIDGE_VERDICT
        bridge = Egress(socket=str(path), port=EGRESS_PORT)
        if _BRIDGE_VERDICT is None:
            _BRIDGE_VERDICT = await probe_egress(ctx, backend, bridge, scratch)
        self.verdict = _BRIDGE_VERDICT
        if not self.verdict.confines:
            log.info("ph.seams.sandbox_local: no egress bridge — %s", self.verdict.because)
            await proxy.aclose()
            return

        def denied(host: str, port: int, agent: str | None) -> None:
            ctx.sandbox.record_denial(
                Denial(kind="network", via="proxy", host=host, port=port), agent=agent
            )

        proxy.on_denied = denied
        ctx.sandbox.register_egress(bridge)
        self.proxy = proxy


class Config(WireModel):
    """Row config for the local confinement backend."""

    root: str | None = None
    """Where the probes do their work. `$PH_HOME/sandbox` by default."""


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

    The egress bridge is gated the same way, after the backend is in: a proxy
    nobody inside can reach is not a door, and registering it would make
    `allowlist` promise a network the command does not have.

    The result reaches `ph doctor` either way: "why is strict refusing to start"
    is a question asked of the tool, and a row that declined in silence is
    indistinguishable from one nobody mounted.
    """
    backend, why = local_backend()
    scratch = default_home_path(config.root, "sandbox")
    result = (
        SandboxProbe(False, why) if backend is None else await probe_sandbox(ctx, backend, scratch)
    )
    egress = _Egress()

    contribute(
        ctx,
        Diagnostic(
            id="sandbox-local",
            title="Local confinement",
            read=lambda: [
                ("backend", why),
                ("confines", "yes" if result.confines else "no"),
                ("because", result.because),
                *egress.rows(),
            ],
            order=15,
        ),
    )
    if backend is None or not result.confines:
        log.info("ph.seams.sandbox_local: declining — %s", result.because)
        return
    ctx.sandbox.register_provider(backend)
    await egress.mount(ctx, backend, scratch)
