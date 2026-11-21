"""`sandbox-local`: the first real confinement backend (P6-04).

The backend is `bwrap` on Linux and `sandbox-exec` on macOS. Both do the same
job in the same shape. They take an argv and return an argv that the **kernel**
bounds. That is the difference between the `sandbox` rung and every rung below
it. A worktree bounds the writes that resolve against cwd, and an overlay bounds
the same set into a delta. Only this rung refuses `open("/etc/passwd", "w")`.

**`confine` only builds an argv, and that is deliberate.** It builds a command
line and runs nothing. So a test can assert every rule that this module encodes
without a kernel that enforces it. The argv *is* the policy, so a test that
reads the argv tests the policy. That also let the Seatbelt half be reviewed
before a Mac ran it. The same holds for the network door. When the effective
policy carries an `Egress`, the argv gains the proxy variables. Under `bwrap` it
also gains a `socat` shim wrapped around the command. Still nothing here runs.

**The probe decides registration, and it is the only reason that a backend
written without a host to run it on is safe.** A confinement backend that
silently fails open is the worst object in this codebase. Every caller believes
that the kernel holds the boundary, and nothing says otherwise. So registration
depends on a canary. The canary writes to an absolute path *outside* the
workspace, and the probe checks that the host copy is unchanged. A backend
whose rules are wrong in the permissive direction fails that check and
declines. It does not claim a tier that it does not occupy.

**The egress bridge is probed the same way, and the row claims it only when it
works.** The row starts the proxy. A confined command then sends one `CONNECT`
to the host that the proxy always refuses. The row registers the bridge on the
seam only if the 403 of the proxy itself comes back. Each backend owns its
**door** (`needs_loopback`, `egress_blocker` and `egress_probe`):

- `bwrap` unshares the network. Its door is a `socat` shim on the loopback of
  the sandbox itself, which forwards to the unix socket of the proxy.
- Seatbelt denies and does not unshare, so the confined command shares the
  loopback of the host. Its door is the `127.0.0.1:<port>` of the proxy itself,
  which is the one remote that the profile allows.

Without `socat` on a `bwrap` host, or with a bridge that does not answer, the
row registers no bridge. `allowlist` then means no network, and
`SandboxSeam.network_posture` says so explicitly.

**Both backends are verified against a real kernel.**

`bwrap`: the prerequisite on Ubuntu 23.10+ is an AppArmor profile at
`/etc/apparmor.d/bwrap`. `kernel.apparmor_restrict_unprivileged_userns=1` is the
default there, and an unprofiled `bwrap` is not setuid. So without a profile it
dies with `setting up uid map: Permission denied`.

Seatbelt: measured on macOS 26.6 on 2026-09-07. The profile that was written
before a Mac ran it was wrong in three places that the probe would have caught,
and in one place that it would not:

1. Seatbelt matches the path that the kernel *resolves*, and `/var` and `/tmp`
   are symlinks into `/private`. So a workspace under `$TMPDIR` was refused its
   own writes, and the probe declined the tier. Every root that pH creates is
   now canonical at its source (`ph.paths.canonical`). That is the only place
   where a spelling can be fixed without the prompt boundary and the enforced
   boundary drifting apart (E6).
2. The unix-socket door had the same defect.
3. A `socat` shim under Seatbelt is not in a PID namespace. So it outlived
   every command, and the bind of the next command found the port taken. That
   is the reason for the loopback door, which needs no shim and no `socat`.
4. What the kernel prints: `Operation not permitted`, with the path on the line
   for a file and without a path for a socket. That was measured across `sh`,
   `mkdir`, `touch`, `rm`, `git`, Python and `socat`. `Seatbelt` is now a
   `DenialReader`, which it deliberately was not before the measurement.

Also measured:

- `SIGINT` to the wrapper lands on the command, with and without the `env`
  prefix.
- fd 3 crosses the wrapper.
- `(remote ip "localhost:<port>")` is the spelling that Seatbelt accepts. A
  literal `127.0.0.1:<port>` is a profile syntax error.

@module ph.seams.sandbox_local
"""

from __future__ import annotations

import logging
import math
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import TypeAlias
from urllib.parse import quote

import anyio

from ..cordis import Context, maybe_await, plugin, releasing
from ..keys import SANDBOX, SUBPROCESS
from ..paths import default_home_path, resolve_roots
from ..resources import temporary_directory
from ..text import around
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
from .subprocess import first_line

__all__ = [
    "BYPASS_VARIABLES",
    "DENIAL_SIGNATURES",
    "EGRESS_PORT",
    "PROXY_PASSWORD",
    "PROXY_VARIABLES",
    "SEATBELT_SIGNATURES",
    "Bubblewrap",
    "LocalBackend",
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
"""The port where the shim listens inside a sandbox.

Every sandbox uses the same number. Each sandbox has its own network namespace,
and a port in one namespace is invisible from another. So the port can collide
only with a listener that the confined command itself opens. The number is the
port of Squid, so a person who sees `HTTPS_PROXY=http://127.0.0.1:3128` inside
knows what kind of thing listens there."""

PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
"""Both cases, because the tools disagree. `curl` reads the lower-case names and
refuses upper-case `HTTP_PROXY` on purpose (CGI hijacking). Go and most Python
clients read either case."""

BYPASS_VARIABLES = ("NO_PROXY", "no_proxy")
"""Unset inside the sandbox, whatever the host had. A host listed in one of
these is a host that the client would dial directly, and inside the namespace
there is nothing to dial."""

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
"""What the silent refusals of `bwrap` look like from inside a command.

The table is here and not on the seam, because every entry is a fact about
**the platform of this backend**: Linux, glibc and `bwrap`. Seatbelt refuses in
other words (`SEATBELT_SIGNATURES`). A table of these sentences in the seam
definition would assert the wording of one kernel on behalf of every backend.
Each backend is its own `DenialReader`.

The entries were measured and not guessed. A redirect onto the read-only tree
prints the first sentence from `sh`. A raw `connect()` inside an unshared
namespace raises `[Errno 101] Network is unreachable`. The three resolver texts
are what `curl`, glibc and Python print when a command bypasses the proxy
variables and asks for DNS that a namespace does not have. The entries are
substrings and not patterns, and each is one that no ordinary command prints on
success. That is also the reason why `ShellService` asks only about a command
that failed.
"""


# The door is the way that a confined command reaches the egress proxy. It is
# **split between a declaration and a behavior**, because the row needs the two
# at different times.
#
# `needs_loopback` has the shape of `enforcement`, for the reason that
# `enforcement` states. The row must know *before* it has confined anything
# whether to open a TCP listener. A caller cannot act beforehand on a property
# that it can discover only by running something. `egress_blocker` and
# `egress_probe` have the shape of `DenialReader`: behavior that the backend
# owns, because what proves that a door works is a fact about that door. An
# earlier design had one `Literal["shim", "loopback"]` switch that the row read.
# The switch conflated the two. To add Landlock, an author had to edit a
# `Literal` and every `if` that matched on it, and nothing checked that the
# author had found them all.


def _earliest(
    output: str, signatures: tuple[tuple[DenialKind, str], ...], *, network: bool
) -> tuple[int, DenialKind, str] | None:
    """The first refusal that any signature shows in `output`, as `(offset, kind,
    signature)`. The signature is returned so that the evidence can be centered
    on all of it.

    This uses `find` over the whole buffer and not a loop over `splitlines()`.
    It runs on every failed confined command, and the seam keeps up to 8 MiB
    *per stream* by default. At that size the line loop measured **51 ms** and
    ~27 MB of transient `str` objects, almost always to find nothing. A handful
    of C-level scans measured **13 ms** and no allocation. The earliest match
    wins, as it did in the line loop. That is why every signature is scanned,
    and the first hit is not returned at once.

    `network=True` means that the command had the network of the host. It skips
    the network signatures. An outage is not the work of the sandbox, and a
    record that said so would be the misattribution that this record must not
    make.
    """
    found: tuple[int, DenialKind, str] | None = None
    for kind, mark in signatures:
        if kind == "network" and network:
            continue
        at = output.find(mark)
        if at >= 0 and (found is None or at < found[0]):
            found = (at, kind, mark)
    return found


EVIDENCE_CHARS = 200
"""How much of a refusal's line a denial record carries."""


def _line_at(output: str, at: int, mark: str) -> tuple[str, str]:
    """The line that holds the signature `mark` at offset `at`, stripped, and
    the evidence that a record carries.

    The evidence is as much of the line as fits around the signature
    (`ph.text.around`). The whole line is returned for a caller that reads it.
    The evidence is a window and not the head of the line, because the first
    `EVIDENCE_CHARS` characters of a long line cut off a signature that is
    further along."""
    start = output.rfind("\n", 0, at) + 1
    end = output.find("\n", at)
    raw = output[start:] if end < 0 else output[start:end]
    line = raw.strip()
    found = max(at - start - (len(raw) - len(raw.lstrip())), 0)
    return line, around(line, found, EVIDENCE_CHARS, stop=found + len(mark))


# ------------------------------------------------------------------ bwrap --


PROXY_PASSWORD = "ph"
"""The password beside the agent in the proxy URL. It is a constant and not a
secret.

It is present because `urllib` sends `Proxy-Authorization` only when *both*
parts of the user info are non-empty (`if user and password:`). A URL with only
the agent reached the proxy with no identity, and the refusal was recorded
against nobody. That was measured in the end-to-end test. The proxy reads the
user and ignores the password."""


def proxy_url(egress: Egress) -> str:
    """`http://<agent>:ph@127.0.0.1:<port>`, with the agent in the user part.

    Every client that honors the URL sends the user back as Basic credentials
    on each request. That is how the proxy knows which session a refusal
    belongs in, without a side channel. The agent is percent-encoded, because
    an agent id is a session id, and a session id is whatever the store made it.
    """
    user = f"{quote(egress.agent, safe='')}:{PROXY_PASSWORD}@" if egress.agent else ""
    return f"http://{user}127.0.0.1:{egress.port}"


def egress_shim(argv: tuple[str, ...], egress: Egress) -> tuple[str, ...]:
    """Wrap `argv` so that a `socat` shim listens on loopback before it runs.

    `sh -c` starts the shim in the background, waits until the port accepts a
    connection, and then `exec`s the command. So the exit status and the
    signals of the command are its own. The shim is a sibling that dies when
    the PID namespace does (`--unshare-pid`, measured: nothing survives the
    command).

    **This is the door of `bwrap` only.** Without a PID namespace the sibling
    *outlives* the command. That was measured under Seatbelt: a shim from one
    test run still held `EGRESS_PORT` an hour later, and the bind of the next
    command found the port taken. That is why the Seatbelt backend has a
    different door and never calls this function.

    The wait is a bounded poll and not a fixed sleep. The shim is up in a few
    milliseconds, and a command that reached for the network in its first
    millisecond would otherwise race it. A shim that never comes up is reported
    on stderr, and the command runs anyway. It then has no network, which is
    what it had before.

    **The interval has two phases.** The listener binds in 1-3 ms, so a flat
    10 ms sleep costs a whole tick on a probe that was always going to miss one
    time. Measured here, as the median of 21 runs of `/bin/true`:

    - `bwrap` alone: **6.1 ms**
    - with a flat 10 ms poll: **12.8 ms**
    - with 1 ms for the first 40 tries and 10 ms after: **10.8 ms**

    So the two phases give **2 ms back on every confined command**, and keep
    the same ~2 s budget for a host where something is actually wrong. (A
    review pass measured against the real proxy and not a bare accepting
    socket. It saw a larger gap, 25.5 ms against 15.8. The direction is the
    same, and the smaller number is the one that this docstring claims.) The
    spawns are not the cost: a loop that used the native `retry=` of `socat`
    measured within a millisecond of this one. The interval is the cost.
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
    needs_loopback: bool = False
    """The namespace belongs to the sandbox alone, so the proxy needs no
    listener on the loopback of the host. Nothing inside could reach it. See
    `egress_probe`."""

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv:
        """Wrap `argv` in a namespace that binds the filesystem read-only.

        `--ro-bind / /` comes first and the writable binds come after. `bwrap`
        applies them in order, and the last bind that covers a path wins. So
        the whole tree is read-only, and the declared roots are bound writable
        again on top of it.

        `--dev` and `--proc` are used, and not binds of the `/dev` and `/proc`
        of the host, **with `--unshare-pid`**. A fresh `/proc` mount alone
        still reports the PID namespace that it is in, so the confined process
        could see and signal every process on the host. Filesystem confinement
        with the process table open is not what this rung claims.

        **Network is a namespace, then a door.** `--unshare-net` is added
        whenever the policy does not say `network=True`. `False` is the default
        of the field, which is the closed reading that a backend owes a policy
        that reached it without the seam. When the effective policy carries an
        `Egress`, the proxy variables are set inside and the command is wrapped
        in `egress_shim`. The namespace is still unshared, so the command
        cannot reach what the shim does not carry.

        `danger-full-access` is still wrapped, with `/` bound writable. The
        mode means "no confinement". The correct way to state that is a sandbox
        that permits everything, and not an unwrapped argv. The contract of the
        seam is that a caller must never be able to mistake absence for
        confinement, and an argv that comes back unchanged is exactly that
        mistake.
        """
        parts = ["bwrap", "--die-with-parent"]
        if policy.mode == "danger-full-access":
            parts += ["--bind", "/", "/"]
        else:
            parts += ["--ro-bind", "/", "/"]
            for path in writable_paths(policy):
                parts += ["--bind", path, path]
        # `--unshare-pid` is what makes the `/proc` statement above true.
        # Measured: with `--proc /proc` alone, the confined process saw **628**
        # PIDs against 626 on the host. That is the whole process table, and
        # the process could signal all of it, because a fresh `/proc` mount
        # still reports the namespace that it is in. With the flag it saw 4.
        # The docstring claimed process isolation that the flags did not
        # deliver, which is the E1 shape one level down.
        parts += ["--dev", "/dev", "--proc", "/proc", "--unshare-pid"]
        # **A session of its own** (J9). This is the terminal half of the claim
        # that `--unshare-pid` makes about the process table. Without the flag,
        # the confined command shares the controlling terminal of the host. It
        # can then push characters into that terminal with `TIOCSTI`, and the
        # shell of the *user* executes that input unconfined. A `^C` in that
        # terminal also signals the whole foreground group and not the sandbox.
        # `bwrap` offers the flag for exactly this case. `ph.seams.subprocess`
        # does the same with `start_new_session` for the children that it
        # spawns directly.
        parts.append("--new-session")
        if not policy.network:
            # The namespace is the enforcement. An unshared net namespace has
            # no interface but loopback, so this is not a filter that a command
            # can bypass.
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
            # Measured: `SIGINT` to this argv kills `bwrap` itself (`rc=-2`),
            # and `--unshare-pid` means that the namespace goes with it. A
            # caller that meant to interrupt would end everything instead.
            forwards_signals=False,
        )

    def egress_blocker(self) -> str | None:
        """Why this host cannot bridge at all, or `None`.

        The question is whether `socat` is on the *host*, because the same
        binary runs inside. The read-only bind of `/` is the filesystem of the
        host, so a binary that is missing on the host is missing inside.
        """
        if shutil.which("socat") is None:
            return "socat is not installed; it is the shim between the sandbox and the proxy"
        return None

    def egress_probe(self, bridge: Egress) -> tuple[str, ...]:
        """One `CONNECT` through the shim, as a command to confine and run.

        The shim must come up and reach the socket through the read-only bind
        before the proxy can answer. So this proves the whole path in one
        spawn. `confine` wraps it in `egress_shim` exactly as it wraps any
        other command.
        """
        request = f"CONNECT {PROBE_HOST}:443 HTTP/1.1\\r\\nHost: {PROBE_HOST}\\r\\n\\r\\n"
        script = f"printf '{request}' | socat -T2 - TCP:127.0.0.1:{bridge.port},connect-timeout=2"
        return ("/bin/sh", "-c", script)

    def read_denial(self, output: str, *, network: bool) -> Denial | None:
        """The first refusal that the words of this kernel show, or `None`.
        This is the `DenialReader` half of the backend (see
        `DENIAL_SIGNATURES`).

        The scan is `_earliest`, shared with `Seatbelt`. The two backends
        differ in the table, and not in the way that it is read.
        """
        found = _earliest(output, DENIAL_SIGNATURES, network=network)
        if found is None:
            return None
        at, kind, mark = found
        _line, evidence = _line_at(output, at, mark)
        return Denial(kind=kind, via="output", evidence=evidence)


# --------------------------------------------------------------- seatbelt --


SEATBELT_SIGNATURES: tuple[tuple[DenialKind, str], ...] = (
    ("filesystem", "Operation not permitted"),
    ("network", "nodename nor servname provided, or not known"),
    ("network", "Could not resolve host"),
)
"""What the refusals of Seatbelt look like from inside a command. They were
measured on macOS 26.6 on 2026-09-07, and they are the reason that `Seatbelt`
is now a `DenialReader`.

The kernel answers `EPERM`, and every tool spells it `Operation not permitted`:

- `sh: /tmp/x: Operation not permitted`
- `mkdir: /tmp/d: Operation not permitted`
- `rm: /etc/hosts: Operation not permitted`
- `git: error: couldn't create cache file '/tmp/xcrun_db-…' (errno=Operation not permitted)`
- Python: `PermissionError: [Errno 1] Operation not permitted: '/tmp/x'`

A refused `connect()` gets the same `EPERM`: `PermissionError: [Errno 1]
Operation not permitted` from Python, and `E connect(… AF=2 127.0.0.1:9 …):
Operation not permitted` from `socat`. So the sentence alone does not say which
boundary refused. `Seatbelt.read_denial` reads the line for a path. The two
resolver texts are what Python (`gaierror`) and `curl` print when a command
bypasses the proxy variables and asks for DNS that the profile denies. `curl`
to a raw address prints its generic `Couldn't connect to server`. It prints the
same text for a server that is down, so that text is deliberately absent.
"""


def seatbelt_profile(policy: SandboxPolicy) -> str:
    """The Seatbelt profile for one policy, in the form that `sandbox-exec -p`
    takes.

    **Deny by default and allow back.** That is the only direction that fails
    closed. A profile that allows by default and denies a list permits every
    rule that somebody forgot.

    Reads stay open. A confinement tier is about what an agent can *change*.
    The read boundary of pH belongs to `ctx.fs`, one layer up. A Seatbelt
    profile that also denied reads would refuse the toolchain its own
    libraries, and the first person who met it would switch it off.

    Writable roots are emitted **as given**, and the boundary depends on that.
    Seatbelt matches `subpath` against the path that the kernel resolves, and
    on macOS `/var` is a symlink into `/private`. A root spelled
    `/var/folders/…` was refused its own writes (measured). The spelling is
    fixed at the source instead. Every root that the workspace seam creates and
    every directory that `sandbox-allow` admits is canonical
    (`ph.paths.canonical`). So this builder, the binds of `bwrap` and the
    prompt boundary of `permissions-fs` all read one string. A resolution here
    would be a second spelling of the set that `writable_roots` alone defines,
    and a small case of the drift in E6.

    The door is **one remote and nothing else**:
    `(remote ip "localhost:<port>")`, the loopback listener of the proxy
    itself. Measured:

    - The allowed port answers.
    - The port beside it gets `Operation not permitted`.
    - DNS is refused.
    - A raw `curl` to an address cannot connect.
    - A `curl` that honors `HTTPS_PROXY` gets the 403 of the proxy itself.

    `localhost` is the spelling that Seatbelt takes. A literal
    `127.0.0.1:<port>` is a syntax error. `localhost` also admits `::1`, where
    nothing listens.

    This is a separate function because it is the part worth reading in a test.
    The argv around it is three tokens, and the profile is the policy.
    """
    if policy.mode == "danger-full-access":
        # This case is here and not in `confine`. This function is documented
        # as *the* profile for a policy, and the profile of one policy was
        # built outside it. That made two places to look for one answer.
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
        lines.append(f"(allow file-write* (subpath {_sexp(path)}))")
    if policy.network:
        # Only the permitting arm is emitted. `(deny network*)` is a no-op
        # under `(deny default)` above. A line that changes nothing is a line
        # that a test can assert while it proves nothing, and that happened.
        lines.append("(allow network*)")
    elif policy.egress is not None:
        lines.append(f'(allow network-outbound (remote ip "localhost:{policy.egress.port}"))')
    return "\n".join(lines)


def _sexp(value: str) -> str:
    """One path as a Seatbelt string literal (J9).

    The profile is a TinyScheme s-expression, and the paths went into it raw.
    So a `"` or a `\\` in a directory name ended the literal early. The rest of
    the path became stray atoms, and `sandbox-exec` rejected the profile, which
    took the rung down. A profile that still parsed was worse: it granted
    `file-write*` on a *prefix* of what was asked for.

    Both characters are legal in a macOS path, and `$PH_HOME` is a path that
    the user chooses. The escaping is the pair that Scheme uses, which is what
    the reader on the other side expects.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass(frozen=True, slots=True)
class Seatbelt:
    """`sandbox-exec` (macOS): Apple's Seatbelt, driven by a generated profile.

    Verified against a real kernel on 2026-09-07 (macOS 26.6):

    - A write inside the workspace lands.
    - An absolute-path write outside it is refused, and the host file is
      untouched.
    - `read-only` refuses the workspace too.
    - fd 3 crosses the wrapper.
    - `SIGINT` sent to the wrapper lands on the command.

    It has been a `DenialReader` since that measurement. See
    `SEATBELT_SIGNATURES` for the words, and `read_denial` for the one decision
    that the words alone do not make.
    """

    enforcement: Enforcement = "full"
    backend: str = "sandbox-exec"
    needs_loopback: bool = True
    """Seatbelt denies and does not unshare, so a confined command shares the
    loopback of the host and the proxy can listen there. That is the whole
    door, and it is why there is no shim to leak. See `egress_probe`."""

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv:
        """Wrap `argv` in `sandbox-exec -p <profile>`.

        Every mode goes through `seatbelt_profile`, `danger-full-access`
        included. That function returns a permissive profile, and this method
        does not return an unwrapped argv. The reason is the one that
        `Bubblewrap` gives: the caller must not be able to mistake absence for
        confinement.

        `sandbox-exec` sets no environment, so an `env` prefix carries the
        proxy variables. `env` execs the command, so a signal still lands on
        the command (measured). **No shim**: the door is the loopback port of
        the proxy itself, which the profile allows and the command reaches
        directly. See `needs_loopback`.
        """
        wrapped = argv
        if policy.egress is not None and not policy.network:
            url = proxy_url(policy.egress)
            assignments = [f"{name}={url}" for name in PROXY_VARIABLES]
            unset = [flag for name in BYPASS_VARIABLES for flag in ("-u", name)]
            wrapped = ("/usr/bin/env", *unset, *assignments, *argv)
        return ConfinedArgv(
            argv=(self.backend, "-p", seatbelt_profile(policy), *wrapped),
            enforcement=self.enforcement,
            backend=self.backend,
            # Measured: `sandbox-exec` execs its target and unshares nothing,
            # so `SIGINT` to the pid of the wrapper reaches the command. A
            # Python handler inside ran and exited 3, with and without the
            # `env` prefix. This is stated because `Bubblewrap` states the
            # opposite.
            forwards_signals=True,
        )

    def egress_blocker(self) -> str | None:
        """Nothing to install. The door is a TCP connect and this interpreter
        makes it, so no host binary can be missing."""
        return None

    def egress_probe(self, bridge: Egress) -> tuple[str, ...]:
        """One `CONNECT` to the loopback port of the proxy, as a command to
        confine.

        The probe uses this interpreter and not `socat`. A Mac has no `socat`
        unless somebody installed one, and the door needs nothing but a TCP
        connect. That also makes it a fair probe of what a confined command
        will do.
        """
        program = (
            "import socket, sys\n"
            f"s = socket.create_connection(('127.0.0.1', {bridge.port}), timeout=2)\n"
            f"s.sendall(b'CONNECT {PROBE_HOST}:443 HTTP/1.1\\r\\nHost: {PROBE_HOST}\\r\\n\\r\\n')\n"
            "sys.stdout.write(s.recv(4096).decode('latin-1'))\n"
        )
        return (sys.executable, "-c", program)

    def read_denial(self, output: str, *, network: bool) -> Denial | None:
        """The first refusal that the words of this kernel show, or `None`.

        **The sentence leaves one decision to this method.** Seatbelt answers
        a refused `open()` and a refused `connect()` with the same `EPERM`. So
        `Operation not permitted` is read with its line. A line that names a
        path is a filesystem refusal, because every tool measured puts the
        path there. A line with no path is a socket refusal when the command
        had no network. When the command *had* the network, an `EPERM` with no
        path belongs to neither boundary that this seam bounds (for example a
        `mach-lookup` or a `process-info` that the profile does not open). It
        is not recorded, because a record that names the wrong boundary is
        worse than none.
        """
        found = _earliest(output, SEATBELT_SIGNATURES, network=network)
        if found is None:
            return None
        at, kind, mark = found
        # The whole line decides the kind, and the window is what is shown. A
        # path past the edge of the window still makes this a filesystem
        # denial.
        line, evidence = _line_at(output, at, mark)
        if kind == "filesystem" and "/" not in line:
            if network:
                return None
            kind = "network"
        return Denial(kind=kind, via="output", evidence=evidence)


LocalBackend: TypeAlias = Bubblewrap | Seatbelt
"""The backends that this row can mount.

The union has a name because three public signatures use it (`local_backend`,
`probe_sandbox` and `probe_egress`), and the summary of P6-04 still owes a
third backend (Landlock), which would otherwise edit each of them. It is not
`SandboxProvider`: that Protocol does not declare `needs_loopback`,
`egress_blocker` or `egress_probe`, which the row and `probe_egress` read from
a backend.
"""


def local_backend() -> tuple[LocalBackend | None, str]:
    """The backend of this platform, or `None` and the reason that there is none.

    The platform is checked first and the binary second. So a Linux host
    without `bwrap` is told that `bwrap` is not installed, and is not told that
    its platform is unsupported.
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

    **There are two checks, and the write inside the workspace is the one that
    stops a useless backend from passing.** A profile that denies everything
    refuses the escape. It would look confining while it was unusable: the
    first command that an agent ran would fail, and somebody would switch the
    tier off. So the probe writes *inside* the workspace, which must succeed,
    and *outside* it by absolute path, which must not.

    One spawn does both writes. `sh` continues after a failed redirect, so both
    writes are attempted and both outcomes can be read. The checks stay
    independent: "refused inside" and "escaped" can still be told apart. And
    the probe does not pay twice for namespace setup on the serial startup
    path.

    **The probe never reads the exit code.** It reads the files. A backend can
    exit 0 after it wrote straight through to the host. A backend can also
    print a whole session banner when the sandbox failed to start and the
    command never ran.

    The decline uses the **words of the backend itself** when it has any. "the
    sandbox refused a write inside the workspace" is true and useless.
    `setting up uid map: Permission denied` is what tells an operator to add an
    AppArmor profile.

    The probe runs one time at each mount, because the result is a property of
    the host. It is deliberately **not** cached across processes.
    `kernel.apparmor_restrict_unprivileged_userns` is a live sysctl, so a
    remembered "yes" is exactly the fail-open that this row exists to prevent.
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
        # **Quoted** (J9). These paths are interpolated into a shell command,
        # and `scratch` comes from `$PH_HOME`. On a Mac that is often under
        # `~/Library/Application Support`. Unquoted, the redirect took the
        # first word, and the write of the probe landed somewhere else. So
        # `landed` was false and the rung *declined* on a host where the
        # sandbox works. A space in a directory name switched off a
        # confinement tier, and the report was "the sandbox refused a write
        # inside the workspace it was given".
        script = (
            f"echo landed > {shlex.quote(str(inside))}; echo escaped > {shlex.quote(str(outside))}"
        )
        confined = backend.confine(("/bin/sh", "-c", script), policy)
        probe = await ctx.require(SUBPROCESS).run(
            confined.spawn_spec(work, env=ctx.require(SUBPROCESS).env())
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
        # `releasing` is used because raw cancellation would otherwise skip
        # this `await` outright (L10). A mount abandoned in the middle of a
        # probe would leave the probe tree under `$PH_HOME` for the next mount
        # to find. The verdict of a probe that finds its own leftovers is about
        # the leftovers.
        with releasing():
            await anyio.to_thread.run_sync(lambda: shutil.rmtree(work, ignore_errors=True))


async def probe_egress(
    ctx: Context, backend: LocalBackend, bridge: Egress, scratch: Path
) -> SandboxProbe:
    """Send one `CONNECT` through the bridge from inside, and expect the proxy's refusal.

    One spawn proves the whole path, whatever the door is. The command must
    reach the proxy from inside the sandbox, and the proxy must answer. The
    command asks for `PROBE_HOST`, the host that the proxy refuses whatever the
    allowances say. So a `403` is the only pass, and it comes from the proxy
    itself. A `502`, a refused connection or silence means that the bridge does
    not work, and the decline quotes it.

    **The door belongs to the backend.** The backend supplies both the
    prerequisite that it needs on this host (`egress_blocker`) and the command
    that proves the door (`egress_probe`). This function knows only that
    something is spawned confined and that a 403 comes back. So a third
    backend adds a door without an edit here.
    """
    blocked = backend.egress_blocker()
    if blocked is not None:
        # This check runs before the directory is made, so a host that was
        # always going to decline does no work and leaves nothing behind. The
        # check was above the `mkdir` originally, and the branch on the door
        # pushed it below for a time.
        return SandboxProbe(False, blocked)
    work = scratch / "egress-probe"
    await anyio.to_thread.run_sync(lambda: work.mkdir(parents=True, exist_ok=True))
    # The policy goes straight to the backend and not through the seam. So
    # `network` is the resolved field at its closed default, and the door is
    # the bridge under test.
    policy = SandboxPolicy(mode="workspace-write", workspace_root=str(work), egress=bridge)
    try:
        confined = backend.confine(backend.egress_probe(bridge), policy)
        probe = await ctx.require(SUBPROCESS).run(
            confined.spawn_spec(work, env=ctx.require(SUBPROCESS).env(), timeout_ms=10_000)
        )
        answer = first_line(probe.stdout)
        if answer.startswith("HTTP/1.1 403"):
            return SandboxProbe(True, f"reachable from inside; the proxy refused {PROBE_HOST}")
        because = first_line(probe.stderr) or answer or "the bridge did not answer"
        return SandboxProbe(False, because)
    except Exception as error:  # pragma: no cover - a host that fails in a new way
        return SandboxProbe(False, f"{type(error).__name__}: {error}")
    finally:
        # `releasing` is used because raw cancellation would otherwise skip
        # this `await` outright (L10). A mount abandoned in the middle of a
        # probe would leave the probe tree under `$PH_HOME` for the next mount
        # to find. The verdict of a probe that finds its own leftovers is about
        # the leftovers.
        with releasing():
            await anyio.to_thread.run_sync(lambda: shutil.rmtree(work, ignore_errors=True))


_SOCKETS = count(1)


async def egress_socket_path(ctx: Context) -> Path:
    """Where the proxy of this mount listens: `$PH_RUNTIME/sandbox/egress-<pid>-<n>.sock`.

    The socket is under `$PH_RUNTIME` for the same reason as the daemon socket
    (`ph.paths`): the directory is per boot, per user and `0700`, and a sync
    never carries it anywhere. The name is per process and per mount, because
    one daemon mounts one profile one time for each root, and each mount is its
    own proxy.

    **It falls back to a private temp directory when the path would not fit.**
    `sun_path` is 108 bytes, and the `$PH_RUNTIME` of a test under the numbered
    tree of pytest can exceed it. The fallback goes through
    `temporary_directory`, so it is `0700` and it unwinds with the scope of the
    row and not with garbage collection (§4.9).
    """
    roots = resolve_roots(create=True)
    directory = roots.runtime / "sandbox"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"egress-{os.getpid()}-{next(_SOCKETS)}.sock"
    if len(str(path).encode()) > SOCKET_PATH_MAX:
        path = await temporary_directory(ctx, prefix="ph-egress-") / "egress.sock"
    return path


# --------------------------------------------------------------------- row --


BRIDGE_RETRY_SECONDS = 300.0
"""How long a failed probe is believed before a mount asks again.

A failure is not believed for the life of the process, as it used to be. One
slow or unlucky probe then declined egress to every root mounted after it. A
failure is not probed again at every mount either. On a host that has the
prerequisites but a bridge that fails, each probe spawns a confined command and
can wait out its connect timeout. A daemon that mounts a burst of roots would
pay that one time for each root."""

_BRIDGE_VERDICT: tuple[SandboxProbe, float] | None = None
"""What `probe_egress` learned, and until when it is believed
(`time.monotonic`). A bridge that it proved is believed for the life of the
process. A bridge that it could not prove is believed for
`BRIDGE_RETRY_SECONDS`.

**The verdict is per process, never persisted, and deliberately unlike the
filesystem probe.** The filesystem probe refuses to be cached at all, because
`kernel.apparmor_restrict_unprivileged_userns` is a live sysctl and a
remembered "yes" would be a fail-open. Nothing that this probe establishes can
change in that direction. It shows that `socat` is installed, that a confined
shell reaches a unix socket under `$PH_RUNTIME` through the read-only bind, and
that the proxy answers. And if the sysctl *did* change, `probe_sandbox`
declines first and this is never reached. `probe_sandbox` is still uncached
and still runs at every mount.

The cache matters because a daemon mounts the profile **one time for each
root**, and the probe costs 19.4 ms of serial startup each time. Eight roots
paid ~155 ms to learn one fact about the host again.
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
        where = str(self.proxy.path)
        if self.proxy.tcp_port is not None:
            where += f" and 127.0.0.1:{self.proxy.tcp_port}"
        return [
            ("egress", f"proxy at {where}; {self.verdict.because}"),
            ("egress refused", str(self.proxy.denied)),
        ]

    async def mount(self, ctx: Context, backend: LocalBackend, scratch: Path) -> None:
        """Start the proxy, prove the bridge, and only then register it.

        The proxy runs before the probe, because the probe needs something to
        answer. `on_denied` is attached *after* the probe, because the refusal
        that the probe itself causes belongs to no agent. A bridge that fails
        its probe is closed again here, and is not left listening for nothing.
        """
        path = await egress_socket_path(ctx)
        # The backend declares the door (`needs_loopback`). A namespace gets a
        # shim on the unix socket. A backend that denies and does not unshare
        # gets the proxy on loopback too.
        proxy = EgressProxy(
            path=path, permits=ctx.require(SANDBOX).permits, loopback=backend.needs_loopback
        )
        # The close is registered before the start. A start that fails or is
        # canceled part-way, with the unix listener up and the loopback one
        # refused, already has its close.
        closing = ctx.add_disposer(proxy.aclose, label="sandbox-local(egress proxy)")
        try:
            await proxy.start()
        except OSError as error:
            # A host that will not let this process listen is a host with no
            # bridge, and not a row that fails to mount. Examples: a sandbox
            # around this process that refuses unix sockets, or a runtime
            # directory that this process cannot write.
            await maybe_await(closing())
            self.verdict = SandboxProbe(False, f"could not listen on {path}: {error}")
            log.info("ph.seams.sandbox_local: no egress bridge — %s", self.verdict.because)
            return
        global _BRIDGE_VERDICT
        # Where the command dials: the loopback port of the proxy itself where
        # the proxy opened one, or else the fixed port of the shim inside the
        # namespace. The value is read from the proxy and not derived again
        # from the backend, so there is one answer.
        bridge = Egress(socket=str(path), port=proxy.tcp_port or EGRESS_PORT)
        known = _BRIDGE_VERDICT
        if known is not None and time.monotonic() < known[1]:
            self.verdict = known[0]
        else:
            self.verdict = await probe_egress(ctx, backend, bridge, scratch)
            until = math.inf if self.verdict.confines else time.monotonic() + BRIDGE_RETRY_SECONDS
            _BRIDGE_VERDICT = (self.verdict, until)
        if not self.verdict.confines:
            log.info("ph.seams.sandbox_local: no egress bridge — %s", self.verdict.because)
            # The close goes through the releaser, so it leaves the scope of
            # the row as it runs. Called bare, it stayed registered for a proxy
            # that was already closed.
            await maybe_await(closing())
            return

        def denied(host: str, port: int, agent: str | None) -> None:
            ctx.require(SANDBOX).record_denial(
                Denial(kind="network", via="proxy", host=host, port=port), agent=agent
            )

        proxy.on_denied = denied
        ctx.require(SANDBOX).register_egress(bridge)
        self.proxy = proxy


class Config(WireModel):
    """Row config for the local confinement backend."""

    root: str | None = None
    """Where the probes do their work. `$PH_HOME/sandbox` by default."""


@plugin("sandbox-local", affects="environment", inject=[SANDBOX, SUBPROCESS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Probe the host, and claim the sandbox slot only if the kernel enforced it.

    **The probe gates registration**, for the reason that `workspace-agentfs`
    gives (`ph.seams.workspace_agentfs`). But the consequence here is more
    serious. `SandboxSeam.confine` raises when no backend is mounted, so a row
    that declines leaves callers with a refusal that they can see. A row that
    *registered* and could not enforce would give every caller a wrapped argv
    that bounds nothing, and `enforcement_of` would tell `containment.strict`
    that the deployment is confined. To fail closed, the row does not claim the
    slot.

    The egress bridge is gated the same way, after the backend is registered.
    A proxy that nobody inside can reach is not a door, and registering it
    would make `allowlist` promise a network that the command does not have.

    The result reaches `phern doctor` in both cases. "why is strict refusing to
    start" is a question that a user asks the tool, and a row that declined in
    silence looks the same as a row that nobody mounted.
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
    ctx.require(SANDBOX).register_provider(backend)
    await egress.mount(ctx, backend, scratch)
