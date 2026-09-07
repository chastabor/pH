"""P6-04 — `sandbox-local`: the first backend the kernel actually enforces.

Gate: *an absolute-path write is refused under `sandbox`.*

**`confine` builds an argv and runs nothing, which is why most of this file is
assertions about strings.** That is not a compromise: the argv *is* the policy,
so reading it is testing the rule, and it is what lets the Seatbelt half be
reviewed on a machine that cannot run it.

The kernel half runs where a kernel will enforce it — on Ubuntu 23.10+ that means
an AppArmor profile at `/etc/apparmor.d/bwrap`, without which an unprofiled,
non-setuid `bwrap` cannot create a user namespace at all. Those tests skip
cleanly elsewhere, and the row declines rather than claiming the tier.

## What the confinement was measured to do, and what the wrapper costs

**Verified against a real kernel** (bwrap, with `/etc/apparmor.d/bwrap` in place):
an absolute-path write **refused with the host file untouched**, a workspace write
**landing**, `read-only` **refusing everywhere**, and **one network interface
against thirteen** on the host.

**And against Seatbelt** (macOS 26.6, 2026-09-07), where the blind profile was wrong
in the ways the module docstring lists: canonical paths, no shim, a loopback door,
and `Operation not permitted` as the kernel's one word for both boundaries. The
network tests ask their question the same way on both platforms — can a confined
command reach a listener this process opened on the host's loopback — because
one backend unshares and the other denies, and only that question is the same.

**Where the time goes, so nobody optimises the wrong end.** Building the argv
costs **0.83 µs**; wrapping a command in `bwrap` costs **~5.8 ms fixed plus
~0.3 ms per writable root**. The Python is **0.024%** of the price, and the best
possible micro-optimisation of it is **0.004%** — 3 323 confined commands to save
one millisecond. `confine` is left exactly as it is on purpose.

**Why the probe asserts isolation rather than availability.** This module has
watched a backend fail open twice next door: `agentfs run --experimental-sandbox`
exits 0 while writing straight through to the host, and a `mount`-based overlay
does not confine the process at all. A confinement backend that silently fails
open is the worst object in this codebase — every caller believes the kernel is
holding the line and nothing says otherwise — so an exit code is never read as
proof of confinement (E13).

## Why the probe is one spawn and not two

Namespace setup is paid per spawn, and the probe runs on the serial startup path
where nothing else can overlap it: **14.5 ms as two spawns against 8.2 ms as one**.
`sh` keeps going after a failed redirect, so both writes are attempted and both
outcomes stay readable — the checks remain independent without paying twice.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from ph.seams.sandbox import Egress, SandboxError, SandboxPolicy, writable_paths
from ph.seams.sandbox_egress import PROBE_HOST
from ph.seams.sandbox_local import (
    BYPASS_VARIABLES,
    DENIAL_SIGNATURES,
    EGRESS_PORT,
    PROXY_PASSWORD,
    PROXY_VARIABLES,
    SEATBELT_SIGNATURES,
    Bubblewrap,
    Seatbelt,
    local_backend,
    proxy_url,
    seatbelt_profile,
)
from ph.seams.subprocess import SubprocessSpawnSpec, scrub_env
from ph.testing.diagnostics import report_section

pytestmark = pytest.mark.anyio

ROW = {"id": "sandbox-local", "disabled": False}
"""The row ships in `ph-base`; the `mount` fixture turns it off for the suite, and
this turns it back on for the tests that are about it."""


def _policy(root: str = "/w", **extra: Any) -> SandboxPolicy:
    return SandboxPolicy(mode="workspace-write", workspace_root=root, **extra)


def _binds(argv: tuple[str, ...]) -> list[str]:
    """Which paths got a writable `--bind`, in order.

    One reader for both bind tests: the index arithmetic the first one used broke
    the moment anything was bound before the workspace.
    """
    return [argv[index + 1] for index, token in enumerate(argv) if token == "--bind"]


# ------------------------------------------------------------ writable set --


def test_the_writable_set_is_exhaustive_over_the_modes() -> None:
    """On the seam beside `SandboxPolicy`, and a `match` rather than an `if`.

    `read-only` drops the workspace root and keeps only what the caller named —
    no implicit `/tmp`, because a temp directory nobody declared is a writable
    hole. `danger-full-access` is not a *set*: it means "everything", which each
    backend spells its own way.
    """
    assert writable_paths(_policy(writable_extra=["/s"])) == ["/w", "/s"]
    assert writable_paths(SandboxPolicy(mode="read-only", workspace_root="/w")) == []
    assert writable_paths(SandboxPolicy(mode="danger-full-access", workspace_root="/w")) == []


# ------------------------------------------------------------------- bwrap --


def test_the_tree_is_read_only_and_the_workspace_is_punched_back_through() -> None:
    """**Order is the enforcement.** bwrap applies binds left to right and the
    last one covering a path wins, so `--ro-bind / /` must come before the
    writable roots or the workspace would be read-only too."""
    argv = Bubblewrap().confine(("cmd",), _policy()).argv

    assert argv.index("--ro-bind") < argv.index("--bind"), "read-only first, then the holes"
    assert _binds(argv) == ["/w"]
    assert argv[-1] == "cmd" and argv[-2] == "--"


def test_read_only_mode_names_no_writable_root_even_when_one_is_given() -> None:
    """The combination P6-05's `readonly-scratch` will produce, pinned here: a
    policy carrying a workspace root that the mode says is not writable."""
    argv = Bubblewrap().confine(("cmd",), SandboxPolicy(mode="read-only", workspace_root="/w")).argv

    assert _binds(argv) == []
    assert "--ro-bind" in argv


def test_extra_writable_paths_are_bound_after_the_workspace() -> None:
    """Later wins in bwrap, so an extra overlapping the workspace takes effect —
    which is what a caller naming it explicitly means."""
    argv = Bubblewrap().confine(("cmd",), _policy(writable_extra=["/scratch"])).argv

    assert _binds(argv) == ["/w", "/scratch"]


def test_the_process_table_is_unshared_along_with_proc() -> None:
    """**A fresh `/proc` alone is not process isolation**, which this row claimed
    before it was measured: with `--proc /proc` and no `--unshare-pid` the
    confined process saw 628 PIDs against the host's 626 — the whole table, and
    signalable. With the flag, 4. Filesystem confinement beside a wide-open
    process table is not what this rung says it buys.
    """
    argv = Bubblewrap().confine(("cmd",), _policy()).argv

    assert "--proc" in argv and "--unshare-pid" in argv


def test_the_network_namespace_is_unshared_unless_the_policy_asks_for_it() -> None:
    """A namespace with no interface but loopback is not a filter that can be
    talked past, which is why this is the mechanism rather than a rule."""
    assert "--unshare-net" in Bubblewrap().confine(("cmd",), _policy()).argv
    assert "--unshare-net" not in Bubblewrap().confine(("cmd",), _policy(network=True)).argv


def test_full_access_is_a_permissive_sandbox_rather_than_an_unwrapped_argv() -> None:
    """**The seam's contract, kept at the mode that means "no confinement".**

    `SandboxSeam.confine` promises never to hand back `argv` unchanged, because a
    caller must not be able to mistake absence for confinement. So the dangerous
    mode is a sandbox that permits everything — the wrapper is still there, and
    `backend` still names who bounded it.
    """
    confined = Bubblewrap().confine(("cmd",), SandboxPolicy(mode="danger-full-access"))

    assert confined.argv[0] == "bwrap"
    assert confined.argv[2:5] == ("--bind", "/", "/")
    assert "--ro-bind" not in confined.argv


# ---------------------------------------------------------------- seatbelt --


def test_the_profile_denies_by_default_and_allows_back() -> None:
    """The only direction that fails closed: an allow-by-default profile permits
    every rule anybody forgot."""
    profile = seatbelt_profile(_policy())

    assert profile.splitlines()[1] == "(deny default)"
    assert '(allow file-write* (subpath "/w"))' in profile


def test_the_network_line_is_the_one_that_changes_something() -> None:
    """`(deny network*)` is a no-op under `(deny default)`, so only the
    permitting arm is emitted — a line that changes nothing is a line a test can
    assert while proving nothing, which is what the first version did."""
    assert "(allow network*)" in seatbelt_profile(_policy(network=True))
    assert "network" not in seatbelt_profile(_policy())


def test_reads_stay_open_because_the_read_boundary_is_not_this_seam_s() -> None:
    """A confinement tier is about what an agent can *change*. A profile that
    also denied reads would refuse the toolchain its own libraries and be
    switched off by the first person who met it."""
    assert "(allow file-read*)" in seatbelt_profile(SandboxPolicy(mode="read-only"))


def test_every_mode_goes_through_one_profile_builder() -> None:
    """Including `danger-full-access`: one function is documented as *the* profile
    for a policy, so a policy whose profile was built elsewhere made that false —
    and made the argv assertion below true only for the modes it did own."""
    for mode in ("read-only", "workspace-write", "danger-full-access"):
        policy = SandboxPolicy(mode=mode, workspace_root="/w")  # type: ignore[arg-type]
        confined = Seatbelt().confine(("cmd", "arg"), policy)
        assert confined.argv[:2] == ("sandbox-exec", "-p")
        assert confined.argv[2] == seatbelt_profile(policy)
        assert confined.argv[3:] == ("cmd", "arg")
    assert seatbelt_profile(SandboxPolicy(mode="danger-full-access")) == (
        "(version 1)\n(allow default)"
    )


# ------------------------------------------------------------- the backend --


def test_the_backend_is_chosen_by_platform_then_by_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All five outcomes, none of them the host's.

    The first version branched on `sys.platform` and asserted
    `isinstance(backend, Bubblewrap) or "bwrap" in why` — where `why` is `"bwrap"`
    on success and `"bwrap is not installed"` on failure, so the substring held
    either way and the test passed even if the wrong class were selected. Two of
    its three arms never ran, and the live one asserted nothing.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert local_backend() == (None, "sandbox-exec is not installed")

    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    assert local_backend() == (Seatbelt(), "sandbox-exec")

    monkeypatch.setattr(sys, "platform", "linux")
    assert local_backend() == (Bubblewrap(), "bwrap")

    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert local_backend() == (None, "bwrap is not installed")

    monkeypatch.setattr(sys, "platform", "sunos5")
    assert local_backend() == (None, "no confinement backend for sunos5")


async def test_the_row_declines_rather_than_claiming_confinement_it_cannot_prove(
    mount: Any,
) -> None:
    """**The whole safety argument of this row.**

    A backend that registered without enforcing would hand every caller a wrapped
    argv that bounds nothing, and `enforcement_of` would tell
    `containment.strict` the deployment is confined — the exact inversion this
    seam exists to prevent. So a host that cannot enforce leaves the slot empty
    and `confine` raising.

    Read off the mounted row rather than by probing again: `apply` has already
    paid for the verdict, and a second probe would compare two independent runs
    rather than assert the row's decision.
    """
    ctx = await mount(ROW)

    if ctx.sandbox.provider is None:
        assert ctx.sandbox.enforcement is None, "no backend is not partial confinement"
        with pytest.raises(SandboxError):
            ctx.sandbox.confine(("cmd",), _policy())
    else:
        assert ctx.sandbox.enforcement == "full"


async def test_the_verdict_reaches_ph_doctor_either_way(mount: Any) -> None:
    """ "Why is strict refusing to start" is asked of the tool, not of the source.

    A decline that says only *"the sandbox refused a write"* is true and useless;
    the backend's own `setting up uid map: Permission denied` is what tells an
    operator to add an AppArmor profile. So the section is asserted, not just the
    registration.
    """
    ctx = await mount(ROW)
    rows = report_section(ctx, "Local confinement")

    assert rows["backend"], "which backend answered, or why there was none"
    assert rows["because"], "a decline with no reason is indistinguishable from no row"
    assert rows["confines"] in ("yes", "no")


# ------------------------------------------------ the kernel, where it does --


async def _run(ctx: Any, argv: tuple[str, ...], cwd: Path) -> tuple[int, str]:
    """One confined argv, run. Code plus output, so a failure says why."""
    outcome = await ctx.subprocess.run(SubprocessSpawnSpec(argv=argv, cwd=cwd, env=scrub_env()))
    return outcome.exit_code, outcome.stdout + outcome.stderr


async def _enforcing(mount: Any, tmp_path: Path) -> tuple[Any, Path]:
    """A mounted row over a host whose kernel enforces, and a workspace, or a skip."""
    ctx = await mount(ROW)
    if ctx.sandbox.provider is None:
        pytest.skip("no enforcing sandbox backend on this host")
    workspace = tmp_path / "work"
    workspace.mkdir()
    return ctx, workspace


async def test_an_absolute_path_write_is_refused(mount: Any, tmp_path: Path) -> None:
    """**P6-04's gate, and the half of E13 no tier below this one can close.**

    `test_containment_ladder.py` asserts the escape at the `worktree` rung and
    names this as its other half. It uses a **raw `open()` from a bare spawn**,
    deliberately — a shell redirect could be caught by something other than the
    kernel, and would then pass "because of that branch rather than because of
    `sandbox`". So the same write is used here, or the pair proves two different
    things.

    Both directions in one test on purpose: a backend refusing *everything* would
    pass the escape half while being unusable, and the workspace write is what
    stops that reading as success.
    """
    ctx, workspace = await _enforcing(mount, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("host", encoding="utf-8")
    policy = SandboxPolicy(mode="workspace-write", workspace_root=str(workspace))

    def raw(target: Path) -> tuple[str, ...]:
        return (sys.executable, "-c", f"open({str(target)!r}, 'w').write('written')")

    landed, output = await _run(
        ctx, ctx.sandbox.confine(raw(workspace / "in.txt"), policy).argv, workspace
    )
    assert landed == 0, output
    assert (workspace / "in.txt").read_text(encoding="utf-8") == "written"

    escaped, output = await _run(ctx, ctx.sandbox.confine(raw(outside), policy).argv, workspace)
    assert escaped != 0, "the kernel must refuse this"
    assert outside.read_text(encoding="utf-8") == "host", "and the host file must be untouched"


async def test_read_only_mode_refuses_the_workspace_too(mount: Any, tmp_path: Path) -> None:
    """The mode `readonly-scratch` will be built on (P6-05): nothing is writable,
    including the directory the agent is standing in."""
    ctx, workspace = await _enforcing(mount, tmp_path)

    confined = ctx.sandbox.confine(
        (sys.executable, "-c", "open('here.txt', 'w').write('x')"),
        SandboxPolicy(mode="read-only"),
    )

    code, _ = await _run(ctx, confined.argv, workspace)
    assert code != 0
    assert not (workspace / "here.txt").exists()


FULL_NETWORK = {"id": "sandbox-allow", "config": {"network": {"mode": "full"}}}
"""The deployment saying confined commands get the host's network."""

# `/proc/net/dev`, not `/sys/class/net`: `/sys` is bind-mounted from the host and
# its sysfs is not namespace-aware, so it reports the host's 13 interfaces even
# inside `--unshare-net`. The netlink-backed `/proc` view is the one the namespace
# actually owns — measured 1 against 13. Linux only, like the namespaces it counts.
LINKS = "print(len(open('/proc/net/dev').read().splitlines()) - 2)"
PIDS = "import os;print(len([p for p in os.listdir('/proc') if p.isdigit()]))"


async def _said(ctx: Any, workspace: Path, script: str, **extra: Any) -> str:
    """What one confined `python -c` printed, run through the **seam** so the
    deployment's allowances are merged in — which is what makes `refuse_network=` a
    request rather than the resolved fact a backend reads.

    The two questions below differ only in how they read this string, so they share
    the three lines that produce it.
    """
    policy = SandboxPolicy(mode="workspace-write", workspace_root=str(workspace), **extra)
    argv = ctx.sandbox.confine((sys.executable, "-c", script), policy).argv
    _, out = await _run(ctx, argv, workspace)
    return out


async def _count(ctx: Any, workspace: Path, script: str, **extra: Any) -> int:
    """A confined command's printed number — the Linux namespace counts."""
    return int((await _said(ctx, workspace, script, **extra)).strip())


async def _isolated(ctx: Any, workspace: Path, **extra: Any) -> bool:
    """Whether a confined command is cut off from the network, asked the one way that
    is the same on every backend: can it reach a listener *this process* opened on
    the host's loopback?

    Under `bwrap --unshare-net` the sandbox has a loopback of its own and the connect
    finds nothing listening; under Seatbelt the profile allows no remote but the
    proxy's port and the connect is `Operation not permitted`. Either way "reached"
    never prints.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    try:
        port = listener.getsockname()[1]
        reach = (
            f"import socket; socket.create_connection(('127.0.0.1', {port}), 3); print('reached')"
        )
        return "reached" not in await _said(ctx, workspace, reach, **extra)
    finally:
        listener.close()


async def test_the_namespaces_are_real(mount: Any, tmp_path: Path) -> None:
    """`--unshare-net` and `--unshare-pid` are namespaces, not filters — nothing
    inside can talk past them, which is why they are the mechanism. Seatbelt has
    no namespaces and *is* a filter, at the kernel; the loopback question below is
    what both must answer the same way.

    **A caller cannot widen the deployment's posture** (§6.5): there is no way to
    ask for the network at all, only to refuse it, and the shipped `sandbox-allow`
    says `allowlist` — still one interface and a door. The host's interfaces appear
    only when the *deployment* says `full`, which is the next test.
    """
    ctx, workspace = await _enforcing(mount, tmp_path)

    assert await _isolated(ctx, workspace), "a listener on the host's loopback is out of reach"
    assert await _isolated(ctx, workspace, refuse_network=True), (
        "and a caller refusing the network still has none"
    )
    if sys.platform.startswith("linux"):
        assert await _count(ctx, workspace, LINKS) == 1, "loopback and nothing else"
        assert await _count(ctx, workspace, PIDS) < 10, "its own processes, not the host's table"


async def test_the_deployment_can_hand_out_the_hosts_network(mount: Any, tmp_path: Path) -> None:
    """`network.mode: full` is the one way to the host's interfaces, and it is the
    deployment's to say — a row, not a per-call flag."""
    ctx = await mount(ROW, FULL_NETWORK)
    if ctx.sandbox.provider is None:
        pytest.skip("no enforcing sandbox backend on this host")
    workspace = tmp_path / "work"
    workspace.mkdir()

    assert not await _isolated(ctx, workspace), "the host's network, when the row says so"
    assert await _isolated(ctx, workspace, refuse_network=True), (
        "and a caller refusing the network still gets none"
    )
    if sys.platform.startswith("linux"):
        assert await _count(ctx, workspace, LINKS) > 1, "the host's interfaces, counted"


# ------------------------------------------------------------------- egress --


def _egress(agent: str | None = "a1") -> Egress:
    return Egress(socket="/run/user/1/ph/sandbox/egress-1-1.sock", port=EGRESS_PORT, agent=agent)


def test_the_egress_door_is_a_shim_and_the_proxy_variables_with_the_namespace_kept() -> None:
    """P6-38's argv, read as the policy it is: the namespace is still unshared, the
    proxy variables point at loopback with the agent in the user part, the bypass
    variables are gone, and the command runs behind a `socat` shim that is waited
    for before it `exec`s."""
    argv = Bubblewrap().confine(("curl", "https://x"), _policy(egress=_egress())).argv

    assert "--unshare-net" in argv, "the door does not open the namespace"
    settings = {argv[i + 1]: argv[i + 2] for i, token in enumerate(argv) if token == "--setenv"}
    assert set(settings) == set(PROXY_VARIABLES)
    assert set(settings.values()) == {f"http://a1:{PROXY_PASSWORD}@127.0.0.1:{EGRESS_PORT}"}
    unset = [argv[i + 1] for i, token in enumerate(argv) if token == "--unsetenv"]
    assert unset == list(BYPASS_VARIABLES)

    command = argv[argv.index("--") + 1 :]
    assert command[:2] == ("/bin/sh", "-c")
    assert command[3:] == ("ph-egress", "curl", "https://x"), "the command is `$@`, untouched"
    script = command[2]
    assert f"TCP-LISTEN:{EGRESS_PORT},bind=127.0.0.1,fork,reuseaddr" in script
    assert "UNIX-CONNECT:/run/user/1/ph/sandbox/egress-1-1.sock" in script
    assert script.rstrip().endswith('exec "$@"')
    assert "until socat" in script, "the shim is waited for, not assumed"


def test_the_backend_names_itself_rather_than_being_named_by_its_class() -> None:
    """`ConfinedArgv.backend` is the provider's declared name, so a consumer that
    wants to tell a person what is confining them never reads a Python class."""
    assert Bubblewrap().backend == "bwrap"
    assert Seatbelt().backend == "sandbox-exec"
    assert Bubblewrap().confine(("true",), _policy()).backend == "bwrap"


def test_the_signatures_are_each_backends_platform_and_not_the_seams() -> None:
    """Every sentence in `DENIAL_SIGNATURES` is Linux/glibc/bwrap and every one in
    `SEATBELT_SIGNATURES` is macOS — measured on each, and disjoint, so neither
    backend records the other kernel's refusals as its own."""
    assert ("filesystem", "Read-only file system") in DENIAL_SIGNATURES
    assert ("filesystem", "Operation not permitted") in SEATBELT_SIGNATURES
    kernel_words = {
        table: {mark for kind, mark in table if kind == "filesystem"}
        for table in (DENIAL_SIGNATURES, SEATBELT_SIGNATURES)
    }
    assert not kernel_words[DENIAL_SIGNATURES] & kernel_words[SEATBELT_SIGNATURES]
    shared = {mark for _, mark in DENIAL_SIGNATURES} & {mark for _, mark in SEATBELT_SIGNATURES}
    assert shared == {"Could not resolve host"}, "curl's sentence, the same on both; not a kernel's"
    assert Bubblewrap().read_denial("sh: /x: Operation not permitted", network=False) is None
    assert Seatbelt().read_denial("sh: /x: Read-only file system", network=False) is None


def test_seatbelt_reads_the_path_on_the_line_to_tell_the_boundaries_apart() -> None:
    """Seatbelt answers a refused `open()` and a refused `connect()` with one `EPERM`,
    and every tool measured puts the path on the line for the first and nothing for
    the second — so the line, not the sentence, names the boundary. Each string here
    is one that was printed under the real profile on 2026-09-07."""
    reader = Seatbelt()
    trace = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 1, in <module>\n'
        "PermissionError: [Errno 1] Operation not permitted: '/tmp/ph_sb_x'\n"
    )
    file = reader.read_denial(trace, network=False)
    assert file is not None and (file.kind, file.via) == ("filesystem", "output")
    assert file.evidence == "PermissionError: [Errno 1] Operation not permitted: '/tmp/ph_sb_x'"
    made = reader.read_denial("mkdir: /tmp/d: Operation not permitted", network=True)
    assert made is not None and made.kind == "filesystem", (
        "a path is a file refusal whatever the network posture"
    )

    bare = "PermissionError: [Errno 1] Operation not permitted\n"
    dialled = reader.read_denial(bare, network=False)
    assert dialled is not None and dialled.kind == "network", "no path, no network: a socket"
    assert reader.read_denial(bare, network=True) is None, (
        "with the host's network a bare EPERM is neither boundary this seam bounds"
    )

    dns = "socket.gaierror: [Errno 8] nodename nor servname provided, or not known"
    resolved = reader.read_denial(dns, network=False)
    assert resolved is not None and resolved.kind == "network"
    assert reader.read_denial("curl: (6) Could not resolve host: x", network=True) is None
    assert reader.read_denial("Couldn't connect to server", network=False) is None, (
        "curl's generic connect failure is what a down server prints too"
    )
    assert reader.read_denial("all fine\n", network=False) is None

    both = f"{dns}\nrm: /etc/hosts: Operation not permitted\n"
    first = reader.read_denial(both, network=False)
    assert first is not None and first.kind == "network", (
        "the earliest refusal wins, as bwrap's reader does"
    )


def test_the_seatbelt_profile_names_the_path_the_kernel_resolves(tmp_path: Path) -> None:
    """Seatbelt matches the path the kernel resolves, and on macOS `/var` and `/tmp`
    are symlinks into `/private`: a workspace under `$TMPDIR` named as given was
    refused its own writes and the probe declined the tier (measured — see
    `_canonical`). So the profile carries the canonical path, never the spelling."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    profile = seatbelt_profile(_policy(str(link), writable_extra=[str(link / "deeper")]))

    assert f'(allow file-write* (subpath "{os.path.realpath(link)}"))' in profile
    assert f'(allow file-write* (subpath "{os.path.realpath(link)}/deeper"))' in profile, (
        "a tail that does not exist yet is kept"
    )
    assert f'"{link}"' not in profile


def test_the_wrapper_dies_with_the_host_and_says_whether_signals_reach_the_child() -> None:
    """Two flags another package now depends on, pinned where they are produced.

    `--die-with-parent` is what the guest's `die_with_parent` leans on: it stopped
    checking whether it had been orphaned, on the argument that inside a PID
    namespace the sandbox propagates the host's death instead. That argument is only
    true while this flag is here, and nothing asserted it.

    `forwards_signals` is the other half: `bwrap` does *not* forward, measured — it
    dies of `SIGINT` itself and takes the namespace with it — so a caller with a
    cooperative stop of its own reads this rather than guessing from "am I
    confined". `sandbox-exec` execs its target and does forward.
    """
    confined = Bubblewrap().confine(("python",), _policy())
    assert "--die-with-parent" in confined.argv
    assert "--unshare-pid" in confined.argv, "the namespace the teardown argument rests on"
    assert confined.forwards_signals is False
    assert Seatbelt().confine(("python",), _policy()).forwards_signals is True


def test_the_proxy_url_carries_no_user_when_there_is_no_agent() -> None:
    assert proxy_url(_egress(agent=None)) == f"http://127.0.0.1:{EGRESS_PORT}"
    assert (
        proxy_url(_egress(agent="s/1")) == f"http://s%2F1:{PROXY_PASSWORD}@127.0.0.1:{EGRESS_PORT}"
    )


def test_a_command_with_the_hosts_network_gets_no_shim() -> None:
    """`full` is the host's network; the door is for `allowlist` only, and a seam
    that hands both to a backend has made a mistake this backend does not repeat."""
    argv = Bubblewrap().confine(("curl",), _policy(network=True, egress=_egress())).argv  # type: ignore[arg-type]
    assert "--unshare-net" not in argv
    assert "--setenv" not in argv
    assert argv[argv.index("--") + 1 :] == ("curl",)


def test_the_seatbelt_profile_opens_only_the_door() -> None:
    """Measured on macOS: the door is the proxy's own loopback port — one `remote ip`
    line, in the spelling Seatbelt accepts — and nothing that would let a command
    dial out itself. **No shim**, because there is no PID namespace for one to die
    with: a `socat` left by the blind design was still holding the port an hour
    later, and the next command's bind found it taken."""
    with_door = seatbelt_profile(_policy(egress=_egress()))
    assert f'(allow network-outbound (remote ip "localhost:{EGRESS_PORT}"))' in with_door
    assert "(allow network*)" not in with_door
    assert "unix-socket" not in with_door and "network-bind" not in with_door, (
        "nothing to bind and nothing to reach by path"
    )
    assert "network" not in seatbelt_profile(_policy()), "no door, no lines"
    assert "remote ip" not in seatbelt_profile(_policy(network=True, egress=_egress()))  # type: ignore[arg-type]

    confined = Seatbelt().confine(("curl",), _policy(egress=_egress()))
    assert confined.argv[3] == "/usr/bin/env", "sandbox-exec sets no environment; env does"
    assert confined.argv[4:8] == ("-u", "NO_PROXY", "-u", "no_proxy"), "the bypass list goes"
    assert f"HTTPS_PROXY=http://a1:{PROXY_PASSWORD}@127.0.0.1:{EGRESS_PORT}" in confined.argv
    assert confined.argv[-1] == "curl", "env execs the command itself; no sh, no socat"
    assert not any("socat" in token for token in confined.argv)
    assert Seatbelt().needs_loopback and not Bubblewrap().needs_loopback


def test_each_backend_owns_its_door_rather_than_the_row_switching_on_it() -> None:
    """The declaration the row reads before it has confined anything, and the two
    behaviours it must not have to know about. A `Literal` switch read by
    `probe_egress` meant a third backend edited the probe; these mean it does not.
    """
    bridge = _egress()
    assert Bubblewrap().egress_probe(bridge)[:2] == ("/bin/sh", "-c")
    assert "socat" in Bubblewrap().egress_probe(bridge)[2]
    assert PROBE_HOST in Bubblewrap().egress_probe(bridge)[2]

    door = Seatbelt().egress_probe(bridge)
    assert door[0] == sys.executable and door[1] == "-c"
    assert PROBE_HOST in door[2] and str(bridge.port) in door[2]
    assert "socat" not in door[2], "the loopback door needs no binary a Mac lacks"
    assert Seatbelt().egress_blocker() is None, "nothing to install for a TCP connect"
