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

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

from ph.seams.sandbox import SandboxError, SandboxPolicy, writable_paths
from ph.seams.sandbox_local import (
    Bubblewrap,
    Seatbelt,
    local_backend,
    seatbelt_profile,
)
from ph.seams.subprocess import SubprocessSpawnSpec, scrub_env
from ph.testing.diagnostics import report_section

pytestmark = pytest.mark.anyio

ROW = {"insert": [{"id": "sandbox-local", "name": "sandbox-local"}]}


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


async def test_the_namespaces_are_real(mount: Any, tmp_path: Path) -> None:
    """`--unshare-net` and `--unshare-pid` are namespaces, not filters — nothing
    inside can talk past them, which is why they are the mechanism."""
    ctx, workspace = await _enforcing(mount, tmp_path)

    async def count(script: str, **extra: Any) -> int:
        policy = SandboxPolicy(mode="workspace-write", workspace_root=str(workspace), **extra)
        argv = ctx.sandbox.confine((sys.executable, "-c", script), policy).argv
        _, out = await _run(ctx, argv, workspace)
        return int(out.strip())

    # `/proc/net/dev`, not `/sys/class/net`: `/sys` is bind-mounted from the
    # host and its sysfs is not namespace-aware, so it reports the host's 13
    # interfaces even inside `--unshare-net`. The netlink-backed `/proc` view is
    # the one the namespace actually owns — measured 1 against 13.
    links = "print(len(open('/proc/net/dev').read().splitlines()) - 2)"
    pids = "import os;print(len([p for p in os.listdir('/proc') if p.isdigit()]))"

    assert await count(links) == 1, "loopback and nothing else"
    assert await count(links, network=True) > 1, "the host's interfaces, when allowed"
    assert await count(pids) < 10, "its own processes, not the host's table"
