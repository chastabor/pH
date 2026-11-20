"""The RLM kernel, confined at the kernel (P6-39).

`cwd` puts a cell in the tree of the agent, so a *relative* write lands there.
Only confinement refuses `open("/etc/passwd", "w")`, and §4.8 names that as the
one thing that no tier below `sandbox` can do. Until this row, the kernel was
spawned unconfined. `permissions-fs` therefore told operators that "a sandbox
provider bounds what a code cell can reach directly" while nothing did.

**The whole design depends on two properties, and both are measured here, not
argued.** fd 3 must survive the wrapper, or the kernel cannot talk to its host at
all. An absolute-path write must actually fail. The first is checked against the
real argv that the backend of this platform builds, with a live socket at the far
end. It is the one that would fail *silently at boot* on a host that nobody
tested. The kernel would hang until `boot_timeout`, and would be reported as a
dead runtime and not as a lost channel.

The enforcement half skips where the kernel cannot enforce: no `bwrap`, no
AppArmor profile on Ubuntu 23.10+, or no `sandbox-exec`. It skips through the
same in-body check that `test_sandbox_local.py` uses. That check asks whether the
row actually registered a provider, and not whether the binary exists.
`needs_backend` is only on the test that drives the backend of the platform
directly without mounting the row, where there is no provider to ask. Both
backends were verified: `bwrap` on 2026-09-01, and Seatbelt on 2026-09-07.
Seatbelt was verified only after the macOS boot hang that `test_boot_report.py`
covers was found, because a kernel that never reports ready confines nothing.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from rlm_fixtures import MountedRuntime, agent_with_workspace
from runtime_helpers import run_in_kernel

from ph.keys import AGENTS, SANDBOX
from ph.orphans import OrphanJournal
from ph.seams.sandbox import DENIED, ConfinedArgv, SandboxPolicy
from ph.seams.sandbox_local import Bubblewrap, Seatbelt, local_backend
from ph.seams.workspace import workspace_policy
from ph.testing import StubSandboxProvider, report_section
from ph_rlm.kernel.manager import Kernel, KernelLimits, PythonCodeRuntime
from ph_rlm.kernel.venv import resolve_interpreter
from ph_rlm.keys import PYTHON_RUNTIME
from ph_runtime.protocol import FD_ENV

pytestmark = pytest.mark.anyio

SANDBOX_ROW: dict[str, Any] = {"id": "sandbox-local", "disabled": False}

WRITABLE_ROW: dict[str, Any] = {"id": "sandbox", "config": {"defaultMode": "workspace-write"}}
"""The posture that a cell needs to write its own workspace. It is stated here
and not assumed.

Since N1, `SandboxSeam.effective` limits the mode of a caller to the resolved
mode of the deployment, and `workspace_policy` no longer guesses
`workspace-write`. The shipped `read-only` default of `ph-base` therefore now
reaches the kernel, which is correct, and a test that states no posture gets a
read-only cell. The RLM profile sets this."""

needs_backend = pytest.mark.skipif(
    local_backend()[0] is None, reason="no local confinement backend on this host"
)


# ------------------------------------------------------- fd 3 crosses it --


@needs_backend
def test_the_framed_channel_survives_the_wrapper(tmp_path: Path) -> None:
    """**The property that the whole design depends on.** `bwrap` passes an
    inherited descriptor through to the command that it execs, and so does
    `sandbox-exec` (both were measured). The fd 3 of the kernel therefore reaches
    the guest.

    The test runs the real argv that the backend of this platform builds, with a
    live socket at the far end. A wrapper that closed the descriptor would fail
    at *boot*: the guest would find nothing on fd 3, the host would wait for the
    whole `boot_timeout`, and the report would say that the runtime died, not
    that the channel never crossed.
    """
    work = tmp_path / "w"
    work.mkdir()
    host_end, child_end = socket.socketpair()
    try:
        child_fd = child_end.fileno()
        speak = f"import os, socket; socket.socket(fileno=int(os.environ[{FD_ENV!r}])).send(b'ok')"
        policy = SandboxPolicy(mode="workspace-write", workspace_root=str(work))
        backend = local_backend()[0]
        assert backend is not None
        argv = backend.confine((sys.executable, "-c", speak), policy).argv

        done = subprocess.run(
            argv,
            env={FD_ENV: str(child_fd), "PATH": "/usr/bin:/bin"},
            pass_fds=(child_fd,),
            capture_output=True,
            timeout=30,
            check=False,
        )
        child_end.close()
        host_end.setblocking(False)
        assert host_end.recv(16) == b"ok", f"fd 3 did not cross: {done.stderr!r}"
    finally:
        host_end.close()


# ------------------------------------------------------- what it resolves --


async def _confined(mounted_runtime: MountedRuntime, tmp_path: Path) -> tuple[Any, Any, Any]:
    """A mounted runtime whose kernels are really confined, plus the agent and its
    workspace. If that is not possible, a skip that says why.

    This has the shape of `test_sandbox_local._enforcing`. The question is whether
    the row *registered* a provider, which is what a probe against this host
    decided. The question is not whether a binary is on PATH.
    """
    ctx, session, agent = await mounted_runtime(extra_rows=[SANDBOX_ROW, WRITABLE_ROW])
    if ctx.require(SANDBOX).provider is None:
        pytest.skip("no enforcing sandbox backend on this host")
    workspace = await agent_with_workspace(ctx, session, agent, tmp_path / "project")
    return ctx, agent, workspace


async def test_a_kernel_is_confined_against_its_own_agents_workspace(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """The writable set is the workspace and the scratch directory of the agent
    itself. That is the policy of `ctx.shell`, from the same `workspace_policy`,
    so a deployment cannot have `tool-bash` confined and `run_code` not
    confined."""
    ctx, session, agent = await mounted_runtime()
    ctx.require(SANDBOX).register_provider(StubSandboxProvider())
    workspace = await agent_with_workspace(ctx, session, agent, tmp_path)
    runtime: PythonCodeRuntime = ctx.require(PYTHON_RUNTIME)

    confine = runtime.confiner(agent.id)

    assert confine is not None
    confined = confine(("python", "-m", "ph_runtime"))
    assert confined.backend == "stub"
    assert confined.policy is not None
    expected = workspace_policy(workspace)
    assert confined.policy.workspace_root == expected.workspace_root == str(workspace.root)
    assert str(workspace.scratch) in (confined.policy.writable_extra or [])


async def test_no_backend_and_no_workspace_are_both_declines_not_passthroughs(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """The answer is `None`, never an unwrapped argv. The seam refuses and does
    not pretend, so a kernel that cannot be confined says so and does not look
    confined."""
    ctx, session, agent = await mounted_runtime()
    runtime: PythonCodeRuntime = ctx.require(PYTHON_RUNTIME)

    await agent_with_workspace(ctx, session, agent, tmp_path)
    assert runtime.confiner(agent.id) is None, "a workspace with no backend is not confinement"

    ctx.require(SANDBOX).register_provider(StubSandboxProvider())
    assert runtime.confiner(agent.id) is not None, "with both, it confines"
    assert runtime.confiner("an-agent-with-no-workspace") is None


async def test_the_runtime_asks_for_the_seam_when_a_kernel_starts(
    mounted_runtime: MountedRuntime,
) -> None:
    """A backend that is layered *after* this row still confines the kernels that
    the runtime spawns. That is the reason why the seam is a resolver and not a
    value read at mount."""
    ctx, _session, _agent = await mounted_runtime()
    runtime: PythonCodeRuntime = ctx.require(PYTHON_RUNTIME)
    assert runtime.sandbox is not None
    assert runtime.sandbox() is ctx.require(SANDBOX)


async def test_doctor_says_whether_cells_are_confined(mounted_runtime: MountedRuntime) -> None:
    """ "Are my code cells bounded" is a question for the tool to answer. The
    answer is conditional: on a backend, and on the agent having a workspace."""
    ctx, _session, _agent = await mounted_runtime()

    assert "no sandbox backend" in report_section(ctx, "Code runtime")["cells confined by"]

    ctx.require(SANDBOX).register_provider(StubSandboxProvider())
    assert report_section(ctx, "Code runtime")["cells confined by"].startswith(
        "a backend is mounted"
    )


# ------------------------------------------------------ what it enforces --


async def test_a_cell_cannot_write_an_absolute_path_outside_its_workspace(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """**The gate.** The kernel refuses the raw `open()` of a cell on an absolute
    path, and the host file is untouched. That is the one thing that `worktree`
    cannot do (§4.8, E13). It is now true of authored code and not only of
    `tool-bash`.

    The test checks both directions, for the reason that `probe_sandbox` gives. A
    kernel that could write *nothing* would pass the escape half and be useless.
    The write inside the workspace stops that from reading as success.
    """
    ctx, agent, workspace = await _confined(mounted_runtime, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("host", encoding="utf-8")

    escaped = await run_in_kernel(ctx, agent.id, f"open({str(outside)!r}, 'w').write('escaped')")
    landed = await run_in_kernel(ctx, agent.id, "open('landed.txt', 'w').write('agent')")

    assert outside.read_text(encoding="utf-8") == "host", "an absolute write escaped the kernel"
    assert escaped.error is not None, f"the kernel must refuse this: {escaped}"
    assert landed.error is None, f"the workspace must stay writable: {landed}"
    assert (workspace.root / "landed.txt").read_text(encoding="utf-8") == "agent"


async def test_a_confined_kernel_reports_the_backend_that_bounds_it(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """`phern doctor` names the backend. It reads the name from the kernels that
    are actually running, and not from what would be true."""
    ctx, agent, _workspace = await _confined(mounted_runtime, tmp_path)

    assert (await run_in_kernel(ctx, agent.id, "x = 1")).error is None

    confined = report_section(ctx, "Code runtime")["cells confined by"]
    assert confined.startswith(f"{ctx.require(SANDBOX).provider.backend} —"), confined


async def test_a_cell_refused_by_the_kernel_leaves_the_same_record_bash_would(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """The rule of the seam reaches cells: a refusal is a record, not a silence.

    The kernel had adopted the *boundary* and not the rule. A cell that wrote
    outside its workspace got a traceback with no way to act on it, while the
    identical refusal from `tool-bash` got a `sandbox/denied` event that carries
    the `/sandbox allow path` line. The asymmetry was visible inside this change:
    *network* refusals were already recorded, because the proxy is told which
    agent it refuses.
    """
    ctx, agent, _workspace = await _confined(mounted_runtime, tmp_path)
    session = ctx.require(AGENTS).get(agent.id).session
    outside = tmp_path / "nope.txt"

    refused = await run_in_kernel(ctx, agent.id, f"open({str(outside)!r}, 'w').write('x')")
    assert refused.error is not None

    denials = [event for event in session.events if event.type == DENIED]
    assert len(denials) == 1, denials
    assert denials[0].data["kind"] == "filesystem"
    assert denials[0].data["via"] == "output"
    assert denials[0].data["agent"] == agent.id
    assert "/sandbox allow path" in denials[0].data["message"]


async def test_a_cell_that_merely_prints_the_words_is_not_a_refusal(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """The requirement of `report_denial` itself, from this caller: a run that
    succeeded was refused nothing, whatever it printed."""
    ctx, agent, _workspace = await _confined(mounted_runtime, tmp_path)
    session = ctx.require(AGENTS).get(agent.id).session

    ran = await run_in_kernel(ctx, agent.id, "print('Read-only file system')")

    assert ran.error is None
    assert not [event for event in session.events if event.type == DENIED]


@pytest.mark.parametrize("forwards", [True, False])
async def test_the_signal_route_follows_the_backend_not_whether_it_confined(
    tmp_path: Path, forwards: bool
) -> None:
    """`_interrupt` asks `ConfinedArgv.forwards_signals`, which is a declared
    fact. It does not derive the decision from "was I confined at all".

    The two differ, and the difference is a lost capability, not a matter of
    tidiness. `bwrap` puts a wrapper and a PID namespace between the host and the
    guest, so a signal to it destroys the namespace and does not interrupt the
    cell. But `sandbox-exec` execs its target, and the test stub wraps nothing.
    On those, the signal is a second cancel route that is worth keeping.

    The test drives `_interrupt` and does not assert on the flag, because a
    correct value of the flag proves nothing about the branch that reads it.
    """
    signaled: list[int] = []
    # The kernel is built and not started. `_interrupt` needs a process to signal
    # and a confinement result to consult. To start a real guest and then replace
    # its process with a fake would leak the guest that was spawned.
    kernel = Kernel(
        namespace="agent-test",
        environment=resolve_interpreter(cache=tmp_path, mode="host"),
        limits=KernelLimits(),
        journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
    )
    setattr(kernel, "_process", _FakeProcess(signaled))  # noqa: B010
    kernel.confined = ConfinedArgv(
        argv=("python",), enforcement="full", backend="stub", forwards_signals=forwards
    )

    await kernel._interrupt(run_id=1)

    assert bool(signaled) is forwards
    if forwards:
        assert signaled == [signal.SIGINT]


@dataclass
class _FakeProcess:
    """The minimum process that `_interrupt` needs: it is alive, and it records signals."""

    signaled: list[int]
    returncode: None = None

    def send_signal(self, number: int) -> None:
        self.signaled.append(number)


def test_the_backends_declare_whether_a_signal_reaches_what_they_wrap() -> None:
    """The values that the branch above depends on, at their source."""
    policy = SandboxPolicy(mode="workspace-write", workspace_root="/w")
    assert Bubblewrap().confine(("python",), policy).forwards_signals is False
    assert Seatbelt().confine(("python",), policy).forwards_signals is True
    assert StubSandboxProvider().confine(("python",), policy).forwards_signals is True
