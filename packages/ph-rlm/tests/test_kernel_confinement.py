"""The RLM kernel, bounded at the kernel (P6-39).

`cwd` puts a cell in the agent's tree, so a *relative* write lands there. Only
confinement refuses `open("/etc/passwd", "w")` — §4.8's own line about what no
tier below `sandbox` can do — and until this row the kernel was spawned
unconfined, so `permissions-fs` was telling operators that "a sandbox provider
bounds what a code cell can reach directly" while nothing did.

**Two properties carry the whole design, and both are measured here rather than
argued.** fd 3 has to survive the wrapper, or the kernel cannot talk to its host at
all; and an absolute-path write has to actually fail. The first is checked against
the real `bwrap` argv with a live socket at the far end, because it is the one that
would fail *silently at boot* on a host nobody tested — the kernel would hang until
`boot_timeout` and be reported as a dead runtime rather than as a lost channel.

The enforcement half skips where the kernel cannot enforce — no `bwrap`, or no
AppArmor profile on Ubuntu 23.10+; no `sandbox-exec` — through the same in-body
check `test_sandbox_local.py` uses, which asks whether the row actually registered
a provider rather than whether the binary exists. `needs_backend` is only on the
test that drives the platform's backend directly without mounting the row, where
there is no provider to ask. Verified against both: `bwrap` on 2026-09-01, Seatbelt
on 2026-09-07 — the latter only after the macOS boot hang `test_boot_report.py`
pins was found, because a kernel that never reports ready confines nothing.
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
from rlm_fixtures import MountedRuntime

from ph.keys import AGENTS, CODE_RUNTIME, SANDBOX, WORKSPACE
from ph.seams.code_runtime import CodeRunRequest
from ph.seams.sandbox import DENIED, ConfinedArgv, SandboxPolicy
from ph.seams.sandbox_local import Bubblewrap, Seatbelt, local_backend
from ph.seams.workspace import workspace_policy
from ph.testing import StubSandboxProvider, report_section
from ph_rlm.kernel.journal import OrphanJournal
from ph_rlm.kernel.manager import Kernel, KernelLimits, PythonCodeRuntime
from ph_rlm.kernel.venv import resolve_interpreter
from ph_rlm.keys import PYTHON_RUNTIME
from ph_runtime.protocol import FD_ENV

pytestmark = pytest.mark.anyio

SANDBOX_ROW: dict[str, Any] = {"id": "sandbox-local", "disabled": False}

needs_backend = pytest.mark.skipif(
    local_backend()[0] is None, reason="no local confinement backend on this host"
)


# ------------------------------------------------------- fd 3 crosses it --


@needs_backend
def test_the_framed_channel_survives_the_wrapper(tmp_path: Path) -> None:
    """**The property the whole design rests on.** `bwrap` passes an inherited
    descriptor through to what it execs, and so does `sandbox-exec` (measured on
    both), so the kernel's fd 3 reaches the guest.

    Against the real argv this platform's backend builds, with a live socket at the far end,
    because a wrapper that closed the descriptor would fail at *boot*: the guest
    would find nothing on fd 3, the host would wait out `boot_timeout`, and the
    report would say the runtime died rather than that the channel never crossed.
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


async def _agent_with_workspace(
    ctx: Any,  # noqa: ANN401
    session: Any,  # noqa: ANN401
    agent: Any,  # noqa: ANN401
    base: Path,
) -> Any:  # noqa: ANN401
    """The lifecycle row acquires at an agent's first step; these tests never take
    one, so the workspace is acquired the way the ladder tests do.

    The base is created because it is where the kernel will `chdir`: the shared
    provider hands back the directory it was given without making it, and a
    missing cwd fails the spawn rather than the confinement.
    """
    base.mkdir(parents=True, exist_ok=True)
    return await ctx.require(WORKSPACE).acquire(session_id=session.id, agent_id=agent.id, base=base)


async def _confined(mounted_runtime: MountedRuntime, tmp_path: Path) -> tuple[Any, Any, Any]:
    """A mounted runtime whose kernels will really be confined, plus the agent and
    its workspace — or a skip that says why not.

    `test_sandbox_local._enforcing`'s shape: the question is whether the row
    *registered* a provider, which is what a probe against this host decided, not
    whether a binary is on PATH.
    """
    ctx, session, agent = await mounted_runtime(extra_rows=[SANDBOX_ROW])
    if ctx.require(SANDBOX).provider is None:
        pytest.skip("no enforcing sandbox backend on this host")
    workspace = await _agent_with_workspace(ctx, session, agent, tmp_path / "project")
    return ctx, agent, workspace


async def test_a_kernel_is_confined_against_its_own_agents_workspace(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """The writable set is the agent's own workspace and scratch — `ctx.shell`'s
    policy, from the same `workspace_policy`, so a deployment cannot end up with
    `tool-bash` confined and `run_code` not."""
    ctx, session, agent = await mounted_runtime()
    ctx.require(SANDBOX).register_provider(StubSandboxProvider())
    workspace = await _agent_with_workspace(ctx, session, agent, tmp_path)
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
    """`None`, never an unwrapped argv: the seam refuses rather than pretending, so
    a kernel that cannot be bounded says so instead of looking bounded."""
    ctx, session, agent = await mounted_runtime()
    runtime: PythonCodeRuntime = ctx.require(PYTHON_RUNTIME)

    await _agent_with_workspace(ctx, session, agent, tmp_path)
    assert runtime.confiner(agent.id) is None, "a workspace with no backend is not confinement"

    ctx.require(SANDBOX).register_provider(StubSandboxProvider())
    assert runtime.confiner(agent.id) is not None, "with both, it confines"
    assert runtime.confiner("an-agent-with-no-workspace") is None


async def test_the_runtime_asks_for_the_seam_when_a_kernel_starts(
    mounted_runtime: MountedRuntime,
) -> None:
    """A backend layered *after* this row still bounds the kernels it spawns —
    the reason the seam is a resolver rather than a value read at mount."""
    ctx, _session, _agent = await mounted_runtime()
    runtime: PythonCodeRuntime = ctx.require(PYTHON_RUNTIME)
    assert runtime.sandbox is not None
    assert runtime.sandbox() is ctx.require(SANDBOX)


async def test_doctor_says_whether_cells_are_confined(mounted_runtime: MountedRuntime) -> None:
    """ "Are my code cells bounded" is a question asked of the tool, and the answer
    is conditional — on a backend, and on the agent having a workspace."""
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
    """**The gate.** A cell's raw `open()` on an absolute path is refused by the
    kernel and the host file is untouched — the one thing `worktree` cannot do
    (§4.8, E13), now true of authored code and not only of `tool-bash`.

    Both directions in one test, for `probe_sandbox`'s reason: a kernel that could
    write *nothing* would pass the escape half while being useless, and the write
    inside the workspace is what stops that reading as success.
    """
    ctx, agent, workspace = await _confined(mounted_runtime, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("host", encoding="utf-8")

    escaped = await _run_cell(ctx, agent.id, f"open({str(outside)!r}, 'w').write('escaped')")
    landed = await _run_cell(ctx, agent.id, "open('landed.txt', 'w').write('agent')")

    assert outside.read_text(encoding="utf-8") == "host", "an absolute write escaped the kernel"
    assert escaped.error is not None, f"the kernel must refuse this: {escaped}"
    assert landed.error is None, f"the workspace must stay writable: {landed}"
    assert (workspace.root / "landed.txt").read_text(encoding="utf-8") == "agent"


async def test_a_confined_kernel_reports_the_backend_that_bounds_it(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """`ph doctor` names it, read from the kernels actually running rather than
    from what would be true."""
    ctx, agent, _workspace = await _confined(mounted_runtime, tmp_path)

    assert (await _run_cell(ctx, agent.id, "x = 1")).error is None

    confined = report_section(ctx, "Code runtime")["cells confined by"]
    assert confined.startswith(f"{ctx.require(SANDBOX).provider.backend} —"), confined


async def _run_cell(ctx: Any, agent_id: str, program: str) -> Any:  # noqa: ANN401
    """One cell in this agent's own namespace — which *is* the agent id, so the
    kernel it reaches is the one confined against that agent's workspace."""
    return await ctx.require(CODE_RUNTIME).run(CodeRunRequest(program=program, namespace=agent_id))


async def test_a_cell_refused_by_the_kernel_leaves_the_same_record_bash_would(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """The seam's rule — a refusal is a record, not a silence — reaches cells.

    The kernel had adopted the *boundary* and not the rule: a cell writing outside
    its workspace got a traceback with no way to act on it, while the identical
    refusal from `tool-bash` got a `sandbox/denied` event carrying the
    `/sandbox allow path` line. The asymmetry was visible inside this change —
    *network* refusals were already recorded, because the proxy is told whose agent
    it is refusing.
    """
    ctx, agent, _workspace = await _confined(mounted_runtime, tmp_path)
    session = ctx.require(AGENTS).get(agent.id).session
    outside = tmp_path / "nope.txt"

    refused = await _run_cell(ctx, agent.id, f"open({str(outside)!r}, 'w').write('x')")
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
    """`report_denial`'s own requirement, from this caller: a run that succeeded was
    refused nothing, whatever it printed."""
    ctx, agent, _workspace = await _confined(mounted_runtime, tmp_path)
    session = ctx.require(AGENTS).get(agent.id).session

    ran = await _run_cell(ctx, agent.id, "print('Read-only file system')")

    assert ran.error is None
    assert not [event for event in session.events if event.type == DENIED]


@pytest.mark.parametrize("forwards", [True, False])
async def test_the_signal_route_follows_the_backend_not_whether_it_confined(
    tmp_path: Path, forwards: bool
) -> None:
    """`_interrupt` asks `ConfinedArgv.forwards_signals`, a declared fact, rather
    than deriving the decision from "was I confined at all".

    The two differ, and the difference is a lost capability rather than a tidiness
    point: `bwrap` puts a wrapper and a PID namespace in the way, so signalling it
    destroys the namespace instead of interrupting the cell — but `sandbox-exec`
    execs its target and the test stub wraps nothing, and on those the signal is a
    second cancel route worth keeping.

    Driven through `_interrupt` rather than asserted on the flag, because the flag
    having the right value proves nothing about the branch that reads it.
    """
    signalled: list[int] = []
    # Built rather than started: `_interrupt` needs a process to signal and a
    # confinement result to consult, and starting a real guest to replace its
    # process with a fake would leak the one it spawned.
    kernel = Kernel(
        namespace="agent-test",
        environment=resolve_interpreter(cache=tmp_path, mode="host"),
        limits=KernelLimits(),
        journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
    )
    setattr(kernel, "_process", _FakeProcess(signalled))  # noqa: B010
    kernel.confined = ConfinedArgv(
        argv=("python",), enforcement="full", backend="stub", forwards_signals=forwards
    )

    await kernel._interrupt(run_id=1)

    assert bool(signalled) is forwards
    if forwards:
        assert signalled == [signal.SIGINT]


@dataclass
class _FakeProcess:
    """Just enough process for `_interrupt`: it is alive, and it records signals."""

    signalled: list[int]
    returncode: None = None

    def send_signal(self, number: int) -> None:
        self.signalled.append(number)


def test_the_backends_declare_whether_a_signal_reaches_what_they_wrap() -> None:
    """The values the branch above depends on, at their source."""
    policy = SandboxPolicy(mode="workspace-write", workspace_root="/w")
    assert Bubblewrap().confine(("python",), policy).forwards_signals is False
    assert Seatbelt().confine(("python",), policy).forwards_signals is True
    assert StubSandboxProvider().confine(("python",), policy).forwards_signals is True
