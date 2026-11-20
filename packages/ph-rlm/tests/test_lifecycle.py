"""Die-with-parent (F3), and why it must be stated at all.

The intuition is that killing a process kills what it started. POSIX does the
opposite: the children of a dead parent are re-parented to PID 1 and continue to
run, and `atexit` never runs under `SIGKILL`. A pH that was killed hard would
therefore leave a live CPython process that holds the namespace of a model. There
would be one for each agent, indefinitely. Nothing would ever reconcile them,
because nobody reopens a session that died.

This test kills a host in the way that the operating system would, and asserts
that the child is gone. That assertion shows that the mechanism in
`ph_runtime.lifecycle` is actually armed. No unit test of that module can show
it.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ph.orphans import process_alive

HOST = """
import anyio, sys
from pathlib import Path
from ph.orphans import OrphanJournal
from ph_rlm.kernel.manager import Kernel, KernelLimits
from ph_rlm.kernel.venv import resolve_interpreter

async def main():
    root = Path({root!r})
    kernel = Kernel(
        namespace="orphan-test",
        environment=resolve_interpreter(cache=root, mode="host"),
        limits=KernelLimits(),
        journal=OrphanJournal(path=root / "processes.jsonl"),
        boot_timeout=60.0,
    )
    await kernel.start([])
    async with anyio.create_task_group() as tasks:
        if {busy!r}:
            # A cell in synchronous Python: the guest's event loop, and with it
            # the socket read that would hear the host's EOF, does not run until
            # it returns. The marker says the cell is past its first line.
            tasks.start_soon(kernel.run, {cell!r}, (), None)
        print(kernel._process.pid, flush=True)
        await anyio.sleep(300)

anyio.run(main)
"""

BLOCKING_CELL = """
import pathlib, time
pathlib.Path({marker!r}).touch()
time.sleep(3600)
"""


def _await(path: Path, *, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


@pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="POSIX re-parenting")
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "in a blocking cell"])
def test_the_runtime_child_does_not_outlive_a_hard_killed_host(tmp_path: Path, busy: bool) -> None:
    """The host is killed while the guest is idle, and while a cell blocks the
    loop of the guest.

    The second case is the one that the socket cannot cover (P12-07). A cell in
    `time.sleep` never yields to the reader that would see the EOF of the host.
    Only a mechanism outside the loop ends the guest: `PR_SET_PDEATHSIG`, or the
    kqueue `NOTE_EXIT` thread on macOS.
    """
    marker = tmp_path / "cell-started"
    cell = textwrap.dedent(BLOCKING_CELL).format(marker=str(marker))
    program = textwrap.dedent(HOST).format(root=str(tmp_path), busy=busy, cell=cell)
    host = subprocess.Popen(
        [sys.executable, "-c", program],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert host.stdout is not None
        line = host.stdout.readline().strip()
        assert line.isdigit(), f"the host did not report a child pid: {host.stderr!r}"
        child = int(line)
        assert process_alive(child)
        if busy:
            _await(marker)

        # Not `terminate()`. `SIGKILL` is the case where none of our cleanup code
        # runs at all, on any platform.
        host.send_signal(signal.SIGKILL)
        host.wait(timeout=10)

        deadline = time.monotonic() + 10
        while process_alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_alive(child), f"the runtime child {child} outlived its host"
    finally:
        if host.poll() is None:  # pragma: no cover
            host.kill()
            host.wait()


def test_the_guest_reports_which_mechanism_it_armed() -> None:
    """`boot-ack` carries the mechanism, so the log says what was in force on
    this host.

    This is not cosmetic. The mechanisms give different guarantees. Until P12-07,
    a session on macOS ran under `getppid-poll`, with a window of one second in
    which a host that was killed hard could leave a stray process. A log that
    names the mechanism says so.
    """
    from ph_runtime.lifecycle import die_with_parent

    expected = {"linux": "pdeathsig", "darwin": "kqueue-exit", "win32": "job-object"}
    assert die_with_parent() == expected[sys.platform]
