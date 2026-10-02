"""Dying with the parent (F3), and why it needs saying at all.

The intuition is that killing a process kills what it started. POSIX does the
opposite: a dead parent's children are re-parented to PID 1 and keep running,
and `atexit` never runs under `SIGKILL`. So a hard-killed pH would otherwise
leave a live CPython holding a model's namespace — one per agent, indefinitely,
with nothing that will ever reconcile them because nobody reopens a session that
died.

This test kills a host the way the OS would and asserts the child is gone. It is
the assertion that the mechanism in `ph_runtime.lifecycle` is actually armed,
which no unit test of that module can show.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

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


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _await(path: Path, *, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} never appeared"
        time.sleep(0.05)


@pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="POSIX re-parenting")
@pytest.mark.parametrize("busy", [False, True], ids=["idle", "in a blocking cell"])
def test_the_runtime_child_does_not_outlive_a_hard_killed_host(tmp_path: Path, busy: bool) -> None:
    """Killed while idle, and killed while a cell blocks the guest's loop.

    The second is the case the socket cannot cover (P12-07): a cell in
    `time.sleep` never yields to the reader that would see the host's EOF, so
    only a mechanism outside the loop — `PR_SET_PDEATHSIG`, or the kqueue
    `NOTE_EXIT` thread on macOS — ends the guest.
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
        assert _alive(child)
        if busy:
            _await(marker)

        # Not `terminate()`: `SIGKILL` is the case where no cleanup code of ours
        # runs at all, on any platform.
        host.send_signal(signal.SIGKILL)
        host.wait(timeout=10)

        deadline = time.monotonic() + 10
        while _alive(child) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _alive(child), f"the runtime child {child} outlived its host"
    finally:
        if host.poll() is None:  # pragma: no cover
            host.kill()
            host.wait()


def test_the_guest_reports_which_mechanism_it_armed() -> None:
    """`boot-ack` carries it, so the log says what was in force on this host.

    Not cosmetic: the mechanisms have genuinely different guarantees. Until
    P12-07, a session on macOS ran under `getppid-poll`, with a one-second window
    where a hard-killed host could leave a stray; a log that names it says so.
    """
    from ph_runtime.lifecycle import die_with_parent

    expected = {"linux": "pdeathsig", "darwin": "kqueue-exit", "win32": "job-object"}
    assert die_with_parent() == expected[sys.platform]
