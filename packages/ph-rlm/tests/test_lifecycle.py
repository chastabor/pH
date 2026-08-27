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
from ph_rlm.kernel.journal import OrphanJournal
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
    print(kernel._process.pid, flush=True)
    await anyio.sleep(300)

anyio.run(main)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


@pytest.mark.skipif(not sys.platform.startswith(("linux", "darwin")), reason="POSIX re-parenting")
def test_the_runtime_child_does_not_outlive_a_hard_killed_host(tmp_path: Path) -> None:
    host = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(HOST).format(root=str(tmp_path))],
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

    Not cosmetic: the three mechanisms have genuinely different guarantees, and
    a session that ran under `getppid-poll` had a one-second window where a
    hard-killed host could leave a stray. That belongs in the record.
    """
    from ph_runtime.lifecycle import die_with_parent

    assert die_with_parent() in {"pdeathsig", "getppid-poll", "job-object"}
