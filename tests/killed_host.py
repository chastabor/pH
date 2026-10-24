"""Run a host process until it says its work is in flight, then `SIGKILL` it.

The restart gates' one boundary: a host script is its own process, so the kill is
the case where no teardown of ours runs at all — the case an in-process "restart"
cannot reach. Each host prints the ids its test needs as one JSON line once
everything is on disk, and then waits to be killed. `mass_restart_host.py` and
`daemon_restart_host.py` are the two.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["killed_when_ready"]


def killed_when_ready(host: Path, *args: str) -> dict[str, Any]:
    """Run `host` until it prints its line, `SIGKILL` it, and return what it said.

    The host inherits this process's environment, so it reads the test's `$PH_HOME`
    and `$PH_RUNTIME`. A host that exits without a line fails the test with its stderr,
    which goes to a file rather than a pipe: a daemon's log is long enough to fill
    one, and a host blocked writing it never prints its line.
    """
    with tempfile.TemporaryFile("w+") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(host), *args],
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        try:
            assert process.stdout is not None
            line = process.stdout.readline()
            if not line:
                process.wait(timeout=10)
                stderr.seek(0)
                raise AssertionError(f"{host.name} never got its work in flight:\n{stderr.read()}")
            ids: dict[str, Any] = json.loads(line)
            process.send_signal(signal.SIGKILL)
            process.wait(timeout=10)
            return ids
        finally:
            if process.poll() is None:  # pragma: no cover
                process.kill()
                process.wait()
