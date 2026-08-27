"""Dying with the parent, per platform (F3).

The OS does not do what one would hope. A parent's death does **not** kill its
children: POSIX re-parents them to PID 1, and `atexit` never runs under
`SIGKILL` — so a host that is hard-killed leaves a Python process holding the
model's namespace and whatever it was doing. Each platform needs its own
mechanism, and only one of the three lives in the guest's gift:

* **Linux** — `prctl(PR_SET_PDEATHSIG, SIGKILL)`, set here because it is a
  property of *this* process. It is armed relative to the parent that was
  current when it was set, so the check right after it is not paranoia: if the
  host died during spawn, the signal would never come.
* **macOS** — no equivalent exists, so a daemon thread watches `os.getppid()`
  and `os._exit`s when it changes. This is what prime-agent's fork-server does,
  for this reason.
* **Windows** — the host's job: a Job Object with `KILL_ON_JOB_CLOSE`. Nothing
  to do here, and it is the tidiest of the three.

@module ph_runtime.lifecycle
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import threading
import time

__all__ = ["POLL_SECONDS", "die_with_parent"]

POLL_SECONDS = 1.0
_PR_SET_PDEATHSIG = 1


def die_with_parent() -> str:
    """Arrange to not outlive the host. Returns the mechanism that took effect."""
    if sys.platform.startswith("linux"):
        if _set_pdeathsig():
            # Armed against the parent as it was a moment ago. If the host died
            # in between, no signal is coming and this is the only chance to
            # notice.
            if os.getppid() == 1:
                os._exit(0)
            return "pdeathsig"
    if sys.platform == "win32":  # pragma: no cover — the host owns the Job Object
        return "job-object"
    _watch_parent()
    return "getppid-poll"


def _set_pdeathsig() -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        applied: int = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):  # pragma: no cover
        return False
    return applied == 0


def _watch_parent() -> None:
    original = os.getppid()

    def watch() -> None:  # pragma: no cover — timing-dependent
        while True:
            time.sleep(POLL_SECONDS)
            if os.getppid() != original:
                os._exit(0)

    threading.Thread(target=watch, name="ph-parent-watch", daemon=True).start()
