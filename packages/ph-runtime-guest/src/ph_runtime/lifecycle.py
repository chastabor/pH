"""Dying with the parent, per platform (F3).

The OS does not do what one would hope. A parent's death does **not** kill its
children: POSIX re-parents them to PID 1, and `atexit` never runs under
`SIGKILL` — so a host that is hard-killed leaves a Python process holding the
model's namespace and whatever it was doing. Each platform needs its own
mechanism, and only one of the three lives in the guest's gift:

* **Linux** — `prctl(PR_SET_PDEATHSIG, SIGKILL)`, set here because it is a
  property of *this* process. It is armed relative to the parent that was
  current when it was set.
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
    """Arrange to not outlive the host. Returns the mechanism that took effect.

    **There is deliberately no "am I already an orphan" check here**, and removing
    one is what let the guest run confined at all.

    It read `os.getppid() == 1` and `os._exit(0)`, guarding the window between the
    fork and `prctl`: a host that died in it would never send the signal. The guard
    was unnecessary and, under a PID namespace, always wrong.

    *Unnecessary*, because of where this is called from. `_serve` reads the boot
    frame **before** calling this, and returns on `None` — so reaching this line
    means a frame was just read from the host, which is proof it was alive after
    the spawn. A host that dies after that closes the socket, `Channel.receive`
    returns `None`, and `serve` returns; that is the same mechanism relied on for
    every later death, and `channel.send`'s own comment already points at it.

    *Wrong*, because inside `bwrap --unshare-pid` this process's parent **is** PID
    1 — the sandbox's init — which is the healthy arrangement rather than evidence
    of an orphan. So the check fired on every confined start and the guest exited
    silently, with no stderr, before sending `boot-ack`; the host could only report
    that the runtime "exited before reporting ready". Dying with the host is the
    sandbox's job there and it is already arranged: `bwrap` holds
    `--die-with-parent` against the host, and the kernel tears down every process
    in the namespace when its init goes.
    """
    if sys.platform.startswith("linux"):
        if _set_pdeathsig():
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
