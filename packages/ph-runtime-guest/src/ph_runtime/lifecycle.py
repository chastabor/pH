"""Dying with the parent, per platform (F3).

The OS does not do what one would hope. A parent's death does **not** kill its
children: POSIX re-parents them to PID 1, and `atexit` never runs under
`SIGKILL` — so a host that is hard-killed leaves a Python process holding the
model's namespace and whatever it was doing. Each platform needs its own
mechanism, and two of the three live in the guest's gift:

* **Linux** — `prctl(PR_SET_PDEATHSIG, SIGKILL)`, set here because it is a
  property of *this* process. It is armed relative to the parent that was
  current when it was set.
* **macOS** — no equivalent exists, so a daemon thread waits on a kqueue
  `EVFILT_PROC` / `NOTE_EXIT` event for the parent and `os._exit`s when it
  arrives (P12-07). It used to check `os.getppid()` once a second; the event
  removes that window, and a thread blocked in `kevent` costs nothing while the
  host lives.
* **Windows** — the host's job: a Job Object with `KILL_ON_JOB_CLOSE`. Nothing
  to do here, and it is the tidiest of the three.

**Why the socket is not enough, and why this is a thread.** The guest's ordinary
"host is gone" signal is the socket's EOF (`Channel.receive` returns `None`), and
that read runs on the guest's event loop. While a cell is in synchronous Python
(a tight loop, a long native call, `time.sleep(3600)`) the reader never runs, so
an orphan would keep executing model code, holding the namespace's memory and
anything the cell opened, and a restarted daemon would start a second guest
beside it. A thread still runs while the cell blocks the loop; `kevent` wakes it
the moment the parent is reaped.

@module ph_runtime.lifecycle
"""

from __future__ import annotations

import ctypes
import os
import select
import signal
import sys
import threading

__all__ = ["die_with_parent"]

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

    The kqueue path is the one place a parent already gone *is* acted on, and only
    because the kernel says so: registering `NOTE_EXIT` on a pid that no longer
    exists fails with `ESRCH`, and an exit that happened before the watch was armed
    would otherwise never be heard.
    """
    if sys.platform.startswith("linux"):
        if _set_pdeathsig():
            return "pdeathsig"
    if sys.platform == "win32":  # pragma: no cover — the host owns the Job Object
        return "job-object"
    _watch_parent()
    return "kqueue-exit"


def _set_pdeathsig() -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        applied: int = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):  # pragma: no cover
        return False
    return applied == 0


def _watch_parent() -> None:
    """Block a daemon thread on the parent's exit event, then end this process.

    The order closes the re-parent race: the parent is read, the event is
    registered against that pid, and the parent is read again. A host that died
    between the first read and the registration is caught by `ESRCH` or by the
    second read showing a different parent; one that dies after is what the event
    is for. `os._exit`, not `sys.exit`, for the reason `Runner._end_runaway`
    gives: the thing being escaped may be a cell that does not yield.
    """
    original = os.getppid()
    kq = select.kqueue()
    exited = select.kevent(
        original,
        filter=select.KQ_FILTER_PROC,
        flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
        fflags=select.KQ_NOTE_EXIT,
    )
    try:
        kq.control([exited], 0, 0)
    except ProcessLookupError:  # pragma: no cover — the host died during the spawn
        os._exit(0)
    if os.getppid() != original:  # pragma: no cover — likewise
        os._exit(0)

    def watch() -> None:  # pragma: no cover — ends the process
        kq.control(None, 1, None)
        os._exit(0)

    threading.Thread(target=watch, name="ph-parent-watch", daemon=True).start()
