"""End the guest when its parent ends, on each platform (F3).

The operating system does not do this by default. The death of a parent does
**not** kill its children: POSIX re-parents them to PID 1, and `atexit` never
runs under `SIGKILL`. A host that is killed hard therefore leaves a Python
process that holds the namespace of the model and continues its work. Each
platform needs its own mechanism, and the guest can provide two of the three:

* **Linux**: `prctl(PR_SET_PDEATHSIG, SIGKILL)`. It is set here because it is a
  property of *this* process. It applies to the parent that was current when it
  was set.
* **macOS**: no equivalent exists. A daemon thread waits on a kqueue
  `EVFILT_PROC` / `NOTE_EXIT` event for the parent, and calls `os._exit` when the
  event arrives (P12-07). An earlier version checked `os.getppid()` one time each
  second. The event removes that delay, and a thread that is blocked in `kevent`
  costs nothing while the host lives.
* **Windows**: this is the job of the host, with a Job Object that has
  `KILL_ON_JOB_CLOSE`. The guest has nothing to do, and this is the cleanest of
  the three. The host does not create a Job Object yet, so on Windows its orphan
  journal is the only protection.

**Why the socket is not sufficient, and why this is a thread.** The ordinary
"host is gone" signal for the guest is the EOF of the socket (`Channel.receive`
returns `None`). That read runs on the event loop of the guest. While a cell is
in synchronous Python (a tight loop, a long native call, `time.sleep(3600)`), the
reader never runs. An orphan would then continue to execute model code. It would
hold the memory of the namespace and everything that the cell opened, and a
restarted daemon would start a second guest beside it. A thread still runs while
the cell blocks the loop, and `kevent` wakes the thread at the moment the parent
is reaped.

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
    """Arrange for this process not to outlive the host. Returns the mechanism
    that took effect.

    **This function deliberately has no "am I already an orphan" check.** The
    removal of that check is what let the guest run confined at all.

    The old check read `os.getppid() == 1` and called `os._exit(0)`. It guarded
    the time between the fork and `prctl`: a host that died in that time would
    never send the signal. The guard was unnecessary, and under a PID namespace
    it was always wrong.

    It was *unnecessary* because of where this function is called. `_serve` reads
    the boot frame **before** it calls this function, and returns on `None`. When
    execution reaches this function, a frame was just read from the host, which
    proves that the host was alive after the spawn. A host that dies after that
    closes the socket, `Channel.receive` returns `None`, and `serve` returns.
    Every later death depends on that same mechanism, and the comment in
    `channel.send` already refers to it.

    It was *wrong* because inside `bwrap --unshare-pid` the parent of this
    process **is** PID 1, the init of the sandbox. That is the healthy
    arrangement, not a sign of an orphan. The check therefore fired on every
    confined start. The guest exited silently, with no stderr, before it sent
    `boot-ack`, and the host could only report that the runtime "exited before
    reporting ready". In that case the sandbox is responsible for ending the
    guest with the host, and that is already arranged: `bwrap` holds
    `--die-with-parent` against the host, and the kernel ends every process in
    the namespace when its init ends.

    The kqueue path is the one place that does act on a parent that is already
    gone, and only because the kernel reports it. Registering `NOTE_EXIT` on a
    pid that no longer exists fails with `ESRCH`. Without that check, an exit
    that happened before the watch was armed would never be detected.
    """
    if sys.platform.startswith("linux"):
        if _set_pdeathsig():
            return "pdeathsig"
    if sys.platform == "win32":  # pragma: no cover — the host owns the Job Object
        return "job-object"
    if sys.platform == "darwin":
        _watch_parent()
        return "kqueue-exit"
    # A Linux system where `prctl` could not be reached (no `libc.so.6`). It has
    # no kqueue to fall back on. When confined, `bwrap --die-with-parent` still
    # applies.
    return "none"  # pragma: no cover


def _set_pdeathsig() -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        applied: int = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):  # pragma: no cover
        return False
    return applied == 0


if sys.platform == "darwin":

    def _watch_parent() -> None:
        """Block a daemon thread on the exit event of the parent, then end this
        process.

        The order closes the re-parent race. The parent pid is read, the event is
        registered against that pid, and the parent pid is read again. A host
        that died between the first read and the registration is caught by
        `ESRCH`, or by a second read that shows a different parent. A host that
        dies later is what the event is for. The exit uses `os._exit`, not
        `sys.exit`, for the reason that `Runner._end_runaway` gives: the code
        being escaped may be a cell that does not yield.
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
