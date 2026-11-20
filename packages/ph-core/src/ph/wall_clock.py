"""Wait for a moment on the wall clock, and count the time that the machine slept.

`anyio.sleep` and every `CancelScope` deadline run on `loop.time()`, which is
`time.monotonic()`: `CLOCK_MONOTONIC` on Linux, and `mach_absolute_time` on macOS
under Python 3.12. **Neither clock counts the time spent suspended.** That is
correct for a grace period inside a running cell, and wrong for an appointment.
Take a scheduler that goes to sleep twenty hours before a daily run, on a laptop
that is then closed for eight hours. It wakes eight hours after the run was due.
The five-second tick that this module replaced limited that delay to five
seconds after the machine woke. A sleep until the next appointment has no such
limit unless its timer is on the wall clock.

A wait for an epoch instant is therefore armed on the wall clock, and the kernel
fires it on resume when its moment has passed:

* **Linux**: a `timerfd` on `CLOCK_REALTIME` with `TFD_TIMER_ABSTIME`, through
  ctypes, because `os.timerfd_create` arrived in 3.13 and the floor is 3.12. An
  absolute `CLOCK_REALTIME` timer also follows a stepped clock (NTP, or a person
  who sets the time) without help, so `TFD_TIMER_CANCEL_ON_SET` is not needed.
  The thing that is waited for is an instant, not an interval.
* **macOS**: a kqueue `EVFILT_TIMER` with `NOTE_ABSOLUTE` and
  `NOTE_MACH_CONTINUOUS_TIME`, which `<sys/event.h>` documents as continuing "to
  tick across sleep, still uses gettimeofday epoch".
* **Any other platform, or when the timer cannot be made**: `anyio.sleep` for
  the time that remains. That is the monotonic behavior with the suspend gap,
  and it is logged. It is still one wait for one deadline, and never a cadence.

Each descriptor is awaited through the free function `anyio.wait_readable`. That
function removes its reader synchronously and is safe to cancel (see
`ph_rlm.kernel.manager.Kernel._pump` on issue 58).

**The suite does not exercise suspend itself**, because no test can close the
lid. `docs/dev-notes/linux-macos-differences.md` says how to check it by hand.

@module ph.wall_clock
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import select
import sys
from contextlib import suppress
from dataclasses import dataclass

import anyio
import anyio.lowlevel

from .libc import LIBC, failed
from .session import now_ms

__all__ = ["Alarm", "sleep_until"]

log = logging.getLogger("ph.wall_clock")


@dataclass(slots=True)
class Alarm:
    """A wall-clock instant to race against events with `ph.cancel.first_of`.

    After the race, `rang` says whether the instant ended it.
    `move_on_after(...).cancelled_caught` used to answer that question. A caller
    needs the answer to tell a wake by the clock from a wake by an event.
    """

    at: int | None
    """Epoch milliseconds, or `None` for an alarm that never rings."""
    rang: bool = False

    async def wait(self) -> None:
        if self.at is None:
            await anyio.sleep_forever()
        else:
            await sleep_until(self.at)
            self.rang = True


async def sleep_until(at: int) -> None:
    """Return when the wall clock reads `at` (epoch ms). Time suspended counts.

    A moment that is already past returns immediately, after a checkpoint, so a
    caller that planned late still yields to the loop.
    """
    if at <= now_ms():
        await anyio.lowlevel.checkpoint()
        return
    try:
        await _wall_wait(at)
    except OSError as error:
        # The process is out of descriptors, or a platform call was refused (a
        # seccomp profile can refuse `timerfd_create`), or the platform has no
        # timer at all. To raise would end the loop of the caller, which for the
        # scheduler is the task group of the daemon.
        remaining = max(0.0, (at - now_ms()) / 1000)
        log.warning(
            "ph.wall_clock: no wall-clock timer (%s); sleeping on the monotonic clock, "
            "so a suspend in the next %.0f s delays this wake",
            error,
            remaining,
        )
        await anyio.sleep(remaining)


if sys.platform == "linux":
    _CLOCK_REALTIME = 0
    _TFD_TIMER_ABSTIME = 1

    class _Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    class _Itimerspec(ctypes.Structure):
        _fields_ = [("it_interval", _Timespec), ("it_value", _Timespec)]

    LIBC.timerfd_create.argtypes = [ctypes.c_int, ctypes.c_int]
    LIBC.timerfd_create.restype = ctypes.c_int
    LIBC.timerfd_settime.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_Itimerspec),
        ctypes.POINTER(_Itimerspec),
    ]
    LIBC.timerfd_settime.restype = ctypes.c_int

    async def _wall_wait(at: int) -> None:
        # `TFD_NONBLOCK` and `TFD_CLOEXEC` are defined as `O_NONBLOCK` and
        # `O_CLOEXEC`, so the `os` constants are right on every architecture.
        fd: int = LIBC.timerfd_create(_CLOCK_REALTIME, os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise failed("timerfd_create")
        try:
            # A zero `it_value` disarms the timer, and a moment at the epoch would
            # be zero. `sleep_until` returns early for any moment that is past, so
            # the value is never zero.
            when = _Itimerspec(_Timespec(0, 0), _Timespec(at // 1000, (at % 1000) * 1_000_000))
            if LIBC.timerfd_settime(fd, _TFD_TIMER_ABSTIME, ctypes.byref(when), None) != 0:
                raise failed("timerfd_settime")
            while True:
                await anyio.wait_readable(fd)
                # This reads the expiry count. Nothing needs the count, but a
                # successful read shows that the timer fired and that the loop
                # did not wake spuriously.
                with suppress(BlockingIOError):
                    os.read(fd, 8)
                    return
        finally:
            os.close(fd)

elif sys.platform == "darwin":
    # From `<sys/event.h>`. `select` exports the filter but none of these flags.
    _NOTE_USECONDS = 0x00000002
    _NOTE_ABSOLUTE = 0x00000008
    _NOTE_MACH_CONTINUOUS_TIME = 0x00000080

    async def _wall_wait(at: int) -> None:
        queue = select.kqueue()
        try:
            alarm = select.kevent(
                0,
                filter=select.KQ_FILTER_TIMER,
                flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                fflags=_NOTE_ABSOLUTE | _NOTE_USECONDS | _NOTE_MACH_CONTINUOUS_TIME,
                data=at * 1000,
            )
            queue.control([alarm], 0, 0)
            while True:
                await anyio.wait_readable(queue.fileno())
                if queue.control(None, 1, 0):
                    return
        finally:
            queue.close()

else:

    async def _wall_wait(at: int) -> None:
        # This goes to the one fallback in `sleep_until`, which logs the error and
        # sleeps on the monotonic clock.
        raise OSError(errno.ENOSYS, f"no wall-clock timer on {sys.platform}")
