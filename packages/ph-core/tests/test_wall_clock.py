"""`ph.wall_clock`: a wait for an epoch instant that counts the time a machine slept.

What is pinned here is everything short of the suspend itself. The wait ends at its
instant and not before. A past instant does not wait. An `Alarm` says whether it
ended a race. The descriptor is closed however the wait ends. A machine with no
timer still wakes on time. And on Linux, the kernel's own account of the timer says
it is on `CLOCK_REALTIME` with an absolute expiry, which is the property that makes
it fire on resume. Closing the lid is a manual check
(`docs/dev-notes/linux-macos-differences.md`).
"""

from __future__ import annotations

import errno
import logging
import os
import sys
from pathlib import Path

import anyio
import pytest

from ph import wall_clock
from ph.cancel import first_of
from ph.session import now_ms
from ph.testing import open_fds, raising, settled
from ph.wall_clock import Alarm, sleep_until

pytestmark = pytest.mark.anyio


def _timerfd_infos() -> list[str]:
    """The kernel's account of each `timerfd` this process holds (Linux): the
    `/proc/self/fdinfo` entries that carry a `clockid`."""
    found: list[str] = []
    for fd in os.listdir("/proc/self/fdinfo"):
        try:
            text = Path(f"/proc/self/fdinfo/{fd}").read_text()
        except OSError:
            continue  # closed between the listing and the read
        if "clockid:" in text:
            found.append(text)
    return found


async def test_the_wait_ends_at_its_instant_and_not_before() -> None:
    at = now_ms() + 150
    with anyio.fail_after(5):
        await sleep_until(at)
    assert now_ms() >= at


async def test_an_instant_already_past_does_not_wait() -> None:
    with anyio.fail_after(0.5):
        await sleep_until(now_ms() - 60_000)


async def test_an_alarm_that_ends_the_race_says_so() -> None:
    alarm = Alarm(now_ms() + 50)
    with anyio.fail_after(5):
        await first_of(anyio.Event(), alarm)
    assert alarm.rang


@pytest.mark.parametrize("ahead", [60_000, None], ids=["later", "never"])
async def test_an_event_that_ends_the_race_leaves_the_alarm_unrung(ahead: int | None) -> None:
    """For an alarm set for later, and for one set for nothing: `None` is "nothing
    planned", which the scheduler sleeps on until told."""
    moved = anyio.Event()
    alarm = Alarm(None if ahead is None else now_ms() + ahead)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(first_of, moved, alarm)
        await anyio.sleep(0.05)
        moved.set()
    assert not alarm.rang


async def test_the_timer_is_closed_whichever_way_the_wait_ends() -> None:
    """A daemon sleeps once per plan for weeks, so a descriptor left per wait is a
    leak that ends at `EMFILE`."""
    before = open_fds()

    with anyio.fail_after(5):
        await sleep_until(now_ms() + 20)
    with anyio.move_on_after(0.05):
        await sleep_until(now_ms() + 60_000)

    assert open_fds() == before


async def test_without_a_timer_the_wait_still_ends_on_time_and_says_why(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An `OSError` out of the platform call must not reach the caller: for the
    scheduler that is the daemon's task group."""
    monkeypatch.setattr(
        wall_clock, "_wall_wait", raising(OSError(errno.EMFILE, "Too many open files"))
    )
    at = now_ms() + 50
    with caplog.at_level(logging.WARNING, logger="ph.wall_clock"), anyio.fail_after(5):
        await sleep_until(at)

    assert now_ms() >= at
    assert "monotonic clock" in caplog.text


@pytest.mark.skipif(sys.platform != "linux", reason="reads the kernel's timerfd fdinfo")
async def test_on_linux_the_timer_is_on_the_wall_clock_and_absolute() -> None:
    """The property suspend depends on, read from the kernel rather than the code.

    `clockid: 0` is `CLOCK_REALTIME`, which keeps counting while the machine is
    suspended, unlike `CLOCK_MONOTONIC` (1). `settime flags: 01` is
    `TFD_TIMER_ABSTIME`, so the expiry is a moment rather than a delay measured from
    when it was armed.
    """
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(sleep_until, now_ms() + 60_000)
        await settled(_timerfd_infos, "the wall-clock timer to be armed")
        found = _timerfd_infos()
        tasks.cancel_scope.cancel()

    assert len(found) == 1, found
    assert "clockid: 0\n" in found[0]
    assert "settime flags: 01\n" in found[0]
