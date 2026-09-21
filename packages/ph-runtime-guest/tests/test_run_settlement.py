"""Every run sends exactly one `done`, including the ones that never started.

The host has no wall clock on a run. It writes `run` and then waits for `done`,
so a run that never sends one wedges the kernel until a person cancels it —
which is why `_execute` reports through a `finally`, and why it carries a second
catch around the settle itself for the ways settling can fail.

A run cancelled *before its first line* runs none of that. `create_task`
schedules a coroutine; it does not start one. So `run` and `cancel` arriving in
the same read chunk — which is what a host cancelling the moment it issues does,
because both frames are written back to back and the guest reads both before
yielding — cancels a task whose body never entered its own `try`.

Driven against `Runner` directly rather than through the kernel: the kernel's
stop ladder sleeps `POLL_SECONDS` before it looks at the token, so by the time
its `cancel` goes out the guest has long since started the task. The window is a
guest-level one and this is where it can be opened deliberately.
"""

from __future__ import annotations

import asyncio
import contextlib
import resource
from typing import Any

import pytest

from ph_runtime.runner import Runner

pytestmark = pytest.mark.anyio


class _Frames:
    """A channel that hands over a scripted list and records what comes back."""

    def __init__(self, inbound: list[dict[str, Any]]) -> None:
        self._inbound = list(inbound)
        self.sent: list[dict[str, Any]] = []

    async def receive(self) -> dict[str, Any] | None:
        return self._inbound.pop(0) if self._inbound else None

    def send(self, frame: dict[str, Any]) -> None:
        self.sent.append(frame)

    async def drain(self) -> None:
        return None


def _runner(inbound: list[dict[str, Any]]) -> tuple[Runner, _Frames]:
    channel = _Frames(inbound)
    boot = {
        "namespaceId": "t",
        "maxLogBytes": 4096,
        "maxValueBytes": 4096,
        "maxSnapshotBytes": 4096,
        "cpuSeconds": 60,
        "idleCpuSeconds": 2,
    }
    return Runner(channel, boot), channel  # type: ignore[arg-type]


async def test_a_run_canceled_in_the_same_chunk_still_sends_done() -> None:
    """F7 — the task is cancelled before it has run a line, so nothing settles it.

    Both frames are queued before `serve` reads either, which is exactly what one
    read chunk looks like from in here: `_begin` creates the task and the loop
    comes straight back for `cancel` without yielding to it. `serve` then returns
    when the script runs out, which is the host going away.
    """
    runner, channel = _runner(
        [
            {"type": "run", "id": 7, "program": "import asyncio\nawait asyncio.sleep(60)"},
            {"type": "cancel"},
        ]
    )

    await asyncio.wait_for(runner.serve(), timeout=5)

    done = [frame for frame in channel.sent if frame.get("type") == "done"]
    assert done, "the run was cancelled without ever settling"
    assert done[0]["id"] == 7
    assert done[0]["error"]["kind"] == "aborted"


async def test_a_run_that_settled_is_not_settled_twice() -> None:
    """The other side of the same bookkeeping.

    `_owed` is what lets `_abort_run` answer for a run that never started, so the
    ordinary settle has to clear it — otherwise a `cancel` arriving after a run
    finished would send a *second* terminal frame for it, and the host pairs
    `done` to a run by id.

    The run is allowed to finish first, which is what distinguishes this from the
    test above: there the cancel lands in the same chunk, here it lands after.
    """
    runner, channel = _runner([{"type": "run", "id": 3, "program": "1 + 1"}])

    await asyncio.wait_for(runner.serve(), timeout=5)
    assert runner._run is not None
    await asyncio.wait_for(runner._run, timeout=5)

    # A late cancel, once there is nothing left to cancel.
    await runner._abort_run()

    done = [frame for frame in channel.sent if frame.get("type") == "done"]
    assert len(done) == 1, f"{len(done)} terminal frames for one run"
    assert done[0]["id"] == 3
    assert done[0].get("error") is None
    assert done[0]["value"] == 2


async def test_a_run_that_dies_above_its_own_guards_still_sends_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hole `_abort_run` could not reach, and the reason this hangs off the task.

    `_execute` reports through a `finally` and carries a second catch around the
    settle, but both live *inside* its own `try`. Anything raised before that —
    `arm_cpu_budget`, building the capped streams, `_RUN.set` — leaves the task
    `done()` with `_owed` still set, and the old check was in `_abort_run` behind
    an early `if run is None or run.done(): return`. So the one class of failure
    the bookkeeping could not answer for was a task that finished *without*
    settling: the kernel then waits for a frame nobody will send.

    **The vehicle is `_CappedStream`, one of the three this docstring names.**
    It was `arm_cpu_budget`, which is now gated on `Runner._catches_budget` and
    so does not run in a process that installed no `SIGXCPU` handler — which is
    this one, and is the point of that gate: arming a process-wide `RLIMIT_CPU`
    here capped pytest itself. Any of the three proves the same property; this
    one is reachable in-process.

    Sabotage: drop `task.add_done_callback(self._answer_if_owed)` from `_begin`
    and no terminal frame is sent at all.
    """

    def exploding(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the capped stream could not be built")

    monkeypatch.setattr("ph_runtime.runner._CappedStream", exploding)
    runner, channel = _runner([{"type": "run", "id": 11, "program": "1 + 1"}])

    await asyncio.wait_for(runner.serve(), timeout=5)
    assert runner._run is not None
    with contextlib.suppress(RuntimeError):
        await asyncio.wait_for(runner._run, timeout=5)
    # The callback is scheduled with `call_soon`, so it lands on the next tick.
    await asyncio.sleep(0)

    done = [frame for frame in channel.sent if frame.get("type") == "done"]
    assert len(done) == 1, "a run ended without settling and the host was never told"
    assert done[0]["id"] == 11
    assert done[0]["error"]["kind"] == "RuntimeError"
    assert "could not be built" in done[0]["error"]["message"]


async def test_a_stale_callback_cannot_settle_the_run_that_followed_it() -> None:
    """The callback answers for its own run, not for whatever is owed now.

    `add_done_callback` fires through `call_soon`, and `_begin`'s guard is
    `self._run.done()` — already true for a task whose callback has not fired.
    So a run that ended without settling could be followed by a new `run` frame,
    and the stale callback would then send `done` for the run that had *just
    started*: the host marks it finished before its first statement, and nothing
    can settle it afterwards.

    Sabotage: have `_answer_if_owed` read `self._owed` instead of comparing it
    to the run it was created for, and the second run is settled here.
    """
    runner, channel = _runner([])

    async def nothing() -> None:
        return None

    finished = asyncio.get_running_loop().create_task(nothing())
    await finished
    # Run 1 ended without settling; run 2 has since been accepted.
    runner._owed = 2

    runner._answer_if_owed(1, finished)

    assert channel.sent == [], "a finished run settled its successor"
    assert runner._owed == 2, "and cleared what that successor still owes"


def test_a_runner_in_this_process_never_arms_a_limit_it_cannot_catch() -> None:
    """`RLIMIT_CPU` is process-wide, and this process is pytest.

    `arm_cpu_budget` sets a soft limit on *the process*, and `SIGXCPU`'s default
    disposition is to terminate. A `Runner` built here has installed no handler
    for it — `install_signal_handlers` is the real entry point's call, not a
    test's — so every arm made in-process points a timer at the test runner.

    It had been pointed for a while: `_execute`'s per-run arm left the limit at
    `used + cpuSeconds`, which the suite reached on a long run and died of,
    reading as a run that reached 100% and printed no summary. The standing
    budget added by M1 is small enough to make it a matter of seconds, which is
    how it was found.

    Ordered last in this module by name, and asserted on the *process* rather
    than on a call count, because what matters is the state left behind rather
    than which line left it.

    Sabotage: ungate either `arm_cpu_budget` call and this reports a finite
    soft limit.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)

    assert soft == resource.RLIM_INFINITY, (
        f"a guest armed RLIMIT_CPU on the test runner: soft={soft}, hard={hard}"
    )
