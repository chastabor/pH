"""Every run sends exactly one `done`, including a run that never started.

The host has no wall clock for a run. It writes `run` and then waits for `done`.
A run that never sends `done` blocks the kernel until a person cancels it. For
that reason `_execute` reports through a `finally`, and has a second catch around
the settle for the ways in which settling can fail.

A run that is cancelled *before its first line* runs none of that code.
`create_task` schedules a coroutine and does not start it. A host that cancels
immediately after it issues a run writes both frames one after the other, and
the guest reads both before it yields. `run` and `cancel` then arrive in the same
read chunk, and the cancel reaches a task whose body never entered its own `try`.

The tests drive `Runner` directly and not through the kernel. Whether the
`cancel` of the kernel lands in the same read chunk as its `run` depends on
scheduling that a test through the kernel does not control. The window is at the
level of the guest, and here a test can open it deliberately.
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
    """A channel that gives out a scripted list of frames and records what comes back."""

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
    """F7: the task is cancelled before it has run a line, so nothing settles it.

    Both frames are queued before `serve` reads either. From inside the guest,
    that is what one read chunk looks like: `_begin` creates the task, and the
    loop returns immediately for `cancel` without yielding to the task. `serve`
    then returns when the script ends, which stands for the host going away.
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

    `_owed` lets `_abort_run` answer for a run that never started. The ordinary
    settle must therefore clear it. If it did not, a `cancel` that arrives after
    a run finished would send a *second* terminal frame for that run, and the
    host pairs `done` with a run by id.

    Here the run is allowed to finish first. That is the difference from the
    test above: there the cancel lands in the same chunk, and here it lands
    later.
    """
    runner, channel = _runner([{"type": "run", "id": 3, "program": "1 + 1"}])

    await asyncio.wait_for(runner.serve(), timeout=5)
    assert runner._run is not None
    await asyncio.wait_for(runner._run, timeout=5)

    # A late cancel, when nothing is left to cancel.
    await runner._abort_run()

    done = [frame for frame in channel.sent if frame.get("type") == "done"]
    assert len(done) == 1, f"{len(done)} terminal frames for one run"
    assert done[0]["id"] == 3
    assert done[0].get("error") is None
    assert done[0]["value"] == 2


async def test_a_run_that_dies_above_its_own_guards_still_sends_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gap that `_abort_run` could not reach, and the reason why the callback
    is attached to the task.

    `_execute` reports through a `finally` and has a second catch around the
    settle, but both are *inside* its own `try`. Anything raised before that
    (`arm_cpu_budget`, the construction of the capped streams, `_RUN.set`) leaves
    the task `done()` with `_owed` still set. The old check was in `_abort_run`,
    behind an early `if run is None or run.done(): return`. The one class of
    failure that the bookkeeping could not answer for was therefore a task that
    finished *without* settling. The kernel then waits for a frame that nothing
    will send.

    **The test uses `_CappedStream`, one of the three that this docstring
    names.** It used to use `arm_cpu_budget`. That function is now gated on
    `Runner._catches_budget`, so it does not run in a process that installed no
    `SIGXCPU` handler. This process is such a process, and that is the purpose of
    the gate: arming a process-wide `RLIMIT_CPU` here capped pytest itself. Any
    of the three proves the same property, and this one can be reached
    in-process.

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
    # The callback is scheduled with `call_soon`, so it runs on the next tick.
    await asyncio.sleep(0)

    done = [frame for frame in channel.sent if frame.get("type") == "done"]
    assert len(done) == 1, "a run ended without settling and the host was never told"
    assert done[0]["id"] == 11
    assert done[0]["error"]["kind"] == "RuntimeError"
    assert "could not be built" in done[0]["error"]["message"]


async def test_a_stale_callback_cannot_settle_the_run_that_followed_it() -> None:
    """The callback answers for its own run, not for whatever is owed now.

    `add_done_callback` fires through `call_soon`, and the guard in `_begin` is
    `self._run.done()`, which is already true for a task whose callback has not
    fired. A run that ended without settling could therefore be followed by a new
    `run` frame. The stale callback would then send `done` for the run that had
    *just started*. The host would mark that run as finished before its first
    statement, and nothing could settle it afterward.

    Sabotage: have `_answer_if_owed` read `self._owed` instead of comparing it
    to the run it was created for, and the second run is settled here.
    """
    runner, channel = _runner([])

    async def nothing() -> None:
        return None

    finished = asyncio.get_running_loop().create_task(nothing())
    await finished
    # Run 1 ended without settling. Run 2 was accepted after that.
    runner._owed = 2

    runner._answer_if_owed(1, finished)

    assert channel.sent == [], "a finished run settled its successor"
    assert runner._owed == 2, "and cleared what that successor still owes"


def test_a_runner_in_this_process_never_arms_a_limit_it_cannot_catch() -> None:
    """`RLIMIT_CPU` is process-wide, and this process is pytest.

    `arm_cpu_budget` sets a soft limit on *the process*, and the default action
    of `SIGXCPU` is to terminate. A `Runner` that is built here has installed no
    handler for it, because `install_signal_handlers` is called by the real entry
    point and not by a test. Every arm made in-process therefore sets a timer on
    the test runner.

    This happened for some time. The per-run arm in `_execute` left the limit at
    `used + cpuSeconds`. A long run of the suite reached that limit and died, and
    it looked like a run that reached 100% and printed no summary. The standing
    budget that M1 added is small enough to make that happen within seconds,
    which is how the problem was found.

    The test is last in this module by name. It asserts on the *process* and not
    on a call count, because the state that is left behind matters, not the line
    that left it.

    Sabotage: ungate either `arm_cpu_budget` call and this reports a finite
    soft limit.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)

    assert soft == resource.RLIM_INFINITY, (
        f"a guest armed RLIMIT_CPU on the test runner: soft={soft}, hard={hard}"
    )
