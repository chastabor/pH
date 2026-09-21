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
