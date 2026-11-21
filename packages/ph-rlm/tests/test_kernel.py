"""The runtime, against a real child process (D1-D5, C1, C3, C10, F4).

These tests would have been the suite of prime-agent itself, if D19 had not
replaced its runtime. Nothing here is mocked. Each test spawns CPython, gives it
fd 3, and drives the protocol, because every property under test is a property
of the process boundary.

## Why guest stdout is coalesced into ~8 KiB frames

One frame for each `write` is quadratic, in two ways. `print` issues two writes
(the text and the newline). If each write becomes a `channel.send`, then on
CPython 3.12 `_SelectorSocketTransport.write` calls `get_write_buffer_size()`,
which is `sum(map(len, self._buffer))` over the pending deque. A cell that never
awaits never lets the transport drain, so that sum grows with everything written
so far. The host then also pays its own **~25 µs** of work for each frame.

Measured: `for i in range(10_000): print(i)` took **2.1-2.6 s**. Buffered into
~8 KiB frames, it takes **~2 ms**.

## Why the frame buffer is a `bytearray` scanned from its tail

With `bytes` and `+=`, a frame of several megabytes copies the whole buffer for
each 64 KiB chunk and scans it again for a newline. A **16 MiB snapshot spent
834 ms of 1160 ms** on exactly that. Appending to a `bytearray` and searching
from the previous length makes both linear: **1160 ms -> 221 ms**.

## Two caps that stop at the cap

**No round trip through JSON when `_serve_call` answers a call.** `encode`
serializes the frame next. To serialize it twice cost **2.8 ms against 1.2 ms
for a 1 MiB result**.

**`_encode_value` never builds what it is about to discard.** `json.dumps` of a
list of 1M elements took **40 ms, and `repr` another 37 ms: 77 ms to produce
64 KiB**. To end the cell with `df` is exactly how models write cells.

## Why sends are serialized behind `_send_lock`

Without the lock, contention arrives in the `OSError`/`ClosedResourceError`
branch as `BusyResourceError`, and is reported as an exit of the child. **Eight
concurrent replies were enough to "kill" a perfectly healthy kernel.**
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn

import anyio
import pytest
from pydantic import ValidationError
from runtime_helpers import namespace

from ph.orphans import process_alive
from ph.seams.code_runtime import CodeBinding, CodeBindingNamespace
from ph.seams.subprocess import EnvScrub
from ph.testing import EMOJI_FAMILY, settled
from ph.tools.code_mode import CodeRunFailure, ToolCallError
from ph_rlm.kernel.manager import (
    MAX_FRAME_BYTES,
    RESET_NOTICE,
    Config,
    Kernel,
    KernelDied,
    KernelLimits,
    frame_cap,
)
from ph_runtime.cell import MAGIC_HINT
from ph_runtime.protocol import FD_ENV, FRAME_BYTES_ENV, truncation_marker

pytestmark = pytest.mark.anyio

MakeKernel = Callable[..., Any]


def tools(**handlers: Callable[..., Any]) -> CodeBindingNamespace:
    """A `tools` namespace whose bindings are the given handlers.

    This is deliberately not the real `DispatchBridge`. This file tests the half
    of C1 that belongs to the *runtime*: a call reaches a host closure, and its
    answer reaches the program. `test_governance_gate.py` tests the half that
    belongs to the pipeline.
    """
    return namespace("tools", **handlers)


# ------------------------------------------------------------------ execution --


async def test_top_level_await_and_return_both_work(make_kernel: MakeKernel) -> None:
    """The two things module-level code cannot normally do (D2's replacement)."""
    kernel = await make_kernel()
    result = await kernel.run("import asyncio\nawait asyncio.sleep(0)\nreturn 7", (), None)
    assert result.error is None
    assert result.value == 7


async def test_a_trailing_expression_is_the_cells_value(make_kernel: MakeKernel) -> None:
    kernel = await make_kernel()
    assert (await kernel.run("2 + 3", (), None)).value == 5


async def test_the_namespace_persists_across_cells(make_kernel: MakeKernel) -> None:
    """The persistence of C1, and the reason why the seam demands snapshots (D6)."""
    kernel = await make_kernel()
    await kernel.run("import math\ncounter = 1\ndef bump(): return counter + 1", (), None)
    result = await kernel.run("counter = bump()\n(counter, math.floor(2.5))", (), None)
    assert result.value == [2, 2]


async def test_a_magic_is_a_syntax_error_that_names_the_governed_route(
    make_kernel: MakeKernel,
) -> None:
    """D2. The magic *was* the bypass, so the message names the tool and does not apologize."""
    kernel = await make_kernel()
    result = await kernel.run("%%bash\necho hello", (), None)
    assert result.error is not None
    assert MAGIC_HINT in result.error
    assert "tools.bash" in result.error


async def test_a_cell_traceback_shows_the_cell_and_not_the_runner(
    make_kernel: MakeKernel,
) -> None:
    kernel = await make_kernel()
    result = await kernel.run("def boom():\n    raise ValueError('nope')\nboom()", (), None)
    assert result.error is not None
    assert "ValueError: nope" in result.error
    assert "<cell>" in result.error
    assert "ph_runtime" not in result.error, "the harness's own frames are not the model's problem"


# --------------------------------------------------------------------- limits --


async def test_output_is_capped_with_the_shared_marker(make_kernel: MakeKernel) -> None:
    """D4: the cap holds, and the marker is the one that the host would have written."""
    kernel = await make_kernel(max_log_bytes=2_048)
    result = await kernel.run("print('x' * 50_000)", (), None)
    assert result.truncated is True
    assert len(result.logs) < 10_000
    assert truncation_marker(0, 2_048).split("—")[0].strip() in result.logs


async def test_a_value_too_large_is_bounded(make_kernel: MakeKernel) -> None:
    kernel = await make_kernel(max_value_bytes=512)
    result = await kernel.run("'y' * 10_000", (), None)
    assert result.truncated is True
    assert isinstance(result.value, str)
    assert len(result.value) <= 512


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
async def test_a_cpu_bomb_hits_its_budget_and_the_kernel_survives(
    make_kernel: MakeKernel,
) -> None:
    """D3, plus the property that makes a *persistent* kernel possible.

    `RLIMIT_CPU` is cumulative over the process, so the budget is armed again at
    each run. Without that, this test would pass and the *next* cell would die
    on a limit that this cell used up.
    """
    kernel = await make_kernel(cpu_seconds=1)
    result = await kernel.run("while True:\n    pass", (), None)
    assert result.error is not None
    assert "CPU budget" in result.error
    # The budget was armed again. The kernel is still usable, which is the
    # purpose.
    assert (await kernel.run("1 + 1", (), None)).value == 2


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
async def test_a_cpu_bomb_on_a_worker_thread_costs_the_cell_and_not_the_guest(
    make_kernel: MakeKernel,
) -> None:
    """The budget must cost one run, wherever the CPU was used (F1).

    `SIGXCPU` had a `signal.signal` handler that raised, and such a handler
    unwinds from wherever the **main thread** is. When the cell is on that
    thread, the raise lands in the cell, which is the design. When a worker uses
    the CPU, the main thread waits in the `select()` of the loop. The exception
    then came out of the internals of `asyncio`, and the guest exited. The
    namespace, the kernel of the session, and every later cell were lost for a
    limit that should end one run.

    The handler now asks where the cell is before it raises, and cancels the run
    task when the answer is "not here". Both routes report the same `cpu` error.
    They differ only in what they are allowed to interrupt.
    """

    kernel = await make_kernel(cpu_seconds=1)

    # This runs under `fail_after`, because the old failure is a *hang* and not a
    # wrong answer. The main thread died of the raise, and the worker that it
    # could not interrupt kept the process alive. The channel therefore never
    # reached EOF. The host has no wall clock for a run, so it waited for a
    # `done` frame that nothing was left to send.
    with anyio.fail_after(60):
        result = await kernel.run(
            "import asyncio\n"
            "def burn() -> None:\n"
            "    while True:\n"
            "        pass\n"
            "await asyncio.to_thread(burn)",
            (),
            None,
        )

    assert result.error is not None
    assert "CPU budget" in result.error, result.error
    # The guest still serves, which is what this regression test checks. A
    # restarted kernel would say so in the logs of the next run.
    revived = await kernel.run("'alive'", (), None)
    assert revived.value == "alive"
    assert RESET_NOTICE not in revived.logs, "the guest died of the budget"


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX")
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="macOS refuses RLIMIT_AS (ValueError: current limit exceeds maximum limit), "
    "so the guest reports the limit as not applied — see test_boot_report.py",
)
async def test_a_memory_bomb_hits_its_address_space_limit(make_kernel: MakeKernel) -> None:
    kernel = await make_kernel(address_space_bytes=1024**3)
    result = await kernel.run("buffer = bytearray(4 * 1024**3)", (), None)
    assert result.error is not None
    assert "MemoryError" in result.error


# --------------------------------------------------------------- cancellation --


async def test_cancel_aborts_a_waiting_cell_and_the_next_run_succeeds(
    make_kernel: MakeKernel,
) -> None:
    """D5. No workaround for the control channel: fd 3 is not the channel that the run occupies."""

    from ph.cancel import CancelToken

    kernel = await make_kernel()
    token = CancelToken()

    async with anyio.create_task_group() as tasks:

        async def cancel_shortly() -> None:
            await anyio.sleep(0.3)
            token.cancel("user")

        tasks.start_soon(cancel_shortly)
        result = await kernel.run("import asyncio\nawait asyncio.sleep(30)", (), token)

    assert result.error is not None
    assert (await kernel.run("'alive'", (), None)).value == "alive"


async def test_a_spinning_cell_is_killed_after_the_grace_period(
    make_kernel: MakeKernel,
) -> None:
    """The case that neither cooperative route can reach.

    A cell that spins in Python starves the loop of the guest, so the `cancel`
    frame is never read and the `SIGINT` callback never runs. The host
    escalates. The test asserts that the outcome is reported *truthfully*: the
    namespace is gone and the result says so. The kernel does not stay stuck
    until the turn times out.
    """

    from ph.cancel import CancelToken

    kernel = await make_kernel(cpu_seconds=600, cancel_grace=0.3)
    token = CancelToken()

    async with anyio.create_task_group() as tasks:

        async def cancel_shortly() -> None:
            await anyio.sleep(0.3)
            token.cancel("user")

        tasks.start_soon(cancel_shortly)
        result = await kernel.run("while True:\n    pass", (), token)

    assert result.error is not None
    assert "namespace is gone" in result.error
    # The session also continues: the next cell gets a fresh kernel and is told
    # so.
    revived = await kernel.run("'back'", (), None)
    assert revived.value == "back"
    assert RESET_NOTICE in revived.logs


async def test_a_cell_blocked_behind_a_large_reply_is_still_killed(
    make_kernel: MakeKernel,
) -> None:
    """The abort ladder must not wait on the channel that it is trying to abandon
    (F2).

    The setup is one background call, a reply too big for the socket buffer, and
    a cell that then spins. The loop of the guest is starved, so it drains
    nothing. The write of the host fills the buffer and waits while it holds
    `_send_lock`, and every later frame queues behind it. That is the state that
    the abort ladder exists for, and the ladder went through the same lock.
    `_watch` asked for the cancel frame, blocked on the lock, and never returned
    to the loop that times the grace, so the escalation to `SIGKILL` could not
    be reached. `aclose` queued behind it too, so the kernel could not even be
    shut down.

    The clock now starts before the ask, and the ask runs in its own task, so
    the grace expires on schedule whatever the channel is doing. The test runs
    under `fail_after` because the regression is a hang: without the fix,
    nothing here ever returns.
    """

    from ph.cancel import CancelToken

    # Well past the buffer of a unix socket. The reply therefore cannot be handed
    # off in one write, and the host is still in the middle of the frame when the
    # cancel lands.
    huge = "x" * (8 * 1024 * 1024)

    async def big(**_arguments: object) -> str:
        return huge

    bindings = tools(big=big)
    kernel = await make_kernel(namespaces=(bindings,), cpu_seconds=600, cancel_grace=0.3)
    token = CancelToken()
    program = (
        "import asyncio\n"
        # The call is issued from a background task, so the cell does not await
        # the reply that it is about to stop reading.
        "asyncio.get_running_loop().create_task(tools.big())\n"
        "await asyncio.sleep(0)\n"
        "while True:\n"
        "    pass\n"
    )

    async with anyio.create_task_group() as tasks:

        async def cancel_shortly() -> None:
            await anyio.sleep(0.5)
            token.cancel("user")

        tasks.start_soon(cancel_shortly)
        with anyio.fail_after(30):
            result = await kernel.run(program, (bindings,), token)

    assert result.error is not None
    assert "namespace is gone" in result.error, result.error
    # The session survives, which is the reason why the ladder has a last step.
    revived = await kernel.run("'back'", (), None)
    assert revived.value == "back"


async def test_a_namespace_larger_than_one_frame_is_still_snapshotted(
    make_kernel: MakeKernel,
) -> None:
    """The per-variable cap and the per-frame cap must be the same cap (F3).

    `max_snapshot_bytes` limits each *value* that the guest encodes, and the
    guest put every changed value in one `snapshot` frame. Three variables just
    under the cap therefore made a frame three times over it. The host refuses
    an oversized frame, and that is correct: a peer that writes megabytes with
    no newline is how a hostile child would try to exhaust the reader (C10). But
    an ordinary namespace then could not be told from an attack. The channel
    closed, the model was told that the runtime had exited, and the namespace
    was gone. Nothing in the account mentioned a snapshot.

    The property is the *ratio* between the two caps. It is now asserted at
    whatever `max_snapshot_bytes` the kernel booted with, and not by lowering
    the constant of the host to meet it (M3). `frame_cap` derives the ceiling of
    the reader from the limit, so a deployment that raises one raises the other.
    The companion test below checks that derivation separately:
    `test_the_readers_ceiling_follows_the_limit_the_kernel_booted_with`.
    """
    kernel = await make_kernel(max_snapshot_bytes=200 * 1024)

    result = await kernel.run(
        "\n".join(f"v{index} = 'x' * 100_000" for index in range(3)), (), None
    )

    assert result.error is None, result.error
    # The kernel is still the one that ran the cell. A refused frame would have
    # closed the channel, and the next run would have said so.
    revived = await kernel.run("v0 == 'x' * 100_000 and v2 == 'x' * 100_000", (), None)
    assert revived.value is True
    assert RESET_NOTICE not in revived.logs, "the namespace was lost to its own snapshot"


async def test_a_task_a_cell_left_behind_cannot_call_into_the_next_run(
    make_kernel: MakeKernel,
) -> None:
    """A run owns its calls. The namespace outlives the cell, but that does not
    mean that the tasks of the cell outlive it (F4).

    The guest is persistent, so the loop keeps running between programs. A task
    that the cell started with `asyncio.create_task` and never awaited continues
    to call bindings after its run has settled. The host could not tell: a
    `call` frame said which *call* it was, and never which *program*. The stray
    call was therefore served against whichever run was open next. It used the
    call budget of that run, was recorded under its dispatch id, and had its
    approval decided for a turn that had nothing to do with it.

    The first cell leaves a task that loops on a binding. The second cell calls
    the same binding one time. Only the call of the second run itself must reach
    that run.
    """

    seen: list[str] = []

    async def note(**arguments: object) -> str:
        seen.append(str(arguments.get("tag")))
        return "ok"

    bindings = tools(note=note)
    kernel = await make_kernel(namespaces=(bindings,))

    first = await kernel.run(
        "import asyncio\n"
        "async def keep_calling() -> None:\n"
        "    while True:\n"
        "        try:\n"
        # The cancel is swallowed on purpose. The guest ends a well-behaved task
        # at settle, and this test is about a task that is not well-behaved.
        # Because of that, the thing under test is the refusal of the *host* and
        # not the cancellation of the guest. It is also the case that the run
        # stamp exists for.
        "            await tools.note(tag='stray')\n"
        "            await asyncio.sleep(0.01)\n"
        "        except asyncio.CancelledError:\n"
        "            pass\n"
        "asyncio.get_running_loop().create_task(keep_calling())\n"
        "await asyncio.sleep(0.05)\n"
        "'done'",
        (bindings,),
        None,
    )
    assert first.value == "done", first.error
    assert "stray" in seen, "the background task never got going, so this proves nothing"

    # Long enough that a surviving task would have called many times.
    await anyio.sleep(0.2)
    seen.clear()
    second = await kernel.run("await tools.note(tag='mine')", (bindings,), None)

    assert second.error is None, second.error
    assert seen == ["mine"], f"a settled run's task reached the next one: {seen}"


# --------------------------------------------------------------- the boundary --


async def test_a_cell_that_kills_the_process_costs_the_namespace_not_the_session(
    make_kernel: MakeKernel,
) -> None:
    """D1. The namespace cannot be recovered. Saying so prevents a wasted turn."""
    kernel = await make_kernel()
    await kernel.run("keep = 'this'", (), None)
    result = await kernel.run("import os\nos._exit(1)", (), None)
    assert result.error is not None

    revived = await kernel.run("'keep' in dir()", (), None)
    assert revived.value is False
    assert RESET_NOTICE in revived.logs


async def test_forged_frames_from_the_cell_do_not_disturb_the_host(
    make_kernel: MakeKernel,
) -> None:
    """C10, end to end: the cell writes onto fd 3 itself.

    A forged `done` for another run id must settle nothing, and a forged `reply`
    with a string id must not land on a pending call.
    """
    kernel = await make_kernel()
    program = f"""
import os
fd = int(os.environ[{FD_ENV!r}])
for line in [
    b'{{"type": "done", "id": 9999}}\\n',
    b'{{"type": "reply", "id": "1", "ok": true}}\\n',
    b'not json at all\\n',
    b'{{"type": "boot-ack", "protocol": 99, "python": "x", "limits": {{}}}}\\n',
]:
    os.write(fd, line)
'forged'
"""
    result = await kernel.run(program, (), None)
    assert result.value == "forged", "the real `done` still settled the run"
    assert (await kernel.run("2 * 21", (), None)).value == 42


# ------------------------------------------------------------------- bindings --


async def test_a_binding_call_round_trips_through_the_host(make_kernel: MakeKernel) -> None:
    """C1 at the runtime layer: one `call` frame out, one `reply` back."""
    seen: list[dict[str, Any]] = []

    async def read(**arguments: object) -> Any:  # noqa: ANN401
        seen.append(arguments)
        return {"text": "file contents"}

    namespace = tools(read=read)
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run(
        "found = await tools.read(path='a.py')\nfound['text']", (namespace,), None
    )
    assert result.error is None
    assert result.value == "file contents"
    assert seen == [{"path": "a.py"}]


async def test_concurrent_binding_calls_overlap(make_kernel: MakeKernel) -> None:
    """`asyncio.gather` in a cell is what makes fan-out cheaper than N native calls."""

    async def slow(**arguments: object) -> Any:  # noqa: ANN401
        await anyio.sleep(0.1)
        return arguments["n"]

    namespace = tools(slow=slow)
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run(
        "import asyncio\n"
        "values = await asyncio.gather(*[tools.slow(n=i) for i in range(8)])\n"
        "sum(values)",
        (namespace,),
        None,
    )
    assert result.value == 28


async def test_a_refusal_ends_the_run_and_the_program_cannot_catch_it(
    make_kernel: MakeKernel,
) -> None:
    """C3, the divergence from dsh that the whole containment argument depends on.

    A program that can `except` a refusal can route around it: it can retry with
    a different path, or fall back to `subprocess`. The refusal is therefore not
    an exception that the cell is offered. It ends the run, and the tool call
    fails with the refusal in it.
    """

    async def refused(**_arguments: object) -> NoReturn:
        raise CodeRunFailure("denied", "tools.edit was refused: outside the workspace")

    namespace = tools(edit=refused)
    kernel = await make_kernel(namespaces=(namespace,))

    with pytest.raises(CodeRunFailure) as raised:
        await kernel.run(
            "try:\n"
            "    await tools.edit(path='/etc/passwd')\n"
            "except BaseException:\n"
            "    pass\n"
            "'the program continued'",
            (namespace,),
            None,
        )
    assert raised.value.kind == "denied"
    assert "refused" in raised.value.message


async def test_a_refused_cell_is_stopped_before_it_can_write_anyway(
    make_kernel: MakeKernel, tmp_path: Path
) -> None:
    """The actual enforcement of C3, which belongs to the host and not to the
    guest.

    `RunStopped` makes a *well-behaved* cell unwind, and this cell is not one.
    It swallows `BaseException` and then writes a file with raw `pathlib`, which
    no waterfall can reach. To report the refusal afterwards would be a tool
    call that failed on paper while the side effect happened anyway. The host
    therefore fires the run-scoped abort that the C3 row of the plan names, and
    the write never runs.

    The `sleep` is the window. Without the abort the write lands, and this test
    fails by finding the file.
    """
    target = tmp_path / "written-after-the-refusal.txt"

    async def refused(**_arguments: object) -> NoReturn:
        raise CodeRunFailure("denied", "tools.edit was refused: outside the workspace")

    namespace = tools(edit=refused)
    kernel = await make_kernel(namespaces=(namespace,), cancel_grace=0.5)

    with pytest.raises(CodeRunFailure):
        await kernel.run(
            "import time\n"
            "from pathlib import Path\n"
            "try:\n"
            "    await tools.edit(path='/etc/passwd')\n"
            "except BaseException:\n"
            "    pass\n"
            "time.sleep(2)\n"
            f"Path({str(target)!r}).write_text('routed around the refusal')\n",
            (namespace,),
            None,
        )
    assert not target.exists(), "the refusal did not stop the program"
    # The session survives the abort. Only that cell was ended.
    assert (await kernel.run("'still here'", (namespace,), None)).value == "still here"


async def test_a_failed_call_is_the_programs_to_handle(make_kernel: MakeKernel) -> None:
    """The other half of C3: a *failure* keeps the semantics of dsh and can still be caught."""

    async def failing(**_arguments: object) -> NoReturn:
        raise ToolCallError("read", "no such file")

    namespace = tools(read=failing)
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run(
        "try:\n"
        "    await tools.read(path='missing')\n"
        "except ToolFailed as error:\n"
        "    outcome = f'handled: {error}'\n"
        "outcome",
        (namespace,),
        None,
    )
    assert result.error is None
    assert "no such file" in str(result.value)


async def test_an_uncaught_failure_stops_the_calls_after_it(make_kernel: MakeKernel) -> None:
    """C14: inside a program, "B needs A" is just `await`.

    The tool pipeline orders calls, but it never makes one call conditional on
    the outcome of another. An exclusive call is a barrier that runs whether or
    not the calls before it succeeded. That is left as it is, on purpose. In one
    step, a model cannot feed the result of A into the arguments of B, so only
    the order of effects is at stake. There, B after a failed A is often what
    shows what broke. This is what makes that decision correct: where a model
    does mean "B only if A succeeded", a program says so, because a failed call
    raises and nothing after it runs.

    Sabotage: have a failed dispatch return its message instead of raising, and
    `b` runs against an `a` that never answered.
    """
    ran: list[str] = []

    async def failing(**_arguments: object) -> NoReturn:
        raise ToolCallError("a", "no answer")

    async def recording(**_arguments: object) -> str:
        ran.append("b")
        return "b ran"

    namespace = tools(a=failing, b=recording)
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run("await tools.a()\nawait tools.b()", (namespace,), None)

    assert result.error is not None and "no answer" in result.error
    assert ran == [], "a call that depended on a failed one still ran"


async def test_an_unknown_binding_says_what_exists(make_kernel: MakeKernel) -> None:
    namespace = tools(read=_ok)
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run("await tools.nonexistent()", (namespace,), None)
    assert result.error is not None
    assert "read" in result.error, "the message names the available bindings"


def _tombstone(received: list[dict[str, object]]) -> CodeBindingNamespace:
    """A `tools.tombstone(sections, reason)` binding that records what it was given."""

    async def record(**arguments: object) -> str:
        received.append(arguments)
        return "ok"

    schema = {"type": "object", "properties": {"sections": {}, "reason": {}}}
    return CodeBindingNamespace(
        name="tools",
        bindings=(
            CodeBinding(name="tombstone", description="", parameters=schema, dispatch=record),
        ),
    )


async def test_a_binding_is_called_the_way_the_listing_shows_it(
    make_kernel: MakeKernel,
) -> None:
    """The SDK listing renders `async def tools.tombstone(sections, reason)`, so
    positional arguments bind in that order. A live model wrote
    `tools.context_tombstone(['S12'], reason=...)` from the listing. The proxy
    took keywords only, so the call was refused, and the model lost the step that
    it had to edit in.

    Sabotage: drop the parameter names from `_declare`, and the call is refused.
    """
    received: list[dict[str, object]] = []
    namespace = _tombstone(received)
    kernel = await make_kernel(namespaces=(namespace,))

    result = await kernel.run(
        "await tools.tombstone(['S1'], reason='a')\nawait tools.tombstone(['S2'], 'b')",
        (namespace,),
        None,
    )

    assert result.error is None, result.error
    assert received == [
        {"sections": ["S1"], "reason": "a"},
        {"sections": ["S2"], "reason": "b"},
    ]


@pytest.mark.parametrize(
    "call",
    ["await tools.tombstone(['S1'], 'b', 'extra')", "await tools.tombstone(['S1'], sections=[])"],
)
async def test_a_call_the_listing_cannot_mean_is_refused_with_it_spelled_out(
    make_kernel: MakeKernel, call: str
) -> None:
    """More positional arguments than parameters, or one parameter given both ways."""
    received: list[dict[str, object]] = []
    namespace = _tombstone(received)
    kernel = await make_kernel(namespaces=(namespace,))

    result = await kernel.run(call, (namespace,), None)

    assert result.error is not None
    assert "tools.tombstone(sections, reason)" in result.error, result.error
    assert received == [], "a refused call reached the host"


async def test_rlm_rejects_an_unknown_keyword_in_the_cell(make_kernel: MakeKernel) -> None:
    """The contract of prime-agent: unknown kwargs fail loudly (§6.0)."""
    namespace = CodeBindingNamespace(
        name="rlm",
        bindings=(
            CodeBinding(
                name="run", description="spawn", parameters={}, dispatch=_ok, counts_as_spawn=True
            ),
        ),
    )
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run("await rlm('investigate', mode='fast')", (namespace,), None)
    assert result.error is not None
    assert "mode" in result.error
    assert "access" in result.error, "the message lists what rlm() does accept"


# ------------------------------------------------------------------ lifecycle --


async def test_the_child_does_not_inherit_credentials(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child is the one that `scrub_env` was written for.

    The docstring of `ph.seams.subprocess` says so: "a child runs code the model
    wrote, so it does not inherit `*KEY*`". A cell that can read `os.environ` can
    print a provider credential into its own output, and the output is then
    logged. The credential was never for the model to see.
    """
    monkeypatch.setenv("PH_TEST_ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("PH_TEST_DB_PASSWORD", "hunter2")
    monkeypatch.setenv("PH_TEST_HARMLESS", "fine")
    kernel = await make_kernel()
    result = await kernel.run(
        "import os\n"
        "[os.environ.get(name) for name in "
        "('PH_TEST_ANTHROPIC_API_KEY', 'PH_TEST_DB_PASSWORD', 'PH_TEST_HARMLESS')]",
        (),
        None,
    )
    assert result.value == [None, None, "fine"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
async def test_a_closed_kernel_leaves_no_zombie(make_kernel: MakeKernel) -> None:
    """F4: a child that exited while the parent lives, and that is never reaped, is a zombie."""
    kernel = await make_kernel()
    pid = kernel._process.pid
    await kernel.run("1", (), None)
    await kernel.aclose()
    state = _process_state(pid)
    assert state != "Z", f"pid {pid} is a zombie"


async def test_close_is_idempotent(make_kernel: MakeKernel) -> None:
    kernel = await make_kernel()
    await kernel.aclose()
    await kernel.aclose()


async def test_a_closed_kernel_starts_no_guest_for_a_run_that_held_it(
    make_kernel: MakeKernel,
) -> None:
    """K1. A kernel that is closed permanently refuses a run and does not start a
    guest.

    `close_namespace` takes the kernel out of the table and closes it. A run
    that already held the reference found `not _alive` and started a guest that
    nothing tracked. The kernel was out of the table, so no later close reached
    the guest.

    Sabotage: drop the `_closed` check from `run`, and a guest is running again.
    """
    kernel = await make_kernel()
    await kernel.aclose()

    with pytest.raises(KernelDied, match="closed"):
        await kernel.run("1", (), None)
    assert kernel._process is None, "a guest was started for a closed kernel"


async def test_a_boot_stopped_from_outside_leaves_no_half_booted_guest(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K2. A hard stop during boot stops the guest and does not leave it running.

    `_alive` is set before `boot-ack`. A stop between the two therefore left a
    guest that the next run treated as ready. The next run skipped the start and
    the restore, and sent its cell to a guest that had not finished booting.

    Sabotage: drop the `except BaseException` around the boot, and the kernel
    reads alive with a guest still there.
    """
    kernel = await make_kernel()
    await kernel.aclose()
    kernel._closed = False
    waiting = anyio.Event()

    async def never_acked(self: Kernel) -> str | None:
        waiting.set()
        await anyio.sleep_forever()
        return None  # pragma: no cover

    monkeypatch.setattr(Kernel, "_await_boot_ack", never_acked)
    async with anyio.create_task_group() as booting:
        booting.start_soon(kernel.start, [])
        await waiting.wait()
        booting.cancel_scope.cancel()

    assert not kernel._alive, "a guest that never acked reads as ready"
    assert kernel._process is None


async def test_a_run_stopped_from_outside_does_not_leave_its_cell_running(
    make_kernel: MakeKernel,
) -> None:
    """K2. A hard stop in the middle of a run ends the guest, so the next run is
    not refused.

    A stop through the token of the run goes through the abort ladder. A stop
    from outside (the task of the tool call was canceled) skipped the ladder.
    The guest then continued the cell, and answered the next run with "a program
    is already running".

    Sabotage: drop the `except BaseException` around the run, and the next run
    meets the cell still running.
    """
    kernel = await make_kernel()
    async with anyio.create_task_group() as running:
        running.start_soon(kernel.run, "import time\ntime.sleep(30)", (), None)
        await anyio.sleep(0.5)
        running.cancel_scope.cancel()

    with anyio.fail_after(10):
        result = await kernel.run("1 + 1", (), None)
    assert result.error is None, result.error
    assert result.value == 2


async def test_a_close_stopped_from_outside_still_kills_and_forgets_the_guest(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K2. The teardown finishes under a stop: the guest is killed, reaped and forgotten.

    The process is detached from the kernel first. A stop during the polite
    shutdown therefore skipped the kill, `_alive = False`, and the removal of
    the journal entry, and nothing would come back for the guest.

    Sabotage: drop the shield from `_teardown`, and the guest outlives the close.
    """
    kernel = await make_kernel()
    pid = kernel._process.pid
    asked = anyio.Event()
    original = Kernel._send

    async def ignored(self: Kernel, frame: Any) -> None:  # noqa: ANN401
        if type(frame).__name__ == "ShutdownFrame":
            asked.set()
            await anyio.sleep(kernel.shutdown_grace * 2)
            return
        await original(self, frame)

    monkeypatch.setattr(Kernel, "_send", ignored)
    async with anyio.create_task_group() as closing:
        closing.start_soon(kernel.aclose)
        await asked.wait()
        closing.cancel_scope.cancel()

    assert not kernel._alive
    assert not process_alive(pid), f"pid {pid} outlived a canceled close"


async def _ok(**_arguments: object) -> None:
    return None


def _process_state(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    tail = raw.rpartition(")")[2].split()
    return tail[0] if tail else None


def test_the_guest_never_imports_the_harness() -> None:
    """The process boundary exists so that model code cannot reach the harness.

    A guest module that imports `ph` would put the harness back within the reach
    of model code. The managed venv would then also need `ph-core`, which is the
    dependency that the venv exists to avoid.
    """
    root = Path(__file__).resolve().parents[3] / "ph-runtime-guest" / "src" / "ph_runtime"
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ph", "from ph."))
        if not line.startswith(("import ph_runtime", "from ph_runtime"))
    ]
    assert offenders == []


async def test_a_running_cell_never_interrupts_the_frame_read(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason why the clock of the abort ladder is a sibling task.

    `_pump` used to run in a 50 ms `move_on_after`, so that it could ask between
    reads whether the caller had canceled. The cost was not one scope for each
    socket read but **one for each frame**. `_recv_line` returns directly from
    its buffer whenever a frame is already there, and a 64 KiB read from a cell
    that prints a lot carries many frames. This was measured at 37.5 ms against
    13.8 ms over 3,000 `log` frames. That is 7.9 µs for each frame, for a
    deadline that was almost never reached.

    The test counts interruptions and does not assert on the structure. "Does
    `_pump` contain a `move_on_after`" is a fact about the source. What matters
    is that a run that lasts many poll intervals interrupts the read **no**
    times.

    **This is not a guard against issue 58**, although it was first written as
    one. The mechanism there belongs to `UNIXSocketStream` and `connect_unix`,
    that is, to the daemon socket. `_RawSocketMixin._wait_until_readable`
    registers `f.set_result` as the reader callback, and removes the reader in a
    done-callback one loop iteration later. A canceled wait can therefore still
    be fired on a canceled future. `_recv_line` calls the *free*
    `anyio.wait_readable`, which is a different implementation. It catches
    `InvalidStateError` and removes the reader synchronously inside the
    callback. Canceling it was always safe.

    Sabotage: put a 50 ms `move_on_after` back around the read, and `canceled`
    counts roughly `duration / 0.05`.
    """
    original = anyio.wait_readable
    canceled = 0
    waits = 0

    async def counting(obj: Any) -> Any:  # noqa: ANN401
        nonlocal canceled, waits
        waits += 1
        try:
            return await original(obj)
        except anyio.get_cancelled_exc_class():
            canceled += 1
            raise

    monkeypatch.setattr(anyio, "wait_readable", counting)

    kernel = await make_kernel()
    # Long enough that the old 50 ms poll would have fired many times.
    slept = 0.6
    result = await kernel.run(f"import asyncio\nawait asyncio.sleep({slept})\n'done'", (), None)

    assert result.value == "done", result.error
    # The positive control. `canceled == 0` also passes if the patch is never
    # reached at all. A future move of `_recv_line` onto a stream would cause
    # that silently.
    assert waits > 0, "the patched readiness wait was never reached; the test proves nothing"
    assert canceled == 0, (
        f"the frame read was interrupted {canceled} times during one run; "
        "the poll is back on the read path and the per-frame scope with it"
    )


def test_a_run_records_how_it_ended_once_and_the_first_writer_wins() -> None:
    """Three places end a run, and they race by construction.

    `_watch` can kill while `_pump` is blocked in the middle of a frame, and the
    socket close in `_teardown` is itself what wakes `_pump`. A `done` frame
    that was already buffered when the kill landed therefore used to replace
    "the runtime was killed" with the value of the program. Each writer assigned
    `error` and `settled` directly, and only `_on_closed` checked first. The
    outcome therefore depended on three call sites that had to run in the
    correct order relative to each other.

    Sabotage: drop the `if self.settled: return False` guard and the second
    call's account replaces the first's.
    """
    from ph_rlm.kernel.manager import _ActiveRun

    active = _ActiveRun(run_id=1, bindings={})
    assert active.settle(error="killed") is True, "the first writer claims it"
    assert active.settled

    assert active.settle(value="done") is False, "and the second is told it did not"
    assert active.error == "killed", "the kill is still the account the caller gets"
    assert active.value is None, "and the late frame's value did not land beside it"


async def test_a_channel_closed_mid_frame_is_reported_as_a_closure(
    make_kernel: MakeKernel,
) -> None:
    """The third way in which this socket reports that it is gone. It was not
    caught.

    `sock.send` and `sock.recv` on a closed socket raise `OSError`, and
    `notify_closing` raises `ClosedResourceError` into a waiter that already
    waits. Both were handled. But `wait_writable` and `wait_readable` that are
    *entered* with a closed socket raise
    `ValueError("Invalid file descriptor: -1")`, because `close()` sets the
    fileno to `-1` and the loop refuses to register it. A frame that is larger
    than the socket buffer goes through that wait many times, so a teardown that
    lands between two turns hits exactly this case.

    This was found as a CI failure on macOS only, in
    `test_a_cell_blocked_behind_a_large_reply_is_still_killed`, where an 8 MB
    reply is being written while the abort ladder kills the guest. Linux passed.
    The difference is buffer sizes and wakeup order, and not anything that the
    code decides. This test therefore checks the *shape* with the socket closed
    deterministically, and does not leave it to a race that reproduces on one
    platform.
    """

    from ph_rlm.kernel.manager import _CHANNEL_GONE

    kernel = await make_kernel()
    assert (await kernel.run("1 + 1", (), None)).value == 2

    # The socket is closed under both waits, with nothing waiting on it. This is
    # the case where a wait is entered again, and `fileno()` is already `-1`.
    sock = kernel._sock
    assert sock is not None
    sock.close()
    assert sock.fileno() == -1

    for wait in (anyio.wait_writable, anyio.wait_readable):
        with pytest.raises(_CHANNEL_GONE):
            await wait(sock)

    # The kernel reports this as a channel that is gone, and does not raise. A
    # send finds the socket closed and settles the run, and the session still
    # restarts.
    outcome = await kernel.run("2 + 2", (), None)
    assert outcome.error is not None or outcome.value == 4
    assert (await kernel.run("3 + 3", (), None)).value == 6


async def test_a_path_argument_reaches_the_host_as_a_path(make_kernel: MakeKernel) -> None:
    """F7: `default=repr` silently loses information for the two types that a
    cell passes most.

    `tools.read(Path("notes.md"))` arrived as the string
    `"PosixPath('notes.md')"`, and a `datetime` arrived as
    `"datetime.datetime(2026, 9, 20, 0, 0)"`. Neither is an error anywhere. The
    tool receives a string where the model wrote a value. It either refuses the
    string with a message about a path that does not exist, or uses it.

    A type with no JSON form still falls back to `repr`. The guest does not know
    the schema of the tool, and to refuse an argument that the tool would have
    accepted is the worse error. In that case, `repr` is what makes the refusal
    of the tool itself readable.
    """
    seen: list[dict[str, object]] = []

    async def record(**arguments: object) -> str:
        seen.append(dict(arguments))
        return "ok"

    bindings = tools(record=record)
    kernel = await make_kernel(namespaces=(bindings,))
    program = (
        "from pathlib import Path\n"
        "from datetime import datetime\n"
        "await tools.record(path=Path('notes.md'), when=datetime(2026, 9, 20), tags={'a'})\n"
    )
    result = await kernel.run(program, (bindings,), None)

    assert result.error is None, result.error
    (arguments,) = seen
    assert arguments["path"] == "notes.md", arguments["path"]
    assert arguments["when"] == "2026-09-20T00:00:00", arguments["when"]
    assert arguments["tags"] == ["a"]


async def test_the_guest_runs_in_a_session_of_its_own(make_kernel: MakeKernel) -> None:
    """J9: no controlling terminal for the process that runs model-authored code.

    The comment on the `open_process` call in `start` says why that matters. The
    assertion is on the guest and not on the sandbox argv, because the property
    is "every guest is a session leader". The `--new-session` of bwrap gives
    that on one backend of three.

    Sabotage: drop `start_new_session=True` from `open_process` and the sid is
    the host's.
    """
    kernel = await make_kernel()

    result = await kernel.run("import os\n(os.getsid(0), os.getpid())", (), None)

    sid, pid = result.value
    assert sid == pid, "the guest is not a session leader"
    assert sid != os.getsid(0), "the guest shares the host's session"


async def test_a_subprocess_a_cell_started_dies_with_the_kernel(
    make_kernel: MakeKernel,
) -> None:
    """The other half of J9: the session that the guest got must be killed as a
    group.

    `ph.orphans.signal_group` says why the two belong together. The behavior is
    measured here on the path that has no other protection: a clean `aclose`,
    where the guest exits by itself and only the sweep is left to look for what
    it started.

    Sabotage: `process.kill()` in place of `_kill_group` and the grandchild is
    still alive after the kernel is gone.
    """
    kernel = await make_kernel()
    started = await kernel.run(
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "child.pid",
        (),
        None,
    )
    grandchild = int(started.value)
    assert process_alive(grandchild), "the cell's subprocess never started"

    await kernel.aclose()

    # This is polled and not asserted at once. The grandchild is killed inside
    # `aclose`, and stays for a moment as a zombie that waits to be re-parented.
    # `kill(0)` still answers `True` for a zombie.
    await settled(lambda: not process_alive(grandchild), "the cell's subprocess to be reaped")


# ------------------------------------------------- the loop's own clock (M2) --


async def test_a_run_the_guest_will_never_finish_is_repaired_not_waited_on(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2: the host waits for `done`, and nothing else was watching.

    Both steps of the abort ladder start only because somebody pressed stop. A
    run that is never canceled and never settles therefore had no clock at all,
    and the host waits for that frame with no wall clock of its own.

    **The remedy is a repair, not a kill.** The probe names the run that the
    host waits for. The guest is the only party that knows whether the frame is
    still coming, so the guest settles the run, through the same `_send_done`
    that every other terminal path takes. The namespace survives, and that is
    the purpose. The predicate establishes that the guest is *alive and
    healthy*. To kill a guest on that evidence costs everything that it held,
    for a fault that is one missing frame.

    **The stuck run is staged at the host**, by losing the *first* terminal
    frame of the run. Every input is then what a guest whose `done` never
    arrived produces: the run is unsettled, the loop is free, and nothing is
    owed. Only the first frame is lost, because the repair is a `done` too, and
    nothing could repair a host that dropped every one. To break the teardown of
    the guest itself would reproduce one cause, and not the shape that the
    causes share. Those causes also leave the run still *owed*, and
    `_answer_if_owed` already answers that case.

    Sabotage: drop the `_probe` call from `_watch` and this waits out
    `fail_after` instead.
    """
    kernel = await make_kernel(probe_seconds=0.05)
    lost: list[int] = []
    real = Kernel._settle

    def lose_the_first(self: Kernel, frame: Any, active: Any) -> None:  # noqa: ANN401
        if not lost:
            lost.append(frame["id"])
            return
        real(self, frame, active)

    monkeypatch.setattr(Kernel, "_settle", lose_the_first)
    with anyio.fail_after(10):
        result = await kernel.run("1 + 1", (), None)
    monkeypatch.undo()

    assert lost, "the staging never fired; this asserted nothing"

    assert result.error is not None
    assert "ended without settling" in result.error, result.error
    # The namespace is still there. A kill would have cost it.
    assert (await kernel.run("'alive'", (), None)).value == "alive"
    assert RESET_NOTICE not in (await kernel.run("1", (), None)).logs


async def test_a_cell_that_starves_the_loop_is_measured_and_not_killed(
    make_kernel: MakeKernel,
) -> None:
    """The reading that the probe is *not* allowed to make (M2).

    A cell that spins in straight-line Python starves the reader task that
    answers a probe. The same starvation makes `cancel` and `SIGINT` both miss
    the cell. That is how a busy cell looks, and a busy cell is allowed to be
    busy: the per-run CPU budget (`cpu_seconds`) limits it, and this clock does
    not. The probe therefore goes unanswered for much longer than
    `probe_seconds`, and the run still finishes by itself.

    The benefit of the unanswered probe is the gauge. `phern doctor` reports the
    worst round trip and the stalls. That is how a person sees *where* the time
    of a slow agent goes, and not only that the agent was slow.

    Sabotage: escalate on an outstanding probe, which `_probe` only counts as a
    stall, and this cell is killed mid-computation.
    """
    kernel = await make_kernel(probe_seconds=0.05)

    with anyio.fail_after(30):
        result = await kernel.run("sum(range(60_000_000))", (), None)

    assert result.error is None, result.error
    # This is the closed form, not a second `sum(range(...))`. To compute the sum
    # again here cost 0.58 s of the 1.0 s of the test, to assert a number that
    # the formula of Gauss gives.
    assert result.value == 59_999_999 * 60_000_000 // 2
    assert kernel.loop_stalls > 0, "a cell that never yielded answered every probe on time"


def _count_pings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The id of every probe that a kernel sends from now on. Each probe is still really sent."""
    sent: list[int] = []
    real = Kernel._ping

    async def counting(self: Kernel, run_id: int, probe: int) -> None:
        sent.append(probe)
        await real(self, run_id, probe)

    monkeypatch.setattr(Kernel, "_ping", counting)
    return sent


async def test_the_probe_reports_how_far_behind_a_guest_loop_is(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gauge on a loop that is *not* starved, which is the ordinary reading.

    A cell that awaits inside asyncio leaves the reader task free, so every
    probe is answered promptly. A person needs that baseline to read the busy
    case as unusual. That is the reason why the loop is measured continuously,
    and not only when something already looks wrong.

    **The probes also go out at the interval that the setting names.** `_watch`
    used to tick twenty times a second. An answer makes the probe idle again. If
    that alone decided when the next probe goes out, a frame would go each way
    on every tick. That is twenty times the traffic that `probe_seconds` says it
    costs, on every open run.

    Sabotage: drop the `probed_at` gate from `_probe` and the count is the poll
    rate rather than the probe rate.
    """
    kernel = await make_kernel(probe_seconds=0.2)
    sent = _count_pings(monkeypatch)
    await kernel.run("import asyncio\nawait asyncio.sleep(1.0)", (), None)

    assert kernel.loop_worst is not None, "no probe was answered"
    assert kernel.loop_worst < 0.2, f"an idle loop answered slowly: {kernel.loop_worst}"
    assert kernel.loop_stalls == 0, "an idle loop left a probe unanswered"
    # The cell runs for one second, and the probe interval is a fifth of a
    # second. The count is bounded loosely on both sides. The exact count belongs
    # to the scheduler, and the claim is the *order* of magnitude. The rate that
    # it must not be is the old poll rate, 20/s.
    assert 2 <= len(sent) <= 10, f"{len(sent)} probes in a second at probe_seconds=0.2"


async def test_an_open_run_wakes_its_clock_once_per_probe(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clock of the abort ladder sleeps until its next question, and not on a
    tick (P12-03).

    `_watch` used to wake twenty times a second for the life of every open run,
    and ask `_probe` whether a probe was due. Now it plans the moment when one
    is due. A cell that runs for one second at a probe interval of a fifth of a
    second therefore wakes it about five times.

    The count is taken at `_probe`, which the clock calls on every wake while
    the probe is on. The count is therefore the wake count, whatever the
    structure of the clock itself.

    Sabotage: put a 50 ms `anyio.sleep` back at the top of `_watch`'s loop, and the
    count is the tick rate.
    """
    kernel = await make_kernel(probe_seconds=0.2)
    asked: list[float] = []
    real = Kernel._probe

    def counting(self: Kernel, active: Any, tasks: Any) -> None:  # noqa: ANN401
        asked.append(anyio.current_time())
        real(self, active, tasks)

    monkeypatch.setattr(Kernel, "_probe", counting)
    await kernel.run("import asyncio\nawait asyncio.sleep(1.0)", (), None)

    assert 2 <= len(asked) <= 10, f"the clock woke {len(asked)} times in a second"


async def test_a_probe_answered_after_a_stall_starts_the_next(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stalled probe has no deadline, so its answer is what wakes the clock.

    The cell starves its loop for some time. The probe that is outstanding when
    the spin begins therefore goes unanswered past its interval, and is counted
    as a stall. Then the cell yields, the answer arrives, and probing must
    resume for the rest of the run.

    Sabotage: drop `_move()` from `_ActiveRun.answered`, and the clock sleeps out
    the run once the stall ends; one or two probes go out, not a dozen.
    """
    kernel = await make_kernel(probe_seconds=0.05)
    sent = _count_pings(monkeypatch)
    with anyio.fail_after(30):
        await kernel.run(
            "import asyncio, time\n"
            "end = time.monotonic() + 0.4\n"
            "while time.monotonic() < end:\n"
            "    pass\n"
            "await asyncio.sleep(0.6)\n",
            (),
            None,
        )

    assert kernel.loop_stalls >= 1, "the spin never starved a probe; this asserted nothing"
    assert len(sent) >= 5, f"probing did not resume after the stall: {len(sent)} sent"


async def test_with_the_probe_off_a_run_sends_no_probe_and_cancel_still_stops_it(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`probeSeconds: null` turns the probe off, and turns off nothing else.

    No probe goes out, and the clock has nothing to wake for. But the cancel of
    the caller is an event (`token.wait()`), so it still reaches the cell at
    once.

    Sabotage: race only `moved` in `_watch`, not the token, and this cancel waits
    for nothing.
    """
    from ph.cancel import CancelToken

    kernel = await make_kernel(probe_seconds=None)
    sent = _count_pings(monkeypatch)
    token = CancelToken()

    async with anyio.create_task_group() as tasks:

        async def cancel_shortly() -> None:
            await anyio.sleep(0.3)
            token.cancel("user")

        tasks.start_soon(cancel_shortly)
        with anyio.fail_after(10):
            result = await kernel.run("import asyncio\nawait asyncio.sleep(30)", (), token)

    assert result.error is not None
    assert sent == [], "a probe went out with the probe off"
    assert kernel.loop_worst is None


async def test_an_abort_a_refusal_began_still_reaches_the_kill_with_the_probe_off(
    make_kernel: MakeKernel,
) -> None:
    """The grace deadline is planned when the abort begins, from any task.

    A refused call starts the abort from `_serve_call`, not from the clock. This
    cell then ignores `SIGINT` and spins, so the cancel frame is never read and
    the signal does nothing. Only the kill at the end of the grace stops the
    cell. With the probe off, the clock had no deadline at all, so `begin_abort`
    setting `moved` is what wakes it to plan the grace.

    Sabotage: drop `_move()` from `begin_abort`, and the clock sleeps on while the
    cell spins.
    """

    async def refused(**_arguments: object) -> NoReturn:
        raise CodeRunFailure("denied", "tools.edit was refused")

    namespace = tools(edit=refused)
    kernel = await make_kernel(
        namespaces=(namespace,), cancel_grace=0.3, probe_seconds=None, cpu_seconds=600
    )

    with pytest.raises(CodeRunFailure), anyio.fail_after(10):
        await kernel.run(
            "import signal\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "try:\n"
            "    await tools.edit(path='x')\n"
            "except BaseException:\n"
            "    pass\n"
            "while True:\n"
            "    pass\n",
            (namespace,),
            None,
        )
    revived = await kernel.run("'back'", (namespace,), None)
    assert revived.value == "back"
    assert RESET_NOTICE in revived.logs, "the spinning cell was not killed"


def test_a_probe_interval_of_zero_is_refused_and_null_turns_it_off() -> None:
    """Zero does not mean "as often as possible": the clock would plan a deadline
    that is always now. `null` is the way to say never."""
    assert Config(probe_seconds=None).probe_seconds is None
    with pytest.raises(ValidationError):
        Config(probe_seconds=0)
    with pytest.raises(ValidationError):
        Config(probe_seconds=-1.0)


def test_the_readers_ceiling_follows_the_limit_the_kernel_booted_with() -> None:
    """M3: the sizing that the constant claimed and that nothing enforced.

    The docstring of `MAX_FRAME_BYTES` said that it was "sized to hold a
    `maxSnapshotBytes` payload with base64 and JSON overhead", and at the
    shipped 16 MiB that was true. It is a claim about a *ratio* between two
    numbers, and a deployment configures one of them. A profile that raised
    `maxSnapshotBytes` past about a third of the constant therefore made the
    claim false. The cost is not a rejected frame but a lost namespace: the host
    refuses the oversized snapshot and closes the channel, and the model is told
    that the runtime exited.

    The assertion is on the derivation. It does not round-trip eighty megabytes,
    which the sibling test above
    (`test_a_namespace_larger_than_one_frame_is_still_snapshotted`) used to need
    a monkeypatch to avoid.

    Sabotage: return `MAX_FRAME_BYTES` unconditionally and the raised limit gets
    a ceiling below its own payload.
    """
    shipped = KernelLimits()
    assert frame_cap(shipped) == MAX_FRAME_BYTES, "the shipped limit needs no more than the floor"

    raised = KernelLimits(max_snapshot_bytes=256 * 1024 * 1024)
    assert frame_cap(raised) > raised.max_snapshot_bytes * 4 // 3, (
        "a base64 payload at the configured limit would not fit the reader's ceiling"
    )
    # The floor still holds for a deployment that lowers the limit.
    assert frame_cap(KernelLimits(max_snapshot_bytes=1024)) == MAX_FRAME_BYTES


async def test_the_guest_reads_with_the_hosts_frame_cap(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8: one expression gives the limit to both readers, and it is passed at
    spawn.

    The ceiling of the host followed `maxSnapshotBytes` while the ceiling of the
    guest stayed at a fixed 64 MiB, so a raised limit lost the namespace on
    restore. `start` puts `frame_cap` in the environment of the child, in the
    same way that the descriptor crosses. The assertion is on a real spawn,
    which the guest then boots from.

    Sabotage: drop the `FRAME_BYTES_ENV` entry from `start`'s environment and
    this finds none.
    """
    handed: dict[str, str] = {}

    apply = EnvScrub.apply

    def recording(
        scrub: EnvScrub, base: Mapping[str, str] | None = None, *, extra: Mapping[str, str]
    ) -> dict[str, str]:
        handed.update(extra)
        return apply(scrub, base, extra=extra)

    monkeypatch.setattr(EnvScrub, "apply", recording)
    raised = 256 * 1024 * 1024

    # `start` raises unless the guest answers `boot`. A return therefore proves
    # that the guest started and reads with the number that it was given.
    await make_kernel(max_snapshot_bytes=raised)

    assert handed[FRAME_BYTES_ENV] == str(frame_cap(KernelLimits(max_snapshot_bytes=raised)))


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
async def test_a_finished_cell_cannot_leave_a_thread_burning_a_core(
    make_kernel: MakeKernel,
) -> None:
    """M1: the budget was turned off and not lowered, and nothing armed it again.

    `relax_cpu_budget` disarms at the first `SIGXCPU`, and it must. The limit is
    cumulative and Linux delivers the signal again every CPU-second, so a second
    signal lands in the teardown of the guest and costs the `done` frame. That
    left the gap that this change closes. A `to_thread` worker is reached by
    neither cooperative route: a cancel does not reach a thread, and the raise
    route fires only where the cell is. The run therefore reported `cpu`, and
    the worker continued. With the budget off, no second signal ever came. The
    worker used a core until the session ended.

    A standing `idle_cpu_seconds` is armed at the moment that a run stops owning
    the process, and a breach with no run open ends the runtime. Nothing
    legitimate uses that budget, because a background thread that a cell left on
    purpose is *waiting*, and waiting costs no CPU.

    The namespace is gone afterwards. That is the real cost, and the test
    asserts it. The namespace already held a thread that nobody could stop. The
    reason that the guest gives reaches the model on the way out, through the
    stderr that the host captures.

    Sabotage: drop the `arm_cpu_budget` from `_send_done` and this waits out
    `fail_after` with the thread still running.
    """
    kernel = await make_kernel(cpu_seconds=30, idle_cpu_seconds=1)

    # The cell finishes immediately. No cancel can reach the worker that it
    # starts.
    left = await kernel.run(
        "import threading\n"
        "def burn():\n"
        "    while True:\n"
        "        pass\n"
        "threading.Thread(target=burn, daemon=True).start()\n"
        "'started'",
        (),
        None,
    )
    assert left.value == "started", left.error

    with anyio.fail_after(30):
        await settled(
            lambda: kernel._process is not None and kernel._process.returncode is not None,
            "the runtime to end itself over the runaway",
        )

    # The next cell finds the dead runtime, and its logs carry the sentence that
    # the guest wrote. The stderr that the host captured tells a person *why* the
    # runtime ended. Without it, the person would see an unexplained exit.
    met = await kernel.run("'alive'", (), None)
    assert met.error is not None
    assert "burning CPU" in met.logs, met.logs

    # The cell after that gets a fresh namespace, and its logs say so.
    revived = await kernel.run("'alive'", (), None)
    assert revived.value == "alive"
    assert RESET_NOTICE in revived.logs, "the namespace went and nobody said so"


async def test_capped_output_ends_between_characters(make_kernel: MakeKernel) -> None:
    """The cap counts bytes, and the cut falls between characters, in the print
    stream of the guest and in the pipe drain of the host alike. The cut is never
    inside a family and never produces U+FFFD.

    Sabotage: slice by index in the guest's `write` again and the kept output
    ends in the first members of a family.
    """
    kernel = await make_kernel(max_log_bytes=2_048)
    result = await kernel.run(f"print({EMOJI_FAMILY!r} * 1000)", (), None)

    kept = result.logs.partition("\n")[0]
    assert result.truncated and "\ufffd" not in result.logs
    assert kept and not kept.replace(EMOJI_FAMILY, "")


async def test_a_character_split_between_two_writes_is_joined(make_kernel: MakeKernel) -> None:
    """What the subprocess of a cell writes reaches the pipe of the host in
    pieces, and a character can be split between two of them. One decoder for
    the stream joins the character. Decoding each piece separately made it two
    U+FFFDs."""
    kernel = await make_kernel()
    cell = (
        "import os, time\n"
        "os.write(1, 'é'.encode()[:1]); time.sleep(0.2); os.write(1, 'é'.encode()[1:])\n"
    )

    result = await kernel.run(cell, (), None)

    assert "é" in result.logs and "\ufffd" not in result.logs


async def test_a_run_reports_every_byte_its_prints_dropped(make_kernel: MakeKernel) -> None:
    """The print stream of the guest marks where it was cut, with no number, and
    counts every write after the cut. It then reports the total on the `done`
    frame that it sends in any case, which is the first moment when the whole
    total is known. When the count was taken at the cut, it held only the write
    that crossed the cap.

    Sabotage: count only the crossing write, as the marker used to, and the
    hundred prints after it go missing from the total.
    """
    kernel = await make_kernel(max_log_bytes=2_048)

    result = await kernel.run("for _ in range(100):\n    print('y' * 1_000)", (), None)

    assert result.truncated
    assert result.dropped == 100 * 1_001 - 2_048, "every byte printed past the cap"
    assert truncation_marker(None, 2_048) in result.logs, "and the cut marked with no count"
    assert result.logs.endswith(truncation_marker(result.dropped, 2_048)), (
        "and the logs end with the total, in the sentence `tool-bash` says it in"
    )


async def test_a_run_reports_every_byte_its_pipes_dropped(make_kernel: MakeKernel) -> None:
    """What the subprocesses of a cell write reaches the pipe drain of the host.
    The drain keeps reading past the cap, so that the child never blocks, and
    counts what it skips into the same total.

    Sabotage: drop the count from the skip in `_drain` and the total is only the
    chunk that crossed the cap.
    """
    kernel = await make_kernel(max_log_bytes=2_048)
    cell = (
        "import os, time\n"
        "for _ in range(10):\n"
        "    os.write(1, b'z' * 5_000); time.sleep(0.02)\n"
        "time.sleep(0.3)\n"
    )

    result = await kernel.run(cell, (), None)

    assert result.truncated
    assert result.dropped == 10 * 5_000 - 2_048
