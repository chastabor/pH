"""The runtime, against a real child process (D1-D5, C1, C3, C10, F4).

These are the tests that would have been prime-agent's own suite, had D19 not
replaced its runtime. Nothing here is mocked: each test spawns CPython, hands it
fd 3, and drives the protocol, because every property under test is a property
of the process boundary.

## Why guest stdout is coalesced into ~8 KiB frames

A frame per `write` is quadratic twice over. `print` issues two writes (the text
and the newline), and each becoming a `channel.send` means CPython 3.12's
`_SelectorSocketTransport.write` calls `get_write_buffer_size()` —
`sum(map(len, self._buffer))` over the pending deque — and a cell that never awaits
never lets the transport drain, so that sum grows with everything written so far.
The host then pays its own **~25 µs** of per-frame work.

Measured: `for i in range(10_000): print(i)` took **2.1-2.6 s**; buffered into
~8 KiB frames it is **~2 ms**.

## Why the frame buffer is a `bytearray` scanned from its tail

With `bytes` and `+=`, a multi-megabyte frame copies the whole buffer per 64 KiB
chunk and re-scans it for a newline. A **16 MiB snapshot spent 834 ms of 1160 ms**
doing exactly that; appending to a `bytearray` and searching from the previous
length makes both linear — **1160 ms -> 221 ms**.

## Two caps that stop at the cap

**No round trip through JSON in `_json_safe`.** `encode` is about to serialize the
frame anyway; doing it twice cost **2.8 ms against 1.2 ms for a 1 MiB result**.

**`_encode_value` never builds what it is about to discard.** `json.dumps` of a
1M-element list took **40 ms and `repr` another 37 ms — 77 ms to produce 64 KiB**,
and "end the cell with `df`" is exactly how models write cells.

## Why sends are serialized behind `_send_lock`

Without it, contention lands in the `OSError`/`ClosedResourceError` branch as
`BusyResourceError` and is reported as the child having exited — **eight concurrent
replies were enough to "kill" a perfectly healthy kernel**.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn

import anyio
import pytest
from runtime_helpers import namespace

from ph.orphans import process_alive
from ph.seams.code_runtime import CodeBinding, CodeBindingNamespace
from ph.seams.subprocess import scrub_env
from ph.testing import settled
from ph.tools.code_mode import CodeRunFailure, ToolCallError
from ph_rlm.kernel.manager import (
    MAX_FRAME_BYTES,
    RESET_NOTICE,
    Kernel,
    KernelLimits,
    frame_cap,
)
from ph_runtime.cell import MAGIC_HINT
from ph_runtime.protocol import FD_ENV, FRAME_BYTES_ENV, truncation_marker

pytestmark = pytest.mark.anyio

MakeKernel = Callable[..., Any]


def tools(**handlers: Callable[..., Any]) -> CodeBindingNamespace:
    """A `tools` namespace whose bindings are the handlers given.

    Deliberately not the real `DispatchBridge`: this file tests the *runtime's*
    half of C1 — that a call reaches a host closure and its answer reaches the
    program — and `test_governance.py` tests the pipeline's half.
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
    """C1's persistence, and the reason the seam demands snapshots (D6)."""
    kernel = await make_kernel()
    await kernel.run("import math\ncounter = 1\ndef bump(): return counter + 1", (), None)
    result = await kernel.run("counter = bump()\n(counter, math.floor(2.5))", (), None)
    assert result.value == [2, 2]


async def test_a_magic_is_a_syntax_error_that_names_the_governed_route(
    make_kernel: MakeKernel,
) -> None:
    """D2. The magic *was* the bypass, so the message is the tool, not an apology."""
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
    """D4: the cap holds and the marker is the one the host would have written."""
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

    `RLIMIT_CPU` is cumulative over the process, so the budget is re-armed at
    each run. Without that, this test would pass and the *next* cell would die
    on a limit this one spent.
    """
    kernel = await make_kernel(cpu_seconds=1)
    result = await kernel.run("while True:\n    pass", (), None)
    assert result.error is not None
    assert "CPU budget" in result.error
    # Re-armed: the kernel is still usable, which is the whole point.
    assert (await kernel.run("1 + 1", (), None)).value == 2


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
async def test_a_cpu_bomb_on_a_worker_thread_costs_the_cell_and_not_the_guest(
    make_kernel: MakeKernel,
) -> None:
    """The budget must cost one run, wherever the CPU was burned (F1).

    `SIGXCPU` was a raising `signal.signal` handler, and a raising handler
    unwinds from wherever the **main thread** happens to be. When the cell is on
    that thread the raise lands in the cell, which is the whole design. When the
    burn is on a worker the main thread is sitting in the loop's `select()`, so
    the exception came out of `asyncio`'s own internals and the guest exited —
    the namespace, the session's kernel and every later cell, for a limit that is
    supposed to end one run.

    The handler now asks where the cell is before it raises, and cancels the run
    task when the answer is "not here". Both routes report the same `cpu` error;
    what differs is what they are allowed to interrupt.
    """

    kernel = await make_kernel(cpu_seconds=1)

    # Under `fail_after` because the old failure is a *hang*, not a wrong
    # answer: the main thread died of the raise, and the worker it could not
    # interrupt kept the process alive — so the channel never reached EOF and
    # the host, which has no wall clock on a run, waited for a `done` frame no
    # one was left to send.
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
    # The guest is still serving — which is the regression. A restarted kernel
    # would say so in the next run's logs.
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
    """D5. No control-channel workaround: fd 3 is not the channel the run occupies."""

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
    """The case neither cooperative route can reach.

    A cell spinning in Python starves the guest's loop, so the `cancel` frame is
    never read and the `SIGINT` callback never runs. The host escalates. What is
    being asserted is the *honesty* of the outcome: the namespace is gone and the
    result says so, rather than the kernel wedging until the turn times out.
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
    # And the session continues: the next cell gets a fresh kernel and is told.
    revived = await kernel.run("'back'", (), None)
    assert revived.value == "back"
    assert RESET_NOTICE in revived.logs


async def test_a_cell_blocked_behind_a_large_reply_is_still_killed(
    make_kernel: MakeKernel,
) -> None:
    """The ladder must not wait on the channel it is trying to give up on (F2).

    One background call, a reply too big for the socket buffer, and a cell that
    then spins: the guest's loop is starved so it drains nothing, the host's
    write fills the buffer and parks holding `_send_lock`, and every later frame
    queues behind it. That is the state the stop ladder exists for — and the
    ladder went through the same lock. `_watch` asked for the cancel frame,
    blocked on the lock, and never returned to the loop that times the grace, so
    the escalation to `SIGKILL` was unreachable. `aclose` queued behind it too,
    which is a kernel that cannot even be shut down.

    The clock now starts before the ask and the ask runs in its own task, so the
    grace expires on schedule whatever the channel is doing. Under `fail_after`
    because the regression is a hang: without the fix nothing here ever returns.
    """

    from ph.cancel import CancelToken

    # Comfortably past a unix socket's buffer, so the reply cannot be handed off
    # in one write and the host is still mid-frame when the cancel lands.
    huge = "x" * (8 * 1024 * 1024)

    async def big(**_arguments: object) -> str:
        return huge

    bindings = tools(big=big)
    kernel = await make_kernel(namespaces=(bindings,), cpu_seconds=600, cancel_grace=0.3)
    token = CancelToken()
    program = (
        "import asyncio\n"
        # Issued from a background task, so the cell does not await the reply it
        # is about to stop reading.
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
    # And the session survives it, which is the reason the ladder has a last rung.
    revived = await kernel.run("'back'", (), None)
    assert revived.value == "back"


async def test_a_namespace_larger_than_one_frame_is_still_snapshotted(
    make_kernel: MakeKernel,
) -> None:
    """The per-variable cap and the per-frame cap have to be the same cap (F3).

    `max_snapshot_bytes` bounds each *value* the guest encodes, and the guest put
    every changed value in one `snapshot` frame — so three variables just under
    the cap made a frame three times over it. The host refuses an oversized
    frame, and rightly: a peer writing megabytes with no newline is how a hostile
    child would try to exhaust the reader (C10). But that made an ordinary
    namespace indistinguishable from an attack — the channel closed, the model
    was told the runtime had exited, and the namespace was gone with no mention
    of a snapshot anywhere in the account.

    The property is the *ratio* between the two caps, and it is now asserted at
    whatever `max_snapshot_bytes` the kernel booted with rather than by lowering
    the host's constant to meet it (M3): `frame_cap` derives the reader's ceiling
    from the limit, so a deployment that raises one raises the other. The
    companion below pins that derivation on its own.
    """
    kernel = await make_kernel(max_snapshot_bytes=200 * 1024)

    result = await kernel.run(
        "\n".join(f"v{index} = 'x' * 100_000" for index in range(3)), (), None
    )

    assert result.error is None, result.error
    # The kernel is still the one that ran the cell: a refused frame closed the
    # channel, and the next run would have said so.
    revived = await kernel.run("v0 == 'x' * 100_000 and v2 == 'x' * 100_000", (), None)
    assert revived.value is True
    assert RESET_NOTICE not in revived.logs, "the namespace was lost to its own snapshot"


async def test_a_task_a_cell_left_behind_cannot_call_into_the_next_run(
    make_kernel: MakeKernel,
) -> None:
    """A run owns its calls, and the namespace outliving the cell is not the same
    thing as its tasks outliving it (F4).

    The guest is persistent, so the loop keeps running between programs: a task
    the cell started with `asyncio.create_task` and never awaited goes on calling
    bindings after its run has settled. The host had no way to tell — a `call`
    frame said which *call* it was, never which *program* — so the stray was
    served against whichever run was open next. It spent that run's call budget,
    was recorded under its dispatch id, and had its approval decided for a turn
    that had nothing to do with it.

    The first cell leaves a task looping on a binding; the second cell calls the
    same binding once. What must reach the second run is its own call and nothing
    else.
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
        # Swallowing the cancel on purpose: a well-behaved task is ended by the
        # guest at settle, and this test is about the one that is not. It is what
        # makes the *host's* refusal the thing under test rather than the guest's
        # cancellation — and it is the case the run stamp exists for.
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

    # Long enough that a surviving task would have called many times over.
    await anyio.sleep(0.2)
    seen.clear()
    second = await kernel.run("await tools.note(tag='mine')", (bindings,), None)

    assert second.error is None, second.error
    assert seen == ["mine"], f"a settled run's task reached the next one: {seen}"


# --------------------------------------------------------------- the boundary --


async def test_a_cell_that_kills_the_process_costs_the_namespace_not_the_session(
    make_kernel: MakeKernel,
) -> None:
    """D1. The namespace is unrecoverable; saying so is what stops a wasted turn."""
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
    """C3, the divergence from dsh that the whole containment argument rests on.

    A program that can `except` a refusal can route around it — retry with a
    different path, fall back to `subprocess`. So the refusal is not an
    exception the cell is offered; it ends the run, and the tool call fails with
    the refusal in it.
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
    """C3's actual enforcement, which is the host's and not the guest's.

    `RunStopped` makes a *well-behaved* cell unwind, and this cell is not one: it
    swallows `BaseException` and then writes a file with raw `pathlib`, which no
    waterfall can reach. Reporting the refusal afterwards would be a tool call
    that failed on paper while the side effect happened anyway — so the host
    fires the run-scoped abort the plan's C3 row names, and the write never runs.

    The `sleep` is the window: without the abort the write lands, and this test
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
    # And the session survives the abort: only that cell was ended.
    assert (await kernel.run("'still here'", (namespace,), None)).value == "still here"


async def test_a_failed_call_is_the_programs_to_handle(make_kernel: MakeKernel) -> None:
    """C3's other half: a *failure* keeps dsh's semantics and stays catchable."""

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


async def test_an_unknown_binding_says_what_exists(make_kernel: MakeKernel) -> None:
    namespace = tools(read=_ok)
    kernel = await make_kernel(namespaces=(namespace,))
    result = await kernel.run("await tools.nonexistent()", (namespace,), None)
    assert result.error is not None
    assert "read" in result.error, "the message names the available bindings"


async def test_rlm_rejects_an_unknown_keyword_in_the_cell(make_kernel: MakeKernel) -> None:
    """Prime-agent's contract: unknown kwargs fail loudly (§6.0)."""
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
    """The child is the one `scrub_env` was written for.

    Its docstring in `ph.seams.subprocess` says so: "a child runs code the model
    wrote, so it does not inherit `*KEY*`". A cell that can read `os.environ` can
    print a provider credential into its own output, which is then logged — and
    the credential was never the model's to see.
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
    """F4: a child that exited while the parent lives and is never reaped."""
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
    """The process boundary exists so model code cannot reach the harness.

    A guest module importing `ph` would put it back inside — and would also make
    the managed venv need `ph-core`, which is the dependency the venv exists to
    avoid.
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
    """The reason the stop ladder's clock is a sibling task.

    `_pump` used to sit in `move_on_after(POLL_SECONDS)` so it could ask
    between reads whether the caller had canceled. The cost was not one scope
    per socket read but **one per frame**: `_recv_line` returns straight out of
    its buffer whenever a frame is already there, and a 64 KiB read of a chatty
    cell carries many. Measured at 37.5 ms against 13.8 ms over 3,000 `log`
    frames — 7.9 us each — for a deadline almost never reached.

    Counted rather than asserted structurally: "does `_pump` contain a
    `move_on_after`" is a fact about the source, and what matters is that a run
    lasting many poll intervals interrupts the read **no** times.

    **Not a guard against issue 58**, which is what this was first written as.
    The mechanism there — `_RawSocketMixin._wait_until_readable` registering
    `f.set_result` as the reader callback and removing the reader in a
    done-callback a loop iteration later, so a canceled wait can still be
    fired on a canceled future — belongs to `UNIXSocketStream` and
    `connect_unix`, i.e. the daemon socket. `_recv_line` calls the *free*
    `anyio.wait_readable`, a different implementation that catches
    `InvalidStateError` and removes the reader synchronously inside the
    callback. Canceling it was always safe.

    Sabotage: put the `move_on_after` back around the read, and `canceled`
    counts roughly `duration / POLL_SECONDS`.
    """

    from ph.cancel import POLL_SECONDS

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
    # Long enough that the old poll would have fired many times over.
    slept = 12 * POLL_SECONDS
    result = await kernel.run(f"import asyncio\nawait asyncio.sleep({slept})\n'done'", (), None)

    assert result.value == "done", result.error
    # The positive control: `canceled == 0` also passes if the patch is never
    # reached at all, which is what a future move of `_recv_line` onto a stream
    # would do silently.
    assert waits > 0, "the patched readiness wait was never reached; the test proves nothing"
    assert canceled == 0, (
        f"the frame read was interrupted {canceled} times during one run; "
        "the poll is back on the read path and the per-frame scope with it"
    )


def test_a_run_records_how_it_ended_once_and_the_first_writer_wins() -> None:
    """Three tasks end a run and they race by construction.

    `_watch` can kill while `_pump` is blocked mid-frame, and `_teardown`
    closing the socket is itself what wakes the pump — so a `done` frame
    already buffered when the kill landed used to overwrite "the runtime was
    killed" with the program's own value. Each writer assigned `error` and
    `settled` directly and only `_on_closed` checked first, which made the
    outcome depend on three call sites ordering themselves around each other.

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
    """The third way this socket says it is gone, which was not caught.

    `sock.send`/`recv` on a closed socket raise `OSError`, and `notify_closing`
    raises `ClosedResourceError` into a waiter already parked — both handled.
    But `wait_writable`/`wait_readable` *entered* with a closed socket raise
    `ValueError("Invalid file descriptor: -1")`, because `close()` sets the
    fileno to `-1` and the loop refuses to register it. A frame bigger than the
    socket buffer goes round that loop many times, so a teardown landing between
    two turns hits exactly it.

    Found as a macOS-only CI failure of the test above, where an 8 MB reply is
    being written while the stop ladder kills the guest. Linux passed, and the
    difference is buffer sizes and wakeup order rather than anything the code
    decides — which is why this pins the *shape* with the socket closed
    deterministically, instead of leaving it to a race that reproduces on one
    platform.
    """

    from ph_rlm.kernel.manager import _CHANNEL_GONE

    kernel = await make_kernel()
    assert (await kernel.run("1 + 1", (), None)).value == 2

    # Closed underneath both waits, with nothing parked on it: this is the
    # re-entry case, and `fileno()` is already `-1`.
    sock = kernel._sock
    assert sock is not None
    sock.close()
    assert sock.fileno() == -1

    for wait in (anyio.wait_writable, anyio.wait_readable):
        with pytest.raises(_CHANNEL_GONE):
            await wait(sock)

    # And the kernel reports it as the channel being gone rather than raising:
    # a send finds it closed, settles the run, and the session still restarts.
    outcome = await kernel.run("2 + 2", (), None)
    assert outcome.error is not None or outcome.value == 4
    assert (await kernel.run("3 + 3", (), None)).value == 6


async def test_a_path_argument_reaches_the_host_as_a_path(make_kernel: MakeKernel) -> None:
    """F7 — `default=repr` is silently lossy for the two types a cell passes most.

    `tools.read(Path("notes.md"))` arrived as the string
    `"PosixPath('notes.md')"`, and a `datetime` as
    `"datetime.datetime(2026, 9, 20, 0, 0)"`. Neither is an error anywhere: the
    tool receives a string where the model wrote a value and either refuses it
    with a message about a path that does not exist, or uses it.

    A type with no JSON form still falls back to `repr`, because the guest does
    not know the tool's schema and refusing an argument the tool would have
    accepted is the worse error — but `repr` is then the thing that makes the
    tool's own refusal readable.
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
    """J9 — no controlling terminal for the process that runs model-authored code.

    Why that matters is on `_start`'s `open_process` call. Asserted on the guest
    rather than on the sandbox argv because the property is "every guest is a
    session leader", which bwrap's `--new-session` gives on one backend of
    three.

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
    """J9's other half: the session the guest got has to be killed as a group.

    Why the two belong together is on `ph.orphans.signal_group`. Measured here
    on the path that has no other backstop — a clean `aclose`, where the guest
    exits by itself and only the sweep is left looking.

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

    # Polled, not asserted outright: the grandchild is killed inside `aclose`
    # and lingers for a tick as a zombie awaiting reparenting, which `kill(0)`
    # still answers `True` for.
    await settled(lambda: not process_alive(grandchild), "the cell's subprocess to be reaped")


# ------------------------------------------------- the loop's own clock (M2) --


async def test_a_run_the_guest_will_never_finish_is_repaired_not_waited_on(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2 — the host waits for `done` and nothing else was watching.

    Both rungs of the stop ladder start only because somebody pressed stop, so a
    run that is never canceled and never settles had no clock at all, and the
    host waits on that frame with no wall clock of its own.

    **The remedy is a repair, not a kill.** The probe names the run the host is
    waiting on, so the guest — which is the only party that knows whether the
    frame is still coming — settles it through the same `_send_done` every
    other terminal path takes. The namespace survives, which is the whole point:
    the predicate establishes that the guest is *alive and healthy*, and killing
    something on that evidence costs everything it was holding for a fault that
    is one missing frame.

    **The wedge is staged at the host**, by losing the run's *first* terminal
    frame. Every input is then what a guest whose `done` never arrived
    produces: the run unsettled, the loop free, and nothing owed. Only the
    first — the repair is a `done` too, and a host that dropped every one could
    not be repaired by anything. Breaking the guest's own teardown would
    reproduce one cause rather than the shape they share, and those causes
    leave the run still *owed*, where `_answer_if_owed` already answers.

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
    # The namespace is still there, which a kill would have cost.
    assert (await kernel.run("'alive'", (), None)).value == "alive"
    assert RESET_NOTICE not in (await kernel.run("1", (), None)).logs


async def test_a_cell_that_starves_the_loop_is_measured_and_not_killed(
    make_kernel: MakeKernel,
) -> None:
    """The reading the probe is *not* allowed to make (M2).

    A cell spinning in straight-line Python starves the reader task that answers
    a probe — the same starvation that makes `cancel` and `SIGINT` both miss it.
    That is what a busy cell looks like, and a busy cell is entitled to be busy:
    `CpuBudget` bounds it, this clock does not. So the probe goes unanswered for
    longer than `idle_grace` and the run still finishes on its own.

    What the unanswered probe buys is the gauge — `phern doctor` reports the
    worst round trip and the stalls, which is how a person sees *where* a slow
    agent's time is going rather than only that it was slow.

    Sabotage: escalate on an outstanding probe instead of on `orphan_since` and
    this cell is killed mid-computation.
    """
    kernel = await make_kernel(probe_seconds=0.05)

    with anyio.fail_after(30):
        result = await kernel.run("sum(range(60_000_000))", (), None)

    assert result.error is None, result.error
    # The closed form, not a second `sum(range(...))`: recomputing it here
    # cost 0.58 s of the test's 1.0 s to assert a number Gauss had.
    assert result.value == 59_999_999 * 60_000_000 // 2
    assert kernel.loop_stalls > 0, "a cell that never yielded answered every probe on time"


async def test_the_probe_reports_how_far_behind_a_guest_loop_is(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gauge, on a loop that is *not* starved — the ordinary reading.

    A cell awaiting inside asyncio leaves the reader task free, so every probe is
    answered promptly. That is the baseline a person needs in order to read the
    busy case as unusual, which is why this is measured continuously rather than
    only when something already looks wrong.

    **And at the cadence the knob names.** `_watch` ticks twenty times a second;
    `probe_sent` is cleared by the answer, so gating the next probe on that
    alone would put a frame each way on every tick — twenty times the traffic
    `probe_seconds` says it costs, on every open run.

    Sabotage: drop the `probed_at` gate from `_probe` and the count is the poll
    rate rather than the probe rate.
    """
    kernel = await make_kernel(probe_seconds=0.2)
    sent: list[int] = []
    real = Kernel._ping

    async def counting(self: Kernel, run_id: int, probe: int) -> None:
        sent.append(probe)
        await real(self, run_id, probe)

    monkeypatch.setattr(Kernel, "_ping", counting)
    await kernel.run("import asyncio\nawait asyncio.sleep(1.0)", (), None)

    assert kernel.loop_worst is not None, "no probe was answered"
    assert kernel.loop_worst < 0.2, f"an idle loop answered slowly: {kernel.loop_worst}"
    assert kernel.loop_stalls == 0, "an idle loop left a probe unanswered"
    # A second of cell against a fifth of a second of cadence. Bounded loosely
    # on both sides: the exact count is the scheduler's, the *order* is the
    # claim, and the poll rate it must not be is 20/s.
    assert 2 <= len(sent) <= 10, f"{len(sent)} probes in a second at probe_seconds=0.2"


def test_the_readers_ceiling_follows_the_limit_the_kernel_booted_with() -> None:
    """M3 — the sizing the constant claimed and nothing enforced.

    `MAX_FRAME_BYTES`'s docstring said it was "sized to hold a `maxSnapshotBytes`
    payload with base64 and JSON overhead", and at the shipped 16 MiB that was
    true. It is a claim about a *ratio* between two numbers, one of which a
    deployment configures — so a profile raising `maxSnapshotBytes` past about a
    third of the constant made it false, and what that costs is not a rejected
    frame but a lost namespace: the host refuses the oversized snapshot, closes
    the channel, and the model is told the runtime exited.

    Asserted on the derivation rather than by round-tripping eighty megabytes,
    which is what the sibling above used to need a monkeypatch to avoid.

    Sabotage: return `MAX_FRAME_BYTES` unconditionally and the raised limit gets
    a ceiling below its own payload.
    """
    shipped = KernelLimits()
    assert frame_cap(shipped) == MAX_FRAME_BYTES, "the shipped limit needs no more than the floor"

    raised = KernelLimits(max_snapshot_bytes=256 * 1024 * 1024)
    assert frame_cap(raised) > raised.max_snapshot_bytes * 4 // 3, (
        "a base64 payload at the configured limit would not fit the reader's ceiling"
    )
    # And the floor still holds underneath a deployment that lowers the limit.
    assert frame_cap(KernelLimits(max_snapshot_bytes=1024)) == MAX_FRAME_BYTES


async def test_the_guest_reads_with_the_hosts_frame_cap(
    make_kernel: MakeKernel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F8 — both readers from one expression, handed over at spawn.

    The host's ceiling followed `maxSnapshotBytes` while the guest's stayed at a
    fixed 64 MiB, so a raised limit lost the namespace on restore. `start` puts
    `frame_cap` in the child's environment, the way the descriptor crosses —
    asserted on a real spawn, which the guest then boots from.

    Sabotage: drop the `FRAME_BYTES_ENV` entry from `start`'s environment and
    this finds none.
    """
    handed: dict[str, str] = {}

    def recording(*, extra: Mapping[str, str]) -> dict[str, str]:
        handed.update(extra)
        return scrub_env(extra=extra)

    monkeypatch.setattr("ph_rlm.kernel.manager.scrub_env", recording)
    raised = 256 * 1024 * 1024

    # `start` raises unless the guest answers `boot`, so returning is the proof
    # that it came up reading with the number it was handed.
    await make_kernel(max_snapshot_bytes=raised)

    assert handed[FRAME_BYTES_ENV] == str(frame_cap(KernelLimits(max_snapshot_bytes=raised)))


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_CPU is POSIX")
async def test_a_finished_cell_cannot_leave_a_thread_burning_a_core(
    make_kernel: MakeKernel,
) -> None:
    """M1 — the budget was switched off rather than down, and nothing re-armed it.

    `relax_cpu_budget` disarms at the first `SIGXCPU`, which it must: the limit
    is cumulative and Linux re-delivers every CPU-second, so a second signal
    lands in the guest's teardown and costs the `done` frame. What that left is
    the gap this closes. A `to_thread` worker reaches neither cooperative route
    — a cancel does not reach a thread, and the raise route only fires where the
    cell is — so the run reported `cpu`, the worker carried on, and with the
    budget off no second signal was ever coming. A core, until the session ended.

    A standing `idle_cpu_seconds` is armed the moment a run stops owning the
    process, and a breach with no run open ends the runtime: nothing legitimate
    spends it, because a background thread a cell left on purpose is *waiting*
    and waiting costs no CPU.

    The namespace is gone afterwards, which is the honest price and is asserted
    — it was already holding a thread nobody could stop. The guest's reason
    reaches the model on the way out, through the stderr the host captures.

    Sabotage: drop the `arm_cpu_budget` from `_send_done` and this waits out
    `fail_after` with the thread still running.
    """
    kernel = await make_kernel(cpu_seconds=30, idle_cpu_seconds=1)

    # Finishes immediately; the worker it starts is not reachable by any cancel.
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

    # The next cell meets the death, and the guest's own sentence rides out with
    # it: the stderr the host captured is what tells a person *why* the runtime
    # went, rather than leaving them with an unexplained exit.
    met = await kernel.run("'alive'", (), None)
    assert met.error is not None
    assert "burning CPU" in met.logs, met.logs

    # And the one after that is a fresh namespace, said out loud.
    revived = await kernel.run("'alive'", (), None)
    assert revived.value == "alive"
    assert RESET_NOTICE in revived.logs, "the namespace went and nobody said so"
