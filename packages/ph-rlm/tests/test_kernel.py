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

## Why sends are serialised behind `_send_lock`

Without it, contention lands in the `OSError`/`ClosedResourceError` branch as
`BusyResourceError` and is reported as the child having exited — **eight concurrent
replies were enough to "kill" a perfectly healthy kernel**.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from runtime_helpers import namespace

from ph.seams.code_runtime import CodeBinding, CodeBindingNamespace
from ph.tools.code_mode import CodeRunFailure, ToolCallError
from ph_rlm.kernel.manager import RESET_NOTICE
from ph_runtime.cell import MAGIC_HINT
from ph_runtime.protocol import FD_ENV, truncation_marker

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
    import anyio

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
    import anyio

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
    import anyio

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

    async def refused(**_arguments: object) -> Any:  # noqa: ANN401
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

    async def refused(**_arguments: object) -> Any:  # noqa: ANN401
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

    async def failing(**_arguments: object) -> Any:  # noqa: ANN401
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


async def _ok(**_arguments: object) -> Any:  # noqa: ANN401
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

    `_pump` used to sit in `move_on_after(_CANCEL_POLL_SECONDS)` so it could ask
    between reads whether the caller had cancelled. The cost was not one scope
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
    done-callback a loop iteration later, so a cancelled wait can still be
    fired on a cancelled future — belongs to `UNIXSocketStream` and
    `connect_unix`, i.e. the daemon socket. `_recv_line` calls the *free*
    `anyio.wait_readable`, a different implementation that catches
    `InvalidStateError` and removes the reader synchronously inside the
    callback. Cancelling it was always safe.

    Sabotage: put the `move_on_after` back around the read, and `cancelled`
    counts roughly `duration / _CANCEL_POLL_SECONDS`.
    """
    import anyio

    from ph_rlm.kernel.manager import _CANCEL_POLL_SECONDS

    original = anyio.wait_readable
    cancelled = 0
    waits = 0

    async def counting(obj: Any) -> Any:  # noqa: ANN401
        nonlocal cancelled, waits
        waits += 1
        try:
            return await original(obj)
        except anyio.get_cancelled_exc_class():
            cancelled += 1
            raise

    monkeypatch.setattr(anyio, "wait_readable", counting)

    kernel = await make_kernel()
    # Long enough that the old poll would have fired many times over.
    slept = 12 * _CANCEL_POLL_SECONDS
    result = await kernel.run(f"import asyncio\nawait asyncio.sleep({slept})\n'done'", (), None)

    assert result.value == "done", result.error
    # The positive control: `cancelled == 0` also passes if the patch is never
    # reached at all, which is what a future move of `_recv_line` onto a stream
    # would do silently.
    assert waits > 0, "the patched readiness wait was never reached; the test proves nothing"
    assert cancelled == 0, (
        f"the frame read was interrupted {cancelled} times during one run; "
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
