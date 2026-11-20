"""The two halves of the boot handshake must agree on what a number is (D7).

This was found on macOS on 2026-09-07, during verification of the Seatbelt
backend. Every kernel start waited for the whole `boot_timeout` and reported "did
not report ready" with nothing to quote, although the guest had answered within
50 ms. macOS refuses `RLIMIT_AS` (`ValueError: current limit exceeds maximum
limit`). The guest reported the soft limit in force, which was `RLIM_INFINITY`,
2**63-1. The lossless-integer rule of the host codec was then a `parse_int` hook
over the whole line. The codec dropped the `boot-ack` as junk, and the host
continued to wait for a frame that had already come.

The root cause is in the codec, and it is fixed there: the bound applies to the
`int` fields that the spec declares, and to nothing that it declines to inspect
(`test_codec.py`). The tests here check three things that are true regardless of
that fix:

* The guest reports `None` for a limit that it could not apply. A report is for
  reading, and `RLIM_INFINITY` is not a limit.
* The host treats anything that is not `boot-ack` or `fault` as a fault to
  report, whether it is unreadable *or* only unexpected. Nothing model-written
  has run before `boot-ack`, so the tolerance of C10 ("junk is skipped, the peer
  is hostile") is not owed yet. A loop on such a frame is how a silent
  `boot_timeout` happens.
* `phern doctor` reports the limits that the guests actually applied, not the
  limits that the host asked for. On macOS those differ. To claim the request
  would be the failure of E1, one layer down from the tier table.
"""

from __future__ import annotations

import json
import os
import resource
import socket
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from ph.orphans import OrphanJournal
from ph_rlm.kernel import codec
from ph_rlm.kernel.manager import Kernel, KernelLimits, PythonCodeRuntime
from ph_rlm.kernel.venv import RuntimeEnvironment, resolve_interpreter
from ph_runtime.limits import apply_limits
from ph_runtime.protocol import PROTOCOL_VERSION

pytestmark = pytest.mark.anyio

INFINITY = resource.RLIM_INFINITY


def _refuse(*_args: object) -> None:
    raise ValueError("current limit exceeds maximum limit")


def _boot_ack(limits: dict[str, Any]) -> bytes:
    frame = {"type": "boot-ack", "protocol": PROTOCOL_VERSION, "python": "3.12", "limits": limits}
    return json.dumps(frame).encode() + b"\n"


def test_a_limit_the_platform_refuses_is_reported_as_none_and_the_frame_stays_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (INFINITY, INFINITY))
    monkeypatch.setattr(resource, "setrlimit", _refuse)

    applied = apply_limits(address_space_bytes=2**31)

    assert applied["addressSpaceBytes"] is None, "not in force, and said so"
    assert codec.decode(_boot_ack(applied)) is not None, "and the host can read the report"


def test_a_finite_soft_limit_already_in_force_is_still_the_number_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The promise of the `apply_limits` docstring ("the host logs the number that
    is in force") holds where there is a number. A refused `setrlimit` under an
    existing finite soft limit reports that limit, not `None`."""
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (2**30, 2**30))
    monkeypatch.setattr(resource, "setrlimit", _refuse)

    assert apply_limits(address_space_bytes=2**31)["addressSpaceBytes"] == 2**30


async def test_an_unreadable_first_frame_is_a_fault_not_a_silence(tmp_path: Path) -> None:
    """The test drives `_await_boot_ack` over a socketpair, with no guest spawned.
    It sends a `boot-ack` in a shape that this host does not read: `protocol` as
    a string, which is the kind of drift that D7 is about. The host answers with
    a fault that quotes the line, and does not wait for `boot_timeout`."""
    kernel = Kernel(
        namespace="agent-test",
        environment=resolve_interpreter(cache=tmp_path, mode="host"),
        limits=KernelLimits(),
        journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
    )
    host, guest = socket.socketpair()
    try:
        host.setblocking(False)
        kernel._sock = host
        drifted = {"type": "boot-ack", "protocol": "two", "python": "3.12", "limits": {}}
        guest.sendall(json.dumps(drifted).encode() + b"\n")

        with anyio.fail_after(5):
            fault = await kernel._await_boot_ack()

        assert fault is not None
        assert "could not be read as protocol" in fault
        assert '"protocol": "two"' in fault, "the offending line is quoted"
    finally:
        guest.close()
        host.close()


async def test_a_frame_that_is_not_boot_ack_before_ready_is_also_a_fault(tmp_path: Path) -> None:
    """The same rule for this phase, for the other way in which a guest can break
    it.

    An unreadable line was made fatal, but a *readable* frame that is not
    `boot-ack` still fell through, and the loop continued. A `log` or `done` that
    arrived early therefore reproduced exactly the silent `boot_timeout` that
    this module exists to end. There is one rule for both shapes.
    """
    kernel = Kernel(
        namespace="agent-test",
        environment=resolve_interpreter(cache=tmp_path, mode="host"),
        limits=KernelLimits(),
        journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
    )
    host, guest = socket.socketpair()
    try:
        host.setblocking(False)
        kernel._sock = host
        guest.sendall(
            json.dumps({"type": "log", "stream": "stdout", "text": "hi"}).encode() + b"\n"
        )

        with anyio.fail_after(5):
            fault = await kernel._await_boot_ack()

        assert fault is not None and "'log' before reporting ready" in fault
    finally:
        guest.close()
        host.close()


async def test_doctor_reports_the_limits_the_guests_applied_not_the_ones_asked_for() -> None:
    """E1, one layer down from the tier table. A row that printed the *request*
    claimed an address-space limit that macOS does not enforce. The weakest live
    kernel decides, as it already does in the confinement row beside this one."""
    runtime = PythonCodeRuntime(
        limits=KernelLimits(), journal=OrphanJournal(path=Path("processes.jsonl")), cache=Path(".")
    )
    asked = dict(runtime.describe())["per-child limits"]
    assert "GiB address space" in asked and "requested" in asked, (
        "with nothing started, the request is all there is — and it says so"
    )

    applied = Kernel(
        namespace="a",
        environment=resolve_interpreter(cache=Path("."), mode="host"),
        limits=KernelLimits(),
        journal=OrphanJournal(path=Path("processes.jsonl")),
    )
    applied.applied_limits = {"addressSpaceBytes": 2**31, "cpu": "per-run"}
    refused = Kernel(
        namespace="b",
        environment=applied.environment,
        limits=KernelLimits(),
        journal=applied.journal,
    )
    refused.applied_limits = {"addressSpaceBytes": None, "cpu": "per-run"}

    runtime._kernels = {"a": applied}
    assert "GiB address space" in dict(runtime.describe())["per-child limits"]

    runtime._kernels = {"a": applied, "b": refused}
    row = dict(runtime.describe())["per-child limits"]
    assert "not applied" in row and "GiB address space" not in row, row


def test_the_design_document_names_the_version_the_guest_declares() -> None:
    """I1: DESIGN.md §1 stated the `PROTOCOL_VERSION` of the kernel as prose.

    The statement was wrong for some time. The document said 2 while the guest
    declared 1. The `2` that a reader would have found with grep is the *wire*
    version of `ph_app.protocol`, which numbers a completely different protocol.
    There were two constants with one name, and one of them was in a paragraph
    that nothing checks.

    The bump has landed since then, so this test is the guard and not the fix.
    The number that a person reads and the number that the host negotiates with
    cannot drift apart again unless a test reports it. The assertion is made
    against the text of the document, because that is the copy with no compiler.
    """
    design = (Path(__file__).resolve().parents[3] / "DESIGN.md").read_text(encoding="utf-8")

    assert f"`PROTOCOL_VERSION = {PROTOCOL_VERSION}`" in design, (
        "DESIGN.md §1 names a kernel protocol version the guest does not declare"
    )


async def test_two_first_runs_on_one_namespace_build_one_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F6: `environment()` suspends, and two first runs both got past the read.

    The namespace *is* the agent id, and a fan-out of parallel tool calls on one
    agent is what Code Mode is for. Two first runs on one namespace are therefore
    ordinary. Both built a `Kernel`, and the second overwrote the first in
    `_kernels`. Both then spawned a guest on their first cell, and
    `close_namespace` could reach only the one that was still in the dict. The
    other ran until the process exited, and held a socket, a pid, and whatever
    the cell had open.

    The test drives `_acquire`, and holds `environment()` open at exactly the
    await that suspends in production. A cold cache that shells out to `uv`
    takes seconds, and that time is the window.

    Sabotage: drop `async with self._acquire_lock:` and `environment()` is
    entered twice, returning two different kernels.
    """
    runtime = PythonCodeRuntime(
        limits=KernelLimits(),
        journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
        cache=tmp_path,
    )
    resolved = resolve_interpreter(cache=tmp_path, mode="host")
    entered = 0
    reached = anyio.Event()
    finish = anyio.Event()

    async def held(_runtime: PythonCodeRuntime) -> RuntimeEnvironment:
        nonlocal entered
        entered += 1
        reached.set()
        await finish.wait()
        return resolved

    monkeypatch.setattr(PythonCodeRuntime, "environment", held)

    built: list[Kernel] = []

    async def acquire() -> None:
        built.append(await runtime._acquire("a", "a"))

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(acquire)
        await reached.wait()  # the first is suspended inside `environment()`
        tasks.start_soon(acquire)
        await anyio.sleep(0)  # and the second has reached the lock
        finish.set()

    assert entered == 1, "the interpreter was resolved twice for one namespace"
    assert built[0] is built[1], "two kernels were built and one became unreachable"
    assert list(runtime._kernels) == ["a"]


class _Pipe:
    """The pipe of a child. The child cannot write past it until the host reads
    it.

    The real failure is a blocking `write(2)`: a guest that puts more than the
    pipe buffer on fd 1 or fd 2 before it sends `boot-ack` stops there. The pipe
    is modeled, and not reproduced with an OS pipe. A real pipe can be unblocked
    only from a worker thread, and a thread that is blocked in `read` cannot be
    canceled. A host that failed this test would then hang the test and not fail
    it, and that is the one outcome that a regression test must not have.

    The model keeps the causal chain that matters. `settled` is set only after
    the host has consumed the stream, and the child sends `boot-ack` only after
    `settled` is set. A host that does not read this pipe never gets the
    `boot-ack`.

    `chunks=0` with no `settled` is the other pipe in the same test. It is open
    and silent, and it is never the reason why boot finishes.
    """

    def __init__(self, settled: anyio.Event | None = None, *, chunks: int = 8) -> None:
        self._settled = settled
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._pump()

    async def _pump(self) -> AsyncIterator[bytes]:
        for _ in range(self._chunks):
            yield b"x" * 4096
        if self._settled is not None:
            self._settled.set()
        # The pipe stays open, like the pipe of a live child. Boot ends when this
        # reader is canceled, not when the pipe runs out.
        await anyio.sleep(3600)


@pytest.mark.parametrize("pipe", ["stderr", "stdout"])
async def test_a_child_that_fills_a_pipe_before_acking_still_boots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipe: str
) -> None:
    """F7(b): during boot, nothing consumed fd 1 or fd 2.

    `_drain` runs only for the duration of a *run*. A guest that writes more than
    the pipe buffer before `boot-ack` therefore blocks on that write and never
    sends it. The host then reported "did not report ready within Ns" for a child
    that was ready and blocked, and quoted whatever fit in one late read. That is
    the reverse of the truth, and it sends whoever reads it to look at the guest.

    The test is parametrized over both pipes, because `_boot_said` quotes both,
    and a wrapper that explains itself on stdout names the same failure. If the
    host drained only the pipe that happened to be tested, the other would still
    deadlock.

    Sabotage: drop either `start_soon(self._collect_boot_noise, ...)` line and
    the matching case waits out `boot_timeout` and raises `KernelDied`.
    """
    settled = anyio.Event()
    opened = anyio.Event()
    held: dict[str, Any] = {}

    async def fake_open_process(*_args: object, **kwargs: object) -> object:
        passed = kwargs["pass_fds"]
        assert isinstance(passed, tuple)
        (child_fd,) = passed
        held["ack"] = os.dup(child_fd)
        streams = {"stdout": _Pipe(chunks=0), "stderr": _Pipe(chunks=0)}
        streams[pipe] = _Pipe(settled)
        held["process"] = SimpleNamespace(pid=None, returncode=None, **streams)
        opened.set()
        return held["process"]

    monkeypatch.setattr(anyio, "open_process", fake_open_process)

    kernel = Kernel(
        namespace="a",
        environment=resolve_interpreter(cache=tmp_path, mode="host"),
        limits=KernelLimits(),
        journal=None,
        boot_timeout=5.0,
    )

    async def ack_once_the_host_has_read_the_pipe() -> None:
        await opened.wait()
        await settled.wait()
        os.write(held["ack"], _boot_ack({}))

    try:
        with anyio.fail_after(30):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(ack_once_the_host_has_read_the_pipe)
                await kernel.start(namespaces=[])
        assert kernel._alive, "the child acked but the host did not come up"
        assert kernel._boot_noise, f"{pipe} was not read while the host waited"
    finally:
        kernel._process = None
        if kernel._sock is not None:
            kernel._sock.close()
            kernel._sock = None
        if "ack" in held:
            with suppress(OSError):
                os.close(held["ack"])
