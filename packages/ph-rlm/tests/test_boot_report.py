"""The boot handshake's two halves have to agree on what a number is (D7).

Found on macOS on 2026-09-07, while verifying the Seatbelt backend: every kernel
start waited out `boot_timeout` and reported "did not report ready" with nothing to
quote — and the guest had answered within 50 ms. macOS refuses `RLIMIT_AS`
(`ValueError: current limit exceeds maximum limit`), the guest reported the soft
limit in force — `RLIM_INFINITY`, 2**63-1 — and the host's codec, whose
lossless-integer rule was then a `parse_int` hook over the whole line, dropped the
`boot-ack` as junk and kept waiting for a frame that had already come.

The root cause is the codec's, and is fixed there: the bound applies to the `int`
fields the spec declares and to nothing it declines to inspect (`test_codec.py`).
Three things pinned here are true regardless of it. The guest reports `None` for a
limit it could not apply, because a report is for reading and `RLIM_INFINITY` is
not a limit. The host treats anything that is not `boot-ack` or `fault` as a fault
to report — unreadable *or* merely unexpected — because nothing model-written has
run before `boot-ack`, so C10's "junk is skipped, the peer is hostile" tolerance is
not yet owed and looping on it is how a silent `boot_timeout` happens. And `ph
doctor` reports the limits the guests actually applied, not the ones the host asked
for, since on macOS those differ and claiming the request would be E1's failure one
layer down from the tier table.
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
    """The docstring's promise — "the host logs the number in force" — holds where
    there is one: a refused `setrlimit` under an existing finite soft limit reports
    that limit, not `None`."""
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (2**30, 2**30))
    monkeypatch.setattr(resource, "setrlimit", _refuse)

    assert apply_limits(address_space_bytes=2**31)["addressSpaceBytes"] == 2**30


async def test_an_unreadable_first_frame_is_a_fault_not_a_silence(tmp_path: Path) -> None:
    """Driven through `_await_boot_ack` over a socketpair, with no guest spawned: a
    `boot-ack` in a shape this host will not read — `protocol` as a string, the
    kind of drift D7 is about — and the host's answer to it is a fault that quotes
    the line rather than a wait for `boot_timeout`."""
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
    """The same phase rule over the other way a guest can break it.

    An unreadable line was made fatal and a *readable* frame that is not `boot-ack`
    was still fallen through and re-looped — so a `log` or `done` arriving early
    reproduced exactly the silent `boot_timeout` this module exists to end. One
    rule, both shapes.
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
    """E1, one layer down from the tier table: a row that printed the *request*
    claimed an address-space bound macOS does not enforce. The weakest live kernel
    wins, as the confinement row beside it already does."""
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
    """I1 — DESIGN.md §1 stated the kernel's `PROTOCOL_VERSION` as prose.

    It was wrong for a while: the document said 2 while the guest declared 1,
    and the `2` a reader would have found by grepping is `ph_app.protocol`'s
    *wire* version, which numbers a different protocol entirely. Two constants
    with one name, one of them in a paragraph nothing checks.

    The bump has since landed, so this is the guard rather than the fix: the
    number a person reads and the number the host negotiates with cannot drift
    apart again without a test saying so. Asserted against the document text
    because that is the copy with no compiler.
    """
    design = (Path(__file__).resolve().parents[3] / "DESIGN.md").read_text(encoding="utf-8")

    assert f"`PROTOCOL_VERSION = {PROTOCOL_VERSION}`" in design, (
        "DESIGN.md §1 names a kernel protocol version the guest does not declare"
    )


async def test_two_first_runs_on_one_namespace_build_one_kernel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F6 — `environment()` suspends, and two first runs both got past the read.

    The namespace *is* the agent id, and a fan-out of parallel tool calls on one
    agent is what Code Mode is for, so two first runs on one namespace is
    ordinary rather than exotic. Both built a `Kernel` and the second overwrote
    the first in `_kernels`; both then spawned a guest on their first cell, and
    only the one still in the dict was reachable by `close_namespace`. The other
    ran until the process exited, holding a socket, a pid and whatever the cell
    had open.

    Driven through `_acquire`, with `environment()` held open at exactly the
    await that suspends in production — a cold cache shelling out to `uv` takes
    seconds, which is the window.

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
        built.append(await runtime._acquire("a"))

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
    """A child's pipe, which it cannot get past until the host reads it.

    The real failure is a blocking `write(2)`: a guest that puts more than the
    pipe buffer on fd 1 or fd 2 before acking stops there. Modelled rather than
    reproduced with an OS pipe, because a real one can only be un-blocked from a
    worker thread, and a thread blocked in `read` is not cancellable — a host
    that failed this test would hang it rather than fail it, which is the one
    outcome a regression test must not have.

    What is kept is the causal chain that matters: `settled` fires only once the
    host has consumed the stream, and the child sends `boot-ack` only once it
    fires. A host that does not read this pipe never gets the ack.

    `chunks=0` and no `settled` is the other pipe in the same test: open, silent,
    and never the reason boot finishes.
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
        # Open, like a live child's: boot ends by cancelling this reader, not by
        # the pipe running out.
        await anyio.sleep(3600)


@pytest.mark.parametrize("pipe", ["stderr", "stdout"])
async def test_a_child_that_fills_a_pipe_before_acking_still_boots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pipe: str
) -> None:
    """F7(b) — during boot, nothing was consuming fd 1 or fd 2.

    `_drain` only runs for the duration of a *run*, so a guest that writes more
    than the pipe buffer before it acks blocks on that write and never acks at
    all. The host then reported "did not report ready within Ns" for a child
    that was ready and stuck, quoting whatever fitted in one late read — the
    reverse of the truth, and it sends whoever reads it looking at the guest.

    Both pipes, parametrized, because `_boot_said` quotes both and a wrapper
    explaining itself on stdout is naming the same failure: draining only the
    one that happened to be tested would leave the other deadlocking.

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
