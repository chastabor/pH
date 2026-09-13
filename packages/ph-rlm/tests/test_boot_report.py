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
import resource
import socket
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph_rlm.kernel import codec
from ph_rlm.kernel.journal import OrphanJournal
from ph_rlm.kernel.manager import Kernel, KernelLimits, PythonCodeRuntime
from ph_rlm.kernel.venv import resolve_interpreter
from ph_runtime.limits import apply_limits
from ph_runtime.protocol import PROTOCOL_VERSION

pytestmark = pytest.mark.anyio

INFINITY = resource.RLIM_INFINITY


def _refuse(*_args: Any) -> None:  # noqa: ANN401
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
