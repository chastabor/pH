"""The runtime conformance suite (P3-22): the protocol, exercised whole.

Every property here has a behavioural test elsewhere — `test_kernel.py` drives
the process boundary, `test_codec.py` fuzzes the decoder, `test_snapshot.py`
folds the namespace. This module exists for the claim those cannot make:
**completeness**. It enumerates the protocol's own vocabulary and the shipped
profile's own namespaces, and fails when one of them has no coverage.

That inversion is the point. A hand-written list of "the frames we test" agrees
with the protocol until someone adds a frame, and then agrees with nothing — the
suite stays green while a frame type nobody exercises ships. Reading
`FRAME_FIELDS` and the mounted registry instead means the *absence* of a test is
what breaks.

Which frames have no producer, and which snapshot kinds are reserved, are facts
about the vocabulary rather than about the tests, so they live beside it
(`ph_runtime.protocol.UNPRODUCED_FRAMES`, `ph_rlm.snapshot.RESERVED_KINDS`) and
are read from here. Whoever closes one of those gaps is reading that module, not
this file.
"""

from __future__ import annotations

import socket
import subprocess
from typing import Any

import anyio
import pytest
from runtime_helpers import run_ipython_cell

from ph.seams.subprocess import scrub_env
from ph_app.profiles import resolve_profile
from ph_rlm.kernel.codec import decode, encode
from ph_rlm.kernel.manager import KernelLimits
from ph_rlm.kernel.venv import resolve_interpreter
from ph_rlm.prompt import DOCTRINE
from ph_rlm.snapshot import RESERVED_KINDS, SNAPSHOT_KINDS
from ph_runtime.cell import MAGIC_HINT, MAGIC_PREFIXES, compile_cell
from ph_runtime.protocol import FD_ENV, FRAME_FIELDS, PROTOCOL_VERSION, UNPRODUCED_FRAMES

pytestmark = pytest.mark.anyio


# ----------------------------------------------------------- the vocabulary --

COVERED_FRAMES: frozenset[str] = frozenset(
    {
        # host → child
        "boot",  # every kernel test: `Kernel.start` sends it, waits for `boot-ack`
        "run",  # every cell test
        "reply",  # test_kernel.py::test_a_binding_call_round_trips_through_the_host
        "restore",  # test_snapshot.py::test_a_new_kernel_gets_the_namespace_back
        "cancel",  # test_kernel.py::test_cancel_aborts_a_waiting_cell_...
        "shutdown",  # test_kernel.py::test_a_closed_kernel_leaves_no_zombie
        # child → host
        "boot-ack",  # test_lifecycle.py::test_the_guest_reports_which_mechanism_it_armed
        "call",  # test_kernel.py::test_a_binding_call_round_trips_through_the_host
        "log",  # test_kernel.py::test_output_is_capped_with_the_shared_marker
        "snapshot",  # test_snapshot.py::test_a_variable_becomes_a_snapshot_event
        "done",  # every cell test: it carries the value or the error
        "fault",  # test_a_protocol_mismatch_is_refused_at_boot, below
    }
)
"""Frames with a test that drives them.

The comments name where; the *set* is what is checked. A test name held as a
string goes stale in silence, while a frame missing from this set fails."""

COVERED_KINDS: frozenset[str] = frozenset(
    {
        "snap",  # test_snapshot.py::test_a_variable_becomes_a_snapshot_event
        "clear",  # test_snapshot.py::test_a_deleted_variable_is_cleared_not_forgotten
    }
)


def test_every_frame_type_is_accounted_for() -> None:
    """The suite's own completeness, read off the protocol rather than a list.

    A frame added to `FRAME_FIELDS` without a test fails here, which is the only
    moment anyone is looking.
    """
    assert set(FRAME_FIELDS) == COVERED_FRAMES | UNPRODUCED_FRAMES, (
        "a frame type has neither a test nor a recorded reason for having none"
    )
    assert not COVERED_FRAMES & UNPRODUCED_FRAMES


def test_every_snapshot_kind_is_accounted_for() -> None:
    """The same completeness claim, one layer up (D17).

    `kernel/snapshot` has its own closed vocabulary, and a kind that gains a
    producer without a test is the same silent gap a frame would be.
    """
    assert set(SNAPSHOT_KINDS) == COVERED_KINDS | RESERVED_KINDS
    assert not COVERED_KINDS & RESERVED_KINDS


def test_the_unproduced_frame_is_still_decodable() -> None:
    """`display` has no producer, so what it *does* have must still hold: both
    twins agree on its shape, and the host will accept one when it arrives."""
    frame = decode('{"type": "display", "mime": "text/plain", "data": "hello"}')

    assert frame is not None
    assert frame["type"] == "display"
    assert frame["mime"] == "text/plain"


# ------------------------------------------------------------- the frames --


async def test_a_protocol_mismatch_is_refused_at_boot(tmp_path: Any) -> None:
    """The `fault` frame's one producer, and D7's whole point.

    A guest that cannot serve this protocol says so once and exits, rather than
    misreading frames one at a time — which is why the version is refused at
    `boot` and not discovered later.

    The frame is **built, not typed**: `to_boot` + `encode` is what the host
    actually sends, so a required field added to `boot` (as `skills` was at
    protocol 2) reaches this test instead of leaving a stale literal that fails
    for the wrong reason. `FD_ENV` and `scrub_env` for the same reason — they are
    the two things a hand-rolled spawn gets wrong, and `Kernel.start` is the
    thing being imitated.
    """
    environment = await anyio.to_thread.run_sync(
        lambda: resolve_interpreter(cache=tmp_path, mode="host")
    )
    frame = KernelLimits().to_boot(namespaces=[], namespace_id=None, skills=[])
    payload = encode(frame.model_copy(update={"protocol": PROTOCOL_VERSION + 1}))

    host_end, child_end = socket.socketpair()
    process = await anyio.open_process(
        [str(environment.python), "-m", "ph_runtime"],
        pass_fds=(child_end.fileno(),),
        env=scrub_env(extra={FD_ENV: str(child_end.fileno())}),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_end.close()
    try:
        host_end.sendall(payload)
        host_end.settimeout(20)
        reply = decode(host_end.recv(65536).decode("utf-8").splitlines()[0])
        assert reply is not None
        assert reply["type"] == "fault"
        assert str(PROTOCOL_VERSION) in reply["message"]
        assert "runtime-venv" in reply["message"], "the fault must say how to fix it"
    finally:
        host_end.close()
        process.terminate()
        await process.wait()


# --------------------------------------------------------- the namespaces --


async def test_every_shipped_namespace_is_callable_from_a_cell(shipped_profile: Any) -> None:
    """One test per binding namespace, enumerated from the *mounted* profile.

    A namespace the SDK block advertises and a cell cannot reach is the failure
    this catches — and reading the registry means a namespace added to the
    bundle without a call here fails rather than passing unnoticed.

    Through `shipped_profile`, which merges the interpreter pin per row: three
    call sites had re-spelled it as a raw patch, and that bypass is how the pin
    once got dropped and a `uv` venv started building over the network.
    """
    ctx, session, agent = await shipped_profile(profile=resolve_profile("rlm"))
    declared = {"tools", *ctx.tools.view(agent.ctx).code_namespaces}

    # The names, not the calls: what is under test is that every namespace the
    # host declares is *bound in the guest*. Which names the profile ought to
    # have is `test_bundle.py`'s claim; the calls are governed and have their own
    # tests.
    program = "\n".join(f"assert {name}, {name!r}" for name in sorted(declared)) + "\n'reached'"
    result = await run_ipython_cell(ctx, program, agent=agent, session=session)

    assert result.is_error is False, result.error
    assert result.value["value"] == "reached"


# ------------------------------------------------ the layers agreeing --


def test_the_doctrine_describes_the_runtime_it_runs_on() -> None:
    """The runtime and the prompt have to describe the same runtime.

    D19's argument is that owning the runtime is what closes the `%%bash` hole —
    "the magic was the bypass; removing the mechanism closes the hole". The
    doctrine is what tells the model that, and for a phase it said the opposite:
    it promised `%%bash` shell cells, ported from prime-agent, which pH's own
    guest raises a `SyntaxError` for. A model reading it would have spent turns
    on a cell that cannot run.

    The prefixes come from the guest's own tuple rather than a third copy, so a
    fourth escape added there reaches the prompt by construction.
    """
    for prefix in MAGIC_PREFIXES:
        assert f"`{prefix}`" in DOCTRINE, f"the doctrine does not mention {prefix}"
    assert "no IPython magics" in DOCTRINE
    assert "await tools.bash(...)" in DOCTRINE
    assert "shell cell" not in DOCTRINE, "the doctrine promises a magic the runtime refuses"


@pytest.mark.parametrize("prefix", MAGIC_PREFIXES)
def test_every_magic_prefix_is_refused(prefix: str) -> None:
    """All three IPython escapes, because a hole left in one of them is the hole.

    Against `compile_cell` directly: refusing a magic is a *compile-time*
    property of the guest, so a child process per prefix would spend three spawns
    to observe what the pure function already decides. `test_kernel.py` drives
    one through a real kernel, which is where that round trip belongs.
    """
    with pytest.raises(SyntaxError) as raised:
        compile_cell(f"{prefix}bash\necho hi")

    assert MAGIC_HINT in str(raised.value)
