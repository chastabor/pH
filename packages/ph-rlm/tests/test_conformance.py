"""The runtime conformance suite (P3-22): the protocol, exercised as a whole.

Every property here has a behavioral test elsewhere. `test_kernel.py` drives the
process boundary, `test_codec.py` fuzzes the decoder, and `test_snapshot.py`
folds the namespace. This module exists for the one claim that those tests
cannot make: **completeness**. It enumerates the vocabulary of the protocol and
the namespaces of the shipped profile, and it fails when one of them has no
coverage.

That reversal is the purpose. A hand-written list of "the frames we test" agrees
with the protocol until someone adds a frame, and then it agrees with nothing.
The suite stays green while a frame type that nobody exercises ships. This
module reads `FRAME_FIELDS` and the mounted registry instead, so the *absence*
of a test is what breaks.

Which frames have no producer, and which snapshot kinds are reserved, are facts
about the vocabulary and not about the tests. They are therefore declared beside
the vocabulary (`ph_runtime.protocol.UNPRODUCED_FRAMES`,
`ph_rlm.snapshot.RESERVED_KINDS`) and read from here. The developer who closes
one of those gaps reads that module, and not this file.
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import anyio
import pytest
from rlm_fixtures import ShippedProfile
from runtime_helpers import run_python_cell

from ph.keys import TOOLS
from ph.seams.subprocess import scrub_env
from ph_app.profiles import resolve_profile
from ph_rlm.kernel.codec import decode, encode
from ph_rlm.kernel.manager import KernelLimits
from ph_rlm.kernel.venv import resolve_interpreter
from ph_rlm.presentation import PYTHON_DESCRIPTION
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
        "ping",  # test_kernel.py::test_the_probe_reports_how_far_behind_a_guest_loop_is
        # child → host
        "boot-ack",  # test_lifecycle.py::test_the_guest_reports_which_mechanism_it_armed
        "call",  # test_kernel.py::test_a_binding_call_round_trips_through_the_host
        "log",  # test_kernel.py::test_output_is_capped_with_the_shared_marker
        "snapshot",  # test_snapshot.py::test_a_variable_becomes_a_snapshot_event
        "done",  # every cell test: it carries the value or the error
        "fault",  # test_a_protocol_mismatch_is_refused_at_boot, below
        "pong",  # test_kernel.py::test_a_run_the_guest_will_never_finish_is_repaired_not_waited_on
    }
)
"""The frames that have a test that drives them.

The comments name the test. The *set* is what is checked. A test name that is
held as a string goes stale silently, but a frame that is missing from this set
fails."""

COVERED_KINDS: frozenset[str] = frozenset(
    {
        "snap",  # test_snapshot.py::test_a_variable_becomes_a_snapshot_event
        "clear",  # test_snapshot.py::test_a_deleted_variable_is_cleared_not_forgotten
    }
)


def test_every_frame_type_is_accounted_for() -> None:
    """The completeness of the suite itself, read from the protocol and not from
    a list.

    A frame that is added to `FRAME_FIELDS` without a test fails here, which is
    the only moment when anyone is looking.
    """
    assert set(FRAME_FIELDS) == COVERED_FRAMES | UNPRODUCED_FRAMES, (
        "a frame type has neither a test nor a recorded reason for having none"
    )
    assert not COVERED_FRAMES & UNPRODUCED_FRAMES


def test_every_snapshot_kind_is_accounted_for() -> None:
    """The same completeness claim, one layer up (D17).

    `kernel/snapshot` has its own closed vocabulary. A kind that gets a producer
    without a test is the same silent gap that a frame would be.
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


async def test_a_protocol_mismatch_is_refused_at_boot(tmp_path: Path) -> None:
    """The one producer of the `fault` frame, and the purpose of D7.

    A guest that cannot serve this protocol says so one time and exits. It does
    not misread frames one at a time. That is why the version is refused at
    `boot` and not discovered later.

    The frame is **built, not typed**. `to_boot` + `encode` is what the host
    actually sends. A required field that is added to `boot` (as `skills` was)
    therefore reaches this test, and does not leave a stale literal that fails
    for the wrong reason. `FD_ENV` and `scrub_env` are used for the same reason.
    They are the two things that a hand-written spawn gets wrong, and this test
    imitates `Kernel.start`.
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


async def test_every_shipped_namespace_is_callable_from_a_cell(
    shipped_profile: ShippedProfile,
) -> None:
    """One test for each binding namespace, enumerated from the *mounted* profile.

    This test catches a namespace that the SDK block advertises and that a cell
    cannot reach. Because it reads the registry, a namespace that is added to
    the bundle without a call here fails, and does not pass unnoticed.

    The test goes through `shipped_profile`, which merges the interpreter pin
    for each row. Three call sites had written the pin again as a raw patch.
    That bypass is how the pin was dropped before, and a `uv` venv started to
    build over the network.
    """
    ctx, session, agent = await shipped_profile(profile=resolve_profile("rlm"))
    declared = {"tools", *ctx.require(TOOLS).view(agent.ctx).code_namespaces}

    # The program checks the names, not the calls. The test is that every
    # namespace that the host declares is *bound in the guest*. Which names the
    # profile must have is the claim of `test_bundle.py`. The calls are governed
    # and have their own tests.
    program = "\n".join(f"assert {name}, {name!r}" for name in sorted(declared)) + "\n'reached'"
    result = await run_python_cell(ctx, program, agent=agent, session=session)

    assert result.is_error is False, result.error
    assert result.value["value"] == "reached"


# ------------------------------------------------ the layers agreeing --


def test_the_transport_describes_the_runtime_it_runs_on() -> None:
    """The runtime and what the model is told about it must describe one runtime.

    The argument of D19 is that owning the runtime closes the `%%bash` hole:
    "the magic was the bypass; removing the mechanism closes the hole". The
    description of the transport tells the model that. For one phase the prompt
    said the opposite. It promised `%%bash` shell cells, ported from
    prime-agent, and the guest of pH raises a `SyntaxError` for those. A model
    that read the prompt would have spent turns on a cell that cannot run. The
    doctrine now points at the description and does not repeat it.

    The prefixes come from the tuple of the guest itself and not from a third
    copy, so a fourth escape that is added there reaches the description by
    construction.
    """
    for prefix in MAGIC_PREFIXES:
        assert f"`{prefix}`" in PYTHON_DESCRIPTION, f"the description does not mention {prefix}"
    assert "not IPython or Jupyter" in PYTHON_DESCRIPTION
    assert "await tools.bash(command=...)" in PYTHON_DESCRIPTION
    assert "shell cell" not in PYTHON_DESCRIPTION + DOCTRINE, "a magic the runtime refuses"


@pytest.mark.parametrize("prefix", MAGIC_PREFIXES)
def test_every_magic_prefix_is_refused(prefix: str) -> None:
    """All three IPython escapes, because a hole in one of them is a hole in the
    rule.

    The test calls `compile_cell` directly. Refusing a magic is a *compile-time*
    property of the guest, so a child process for each prefix would spend three
    spawns to observe what the pure function already decides. `test_kernel.py`
    drives one through a real kernel, which is where that round trip belongs.
    """
    with pytest.raises(SyntaxError) as raised:
        compile_cell(f"{prefix}bash\necho hi")

    assert MAGIC_HINT in str(raised.value)
