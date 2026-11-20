"""The persistent namespace, in the log (D17).

dsh withheld a persistent Python REPL for one stated reason: *"cross-call state
would be invisible to the log"*. The seam admits one only from a provider that
promised, at registration, to keep that state visible (D6). These tests check
that the promise is kept, and that keeping it has the properties that the promise
implies:

- A fork sees the namespace as of its boundary.
- An unchanged variable costs nothing.
- A blob that cannot be trusted is not unpickled.

## Why the guest tries the C pickler before `dill`

`dill.Pickler` subclasses `pickle._Pickler`, the **pure-Python** pickler, so it
serializes a large container in Python bytecode. A **1M-element list measured
463 ms per cell against 6.8 ms for `pickle`**. The digest comparison alone saves
the *send* and not the serialization, which runs after every cell for every name.
So the choice of pickler is on the hot path even when nothing is emitted.

**A variable over the cap stops at the cap.** The pickler writes into a sink that
raises when it passes `max_value_bytes`. The 200 MiB DataFrame in the example of
the module therefore costs **16 MiB of work per cell and not 200**, and reports
`too-large` either way.

## Two passes that the snapshot path does not make

**The host decodes the payload one time.** The payload used to be base64-decoded
for the tag, decoded again to write the blob, and hashed a third time to derive
the path. That was **35 ms of the 64 ms that a 16 MiB variable cost**, spent on
recomputing what was already available.

**The guest encodes to base64 only when the record is certain to go out.** To
encode a **10 MiB payload that turns out to be unchanged measured 11 ms per
cell**.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rlm_fixtures import MountedRuntime
from runtime_helpers import run_cell

from ph.cordis import DEPLOYMENT, InactiveScopeError
from ph.json import as_seq, as_str
from ph.keys import AGENTS, COMPACTION, MOUNT, SESSIONS, SPILL_STORE
from ph.llm.types import text_of
from ph.seams import spill as spill_module
from ph.session import IGNORABLE_SESSION_EVENT_TYPES, SurfaceIntent
from ph.session.events import SurfaceReplace
from ph.testing import (
    FAKE_OPTIONS,
    full_disk_batch,
    log_event,
    plugin_payload,
    prefix_of,
    user_payload,
)
from ph_rlm.kernel.manager import Kernel
from ph_rlm.keys import KERNEL_SNAPSHOTS, PYTHON_RUNTIME
from ph_rlm.snapshot import (
    KernelSnapshotPolicy,
    fold_namespace,
    render_live_variables,
)

pytestmark = pytest.mark.anyio


def _snapshots(session: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    return [
        dict(event.data["record"]) for event in session.events if event.type == "kernel/snapshot"
    ]


# ------------------------------------------------------------------ recording --


async def test_a_variable_becomes_a_snapshot_event(mounted_runtime: MountedRuntime) -> None:
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "answer = 42", agent=agent, session=session)
    records = _snapshots(session)
    assert [record["var"] for record in records] == ["answer"]
    assert records[0]["kind"] == "snap"
    assert records[0]["digest"]
    assert records[0]["tag"], "the payload is tagged against the session"


async def test_the_records_are_ignorable(mounted_runtime: MountedRuntime) -> None:
    """A different build may skip them and still read the session correctly.

    They describe state, not conversation. An older pH that reads this log should
    therefore not refuse it outright, which is what an unknown *required* type
    causes.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "answer = 42", agent=agent, session=session)
    kernel_events = [event for event in session.events if event.type.startswith("kernel/")]
    assert kernel_events
    assert all(event.ignorable for event in kernel_events)
    # The flag is stamped from the type and not passed on each call, so no two
    # call sites can disagree about one type. Conversation events stay required.
    assert {"kernel/snapshot", "kernel/restored"} <= IGNORABLE_SESSION_EVENT_TYPES
    conversation = [event for event in session.events if event.type == "turn/start"]
    assert all(not event.ignorable for event in conversation)


async def test_an_unchanged_variable_costs_nothing(mounted_runtime: MountedRuntime) -> None:
    """The digest of each variable is what keeps growth linear and not quadratic.

    A snapshot of the namespace as one blob would append `kept` again on every
    cell that changed anything.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "kept = list(range(100))", agent=agent, session=session, call_id="c1")
    after_first = len(_snapshots(session))
    await run_cell(ctx, "unrelated = 1", agent=agent, session=session, call_id="c2")
    records = _snapshots(session)
    assert after_first == 1
    assert [record["var"] for record in records[after_first:]] == ["unrelated"]


async def test_a_deleted_variable_is_cleared_not_forgotten(mounted_runtime: MountedRuntime) -> None:
    """A restore must *say* that the name is gone.

    A model that finds a name undefined in the middle of a session reads that as
    its own bug, and uses a turn to work around it.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "temporary = 'here'", agent=agent, session=session, call_id="c1")
    await run_cell(ctx, "del temporary", agent=agent, session=session, call_id="c2")
    # `to_wire` omits absent optionals, so a `clear` is exactly these three keys.
    assert _snapshots(session)[-1] == {"kind": "clear", "var": "temporary", "reason": "deleted"}
    assert "temporary" not in fold_namespace(session, agent.id)


# ------------------------------------------------------------------ restoring --


async def test_a_new_kernel_gets_the_namespace_back(mounted_runtime: MountedRuntime) -> None:
    """The resume path: a fresh child, the same namespace.

    When a kernel is closed, its close leaves the scope of the agent with it. So
    the scope holds one close, for the kernel that it has. Sabotage: skip the drop
    in `close_namespace`, and it holds two, the first of which finds nothing when
    the agent goes."""
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "carried = {'over': [1, 2, 3]}", agent=agent, session=session, call_id="c1")

    # Close the child the way disposal would, then run again. A new kernel for
    # the same namespace must be given the state that the log remembers.
    runtime = ctx.require(PYTHON_RUNTIME)
    await runtime.close_namespace(agent.id)

    result = await run_cell(ctx, "carried['over']", agent=agent, session=session, call_id="c2")
    assert result.value["value"] == [1, 2, 3]
    restored = [event for event in session.events if event.type == "kernel/restored"]
    assert restored
    assert "carried" in as_seq(restored[-1].data["restored"])
    closes = [one for one in agent.ctx._effects if one.label == f"code-runtime:{agent.id}"]
    assert len(closes) == 1


async def test_no_kernel_is_built_for_an_agent_that_is_gone(
    mounted_runtime: MountedRuntime,
) -> None:
    """The close of a kernel is an effect of the scope of its agent. An agent that
    went while the kernel was being built therefore has nowhere to put the close.
    The build is refused and the table is left as it was. When the kernel was
    built anyway, it had no owner. Its guest ran until the row unmounted, and an
    agent that was brought back under the same id ran its cells in that kernel.

    Sabotage: answer `None` for an agent that is gone, and the kernel is in the table.
    """
    ctx, _session, agent = await mounted_runtime(session_id="kernel-state")
    runtime = ctx.require(PYTHON_RUNTIME)
    await ctx.require(AGENTS).dispose(agent.id)

    with pytest.raises(InactiveScopeError):
        await runtime._acquire(agent.id, agent.id)

    assert agent.id not in runtime._kernels
    assert agent.id not in runtime._closers


async def test_a_snapshot_policy_goes_with_its_row(mounted_runtime: MountedRuntime) -> None:
    """The row takes its policy back from the runtime as it unmounts.

    The row set `runtime.snapshots` and nothing took it back. A row that was let
    go left the runtime recording into a policy whose row was gone.

    Sabotage: drop the take-back from the row, and the policy outlives it.
    """
    ctx, _session, _agent = await mounted_runtime(session_id="snapshot-row")
    runtime = ctx.require(PYTHON_RUNTIME)
    assert runtime.snapshots is not None
    await ctx.require(MOUNT).forks["rlm-kernel-snapshot"].dispose()

    assert runtime.snapshots is None


async def test_a_crash_does_not_silently_reconstitute_the_namespace(
    mounted_runtime: MountedRuntime,
) -> None:
    """After `os._exit`, the model was told that the namespace is empty. It must
    be empty.

    To restore half of it without telling the model would be worse than the empty
    namespace that the reset notice announces. The model would see some names and
    not others, with no way to tell which.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "before = 'crash'", agent=agent, session=session, call_id="c1")
    await run_cell(ctx, "import os\nos._exit(1)", agent=agent, session=session, call_id="c2")
    result = await run_cell(ctx, "'before' in dir()", agent=agent, session=session, call_id="c3")
    assert result.value["value"] is False


async def test_a_tag_from_another_session_is_refused(
    mounted_runtime: MountedRuntime, tmp_path: Path
) -> None:
    """`dill.loads` on arbitrary bytes executes arbitrary code.

    So a payload whose tag does not verify is not unpickled. This is provenance,
    not secrecy. Anyone who can write the log can write the tag. What the tag
    actually prevents is a blob restored into the wrong session, or a
    half-written file read as sound.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    policy: KernelSnapshotPolicy = ctx.require(KERNEL_SNAPSHOTS)
    await run_cell(ctx, "value = 'mine'", agent=agent, session=session, call_id="c1")

    # The record as it stands verifies against its own session.
    [record] = list(fold_namespace(session, agent.id).values())
    assert await policy._payload(session, record) is not None

    # Under a different session id the same bytes do not verify, so nothing is
    # returned to be unpickled.
    forged = ctx.require(SESSIONS).create("someone-else")
    assert await policy._payload(forged, record) is None


# ---------------------------------------------------------------------- spill --


async def test_a_large_payload_goes_to_spill_and_the_event_names_it(
    mounted_runtime: MountedRuntime,
) -> None:
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'x' * 20_000", agent=agent, session=session)
    [record] = [record for record in _snapshots(session) if record["var"] == "big"]
    assert "blob" not in record, "the payload is not inline"
    assert record["locator"]
    # The event names the locator that the store derived before the write, so
    # this asserts that the derived path is the real one.
    assert Path(record["locator"]).exists()
    assert Path(record["locator"]).stat().st_size == record["bytes"]


async def test_a_spilled_variable_still_restores(mounted_runtime: MountedRuntime) -> None:
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'y' * 20_000", agent=agent, session=session, call_id="c1")
    await ctx.require(PYTHON_RUNTIME).close_namespace(agent.id)
    result = await run_cell(ctx, "len(big)", agent=agent, session=session, call_id="c2")
    assert result.value["value"] == 20_000


async def test_a_blob_that_is_gone_is_reported_rather_than_dropped(
    mounted_runtime: MountedRuntime,
) -> None:
    """S11: a missing blob was skipped, so it was neither restored nor failed.

    With nothing restored, no `kernel/restored` was written at all. The variable
    was undefined after a restart, with no notice. Now it is named as failed,
    even when it was the only variable.

    Sabotage: skip unreadable payloads silently in `materialize` again, and nothing
    records the loss.
    """
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'z' * 20_000", agent=agent, session=session, call_id="c1")
    [record] = [record for record in _snapshots(session) if record["var"] == "big"]
    Path(record["locator"]).unlink()
    await ctx.require(PYTHON_RUNTIME).close_namespace(agent.id)

    result = await run_cell(ctx, "'big' in dir()", agent=agent, session=session, call_id="c2")

    assert result.value["value"] is False
    restored = [event for event in session.events if event.type == "kernel/restored"]
    assert restored, "the loss was not recorded at all"
    assert "big" in as_seq(restored[-1].data["failed"])


async def test_a_blob_the_store_cannot_take_is_recorded_as_cleared(
    mounted_runtime: MountedRuntime,
) -> None:
    """S11: the record named a blob that the store had refused.

    When the blob was written after the record, a blob that failed to write left
    a `snap` that pointed at nothing. The blob is now written first, so the
    refusal is known before anything is recorded, and the record says that the
    value is gone.

    Sabotage: record the `snap` whether or not the blob was written, and the record
    names a file that is not there.
    """
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )

    with patch.object(spill_module, "write_atomic_all", full_disk_batch):
        await run_cell(ctx, "big = 'q' * 20_000", agent=agent, session=session)

    [record] = [record for record in _snapshots(session) if record["var"] == "big"]
    assert record["kind"] == "clear" and "could not be stored" in record["reason"]


def _kernel_locators(session: Any) -> set[str]:  # noqa: ANN401
    """The spill locators that the `kernel/snapshot` events of this log still name.

    The check is for `Mapping`, not `dict`. A frozen payload is a
    `MappingProxyType`, and a `dict` test here read every live event as having no
    record. That is the trap that `ph_app.wire` exists to name, and this helper
    met it one more time.
    """
    return {
        record["locator"]
        for event in session.select("kernel/snapshot")
        if isinstance(record := event.data.get("record"), Mapping)
        and isinstance(record.get("locator"), str)
    }


async def test_unreferenced_blobs_are_swept(mounted_runtime: MountedRuntime) -> None:
    """F7: a blob whose event never landed is otherwise never reconciled."""
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'z' * 20_000", agent=agent, session=session)
    directory = Path(ctx.require(SPILL_STORE).root) / f"kernel/{agent.id}"
    orphan = directory / "deadbeefdeadbeef-nothing.dill"
    orphan.write_bytes(b"nobody points at this")
    kept = _kernel_locators(session)

    removed = await ctx.require(SPILL_STORE).sweep_session(session)
    assert str(orphan) in removed
    assert not orphan.exists()
    assert all(Path(locator).exists() for locator in kept)


async def test_a_sweep_leaves_another_sessions_blobs_alone(mounted_runtime: MountedRuntime) -> None:
    """The namespaces to visit come from the same log as the locators to keep.

    A sweep of every namespace that the process had seen, against the reference
    set of *one* session, deleted the blobs of the other sessions. For them the
    set was empty, so everything that they owned looked unreferenced.
    """
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "mine = 'a' * 20_000", agent=agent, session=session)
    ours = _kernel_locators(session)
    assert ours

    # A second session in the same process, with its own agent and its own blob.
    other_session = ctx.require(SESSIONS).create("another-session")
    other_agent = ctx.require(AGENTS).create(other_session, FAKE_OPTIONS)
    await run_cell(ctx, "theirs = 'b' * 20_000", agent=other_agent, session=other_session)
    theirs = _kernel_locators(other_session)
    assert theirs and theirs != ours

    # To open either session must not collect the blobs of the other session.
    assert await ctx.require(SPILL_STORE).sweep_session(other_session) == []
    assert all(Path(locator).exists() for locator in ours | theirs)


# ----------------------------------------------------------------------- fold --


async def test_the_fold_reconstructs_the_namespace_as_of_a_boundary(
    mounted_runtime: MountedRuntime,
) -> None:
    """What a side file could not express, and the reason that D17 chose events.

    A fork inherits the namespace *as it was then*. A side file would give every
    fork the latest namespace of the parent.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "first = 1", agent=agent, session=session, call_id="c1")
    boundary = len(session.events)
    await run_cell(ctx, "second = 2", agent=agent, session=session, call_id="c2")

    assert set(fold_namespace(session, agent.id)) == {"first", "second"}
    # The prefix of the log that a fork at `boundary` would carry.
    assert set(fold_namespace(prefix_of(session, boundary), agent.id)) == {"first"}


async def test_a_foreign_record_does_not_break_the_fold(mounted_runtime: MountedRuntime) -> None:
    _ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    log_event(
        session,
        "kernel/snapshot",
        {"namespace": agent.id, "run": 1, "record": {"kind": "invented", "var": 7}},
    )
    assert fold_namespace(session, agent.id) == {}


async def test_a_namespace_only_sees_its_own_records(mounted_runtime: MountedRuntime) -> None:
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "mine = 1", agent=agent, session=session)
    assert set(fold_namespace(session, agent.id)) == {"mine"}
    assert fold_namespace(session, "some-other-agent") == {}


# ------------------------------------------------------- surviving a summary --
# G10. Compaction rewrites a *surface*. A REPL is not on a surface, so a
# namespace crosses a summary untouched. The risk is not that the state is lost.
# The risk is that the model is told that the conversation was summarized, and
# concludes that its variables went too. It then spends the next cell on
# recomputing a DataFrame that it still has.


async def test_the_note_names_the_variables_the_kernel_still_holds(
    mounted_runtime: MountedRuntime,
) -> None:
    """The note is read from the same fold that `materialize` restores from. What
    a summary promises and what a restart delivers therefore cannot disagree."""
    ctx, session, agent = await mounted_runtime(session_id="live-variables")
    await run_cell(ctx, "frame = list(range(64))\nlabel = 'x'", agent=agent, session=session)

    text = render_live_variables(session)

    assert "`frame`" in text and "`label`" in text
    assert "live-variables" in text, "the namespace is named, because a fork has more than one"
    assert "still defined" in text


async def test_a_variable_that_is_gone_is_not_promised(
    mounted_runtime: MountedRuntime,
) -> None:
    """The failure to guard against. A note is a *claim* that the model will act
    on, so a name that the fold has cleared must not appear in it."""
    ctx, session, agent = await mounted_runtime(session_id="cleared")
    await run_cell(ctx, "temporary = 1", agent=agent, session=session)
    await run_cell(ctx, "del temporary", agent=agent, session=session, call_id="call-2")

    assert render_live_variables(session) == ""


async def test_an_empty_namespace_contributes_no_note(mounted_runtime: MountedRuntime) -> None:
    """Empty means absent. A session that has run no cell spends no prompt on a
    paragraph that says its kernel holds nothing."""
    _ctx, session, _agent = await mounted_runtime(session_id="nothing-yet")
    assert render_live_variables(session) == ""


async def test_the_note_is_registered_on_the_compaction_seam(
    mounted_runtime: MountedRuntime,
) -> None:
    """The wiring, which is `ctx.inject` and not the `inject` list of this row.

    This is deliberate. The compaction seam is a nice-to-have here, and a
    deployment that dropped the compaction row must not lose kernel snapshots
    because of it. This test asserts the inverse. With the seam present (it ships
    in `ph-base`), the note is there for whatever engine a profile layers on top.
    """
    ctx, session, agent = await mounted_runtime(session_id="wired")
    await run_cell(ctx, "answer = 42", agent=agent, session=session)

    assert any(
        "`answer`" in note for note in ctx.require(COMPACTION).notes(session, scope=DEPLOYMENT)
    )


async def test_the_namespace_outlives_a_compaction_of_the_conversation(
    mounted_runtime: MountedRuntime,
) -> None:
    """ "Runtime state itself is untouched" (G10), with a real kernel.

    The replacement is built by hand here, because the test is about the
    *independence of the kernel from the surface* and not about the engine that
    writes a surface. `ph-stabilize` owns that half and tests it against its own
    gates. To shadow every message that the conversation had is the strongest
    form of the question. The cell after it still sees the variable, because a
    surface rewrite appends events and never speaks to the runtime.
    """
    ctx, session, agent = await mounted_runtime(session_id="survives")
    log_event(session, "user/message", user_payload("build the frame"), SurfaceIntent("append"))
    await run_cell(ctx, "frame = list(range(8))", agent=agent, session=session)
    nodes = session.surface.nodes
    log_event(
        session,
        "user/message",
        plugin_payload(
            "(the conversation, summarized)",
            "summary",
            plugin="compaction-summarize",
            form="compaction",
            summary="history summarized",
        ),
        SurfaceIntent(SurfaceReplace(replaces=tuple(nodes)), nodes),
    )

    result = await run_cell(ctx, "print(sum(frame))", agent=agent, session=session, call_id="c2")

    assert "28" in text_of(result.content), "the namespace did not survive the summary"
    assert "`frame`" in render_live_variables(session), "the fold lost it"


async def test_a_namespace_is_restored_one_variable_to_a_frame(
    mounted_runtime: MountedRuntime,
) -> None:
    """O3: the defect of F3 survived, mirrored, in the direction that F3 did not
    reach.

    `maxSnapshotBytes` limits each snapshotted *value*, and F3 split the way back
    so that one frame carries one value. The way *in* still carried the whole
    namespace in a single `restore`, with no limit. This direction is the worse
    of the two. The guest reads with a fixed limit, which it cannot size from a
    `boot` frame that it has not read yet. A line over the limit is therefore a
    `ValueError` in its reader, `receive` answers `None`, and the guest exits.
    The namespace is lost on the way back in, and to bring it back was the
    purpose of the restore.

    Three variables are each near the cap for one variable. One frame for each
    is fine, and one frame for all three is three times the cap. The test
    asserts the count as well as the outcome. A host that sent one frame and
    succeeded on a small namespace would pass on the values alone.

    Sabotage: send a single `RestoreFrame` with every variable and the frame
    count is 1.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    for index in range(3):
        await run_cell(
            ctx,
            f"kept{index} = 'x' * 100_000",
            agent=agent,
            session=session,
            call_id=f"c{index}",
        )
    runtime = ctx.require(PYTHON_RUNTIME)
    await runtime.close_namespace(agent.id)

    sent: list[int] = []
    kernel_send = Kernel._send

    async def counting(self: Kernel, frame: Any) -> None:  # noqa: ANN401
        if getattr(frame, "type", None) == "restore":
            sent.append(len(frame.variables))
        await kernel_send(self, frame)

    with patch.object(Kernel, "_send", counting):
        result = await run_cell(
            ctx, "len(kept0) + len(kept2)", agent=agent, session=session, call_id="back"
        )

    assert result.value["value"] == 200_000, "the namespace did not come back"
    assert sent == [1, 1, 1], f"the restore was not split per variable: {sent}"
    restored = [event for event in session.events if event.type == "kernel/restored"]
    # One `restore` is still one report, whatever it took on the wire.
    names = sorted(as_str(one) for one in as_seq(restored[-1].data["restored"]))
    assert names == ["kept0", "kept1", "kept2"]
