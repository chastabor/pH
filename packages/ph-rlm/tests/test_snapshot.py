"""The persistent namespace, in the log (D17).

dsh withheld a persistent Python REPL for one stated reason — *"cross-call state
would be invisible to the log"* — and the seam admits one only from a provider
that promised, at registration, to keep it visible (D6). These are the tests that
the promise is kept, and that keeping it has the properties the promise implies:
a fork sees the namespace as of its boundary, an unchanged variable costs
nothing, and a blob that cannot be trusted is not unpickled.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from runtime_helpers import run_cell

from ph.session import IGNORABLE_SESSION_EVENT_TYPES
from ph.testing import FAKE_OPTIONS
from ph_rlm.snapshot import KernelSnapshotPolicy, fold_namespace, referenced_locators

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]


def _snapshots(session: Any) -> list[dict[str, Any]]:
    return [
        dict(event.data["record"]) for event in session.events if event.type == "kernel/snapshot"
    ]


# ------------------------------------------------------------------ recording --


async def test_a_variable_becomes_a_snapshot_event(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "answer = 42", agent=agent, session=session)
    records = _snapshots(session)
    assert [record["var"] for record in records] == ["answer"]
    assert records[0]["kind"] == "snap"
    assert records[0]["digest"]
    assert records[0]["tag"], "the payload is tagged against the session"


async def test_the_records_are_ignorable(mounted_runtime: Mounted) -> None:
    """A different build may skip them without misreading the session.

    They describe state, not conversation, so an older pH reading this log should
    not refuse it outright — which is what an unknown *required* type does.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "answer = 42", agent=agent, session=session)
    kernel_events = [event for event in session.events if event.type.startswith("kernel/")]
    assert kernel_events
    assert all(event.ignorable for event in kernel_events)
    # Stamped from the type, not passed per call, so no two call sites can
    # disagree about one type — and conversation events stay required.
    assert {"kernel/snapshot", "kernel/restored"} <= IGNORABLE_SESSION_EVENT_TYPES
    conversation = [event for event in session.events if event.type == "turn/start"]
    assert all(not event.ignorable for event in conversation)


async def test_an_unchanged_variable_costs_nothing(mounted_runtime: Mounted) -> None:
    """Per-variable digesting is what keeps growth linear rather than quadratic.

    Snapshotting the namespace as one blob would re-append `kept` on every cell
    that touched anything at all.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "kept = list(range(100))", agent=agent, session=session, call_id="c1")
    after_first = len(_snapshots(session))
    await run_cell(ctx, "unrelated = 1", agent=agent, session=session, call_id="c2")
    records = _snapshots(session)
    assert after_first == 1
    assert [record["var"] for record in records[after_first:]] == ["unrelated"]


async def test_a_deleted_variable_is_cleared_not_forgotten(mounted_runtime: Mounted) -> None:
    """A restore must *say* the name is gone.

    A model that finds a name undefined mid-session reads it as its own bug and
    spends a turn working around it.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "temporary = 'here'", agent=agent, session=session, call_id="c1")
    await run_cell(ctx, "del temporary", agent=agent, session=session, call_id="c2")
    # `to_wire` omits absent optionals, so a `clear` is exactly these three keys.
    assert _snapshots(session)[-1] == {"kind": "clear", "var": "temporary", "reason": "deleted"}
    assert "temporary" not in fold_namespace(session, agent.id)


# ------------------------------------------------------------------ restoring --


async def test_a_new_kernel_gets_the_namespace_back(mounted_runtime: Mounted) -> None:
    """The resume path: a fresh child, the same namespace."""
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "carried = {'over': [1, 2, 3]}", agent=agent, session=session, call_id="c1")

    # Close the child the way disposal would, then run again: a new kernel for
    # the same namespace has to be handed the state the log remembers.
    runtime = ctx.python_runtime
    await runtime.close_namespace(agent.id)

    result = await run_cell(ctx, "carried['over']", agent=agent, session=session, call_id="c2")
    assert result.value["value"] == [1, 2, 3]
    restored = [event for event in session.events if event.type == "kernel/restored"]
    assert restored
    assert "carried" in restored[-1].data["restored"]


async def test_a_crash_does_not_silently_reconstitute_the_namespace(
    mounted_runtime: Mounted,
) -> None:
    """After `os._exit`, the model was told the namespace is empty. It must be.

    Restoring half of it behind the model's back would be worse than the empty
    namespace the reset notice announces — the model would see some names and not
    others with no way to tell which.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "before = 'crash'", agent=agent, session=session, call_id="c1")
    await run_cell(ctx, "import os\nos._exit(1)", agent=agent, session=session, call_id="c2")
    result = await run_cell(ctx, "'before' in dir()", agent=agent, session=session, call_id="c3")
    assert result.value["value"] is False


async def test_a_tag_from_another_session_is_refused(
    mounted_runtime: Mounted, tmp_path: Path
) -> None:
    """`dill.loads` on arbitrary bytes executes arbitrary code.

    So a payload whose tag does not verify is not unpickled. This is provenance,
    not secrecy: anyone who can write the log can write the tag, and what it
    actually prevents is a blob restored into the wrong session or a half-written
    file read as sound.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    policy: KernelSnapshotPolicy = ctx.kernel_snapshots
    await run_cell(ctx, "value = 'mine'", agent=agent, session=session, call_id="c1")

    # The record as it stands verifies against its own session.
    [record] = list(fold_namespace(session, agent.id).values())
    assert await policy._payload(session, record) is not None

    # Under a different session id the same bytes do not, so nothing is handed
    # back to be unpickled.
    forged = ctx.sessions.create("someone-else")
    assert await policy._payload(forged, record) is None


# ---------------------------------------------------------------------- spill --


async def test_a_large_payload_goes_to_spill_and_the_event_names_it(
    mounted_runtime: Mounted,
) -> None:
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'x' * 20_000", agent=agent, session=session)
    [record] = [record for record in _snapshots(session) if record["var"] == "big"]
    assert "blob" not in record, "the payload is not inline"
    assert record["locator"]
    # The event named the locator *before* the blob was written (write-ahead
    # ordering, §4.9), so this asserts the derived path is the real one.
    assert Path(record["locator"]).exists()
    assert Path(record["locator"]).stat().st_size == record["bytes"]


async def test_a_spilled_variable_still_restores(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'y' * 20_000", agent=agent, session=session, call_id="c1")
    await ctx.python_runtime.close_namespace(agent.id)
    result = await run_cell(ctx, "len(big)", agent=agent, session=session, call_id="c2")
    assert result.value["value"] == 20_000


async def test_unreferenced_blobs_are_swept(mounted_runtime: Mounted) -> None:
    """F7: a blob whose event never landed is otherwise never reconciled."""
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "big = 'z' * 20_000", agent=agent, session=session)
    policy: KernelSnapshotPolicy = ctx.kernel_snapshots

    directory = Path(ctx.spill_store.root) / f"kernel/{agent.id}"
    orphan = directory / "deadbeefdeadbeef-nothing.dill"
    orphan.write_bytes(b"nobody points at this")
    kept = referenced_locators(session)

    removed = await policy.sweep(session)
    assert str(orphan) in removed
    assert not orphan.exists()
    assert all(Path(locator).exists() for locator in kept)


async def test_a_sweep_leaves_another_sessions_blobs_alone(mounted_runtime: Mounted) -> None:
    """The namespaces to visit come from the same log as the locators to keep.

    Sweeping every namespace the process had seen against *one* session's
    reference set deleted the other sessions' blobs: for them the set was empty,
    so everything they owned looked unreferenced.
    """
    ctx, session, agent = await mounted_runtime(
        session_id="kernel-state", snapshot_config={"inlineBlobMax": 256}
    )
    await run_cell(ctx, "mine = 'a' * 20_000", agent=agent, session=session)
    policy: KernelSnapshotPolicy = ctx.kernel_snapshots
    ours = referenced_locators(session)
    assert ours

    # A second session in the same process, with its own agent and its own blob.
    other_session = ctx.sessions.create("another-session")
    other_agent = ctx.agents.create(other_session, FAKE_OPTIONS)
    await run_cell(ctx, "theirs = 'b' * 20_000", agent=other_agent, session=other_session)
    theirs = referenced_locators(other_session)
    assert theirs and theirs != ours

    # Opening either session must not collect the other's blobs.
    assert await policy.sweep(other_session) == []
    assert all(Path(locator).exists() for locator in ours | theirs)


# ----------------------------------------------------------------------- fold --


async def test_the_fold_reconstructs_the_namespace_as_of_a_boundary(
    mounted_runtime: Mounted,
) -> None:
    """What a side file could not express, and the reason D17 chose events.

    A fork inherits the namespace *as it was then*; a side file would hand every
    fork the parent's latest.
    """
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "first = 1", agent=agent, session=session, call_id="c1")
    boundary = len(session.events)
    await run_cell(ctx, "second = 2", agent=agent, session=session, call_id="c2")

    assert set(fold_namespace(session, agent.id)) == {"first", "second"}

    class _AsOf:
        """The prefix of the log a fork at `boundary` would carry."""

        id = session.id
        events = session.events[:boundary]

    assert set(fold_namespace(_AsOf(), agent.id)) == {"first"}


async def test_a_foreign_record_does_not_break_the_fold(mounted_runtime: Mounted) -> None:
    _ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    session.append(
        "kernel/snapshot",
        {"namespace": agent.id, "run": 1, "record": {"kind": "invented", "var": 7}},
    )
    assert fold_namespace(session, agent.id) == {}


async def test_a_namespace_only_sees_its_own_records(mounted_runtime: Mounted) -> None:
    ctx, session, agent = await mounted_runtime(session_id="kernel-state")
    await run_cell(ctx, "mine = 1", agent=agent, session=session)
    assert set(fold_namespace(session, agent.id)) == {"mine"}
    assert fold_namespace(session, "some-other-agent") == {}
