"""A child's records, through the seam's doors, in the child's own log (S10, Phase 11).

Every fact about a sub-agent is in its own log — its admission, each start, each
wait, its ending and its tombstone — and nothing about it is written to its parent's.
Some of those records carry a rule about *when* or *how* they reach disk: a restart
must be written before the attempt it counts, an ending before the parent is handed
the result, and a tombstone in one batch with the `canceled` that ends it.
`ph.seams.subagents` keeps those rules in doors every provider calls, and
`test_log_writers` holds every provider to them, since none may append one of these
records itself.
"""

from __future__ import annotations

import pytest

from ph.cordis import Context
from ph.keys import SESSIONS
from ph.seams.subagents import (
    ADMITTED,
    DELETED,
    SUSPENDED_DETAIL,
    child_is_live,
    child_state,
    record_deleted,
    record_ended,
    record_started,
    record_waiting,
    restarts_since_progress,
)
from ph.session import Session, SurfaceIntent
from ph.testing import MountProfile, admitted_child, assistant_payload, log_event, stored_types

pytestmark = pytest.mark.anyio


def _child(ctx: Context, run_id: str = "r1", *, admitted: bool = True) -> Session:
    """A child's own log, naming its parent the way a provider opens one."""
    sessions = ctx.require(SESSIONS)
    parent = sessions.get("parent") or sessions.create("parent")
    return admitted_child(ctx, parent, run_id, {"name": "scout"}, admitted=admitted)


def _answer(child: Session, text: str = "done") -> None:
    log_event(
        child,
        "assistant/message",
        {**assistant_payload(text, "m1"), "usage": {"inputTokens": 3, "outputTokens": 5}},
        SurfaceIntent("append", ()),
    )


async def test_a_restart_is_on_the_childs_disk_when_its_door_returns(mount: MountProfile) -> None:
    """The ladder counts restarts, so a restart written only to memory before the
    child took the daemon down again was never counted — a crash loop that never
    tripped `CHILD_RETRY_LIMIT`. On the child's own disk now, which is where the
    ladder reads it.

    Sabotage: drop the flush from `record_started`, and nothing is on disk.
    """
    ctx = await mount()
    child = _child(ctx)

    await record_started(ctx, child, cause="resumed")

    assert stored_types(ctx, child.id) == [ADMITTED, "subagent/status"]


async def test_a_first_start_rides_the_next_flush(mount: MountProfile) -> None:
    """A first start counts nothing, so it pays for no write of its own — the child's
    first model request, which comes after it, writes it."""
    ctx = await mount()
    child = _child(ctx)

    event = await record_started(ctx, child)

    assert event.data == {"status": "running"}
    assert stored_types(ctx, child.id) == []


async def test_an_ending_is_on_the_childs_disk_before_the_result_is_handed_over(
    mount: MountProfile,
) -> None:
    """F1, in one log. A parent handed "done, here is the answer" by a child whose log
    ended before the answer is one repair would call interrupted on the next open —
    and readmit to do the work again. The door returns with the ending, and the answer
    before it, on the child's disk.

    Sabotage: drop the flush from `record_ended`, and the child's store is empty.
    """
    ctx = await mount()
    child = _child(ctx)
    log_event(child, "turn/start", {"turn": 1})
    _answer(child, "42")

    ended = await record_ended(ctx, child, "done", answerPreview="42")

    assert stored_types(ctx, child.id)[-2:] == ["assistant/message", "subagent/status"]
    assert ended.data == {"status": "done", "answerPreview": "42"}


async def test_a_childs_state_folds_what_the_doors_write(mount: MountProfile) -> None:
    """One writer, one shape: the ladder's count comes out of the child's own fold as
    the doors wrote it."""
    ctx = await mount()
    child = _child(ctx)

    record_waiting(child, slots=2)
    await record_started(ctx, child)
    await record_started(ctx, child, cause="resumed")
    await record_started(ctx, child, cause="resumed")

    state = child_state(child)
    assert (state.status, state.starts, restarts_since_progress(state)) == ("running", 3, 2)
    assert (state.run_id, state.name, state.owner, state.parent_id) == (
        "r1",
        "scout",
        "stub",
        "parent",
    )


async def test_a_later_status_does_not_keep_an_earlier_cause(mount: MountProfile) -> None:
    """Each status carries its own `cause` and `detail`, or none. The roster this
    replaced merged each status into its row, so a child woken after a restart went on
    reading `resumed`, and a queued one kept the detail of a wait long over.

    Sabotage: merge a status into the state instead of replacing its fields.
    """
    ctx = await mount()
    child = _child(ctx)

    await record_started(ctx, child, cause="resumed")
    record_waiting(child, detail=SUSPENDED_DETAIL)
    await record_started(ctx, child)

    state = child_state(child)
    assert (state.status, state.cause, state.detail) == ("running", None, None)


async def test_an_answer_forgives_the_restarts_before_it(mount: MountProfile) -> None:
    """The ladder's rule: a child stopped, working, then stopped again met two
    incidents, not a lifetime's. The answer is in the child's own log — nothing had to
    be copied to its parent for the ladder to see it, or caught up after a crash (L5).
    """
    ctx = await mount()
    child = _child(ctx)

    await record_started(ctx, child, cause="resumed")
    await record_started(ctx, child, cause="resumed")
    _answer(child)
    before = child_state(child)
    await record_started(ctx, child, cause="resumed")

    assert restarts_since_progress(before) == 0
    assert restarts_since_progress(child_state(child)) == 1
    assert before.tokens == 8


@pytest.mark.parametrize(
    ("ended", "types"),
    [(False, ["subagent/status", "subagent/deleted"]), (True, ["subagent/deleted"])],
)
async def test_a_tombstone_lands_whole_with_its_ending(
    mount: MountProfile, ended: bool, types: list[str]
) -> None:
    """S14, in the child's own log. A child revoked before it ended gets its
    `canceled` in the tombstone's batch — apart, a flush between them left it
    `canceled` and not deleted — and one that already settled is not settled again:
    whether it had is read from its own log. Both are on its disk when the door
    returns, before anything is let go.

    Sabotage: append the two outside a batch, and they carry no batch ref.
    """
    ctx = await mount()
    child = _child(ctx)
    if ended:
        await record_ended(ctx, child, "done")
    written_before = child.seq

    await record_deleted(ctx, child, "parent-teardown")

    written = [one for one in child.events if one.seq >= written_before]
    assert [one.type for one in written] == types
    if not ended:
        assert written[0].batch is not None and written[0].batch == written[1].batch
    state = child_state(child)
    assert (state.deleted, state.status) == (True, "done" if ended else "canceled")
    assert not child_is_live(state)
    assert stored_types(ctx, child.id)[-len(types) :] == types


async def test_a_child_is_tombstoned_once(mount: MountProfile) -> None:
    """A revocation that reaches a child already revoked — a parent's teardown and
    the cascade from the child above it both come to one grandchild — leaves the
    first tombstone standing rather than writing a second.

    Sabotage: drop the `deleted` check from `_tombstone`, and there are two.
    """
    ctx = await mount()
    child = _child(ctx)

    await record_deleted(ctx, child, "not needed")
    await record_deleted(ctx, child, "parent-teardown")

    assert [one.data["reason"] for one in child.events if one.type == DELETED] == ["not needed"]


async def test_a_log_with_no_admission_is_not_a_child(mount: MountProfile) -> None:
    """A workspace can reach a child's disk before its admission does, and a spawn
    refused before it was admitted leaves such a log. It names no child, so no reader
    counts it as one."""
    ctx = await mount()
    child = _child(ctx, admitted=False)
    await record_started(ctx, child)

    assert not child_state(child).admitted
