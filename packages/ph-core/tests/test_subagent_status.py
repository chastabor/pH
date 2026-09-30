"""S10 — a child's roster records, through the seam's doors.

A provider reports how its child moves in the parent's log, and some of those
reports carry a rule about *when* or *how* they reach disk: a restart must be written
before the attempt it counts, an ending must follow the child's own account of it,
and a tombstone must land in one batch with the `canceled` that ends it. Each provider
used to keep those rules by hand. `ph.seams.subagents` keeps them now, in doors every
provider calls — and `test_log_writers` holds every provider to them, since none may
append a roster record of its own.
"""

from __future__ import annotations

import pytest

from ph.keys import SESSIONS
from ph.seams.subagents import (
    record_deleted,
    record_settled,
    record_started,
    record_status,
    restarts_since_progress,
    subagent_roster,
    usage_mirror,
)
from ph.session import SurfaceIntent
from ph.testing import MountProfile, assistant_payload, log_event, stored_types

pytestmark = pytest.mark.anyio


async def test_a_restart_is_on_disk_when_its_door_returns(mount: MountProfile) -> None:
    """The ladder counts restarts, so a restart written only to memory before the
    child took the daemon down again was never counted — a crash loop that never
    tripped `CHILD_RETRY_LIMIT`.

    Sabotage: drop the flush from `record_started`, and nothing is on disk.
    """
    ctx = await mount()
    parent = ctx.require(SESSIONS).create("parent")

    await record_started(ctx, parent, "r1", cause="resumed")

    assert stored_types(ctx, "parent") == ["subagent/status"]


async def test_a_first_start_rides_the_next_flush(mount: MountProfile) -> None:
    """A first start counts nothing, so it pays for no write of its own."""
    ctx = await mount()
    parent = ctx.require(SESSIONS).create("parent")

    event = await record_started(ctx, parent, "r1")

    assert event.data == {"runId": "r1", "status": "running"}
    assert stored_types(ctx, "parent") == []


async def test_an_ending_follows_the_childs_own_account_of_it(mount: MountProfile) -> None:
    """F1. A parent that read "done, here is a preview" while the child's log ended
    before the answer had repair call the child interrupted on the next open.

    Sabotage: drop the child's flush from `record_settled`, and its answer is not on
    disk when its parent says it finished.
    """
    ctx = await mount()
    parent = ctx.require(SESSIONS).create("parent")
    child = ctx.require(SESSIONS).create("child")
    log_event(child, "turn/start", {"turn": 1})

    ended = await record_settled(ctx, parent, "r1", "done", child=child, answerPreview="42")

    assert stored_types(ctx, "child") == ["turn/start"]
    assert ended.data == {"runId": "r1", "status": "done", "answerPreview": "42"}


async def test_the_roster_folds_what_the_doors_write(mount: MountProfile) -> None:
    """One writer, one shape: the ladder's count comes out of the fold as written."""
    ctx = await mount()
    parent = ctx.require(SESSIONS).create("parent")
    log_event(parent, "subagent/admitted", {"runId": "r1", "name": "scout"})

    record_status(parent, "r1", "queued", slots=2)
    await record_started(ctx, parent, "r1")
    await record_started(ctx, parent, "r1", cause="resumed")
    await record_started(ctx, parent, "r1", cause="resumed")

    row = subagent_roster(parent)["r1"]
    assert (row["status"], row["starts"], restarts_since_progress(row)) == ("running", 3, 2)


@pytest.mark.parametrize(
    ("ended", "types"),
    [(False, ["subagent/status", "subagent/deleted"]), (True, ["subagent/deleted"])],
)
async def test_a_tombstone_lands_with_its_ending(
    mount: MountProfile, ended: bool, types: list[str]
) -> None:
    """S14. A child revoked before it ended gets its `canceled` in the tombstone's
    batch — apart, a flush between them left it `canceled` and not deleted — and one
    that already settled is not settled again.

    Sabotage: append the two outside a batch, and they carry no batch ref.
    """
    ctx = await mount()
    parent = ctx.require(SESSIONS).create("parent")
    log_event(parent, "subagent/admitted", {"runId": "r1", "name": "scout"})

    record_deleted(parent, "r1", "parent-teardown", ended=ended)

    written = [one for one in parent.events if one.type.startswith("subagent/") and one.seq > 0]
    assert [one.type for one in written] == types
    if not ended:
        assert written[0].batch is not None and written[0].batch == written[1].batch
    assert subagent_roster(parent)["r1"]["deleted"] is True


async def test_the_usage_mirror_charges_each_answer_to_the_parent(mount: MountProfile) -> None:
    """The live half of L5: an answer the child makes is charged as it is made, and
    nothing else the child logs is."""
    ctx = await mount()
    parent = ctx.require(SESSIONS).create("parent")
    child = ctx.require(SESSIONS).create("child")
    log_event(parent, "subagent/admitted", {"runId": "r1", "name": "scout"})
    child.observe(usage_mirror(parent, "r1"))

    log_event(child, "turn/start", {"turn": 1})
    answer = log_event(
        child,
        "assistant/message",
        {**assistant_payload("done", "m1"), "usage": {"inputTokens": 3, "outputTokens": 5}},
        SurfaceIntent("append", ()),
    )

    (charged,) = [one for one in parent.events if one.type == "subagent/usage-attributed"]
    assert charged.data["targetSeq"] == answer.seq
    assert charged.data["origin"] == "spawn_task"
