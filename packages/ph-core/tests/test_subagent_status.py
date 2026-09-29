"""S10 — a child's status, through the seam's doors.

A provider reports how its child moves in the parent's log, and two of those
reports carry a rule about *when* they reach disk: a restart must be written before
the attempt it counts, and an ending must follow the child's own account of it. Each
provider used to keep those rules by hand. `ph.seams.subagents` keeps them now, in
three doors every provider calls — and `test_log_writers` holds every provider to
them, since none may append a `subagent/status` of its own.
"""

from __future__ import annotations

import pytest

from ph.keys import SESSIONS
from ph.seams.subagents import (
    record_settled,
    record_started,
    record_status,
    restarts_since_progress,
    subagent_roster,
)
from ph.testing import MountProfile, log_event, stored_types

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
