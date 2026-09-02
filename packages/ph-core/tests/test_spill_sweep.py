"""P6-15 — one open-time sweep, and every producer contributing to its fold (F7).

Gate: *a blob whose event never landed is gone after the next session open; one
whose event did land is not.*

**The sweep began as one producer's private listener, and that is the defect.**
`ph_rlm.snapshot` shipped `session/created` → sweep for `kernel/<namespace>`
owners, folding the locators its *own* events named. Three producers arrived
afterwards — the tool-result offload, the input offload, the compaction history
— all writing under `session.id`, and nothing ever collected any of them. A crash
between a blob write and the append that names it leaked a file permanently, with
nothing to notice: the store has no index, so an orphan is indistinguishable from
a file somebody still wants.

**The fix is not a second sweep, and the first test is why.** Those three
producers share one owner directory. A per-producer sweep there would have each
of them delete the other two's files — correctly, by its own lights, since it
would find files its own events do not name. So the seam unions every claim's
references *before* visiting any owner.

**And it is one pass.** The first version of `SpillClaim` carried two whole-log
callables per claim, so four claims cost eight passes over the log at every
session open: measured **27.4 ms at 500 000 events**, on the event loop, and
again on every fork (whose child is seeded with the whole parent prefix). A claim
now names its event type and reads one event's owner and locator, so the seam
dispatches by type in a single pass and runs the whole sweep on a worker thread:
**15.4 ms including the thread hop and the directory walks**, off the loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.seams.spill import SpillClaim, SpillStore
from ph.session import Session

pytestmark = pytest.mark.anyio

SPILLED = "offload/spilled"
INPUT = "offload/input-spilled"


def _store(tmp_path: Path) -> SpillStore:
    return SpillStore(ctx=Context(), root=tmp_path / "spill")


def _claim(label: str, owner: str, event_type: str = SPILLED) -> SpillClaim:
    return SpillClaim(label=label, event_type=event_type, owners=lambda _session: {owner})


def _named(session: Session, event_type: str, locator: str) -> None:
    """The event a producer appends after a successful spill."""
    session.append(event_type, {"locator": locator, "bytes": 1})


# ------------------------------------------------------------------ the union --


async def test_two_producers_sharing_an_owner_do_not_collect_each_other(
    tmp_path: Path,
) -> None:
    """`tool-result-offload`, `input-offload` and `compaction-summarize` all write
    under `session.id`. Swept per producer, each walks that directory, finds the
    other two's files unreferenced *by its own events*, and deletes them — every
    one behaving correctly and the result being data loss."""
    store = _store(tmp_path)
    session = Session("s1")
    mine = await store.save_text(owner=session.id, source="a", suggested_name="a", content="mine")
    yours = await store.save_text(owner=session.id, source="b", suggested_name="b", content="yours")
    orphan = await store.save_text(owner=session.id, source="c", suggested_name="c", content="none")
    _named(session, SPILLED, mine.locator)
    _named(session, INPUT, yours.locator)

    store.claim(_claim("first", session.id, SPILLED))
    store.claim(_claim("second", session.id, INPUT))

    removed = await store.sweep_session(session)

    assert removed == [orphan.locator], removed
    assert Path(mine.locator).exists() and Path(yours.locator).exists()


async def test_an_owner_no_claim_names_is_never_visited(tmp_path: Path) -> None:
    """A directory nobody claims is somebody else's, or nobody's yet — deleting in
    it on the strength of an empty reference set is the kernel sweep's old failure
    from the other side, where visiting every namespace the process had seen
    deleted another session's blobs."""
    store = _store(tmp_path)
    session = Session("s1")
    theirs = await store.save_text(
        owner="someone-else", source="x", suggested_name="x", content="theirs"
    )
    store.claim(_claim("ours", session.id))

    assert await store.sweep_session(session) == []
    assert Path(theirs.locator).exists()


# ------------------------------------------------------------------- refusal --


async def test_a_claim_that_raises_stops_the_sweep_rather_than_narrowing_it(
    tmp_path: Path,
) -> None:
    """A reference set assembled from *some* of the claims is smaller than the
    truth, and a small reference set does not skip work — it deletes live blobs.
    Losing a sweep costs disk until the next open; running a partial one costs the
    conversation."""
    store = _store(tmp_path)
    session = Session("s1")
    kept = await store.save_text(owner=session.id, source="a", suggested_name="a", content="live")
    _named(session, SPILLED, kept.locator)
    _named(session, INPUT, "irrelevant")

    def explode(_data: Any) -> str | None:
        raise RuntimeError("this producer cannot answer")

    store.claim(_claim("healthy", session.id, SPILLED))
    store.claim(
        SpillClaim(label="broken", event_type=INPUT, owners=lambda _s: set(), locator=explode)
    )

    assert await store.sweep_session(session) == []
    assert Path(kept.locator).exists(), "a partial fold must not be used to delete"


async def test_a_withdrawn_claim_stops_contributing(tmp_path: Path) -> None:
    """The registration is an effect, so a row that unmounts stops being asked.

    Observable only with a second claim still holding the owner open: while both
    are registered the first one's reference keeps the file; once it is released
    the owner is still visited — the second claim names it — and the file it
    alone referenced is collected. A `release()` that did nothing would leave the
    file in place and fail here.
    """
    store = _store(tmp_path)
    session = Session("s1")
    ref = await store.save_text(owner=session.id, source="a", suggested_name="a", content="x")
    _named(session, SPILLED, ref.locator)
    release = store.claim(_claim("temporary", session.id, SPILLED))
    store.claim(_claim("permanent", session.id, INPUT))

    assert await store.sweep_session(session) == []

    release()

    assert await store.sweep_session(session) == [ref.locator]
