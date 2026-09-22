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

import logging
from pathlib import Path

import anyio.from_thread
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


async def test_a_write_in_flight_survives_a_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D17 — a session opening mid-write deleted the write's temp.

    `save_bytes` writes `<name>.<hex>.tmp` beside the locator and renames it into
    place; the open-time sweep runs off-thread on every `session/created` and
    collected every file the log did not name — the temp included — so the
    rename failed with `FileNotFoundError`. `ph_rlm.snapshot` writes a kernel
    variable's blob this way *after* its event is durable, which made the loss a
    variable that would not restore. Reproduced here exactly: the sweep runs
    between the temp's write and its rename.

    Sabotage: drop the `is_atomic_temp` guard from `_remove_unreferenced` and
    the write raises.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("ours", session.id))
    rename = Path.replace
    swept: list[list[str]] = []

    def sweep_then_rename(self: Path, target: Path) -> Path:
        if self.name.endswith(".tmp"):
            swept.append(anyio.from_thread.run(store.sweep_session, session))
        return rename(self, target)

    monkeypatch.setattr(Path, "replace", sweep_then_rename)

    ref = await store.save_text(owner=session.id, source="x", suggested_name="x", content="kept")

    assert swept == [[]], "the sweep ran mid-write and collected nothing"
    assert Path(ref.locator).read_text() == "kept"


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

    def explode(_data: object) -> str | None:
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


# ---------------------------------------------- the ordering the sweep needs --


async def test_a_reserved_blob_is_not_collected_before_its_event_lands(
    tmp_path: Path,
) -> None:
    """The window a producer used to leave open, and the sweep walked into.

    A blob is garbage exactly when the log does not name it, so a producer that
    writes first and appends second is indistinguishable from a leak for as long
    as that takes — and the open-time sweep folds the log on another task. This
    is not hypothetical: it deleted the input offload's own history file often
    enough to fail its test under load, before `reserve_bytes` existed.

    `reserve` is the write and `commit` is the rename, so the blob never exists
    at its locator unreferenced: the sweep either finds nothing there, or finds
    it already named.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))

    ref = await store.reserve_text(owner=session.id, source="a", suggested_name="a", content="x")

    # The state a producer is in between the two calls: content on disk, log
    # silent. The sweep must not read that as garbage.
    assert await store.sweep_session(session) == []
    assert not Path(ref.locator).exists(), "reserved, not yet published"

    _named(session, SPILLED, ref.locator)
    assert await store.commit(ref) is True
    assert Path(ref.locator).read_text(encoding="utf-8") == "x"
    assert await store.sweep_session(session) == [], "and it stays, now that the log names it"


async def test_a_write_interrupted_before_its_rename_is_completed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The repair half of write-ahead, and the reason no bookkeeping is needed.

    A run that dies between appending the locator and renaming the bytes leaves
    the log naming a blob that is not at its locator — with the bytes one
    directory down, under the name they were always going to take. The sweep
    holds both halves of that comparison already, so finishing the rename is the
    whole repair, and the log is the only record of intent it needs.

    This is what replaced a set of in-flight reservations. That set existed so
    the sweep could delete abandoned staged files without eating a live one, and
    it could not tell them apart for the reason write-ahead exists: neither is
    referenced yet. Recovering instead of collecting removes the question.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))
    ref = await store.reserve_text(owner=session.id, source="a", suggested_name="a", content="x")
    _named(session, SPILLED, ref.locator)  # the dead run got this far and no further

    with caplog.at_level(logging.INFO, logger="ph.seams.spill"):
        assert await store.sweep_session(session) == []

    assert Path(ref.locator).read_text(encoding="utf-8") == "x", "the rename was finished"
    assert not store._staging_for(ref.locator).exists()
    assert "completed an interrupted write" in caplog.text


async def test_a_blob_the_log_names_and_nothing_holds_is_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The half a sweep that only deletes cannot see.

    The fold knows every locator the log names and every file on disk, and used
    one direction of that: files nothing names. The other direction is a log
    naming a file nothing has, which is what a failed write leaves once the
    append is already durable — and the first thing that meets it is the model,
    following a path that does not resolve.

    Said out loud at the one moment it is cheap to notice, rather than left for
    the read that fails.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))
    _named(session, SPILLED, str(tmp_path / "spill" / session.id / "never-written.md"))

    with caplog.at_level(logging.WARNING, logger="ph.seams.spill"):
        assert await store.sweep_session(session) == []

    assert "names a blob that is not there" in caplog.text
    assert "never-written.md" in caplog.text


async def test_a_staged_blob_no_run_will_commit_is_left_alone(
    tmp_path: Path,
) -> None:
    """Nothing is deleted from staging, and the leak that leaves is the old one.

    A reservation in flight and one abandoned by a dead run look identical:
    neither is referenced, which is the whole point of appending the locator
    second. Collecting the second would therefore race the first, so the sweep
    collects neither — and what is left behind is exactly the leak `SpillClaim`
    already describes, one file per run that died between the write and the
    append. Recovering the *referenced* ones is the case worth having, and it is
    the test above.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))
    ref = await store.reserve_text(owner=session.id, source="a", suggested_name="a", content="x")
    staged = store._staging_for(ref.locator)

    assert await store.sweep_session(session) == []

    assert staged.exists(), "an unreferenced staged blob is nobody's to judge"
