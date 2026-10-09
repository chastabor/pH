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

import ast
import logging
import os
from pathlib import Path

import pytest
from workspace_layout import parsed_modules

from ph.cordis import Context
from ph.keys import SESSIONS, SPILL_STORE
from ph.seams import spill as spill_module
from ph.seams.spill import SpillClaim, SpillRef, SpillStore
from ph.session import Session, SessionStore
from ph.testing import MountProfile, log_event

pytestmark = pytest.mark.anyio

SPILLED = "offload/spilled"
INPUT = "offload/input-spilled"
KERNEL = "kernel/snapshot"


def _store(tmp_path: Path) -> SpillStore:
    return SpillStore(ctx=Context(), root=tmp_path / "spill")


async def _reserve(store: SpillStore, content: bytes) -> SpillRef:
    return await store.reserve(
        store.plan(owner="s1", suggested_name="a", content=content), source="a"
    )


def _claim(label: str, owner: str, event_type: str = SPILLED) -> SpillClaim:
    return SpillClaim(label=label, event_type=event_type, owners=lambda _session: {owner})


def _named(session: Session, event_type: str, locator: str) -> None:
    """The event a producer appends after a successful spill."""
    log_event(session, event_type, {"locator": locator, "bytes": 1})


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


async def test_a_producer_this_profile_does_not_mount_keeps_its_blobs(tmp_path: Path) -> None:
    """S7 — the reference set was only as wide as the claims mounted *now*.

    A session resumed under a profile that drops `input-offload` but keeps
    `tool-result-offload` swept the directory the two share, found the paste's blob
    named by no mounted claim, and deleted it — while the log still pointed the model
    at it. The seam's own locator convention names a blob whoever wrote it.

    Sabotage: fold only the mounted claims' readers, and the paste's blob is gone.
    """
    store = _store(tmp_path)
    session = Session("s1")
    mounted = await store.save_text(owner=session.id, source="a", suggested_name="a", content="a")
    unmounted = await store.save_text(owner=session.id, source="b", suggested_name="b", content="b")
    orphan = await store.save_text(owner=session.id, source="c", suggested_name="c", content="c")
    _named(session, SPILLED, mounted.locator)
    _named(session, INPUT, unmounted.locator)
    store.claim(_claim("tool-result-offload", session.id, SPILLED))

    removed = await store.sweep_session(session)

    assert removed == [orphan.locator]
    assert Path(unmounted.locator).exists(), "a blob the log still names was deleted"


async def test_a_fork_does_not_sweep_a_directory_it_only_inherited(tmp_path: Path) -> None:
    """S7 — owners were folded from the whole log, the seeded prefix included.

    A kernel's blobs live under `kernel/<namespace>`, owned by whichever records name
    it. A fork's prefix names its parent's, so the fork's open-time sweep visited the
    parent's directory and deleted every blob the parent wrote after the fork point:
    variables the parent's next kernel start could no longer restore.

    Sabotage: take owners from every record the fork holds, and the parent's later
    blob is gone.
    """
    spill = _store(tmp_path)
    sessions = SessionStore(ctx=Context())
    parent = sessions.create("parent")
    spill.claim(
        SpillClaim(
            label="rlm-kernel-snapshot",
            event_type=KERNEL,
            owner=lambda data: str(data["namespace"]),
            locator=lambda data: str(data["locator"]),
        )
    )
    early = await spill.save_text(owner="kernel/ns", source="v", suggested_name="a", content="a")
    log_event(parent, KERNEL, {"namespace": "kernel/ns", "locator": early.locator})
    child = sessions.fork(parent, parent.events[-1].seq, "child")
    later = await spill.save_text(owner="kernel/ns", source="v", suggested_name="b", content="b")
    log_event(parent, KERNEL, {"namespace": "kernel/ns", "locator": later.locator})

    assert await spill.sweep_session(child) == []
    assert Path(later.locator).exists(), "the fork deleted its parent's blob"


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

    Its reference is in a field of its own, as a kernel snapshot's is: one in the
    seam's `locator` convention names its blob whether or not a claim is mounted
    (S7), so only a claim's own reader can stop contributing.
    """
    store = _store(tmp_path)
    session = Session("s1")
    ref = await store.save_text(owner=session.id, source="a", suggested_name="a", content="x")
    log_event(session, KERNEL, {"record": {"locator": ref.locator}})
    release = store.claim(
        SpillClaim(
            label="temporary",
            event_type=KERNEL,
            owners=lambda _session: {session.id},
            locator=lambda data: str(data["record"]["locator"]),
        )
    )
    store.claim(_claim("permanent", session.id, INPUT))

    assert await store.sweep_session(session) == []

    release()

    assert await store.sweep_session(session) == [ref.locator]


# ---------------------------------------------- the ordering the sweep needs --


async def test_a_reserved_blob_appears_only_once_committed(tmp_path: Path) -> None:
    """`reserve` is the write and `commit` is the rename, so the blob never sits at
    its locator before the log names it — and a sweep keeps it once it does."""
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))

    ref = await store.reserve_text(owner=session.id, source="a", suggested_name="a", content="x")
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


async def test_a_stored_log_is_handed_out_with_its_interrupted_writes_finished(
    mount: MountProfile,
) -> None:
    """The repair, before anything runs in the session — and the sweep's wiring.

    The blob its log names is at its locator before anyone holds the session — not
    still in `.staging` when a resumed model follows it, as it could be while the
    sweep was detached (`SessionStore.loaded`).

    Through the mounted row rather than `sweep_session`, because a fold that is
    wrong and a fold nobody calls fail differently: the second deletes nothing and
    looks exactly like a clean store, which is what P6-15 was.

    Sabotage: put `spill-local`'s listener back on `session/created`, and the blob
    is still staged when `loaded` returns.
    """
    ctx = await mount()
    store = ctx.require(SPILL_STORE)
    sessions = ctx.require(SESSIONS)
    crashed = Session("s1")
    store.claim(_claim("producer", crashed.id))
    ref = await _reserve(store, b"x")
    _named(crashed, SPILLED, ref.locator)  # the dead run got this far and no further

    session = sessions.adopt(Session(crashed.id, seed=list(crashed.events)))
    await sessions.loaded(session)

    assert Path(ref.locator).read_text(encoding="utf-8") == "x", "handed out still staged"


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


async def test_a_staged_blob_no_run_will_commit_is_collected(tmp_path: Path) -> None:
    """A dead run's reservation that never reached its append is collected.

    It used to be left, one file per run that died between the write and the
    append, because a reservation in flight looks the same — and the sweep ran
    beside producers. It runs only where nothing writes now (`sweep_session`), so
    a stage nothing names can only be a dead run's.

    Sabotage: list only the owner directory and not its `.staging`, and the stage
    is still there.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))
    ref = await _reserve(store, b"x")
    staged = store._staging_for(ref.locator)

    assert await store.sweep_session(session) == [str(staged)]

    assert not staged.exists()


async def test_a_temp_a_killed_write_left_is_collected(tmp_path: Path) -> None:
    """D17's leftover, collected now that nothing writes beside the sweep.

    A kill between `write_atomic`'s write and its rename leaves `<name>.<hex>.tmp`
    beside the locator. It was passed over while the sweep could meet a write in
    flight, and so leaked; the sweep's one caller runs before anything writes the
    session, so a temp it finds is a dead run's.
    """
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))
    temp = store.owner_root(session.id) / "0123abcd-a.0badf00d.tmp"
    temp.parent.mkdir(parents=True)
    temp.write_bytes(b"half a blo")

    assert await store.sweep_session(session) == [str(temp)]


def _called(attribute: str) -> list[str]:
    """Every shipped module that calls a method named `attribute`, once per call —
    or that hands the method on uncalled, the alias a text match would miss."""
    return [
        module
        for module, parsed in parsed_modules().items()
        for node in ast.walk(parsed.tree)
        if isinstance(node, ast.Attribute) and node.attr == attribute
    ]


def _dispatches(event: str) -> list[str]:
    """Every shipped module that dispatches `event` by name, once per call."""
    return [
        module
        for module, parsed in parsed_modules().items()
        for node in ast.walk(parsed.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"emit", "serial", "parallel", "waterfall"}
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == event
    ]


def test_the_sweep_runs_only_where_a_stored_log_is_read() -> None:
    """**The precondition the sweep's deletions rest on, held where it can fail.**

    `sweep_session` collects everything a session's log does not name, which is
    sound only where nothing is writing that session. That holds along one chain,
    and each link is pinned: the spill row's listener is the sweep's one caller;
    the listener hears only `session/loaded`; `SessionStore.loaded` is that event's
    one dispatcher; and its callers are the two readers of a stored log, each before
    the session is handed out. A new link anywhere — a catch-up over live sessions,
    a `gc` command, a second dispatcher — would sweep beside producers and delete
    what they have written and not yet named (S7, D17). Add one only with the guards
    it would need.
    """
    assert _called("sweep_session") == ["ph.seams.spill"]
    assert _dispatches("session/loaded") == ["ph.session.store"]
    assert sorted(_called("loaded")) == ["ph.persistence.jsonl", "ph.persistence.opening"]


# ------------------------------------------------------- planned once, linked --


async def test_a_planned_blob_is_hashed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The locator a record names and the stage that fills it share one digest.

    A producer that must name a blob before storing it used to derive the locator
    for its record and then reserve the bytes, which derived it again: a second
    sha256 of a multi-megabyte result, on the event loop, for an answer it had.

    Sabotage: derive the locator again in `reserve`, and this counts two.
    """
    store = _store(tmp_path)
    derived: list[str] = []
    original = SpillStore.locator_for

    def counting(self: SpillStore, **kwargs: object) -> Path:
        derived.append(str(kwargs["suggested_name"]))
        return original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(SpillStore, "locator_for", counting)
    planned = store.plan(owner="s1", suggested_name="a", content=b"x" * 64)
    ref = await store.reserve(planned, source="a")

    assert derived == ["a"]
    assert ref.locator == str(planned.locator)


async def test_bytes_already_published_are_linked_rather_than_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blob whose digest is already at its locator costs a link, not a write.

    The name carries the digest, so the file there is the write a reservation would
    make — a kernel variable back at earlier bytes, a tool repeating its output.
    Linked rather than skipped, so the sweep deleting the published copy before the
    record lands cannot take the bytes; and the link is gone once committed, or it
    would pin them after the sweep collects the blob.
    """
    store = _store(tmp_path)
    first = await _reserve(store, b"same")
    assert await store.commit(first)
    written: list[Path] = []
    monkeypatch.setattr(spill_module, "_write", lambda path, _payload: written.append(path))

    again = await _reserve(store, b"same")
    staged = store._staging_for(again.locator)

    assert written == [], "the published bytes were written a second time"
    assert os.path.samefile(staged, again.locator)
    assert await store.commit(again)
    assert not staged.exists(), "the staging link was left to pin the blob"
    assert Path(again.locator).read_bytes() == b"same"


async def test_two_reservations_of_the_same_bytes_both_commit(tmp_path: Path) -> None:
    """They share one staged name, so the first commit takes it — and the second
    finds its bytes published, which is success rather than a missing stage."""
    store = _store(tmp_path)
    one = await _reserve(store, b"same")
    two = await _reserve(store, b"same")

    assert await store.commit(one) and await store.commit(two)
    assert Path(two.locator).read_bytes() == b"same"


async def test_a_redundant_stage_the_log_names_is_dropped(tmp_path: Path) -> None:
    """A dead run that linked a stage and appended its record, then died before the
    commit that would have dropped the link: the blob is published and named, so
    the stage is redundant — and kept, it would pin the bytes past the blob."""
    store = _store(tmp_path)
    session = Session("s1")
    store.claim(_claim("producer", session.id))
    first = await store.reserve_text(owner=session.id, source="a", suggested_name="a", content="x")
    await store.commit(first)
    again = await store.reserve_text(owner=session.id, source="a", suggested_name="a", content="x")
    _named(session, SPILLED, again.locator)

    assert await store.sweep_session(session) == []

    assert not store._staging_for(again.locator).exists()
    assert Path(again.locator).read_text(encoding="utf-8") == "x"
