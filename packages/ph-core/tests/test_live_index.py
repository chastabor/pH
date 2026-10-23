"""P13-01 — the live-roots index: which roots a host holds busy, and why.

The index is a cache over facts the logs still hold, so the gates are the schedule
index's, one level over: the fold that derives a root's holds from its log agrees
with the projection the agent keeps live (`replay_pending` is the inbox's own
constructor fold), a rebuild writes what every stored root's log says and keeps the
entry of a log it cannot read, and a file that cannot vouch for itself reads as
empty and untrusted rather than as an exception on a daemon's boot path.
"""

from __future__ import annotations

from pathlib import Path

from ph.agent.inbox import Inbox, InboxNotifications, replay_pending
from ph.seams.goals import SET, SETTLED, Goal
from ph.seams.live_index import LiveEntry, LiveIndex, holds_of, rebuild_live_index
from ph.seams.subagents import ADMITTED, STATUS
from ph.session import Session, SessionHeader
from ph.session.store import child_session_id
from ph.testing import StoredLogs, log_event, user_payload


def _root(session_id: str = "root") -> Session:
    return Session(session_id, header=SessionHeader(id=session_id, created_at=1, cwd="/tmp/work"))


def _child(parent: Session, run_id: str, *, status: str) -> Session:
    child_id = child_session_id(parent.id, run_id)
    child = Session(
        child_id,
        header=SessionHeader(
            id=child_id,
            created_at=2,
            cwd=parent.header.cwd,
            parent_session=parent.id,
            origin="subagent",
            family=parent.header.family,
        ),
    )
    log_event(child, ADMITTED, {"runId": run_id, "name": run_id, "owner": "stub"})
    log_event(child, STATUS, {"status": status})
    return child


def _splice(session: Session, text: str, message_id: str) -> None:
    log_event(
        session,
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": [user_payload(text, message_id)]},
    )


# ------------------------------------------------------------------- the fold --


def test_holds_of_reads_every_hold_off_a_log() -> None:
    """An open turn, an open goal, a pending prompt and a live child are the four
    reasons a root still owes work, and each is read from the record that says so."""
    root = _root()
    log_event(root, "turn/start", {"turn": 1})
    log_event(root, SET, Goal(id="g1", objective="ship it").to_wire())
    _splice(root, "and then this", "m1")
    assert holds_of(root.header, root.events, children_live=True) == (
        "turn",
        "goal",
        "children",
        "inbox",
    )


def test_a_root_that_owes_nothing_has_no_holds() -> None:
    """A closed turn, a settled goal and a claimed prompt leave nothing to resume."""
    root = _root()
    log_event(root, "turn/start", {"turn": 1})
    log_event(root, SET, Goal(id="g1", objective="ship it").to_wire())
    _splice(root, "hi", "m1")
    log_event(
        root,
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": [], "removedCount": 1},
    )
    log_event(root, SETTLED, {"id": "g1", "outcome": "achieved", "detail": ""})
    log_event(root, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert holds_of(root.header, root.events, children_live=False) == ()


def test_replay_pending_is_the_inbox_constructors_own_fold() -> None:
    """The index reads a stored log with no `Session`, so the fold it uses must be
    the one the live projection is built from — or the two drift (A11)."""
    root = _root()
    _splice(root, "first", "m1")
    _splice(root, "second", "m2")
    log_event(
        root,
        "agent/inbox/spliced",
        {"target": "next-turn", "start": 0, "inserted": [], "removedCount": 1},
    )
    quiet = InboxNotifications(
        inserted=lambda _m: None, discarded=lambda _m: None, claimed=lambda _m, _t: None
    )
    live = Inbox(root, quiet)
    folded = replay_pending(root.own_events())
    assert [m.id for m in live.next_turn] == [m.id for m in folded["next-turn"]] == ["m1"]


# ------------------------------------------------------------------ the file --


def test_the_index_records_clears_and_vouches(tmp_path: Path) -> None:
    index = LiveIndex(tmp_path)
    assert index.read() == {}
    assert not index.survey().complete, "a missing file cannot vouch for itself"
    entry = LiveEntry("root", ("turn", "children"), updated=5, family="f", cwd="/w")
    assert index.record("root", entry)
    assert index.read() == {"root": entry}
    assert not index.survey().complete, "a writer keeps the completeness it found"
    assert index.record("root", None)
    assert index.read() == {}
    assert index.replace({"root": entry}, since=index.survey())
    assert index.survey().complete, "only a rebuild marks the file complete"
    assert index.survey().entries == {"root": entry}


def test_a_damaged_index_reads_as_empty_and_untrusted(tmp_path: Path) -> None:
    index = LiveIndex(tmp_path)
    index.path.write_text("{not json", encoding="utf-8")
    assert index.read() == {}
    assert not index.survey().complete


def test_a_hold_this_build_does_not_know_is_dropped_on_read(tmp_path: Path) -> None:
    """A newer daemon's hold is not carried as a string nothing branches on."""
    index = LiveIndex(tmp_path)
    index.path.write_text(
        '{"version": 1, "complete": true, "sessions": {"root": {"holds": ["turn", "dreams"], '
        '"updated": 3}}}',
        encoding="utf-8",
    )
    assert index.read()["root"].holds == ("turn",)


# --------------------------------------------------------------- the rebuild --


def test_a_rebuild_writes_what_every_root_log_says(tmp_path: Path) -> None:
    """A root killed mid-turn with a child still working is owed; a balanced root is
    not, and a child's log is never a root of its own."""
    busy, done = _root("busy"), _root("done")
    log_event(busy, "turn/start", {"turn": 1})
    child = _child(busy, "c1", status="running")
    log_event(done, "turn/start", {"turn": 1})
    log_event(done, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    index = LiveIndex(tmp_path)
    index.record("done", LiveEntry("done", ("turn",), updated=1))
    assert rebuild_live_index(index, StoredLogs(busy, done, child), now=9, since=index.survey())
    survey = index.survey()
    assert survey.complete
    assert set(survey.entries) == {"busy"}
    assert survey.entries["busy"].holds == ("turn", "children")
    assert survey.entries["busy"].family == busy.header.family
    assert survey.entries["busy"].cwd == "/tmp/work"


def test_a_rebuild_keeps_the_entry_of_a_log_it_cannot_read(tmp_path: Path) -> None:
    """A torn log is no evidence the work is done: its entry stands as the last host
    left it, and the rebuild still answers for every other root."""
    torn, fine = _root("torn"), _root("fine")
    log_event(fine, "turn/start", {"turn": 1})
    index = LiveIndex(tmp_path)
    kept = LiveEntry("torn", ("goal",), updated=1)
    index.record("torn", kept)
    assert rebuild_live_index(
        index, StoredLogs(torn, fine, broken="torn"), now=9, since=index.survey()
    )
    entries = index.read()
    assert entries["torn"] == kept
    assert entries["fine"].holds == ("turn",)


def test_an_entry_that_differs_only_in_when_it_was_said_is_no_change(tmp_path: Path) -> None:
    """`updated` is not compared, so a host re-stating a root's holds writes nothing
    and the entry keeps the moment the root began to owe them."""
    index = LiveIndex(tmp_path)
    first = LiveEntry("root", ("turn",), updated=1)
    assert index.record("root", first)
    before = index.path.stat().st_mtime_ns
    assert index.record("root", LiveEntry("root", ("turn",), updated=99))
    assert index.path.stat().st_mtime_ns == before
    assert index.read()["root"].updated == 1


def test_a_rebuild_takes_a_mounted_roots_entry_from_its_host(tmp_path: Path) -> None:
    """A mounted root's log lags its host until the next barrier, so what the host
    says of it wins over what its log says."""
    root = _root()
    index = LiveIndex(tmp_path)
    said = LiveEntry("root", ("turn",), updated=5, profile="work")
    assert rebuild_live_index(
        index, StoredLogs(root), now=9, since=index.survey(), live={"root": said}
    )
    assert index.read()["root"].profile == "work"
