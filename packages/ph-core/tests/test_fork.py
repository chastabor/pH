"""P0-11 — seed, fork and resume.

Gates: *a fork inside an open turn is refused; a fork at a boundary replays
identically.*

Branching in pH is `fork(source, boundary)` plus `seed_length`, not a message
tree (D2). The open-turn refusal is what keeps that honest: a fork that ended
mid-turn would inherit a half-executed step whose tool results never arrive.

## What reference-forking bought, and what it did not

**On disk it is the whole point.** Ten forks of a 5 000-event session wrote
**4.21 MiB across ten files** before, and **1 980 bytes in total** after —
**2 231x less** — while reading any child back still yields the same 5 001
contiguous events.

**In memory it is not free, and an earlier docstring said it was** ("pointers to
the parent's own immutable events, so sharing costs nothing"). `Session.__init__`
runs `_readmit`, which calls `SessionEvent.readmitted()` on every seeded event: a
fresh object plus a full re-freeze of the payload. Measured on a 5 000-event
parent: **18.2 ms for the fork, 14.4 ms of it (79%) in `readmitted`, 2.2 MB
allocated.** So a fork is not yet O(1) end to end.

The re-freeze is genuinely needed on the replay and wire paths it was written for,
and is duplicate work here, where the seed comes from a live log every event of
which `append` already froze. Left alone deliberately — a trusted-seed path is a
change to the session model, not to `fork` — but `roll` turns forking into a
routine operation, so the next person should know where the time goes.

## The read amplification `upto` removes

`read_one` was a whole-log read at first, so an ancestor contributing **50 events
out of 10 000** was parsed and validated in full: **~170x the useful work at an
early fork boundary**, though only **2.6% at the tip**, where the child wants
nearly all of it anyway. `upto` carries the boundary now — Turso adds
`WHERE seq < ?`, JSONL stops reading lines — and the tail costs a dict lookup
instead of a validate and freeze.

`upto` is deliberately **a hint, not a contract**: the walk re-filters what it
takes, so a backend that ignores it is slower rather than wrong. Making the bound
load-bearing would mean a backend quietly ignoring it produced a *wrong* log,
which is exactly how reference-forking came to be a silent no-op on Turso once
already.

## Why `fork_boundaries` folds the whole log in one pass

Asking `open_turn_at` per record rescans the prefix each time, which is quadratic:
an **8 000-event session cost 467 ms to fold**, and that became a frozen UI once
the trajectory could be opened from a running chat (P4-17).
"""

from __future__ import annotations

import pytest

from ph.cordis import Context
from ph.session import (
    Session,
    SessionForkError,
    SurfaceIntent,
    fork_boundaries,
    is_fork_boundary,
)
from ph.session.events import SessionEvent
from ph.session.store import SessionStore
from ph.testing import user_payload


def _store() -> SessionStore:
    return SessionStore(ctx=Context())


def _closed_turn(session: Session, turn: int, text: str) -> None:
    session.append("turn/start", {"turn": turn})
    session.append("step/start", {"turn": turn, "step": 1})
    session.append("user/message", user_payload(text, f"m{turn}"), SurfaceIntent("append"))
    session.append("step/end", {"turn": turn, "step": 1})
    session.append("turn/end", {"turn": turn, "reason": {"kind": "completed"}})


def test_seeding_marks_the_end_of_the_seed() -> None:
    parent = Session("p")
    _closed_turn(parent, 1, "hello")
    child = Session("c", seed=list(parent.events))
    assert child.first_live_seq == len(parent.events)
    assert child.events[-1].type == "session/end-seed"

    # Reopening an untouched session must not grow its log per open.
    reopened = Session("c2", seed=list(child.events))
    assert reopened.events[-1].type == "session/end-seed"
    assert len(reopened.events) == len(child.events)


def test_a_seed_is_validated_to_the_same_rules_as_an_append() -> None:
    bad = SessionEvent(type="turn/start", seq=3, time=1, data={"turn": 1})
    with pytest.raises(ValueError, match="seed must be contiguous"):
        Session("c", seed=[bad])

    unmarked = SessionEvent(type="user/message", seq=0, time=1, data=user_payload("x"))
    with pytest.raises(ValueError, match="requires a surfaceOp"):
        Session("c", seed=[unmarked])

    # A hand-built event's plain-dict payload is frozen on the way in, so the
    # log never holds a mutable tree.
    plain = SessionEvent(type="turn/start", seq=0, time=1, data={"turn": 1})
    with pytest.raises(TypeError):
        Session("c", seed=[plain]).events[0].data["turn"] = 2  # type: ignore[index]


def test_an_unrecognized_required_event_refuses_the_seed() -> None:
    future = SessionEvent(type="quantum/entangle", seq=0, time=1, data={})
    # Silently skipping it would reconstruct a WRONG session, not a partial one —
    # and this refusal is on the one path every seed takes, not in one backend.
    with pytest.raises(ValueError, match="unrecognized required type"):
        Session("f", seed=[future])


def test_an_unrecognized_ignorable_event_is_accepted() -> None:
    note = SessionEvent(type="telemetry/note", seq=0, time=1, data={}, ignorable=True)
    session = Session("f", seed=[note])
    assert [event.type for event in session.events] == ["telemetry/note", "session/end-seed"]


def test_fork_at_a_closed_turn_replays_identically() -> None:
    store = _store()
    parent = store.create("parent")
    _closed_turn(parent, 1, "first")
    _closed_turn(parent, 2, "second")
    boundary = parent.events[-1].seq

    child = store.fork(parent, boundary, "child")
    assert child.header.parent_session == "parent"
    assert child.header.seed_length == boundary + 1
    # Identical derivation is the whole point: the fork continues the same
    # conversation the model was in, as of that boundary.
    assert child.derive_messages() == parent.derive_messages()


def test_fork_at_an_earlier_boundary_takes_only_that_prefix() -> None:
    store = _store()
    parent = store.create("parent")
    _closed_turn(parent, 1, "first")
    first_turn_end = parent.events[4].seq
    _closed_turn(parent, 2, "second")

    child = store.fork(parent, first_turn_end, "child")
    assert [m.content[0].text for m in child.derive_messages()] == ["first"]
    # The parent is untouched by the fork.
    assert [m.content[0].text for m in parent.derive_messages()] == ["first", "second"]


def test_fork_inside_an_open_turn_is_refused() -> None:
    store = _store()
    parent = store.create("parent")
    parent.append("turn/start", {"turn": 1})
    parent.append("step/start", {"turn": 1, "step": 1})
    parent.append("user/message", user_payload("mid-turn", "m1"), SurfaceIntent("append"))

    with pytest.raises(SessionForkError) as caught:
        store.fork(parent, parent.events[-1].seq, "child")
    assert caught.value.code == "OPEN_TURN"


def test_fork_boundary_must_exist() -> None:
    store = _store()
    parent = store.create("parent")
    _closed_turn(parent, 1, "first")
    for boundary in (-1, 999):
        with pytest.raises(SessionForkError) as caught:
            store.fork(parent, boundary, f"child{boundary}")
        assert caught.value.code == "INVALID_BOUNDARY"


def test_fork_of_an_unknown_session_is_refused() -> None:
    with pytest.raises(SessionForkError) as caught:
        _store().fork("ghost")
    assert caught.value.code == "SESSION_NOT_FOUND"


def test_forking_onto_an_existing_id_is_refused() -> None:
    store = _store()
    parent = store.create("parent")
    _closed_turn(parent, 1, "first")
    store.create("taken")
    with pytest.raises(SessionForkError) as caught:
        store.fork(parent, None, "taken")
    assert caught.value.code == "SESSION_ALREADY_EXISTS"


def test_forking_an_empty_session_yields_an_empty_child() -> None:
    store = _store()
    child = store.fork(store.create("parent"), None, "child")
    # An empty fork still records where its (empty) seed ended, so a consumer
    # reading STORED history can tell inherited prefix from live work without
    # consulting the header.
    assert [event.type for event in child.events] == ["session/end-seed"]
    assert child.header.seed_length == 0
    assert child.first_live_seq == 0
    assert child.derive_messages() == ()


def test_fork_boundaries_agrees_with_the_per_boundary_rule() -> None:
    """A6 has two statements now, and they must not drift.

    `is_fork_boundary` answers for one seq by rescanning the prefix;
    `fork_boundaries` answers for the whole log in one pass, because the reader
    that *marks* which records a fork may aim at asks for every one of them and
    the rescan is quadratic — 8 000 events cost 467 ms, on a keypress. A second
    statement of a rule is exactly how the trajectory came to mark one legal
    boundary in four while citing A6, so the two are pinned to each other here
    over a log holding open turns, closed turns and events between them.
    """
    session = Session("boundaries")
    _closed_turn(session, 1, "first")
    session.append("approval/policy", {"mode": "ask"})
    _closed_turn(session, 2, "second")
    session.append("turn/start", {"turn": 3})
    session.append("user/message", user_payload("mid-turn"), SurfaceIntent("append"))
    log = session.events

    fast = fork_boundaries(log)
    slow = {event.seq for event in log if is_fork_boundary(log, event.seq)}

    assert fast == slow
    # And it is answering something, rather than agreeing by being empty: the
    # targets are the two closed turns and the event between them, and nothing
    # inside the turn that never closed.
    assert {log[seq].type for seq in fast} == {"turn/end", "approval/policy"}
    assert not fast & {event.seq for event in log[-2:]}, "an open turn is not a target"


# ------------------------------------------------- segmentation (§7 step 6) --


def _rolled() -> tuple[SessionStore, Session, Session]:
    """A store, a session with one closed turn, and the segment continuing it."""
    store = _store()
    parent = store.create("p")
    _closed_turn(parent, 1, "hello")
    return store, parent, store.roll(parent, "p2")


def test_a_roll_continues_the_log_in_a_fresh_session() -> None:
    """**A fork at the tip with no divergence**, which is the whole mechanism.

    The child inherits every event and adds nothing of its own yet, so the two
    logs agree completely up to the seam. What makes it a *segment* rather than a
    branch is only that the parent stops — and the marker is what says so.
    """
    _, parent, child = _rolled()

    assert child.header.parent_session == "p"
    assert child.header.seed_length == 5, "everything up to the tip"
    assert [event.type for event in child.events[:5]] == [event.type for event in parent.events[:5]]
    assert child.events[-1].type == "session/end-seed"


def test_the_segment_marker_is_the_parents_own_and_is_not_inherited() -> None:
    """Appended after the fork, so it is the parent's terminal record.

    The link exists in both directions from state that already exists — the
    parent names its continuation, the child's header names its origin — and
    neither needs a field that is not already written. Inheriting it would make
    the child claim to have been segmented, which is the opposite of true.
    """
    _, parent, child = _rolled()

    assert parent.events[-1].type == "session/segmented"
    assert parent.events[-1].data["continues"] == "p2"
    assert "session/segmented" not in [event.type for event in child.events]


def test_a_roll_inside_an_open_turn_is_refused() -> None:
    """Inherited from `_fork_seed`, and the right rule to inherit: a log that
    begins mid-step is not resumable, however it came to begin there."""
    store = _store()
    session = store.create("p")
    session.append("turn/start", {"turn": 1})

    with pytest.raises(SessionForkError) as caught:
        store.roll(session)
    assert caught.value.code == "OPEN_TURN"


def test_a_parent_that_keeps_writing_after_a_roll_has_made_a_branch() -> None:
    """**The parent is left live on purpose.**

    Disposing it would unhook the persistence observer while the `Session` object
    stayed perfectly usable, so an append through a stale reference would vanish
    with nothing raised. Left published, an in-flight append still lands — and a
    caller that goes on writing has made a branch, which is a legitimate thing to
    have done and is visible as one, its events sitting after the marker.
    """
    store, parent, _ = _rolled()

    assert store.get("p") is parent
    _closed_turn(parent, 2, "and again")
    types = [event.type for event in parent.events]
    assert types[5] == "session/segmented"
    assert types[6:], "the branch is in the log, not lost"


def test_a_child_inherits_its_parents_lineage_however_it_was_made() -> None:
    """**Not only through `fork` and `roll`.**

    Inheritance was enforced inside `_branch`, and the other caller that creates
    a child — the subagent spawn, which goes straight to `create` with a
    `parentSession` in its meta — was not updated. Every subagent got a family
    directory of its own, so the layout's central claim, one conversation and
    everything it spawned in one directory, was false for the thing that spawns
    most. The rule lives at the single construction gate now, where a third
    branching caller cannot forget it.
    """
    store = _store()
    root = store.create("r")
    _closed_turn(root, 1, "hello")
    branch = store.fork(root, None, "b")
    segment = store.roll(root, "s")
    subagent = store.create("kid", meta={"parentSession": root.id, "origin": "subagent"})
    grandchild = store.create("g", meta={"parentSession": branch.id})

    assert root.header.family == "r", "a root heads its own lineage"
    assert [one.header.family for one in (branch, segment, subagent, grandchild)] == ["r"] * 4
