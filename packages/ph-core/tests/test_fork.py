"""P0-11 — seed, fork and resume.

Gates: *a fork inside an open turn is refused; a fork at a boundary replays
identically.*

Branching in pH is `fork(source, boundary)` plus `seed_length`, not a message
tree (D2). The open-turn refusal is what keeps that honest: a fork that ended
mid-turn would inherit a half-executed step whose tool results never arrive.
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
