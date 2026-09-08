"""`Session.admit` — the replica's door into a log (P6-44).

`append` is for the process that owns a log: it mints `seq` and stamps `time`. A
front end mirroring a daemon's session over the wire owns nothing; it receives
events the daemon already stamped and must keep them as they are. Before this
door existed the TUI kept a bare list and rebuilt `Session(seed=…)` from it on
every read — re-validating the whole log each time, and growing a
`session/end-seed` marker the daemon's log never had.

What is pinned: the seed path's acceptance rules apply (contiguity, known types),
the daemon's stamps survive, the surface holds a replica to the same rules it
holds an owner, observers cannot tell which door an event used, and a log
admitted event by event *is* the log — the same surface, `stale()` clean.
"""

from __future__ import annotations

import pytest

from ph.session import Session, SessionEvent, SurfaceError, SurfaceIntent
from ph.session.session import SessionHeader
from ph.testing import assistant_payload, user_payload


def _owner() -> Session:
    """A log with an owner, whose events carry the owner's stamps.

    A user message, a chunk, and the assistant message that cites the chunk —
    the smallest log with a surface worth comparing.
    """
    owner = Session("s", header=SessionHeader(id="s", created_at=1_700_000_000_000))
    owner.append("user/message", user_payload("hello", "m1"), SurfaceIntent("append"))
    owner.append("assistant/chunk", {"turn": 1, "step": 1, "chunk": {"type": "usage"}})
    owner.append("assistant/message", assistant_payload("hi", "m2"), SurfaceIntent("append", (1,)))
    return owner


def test_an_admitted_log_is_the_owners_log() -> None:
    """Event by event, the mirror reaches the same surface as the owner — and,
    unlike `Session(seed=…)`, without a marker of its own appended on the end."""
    owner = _owner()
    mirror = Session("s", header=owner.header)

    for event in owner.events:
        mirror.admit(event)

    assert mirror.seq == owner.seq, "no marker grown on the end"
    assert [e.type for e in mirror.events] == [e.type for e in owner.events]
    assert mirror.surface.nodes == owner.surface.nodes
    assert mirror.derive_messages() == owner.derive_messages()
    assert mirror.stale() == [], "the incremental fold agrees with its replay"


def test_the_owners_stamps_survive() -> None:
    """`time` is the daemon's clock, and the trajectory's timings read it; a mirror
    that re-stamped would put this client's clock on the daemon's record."""
    owner = _owner()
    mirror = Session("s", header=owner.header)

    admitted = [mirror.admit(event) for event in owner.events]

    assert [e.time for e in admitted] == [e.time for e in owner.events]
    assert [e.seq for e in admitted] == [e.seq for e in owner.events]


def test_a_hole_is_refused_with_the_seed_paths_sentence() -> None:
    """A replica that skipped a frame must stop, not admit a log with a gap: the
    next seq no longer matches, and that is the seed rule, through the seed
    function, rather than a second copy of it."""
    owner = _owner()
    mirror = Session("s", header=owner.header)
    mirror.admit(owner.events[0])

    with pytest.raises(ValueError, match=r"seq 2 \(expected 1\)"):
        mirror.admit(owner.events[2])

    assert mirror.seq == 1, "the refusal left the log exactly as it was"
    assert mirror.stale() == []


def test_an_unknown_required_type_is_refused_and_an_ignorable_one_is_not() -> None:
    """The other seed rule: a type this build does not know may change how the
    rest of the log reads, so it is refused unless the writer marked it skippable."""
    mirror = Session("s")
    required = SessionEvent(type="future/thing", seq=0, time=1, data={})
    skippable = SessionEvent(type="future/thing", seq=0, time=1, data={}, ignorable=True)

    with pytest.raises(ValueError, match="unrecognized required type"):
        mirror.admit(required)
    mirror.admit(skippable)
    assert mirror.seq == 1


def test_observers_cannot_tell_which_door_an_event_used() -> None:
    """One publish tail for both. A store or a status projection watching the
    mirror sees the same call an owner's observer sees."""
    owner = _owner()
    mirror = Session("s", header=owner.header)
    seen: list[tuple[str, int]] = []
    mirror.observe(lambda session, event: seen.append((event.type, event.seq)))

    for event in owner.events:
        mirror.admit(event)

    assert seen == [(e.type, e.seq) for e in owner.events]
    assert len(mirror.events) == owner.seq, "the events snapshot was invalidated"


def test_admit_holds_the_surface_to_the_same_rules_as_append() -> None:
    """A chunk is not surface-eligible, so one arriving with a `surfaceOp` is refused
    at the door — whichever door — and leaves both the log and the surface as they
    were. The same `_surface_op_of` rule `append` meets, reached through `_commit`."""
    mirror = Session("s")
    mirror.admit(
        SessionEvent(
            type="user/message", seq=0, time=1, data=user_payload("x", "m1"), surface_op="append"
        )
    )
    bogus = SessionEvent(type="assistant/chunk", seq=1, time=2, data={}, surface_op="append")

    with pytest.raises(SurfaceError):
        mirror.admit(bogus)

    assert mirror.seq == 1
    assert mirror.surface.nodes == (0,)
    assert mirror.stale() == []
