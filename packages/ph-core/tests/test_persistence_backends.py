"""P5-08's gate: the same `SessionPersistence` tests pass on both backends.

One suite, parametrized over the two implementations, which is the only way the
Protocol means anything — a second backend tested by its own tests would agree
with itself and nothing else. Every test here asks a question through the
Protocol; none reaches for a path, a directory or a table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.persistence.jsonl import JsonlSessionStore
from ph.persistence.protocol import SessionPersistence
from ph.session import Session, SessionHeader, SurfaceIntent
from ph.testing import user_payload

pytestmark = pytest.mark.anyio

BACKENDS = ("jsonl", "turso")

APPEND = SurfaceIntent("append")
"""`user/message` is surface-eligible, so an append of one owes a marker."""


@pytest.fixture(params=BACKENDS)
def store(request: Any, tmp_path: Path) -> SessionPersistence:
    """A mounted-shape store of each kind, built directly.

    Directly rather than through `mount`, because what is under test is the
    *backend*, and going through a profile would test the row wiring twice and
    the storage once.
    """
    if request.param == "jsonl":
        return JsonlSessionStore(ctx=None, root=tmp_path)  # type: ignore[arg-type]
    from ph.persistence.turso import TursoSessionStore

    return TursoSessionStore(ctx=None, root=tmp_path)  # type: ignore[arg-type]


def _session(store: SessionPersistence, session_id: str = "s1", **header: Any) -> Session:
    """A tracked session. `track` is what the row does on `session/created`."""
    session = Session(session_id, header=SessionHeader(id=session_id, created_at=1, **header))
    store.track(session)
    return session


def _append(
    store: SessionPersistence,
    session: Session,
    kind: str,
    data: Any,
    intent: SurfaceIntent | None = None,
) -> None:
    """Append and record, which is the pair the row wires to the firehose.

    `record` is not something `Session.append` calls — the plugin subscribes it
    to `session/event` — so a test driving a store directly has to do both, and
    doing only the first is how these tests first read back an empty log from
    *both* backends and looked like two bugs instead of one mistake.
    """
    store.record(session, session.append(kind, data, intent))


async def test_a_tracked_session_round_trips_through_the_backend(
    store: SessionPersistence,
) -> None:
    """Write, flush, read back — the whole contract in one property.

    The events that come back must equal the events that went in, in order and
    with their payloads intact, or every fold over a resumed log is a different
    projection from the live one it replaced (A11).
    """
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    _append(
        store, session, "tool/call", {"callId": "c1", "name": "read", "arguments": {"path": "x"}}
    )
    await store.flush(session)

    assert store.exists("s1")
    header, events = store.read("s1")
    assert header.id == "s1"
    assert [(event.seq, event.type) for event in events] == [(0, "turn/start"), (1, "tool/call")]
    assert events[1].data["arguments"]["path"] == "x", "a nested payload did not survive"


async def test_what_comes_back_can_seed_a_session_again(store: SessionPersistence) -> None:
    """The assertion that was missing, and the defect it would have caught.

    Reading events back is not the contract — *re-seeding a `Session` from them*
    is, because that is what `resume_session` does and what every consumer sees.
    The Turso backend stored only the `data` column, so `surfaceOp`, `ignorable`
    and `sourceEventSeqs` were dropped and any session holding a real
    `user/message` came back unmarked: `Session(seed=…)` refuses it, which is
    every resumable session there is.

    The round-trip test above missed it by using `turn/start` alone — an event
    with no envelope fields to lose. A surface-eligible one is what makes the
    envelope matter.
    """
    session = _session(store, "reseedable")
    _append(store, session, "user/message", user_payload("hello", "m1"), APPEND)
    _append(store, session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)

    header, events = store.read("reseedable")
    revived = Session("reseedable", seed=events, header=header)
    assert [event.type for event in revived.events_from(0)][:2] == ["user/message", "turn/end"]
    assert revived.derive_messages(), "the transcript did not survive the round trip"


async def test_an_unknown_session_is_absent_rather_than_empty(
    store: SessionPersistence,
) -> None:
    """`exists` is False and `read` raises — never an empty log.

    A backend that answered "no events" for a session it has never heard of
    would let a resume silently start over, which is P5-01's own bug wearing a
    different hat.
    """
    assert store.exists("never-written") is False
    with pytest.raises((FileNotFoundError, OSError)):
        store.read("never-written")


async def test_appends_after_a_flush_are_added_not_rewritten(
    store: SessionPersistence,
) -> None:
    """A second flush carries only what is new, and the log stays contiguous.

    Contiguity is A1 — `seq == len(log)` — and it is what a resumed session's
    seed is validated against, so a backend that duplicated or dropped a record
    here produces a log the reader refuses.
    """
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    await store.flush(session)
    _append(store, session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)

    _, events = store.read("s1")
    assert [event.seq for event in events] == [0, 1]
    assert [event.type for event in events] == ["turn/start", "turn/end"]


async def test_a_resume_writes_what_it_synthesized_on_top_of_what_it_read(
    store: SessionPersistence,
) -> None:
    """A session resumes **repeatedly**, and each reopen leaves the log contiguous.

    The test the suite was missing, and the shape of the bug it missed. A resume
    seeds the stored events, then adds two things nobody wrote: the repair
    closers, and the `session/end-seed` the constructor appends. Both are in the
    log before the store is ever asked to track it — so a backend that inferred
    "the file exists, therefore its contents are what I have" dropped them and
    wrote only what came afterwards. That leaves a hole in the seq space, and
    `_readmit` refuses a hole, so the *second* resume raised.

    **Twice, because once passes either way.** The gap is created by the first
    resume and only detected by the second, which is precisely why every test
    that touched this stopped one step short — including the one below named for
    resuming, which reads a stored log back but never reopens it.

    Both backends, because they disagreed here and the Protocol is the only
    thing that makes them answer alike: `TursoSessionStore` upserts by `seq` and
    queues its whole log, so it was always correct; `JsonlSessionStore` appends
    and has to be told what it already holds.
    """
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    _append(store, session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)

    for reopen in (1, 2, 3):
        # A reopen is a new *process*, and `forget` is what process death does to
        # a backend's memory. Without it `track` early-returns on the buffer the
        # fixture's store still holds, and the test reproduces the bug in its own
        # harness rather than exercising the code.
        store.forget("s1")
        header, events = store.read("s1")
        assert [event.seq for event in events] == list(range(len(events))), (
            f"reopen {reopen}: the stored seq space has a hole, so nothing can seed from it"
        )
        # What `resume_session` does, without a mounted profile: seed from the
        # store, let the constructor mark the boundary, then record the reopen.
        revived = Session("s1", seed=list(events), header=header)
        revived.durable_length = len(events)
        store.track(revived)
        store.record(revived, revived.append("session/resumed", {"events": len(events)}))
        await store.flush(revived)

    _, events = store.read("s1")
    assert [event.seq for event in events] == list(range(len(events)))
    assert [event.type for event in events] == [
        "turn/start",
        "turn/end",
        # One boundary marker and one record per reopen. Two events, not one:
        # `session/resumed` always lands after the marker, so the constructor's
        # "a seed already ending in one is not re-marked" guard cannot fire.
        "session/end-seed",
        "session/resumed",
        "session/end-seed",
        "session/resumed",
        "session/end-seed",
        "session/resumed",
    ]


async def test_a_repaired_tail_is_written_not_only_repaired(
    store: SessionPersistence,
) -> None:
    """The closers reach the store, so the repair is durable.

    `interrupted_turn_closers` synthesizes them on the seed rather than after
    publication, which is what makes a resumed session provider-valid on the
    first read. But synthesizing is not storing: a backend that dropped them
    left an unclosed turn on disk forever, so every future resume re-repaired
    the same tail and any reader that does *not* repair — anything reading the
    stored log directly — saw a turn with an unresolved tool call.
    """
    from ph.persistence.repair import interrupted_turn_closers

    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    _append(store, session, "step/start", {"turn": 1, "step": 1})
    _append(store, session, "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "read"})
    await store.flush(session)
    crashed_at = session.events[-1].time

    store.forget("s1")  # a new process; see the reopen loop above
    header, events = store.read("s1")
    closers = interrupted_turn_closers(events)
    assert closers, "the fixture is not a crashed log"
    revived = Session("s1", seed=[*events, *closers], header=header)
    revived.durable_length = len(events)
    store.track(revived)
    await store.flush(revived)

    _, stored = store.read("s1")
    assert not interrupted_turn_closers(stored), "the repair did not survive the flush"
    # Backdated, which is what distinguishes writing the closers from
    # re-appending them: repair stamps the last real event's time, never `now()`.
    written = {event.type: event.time for event in stored}
    assert written["turn/end"] == crashed_at
    assert written["step/end"] == crashed_at


async def test_a_flush_with_nothing_pending_is_a_no_op(store: SessionPersistence) -> None:
    """Called on every checkpoint barrier, so it must be free when idle."""
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    await store.flush(session)
    before = store.read("s1")[1]

    await store.flush(session)
    await store.flush(session)
    assert store.read("s1")[1] == before


async def test_stored_lists_what_was_written_most_recent_first(
    store: SessionPersistence,
) -> None:
    """The listing a picker reads, and the repo scoping P5-14 needs.

    `cwd` comes from the header rather than from where the backend happens to
    keep bytes, which is what lets "sessions from this repo" be a question at
    all — a path-shaped answer would be unavailable to a backend with no files.
    """
    for index, name in enumerate(("older", "newer")):
        session = _session(store, name, cwd=f"/repo/{name}")
        _append(store, session, "turn/start", {"turn": index + 1})
        await store.flush(session)

    listed = store.stored()
    assert {row.session_id for row in listed} == {"older", "newer"}
    assert {row.cwd for row in listed} == {"/repo/older", "/repo/newer"}
    assert store.stored(limit=1) != [], "a limit of one returned nothing"
    assert len(store.stored(limit=1)) == 1


async def test_locate_is_honest_about_whether_there_is_a_file(
    store: SessionPersistence,
) -> None:
    """A real path, from both backends, which is why the lease works at all.

    P5-03's lease locks what `locate` returns. The first Turso draft kept one
    shared database, so it had no per-session path, answered `None`, and I-5 was
    silently not enforced — two daemons could open one session and nothing would
    refuse. One database per session restores the property *and* makes deleting
    a session a file deletion, the same as JSONL.
    """
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    await store.flush(session)

    where = store.locate("s1")
    assert where is not None and where.is_file(), (
        "a backend that keeps one file per session must be able to point at it"
    )


async def test_forget_drops_memory_without_dropping_the_log(
    store: SessionPersistence,
) -> None:
    """Disposal frees buffers; it does not delete anything written."""
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    await store.flush(session)

    store.forget("s1")
    assert store.exists("s1"), "forgetting a session deleted its log"
    assert store.read("s1")[1], "forgetting a session lost its events"


def test_both_backends_satisfy_the_protocol(store: SessionPersistence) -> None:
    """The `runtime_checkable` guard, which is weaker than it looks.

    `isinstance` against a Protocol checks that the *names* exist, not their
    signatures — so this catches a backend that forgot a method and nothing
    else. The suite above is what actually holds them to the same behaviour;
    this is here so a missing method fails as "does not satisfy the Protocol"
    rather than as an `AttributeError` three tests later.
    """
    assert isinstance(store, SessionPersistence)


@pytest.mark.parametrize("backend", BACKENDS)
async def test_a_profile_resumes_through_whichever_backend_it_mounted(
    mount: Any, backend: str
) -> None:
    """The row's point, end to end: a session survives a remount, either way.

    Through `mount` rather than by building a store, because what is under test
    here is the *seam* — that `resume_session` and everything above it ask the
    Protocol instead of rebuilding a filename. Before this row those callers
    read `store.root` and `session_path`, which is a JSONL fact four consumers
    deep and the reason a second backend could not be added without breaking
    all four.
    """
    # The row's *id* is `session-persistence` and its *name* is the backend, so
    # a profile swaps one for the other by disabling the row and inserting its
    # replacement. jsonl is base's own, so that case mounts unchanged.
    overlays: tuple[dict[str, Any], ...] = (
        ()
        if backend == "jsonl"
        else (
            {"id": "session-persistence", "disabled": True},
            {"insert": [{"id": "session-persistence-turso", "name": "session-persistence-turso"}]},
        )
    )

    ctx = await mount(*overlays)
    session = ctx.sessions.create("carried")
    session.append("turn/start", {"turn": 1})
    await ctx.sessions.flush(session)

    store = ctx.session_persistence
    assert store.exists("carried"), f"{backend} did not store the session"
    header, events = store.read("carried")
    assert header.id == "carried"
    assert [event.type for event in events] == ["turn/start"]
