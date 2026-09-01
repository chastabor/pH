"""P5-08's gate: the same `SessionPersistence` tests pass on both backends.

One suite, parametrized over the two implementations, which is the only way the
Protocol means anything — a second backend tested by its own tests would agree
with itself and nothing else. Every test here asks a question through the
Protocol; none reaches for a path, a directory or a table.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest

from ph.persistence import MAX_DEPTH, LineageError, lineage_faults, materialise
from ph.persistence.jsonl import JsonlSessionStore
from ph.persistence.protocol import SessionPersistence, StoredSession
from ph.session import Session, SessionEvent, SessionHeader, SurfaceIntent
from ph.testing import reference_fork, user_payload

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


# ------------------------------------------------- lineage (feasibility §5.3) --


def _reference_fork(
    store: SessionPersistence, child: str, parent: str, boundary: int, family: str | None = None
) -> Session:
    """`reference_fork` put through this backend.

    `family` because a lineage shares one directory, which `SessionStore.create`
    settles for anything it builds — a header assembled here has to say it, and a
    synthetic chain with no real root has to pick one.
    """
    header, own = reference_fork(child, parent, boundary=boundary, family=family)
    session = Session(child, header=header)
    store.track(session)
    for event in own:
        store.record(session, event)
    return session


async def test_a_log_that_starts_at_zero_is_read_unchanged(store: SessionPersistence) -> None:
    """The no-op that makes this safe to land before anything writes a reference.

    Every log written so far begins at seq 0 — a root because it is one, and
    today's copy-forked child because the copy is right there. So chain-aware
    reading must be invisible until a file says otherwise, and the thing that
    says otherwise is a first event above 0.
    """
    session = _session(store)
    _append(store, session, "turn/start", {"turn": 1})
    _append(store, session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(session)

    header, events = store.read("s1")
    assert header.id == "s1"
    assert [event.seq for event in events] == [0, 1]


async def test_a_child_holding_only_its_own_events_materialises_its_lineage(
    store: SessionPersistence,
) -> None:
    """The mechanism: a file that begins above 0 is owed exactly that many events.

    The child stores two events and reads back six. Its header is its own — a
    lineage supplies history, never identity — and the assembled log is dense
    from 0, which is what every reader above this already requires.
    """
    parent = _session(store, "p")
    for turn in range(3):
        _append(store, parent, "turn/start", {"turn": turn})
        _append(store, parent, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    await store.flush(parent)

    child = _reference_fork(store, "c", "p", boundary=4)
    await store.flush(child)

    header, events = store.read("c")
    assert header.id == "c", "the lineage supplies events, not identity"
    assert [event.seq for event in events] == [0, 1, 2, 3, 4, 5]
    assert [event.type for event in events[:4]] == [event.type for event in parent.events[:4]]


async def test_a_growing_parent_never_disturbs_a_child(store: SessionPersistence) -> None:
    """**The safety property, and the reason this needs no lock.**

    A child depends on `parent[0:n]`, and a prefix of an append-only log is
    immutable. The parent may run forever without invalidating a descendant —
    which is what makes one writer per file structural rather than enforced, and
    is precisely what tau's shared-file design cannot offer.
    """
    parent = _session(store, "p")
    for turn in range(3):
        _append(store, parent, "turn/start", {"turn": turn})
        _append(store, parent, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    await store.flush(parent)
    child = _reference_fork(store, "c", "p", boundary=4)
    await store.flush(child)
    before = [event.seq for event in store.read("c")[1]]

    for turn in (9, 10):
        _append(store, parent, "turn/start", {"turn": turn})
        _append(store, parent, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    await store.flush(parent)

    assert [event.seq for event in store.read("c")[1]] == before
    assert len(store.read("p")[1]) == 10, "the parent did grow"


async def test_a_child_that_claims_completeness_but_is_short_is_refused(
    store: SessionPersistence,
) -> None:
    """**The trap step 4 walks into**, caught before it can return a wrong log.

    `Session.append` mints `seq = len(self._log)`, so a reference-forked child
    built with an empty seed writes its first event at **0** — and a file
    starting at 0 is exactly what this reader takes as "complete". It would hand
    back a two-event log as a whole session and drop the inherited prefix in
    silence, which is worse than any refusal: the caller cannot tell.

    The header is what disagrees. It says the child inherits four events; the
    file holds two and starts at zero. Those cannot both be true, so `fork` must
    seed the child's counter at `seed_length` — and until it does, this says so.
    """
    header = SessionHeader(id="short", created_at=1, parent_session="p", seed_length=4)
    session = Session("short", header=header)
    store.track(session)
    for seq in (0, 1):
        store.record(session, SessionEvent(type="turn/start", seq=seq, time=1, data={"turn": seq}))
    await store.flush(session)

    with pytest.raises(LineageError, match="claims to hold its own history") as caught:
        store.read("short")

    assert caught.value.code == "TRUNCATED"
    assert caught.value.session_id == "short"


async def test_a_missing_ancestor_fails_loudly_and_names_it(store: SessionPersistence) -> None:
    """A broken chain must refuse, not return a partial log.

    A silent partial read reconstructs a **wrong** session rather than an
    incomplete one — the failure `_readmit`'s unknown-type refusal exists to
    prevent, one layer down. The refusal names the ancestor because that is the
    only thing the reader can act on.
    """
    child = _reference_fork(store, "orphan", "vanished", boundary=4)
    await store.flush(child)

    with pytest.raises(LineageError, match="vanished") as caught:
        store.read("orphan")

    assert caught.value.code == "MISSING_ANCESTOR"


async def test_a_log_above_zero_that_names_no_parent_is_refused(
    store: SessionPersistence,
) -> None:
    """The other corruption: a file that owes a prefix and says nothing about it.

    Distinct from a missing ancestor, and worth its own sentence — one is a
    deleted file, the other a header that never recorded where its history came
    from, and only the second is unrecoverable.
    """
    header = SessionHeader(id="rootless", created_at=1)
    session = Session("rootless", header=header)
    store.track(session)
    store.record(session, SessionEvent(type="turn/start", seq=4, time=1, data={"turn": 4}))
    await store.flush(session)

    with pytest.raises(LineageError, match="names no parent") as caught:
        store.read("rootless")

    assert caught.value.code == "NO_PARENT"


async def test_a_cycle_in_the_lineage_is_refused_and_shown(store: SessionPersistence) -> None:
    """Two headers naming each other must meet a bound, not the stack.

    The chain is data read off disk, so a hand-edited or corrupted header can
    name its own descendant. `RecursionError` would be a true statement about the
    interpreter and a useless one about the log, so the walk reports the cycle it
    found instead.
    """
    for name, parent in (("a", "b"), ("b", "a")):
        session = _reference_fork(store, name, parent, boundary=4, family="a")
        await store.flush(session)

    with pytest.raises(LineageError, match="cycle") as caught:
        store.read("a")

    assert caught.value.code == "CYCLE"


async def test_a_chain_deeper_than_the_bound_is_refused(store: SessionPersistence) -> None:
    """The syscall bound, which is not a correctness one.

    A legal chain of any depth reconstructs correctly and reads the same bytes,
    since each file contributes a disjoint slice — so this exists to stop a
    pathological fork-of-fork from turning one read into hundreds of file opens,
    and to give the cycle guard a backstop when the cycle is longer than the
    walk's memory of it.
    """
    # Each link owes the one above it and supplies nothing, so the walk cannot
    # terminate before the bound does.
    for depth in range(MAX_DEPTH + 2):
        session = _reference_fork(store, f"d{depth}", f"d{depth + 1}", boundary=4, family="d0")
        await store.flush(session)

    with pytest.raises(LineageError, match=f"more than {MAX_DEPTH} deep") as caught:
        store.read("d0")

    assert caught.value.code == "TOO_DEEP"


async def test_a_short_ancestor_is_caught_as_a_gap_not_passed_on(
    store: SessionPersistence,
) -> None:
    """An ancestor that cannot supply what it was asked for.

    The parent starts at 0, so the walk stops there — but it holds fewer events
    than the child was owed, leaving a hole in the middle of the assembled log.
    `_readmit` would refuse that hole one layer up, with no idea which ancestor
    came up short; catching it here is what turns "seed must be contiguous from
    0" into a sentence naming the lineage.
    """
    parent = _session(store, "short")
    _append(store, parent, "turn/start", {"turn": 1})
    _append(store, parent, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    await store.flush(parent)  # seqs 0 and 1 only

    child = _reference_fork(store, "wants4", "short", boundary=4)
    await store.flush(child)

    with pytest.raises(LineageError, match="gap at index 2") as caught:
        store.read("wants4")

    assert caught.value.code == "GAP"


# ---------------------------------------- referential integrity (§5.4a) --


async def test_a_broken_chain_is_visible_from_the_listing_alone(
    store: SessionPersistence,
) -> None:
    """**The survey `materialise` cannot do.**

    `materialise` refuses a missing ancestor at the point of use: one session, at
    resume, long after whatever removed the file. This asks the same question of
    the whole store and costs no reads — both backends fill `StoredSession.parent`
    from the one-line header peek they already pay for, so the answer is a
    listing plus one `exists` per session that names a parent.
    """
    child = _reference_fork(store, "orphan", "vanished", boundary=4)
    await store.flush(child)

    listed = store.stored()
    assert {one.session_id: one.parent for one in listed} == {"orphan": "vanished"}
    assert lineage_faults(((one.session_id, one.parent) for one in listed), store.exists) == [
        ("orphan", "ancestor vanished is missing")
    ]


async def test_a_healthy_store_reports_no_faults(store: SessionPersistence) -> None:
    """Silent when there is nothing wrong, which is what keeps the section useful."""
    parent = _session(store, "p")
    for turn in range(3):
        _append(store, parent, "turn/start", {"turn": turn})
        _append(store, parent, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    await store.flush(parent)
    await store.flush(_reference_fork(store, "c", "p", boundary=4))

    listed = store.stored()
    assert len(listed) == 2
    assert lineage_faults(((one.session_id, one.parent) for one in listed), store.exists) == []


async def test_a_seeded_child_writes_only_what_this_store_lacks(
    store: SessionPersistence,
) -> None:
    """**The gate this suite was missing, and one backend was failing it.**

    `_reference_fork` above builds its child with *no* seed, so `track` queues
    the same thing whichever rule it follows — which is exactly why both backends
    passed while Turso was queueing `session.events` whole. Every fork under it
    got a full copy of its prefix written into the child, `materialise` then read
    the first seq as 0 and called the file complete, and reference-forking was
    silently not happening on that backend at all.

    A child seeded the way `SessionStore.fork` seeds one is the only shape that
    tells the two rules apart, which makes this the test the feature actually
    needed rather than the one it was easy to write.
    """
    parent = _session(store, "p")
    for turn in range(3):
        _append(store, parent, "turn/start", {"turn": turn})
        _append(store, parent, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    await store.flush(parent)

    inherited = len(parent.events)
    child = Session(
        "c",
        seed=list(parent.events),
        header=SessionHeader(
            id="c", created_at=1, parent_session="p", seed_length=inherited, family="p"
        ),
        durable=inherited,
    )
    store.track(child)
    await store.flush(child)

    own = store.read_own("c")[1]
    assert [event.type for event in own] == ["session/end-seed"], "the prefix was re-written"
    assert own[0].seq == inherited, "and the first seq is what marks the file as owing one"
    assert [event.seq for event in store.read("c")[1]] == list(range(inherited + 1))


async def test_a_listing_row_says_the_same_thing_from_either_backend(
    store: SessionPersistence,
) -> None:
    """**One projection, so a new listing field cannot reach one backend only.**

    `kind` was added to `StoredSession`, filled by JSONL, missed by Turso, and
    read by nobody — so a Turso-backed picker would have gone back to drawing a
    rolled session as a staircase with every test green. The field set is pinned
    here because that is the part that drifted: both backends now build rows
    through `stored_row`, and a field added outside it fails this before it can
    reach one of them.
    """
    assert {field.name for field in fields(StoredSession)} == {
        "session_id",
        "modified",
        "cwd",
        "parent",
    }, "a new listing field belongs in `stored_row`, where both backends get it"

    parent = _session(store, "p", cwd="/work")
    _append(store, parent, "turn/start", {"turn": 0})
    _append(store, parent, "turn/end", {"turn": 0, "reason": {"kind": "completed"}})
    await store.flush(parent)
    await store.flush(_reference_fork(store, "c", "p", boundary=2))

    rows = {row.session_id: row for row in store.stored()}
    assert rows["p"].parent is None and rows["p"].cwd == "/work"
    assert rows["c"].parent == "p"


async def test_a_bounded_read_stops_at_the_boundary(store: SessionPersistence) -> None:
    """**What keeps a chained read from parsing an ancestor whole.**

    A child forked early in a long parent inherits a sliver of it, and without a
    bound the whole ancestor was still parsed, validated and frozen to produce
    it. Measured on a 10 000-event ancestor contributing 50 events: 53.7 ms
    unbounded against 0.3 ms bounded (JSONL, 179x), 61.3 against 0.4 (Turso,
    173x). At the *tip* — the common fork — it changes nothing, because the child
    wants nearly all of the ancestor anyway.
    """
    parent = _session(store, "p")
    for turn in range(3):
        _append(store, parent, "turn/start", {"turn": turn})
        _append(store, parent, "turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    await store.flush(parent)

    whole = store.read_own("p")[1]
    bounded = store.read_own("p", 4)[1]

    assert [event.seq for event in whole] == [0, 1, 2, 3, 4, 5]
    assert [event.seq for event in bounded] == [0, 1, 2, 3], (
        "everything below the bound, and no more"
    )
    assert store.read_own("p", 0)[1] == [], "a bound of zero asks for nothing"


def test_the_walk_asks_each_ancestor_for_exactly_what_it_still_owes() -> None:
    """The bound is the count the next generation down is short, nothing else."""
    asked: list[tuple[str, int | None]] = []

    def read_one(
        session_id: str, upto: int | None, family: str | None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        asked.append((session_id, upto))
        return _segment(session_id)

    materialise(read_one, "c")
    assert asked == [("c", None), ("b", 6), ("a", 3)], (
        "the target whole, then each ancestor bounded by what remained"
    )


def test_a_backend_that_ignores_the_bound_still_reads_correctly() -> None:
    """**`upto` is a hint, and the walk must not depend on it being honoured.**

    Making the bound load-bearing would mean a backend that quietly ignored it
    produced a *wrong* log rather than a slow one — which is precisely how
    reference-forking came to be a silent no-op on Turso once already. So the
    walk re-checks what it takes, and a backend that over-returns costs time.
    """

    def ignores_the_bound(
        session_id: str, upto: int | None, family: str | None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        return _segment(session_id)  # every event it has, bound or no bound

    header, events = materialise(ignores_the_bound, "c")
    assert header.id == "c"
    assert [event.seq for event in events] == list(range(9))


def _segment(session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
    """A three-file lineage a -> b -> c, each holding three of nine events."""
    order = {"a": 0, "b": 3, "c": 6}
    start = order[session_id]
    parent = {"b": "a", "c": "b"}.get(session_id)
    header = SessionHeader(
        id=session_id,
        created_at=1,
        parent_session=parent,
        seed_length=start or None,
        family="a",
    )
    return header, [
        SessionEvent(type="turn/start", seq=seq, time=1, data={"turn": seq})
        for seq in range(start, start + 3)
    ]


def test_the_survey_and_the_reader_agree_on_how_deep_is_too_deep() -> None:
    """One bound, stated twice, pinned to itself.

    `materialise` refuses past `MAX_DEPTH`; the survey has to call those chains
    unreadable or it goes quiet on exactly the shape `roll` manufactures — one
    generation per segment, so a long-running segmented session reaches the bound
    by ordinary use, and `ph doctor` would report a healthy store while every
    read of the newest segment raised.
    """

    def chain(length: int) -> list[tuple[str, str | None]]:
        return [(f"s{n}", f"s{n + 1}" if n + 1 < length else None) for n in range(length)]

    assert lineage_faults(chain(MAX_DEPTH), lambda session_id: True) == []
    faults = dict(lineage_faults(chain(MAX_DEPTH + 1), lambda session_id: True))
    assert f"more than {MAX_DEPTH} deep" in faults["s0"]


def test_a_parent_below_the_listing_cut_is_not_reported_as_missing() -> None:
    """**The check that would otherwise cry wolf on every large store.**

    `stored()` takes a limit, so a parent older than the cut is absent from the
    listing while being perfectly present on disk. Membership in the listing is
    therefore not the test; `exists` is.
    """
    assert lineage_faults([("kid", "root")], lambda session_id: True) == []
    assert lineage_faults([("kid", "root")], lambda session_id: False) == [
        ("kid", "ancestor root is missing")
    ]


def test_a_cycle_that_closes_inside_the_listing_is_named() -> None:
    """A hand-edited header can point at its own descendant. Both ends are
    unreadable, so both are reported rather than one arbitrarily chosen."""
    assert lineage_faults([("a", "b"), ("b", "a")], lambda session_id: True) == [
        ("a", "lineage cycles back through a"),
        ("b", "lineage cycles back through b"),
    ]
