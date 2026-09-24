"""P0-08 — `Session.append`, the acceptance boundary.

Gates: *`seq == len(log)` property; losslessness rejections; a raising listener
does not un-append.*

The last one is the subtle one. Once an event is in the log the append is
**committed** — an observer that throws is a bug in the observer, not a reason
to pretend the event never happened. dsh contains observer failures per
listener for exactly this reason, and persistence depends on it.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from ph.json import JsonValue, as_obj, as_seq, thaw_json
from ph.session import (
    BatchRef,
    LogTypeError,
    Session,
    SessionEvent,
    SessionFoldCache,
    SurfaceError,
    SurfaceIntent,
    SurfaceReplace,
    UnknownEventTypeError,
    declare_log_type,
)
from ph.session.json import InvalidJsonValueError
from ph.testing import check_fold_laws, log_event, prefix_of, user_payload


def test_seq_always_equals_log_length() -> None:
    session = Session("s")
    for index in range(25):
        event = log_event(session, "turn/start", {"turn": index})
        assert event.seq == index
        assert session.seq == index + 1
        assert len(session.events) == index + 1
    assert [event.seq for event in session.events] == list(range(25))


def test_appended_data_is_detached_from_the_caller() -> None:
    session = Session("s")
    # Built through named locals rather than one literal, because the mutations
    # below are the point and they have to be typed: a bare
    # `{"nested": {"list": [1, 2]}}` infers `dict[str, object]`, which `append`
    # refuses — and reaching the inner list back through `payload` yields a
    # `JsonValue`, which has no `.append`. Holding it directly says what the
    # test is doing: mutate the caller's own buffer, after the append.
    inner: list[JsonValue] = [1, 2]
    payload: dict[str, JsonValue] = {"turn": 1, "nested": {"list": inner}}
    event = log_event(session, "turn/start", payload)
    payload["turn"] = 99
    inner.append(3)
    # The log holds the value at append time, not a live view of the caller's
    # buffer — a stateful producer cannot rewrite history after the fact.
    assert event.data["turn"] == 1
    assert list(as_seq(as_obj(event.data["nested"])["list"])) == [1, 2]


def test_logged_data_is_not_writable() -> None:
    session = Session("s")
    event = log_event(session, "turn/start", {"turn": 1})
    with pytest.raises(TypeError):
        event.data["turn"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": math.inf},
        {"value": -0.0},
        {"value": (1, 2)},
        {"value": {1: "int key"}},
        {"value": 2**53},
        {"value": {"set"}},
        {"value": object()},
    ],
)
def test_non_lossless_payloads_are_refused(payload: dict[str, Any]) -> None:
    session = Session("s")
    # `Any`, and deliberately: every row here is a value `append` must
    # **refuse**, so annotating it as what `append` accepts would be a
    # claim the test exists to disprove.
    with pytest.raises(InvalidJsonValueError):
        log_event(session, "turn/start", payload)
    assert session.seq == 0


def test_a_non_object_payload_is_refused_at_the_write_door() -> None:
    """The read door refuses one (`_EventWire.data` is an object), and the two
    must agree. They did not at first: `SessionEvent.data` was declared a
    `JsonObject` and nothing enforced it here, so a producer reaching this with
    `Any` — a test tree outside mypy, a plugin built against an older signature —
    appended a list, the bytes reached disk, and `from_wire` then refused the
    record on resume. A log that cannot be reconstructed is the one failure A1
    exists to prevent, so the refusal belongs on both doors."""
    session = Session("s")
    for payload in ([1, 2], "text", 7, None):
        with pytest.raises(InvalidJsonValueError, match="must be a JSON object"):
            log_event(session, "turn/start", cast(Any, payload))
    assert session.events == ()


def test_cyclic_payloads_are_refused() -> None:
    session = Session("s")
    # `Any` for the same reason as the refusal table above: a dict that holds
    # itself is not a `JsonValue`, which is what this asserts.
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    with pytest.raises(InvalidJsonValueError, match="circular"):
        log_event(session, "turn/start", cycle)


def test_rejections_name_the_offending_path() -> None:
    with pytest.raises(InvalidJsonValueError, match=r"^a\.b\[1\]\.c: "):
        log_event(Session("s"), "turn/start", {"a": {"b": [0, {"c": math.nan}]}})


def test_a_raising_observer_cannot_un_append() -> None:
    session = Session("s")
    seen: list[str] = []

    def bad_observer(_session: Session, _event: object) -> None:
        raise RuntimeError("bad observer")

    session.observe(bad_observer)
    session.observe(lambda _s, event: seen.append(event.type))
    event = log_event(session, "turn/start", {"turn": 1})
    assert event.seq == 0
    assert len(session.events) == 1
    # The failing observer neither removed the event nor stopped the next one
    # from seeing it.
    assert seen == ["turn/start"]


def test_reentrant_append_is_refused() -> None:
    session = Session("s")
    caught: list[BaseException] = []

    def reenter(source: Session, _event: object) -> None:
        try:
            log_event(source, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
        except RuntimeError as error:
            caught.append(error)

    session.observe(reenter)
    log_event(session, "turn/start", {"turn": 1})
    # A reentrant append would assign a seq inside another event's publication,
    # so observers would watch the log grow underneath them.
    assert len(caught) == 1
    assert "cannot reenter" in str(caught[0])
    assert len(session.events) == 1


def test_surface_metadata_is_required_and_forbidden_by_type() -> None:
    session = Session("s")
    with pytest.raises(ValueError, match="requires a surfaceOp"):
        log_event(session, "user/message", user_payload("hi"))
    with pytest.raises(ValueError, match="not surface-eligible"):
        log_event(session, "turn/start", {"turn": 1}, SurfaceIntent("append"))


def test_events_snapshot_does_not_grow_under_a_holder() -> None:
    session = Session("s")
    log_event(session, "turn/start", {"turn": 1})
    held = session.events
    log_event(session, "step/start", {"turn": 1, "step": 1})
    assert len(held) == 1
    assert len(session.events) == 2


# ---------------------------------------------------------- the vocabulary --


@pytest.fixture
def vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A registry of declared types this test can add to without leaking.

    The table is module-level because a declaration is an import-time fact
    about a package, which is exactly what makes a test's declarations outlive
    the test unless it is swapped out here.
    """
    from ph.session import known_event_types

    monkeypatch.setattr(known_event_types, "_DECLARED", {})


def test_the_write_door_refuses_a_type_the_read_door_would() -> None:
    """F11. A type this build cannot read back is refused where it is written.

    `_readmit` refuses an unknown *required* type on every seed — resume, fork,
    replay — so a write door that accepted one wrote a log that resumed nowhere,
    with the append itself succeeding. Reproduced: `append("myplugin/thing")`
    returned an event with `ignorable=False`, and re-seeding the same log raised.

    Sabotage: drop the `is_known` check from `append` and the log grows.
    """
    session = Session("s")
    with pytest.raises(UnknownEventTypeError, match="declare_log_type"):
        log_event(session, "sample/thing", {"n": 1})
    assert session.events == (), "a refused append must leave the log as it was"


def test_a_declared_type_is_written_stamped_and_read_back(vocabulary: None) -> None:
    """The door a package outside ph-core writes its own types through.

    Ignorability comes from the declaration, the way it comes from
    `IGNORABLE_SESSION_EVENT_TYPES` for ph-core's, so a build without the package
    skips the record rather than refusing the whole log.
    """
    declare_log_type("sample/note", owner="sample.plugin", ignorable=True)
    declare_log_type("sample/state", owner="sample.plugin", ignorable=False)

    session = Session("s")
    note = log_event(session, "sample/note", {"n": 1})
    state = log_event(session, "sample/state", {"n": 2})
    assert (note.ignorable, state.ignorable) == (True, False)

    reopened = Session("s", seed=list(session.events))
    assert [event.type for event in reopened.events][:2] == ["sample/note", "sample/state"]


def test_a_required_declared_type_opens_only_where_it_is_declared(vocabulary: None) -> None:
    """What `ignorable=False` promises, stated as a test: a build without the
    declaring package refuses the log, because skipping a required record can
    change how the rest of it reads."""
    from ph.session import known_event_types

    declare_log_type("sample/state", owner="sample.plugin", ignorable=False)
    session = Session("s")
    log_event(session, "sample/state", {"n": 1})

    del known_event_types._DECLARED["sample/state"]  # a build without the package
    with pytest.raises(ValueError, match="unrecognized required type"):
        Session("s", seed=list(session.events))


def test_a_declaration_cannot_disagree_with_itself_or_ph_core(vocabulary: None) -> None:
    """The refusals `EventRegistry.declare` makes for a bus event, for a log type.

    Two statements of one type that disagree are a log two builds read two ways.
    """
    first = declare_log_type("sample/note", owner="sample.plugin", ignorable=True)
    assert declare_log_type("sample/note", owner="sample.plugin", ignorable=True) is first

    with pytest.raises(LogTypeError, match="ph-core type"):
        declare_log_type("tool/call", owner="sample.plugin", ignorable=True)
    with pytest.raises(LogTypeError, match="already declared by"):
        declare_log_type("sample/note", owner="another.plugin", ignorable=True)
    with pytest.raises(LogTypeError, match="ignorability cannot change"):
        declare_log_type("sample/note", owner="sample.plugin", ignorable=False)
    with pytest.raises(LogTypeError, match="namespace/type"):
        declare_log_type("SampleNote", owner="sample.plugin", ignorable=True)
    with pytest.raises(LogTypeError, match="needs an owner"):
        declare_log_type("sample/other", owner="", ignorable=True)


# ------------------------------------------------------------- fold caches --


def test_a_fold_cache_recomputes_only_when_the_log_grew() -> None:
    """`seq` is an exact invalidation key because the log is append-only (A1)."""
    session = Session("s")
    calls: list[int] = []

    def count_turns(log: Session) -> int:
        calls.append(log.seq)
        return sum(1 for event in log.events if event.type == "turn/start")

    cache: SessionFoldCache[int] = SessionFoldCache(count_turns)
    log_event(session, "turn/start", {"turn": 1})

    assert cache.read(session) == 1
    assert cache.read(session) == 1
    assert calls == [1], "the fold ran twice for one log"

    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert cache.read(session) == 1
    assert calls == [1, 2], "an appended event did not invalidate the fold"


def test_a_fold_cache_holds_one_value_per_session() -> None:
    """Bounded by live sessions, not by history: entries are replaced."""
    cache: SessionFoldCache[int] = SessionFoldCache(lambda log: log.seq)
    first, second = Session("a"), Session("b")
    for index in range(50):
        log_event(first, "turn/start", {"turn": index})
        assert cache.read(first) == index + 1
    assert cache.read(second) == 0
    assert len(cache._entries) == 2

    cache.forget("a")
    assert len(cache._entries) == 1


def test_a_fold_cache_leaves_the_fold_callable_on_a_slice() -> None:
    """The property that ruled out attaching folds to `Session`.

    A fork reconstructs state *as of its boundary* (D17), and the trajectory view
    projects a stored log with nothing mounted — so the fold has to answer for a
    prefix that is not the live log. A cache is a separate thing a consumer owns;
    the fold itself stays a pure function.
    """
    session = Session("s")
    log_event(session, "turn/start", {"turn": 1})
    boundary = session.seq
    log_event(session, "turn/start", {"turn": 2})

    def count_turns(log: Session) -> int:
        return sum(1 for event in log.events if event.type == "turn/start")

    assert count_turns(session) == 2
    assert count_turns(prefix_of(session, boundary)) == 1


# --------------------------------------------------------------- projections --


def _turn_number(event: SessionEvent) -> int | None:
    """A `turn/start`'s number, and `None` for the one that carries none."""
    turn = event.data.get("turn")
    return turn if isinstance(turn, int) else None


def _turn_label(event: SessionEvent) -> str | None:
    """The same event read as something else, for the key's sake."""
    number = _turn_number(event)
    return None if number is None else f"turn {number}"


def test_a_projection_answers_with_the_newest_event_it_can_parse() -> None:
    """A parser that says `None` means "not this one", not "nothing".

    The rule the whole family rests on: `fold_latest`, `_LatestFold` and every
    seam fold built on `projection` keep the last event the parser *accepted*, so
    a frame that carries nothing of the question cannot erase the answer.
    """
    session = Session("s")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "turn/start", {"turn": 2})

    assert session.projection("turn/start", _turn_number) == 2

    log_event(session, "turn/start", {"turn": "not a number"})
    assert session.projection("turn/start", _turn_number) == 2


def test_a_projection_parses_only_what_arrived_since_it_last_answered() -> None:
    """The whole reason this is not `for event in reversed(session.events)`.

    The walk it replaced materialized a snapshot of the log to read one field and
    grew more expensive the longer a conversation ran — asked, in the meter's
    case, on every pressure check.
    """
    session = Session("s")
    seen: list[int] = []

    def counted(event: SessionEvent) -> int | None:
        seen.append(event.seq)
        return _turn_number(event)

    for turn in range(1, 4):
        log_event(session, "turn/start", {"turn": turn})
    assert session.projection("turn/start", counted) == 3
    assert seen == [2], "the first read walked past the event that answered"

    log_event(session, "assistant/chunk", {"text": "hi"})
    assert session.projection("turn/start", counted) == 3
    assert seen == [2], "an event of another type was handed to the parser"

    log_event(session, "turn/start", {"turn": 4})
    assert session.projection("turn/start", counted) == 4
    assert seen == [2, 4], "the second read re-parsed events it had already seen"


def test_two_parsers_of_one_event_type_do_not_share_a_fold() -> None:
    """The parser is half the key, which is what makes the erased type sound.

    Keyed on the event type alone — or on a name a caller typed — the second of
    these would be handed the first one's *value*: an `int` where a `str` was
    annotated, with no error until something used it.
    """
    session = Session("s")
    log_event(session, "turn/start", {"turn": 1})

    assert session.projection("turn/start", _turn_number) == 1
    assert session.projection("turn/start", _turn_label) == "turn 1"
    assert len(session._latest) == 2


def test_one_parser_over_two_event_types_does_not_share_a_fold() -> None:
    """The other half. `assistant/chunk` carries no turn, so a shared fold would
    answer this with the `turn/start` one."""
    session = Session("s")
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "assistant/chunk", {"text": "hi"})

    assert session.projection("turn/start", _turn_number) == 1
    assert session.projection("assistant/chunk", _turn_number) is None


def test_a_stable_parser_reuses_its_fold_rather_than_rebuilding_one() -> None:
    """The hazard the key trades for: a parser that is a new object per call is
    a new fold per call, and each one reparses the log from the start.

    `latest` is the case that would have paid it — it passed an inline `lambda`
    before the key included the parser — so `_the_event` is module-level and
    pinned here. A frozen parser *object* is the same rule read the other way:
    `_CheckpointOf("a")` hashes by value, so `workspace.latest_checkpoint` gets
    one fold per agent and not one per call.
    """
    session = Session("s")
    for turn in range(1, 4):
        log_event(session, "turn/start", {"turn": turn})

    for _ in range(5):
        assert session.latest("turn/start") is not None
        assert session.projection("turn/start", _turn_number) == 3
    assert len(session._latest) == 2, "a fold was rebuilt instead of reused"


# ------------------------------------------------------------------ batches --


def _conversation() -> Session:
    session = Session("s")
    log_event(session, "user/message", user_payload("one", "m1"), SurfaceIntent())
    log_event(session, "user/message", user_payload("two", "m2"), SurfaceIntent())
    return session


def _summary(replaces: tuple[int, ...]) -> SurfaceIntent:
    return SurfaceIntent(SurfaceReplace(replaces=replaces), replaces)


def test_a_refused_event_leaves_none_of_its_batch() -> None:
    """P10-14. The accounting record and the replacement it describes land together.

    Appended one at a time, a replacement the surface refuses left the record
    before it in the log, describing a rewrite that never happened. In a batch
    the refusal is found on exit, before anything is pushed — a refused plan,
    an unknown type, and an exception out of the block all leave the log, the
    surface and every observer exactly as they were.
    """
    session = _conversation()
    seen: list[int] = []
    session.observe(lambda _session, event: seen.append(event.seq))
    before = (session.seq, session.surface.nodes)

    refused = pytest.raises(SurfaceError, match="seq 7 is not a current surface node")
    with refused, session.batch() as batch:
        log_event(batch, "compaction/summarized", {"shadowedSeqs": [7]})
        log_event(batch, "user/message", user_payload("summary", "m3"), _summary((7,)))
    with pytest.raises(UnknownEventTypeError), session.batch() as batch:
        log_event(batch, "compaction/summarized", {"shadowedSeqs": [0]})
        log_event(batch, "quantum/entangle", {})
    with pytest.raises(LookupError), session.batch() as batch:
        log_event(batch, "compaction/summarized", {"shadowedSeqs": [0]})
        raise LookupError("the block failed before it finished")

    assert (session.seq, session.surface.nodes) == before
    assert seen == []
    assert session.stale() == []


def test_observers_see_a_batch_in_order_after_it_validates() -> None:
    """Nothing is published while the block runs; then every member, in order,
    each seen with the log ending at it — the view a single append gives. An
    observer cannot append between two members: the one reentrancy guard spans
    the whole publication."""
    session = _conversation()
    seen: list[tuple[int, int]] = []
    refused: list[str] = []

    def watch(watched: Session, event: SessionEvent) -> None:
        seen.append((event.seq, len(watched.events)))
        try:
            log_event(watched, "turn/start", {"turn": 9})
        except RuntimeError as error:
            refused.append(str(error))

    session.observe(watch)
    with session.batch() as batch:
        accounting = log_event(batch, "compaction/summarized", {"shadowedSeqs": [0, 1]})
        log_event(batch, "user/message", user_payload("summary", "m3"), _summary((0, 1)))
        assert seen == [], "published before the block exited"
        assert session.seq == 2, "pushed before the block exited"

    assert accounting.seq == 2
    assert seen == [(2, 3), (3, 4)]
    assert len(refused) == 2 and "cannot reenter" in refused[0]
    assert session.surface.nodes == (3,)


def test_a_batch_obeys_the_fold_laws() -> None:
    """A batch is a way to append, not a second kind of log: its events fold
    exactly as the same events appended one at a time, including a member that
    replaces a node an earlier member added — which only a plan against the
    state the earlier members leave can accept."""
    batched = _conversation()
    with batched.batch() as batch:
        log_event(batch, "compaction/summarized", {"shadowedSeqs": [0]})
        log_event(batch, "user/message", user_payload("summary", "m3"), _summary((0,)))
        log_event(batch, "user/message", user_payload("three", "m4"), SurfaceIntent())
        log_event(batch, "user/message", user_payload("three, shorter", "m5"), _summary((4,)))

    single = _conversation()
    for event in batched.events[2:]:
        surface = (
            None
            if event.surface_op is None
            else SurfaceIntent(event.surface_op, event.source_event_seqs)
        )
        log_event(single, event.type, thaw_json(event.data), surface)

    assert batched.surface.nodes == single.surface.nodes == (3, 1, 5)
    assert batched.derive_messages() == single.derive_messages()
    assert batched.stale() == []
    assert check_fold_laws(batched, lambda log: log.surface.nodes) == []


def test_a_batch_whose_log_moved_is_refused_whole() -> None:
    """The members were stamped against the log as the block found it. An
    `await` inside the block lets another task append; landing the batch then
    would put its members at seqs somebody else already holds."""
    session = _conversation()
    moved = pytest.raises(RuntimeError, match="moved from seq 2 to 3 while a batch was open")
    with moved, session.batch() as batch:
        log_event(batch, "compaction/summarized", {"shadowedSeqs": [0]})
        log_event(session, "turn/start", {"turn": 1})
    assert [event.type for event in session.events][2:] == ["turn/start"]


def test_a_batch_is_one_at_a_time_and_closes_behind_itself() -> None:
    session = _conversation()
    with session.batch() as outer:
        with pytest.raises(RuntimeError, match="inside another"), session.batch():
            pass
        log_event(outer, "turn/start", {"turn": 1})
    with pytest.raises(RuntimeError, match="has closed"):
        log_event(outer, "turn/start", {"turn": 2})
    with session.batch():
        pass
    assert [event.type for event in session.events][2:] == ["turn/start"]


def test_every_member_of_a_batch_says_which_batch_and_one_alone_is_not_stamped() -> None:
    """P10-15. The committed members carry the batch's membership — which is what
    lets a reader drop a torn batch whole — and a batch of one carries none, having
    nothing to keep together. The wire form round-trips it."""
    session = _conversation()
    with session.batch() as batch:
        log_event(batch, "compaction/summarized", {"shadowedSeqs": [0]})
        log_event(batch, "user/message", user_payload("summary", "m3"), _summary((0,)))
    with session.batch() as alone:
        log_event(alone, "compaction/args-truncated", {"seqs": []})

    pair, single = session.events[2:4], session.events[4]
    assert {event.batch for event in pair} == {BatchRef(first=2, count=2)}
    assert single.batch is None
    wire = pair[0].to_wire()
    assert wire["batch"] == {"first": 2, "count": 2}
    assert SessionEvent.from_wire(wire) == pair[0]
    assert "batch" not in single.to_wire()
