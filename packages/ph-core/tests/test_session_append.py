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
import re
from pathlib import Path
from typing import Any, cast

import pytest

from ph.session import (
    KNOWN_SESSION_EVENT_TYPES,
    Session,
    SessionEvent,
    SessionFoldCache,
    SurfaceIntent,
)
from ph.session.json import InvalidJsonValueError, JsonValue, as_obj, as_seq
from ph.testing import prefix_of, user_payload


def test_seq_always_equals_log_length() -> None:
    session = Session("s")
    for index in range(25):
        event = session.append("turn/start", {"turn": index})
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
    event = session.append("turn/start", payload)
    payload["turn"] = 99
    inner.append(3)
    # The log holds the value at append time, not a live view of the caller's
    # buffer — a stateful producer cannot rewrite history after the fact.
    assert event.data["turn"] == 1
    assert list(as_seq(as_obj(event.data["nested"])["list"])) == [1, 2]


def test_logged_data_is_not_writable() -> None:
    session = Session("s")
    event = session.append("turn/start", {"turn": 1})
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
        session.append("turn/start", payload)
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
            session.append("turn/start", cast(Any, payload))
    assert session.events == ()


def test_cyclic_payloads_are_refused() -> None:
    session = Session("s")
    # `Any` for the same reason as the refusal table above: a dict that holds
    # itself is not a `JsonValue`, which is what this asserts.
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    with pytest.raises(InvalidJsonValueError, match="circular"):
        session.append("turn/start", cycle)


def test_rejections_name_the_offending_path() -> None:
    with pytest.raises(InvalidJsonValueError, match=r"^a\.b\[1\]\.c: "):
        Session("s").append("turn/start", {"a": {"b": [0, {"c": math.nan}]}})


def test_a_raising_observer_cannot_un_append() -> None:
    session = Session("s")
    seen: list[str] = []

    def bad_observer(_session: Session, _event: object) -> None:
        raise RuntimeError("bad observer")

    session.observe(bad_observer)
    session.observe(lambda _s, event: seen.append(event.type))
    event = session.append("turn/start", {"turn": 1})
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
            source.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
        except RuntimeError as error:
            caught.append(error)

    session.observe(reenter)
    session.append("turn/start", {"turn": 1})
    # A reentrant append would assign a seq inside another event's publication,
    # so observers would watch the log grow underneath them.
    assert len(caught) == 1
    assert "cannot reenter" in str(caught[0])
    assert len(session.events) == 1


def test_surface_metadata_is_required_and_forbidden_by_type() -> None:
    session = Session("s")
    with pytest.raises(ValueError, match="requires a surfaceOp"):
        session.append("user/message", user_payload("hi"))
    with pytest.raises(ValueError, match="not surface-eligible"):
        session.append("turn/start", {"turn": 1}, SurfaceIntent("append"))


def test_events_snapshot_does_not_grow_under_a_holder() -> None:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    held = session.events
    session.append("step/start", {"turn": 1, "step": 1})
    assert len(held) == 1
    assert len(session.events) == 2


def test_every_appended_type_is_a_known_event_type() -> None:
    """A type this build can write but would refuse to read back is a trap.

    `KNOWN_SESSION_EVENT_TYPES` gates the seed path, so every literal `append(`
    call site in `ph-core` must be in the set (or a plugin-owned type that
    marks itself ignorable — none exist yet).
    """
    import ph

    pattern = re.compile(r'\.append\(\s*"([a-z][a-z0-9-]*/[a-z0-9/-]+)"')
    appended = {
        match
        for path in Path(ph.__path__[0]).rglob("*.py")
        for match in pattern.findall(path.read_text(encoding="utf-8"))
    }
    assert appended, "the scan found no append call sites — the regex is stale"
    assert appended <= KNOWN_SESSION_EVENT_TYPES, appended - KNOWN_SESSION_EVENT_TYPES


# ------------------------------------------------------------- fold caches --


def test_a_fold_cache_recomputes_only_when_the_log_grew() -> None:
    """`seq` is an exact invalidation key because the log is append-only (A1)."""
    session = Session("s")
    calls: list[int] = []

    def count_turns(log: Session) -> int:
        calls.append(log.seq)
        return sum(1 for event in log.events if event.type == "turn/start")

    cache: SessionFoldCache[int] = SessionFoldCache(count_turns)
    session.append("turn/start", {"turn": 1})

    assert cache.read(session) == 1
    assert cache.read(session) == 1
    assert calls == [1], "the fold ran twice for one log"

    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert cache.read(session) == 1
    assert calls == [1, 2], "an appended event did not invalidate the fold"


def test_a_fold_cache_holds_one_value_per_session() -> None:
    """Bounded by live sessions, not by history: entries are replaced."""
    cache: SessionFoldCache[int] = SessionFoldCache(lambda log: log.seq)
    first, second = Session("a"), Session("b")
    for index in range(50):
        first.append("turn/start", {"turn": index})
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
    session.append("turn/start", {"turn": 1})
    boundary = session.seq
    session.append("turn/start", {"turn": 2})

    def count_turns(log: Any) -> int:
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
    session.append("turn/start", {"turn": 1})
    session.append("turn/start", {"turn": 2})

    assert session.projection("turn/start", _turn_number) == 2

    session.append("turn/start", {"turn": "not a number"})
    assert session.projection("turn/start", _turn_number) == 2


def test_a_projection_parses_only_what_arrived_since_it_last_answered() -> None:
    """The whole reason this is not `for event in reversed(session.events)`.

    The walk it replaced materialised a snapshot of the log to read one field and
    grew more expensive the longer a conversation ran — asked, in the meter's
    case, on every pressure check.
    """
    session = Session("s")
    seen: list[int] = []

    def counted(event: SessionEvent) -> int | None:
        seen.append(event.seq)
        return _turn_number(event)

    for turn in range(1, 4):
        session.append("turn/start", {"turn": turn})
    assert session.projection("turn/start", counted) == 3
    assert seen == [2], "the first read walked past the event that answered"

    session.append("assistant/chunk", {"text": "hi"})
    assert session.projection("turn/start", counted) == 3
    assert seen == [2], "an event of another type was handed to the parser"

    session.append("turn/start", {"turn": 4})
    assert session.projection("turn/start", counted) == 4
    assert seen == [2, 4], "the second read re-parsed events it had already seen"


def test_two_parsers_of_one_event_type_do_not_share_a_fold() -> None:
    """The parser is half the key, which is what makes the erased type sound.

    Keyed on the event type alone — or on a name a caller typed — the second of
    these would be handed the first one's *value*: an `int` where a `str` was
    annotated, with no error until something used it.
    """
    session = Session("s")
    session.append("turn/start", {"turn": 1})

    assert session.projection("turn/start", _turn_number) == 1
    assert session.projection("turn/start", _turn_label) == "turn 1"
    assert len(session._latest) == 2


def test_one_parser_over_two_event_types_does_not_share_a_fold() -> None:
    """The other half. `assistant/chunk` carries no turn, so a shared fold would
    answer this with the `turn/start` one."""
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("assistant/chunk", {"text": "hi"})

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
        session.append("turn/start", {"turn": turn})

    for _ in range(5):
        assert session.latest("turn/start") is not None
        assert session.projection("turn/start", _turn_number) == 3
    assert len(session._latest) == 2, "a fold was rebuilt instead of reused"
