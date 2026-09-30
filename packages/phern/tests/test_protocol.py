"""The envelope, pinned at runtime (P8-07).

`ph_app.protocol`'s frames are `TypedDict`s — a claim mypy checks against every
builder and no test can. What a test *can* pin is the runtime shape those types
describe, so the two cannot drift: a frame with no `id` is a notification, a
reply carries `result` or `error` and never both, a refusal that named itself
reaches `error.data.reason`, and an id-less request still runs its body.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ph.session import Session, SessionHeader
from ph.testing import log_event
from ph.wire import WireModel
from ph_app.protocol import (
    Cursor,
    DaemonError,
    ErrorFrame,
    InvalidParams,
    Refusal,
    cursor_of,
    notification,
    parse_cursor,
    parse_params,
    request,
    respond,
    result_of,
    resume_at,
)

pytestmark = pytest.mark.anyio


def test_a_notification_has_no_id_and_a_request_has_one() -> None:
    assert notification("session.event", {"a": 1}) == {
        "jsonrpc": "2.0",
        "method": "session.event",
        "params": {"a": 1},
    }
    assert request("c1", "session/status", {}) == {
        "jsonrpc": "2.0",
        "id": "c1",
        "method": "session/status",
        "params": {},
    }


class _Named(Refusal):
    code = "named_refusal"


async def test_respond_shapes_a_result_an_error_and_nothing_for_an_id_less_frame() -> None:
    ran: list[str] = []

    async def dispatch(method: str, params: dict[str, Any]) -> Any:  # noqa: ANN401
        ran.append(method)
        if method == "refuse":
            raise _Named("no")
        if method == "crash":
            raise RuntimeError("boom")
        return {"ok": True}

    assert await respond({"id": 1, "method": "fine", "params": {}}, dispatch) == {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"ok": True},
    }
    refused = await respond({"id": 2, "method": "refuse"}, dispatch)
    assert refused == {
        "jsonrpc": "2.0",
        "id": 2,
        "error": {"code": -32000, "message": "no", "data": {"reason": "named_refusal"}},
    }
    crashed = await respond({"id": 3, "method": "crash"}, dispatch)
    assert crashed is not None and "error" in crashed
    failed = cast("ErrorFrame", crashed)
    assert "data" not in failed["error"], "an unnamed failure carries no reason to branch on"
    # A notification's body runs; nothing comes back.
    assert await respond({"method": "fine", "params": {}}, dispatch) is None
    assert ran == ["fine", "refuse", "crash", "fine"]


def test_result_of_raises_the_named_refusal() -> None:
    assert result_of({"jsonrpc": "2.0", "id": 1, "result": {"x": 1}}) == {"x": 1}
    with pytest.raises(DaemonError) as raised:
        result_of(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32000, "message": "no", "data": {"reason": "why"}},
            }
        )
    assert raised.value.reason == "why"


class _Params(WireModel):
    session_id: str
    count: int = 0


def test_parse_params_names_the_method_and_the_field() -> None:
    parsed = parse_params("m", _Params, {"sessionId": "s", "count": 2})
    assert (parsed.session_id, parsed.count) == ("s", 2)
    with pytest.raises(InvalidParams) as missing:
        parse_params("session/status", _Params, {})
    assert missing.value.code == "invalid_params"
    assert "session/status" in str(missing.value) and "sessionId" in str(missing.value)
    with pytest.raises(InvalidParams) as extra:
        parse_params("m", _Params, {"sessionId": "s", "cursor": 1})
    assert "cursor" in str(extra.value), "a field the method does not take is refused, by name"
    with pytest.raises(InvalidParams) as wrong:
        parse_params("m", _Params, {"sessionId": "s", "count": "many"})
    assert "count" in str(wrong.value)


def test_a_cursor_has_one_spelling() -> None:
    """Built by `cursor_of`, parsed by `parse_cursor`, read by `resume_at` — one
    model, so the dict a client sends is the dict the server's model accepts."""
    session = Session("s")
    for index in range(4):
        log_event(session, "turn/start", {"turn": index})
    generation = str(session.header.created_at)

    built = cursor_of(session)
    assert built == Cursor(generation=generation, sequence=4)
    assert built.to_wire() == {"generation": generation, "sequence": 4}
    # Parsed back to the same model, from either spelling a person may type.
    assert parse_cursor("2", built.to_wire()) == Cursor(generation=generation, sequence=2)
    assert parse_cursor(f"{generation}:2", {}) == Cursor(generation=generation, sequence=2)
    assert parse_cursor("not-a-cursor", built.to_wire()) is None

    assert resume_at(session, None) == 0
    assert resume_at(session, Cursor(generation="another", sequence=2)) == 0, "stale reads as 0"
    assert resume_at(session, Cursor(generation=generation, sequence=2)) == 2
    assert resume_at(session, Cursor(generation=generation, sequence=99)) == 4, "clamped"


def _resumed(before: Session, durable: int) -> Session:
    """`before` as a restart finds it: the store kept its first `durable` events, and
    the resume recorded so (`resume_session`, without the store around it)."""
    revived = Session(
        before.id, seed=before.events[:durable], header=before.header, durable=durable
    )
    log_event(revived, "session/resumed", {"events": durable})
    return revived


def test_a_cursor_from_before_a_resume_holds_only_as_far_as_the_store_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S17: events reach a reader before they are durable, so a crash takes back ones
    it saw and the resume writes others at those seqs. Honored whole, the old cursor
    skipped the resume's own records and kept events the log no longer has."""
    clock = iter(range(1_000, 2_000))
    monkeypatch.setattr("ph.session.session.now_ms", lambda: next(clock))
    first = Session("s")
    for index in range(5):
        log_event(first, "turn/start", {"turn": index})
    seen = cursor_of(first)
    assert seen.sequence == 5

    second = _resumed(first, durable=3)
    for index in range(5):
        log_event(second, "turn/start", {"turn": index})
    assert cursor_of(second).generation != seen.generation, "a resume begins an incarnation"
    assert resume_at(second, seen) == 3, "as far as the store kept, not as far as it saw"
    assert resume_at(second, Cursor(generation=seen.generation, sequence=2)) == 2
    assert resume_at(second, cursor_of(second, 7)) == 7, "this incarnation's own count holds"

    third = _resumed(second, durable=8)
    assert resume_at(third, seen) == 3, "bound by every resume since, not the latest"
    assert resume_at(third, cursor_of(second)) == 8
    assert resume_at(third, Cursor(generation="999", sequence=4)) == 0, "never this log's"


def test_a_fork_does_not_take_its_sources_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fork's seed carries its source's `session/resumed`; read as the fork's own,
    the fork would answer to cursors counted in its source."""
    clock = iter(range(1_000, 2_000))
    monkeypatch.setattr("ph.session.session.now_ms", lambda: next(clock))
    source = _resumed(Session("s"), durable=0)
    log_event(source, "turn/start", {"turn": 0})
    header = SessionHeader(
        id="f", created_at=5_000, parent_session="s", seed_length=len(source.events)
    )
    fork = Session("f", seed=source.events, header=header)
    assert cursor_of(fork).generation == "5000"
    assert resume_at(fork, cursor_of(source)) == 0
