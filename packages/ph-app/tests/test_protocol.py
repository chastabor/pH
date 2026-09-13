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

from ph.session import Session
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
        session.append("turn/start", {"turn": index})
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
