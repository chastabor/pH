"""P1-12 — crash repair.

Gates: *open-turn fixtures resume cleanly; the synthetic vocabulary matches
dsh's.*

The vocabulary matters because a resumed **model** reads it. `TOOL_NOT_STARTED`
tells it nothing ran, so retry freely; `TOOL_OUTCOME_UNKNOWN` tells it the call
may have completed, so reason from the tool's semantics. Collapsing them into
one message would make a blind retry of a non-idempotent operation the obvious
move — which is how one crash becomes two side effects.
"""

from __future__ import annotations

import pytest

from ph.persistence.repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    interrupted_turn_closers,
    repaired,
)
from ph.session import Session, SurfaceIntent
from ph.testing import assistant_payload, user_payload


def _assistant_with_call(call_id: str, *, turn: int = 1, step: int = 1) -> dict:
    payload = assistant_payload("working on it", "m2", turn=turn, step=step)
    payload["message"]["content"].append(
        {"type": "tool-call", "id": call_id, "name": "edit", "arguments": "{}"}
    )
    return payload


def _open_turn_with_unstarted_call() -> Session:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("user/message", user_payload("do it", "m1"), SurfaceIntent("append"))
    session.append("assistant/message", _assistant_with_call("c1"), SurfaceIntent("append", ()))
    return session


def test_a_balanced_log_needs_no_repair() -> None:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert interrupted_turn_closers(session.events) == []
    # Reopening a clean session must not grow its log.
    assert len(repaired(session.events)) == len(session.events)


def test_an_empty_log_needs_no_repair() -> None:
    assert interrupted_turn_closers(()) == []


def test_a_call_that_never_started_is_closed_as_not_started() -> None:
    session = _open_turn_with_unstarted_call()
    closers = interrupted_turn_closers(session.events)
    assert [event.type for event in closers] == ["tool/result", "step/end", "turn/end"]

    result = closers[0]
    assert result.data["error"]["code"] == TOOL_NOT_STARTED
    assert (
        "before the Harness recorded it as started"
        in (result.data["message"]["content"][0]["content"][0]["text"])
    )
    # Nothing ran, so it cites no call event.
    assert result.source_event_seqs is None


def test_a_recorded_call_is_closed_as_outcome_unknown() -> None:
    session = _open_turn_with_unstarted_call()
    call_seq = session.append(
        "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "edit", "arguments": "{}"}
    ).seq

    closers = interrupted_turn_closers(session.events)
    result = closers[0]
    assert result.data["error"]["code"] == TOOL_OUTCOME_UNKNOWN
    text = result.data["message"]["content"][0]["content"][0]["text"]
    # The model is told to reason from the tool, not to retry blindly.
    assert "Do not retry blindly." in text
    assert "read-only or idempotent" in text
    assert result.source_event_seqs == (call_seq,)


def test_a_completed_call_is_not_re_closed() -> None:
    session = _open_turn_with_unstarted_call()
    call_seq = session.append(
        "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "edit", "arguments": "{}"}
    ).seq
    session.append(
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "r1",
                "role": "user",
                "source": {"kind": "tool", "callId": "c1"},
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "c1",
                        "isError": False,
                        "content": [{"type": "text", "text": "done"}],
                    }
                ],
            },
        },
        SurfaceIntent("append", (call_seq,)),
    )
    closers = interrupted_turn_closers(session.events)
    # Only the boundaries are missing now.
    assert [event.type for event in closers] == ["step/end", "turn/end"]


def test_closers_continue_the_log_and_reuse_the_last_timestamp() -> None:
    session = _open_turn_with_unstarted_call()
    last = session.events[-1]
    closers = interrupted_turn_closers(session.events)
    assert [event.seq for event in closers] == [last.seq + 1, last.seq + 2, last.seq + 3]
    # Never `now()`: a repair that invented a future time would make the log say
    # the recovery happened during the crash.
    assert {event.time for event in closers} == {last.time}


def test_the_turn_ends_as_interrupted() -> None:
    session = _open_turn_with_unstarted_call()
    closers = interrupted_turn_closers(session.events)
    assert closers[-1].data["reason"] == {"kind": "interrupted"}


def test_a_repaired_log_seeds_a_resumable_session() -> None:
    session = _open_turn_with_unstarted_call()
    resumed = Session("s", seed=repaired(session.events))
    # Provider-valid: the assistant's tool call now has its result, so nothing
    # rejects the transcript.
    messages = resumed.derive_messages()
    assert [message.role for message in messages] == ["user", "assistant", "user"]
    tool_result = messages[-1].content[0]
    assert tool_result.type == "tool-result"
    assert tool_result.is_error is True
    assert resumed.events[-1].type == "session/end-seed"


def test_an_earlier_completed_turn_is_untouched() -> None:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append("turn/start", {"turn": 2})
    session.append("step/start", {"turn": 2, "step": 1})
    closers = interrupted_turn_closers(session.events)
    assert [event.type for event in closers] == ["step/end", "turn/end"]
    assert closers[-1].data["turn"] == 2


def test_a_turn_with_no_open_step_closes_only_the_turn() -> None:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    closers = interrupted_turn_closers(session.events)
    assert [event.type for event in closers] == ["turn/end"]


@pytest.mark.anyio
async def test_resume_repairs_a_crashed_log_on_load(mount, tmp_path) -> None:
    """The repair is on the load path, so nothing downstream sees an open turn."""
    from ph.persistence import resume_session

    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.sessions.create("crashed")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("assistant/message", _assistant_with_call("c1"), SurfaceIntent("append", ()))
    await ctx.sessions.flush(session)
    ctx.sessions.dispose("crashed")

    revived = await resume_session(ctx, "crashed")
    types = [event.type for event in revived.events]
    # The closers, then the record that this was a resume: the repair is what
    # makes the log readable, and `session/resumed` is what makes the *seam*
    # visible to anything reading it afterwards.
    assert types[-5:] == [
        "tool/result",
        "step/end",
        "turn/end",
        "session/end-seed",
        "session/resumed",
    ]
    assert revived.events[-1].data["interrupted"] is True, "a crashed tail was not reported"
