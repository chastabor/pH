"""Where the surface may be cut without orphaning a tool call or its result.

`ph.session.balance` is the one statement of that rule, shared by every producer that
shadows a range: compaction's summary and a model's own context edits.
"""

from __future__ import annotations

from ph.llm.types import Message
from ph.session import Session, SurfaceIntent, balanced_cuts, derive_event_message, safe_cutoff
from ph.testing import assistant_payload, log_event, tool_result_payload, user_payload


def _paired_session() -> Session:
    """user · assistant(tool-call) · tool/result · user · assistant."""
    session = Session("paired")
    log_event(session, "user/message", user_payload("do it", "m1"), SurfaceIntent("append"))
    log_event(
        session,
        "assistant/message",
        assistant_payload(
            "",
            "m2",
            content=[{"type": "tool-call", "id": "c1", "name": "read", "arguments": "{}"}],
        ),
        SurfaceIntent("append"),
    )
    log_event(
        session, "tool/result", tool_result_payload("the file", "m3", "c1"), SurfaceIntent("append")
    )
    log_event(session, "user/message", user_payload("thanks", "m4"), SurfaceIntent("append"))
    log_event(
        session, "assistant/message", assistant_payload("done", "m5"), SurfaceIntent("append")
    )
    return session


def _projected(session: Session) -> list[Message | None]:
    """The surface as messages — what `safe_cutoff` takes."""
    events = session.events
    return [derive_event_message(events[seq]) for seq in session.surface.nodes]


def test_a_cut_between_a_call_and_its_result_is_unbalanced() -> None:
    """The fold itself: five nodes, six cuts, one of them straddling the pair."""
    assert balanced_cuts(_paired_session()) == (True, True, False, True, True, True)


def test_the_cut_never_separates_a_call_from_its_result() -> None:
    """The row's gate. Asked for a cutoff that lands *between* the assistant's
    tool call and the result answering it; the answer moves back to before the
    call, so the pair travels into the summary together.

    Backward, not forward: advancing would summarize the call and leave the
    model holding a `tool-result` for a call it can no longer see, which several
    providers reject outright.
    """
    session = _paired_session()
    assert safe_cutoff(_projected(session), 2) == 1, "the cut moved back past the tool call"
    assert safe_cutoff(_projected(session), 3) == 3, "a balanced target is left where it is"


def test_a_conversation_with_no_balanced_cut_is_not_compacted() -> None:
    """An outstanding call across the whole surface: `0`, meaning no safe range.

    Not an error and not a partial compaction — dsh states the same contract:
    a single oversized retained unit cannot be repaired by replacing a surface.
    """
    session = Session("unbalanced")
    log_event(
        session,
        "assistant/message",
        assistant_payload(
            "",
            "m1",
            content=[{"type": "tool-call", "id": "c1", "name": "read", "arguments": "{}"}],
        ),
        SurfaceIntent("append"),
    )
    log_event(session, "user/message", user_payload("hello", "m2"), SurfaceIntent("append"))
    assert safe_cutoff(_projected(session), 2) == 0
