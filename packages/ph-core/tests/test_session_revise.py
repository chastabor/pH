"""`ph.session.revise` — the one door for surface revisions, tested once.

Compaction, input-offload and ph-clm write every replacement through `substitute`
and `rewrite`, so the rules each replacement keeps are held here rather than in
three producers' suites: a substitution is a user message of its own; a rewrite
keeps the message id, keeps a result's every field but content, and leaves an
assistant reply's usage and step behind (D8). And a static gate holds that no
other shipped module constructs a `SurfaceReplace`, so the door stays the only one.
"""

from __future__ import annotations

import pytest

from ph.json import JsonValue, thaw_json
from ph.llm.types import Message, create_user_message
from ph.session import (
    Session,
    SessionEvent,
    SurfaceError,
    SurfaceIntent,
    editable_message,
    is_in_place_rewrite,
    is_replacement_surface_event,
    origin_of,
    originals,
    rewrite,
    shadowed_by,
    substitute,
)
from ph.testing import assistant_payload, log_event, tool_result_payload, user_payload


def _conversation() -> tuple[Session, SessionEvent, SessionEvent, SessionEvent]:
    """user · assistant(call, with usage) · tool/result."""
    session = Session("revise")
    asked = log_event(
        session, "user/message", user_payload("read it", "m1"), SurfaceIntent("append")
    )
    call = {"type": "tool-call", "id": "c1", "name": "read", "arguments": '{"path": "a.md"}'}
    replied = log_event(
        session,
        "assistant/message",
        {
            **assistant_payload(
                "reading a.md", "m2", content=[{"type": "text", "text": "reading a.md"}, call]
            ),
            "usage": {"inputTokens": 40, "outputTokens": 9},
        },
        SurfaceIntent("append"),
    )
    result = log_event(
        session,
        "tool/result",
        {**tool_result_payload("the whole file", "m3", "c1"), "meta": {"bytes": 14}},
        SurfaceIntent("append"),
    )
    return session, asked, replied, result


def _summary(text: str = "a summary") -> Message:
    return create_user_message(
        content=[{"type": "text", "text": text}],
        source={"kind": "plugin", "plugin": "test", "form": "compaction"},
    )


# --------------------------------------------------------------- substitute --


def test_a_substitution_stands_in_for_the_run_and_the_log_keeps_it() -> None:
    session, asked, replied, result = _conversation()

    stand_in = substitute(session, (replied.seq, result.seq), _summary())

    assert stand_in.type == "user/message"
    assert shadowed_by(stand_in) == (replied.seq, result.seq)
    assert session.surface.nodes == (asked.seq, stand_in.seq)
    assert len(session.transcript()) == 3, "the person's transcript lost the originals"


def test_a_substitution_for_one_message_is_not_a_rewrite() -> None:
    """The shape a structural test could not tell from a rewrite — one node, cited as
    the source — told apart by type, which the door makes exact."""
    session, asked, _, _ = _conversation()

    stand_in = substitute(session, (asked.seq,), _summary())

    assert is_replacement_surface_event(stand_in)
    assert not is_in_place_rewrite(stand_in)
    assert origin_of(session, stand_in.seq) == stand_in.seq, "a substitution kept the old name"


def test_a_substitution_is_a_user_message_standing_for_something() -> None:
    session, asked, replied, _ = _conversation()
    as_assistant = create_user_message(
        content=[{"type": "text", "text": "x"}], source={"kind": "user"}
    ).model_copy(update={"role": "assistant"})

    with pytest.raises(SurfaceError, match="user message"):
        substitute(session, (replied.seq,), as_assistant)
    with pytest.raises(ValueError, match="at least one surface node"):
        substitute(session, (), _summary())
    assert session.surface.nodes[0] == asked.seq and len(session.events) == 3


# ------------------------------------------------------------------ rewrite --


def test_a_rewritten_reply_keeps_its_id_and_leaves_usage_and_step_behind() -> None:
    session, _, replied, _ = _conversation()
    message = editable_message(replied)
    message["content"][0] = {"type": "text", "text": "reading"}

    rewritten = rewrite(session, replied, message)

    assert rewritten.type == "assistant/message"
    assert is_in_place_rewrite(rewritten)
    assert not {"usage", "turn", "step"} & set(rewritten.data)
    assert editable_message(rewritten)["id"] == "m2"
    assert origin_of(session, rewritten.seq) == replied.seq


def test_a_rewritten_result_keeps_every_field_but_its_content() -> None:
    session, _, _, result = _conversation()
    message = editable_message(result)
    message["content"][0]["content"] = [{"type": "text", "text": "(cut)"}]

    rewritten = rewrite(session, result, message)

    assert is_in_place_rewrite(rewritten)
    unchanged = {key: value for key, value in rewritten.data.items() if key != "message"}
    assert unchanged == {key: value for key, value in result.data.items() if key != "message"}


def test_a_rewrite_that_changes_the_message_is_refused() -> None:
    """A new id is a new message — a substitution — and a result whose call id
    moved would rewrite what the model is told happened (core's own check)."""
    session, asked, replied, result = _conversation()
    renamed: dict[str, JsonValue] = {**editable_message(replied), "id": "m9"}
    moved = editable_message(result)
    moved["content"][0]["toolCallId"] = "c9"

    with pytest.raises(SurfaceError, match="must keep the message id"):
        rewrite(session, replied, renamed)
    with pytest.raises(SurfaceError, match="may change only content"):
        rewrite(session, result, moved)
    with pytest.raises(SurfaceError, match='not "user/message"'):
        rewrite(session, asked, thaw_json(asked.data))
    assert len(session.events) == 3, "a refused rewrite wrote to the log"


def test_the_door_writes_into_a_producers_batch() -> None:
    """So a producer's record of why lands with the revision it describes."""
    session, _, replied, result = _conversation()

    with session.batch() as batch:
        stand_in = substitute(batch, (replied.seq, result.seq), _summary())
        record = log_event(batch, "compaction/declined", {"trigger": "manual", "code": "x"})

    landed, recorded = session.at(stand_in.seq), session.at(record.seq)
    assert landed is not None and recorded is not None
    assert landed.batch is not None and landed.batch == recorded.batch


# ------------------------------------------------------------------ reading --


def test_originals_read_back_through_every_revision() -> None:
    """A substitution over a rewrite reads back to what was first appended."""
    session, _, replied, result = _conversation()
    message = editable_message(result)
    message["content"][0]["content"] = [{"type": "text", "text": "(cut)"}]
    rewritten = rewrite(session, result, message)
    stand_in = substitute(session, (replied.seq, rewritten.seq), _summary())

    found = list(originals(session, stand_in.seq))

    assert [one.id for one in found] == ["m2", "m3"]
    assert "the whole file" in str(found[1].content), "the walk stopped at the rewrite"
