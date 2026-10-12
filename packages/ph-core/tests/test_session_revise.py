"""`ph.session.revise` — the one door for surface revisions, tested once.

Compaction, input-offload and ph-clm write every replacement through `substitute`
and `rewrite`, so the rules each replacement keeps are held here rather than in
three producers' suites: a substitution is a user message of its own; a rewrite
keeps the message id, keeps a result's every field but content, and leaves an
assistant reply's usage and step behind (D8); and every revision carries its
producer's record, which the door writes after it, stamped with what it revises, so
`Session.records_of` pairs the two by lookup. And a static gate holds that no other
shipped module constructs a `SurfaceReplace`, so the door stays the only one.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.json import JsonValue, thaw_json
from ph.llm.types import Message, create_user_message
from ph.session import (
    RevisionSize,
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
    rewrite_many,
    shadowed_by,
    substitute,
)
from ph.session.writers import LogWriteError
from ph.testing import (
    assistant_payload,
    log_event,
    revision_record,
    tool_result_payload,
    user_payload,
)

WHY = revision_record("compaction/summarized", {"trigger": "manual"})
"""A producer's record, for a revision whose record this test does not look at."""


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


def _cut(result: SessionEvent) -> dict[str, Any]:
    """`result`'s message with its output cut to "(cut)", for a rewrite of it."""
    message = editable_message(result)
    message["content"][0]["content"] = [{"type": "text", "text": "(cut)"}]
    return message


def _summary(text: str = "a summary") -> Message:
    return create_user_message(
        content=[{"type": "text", "text": text}],
        source={"kind": "plugin", "plugin": "test", "form": "compaction"},
    )


# --------------------------------------------------------------- substitute --


def test_a_substitution_stands_in_for_the_run_and_the_log_keeps_it() -> None:
    session, asked, replied, result = _conversation()

    stand_in = substitute(session, (replied.seq, result.seq), _summary(), WHY)

    assert stand_in.type == "user/message"
    assert shadowed_by(stand_in) == (replied.seq, result.seq)
    assert session.surface.nodes == (asked.seq, stand_in.seq)
    assert len(session.transcript()) == 3, "the person's transcript lost the originals"


def test_a_substitution_for_one_message_is_not_a_rewrite() -> None:
    """The shape a structural test could not tell from a rewrite — one node, cited as
    the source — told apart by type, which the door makes exact."""
    session, asked, _, _ = _conversation()

    stand_in = substitute(session, (asked.seq,), _summary(), WHY)

    assert is_replacement_surface_event(stand_in)
    assert not is_in_place_rewrite(stand_in)
    assert origin_of(session, stand_in.seq) == stand_in.seq, "a substitution kept the old name"


def test_a_substitution_is_a_user_message_standing_for_something() -> None:
    session, asked, replied, _ = _conversation()
    as_assistant = create_user_message(
        content=[{"type": "text", "text": "x"}], source={"kind": "user"}
    ).model_copy(update={"role": "assistant"})

    with pytest.raises(SurfaceError, match="user message"):
        substitute(session, (replied.seq,), as_assistant, WHY)
    with pytest.raises(ValueError, match="at least one surface node"):
        substitute(session, (), _summary(), WHY)
    assert session.surface.nodes[0] == asked.seq and len(session.events) == 3


# ------------------------------------------------------------------ rewrite --


def test_a_rewritten_reply_keeps_its_id_and_leaves_usage_and_step_behind() -> None:
    session, _, replied, _ = _conversation()
    message = editable_message(replied)
    message["content"][0] = {"type": "text", "text": "reading"}

    rewritten = rewrite(session, replied, message, WHY)

    assert rewritten.type == "assistant/message"
    assert is_in_place_rewrite(rewritten)
    assert not {"usage", "turn", "step"} & set(rewritten.data)
    assert editable_message(rewritten)["id"] == "m2"
    assert origin_of(session, rewritten.seq) == replied.seq


def test_a_rewritten_result_keeps_every_field_but_its_content() -> None:
    session, _, _, result = _conversation()
    message = _cut(result)

    rewritten = rewrite(session, result, message, WHY)

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
        rewrite(session, replied, renamed, WHY)
    with pytest.raises(SurfaceError, match="may change only content"):
        rewrite(session, result, moved, WHY)
    with pytest.raises(SurfaceError, match='not "user/message"'):
        rewrite(session, asked, thaw_json(asked.data), WHY)
    assert len(session.events) == 3, "a refused rewrite wrote to the log"


def test_the_door_writes_into_a_producers_batch() -> None:
    """So a producer's record of why lands with the revision it describes."""
    session, _, replied, result = _conversation()

    with session.batch() as batch:
        stand_in = substitute(batch, (replied.seq, result.seq), _summary(), WHY)

    landed, (record,) = session.at(stand_in.seq), session.records_of(stand_in.seq)
    assert landed is not None and landed.batch is not None and landed.batch == record.batch


# ------------------------------------------------------------------ records --


def test_each_revision_s_record_is_found_by_the_replacement_it_explains() -> None:
    """The door writes the producer's record right after the replacement and stamps it
    with it, so the pairing is a lookup rather than a guess from where it sat.
    Sabotage: leave the stamp off, or stamp what was replaced."""
    session, asked, _, result = _conversation()
    message = _cut(result)
    spilled = revision_record("offload/spilled", {"callId": "c1"})

    clipped = rewrite(session, result, message, spilled)
    summary = substitute(session, (asked.seq,), _summary(), WHY)

    (by_clip,) = session.records_of(clipped.seq)
    (by_summary,) = session.records_of(summary.seq)
    assert (by_clip.type, by_clip.seq, by_clip.data["revises"]) == (
        "offload/spilled",
        clipped.seq + 1,
        (clipped.seq,),
    )
    assert (by_summary.type, by_summary.data["revises"]) == (
        "compaction/summarized",
        (summary.seq,),
    )
    assert session.records_of(result.seq) == (), "the record named what was replaced"
    landed = session.at(summary.seq)
    assert landed is not None and landed.batch is not None, "handed a session, it batched none"
    assert landed.batch == by_summary.batch, "the pair can tear apart"


def test_one_record_for_many_rewrites_names_each() -> None:
    """Argument elision's shape: every rewrite, then one record that explains them
    all. Sabotage: stamp only the last."""
    session, _, replied, result = _conversation()
    reply = editable_message(replied)
    reply["content"][0] = {"type": "text", "text": "reading"}
    output = _cut(result)
    elided = revision_record("compaction/args-truncated", {"savedChars": 9})

    with session.batch() as batch:
        first, second = rewrite_many(batch, [(replied, reply), (result, output)], elided)

    assert session.records_of(first.seq) == session.records_of(second.seq)
    (record,) = session.records_of(first.seq)
    assert record.data["revises"] == (first.seq, second.seq)
    with pytest.raises(ValueError, match="at least one"):
        rewrite_many(session, [], elided)


def test_a_record_s_size_is_written_by_the_door() -> None:
    """One key for every producer's size, so a reader asks every record the same way;
    data that already carries one would read as the door's. Sabotage: drop the size,
    or let the data carry it."""
    session, asked, _, _ = _conversation()
    sized = revision_record("compaction/summarized", {}, RevisionSize("tokens", 40, 6))

    summary = substitute(session, (asked.seq,), _summary(), sized)

    (record,) = session.records_of(summary.seq)
    assert record.data["size"] == {"unit": "tokens", "before": 40, "after": 6}
    with pytest.raises(ValueError, match="door's to write"):
        revision_record("compaction/summarized", {"size": 3})


def test_only_the_door_stamps_what_a_record_revises() -> None:
    """A record that claimed a replacement it was not written with would pair the two
    for every reader. Sabotage: drop the writer's refusal."""
    session, asked, _, _ = _conversation()

    with pytest.raises(LogWriteError, match="only the revision door"):
        log_event(session, "compaction/summarized", {"revises": [asked.seq]})
    assert session.records_of(asked.seq) == ()


# ------------------------------------------------------------------ reading --


def test_a_node_s_history_is_every_revision_behind_it_oldest_first() -> None:
    """What `/diff` groups by and ph-clm's `context_diff` lists: a summary over a rewrite
    has both behind it, and a node no revision touched has none. Sabotage: stop at the
    node itself."""
    session, asked, replied, result = _conversation()
    message = _cut(result)
    rewritten = rewrite(session, result, message, WHY)

    stand_in = substitute(session, (replied.seq, rewritten.seq), _summary(), WHY)

    assert [one.seq for one in session.history(stand_in.seq)] == [rewritten.seq, stand_in.seq]
    assert [one.seq for one in session.history(rewritten.seq)] == [rewritten.seq]
    assert session.history(asked.seq) == ()


def test_originals_read_back_through_every_revision() -> None:
    """A substitution over a rewrite reads back to what was first appended."""
    session, _, replied, result = _conversation()
    message = _cut(result)
    rewritten = rewrite(session, result, message, WHY)
    stand_in = substitute(session, (replied.seq, rewritten.seq), _summary(), WHY)

    found = list(originals(session, stand_in.seq))

    assert [one.id for one in found] == ["m2", "m3"]
    assert "the whole file" in str(found[1].content), "the walk stopped at the rewrite"
