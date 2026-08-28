"""P0-09 — the ordered surface.

Gates: *replace leaves the log intact; an unknown non-ignorable type refuses the
log.*

The first is invariant I4 stated as a test: compaction removes nodes from the
*derivation*, never from history. It is why offload, rollback and fork all reuse
one mechanism, and why a checkpointer could not have substituted.
"""

from __future__ import annotations

import pytest

from ph.session import (
    Session,
    SurfaceIntent,
    SurfaceReplace,
    fold_surface,
    is_in_place_rewrite,
)
from ph.session.surface import (
    SurfaceError,
    is_append_surface_event,
    is_replacement_surface_event,
    is_surface_event,
)
from ph.testing import assistant_payload, tool_result_payload, user_payload


def _conversation() -> Session:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user_payload("one", "m1"), SurfaceIntent("append"))
    session.append("assistant/message", assistant_payload("two", "m2"), SurfaceIntent("append", ()))
    session.append("user/message", user_payload("three", "m3"), SurfaceIntent("append"))
    return session


def test_only_surface_events_join_the_surface() -> None:
    assert _conversation().surface.nodes == (1, 2, 3)


def test_replace_shadows_nodes_and_leaves_the_log_intact() -> None:
    session = _conversation()
    before = len(session.events)
    session.append(
        "user/message",
        user_payload("summary of one and two", "m4"),
        SurfaceIntent(SurfaceReplace(start=1, end=2), (1, 2)),
    )
    assert session.surface.nodes == (4, 3)
    assert len(session.events) == before + 1
    # The shadowed events are still there, byte-for-byte.
    assert session.events[1].data["content"][0]["text"] == "one"
    assert session.surface.replace_generation == 1


def test_replace_must_cite_every_shadowed_node() -> None:
    session = _conversation()
    with pytest.raises(SurfaceError, match="must include every shadowed"):
        session.append(
            "user/message",
            user_payload("summary", "m4"),
            SurfaceIntent(SurfaceReplace(start=1, end=2), (1,)),
        )
    # A refused candidate leaves the surface exactly as it was.
    assert session.surface.nodes == (1, 2, 3)
    assert len(session.events) == 4


def test_replace_bounds_must_exist_and_be_ordered() -> None:
    session = _conversation()
    with pytest.raises(SurfaceError, match="start seq 99 not found"):
        session.append(
            "user/message",
            user_payload("x", "m4"),
            SurfaceIntent(SurfaceReplace(start=99, end=2), (99, 2)),
        )
    with pytest.raises(SurfaceError, match="is after end seq"):
        session.append(
            "user/message",
            user_payload("x", "m5"),
            SurfaceIntent(SurfaceReplace(start=3, end=1), (1, 2, 3)),
        )


def test_source_seqs_must_be_earlier_and_unique() -> None:
    session = _conversation()
    with pytest.raises(SurfaceError, match="must reference earlier events"):
        session.append(
            "assistant/message", assistant_payload("x", "m4"), SurfaceIntent("append", (99,))
        )
    with pytest.raises(SurfaceError, match="duplicates"):
        session.append(
            "assistant/message", assistant_payload("x", "m5"), SurfaceIntent("append", (1, 1))
        )


def test_only_assistant_messages_may_cite_an_empty_source_set() -> None:
    session = _conversation()
    session.append("assistant/message", assistant_payload("x", "m4"), SurfaceIntent("append", ()))
    with pytest.raises(SurfaceError, match="must not be empty except"):
        session.append("user/message", user_payload("x", "m5"), SurfaceIntent("append", ()))


def test_tool_result_replacement_may_change_only_content() -> None:
    session = Session("s")
    session.append("tool/result", tool_result_payload("full output", "r1"), SurfaceIntent("append"))
    # Offload rewrites the result content in place; everything else must match.
    session.append(
        "tool/result",
        tool_result_payload("preview…", "r1"),
        SurfaceIntent(SurfaceReplace(start=0, end=0), (0,)),
    )
    assert session.surface.nodes == (1,)
    with pytest.raises(SurfaceError, match="may change only content"):
        session.append(
            "tool/result",
            tool_result_payload("preview…", "r2", call_id="OTHER"),
            SurfaceIntent(SurfaceReplace(start=1, end=1), (1,)),
        )


def test_tool_result_replacement_targets_exactly_one_node() -> None:
    session = Session("s")
    session.append("tool/result", tool_result_payload("a", "r1"), SurfaceIntent("append"))
    session.append("tool/result", tool_result_payload("b", "r2", "c2"), SurfaceIntent("append"))
    with pytest.raises(SurfaceError, match="exactly one current node"):
        session.append(
            "tool/result",
            tool_result_payload("merged", "r3"),
            SurfaceIntent(SurfaceReplace(start=0, end=1), (0, 1)),
        )


def test_fold_matches_the_incremental_manager() -> None:
    session = _conversation()
    session.append(
        "user/message",
        user_payload("summary", "m4"),
        SurfaceIntent(SurfaceReplace(start=1, end=2), (1, 2)),
    )
    folded = fold_surface(session.events)
    # The offline reconstructor and the live manager must agree, or "replay the
    # log" and "read the session" are two different answers.
    assert folded.nodes == session.surface.nodes
    assert len(folded.replacements) == 1
    assert folded.replacements[0].shadowed_seqs == (1, 2)


def test_incremental_reads_see_only_the_delta() -> None:
    session = _conversation()
    assert session.surface.node_count == 3
    assert session.surface.nodes_from(2) == (3,)
    session.append("user/message", user_payload("four", "m4"), SurfaceIntent("append"))
    assert session.surface.nodes_from(3) == (4,)


def test_surface_predicates() -> None:
    session = _conversation()
    session.append(
        "user/message",
        user_payload("summary", "m4"),
        SurfaceIntent(SurfaceReplace(start=1, end=2), (1, 2)),
    )
    assert not is_surface_event(session.events[0])
    assert is_append_surface_event(session.events[1])
    assert not is_replacement_surface_event(session.events[1])
    # A human transcript reads append-origin events, so a landed compaction
    # never erases conversation the user already saw.
    assert is_replacement_surface_event(session.events[4])
    assert not is_append_surface_event(session.events[4])


def test_an_in_place_rewrite_is_told_apart_from_a_substitution() -> None:
    """The two shapes a `replace` comes in, distinguished structurally.

    A consumer that keyed on `is_replacement_surface_event` alone had to carry a
    list of which producers do which. A node replacing *itself* — a truncated
    argument, a relocated tool result — removes nothing from the conversation and
    should update the row a reader already has; a substitution stands in for a
    range and shadows it.
    """
    session = Session("shapes")
    first = session.append("user/message", user_payload("one", "m1"), SurfaceIntent("append"))
    second = session.append("user/message", user_payload("two", "m2"), SurfaceIntent("append"))

    in_place = session.append(
        "user/message",
        user_payload("one, elided", "m3"),
        SurfaceIntent(SurfaceReplace(start=first.seq, end=first.seq), (first.seq,)),
    )
    substitution = session.append(
        "user/message",
        user_payload("a summary of both", "m4"),
        SurfaceIntent(
            SurfaceReplace(start=in_place.seq, end=second.seq), (in_place.seq, second.seq)
        ),
    )

    assert is_in_place_rewrite(in_place)
    assert not is_in_place_rewrite(substitution)
    assert is_replacement_surface_event(substitution), "both are still replacements"
    assert not is_in_place_rewrite(first), "an append is not a rewrite"
