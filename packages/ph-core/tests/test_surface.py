"""P0-09 — the ordered surface.

Gates: *replace leaves the log intact; an unknown non-ignorable type refuses the
log.*

The first is invariant I4 stated as a test: compaction removes nodes from the
*derivation*, never from history. It is why offload, rollback and fork all reuse
one mechanism, and why a checkpointer could not have substituted.

## Why `_shadowed` resolves names with a set rather than `.index`

The first version resolved each named seq by scanning the node list from position
0, which is O(names x nodes). A compaction shadowing **1 000 of 2 000 nodes
measured 4.5 ms against 0.13 ms** for the range op it replaced — **35x**, and
quadratic in the number of names, which is precisely the direction compaction
grows.
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
        SurfaceIntent(SurfaceReplace(replaces=(1, 2)), (1, 2)),
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
            SurfaceIntent(SurfaceReplace(replaces=(1, 2)), (1,)),
        )
    # A refused candidate leaves the surface exactly as it was.
    assert session.surface.nodes == (1, 2, 3)
    assert len(session.events) == 4


def test_a_replace_must_name_nodes_that_are_on_the_surface_now() -> None:
    """Membership, and nothing in between.

    A range could be *bounded* by two real nodes and still sweep up whatever sat
    between them; a name either is a current node or the whole operation is
    refused. That refusal is the branch-safety property: the same replacement
    applied to a surface that has moved on cannot quietly shadow different
    messages.
    """
    session = _conversation()
    with pytest.raises(SurfaceError, match="seq 99 is not a current surface node"):
        session.append(
            "user/message",
            user_payload("x", "m4"),
            SurfaceIntent(SurfaceReplace(replaces=(99, 2)), (99, 2)),
        )


def test_the_order_the_names_are_given_in_does_not_matter() -> None:
    """A **set**, so `(2, 1)` and `(1, 2)` are one operation.

    Under a range this was an error — "start is after end" — a rule about the
    encoding rather than about the conversation. The replacement lands where the
    earliest named node was, and the shadowed set comes back in surface order
    whichever way it was written.
    """
    session = _conversation()
    event = session.append(
        "user/message",
        user_payload("x", "m5"),
        SurfaceIntent(SurfaceReplace(replaces=(2, 1)), (1, 2)),
    )

    assert session.surface.nodes == (event.seq, 3)
    assert fold_surface(session.events).replacements[-1].shadowed_seqs == (1, 2)


def test_a_replace_must_name_something_and_name_it_once() -> None:
    """Refused at construction: an empty set shadows nothing while still taking a
    surface slot, and a repeat means the writer built the set by accident."""
    with pytest.raises(ValueError, match="at least one surface node"):
        SurfaceReplace(replaces=())
    with pytest.raises(ValueError, match="must not name a surface node twice"):
        SurfaceReplace(replaces=(1, 1))


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
        SurfaceIntent(SurfaceReplace(replaces=(0,)), (0,)),
    )
    assert session.surface.nodes == (1,)
    with pytest.raises(SurfaceError, match="may change only content"):
        session.append(
            "tool/result",
            tool_result_payload("preview…", "r2", call_id="OTHER"),
            SurfaceIntent(SurfaceReplace(replaces=(1,)), (1,)),
        )


def test_tool_result_replacement_targets_exactly_one_node() -> None:
    session = Session("s")
    session.append("tool/result", tool_result_payload("a", "r1"), SurfaceIntent("append"))
    session.append("tool/result", tool_result_payload("b", "r2", "c2"), SurfaceIntent("append"))
    with pytest.raises(SurfaceError, match="exactly one current node"):
        session.append(
            "tool/result",
            tool_result_payload("merged", "r3"),
            SurfaceIntent(SurfaceReplace(replaces=(0, 1)), (0, 1)),
        )


def test_fold_matches_the_incremental_manager() -> None:
    session = _conversation()
    session.append(
        "user/message",
        user_payload("summary", "m4"),
        SurfaceIntent(SurfaceReplace(replaces=(1, 2)), (1, 2)),
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
        SurfaceIntent(SurfaceReplace(replaces=(1, 2)), (1, 2)),
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
        SurfaceIntent(SurfaceReplace(replaces=(first.seq,)), (first.seq,)),
    )
    substitution = session.append(
        "user/message",
        user_payload("a summary of both", "m4"),
        SurfaceIntent(
            SurfaceReplace(replaces=(in_place.seq, second.seq)), (in_place.seq, second.seq)
        ),
    )

    assert is_in_place_rewrite(in_place)
    assert not is_in_place_rewrite(substitution)
    assert is_replacement_surface_event(substitution), "both are still replacements"
    assert not is_in_place_rewrite(first), "an append is not a rewrite"
