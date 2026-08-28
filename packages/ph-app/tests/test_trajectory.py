"""The auditor's projection (P3-24).

Two gates carry this row, and they are the same two the transcript's tests
carry, one layer over:

* **no silent omissions** — every type in `KNOWN_SESSION_EVENT_TYPES` either
  produces a record or is named record-less, enumerated from the vocabulary
  rather than from a list somebody maintains; and
* **the projection equals its fold** — a stored log and a live one produce
  identical records, which is what makes this an auditor's instrument rather
  than a second story about what happened.

The rest is what the records have to carry to be worth reading: the prompt
snapshot *and* the one it replaced, the tool catalog as it was at call time,
timings derived from event times rather than measured at render, and a fork
point that is only ever a closed turn (A6).
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.cordis import Context
from ph.llm.types import PluginSource
from ph.persistence import read_session
from ph.persistence.jsonl import session_path
from ph.session import Session, SurfaceIntent, SurfaceReplace, is_fork_boundary
from ph.session.json import thaw_json
from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES
from ph.testing import assistant_payload, user_payload
from ph_app.tui.adapter import FORWARD_REFERENCES
from ph_app.tui.adapter import RECORDLESS as TRANSCRIPT_RECORDLESS
from ph_app.tui.trajectory import HANDLERS, RECORDLESS, TrajectoryRecord, build_trajectory

pytestmark = pytest.mark.anyio


def kinds(records: list[TrajectoryRecord]) -> list[str]:
    return [record.kind for record in records]


def by_kind(records: list[TrajectoryRecord], kind: str) -> list[TrajectoryRecord]:
    return [record for record in records if record.kind == kind]


# ---------------------------------------------------------------- the gate --


def test_every_known_event_type_produces_a_record_or_is_classified() -> None:
    """A11: no silent omissions, as a claim that can fail.

    The first version of this drove every type through the projection and
    checked whether a record appeared — which a catch-all `else` branch made a
    tautology: everything outside `RECORDLESS` produced one *because* the
    fallback produced one, so a type added to the vocabulary could never fail
    here. It is a set equality now, the shape `adapter.py` has used all along,
    and the dispatch table has no `else` for it to hide behind.
    """
    assert (set(HANDLERS) | RECORDLESS) - FORWARD_REFERENCES == KNOWN_SESSION_EVENT_TYPES, (
        "a known event type has neither a handler nor a record-less classification"
    )
    assert not set(HANDLERS) & RECORDLESS, "a type cannot both produce a record and not"
    # The same forward reference the transcript carries: Phase 4 adds the type
    # and the tool, and both projections are ready for it.
    assert set(HANDLERS) >= FORWARD_REFERENCES


def test_the_gate_fails_when_a_type_goes_unclassified() -> None:
    """The gate's own falsifiability, asserted rather than assumed.

    Written because the version this replaced could not fail, and nothing said
    so — the same way P3-23's diff triage could not fail. A gate that has never
    been shown to reject anything is a gate nobody has tested.
    """
    invented = KNOWN_SESSION_EVENT_TYPES | {"future/thing"}
    assert (set(HANDLERS) | RECORDLESS) - FORWARD_REFERENCES != invented


def test_recordless_is_a_subset_of_the_vocabulary() -> None:
    """A name in `RECORDLESS` that no producer emits is a stale exemption."""
    assert RECORDLESS <= KNOWN_SESSION_EVENT_TYPES


def test_the_auditor_renders_what_the_transcript_does_not() -> None:
    """The types P3-24 exists for.

    Record-less in the *conversation* view on purpose — they are not transcript
    defects — and rendering them is the whole reason this projection is
    separate. Stated as the four names rather than as a set difference with
    `RECORDLESS` subtracted from both sides, which cancelled two of them and
    read as six.
    """
    assert {
        "request/header",
        "approval/policy",
        "fs/observed",
        "session/end-seed",
    } == TRANSCRIPT_RECORDLESS - RECORDLESS
    # And the reverse: what this view skips that the transcript renders. The two
    # sets are the same size, so "smaller" was never the relationship.
    assert {
        "assistant/chunk",
        "tool/code-dispatch-start",
        "subagent/status",
        "subagent/usage-attributed",
    } == RECORDLESS - TRANSCRIPT_RECORDLESS


# ------------------------------------------------------------- the records --


def _conversation() -> Session:
    """One turn: a header, a user message, a step, an assistant reply, a tool."""
    session = Session("trajectory")
    session.append("turn/start", {"turn": 1})
    session.append(
        "request/header",
        {
            "header": {
                "config": {"provider": "fake", "model": "fake-1"},
                "system": "You are pH.",
                "tools": [{"name": "read", "description": "read a file", "parameters": {}}],
            }
        },
    )
    session.append("user/message", user_payload("what is in a.py?"), SurfaceIntent("append"))
    session.append("step/start", {"turn": 1, "step": 0})
    session.append("assistant/chunk", {"turn": 1, "step": 0, "delta": "look"})
    session.append(
        "assistant/message",
        {
            **assistant_payload("looking now", "a1"),
            "usage": {"inputTokens": 10, "outputTokens": 20},
        },
        SurfaceIntent("append"),
    )
    session.append("tool/call", {"callId": "c1", "name": "read", "arguments": '{"path": "a.py"}'})
    session.append("step/end", {"turn": 1, "step": 0})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    return session


def test_the_record_set_is_dshs_closed_vocabulary() -> None:
    session = _conversation()
    records = build_trajectory(session)

    assert kinds(records) == [
        "event",  # turn/start
        "system",  # request/header
        "user",
        "message",  # assistant
        "tool",
        "event",  # turn/end
    ]
    # 1-based `#N`, contiguous, and each pointing back at the event it projects
    # — the join the two views cross-navigate by.
    assert [record.index for record in records] == list(range(1, len(records) + 1))
    for record in records:
        assert session.events[record.source_seq].seq == record.source_seq


def test_a_system_record_carries_the_snapshot_and_the_one_it_replaced() -> None:
    """dsh's own requirement: a prompt change has to read as a diff.

    `request/header` is logged only when it changed (A12), so every one of these
    is a real change — and the previous text is what makes it legible.
    """
    session = _conversation()
    session.append(
        "request/header",
        {"header": {"config": {"provider": "fake", "model": "fake-1"}, "system": "You are pH v2."}},
    )
    first, second = by_kind(build_trajectory(session), "system")

    assert first.detail == "You are pH."
    assert first.replaced == "", "the first snapshot replaced nothing"
    assert second.detail == "You are pH v2."
    assert second.replaced == "You are pH.", "the diff's other half is missing"


def test_a_system_record_carries_the_catalog_as_it_was_at_call_time() -> None:
    """Not as it is now: a tool registered later must not appear in the record
    of a call that could not have used it."""
    (record,) = by_kind(build_trajectory(_conversation()), "system")

    assert record.tools == ["read"]
    assert "1 tool(s)" in record.summary


def test_timings_are_derived_from_the_log_not_the_clock() -> None:
    """A11: the same log has to yield the same numbers on replay as live, so a
    clock read at render time would be wrong by construction."""
    (message,) = by_kind(build_trajectory(_conversation()), "message")

    assert message.timing is not None
    assert message.timing.output_tokens == 20
    assert message.timing.total_ms is not None and message.timing.total_ms >= 0
    # The step had a chunk before its message, so time-to-first-token is known.
    assert message.timing.time_to_first_token_ms is not None


def test_a_timing_says_nothing_rather_than_guessing() -> None:
    """A message outside a step has no timings to report, and reports none."""
    session = Session("no-step")
    session.append("assistant/message", assistant_payload("hi", "a1"), SurfaceIntent("append"))
    (record,) = build_trajectory(session)

    assert record.timing is not None
    assert record.timing.total_ms is None
    assert record.timing.decode_tokens_per_second is None


def test_a_plugin_message_is_context_attributed_to_its_producer() -> None:
    """ "Inspect these records by source" is what `PluginSource` is for — the
    producer and the *form*, so an auditor can ask for every snapshot."""
    session = Session("context")
    session.append(
        "user/message",
        {
            **user_payload("# Loaded context"),
            "source": PluginSource(
                plugin="rlm-context-loader", form="snapshot", sections=[]
            ).to_wire(),
        },
        SurfaceIntent("append"),
    )
    (record,) = build_trajectory(session)

    assert record.kind == "context", "a plugin's injection is not the user speaking"
    assert record.source.name == "rlm-context-loader"
    assert record.source.form == "snapshot"
    assert "rlm-context-loader" in record.title


def test_a_compaction_is_its_own_kind() -> None:
    """A summary that shadows a range is not an ordinary message that appeared."""
    session = Session("compacted")
    first = session.append("user/message", user_payload("original", "m1"), SurfaceIntent("append"))
    session.append(
        "user/message",
        user_payload("(summary of earlier)", "m2"),
        SurfaceIntent(SurfaceReplace(start=0, end=0), (first.seq,)),
    )
    assert kinds(build_trajectory(session)) == ["user", "compacted"]


def test_a_code_mode_sub_dispatch_is_a_subtool() -> None:
    """C2's records, in the view whose job is showing them: one cell, many calls."""
    session = Session("subtool")
    session.append("tool/call", {"callId": "c1", "name": "ipython", "arguments": "{}"})
    for index in range(3):
        session.append(
            "tool/code-dispatch",
            {"parentCallId": "c1", "subCallId": f"s{index}", "name": "read", "content": []},
        )
    records = build_trajectory(session)

    assert kinds(records) == ["tool", "subtool", "subtool", "subtool"]


def test_an_unrecognized_harness_event_still_gets_a_row() -> None:
    """A view that hid what it could not name would be the silent omission A11
    forbids — so an event with no phrase for it renders from its payload."""
    session = Session("unknown")
    session.append("fs/observed", {"path": "/tmp/a.py"})
    (record,) = build_trajectory(session)

    assert record.kind == "event"
    assert "/tmp/a.py" in record.summary


# --------------------------------------------------------------- the fork --


def test_fork_points_are_exactly_what_the_store_would_accept() -> None:
    """A6 is ph-core's rule, and this view marks rows against it.

    The first version marked `turn/end` only — one legal boundary in four — and
    told the reader the other three were refused "(A6)", a claim the layer that
    owns A6 does not make. Now the marks *are* `is_fork_boundary`, so the table
    cannot advertise a target the store rejects or hide one it accepts.
    """
    session = _conversation()
    # A between-turn event *after* a closed turn: legal to fork at, and the
    # kind of row the `turn/end`-only rule refused.
    session.append("fs/observed", {"path": "a.py"})
    records = build_trajectory(session)

    for record in records:
        assert record.fork_point == is_fork_boundary(session.events, record.source_seq), (
            f"#{record.index} ({record.title}) disagrees with the store"
        )
    # Not just `turn/end`: the between-turn events after a closed turn are legal
    # boundaries too, and the seed record is the one an auditor most wants.
    assert sum(1 for record in records if record.fork_point) > 1


def test_a_record_inside_an_open_turn_is_not_a_fork_point() -> None:
    """The rule's actual content: a turn that has not closed cannot be cut."""
    session = Session("open")
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user_payload("mid-turn"), SurfaceIntent("append"))
    records = build_trajectory(session)

    assert [record.fork_point for record in records] == [False, False]


# --------------------------------------------------------------- the fold --


async def test_a_stored_log_and_a_live_one_project_identically(mount: Any, tmp_path: Any) -> None:
    """The P2-01 gate, for the auditor's view.

    The point of the whole projection: it is derived from the log and nothing
    else, so reading a session off disk with **nothing mounted** — no agent, no
    provider, no answerers — gives the records the live session gave. That is
    what P3-25's harness-free entry point stands on.
    """
    ctx: Context = await mount()
    live = ctx.sessions.create("round-trip")
    for event in _conversation().events:
        # `thaw_json`, because a logged payload is frozen — its lists are tuples,
        # which the lossless-JSON guard refuses on the way back in.
        live.append(
            event.type,
            thaw_json(event.data),
            SurfaceIntent("append") if event.surface_op else None,
        )
    await ctx.sessions.flush(live)

    header, events = read_session(session_path(ctx.session_persistence.root, live.id))
    stored = Session(live.id, seed=events, header=header)

    replayed = build_trajectory(stored)
    # A seeded session appends `session/end-seed` to mark where its history
    # ended and this run began — a real event the live session never had, and
    # one this view renders precisely because an auditor wants to see the
    # boundary. So the claim is that the *shared* events project identically,
    # not that the two logs are the same log.
    boundary = [record for record in replayed if record.title == "session/end-seed"]
    assert len(boundary) == 1, "a resumed session should record its own seed boundary"
    shared = [record for record in replayed if record.title != "session/end-seed"]

    live_records = build_trajectory(live)
    assert [record.index for record in shared] == [record.index for record in live_records]
    for replayed_record, live_record in zip(shared, live_records, strict=True):
        assert replayed_record == live_record
