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

## Why `ph_app.wire` is not under `ph_app.tui`

`ph_app.tui.__init__` imports the Textual app, so a module under it cannot be read
from without paying **278 ms** of terminal framework import — which a headless
`phern agents attach` should never do to render a line of a log it just received.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ph.cordis import Context
from ph.json import thaw_json
from ph.keys import SESSIONS
from ph.llm.types import PluginSource
from ph.persistence import read_session
from ph.session import Session, SurfaceIntent, SurfaceReplace, is_fork_boundary
from ph.session.known_event_types import (
    KNOWN_SESSION_EVENT_TYPES,
    declare_log_type,
    is_audit_only,
    known_log_types,
)
from ph.testing import (
    MountProfile,
    assistant_payload,
    isolated_log_types,
    log_event,
    store_root,
    stored_log,
    user_payload,
)
from ph_app.tui.adapter import RECORDLESS as TRANSCRIPT_RECORDLESS
from ph_app.tui.trajectory import (
    HANDLERS,
    RECORDLESS,
    TrajectoryRecord,
    _on_harness_event,
    build_trajectory,
)
from ph_app.wire import describe
from ph_clm.kinds import DECLINED, REVISED

pytestmark = pytest.mark.anyio


def kinds(records: list[TrajectoryRecord]) -> list[str]:
    return [record.kind for record in records]


def by_kind(records: list[TrajectoryRecord], kind: str) -> list[TrajectoryRecord]:
    return [record for record in records if record.kind == kind]


# ---------------------------------------------------------------- the gate --


def _audit_only() -> set[str]:
    return {kind for kind in known_log_types() if is_audit_only(kind)}


def test_every_known_event_type_produces_a_record_or_is_classified() -> None:
    """A11: no silent omissions, as a claim that can fail.

    The first version of this drove every type through the projection and
    checked whether a record appeared — which a catch-all `else` branch made a
    tautology: everything outside `RECORDLESS` produced one *because* the
    fallback produced one, so a type added to the vocabulary could never fail
    here. It is a set equality now, the shape `adapter.py` has used all along.
    The auditor's records the vocabulary declares render generically without an
    entry, which is a decision made where each type is declared — so a type that
    is neither listed nor declared still fails.
    """
    vocabulary = known_log_types()
    assert {REVISED, DECLINED} <= vocabulary, "ph-clm's declarations did not run"
    assert set(HANDLERS) | RECORDLESS | _audit_only() == vocabulary, (
        "a known event type has neither a handler nor a record-less classification"
    )
    assert not set(HANDLERS) & RECORDLESS, "a type cannot both produce a record and not"
    assert not RECORDLESS & _audit_only(), "an auditor's record cannot be record-less"


def test_an_auditors_record_needs_no_entry_of_its_own() -> None:
    """The table names a type an auditor's record only to give it a handler of its
    own; a generic entry would be the declaration said twice."""
    generic = {kind for kind, handler in HANDLERS.items() if handler is _on_harness_event}
    assert not generic & _audit_only()


def test_the_gate_fails_when_a_type_goes_unclassified() -> None:
    """The gate's own falsifiability, asserted rather than assumed.

    Written because the version this replaced could not fail, and nothing said
    so — the same way P3-23's diff triage could not fail. A gate that has never
    been shown to reject anything is a gate nobody has tested.
    """
    invented = known_log_types() | {"future/thing"}
    assert set(HANDLERS) | RECORDLESS | _audit_only() != invented


def test_recordless_is_a_subset_of_the_vocabulary() -> None:
    """A name in `RECORDLESS` that no producer emits is a stale exemption."""
    assert RECORDLESS <= KNOWN_SESSION_EVENT_TYPES


def test_the_auditor_renders_what_the_transcript_does_not() -> None:
    """The types P3-24 exists for: every record only an auditor reads is a record
    here, ph-clm's included, drawn from its payload when the view has no handler of
    its own for it. `request/header` and the profile records have their own.

    And the reverse: what this view skips that the transcript renders — a stream's
    chunks and a dispatch's opening half, which another record already carries.
    """
    unrendered = []
    for kind in sorted(_audit_only() - set(HANDLERS)):
        session = Session("audit")
        log_event(session, kind, {})
        records = build_trajectory(session)
        if [(record.kind, record.type) for record in records] != [("event", kind)]:
            unrendered.append(kind)
    assert not unrendered, unrendered
    assert {"assistant/chunk", "tool/code-dispatch-start"} == RECORDLESS - TRANSCRIPT_RECORDLESS
    assert TRANSCRIPT_RECORDLESS <= RECORDLESS


def test_a_type_this_process_does_not_know_is_rendered_from_its_payload() -> None:
    """Another build's type, or a package's this process never imported — a stored log
    read with nothing mounted. It reached the viewer because it is ignorable, and it is
    shown rather than skipped, with no bundle imported to know it.

    Sabotage: drop `not is_known(event_type)` from `_handler_of`.
    """
    with isolated_log_types():
        declare_log_type("sample/note", owner="sample.plugin", ignorable=True, audit_only=False)
        session = Session("elsewhere")
        log_event(session, "sample/note", {"n": 1})
    stored = Session("elsewhere", seed=list(session.events))

    (record,) = build_trajectory(stored)

    assert (record.kind, record.type) == ("event", "sample/note")


# ------------------------------------------------------------- the records --


def _conversation() -> Session:
    """One turn: a header, a user message, a step, an assistant reply, a tool."""
    session = Session("trajectory")
    log_event(session, "turn/start", {"turn": 1})
    log_event(
        session,
        "request/header",
        {
            "header": {
                "config": {"provider": "fake", "model": "fake-1"},
                "system": "You are pH.",
                "tools": [{"name": "read", "description": "read a file", "parameters": {}}],
            }
        },
    )
    log_event(session, "user/message", user_payload("what is in a.py?"), SurfaceIntent("append"))
    log_event(session, "step/start", {"turn": 1, "step": 0})
    log_event(session, "assistant/chunk", {"turn": 1, "step": 0, "delta": "look"})
    log_event(
        session,
        "assistant/message",
        {
            **assistant_payload("looking now", "a1"),
            "usage": {"inputTokens": 10, "outputTokens": 20},
        },
        SurfaceIntent("append"),
    )
    log_event(
        session, "tool/call", {"callId": "c1", "name": "read", "arguments": '{"path": "a.py"}'}
    )
    log_event(session, "step/end", {"turn": 1, "step": 0})
    log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    return session


def test_a_childs_own_log_tells_its_whole_story() -> None:
    """A sub-agent's log read here says what it was asked, each start and ending,
    and its revocation — every `subagent/*` record, and nothing about it from
    anywhere else (Phase 11).

    The trajectory view is where a child is read: its log is never attached as a
    root (P11-08), and a root's transcript never meets these records. Its status
    changes were a panel's concern while they lived in the parent's log; in the
    child's own, they are the audit.

    Sabotage: classify `subagent/status` as record-less again, and the child's
    ending is missing from its own audit.
    """
    session = Session("lead-r1")
    log_event(session, "subagent/admitted", {"runId": "r1", "name": "scout", "model": "fake-1"})
    log_event(session, "subagent/status", {"status": "running"})
    log_event(session, "subagent/status", {"status": "error", "detail": "boom"})
    log_event(session, "subagent/deleted", {"reason": "user"})

    records = build_trajectory(session)

    assert [record.type for record in records] == [
        "subagent/admitted",
        "subagent/status",
        "subagent/status",
        "subagent/deleted",
    ]
    assert "boom" in records[2].summary


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
    log_event(
        session,
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
    log_event(session, "assistant/message", assistant_payload("hi", "a1"), SurfaceIntent("append"))
    (record,) = build_trajectory(session)

    assert record.timing is not None
    assert record.timing.total_ms is None
    assert record.timing.decode_tokens_per_second is None


def test_a_plugin_message_is_context_attributed_to_its_producer() -> None:
    """ "Inspect these records by source" is what `PluginSource` is for — the
    producer and the *form*, so an auditor can ask for every snapshot."""
    session = Session("context")
    log_event(
        session,
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
    first = log_event(
        session, "user/message", user_payload("original", "m1"), SurfaceIntent("append")
    )
    log_event(
        session,
        "user/message",
        user_payload("(summary of earlier)", "m2"),
        SurfaceIntent(SurfaceReplace(replaces=(first.seq,)), (first.seq,)),
    )
    assert kinds(build_trajectory(session)) == ["user", "compacted"]


def test_the_generic_reading_never_emits_a_python_repr() -> None:
    """The fallback must degrade to *less informative*, never to garbage.

    Fifty-six event types reach `_describe` rather than a phrase of their own,
    and `phern agents attach` reads the same sentences. So the failure mode of
    "nobody wrote a dedicated handler" has to be a thinner line, not a Python
    literal in a document a person reads — which is what `f"{key}={value}"` gave
    for every payload with structure in it.

    `agent/inbox/spliced` is the worst real case and the reason this is a gate
    rather than a tidy-up: its payload carries whole `Message` objects, so the
    auditor's one-line summary was a uuid, a role and nested content blocks,
    truncated mid-token.

    Sabotage: render a value with `str()` and the literal markers below appear.
    """
    payloads = [
        {"target": "next-step", "inserted": [{"id": "m1", "role": "user", "content": [{}]}]},
        {"todos": [{"content": "fix it", "status": "pending"}]},
        {"limit": "turns", "spent": {"turns": 9}, "cap": 8},
        {"a": {"b": {"c": 1, "d": 2}}},
    ]
    for payload in payloads:
        line = describe(payload)
        assert "{'" not in line and "[{" not in line, f"a Python literal reached a reader: {line}"
        assert "': " not in line, f"a dict repr reached a reader: {line}"


def test_the_generic_reading_keeps_the_facts_that_fit() -> None:
    """Bounded, but not so bounded it stops answering anything.

    A mapping is expanded one level because that is where the readable facts
    usually are — `spent={turns=9}` is what somebody wanted to know — while a
    list is counted, since a list is never one-line material. Named rather than
    dropped, which is `block_marker`'s rule one layer up: a reader has to be able
    to see that something was there.
    """
    assert describe({"limit": "turns", "spent": {"turns": 9}, "cap": 8}) == (
        "limit=turns, spent={turns=9}, cap=8"
    )
    assert describe({"todos": [1, 2, 3]}) == "todos=[3 items]"
    assert describe({"todos": [1]}) == "todos=[1 item]", "and it agrees with itself on one"
    assert describe({"a": {"b": {"c": 1, "d": 2}}}) == "a={b={2 fields}}"
    assert describe({"turn": 1, "reason": "completed"}) == "turn=1, reason=completed"


def test_both_invariant_transitions_read_as_themselves() -> None:
    """One event type, two opposite facts, and the generic reading carries neither.

    `verify_root` records a clearing as `supervisor/violated` with an empty
    list, so the fallback renderer printed the good news as `violations=[],
    pid=9` — nothing a reader could recognize as reassurance — and the bad news
    as a Python dict repr in a document a person reads.

    **This is the second renderer of that record**, which is why it needed its
    own entry rather than the fallback: `ph_app.tui.adapter` draws the same two
    transitions for the terminal, and a transcript that disagrees with the
    session it transcribes is the failure `ph.text` exists to prevent.
    """
    session = Session("invariants")
    log_event(
        session,
        "supervisor/violated",
        {
            "violations": [
                {"invariant": "session-log", "detail": "derive_messages holds 0 where 1"},
                {"invariant": "tools-view", "detail": "differs from a rebuild"},
            ],
            "pid": 9,
        },
    )
    log_event(session, "supervisor/violated", {"violations": [], "pid": 9})

    broke, cleared = by_kind(build_trajectory(session), "event")

    assert "2 invariants violated" in broke.summary
    assert "session-log" in broke.summary and "tools-view" in broke.summary
    assert "derive_messages holds 0 where 1" in broke.detail, "the counts an auditor came for"
    assert "violations=" not in broke.summary, "never the raw payload"

    assert cleared.summary == "hold again"
    assert "violated" not in cleared.summary, "the clearing must not read as a violation"


def test_a_code_mode_sub_dispatch_is_a_subtool() -> None:
    """C2's records, in the view whose job is showing them: one cell, many calls."""
    session = Session("subtool")
    log_event(session, "tool/call", {"callId": "c1", "name": "ipython", "arguments": "{}"})
    for index in range(3):
        log_event(
            session,
            "tool/code-dispatch",
            {"parentCallId": "c1", "subCallId": f"s{index}", "name": "read", "content": []},
        )
    records = build_trajectory(session)

    assert kinds(records) == ["tool", "subtool", "subtool", "subtool"]


def test_an_unrecognized_harness_event_still_gets_a_row() -> None:
    """A view that hid what it could not name would be the silent omission A11
    forbids — so an event with no phrase for it renders from its payload."""
    session = Session("unknown")
    log_event(session, "fs/observed", {"path": "/tmp/a.py"})
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
    log_event(session, "fs/observed", {"path": "a.py"})
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
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "user/message", user_payload("mid-turn"), SurfaceIntent("append"))
    records = build_trajectory(session)

    assert [record.fork_point for record in records] == [False, False]


# --------------------------------------------------------------- the fold --


async def test_a_stored_log_and_a_live_one_project_identically(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The P2-01 gate, for the auditor's view.

    The point of the whole projection: it is derived from the log and nothing
    else, so reading a session off disk with **nothing mounted** — no agent, no
    provider, no answerers — gives the records the live session gave. That is
    what P3-25's harness-free entry point stands on.
    """
    ctx: Context = await mount()
    live = ctx.require(SESSIONS).create("round-trip")
    for event in _conversation().events:
        # `thaw_json`, because a logged payload is frozen — its lists are tuples,
        # which the lossless-JSON guard refuses on the way back in.
        log_event(
            live,
            event.type,
            thaw_json(event.data),
            SurfaceIntent("append") if event.surface_op else None,
        )
    await ctx.require(SESSIONS).flush(live)

    header, events = read_session(stored_log(store_root(ctx), live.id))
    stored = Session(live.id, seed=events, header=header)

    # The same records, and no more: a log opened to be viewed is the log on disk.
    assert build_trajectory(stored) == build_trajectory(live)
