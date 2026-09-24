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

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ph.json import JsonObject, as_obj, as_seq, thaw_json
from ph.keys import FS, INTENTS, SESSION_PERSISTENCE, SESSIONS
from ph.llm.types import content_from_wire, text_of
from ph.persistence.repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    interrupted_turn_closers,
    repaired,
)
from ph.seams.approval import INTERRUPTED, pending_approvals
from ph.seams.user_questions import pending_questions
from ph.session import (
    Claim,
    IntentError,
    IntentKind,
    Session,
    SessionEvent,
    SurfaceIntent,
    SurfaceReplace,
    Unsettled,
    declare_intent,
)
from ph.testing import (
    MountProfile,
    assistant_payload,
    isolated_intent_kinds,
    tool_result_payload,
    user_payload,
)


def _assistant_with_call(call_id: str, *, turn: int = 1, step: int = 1) -> dict[str, Any]:
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


def _parked_turn(*, recorded_call: bool) -> Session:
    """A turn stopped on an unanswered approval.

    `recorded_call` is the one difference between a log written before P7-15,
    when `tool/call` preceded the gate, and one written after, when it does not.
    """
    session = _open_turn_with_unstarted_call()
    if recorded_call:
        session.append("tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "edit"})
    session.append("approval/asked", {"toolName": "edit", "callId": "c1"})
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
    assert as_obj(result.data["error"])["code"] == TOOL_NOT_STARTED
    text = as_obj(
        as_seq(as_obj(as_seq(as_obj(result.data["message"])["content"])[0])["content"])[0]
    )["text"]
    assert "before the Harness recorded it as started" in str(text)
    # Nothing ran, so it cites no call event.
    assert result.source_event_seqs is None


def test_a_recorded_call_is_closed_as_outcome_unknown() -> None:
    session = _open_turn_with_unstarted_call()
    call_seq = session.append(
        "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "edit", "arguments": "{}"}
    ).seq

    closers = interrupted_turn_closers(session.events)
    result = closers[0]
    assert as_obj(result.data["error"])["code"] == TOOL_OUTCOME_UNKNOWN
    text = as_obj(
        as_seq(as_obj(as_seq(as_obj(result.data["message"])["content"])[0])["content"])[0]
    )["text"]
    # The model is told to reason from the tool, not to retry blindly.
    assert "Do not retry blindly." in str(text)
    assert "read-only or idempotent" in str(text)
    assert result.source_event_seqs == (call_seq,)


def test_a_replaced_assistant_message_does_not_reopen_answered_calls() -> None:
    """A rewrite is not a new message, and repair used to read it as one.

    `compaction`'s argument truncation runs at `agent/pre-step` — inside an open
    turn, after `step/end` has cleared the pending set — and its replacement
    carries the same `tool-call` blocks as the message it rewrites. Registering
    those again re-opens calls whose results are already in the log, so the
    closers below synthesized a *second* `tool/result` for one of them.

    That is worse than a spurious event: a message carrying two results for one
    `tool_use` id is a log several providers reject outright, and repair has
    closed the turn by the time anything could notice.

    The step is closed before the rewrite lands, which is where the real one
    runs: this is the ordering that makes a replacement look like fresh work.
    """
    session = _open_turn_with_unstarted_call()
    assistant_seq = session.events[-1].seq
    call_seq = session.append(
        "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": "edit", "arguments": "{}"}
    ).seq
    session.append(
        "tool/result", tool_result_payload("done", "r1", "c1"), SurfaceIntent("append", (call_seq,))
    )
    session.append("step/end", {"turn": 1, "step": 1})
    session.append("step/start", {"turn": 1, "step": 2})
    # The elision: the same call, its arguments shortened, standing in place of
    # the message already on the surface.
    session.append(
        "assistant/message",
        _assistant_with_call("c1", step=1),
        SurfaceIntent(
            surface_op=SurfaceReplace(replaces=(assistant_seq,)),
            source_event_seqs=(assistant_seq,),
        ),
    )

    closers = interrupted_turn_closers(session.events)

    assert [event.type for event in closers] == ["step/end", "turn/end"], (
        "the rewrite re-opened a call the log had already answered"
    )


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


def test_a_turn_parked_on_a_human_settles_the_question_it_was_parked_on() -> None:
    """P5-13's resume half: the question is answered, not left hanging.

    **This test used to be a gate on a non-guarantee.** It asserted that repair
    closed a parked turn and left the ask open, and its docstring said that when
    the resume half landed it would fail, "which is the point". It landed; this
    is what replaced it.

    What was actually wrong: an `approval/asked` with no `approval/decided` is
    the pending state *by design* — the log is the truth, so a crash between the
    two cannot lose the question. But nothing ever wrote the second half. The
    fold outlived the process, so every future reader of a resumed log was told a
    decision was outstanding when nothing was waiting for one, and the transcript
    never said what became of the person's question.

    **What did not change, and must not.** The turn still closes `interrupted`
    and the tool result is still synthesized `TOOL_NOT_STARTED` — the model's
    `tool_use` block is unanswered, and a message carrying one with no matching
    `tool_result` is a log several providers reject outright. Settling the ask
    does not make it safe to leave the turn open; it makes the *question* stop
    being reported as live.

    The work resumes the way the harness resumes any interrupted work: the model
    reads "not started" and asks again, to whoever is attached by then.
    """
    session = _parked_turn(recorded_call=False)

    asked = [one.call_id for one in pending_approvals(session.events)]
    assert asked == ["c1"], "the ask is pending"

    closers = interrupted_turn_closers(session.events)

    assert [event.type for event in closers] == [
        "approval/decided",
        "tool/result",
        "step/end",
        "turn/end",
    ], "the question settles before the call it was gating"

    decided = closers[0]
    assert decided.data["outcome"] == INTERRUPTED, "and not `canceled`, which claims a person"
    assert decided.data["automatic"] is True, "nobody decided this"
    assert decided.data["callId"] == "c1" and decided.data["toolName"] == "edit"
    assert decided.surface_op is None and decided.source_event_seqs is None, (
        "`approval/decided` is not surface-eligible, so it may carry neither"
    )

    assert as_obj(closers[-1].data["reason"])["kind"] == "interrupted", (
        "the turn still reads as interrupted"
    )
    # Not started — because nothing was. The synthesized result is what keeps the
    # rebuilt log something a provider will accept.
    assert as_obj(closers[1].data["error"])["code"] == TOOL_NOT_STARTED
    assert as_obj(as_obj(closers[1].data["message"])["source"])["callId"] == "c1"


def test_the_repaired_log_reports_no_pending_approval() -> None:
    """The fold is the thing being fixed, so the fold is what the test reads.

    `pending_approvals` is the one reader that matters here: it is what a resume,
    a UI listing "what needs your attention", or an operator would ask. Asserting
    on the closers alone would pass while the fold still reported a ghost.
    """
    session = _parked_turn(recorded_call=False)
    assert pending_approvals(session.events), "pending before repair"

    repaired_session = Session("s2", seed=repaired(session.events))

    assert pending_approvals(repaired_session.events) == [], "and settled after it"


def test_repair_settles_every_parked_ask_not_only_the_first() -> None:
    """A parallel batch parks more than one question.

    `ToolRuntime` gates each call in a batch, so two calls awaiting a person is
    the ordinary shape of a crash mid-batch rather than an exotic one. A closer
    for the first would leave the rest in the fold forever — which is the bug
    this whole row is about, surviving in the narrower case.
    """
    session = _open_turn_with_unstarted_call()
    session.append("approval/asked", {"toolName": "edit", "callId": "c1"})
    session.append("approval/asked", {"toolName": "bash", "callId": "c2"})

    settled = [
        one for one in interrupted_turn_closers(session.events) if one.type == "approval/decided"
    ]

    assert [one.data["callId"] for one in settled] == ["c1", "c2"], "both, in the order asked"


def test_a_parked_user_question_is_settled_too() -> None:
    """The other dangling-ask fold, which DESIGN names in the same gap row.

    `pending_questions` is `pending_approvals` with different nouns — asked,
    never answered, log is the pending state — and it had the identical bug.
    Fixing only the approval half would have left `phern doctor` declaring the gap
    that remained while the one that closed went unmentioned, and a resumed log
    keeping a ghost question that "what needs your attention" lists forever.

    **`interrupted`, pointedly not `declined`.** `user_questions` opens by
    refusing to write `declined` for an exchange that never happened — "the log
    would then say a person was asked and declined, which is a different and
    false claim" — and a person the process never reached is exactly that case.
    """
    session = _open_turn_with_unstarted_call()
    session.append("question/asked", {"askId": "q1", "question": "which one?"})

    settled = [
        one for one in interrupted_turn_closers(session.events) if one.type == "question/answered"
    ]

    (answered,) = settled
    assert answered.data["askId"] == "q1"
    assert answered.data["interrupted"] is True
    assert "declined" not in answered.data, "nobody was there to decline"
    assert "answer" not in answered.data, "and nobody answered"

    resumed = Session("s2", seed=repaired(session.events))
    assert pending_questions(resumed.events) == [], "the fold stops reporting it"


def test_an_answered_question_is_not_settled_twice() -> None:
    """`pending_questions`' own pop rule, relied on rather than re-derived."""
    session = _open_turn_with_unstarted_call()
    session.append("question/asked", {"askId": "q1", "question": "which one?"})
    session.append("question/answered", {"askId": "q1", "answer": "this one"})

    closers = interrupted_turn_closers(session.events)

    assert not [one for one in closers if one.type == "question/answered"]


def test_an_answered_ask_is_not_settled_twice() -> None:
    """The half that stops this from being "append a decision on every resume".

    A question a person actually answered before the crash has its
    `approval/decided` already, and a second one would put two answers in the log
    for one ask — with the repair's, arriving later, the one a last-writer fold
    would believe.
    """
    session = _open_turn_with_unstarted_call()
    session.append("approval/asked", {"toolName": "edit", "callId": "c1"})
    session.append("approval/decided", {"toolName": "edit", "callId": "c1", "outcome": "rejected"})

    closers = interrupted_turn_closers(session.events)

    assert not [one for one in closers if one.type == "approval/decided"], "already answered"
    assert [one.type for one in closers] == ["tool/result", "step/end", "turn/end"]


def test_an_ask_with_no_call_id_settles_by_tool_name() -> None:
    """`pending_approvals` keys on `callId or toolName`, so repair must too.

    An approval can be asked without a call id — a row gating something that is
    not a tool call. Keying the closer differently from the fold would settle a
    question the fold was not reporting and leave the one it was.
    """
    session = _open_turn_with_unstarted_call()
    session.append("approval/asked", {"toolName": "deploy"})

    (settled,) = [
        one for one in interrupted_turn_closers(session.events) if one.type == "approval/decided"
    ]

    assert settled.data["toolName"] == "deploy"
    assert "callId" not in settled.data, "none was asked with, so none is invented"


def test_a_log_parked_before_p7_15_still_repairs_honestly() -> None:
    """Old logs have the old shape, and repair must read both.

    Before P7-15 a parked call had already written its `tool/call`, so a log on
    disk from then holds `tool/call` → `approval/asked` and stops. Repair cannot
    know from that log alone that the call never ran, and it does not pretend to:
    `TOOL_OUTCOME_UNKNOWN` is the honest word for a record with no result. The
    new shape gets the better answer; the old one keeps the safe one.
    """
    closers = interrupted_turn_closers(_parked_turn(recorded_call=True).events)
    (result,) = [one for one in closers if one.type == "tool/result"]

    assert as_obj(result.data["error"])["code"] == TOOL_OUTCOME_UNKNOWN
    assert result.source_event_seqs is not None, "and it cites the record it found"


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
async def test_resume_repairs_a_crashed_log_on_load(mount: MountProfile, tmp_path: Path) -> None:
    """The repair is on the load path, so nothing downstream sees an open turn."""
    from ph.persistence import resume_session

    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.require(SESSIONS).create("crashed")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("assistant/message", _assistant_with_call("c1"), SurfaceIntent("append", ()))
    await ctx.require(SESSIONS).flush(session)
    ctx.require(SESSIONS).dispose("crashed")

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


@pytest.mark.anyio
async def test_a_resumed_log_holds_no_question_nobody_can_answer(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The whole point of P5-13's resume half, through the path the daemon uses.

    The unit tests above prove `interrupted_turn_closers` *returns* the settling
    decision. That is not the same claim, and the difference is exactly the bug
    this row fixes: the fold existed and the closers existed and nothing wired
    them together, so a parked ask sat in every resumed log forever.

    So this drives the real `resume_session` against a real store. What it asserts
    is not the closer but the **fold** — `pending_approvals` on the revived
    session — because that is what a resume, a UI listing what needs attention, or
    an operator would actually ask.
    """
    from ph.persistence import resume_session

    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.require(SESSIONS).create("parked")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("assistant/message", _assistant_with_call("c1"), SurfaceIntent("append", ()))
    session.append("approval/asked", {"toolName": "edit", "callId": "c1"})
    assert pending_approvals(session.events), "parked before the crash"
    await ctx.require(SESSIONS).flush(session)
    ctx.require(SESSIONS).dispose("parked")

    revived = await resume_session(ctx, "parked")

    assert pending_approvals(revived.events) == [], "the question is settled, not still being asked"
    (decided,) = [one for one in revived.events if one.type == "approval/decided"]
    assert decided.data["outcome"] == INTERRUPTED
    assert decided.data["automatic"] is True

    # Durable, not just live: the next reader off disk must see it settled too,
    # or the fold comes back wrong on the resume after this one.
    await ctx.require(SESSIONS).flush(revived)
    _, stored = ctx.require(SESSION_PERSISTENCE).read("parked")
    assert pending_approvals(stored) == []


@pytest.mark.anyio
async def test_a_session_can_be_resumed_more_than_once(mount: MountProfile, tmp_path: Path) -> None:
    """The seam's wiring: `resume_session` tells the store what it already holds.

    A resume adds two events nobody wrote — the repair closers and the
    `session/end-seed` the constructor appends — and a backend that appends has
    no way to know they are owed. `resume_session` is the only place that knows,
    because it is the only place holding both the events it read and the log it
    built from them, so it states the boundary with `durable_length`.

    **Twice, because once passes either way.** The first resume creates the gap;
    only the second is refused by `_readmit`. Measured before this landed: a
    daemon root, which resumes on every start and every wake from passivation,
    survived exactly two lifetimes and then could not be started at all.

    The parity suite holds the backend half of this (both stores write what they
    are owed). This is the half that cannot live there: it exercises the real
    `resume_session`, and the parity tests set the boundary themselves.
    """
    from ph.persistence import resume_session

    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.require(SESSIONS).create("reopened")
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    session.append("assistant/message", _assistant_with_call("c1"), SurfaceIntent("append", ()))
    await ctx.require(SESSIONS).flush(session)
    ctx.require(SESSIONS).dispose("reopened")

    store = ctx.require(SESSION_PERSISTENCE)
    for reopen in (1, 2, 3):
        revived = await resume_session(ctx, "reopened")
        assert revived.durable_length > 0, f"reopen {reopen}: nothing was declared durable"
        await ctx.require(SESSIONS).flush(revived)
        _, stored = store.read("reopened")
        assert [event.seq for event in stored] == list(range(len(stored))), (
            f"reopen {reopen} left a hole in the seq space; the next resume would be refused"
        )
        ctx.require(SESSIONS).dispose("reopened")

    # The repair became durable on the way through, so the stored log no longer
    # reads as crashed — and a later reopen is a clean one.
    _, stored = store.read("reopened")
    assert not interrupted_turn_closers(stored), "the repair never reached the store"


# ------------------------------------------------------- declared intents --
# P10-07. Every declared kind's orphans, settled by the kind's own closer —
# whether or not a turn is open, since most of them happen between turns.


def _command_key(event: SessionEvent) -> str | None:
    value = event.data.get("id")
    return value if isinstance(value, str) else None


def _unknown(opened: SessionEvent, why: Unsettled) -> JsonObject:
    return {"id": opened.data["id"], "ok": False, "why": why}


COMMAND = IntentKind(
    # A pair no ph-core kind declares, so the test's closer is the only one.
    opened="command/run",
    settled="command/done",
    opened_key=_command_key,
    settled_key=_command_key,
    orphan="outcome-unknown",
    closer=_unknown,
    owner="tests",
)


@pytest.fixture
def kinds() -> Iterator[None]:
    """The intent registry, isolated so a test's kinds do not outlive it, with
    ph-core's own — the asks, the shell — still settled beside them."""
    with isolated_intent_kinds(core=True):
        yield


def _between_turns() -> Session:
    """A finished turn, then a command run outside any turn and never settled."""
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append("command/run", {"id": "x1", "command": "make"})
    return session


@pytest.mark.usefixtures("kinds")
def test_an_orphan_outside_any_turn_is_settled() -> None:
    """F13. The turn is balanced, so repair used to return `[]` and leave the
    command running forever in the eyes of every reader."""
    declare_intent(COMMAND)
    closers = interrupted_turn_closers(_between_turns().events)

    assert [(event.type, dict(event.data)) for event in closers] == [
        ("command/done", {"id": "x1", "ok": False, "why": "outcome-unknown"})
    ]


@pytest.mark.anyio
@pytest.mark.usefixtures("kinds")
async def test_an_orphan_outside_any_turn_is_settled_on_resume(
    mount: MountProfile, tmp_path: Path
) -> None:
    """End to end, and twice: the second resume finds nothing open and writes
    nothing, which is what keeps reopening a session from growing it."""
    from ph.persistence import resume_session

    declare_intent(COMMAND)
    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.require(SESSIONS).create("between")
    for event in _between_turns().events:
        session.append(event.type, thaw_json(event.data))
    await ctx.require(SESSIONS).flush(session)
    ctx.require(SESSIONS).dispose("between")

    revived = await resume_session(ctx, "between")
    assert [event.type for event in revived.events][-3:] == [
        "command/done",
        "session/end-seed",
        "session/resumed",
    ]
    assert revived.events[-1].data["closed"] == 1
    await ctx.require(SESSIONS).flush(revived)
    ctx.require(SESSIONS).dispose("between")

    again = await resume_session(ctx, "between")
    assert again.events[-1].data["closed"] == 0


@pytest.mark.usefixtures("kinds")
def test_an_owner_settles_kind_is_left_for_its_owner() -> None:
    """The owner can look — a tree is on disk or it is not — so a guess from
    repair would be worse than the owner's reconcile."""
    # With a closer, so the policy is what leaves it and not the absence of a
    # way to settle it.
    declare_intent(replace(COMMAND, orphan="owner-settles"))
    assert interrupted_turn_closers(_between_turns().events) == []


@pytest.mark.usefixtures("kinds")
def test_a_balanced_log_still_resumes_with_no_closers() -> None:
    declare_intent(COMMAND)
    session = _between_turns()
    session.append("command/done", {"id": "x1", "ok": True})
    assert interrupted_turn_closers(session.events) == []


@pytest.mark.usefixtures("kinds")
def test_closers_are_deterministic_and_backdated() -> None:
    """Seqs continue the log, the time is the last real event's, the same log
    repairs the same way twice — and inside a turn, intents settle after the
    asks and before the tool results, step and turn."""
    declare_intent(COMMAND)
    session = _parked_turn(recorded_call=True)
    session.append("command/run", {"id": "x1", "command": "make"})
    last = session.events[-1]

    closers = interrupted_turn_closers(session.events)

    assert closers == interrupted_turn_closers(session.events)
    assert [event.seq for event in closers] == list(
        range(last.seq + 1, last.seq + 1 + len(closers))
    )
    assert {event.time for event in closers} == {last.time}
    assert [event.type for event in closers] == [
        "approval/decided",
        "command/done",
        "tool/result",
        "step/end",
        "turn/end",
    ]


@pytest.mark.usefixtures("kinds")
def test_a_closer_that_does_not_settle_its_own_key_is_refused() -> None:
    """Otherwise the intent stays open and every resume writes another settle."""
    declare_intent(replace(COMMAND, closer=lambda opened, why: {"id": "someone-else"}))
    with pytest.raises(IntentError, match="does not settle 'x1'"):
        interrupted_turn_closers(_between_turns().events)


@pytest.mark.anyio
async def test_a_command_the_daemon_died_during_is_settled_on_resume(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P10-08, the real kind end to end: the journal puts the command on disk,
    the daemon dies before the result, and the next resume settles it as
    `outcome-unknown` — once."""
    from ph.persistence import resume_session
    from ph.seams.shell import SHELL_COMMAND

    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.require(SESSIONS).create("died")
    held = await ctx.require(INTENTS).open(
        session, SHELL_COMMAND, {"command": "make", "surface": False}
    )
    assert isinstance(held, Claim)
    # The daemon dies here: the command is on disk and its result never is.
    ctx.require(SESSIONS).dispose("died")

    revived = await resume_session(ctx, "died")
    result = revived.latest("shell/result")
    assert result is not None
    assert dict(result.data) == {
        "commandSeq": held.opened.seq,
        "ok": False,
        "interrupted": "outcome-unknown",
    }
    assert ctx.require(INTENTS).pending(revived, SHELL_COMMAND) == ()
    assert revived.events[-1].data["closed"] == 1


def test_repair_no_longer_knows_the_ask_shapes() -> None:
    """P10-09. The asks are kinds; their keying and their closers' words are the
    seams' to state, once. Repair imports the declaring seams and reads nothing
    of them — no type, no fold, no field — so a second spelling cannot return
    unnoticed."""
    import ast
    import inspect

    from ph.persistence import repair

    tree = ast.parse(inspect.getsource(repair))
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    shapes = ("approval/", "question/", "shell/", "tool/code-dispatch")
    code = [s for s in strings if s.startswith(shapes)]
    assert code == [], code
    declarers = {"approval", "shell", "user_questions", "code_mode"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & declarers, "a declaring module is read"
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in ("seams", "tools")
        for alias in node.names
    }
    assert imported == declarers, imported


def test_importing_repair_declares_every_core_kind() -> None:
    """The guarantee the declaring imports buy: a process that imported only
    repair — the trajectory viewer, a bare resume — settles every kind ph-core
    declares. Asked of a fresh interpreter, since this one has imported
    everything by now, and one that imports `ph.orphans` first: that order made
    a module-level declaring import a cycle, which no in-process test sees."""
    import ast
    import subprocess
    import sys

    import ph

    declared = set()
    for path in Path(ph.__path__[0]).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "IntentKind":
                for keyword in node.keywords:
                    if keyword.arg == "opened" and isinstance(keyword.value, ast.Constant):
                        declared.add(keyword.value.value)
    probe = (
        "import ph.orphans\n"
        "from ph.persistence.repair import interrupted_turn_closers\n"
        "from ph.session import Session, declared_intents\n"
        "log = Session('s')\n"
        "log.append('turn/start', {'turn': 1})\n"
        "interrupted_turn_closers(log.events)\n"
        "print('\\n'.join(sorted(kind.opened for kind in declared_intents())))\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    ).stdout.split()
    assert declared, "the walk found no declaration"
    assert set(found) == declared


# ------------------------------------------------------------- reconcile --
# P10-13. The tool is asked about its own started, unresolved call — mounted, on
# resume — and repair is handed the answer, so it stays a pure fold.


async def _crashed_write(
    mount: MountProfile, tmp_path: Path, session_id: str, *, tool: str = "write"
) -> tuple[Any, Path]:
    """A turn that recorded a `write` as started and died before its result."""
    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    session = ctx.require(SESSIONS).create(session_id)
    session.append("turn/start", {"turn": 1})
    session.append("step/start", {"turn": 1, "step": 1})
    arguments = json.dumps({"path": "notes.md", "content": "the plan"})
    call = {"type": "tool-call", "id": "c1", "name": tool, "arguments": arguments}
    session.append(
        "assistant/message",
        assistant_payload("", "m1", content=[call]),
        SurfaceIntent("append", ()),
    )
    session.append(
        "tool/call", {"turn": 1, "step": 1, "callId": "c1", "name": tool, "arguments": arguments}
    )
    await ctx.require(SESSIONS).flush(session)
    ctx.require(SESSIONS).dispose(session_id)
    return ctx, ctx.require(FS).root / "notes.md"


def _result(session: Session) -> Any:  # noqa: ANN401
    event = session.latest("tool/result")
    assert event is not None
    block = as_obj(as_seq(as_obj(event.data["message"])["content"])[0])
    return event, block, text_of(content_from_wire(block.get("content")))


@pytest.mark.anyio
async def test_a_write_that_landed_is_reported_done_after_a_crash(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The file holds exactly the call's bytes, so the write happened — and the
    model reads the result the tool renders, not `TOOL_OUTCOME_UNKNOWN`."""
    from ph.persistence import resume_session

    ctx, target = await _crashed_write(mount, tmp_path, "landed")
    target.write_text("the plan", encoding="utf-8")

    revived = await resume_session(ctx, "landed")
    event, block, text = _result(revived)

    assert block["isError"] is False
    assert text == "Wrote notes.md (8 bytes)"
    assert dict(as_obj(event.data["meta"])) == {"reconciled": True}
    assert "error" not in event.data


@pytest.mark.anyio
async def test_a_write_that_did_not_land_is_reported_not_started(
    mount: MountProfile, tmp_path: Path
) -> None:
    from ph.persistence import resume_session

    ctx, target = await _crashed_write(mount, tmp_path, "missed")
    assert not target.exists()

    revived = await resume_session(ctx, "missed")
    event, block, text = _result(revived)

    assert block["isError"] is True
    assert as_obj(event.data["error"])["code"] == TOOL_NOT_STARTED
    assert "it did not happen" in text


@pytest.mark.anyio
async def test_a_tool_that_cannot_answer_keeps_the_unknown_text(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`edit` declares no `reconcile`, so its crashed call reads as it always has."""
    from ph.persistence import resume_session

    ctx, _ = await _crashed_write(mount, tmp_path, "unanswered", tool="edit")

    revived = await resume_session(ctx, "unanswered")
    event, _block, _text = _result(revived)

    assert as_obj(event.data["error"])["code"] == TOOL_OUTCOME_UNKNOWN
    assert "meta" not in event.data


@pytest.mark.anyio
async def test_repair_is_still_a_pure_fold_over_a_stored_log(
    mount: MountProfile, tmp_path: Path
) -> None:
    """With no answers handed in — the trajectory viewer, `repaired()` — the same
    stored log repairs the same way, whatever is on disk, with nothing asked."""
    ctx, target = await _crashed_write(mount, tmp_path, "pure")
    target.write_text("the plan", encoding="utf-8")
    _, events = ctx.require(SESSION_PERSISTENCE).read("pure")

    closers = interrupted_turn_closers(events)

    assert closers == interrupted_turn_closers(events)
    (result,) = [one for one in closers if one.type == "tool/result"]
    assert as_obj(result.data["error"])["code"] == TOOL_OUTCOME_UNKNOWN
