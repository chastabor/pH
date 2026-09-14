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

from pathlib import Path
from typing import Any

import pytest

from ph.json import as_obj, as_seq
from ph.keys import SESSION_PERSISTENCE, SESSIONS
from ph.persistence.repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    interrupted_turn_closers,
    repaired,
)
from ph.seams.approval import INTERRUPTED, pending_approvals
from ph.seams.user_questions import pending_questions
from ph.session import Session, SurfaceIntent
from ph.testing import MountProfile, assistant_payload, user_payload


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
    assert decided.data["outcome"] == INTERRUPTED, "and not `cancelled`, which claims a person"
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
    Fixing only the approval half would have left `ph doctor` declaring the gap
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
