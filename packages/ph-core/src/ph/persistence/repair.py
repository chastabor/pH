"""Crash repair: closing a turn the process died inside (P1-12, A5).

A log whose last turn is half-written is not just untidy — it is *invalid* to a
provider. An assistant message carrying a tool call with no matching result is
rejected outright by several APIs, so a session that crashed mid-batch could
never be resumed at all without this.

The synthesized closers are deterministic: sequences continue the log, and the
timestamp is the **last real event's**, never `now()`. A repair that invented a
future time would make the log say the recovery happened during the crash.

Two failure shapes, and the difference is the whole point of having two codes:

* `TOOL_NOT_STARTED` — the assistant asked, but no `tool/call` was ever
  recorded. Nothing ran, so retrying is safe and the text says so.
* `TOOL_OUTCOME_UNKNOWN` — the call *was* recorded, and then the process died.
  It may have completed. The text tells the model to reason from the tool's own
  semantics rather than retry blindly, because a blind retry of a non-idempotent
  operation is how one crash becomes two side effects.

**Dangling asks, settled here for the same reason** (P5-13). An `approval/asked`
with no `approval/decided` — and a `question/asked` with no `question/answered` —
is a question put to a person that the process stopped existing before
answering. Without a closer those pairs stay half-written *forever*: every future
reader of a resumed log is told a decision is outstanding when nothing is waiting
for one, and the transcript never says what became of the person's question.

**Through the seams' own folds, never a copy of their keying.** `pending_approvals`
and `pending_questions` take a `Sequence[SessionEvent]` precisely so this module
can call them: repair writes the record that makes an ask stop being pending, so
a second spelling of "what counts as pending" here is a second spelling of what
repair must settle — and the two would drift silently, in the one direction
nothing fails. Both seams are imported rather than dispatched through a registry:
there are two ask-shaped pairs in the vocabulary, and a registry for two is
indirection with no second reader.

The turn is still closed `interrupted` and the tool result is still synthesized
`TOOL_NOT_STARTED` — nothing about that changes, and it is what keeps the
rebuilt log something a provider will accept. What changes is that the question
is now *settled* rather than left hanging. The work resumes the way the harness
resumes any interrupted work: the model reads "not started" and asks again, to
whoever is attached then.

Ported from dsh `packages/core/session/src/repair.ts`, message texts included:
this vocabulary is what a resumed model reads, and paraphrasing it would change
behavior that was tuned deliberately.

@module ph.persistence.repair
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..json import as_int, as_obj, as_seq, as_str
from ..seams.approval import INTERRUPTED, pending_approvals
from ..seams.user_questions import pending_questions
from ..session import SessionEvent, is_in_place_rewrite
from ..session.json import freeze_json_value

__all__ = [
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "interrupted_turn_closers",
]

TOOL_NOT_STARTED = "TOOL_NOT_STARTED"
"""An assistant tool request that never reached a recorded call start."""

TOOL_OUTCOME_UNKNOWN = "TOOL_OUTCOME_UNKNOWN"
"""A recorded call whose completed outcome was never durably recorded."""

_NOT_STARTED_TEXT = (
    "The tool call was interrupted before the Harness recorded it as started. "
    "Retry it if it is still needed."
)

_OUTCOME_UNKNOWN_TEXT = (
    "The tool call was interrupted after it was recorded, but no result was durably "
    "recorded. Its outcome is unknown. Decide whether to retry from the tool "
    "semantics: retry only if the operation is read-only or idempotent; if it may "
    "have side effects, first verify external state or ask the user. Do not retry "
    "blindly."
)


def _settled_asks(events: Sequence[SessionEvent]) -> list[dict[str, Any]]:
    """The closers for every ask this log put to a person and never answered.

    Both vocabularies in one place because they are one rule wearing two
    spellings: asked, never answered, process gone. Each seam owns *what counts
    as pending* — this owns only what settling it looks like.

    `automatic` on the approval is the same flag a policy of `never` sets: the
    field separating a decision no person made from one somebody did, which is
    the whole question a reader has here. The question's closer says
    `interrupted` and pointedly **not** `declined`: `user_questions` is explicit
    that declined means "somebody was there and declined", and claiming that of a
    person who was never reached is the false statement that module opens by
    refusing to make.
    """
    settled: list[dict[str, Any]] = [
        {
            "type": "approval/decided",
            "data": freeze_json_value(
                {
                    "toolName": one.tool_name,
                    **({"callId": one.call_id} if one.call_id is not None else {}),
                    "outcome": INTERRUPTED,
                    "automatic": True,
                }
            ),
        }
        for one in pending_approvals(events)
    ]
    settled.extend(
        {
            "type": "question/answered",
            "data": freeze_json_value({"askId": one.ask_id, "interrupted": True}),
        }
        for one in pending_questions(events)
    )
    return settled


@dataclass(slots=True)
class _Pending:
    step: int
    call_seq: int | None = None


def interrupted_turn_closers(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """The synthetic events that close an open tail turn, in order.

    Returns `[]` for a balanced log, so a clean resume appends nothing and
    reopening a session does not grow it.
    """
    open_turn: int | None = None
    open_step: int | None = None
    pending: dict[str, _Pending] = {}

    for event in events:
        if event.type == "turn/start":
            open_turn = as_int(event.data.get("turn"))
            open_step = None
            pending.clear()
        elif event.type == "turn/end":
            open_turn = None
            open_step = None
            pending.clear()
        elif event.type == "step/start":
            open_step = as_int(event.data.get("step"))
        elif event.type == "step/end":
            pending.clear()
            open_step = None
        elif event.type == "assistant/message":
            if is_in_place_rewrite(event):
                # A near-copy of a message already in the log — argument
                # truncation is the one that does this — so its `tool-call`
                # blocks are answered behind it and registering them again would
                # have this fold write a second `tool/result` for one id.
                #
                # `is_in_place_rewrite` rather than `is_replacement_surface_event`:
                # the narrower predicate is the one that means "not new work". A
                # substitution putting a genuinely new assistant message in place
                # of a range would carry calls that *do* need closing, and the
                # coarse test would skip those too.
                continue
            content = as_seq(as_obj(event.data.get("message")).get("content"))
            for block in (as_obj(one) for one in content):
                if block.get("type") == "tool-call":
                    pending[as_str(block.get("id"))] = _Pending(step=as_int(event.data.get("step")))
        elif event.type == "tool/call":
            entry = pending.get(as_str(event.data.get("callId")))
            if entry is not None:
                entry.call_seq = event.seq
        elif event.type == "tool/result":
            source = as_obj(as_obj(event.data.get("message")).get("source"))
            pending.pop(as_str(source.get("callId")), None)

    if open_turn is None or not events:
        return []

    last = events[-1]
    next_seq = last.seq + 1
    time = last.time
    closers: list[SessionEvent] = []

    # Asks settle before the call they were gating. Nothing downstream depends on
    # the order — neither closer is surface-eligible, so neither derives a message
    # or moves the provider-facing sequence — but the log reads in the order
    # things happened, and the ask came first.
    #
    # No `source_event_seqs` back to the ask on either, tempting as the symmetry
    # with the tool closer below is: `SURFACE_EVENT_TYPES` permits that field only
    # on the three types that carry a `surfaceOp`, and neither of these is one.
    for settled in _settled_asks(events):
        closers.append(SessionEvent(seq=next_seq, time=time, **settled))
        next_seq += 1

    # Calls close before their step: a provider rejects a dangling assistant
    # call, and insertion order preserves the transcript order the model saw.
    for call_id, entry in pending.items():
        started = entry.call_seq is not None
        message = {
            "id": f"interrupted-tool-result-{call_id}-{next_seq}",
            "role": "user",
            "source": {"kind": "tool", "callId": call_id},
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "isError": True,
                    "content": [
                        {
                            "type": "text",
                            "text": _OUTCOME_UNKNOWN_TEXT if started else _NOT_STARTED_TEXT,
                        }
                    ],
                }
            ],
        }
        data: dict[str, Any] = {
            "turn": open_turn,
            "step": entry.step,
            "message": message,
            "error": (
                {"name": "ToolOutcomeUnknownError", "code": TOOL_OUTCOME_UNKNOWN}
                if started
                else {"name": "ToolNotStartedError", "code": TOOL_NOT_STARTED}
            ),
        }
        closers.append(
            SessionEvent(
                type="tool/result",
                seq=next_seq,
                time=time,
                data=freeze_json_value(data),
                surface_op="append",
                source_event_seqs=(entry.call_seq,) if entry.call_seq is not None else None,
            )
        )
        next_seq += 1

    # An open step must close before its turn: `turn/end` while a step is open
    # is itself an invariant violation, so repairing one must not create another.
    if open_step is not None:
        closers.append(
            SessionEvent(
                type="step/end",
                seq=next_seq,
                time=time,
                data=freeze_json_value({"turn": open_turn, "step": open_step}),
            )
        )
        next_seq += 1
    closers.append(
        SessionEvent(
            type="turn/end",
            seq=next_seq,
            time=time,
            data=freeze_json_value({"turn": open_turn, "reason": {"kind": "interrupted"}}),
        )
    )
    return closers


def repaired(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """The log plus whatever closes it — what a resume should seed with."""
    return [*events, *interrupted_turn_closers(events)]
