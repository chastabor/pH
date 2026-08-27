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

Ported from dsh `packages/core/session/src/repair.ts`, message texts included:
this vocabulary is what a resumed model reads, and paraphrasing it would change
behaviour that was tuned deliberately.

@module ph.persistence.repair
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..session import SessionEvent
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
            open_turn = int(event.data.get("turn", 0))
            open_step = None
            pending.clear()
        elif event.type == "turn/end":
            open_turn = None
            open_step = None
            pending.clear()
        elif event.type == "step/start":
            open_step = int(event.data.get("step", 0))
        elif event.type == "step/end":
            pending.clear()
            open_step = None
        elif event.type == "assistant/message":
            message = event.data.get("message") or {}
            for block in message.get("content", ()):
                if block.get("type") == "tool-call":
                    pending[str(block.get("id"))] = _Pending(step=int(event.data.get("step", 0)))
        elif event.type == "tool/call":
            entry = pending.get(str(event.data.get("callId")))
            if entry is not None:
                entry.call_seq = event.seq
        elif event.type == "tool/result":
            source = (event.data.get("message") or {}).get("source") or {}
            pending.pop(str(source.get("callId")), None)

    if open_turn is None or not events:
        return []

    last = events[-1]
    seq = last.seq + 1
    time = last.time
    closers: list[SessionEvent] = []

    # Calls close before their step: a provider rejects a dangling assistant
    # call, and insertion order preserves the transcript order the model saw.
    for call_id, entry in pending.items():
        started = entry.call_seq is not None
        message = {
            "id": f"interrupted-tool-result-{call_id}-{seq}",
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
                seq=seq,
                time=time,
                data=freeze_json_value(data),
                surface_op="append",
                source_event_seqs=(entry.call_seq,) if entry.call_seq is not None else None,
            )
        )
        seq += 1

    # An open step must close before its turn: `turn/end` while a step is open
    # is itself an invariant violation, so repairing one must not create another.
    if open_step is not None:
        closers.append(
            SessionEvent(
                type="step/end",
                seq=seq,
                time=time,
                data=freeze_json_value({"turn": open_turn, "step": open_step}),
            )
        )
        seq += 1
    closers.append(
        SessionEvent(
            type="turn/end",
            seq=seq,
            time=time,
            data=freeze_json_value({"turn": open_turn, "reason": {"kind": "interrupted"}}),
        )
    )
    return closers


def repaired(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """The log plus whatever closes it — what a resume should seed with."""
    return [*events, *interrupted_turn_closers(events)]
