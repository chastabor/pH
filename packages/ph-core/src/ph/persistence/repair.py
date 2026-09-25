"""Crash repair: closing a turn the process died inside, and every intent it left open (P1-12, A5).

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
Since P10-09 they are two declared intent kinds like any other, so this module
knows neither's keying nor its closer's words: `ph.session.kinds` states both once.

**Every declared intent, in a turn or out of one** (P10-07, F13). The asks above are
two of the pairs `ph.session.intents` declares; the others — a shell command the
daemon died during, a daemon verb — happen *between* turns, so a repair that
returned early for a balanced turn left them open forever, and every reader was
told a command was still running. One pass over `declared_intents()` settles each
open intent of a kind that is neither `owner-settles` (its owner looks, and
reconciles on open) nor model-visible when settled: the tool pair's closer is a
surface event a provider validates, which is the turn repair's job below and
answers a different requirement. The kind's own `closer` writes the payload, so
this module knows no kind's keying: a second spelling of "what counts as open"
here would be a second spelling of what repair must settle, drifting in the one
direction nothing fails.

**A kind this process never declared is refused, by name** (T4). Repair settles the
kinds declared *in the resuming process*, and a kind is declared by importing the
leaf that holds it: ph-core's by importing `ph.session`, which this module does, and
a package's by importing the package (`ph_app` imports `ph_app.kinds`). The
vocabulary's `INTENT_PAIRS` names every type that opens an intent and the leaf that
declares its kind, so a log holding an open intent of a kind this process lacks —
an open `client/command`, resumed without `ph_app` — raises `UndeclaredIntentError`
naming the type and the leaf, where it used to be left open with nothing said.

Not enforced: *which* intent is open, for a kind this process lacks. Without the
kind's keys its records cannot be paired, so they are counted — more openings than
settles means one is open. Exact, because the journal writes a settle only through a
claim on an open intent: T3 made each settle close exactly one, and nothing writes a
settle with no opening (a refused Code Mode dispatch has its start too, P7-15). A
journal door that settled with no opening would make the count miss an open intent.

For a turn parked on an ask, the turn is still closed `interrupted` and the tool
result is still synthesized `TOOL_NOT_STARTED` — that is what keeps the rebuilt log
something a provider will accept. The question is *settled* rather than left
hanging. The work resumes the way the harness
resumes any interrupted work: the model reads "not started" and asks again, to
whoever is attached then.

Ported from dsh `packages/core/session/src/repair.ts`, message texts included:
this vocabulary is what a resumed model reads, and paraphrasing it would change
behavior that was tuned deliberately.

@module ph.persistence.repair
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..json import JsonObject, JsonValue, as_int, as_obj, as_seq, as_str
from ..llm.types import content_from_wire, text_of
from ..session import (
    IntentError,
    IntentKind,
    IntentRecord,
    SessionEvent,
    abandoned,
    declared_intents,
    extend_index,
    is_in_place_rewrite,
    key_of,
    open_intents,
)
from ..session.json import freeze_json_value
from ..session.known_event_types import INTENT_PAIRS, IntentPair

__all__ = [
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "CallOutcome",
    "UndeclaredIntentError",
    "interrupted_turn_closers",
    "unresolved_calls",
]

TOOL_NOT_STARTED = "TOOL_NOT_STARTED"
"""An assistant tool request that never reached a recorded call start."""

TOOL_OUTCOME_UNKNOWN = "TOOL_OUTCOME_UNKNOWN"
"""A recorded call whose completed outcome was never durably recorded."""

_NOT_STARTED_TEXT = (
    "The tool call was interrupted before the Harness recorded it as started. "
    "Retry it if it is still needed."
)

_NOT_DONE_TEXT = (
    "The tool call was interrupted after it was recorded, and the tool has since "
    "checked: it did not happen. Retry it if it is still needed."
)

_OUTCOME_UNKNOWN_TEXT = (
    "The tool call was interrupted after it was recorded, but no result was durably "
    "recorded. Its outcome is unknown. Decide whether to retry from the tool "
    "semantics: retry only if the operation is read-only or idempotent; if it may "
    "have side effects, first verify external state or ask the user. Do not retry "
    "blindly."
)


_INSIDE_TEXT = (
    "Inside this call, before it was interrupted, these tools have since checked their own calls:"
)
"""Heads the list of a call's dispatches whose tools answered on resume (L6b)."""


class UndeclaredIntentError(IntentError):
    """A log holds open intents of a kind this process never declared (T4)."""


def _kinds() -> list[IntentKind]:
    """Every declared kind, in type order. ph-core's own are always among them:
    `ph.session` imports their leaf, and this module imports `ph.session`."""
    return sorted(declared_intents(), key=lambda kind: kind.opened)


def _refuse_undeclared(counted: Mapping[str, int], missing: Mapping[str, IntentPair]) -> None:
    """Refuse, by name, a log with an open intent of a kind nothing here declares.

    Counted, since the kind's keys are not here to pair its records with: more
    openings than settles means one is open.
    """
    for opened, pair in sorted(missing.items()):
        left = counted.get(opened, 0) - counted.get(pair.settled, 0)
        if left > 0:
            raise UndeclaredIntentError(
                f'this log holds {left} open "{opened}" intent(s), and this process declares '
                f"no kind for them: they are declared in {pair.leaf}, which nothing here "
                "imported. Resuming without it would leave them open; resume the log where "
                "that package is installed."
            )


def _settled_intents(
    events: Sequence[SessionEvent], answers: Mapping[str, Mapping[str, CallOutcome]]
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, CallOutcome]]]]:
    """The settles for every open intent of a kind repair is told to settle, and what
    the answered ones inside a call were found to have done, by that call's id.

    Kind by kind **in type order**, then in opening order, so the same log repairs
    the same way in every process — declaration order is import order, which a
    daemon and a trajectory viewer need not share. Folded in **one pass** over the
    log for every kind, since a resume runs this over the whole of it. Each closer
    is checked against the kind's own `settled_key` before it is returned: a closer
    that wrote a settle the fold does not pair would leave the intent open, and
    every resume after would write another — the log growing on each reopen, which
    is the one thing a repair must never do.

    `answers` are what tools said on resume about open intents of their own, by the
    kind's opening type and then the intent's key (L6b). For a kind that declares
    `reconciled`, a done answer is settled by that builder and a not-done one by the
    closer as `not-started`, both marked `reconciled`; a kind that declares `within`
    also names the call it ran inside.

    :raises IntentError: when a kind's closer does not settle its own key.
    :raises UndeclaredIntentError: when the log holds an open intent of a type
        `INTENT_PAIRS` names and no declared kind opens.
    """
    declared = _kinds()
    kinds = [kind for kind in declared if kind.orphan != "owner-settles"]
    by_type: dict[str, list[IntentKind]] = {}
    for kind in kinds:
        by_type.setdefault(kind.opened, []).append(kind)
        by_type.setdefault(kind.settled, []).append(kind)
    opens = {kind.opened for kind in declared}
    missing = {opened: pair for opened, pair in INTENT_PAIRS.items() if opened not in opens}
    watched = {name for opened, pair in missing.items() for name in (opened, pair.settled)}
    counted: Counter[str] = Counter()
    indexes: dict[IntentKind, dict[str, IntentRecord]] = {kind: {} for kind in kinds}
    for event in events:
        if event.type in watched:
            counted[event.type] += 1
        for kind in by_type.get(event.type, ()):
            extend_index(indexes[kind], (event,), kind)
    _refuse_undeclared(counted, missing)

    settled: list[dict[str, Any]] = []
    inside: dict[str, list[tuple[str, CallOutcome]]] = {}
    for kind in kinds:
        why = kind.orphan
        if why == "owner-settles":  # filtered out above; this narrows `why` to `Unsettled`
            continue
        told = answers.get(kind.opened, {})
        for intent in open_intents(indexes[kind], kind):
            answer = told.get(intent.key)
            built: JsonObject
            if answer is not None and kind.reconciled is not None:
                built = {
                    **(
                        kind.reconciled(intent.opened, answer.content)
                        if answer.done
                        else abandoned(kind, intent.opened, "not-started", "repair")
                    ),
                    "reconciled": True,
                }
                where = kind.within(intent.opened) if kind.within is not None else None
                if where is not None:
                    inside.setdefault(where[0], []).append((where[1], answer))
            else:
                built = abandoned(kind, intent.opened, why, "repair")
            data = freeze_json_value(built)
            if key_of(kind.settled_key, kind.settled, data, intent.opened.seq) != intent.key:
                raise IntentError(
                    f"the {kind.opened} closer declared by {kind.owner!r} does not settle "
                    f"{intent.key!r}; repair would reopen it on every resume"
                )
            settled.append({"type": kind.settled, "data": data})
    return settled, inside


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """What a tool said about a started call a crash left unresolved (P10-13).

    **Data, handed in** — which is what keeps repair a pure fold. Asking a tool
    needs a mounted deployment, and `interrupted_turn_closers` also runs over a
    stored log with nothing mounted; so the question is asked by the caller that
    is mounted (`resume_session`) and only the answers reach here. `content` is
    the call's result as the tool renders it, in wire form, when `done`.
    """

    done: bool
    content: tuple[JsonValue, ...] = ()


@dataclass(slots=True)
class _Pending:
    step: int
    call_seq: int | None = None


@dataclass(slots=True)
class _Tail:
    """The open tail turn, if any: its step, and its calls with no result."""

    turn: int | None = None
    step: int | None = None
    pending: dict[str, _Pending] = field(default_factory=dict)


def unresolved_calls(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """The `tool/call` records of the open tail turn that have no result.

    The calls a crash left *started* and unresolved — the ones worth asking the
    tool about (`ToolDefinition.reconcile`), since a call never recorded as
    started is not-started by construction.
    """
    tail = _tail(events)
    if tail.turn is None:
        return []
    return [events[entry.call_seq] for entry in tail.pending.values() if entry.call_seq is not None]


def _tail(events: Sequence[SessionEvent]) -> _Tail:
    tail = _Tail()
    pending = tail.pending

    for event in events:
        if event.type == "turn/start":
            tail.turn = as_int(event.data.get("turn"))
            tail.step = None
            pending.clear()
        elif event.type == "turn/end":
            tail.turn = None
            tail.step = None
            pending.clear()
        elif event.type == "step/start":
            tail.step = as_int(event.data.get("step"))
        elif event.type == "step/end":
            pending.clear()
            tail.step = None
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
    return tail


def interrupted_turn_closers(
    events: Sequence[SessionEvent],
    answers: Mapping[str, CallOutcome] | None = None,
    *,
    intents: Mapping[str, Mapping[str, CallOutcome]] | None = None,
) -> list[SessionEvent]:
    """The synthetic events that close an open tail turn and settle every open
    intent, in order.

    Returns `[]` for a balanced log — no open turn and no open intent — so a
    clean resume appends nothing and reopening a session does not grow it.

    `answers` are what tools said about their started, unresolved calls, by call
    id (`CallOutcome`); a call with none keeps `TOOL_OUTCOME_UNKNOWN`. `intents` are
    what tools said about open intents of their own, by opening type and key: a Code
    Mode dispatch's tool, asked as a top-level call's is (L6b). A call left
    `TOOL_OUTCOME_UNKNOWN` says what the answered dispatches inside it did, since
    the model reads only the call's result. Without either this is the same pure
    fold it always was.
    """
    tail = _tail(events)
    open_turn, open_step, pending = tail.turn, tail.step, tail.pending
    if not events:
        return []
    orphans, inside = _settled_intents(events, intents or {})
    if open_turn is None and not orphans:
        return []

    last = events[-1]
    next_seq = last.seq + 1
    time = last.time
    closers: list[SessionEvent] = []

    # Intents — the asks among them — settle before the call they were gating.
    # Nothing downstream depends on the order — no settle written here is
    # surface-eligible, so none derives a message or moves the provider-facing
    # sequence — but the log reads in the order things happened, and the ask
    # came first.
    #
    # No `source_event_seqs` back to the opening record, tempting as the symmetry
    # with the tool closer below is: `SURFACE_EVENT_TYPES` permits that field only
    # on the three types that carry a `surfaceOp`, and none of these is one.
    for settled in orphans:
        closers.append(SessionEvent(seq=next_seq, time=time, **settled))
        next_seq += 1
    if open_turn is None:
        return closers

    # Calls close before their step: a provider rejects a dangling assistant
    # call, and insertion order preserves the transcript order the model saw.
    for call_id, entry in pending.items():
        started = entry.call_seq is not None
        answer = (answers or {}).get(call_id) if started else None
        within = inside.get(call_id, [])
        closers.append(_tool_result(call_id, entry, open_turn, next_seq, time, answer, within))
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


_NOT_STARTED_ERROR = {"name": "ToolNotStartedError", "code": TOOL_NOT_STARTED}
_OUTCOME_UNKNOWN_ERROR = {"name": "ToolOutcomeUnknownError", "code": TOOL_OUTCOME_UNKNOWN}


def _tool_result(
    call_id: str,
    entry: _Pending,
    turn: int,
    seq: int,
    time: int,
    answer: CallOutcome | None,
    inside: Sequence[tuple[str, CallOutcome]],
) -> SessionEvent:
    """The `tool/result` that closes one call the turn left without one.

    With no `answer`, today's closer: `TOOL_NOT_STARTED` for a call never recorded
    as started, `TOOL_OUTCOME_UNKNOWN` for one that was. With one, what the tool
    said (P10-13) — the result it vouched for, or the not-started closer it made
    true — marked `reconciled` in `meta`, so a reader can tell a result the tool
    produced from one it reported afterwards about a call it had not seen finish.

    `inside` are the intents opened inside this call whose tools answered on resume,
    by name (L6b). A call left `TOOL_OUTCOME_UNKNOWN` lists them, because the model reads
    only this result: a program interrupted after a send the receiver's log holds
    should not send it again.
    """
    started = entry.call_seq is not None
    error: dict[str, str] | None
    if answer is not None and answer.done:
        content, error = list(answer.content), None
    elif answer is not None:
        content, error = [{"type": "text", "text": _NOT_DONE_TEXT}], _NOT_STARTED_ERROR
    elif started:
        text = f"{_OUTCOME_UNKNOWN_TEXT}\n\n{_inside(inside)}" if inside else _OUTCOME_UNKNOWN_TEXT
        content, error = [{"type": "text", "text": text}], _OUTCOME_UNKNOWN_ERROR
    else:
        content, error = [{"type": "text", "text": _NOT_STARTED_TEXT}], _NOT_STARTED_ERROR
    prefix = "interrupted" if answer is None else "reconciled"
    data: dict[str, Any] = {
        "turn": turn,
        "step": entry.step,
        "message": {
            "id": f"{prefix}-tool-result-{call_id}-{seq}",
            "role": "user",
            "source": {"kind": "tool", "callId": call_id},
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "isError": error is not None,
                    "content": content,
                }
            ],
        },
    }
    if error is not None:
        data["error"] = error
    if answer is not None:
        data["meta"] = {"reconciled": True}
    return SessionEvent(
        type="tool/result",
        seq=seq,
        time=time,
        data=freeze_json_value(data),
        surface_op="append",
        source_event_seqs=(entry.call_seq,) if entry.call_seq is not None else None,
    )


def _inside(answered: Sequence[tuple[str, CallOutcome]]) -> str:
    """What the tools inside an interrupted call said about their own calls."""
    lines = [_INSIDE_TEXT]
    for name, said in answered:
        if said.done:
            shown = text_of(content_from_wire(list(said.content)), separator=" ")
            lines.append(f"- `{name}` happened" + (f": {shown}" if shown else "."))
        else:
            lines.append(f"- `{name}` did not happen.")
    return "\n".join(lines)


def repaired(events: Sequence[SessionEvent]) -> list[SessionEvent]:
    """The log plus whatever closes it — what a resume should seed with."""
    return [*events, *interrupted_turn_closers(events)]
