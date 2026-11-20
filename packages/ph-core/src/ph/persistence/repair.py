"""Crash repair: closing a turn the process died inside, and every intent it left open (P1-12, A5).

A log whose last turn is half-written is not only untidy. It is *invalid* to a
provider. Several APIs reject outright an assistant message that carries a tool
call with no matching result. So without this module, a session that crashed in
the middle of a batch could never be resumed.

The synthesized closers are deterministic. Their sequences continue the log,
and their timestamp is that of the **last real event**, never `now()`. A repair
that invented a future time would make the log say that the recovery happened
during the crash.

There are two failure shapes, and the two codes exist to tell them apart:

* `TOOL_NOT_STARTED`: the assistant asked, but no `tool/call` was ever
  recorded. Nothing ran, so a retry is safe, and the text says so.
* `TOOL_OUTCOME_UNKNOWN`: the call *was* recorded, and then the process died.
  The call may have completed. The text tells the model to reason from the
  semantics of the tool and not to retry blindly. A blind retry of a
  non-idempotent operation turns one crash into two side effects.

**Dangling asks are settled here for the same reason** (P5-13). An
`approval/asked` with no `approval/decided`, or a `question/asked` with no
`question/answered`, is a question that was put to a person, and the process
died before the answer came. Without a closer, those pairs stay half-written
*forever*. Every future reader of a resumed log is told that a decision is
outstanding when nothing waits for one, and the transcript never says what
became of the question. Since P10-09 they are two declared intent kinds like any
other. So this module knows neither the keying of a kind nor the words of its
closer: `ph.session.kinds` states both in one place.

**Every declared intent is settled, in a turn or out of one** (P10-07, F13).
The asks above are two of the declared kinds. Some of the others, such as a
shell command that the daemon died during or a daemon verb, happen *between*
turns. A repair that returned early for a balanced turn left them open forever,
and every reader was told that a command was still running. One pass over
`declared_intents()` settles each open intent of every kind that is not
`owner-settles` (the owner of such a kind looks, and reconciles on open). A
pair whose settle is model-visible is not a kind, because `declare_intent`
refuses it. The closer of the tool pair is a surface event that a provider
validates. Writing it is the work of the turn repair below, which answers a
different requirement. The `closer` of the kind writes the payload, so this
module knows the keying of no kind. A second spelling here of "what counts as
open" would be a second spelling of what repair must settle. It would drift in
the one direction in which nothing fails.

**A kind that this process never declared is refused, by name** (T4). Repair
settles the kinds that are declared *in the resuming process*. A kind is
declared by importing the leaf that holds it. The kinds of ph-core are declared
by importing `ph.session`, which this module does. The kinds of a package are
declared by importing the package (`ph_app` imports `ph_app.kinds`).
`INTENT_PAIRS` in the vocabulary names every type that opens an intent and the
leaf that declares its kind. So a log that holds an open intent of a kind that
this process lacks raises `UndeclaredIntentError`, which names the type and the
leaf. An example is an open `client/command`, resumed without `ph_app`.
Formerly such an intent was left open and nothing was reported.

Not enforced: *which* intent is open, for a kind that this process lacks.
Without the keys of the kind, its records cannot be paired, so they are
counted. More openings than settles means that one is open. The count is exact,
because the journal writes a settle only through a claim on an open intent. T3
made each settle close exactly one intent, and nothing writes a settle with no
opening (a refused Code Mode dispatch has its start too, P7-15). A journal
method that wrote a settle with no opening would make the count miss an open
intent.

For a turn parked on an ask, the turn is still closed `interrupted`, and the
tool result is still synthesized as `TOOL_NOT_STARTED`. That keeps the rebuilt
log acceptable to a provider. The question is *settled* and not left open. The
work resumes in the way that the harness resumes any interrupted work: the model
reads "not started" and asks again, and whoever is attached at that time gets
the question.

Ported from dsh `packages/core/session/src/repair.ts`, with the message texts.
This vocabulary is what a resumed model reads. A paraphrase would change
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
    settle_of,
)
from ..session.json import freeze_json_value
from ..session.known_event_types import INTENT_PAIRS, IntentPair

__all__ = [
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "CallOutcome",
    "UndeclaredIntentError",
    "interrupted_turn_closers",
    "repaired",
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
    """Every declared kind, in type order. The kinds of ph-core are always among
    them: `ph.session` imports their leaf, and this module imports `ph.session`."""
    return sorted(declared_intents(), key=lambda kind: kind.opened)


def _refuse_undeclared(counted: Mapping[str, int], missing: Mapping[str, IntentPair]) -> None:
    """Refuse, by name, a log with an open intent of a kind that nothing here declares.

    The records are counted, because the keys of the kind are not here to pair
    them with. More openings than settles means that one is open.
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
    """The settles for every open intent of a kind that repair is told to settle. Also
    what the answered intents inside a call were found to have done, by the id of
    that call.

    The kinds are taken **in type order**, and the intents of each kind in
    opening order, so the same log repairs the same way in every process.
    Declaration order is import order, and a daemon and a trajectory viewer need
    not share it. Every kind is folded in **one pass** over the log, because a
    resume runs this function over the whole log. Each closer is checked against
    the `settled_key` of its kind before it is returned. A closer that wrote a
    settle that the fold does not pair would leave the intent open, and every
    later resume would write another settle. The log would grow on each reopen,
    which a repair must never cause.

    `answers` are what tools said on resume about open intents of their own, by
    the opening type of the kind and then by the key of the intent (L6b). For a
    kind that declares `reconciled`, that builder settles a done answer, and the
    closer settles a not-done answer as `not-started`. Both settles are marked
    `reconciled`. A kind that declares `within` also names the call that the
    intent ran inside.

    :raises IntentError: when the closer of a kind does not settle its own key.
    :raises UndeclaredIntentError: when the log holds an open intent of a type
        that `INTENT_PAIRS` names and that no declared kind opens.
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
    """What a tool said about a started call that a crash left unresolved (P10-13).

    **The answer is data that is handed in**, and that keeps repair a pure fold.
    Asking a tool needs a mounted deployment, and `interrupted_turn_closers`
    also runs over a stored log with nothing mounted. So the caller that is
    mounted (`resume_session`) asks the question, and only the answers reach
    this module. `content` is the result of the call as the tool renders it, in
    wire form, when `done` is true.
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

    These are the calls that a crash left *started* and unresolved. They are the
    calls worth asking the tool about (`ToolDefinition.reconcile`), because a
    call that was never recorded as started is not-started by construction.
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
                # This is a near-copy of a message that is already in the log.
                # Argument truncation is the one that does this. Its `tool-call`
                # blocks are answered behind it, so registering them again would
                # make this fold write a second `tool/result` for one id.
                #
                # The check is `is_in_place_rewrite` and not
                # `is_replacement_surface_event`, because the narrower predicate
                # is the one that means "not new work". The commit refuses an
                # assistant replacement that is anything else
                # (`surface._assert_assistant_rewrite`). New speech in place of
                # a range would carry calls that *do* need closing. So the two
                # predicates agree today, and the narrow one stays right if that
                # ever changes.
                continue
            content = as_seq(as_obj(event.data.get("message")).get("content"))
            for block in (as_obj(one) for one in content):
                if block.get("type") == "tool-call":
                    pending[as_str(block.get("id"))] = _Pending(step=as_int(event.data.get("step")))
        elif event.type == "tool/call":
            entry = pending.get(as_str(event.data.get("callId")))
            if entry is not None:
                entry.call_seq = event.seq
        elif (settled := settle_of(event)) is not None:
            pending.pop(settled[0], None)
    return tail


def interrupted_turn_closers(
    events: Sequence[SessionEvent],
    answers: Mapping[str, CallOutcome] | None = None,
    *,
    intents: Mapping[str, Mapping[str, CallOutcome]] | None = None,
) -> list[SessionEvent]:
    """The synthetic events that close an open tail turn and settle every open
    intent, in order.

    Returns `[]` for a balanced log, which has no open turn and no open intent.
    So a clean resume appends nothing, and reopening a session does not grow it.

    `answers` are what tools said about their started, unresolved calls, by call
    id (`CallOutcome`). A call with no answer keeps `TOOL_OUTCOME_UNKNOWN`.
    `intents` are what tools said about open intents of their own, by opening
    type and key. The tool of a Code Mode dispatch is asked in the same way as
    the tool of a top-level call (L6b). A call left `TOOL_OUTCOME_UNKNOWN` says
    what the answered dispatches inside it did, because the model reads only the
    result of the call. With neither argument, this is the same pure fold as
    before.
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

    # Intents, the asks among them, settle before the call that they were gating.
    # Nothing downstream depends on the order: no settle written here is
    # surface-eligible, so none derives a message or moves the provider-facing
    # sequence. But the log reads in the order in which things happened, and the
    # ask came first.
    #
    # The settles carry no `source_event_seqs` back to the opening record. The
    # symmetry with the tool closer below is tempting, but `SURFACE_EVENT_TYPES`
    # permits that field only on the three types that carry a `surfaceOp`, and
    # none of these types is one of the three.
    for settled in orphans:
        closers.append(SessionEvent(seq=next_seq, time=time, **settled))
        next_seq += 1
    if open_turn is None:
        return closers

    # Calls close before their step. A provider rejects a dangling assistant
    # call, and insertion order preserves the transcript order that the model saw.
    for call_id, entry in pending.items():
        started = entry.call_seq is not None
        answer = (answers or {}).get(call_id) if started else None
        within = inside.get(call_id, [])
        closers.append(_tool_result(call_id, entry, open_turn, next_seq, time, answer, within))
        next_seq += 1

    # An open step must close before its turn. A `turn/end` while a step is open
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
    """The `tool/result` that closes one call that the turn left without one.

    With no `answer`, this is the plain closer: `TOOL_NOT_STARTED` for a call
    that was never recorded as started, and `TOOL_OUTCOME_UNKNOWN` for one that
    was. With an `answer`, it is what the tool said (P10-13): the result that
    the tool vouched for, or the not-started closer, which the answer of the
    tool makes true. It is marked `reconciled` in `meta`. So a reader can tell a
    result that the tool produced from one that it reported afterwards about a
    call that it had not seen finish.

    `inside` are the intents opened inside this call whose tools answered on
    resume, by name (L6b). A call left `TOOL_OUTCOME_UNKNOWN` lists them,
    because the model reads only this result. A program that was interrupted
    after a send that the log of the receiver holds should not send it again.
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
    """The log plus whatever closes it. A resume should seed with this."""
    return [*events, *interrupted_turn_closers(events)]
