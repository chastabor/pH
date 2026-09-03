"""The auditor's projection: one record per thing that happened (P3-24).

The transcript (`adapter.py`) is the *reader's* projection — what the
conversation looked like. This is the second projection of the same fold (I6):
what the harness did, including the six event types the transcript deliberately
does not render, because they are an auditor's records rather than a reader's.

Two disciplines make it a projection rather than a second data path.

**Every record is derived, nothing is remembered.** A record's timings come from
`step/*` event times, its prompt snapshot from `request/header`, its producer
from `PluginSource` — all already in the log because something else needed them.
That is what makes this view addable at zero cost and, more importantly, what
makes it agree with the transcript by construction rather than by discipline.

**The vocabulary is closed and total.** `RecordKind` is dsh's set, and
`RECORDLESS` names the types that deliberately produce none. Together they must
cover `KNOWN_SESSION_EVENT_TYPES` exactly — a test holds that, so a new event
type cannot land unclassified. The transcript has the same rule for the same
reason; this is the auditor's half of it.

**`sourceSeq` is the join.** Every record points back at the event that produced
it, which is what lets the two views cross-navigate and what a fork aims at.

@module ph_app.tui.trajectory
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ph.session import Session, SessionEvent, fork_boundaries, is_replacement_surface_event
from ph.session.request_header import parse_request_header

from ..wire import describe, message_of, obj, one_line, result_block, source_of, text_of_wire

__all__ = [
    "HANDLERS",
    "RECORDLESS",
    "RecordKind",
    "SourceRef",
    "Timing",
    "TrajectoryRecord",
    "build_trajectory",
]

RecordKind: TypeAlias = Literal[
    "system", "user", "context", "compacted", "message", "tool", "subtool", "event"
]
"""dsh's closed set, plus `event`.

`event` is pH's addition and it is the honest one: dsh's vocabulary describes a
*conversation*, and pH's log also carries harness facts a conversation has no
kind for — a policy change, a file read, the seed boundary. Folding those into
`context` would say they were shown to the model, which is exactly the kind of
lie this view exists to make impossible."""

RECORDLESS: frozenset[str] = frozenset(
    {
        # Chunks are the streamed halves of `assistant/message`, which *is* a
        # record. One record per delta would be a keystroke log, not a trajectory.
        "assistant/chunk",
        # Both are folded into the records they belong to rather than standing
        # alone: `step/start` opens the timing that `step/end` closes, and the
        # pair surfaces on the `message` record as `Timing`.
        "step/start",
        "step/end",
        # The dispatch's own start; its settled half carries the record.
        "tool/code-dispatch-start",
        # Kernel bookkeeping: one per changed variable per cell. `kernel/restored`
        # *is* a record, because a variable that did not come back is news.
        "kernel/snapshot",
        # The panel's, in both views — a child's status and token attribution are
        # a projection beside the trajectory, not entries in it.
        "subagent/status",
        "subagent/usage-attributed",
    }
)
"""Types that deliberately produce no record.

*Differently shaped* from the transcript's set, not smaller — both hold seven,
and four of these are types the transcript renders. The difference is the point:
`request/header`, `approval/policy`, `fs/observed` and `session/end-seed` are
record-less for a *reader* and are exactly what an auditor came for, while
`assistant/chunk` and the dispatch's opening half are rows in a conversation and
duplicates in an audit. What stays out here is only what another record already
carries, or bookkeeping so fine-grained it would bury what it describes."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Who produced a record's content."""

    kind: str = ""
    """`user` | `model` | `plugin` | `tool`, from the message's own source."""
    name: str = ""
    """The plugin, provider or tool that produced it."""
    form: str = ""
    """A `PluginSource.form` — `instructions`, `snapshot`, `notice`, … — which is
    what "inspect these records by source" filters on."""

    def label(self) -> str:
        parts = [part for part in (self.kind, self.name) if part]
        return f"{':'.join(parts)}{f' ({self.form})' if self.form else ''}"


@dataclass(frozen=True, slots=True)
class Timing:
    """What a step cost, derived from event times alone.

    `time_to_first_token` is the wait a person feels; `decode_tokens_per_second`
    is what the rest of the step spent. Both are `None` when the log does not
    carry enough to say so rather than being guessed — an auditor's view that
    invents a number is worse than one that admits it cannot.
    """

    total_ms: int | None = None
    time_to_first_token_ms: int | None = None
    decode_tokens_per_second: float | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class TrajectoryRecord:
    """One row of the auditor's view."""

    index: int
    """1-based `#N`, stable for a given log."""
    kind: RecordKind
    source_seq: int
    """The event this record projects — the join with the transcript."""
    type: str = ""
    """The session-log type this projects, verbatim — `workspace/acquired`, not
    `event` (P6-33).

    `kind` is the *view's* eight-value vocabulary and is what a row is coloured
    and grouped by; this is the log's own 63-type one, and the two are not
    interchangeable: forty-odd harness types share the single kind `event`, so a
    reader wanting "just the retained worktrees" cannot ask for it in `kind`.
    Carried on the record rather than re-derived from `source_seq`, because this
    view folds a stored log with nothing mounted — the filter must not need the
    log back to answer.
    """
    title: str = ""
    """What the row is called — the table's own column, so an `event` reads as
    "turn end" rather than as the type that produced it."""
    summary: str = ""
    """One line, for the table."""
    detail: str = ""
    """The full text, for the panel."""
    source: SourceRef = field(default_factory=SourceRef)
    turn: int = 0
    timing: Timing | None = None
    replaced: str = ""
    """For a `system` record: the snapshot this one replaced, so a prompt change
    reads as a diff rather than as a new wall of text (dsh's own requirement)."""
    tools: list[str] = field(default_factory=list)
    """Tool *names* as the catalog held them at call time, not as it holds them
    now. Names rather than schemas: the panel shows what was callable, and
    keeping the schemas would have held a copy of every catalog for the life of
    the view to render a comma-separated list."""
    fork_point: bool = False
    """Whether `ctx.sessions.fork` may aim here (A6). Only closed-turn
    boundaries qualify, so the table can show which rows are targets instead of
    letting a person aim anywhere and be refused."""


def _source_ref(message: Mapping[str, Any]) -> SourceRef:
    kind, name, form = source_of(message)
    return SourceRef(kind=kind, name=name, form=form)


def _text(blocks: Any) -> str:
    """The visible text of wire content, with other blocks named rather than
    dropped — an auditor wants to see that an image was there."""
    return text_of_wire(blocks, placeholder=lambda kind: f"[{kind}]")


@dataclass(slots=True)
class _Builder:
    """Folds a log into records. One pass, no lookahead."""

    records: list[TrajectoryRecord] = field(default_factory=list)
    turn: int = 0
    _step_started: int | None = None
    _first_chunk: int | None = None
    _system: str = ""

    def add(self, **fields: Any) -> TrajectoryRecord:
        record = TrajectoryRecord(index=len(self.records) + 1, **fields)
        self.records.append(record)
        return record

    # ------------------------------------------------------------ handlers --

    def on_request_header(self, event: SessionEvent) -> None:
        """The prompt-and-tool-catalog snapshot, plus the one it replaced.

        `request/header` is logged only when it *changed* (A12), so every one of
        these is a real change and the previous text is what makes it readable.
        """
        try:
            # The model ph-core already parses this into, so the wire keys are
            # spelled once — a renamed one is an error here rather than a
            # silently empty catalog and a `?/?` route.
            header = parse_request_header(event)
        except Exception:
            return
        system = header.system or ""
        tools = [schema.name for schema in header.tools or ()]
        self.add(
            kind="system",
            source_seq=event.seq,
            title="system prompt",
            summary=(
                f"{len(system)} chars, {len(tools)} tool(s) · "
                f"{header.config.provider}/{header.config.model}"
            ),
            detail=system,
            replaced=self._system,
            tools=tools,
            source=SourceRef(kind="harness", name="request/header"),
            turn=self.turn,
        )
        self._system = system

    def on_message(self, event: SessionEvent, kind: RecordKind) -> None:
        outer = obj(event.data)
        message = message_of(event)
        source = _source_ref(message)
        text = _text(message.get("content"))
        # A compaction summary shadows the range it stands for; the record says
        # so rather than looking like an ordinary message that appeared.
        record_kind: RecordKind = "compacted" if is_replacement_surface_event(event) else kind
        if record_kind == "user" and source.kind == "plugin":
            record_kind = "context"
        self.turn = int(outer.get("turn", self.turn)) or self.turn
        self.add(
            kind=record_kind,
            source_seq=event.seq,
            title=source.label() or record_kind,
            summary=one_line(text),
            detail=text,
            source=source,
            turn=self.turn,
            timing=self._close_timing(event, outer) if kind == "message" else None,
        )

    def on_tool_call(self, event: SessionEvent) -> None:
        data = obj(event.data)
        name = str(data.get("name") or "?")
        self.add(
            kind="tool",
            source_seq=event.seq,
            title=name,
            summary=one_line(str(data.get("arguments") or "")),
            detail=str(data.get("arguments") or ""),
            source=SourceRef(kind="tool", name=name),
            turn=self.turn,
        )

    def on_tool_result(self, event: SessionEvent) -> None:
        message = obj(event.data.get("message"))
        result = result_block(message)
        _kind, call_id, _form = source_of(message)
        text = _text(result.get("content"))
        self.add(
            kind="tool",
            source_seq=event.seq,
            title=f"result {call_id}" if call_id else "result",
            summary=one_line(text),
            detail=text,
            source=SourceRef(kind="tool", name=call_id),
            turn=self.turn,
        )

    def on_sub_dispatch(self, event: SessionEvent) -> None:
        """A Code Mode sub-call: `subtool`, the kind C2 exists to make visible."""
        data = obj(event.data)
        name = str(data.get("name") or "?")
        text = _text(data.get("content"))
        self.add(
            kind="subtool",
            source_seq=event.seq,
            title=name,
            summary=one_line(text),
            detail=text,
            source=SourceRef(kind="tool", name=name),
            turn=self.turn,
        )

    def on_event(self, event: SessionEvent, title: str, summary: str) -> None:
        """A harness fact with no conversational kind."""
        self.add(
            kind="event",
            source_seq=event.seq,
            title=title,
            summary=summary,
            detail=summary,
            source=SourceRef(kind="harness", name=event.type),
            turn=self.turn,
        )

    # -------------------------------------------------------------- timing --

    def _close_timing(self, event: SessionEvent, message: Mapping[str, Any]) -> Timing:
        """What the step cost, from event times and the provider's own usage.

        Derived, never measured here: the view must produce the same numbers on
        a replayed log as on a live one, and a clock read at render time would
        not (A11).
        """
        usage = obj(message.get("usage"))
        output = usage.get("outputTokens")
        tokens = int(output) if isinstance(output, int) else None
        started, first_chunk = self._step_started, self._first_chunk
        total = event.time - started if started is not None else None
        ttft = first_chunk - started if started is not None and first_chunk else None
        rate: float | None = None
        if tokens and first_chunk is not None and event.time > first_chunk:
            rate = round(tokens / ((event.time - first_chunk) / 1000), 1)
        return Timing(
            total_ms=total,
            time_to_first_token_ms=ttft,
            decode_tokens_per_second=rate,
            output_tokens=tokens,
        )


def _on_turn_start(builder: _Builder, event: SessionEvent) -> None:
    builder.turn = int(obj(event.data).get("turn", 0))
    builder.on_event(event, "turn start", f"turn {builder.turn}")


def _on_turn_end(builder: _Builder, event: SessionEvent) -> None:
    reason = obj(obj(event.data).get("reason")).get("kind") or "completed"
    builder.on_event(event, "turn end", f"turn {builder.turn} — {reason}")


def _on_harness_event(builder: _Builder, event: SessionEvent) -> None:
    """A harness fact with no conversational kind, rendered from its payload."""
    builder.on_event(event, event.type, _describe(event))


Handler = Callable[["_Builder", SessionEvent], None]

HANDLERS: Mapping[str, Handler] = {
    "turn/start": _on_turn_start,
    "turn/end": _on_turn_end,
    "request/header": lambda builder, event: builder.on_request_header(event),
    "user/message": lambda builder, event: builder.on_message(event, "user"),
    "assistant/message": lambda builder, event: builder.on_message(event, "message"),
    "tool/call": lambda builder, event: builder.on_tool_call(event),
    "tool/result": lambda builder, event: builder.on_tool_result(event),
    "tool/code-dispatch": lambda builder, event: builder.on_sub_dispatch(event),
    # Every remaining known type renders generically, and each is listed rather
    # than caught by an `else`. The `else` is what made the completeness gate
    # unfalsifiable: with a catch-all, a type added to the vocabulary silently
    # got a generic row and no one ever decided whether it deserved a kind of
    # its own. Explicit keys mean the *set* is checkable, which is the shape
    # `adapter.py` uses for exactly this reason.
    "request/context": _on_harness_event,
    "approval/asked": _on_harness_event,
    "approval/decided": _on_harness_event,
    "approval/mode": _on_harness_event,
    "approval/policy": _on_harness_event,
    "workspace/acquired": _on_harness_event,
    "workspace/disposed": _on_harness_event,
    "workspace/retained": _on_harness_event,
    "workspace/provisioned": _on_harness_event,
    "workspace/checkpoint": _on_harness_event,
    "permission/preset": _on_harness_event,
    "sandbox/mode": _on_harness_event,
    "command/run": _on_harness_event,
    "command/done": _on_harness_event,
    "fs/observed": _on_harness_event,
    "llm/retry": _on_harness_event,
    # Records, never `RECORDLESS`: a reader that skipped these would find a
    # transcript resuming mid-thought with no account of why, and a session that
    # stopped working would look like one that simply went quiet. For an
    # unattended run that is the whole point of reading the trace.
    "supervisor/retry": _on_harness_event,
    "supervisor/failed": _on_harness_event,
    "supervisor/recovered": _on_harness_event,
    "supervisor/passivated": _on_harness_event,
    "supervisor/unreachable": _on_harness_event,
    "schedule/created": _on_harness_event,
    "schedule/cancelled": _on_harness_event,
    "schedule/tick": _on_harness_event,
    "schedule/heartbeat": _on_harness_event,
    "goal/set": _on_harness_event,
    "goal/continued": _on_harness_event,
    "goal/gate": _on_harness_event,
    "goal/settled": _on_harness_event,
    # A record, not `RECORDLESS`: an auditor reading a transcript needs to know
    # where somebody else's work ends and this run's begins, and that the turn
    # above the seam may have been closed by the repair rather than by the model.
    "session/resumed": _on_harness_event,
    # The mirror of the line above, and a record for the mirrored reason: the
    # seam at the *end* of a log is as invisible as the one at the start, and an
    # auditor who reaches the last row of a segment has reached the end of a
    # file rather than the end of the work.
    "session/segmented": _on_harness_event,
    # A record: "who asked for this, and had they asked before" is exactly the
    # provenance an auditor reading a daemon-driven run needs.
    "client/command": _on_harness_event,
    "kernel/restored": _on_harness_event,
    "subagent/admitted": _on_harness_event,
    "subagent/deleted": _on_harness_event,
    "harness/refined": _on_harness_event,
    "harness/refine-considered": _on_harness_event,
    "context/loaded": _on_harness_event,
    "agent/inbox/spliced": _on_harness_event,
    "todo/write": _on_harness_event,
    "offload/spilled": _on_harness_event,
    "offload/input-spilled": _on_harness_event,
    # Both, and generically: an auditor came for exactly this — what a summary
    # shadowed, what it cost, which model wrote it, and every attempt that
    # decided not to. The transcript renders neither (the summary rides on its
    # replacement `user/message`), which is the split this view exists for.
    "compaction/summarized": _on_harness_event,
    "compaction/declined": _on_harness_event,
    "compaction/args-truncated": _on_harness_event,
    "attachment/degraded": _on_harness_event,
    "attachment/oversized": _on_harness_event,
    "attachment/uploaded": _on_harness_event,
    "limits/exceeded": _on_harness_event,
    "limits/breaker-tripped": _on_harness_event,
    "session/end-seed": _on_harness_event,
}
"""Event type → the record it produces.

Explicit, so `set(HANDLERS) | RECORDLESS` is a value a test can hold against
`KNOWN_SESSION_EVENT_TYPES` — which is the only version of "no silent
omissions" that fails when a type is added."""


def build_trajectory(session: Session) -> list[TrajectoryRecord]:
    """Fold a log into the auditor's records.

    Takes a `Session`, which a *stored* log becomes through
    `ph.persistence.read_session` with nothing mounted — the property P3-25's
    harness-free entry point rests on.

    Fork points are decided against `fork_boundaries`, the store's own rule, so
    the rows this marks are exactly the ones `ctx.sessions.fork` accepts. Asked
    once for the whole log rather than once per record: the per-record form
    rescanned the prefix each time, which is quadratic and became a frozen UI
    when P4-17 put this fold on a keypress.
    """
    builder = _Builder()
    log = session.events
    boundaries = fork_boundaries(log)
    for event in log:
        handler = HANDLERS.get(event.type)
        if handler is None:
            # Record-less, or a type from a newer build. Two of the record-less
            # ones are still *read*: a step opens the clock and the first chunk
            # marks time-to-first-token, which is why neither is a record.
            if event.type == "step/start":
                builder._step_started = event.time
                builder._first_chunk = None
            elif event.type == "assistant/chunk" and builder._first_chunk is None:
                builder._first_chunk = event.time
            continue
        before = len(builder.records)
        handler(builder, event)
        # **The records this handler added**, which is not always exactly one:
        # `on_request_header` returns without adding when the payload will not
        # parse. `records[-1]` therefore stamped a *previous, unrelated* record —
        # with this event's fork status and, on a log whose first record-producing
        # event was that one, with an `IndexError`. A slice is correct for zero,
        # one and many, which is what the handler table actually contains.
        for record in builder.records[before:]:
            record.type = event.type
            record.fork_point = event.seq in boundaries
    return builder.records


def _describe(event: SessionEvent) -> str:
    """A one-line account of a harness event, from its own payload.

    The generic reading itself now lives in `ph_app.wire`, because `ph agents
    attach` needs the same fallback for the same reason and cannot import
    anything under `ph_app.tui` without paying for Textual. What stays here is
    the type name, which is this view's answer for a payload with nothing in it.
    """
    return describe(event.data) or event.type
