"""The projection for an auditor: one record for each thing that happened (P3-24).

The transcript (`adapter.py`) is the projection for a *reader*: what the
conversation looked like. This module is the second projection of the same fold
(I6): what the harness did. It includes the event types that the transcript
deliberately does not render, because they are records for an auditor and not
for a reader. ph-core declares those types in `AUDIT_ONLY_SESSION_EVENT_TYPES`.

Two rules make this a projection and not a second data path.

**Every record is derived, and nothing is remembered.** The timings of a record
come from the times of `step/*` events, its prompt snapshot comes from
`request/header`, and its producer comes from `PluginSource`. All of these are
already in the log because something else needed them. For that reason this
view costs nothing to add. More importantly, it agrees with the transcript by
construction and not by discipline.

**The vocabulary is closed and total.** `RecordKind` is the set from dsh.
`HANDLERS` maps the types that have a record of their own, and `RECORDLESS`
names the types that deliberately produce none. The remaining types are the
auditor's records that the vocabulary declares (`is_audit_only`), and they are
rendered from their payload. Together these must cover the vocabulary exactly.
A test holds that, so a new event type cannot be added unclassified. The
transcript has the same rule for the same reason, and this is the auditor's
half of it.

**A type that this process does not know is also rendered from its payload.**
Such a type came from another build, or from a package that this process never
imported, as when a stored log is read with nothing mounted. It reached this
module only because it is ignorable, since the seed refuses an unknown required
type. To skip it would be the silent omission that this view exists to prevent,
and the app may not import a bundle to learn its types.

**`sourceSeq` is the join.** Every record points back at the event that
produced it. That lets the two views cross-navigate, and it is what a fork aims
at.

@module ph_app.tui.trajectory
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from ph.json import as_int, as_obj, as_seq, as_str
from ph.seams.skills import read_summary
from ph.session import (
    Session,
    SessionEvent,
    fork_boundaries,
    is_audit_only,
    is_replacement_surface_event,
)
from ph.session.known_event_types import is_known
from ph.session.request_header import parse_request_header
from ph.session_profile import EnvironmentFold, environment_listing, record_summary
from ph.text import block_marker, count_of, one_line

from ..wire import describe, message_of, result_block, source_of, text_of_wire

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
"""The closed set from dsh, plus `event`.

`event` is the addition that pH made, and it is the accurate one. The
vocabulary of dsh describes a *conversation*. The log of pH also carries
harness facts that a conversation has no kind for: a policy change, a file
read, the seed boundary. To fold those into `context` would say that they were
shown to the model. This view exists to make exactly that kind of false
statement impossible."""

RECORDLESS: frozenset[str] = frozenset(
    {
        # Chunks are the streamed halves of `assistant/message`, which *is* a
        # record. One record for each delta would be a keystroke log, not a
        # trajectory.
        "assistant/chunk",
        # Neither is a record of its own. `step/start` opens the timing, and
        # the `assistant/message` of the step closes it, so the timing appears
        # on the `message` record as `Timing`. Nothing here reads `step/end`.
        "step/start",
        "step/end",
        # The start of the dispatch. Its settled half carries the record.
        "tool/code-dispatch-start",
        # Kernel bookkeeping: one for each changed variable in each cell.
        # `kernel/restored` *is* a record, because a variable that did not come
        # back is news.
        "kernel/snapshot",
    }
)
"""Types that deliberately produce no record.

This is the set of the transcript plus two more: `assistant/chunk` and the
opening half of a dispatch. Those two are rows in a conversation and duplicates
in an audit. The only types listed here are those that another record already
carries, and bookkeeping so fine-grained that it would bury what it describes.
The types that a reader has no row for and that an auditor came for, such as
`request/header` and `approval/policy`, are not here. The vocabulary declares
them (`is_audit_only`)."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Who produced a record's content."""

    kind: str | None = None
    """`user` | `model` | `plugin` | `tool`, from the source of the message
    itself, or `harness` for a record that the harness produced. `None` when the
    message names no source."""
    name: str | None = None
    """The plugin, provider or tool that produced it, or `None` when the source
    names none."""
    form: str | None = None
    """A `PluginSource.form`, such as `instructions`, `snapshot` or `notice`.
    "Inspect these records by source" filters on it. `None` for a source that
    has no form."""

    def label(self) -> str:
        parts = [part for part in (self.kind, self.name) if part is not None]
        return f"{':'.join(parts)}{f' ({self.form})' if self.form is not None else ''}"


@dataclass(frozen=True, slots=True)
class Timing:
    """What a step cost, derived from event times and the usage that the provider reported.

    `time_to_first_token_ms` is the wait that a user feels.
    `decode_tokens_per_second` is the rate over the rest of the step. Each is
    `None` when the log does not carry enough to give it, and is never guessed.
    An auditor's view that invents a number is worse than one that admits that
    it has none.
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
    """The event that this record projects. It is the join with the transcript."""
    type: str = ""
    """The session-log type that this record projects, verbatim: `workspace/acquired`,
    not `event` (P6-33).

    `kind` is the eight-value vocabulary of the *view*, and a row is colored and
    grouped by it. This field is the vocabulary of the log
    (`KNOWN_SESSION_EVENT_TYPES` in ph-core, plus what each plugin declares),
    which is more than ten times as large. The two are not interchangeable.
    Most types in the log share the single kind `event`, so a reader who wants
    "just the retained worktrees" cannot ask for that in `kind`. The type is
    carried on the record and not derived again from `source_seq`, because this
    view folds a stored log with nothing mounted. The filter must not need the
    log again to answer.
    """
    title: str = ""
    """What the row is called. It is the table's own column, so an `event` reads
    as "turn end" and not as the type that produced it."""
    summary: str = ""
    """One line, for the table."""
    detail: str = ""
    """The full text, for the panel."""
    source: SourceRef = field(default_factory=SourceRef)
    turn: int = 0
    timing: Timing | None = None
    replaced: str = ""
    """For a `system` record: the snapshot that this one replaced. A prompt
    change then reads as a diff and not as a new wall of text (a requirement
    that comes from dsh)."""
    tools: list[str] = field(default_factory=list)
    """Tool *names* as the catalog held them at call time, not as it holds them
    now. These are names and not schemas. The panel shows what was callable.
    Schemas would hold a copy of every catalog for the life of the view, only
    to render a comma-separated list."""
    fork_point: bool = False
    """Whether `ctx.sessions.fork` may aim here (A6). Only closed-turn
    boundaries qualify. The table can therefore show which rows are targets,
    and a user does not aim anywhere and get refused."""


def _source_ref(message: Mapping[str, Any]) -> SourceRef:
    kind, name, form = source_of(message)
    return SourceRef(kind=kind, name=name, form=form)


def _text(blocks: object) -> str:
    """The visible text of wire content. Other blocks are named and not dropped,
    because an auditor wants to see that an image was there."""
    return text_of_wire(blocks, placeholder=block_marker)


@dataclass(slots=True)
class _Builder:
    """Folds a log into records. One pass, no lookahead."""

    records: list[TrajectoryRecord] = field(default_factory=list)
    turn: int = 0
    _step_started: int | None = None
    _first_chunk: int | None = None
    _system: str = ""
    _profile: EnvironmentFold = field(default_factory=EnvironmentFold)
    """The `profile/*` records so far, folded one at a time. This is the
    environment at the record that is being built (S8)."""

    def add(self, **fields: Any) -> TrajectoryRecord:  # noqa: ANN401
        record = TrajectoryRecord(index=len(self.records) + 1, **fields)
        self.records.append(record)
        return record

    # ------------------------------------------------------------ handlers --

    def on_request_header(self, event: SessionEvent) -> None:
        """The snapshot of the prompt and the tool catalog, plus the one that it replaced.

        `request/header` is logged only when it *changed* (A12). So each one is
        a real change, and the previous text is what makes it readable.
        """
        try:
            # This is the model that ph-core already parses the header into, so
            # the wire keys are spelled one time. A renamed key is an error
            # here, and not a silently empty catalog and a `?/?` route.
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
        outer = event.data
        message = message_of(outer)
        source = _source_ref(message)
        text = _text(message.get("content"))
        # A compaction summary shadows the range that it stands for. The record
        # says so, and does not look like an ordinary message that appeared.
        record_kind: RecordKind = "compacted" if is_replacement_surface_event(event) else kind
        if record_kind == "user" and source.kind == "plugin":
            record_kind = "context"
        self.turn = as_int(outer.get("turn")) or self.turn
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
        data = event.data
        name = as_str(data.get("name"), "?")
        arguments = as_str(data.get("arguments"))
        self.add(
            kind="tool",
            source_seq=event.seq,
            title=name,
            summary=one_line(arguments),
            detail=arguments,
            source=SourceRef(kind="tool", name=name),
            turn=self.turn,
        )

    def on_tool_result(self, event: SessionEvent) -> None:
        message = as_obj(event.data.get("message"))
        result = result_block(message)
        _kind, call_id, _form = source_of(message)
        text = _text(result.get("content"))
        self.add(
            kind="tool",
            source_seq=event.seq,
            title=f"result {call_id}" if call_id is not None else "result",
            summary=one_line(text),
            detail=text,
            source=SourceRef(kind="tool", name=call_id),
            turn=self.turn,
        )

    def on_sub_dispatch(self, event: SessionEvent) -> None:
        """A Code Mode sub-call: `subtool`, the kind that C2 exists to make visible."""
        data = event.data
        name = as_str(data.get("name"), "?")
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

    def on_event(
        self, event: SessionEvent, title: str, summary: str, detail: str | None = None
    ) -> None:
        """A harness fact with no conversational kind. Its detail is its summary
        unless it has more to say."""
        self.add(
            kind="event",
            source_seq=event.seq,
            title=title,
            summary=summary,
            detail=summary if detail is None else detail,
            source=SourceRef(kind="harness", name=event.type),
            turn=self.turn,
        )

    # -------------------------------------------------------------- timing --

    def _close_timing(self, event: SessionEvent, message: Mapping[str, Any]) -> Timing:
        """What the step cost, from event times and the usage that the provider reported.

        The numbers are derived and never measured here. The view must produce
        the same numbers on a replayed log as on a live one, and a clock that
        is read at render time would not (A11).
        """
        usage = as_obj(message.get("usage"))
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
    builder.turn = as_int(event.data.get("turn"))
    builder.on_event(event, "turn start", f"turn {builder.turn}")


def _on_turn_end(builder: _Builder, event: SessionEvent) -> None:
    reason = as_obj(event.data.get("reason")).get("kind") or "completed"
    builder.on_event(event, "turn end", f"turn {builder.turn} — {reason}")


def _on_supervisor_violated(builder: _Builder, event: SessionEvent) -> None:
    """Invariants stopped holding, or started to hold again (I6).

    **One type, two opposite facts.** The generic reading cannot carry that.
    `verify_root` records the clearing as the same type with an empty list, so
    `_describe` rendered the good news as `violations=[], pid=9`. A reader has
    no way to tell that this is reassurance. The bad news arrived as a Python
    dict repr in a document that a person reads.

    This is the second renderer of this record, and that is why it needed its
    own entry and not the fallback. `ph_app.tui.adapter` draws the same two
    transitions for the terminal. A transcript that disagrees with the session
    that it transcribes is the failure that `ph.text` exists to prevent.

    The ids go in the summary and the details go in the `detail` line. The
    view of the auditor is where the node counts belong, and they are the half
    that a reader who is chasing the drift needs.
    """
    violations = [as_obj(one) for one in as_seq(event.data.get("violations"))]
    broken = [as_str(one.get("invariant")) for one in violations]
    if not broken:
        builder.on_event(event, "invariants", "hold again")
        return
    builder.add(
        kind="event",
        source_seq=event.seq,
        title="invariants",
        summary=f"{count_of(len(broken), 'invariant')} violated: {', '.join(broken)}",
        detail="\n".join(f"{one.get('invariant')}: {one.get('detail')}" for one in violations),
        source=SourceRef(kind="harness", name=event.type),
        turn=builder.turn,
    )


def _on_harness_event(builder: _Builder, event: SessionEvent) -> None:
    """A harness fact with no conversational kind, rendered from its payload."""
    builder.on_event(event, event.type, _describe(event))


def _on_profile_record(builder: _Builder, event: SessionEvent) -> None:
    """A change to the environment, and the environment that it leads to (S8, item 11).

    The detail is the environment at this seq: the base in force and the
    overrides logged up to here, folded as a restart would fold them. So an
    auditor reads what the agent ran in from this point, and not only what
    changed.
    """
    builder._profile.step(event.seq, event.type, event.data)
    summary = record_summary(event.type, event.data)
    listing = environment_listing(builder._profile.environment)
    builder.on_event(event, event.type, summary, "\n".join([summary, "", *listing]))


def _on_skill_read(builder: _Builder, event: SessionEvent) -> None:
    """Which text of a skill an agent read, by hash (S8, decision 12)."""
    builder.on_event(event, event.type, f"read skill {read_summary(event.data)}")


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
    # Every remaining known type that draws a transcript row renders
    # generically, and each one is listed and not caught by an `else`. The
    # `else` made the completeness gate unfalsifiable. With a catch-all, a type
    # that was added to the vocabulary silently got a generic row, and nobody
    # decided whether it deserved a kind of its own. Explicit keys make the
    # *set* checkable. `adapter.py` uses the same shape for the same reason.
    #
    # A type that the vocabulary declares an auditor's record (`is_audit_only`)
    # needs no entry. It renders generically unless it has an entry in this
    # table. That decision is made where the type is declared, and it is not a
    # catch-all, so a type that is neither listed nor declared still fails the
    # gate.
    "request/context": _on_harness_event,
    "approval/asked": _on_harness_event,
    "approval/decided": _on_harness_event,
    "shell/command": _on_harness_event,
    "shell/result": _on_harness_event,
    "question/asked": _on_harness_event,
    "question/answered": _on_harness_event,
    "workspace/restored": _on_harness_event,
    # The environment that the session started in (session profiles, S3). It is
    # the first thing that an audit of a run asks, and a conversation shows
    # none of it.
    "profile/base": _on_profile_record,
    # And each deviation from it (S4): what asked, and what changed.
    "profile/override": _on_profile_record,
    # And what a person decided about a named profile that moved (S6).
    "profile/adopted": _on_profile_record,
    "profile/declined": _on_profile_record,
    "profile/override-cleared": _on_profile_record,
    # And an adopted version a start could not mount, taken back with why.
    "profile/withdrawn": _on_profile_record,
    # And a change the log could not hold, refused and so never in force.
    "profile/refused": _on_profile_record,
    # A named profile written from this session's environment (S7).
    "profile/saved": _on_profile_record,
    # A skill's body read at runtime, hashed (S8, decision 12).
    "skill/read": _on_skill_read,
    "sandbox/denied": _on_harness_event,
    "command/run": _on_harness_event,
    "command/done": _on_harness_event,
    "llm/retry": _on_harness_event,
    # These are records and never `RECORDLESS`. A reader that skipped them
    # would find a transcript that resumes mid-thought with no account of why,
    # and a session that stopped working would look like one that only went
    # quiet. For an unattended run, that is the purpose of reading the trace.
    "supervisor/retry": _on_harness_event,
    "supervisor/failed": _on_harness_event,
    "supervisor/recovered": _on_harness_event,
    "supervisor/passivated": _on_harness_event,
    "supervisor/unreachable": _on_harness_event,
    "supervisor/violated": _on_supervisor_violated,
    # Records for the same reason (T5). Without them, a child that paused on a
    # missing key reads the same as one that only went quiet.
    "credential/needed": _on_harness_event,
    "credential/supplied": _on_harness_event,
    "schedule/tick": _on_harness_event,
    "goal/set": _on_harness_event,
    "goal/settled": _on_harness_event,
    # A record, not `RECORDLESS`. An auditor who reads a transcript needs to
    # know where the work of another run ends and the work of this run begins.
    # The auditor also needs to know that the repair, and not the model, may
    # have closed the turn above the seam.
    "session/resumed": _on_harness_event,
    # The mirror of the line above, and a record for the mirrored reason. The
    # seam at the *end* of a log is as invisible as the one at the start. An
    # auditor who reaches the last row of a segment has reached the end of a
    # file and not the end of the work.
    "session/segmented": _on_harness_event,
    "kernel/restored": _on_harness_event,
    "harness/refined": _on_harness_event,
    "harness/refine-considered": _on_harness_event,
    "context/loaded": _on_harness_event,
    "agent/inbox/spliced": _on_harness_event,
    "todo/write": _on_harness_event,
    "offload/spilled": _on_harness_event,
    "offload/input-spilled": _on_harness_event,
    # Every compaction attempt that decided not to compact, and the arguments
    # that one elided. What a landed summary shadowed and cost is
    # `compaction/summarized`, an auditor's record that the vocabulary declares.
    "compaction/declined": _on_harness_event,
    "compaction/args-truncated": _on_harness_event,
    "attachment/degraded": _on_harness_event,
    "attachment/oversized": _on_harness_event,
    "attachment/uploaded": _on_harness_event,
    "limits/exceeded": _on_harness_event,
    "limits/breaker-tripped": _on_harness_event,
}
"""Event type → the record that it produces. This covers every type except the
auditor's records and the unknown types that render generically (`_handler_of`).

The table is explicit. So `set(HANDLERS) | RECORDLESS`, with the types that the
vocabulary declares audit-only, is a value that a test can compare with the
vocabulary. That is the only form of "no silent omissions" that fails when a
type is added."""


def _handler_of(event_type: str) -> Handler | None:
    """The handler that the events of a type go to.

    A type with an entry in `HANDLERS` gets that entry. The generic handler
    takes a type that the vocabulary declares as read only by an auditor, and a
    type that this process does not know (the module docstring says why). A
    `RECORDLESS` type gets none."""
    handler = HANDLERS.get(event_type)
    if handler is None and (is_audit_only(event_type) or not is_known(event_type)):
        return _on_harness_event
    return handler


def build_trajectory(session: Session) -> list[TrajectoryRecord]:
    """Fold a log into the auditor's records.

    The argument is a `Session`. A *stored* log becomes one through
    `ph.persistence.read_session` with nothing mounted, and the harness-free
    entry point of P3-25 depends on that property.

    Fork points are decided against `fork_boundaries`, which is the rule of the
    store itself. The rows that this marks are therefore exactly the rows that
    `ctx.sessions.fork` accepts. The boundaries are asked for one time for the
    whole log and not one time for each record. The per-record form scanned the
    prefix again each time. That is quadratic, and it froze the UI when P4-17
    put this fold on a keypress.
    """
    builder = _Builder()
    log = session.events
    boundaries = fork_boundaries(log)
    for event in log:
        handler = _handler_of(event.type)
        if handler is None:
            # Record-less. Two of these types are still *read*: a step opens
            # the clock, and the first chunk marks time-to-first-token. That is
            # why neither is a record.
            if event.type == "step/start":
                builder._step_started = event.time
                builder._first_chunk = None
            elif event.type == "assistant/chunk" and builder._first_chunk is None:
                builder._first_chunk = event.time
            continue
        before = len(builder.records)
        handler(builder, event)
        # **Stamp the records that this handler added.** That is not always
        # exactly one: `on_request_header` returns without adding a record when
        # the payload does not parse. `records[-1]` therefore stamped a
        # *previous, unrelated* record with the fork status of this event. On a
        # log whose first record-producing event was that one, it raised an
        # `IndexError`. A slice is correct for zero, one and many, which are
        # the cases that the handler table contains.
        for record in builder.records[before:]:
            record.type = event.type
            record.fork_point = event.seq in boundaries
    return builder.records


def _describe(event: SessionEvent) -> str:
    """A one-line account of a harness event, from its own payload.

    The generic reading itself is now in `ph_app.wire`. `phern agents attach`
    needs the same fallback for the same reason, and it cannot import anything
    under `ph_app.tui` without paying for Textual. What stays here is the type
    name, which is the answer of this view for a payload with nothing in it.
    """
    return describe(event.data) or event.type
