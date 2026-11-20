"""The state of the Continual Harness: a fold over events, at two scopes (P3-16, D14).

Prime-agent keeps this state in a JSON file that it reads and writes. pH keeps
it as a **fold over an append-only log**. That one change gives three properties
that the file could not express:

* **Fork and resume need no extra work.** `ctx.sessions.fork(source, boundary)`
  folds only the `harness/*` events at or before the boundary, so a fork
  inherits the harness as it was *then*. A file gives every fork the latest
  state of the parent.
* **Rollback is derivable.** Each apply records its own before and after
  snapshots. The inverse of a refinement is therefore a function of its event
  (H6), and not a second file that someone must keep.
* **There is one writer and no reload rule.** A file that the host reads and the
  model writes needs an mtime guard and a conflict policy. A log needs neither.

**Two scopes, one shape.** Local state folds the events of this session. Global
state folds `$PH_HOME/harness/events.jsonl`, which is a separate append-only log
with the same records. So "state is a fold over an append-only log" holds at
both scopes, and the global scope does not bring a file back one level up.

**One event type, not two.** The plan describes a fold of `harness/refined` and
`harness/rolled-back`. A rollback is an inverse that is *applied*, and the
refined record already carries `rollbackOf`. Two types would mean two fold cases
that must agree about one operation.

@module ph_rlm.harness.state
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

from pydantic import Field

from ph.persistence import read_records
from ph.session import Session, SessionFoldCache
from ph.wire import WireModel

__all__ = [
    "GLOBAL_LOG_NAME",
    "KINDS",
    "PROJECTION_NAME",
    "REFINED",
    "RESERVED_IDS",
    "AppliedEdit",
    "HarnessEdit",
    "HarnessEntry",
    "HarnessKind",
    "HarnessReference",
    "HarnessScope",
    "HarnessState",
    "RefinementProposal",
    "RefinementRecord",
    "entry_label",
    "extend_session",
    "fold_events",
    "fold_session",
    "read_global_events",
    "refinement_line",
]

REFINED = "harness/refined"
"""The one durable record of a refinement, rollbacks included."""

GLOBAL_LOG_NAME = "events.jsonl"
PROJECTION_NAME = "harness_state.json"

HarnessScope: TypeAlias = Literal["local", "global"]
HarnessKind: TypeAlias = Literal["note", "procedure", "skill"]
"""What a refinement may write.

The set is closed, and on purpose no kind is a capability. A `note` is something
learned. A `procedure` is how to do something with what already exists. A
`skill` *points at* installed capability. Invariant I7 says that `/refine`
writes procedure and never capability. That is why skills and the harness share
a word but not a mechanism (Q13)."""

KINDS: tuple[HarnessKind, ...] = get_args(HarnessKind)
"""The kinds in rendering order. The tuple is derived from the type, so a
renderer cannot list a kind that the fold does not hold."""

RESERVED_IDS: frozenset[str] = frozenset({"base_system_prompt"})
"""Ids that a refinement may not touch (H5).

The harness may not rewrite the doctrine. Prime-agent learned this rule by
experience. The validator refuses the id with the wording of prime-agent."""


class HarnessReference(WireModel):
    """What an entry points at, so H1 can check that it exists."""

    type: Literal["python"] = "python"
    module: str
    callable: str

    def probe(self) -> str:
        """A program that fails unless this reference resolves.

        The program runs in the *runtime that the model uses*. It is not checked
        against this process, because an entry is true only if the kernel can
        reach it. The program is one statement, and it binds nothing.
        """
        return (
            f"assert callable(getattr(__import__('importlib').import_module({self.module!r}), "
            f"{self.callable!r})), {self.callable!r} + ' is not callable'\n"
        )


class HarnessEntry(WireModel):
    """One thing the harness knows."""

    kind: HarnessKind
    id: str
    title: str
    content: str
    version: int = 1
    scope: HarnessScope = "local"
    path: str | None = None
    reference: HarnessReference | None = None
    call_pattern: str | None = None
    """How the model should invoke what this entry describes (H2).

    The service renders this text (`HarnessService.render_call_pattern`) and
    never accepts it from a proposal. A refinement that could write its own call
    pattern could steer the model onto the ungoverned raw-namespace path, and C2
    exists to close that path."""
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessEdit(WireModel):
    """One change a refinement proposes."""

    action: Literal["create", "update", "delete"]
    kind: HarnessKind
    id: str | None = None
    title: str = ""
    content: str = ""
    path: str | None = None
    reference: HarnessReference | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class RefinementProposal(WireModel):
    """What a planner proposes, before anything is validated or applied."""

    summary: str
    rationale: str = ""
    expected_outcome: str = ""
    edits: list[HarnessEdit] = Field(default_factory=list)


class AppliedEdit(WireModel):
    """One applied change, with the snapshots that rollback is derived from (H6)."""

    action: Literal["create", "update", "delete"]
    kind: HarnessKind
    id: str
    before: HarnessEntry | None = None
    after: HarnessEntry | None = None


class RefinementRecord(WireModel):
    """A refinement as the fold sees it."""

    refine_id: str
    scope: HarnessScope
    summary: str
    rationale: str = ""
    expected_outcome: str = ""
    """With `rationale`, the stated case of the planner. Both are recorded so
    that the event carries the proposal that produced it and not only its
    effects."""
    applied_edits: list[AppliedEdit] = Field(default_factory=list)
    rollback_of: str | None = None
    rejected: list[str] = Field(default_factory=list)
    """The edits that validation refused, each with the reason. They are
    recorded on the event so that a rejection can be audited and is not a silent
    no-op (H1)."""
    approved_in: str | None = None
    """For a global refinement, the session whose log holds the approval that
    allowed it. The `approval/decided` event of that session carries the
    `refine_id` of this record as `callId` (L4). The value is `None` for a local
    refinement, whose record is in its session."""


class HarnessState(WireModel):
    """The folded state: the entries by kind, and the refinements that produced them."""

    schema_version: int = 1
    entries: dict[str, dict[str, HarnessEntry]] = Field(default_factory=dict)
    refinements: list[RefinementRecord] = Field(default_factory=list)

    def entry(self, kind: HarnessKind, entry_id: str) -> HarnessEntry | None:
        return self.entries.get(kind, {}).get(entry_id)

    def of_kind(self, kind: HarnessKind) -> list[HarnessEntry]:
        """Entries of one kind, in id order so a prompt is byte-stable (A12)."""
        return [self.entries[kind][key] for key in sorted(self.entries.get(kind, {}))]

    def merged_with(self, other: HarnessState) -> HarnessState:
        """This state layered over `other`: local over global.

        A local entry shadows a global entry of the same kind and id. The
        session that learned something specific must not be overruled by a
        deployment-wide note.
        """
        entries: dict[str, dict[str, HarnessEntry]] = {
            kind: dict(rows) for kind, rows in other.entries.items()
        }
        for kind, rows in self.entries.items():
            entries.setdefault(kind, {}).update(rows)
        return HarnessState(entries=entries, refinements=[*other.refinements, *self.refinements])


def entry_label(entry: HarnessEntry, detail: str = "") -> str:
    """`[scope:id] title (vN)`: one identity for every rendering of an entry.

    The label is shared because the planner tells the model to use the exact id
    of an entry, and the model reads that id in one rendering while the prompt
    section shows another. If the id format differed between the two, the model
    would edit entries that do not exist.
    """
    return f"[{entry.scope}:{entry.id}] {entry.title} (v{entry.version}{detail})"


def refinement_line(record: RefinementRecord) -> str:
    """One refinement, as the prompt section and the planner history both show it."""
    return f"- [{record.refine_id}] {record.summary}"


def _apply_record(state: HarnessState, record: RefinementRecord) -> None:
    """Fold one refinement into `state`, in place."""
    for edit in record.applied_edits:
        rows = state.entries.setdefault(edit.kind, {})
        if edit.after is None:
            rows.pop(edit.id, None)
        else:
            rows[edit.id] = edit.after
        if not rows:
            state.entries.pop(edit.kind, None)
    state.refinements.append(record)


def _fold_into(state: HarnessState, payloads: Iterable[Mapping[str, Any]]) -> HarnessState:
    for payload in payloads:
        try:
            record = RefinementRecord.model_validate(payload)
        except Exception:
            # A record that this build cannot read is skipped and is not fatal.
            # If one refinement from a newer pH made the fold refuse the whole
            # harness, the rest of the harness would be lost.
            continue
        _apply_record(state, record)
    return state


def fold_events(events: Iterable[Mapping[str, Any]]) -> HarnessState:
    """The state that a sequence of refinement payloads folds to.

    The function takes payloads and not `SessionEvent`s. The records of the
    global scope come from a JSONL file and must fold through the same code. Two
    implementations of one fold are the disagreement that A11 forbids.
    """
    return _fold_into(HarnessState(), events)


def fold_session(session: Session) -> HarnessState:
    """Local state: the fold over this session's own `harness/refined` events."""
    return fold_events([event.data for event in session.events if event.type == REFINED])


def extend_session(state: HarnessState, session: Session, from_seq: int) -> HarnessState:
    """`fold_session`, resumed from a prefix that is already folded.

    This is the contract that `SessionFoldCache` requires: extending the fold of
    a prefix equals folding the whole log. Almost every appended event is a
    chunk, so the scan of the new slice usually finds nothing and returns the
    same state object. That also lets `HarnessService.state` treat identity as
    "nothing changed". When the slice does hold refinements, the fold runs on a
    copy, so a holder of the previous value keeps what it read.
    """
    fresh = [event.data for event in session.events[from_seq:] if event.type == REFINED]
    if not fresh:
        return state
    extended = HarnessState(
        entries={kind: dict(rows) for kind, rows in state.entries.items()},
        refinements=list(state.refinements),
    )
    return _fold_into(extended, fresh)


def read_global_events(directory: Path) -> list[dict[str, Any]]:
    """The payloads of the global log, in order. A missing log is an empty harness.

    `read_records` holds the tolerance and the reason for it: a torn last line
    is a crash in the middle of an append, and the refinements before it are
    still sound. The orphan journal has the same shape, and it now shares the
    rule and does not restate it.
    """
    return list(read_records(directory / GLOBAL_LOG_NAME))


def local_fold_cache() -> SessionFoldCache[HarnessState]:
    """The per-session cache that the prompt reads through.

    The cost of the fold grows with the length of the log, and the prompt asks
    for the fold on every model step. `session.seq` increases on every event,
    chunks included, so the key alone would miss on nearly every read.
    `extend_session` makes that miss cheap: one scan of the new slice, and not a
    new fold of the whole log. `fold_session` stays a pure function that a fork
    slice or a stored log can use.
    """
    return SessionFoldCache(fold_session, extend=extend_session)
