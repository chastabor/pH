"""The Continual Harness state: a fold over events, at two scopes (P3-16, D14).

Prime Agent keeps this in a JSON file it reads and writes. pH keeps it as a
**fold over an append-only log**, and that single change is what buys three
properties the file could not express:

* **Fork and resume come free.** `ctx.sessions.fork(source, boundary)` folds only
  the `harness/*` events at or before the boundary, so a fork inherits the
  harness as it was *then*. A file hands every fork the parent's latest.
* **Rollback is derivable.** Each apply records its own before/after snapshots,
  so the inverse of a refinement is a function of its event (H6) rather than a
  second file someone has to keep.
* **There is one writer and no reload rule.** A file read by the host and written
  by the model needs an mtime guard and a conflict policy; a log has neither.

**Two scopes, one shape.** Local state folds this session's own events. Global
state folds `$PH_HOME/harness/events.jsonl` — its own append-only log with the
same records, so "state is a fold over an append-only log" holds at both scopes
rather than putting a file back one level up.

**One event type, not two.** The plan describes folding `harness/refined` and
`harness/rolled-back`; a rollback is an inverse proposal *applied*, and the
refined record already carries `rollbackOf`. Two types would mean two fold cases
that have to agree about one operation.

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

A closed set, and deliberately none of them a capability: a `note` is something
learned, a `procedure` is how to do something with what already exists, and a
`skill` *points at* installed capability. `/refine` writing procedure and never
capability is invariant I7 — the reason skills and the harness share a word but
not a mechanism (Q13)."""

KINDS: tuple[HarnessKind, ...] = get_args(HarnessKind)
"""The kinds in rendering order — derived from the type, so a renderer cannot
enumerate a kind the fold does not hold."""

RESERVED_IDS: frozenset[str] = frozenset({"base_system_prompt"})
"""Ids a refinement may not touch (H5).

The doctrine is not the harness's to rewrite. Prime Agent learned this one the
hard way; the validator refuses the id with its wording."""


class HarnessReference(WireModel):
    """What an entry points at, so H1 can check that it exists."""

    type: Literal["python"] = "python"
    module: str
    callable: str

    def probe(self) -> str:
        """A program that fails unless this reference resolves.

        Run in the *runtime the model actually uses* rather than checked against
        this process: an entry is only true if the kernel can reach it.
        """
        return (
            f"import importlib\n"
            f"_m = importlib.import_module({self.module!r})\n"
            f"_c = getattr(_m, {self.callable!r})\n"
            f"assert callable(_c), {self.callable!r} + ' is not callable'\n"
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

    Rendered by the validator, never accepted from a proposal: a refinement that
    could write its own call pattern could steer the model onto the ungoverned
    raw-namespace path, which is the whole thing C2 exists to close."""
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
    """One applied change, with the snapshots rollback is derived from (H6)."""

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
    """With `rationale`: the planner's stated case, recorded so the event
    carries the proposal that produced it and not only its effects."""
    applied_edits: list[AppliedEdit] = Field(default_factory=list)
    rollback_of: str | None = None
    rejected: list[str] = Field(default_factory=list)
    """Edits refused by validation, with the reason — recorded on the event so a
    rejection is auditable rather than a silent no-op (H1)."""


class HarnessState(WireModel):
    """The folded state: entries by kind, and what got it here."""

    schema_version: int = 1
    entries: dict[str, dict[str, HarnessEntry]] = Field(default_factory=dict)
    refinements: list[RefinementRecord] = Field(default_factory=list)

    def entry(self, kind: HarnessKind, entry_id: str) -> HarnessEntry | None:
        return self.entries.get(kind, {}).get(entry_id)

    def of_kind(self, kind: HarnessKind) -> list[HarnessEntry]:
        """Entries of one kind, in id order so a prompt is byte-stable (A12)."""
        return [self.entries[kind][key] for key in sorted(self.entries.get(kind, {}))]

    def merged_with(self, other: HarnessState) -> HarnessState:
        """This state layered over `other` — local over global.

        A local entry shadows a global one of the same kind and id, because the
        session that learned something specific should not be overruled by a
        deployment-wide note.
        """
        entries: dict[str, dict[str, HarnessEntry]] = {
            kind: dict(rows) for kind, rows in other.entries.items()
        }
        for kind, rows in self.entries.items():
            entries.setdefault(kind, {}).update(rows)
        return HarnessState(entries=entries, refinements=[*other.refinements, *self.refinements])


def entry_label(entry: HarnessEntry, detail: str = "") -> str:
    """`[scope:id] title (vN)` — one identity for every rendering of an entry.

    Shared because the planner tells the model to edit "using its exact id"
    against one rendering while the prompt section shows another; an id format
    that drifted between them would have the model editing entries that do not
    exist.
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
            # A record this build cannot read is skipped rather than fatal: the
            # type is ignorable, and refusing the whole harness because one
            # refinement came from a newer pH would lose the rest of it.
            continue
        _apply_record(state, record)
    return state


def fold_events(events: Iterable[Mapping[str, Any]]) -> HarnessState:
    """The state a sequence of refinement payloads folds to.

    Takes payloads rather than `SessionEvent`s, because the global scope's
    records come off a JSONL file and must fold through the same code — two
    implementations of one fold is the disagreement A11 forbids.
    """
    return _fold_into(HarnessState(), events)


def fold_session(session: Session) -> HarnessState:
    """Local state: the fold over this session's own `harness/refined` events."""
    return fold_events([event.data for event in session.events if event.type == REFINED])


def extend_session(state: HarnessState, session: Session, from_seq: int) -> HarnessState:
    """`fold_session`, resumed from an already-folded prefix.

    The contract `SessionFoldCache` requires: extending the fold of a prefix
    equals folding the whole log. Almost every appended event is a chunk, so the
    scan of the new slice usually finds nothing and hands back the same state
    object — which is also what lets `HarnessService.state` treat identity as
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
    """The global log's payloads, in order. A missing log is an empty harness.

    `read_records` carries the tolerance and the reason for it: a torn last line
    is a crash mid-append, and the refinements before it are still sound. The
    orphan journal is the same shape and now shares the rule rather than
    restating it.
    """
    return list(read_records(directory / GLOBAL_LOG_NAME))


def local_fold_cache() -> SessionFoldCache[HarnessState]:
    """The per-session cache the prompt reads through.

    The fold is O(log) and the prompt asks for it every model step — and
    `session.seq` bumps on every event, chunks included, so the key alone would
    miss on nearly every read. `extend_session` is what makes that miss cheap:
    one scan of the new slice instead of a refold of the whole log, with
    `fold_session` staying a pure function a fork slice or stored log can use.
    """
    return SessionFoldCache(fold_session, extend=extend_session)
