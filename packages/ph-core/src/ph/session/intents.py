"""Intents: an act the log records before it happens, and settles after.

Seven pairs already work this way — `approval/asked` and `approval/decided`,
`question/asked` and `question/answered`, `shell/command` and `shell/result`,
`tool/call` and `tool/result`, a Code Mode dispatch and its result, a daemon verb
and its outcome, a workspace tree acquired and disposed. Each has its own fold of
"opened and never settled", and three different places settle the orphans a crash
leaves. Each fold is correct; what was missing is **one statement** of the rule
that repair, the dedupe index and every reader share.

This module is that statement, and nothing else: declarations and pure functions
over a sequence of events. No service — `ctx.intents` (P10-06) is where appending,
flushing and deduping live. Pure for `folds.py`'s reason: repair folds a log while
it is being rebuilt, and the trajectory view folds a stored one with nothing
mounted, so a fold attached to a live `Session` could answer neither.

**Keys stay in the payloads.** A kind reads its key off its records: the opening
record's own seq for the asks and the shell (whose settles carry it back as
`askSeq` and `commandSeq`, T3), a dispatch's `subCallId`, an effect's key, a
daemon verb's command id. Those fields are data their readers already use, and a
second copy on the envelope would be a second carrier of one fact.

**Kinds are declared in leaves** (T4): ph-core's in `ph.session.kinds`, which this
package imports, and a package's in its own `kinds`, which that package imports.

**One fold, `fold_intents`**: per key, the latest intent opened under it and the
settle that closed it. `open_intents` and `settled_record` are both read off it, so
"open" and "settled" cannot come to mean two different things. Its rule is the one
`pending_approvals` and `pending_questions` already had, so moving them onto it
changes nothing a reader sees: an open of a key replaces the intent before it (the
later ask is the live one), a settle closes only its own key, a record whose key
cannot be read is not part of the pair, and open intents come back in the order
they were opened.

@module ph.session.intents
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Literal, TypeAlias

from ..json import JsonObject, as_obj, as_str
from ..wire import literal_lookup
from .events import SessionEvent, is_surface_eligible_type
from .known_event_types import is_known
from .writers import LogWriter

__all__ = [
    "UNSETTLED",
    "Barrier",
    "IntentError",
    "IntentKind",
    "IntentOrphan",
    "IntentRecord",
    "OpenIntent",
    "Outcome",
    "SettledBy",
    "Unsettled",
    "abandoned",
    "declare_intent",
    "declared_intents",
    "extend_index",
    "fold_intents",
    "is_declared",
    "key_of",
    "open_intents",
    "outcome_of",
    "settled_record",
    "unsettled",
    "unsettled_why",
]

Unsettled: TypeAlias = Literal["outcome-unknown", "not-started"]
"""What a settle written by anyone but the act itself says: that it may have
happened, or that it cannot have. Passed to a kind's `closer`."""

IntentOrphan: TypeAlias = Literal[Unsettled, "owner-settles"]
"""What an intent a crash left open means, and so who settles it (decision 2).

* `outcome-unknown` — the act may have happened. Repair writes the settle saying
  so, which is the one honest thing a log can say about an effect it did not see
  finish: a shell command, a daemon verb.
* `not-started` — the act cannot have happened without its settle: the process
  that would have acted died holding the question. Repair writes it as not
  started: an approval nobody answered, a question nobody saw.
* `owner-settles` — repair leaves it. The owning seam can *look* — a workspace
  tree is on disk or it is not — and reconciles on open, as
  `WorkspaceSeam.reconcile` already does. A guess from repair would be worse
  than the owner's look.
"""

Barrier: TypeAlias = Literal["durable", "buffered", "tools-execute"]
"""When the opening record must be on disk (decision 3).

* `durable` — before the act: the journal flushes before handing out the claim,
  and refuses the claim if the flush fails.
* `buffered` — whenever the next flush happens: accounting, where losing the
  record in a crash loses a line of history and never an act.
* `tools-execute` — the checkpoint policy's barrier before tools execute, which
  must run *after* every pre-execute gate. The two tool kinds declare it so the
  journal does not place a second one; it is placed where it already is.
"""


Outcome: TypeAlias = Literal["done", "failed", "outcome-unknown", "not-started"]
"""What became of an intent, read the same way by every consumer (T2).

`done` — the act settled it. `failed` — the act settled it and said it failed, so a
retry is its own decision (`IntentKind.failed`). `outcome-unknown` and `not-started`
— somebody other than the act settled it, and these say which half is known
(`Unsettled`)."""

SettledBy: TypeAlias = Literal["repair", "process"]
"""Who wrote a settle the act did not: `repair`, on resume, for an intent a dead
process left open; or the live `process` on the act's behalf — a barrier that
failed, a body that raised or was canceled."""

UNSETTLED = "unsettled"
"""The payload field every settle **not written by the act** carries, and nothing
else does: `{"unsettled": {"why": <Unsettled>, "by": <SettledBy>}}`.

One field for every kind, written by the journal and repair rather than by each
closer, so "did this finish?" has one reader (`outcome_of`) instead of the four
spellings Phase 10 grew — `interrupted: why`, `interrupted: true`,
`outcome: "unknown"`, `failed: true`. A closer still writes its kind's own payload
(an approval's `outcome`, a question's `resolution`), which readers of that kind
use; this is the part every reader of every kind shares."""


_UNSETTLED_WHY: Mapping[str, Unsettled] = literal_lookup(Unsettled)


def unsettled(why: Unsettled, by: SettledBy) -> JsonObject:
    """The marker, ready to merge into a closer's payload."""
    return {UNSETTLED: {"why": why, "by": by}}


def unsettled_why(data: JsonObject) -> Unsettled | None:
    """The marker's reason, or `None` for a settle the act wrote itself."""
    return _UNSETTLED_WHY.get(as_str(as_obj(data.get(UNSETTLED)).get("why")))


class IntentError(ValueError):
    """A kind that cannot be declared as stated."""


@dataclass(frozen=True, slots=True)
class IntentKind:
    """One pair: the record that opens an intent, and the one that settles it.

    `closer(opened, why)` builds a settle payload from the opening record for a
    settle the act did not write: repair's for an orphan (`why` is the kind's
    `orphan`), and the journal's for an intent whose barrier failed
    (`"not-started"`) or whose body raised (`"outcome-unknown"`). It must carry
    the key where `settled_key` reads it. Required unless the kind is
    `owner-settles`, since repair must be able to write *something* for every
    orphan it is told to settle.

    `writer` is the declaring leaf's own (`log_writer(__name__)`), and the journal
    writes the pair through it (T6): so a kind is declared only by a module that is
    the writer of record for both of its types, and the journal writes nothing a
    kind's declarer could not.
    """

    opened: str
    settled: str
    opened_key: Callable[[SessionEvent], str | None]
    settled_key: Callable[[SessionEvent], str | None]
    orphan: IntentOrphan
    barrier: Barrier = "durable"
    closer: Callable[[SessionEvent, Unsettled], JsonObject] | None = None
    failed: Callable[[SessionEvent], bool] | None = None
    """Whether the act's own settle says the act failed — read by `outcome_of`. `None`
    for a kind whose act, once it settles, has done what it does: a shell command
    that exits non-zero still ran."""
    reopen: frozenset[Outcome] = frozenset()
    """What a prior may be, for `open` to open the key again rather than answer
    with the prior (T2). A tool effect that failed, or never started, is worth
    another attempt; one that was done is answered from the log; one whose outcome
    is unknown is its caller's to decide (`reconcile` first). Empty — the default —
    answers every prior."""
    _: KW_ONLY
    writer: LogWriter

    @property
    def owner(self) -> str:
        """The module that declares the pair — its writer's owner."""
        return self.writer.owner


def outcome_of(kind: IntentKind, settled: SessionEvent) -> Outcome:
    """What a settle says became of its intent — the one reader of it (T2)."""
    why = unsettled_why(settled.data)
    if why is not None:
        return why
    return "failed" if kind.failed is not None and kind.failed(settled) else "done"


def abandoned(kind: IntentKind, opened: SessionEvent, why: Unsettled, by: SettledBy) -> JsonObject:
    """The settle of an intent its act did not settle: the kind's closer, marked.

    The one spelling of it, for repair (`by="repair"`) and for the journal's own
    settles on an act's behalf (`by="process"`).

    :raises IntentError: when the kind has no closer to write it with.
    """
    if kind.closer is None:
        raise IntentError(f'"{kind.opened}" has no closer, so nothing can settle it but its owner')
    return {**kind.closer(opened, why), **unsettled(why, by)}


@dataclass(frozen=True, slots=True)
class OpenIntent:
    """An intent the log opened and has not settled: its key and its record."""

    key: str
    opened: SessionEvent


@dataclass(frozen=True, slots=True)
class IntentRecord:
    """One key's latest intent: the record that opened it, and its settle, if any.

    A settle under a key nothing opened is not an intent, and is not folded — a Code
    Mode dispatch refused before it started writes one, and there is nothing open for
    it to close.
    """

    opened: SessionEvent
    settled: SessionEvent | None


_KINDS: dict[str, IntentKind] = {}
"""Every declared kind, by the type that opens it. Import-time facts, like
`known_event_types._DECLARED`, so a test that declares one swaps this out."""


def declare_intent(kind: IntentKind) -> IntentKind:
    """Declare a pair. Call at import, in a package's `kinds` leaf (T4).

    Refused:

    * a type the vocabulary does not know — a pair this build could write and
      refuse to read back is F11 one level up;
    * a second kind opened by the same type — one record, one meaning;
    * a kind repair is told to settle with no `closer` to settle it with;
    * the same type opening and settling — no record can be both;
    * a settle the model sees (`SURFACE_EVENT_TYPES`) — it needs a surface placement
      the journal does not write and provider rules repair keeps for the turn it
      closes, so such a pair belongs to the turn repair, not to a kind;
    * a writer that may not write both types (T6) — the journal writes the pair
      through it, so a kind cannot carry writes its declarer could not make.

    Declaring the same kind twice returns the first, so a module imported twice
    does not refuse itself.
    """
    for name in (kind.opened, kind.settled):
        if not is_known(name):
            raise IntentError(f'"{name}" is not a session event type this build can read back')
    if kind.opened == kind.settled:
        raise IntentError(f'"{kind.opened}" cannot both open and settle an intent')
    for name in (kind.opened, kind.settled):
        if not kind.writer.owns(name):
            raise IntentError(
                f'{kind.owner} is not a writer of record for "{name}", so it cannot '
                "declare a kind that writes it"
            )
    if is_surface_eligible_type(kind.settled):
        raise IntentError(
            f'"{kind.settled}" is model-visible; a pair it settles is the turn repair\'s, '
            "not an intent kind"
        )
    if kind.orphan != "owner-settles" and kind.closer is None:
        raise IntentError(
            f'"{kind.opened}" orphans are "{kind.orphan}", so repair settles them and needs a '
            "closer to write the settle with"
        )
    existing = _KINDS.get(kind.opened)
    if existing is not None:
        if existing == kind:
            return existing
        raise IntentError(f'"{kind.opened}" already opens an intent declared by {existing.owner!r}')
    _KINDS[kind.opened] = kind
    return kind


def declared_intents() -> tuple[IntentKind, ...]:
    """Every declared kind, in declaration order — what repair walks (P10-07)."""
    return tuple(_KINDS.values())


def is_declared(kind: IntentKind) -> bool:
    """Whether `kind` is the one declared for its opening type."""
    return _KINDS.get(kind.opened) == kind


def fold_intents(
    events: Iterable[SessionEvent],
    kind: IntentKind,
    since: Mapping[str, IntentRecord] | None = None,
) -> Mapping[str, IntentRecord]:
    """Per key, the latest intent of `kind` this log opened, and how it settled.

    `since` is this fold of an earlier prefix, and `events` the rest of the log
    after it: extending from a prefix equals folding the whole log, which is the
    law `SessionFoldCache`'s `extend` needs and `ph.testing.folds` checks. The
    prefix's value is copied, never changed, and returned as it was when the
    slice holds nothing of this kind. A cache that owns its value and hands it to
    nobody extends it in place instead (`extend_index`), and so pays for the slice
    rather than for every key it holds.
    """
    index: dict[str, IntentRecord] | None = None
    for event in events:
        if event.type == kind.opened or event.type == kind.settled:
            if index is None:
                index = dict(since or {})
            _fold(index, event, kind)
    if index is not None:
        return index
    return {} if since is None else since


def extend_index(
    index: dict[str, IntentRecord], events: Iterable[SessionEvent], kind: IntentKind
) -> None:
    """`fold_intents`, applied to `index` in place — for a cache that owns it."""
    for event in events:
        _fold(index, event, kind)


def _fold(index: dict[str, IntentRecord], event: SessionEvent, kind: IntentKind) -> None:
    """The one rule, for one record: an open replaces, a settle closes its own key."""
    if event.type == kind.opened:
        key = kind.opened_key(event)
        if key is not None:
            index[key] = IntentRecord(opened=event, settled=None)
    elif event.type == kind.settled:
        key = kind.settled_key(event)
        record = None if key is None else index.get(key)
        if key is not None and record is not None:
            index[key] = IntentRecord(opened=record.opened, settled=event)


def key_of(
    read: Callable[[SessionEvent], str | None], event_type: str, data: JsonObject, seq: int
) -> str | None:
    """The key `read` finds in a record not yet in the log, as the fold would read it.

    A shallow probe: key functions read a field or two off `data`, so it is handed
    as given rather than frozen first — the append that follows freezes it once.
    """
    return read(SessionEvent(type=event_type, seq=seq, time=0, data=data))


def open_intents(
    events: Iterable[SessionEvent] | Mapping[str, IntentRecord], kind: IntentKind
) -> tuple[OpenIntent, ...]:
    """The intents of `kind` this log opened and never settled, in opening order.

    Handed a log, or the `fold_intents` of one — the journal's cached index."""
    index = events if isinstance(events, Mapping) else fold_intents(events, kind)
    found = [
        OpenIntent(key=key, opened=record.opened)
        for key, record in index.items()
        if record.settled is None
    ]
    return tuple(sorted(found, key=lambda intent: intent.opened.seq))


def settled_record(
    events: Iterable[SessionEvent] | Mapping[str, IntentRecord], kind: IntentKind, key: str
) -> SessionEvent | None:
    """The settle of the latest intent opened under `key`, or `None`.

    `None` when that intent is still open, and when the key was never used."""
    index = events if isinstance(events, Mapping) else fold_intents(events, kind)
    record = index.get(key)
    return None if record is None else record.settled
