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

**Keys stay in the payloads.** A kind reads its key off its records the way each
fold did before it: `callId or toolName`, `askId`, the command id. Those fields are
data their readers already use, and a second copy on the envelope would be a
second carrier of one fact.

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
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..json import JsonObject
from .events import SessionEvent, is_surface_eligible_type
from .known_event_types import is_known

__all__ = [
    "Barrier",
    "IntentError",
    "IntentKind",
    "IntentOrphan",
    "IntentRecord",
    "OpenIntent",
    "Unsettled",
    "declare_intent",
    "declared_intents",
    "extend_index",
    "fold_intents",
    "is_declared",
    "key_of",
    "open_intents",
    "settled_record",
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
    orphan it is told to settle. `owner` is the module that declares and writes
    the pair, for the same reason `declare_log_type` asks for one.
    """

    opened: str
    settled: str
    opened_key: Callable[[SessionEvent], str | None]
    settled_key: Callable[[SessionEvent], str | None]
    orphan: IntentOrphan
    barrier: Barrier = "durable"
    closer: Callable[[SessionEvent, Unsettled], JsonObject] | None = None
    owner: str = ""
    dedupe: bool = True
    """Whether a key names one act for the life of the log, so opening it again
    is answered with the `Prior` rather than done again. False for a kind whose
    key is reused on purpose — an approval keyed by call id *or tool name*, a
    question re-posed under its own id after a resume — where a second open is a
    new intent that replaces the first, as the fold already reads it. Such a kind
    also cannot tell two concurrent intents under one key apart, so the journal
    holds its settles to the key alone (`IntentJournal.settle`)."""


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
    """Declare a pair. Call at import, beside the code that writes it.

    Refused:

    * a type the vocabulary does not know — a pair this build could write and
      refuse to read back is F11 one level up;
    * a second kind opened by the same type — one record, one meaning;
    * a kind repair is told to settle with no `closer` to settle it with;
    * the same type opening and settling — no record can be both;
    * a settle the model sees (`SURFACE_EVENT_TYPES`) — it needs a surface placement
      the journal does not write and provider rules repair keeps for the turn it
      closes, so such a pair belongs to the turn repair, not to a kind.

    Declaring the same kind twice returns the first, so a module imported twice
    does not refuse itself.
    """
    for name in (kind.opened, kind.settled):
        if not is_known(name):
            raise IntentError(f'"{name}" is not a session event type this build can read back')
    if kind.opened == kind.settled:
        raise IntentError(f'"{kind.opened}" cannot both open and settle an intent')
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
