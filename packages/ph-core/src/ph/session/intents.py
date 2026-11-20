"""Intents: an act that the log records before it happens, and settles after.

Pairs of records worked this way before this module existed: `approval/asked`
and `approval/decided`, `question/asked` and `question/answered`,
`shell/command` and `shell/result`, `tool/call` and `tool/result`, a Code Mode
dispatch and its result, a daemon verb and its outcome, and a workspace tree
that was acquired and disposed. Each pair had its own fold of "opened and never
settled", and three different places settled the orphans that a crash leaves.
Each fold was correct. What was missing was **one statement** of the rule that
repair, the dedupe index and every reader share.

This module is that statement and nothing else. It holds declarations and pure
functions over a sequence of events. It holds no service: `ctx.intents`
(P10-06) does the appending, the flushing and the deduping. The functions are
pure for the same reason as the functions of `folds.py`. Repair folds a log
while the log is being rebuilt, and the trajectory view folds a stored log with
nothing mounted. A fold that was attached to a live `Session` could serve
neither.

**Keys stay in the payloads.** A kind reads its key from its records:

* the seq of the opening record, for the asks and the shell (their settles carry
  the seq back as `askSeq` and `commandSeq`, T3)
* the `subCallId` of a dispatch
* the key of an effect
* the command id of a daemon verb

The readers of those records already use these fields as data. A second copy on
the envelope would be a second carrier of one fact.

**Kinds are declared in leaves** (T4). The kinds of ph-core are in
`ph.session.kinds`, which this package imports. The kinds of another package are
in its own `kinds` module, which that package imports. `declared_intents` lists
them. Not every pair above is a kind: `tool/call` and `tool/result`, and a
workspace tree, keep their own folds.

**There is one fold, `fold_intents`.** For each key, it gives the latest intent
that was opened under the key and the settle that closed it. `open_intents` and
the `Prior` of the journal both read from this fold, so "open" and "settled"
cannot come to mean two different things. The rule of the fold:

* An open of a key replaces the intent before it. The later intent is the live
  one.
* A settle closes only its own key.
* A record whose key cannot be read is not part of the pair.
* Open intents come back in the order in which they were opened.

@module ph.session.intents
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import Literal, TypeAlias

from ..json import JsonObject, JsonValue, as_obj, as_str
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
    "opened_seq",
    "outcome_of",
    "seq_field",
    "unsettled",
    "unsettled_why",
]

Unsettled: TypeAlias = Literal["outcome-unknown", "not-started"]
"""What a settle says when something other than the act itself wrote it: that the
act may have happened, or that it cannot have happened. The value is passed to the
`closer` of a kind."""

IntentOrphan: TypeAlias = Literal[Unsettled, "owner-settles"]
"""What an intent that a crash left open means, and therefore who settles it
(decision 2).

* `outcome-unknown`: the act may have happened. Repair writes a settle that says
  so. That is the only correct thing that a log can say about an effect that it
  did not see finish: a shell command, a daemon verb, or an approval or a
  question that a crash cut short. Every kind that repair settles is declared
  this way.
* `not-started`: the act cannot have happened without its settle, and repair
  writes the settle as not started. No kind is declared this way. The journal
  writes `not-started` for a durable kind whose opening record could not be
  flushed (`IntentJournal.open`). In that case nobody was asked and nothing ran.
* `owner-settles`: repair leaves the intent open. The seam that owns it can
  *look*, and it reconciles on open. An example is a credential hold, whose
  resume check sees whether the name has arrived. A guess from repair would be
  worse than that check by the owner.
"""

Barrier: TypeAlias = Literal["durable", "buffered", "tools-body"]
"""When the opening record must be on disk (decision 3).

* `durable`: before the act. The journal flushes before it hands out the claim,
  and it refuses the claim if the flush fails. An opener whose act carries a
  durable intent of its own can open with `open_deferred` instead. The durable
  intent of the act then flushes the record.
* `buffered`: at the next flush, whenever that happens. This is for accounting,
  where a crash that loses the record loses a line of history and never an act.
* `tools-body`: at the barrier that the checkpoint policy places before the body
  of a tool (`tools/body`). That barrier runs *after* every pre-execute gate and
  every `tools/execute` wrapper. The two tool kinds declare this value so that
  the journal does not place a second barrier. The barrier stays where it
  already is.
"""


Outcome: TypeAlias = Literal["done", "failed", "outcome-unknown", "not-started"]
"""What became of an intent. Every consumer reads it the same way (T2).

* `done`: the act settled the intent.
* `failed`: the act settled the intent and said that it failed, so a retry is a
  decision of its own (`IntentKind.failed`).
* `outcome-unknown` and `not-started`: something other than the act settled the
  intent. The value says which half is known (`Unsettled`).
"""

SettledBy: TypeAlias = Literal["repair", "process"]
"""Who wrote a settle that the act did not write.

* `repair`: on resume, for an intent that a dead process left open.
* `process`: the live process, on behalf of the act, after a barrier failed or
  after a body raised or was canceled.
"""

UNSETTLED = "unsettled"
"""The name of the marker: a payload field that every settle **not written by the
act** carries, and that nothing else carries. Its shape is
`{"unsettled": {"why": <Unsettled>, "by": <SettledBy>}}`.

There is one field for every kind. The journal and repair write it, and the
closers do not. The question "did this finish?" therefore has one reader
(`outcome_of`). Phase 10 had grown four spellings of it: `interrupted: why`,
`interrupted: true`, `outcome: "unknown"` and `failed: true`. A closer still
writes the payload of its own kind (the `outcome` of an approval, the
`resolution` of a question), and the readers of that kind use it. This field is
the part that every reader of every kind shares."""


_UNSETTLED_WHY: Mapping[str, Unsettled] = literal_lookup(Unsettled)


def unsettled(why: Unsettled, by: SettledBy) -> JsonObject:
    """The marker, ready to merge into a closer's payload."""
    return {UNSETTLED: {"why": why, "by": by}}


def unsettled_why(data: JsonObject) -> Unsettled | None:
    """The reason in the marker, or `None` for a settle that the act wrote itself."""
    return _UNSETTLED_WHY.get(as_str(as_obj(data.get(UNSETTLED)).get("why")))


class IntentError(ValueError):
    """A kind that cannot be declared as stated."""


@dataclass(frozen=True, slots=True)
class IntentKind:
    """One pair: the record that opens an intent, and the record that settles it.

    `closer(opened, why)` builds a settle payload from the opening record, for a
    settle that the act did not write. Repair uses it for an orphan (`why` is
    the `orphan` of the kind). The journal uses it for an intent whose barrier
    failed (`"not-started"`) or whose body raised (`"outcome-unknown"`). The
    payload must carry the key where `settled_key` reads it. A closer is
    required unless the kind is `owner-settles`, because repair must be able to
    write *something* for every orphan that it is told to settle.

    `writer` is the writer of the leaf that declares the kind
    (`log_writer(__name__)`), and the journal writes the pair through it (T6).
    A kind can therefore be declared only by a module that is the writer of
    record for both of its types, and the journal writes nothing that the
    declarer of a kind could not write.
    """

    opened: str
    settled: str
    opened_key: Callable[[SessionEvent], str | None]
    settled_key: Callable[[SessionEvent], str | None]
    orphan: IntentOrphan
    barrier: Barrier = "durable"
    closer: Callable[[SessionEvent, Unsettled], JsonObject] | None = None
    failed: Callable[[SessionEvent], bool] | None = None
    """Whether the settle that the act wrote says that the act failed. `outcome_of`
    reads it. `None` for a kind whose act has done its work when it settles: a
    shell command that exits non-zero still ran."""
    reopen: frozenset[Outcome] = frozenset()
    """The outcomes of a prior for which `open_once` opens the key again and does
    not answer with the prior (T2). A tool effect that failed, or never started,
    is worth another attempt. An effect that was done is answered from the log.
    For an effect whose outcome is unknown, the caller decides (`reconcile`
    first). Empty, the default, answers every prior."""
    reconciled: Callable[[SessionEvent, tuple[JsonValue, ...]], JsonObject] | None = None
    """Builds the settle for an intent whose tool said, after a crash, that the act
    happened (L6b). It is given the opening record and the content that the tool
    rendered. A kind that declares it has open intents that are worth asking its
    tool about on resume. The payload must carry the key where `settled_key`
    reads it, as the payload of the closer does. When a tool says that the act
    did *not* happen, the closer writes the settle, as `not-started`. `None` for
    a kind that no tool is asked about."""
    within: Callable[[SessionEvent], tuple[str, str] | None] | None = None
    """The top-level call that an intent was opened inside, and the name of the
    intent there: the `rootCallId` and the tool name of a Code Mode dispatch
    (L6b). The repaired result of that call names what the intents inside it
    were found to have done, because the model reads that result. `None` for a
    kind that no call contains."""
    _: KW_ONLY
    writer: LogWriter

    @property
    def owner(self) -> str:
        """The module that declares the pair: the owner of its writer."""
        return self.writer.owner


def outcome_of(kind: IntentKind, settled: SessionEvent) -> Outcome:
    """What a settle says became of its intent. This function is the only reader of
    that answer (T2)."""
    why = unsettled_why(settled.data)
    if why is not None:
        return why
    return "failed" if kind.failed is not None and kind.failed(settled) else "done"


def abandoned(kind: IntentKind, opened: SessionEvent, why: Unsettled, by: SettledBy) -> JsonObject:
    """Build the settle of an intent that its act did not settle: the payload of the
    closer of the kind, with the marker.

    This is the one spelling of that settle. Repair uses it (`by="repair"`), and
    the journal uses it when it settles on behalf of an act (`by="process"`).

    :raises IntentError: when the kind has no closer to write the settle with.
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
    """The latest intent of one key: the record that opened it, and its settle, if any.

    A settle under a key that nothing opened is not an intent, and the fold skips
    it, because there is nothing open for it to close. The journal writes no such
    settle, but a log need not have come from this journal.
    """

    opened: SessionEvent
    settled: SessionEvent | None


_KINDS: dict[str, IntentKind] = {}
"""Every declared kind, by the type that opens it. These are import-time facts, like
`known_event_types._DECLARED`, so a test that declares a kind swaps this dict out."""


def declare_intent(kind: IntentKind) -> IntentKind:
    """Declare a pair. Call it at import, in the `kinds` leaf of a package (T4).

    The function refuses:

    * A type that the vocabulary does not know. A pair that this build could
      write and then refuse to read back is F11 one level up.
    * A second kind that the same type opens. One record has one meaning.
    * A kind that repair is told to settle, with no `closer` to settle it with.
    * One type that both opens and settles. No record can be both.
    * A settle that the model sees (`SURFACE_EVENT_TYPES`). Such a settle needs a
      surface placement, which the journal does not write, and provider rules,
      which repair keeps for the turn that it closes. Such a pair therefore
      belongs to the turn repair and not to a kind.
    * A writer that may not write both types (T6). The journal writes the pair
      through the writer, so a kind cannot carry writes that its declarer could
      not make.

    A second declaration of the same kind returns the first one, so a module that
    is imported twice does not refuse itself.
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
    """Every declared kind, in declaration order. Repair walks this tuple (P10-07)."""
    return tuple(_KINDS.values())


def is_declared(kind: IntentKind) -> bool:
    """Whether `kind` is the one declared for its opening type."""
    return _KINDS.get(kind.opened) == kind


def fold_intents(
    events: Iterable[SessionEvent],
    kind: IntentKind,
    since: Mapping[str, IntentRecord] | None = None,
) -> Mapping[str, IntentRecord]:
    """For each key, the latest intent of `kind` that this log opened, and how it settled.

    `since` is this fold of an earlier prefix, and `events` is the rest of the log
    after that prefix. Extending from a prefix equals folding the whole log. The
    `extend` of `SessionFoldCache` needs that law, and `ph.testing.folds` checks
    it. The function copies the value of the prefix and never changes it. When
    the slice holds nothing of this kind, the function returns the value of the
    prefix as it was. A cache that owns its value and hands it to no other code
    extends the value in place instead (`extend_index`). Such a cache pays for
    the slice, and not for every key that it holds.
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
    """`fold_intents`, applied to `index` in place. For a cache that owns `index`."""
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


def opened_seq(event: SessionEvent) -> str:
    """The key of an intent that is keyed by the seq of the record that opened it.
    Such a key is unique by construction.

    The function is here and not in a kinds leaf, because every leaf keys some
    kind this way and a leaf may not import another leaf (T4). Two copies of a
    key rule are two chances for a settle to stop matching the record that it
    closes.
    """
    return str(event.seq)


def seq_field(event: SessionEvent, name: str) -> str | None:
    """The seq of the opening record that a settle carries under `name`, as a key.
    `None` when the settle carries none. A `bool` is not a seq, although Python
    calls it an `int`."""
    seq = event.data.get(name)
    return str(seq) if isinstance(seq, int) and not isinstance(seq, bool) else None


def key_of(
    read: Callable[[SessionEvent], str | None], event_type: str, data: JsonObject, seq: int
) -> str | None:
    """The key that `read` finds in a record that is not yet in the log, as the fold
    would read it.

    This is a shallow probe. Key functions read one or two fields of `data`, so
    `data` is passed as given and is not frozen first. The append that follows
    freezes it one time.
    """
    return read(SessionEvent(type=event_type, seq=seq, time=0, data=data))


def open_intents(
    events: Iterable[SessionEvent] | Mapping[str, IntentRecord], kind: IntentKind
) -> tuple[OpenIntent, ...]:
    """The intents of `kind` that this log opened and never settled, in opening order.

    `events` is a log, or the `fold_intents` of a log: the cached index of the journal."""
    index = events if isinstance(events, Mapping) else fold_intents(events, kind)
    found = [
        OpenIntent(key=key, opened=record.opened)
        for key, record in index.items()
        if record.settled is None
    ]
    return tuple(sorted(found, key=lambda intent: intent.opened.seq))
