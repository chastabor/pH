"""`ctx.intents`: record before an act, settle after it, and answer for the key (P10-06).

Seven sites wrote an intent pair, and each spelled its own append, its own flush
and its own failure. This is where that spelling lives once, so a component that
must record before an effect gets it right by calling one method:

```python
async with ctx.intents.claim(session, SHELL, {"commandId": id, "command": text}) as held:
    if isinstance(held, Prior):
        return held.settled  # this key already ran; say what happened, do not rerun
    ctx.intents.settle(session, held, {"commandId": id, "exitCode": await run(text)})
```

**The log is the journal.** Nothing here stores anything: an intent is its opening
record, its outcome is its settle, and what is open is `ph.session.intents`' fold of
the log — cached per session and kind, and held to that fold by the
`intent-fold-cache` invariant like every other cache over the log.

**A kind must be declared** (`declare_intent`) before the journal will open one,
because repair settles only the kinds it knows: an intent of an undeclared kind is
one whose orphan nobody would ever close.

@module ph.session.journal
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..cordis import Context
from ..json import JsonObject
from ..keys import INTENTS
from .events import SessionEvent
from .folds import SessionFoldCache
from .intents import (
    IntentError,
    IntentKind,
    IntentRecord,
    OpenIntent,
    Outcome,
    Unsettled,
    abandoned,
    extend_index,
    fold_intents,
    is_declared,
    key_of,
    open_intents,
    outcome_of,
)
from .session import Session

if TYPE_CHECKING:
    from .store import SessionStore

__all__ = ["Claim", "IntentJournal", "IntentNotDurable", "Prior", "intents_of"]


class IntentNotDurable(RuntimeError):
    """A durable intent's opening record could not be written, so no claim was
    handed out. The act must not happen: the log could not say it was about to."""


@dataclass(frozen=True, slots=True)
class Claim:
    """An intent opened by this caller and not yet settled: permission to act."""

    kind: IntentKind
    key: str
    opened: SessionEvent


@dataclass(frozen=True, slots=True)
class Prior:
    """A key this log had already opened, so nothing was: what it opened then, how
    it settled, and what that means — read once, here, so no caller decodes a
    payload to find out (T2)."""

    opened: SessionEvent
    settled: SessionEvent | None
    """`None` while the intent is open."""
    outcome: Outcome | None
    """`outcome_of` its settle; `None` while it is open."""
    here: bool
    """Whether this process opened it (its seq is at or after `first_live_seq`)."""

    @property
    def running_here(self) -> bool:
        """Open, and opened by this process: still being carried out, now."""
        return self.settled is None and self.here


_Index = Mapping[str, IntentRecord]


@dataclass(slots=True)
class IntentJournal:
    """The service published as `ctx.intents`, by the `session` row (decision 1)."""

    sessions: SessionStore | None
    """Whose flush is the barrier. `None` in a context with no session row, where
    nothing writes a log and so nothing can fail to — `session_written`'s reading."""
    _indexes: dict[IntentKind, SessionFoldCache[_Index]] = field(default_factory=dict)

    async def open(self, session: Session, kind: IntentKind, data: JsonObject) -> Claim:
        """Open a new intent under the key `data` carries.

        **Always a new one** (T3): this door does not look for a prior. Keys that
        name one act for the life of the log are `open_once`'s; a kind whose every
        act is its own — an ask, a command — opens here, keyed by its own record.

        **The barrier**, for a `durable` kind: the opening record is flushed before
        the claim is handed out, and a flush that fails hands out none — fail-closed,
        as the checkpoint policy's barriers are. The intent is then settled
        `not-started` in memory (when the kind has a closer), so the next flush that
        works writes an honest pair rather than an orphan for repair. A `buffered` or
        `tools-execute` kind is not flushed here; see `Barrier`.

        :raises IntentNotDurable: when a durable kind's record could not be written.
        :raises IntentError: when the kind is undeclared or `data` carries no key.
        """
        self._require_declared(kind)
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        return await self._opened(session, kind, key, data)

    async def open_once(
        self, session: Session, kind: IntentKind, data: JsonObject
    ) -> Claim | Prior:
        """Open an intent under the key `data` carries — unless the log already has.

        A key this log already opened, settled or not, is answered with its `Prior`
        and nothing is appended, so a retried act is answered from the log rather
        than done twice. Except where the kind says a prior of that outcome may be
        tried again (`IntentKind.reopen`): then this opens anew, as `open` does.

        A key names its act for the life of the log. There was a `process` scope for
        the one act whose effect lived in process memory — a credential handed to the
        daemon — and it went when that verb stopped taking a key (T5).

        :raises IntentNotDurable: when a durable kind's record could not be written.
        :raises IntentError: when the kind is undeclared or `data` carries no key.
        """
        self._require_declared(kind)
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        record = self._index(session, kind).get(key)
        if record is not None:
            prior = _prior(kind, record, session)
            if prior.outcome is None or prior.outcome not in kind.reopen:
                return prior
        return await self._opened(session, kind, key, data)

    async def _opened(
        self, session: Session, kind: IntentKind, key: str, data: JsonObject
    ) -> Claim:
        claim = Claim(kind=kind, key=key, opened=kind.writer.append(session, kind.opened, data))
        if kind.barrier != "durable" or self.sessions is None:
            return claim
        try:
            durable = await self.sessions.written(session)
        except BaseException:
            # Canceled while the record was being written: the act was never
            # started, and `append` is synchronous, so the pair still closes.
            self._not_started(session, claim)
            raise
        if not durable:
            self._not_started(session, claim)
            raise IntentNotDurable(
                f"session {session.id}: {kind.opened} {key!r} could not be written, so it "
                "was not started"
            )
        return claim

    def record(self, session: Session, kind: IntentKind, data: JsonObject) -> Claim:
        """`open` with no barrier — so synchronous.

        For the `tools-execute` kinds, whose records are written by the tool batch
        and whose flush the checkpoint policy places after every pre-execute gate —
        so a flush here would be the wrong one, in the wrong place. Refused for a
        `durable` kind, which `open` exists to make durable.

        :raises IntentError: when the kind is durable or undeclared, or `data`
            carries no key.
        """
        self._require_declared(kind)
        if kind.barrier == "durable":
            raise IntentError(
                f'"{kind.opened}" is durable; open it with `open`, which writes it first'
            )
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        return Claim(kind=kind, key=key, opened=kind.writer.append(session, kind.opened, data))

    def open_settled(
        self,
        session: Session,
        kind: IntentKind,
        data: JsonObject,
        settle: Callable[[SessionEvent], JsonObject],
    ) -> tuple[SessionEvent, SessionEvent]:
        """Open an intent and settle it together: an act decided as it was asked (T6).

        For the approval `never` policy, which has answered every ask in advance: the
        ask is recorded so the log says why, and its decision beside it. One
        `Session.batch()`, so the pair lands whole or not at all; and no barrier,
        because nothing acts between the two, so there is no "before the act" for the
        opening to be on disk ahead of. `settle` builds the settle from the opening
        record, which is how it learns the seq a seq-keyed kind is keyed by.

        :raises IntentError: when the kind is undeclared, `data` carries no key, or
            the settle names another.
        """
        self._require_declared(kind)
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        with session.batch() as batch:
            opened = kind.writer.append(batch, kind.opened, data)
            settled_data = settle(opened)
            closes = _key(kind.settled, kind.settled_key, settled_data, opened.seq + 1)
            if closes != key:
                raise IntentError(
                    f"a settle for {closes!r} cannot close the intent opened as {key!r}"
                )
            settled = kind.writer.append(batch, kind.settled, settled_data)
        return opened, settled

    def settle(self, session: Session, claim: Claim, data: JsonObject) -> SessionEvent:
        """Append the settle of `claim`.

        **Only the claim's own intent, and only once**: the settle must carry the
        claim's key where the kind reads it, and the intent must still be the open
        one under that key — not settled already, not opened again since.

        :raises IntentError: when `data` names another key, or the intent is not open.
        """
        kind = claim.kind
        key = _key(kind.settled, kind.settled_key, data, session.seq)
        if key != claim.key:
            raise IntentError(
                f"a settle for {key!r} cannot close the intent opened as {claim.key!r}"
            )
        if not self._is_open(session, claim):
            raise IntentError(f"{kind.opened} {claim.key!r} is not open, so it cannot be settled")
        return kind.writer.append(session, kind.settled, data)

    def held(self, session: Session, kind: IntentKind) -> tuple[Claim, ...]:
        """Every open intent of `kind`, as claims its owner can settle — for an
        `owner-settles` kind only (T5), in opening order.

        Such a kind's intents are settled by their owner when it looks, which may be
        in a later process than the one that opened them, or in another call than
        the one holding the claim. So the owner asks the log for its claims rather
        than keeping them. Any other kind's orphans are repair's to settle, and its
        live intents are settled by the caller that opened them.

        :raises IntentError: when the kind is undeclared or not `owner-settles`.
        """
        self._require_declared(kind)
        if kind.orphan != "owner-settles":
            raise IntentError(
                f'"{kind.opened}" is settled by repair or by its opener, not claimed back '
                "from the log"
            )
        return tuple(
            Claim(kind=kind, key=intent.key, opened=intent.opened)
            for intent in self.pending(session, kind)
        )

    @asynccontextmanager
    async def claim(
        self, session: Session, kind: IntentKind, data: JsonObject
    ) -> AsyncIterator[Claim]:
        """`open`, and a settle nobody can forget when the body raises.

        A body that raises — or is canceled — while its intent is open has it
        settled `outcome-unknown`, the way `step/end` is written in a `finally`, so
        a caller cannot leave a pair open by raising. A body that returns without
        settling leaves it open: that is a caller that settles later, which only
        its owner can do.
        """
        held = await self.open(session, kind, data)
        async with self._settling_on_failure(session, held):
            yield held

    @asynccontextmanager
    async def claim_once(
        self, session: Session, kind: IntentKind, data: JsonObject
    ) -> AsyncIterator[Claim | Prior]:
        """`open_once`, with `claim`'s settle on failure. A `Prior` is handed through
        untouched — there is nothing of this caller's to settle."""
        held = await self.open_once(session, kind, data)
        if isinstance(held, Prior):
            yield held
            return
        async with self._settling_on_failure(session, held):
            yield held

    @asynccontextmanager
    async def _settling_on_failure(self, session: Session, held: Claim) -> AsyncIterator[None]:
        try:
            yield
        except BaseException:
            self.abandon(session, held, "outcome-unknown")
            raise

    def _not_started(self, session: Session, claim: Claim) -> None:
        self.abandon(session, claim, "not-started")

    def abandon(self, session: Session, claim: Claim, why: Unsettled) -> SessionEvent | None:
        """Settle `claim` on its act's behalf — the act raised, was canceled, or never
        started — with the kind's closer, marked `by: "process"` (T2).

        `None`, writing nothing, when the intent is no longer open or its kind has
        no closer (an `owner-settles` kind, which only its owner settles).
        """
        if claim.kind.closer is None or not self._is_open(session, claim):
            return None
        return self.settle(session, claim, abandoned(claim.kind, claim.opened, why, "process"))

    def _is_open(self, session: Session, claim: Claim) -> bool:
        """Whether `claim` is still the open intent under its key."""
        record = self._index(session, claim.kind).get(claim.key)
        return (
            record is not None and record.opened.seq == claim.opened.seq and record.settled is None
        )

    def pending(self, session: Session, kind: IntentKind) -> tuple[OpenIntent, ...]:
        """Intents of `kind` this log opened and never settled, in opening order."""
        return open_intents(self._index(session, kind), kind)

    def stale(self, sessions: Iterable[Session]) -> list[str]:
        """Every cached index that no longer equals its fold (I6), by kind."""
        live = tuple(sessions)
        return [
            f"{kind.opened}: {finding}"
            for kind, cache in self._indexes.items()
            for finding in cache.stale(live)
        ]

    def forget(self, session_id: str) -> None:
        for cache in self._indexes.values():
            cache.forget(session_id)

    def _index(self, session: Session, kind: IntentKind) -> _Index:
        cache = self._indexes.get(kind)
        if cache is None:
            cache = self._indexes[kind] = _index_cache(kind)
        return cache.read(session)

    @staticmethod
    def _require_declared(kind: IntentKind) -> None:
        if not is_declared(kind):
            raise IntentError(
                f'"{kind.opened}" is not a declared intent kind; repair settles only declared '
                "kinds, so declare it with ph.session.declare_intent"
            )


def intents_of(ctx: Context) -> IntentJournal:
    """The journal a seam should open its intents through.

    `ctx.intents` where the `session` row is mounted. Elsewhere — a seam stood up
    on a bare `Context`, a test holding a `Session` it built by hand — a journal
    with no store, and so no barrier: nothing there writes a log, which is what
    `session_written` answers for the same case. A fresh one each time, never a
    shared one, since its index is keyed by session id and a hand-built `Session`
    reuses ids freely.
    """
    return ctx.get(INTENTS) or IntentJournal(sessions=None)


def _index_cache(kind: IntentKind) -> SessionFoldCache[_Index]:
    """A kind's index, folded once and then extended **in place** per slice.

    In place because this cache owns the value and hands it to nobody: the journal
    reads it and answers with records, never the dict. A copy per extend would cost
    every key the log has ever used, once per keyed append — quadratic over a
    session of Code Mode dispatches. `stale()` still compares against a fresh fold.
    """

    def compute(log: Session) -> _Index:
        return dict(fold_intents(log.events, kind))

    def extend(previous: _Index, log: Session, start: int) -> _Index:
        index = previous if isinstance(previous, dict) else dict(previous)
        extend_index(index, log.events_from(start), kind)
        return index

    return SessionFoldCache(compute, extend=extend)


def _prior(kind: IntentKind, record: IntentRecord, session: Session) -> Prior:
    settled = record.settled
    return Prior(
        opened=record.opened,
        settled=settled,
        outcome=None if settled is None else outcome_of(kind, settled),
        here=record.opened.seq >= session.first_live_seq,
    )


def _key(
    event_type: str, read: Callable[[SessionEvent], str | None], data: JsonObject, seq: int
) -> str:
    """The key `read` finds in `data`, asked before anything is appended, so a
    record whose key cannot be read is refused rather than written as one no fold
    would pair."""
    key = key_of(read, event_type, data, seq)
    if not key:
        # Empty too: a key every keyless record shares pairs them all with each
        # other, which is the collision a key exists to prevent.
        raise IntentError(f"a {event_type} record must carry its key; this one has none")
    return key
