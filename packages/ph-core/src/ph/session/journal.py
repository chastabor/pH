"""`ctx.intents`: record before an act, settle after it, and answer for the key (P10-06).

Seven sites wrote an intent pair, and each site wrote its own append, its own
flush and its own failure handling. This module now holds that code in one
place. A component that must record before an effect gets it right with one
method call:

```python
async with ctx.intents.claim_once(session, CLIENT_COMMAND, {"command": key}) as held:
    if isinstance(held, Prior):
        return repeated(held.outcome)  # This key already ran. Say what became of it.
    reply = await act()
    ctx.intents.settle(session, held, command_settled(key))
```

`claim_once` is for a key that names one act for the life of the log, as the key
of a daemon verb does. A kind in which every act is separate (an ask, a `!!`
command) opens with `claim`. `claim` always opens a new intent and never
returns a `Prior`.

**The log is the journal.** Nothing here stores anything. An intent is its
opening record, and its outcome is its settle. What is open is the fold of the
log that `ph.session.intents` defines. The fold is cached for each session and
kind, and the `intent-fold-cache` invariant holds the cache equal to that fold,
as it does for every other cache over the log.

**A kind must be declared** (`declare_intent`) before the journal opens an
intent of that kind, because repair settles only the kinds that it knows. An
intent of an undeclared kind would leave an orphan that nothing would ever
close.

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
    """The opening record of a durable intent could not be written, so no claim was
    handed out. The act must not happen, because the log could not say that the
    act was about to happen."""


@dataclass(frozen=True, slots=True)
class Claim:
    """An intent opened by this caller and not yet settled: permission to act."""

    kind: IntentKind
    key: str
    opened: SessionEvent


@dataclass(frozen=True, slots=True)
class Prior:
    """A key that this log had already opened, so nothing was opened now. It holds
    what the log opened then, how that intent settled, and what the settle means.
    The meaning is read one time, here, so no caller decodes a payload to find it
    (T2)."""

    opened: SessionEvent
    settled: SessionEvent | None
    """`None` while the intent is open."""
    outcome: Outcome | None
    """The `outcome_of` its settle. `None` while the intent is open."""
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
    """The store whose flush is the barrier. `None` in a context with no session row.
    Nothing writes a log there, so no write can fail. `session_written` reads that
    case the same way."""
    _indexes: dict[IntentKind, SessionFoldCache[_Index]] = field(default_factory=dict)

    async def open(self, session: Session, kind: IntentKind, data: JsonObject) -> Claim:
        """Open a new intent under the key that `data` carries.

        **The intent is always new** (T3). This method does not look for a prior.
        `open_once` is for a key that names one act for the life of the log. A
        kind in which every act is separate (an ask, a command) opens here, keyed
        by its own record.

        **The barrier**, for a `durable` kind: the method flushes the opening
        record before it hands out the claim. If the flush fails, it hands out
        no claim. This is fail-closed, as the barriers of the checkpoint policy
        are. The intent is then settled `not-started` in memory, when the kind
        has a closer. The next flush that works therefore writes a correct pair,
        and not an orphan for repair. A `buffered` or `tools-body` kind is not
        flushed here. See `Barrier`.

        :raises IntentNotDurable: when the record of a durable kind could not be written.
        :raises IntentError: when the kind is undeclared or `data` carries no key.
        """
        self._require_declared(kind)
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        return await self._opened(session, kind, key, data)

    async def open_once(
        self, session: Session, kind: IntentKind, data: JsonObject
    ) -> Claim | Prior:
        """Open an intent under the key that `data` carries, unless the log already did.

        When this log already opened the key, settled or not, the method answers
        with the `Prior` of the key and appends nothing. A retried act is
        therefore answered from the log and is not done twice. There is one
        exception: the kind may say that a prior with that outcome can be tried
        again (`IntentKind.reopen`). The method then opens a new intent, as
        `open` does.

        A key names its act for the life of the log. There was a `process` scope
        for the one act whose effect lived in process memory: a credential that
        was handed to the daemon. That scope was removed when that verb stopped
        taking a key (T5).

        :raises IntentNotDurable: when the record of a durable kind could not be written.
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
            # The task was canceled while the record was being written. The act
            # was never started, and `append` is synchronous, so the pair still
            # closes.
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
        """`open` with no barrier, and therefore synchronous.

        This method is for the `tools-body` kinds. The tool batch writes their
        records, and the checkpoint policy places their flush after every
        pre-execute gate. A flush here would be the wrong flush in the wrong
        place. The method refuses a `durable` kind, because `open` exists to
        make such a kind durable.

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

    def open_deferred(self, session: Session, kind: IntentKind, data: JsonObject) -> Claim:
        """`open` for a durable kind whose barrier is the barrier of its act, and
        therefore synchronous.

        This method is for an opener whose act is itself preceded by a durable
        intent. `/revert <seq>` restores through `WORKSPACE_RESTORE`. The barrier
        of that kind writes the log whole and in append order, so this record
        reaches disk with it before the tree is touched. A barrier here as well
        was a second fsync for one line, with only checks in memory between the
        two. An opener that ends without acting (a refusal) leaves the pair to
        the next flush. A record with no act behind it needs no more than that.

        **The caller is responsible for that promise.** That is why this is a
        separate method and not a flag on `open`. An act with no durable intent
        ahead of it would run while this record was still in memory.

        :raises IntentError: when the kind is not durable or undeclared, or `data`
            carries no key.
        """
        self._require_declared(kind)
        if kind.barrier != "durable":
            raise IntentError(f'"{kind.opened}" has no barrier to defer; record it with `record`')
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        return Claim(kind=kind, key=key, opened=kind.writer.append(session, kind.opened, data))

    def open_settled(
        self,
        session: Session,
        kind: IntentKind,
        data: JsonObject,
        settle: Callable[[SessionEvent], JsonObject],
    ) -> tuple[SessionEvent, SessionEvent]:
        """Open an intent and settle it together, for an act that was decided as it
        was asked (T6).

        The approval `never` policy uses this method. That policy has answered
        every ask in advance. The ask is recorded so that the log says why, with
        its decision beside it. The command registry also uses it, for a command
        whose record mode is `after`: the pair is written after the body has
        run.

        The pair is written in one `Session.batch()`, so it lands whole or not at
        all. There is no barrier, because nothing acts between the two records.
        There is therefore no "before the act" that the opening record must be
        on disk ahead of. `settle` builds the settle from the opening record.
        That is how it learns the seq that keys a seq-keyed kind.

        :raises IntentError: when the kind is undeclared, `data` carries no key, or
            the settle names another key.
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

        **The method settles only the intent of the claim, and only one time.**
        The settle must carry the key of the claim where the kind reads it. The
        intent must still be the open one under that key: not settled already,
        and not opened again since.

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
        """Every open intent of `kind`, as claims that its owner can settle, in
        opening order. For an `owner-settles` kind only (T5).

        The owner of such a kind settles its intents when it looks. That may be
        in a later process than the one that opened them, or in another call
        than the one that holds the claim. The owner therefore asks the log for
        its claims and does not keep them. For any other kind, repair settles
        the orphans, and the caller that opened a live intent settles it.

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
        """`open`, with a settle that the caller cannot forget when the body raises.

        When a body raises or is canceled while its intent is open, the intent
        is settled `outcome-unknown`, in the way that `step/end` is written in a
        `finally`. A caller therefore cannot leave a pair open by raising. A
        body that returns without settling leaves the intent open. That caller
        settles later, which only the owner of the intent can do.
        """
        held = await self.open(session, kind, data)
        async with self.settling_on_failure(session, held):
            yield held

    @asynccontextmanager
    async def claim_once(
        self, session: Session, kind: IntentKind, data: JsonObject
    ) -> AsyncIterator[Claim | Prior]:
        """`open_once`, with the settle on failure of `claim`. A `Prior` is passed
        through unchanged, because this caller has nothing of its own to settle."""
        held = await self.open_once(session, kind, data)
        if isinstance(held, Prior):
            yield held
            return
        async with self.settling_on_failure(session, held):
            yield held

    @asynccontextmanager
    async def settling_on_failure(self, session: Session, held: Claim) -> AsyncIterator[None]:
        """Settle `held` as `outcome-unknown` if the body raises or is canceled. This
        is the settle of `claim`, for a caller that opened its intent another way
        (`record`)."""
        try:
            yield
        except BaseException:
            self.abandon(session, held, "outcome-unknown")
            raise

    def _not_started(self, session: Session, claim: Claim) -> None:
        self.abandon(session, claim, "not-started")

    def abandon(self, session: Session, claim: Claim, why: Unsettled) -> SessionEvent | None:
        """Settle `claim` on behalf of its act, when the act raised, was canceled, or
        never started. The settle is the closer of the kind, marked
        `by: "process"` (T2).

        Returns `None` and writes nothing when the intent is no longer open, or
        when its kind has no closer (an `owner-settles` kind, which only its
        owner settles).
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
    """The journal through which a seam should open its intents.

    Where the `session` row is mounted, this is `ctx.intents`. Elsewhere it is a
    journal with no store, and therefore with no barrier. Examples are a seam
    that was set up on a bare `Context`, and a test that holds a `Session` that
    it built by hand. Nothing writes a log there, and `session_written` gives
    the same answer for the same case. Each such call returns a new journal and
    never a shared one.
    """
    return ctx.get(INTENTS) or IntentJournal(sessions=None)


def _index_cache(kind: IntentKind) -> SessionFoldCache[_Index]:
    """The index of a kind, folded one time and then extended **in place** for each slice.

    The extension is in place because this cache owns the value and hands it to
    no other code. The journal reads the value and answers with records, never
    with the dict. A copy for each extension would cost every key that the log
    has ever used, one time for each keyed append. That is quadratic over a
    session of Code Mode dispatches. `stale()` still compares against a fresh
    fold.
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
    """The key that `read` finds in `data`. It is read before anything is appended,
    so a record whose key cannot be read is refused. Such a record would
    otherwise be written as one that no fold would pair."""
    key = key_of(read, event_type, data, seq)
    if not key:
        # An empty key is refused too. Every keyless record would share it, and
        # it would pair them all with each other. A key exists to prevent that
        # collision.
        raise IntentError(f"a {event_type} record must carry its key; this one has none")
    return key
