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
from typing import TYPE_CHECKING, Literal, TypeAlias

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
    extend_index,
    fold_intents,
    is_declared,
    key_of,
    open_intents,
    settled_record,
)
from .session import Session

if TYPE_CHECKING:
    from .store import SessionStore

__all__ = ["Claim", "IntentJournal", "IntentNotDurable", "IntentScope", "Prior", "intents_of"]


IntentScope: TypeAlias = Literal["log", "process"]
"""How long a key names its act: for the life of the **log**, or of the **process**
that opened it.

`process` is for an act whose effect lives in process memory — a credential handed
to the daemon (L1). Its key is on the log after a restart and its effect is not, so
refusing a re-send as repeated would refuse the only way the value comes back. The
opened record carries `"scope": "process"`, and a key opened before this process's
`first_live_seq` is no prior."""


class IntentNotDurable(RuntimeError):
    """A durable intent's opening record could not be written, so no claim was
    handed out. The act must not happen: the log could not say it was about to."""


@dataclass(frozen=True, slots=True)
class Claim:
    """An intent opened by this caller and not yet settled: permission to act."""

    kind: IntentKind
    key: str
    opened: SessionEvent


Prior: TypeAlias = IntentRecord
"""A key this log had already opened, so nothing was: the fold's own record of it —
what it opened then, and its settle, `None` while that intent is still open."""

_Index = Mapping[str, IntentRecord]


@dataclass(slots=True)
class IntentJournal:
    """The service published as `ctx.intents`, by the `session` row (decision 1)."""

    sessions: SessionStore | None
    """Whose flush is the barrier. `None` in a context with no session row, where
    nothing writes a log and so nothing can fail to — `session_written`'s reading."""
    _indexes: dict[IntentKind, SessionFoldCache[_Index]] = field(default_factory=dict)

    async def open(
        self,
        session: Session,
        kind: IntentKind,
        data: JsonObject,
        *,
        key_scope: IntentScope = "log",
    ) -> Claim | Prior:
        """Open an intent under the key `data` carries, unless the log already has.

        **Dedupe first.** A key this log already opened — settled or not — returns
        `Prior` and appends nothing, so a retried act is answered from the log
        rather than done twice.

        **Then the barrier**, for a `durable` kind: the opening record is flushed
        before the claim is handed out, and a flush that fails hands out none —
        fail-closed, as the checkpoint policy's barriers are. The intent is then
        settled `not-started` in memory (when the kind has a closer), so the next
        flush that works writes an honest pair rather than an orphan for repair.
        A `buffered` or `tools-execute` kind is not flushed here; see `Barrier`.

        A kind that does not `dedupe` skips the first step: its keys are reused
        on purpose, and a second open is a new intent. A `process`-scoped key
        opened by an earlier process is no prior either (`IntentScope`).

        :raises IntentNotDurable: when a durable kind's record could not be written.
        :raises IntentError: when the kind is undeclared or `data` carries no key.
        """
        self._require_declared(kind)
        if key_scope == "process":
            data = {**data, "scope": "process"}
        key = _key(kind.opened, kind.opened_key, data, session.seq)
        if kind.dedupe:
            prior = self._index(session, kind).get(key)
            if prior is not None and _lives(prior.opened, session):
                return prior
        claim = Claim(kind=kind, key=key, opened=session.append(kind.opened, data))
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
        """Open an intent with no barrier and no dedupe.

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
        return Claim(kind=kind, key=key, opened=session.append(kind.opened, data))

    def settle(self, session: Session, claim: Claim, data: JsonObject) -> SessionEvent:
        """Append the settle of `claim`.

        **Only the claim's own intent, and only once**: the settle must carry the
        claim's key where the kind reads it, and the intent must still be the open
        one under that key — not settled already, not opened again since.

        **A kind that reuses keys is held to the first half only.** Two asks of one
        tool with no call id share a key, and when they run concurrently the
        second open replaces the first in the fold — which could never tell them
        apart, and does not now. Refusing the first ask's settle would turn that
        old ambiguity into an error raised from a `finally`; so for a kind that
        does not `dedupe`, the settle closes its key, as the fold reads it.

        :raises IntentError: when `data` names another key, or the intent is not open.
        """
        kind = claim.kind
        key = _key(kind.settled, kind.settled_key, data, session.seq)
        if key != claim.key:
            raise IntentError(
                f"a settle for {key!r} cannot close the intent opened as {claim.key!r}"
            )
        if kind.dedupe and not self.is_open(session, claim):
            raise IntentError(f"{kind.opened} {claim.key!r} is not open, so it cannot be settled")
        return session.append(kind.settled, data)

    @asynccontextmanager
    async def claim(
        self,
        session: Session,
        kind: IntentKind,
        data: JsonObject,
        *,
        key_scope: IntentScope = "log",
    ) -> AsyncIterator[Claim | Prior]:
        """`open`, and a settle nobody can forget when the body raises.

        A body that raises — or is canceled — while its intent is open has it
        settled `outcome-unknown` with `"failed": true`, the way `step/end` is
        written in a `finally`, so a caller cannot leave a pair open by raising.
        A body that returns without settling leaves it open: that is a caller that
        settles later, which only its owner can do. A `Prior` is handed through
        untouched — there is nothing of this caller's to settle.
        """
        held = await self.open(session, kind, data, key_scope=key_scope)
        if isinstance(held, Prior):
            yield held
            return
        try:
            yield held
        except BaseException:
            if kind.closer is not None and self.is_open(session, held):
                self.settle(
                    session, held, {**kind.closer(held.opened, "outcome-unknown"), "failed": True}
                )
            raise

    def _not_started(self, session: Session, claim: Claim) -> None:
        if claim.kind.closer is not None:
            self.settle(session, claim, claim.kind.closer(claim.opened, "not-started"))

    def is_open(self, session: Session, claim: Claim) -> bool:
        """Whether `claim` is still the open intent under its key."""
        record = self._index(session, claim.kind).get(claim.key)
        return (
            record is not None and record.opened.seq == claim.opened.seq and record.settled is None
        )

    def pending(self, session: Session, kind: IntentKind) -> tuple[OpenIntent, ...]:
        """Intents of `kind` this log opened and never settled, in opening order."""
        return open_intents(self._index(session, kind), kind)

    def outcome(self, session: Session, kind: IntentKind, key: str) -> SessionEvent | None:
        """The settle of the latest intent opened under `key`, or `None`."""
        return settled_record(self._index(session, kind), kind, key)

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


def _lives(opened: SessionEvent, session: Session) -> bool:
    """Whether a key opened by `opened` still names its act in this process."""
    return opened.data.get("scope") != "process" or opened.seq >= session.first_live_seq


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
