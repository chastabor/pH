"""What a row keeps per session, kept only while the store holds the session.

A row that remembers something for each session it is asked about (a cached fold,
the files it wrote, a count it reached) owes two things, and each was written by
hand, row by row, and forgotten the same way:

* **the entry goes when the session does.** Thirteen rows subscribed to
  `session/disposed` to pop their own dict, and a fourteenth did not, so it kept
  one entry per session for the life of its mount;
* **nothing is kept for a session that has already gone.** A reader still holding
  a disposed session (a status line drawn once more, a listener running during the
  disposal itself) asked again, and the row filed a new entry that nothing was
  left to forget. Forgetting on the event does not cover this; only the write can.

`SessionTable` is both, in one place (DESIGN.md §2.5, "Per-session tables"). **The
session is the scope**: filing an entry asks the session to release it when its
store lets it go (`Session.when_disposed`), and a session that has already gone
says so and nothing is filed. The value is handed back either way, so a reader of
a gone session gets its answer and the table stays as it was. So a table needs no
context and no declaring: one built anywhere is bounded by the sessions it holds.

**Keyed by the session, not its id.** A log opened a second time in one process is
another object under the same id, and one copy's entry is not the other's.

A session no store ever held is never let go, and its entries live as long as it
does, which is what a test or an offline tool building sessions by hand wants.

**What it is not for.** State that is *meant* to outlive its session (the
sub-agent seam's last state of a let-go child) and state not keyed by one session
(a rate limit per pair of agents) keep their own listener, and
`tests/test_session_tables.py` makes each say why.

@module ph.session.tables
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from .session import Session

__all__ = ["SessionTable"]


class SessionTable[V]:
    """One entry per session, gone when the store lets the session go.

    `released` is what else goes with an entry: it is handed the value as the entry
    is forgotten, by a disposal or by `forget`, so a table of files can delete them
    through the same door that drops the record (Holdings rule 2).
    """

    __slots__ = ("_entries", "_released")

    def __init__(self, *, released: Callable[[V], None] | None = None) -> None:
        self._entries: dict[Session, V] = {}
        self._released = released

    @overload
    def get(self, session: Session) -> V | None: ...
    @overload
    def get[D](self, session: Session, default: D) -> V | D: ...
    def get(self, session: Session, default: object = None) -> object:
        return self._entries.get(session, default)

    def keep(self, session: Session, value: V) -> V:
        """File `value` as `session`'s entry unless the session has gone, and hand it
        back either way."""
        if session in self._entries or session.when_disposed(self._left):
            self._entries[session] = value
        return value

    def setdefault(self, session: Session, make: Callable[[], V]) -> V:
        """`session`'s entry, made and kept if there is none. For a session that has
        gone this is a new value each time, and nothing is filed."""
        try:
            return self._entries[session]
        except KeyError:
            return self.keep(session, make())

    def forget(self, session: Session) -> None:
        """Drop one session's entry, and release what it held."""
        if session not in self._entries:
            return
        value = self._entries.pop(session)
        if self._released is not None:
            self._released(value)

    def __contains__(self, session: object) -> bool:
        return session in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def _left(self, session: Session) -> None:
        self.forget(session)
