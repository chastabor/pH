"""What a row keeps for each session, kept only while the store holds the session.

A row may remember something for each session that it is asked about: a cached
fold, the files that it wrote, or a count that it reached. Such a row has two
duties. Each row used to write both duties by hand, and each row could forget
them in the same way:

* **The entry goes when the session goes.** Thirteen rows subscribed to
  `session/disposed` to pop their own dict. A fourteenth row did not, so it kept
  one entry for each session for the life of its mount.
* **Nothing is kept for a session that the store has already disposed.** A
  reader can still hold a disposed session: a status line that is drawn one more
  time, or a listener that runs during the disposal itself. Such a reader asked
  again, and the row filed a new entry that nothing was left to forget. A
  listener on the event does not cover this case. Only the write can.

`SessionTable` does both in one place (DESIGN.md §2.5, "Per-session tables").
**The session is the scope.** When a table files an entry, it asks the session
to release the entry when the store disposes the session
(`Session.when_disposed`). A session that is already disposed reports that, and
the table files nothing. The table returns the value in both cases, so a reader
of a disposed session gets its answer and the table stays as it was. A table
therefore needs no context and no declaration. A table that is built anywhere
is bounded by the sessions that it holds.

**The key is the session, not the id of the session.** A log that is opened a
second time in one process is another object under the same id, and the entry
of one copy is not the entry of the other.

A session that no store ever held is never disposed, and its entries live as
long as the session does. A test or an offline tool that builds sessions by
hand needs this behavior.

**What a table is not for.** Two kinds of state keep their own listener:

* State that is *meant* to outlive its session: the last state of a disposed
  child, which the sub-agent seam keeps.
* State that is not keyed by one session: a rate limit for each pair of agents.

`tests/test_session_tables.py` makes each such listener say why.

@module ph.session.tables
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

from .session import Session

__all__ = ["SessionTable"]


class SessionTable[V]:
    """One entry for each session. The entry is dropped when the store disposes
    the session.

    `released` names what else goes with an entry. The table passes the value to
    `released` when it forgets the entry, at a disposal or through `forget`. A
    table of files can therefore delete the files in the same call that drops
    the record (Holdings rule 2).
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
        """File `value` as the entry of `session`, unless the session is disposed.
        Return `value` in both cases."""
        if session in self._entries or session.when_disposed(self._left):
            self._entries[session] = value
        return value

    def setdefault(self, session: Session, make: Callable[[], V]) -> V:
        """Return the entry of `session`. Make and keep one if there is none. For a
        disposed session, each call makes a new value and files nothing."""
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
