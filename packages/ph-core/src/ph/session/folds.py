"""A cache for the fold of a plugin over a session log, without ownership of the log.

Several plugins project the log into something that they need per turn: the
subagent roster, the Continual Harness state (P3-16), the kernel snapshot of a
namespace. Each is a fold, and each is O(log). The log of a long session is
mostly `assistant/chunk`. So a projection that is read one time per model step
must not scan the log again.

The obvious answer is to attach folds to `Session`, as `Session.projection`
attaches them. That answer is wrong for this family, because of a property that
is worth protecting: **these folds must stay callable on a log that is not the
live one.** `fold_namespace` is what makes `ctx.sessions.fork(source, boundary)`
reconstruct a namespace *as of the boundary* (D17). The trajectory view of P3-24
projects a stored log with nothing mounted. A fold that is attached to a live
`Session` is monotonic in that log and cannot answer for an earlier prefix. So
to attach it would give up the purpose in exchange for the speed.

So the fold stays a pure function of a log, and the *cache* is a separate object
that a consumer owns. `session.seq` is an exact invalidation key because the log
is append-only (A1): if the log has not grown, no fold over it can have changed.

**Which of the two to use.** `Session.projection` is the narrow family: "the
latest event of type X, read as Y". The answer is one event, and the fold can
start from the point where it stopped. `SessionFoldCache` is the general one: an
arbitrary fold over the whole log. That is why the fold must be a pure function,
and why it is the one held to the laws of `ph.testing.folds`. A projection that
grows past one event belongs here and not there.

**The one requirement**, and the only way to misuse this cache: the cached
function must be a pure fold of the prefix. A function that also reads the
clock, the filesystem, or a mutable table can change its answer when the log has
not grown, and the cache will not notice. Nothing here can check that *ahead* of
the fact. `ph.testing.folds` can, from a process that holds the whole log and
therefore every prefix of it. The tests of each consumer hold its fold to those
laws before the cache relies on them.

`stale` adds the part that a test cannot reach: whether the answers that this
cache serves **right now**, in a running deployment, still equal the fold. Each
consumer polls it through its own invariant row. So a report names the cache
that drifted, and does not say only that one drifted.

@module ph.session.folds
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .session import Session
from .tables import SessionTable

__all__ = ["SessionFoldCache"]


class SessionFoldCache[T]:
    """One cached fold per session, recomputed only when the log grew.

    The cache is bounded by live sessions and not by history. An entry is
    *replaced* and not accumulated, so a million-event session holds one value.
    "Live" has the meaning that the store gives it: the entries are a
    `SessionTable`, so a session that the store lets go takes its entry with it.
    A read of a session that has already gone is answered and not filed.
    """

    __slots__ = ("_compute", "_entries", "_extend")

    def __init__(
        self,
        compute: Callable[[Any], T],
        *,
        extend: Callable[[T, Any, int], T] | None = None,
    ) -> None:
        self._compute = compute
        self._extend = extend
        self._entries: SessionTable[tuple[int, T]] = SessionTable()

    def read(self, session: Session) -> T:
        """The fold over this session, folded at most one time per appended event.

        With `extend`, a miss folds only the new slice. `session.seq` increases
        on every event of any type, and a long session is mostly chunks. So a
        projection that is read at each model step misses the key almost every
        time, although the events that it folds are rare.
        `extend(previous, session, from_seq)` carries the same requirement as
        `compute`, resumed from a prefix: to extend the fold of a prefix must
        equal a fold of the whole log. Without `extend`, a miss folds again from
        zero, which is still correct.
        """
        cached = self._entries.get(session)
        if cached is not None and cached[0] == session.seq:
            return cached[1]
        if cached is not None and self._extend is not None and session.seq > cached[0]:
            value = self._extend(cached[1], session, cached[0])
        else:
            value = self._compute(session)
        self._entries.keep(session, (session.seq, value))
        return value

    def stale(self, sessions: Iterable[Session]) -> list[str]:
        """Every cached answer that no longer equals the fold of its log (I6).

        The check is here and not in the invariant row that declares it, for the
        reason that `ToolRuntime.stale_views` gives. `_entries` is private to
        this class. A rename would disable a check that was written against it
        from outside, and nobody would notice.

        **Only entries whose key still matches are compared.** An entry below
        the current `seq` of the session is not drift. It is the ordinary state
        of a cache between reads, and `read` folds the new slice before it
        serves the entry. This check catches the failures that the key cannot
        see: a fold that answered from something other than the log, and a
        reader that mutated the value that it was given and so corrupted the
        entry for every later reader.

        A cached session that is no longer live is skipped and not reported.
        There is no log left to fold, so nothing here can say whether it
        drifted.

        The cost is O(events) per cached session. That is why this is polled and
        never run on `read`: on `read` it would cost exactly the memoization
        that it checks.
        """
        found: list[str] = []
        for session in sessions:
            cached = self._entries.get(session)
            if cached is None or cached[0] != session.seq:
                continue
            fresh = self._compute(session)
            if cached[1] != fresh:
                found.append(
                    f"session {session.id}: the value cached at seq {session.seq} "
                    f"does not equal the fold of its log"
                )
        return found

    def forget(self, session: Session) -> None:
        self._entries.forget(session)

    def __len__(self) -> int:
        """How many sessions have a cached answer."""
        return len(self._entries)
