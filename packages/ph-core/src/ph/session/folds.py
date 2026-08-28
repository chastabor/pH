"""Caching a plugin's own fold over a session log, without owning the log.

Several plugins project the log into something they need per turn: the subagent
roster, the Continual Harness state (P3-16), a namespace's kernel snapshot. Each
is a fold, each is O(log), and a long session's log is mostly
`assistant/chunk` — so a projection read once per model step must not re-scan it.

The obvious answer, and the wrong one, is to attach folds to `Session` the way
`Session.latest` is attached. It does not work for this family, and the reason is
a property worth protecting: **these folds must stay callable on a log that is
not the live one.** `fold_namespace` is what makes `ctx.sessions.fork(source,
boundary)` reconstruct a namespace *as of the boundary* (D17), and P3-24's
trajectory view projects a stored log with nothing mounted. A fold attached to a
live `Session` is monotonic in that log and cannot answer for an earlier prefix,
so attaching it would trade the whole point for the speed.

So the fold stays a pure function of a log, and the *cache* is a separate thing a
consumer owns. `session.seq` is an exact invalidation key because the log is
append-only (A1): if it has not grown, no fold over it can have changed.

**The one requirement**, and the only way to misuse this: the cached function
must be a pure fold of the prefix. A function that also reads the clock, the
filesystem, or a mutable table can change its answer without the log growing, and
the cache will not notice. Nothing here can check that.

@module ph.session.folds
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

__all__ = ["SessionFoldCache"]


class _Log(Protocol):
    """What a fold cache needs: an identity and a length."""

    @property
    def id(self) -> str: ...

    @property
    def seq(self) -> int: ...


class SessionFoldCache[T]:
    """One cached fold per session, recomputed only when the log grew.

    Bounded by live sessions rather than by history: an entry is *replaced*, not
    accumulated, so a million-event session holds one value. `forget` is for a
    consumer that also tracks session disposal and would rather not keep the last
    projection of a session nobody can reach.
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
        self._entries: dict[str, tuple[int, T]] = {}

    def read(self, session: _Log) -> T:
        """The fold over this session, folded at most once per appended event.

        With `extend`, a miss folds only the new slice: `session.seq` bumps on
        every event of any type — a long session is mostly chunks — so a
        projection read each model step misses the key almost every time even
        though the events it folds are rare. `extend(previous, session,
        from_seq)` carries the same requirement as `compute`, resumed from a
        prefix: extending the fold of a prefix must equal folding the whole log.
        Without it a miss refolds from zero, which is still correct.
        """
        cached = self._entries.get(session.id)
        if cached is not None and cached[0] == session.seq:
            return cached[1]
        if cached is not None and self._extend is not None and session.seq > cached[0]:
            value = self._extend(cached[1], session, cached[0])
        else:
            value = self._compute(session)
        self._entries[session.id] = (session.seq, value)
        return value

    def forget(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def clear(self) -> None:
        self._entries.clear()
