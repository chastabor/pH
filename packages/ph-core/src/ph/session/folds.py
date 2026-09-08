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
the cache will not notice. Nothing here can check that *ahead* of the fact;
`ph.testing.folds` can, from a process that holds the whole log and so every
prefix of it, and each consumer's tests hold their fold to those laws before the
cache ever relies on them.

What `stale` adds is the half a test cannot reach: whether the answers this cache
is serving **right now**, in a running deployment, still equal the fold. Each
consumer polls it through its own invariant row, so a report names which cache
drifted rather than that one did.

@module ph.session.folds
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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

    def stale(self, sessions: Iterable[Any]) -> list[str]:
        """Every cached answer that no longer equals the fold of its log (I6).

        Here rather than in the invariant row that declares it, for
        `ToolRuntime.stale_views`' reason: `_entries` is this class's own secret,
        and a check written against it from outside is one a rename disables
        without anybody noticing.

        **Only entries whose key still matches are compared.** An entry below the
        session's current `seq` is not drift — it is the ordinary state of a cache
        between reads, and `read` will fold the new slice before serving it. What
        this catches is the failure the key cannot see: a fold that answered from
        something other than the log, and a reader that mutated the value it was
        handed and so poisoned the entry for everyone after it.

        A cached session that is no longer live is skipped rather than reported:
        there is no log left to fold, so nothing here can say whether it drifted.

        O(events) per cached session, which is why this is polled and never run on
        `read` — doing it there would cost exactly the memoization it checks.
        """
        found: list[str] = []
        for session in sessions:
            cached = self._entries.get(session.id)
            if cached is None or cached[0] != session.seq:
                continue
            fresh = self._compute(session)
            if cached[1] != fresh:
                found.append(
                    f"session {session.id}: the value cached at seq {session.seq} "
                    f"does not equal the fold of its log"
                )
        return found

    def forget(self, session_id: str) -> None:
        self._entries.pop(session_id, None)

    def clear(self) -> None:
        self._entries.clear()
