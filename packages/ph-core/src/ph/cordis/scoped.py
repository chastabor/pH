"""One table shape for registrations bucketed by the isolation scope they narrow or shadow.

A registry that answers per agent keys what it holds by scope: the layers of the
tool registry, the restrictions of the skill registry, the per-child path limits
of the sandbox. Each registry wrote the same three things by hand, and each got
one of them wrong at some point:

1. **The key is the layer of the registration** (`Running.layer.isolation`). A
   registration made for an agent is therefore keyed by that agent and not by
   the scope of the row.
2. **The release runs when the row or that scope ends, whichever is first**
   (`Running.add_disposer`). The disposer that the scope holds is the one that
   announces the change. A registry that wrapped the returned disposer told
   nobody when a scope unwound. **The bucket is also dropped with its last
   entry.** A bucket left keyed by a settled scope holds that scope, and a
   table that never empties keeps every reader off its fast path for "nothing
   registered".
3. **A read walks an isolation chain**, in the order of the chain.

`ScopedTable` is the shape, with a bucket of any kind. `ScopedEntries` is the
common case: a list of values, each removed by identity.

@module ph.cordis.scoped
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .context import remove_identical

if TYPE_CHECKING:
    from .context import Context, Disposer, Releaser, Running

__all__ = ["ScopedEntries", "ScopedTable"]


class ScopedTable[B]:
    """One bucket per isolation scope, made on the first registration and dropped
    with the last."""

    __slots__ = ("_buckets", "_empty", "_new")

    def __init__(self, new: Callable[[], B], empty: Callable[[B], bool]) -> None:
        self._new = new
        self._empty = empty
        self._buckets: dict[Context | None, B] = {}

    def claim(
        self,
        by: Running,
        mutate: Callable[[B], None],
        undo: Callable[[B], object],
        *,
        label: str,
        then: Callable[[], None] | None = None,
    ) -> Disposer:
        """Apply `mutate` to the bucket of the layer of `by`. The disposer undoes it.

        The disposer belongs to `by`, so it has the lifetime of the pair. It runs
        `undo`, drops the bucket when the bucket is `empty`, and then runs
        `then`. A registry that announces its changes therefore also announces
        the change that an unwinding scope makes. `undo` is applied to the
        bucket that `mutate` was applied to, even if the table has since keyed a
        new bucket there.
        """
        key = by.layer.isolation
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = self._new()

        def release() -> None:
            undo(bucket)
            if self._buckets.get(key) is bucket and self._empty(bucket):
                del self._buckets[key]
            if then is not None:
                then()

        releaser: Releaser | None = None
        try:
            # The teardown is registered before the change that it undoes. A
            # pair that is down refuses here. When the change was applied first,
            # it stayed in a bucket that nothing would ever empty.
            releaser = by.add_disposer(release, label=label)
            mutate(bucket)
        except BaseException:
            if releaser is not None:
                releaser.drop()
            # A refused registration (a duplicate name, an owner that is down)
            # leaves no empty bucket.
            if self._empty(bucket) and self._buckets.get(key) is bucket:
                del self._buckets[key]
            raise
        return releaser

    def get(self, key: Context | None) -> B | None:
        """The bucket of one scope. The key `None` gives the bucket of the global layer."""
        return self._buckets.get(key)

    def along(self, chain: Iterable[Context | None]) -> list[B]:
        """The buckets that an isolation chain resolves to, in the order of the chain."""
        return [bucket for key in chain if (bucket := self._buckets.get(key)) is not None]

    def __len__(self) -> int:
        return len(self._buckets)


class ScopedEntries[T](ScopedTable[list[T]]):
    """A `ScopedTable` of lists (filters, limits). Each value is removed by identity
    (`remove_identical`), as `claim_entry` removes. Otherwise, when two rows
    contribute an equal value, the disposer of one would remove the value of the
    other."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(list, lambda entries: not entries)

    def add(
        self, by: Running, value: T, *, label: str, then: Callable[[], None] | None = None
    ) -> Disposer:
        """Append `value` to the bucket of the layer of `by` (`claim`)."""
        return self.claim(
            by,
            lambda entries: entries.append(value),
            lambda entries: remove_identical(entries, value),
            label=label,
            then=then,
        )

    def gathered(self, chain: Iterable[Context | None]) -> list[T]:
        """Every value along an isolation chain, bucket by bucket in the order of the chain."""
        return [value for bucket in self.along(chain) for value in bucket]
