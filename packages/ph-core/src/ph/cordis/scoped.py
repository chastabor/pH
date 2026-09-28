"""Registrations bucketed by the isolation scope they narrow or shadow — one table shape.

A registry that answers per agent keys what it holds by scope: the tool registry's
layers, the skill registry's restrictions, the sandbox's per-child path limits. Each
wrote the same three things by hand, and each got one of them wrong at some point:

1. **the key is the registration's layer** (`Running.layer.isolation`), so a
   registration made for an agent lands on that agent and not on the row's scope;
2. **the release runs on whichever of the row and that scope ends first**
   (`Running.add_disposer`), and it is the disposer the scope holds that announces the
   change — a registry that wrapped the returned disposer told nobody when a scope
   unwound — and **the bucket goes with its last entry**: one left keyed by a settled
   scope holds that scope, and a table that never empties keeps every reader off its
   nothing-registered fast path;
3. **a read walks an isolation chain**, in the chain's order.

`ScopedTable` is the shape, with a bucket of any kind; `ScopedEntries` is the common
case, a list of values each removed by identity.

@module ph.cordis.scoped
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from .context import remove_identical

if TYPE_CHECKING:
    from .context import Context, Disposer, Running

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
        """Apply `mutate` to the bucket of `by`'s layer; the disposer undoes it.

        The disposer is `by`'s — the pair's lifetime — and it runs `undo`, drops the
        bucket once `empty`, then `then`, so a registry that announces its changes
        announces the one an unwinding scope makes too. `undo` is applied to the
        bucket `mutate` was, even if the table has since keyed a new one there.
        """
        key = by.layer.isolation
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = self._buckets[key] = self._new()
        try:
            mutate(bucket)
        except BaseException:
            # A registration refused — a duplicate name — leaves no empty bucket.
            if self._empty(bucket) and self._buckets.get(key) is bucket:
                del self._buckets[key]
            raise

        def release() -> None:
            undo(bucket)
            if self._buckets.get(key) is bucket and self._empty(bucket):
                del self._buckets[key]
            if then is not None:
                then()

        return by.add_disposer(release, label=label)

    def get(self, key: Context | None) -> B | None:
        """The bucket of one scope — `None` for the global layer's."""
        return self._buckets.get(key)

    def along(self, chain: Iterable[Context | None]) -> list[B]:
        """The buckets an isolation chain resolves to, in the chain's order."""
        return [bucket for key in chain if (bucket := self._buckets.get(key)) is not None]

    def __len__(self) -> int:
        return len(self._buckets)


class ScopedEntries[T](ScopedTable[list[T]]):
    """A `ScopedTable` of lists — filters, limits — each value removed by identity
    (`remove_identical`), as `claim_entry` removes: two rows contributing an equal
    value would otherwise have one disposer take the other's."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(list, lambda entries: not entries)

    def add(
        self, by: Running, value: T, *, label: str, then: Callable[[], None] | None = None
    ) -> Disposer:
        """Append `value` to the bucket of `by`'s layer (`claim`)."""
        return self.claim(
            by,
            lambda entries: entries.append(value),
            lambda entries: remove_identical(entries, value),
            label=label,
            then=then,
        )

    def gathered(self, chain: Iterable[Context | None]) -> list[T]:
        """Every value along an isolation chain, bucket by bucket in the chain's order."""
        return [value for bucket in self.along(chain) for value in bucket]
