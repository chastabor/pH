"""Which records explain a replacement, and what a node stands for: the index behind
`Session.records_of` and `Session.history`.

Apart from `ph.session.revise`, which writes what this reads, so `Session` can hold
one without importing the door that imports it.

@module ph.session.revision_index
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .events import SessionEvent
from .surface import is_replacement_surface_event, shadowed_by
from .writers import REVISES

__all__ = ["RevisionIndex", "revises_of"]


def revises_of(record: SessionEvent) -> tuple[int, ...]:
    """The replacements the door stamped `record` with, in the order it named them:
    empty for a record no revision wrote."""
    revised = record.data.get(REVISES)
    if not isinstance(revised, Sequence) or isinstance(revised, str):
        return ()
    return tuple(node for node in revised if isinstance(node, int) and not isinstance(node, bool))


class RevisionIndex:
    """Each replacement's records, by the `revises` the door stamped on them, and each
    node's history: the index behind `Session.records_of` and `Session.history`, extended
    as the log grows rather than rescanned."""

    __slots__ = ("_by_node", "_history", "_seen")

    def __init__(self) -> None:
        self._by_node: dict[int, tuple[SessionEvent, ...]] = {}
        self._history: dict[int, tuple[SessionEvent, ...]] = {}
        """Kept for good once worked out, for a node with any: what a node stands for
        is fixed when it is written, so its history never changes."""
        self._seen = 0

    def read(self, log: Sequence[SessionEvent], seq: int) -> tuple[SessionEvent, ...]:
        """The records that revise node `seq`, in log order."""
        for index in range(self._seen, len(log)):
            event = log[index]
            # Most events are chunks with no stamp, so they cost one lookup.
            if REVISES not in event.data:
                continue
            for node in revises_of(event):
                self._by_node[node] = (*self._by_node.get(node, ()), event)
        self._seen = len(log)
        return self._by_node.get(seq, ())

    def history(
        self, at: Callable[[int], SessionEvent | None], seq: int
    ) -> tuple[SessionEvent, ...]:
        """Every replacement among node `seq` and what it stands for, oldest first."""
        known = self._history.get(seq)
        if known is not None:
            return known
        found: dict[int, SessionEvent] = {}
        pending = [seq]
        while pending:
            node = pending.pop()
            earlier = self._history.get(node)
            if earlier is not None:
                found.update((one.seq, one) for one in earlier)
                continue
            event = at(node)
            if event is not None and is_replacement_surface_event(event):
                found[event.seq] = event
                pending.extend(shadowed_by(event))
        history = tuple(found[one] for one in sorted(found))
        if history:
            self._history[seq] = history
        return history
