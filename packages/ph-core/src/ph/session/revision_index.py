"""Which records explain a replacement, and what a node stands for. This is the
index behind `Session.records_of` and `Session.history`.

The index is in a module apart from `ph.session.revise`, which writes what this
module reads. So `Session` can hold an index without importing the revision
module, which imports `Session`.

@module ph.session.revision_index
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .events import SessionEvent
from .surface import is_replacement_surface_event, shadowed_by
from .writers import REVISES

__all__ = ["RevisionIndex", "revises_of"]


def revises_of(record: SessionEvent) -> tuple[int, ...]:
    """The replacements that a revision function (`ph.session.revise`) stamped on
    `record`, in the order in which it named them. Empty for a record that no
    revision wrote."""
    revised = record.data.get(REVISES)
    if not isinstance(revised, Sequence) or isinstance(revised, str):
        return ()
    return tuple(node for node in revised if isinstance(node, int) and not isinstance(node, bool))


class RevisionIndex:
    """The records of each replacement, by the `revises` key that a revision
    function stamped on them, and the history of each node. This is the index
    behind `Session.records_of` and `Session.history`. It is extended as the log
    grows and is not rebuilt by a new scan."""

    __slots__ = ("_by_node", "_history", "_seen")

    def __init__(self) -> None:
        self._by_node: dict[int, tuple[SessionEvent, ...]] = {}
        self._history: dict[int, tuple[SessionEvent, ...]] = {}
        """Kept permanently after it is worked out, for a node that has any
        history. What a node stands for is fixed when the node is written, so
        its history never changes."""
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
