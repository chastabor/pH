"""Following one session on the daemon: catch up by paging, then go live, never twice.

`session/attach` subscribes before the history is fetched — deliberately, since
the other order drops everything that happens in between — so live frames arrive
while `session/snapshot` is still paging. Holding them and releasing them after,
with anything the pages already showed discarded by `seq`, is what stops one event
landing twice and a later one landing first.

That rule was written twice — once for `ph agents attach`, once for the TUI over a
socket — and the invariant it protects was tested once. This is the one copy; a
caller supplies the sink and keeps whatever is its own (a console and a `--type`
filter for the CLI, a transcript fold for the TUI).

@module ph_app.daemon.follow
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio

from ..wire import obj, seq
from .client import DaemonClient

__all__ = ["Followed", "first_of"]

Sink = Callable[[Sequence[tuple[Mapping[str, Any], Any]]], None]
"""Called with `(event, view)` pairs — the card the daemon rendered kept *beside*
the event, never merged into it: `_EventWire` forbids extras, so an event carrying
a `presentation` key fails validation and is dropped by the `except` that exists
for genuinely unreadable frames."""


@dataclass(slots=True)
class Followed:
    """One session's feed: buffered until catch-up finishes, then live.

    `pending` is the buffer *and* the phase — `None` means live — because a
    separate flag beside it was a second copy of one fact that could be set apart
    from it.
    """

    session_id: str
    on_events: Sink
    on_status: Callable[[Mapping[str, Any]], None]
    """The raw `session.status` params; the caller reads what it wants from them."""
    seen: int = 0
    pending: list[tuple[str, dict[str, Any]]] | None = field(default_factory=list)

    def __call__(self, method: str, params: dict[str, Any]) -> None:
        """The client's notification callback. Sync, because the pump is."""
        if self.pending is not None:
            self.pending.append((method, params))
            return
        if params.get("sessionId") != self.session_id:
            return
        if method == "session.status":
            self.on_status(params)
            return
        if method != "session.event":
            return
        event = obj(params.get("event"))
        at = int(event.get("seq", -1))
        if at <= self.seen:
            # Already shown by a snapshot page. Dropped by `seq` rather than by
            # remembering which frames were buffered, which is what makes the two
            # sources idempotent against each other.
            return
        self.seen = at
        self.on_events([(event, params.get("presentation"))])

    def live(self) -> None:
        """Catch-up is done: go live, then release what arrived during it."""
        held, self.pending = self.pending or [], None
        for method, params in held:
            self(method, params)

    async def catch_up(self, client: DaemonClient, cursor: Any) -> None:
        """Page from `cursor` to the head, one sink call per page.

        Paged because `session/snapshot` is the only mechanism that catches up
        (`session/attach` deliberately does not replay: streaming a gap of unknown
        size into a bounded outbox fails at exactly the moment it matters), and a
        page at a time is one write at a time, which is what keeps a resumed
        root's whole log from being rendered event by event.
        """
        while True:
            page = await client.call("session/snapshot", sessionId=self.session_id, cursor=cursor)
            events = [obj(wire) for wire in seq(page.get("events"))]
            # Sparse and keyed by seq, which is how the daemon sends it: a page
            # is 2048 events and a turn contributes a handful of cards.
            views = obj(page.get("presentations"))
            self.on_events([(one, views.get(str(one.get("seq")))) for one in events])
            for event in events:
                self.seen = max(self.seen, int(event.get("seq", self.seen)))
            if not page.get("more"):
                return
            cursor = page.get("cursor")


async def first_of(*events: anyio.Event) -> None:
    """Wait for whichever of these happens first.

    Two ways a follow ends — the root went idle, or the daemon went away — and
    waiting on only the first is a hang whenever it is the second that happens.
    """
    async with anyio.create_task_group() as tasks:

        async def stop_on(event: anyio.Event) -> None:
            await event.wait()
            tasks.cancel_scope.cancel()

        for event in events:
            tasks.start_soon(stop_on, event)
