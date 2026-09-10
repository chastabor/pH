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

**The attach reply is the first status frame, and `seed` is what says so.** The
same gap has a second half nobody named: a root that was already idle when the
attach landed announces *nothing* afterwards, so that reply is the only status the
feed will ever see. Both callers compensated for it and they did it differently —
the TUI replayed the reply through its own handler, while the CLI made a second
`session/status` round trip for a fact the daemon had already handed it, and then
re-decided "idle means done" outside the handler that decides it. `seed` is the
one compensation.

@module ph_app.daemon.follow
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio

from ..params import SnapshotParams
from ..payloads import (
    FED,
    AttachReply,
    SessionEventNotice,
    SessionStatusNotice,
    SnapshotPage,
    StatusFacts,
    notice_of,
)
from ..protocol import Cursor
from ..wire import as_int
from .client import DaemonClient

__all__ = ["Followed", "first_of"]

log = logging.getLogger("ph_app.daemon.follow")

Sink = Callable[[Sequence[tuple[Mapping[str, Any], Any]], bool], None]
"""Called with `(event, view)` pairs and whether they are arriving **live**.

The pairs keep the card the daemon rendered *beside* the event, never merged into
it: `_EventWire` forbids extras, so an event carrying a `presentation` key fails
validation and is dropped by the `except` that exists for unreadable frames.

The flag says whether they are arriving live or being rebuilt — a distinction
`TuiEventAdapter.Frame(live=…)` has drawn since P3, and one a consumer must not
assume. `test_a_caught_up_page_is_folded_as_history_and_a_frame_as_live` is why."""

StatusSink = Callable[[StatusFacts], None]
"""Where a status frame goes. `StatusFacts` and not a mapping, because the three
shapes that reach it — the attach reply, an `announce`, a bare `passivated` —
are one type with optional fields rather than three dicts a reader must sniff."""


@dataclass(slots=True)
class Followed:
    """One session's feed: buffered until catch-up finishes, then live.

    `pending` is the buffer *and* the phase — `None` means live — because a
    separate flag beside it was a second copy of one fact that could be set apart
    from it.
    """

    session_id: str
    on_events: Sink
    on_status: StatusSink
    """The raw `session.status` params; the caller reads what it wants from them."""
    seen: int = -1
    """The highest seq already shown. **`-1`, not `0`**: a log's first event *is*
    seq 0, so a zero sentinel conflated "nothing seen" with "seen the first one"
    and silently dropped the opening event of every session that had none before
    the attach — which is every new one. It surfaced only where the log was later
    rebuilt as a `Session`, whose seed must be contiguous from 0."""
    pending: list[tuple[str, dict[str, Any]]] | None = field(default_factory=list)

    def __call__(self, method: str, params: dict[str, Any]) -> None:
        """The client's notification callback. Sync, because the pump is."""
        if self.pending is not None:
            self.pending.append((method, params))
            return
        # Read off the raw frame, before any model sees it: "is this mine"
        # is the one question that must be answered *without* validating,
        # because a client watching one root receives notices for the
        # others and parsing them to discard them is the work this skips.
        if params.get("sessionId") != self.session_id:
            return
        # Gated before the table: this reader wants two of the six notices, and
        # `Root.publish` fans all of them to every subscriber — so without this
        # a palette republish was fully validated and then thrown away.
        if method not in FED:
            return
        # Through the table that owns method → payload, then narrowed by type:
        # a notice this build has no model for is `None`, which is a daemon
        # newer than this client and a row to drop rather than a crash. Events
        # first, because they are the hot one and an `isinstance` *hit* on the
        # exact type takes a fast path a miss does not.
        notice = notice_of(method, params)
        if not isinstance(notice, SessionEventNotice):
            if isinstance(notice, SessionStatusNotice):
                self.on_status(notice)
            return
        event = notice.event
        at = as_int(event.get("seq"), -1)
        if at < 0:
            # The one reader here whose input never passed `freeze_json_value`:
            # `SessionEventNotice.event` is carried by reference off the socket
            # and nothing in this process validates its contents. A junk `seq`
            # used to raise out of this call and `Peer._watch` logged it as "a
            # notification this end could not read"; reading it as a default
            # silently drops the frame instead, in a module where every other
            # drop explains itself. Say so, once per frame, and keep the drop.
            log.warning("ph_app.daemon.follow: a session event arrived with no readable seq")
            return
        if at <= self.seen:
            # Already shown by a snapshot page. Dropped by `seq` rather than by
            # remembering which frames were buffered, which is what makes the two
            # sources idempotent against each other.
            return
        self.seen = at
        self.on_events([(event, notice.presentation)], True)

    def seed(self, attached: AttachReply) -> None:
        """Deliver the attach reply as this feed's first status.

        The reply is `root.describe()` plus the footer — the *same shape*
        `session.status` sends, which the server says in as many words — so it goes
        to the same sink rather than to a caller-side special case.

        **Delivered now rather than held until `live()`, because status has no
        catch-up phase to wait for.** `session/snapshot` pages *events*; the reply
        is already the whole of the status history, so there is nothing for it to
        arrive out of order with. That is also what lets a front end draw its
        footer while a long log is still paging, and it makes the busy and
        already-idle cases print in one order instead of two — the difference that
        used to be settled by which side of a race a host landed on.
        """
        self.on_status(attached.facts())

    def live(self) -> None:
        """Catch-up is done: go live, then release what arrived during it."""
        held, self.pending = self.pending or [], None
        for method, params in held:
            self(method, params)

    async def catch_up(self, client: DaemonClient, cursor: Cursor | None) -> int:
        """Page from `cursor` to the head, one sink call per page.

        Paged because `session/snapshot` is the only mechanism that catches up
        (`session/attach` deliberately does not replay: streaming a gap of unknown
        size into a bounded outbox fails at exactly the moment it matters), and a
        page at a time is one write at a time, which is what keeps a resumed
        root's whole log from being rendered event by event.

        **Returns the seq the daemon actually started from**, which is the reply's
        own `from`. A cursor names a position *in one incarnation of the log*;
        `resume_at` answers a cursor from another incarnation with "you have seen
        nothing of this one" and pages from 0 — the safe reading, but a silent one
        unless the caller can see that it happened. Read rather than inferred from
        the first event's seq, which is what `resume_at`'s docstring already
        promised and what an empty page cannot answer at all. `seen` is advanced
        from what arrives rather than pre-set from what was asked, so a fallback to
        0 is *shown* instead of having its first `since` events dropped as already
        seen.
        """
        started: int | None = None
        while True:
            page = SnapshotPage.model_validate(
                await client.call(
                    "session/snapshot",
                    SnapshotParams(session_id=self.session_id, cursor=cursor),
                )
            )
            if started is None:
                started = page.started_at
            # Sparse and keyed by seq, which is how the daemon sends it: a page
            # is 2048 events and a turn contributes a handful of cards.
            views = page.presentations
            self.on_events([(one, views.get(str(one.get("seq")))) for one in page.events], False)
            for event in page.events:
                self.seen = max(self.seen, as_int(event.get("seq"), self.seen))
            if not page.more:
                return started
            cursor = page.cursor


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
