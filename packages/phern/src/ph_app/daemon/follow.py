"""Follow one session on the daemon: page to catch up, then go live, with no event shown twice.

`session/attach` subscribes before the history is fetched. That is deliberate,
because the other order drops everything that happens in between. Live frames
therefore arrive while `session/snapshot` is still paging. This module holds
them and releases them afterwards, and discards by `seq` anything that the pages
already showed. That stops one event from arriving twice and a later event from
arriving first.

That rule was written twice, for `phern agents attach` and for the TUI over a
socket, and the invariant that it protects was tested one time. This is the one
copy. A caller supplies the sink and keeps what is its own: a console and a
`--type` filter for the CLI, a transcript fold for the TUI.

**The attach reply is the first status frame, and `seed` is what says so.** The
same gap has a second half that nobody named. A root that was already idle when
the attach arrived announces *nothing* afterwards, so that reply is the only
status that the feed will ever see. Both callers compensated for it, and in
different ways. The TUI replayed the reply through its own handler. The CLI made
a second `session/status` round trip for a fact that the daemon had already
given it, and then decided "idle means done" again, outside the handler that
decides it. `seed` is the one compensation.

@module ph_app.daemon.follow
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ph.json import JsonObject, JsonValue, as_int

from .. import verbs
from ..params import SnapshotParams
from ..payloads import (
    FED,
    AttachReply,
    SessionEventNotice,
    SessionStatusNotice,
    StatusFacts,
    notice_of,
)
from ..protocol import Cursor
from .client import DaemonClient

__all__ = ["EventFrame", "Followed"]

log = logging.getLogger("ph_app.daemon.follow")

type EventFrame = tuple[JsonObject, JsonValue]
"""One session event as it came off the wire, and the card that was rendered
beside it.

**Neither half is an event or a card yet.** The two halves have different types
because the two producers disagree. The envelope is an object in both cases, and
`SessionEvent.from_wire` is what proves that it is an *event*. The card is a
bare `JsonValue` because only one of its two sources says more. A live frame
arrives as `SessionEventNotice.presentation: dict[str, Any] | None`, and
pydantic has already checked that it is an object. A rebuilt card is one of the
*values* of `SnapshotPage.presentations`. Those are typed `Any`, so nothing has
looked at them. `JsonValue` is what is true of both.

A tighter `presentations` type, `dict[str, dict[str, Any]]`, would make the pair
`JsonObject | None` and retire the `isinstance` in `view_of`. It was rejected.
It moves the failure from "this one card renders generic" to "this whole
2048-event page fails validation", which is the opposite of what `view_of` is
written for.

This pair was `tuple[Mapping[str, Any], Any]`, which said none of that. Every
implementer inherited two `Any`s, and a consumer could not tell "arbitrary JSON"
from "not typed yet"."""

Sink = Callable[[Sequence[EventFrame], bool], None]
"""Called with `EventFrame`s and whether they arrive **live**.

The pairs keep the card that the daemon rendered *beside* the event, and never
merge it into the event. `_EventWire` forbids extras, so an event with a
`presentation` key fails validation, and the `except` that exists for unreadable
frames drops it.

The flag says whether the frames arrive live or are rebuilt. `Frame(live=…)` in
`ph_app.tui.adapter` has drawn that distinction since P3, and a consumer must
not assume it. `test_a_caught_up_page_is_folded_as_history_and_a_frame_as_live`
shows why."""

StatusSink = Callable[[StatusFacts], None]
"""Where a status frame goes. It takes `StatusFacts` and not a mapping. The three
shapes that reach it (the attach reply, an `announce`, a bare `passivated`) are
one type with optional fields, and not three dicts that a reader must inspect."""


def _seq_of(event: dict[str, Any]) -> int | None:
    """The seq of a wire event, or `None` when the event has no readable seq.

    A seq is never negative, so a negative number is unreadable too. The event
    comes from the socket by reference, and nothing in this process has
    validated its contents."""
    at = as_int(event.get("seq"), -1)
    return at if at >= 0 else None


@dataclass(slots=True)
class Followed:
    """The feed of one session: buffered until catch-up finishes, then live.

    `pending` is the buffer *and* the phase, where `None` means live. A separate
    flag was a second copy of one fact, and it could be set separately from the
    buffer.
    """

    session_id: str
    on_events: Sink
    on_status: StatusSink
    """Where a status goes. The caller reads what it wants from the `StatusFacts`."""
    seen: int | None = None
    """The highest seq already shown, or `None` when nothing has been shown.

    **"Nothing shown" is `None` and not a number.** The first event of a log
    *is* seq 0. When this field started at `0`, "nothing seen" and "seen the
    first one" were the same value. That silently dropped the opening event of
    every session that had no events before the attach, which is every new
    session. The fault appeared only where the log was later rebuilt as a
    `Session`, whose seed must be contiguous from 0. The field then started at
    `-1`, which worked but was still a number that stood for "no number"."""
    pending: list[tuple[str, dict[str, Any]]] | None = field(default_factory=list)

    def __call__(self, method: str, params: dict[str, Any]) -> None:
        """The client's notification callback. Sync, because the pump is."""
        if self.pending is not None:
            self.pending.append((method, params))
            return
        # This is read from the raw frame, before any model sees it. "Is this
        # mine" is the one question that must be answered *without* validation.
        # A client that watches one root receives notices for the other roots,
        # and this check skips the work of parsing them only to discard them.
        if params.get("sessionId") != self.session_id:
            return
        # This gate comes before the table. This reader wants two of the six
        # notices, and `Root.publish` fans all of them out to every subscriber.
        # Without the gate, a palette republish was fully validated and then
        # thrown away.
        if method not in FED:
            return
        # The frame goes through the table that owns the mapping from method to
        # payload, and is then narrowed by type. A notice for which this build
        # has no model is `None`. That means a daemon newer than this client,
        # and a row to drop, not a crash. Events are tested first, because they
        # are the hot path, and an `isinstance` *hit* on the exact type takes a
        # fast path that a miss does not.
        notice = notice_of(method, params)
        if not isinstance(notice, SessionEventNotice):
            if isinstance(notice, SessionStatusNotice):
                self.on_status(notice)
            return
        event = notice.event
        at = _seq_of(event)
        if at is None:
            # This is the one reader here whose input never passed
            # `freeze_json_value`. `SessionEventNotice.event` is carried by
            # reference from the socket, and nothing in this process validates
            # its contents. A junk `seq` used to raise out of this call, and
            # `Peer._watch` logged it as "a notification this end could not
            # read". A read with a default drops the frame silently instead, in
            # a module where every other drop explains itself. So this logs a
            # warning, one time for each frame, and keeps the drop.
            log.warning("ph_app.daemon.follow: a session event arrived with no readable seq")
            return
        if self.seen is not None and at <= self.seen:
            # A snapshot page already showed it. The drop is by `seq`, and not by
            # a record of which frames were buffered. That makes the two sources
            # idempotent against each other.
            return
        self.seen = at
        self.on_events([(event, notice.presentation)], True)

    def seed(self, attached: AttachReply) -> None:
        """Deliver the attach reply as the first status of this feed.

        The reply is `root.describe()` plus the footer. That is the *same shape*
        that `session.status` sends, as the server states explicitly. So it goes
        to the same sink, and not to a special case on the caller side.

        **It is delivered now and not held until `live()`, because status has no
        catch-up phase to wait for.** `session/snapshot` pages *events*. The
        reply is already the whole status history, so it cannot arrive out of
        order with anything. This also lets a front end draw its footer while a
        long log is still paging. And the busy case and the already-idle case
        now print in one order and not in two. Which side of a race a host
        landed on used to decide that difference.
        """
        self.on_status(attached.facts())

    def live(self) -> None:
        """Catch-up is done: go live, then release what arrived during it."""
        held, self.pending = self.pending or [], None
        for method, params in held:
            self(method, params)

    async def catch_up(self, client: DaemonClient, cursor: Cursor | None) -> int:
        """Page from `cursor` to the head, with one sink call for each page.

        The catch-up is paged because `session/snapshot` is the only mechanism
        that catches up. `session/attach` deliberately does not replay: a stream
        of a gap of unknown size into a bounded outbox fails at exactly the
        moment when it matters. One page at a time is also one write at a time,
        so the whole log of a resumed root is not rendered event by event.

        **Returns the seq from which the daemon actually started**, which is the
        `from` of the reply itself. A cursor names a position *in one
        incarnation of the log*. `resume_at` honors a cursor from an earlier
        incarnation only as far as every resume since then kept. It answers a
        cursor from another log with "you have seen nothing of this one" and
        pages from 0. These are the safe readings, but they are silent unless
        the caller can see that they happened. The value is read from the reply
        and not inferred from the seq of the first event. The docstring of
        `resume_at` already promised that, and an empty page cannot answer it at
        all. `seen` is advanced from what arrives and is not set in advance from
        what was asked. A fallback to 0 is therefore *shown*, and its first
        `since` events are not dropped as already seen.
        """
        started: int | None = None
        while True:
            page = await client.call(
                verbs.SESSION_SNAPSHOT,
                SnapshotParams(session_id=self.session_id, cursor=cursor),
            )
            if started is None:
                started = page.started_at
            # The cards are sparse and keyed by seq, which is how the daemon sends
            # them: a page is 2048 events and a turn contributes a few cards.
            views = page.presentations
            self.on_events([(one, views.get(str(one.get("seq")))) for one in page.events], False)
            for event in page.events:
                at = _seq_of(event)
                if at is not None and (self.seen is None or at > self.seen):
                    self.seen = at
            if not page.more:
                return started
            cursor = page.cursor
