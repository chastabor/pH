"""One end of a two-way JSON-RPC connection, for both ends to be.

`protocol.py` holds the *stateless* half of the vocabulary — how a request, a
notification and a reply are shaped — and deliberately imports nothing. This is
the stateful half: minting ids, remembering what is outstanding, deciding which
direction an inbound frame is going, serialising what goes out, and waking
everybody when the socket ends.

**It exists because that half was written twice.** Until P5-13 the protocol only
went one way for anything expecting an answer, so the client had a pending table
and the server did not. Making the daemon able to ask a person gave the server
one too — and the two copies immediately disagreed about the case that matters:
a connection dying mid-request raised `DaemonGone` on one side and returned an
empty `{}` on the other, which reads downstream as a successful answer with no
fields and denied the call. One object, one answer.

Not a seam: nothing swaps it, it is built before any `Context` exists, and both
ends run *the same code* rather than talking through it — the socket is what they
talk through.

## The three buffers, which are not the same buffer

A duplex peer queues in three places and they have different jobs, different
sizes and — the reason a shared object needs saying out loud — different
overflow policies:

* **Outbound frames** go through one memory stream drained by one writer task, so
  a notification arriving while a reply is half-written cannot interleave two
  frames on the wire. Two ways in, because the two ends need opposite things when
  it is full: `tell()` refuses (`WouldBlock`) so the daemon can *drop* a watcher
  that cannot keep up — a subscriber must never become the thing the work waits
  on — while `send()` blocks, which is right for a client, who has nobody to drop
  but itself.
* **Inbound requests** are bounded by a semaphore held from the read loop and
  released by the handler. A handler may park on a human, so the loop cannot
  await one; without a ceiling "do not await" silently becomes "accept without
  limit", and a peer that pipelines parks tens of megabytes of handler tasks. The
  limiter hands backpressure back to the socket, which is where it was before
  dispatching moved off the loop.
* **Outstanding asks** are a correlation table, not a queue: id → the event its
  caller is parked on. It shrinks as answers land and is emptied on close, which
  is what stops a caller waiting for a frame that is never coming.

@module ph_app.daemon.duplex
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import anyio
from anyio.abc import ByteStream

from ..protocol import DaemonGone, Dispatch, notification, request, respond, result_of
from .framing import FramingError, read_frames, write_frame

__all__ = ["IN_FLIGHT", "OUTBOX", "Handler", "Notification", "Peer"]

log = logging.getLogger("ph_app.daemon.duplex")

Notification = Callable[[str, dict[str, Any]], None]
Handler = Callable[[dict[str, Any]], Awaitable[Any]]

OUTBOX = 1024
"""Frames that may be queued for the wire before `tell` refuses.

Large enough that a burst of `session.event` during a fast turn is absorbed, and
finite so that "this peer is not reading" is a thing the sender can find out."""

IN_FLIGHT = 64
"""Requests from one peer that may be handled at once.

Generous for any real client — a TUI has one call outstanding and `ph agents
attach` pages snapshots serially — and finite, which is the property that
matters."""


@dataclass(slots=True)
class _Pending:
    """One ask this end is waiting on, and the reply when it lands."""

    answered: anyio.Event
    reply: dict[str, Any] | None = None


@dataclass(slots=True)
class Peer:
    """One end of a connection: what it can ask, what it will answer, what it owes."""

    stream: ByteStream
    dispatch: Dispatch
    """What this end does with an inbound *request*. `respond` shapes the reply,
    so a handler that raises becomes an error frame and the asker is settled
    either way."""
    on_notify: Notification | None = None
    id_prefix: str = ""
    """What this end's ids start with, so a frame log says which side asked.

    Both ends mint ids now, and they must not collide: the client uses `c`, the
    daemon `s`."""
    outbox_size: int = OUTBOX
    in_flight: int = IN_FLIGHT
    closed: anyio.Event = field(default_factory=anyio.Event)
    """Set when the loop stops, whichever end stopped it.

    "The peer went away" is a thing callers have to be able to *wait for* rather
    than only notice — `shutdown` is a notification by contract, so the only
    honest confirmation is the connection closing."""
    _asked: int = 0
    _pending: dict[str, _Pending] = field(default_factory=dict)
    _outbox: Any = None
    _inbox: Any = None
    _tasks: Any = None
    _limit: anyio.Semaphore | None = None

    # ------------------------------------------------------------- outbound --

    def _queue(self) -> Any:
        """The outbound stream, built on first use.

        **Not in `serve()`**, which is the obvious place and is a race: a caller
        may `call()` the moment after `start_soon(peer.serve)` and before that
        task has had a turn, and the frame has to have somewhere to go. The queue
        is part of what this end *is*; draining it is what `serve` does.
        """
        if self._outbox is None and self._inbox is None:
            self._outbox, self._inbox = anyio.create_memory_object_stream[dict[str, Any]](
                max_buffer_size=self.outbox_size
            )
        if self._outbox is None:
            raise DaemonGone
        return self._outbox

    def tell(self, method: str, params: dict[str, Any]) -> None:
        """Queue a notification, or **raise** so the caller drops this peer.

        Raising is the point, and the daemon's watcher policy depends on it:
        catching `WouldBlock` here and logging "dropped" drops nothing, and the
        peer that cannot keep up re-pays the whole fan-out for every later event.
        The subscriber list belongs to whoever owns it, so this only has to fail
        loudly enough to be noticed.
        """
        self._queue().send_nowait(notification(method, params))

    async def send(self, frame: dict[str, Any]) -> None:
        """Queue any frame, waiting for room. For an end with nobody to drop."""
        await self._queue().send(frame)

    async def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Put a request to the other end and wait for its answer.

        **No timeout.** On the daemon's side the answer is a person, and a
        deadline would turn "they went to lunch" into a denial. What bounds it is
        the connection: if the peer goes away this raises, and the caller decides
        what that means — for an approval it means the question stays open for
        whoever attaches next.
        """
        self._asked += 1
        ask_id = f"{self.id_prefix}{self._asked}"
        entry = _Pending(answered=anyio.Event())
        self._pending[ask_id] = entry
        try:
            await self.send(request(ask_id, method, params))
            await entry.answered.wait()
        finally:
            self._pending.pop(ask_id, None)
        if entry.reply is None:
            # Woken by the connection ending rather than by an answer. Named,
            # because an empty `{}` reads as a successful reply with no fields
            # in it — and named as a *disconnection*, because nobody refused.
            raise DaemonGone
        return result_of(entry.reply)

    # -------------------------------------------------------------- the loop --

    async def serve(self) -> None:
        """Read until the socket closes, answering and settling as frames arrive."""
        send, receive = self._queue(), self._inbox
        self._limit = anyio.Semaphore(self.in_flight)
        try:
            async with anyio.create_task_group() as tasks:
                self._tasks = tasks
                tasks.start_soon(self._write, receive)
                # Suppressed *inside* the group, and that is load-bearing: anyio
                # wraps even a single exception leaving a task group into an
                # `ExceptionGroup` that no `except` naming the original can see.
                with suppress(
                    anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream
                ):
                    await self._read()
                # A handler still parked on a person is the one thing expected to
                # be in flight: the socket is gone, so its answer has nowhere to
                # go and the question belongs to whoever owns it.
                tasks.cancel_scope.cancel()
        finally:
            # In `finally`, so a waiter is woken by a cancellation and a crash as
            # well as by an orderly end. A wait that only completes on the happy
            # path is a hang wearing a timeout.
            self._outbox = None
            self.closed.set()
            for entry in self._pending.values():
                entry.answered.set()
            self._pending.clear()
            with suppress(anyio.ClosedResourceError):
                await send.aclose()

    async def _write(self, receive: Any) -> None:
        async with receive:
            async for frame in receive:
                try:
                    await write_frame(self.stream, frame)
                except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                    return

    async def _read(self) -> None:
        """Route each frame by direction, and never handle one inline.

        `method` is the discriminator, not `id` — see `protocol.request`. Both
        ends were written assuming otherwise, and each got the same bug from its
        own side: the daemon dispatched a client's *answer* as method `""` and
        bounced it back as `unknown_method`, while the client filed the daemon's
        *request* as the answer to a call nobody made.
        """
        try:
            async for frame in read_frames(self.stream):
                if "method" not in frame:
                    self._settle(frame)
                    continue
                if frame.get("id") is None and self.on_notify is not None:
                    # **An id-less frame means different things to the two ends,
                    # and `on_notify` is which end this is.** To a client a
                    # notification is an event to watch — `session.event`,
                    # `session.status` — with no body to run. To the daemon it is
                    # a *method whose answer nobody wants*: `shutdown` carries no
                    # id by contract, because a reply would have the caller
                    # waiting on a frame the daemon is losing the ability to
                    # write. So an end with an observer observes, and an end
                    # without one dispatches — and `respond` returns `None` for
                    # an id-less frame, so the body runs and nothing is written
                    # back.
                    self.on_notify(str(frame.get("method") or ""), frame.get("params") or {})
                    continue
                assert self._limit is not None and self._tasks is not None
                await self._limit.acquire()
                self._tasks.start_soon(self._handle, frame)
        except FramingError as error:
            # Unreadable framing ends the connection: after a bad frame there is
            # no way to know where the next one starts.
            log.info("ph_app.daemon: closing a connection — %s", error)

    async def _handle(self, frame: dict[str, Any]) -> None:
        try:
            reply = await respond(frame, self.dispatch)
        finally:
            if self._limit is not None:
                self._limit.release()
        if reply is not None and self._outbox is not None:
            with suppress(anyio.WouldBlock, anyio.ClosedResourceError):
                self._outbox.send_nowait(reply)

    def _settle(self, frame: dict[str, Any]) -> None:
        """An answer to something this end asked.

        An unknown id is dropped rather than raised on: an ask already given up
        on — because the asker was cancelled, or the connection is closing — gets
        answered by a peer that could not have known, and ending the connection
        over a late reply would punish it for a race it did not cause.
        """
        entry = self._pending.pop(str(frame.get("id")), None)
        if entry is None:
            return
        entry.reply = frame
        entry.answered.set()
