"""One end of a two-way JSON-RPC connection, for both ends to be.

`protocol.py` holds the *stateless* half of the vocabulary — how a request, a
notification and a reply are shaped — and deliberately imports nothing of the
daemon's. This is the stateful half: minting ids, remembering what is outstanding, deciding which
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

from ph.json import as_str
from ph.wire import WireModel

from ..payloads import SessionAsk
from ..protocol import (
    DaemonGone,
    Dispatch,
    Frame,
    notification,
    parse_params,
    request,
    respond,
    result_of,
)
from .framing import FramingError, read_frames, write_frame

__all__ = ["IN_FLIGHT", "OUTBOX", "Handler", "Notification", "Peer", "answering"]

log = logging.getLogger("ph_app.daemon.duplex")

Notification = Callable[[str, dict[str, Any]], None]
"""What this end does with an id-less frame the peer sent.

`dict[str, Any]` because a frame the peer sent is a claim — `ph_app.protocol`'s
module docstring is where that rule lives, and this is it applied to the
observer. The models exist and the readers use them (`payloads.notice_of`);
what the transport does not do is pretend a dict is one of them.

The method travels beside the payload rather than inside it because JSON-RPC
already carries it in the envelope, and this signature is the envelope's."""

Handler = Callable[[dict[str, Any]], Awaitable[Any]]
"""What this end will answer when the *peer* asks it something (P5-13).

`dict` in for `Notification`'s reason. `answering` below is the typed door, so
a handler body takes a model and returns one and this alias stays the
transport's business."""


def answering[A: SessionAsk](ask: type[A], answer: Callable[[A], Awaitable[WireModel]]) -> Handler:
    """One typed ask handler, as the `Handler` the transport registers.

    The client's mirror of the daemon's `parse_params` — literally, not by
    analogy: it *is* `parse_params`, so a bad ask is refused with the same
    named `invalid_params` and the same one-line sentence a bad request gets,
    whichever direction it was travelling. Written by hand it was
    `ApprovalAsk.model_validate(params)` at the top of each handler, whose bare
    `ValidationError` carries no `code` — so `respond` sent it back as an
    unnamed `-32000` with a multi-line pydantic dump for a message, which is the
    error shape P8-07 spent a row deleting on the daemon's side.

    Nothing dumps the reply here. `respond` turns a `WireModel` into a frame at
    the one point a result becomes one, and its own comment says none of its
    handlers spells `.to_wire()` at the `return` — this is a handler.

    `A: SessionAsk` rather than `WireModel` for the `METHOD` the refusal names,
    and because a *notice* has no reply to give: the bound is what stops one
    being registered here.
    """

    async def handler(params: dict[str, Any]) -> WireModel:
        return await answer(parse_params(ask.METHOD, ask, params))

    return handler


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
    dispatch_notifications: bool = False
    """Whether an id-less frame is a *method to run* on this end.

    **The one thing the two ends genuinely disagree about, said rather than
    sniffed.** To a client a notification is an event to watch — `session.event`,
    `session.status` — with no body to run. To the daemon it is a method whose
    answer nobody wants: `shutdown` carries no id by contract, because a reply
    would have the caller waiting on a frame the daemon is losing the ability to
    write.

    Inferring it from `on_notify is not None` read the same way and failed
    quietly in one direction: a client built without an observer — `ph agents
    attach` follows a log and needs no callback — then *dispatched* every
    `session.event` into a handler table that has none, spending a task and one
    of `IN_FLIGHT` per event to raise a `LookupError` that `respond` swallows
    because there is no id to answer. Declared, that client drops them for free.
    """
    id_prefix: str = ""
    """What this end's ids start with, so a frame log says which side asked.

    Both ends mint ids now, and they must not collide: the client uses `c`, the
    daemon `s`."""
    closed: anyio.Event = field(default_factory=anyio.Event)
    """Set when the loop stops, whichever end stopped it.

    "The peer went away" is a thing callers have to be able to *wait for* rather
    than only notice — `shutdown` is a notification by contract, so the only
    honest confirmation is the connection closing."""
    _asked: int = 0
    _unwatchable: int = 0
    """How many notifications this end has failed to read. The first is logged
    with its traceback and the rest at debug — see `_watch`."""
    _pending: dict[str, _Pending] = field(default_factory=dict)
    _outbox: Any = None
    _inbox: Any = None

    # ------------------------------------------------------------- outbound --

    def _queue(self) -> Any:
        """The outbound stream, built on first use.

        **Not in `serve()`**, which is the obvious place and is a race: a caller
        may `call()` the moment after `start_soon(peer.serve)` and before that
        task has had a turn, and the frame has to have somewhere to go. The queue
        is part of what this end *is*; draining it is what `serve` does.
        """
        if self.closed.is_set():
            raise DaemonGone
        if self._outbox is None:
            self._outbox, self._inbox = anyio.create_memory_object_stream[Frame](
                max_buffer_size=OUTBOX
            )
        return self._outbox

    def tell(self, method: str, params: dict[str, Any]) -> None:
        """Queue a notification, or **raise** so the caller drops this peer.

        Raising is the point, and the daemon's watcher policy depends on it:
        catching `WouldBlock` here and logging "dropped" drops nothing, and the
        peer that cannot keep up re-pays the whole fan-out for every later event.
        The subscriber list belongs to whoever owns it, so this only has to fail
        loudly enough to be noticed.
        """
        # The relay calls this once per event per watcher, so the built queue is
        # read straight off the field and `_queue()` is only the first time.
        outbox = self._outbox or self._queue()
        outbox.send_nowait(notification(method, params))

    async def send(self, frame: Frame) -> None:
        """Queue any frame, waiting for room. For an end with nobody to drop.

        A `Frame` — one of the four shapes this side builds — and not a dict:
        what goes *out* is ours to get right, and the type is what checks it.
        What comes *in* stays a `dict[str, Any]` (`_settle`, `result_of`),
        because a peer's frame is a claim."""
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
        limit = anyio.Semaphore(IN_FLIGHT)
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(self._write, receive)
                # Suppressed *inside* the group, and that is load-bearing: anyio
                # wraps even a single exception leaving a task group into an
                # `ExceptionGroup` that no `except` naming the original can see.
                with suppress(
                    anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream
                ):
                    await self._read(tasks, limit)
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

    async def _read(self, tasks: Any, limit: anyio.Semaphore) -> None:
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
                if frame.get("id") is None and not self.dispatch_notifications:
                    # An event to watch, and nothing to run — see
                    # `dispatch_notifications`. An end with no observer drops it
                    # here rather than spending a task to find out it has no body.
                    if self.on_notify is not None:
                        self._watch(frame)
                    continue
                # `respond` returns `None` for an id-less frame, so a dispatched
                # notification runs its body and writes nothing back.
                await limit.acquire()
                tasks.start_soon(self._handle, frame, limit)
        except FramingError as error:
            # Unreadable framing ends the connection: after a bad frame there is
            # no way to know where the next one starts.
            log.info("ph_app.daemon: closing a connection — %s", error)

    def _watch(self, frame: dict[str, Any]) -> None:
        """Hand one notification to the observer, and survive what it does.

        **A notification body must not be able to end the connection.** This
        runs *inline on the read loop* — deliberately, since an observer is
        cheap and spending a task per event is not — so anything it raises
        unwinds `_read`, leaves the task group, and takes the socket with it.
        The reader is a client's own code parsing a frame it may not recognise:
        a notice carrying a field this build has never heard of raises out of
        `model_validate`, and before this guard that killed `ph agents attach`
        over a `session.staged` it does not even read.

        The same judgement `_settle` makes about a late reply — "ending the
        connection over it would punish a peer for a race it did not cause" —
        applied to the other id-less path. A bad frame costs a frame, and says
        so once: `exc_info` on the first, then quiet, because a daemon sending
        one unreadable notice will send the next one too and a follower's
        terminal is not the place to print the same traceback per event.
        """
        try:
            self.on_notify(as_str(frame.get("method")), frame.get("params") or {})  # type: ignore[misc]
        except Exception:
            log.log(
                logging.WARNING if not self._unwatchable else logging.DEBUG,
                "ph_app.daemon: dropping a notification this end could not read (%s)",
                frame.get("method"),
                exc_info=not self._unwatchable,
            )
            self._unwatchable += 1

    async def _handle(self, frame: dict[str, Any], limit: anyio.Semaphore) -> None:
        try:
            reply = await respond(frame, self.dispatch)
        finally:
            limit.release()
        if reply is None:
            return
        # **Through `send`, so a reply waits for room rather than being dropped.**
        # A full outbox is a slow *reader*, and discarding a reply strands the
        # asker on the other end until the connection closes — the one frame
        # where dropping is least defensible, and the only one nobody chose to
        # drop. Blocking here is the backpressure working: this task holds one of
        # `IN_FLIGHT`, so a peer that stops reading stops being served rather
        # than being quietly lied to.
        with suppress(DaemonGone, anyio.ClosedResourceError, anyio.BrokenResourceError):
            await self.send(reply)

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
