"""One end of a two-way JSON-RPC connection. Both ends use this class.

`protocol.py` holds the *stateless* half of the vocabulary: the shape of a
request, a notification and a reply. It deliberately imports nothing from the
daemon. This module is the stateful half. It mints ids, remembers what is
outstanding, decides which direction an inbound frame goes, serializes what goes
out, and wakes every waiter when the socket ends.

**This module exists because that half was written twice.** Until P5-13 the
protocol went only one way for anything that expected an answer, so the client
had a pending table and the server did not. When the daemon became able to ask a
person, the server got a pending table too. The two copies immediately disagreed
about the case that matters. When a connection died in the middle of a request,
one side raised `DaemonGone` and the other side returned an empty `{}`.
Downstream, the empty `{}` reads as a successful answer with no fields, and it
denied the call. One object gives one answer.

This is not a seam. Nothing swaps it, and it is built before any `Context`
exists. Both ends run *the same code* and do not talk through it: they talk
through the socket.

## The three buffers, which are not the same buffer

A duplex peer queues in three places. The three have different jobs and
different sizes. They also have different overflow policies, which is the reason
why a shared object must state them explicitly:

* **Outbound frames** go through one memory stream that one writer task drains.
  A notification that arrives while a reply is half-written therefore cannot
  interleave two frames on the wire. There are two ways in, because the two ends
  need opposite things when the stream is full. `tell()` refuses (`WouldBlock`),
  so that the daemon can *drop* a watcher that cannot keep up: the work must
  never wait on a subscriber. `send()` blocks, which is right for a client,
  because a client has nobody to drop except itself.
* **Inbound requests** are bounded by a semaphore that the read loop acquires
  and the handler releases. A handler may wait for a human, so the loop cannot
  await a handler. Without a ceiling, "do not await" silently becomes "accept
  without limit", and a peer that pipelines requests parks tens of megabytes of
  handler tasks. The limiter returns the backpressure to the socket, which is
  where it was before dispatching moved off the loop.
* **Outstanding asks** are a correlation table, not a queue: each id maps to the
  event that its caller waits on. The table shrinks as answers arrive and is
  emptied on close. That stops a caller from waiting for a frame that will never
  come.

@module ph_app.daemon.duplex
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import anyio
from anyio.abc import ByteStream, TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from ph.framing import FramingError, read_frames, write_frame
from ph.json import as_str
from ph.wire import WireModel

from ..payloads import SessionAsk
from ..protocol import (
    DaemonGone,
    Dispatch,
    Frame,
    MethodResult,
    notification,
    parse_params,
    request,
    respond,
    result_of,
)

__all__ = ["IN_FLIGHT", "OUTBOX", "Handler", "Notification", "Peer", "answering"]

log = logging.getLogger("ph_app.daemon.duplex")

Notification = Callable[[str, dict[str, Any]], None]
"""What this end does with a frame with no id that the peer sent.

The type is `dict[str, Any]` because a frame that the peer sent is a claim. The
module docstring of `ph_app.protocol` states that rule, and this is the rule
applied to the observer. The models exist and the readers use them
(`payloads.notice_of`). The transport does not pretend that a dict is one of
them.

The method travels next to the payload and not inside it, because JSON-RPC
already carries the method in the envelope, and this signature is the signature
of the envelope."""

Handler = Callable[[dict[str, Any]], Awaitable[MethodResult]]
"""What this end answers when the *peer* asks it something (P5-13).

It takes a `dict` for the same reason as `Notification`. `answering` below is the
typed entry point. A handler body therefore takes a model and returns a model,
and this alias remains the concern of the transport."""


def answering[A: SessionAsk](ask: type[A], answer: Callable[[A], Awaitable[WireModel]]) -> Handler:
    """One typed ask handler, as the `Handler` that the transport registers.

    This is the client counterpart of `parse_params` on the daemon, and literally
    so: it *is* `parse_params`. A bad ask is therefore refused with the same
    named `invalid_params` and the same one-line sentence as a bad request, in
    whichever direction it traveled. The handwritten version was
    `ApprovalAsk.model_validate(params)` at the top of each handler. Its bare
    `ValidationError` carries no `code`, so `respond` sent it back as an unnamed
    `-32000` with a multi-line pydantic dump as the message. That is the error
    shape that P8-07 spent a row to delete on the daemon side.

    Nothing dumps the reply here. `respond` turns a `WireModel` into a frame at
    the one point where a result becomes a frame. Its own comment says that none
    of its handlers writes `.to_wire()` at the `return`, and this is a handler.

    The bound is `A: SessionAsk` and not `WireModel`, for the `METHOD` that the
    refusal names. Also, a *notice* has no reply to give, and the bound stops a
    notice from being registered here.
    """

    async def handler(params: dict[str, Any]) -> WireModel:
        return await answer(parse_params(ask.METHOD, ask, params))

    return handler


OUTBOX = 1024
"""Frames that may be queued for the wire before `tell` refuses.

The number is large enough to absorb a burst of `session.event` during a fast
turn. It is finite so that the sender can find out that "this peer is not
reading"."""

IN_FLIGHT = 64
"""Requests from one peer that may be handled at the same time.

The number is generous for any real client: a TUI has one call outstanding, and
`phern agents attach` pages snapshots serially. It is also finite, which is the
property that matters."""


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
    so a handler that raises becomes an error frame, and the asker is settled in
    both cases."""
    on_notify: Notification | None = None
    dispatch_notifications: bool = False
    """Whether a frame with no id is a *method to run* on this end.

    **This is the one thing on which the two ends really disagree, so it is
    declared and not inferred.** To a client, a notification is an event to
    watch (`session.event`, `session.status`) with no body to run. To the
    daemon, it is a method whose answer nobody wants. `shutdown` carries no id by
    contract, because a reply would make the caller wait for a frame that the
    daemon is losing the ability to write.

    An inference from `on_notify is not None` read the same way, and it failed
    quietly in one direction. A client that is built without an observer
    (`phern agents attach` follows a log and needs no callback) then
    *dispatched* every `session.event` into a handler table that has no handler
    for it. Each event cost a task and one of `IN_FLIGHT` to raise a
    `LookupError`, which `respond` swallows because there is no id to answer.
    With the declaration, that client drops the events at no cost.
    """
    id_prefix: str = ""
    """The prefix of the ids that this end mints, so that a frame log shows which
    side asked.

    Both ends mint ids now, and the ids must not collide: the client uses `c`,
    and the daemon uses `s`."""
    closed: anyio.Event = field(default_factory=anyio.Event)
    """Set when the loop stops, whichever end stopped it.

    Callers must be able to *wait for* "the peer went away", and not only notice
    it. `shutdown` is a notification by contract, so the only reliable
    confirmation is that the connection closes."""
    _asked: int = 0
    _unwatchable: int = 0
    """How many notifications this end has failed to read. The first one is logged
    with its traceback, and the rest are logged at debug. See `_watch`."""
    _pending: dict[str, _Pending] = field(default_factory=dict)
    _streams: tuple[MemoryObjectSendStream[Frame], MemoryObjectReceiveStream[Frame]] | None = None
    """Both halves of the outbound queue, or neither. `_queue` builds the pair and
    `serve` drains it, so nothing here can act on a state with only one half."""

    # ------------------------------------------------------------- outbound --

    def _queue(self) -> tuple[MemoryObjectSendStream[Frame], MemoryObjectReceiveStream[Frame]]:
        """The outbound stream, built on first use.

        **It is not built in `serve()`.** That is the obvious place, but it is a
        race. A caller may `call()` just after `start_soon(peer.serve)` and
        before that task has run, and the frame must have somewhere to go. The
        queue is part of what this end *is*. `serve` only drains it.
        """
        if self.closed.is_set():
            raise DaemonGone
        if self._streams is None:
            self._streams = anyio.create_memory_object_stream[Frame](max_buffer_size=OUTBOX)
        return self._streams

    def tell(self, method: str, params: dict[str, Any]) -> None:
        """Queue a notification, or **raise** so that the caller drops this peer.

        The raise is the purpose, and the watcher policy of the daemon depends on
        it. Code that caught `WouldBlock` here and logged "dropped" would drop
        nothing, and the peer that cannot keep up would cost the whole fan-out
        again for every later event. The subscriber list belongs to its owner,
        so this method only has to fail loudly enough to be noticed.
        """
        # The relay calls this one time for each event and each watcher. So the
        # built pair is read directly from the field, and `_queue()` runs only
        # the first time.
        outbox, _ = self._streams or self._queue()
        outbox.send_nowait(notification(method, params))

    async def send(self, frame: Frame) -> None:
        """Queue any frame, and wait for room. For an end that has nobody to drop.

        The argument is a `Frame`, one of the four shapes that this side builds,
        and not a dict. This side must get right what goes *out*, and the type
        checks it. What comes *in* stays a `dict[str, Any]` (`_settle`,
        `result_of`), because a frame from a peer is a claim."""
        await self._queue()[0].send(frame)

    async def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the other end and wait for its answer.

        **There is no timeout.** On the daemon side the answer comes from a
        person, and a deadline would turn "they went to lunch" into a denial.
        The connection is the bound: if the peer goes away, this raises, and the
        caller decides what that means. For an approval, it means that the
        question stays open for whoever attaches next.
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
            # The end of the connection woke this wait, and no answer arrived.
            # The error is named, because an empty `{}` reads as a successful
            # reply with no fields in it. It is named as a *disconnection*,
            # because nobody refused.
            raise DaemonGone
        return result_of(entry.reply)

    # -------------------------------------------------------------- the loop --

    async def serve(self) -> None:
        """Read until the socket closes, answering and settling as frames arrive."""
        send, receive = self._queue()
        limit = anyio.Semaphore(IN_FLIGHT)
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(self._write, receive)
                # The suppression is *inside* the group, and it must be. anyio
                # wraps even a single exception that leaves a task group into an
                # `ExceptionGroup`, which no `except` that names the original
                # exception can see.
                with suppress(
                    anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream
                ):
                    await self._read(tasks, limit)
                # A handler that still waits for a person is the one thing that is
                # expected to be in flight. The socket is gone, so its answer has
                # nowhere to go, and the question belongs to its owner.
                tasks.cancel_scope.cancel()
        finally:
            # This is in `finally`, so a cancellation or a crash wakes a waiter in
            # the same way as an orderly end. A wait that completes only on the
            # success path is a hang that looks like a timeout.
            self._streams = None
            self.closed.set()
            for entry in self._pending.values():
                entry.answered.set()
            self._pending.clear()
            with suppress(anyio.ClosedResourceError):
                await send.aclose()

    async def _write(self, receive: MemoryObjectReceiveStream[Frame]) -> None:
        async with receive:
            async for frame in receive:
                try:
                    await write_frame(self.stream, frame)
                except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                    return

    async def _read(self, tasks: TaskGroup, limit: anyio.Semaphore) -> None:
        """Route each frame by direction, and never handle a frame inline.

        The discriminator is `method`, not `id`. See `protocol.request`. Both
        ends were written with the opposite assumption, and each got the same
        bug from its own side. The daemon dispatched the *answer* of a client as
        method `""` and returned it as `unknown_method`. The client filed the
        *request* of the daemon as the answer to a call that nobody made.
        """
        try:
            async for frame in read_frames(self.stream):
                if "method" not in frame:
                    self._settle(frame)
                    continue
                if frame.get("id") is None and not self.dispatch_notifications:
                    # This is an event to watch, with nothing to run. See
                    # `dispatch_notifications`. An end with no observer drops it
                    # here, and does not spend a task to learn that it has no body.
                    if self.on_notify is not None:
                        self._watch(frame)
                    continue
                # `respond` returns `None` for a frame with no id, so a dispatched
                # notification runs its body and writes nothing back.
                await limit.acquire()
                tasks.start_soon(self._handle, frame, limit)
        except FramingError as error:
            # Unreadable framing ends the connection. After a bad frame, there is
            # no way to know where the next frame starts.
            log.info("ph_app.daemon: closing a connection — %s", error)

    def _watch(self, frame: dict[str, Any]) -> None:
        """Pass one notification to the observer, and survive what the observer
        does.

        **A notification body must not be able to end the connection.** This
        runs *inline on the read loop*. That is deliberate: an observer is
        cheap, and a task for each event is not. So anything that the observer
        raises would unwind `_read`, leave the task group, and close the socket.
        The reader is the code of a client, which parses a frame that it may not
        recognize. A notice with a field that this build does not know raises
        out of `model_validate`. Before this guard, that killed
        `phern agents attach` over a `session.staged` that it does not even
        read.

        `_settle` makes the same judgment about a late reply: to end the
        connection over it would punish a peer for a race that it did not cause.
        Here that judgment is applied to the other path for frames with no id.
        A bad frame costs one frame, and is reported in full one time: the first
        is logged with `exc_info`, and the later ones quietly. A daemon that
        sends one unreadable notice will send the next one too, and the terminal
        of a follower is not the place to print the same traceback for each
        event.
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
        """Serve one request, and **hold the slot until its reply is out** (E4).

        The release used to be a `finally` around the dispatch alone. That made
        the last sentence of the comment below false. The slot came back at the
        moment the handler returned, so a peer that pipelined requests and
        stopped reading got every one of them dispatched. The bound counted work
        *started* and not work outstanding. Both the tasks and the frames that
        they held piled up without bound against a reader that would never drain
        them.
        """
        try:
            reply = await respond(frame, self.dispatch)
            if reply is None:
                return
            # **The reply goes through `send`, so it waits for room and is not
            # dropped.** A full outbox means a slow *reader*. A discarded reply
            # leaves the asker on the other end waiting until the connection
            # closes. A reply is the frame for which a drop is least defensible,
            # and the only frame that nobody chose to drop. To block here is the
            # backpressure at work: this task holds one of `IN_FLIGHT`, so a peer
            # that stops reading stops being served, and is not quietly misled.
            with suppress(DaemonGone, anyio.ClosedResourceError, anyio.BrokenResourceError):
                await self.send(reply)
        finally:
            limit.release()

    def _settle(self, frame: dict[str, Any]) -> None:
        """An answer to something that this end asked.

        An unknown id is dropped and does not raise. An ask that was already
        abandoned, because the asker was canceled or the connection is closing,
        gets an answer from a peer that could not have known. To end the
        connection over a late reply would punish the peer for a race that it
        did not cause.
        """
        entry = self._pending.pop(str(frame.get("id")), None)
        if entry is None:
            return
        entry.reply = frame
        entry.answered.set()
