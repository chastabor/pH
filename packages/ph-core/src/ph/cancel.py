"""Cooperative cancellation, fused across nested owners.

dsh passes an `AbortSignal` through the tool pipeline. The caller owns one
signal. A `tools/execute` wrapper may *replace* it for the lifetime that is
delegated to the wrapper, which is how a timeout policy works. The registry
fuses every replacement with the captured signal of the caller, so a wrapper can
narrow the lifetime but can never widen it.

`CancelToken` is that contract. A child is canceled when it is canceled *or
when any ancestor is*. The fusion is therefore part of the structure, and no
wrapper has to remember it.

A bare `anyio.CancelScope` is not enough. A scope cancels the task that awaits
inside it. The pipeline also needs to *ask* whether cancellation happened, at
points where no await is pending. The answer decides between "aborted before
dispatch" (the call had no effect) and "aborted" (the body ran). A token answers
that question. A scope only acts on it.

@module ph.cancel
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import anyio

__all__ = [
    "CancelToken",
    "Canceled",
    "Cancellation",
    "Waitable",
    "first_of",
    "is_canceled",
    "raced",
    "reason_of",
    "until_canceled",
]


class Canceled(Exception):
    """Raised by `raise_if_canceled()`. It carries the reason for the record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Cancellation(Protocol):
    """What a *reader* needs of a cancellation: whether it happened, and why.

    `CancelToken` satisfies this protocol. The protocol is declared because most
    holders only ask. The docstring of `AgentDriver` gives the argument about
    the method that this protocol withholds: a seam that can reach `cancel`
    through a parameter that is typed for reading will call it at some time.
    `cancel()` and `child()` belong to the owner of the lifetime.
    `AgentHandle.signal` hands out this protocol instead, so a row that prompts
    can ask whether the work is still wanted, and cannot end it.
    """

    @property
    def canceled(self) -> bool: ...
    @property
    def cancel_reason(self) -> str | None: ...
    def raise_if_canceled(self) -> None: ...
    async def wait(self) -> None:
        """Return when this is canceled. This is still a read: a wait ends nothing."""
        ...


@dataclass(slots=True)
class CancelToken:
    """A cooperative cancellation flag that inherits from its parent."""

    reason: str | None = None
    parent: CancelToken | None = None
    _woken: anyio.Event = field(default_factory=anyio.Event, compare=False, repr=False)
    """`cancel` sets this event, so a waiter is told and does not have to ask.

    **One event for each node, because `cancel` touches one node.** Cancellation
    is inherited *upward*: `canceled` walks `parent`. A parent that cancels
    therefore does not touch the state of its children at all. An event on the
    canceled token alone would leave every waiter on a child asleep. `wait`
    solves this by watching the whole chain. It does not keep a downward
    registry of children, because a long-lived root would add one entry to such
    a registry for each tool call and never remove one.

    The event is private, and it is not on `Cancellation`. A reader is offered
    `wait()`. To hand out the event would hand out `set()`, which is `cancel`
    under another name. `Cancellation` exists to keep that distinction.
    """

    @property
    def canceled(self) -> bool:
        node: CancelToken | None = self
        while node is not None:
            if node.reason is not None:
                return True
            node = node.parent
        return False

    @property
    def cancel_reason(self) -> str | None:
        node: CancelToken | None = self
        while node is not None:
            if node.reason is not None:
                return node.reason
            node = node.parent
        return None

    def cancel(self, reason: str = "canceled") -> None:
        if self.reason is None:
            self.reason = reason
            self._woken.set()

    def child(self, reason: str | None = None) -> CancelToken:
        """A narrower token: canceled by itself or by anything above it."""
        return CancelToken(reason=reason, parent=self)

    def raise_if_canceled(self) -> None:
        reason = self.cancel_reason
        if reason is not None:
            raise Canceled(reason)

    async def wait(self) -> None:
        """Return when this token is canceled, by itself or by anything above it.

        **The wait covers the chain, not only this node.** `canceled` is
        inherited upward and `cancel` sets one node. A wait on the event of this
        token alone would therefore sleep through the cancel of a parent. That
        is the common case, because the token of a tool is a child of the token
        of the turn.

        The wait is cheap for the shape that is almost always in use: a root has
        no parent, so it waits on one event with no task group at all.
        """
        if self.canceled:
            return
        if self.parent is None:
            await self._woken.wait()
            return
        async with anyio.create_task_group() as tasks:

            async def woken(node: CancelToken) -> None:
                await node._woken.wait()
                tasks.cancel_scope.cancel()

            ancestor: CancelToken | None = self
            while ancestor is not None:
                tasks.start_soon(woken, ancestor)
                ancestor = ancestor.parent


def is_canceled(token: Cancellation | None) -> bool:
    """`False` for no token. This is the one place that states that rule."""
    return token is not None and token.canceled


def reason_of(stop: BaseException) -> str:
    """What a stop, or any failure, reports about itself. For a token or a scope
    it is the reason. For anything else it is the type and the message. For a
    group it is what each member reports.

    A canceled anyio scope passes its reason inside a message that also names
    the id of the scope and the task that canceled it. Only the reason is kept,
    because the rest differs on every run and tells a reader nothing. That
    message belongs to the asyncio backend and is not public API, so if it
    changes, the result is the bare type name. The stop of a task group reports
    what the stops that it holds report. The function needs a running event
    loop, as `anyio.get_cancelled_exc_class` does.
    """
    if isinstance(stop, Canceled):
        return stop.reason
    if isinstance(stop, BaseExceptionGroup):
        return "; ".join(reason_of(inner) for inner in stop.exceptions)
    name, text = type(stop).__name__, str(stop)
    if isinstance(stop, anyio.get_cancelled_exc_class()):
        return text.partition("; reason: ")[2] or name
    return f"{name}: {text}" if text else name


async def until_canceled(signal: Cancellation | None, scope: anyio.CancelScope) -> None:
    """Cancel `scope` when `signal` trips, with the reason of the signal. Wait
    forever when there is no signal.

    `None` waits and does not return, so `raced` can start this function
    unconditionally and let the work be what finishes the scope. The reason
    goes with the cancel, so a turn that the scope cuts records why
    (`reason_of`).
    """
    if signal is None:
        await anyio.sleep_forever()
        return  # `sleep_forever` never returns; this is for the checker
    await signal.wait()
    scope.cancel(signal.cancel_reason)


class Waitable(Protocol):
    """Anything that can be waited on to happen: an `anyio.Event`, a
    `Cancellation`, a `ph.wall_clock.Alarm`."""

    async def wait(self) -> None: ...


async def first_of(*events: Waitable) -> None:
    """Wait for whichever of these happens first.

    Some waits have several possible endings. A follow ends when its root goes
    idle or when its daemon goes away. A scheduler wakes when its plan moves or
    when its daemon stops. A wait on only one ending hangs whenever another
    ending happens. A deadline is one more ending: pass a `ph.wall_clock.Alarm`
    for an epoch instant, or bound the call with `anyio.move_on_after` for a
    short relative one.
    """
    async with anyio.create_task_group() as tasks:

        async def stop_on(event: Waitable) -> None:
            await event.wait()
            tasks.cancel_scope.cancel()

        for event in events:
            tasks.start_soon(stop_on, event)


async def raced[T](signal: Cancellation | None, work: Callable[[], Awaitable[T]]) -> T | None:
    """Run `work`, and give it up if `signal` trips first. `None` means that the
    signal tripped first.

    **This is the whole shape, not only the sleep.** Two seams needed this
    within a week: a running child and a delegated subagent. When only the
    polling was extracted, both seams still wrote the same task group, the same
    `start_soon`, and the same "the work won, cancel the watcher" line. One of
    them also had a `nonlocal` and a nested `async def` whose only job was to
    carry a result out of a task group. A third caller would have written it a
    third way.

    "Canceled" is `None` and not an exception, because each caller does a
    different thing next. Of those two seams, one reports an exit code, and the
    other releases a child and then raises. Both would have to catch and
    translate a shared exception.
    """
    outcome: T | None = None

    async with anyio.create_task_group() as tasks:

        async def run() -> None:
            nonlocal outcome
            outcome = await work()
            # Only the watcher is left. Without this cancel it would wait until
            # the owner of the token went away.
            tasks.cancel_scope.cancel()

        tasks.start_soon(until_canceled, signal, tasks.cancel_scope)
        tasks.start_soon(run)

    return outcome
