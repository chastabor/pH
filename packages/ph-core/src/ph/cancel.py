"""Cooperative cancellation, fused across nested owners.

dsh threads an `AbortSignal` through the tool pipeline: the caller owns one, a
`tools/execute` wrapper may *replace* it for its delegated lifetime (that is how
a timeout policy works), and the registry fuses every replacement with the
captured caller signal so a wrapper can narrow the lifetime but never widen it.

`CancelToken` is that contract. A child is canceled when it is canceled *or
when any ancestor is*, which makes the fusion structural rather than something
each wrapper has to remember.

Why not a bare `anyio.CancelScope`: a scope cancels the task that awaits inside
it, and the pipeline needs to *ask* whether cancellation happened at points
where no await is pending — to decide between "aborted before dispatch" (the
call had no effect) and "aborted" (the body ran). A token answers that question;
a scope only acts on it.

@module ph.cancel
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

import anyio

__all__ = [
    "POLL_SECONDS",
    "CancelToken",
    "Canceled",
    "Cancellation",
    "is_canceled",
    "raced",
    "until_canceled",
]


class Canceled(Exception):
    """Raised by `raise_if_canceled()`; carries the reason for the record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Cancellation(Protocol):
    """What a *reader* needs of a cancellation: whether, and why.

    `CancelToken` satisfies it. Declared because most holders only ask — and
    `AgentDriver`'s docstring makes the argument about the verb it withholds:
    "a seam that could reach `cancel` through a parameter typed for reading is a
    seam that will, eventually." `cancel()` and `child()` belong to whoever owns
    the lifetime; `AgentHandle.signal` hands out this instead, so a row that
    prompts can ask whether the work is still wanted and cannot end it.
    """

    @property
    def canceled(self) -> bool: ...
    @property
    def cancel_reason(self) -> str | None: ...
    def raise_if_canceled(self) -> None: ...
    async def wait(self) -> None:
        """Return once this is canceled. Still a reader: waiting ends nothing."""
        ...


@dataclass(slots=True)
class CancelToken:
    """A cooperative cancellation flag that inherits from its parent."""

    reason: str | None = None
    parent: CancelToken | None = None
    _woken: anyio.Event = field(default_factory=anyio.Event, compare=False, repr=False)
    """Set by `cancel`, so a waiter is told rather than having to ask.

    **One event per node, because `cancel` touches one node.** Cancellation is
    inherited *upward* — `canceled` walks `parent` — so a parent that cancels
    does not touch its children's state at all, and an event on the cancelled
    token alone would leave every waiter on a child asleep. `wait` closes that by
    watching the whole chain rather than by keeping a downward registry of
    children, which a long-lived root would grow one entry per tool call and
    never shrink.

    Private, and not on `Cancellation`: what a reader is offered is `wait()`.
    Handing out the event would hand out `set()`, which is `cancel` by another
    name — the distinction `Cancellation` exists to keep.
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
        """Return once this token is canceled — by itself or by anything above it.

        **The chain, not just this node.** `canceled` is inherited upward and
        `cancel` sets one node, so waiting on this token's own event alone would
        sleep through a parent being canceled — which is the common case, since a
        tool's token is a child of the turn's.

        Cheap for the shape that is almost always in hand: a root has no parent
        and waits on one event with no task group at all.
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
    """`False` for no token — the one place that rule is spelled out."""
    return token is not None and token.canceled


POLL_SECONDS = 0.05
"""The stop ladder's cadence — how often a *deadline* is re-checked.

**Not how a cancellation is noticed any more.** `Cancellation.wait` is an event,
so a waiter is told; what is left polling is the kernel's `_watch`, which asks a
second question on the same tick — has the abort grace expired — and a deadline
needs a clock whatever the flag does. Well under the time a person takes to
notice that nothing has happened, which is the bound that matters for a ladder.
"""


async def until_canceled(signal: Cancellation | None, scope: anyio.CancelScope) -> None:
    """Cancel `scope` once `signal` trips; park forever when there is none.

    `None` parks rather than returning, so `raced` can start this
    unconditionally and let the work be what finishes the scope.
    """
    if signal is None:
        await anyio.sleep_forever()
        return  # `sleep_forever` never returns; this is for the checker
    await signal.wait()
    scope.cancel()


async def raced[T](signal: Cancellation | None, work: Callable[[], Awaitable[T]]) -> T | None:
    """Run `work`, giving it up if `signal` trips first. `None` means it was.

    **The whole shape, not just the sleep.** Two seams needed this within a week
    — a running child and a delegated subagent — and extracting only the polling
    left both of them writing the same task group, the same `start_soon`, the
    same "the work won, cancel the watcher" line, and in one case a `nonlocal`
    and a nested `async def` whose only job was to carry a result out of a task
    group. A third caller would have written it a third way.

    `None` for "canceled" rather than an exception, because what each caller does
    next differs: one reports an exit code, the other releases a child and then
    raises. A shared exception would have to be caught and translated at both.
    """
    outcome: T | None = None

    async with anyio.create_task_group() as tasks:

        async def run() -> None:
            nonlocal outcome
            outcome = await work()
            # The watcher is all that is left; it would otherwise poll until the
            # token's owner went away.
            tasks.cancel_scope.cancel()

        tasks.start_soon(until_canceled, signal, tasks.cancel_scope)
        tasks.start_soon(run)

    return outcome
