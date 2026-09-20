"""`ctx`-free cancellation: the token, and what it now lets a caller wait on.

`CancelToken` began as a flag — `canceled`, `cancel_reason`,
`raise_if_canceled`, all synchronous — and everything that needed to *wait* for
one polled it at `POLL_SECONDS`. That was never a property of the system; it was
a property of a type with no notification channel, and this is the module where
that stopped being true.

The half worth testing is the direction. Cancellation is inherited **upward**:
`canceled` walks `parent`, and `cancel()` sets one node. So an event on the
cancelled token alone would leave every waiter on a *child* asleep — which is
the common case, because a tool's token is a child of the turn's. `wait()`
watches the chain, and that is what these pin.
"""

from __future__ import annotations

import anyio
import pytest

from ph.cancel import POLL_SECONDS, Canceled, CancelToken, is_canceled, raced, until_canceled

pytestmark = pytest.mark.anyio


async def test_a_waiter_is_told_rather_than_asking() -> None:
    """The point of the change: no cadence between the cancel and the waiter.

    The bound is deliberately far below `POLL_SECONDS`. A poll could not pass
    it, and asserting "faster than the old mechanism" is the only way to say
    "this is an event" in a test that does not reach into the implementation.
    """
    token = CancelToken()

    async with anyio.create_task_group() as tasks:
        woken = anyio.Event()

        async def waiting() -> None:
            await token.wait()
            woken.set()

        tasks.start_soon(waiting)
        await anyio.sleep(0)
        token.cancel("user")
        with anyio.fail_after(POLL_SECONDS / 5):
            await woken.wait()


async def test_a_child_is_woken_by_its_parent() -> None:
    """The direction that makes this more than one `Event` (the chain).

    `cancel()` touches one node and `canceled` reads upward, so a parent that
    cancels never writes to its children. A waiter on the child's own event
    would sleep through the very cancellation that applies to it — and a tool's
    token is a child of the turn's, so that is the ordinary case rather than a
    corner.
    """
    root = CancelToken()
    turn = root.child()
    tool = turn.child()

    async with anyio.create_task_group() as tasks:
        woken = anyio.Event()

        async def waiting() -> None:
            await tool.wait()
            woken.set()

        tasks.start_soon(waiting)
        await anyio.sleep(0)
        root.cancel("user")
        with anyio.fail_after(1):
            await woken.wait()

    assert tool.canceled and tool.cancel_reason == "user"
    assert root.canceled, "and the ancestor still reads as canceled itself"


async def test_a_token_already_canceled_does_not_wait() -> None:
    """Including one canceled through an ancestor, and one born canceled.

    `child(reason=…)` sets `reason` in the constructor, so `cancel()` never runs
    and the event is never set — the `canceled` check at the top of `wait` is
    what keeps that from being a hang.
    """
    root = CancelToken()
    root.cancel("user")

    with anyio.fail_after(1):
        await root.wait()
        await root.child().wait()
        await CancelToken(reason="born canceled").wait()


async def test_waiting_ends_nothing() -> None:
    """`Cancellation` is a reader, and `wait` had to stay one.

    The protocol withholds `cancel()` on purpose — "a seam that could reach
    `cancel` through a parameter typed for reading is a seam that will,
    eventually" — so the event is private and what a reader is offered is a
    method that returns when somebody *else* decides.
    """
    token = CancelToken()

    with anyio.move_on_after(0.01):
        await token.wait()

    assert not token.canceled, "waiting is not asking for it"
    assert not is_canceled(token)


async def test_raced_gives_up_the_work_when_the_token_wins() -> None:
    """The shape both new callers use, end to end."""
    token = CancelToken()

    async def parked() -> str:
        await anyio.sleep(30)
        return "finished"

    async with anyio.create_task_group() as tasks:

        async def cancel_shortly() -> None:
            await anyio.sleep(0.01)
            token.cancel("user")

        tasks.start_soon(cancel_shortly)
        with anyio.fail_after(1):
            assert await raced(token, parked) is None, "the work outlived the cancel"

    # And the other way: the work wins, and its value comes back.
    assert await raced(CancelToken(), lambda: _answered("done")) == "done"


async def _answered(value: str) -> str:
    return value


async def test_until_canceled_ends_the_scope_it_was_given() -> None:
    """`raced`'s half, on its own — a scope nothing else would end."""
    token = CancelToken()

    with anyio.fail_after(1):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(until_canceled, token, tasks.cancel_scope)
            await anyio.sleep(0.01)
            token.cancel("user")


async def test_a_token_with_no_signal_parks_rather_than_ending_the_scope() -> None:
    """`None` is "nothing to watch", not "already canceled".

    `raced` starts the watcher unconditionally, so a `None` that returned would
    cancel the scope immediately and every caller would read its work as
    canceled before it began.
    """
    with anyio.move_on_after(0.02) as scope:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(until_canceled, None, tasks.cancel_scope)

    assert scope.cancelled_caught, "the watcher returned instead of parking"


def test_the_reason_travels_with_the_refusal() -> None:
    """Unchanged by this work, and asserted because `wait` sits beside it."""
    token = CancelToken()
    token.cancel("user")

    with pytest.raises(Canceled) as caught:
        token.child().raise_if_canceled()

    assert caught.value.reason == "user"
