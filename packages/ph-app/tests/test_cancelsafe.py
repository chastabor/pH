"""The cancelled-readiness-wait guard, and what keeps it honest (issue 58).

Driven **deterministically** rather than by racing. The defect is a window
between the event loop queueing a ready handle and running it, so a test that
tried to *hit* the window would be the flake it exists to remove — the original
was roughly 2 occurrences in 30 full-suite runs.

**The first two drafts of this file both passed with the guard removed**, which
is the reason the drive below is spelled out rather than staged through a real
listener. The first watched the ordering through `add_done_callback`, which is
itself `call_soon`'d and so reads the selector an iteration *after* resolution,
by which time every ordering looks identical. The second cancelled a listener's
`accept()` after connecting to it — but `await connect_unix(...)` yields, so the
queued accept callback had already resolved and the wait being cancelled was a
fresh one on an idle fd. It never entered the window at all: 0 failures in 25
runs with the guard on, with it off, and under the sabotage its own docstring
named.

What enters the window is `loop.call_soon(future.cancel)`. `_run_once` takes
`ntodo = len(self._ready)` *after* `_process_events` has appended the ready I/O
handles, so a callback queued before the poll and the readiness handle run in
the **same** iteration, in that order: the cancel lands after the loop has
committed to firing the wait, and the `remove_reader` that anyio's done-callback
would do is queued for the iteration after. That is the defect exactly, and it
reproduces 20 times in 20.

`ph_app.daemon.cancelsafe` states what the guard does and what it refuses.
"""

from __future__ import annotations

import asyncio
import socket
from asyncio.selector_events import BaseSelectorEventLoop
from pathlib import Path

import anyio
import pytest

from ph_app.daemon.cancelsafe import (
    apply_cancel_safe_socket_waits,
    guarded,
    resolves_a_future,
)

pytestmark = pytest.mark.anyio


def watch_list(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> bool:
    """Whether the loop is still watching this socket's fd."""
    return sock.fileno() in loop._selector.get_map()  # type: ignore[attr-defined]


def test_the_guard_is_in_force_from_importing_the_daemon_package() -> None:
    """Importing anything under `ph_app.daemon` is what applies it.

    Not a call a test makes for itself: the guard has to be in force before the
    first socket exists, and the daemon binds its listener while three CLI paths
    connect — so an explicit call would be four call sites that must agree.

    Sabotage: delete the `apply_cancel_safe_socket_waits()` line from
    `ph_app/daemon/__init__.py` and this fails.
    """
    import ph_app.daemon  # noqa: F401  - the import *is* the subject

    assert guarded(BaseSelectorEventLoop.add_reader), "add_reader is unguarded"
    assert guarded(BaseSelectorEventLoop.add_writer), "add_writer is unguarded"
    assert apply_cancel_safe_socket_waits() is False, "a second apply re-patched"


async def test_only_a_bound_future_set_result_is_recognised() -> None:
    """The predicate that keeps the blast radius benign.

    Any callback that is *not* the failing shape must pass through untouched —
    anyio's own hardened `cb` closure included, which already swallows
    `InvalidStateError` for itself and would gain nothing but a frame.

    Recognised by shape rather than identity on purpose: `set_result` on the C
    accelerator is a `builtin_method` with no `__func__`, so an identity test
    stops recognising every real anyio wait, and a `Future` subclass that
    overrides `set_result` would slip past as well.
    """
    future = asyncio.get_running_loop().create_future()
    assert resolves_a_future(future.set_result)
    assert not resolves_a_future(future.cancel), "only `set_result` is the failing shape"
    assert not resolves_a_future(lambda: None), "a plain closure is not the failing shape"
    assert not resolves_a_future(print), "a builtin is not the failing shape"
    future.cancel()


def test_anyio_still_registers_the_shape_the_guard_exists_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shelf-life guard, and the reason this file exists as well as the fix.

    A monkeypatch survives an upgrade *silently*. If anyio adopts its own
    hardened pattern at these sites — which is the fix we want upstream — the
    guard will still report itself as applied while having nothing left to do,
    and `cancelsafe` should then be deleted rather than left to look effective.

    So this asks the question `guarded` cannot: does a real
    `UNIXSocketListener.accept` still hand the loop a bound `Future.set_result`?
    Asserted by watching the registration rather than by reading anyio's source,
    so it stays true however the source is spelled.
    """
    seen: list[bool] = []
    original = BaseSelectorEventLoop.add_reader

    def watching(self: object, fd: object, callback: object, *args: object) -> object:
        seen.append(resolves_a_future(callback))
        return original(self, fd, callback, *args)  # type: ignore[arg-type]

    async def accept_once() -> None:
        listener = await anyio.create_unix_listener(tmp_path / "s.sock")
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(listener.serve, lambda stream: anyio.sleep(30))
                await anyio.sleep(0.05)
                tasks.cancel_scope.cancel()
        finally:
            # `SocketListener.serve` does not close on cancellation, and a
            # listening fd held to GC is the leak this suite forbids.
            await listener.aclose()

    monkeypatch.setattr(BaseSelectorEventLoop, "add_reader", watching)
    anyio.run(accept_once)

    assert any(seen), (
        "no readiness wait registered a bound `Future.set_result` — anyio may have "
        "adopted its own hardened pattern, in which case `cancelsafe` has nothing "
        "left to do and should be deleted (issue 58)"
    )


async def test_a_wait_cancelled_after_the_loop_committed_to_firing_it_is_swallowed() -> None:
    """The defect itself, driven rather than raced.

    `fired` is the positive control, and it is the half a green result needs:
    `raised == []` also passes when the guard is never reached, which is how
    both earlier drafts of this file came to pass at HEAD. Recording
    `self.cancelled()` from inside `set_result` proves the loop really did fire
    a wait that had already been cancelled — the window was entered — and
    `raised` then says the guard absorbed it.

    Sabotage: delete the `except asyncio.InvalidStateError` from
    `cancelsafe._guarding` and `raised` holds
    `Exception in callback ...set_result(None): InvalidStateError`, which is the
    production traceback verbatim.
    """
    raised: list[str] = []
    fired: list[bool] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: raised.append(
            f"{context.get('message')}: {context.get('exception')!r}"
        )
    )

    class Counting(asyncio.Future[None]):
        """Records whether it was already cancelled when the loop fired on it."""

        def set_result(self, value: None, /) -> None:
            fired.append(self.cancelled())
            super().set_result(value)

    left, right = socket.socketpair()
    left.setblocking(False)
    try:
        future: asyncio.Future[None] = Counting(loop=loop)
        # Registered exactly as anyio's five unguarded sites do it, done-callback
        # included, so the guard is what is under test and nothing else.
        loop.add_reader(left, future.set_result, None)
        future.add_done_callback(lambda _: loop.remove_reader(left))

        right.send(b"x")
        loop.call_soon(future.cancel)
        with pytest.raises(asyncio.CancelledError):
            await future
        await asyncio.sleep(0.05)

        assert fired == [True], f"the loop never fired a cancelled wait: {fired}"
        assert raised == [], f"a cancelled wait reached the loop handler: {raised}"
    finally:
        loop.set_exception_handler(previous)
        left.close()
        right.close()


async def test_a_second_firing_on_an_already_resolved_future_is_absorbed_too() -> None:
    """The other state `set_result` refuses, and the reason the check is `done()`.

    `InvalidStateError` has two causes, reached by different paths: the wait was
    **cancelled** (issue 58's race), or the future was **already resolved** and
    the fd fired again before anything deregistered it. They matter separately
    because a pre-check spelled `if not future.cancelled()` — the obvious
    reading, and the one asked about — passes the second straight through:
    a resolved future reports `cancelled() is False`.

    Under anyio's own shape the done-callback wins and this fires once, so this
    drives the unbounded case directly by registering no removal at all. That
    is also why the guard catches rather than checks: the exception is paid only
    on the firing that races, where a check is paid on every readiness event.

    Sabotage: narrow the `except` in `cancelsafe._guarding` to
    `except asyncio.CancelledError` and this reports the swallowed errors.
    """
    raised: list[str] = []
    fired: list[bool] = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: raised.append(
            f"{context.get('message')}: {context.get('exception')!r}"
        )
    )

    class Counting(asyncio.Future[None]):
        def set_result(self, value: None, /) -> None:
            fired.append(self.done())
            super().set_result(value)

    left, right = socket.socketpair()
    left.setblocking(False)
    try:
        future: asyncio.Future[None] = Counting(loop=loop)
        # Deliberately no done-callback, and the byte is never read, so the fd
        # stays readable and registered and the loop re-fires it every pass.
        loop.add_reader(left, future.set_result, None)
        right.send(b"x")
        await future
        await asyncio.sleep(0.01)
        loop.remove_reader(left)

        assert len(fired) > 1, f"the fd was only fired once, so nothing re-fired: {len(fired)}"
        assert fired[0] is False and fired[1] is True, (
            f"expected a pending firing then an already-resolved one: {fired[:2]}"
        )
        assert raised == [], f"a repeat firing reached the loop handler: {raised[:2]}"
    finally:
        loop.set_exception_handler(previous)
        left.close()
        right.close()


async def test_the_guard_leaves_the_fd_for_anyio_to_deregister() -> None:
    """The removal is anyio's, and doing it here cost 9.9% of a round trip.

    An earlier guard deregistered the fd inline before resolving, on the reading
    that this is what anyio's hardened `wait_readable` does. It is not: that one
    guards its removal behind a `read_events` bookkeeping dict precisely so it
    never removes twice. Removing here made anyio's own done-callback a *miss* —
    `selectors` raises `KeyError` twice, nested, and formats a `repr(socket)`
    that costs `getsockname` and `getpeername` — on every wait rather than only
    cancelled ones: measured at **+16.8 µs on every daemon round trip, +9.9%**,
    against +1.0% for swallowing alone.

    Sabotage: put `loop.remove_reader(fd)` back at the top of
    `cancelsafe._guarding`'s `fire` and this reports `[False]`.
    """
    loop = asyncio.get_running_loop()
    watched_at_resolution: list[bool] = []

    class Observing(asyncio.Future[None]):
        def set_result(self, value: None, /) -> None:
            watched_at_resolution.append(watch_list(loop, left))
            super().set_result(value)

    left, right = socket.socketpair()
    left.setblocking(False)
    try:
        future: asyncio.Future[None] = Observing(loop=loop)
        loop.add_reader(left, future.set_result, None)
        future.add_done_callback(lambda _: loop.remove_reader(left))

        right.send(b"x")
        await future
        await asyncio.sleep(0.05)

        assert watched_at_resolution == [True], (
            "the guard deregistered the fd itself, which turns anyio's own "
            "removal into a KeyError miss on every wait"
        )
        assert not watch_list(loop, left), "anyio's done-callback never removed the fd"
    finally:
        left.close()
        right.close()
