"""Make a **canceled** socket readiness wait safe, on every platform (issue 58).

anyio waits on a socket in two shapes. The first shape is hardened.
`AsyncIOBackend.wait_readable` uses it, and `ph_rlm.kernel.manager` reaches it
through the free function `anyio.wait_readable`. Its callback ends with:

    try:
        fut.set_result(True)
    except asyncio.InvalidStateError:
        pass

The second shape registers the bound `future.set_result` itself and has no such
guard. It has five copies: in `_RawSocketMixin._wait_until_readable` and
`_wait_until_writable`, in `UNIXSocketListener.accept`, and inline in
`AsyncIOBackend.connect_unix` and `create_unix_datagram_socket`:

    f = asyncio.Future()
    loop.add_reader(sock, f.set_result, None)
    f.add_done_callback(lambda _: loop.remove_reader(sock))
    await f

Suppose that this wait is canceled after the loop has already queued the ready
handle. `set_result` then runs on a canceled future and raises
`InvalidStateError` from a bare loop handle, with **no frames of pH on the
stack**. That is the reason why the captured traceback was one line of
`asyncio/events.py` and nothing else. The removal of the reader cannot close the
window. Done-callbacks are scheduled with `call_soon`, so they run one iteration
later than the firing that they would prevent.

**This module supplies the missing guard, and only that guard.** Where a bound
`Future.set_result` is registered as an I/O callback, the module substitutes a
wrapper that tolerates a future that is already settled. The module does **not**
deregister the fd. The done-callback above already does that, exactly one time,
and the hardened shape guards its own removal behind a bookkeeping dict for the
same reason. If a guard removed the fd itself, the removal by anyio would miss.
`selectors` then raises `KeyError` twice, nested, with a `repr(socket)` that
costs two more syscalls. That would happen on every wait, and not only on a
canceled wait.

**This is not a bug in pH.** No package here uses `call_soon`, `call_later`,
`add_reader` or `add_writer`. The one `future.set_result` in pH
(`ph_runtime.runner._resolve`) already guards with `future.done()`. pH only
cancels a pending wait, which is a supported operation on the public API of
anyio.

**The guard is applied to the loop, not to anyio, because there are five
copies.** All five pass through `BaseSelectorEventLoop.add_reader`/`add_writer`,
so a guard there is one patch. The alternative is five vendored method bodies
that must be checked again on every upgrade. The patch is on the **class**, so
it also covers a loop that anyio has already created.

**One fix covers every platform, and that is the purpose.** The race is in
`BaseSelectorEventLoop`, *above* the selector: `_process_events` queues the
ready handle, and `Handle._run` executes it later in the same iteration. macOS
uses that loop class with `KqueueSelector`, and Linux uses it with
`EpollSelector`. The window is therefore identical on both, and only the hit
*rate* can differ. The guard tests the shape and does not branch on
`sys.platform`, so this is one fix and not one fix for each platform. Windows
is not covered and does not need to be: its default `ProactorEventLoop` has no
`add_reader`, and `AF_UNIX` is not the transport of the daemon there.

**This module is temporary.** `test_cancelsafe` fails when anyio stops
registering the shape that this module recognizes. That failure will show that
the fix has landed upstream and that this module can be deleted.

@module ph_app.daemon.cancelsafe
"""

from __future__ import annotations

import asyncio
from asyncio.selector_events import BaseSelectorEventLoop
from collections.abc import Callable
from typing import Any

__all__ = ["apply_cancel_safe_socket_waits", "guarded", "resolves_a_future"]


def resolves_a_future(callback: object) -> bool:
    """Whether this I/O callback is a bound `Future.set_result`.

    This is the one shape that fails. It is a named predicate so that the test
    suite can assert on it directly. If anyio ever stops registering the shape,
    the guard has no work left. This module should then be deleted, and not left
    in place where it looks effective.

    The test is deliberately a test of *shape* and not of identity. `set_result`
    on the C accelerator is a `builtin_method` with no `__func__`, and a `Future`
    subclass may override it. An identity comparison stops recognizing both. A
    callback that is not recognized is passed through, so a wrong answer always
    gives the current behavior and never a changed behavior.
    """
    return isinstance(getattr(callback, "__self__", None), asyncio.Future) and (
        getattr(callback, "__name__", "") == "set_result"
    )


def guarded(method: object) -> bool:
    """Whether this loop method is the replacement from this module.

    The answer is read from `__module__`, which the replacement carries without
    extra work. The alternative is a marker attribute that someone must remember
    to set.
    """
    return bool(getattr(method, "__module__", "") == __name__)


def _guarding(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap `add_reader` or `add_writer` so that it is safe for a one-shot future.

    The two differ only in the original that they wrap, so they share this body.
    A second hand-written copy is the kind of code that drifts, and that is how
    anyio came to have five copies of the defect.
    """

    def add(self: Any, fd: Any, callback: Any, *args: Any) -> Any:  # noqa: ANN401
        if not resolves_a_future(callback):
            return original(self, fd, callback, *args)

        def fire() -> None:
            # The exception is caught and not *pre-checked*, and both options
            # were measured. Against a bare `set_result`, the guard costs
            # +9.5 ns as a `try` and +53 ns as `if not future.done()`. The
            # exception is paid only on the firing that races, and the check is
            # paid on every readiness event. The check wins only if about one
            # wait in three is canceled inside the window. The observed rate is
            # nearer two per thirty full-suite runs. The hardened
            # `wait_readable` of anyio catches for the same reason.
            #
            # A pre-check would also have to ask `done()`, not `canceled()`.
            # `set_result` refuses two states. The second state is a future
            # that is *already resolved* and is fired on again before the
            # done-callback of anyio deregisters the fd. That future reports
            # `canceled() is False`. It is the rarer of the two here (measured:
            # 1 firing under the shape of anyio), but it is the unbounded one
            # if a deregistration is ever missed.
            #
            # `try`/`except` is used and not `contextlib.suppress`, by the same
            # reasoning one level down: 5.7 ns against 229 ns for identical
            # behavior, on the hottest path of the daemon.
            try:  # noqa: SIM105
                callback(*args)
            except asyncio.InvalidStateError:
                pass

        return original(self, fd, fire)

    return add


def apply_cancel_safe_socket_waits() -> bool:
    """Install the guard. Return whether this call was the one that installed it.

    The function is idempotent. Importing any module under `ph_app.daemon`
    applies the guard one time, and a second import costs nothing.
    """
    if guarded(BaseSelectorEventLoop.add_reader):
        return False
    # Each assignment has two ignore codes. typeshed declares both methods on
    # `BaseEventLoop` and not on this subclass, so mypy reads the rebind as an
    # assignment to an inherited attribute and also as an override.
    BaseSelectorEventLoop.add_reader = _guarding(  # type: ignore[method-assign,assignment]
        BaseSelectorEventLoop.add_reader
    )
    BaseSelectorEventLoop.add_writer = _guarding(  # type: ignore[method-assign,assignment]
        BaseSelectorEventLoop.add_writer
    )
    return True
