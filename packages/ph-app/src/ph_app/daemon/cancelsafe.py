"""Make a **cancelled** socket readiness wait safe, on every platform (issue 58).

anyio has two shapes for waiting on a socket. The hardened one, which
`AsyncIOBackend.wait_readable` uses and `ph_rlm.kernel.manager` reaches through
the free `anyio.wait_readable`, ends its callback with:

    try:
        fut.set_result(True)
    except asyncio.InvalidStateError:
        pass

The other shape, in `_RawSocketMixin._wait_until_readable` and
`_wait_until_writable`, in `UNIXSocketListener.accept`, and inline in
`AsyncIOBackend.connect_unix` and `create_unix_datagram_socket` — five copies —
registers the bound `future.set_result` itself and has no such guard:

    f = asyncio.Future()
    loop.add_reader(sock, f.set_result, None)
    f.add_done_callback(lambda _: loop.remove_reader(sock))
    await f

Cancel that wait after the loop has already queued the ready handle and
`set_result` runs on a cancelled future, raising `InvalidStateError` from a bare
loop handle with **no frames of ours on the stack** — which is why the captured
traceback was one line of `asyncio/events.py` and nothing else. The removal
cannot close the window: done-callbacks are `call_soon`'d, so they run an
iteration later than the firing they would prevent.

**This module supplies the missing half, and only that half.** Where a bound
`Future.set_result` is being registered as an I/O callback, it substitutes a
wrapper that tolerates the future already being settled. It does **not**
deregister the fd: the done-callback above already does, exactly once, and the
hardened shape guards its own removal behind a bookkeeping dict for the same
reason. A guard that removed the fd itself would make anyio's removal a miss —
`selectors` raises `KeyError` twice, nested, with a `repr(socket)` that costs
two more syscalls, on every wait rather than only cancelled ones.

**This is not a bug in pH.** No package here uses `call_soon`, `call_later`,
`add_reader` or `add_writer`, and pH's one `future.set_result`
(`ph_runtime.runner._resolve`) already guards with `future.done()`. All pH does
is cancel a pending wait, which is a supported operation on anyio's public API.

**Applied to the loop, not to anyio, because there are five copies.** All of
them pass through `BaseSelectorEventLoop.add_reader`/`add_writer`, so guarding
there is one patch instead of five vendored method bodies to re-check on every
upgrade. Patched on the **class**, so a loop anyio has already created is
covered too.

**One fix, every platform, which is the point.** The race lives in
`BaseSelectorEventLoop`, *above* the selector: `_process_events` queues the
ready handle and `Handle._run` executes it later in the same iteration. macOS
uses that same loop class with `KqueueSelector` where Linux uses
`EpollSelector`, so the window exists identically on both and only the hit
*rate* can differ. Guarding the shape rather than branching on `sys.platform`
is what makes this a single fix rather than one per platform. Windows is moot
rather than covered: its default `ProactorEventLoop` has no `add_reader`, and
`AF_UNIX` is not the daemon's transport there.

**Interim, with a shelf life.** `test_cancelsafe` fails when anyio stops
registering the shape this recognises, which is how we would learn the fix had
landed upstream and this module could go.

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

    The one shape that fails, named as a predicate so the test suite can assert
    on it directly: if anyio ever stops registering it, the guard has nothing
    left to do and this module should be deleted rather than left to look
    effective.

    Deliberately a *shape* test and not an identity one. `set_result` on the C
    accelerator is a `builtin_method` with no `__func__`, and a `Future`
    subclass may override it — both of which an identity comparison stops
    recognising. Anything unrecognised is passed through, so the failure is
    always to today's behaviour rather than to a changed one.
    """
    return isinstance(getattr(callback, "__self__", None), asyncio.Future) and (
        getattr(callback, "__name__", "") == "set_result"
    )


def guarded(method: object) -> bool:
    """Whether this loop method is ours rather than asyncio's.

    Read off `__module__`, which the replacement carries for free, rather than
    a marker attribute someone has to remember to set.
    """
    return bool(getattr(method, "__module__", "") == __name__)


def _guarding(original: Callable[..., Any]) -> Callable[..., Any]:
    """`add_reader` or `add_writer`, made safe for a one-shot future.

    The two differ only in which original they wrap, so they share this body
    rather than having a second hand-written copy — which is the shape that
    drifts, and is how anyio came to have five copies of the defect.
    """

    def add(self: Any, fd: Any, callback: Any, *args: Any) -> Any:  # noqa: ANN401
        if not resolves_a_future(callback):
            return original(self, fd, callback, *args)

        def fire() -> None:
            # Caught rather than *pre-checked*, and both halves of that were
            # measured. Against a bare `set_result`, the guard costs +9.5 ns as
            # a `try` and +53 ns as `if not future.done()` — because the
            # exception is only paid on the firing that races, while the check
            # is paid on every readiness event. The check wins only if about
            # one wait in three is cancelled inside the window; the observed
            # rate is nearer two per thirty full-suite runs. anyio's own
            # hardened `wait_readable` catches for the same reason.
            #
            # A pre-check would also have to ask `done()`, not `cancelled()`:
            # `set_result` refuses two states, and the second — an *already
            # resolved* future, fired on again before anyio's done-callback
            # deregisters the fd — reports `cancelled() is False`. It is the
            # rarer of the two here (measured: 1 firing under anyio's shape)
            # but the unbounded one if a deregistration is ever missed.
            #
            # `try`/`except` and not `contextlib.suppress`, same reasoning one
            # level down: 5.7 ns against 229 ns for identical behaviour, on the
            # hottest path the daemon has.
            try:  # noqa: SIM105
                callback(*args)
            except asyncio.InvalidStateError:
                pass

        return original(self, fd, fire)

    return add


def apply_cancel_safe_socket_waits() -> bool:
    """Install the guard. Returns whether this call was the one that did it.

    Idempotent, so importing any module under `ph_app.daemon` applies it once
    and a second import is free.
    """
    if guarded(BaseSelectorEventLoop.add_reader):
        return False
    # Two ignore codes each: typeshed declares both methods on `BaseEventLoop`
    # rather than on this subclass, so mypy reads the rebind as an assignment to
    # an inherited attribute as well as an override.
    BaseSelectorEventLoop.add_reader = _guarding(  # type: ignore[method-assign,assignment]
        BaseSelectorEventLoop.add_reader
    )
    BaseSelectorEventLoop.add_writer = _guarding(  # type: ignore[method-assign,assignment]
        BaseSelectorEventLoop.add_writer
    )
    return True
