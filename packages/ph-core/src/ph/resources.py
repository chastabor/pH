"""Resource ownership: every artifact unwinds with its scope (§4.9, I2).

Phase 0 made *registrations* effects. This makes **artifacts** effects too —
child processes, temp directories, locks, worktrees — so cleanup is structural
rather than remembered. The rule that keeps it honest is a lint, not a
convention: `subprocess.Popen` and `tempfile.mkdtemp` outside the seams are a
test failure, because the fiftieth plugin author will not have read §4.9.

Shutdown is the other half. A harness that leaves child processes behind on
`SIGTERM` is a harness that leaks a runtime per crash, so:

* `atexit` disposes the root on a normal exit;
* `SIGTERM`/`SIGINT` dispose it with a grace period, then **self-`SIGKILL`** —
  because a shutdown path that can hang is a shutdown path that will.

`SIGKILL` itself runs nothing, on any platform (N7). That is why the crash
layer exists separately: paired events and the orphan journal (Phase 3).

@module ph.resources
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import shutil
import signal
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio

from .cordis import Context, Disposer

__all__ = [
    "GRACE_SECONDS",
    "install_lifecycle",
    "temporary_directory",
]

log = logging.getLogger("ph.resources")

GRACE_SECONDS = 10.0
"""How long an orderly shutdown may take before pH stops waiting on itself."""


async def temporary_directory(ctx: Context, *, prefix: str = "ph-") -> Path:
    """An unguessable 0700 temp directory that disposes with `ctx`.

    Deliberately not `TemporaryDirectory`: its cleanup is a `weakref.finalize`,
    so the directory survives until the object is collected — GC-timed cleanup
    of a path something else may reuse. Acquired through `ctx.effect()` like
    every other artifact (§4.9), so acquisition and its disposer are one step: a
    failure between the two cannot leave the directory unregistered.
    """
    created: list[Path] = []

    def enter() -> Disposer:
        path = Path(tempfile.mkdtemp(prefix=prefix))
        path.chmod(0o700)
        created.append(path)
        return lambda: shutil.rmtree(path, ignore_errors=True)

    await ctx.effect(enter, label="tempdir")
    return created[0]


def install_lifecycle(
    ctx: Context,
    *,
    grace_seconds: float = GRACE_SECONDS,
    on_signal: Callable[[int], None] | None = None,
) -> Disposer:
    """Dispose `ctx` on exit and on `SIGTERM`/`SIGINT`.

    A signal handler runs on the main thread, *interrupting* whatever the event
    loop was doing — so it cannot block waiting for an async teardown without
    deadlocking the loop it needs. Instead it schedules the teardown as a task
    and returns; the loop then runs it, and the shutdown task is what finally
    leaves.

    Past the grace period pH stops trusting its own teardown and re-raises the
    signal with the default handler: a shutdown path that can hang is a shutdown
    path that will. `SIGKILL` runs nothing on any platform (N7), which is why the
    crash-recovery layer exists separately.

    Returns a disposer that removes the handlers, so a test or an embedded host
    can install and remove them without leaking global state.
    """
    finished = threading.Event()
    # A task with no reference can be garbage-collected mid-flight, which would
    # abandon the very teardown this exists to run.
    pending: set[asyncio.Task[None]] = set()

    async def unwind(signum: int) -> None:
        try:
            with anyio.move_on_after(grace_seconds, shield=True):
                await ctx.dispose()
        except Exception:
            log.exception("ph.resources: orderly disposal failed")
        finally:
            finished.set()
            _leave(signum)

    def dispose_blocking(reason: str) -> None:
        """The no-loop path: `atexit`, or a signal before the loop started."""
        if finished.is_set():
            return
        finished.set()
        log.debug("ph.resources: disposing the root scope (%s)", reason)
        try:
            anyio.run(_dispose_within, ctx, grace_seconds)
        except Exception:
            log.exception("ph.resources: orderly disposal failed")

    def handle(signum: int, _frame: Any) -> None:
        if on_signal is not None:
            on_signal(signum)
        if finished.is_set():  # pragma: no cover - a second signal
            _leave(signum)
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            dispose_blocking(signal.Signals(signum).name)
            _leave(signum)
            return
        task = loop.create_task(unwind(signum))
        pending.add(task)
        task.add_done_callback(pending.discard)

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[signum] = signal.signal(signum, handle)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            log.debug("ph.resources: cannot install a handler for %s here", signum)

    atexit.register(dispose_blocking, "atexit")

    def release() -> None:
        atexit.unregister(dispose_blocking)
        for signum, handler in previous.items():
            with suppress(ValueError, OSError):  # pragma: no cover
                signal.signal(signum, handler)

    return release


def _leave(signum: int) -> None:
    """Re-raise `signum` with the default handler, so the exit code is honest."""
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except (ValueError, OSError):  # pragma: no cover
        raise SystemExit(128 + signum) from None


async def _dispose_within(ctx: Context, grace_seconds: float) -> None:
    with anyio.move_on_after(grace_seconds):
        await ctx.dispose()
