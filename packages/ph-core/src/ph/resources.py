"""Resource ownership: every artifact unwinds with its scope (§4.9, I2).

Phase 0 made *registrations* effects. This makes **artifacts** effects too —
child processes, temp directories, locks, worktrees — so cleanup is structural
rather than remembered. The rule that keeps it honest is a lint, not a
convention: `subprocess.Popen` and `tempfile.mkdtemp` outside the seams are a
test failure, because the fiftieth plugin author will not have read §4.9.

Shutdown is the other half. A harness that leaves child processes behind on
`SIGTERM` is a harness that leaks a runtime per crash, so every host (the daemon,
`phern -p`, `--mode rpc`) takes `SIGTERM`/`SIGINT` as its orderly stop, with a
**hard stop** behind it that kills what the process still owns and leaves
(`stop_on_signals`), because a shutdown path that can hang is a shutdown path that
will.

`SIGKILL` itself runs nothing, on any platform (N7). That is why the crash
layer exists separately: paired events and the orphan journal (Phase 3).

@module ph.resources
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import NoReturn

import anyio
from anyio.abc import TaskStatus

from .cordis import GRACE_SECONDS as _GRACE_SECONDS
from .cordis import Context, Disposer
from .orphans import OrphanJournal, host_journal

__all__ = [
    "EXIT_SECONDS",
    "GRACE_SECONDS",
    "HARD_STOP_THREAD",
    "SHUTDOWN_SECONDS",
    "Stopped",
    "leave_on",
    "run_until_signaled",
    "stop_on_signals",
    "temporary_directory",
    "until_signaled",
]

log = logging.getLogger("ph.resources")

GRACE_SECONDS = _GRACE_SECONDS
"""How long an orderly shutdown may take before pH stops waiting on itself.

Re-exported from `ph.cordis`, which now owns the number because `Context.dispose`
applies it itself. The name stays here because three shutdown paths import it
from this module, and because "how long shutdown may take" is a statement about
this module's subject even when the enforcement moved down a layer."""

EXIT_SECONDS = 5.0
"""How long a process may take to leave once its roots are down.

A literal rather than a share of something else, because it is not a share of
anything: the socket closing, the last flush reaching disk and the interpreter
exiting are not the unwind's work, and borrowing `DRAIN_SECONDS` for them — which
this did — meant that widening the drain silently widened an observer's patience
with a process that had stopped unwinding.
"""

SHUTDOWN_SECONDS = _GRACE_SECONDS + EXIT_SECONDS
"""How long to allow a whole process to stop in, from the outside.

The teardown's budget plus the leaving. Derived, because a third literal
agreeing with the other two by inspection is how a bound stops being the one
that fires — `daemon/server.py` names that hazard about its own pair.

For a caller *waiting on* a shutdown rather than performing one, so that "it has
not stopped yet" means it rather than meaning the wait was shorter than the work.
That is `phern shutdown`, and the hard stop `stop_on_signals` arms, which watches
its own process from a thread.
"""

HARD_STOP_THREAD = "ph-hard-stop"
"""The name of the thread `stop_on_signals` arms, so a test can see whether one is
still armed after the stop it bounds has finished."""


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


async def stop_on_signals(
    stop: Callable[[int], bool],
    *,
    within: float = SHUTDOWN_SECONDS,
    task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
) -> None:
    """For a host's whole life: the first `SIGTERM` or `SIGINT` asks for its orderly
    stop, and a hard stop ends the process if that stop has not finished `within`
    seconds later.

    `stop` is the host's own stop, called with the signal: the daemon's `stop`
    event, or a one-shot's cancel scope (`until_signaled`). The unwind does the
    work: sub-agents are suspended in their own logs, child processes are stopped by
    their disposers, logs are written and leases given back. `stop` returns `False`
    when a stop was already under way, a daemon's `shutdown` frame say, and then
    this signal counts as the second.

    **The hard stop is a thread.** The first signal arms it, and it fires `within`
    seconds later unless this task ends first. A host ends it by canceling this task
    once its unwind is over. It has to be a thread because it exists for a host that
    has stopped answering, a loop held by a call that never yields, where a task
    would never run. It kills what this process still has in the orphan journal
    (`OrphanJournal.kill_owned`) and exits with `128 + signum`. Nothing in it logs:
    a stuck loop may hold the lock of the very handler it would write through.

    **A second signal is the same stop at once**, leaving through the signal's own
    default action (`leave_on`), so a person who asks twice does not wait out the
    bound.

    Start it with `TaskGroup.start`, so the receiver is in place before the host's
    work begins. A signal that arrives earlier takes the default action.
    """
    hard: threading.Timer | None = None
    journal: OrphanJournal | None = None
    try:
        with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as arriving:
            task_status.started()
            async for signum in arriving:
                if hard is None:
                    # Resolved on the first signal, on the loop: a host that is never
                    # signaled never needs the journal, and the thread must not be
                    # the one to create `$PH_RUNTIME`.
                    journal = host_journal()
                    hard = threading.Timer(within, _leave_now, (signum, journal))
                    hard.name = HARD_STOP_THREAD
                    hard.daemon = True
                    hard.start()
                    if stop(signum):
                        continue
                log.warning(
                    "ph.resources: %s while stopping; leaving at once", signal.Signals(signum).name
                )
                _leave_now(signum, journal, leave=leave_on)
    finally:
        if hard is not None:
            hard.cancel()


def _exit_on(signum: int) -> NoReturn:
    os._exit(128 + signum)


def _leave_now(
    signum: int, journal: OrphanJournal | None, *, leave: Callable[[int], NoReturn] = _exit_on
) -> NoReturn:
    """The hard stop: what the process still owns is killed, and the process leaves
    without running another line of its unwind.

    On its own thread it leaves by `os._exit` rather than `leave_on`, because only
    the main thread may set a signal's handler; a second signal, on the loop, passes
    `leave_on`. Leaving is in the `finally` so that nothing the kill raises can keep
    a process alive past its bound."""
    try:
        if journal is not None:
            journal.kill_owned()
    finally:
        leave(signum)


@dataclass(frozen=True, slots=True)
class Stopped:
    """What `until_signaled` returns in place of its run's value: the signal that
    stopped the run."""

    signum: int


async def until_signaled[T](
    run: Callable[[], Awaitable[T]], *, within: float = SHUTDOWN_SECONDS
) -> T | Stopped:
    """Run `run` for a one-shot host (`phern -p`, `--mode json`, `--mode rpc`), and
    let a signal stop it the way it stops the daemon (`stop_on_signals`).

    The first `SIGTERM` or `SIGINT` cancels `run`, so the mount it holds unwinds as
    it does on any exit, and what comes back is `Stopped`. Before this the default
    action ended the process at once, which is a crash under another name: children
    were left `running`, which spends one of their restart attempts at the next
    start, and the log's unwritten tail was lost.

    A host must then leave through the signal rather than return; from synchronous
    code, `run_until_signaled` is this with that step included.
    """
    signaled: list[int] = []
    body = anyio.CancelScope()

    def stop(signum: int) -> bool:
        signaled.append(signum)
        body.cancel()
        return True

    finished: tuple[T] | None = None
    failed: Exception | None = None
    async with anyio.create_task_group() as tasks:
        await tasks.start(partial(stop_on_signals, stop, within=within))
        with body:
            try:
                finished = (await run(),)
            except Exception as error:
                # Raised past the group rather than through it: a task group wraps
                # what its body raises in an `ExceptionGroup`, and the host's
                # `except SessionBusy` would no longer match the refusal it names.
                failed = error
        tasks.cancel_scope.cancel()
    if failed is not None:
        raise failed
    if finished is None:
        return Stopped(signaled[0])
    return finished[0]


def run_until_signaled[T](
    run: Callable[[], Awaitable[T]], *, on_stop: Callable[[int], None] | None = None
) -> T:
    """A one-shot host's whole life from synchronous code: `run` under
    `until_signaled`, and a run a signal stopped never returns.

    Once its unwind is over, `on_stop` is told the signal (a host's one line on
    stderr), stdio is flushed, and the process leaves through the signal
    (`leave_on`), so whoever sent it reads the exit it asked for. Through the signal
    rather than by returning, because `--mode rpc` can still have a worker thread
    blocked reading stdin, and the interpreter would wait for it at exit. Flushed
    first, since the default action writes nothing more: `--mode json` has already
    streamed its events to stdout. What `run` raises comes back as itself.
    """
    outcome = anyio.run(until_signaled, run)
    if not isinstance(outcome, Stopped):
        return outcome
    if on_stop is not None:
        on_stop(outcome.signum)
    sys.stdout.flush()
    sys.stderr.flush()
    leave_on(outcome.signum)


def leave_on(signum: int) -> NoReturn:
    """Re-raise `signum` with the default handler, so the exit code is honest.

    Public for the hosts that leave this way once their unwind is over, or when a
    second signal says not to wait for it (`stop_on_signals`). The default action
    ends the process before `os.kill` returns; where it cannot be sent, the same
    code is the exit's own."""
    with suppress(ValueError, OSError):
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(128 + signum)
