"""Resource ownership: every artifact unwinds with its scope (§4.9, I2).

Phase 0 made *registrations* effects. This module makes **artifacts** effects
too: child processes, temp directories, locks, and worktrees. Cleanup is then
part of the structure, and nobody has to remember it. A lint enforces the rule,
not a convention: `subprocess.Popen` and `tempfile.mkdtemp` outside the seams
are a test failure, because the fiftieth plugin author will not have read §4.9.

Shutdown is the other half. A harness that leaves child processes behind on
`SIGTERM` leaks a runtime for each crash. So every host (the daemon, `phern -p`,
`--mode rpc`) takes `SIGTERM` or `SIGINT` as its orderly stop. A **hard stop**
stands behind the orderly stop (`stop_on_signals`). It kills what the process
still owns, and then the process leaves. The hard stop exists because a
shutdown path that can hang will hang at some time.

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
"""How long an orderly shutdown may take before pH stops waiting for it.

The value is re-exported from `ph.cordis`. That package now owns the number,
because `Context.dispose` applies it. The name stays here for two reasons. The
shutdown paths of the daemon and of the web server import it from this module.
Also, "how long shutdown may take" is a statement about the subject of this
module, even though the enforcement moved down a layer."""

EXIT_SECONDS = 5.0
"""How long a process may take to leave after its roots are down.

The value is a literal and not a share of another number, because it is not a
share of anything. The close of the socket, the last flush to disk, and the exit
of the interpreter are not the work of the unwind. This module once borrowed
`DRAIN_SECONDS` for them. A wider drain then silently made an observer wait
longer for a process that had stopped unwinding.
"""

SHUTDOWN_SECONDS = _GRACE_SECONDS + EXIT_SECONDS
"""How long an outside observer allows a whole process to stop.

The value is the budget of the teardown plus the time to leave. It is derived.
A third literal would agree with the other two only by inspection, and that is
how a bound stops being the bound that fires. `daemon/server.py` names that
hazard about its own pair.

The value is for a caller that *waits for* a shutdown and does not perform one.
With it, "it has not stopped yet" is a true statement, and does not mean that
the wait was shorter than the work. Such a caller is `phern shutdown`. Another
is the hard stop that `stop_on_signals` arms, which watches its own process from
a thread.
"""

HARD_STOP_THREAD = "ph-hard-stop"
"""The name of the thread that `stop_on_signals` arms. A test uses the name to
see whether a thread is still armed after the stop that it bounds has finished."""


async def temporary_directory(ctx: Context, *, prefix: str = "ph-") -> Path:
    """Make an unguessable 0700 temp directory that is disposed with `ctx`.

    This is deliberately not `TemporaryDirectory`. The cleanup of that class is
    a `weakref.finalize`, so the directory survives until the object is
    collected. The garbage collector would then decide when to remove a path
    that something else may reuse. The directory is acquired through
    `ctx.effect()` like every other artifact (§4.9), so the acquisition and its
    disposer are one step. A failure between the two cannot leave the directory
    unregistered.
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
    """Handle signals for the whole life of a host. The first `SIGTERM` or
    `SIGINT` asks for the orderly stop of the host. A hard stop ends the process
    if that stop has not finished `within` seconds later.

    `stop` is the stop of the host itself, called with the signal: the `stop`
    event of the daemon, or the cancel scope of a one-shot host
    (`until_signaled`). The unwind does the work. Sub-agents are suspended in
    their own logs, the disposers of child processes stop them, logs are
    written, and leases are given back. `stop` returns `False` when a stop was
    already under way, for example after a `shutdown` frame of the daemon. This
    signal then counts as the second signal.

    **The hard stop is a thread.** The first signal arms it. It fires `within`
    seconds later unless this task ends first. A host ends it by canceling this
    task when its unwind is over. It must be a thread because it exists for a
    host that has stopped answering: a loop held by a call that never yields,
    where a task would never run. It kills what this process still has in the
    orphan journal (`OrphanJournal.kill_owned`) and exits with `128 + signum`.
    Nothing in it logs, because a stuck loop may hold the lock of the same
    handler that it would write through.

    **A second signal runs the same hard stop immediately.** The process then
    leaves through the default action of the signal (`leave_on`), so a user who
    asks twice does not wait for the bound.

    Start it with `TaskGroup.start`, so that the receiver is in place before the
    work of the host begins. A signal that arrives earlier takes the default
    action.
    """
    hard: threading.Timer | None = None
    journal: OrphanJournal | None = None
    try:
        with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as arriving:
            task_status.started()
            async for signum in arriving:
                if hard is None:
                    # The journal is resolved on the first signal, on the loop. A
                    # host that is never signaled never needs the journal, and the
                    # thread must not be the one that creates `$PH_RUNTIME`.
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
    """The hard stop: kill what the process still owns, then leave without running
    another line of the unwind.

    On its own thread, the hard stop leaves by `os._exit` and not by `leave_on`,
    because only the main thread may set the handler of a signal. A second
    signal arrives on the loop, and that call passes `leave_on`. The leave is in
    the `finally`, so nothing that the kill raises can keep a process alive past
    its bound."""
    try:
        if journal is not None:
            journal.kill_owned()
    finally:
        leave(signum)


@dataclass(frozen=True, slots=True)
class Stopped:
    """What `until_signaled` returns in place of the value of its run: the signal
    that stopped the run."""

    signum: int


async def until_signaled[T](
    run: Callable[[], Awaitable[T]], *, within: float = SHUTDOWN_SECONDS
) -> T | Stopped:
    """Run `run` for a one-shot host (`phern -p`, `--mode json`, `--mode rpc`), and
    let a signal stop it in the way that a signal stops the daemon
    (`stop_on_signals`).

    The first `SIGTERM` or `SIGINT` cancels `run`. The mount that `run` holds
    then unwinds as it does on any exit, and this function returns `Stopped`.
    Before this function existed, the default action ended the process
    immediately, which had the effect of a crash. Children were left `running`,
    and each such child spends one of its restart attempts at the next start.
    The unwritten tail of the log was lost.

    After that, a host must leave through the signal and must not return. From
    synchronous code, `run_until_signaled` does the same and includes that step.
    """
    signaled: list[int] = []
    body = anyio.CancelScope()

    def stop(signum: int) -> bool:
        signaled.append(signum)
        # The cancel is named, so a turn that it cuts records which signal it
        # was. Otherwise the Ctrl-C of a user and the `SIGTERM` of a supervisor
        # are the same cancel.
        body.cancel(signal.Signals(signum).name)
        return True

    finished: tuple[T] | None = None
    failed: Exception | None = None
    async with anyio.create_task_group() as tasks:
        await tasks.start(partial(stop_on_signals, stop, within=within))
        with body:
            try:
                finished = (await run(),)
            except Exception as error:
                # The error is raised after the group and not through it. A task
                # group wraps what its body raises in an `ExceptionGroup`, and
                # the `except SessionBusy` of the host would then not match the
                # refusal that it names.
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
    """Run the whole life of a one-shot host from synchronous code: `run` under
    `until_signaled`. A run that a signal stopped never returns.

    When the unwind is over, `on_stop` receives the signal (a host prints its
    one line on stderr), stdio is flushed, and the process leaves through the
    signal (`leave_on`). The sender of the signal therefore reads the exit that
    it asked for. The process leaves through the signal and does not return,
    because `--mode rpc` can still have a worker thread blocked on a read of
    stdin, and the interpreter would wait for that thread at exit. Stdio is
    flushed first, because the default action writes nothing more, and
    `--mode json` has already streamed its events to stdout. An exception that
    `run` raises propagates unchanged.
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
    """Re-raise `signum` with the default handler, so that the exit code is correct.

    The function is public for the hosts that leave this way when their unwind
    is over, or when a second signal says not to wait for the unwind
    (`stop_on_signals`). The default action ends the process before `os.kill`
    returns. Where the signal cannot be sent, the process exits with the same
    code."""
    with suppress(ValueError, OSError):
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    raise SystemExit(128 + signum)
