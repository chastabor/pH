"""Getting a daemon, from a UI that would rather not know there is one (P7-08).

The harness runs in the daemon now, so every interactive front end needs one to
exist. Requiring a person to start it first would make the ordinary case — open a
terminal, type a prompt — two commands, and the second one would be the kind
nobody remembers until the first fails. So a UI starts one itself when the socket
is absent; `DaemonServer.ephemeral` says why the one it starts is not a service.

**Absent is not the same as unresponsive, and the distinction is the whole of the
error handling here.** No socket means no daemon was ever started, or the one
that was has left — start one. A socket that exists and refuses is the aftermath
of a crash, or a logout that reaped the door out from under a process still
holding every session lease it took (P5-11) — and starting a second daemon there
is precisely the wrong move. This module's job ends at "there is now something
listening"; the connect that follows raises through the diagnosis `ph agents`
already has.

**Two UIs opening at once must start one daemon, not two.** Both find no socket,
both spawn, and the loser's `serve` hits `_clear_stale`, finds a *live* socket and
refuses — leaving a person who opened two terminals looking at a traceback from
the second. The lock makes the check-and-spawn one step, and the re-check *inside*
it is what turns the loser into a client instead of a casualty.

@module ph_app.daemon.launch
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio
from filelock import FileLock, Timeout

from ph.paths import PathRoots, resolve_roots

from ..protocol import Refusal

__all__ = ["DaemonAbsent", "Started", "ensure_daemon", "listening"]

log = logging.getLogger("ph_app.daemon.launch")

SPAWN_TIMEOUT = 30.0
"""How long to wait for a daemon we started to answer on its socket.

Measured at ~0.24 s on a warm cache; generous because the cost of being wrong is
asymmetric — a timeout that fires early tells a person their daemon failed to
start while it is still starting, and they cannot tell that from the truth.
"""

LOCK_TIMEOUT = SPAWN_TIMEOUT + 5.0
"""How long the loser of a race waits for the winner's spawn — longer than the
spawn it is serialising against, so one slow start is not two failures."""


class DaemonAbsent(Refusal):
    """There is no daemon and this caller will not, or could not, start one.

    Its own type because the caller's next step is a command they can type, and
    because `--no-spawn` is a deliberate posture rather than a failure: a person
    running a daemon under systemd wants to know their UI did not quietly start a
    second one beside it.
    """

    code = "daemon_absent"


@dataclass(frozen=True, slots=True)
class Started:
    """What `ensure_daemon` did, so a UI can say "starting a supervisor…" rather
    than looking hung for the seconds a spawn takes."""

    path: Path
    spawned: bool


async def listening(path: Path) -> bool:
    """Whether something answers on this path right now.

    A connect and an immediate close — the only test that distinguishes "a socket
    file exists" from "a daemon is behind it", which is the distinction this
    module and `_clear_stale` are both built on. One copy, so a change here (a
    connect timeout on a hung listener, say) reaches both.
    """
    try:
        stream = await anyio.connect_unix(str(path))
    except OSError:
        return False
    await stream.aclose()
    return True


async def ensure_daemon(
    *, argv: Sequence[str], roots: PathRoots | None = None, spawn: bool = True
) -> Started:
    """Make sure a daemon is listening, starting one with `argv` if not.

    Returns without doing anything when one already answers, which is the
    common case and must stay cheap: a second terminal, a browser tab, a
    `ph agents` call. `argv` is the caller's — this module knows nothing about
    profiles or providers; `ph_app.cli.spawn_command` spells the command whose
    options those are.

    `spawn=False` is `--no-spawn`: refuse rather than start one, for a deployment
    that runs `ph daemon` under an init system.
    """
    resolved = roots if roots is not None else resolve_roots(create=True)
    path = resolved.daemon_socket()
    if await listening(path):
        return Started(path=path, spawned=False)
    if not spawn:
        raise DaemonAbsent(f"no daemon at {path}; start one with `ph daemon`")

    # `thread_local=False` for the reason the session lease uses it: filelock
    # keeps its re-entrancy counter in a thread-local, and anyio runs the blocking
    # acquire on a worker thread while the release happens on the event loop —
    # where the counter is zero and releasing silently does nothing.
    lock = FileLock(str(resolved.runtime / "daemon.lock"), timeout=LOCK_TIMEOUT, thread_local=False)
    try:
        await anyio.to_thread.run_sync(lock.acquire)
    except Timeout as error:
        # A live holder inside the lock longer than a spawn can take. Not a
        # reason to spawn anyway — that is the race the lock exists to stop.
        raise DaemonAbsent(
            f"another launcher has held {lock.lock_file} for {LOCK_TIMEOUT:g}s; "
            "run `ph daemon` to see why"
        ) from error
    try:
        # **Inside the lock**, and this is the line that makes the race benign:
        # the loser of two simultaneous launches gets here after the winner's
        # daemon is already answering, sees it, and becomes a client.
        if await listening(path):
            return Started(path=path, spawned=False)
        log.info("ph_app.daemon: no daemon at %s; starting one", path)
        _detach(list(argv))
        await _await_socket(path)
        return Started(path=path, spawned=True)
    finally:
        await anyio.to_thread.run_sync(lock.release)


def _detach(argv: list[str]) -> None:
    """Start the daemon as a process that outlives this one.

    The whole point of a daemon is that closing the terminal does not end the
    session, so it must not be a child that dies with its parent's process group
    on a Ctrl-C. `start_new_session` puts it in one of its own — the portable
    half of the double-fork idiom, and enough here because pH's daemon needs no
    controlling terminal.

    Output goes to the null device rather than being inherited: the daemon prints
    its socket path and its linger warning on the way up, and those landing in the
    middle of a Textual screen would corrupt the UI that started it. What it says
    afterwards belongs in the logs and in `ph agents doctor`.
    """
    # argv is built by the caller's `spawn_command`, never from anything typed.
    subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


async def _await_socket(path: Path) -> None:
    """Wait for something to answer, or say what we waited for and gave up on.

    Polls the *connect* rather than the file, because the file exists before
    `serve()` is listening — exactly the window a `path.exists()` poll lands in,
    and the reason `serve` takes a `ready` event for callers in the same process.
    A spawned daemon is in another one, so this is the honest test.
    """
    with anyio.move_on_after(SPAWN_TIMEOUT):
        while not await listening(path):
            await anyio.sleep(0.02)
        return
    raise DaemonAbsent(
        f"started a daemon but nothing is listening on {path} after {SPAWN_TIMEOUT:g}s; "
        "run `ph daemon` in another terminal to see why"
    )
