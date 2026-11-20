"""Get a daemon for a UI, so that the user does not have to start one (P7-08).

The harness now runs in the daemon, so every interactive front end needs a
daemon to exist. If the user had to start the daemon first, the ordinary case
(open a terminal, type a prompt) would take two commands. Nobody would remember
the second command until the first one failed. A UI therefore starts a daemon
itself when no daemon answers on the socket. `DaemonServer.ephemeral` explains
why the daemon that a UI starts is not a service.

**This module asks whether a daemon answers. It does not ask why none does.**
When nothing answers on the socket, the module starts a daemon. That is correct
when no daemon was ever started, when the daemon has exited, and when a crash
left a socket file behind: `_clear_stale` in the new daemon removes that file.
One case is different. A logout can remove the socket while the process behind
it still holds every session lease that it took (P5-11). A second daemon is the
wrong answer there, but this module does not diagnose that case. The job of
this module ends when something is listening. The connect that follows raises
through the diagnosis that `phern agents` already has.

**Two UIs that open at the same time must start one daemon, not two.** Without
a lock, both find no socket and both spawn. The `serve` of the loser reaches
`_clear_stale`, finds a *live* socket, and refuses. The user who opened two
terminals then sees a traceback from the second terminal. The lock makes the
check and the spawn one step. The second check *inside* the lock makes the
loser a client and not a failure.

@module ph_app.daemon.launch
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio

from ph.framing import listening
from ph.locks import LockBusy, acquire_file_lock
from ph.paths import PathRoots, resolve_roots

from ..protocol import Refusal

__all__ = ["DaemonAbsent", "Started", "ensure_daemon", "phern_argv"]

log = logging.getLogger("ph_app.daemon.launch")

SPAWN_TIMEOUT = 30.0
"""How long to wait for a daemon that this module started to answer on its socket.

A start was measured at about 0.24 s on a warm cache. The timeout is generous
because the two possible errors do not cost the same. A timeout that fires early
tells the user that their daemon failed to start while it is still starting, and
the user cannot tell that message from a true one.
"""

LOCK_TIMEOUT = SPAWN_TIMEOUT + 5.0
"""How long the loser of a race waits for the spawn of the winner. The wait is
longer than the spawn that it serializes against, so that one slow start does
not become two failures."""

TAIL_LINES = 12
"""How many of the last stderr lines of a daemon go into the refusal when the
daemon exits before it listens. The number is enough for a usage error or the end
of a traceback, and small enough to read in a modal."""

DETACHED = "PH_DAEMON_DETACHED"
"""Set in the environment of a daemon that `_detach` starts. The stderr of that
daemon is the file of the launcher, which is read only until the socket answers.
`phern daemon` reads this variable and keeps its warnings out of that file after
it is up (`configure_daemon_logging(to_stderr=…)`). The rotating log already
holds every warning, and this file does not rotate."""


def phern_argv(*args: str) -> list[str]:
    """Return the argv that starts another process of this pH, given its words.

    The argv is `sys.executable -m ph_app` and not a bare `phern`. The caller may
    run from a virtualenv that is not on `PATH`, or from a checkout with no
    console script installed. A pH that a *different* pH started has a profile,
    an event vocabulary and a wire version that nobody chose. The spelling is
    written one time, here, for the three callers that start a process: the
    daemon that a UI spawns and the terminal that a browser tab runs (both
    through `ph_app.cli.reinvoke`), and the daemon that `phern service` starts.
    """
    return [sys.executable, "-m", "ph_app", *args]


class DaemonAbsent(Refusal):
    """There is no daemon, and this caller will not start one or could not.

    The refusal has its own type for two reasons. The next step of the caller is
    a command that they can type. Also, `--no-spawn` is a deliberate choice and
    not a failure: a user who runs a daemon under systemd wants to know that
    their UI did not quietly start a second daemon beside it.
    """

    code = "daemon_absent"


@dataclass(frozen=True, slots=True)
class Started:
    """What `ensure_daemon` did. A UI can then show "starting a supervisor…" and
    does not look hung for the seconds that a spawn takes."""

    path: Path
    spawned: bool


async def ensure_daemon(
    *, argv: Sequence[str], roots: PathRoots | None = None, spawn: bool = True
) -> Started:
    """Make sure that a daemon is listening. Start one with `argv` if none is.

    The function returns without other work when a daemon already answers. That
    is the common case and must stay cheap: a second terminal, a browser tab, a
    `phern agents` call. The caller supplies `argv`. This module knows nothing
    about profiles or providers. `ph_app.cli.spawn_command` builds the command
    that has those options.

    `spawn=False` is `--no-spawn`: refuse and do not start a daemon. It is for a
    deployment that runs `phern daemon` under an init system.
    """
    resolved = roots if roots is not None else resolve_roots(create=True)
    path = resolved.daemon_socket()
    if await listening(path):
        return Started(path=path, spawned=False)
    if not spawn:
        raise DaemonAbsent(f"no daemon at {path}; start one with `phern daemon`")

    # The lock is acquired on a worker thread because the acquire blocks. That is
    # the reason why `ph.locks` sets `thread_local=False`: the release below runs
    # on the event loop, where a thread-local counter would be zero and the
    # release would silently do nothing.
    lock_path = resolved.runtime / "daemon.lock"
    try:
        release = await anyio.to_thread.run_sync(
            partial(acquire_file_lock, lock_path, timeout=LOCK_TIMEOUT, what="the daemon launcher")
        )
    except LockBusy as busy:
        # A live holder has been inside the lock for longer than a spawn can
        # take. That is not a reason to spawn anyway: the lock exists to stop
        # that race.
        raise DaemonAbsent(
            f"another launcher has held {lock_path} for {LOCK_TIMEOUT:g}s; "
            "run `phern daemon` to see why"
        ) from busy
    try:
        # This check is **inside the lock**, and it makes the race harmless. The
        # loser of two simultaneous launches arrives here after the daemon of
        # the winner already answers. The loser sees it and becomes a client.
        if await listening(path):
            return Started(path=path, spawned=False)
        log.info("ph_app.daemon: no daemon at %s; starting one", path)
        stderr = resolved.daemon_stderr()
        process = _detach(list(argv), stderr)
        await _await_socket(path, process, stderr)
        return Started(path=path, spawned=True)
    finally:
        await anyio.to_thread.run_sync(release)


def _detach(argv: list[str], stderr: Path) -> subprocess.Popen[bytes]:
    """Start the daemon as a process that outlives this one.

    The purpose of a daemon is that closing the terminal does not end the
    session. The daemon must therefore not be a child that dies with the process
    group of its parent on a Ctrl-C. `start_new_session` puts the daemon in a
    process group of its own. That is the portable half of the double-fork idiom,
    and it is enough here because the daemon of pH needs no controlling terminal.

    The daemon does not inherit the output streams. It prints its socket path and
    its linger warning during startup, and those lines in the middle of a Textual
    screen would corrupt the UI that started it. Stdout goes to the null device.
    **Stderr goes to `stderr`, which is truncated first.** A daemon that exits
    before it listens writes the reason to that stream. When the stream went to
    the null device, a UI could report only the exit status, with advice to run
    the command again in a place where the output could be seen. `_await_socket`
    reads the file back into its refusal. After the daemon is up, its warnings go
    only to its own rotating log (`DETACHED`). This file therefore keeps the
    startup output of the daemon and, if the daemon crashes, its last output.
    """
    stderr.parent.mkdir(parents=True, exist_ok=True)
    # The caller builds argv. A UI builds it with `spawn_command`, and
    # `phern service start` appends the daemon options that the user typed.
    with stderr.open("wb") as captured:
        # Popen keeps its own descriptor. This handle closes with the block.
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=captured,
            start_new_session=True,
            env={**os.environ, DETACHED: "1"},
        )


async def _await_socket(path: Path, process: subprocess.Popen[bytes], stderr: Path) -> None:
    """Wait until a daemon answers. If none does, report what the wait was for.

    The loop polls the *connect* and not the file, because the file exists before
    `serve()` listens. A `path.exists()` poll would succeed in that window. The
    same window is the reason why `serve` takes a `ready` event for callers in
    the same process. A spawned daemon is in another process, so the connect is
    the correct test.

    A daemon can exit before it listens: a profile that does not compose, a flag
    that it does not take, a socket that it cannot bind. The exit is reported
    immediately and not after the full wait, and **in the words of the daemon**.
    `_detach` pointed the stderr of the daemon at `stderr`, so this function
    reads back the refusal that the daemon printed and puts it in the message.
    Before that change, the refusal went to the null device with the rest of the
    output. The user got an exit status and advice to run the command again in a
    terminal, to read what the daemon had already printed one time.
    """
    with anyio.move_on_after(SPAWN_TIMEOUT):
        while not await listening(path):
            code = process.poll()
            if code is not None:
                raise DaemonAbsent(
                    _refusal(
                        f"the daemon exited with status {code} before listening on {path}",
                        stderr,
                    )
                )
            await anyio.sleep(0.02)
        return
    raise DaemonAbsent(
        _refusal(
            f"started a daemon but nothing is listening on {path} after {SPAWN_TIMEOUT:g}s",
            stderr,
        )
    )


def _refusal(what: str, stderr: Path) -> str:
    """Return `what`, then the last `TAIL_LINES` non-blank lines of the daemon if any.

    The function reads the whole file. The file was truncated at this spawn, so
    it holds only what the daemon wrote during startup. The function takes the
    tail because the reason in a traceback is on its last line.
    """
    try:
        text = stderr.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    said = [line.rstrip() for line in text.splitlines() if line.strip()][-TAIL_LINES:]
    if said:
        return f"{what}; it said:\n" + "\n".join(said)
    return (
        f"{what} and said nothing; run `phern daemon` with the same options in a "
        "terminal to see why"
    )
