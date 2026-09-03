"""Standing a supervisor up, written once.

`running` is the whole of `serve()`'s startup contract — the task group, the
`ready` event, the `started` handle, and the cancel behind teardown — and that
contract changed under P5-10 (the listener moved out of the group; the socket
path and the cadences moved onto `DaemonServer`). Two copies of it in two test
files is one copy that gets missed, and a missed one fails as a hang rather
than as a diff.

Waits on `ready` rather than for the socket file to appear: the path exists
before `serve()` is listening, which is exactly the window a poll would land in.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from ph.bundles import BASE, HEADLESS
from ph.cordis import Profile
from ph.paths import resolve_roots
from ph_app.daemon import DaemonClient, serve

__all__ = ["PROFILE", "Daemon", "running"]

PROFILE = Profile.from_paths([BASE, HEADLESS])


@dataclass(slots=True)
class _Daemon:
    """A running supervisor and the socket it answers on."""

    path: Path
    tasks: Any
    server: Any = None
    """The `DaemonServer` behind the socket, for the tests whose subject is the
    supervisor itself rather than the wire."""

    async def client(self, on_notify: Any = None, *capabilities: str) -> DaemonClient:
        """One connected, pumping client. `capabilities` are what it declares.

        Passing `"asks"` is what makes it a front end — see `AskDesk`."""
        client = await DaemonClient.connect(self.path, on_notify)
        self.tasks.start_soon(client.pump)
        if capabilities:
            await client.initialize(*capabilities)
        return client


Daemon = _Daemon


@asynccontextmanager
async def running(
    tmp_path: Path, *, name: str = "", path: Path | None = None, **options: Any
) -> AsyncIterator[_Daemon]:
    """A daemon, started and accepting, torn down when the block ends.

    Teardown is a cancel; tests that want the real path send `shutdown` through
    the socket themselves, so that path is exercised by the tests whose subject
    it is rather than by every one of them.

    `name` is how a test runs *two* daemons over one `$PH_HOME` — the only way
    to reach P5-03's question, since `_clear_stale` makes one socket refuse a
    second listener before a lease could be asked for. `path` is the other
    direction: an explicit socket, for a test that has pinned `$PH_RUNTIME` and
    wants both halves to derive the same one.
    """
    async with anyio.create_task_group() as tasks:
        ready = anyio.Event()
        started: list[Any] = []
        socket = path if path is not None else tmp_path / name / "daemon.sock"
        socket.parent.mkdir(parents=True, exist_ok=True)
        # `passivate_after=None` by default: the sweeper is a background timer,
        # and a test that did not ask about passivation should not have one
        # racing its assertions. P5-05's own tests opt in.
        options.setdefault("passivate_after", None)
        tasks.start_soon(
            lambda: serve(PROFILE, path=socket, ready=ready, started=started.append, **options)
        )
        await ready.wait()
        try:
            yield _Daemon(path=socket, tasks=tasks, server=started[0])
        finally:
            tasks.cancel_scope.cancel()


def daemon_socket() -> Path:
    """Where a client will look, given whatever `$PH_RUNTIME` currently says."""
    return resolve_roots().ensure().daemon_socket()
