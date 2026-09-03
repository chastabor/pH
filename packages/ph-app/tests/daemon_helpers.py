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
import pytest

from ph.bundles import BASE, HEADLESS
from ph.cordis import Profile
from ph.paths import resolve_roots
from ph_app.daemon import DaemonClient, serve

__all__ = ["PROFILE", "Daemon", "running", "shut_down", "until"]

PROFILE = Profile.from_paths([BASE, HEADLESS])


@dataclass(slots=True)
class _Daemon:
    """A running supervisor and the socket it answers on."""

    path: Path
    tasks: Any
    server: Any = None
    """The `DaemonServer` behind the socket, for the tests whose subject is the
    supervisor itself rather than the wire."""

    async def client(self, *capabilities: str, on_notify: Any = None) -> DaemonClient:
        """One connected, pumping client. `capabilities` are what it declares.

        Passing `"asks"` is what makes it a front end — see `AskDesk`. The
        observer is keyword-only because almost nothing passes one, and leading
        with it made every front end in the suite open with a `None` placeholder."""
        client = await DaemonClient.connect(self.path, on_notify)
        self.tasks.start_soon(client.pump)
        if capabilities:
            await client.initialize(*capabilities)
        return client

    async def root(self, session_id: str = "root") -> Any:
        """One live root, started the way `session/attach` starts one."""
        return await self.server.supervisor.start(session_id)


Daemon = _Daemon


async def until(done: Any, *, what: str, seconds: float = 10.0) -> None:
    """Poll until `done()`, or fail saying what was being waited for.

    Here rather than in one test file because four of them wrote the loop out —
    and this module's own docstring makes the argument: a copy that gets missed
    fails as a *hang*, which is the least legible failure a suite has. The
    message matters as much as the wait: `fail_after` raises a bare
    `TimeoutError`, so a call site's `what=` reached nothing without this.
    """
    try:
        with anyio.fail_after(seconds):
            while not done():
                await anyio.sleep(0.01)
    except TimeoutError:
        pytest.fail(f"timed out waiting for {what}")


@asynccontextmanager
async def running(
    tmp_path: Path,
    *,
    name: str = "",
    path: Path | None = None,
    profile: Profile | None = None,
    **options: Any,
) -> AsyncIterator[_Daemon]:
    """A daemon, started and accepting, torn down when the block ends.

    Teardown is a cancel; tests that want the real path send `shutdown` through
    the socket themselves, so that path is exercised by the tests whose subject
    it is rather than by every one of them.

    `profile` composes a different deployment — for a test whose subject is what
    happens when a row is *absent*, which cannot be reached by taking a seam off
    a mounted root (`ctx.provide` refuses a second claim in the same realm, and
    rightly).

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
            lambda: serve(
                profile or PROFILE, path=socket, ready=ready, started=started.append, **options
            )
        )
        await ready.wait()
        try:
            yield _Daemon(path=socket, tasks=tasks, server=started[0])
        finally:
            tasks.cancel_scope.cancel()


async def shut_down(path: Path) -> None:
    """Stop a daemon that is *not* in this process, through its only handle.

    Connect, pump, `initialize`, `shutdown`, wait for the close: the sequence a
    spawned daemon leaves a test no alternative to, written once rather than in
    every test that spawns one.
    """
    client = await DaemonClient.connect(path)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(client.pump)
        await client.call("initialize")
        await client.notify("shutdown")
        with anyio.fail_after(20):
            await client.closed.wait()


def daemon_socket() -> Path:
    """Where a client will look, given whatever `$PH_RUNTIME` currently says."""
    return resolve_roots().ensure().daemon_socket()
