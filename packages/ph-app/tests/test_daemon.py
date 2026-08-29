"""P5-01 — the supervisor, and the property every earlier mode lacked.

Every mode before this ties an agent's life to a connection: `ph -p` exits with
the turn, the TUI's root dies with the terminal, `--mode rpc` lives as long as
stdin. The gate here is the negation of that — *TUI close leaves the root
running* — so the tests that matter are the ones where a client goes away and
the work does not.

The socket is a real unix socket under `tmp_path` and the frames are real JSONL;
there is no in-process shortcut, because what this row delivers is precisely the
transport and a fake one would agree with whatever the code did.
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
from ph_app.daemon import DaemonClient, serve

pytestmark = pytest.mark.anyio

PROFILE = [BASE, HEADLESS]


@dataclass(slots=True)
class _Daemon:
    """A running supervisor and the socket it answers on."""

    path: Path
    tasks: Any

    async def client(self, on_notify: Any = None) -> DaemonClient:
        client = await DaemonClient.connect(self.path, on_notify)
        self.tasks.start_soon(client.pump)
        return client


@asynccontextmanager
async def running(tmp_path: Path) -> AsyncIterator[_Daemon]:
    """A daemon, started and accepting, torn down when the block ends.

    Waits on the `ready` event rather than for the socket file to appear: the
    path exists before `serve()` is listening, which is exactly the window a
    poll would land in and the flake a test like this otherwise ships with.

    Teardown is `shutdown` through the socket — the same path a person uses, so
    every test exercises it — with a cancel behind it only for the tests that
    fail before they get there.
    """
    async with anyio.create_task_group() as tasks:
        ready = anyio.Event()
        path = tmp_path / "daemon.sock"
        tasks.start_soon(lambda: serve(PROFILE, path=path, ready=ready))
        await ready.wait()
        try:
            yield _Daemon(path=path, tasks=tasks)
        finally:
            tasks.cancel_scope.cancel()


async def _settled(client: DaemonClient, root_id: str, *, events: int) -> dict[str, Any]:
    """Poll the root until its log has grown and it is idle again."""
    with anyio.fail_after(10):
        while True:
            listed = await client.call("roots/list")
            row = next(one for one in listed["roots"] if one["rootId"] == root_id)
            if row["status"] == "idle" and row["events"] >= events:
                return row
            await anyio.sleep(0.01)


# ------------------------------------------------------------------ the gate --


async def test_a_root_keeps_running_after_its_client_disconnects(tmp_path: Path) -> None:
    """P5-01's gate. The client is not the thing doing the work.

    A prompt is queued, the client *disconnects entirely* — the socket closes,
    which is what a closed terminal looks like from here — and a second client
    connects to find the same root, still there, with the turn it was given.
    """
    async with running(tmp_path) as daemon:
        first = await daemon.client()
        await first.call("root/start", rootId="alpha")
        await first.call("root/prompt", rootId="alpha", prompt="keep going")
        await first.aclose()

        second = await daemon.client()
        row = await _settled(second, "alpha", events=1)

        assert row["rootId"] == "alpha"
        assert row["watchers"] == 0, "a disconnected client is still counted as watching"
        assert row["events"] > 0, "the root lost the turn its client queued"
        await second.notify("shutdown")


async def test_attaching_streams_events_and_detaching_stops_them(tmp_path: Path) -> None:
    """Attach and detach are symmetric, and neither touches the work.

    The root is prompted *after* the detach, so the second silence is evidence
    the subscription ended rather than that nothing happened.
    """
    async with running(tmp_path) as daemon:
        seen: list[str] = []
        client = await daemon.client(lambda method, params: seen.append(method))
        await client.call("root/start", rootId="beta")
        await client.call("root/attach", rootId="beta")
        await client.call("root/prompt", rootId="beta", prompt="first")
        await _settled(client, "beta", events=1)
        assert "session.event" in seen, "attaching did not stream anything"

        await client.call("root/detach", rootId="beta")
        seen.clear()
        await client.call("root/prompt", rootId="beta", prompt="second")
        await _settled(client, "beta", events=2)

        assert seen == [], "a detached client was still being sent events"
        await client.notify("shutdown")


async def test_a_reattaching_client_can_replay_what_it_missed(tmp_path: Path) -> None:
    """The work happened while nobody watched, and the log is what proves it.

    A count rather than a cursor — `{generation, sequence}` is P5-02's — but the
    property is the one that matters here: the root's events are the root's, not
    the connection's.
    """
    async with running(tmp_path) as daemon:
        starter = await daemon.client()
        await starter.call("root/start", rootId="gamma")
        await starter.call("root/prompt", rootId="gamma", prompt="unwatched")
        await _settled(starter, "gamma", events=1)
        await starter.aclose()

        seen: list[dict[str, Any]] = []
        watcher = await daemon.client(lambda method, params: seen.append(params))
        await watcher.call("root/attach", rootId="gamma", replay=50)
        with anyio.fail_after(5):
            while not seen:
                await anyio.sleep(0.01)

        assert all(one["rootId"] == "gamma" for one in seen)
        await watcher.notify("shutdown")


# ------------------------------------------------------------ many and one --


async def test_two_roots_are_two_deployments(tmp_path: Path) -> None:
    """One task each, one mounted profile each. Two roots are not two agents in
    one deployment: separate sessions, separate seams, separate everything a row
    provides — sharing a `Context` would make one root's registration visible to
    a root that never asked for it."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("root/prompt", rootId="one", prompt="a")
        await client.call("root/prompt", rootId="two", prompt="b")
        await _settled(client, "one", events=1)
        await _settled(client, "two", events=1)

        listed = await client.call("roots/list")

        assert {row["rootId"] for row in listed["roots"]} == {"one", "two"}
        assert {row["sessionId"] for row in listed["roots"]} == {"one", "two"}
        await client.notify("shutdown")


async def test_a_second_daemon_refuses_a_live_socket(tmp_path: Path) -> None:
    """Two supervisors both believing they own this user's roots is I-5's
    question, and taking the socket would answer it wrongly and silently. The
    refusal is this row's; the lease that arbitrates properly is P5-03."""
    async with running(tmp_path) as daemon:
        with pytest.raises(RuntimeError, match="already listening"):
            await serve(PROFILE, path=daemon.path)

        client = await daemon.client()
        await client.notify("shutdown")


async def test_a_stale_socket_is_cleared_rather_than_inherited(tmp_path: Path) -> None:
    """The ordinary aftermath of a crash. A path nobody answers makes every
    client hang on a connect that is never completed, so it is removed — the
    opposite of the live case, and distinguishable only by trying it."""
    stale = tmp_path / "daemon.sock"
    stale.write_bytes(b"")
    assert stale.exists()

    async with running(tmp_path) as daemon:
        client = await daemon.client()

        assert (await client.call("initialize"))["capabilities"]["attach"] is True

        await client.notify("shutdown")


# ------------------------------------------------------------------- resume --


async def test_a_restarted_daemon_continues_the_log_rather_than_appending_to_it(
    tmp_path: Path,
) -> None:
    """The defect this row shipped for one commit.

    `sessions.create` mints a *fresh* session and the JSONL store appends, so a
    daemon restarted with the same root id concatenated a second session onto
    the first: one file, one header, and `seq` restarting at zero halfway
    through — which breaks A1 and makes every fold over that file double-count.

    Resuming is also what lets P4-14's reconciliation fire for a daemon root:
    `session/created` is published for an adopted session too, so a root that
    died holding a worktree gets it reclaimed on the way back up.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("root/prompt", rootId="delta", prompt="first")
        first = await _settled(client, "delta", events=1)
        await client.notify("shutdown")

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("root/start", rootId="delta")
        second = await _settled(client, "delta", events=first["events"])

        assert second["events"] > first["events"], "the resumed root lost its history"
        await client.notify("shutdown")


async def test_a_resumed_root_says_so_in_its_own_log(tmp_path: Path) -> None:
    """The durable half of the notice. stderr is for whoever is watching; a
    cron-started agent has nobody watching, and "this picked up somebody else's
    work" is a fact about provenance that belongs in the trace."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("root/prompt", rootId="epsilon", prompt="first")
        await _settled(client, "epsilon", events=1)
        await client.notify("shutdown")

    seen: list[dict[str, Any]] = []
    async with running(tmp_path) as daemon:
        client = await daemon.client(lambda method, params: seen.append(params))
        await client.call("root/start", rootId="epsilon")
        await client.call("root/attach", rootId="epsilon", replay=200)
        with anyio.fail_after(5):
            while not any(one.get("event", {}).get("type") == "session/resumed" for one in seen):
                await anyio.sleep(0.01)
        await client.notify("shutdown")
