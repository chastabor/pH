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
from ph_app.daemon import DaemonClient, serve, server

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


async def _history(
    client: DaemonClient, session_id: str, cursor: Any = None
) -> list[dict[str, Any]]:
    """Everything from `cursor` to now, paged the way a client must page it.

    Catch-up has one mechanism — `session/snapshot` — because `session/attach`
    deliberately does not replay: streaming a gap of unknown size into a bounded
    outbox is how a reattach fails at exactly the moment it matters.
    """
    collected: list[dict[str, Any]] = []
    page = await client.call("session/snapshot", sessionId=session_id, cursor=cursor)
    collected.extend(page["events"])
    while page["more"]:
        page = await client.call("session/snapshot", sessionId=session_id, cursor=page["cursor"])
        collected.extend(page["events"])
    return collected


async def _settled(client: DaemonClient, root_id: str, *, events: int) -> dict[str, Any]:
    """Poll the root until its log has grown and it is idle again."""
    with anyio.fail_after(10):
        while True:
            listed = await client.call("sessions/list")
            row = next(one for one in listed["sessions"] if one["sessionId"] == root_id)
            if row["status"] == "idle" and row["cursor"]["sequence"] >= events:
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
        await first.call("session/new", sessionId="alpha")
        await first.call("session/prompt", sessionId="alpha", prompt="keep going")
        await first.aclose()

        second = await daemon.client()
        row = await _settled(second, "alpha", events=1)

        assert row["sessionId"] == "alpha"
        assert row["watchers"] == 0, "a disconnected client is still counted as watching"
        assert row["cursor"]["sequence"] > 0, "the root lost the turn its client queued"
        await second.notify("shutdown")


async def test_attaching_streams_events_and_detaching_stops_them(tmp_path: Path) -> None:
    """Attach and detach are symmetric, and neither touches the work.

    The root is prompted *after* the detach, so the second silence is evidence
    the subscription ended rather than that nothing happened.
    """
    async with running(tmp_path) as daemon:
        seen: list[str] = []
        client = await daemon.client(lambda method, params: seen.append(method))
        await client.call("session/new", sessionId="beta")
        await client.call("session/attach", sessionId="beta")
        await client.call("session/prompt", sessionId="beta", prompt="first")
        await _settled(client, "beta", events=1)
        assert "session.event" in seen, "attaching did not stream anything"

        await client.call("session/detach", sessionId="beta")
        seen.clear()
        await client.call("session/prompt", sessionId="beta", prompt="second")
        await _settled(client, "beta", events=2)

        assert seen == [], "a detached client was still being sent events"
        await client.notify("shutdown")


async def test_a_reattaching_client_reads_what_it_missed(tmp_path: Path) -> None:
    """The work happened while nobody watched, and the log is what proves it.

    Through `session/snapshot`, which is the only catch-up path: the root's
    events are the root's, not the connection's, and a client that was away
    reads them back at its own pace.
    """
    async with running(tmp_path) as daemon:
        starter = await daemon.client()
        await starter.call("session/new", sessionId="gamma")
        await starter.call("session/prompt", sessionId="gamma", prompt="unwatched")
        await _settled(starter, "gamma", events=1)
        await starter.aclose()

        watcher = await daemon.client()
        history = await _history(watcher, "gamma")

        assert history, "the session's history was not readable"
        assert [one["seq"] for one in history] == sorted(one["seq"] for one in history)
        await watcher.notify("shutdown")


# ------------------------------------------------------------ many and one --


async def test_two_roots_are_two_deployments(tmp_path: Path) -> None:
    """One task each, one mounted profile each. Two roots are not two agents in
    one deployment: separate sessions, separate seams, separate everything a row
    provides — sharing a `Context` would make one root's registration visible to
    a root that never asked for it."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="one", prompt="a")
        await client.call("session/prompt", sessionId="two", prompt="b")
        await _settled(client, "one", events=1)
        await _settled(client, "two", events=1)

        listed = await client.call("sessions/list")

        assert {row["sessionId"] for row in listed["sessions"]} == {"one", "two"}
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
        await client.call("session/prompt", sessionId="delta", prompt="first")
        first = await _settled(client, "delta", events=1)
        await client.notify("shutdown")

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="delta")
        second = await _settled(client, "delta", events=first["cursor"]["sequence"])

        assert second["cursor"]["sequence"] > first["cursor"]["sequence"], (
            "the resumed root lost its history"
        )
        await client.notify("shutdown")


async def test_a_resumed_root_says_so_in_its_own_log(tmp_path: Path) -> None:
    """The durable half of the notice. stderr is for whoever is watching; a
    cron-started agent has nobody watching, and "this picked up somebody else's
    work" is a fact about provenance that belongs in the trace."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="epsilon", prompt="first")
        await _settled(client, "epsilon", events=1)
        await client.notify("shutdown")

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="epsilon")

        history = await _history(client, "epsilon")

        assert any(one["type"] == "session/resumed" for one in history)
        await client.notify("shutdown")


# --------------------------------------------------------------- P5-02 gate --


async def test_a_cursor_resumes_reading_exactly_where_it_stopped(tmp_path: Path) -> None:
    """Half the gate: *reattach preserves streaming position*.

    The client stores the cursor from what it has read, work happens while it is
    away, and it comes back to be handed exactly the gap — not one event more,
    and not the whole log again. A count would have made the client guess; the
    cursor is what it can actually prove about what it has seen.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="zeta", prompt="first")
        first = await _settled(client, "zeta", events=1)
        cursor = first["cursor"]

        await client.call("session/prompt", sessionId="zeta", prompt="second")
        second = await _settled(client, "zeta", events=first["cursor"]["sequence"] + 1)

        missed = await _history(client, "zeta", cursor)

        assert missed, "the cursor read back nothing"
        assert missed[0]["seq"] == cursor["sequence"], "the gap did not start at the cursor"
        assert len(missed) == second["cursor"]["sequence"] - cursor["sequence"]
        await client.notify("shutdown")


async def test_a_cursor_from_another_log_reads_from_the_start(tmp_path: Path) -> None:
    """A sequence only means something against the log that counted it.

    Refusing would strand a client that did nothing wrong; honouring it would
    skip events it never saw. So a stale generation reads as "you have seen
    nothing of *this* log" — and `session/attach` says where it actually
    resumed, so the client is not left inferring it from sequence numbers.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="eta", prompt="first")
        settled = await _settled(client, "eta", events=1)
        stale = {"generation": "not-this-log", "sequence": settled["cursor"]["sequence"]}

        history = await _history(client, "eta", stale)
        attached = await client.call("session/attach", sessionId="eta", cursor=stale)

        assert len(history) == settled["cursor"]["sequence"], "a stale cursor skipped events"
        assert attached["from"] == 0, "attach did not say it was resuming from the start"
        await client.notify("shutdown")


async def test_the_same_command_twice_runs_once(tmp_path: Path) -> None:
    """The other half: *duplicate command is idempotent*.

    A reconnecting client cannot know whether its last `session/prompt` landed,
    so it sends it again. Answering "yes, that one" is what makes asking twice
    safe — and the record is in the log, so it survives the restart that caused
    the retry.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call(
            "session/prompt", sessionId="theta", prompt="once", clientId="c1", commandId="k1"
        )
        settled = await _settled(client, "theta", events=1)

        await client.call(
            "session/prompt", sessionId="theta", prompt="once", clientId="c1", commandId="k1"
        )
        await anyio.sleep(0.05)
        again = await _settled(client, "theta", events=settled["cursor"]["sequence"])

        assert again["cursor"]["sequence"] == settled["cursor"]["sequence"], (
            "the duplicate ran a second turn"
        )
        await client.notify("shutdown")


async def test_a_snapshot_is_paged_rather_than_one_huge_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed root can hold hundreds of thousands of events, and one reply
    carrying all of them would trip the transport's own frame bound.

    The page size is patched down because the first version of this test looped
    on `page["more"]` over a sixteen-event session — `more` was always `False`,
    the loop body never ran, and the bound it claimed to exercise was never
    reached by any test in the suite.
    """
    monkeypatch.setattr(server, "SNAPSHOT_EVENTS", 4)
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("iota", "first")
        settled = await _settled(client, "iota", events=5)

        pages = 0
        collected: list[dict[str, Any]] = []
        page = await client.call("session/snapshot", sessionId="iota")
        while True:
            pages += 1
            collected.extend(page["events"])
            if not page["more"]:
                break
            page = await client.call("session/snapshot", sessionId="iota", cursor=page["cursor"])

        assert pages > 1, "the page bound was never reached"
        assert all(len(one) <= 4 for one in [page["events"]]), "a page exceeded the bound"
        assert len(collected) == settled["cursor"]["sequence"], "paging lost or duplicated events"
        assert [one["seq"] for one in collected] == sorted(one["seq"] for one in collected)
        await client.notify("shutdown")


async def test_the_client_makes_its_own_retries_safe(tmp_path: Path) -> None:
    """Idempotence as a property of the protocol rather than of a caller's
    discipline: `DaemonClient.prompt` mints the ids, so the same call twice is
    two commands and a *replayed* call is one."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        first = await client.prompt("kappa", "once")
        settled = await _settled(client, "kappa", events=1)

        # The retry a reconnect forces: the same command id, sent again.
        await client.call(
            "session/prompt",
            sessionId="kappa",
            prompt="once",
            clientId=client.id,
            commandId="1",
        )
        await anyio.sleep(0.05)
        again = await _settled(client, "kappa", events=settled["cursor"]["sequence"])

        assert first["sessionId"] == "kappa"
        assert again["cursor"]["sequence"] == settled["cursor"]["sequence"], (
            "the replayed command ran a second turn"
        )
        await client.notify("shutdown")
