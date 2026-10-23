"""P13-13 — `phern service`, the client an orchestrator drives.

Every command answers in one vocabulary — exit 0 ready, 1 down, 2 degraded — so
the gates here are about that vocabulary holding against a real daemon: a socket
with nothing behind it is down and `stop` has nothing to do; a root the daemon
could not write, a root parked on a person, and a stored root that owes work are
each degraded for a reason the output names; `start` spawns a real `phern daemon`
with the options it was handed, waits for ready, resumes and reports, and a refused
spawn is down; `watch --once` is one pass of the loop, and the loop's waits climb
the scheduler's own ladder.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from daemon_helpers import left_mid_turn, private_runtime, route_needs, run_phern, serving

from ph.paths import PathRoots, resolve_roots
from ph_app import service
from ph_app.cli import app
from ph_app.daemon.launch import DaemonAbsent
from ph_app.daemon.recovery import WAKE_RETRY_DELAYS
from ph_app.payloads import OwedListing, OwedRoot, ProbeReply
from ph_app.service import EXIT, Observed

pytestmark = pytest.mark.anyio


def _observed(*rows: OwedRoot, reasons: tuple[str, ...] = ()) -> Observed:
    ready = ProbeReply(name="ready", ok=not reasons, reasons=list(reasons), checked_at=1)
    owed = OwedListing(sessions=list(rows), trusted=True, rebuilt=False)
    return Observed.of(pid=1, uptime_ms=1000, ready=ready, owed=owed)


def _row(
    session_id: str, *, mounted: bool, status: str, failed: bool = False, held: bool = False
) -> OwedRoot:
    return OwedRoot(
        session_id=session_id,
        holds=["turn"],
        updated=1,
        mounted=mounted,
        status=status,
        failed=failed,
        held=held,
    )


# ----------------------------------------------------------------- the verdict --


def test_the_verdict_names_each_way_a_daemon_is_degraded() -> None:
    """One fold, from the readiness answer and the owed listing, into the sentences
    an operator reads: a reason the probe gave, a root whose ladder is spent, one the
    daemon says is held on a person, one that owes work and is not mounted. Nothing
    owed and a yes from the probe is ready, and so is a mounted root at work."""
    assert _observed().verdict == "ready"
    unwritten = _observed(reasons=("root(s) could not be written: a",))
    assert unwritten.degraded == ("root(s) could not be written: a",)
    spent = _observed(_row("a", mounted=False, status="stored", failed=True))
    assert "gave up" in spent.degraded[0]
    assert spent.unmounted == (), "a spent root is not one a pass without force would mount"
    held = _observed(_row("a", mounted=True, status="needs-credential", held=True))
    assert "held on a person: a (needs-credential)" in held.degraded[0]
    owed = _observed(_row("a", mounted=False, status="stored"))
    assert owed.unmounted == ("a",) and "not mounted: a" in owed.degraded[0]
    busy = _observed(_row("a", mounted=True, status="running"))
    assert busy.verdict == "ready"
    # The contract an orchestrator branches on, stated rather than read back.
    assert EXIT == {"ready": 0, "down": 1, "degraded": 2}


def test_the_watchers_waits_climb_the_schedulers_ladder() -> None:
    """Ready waits the first rung and resets; anything else climbs one rung per pass
    and stays on the last — the scheduler's own backoff for a wake that could not
    mount, because the shape of the wait is the same."""
    assert service._delay("ready", 3) == (WAKE_RETRY_DELAYS[0], 0)
    delay, rung = service._delay("down", 0)
    assert (delay, rung) == (WAKE_RETRY_DELAYS[0], 1)
    delays = []
    for _ in range(len(WAKE_RETRY_DELAYS) + 2):
        delay, rung = service._delay("degraded", rung)
        delays.append(delay)
    assert delays == [*WAKE_RETRY_DELAYS[1:], *[WAKE_RETRY_DELAYS[-1]] * 3]


def test_every_service_command_is_registered() -> None:
    """The six names, held against the app rather than against a docstring — the
    guard `test_agents_cli.py` keeps for its group, for this one."""
    names = {
        command.name or (command.callback.__name__ if command.callback else "<no callback>")
        for command in service.service_app.registered_commands
    }
    assert names == {"start", "stop", "restart", "status", "resume", "watch"}
    assert "service" in {group.name for group in app.registered_groups}


# -------------------------------------------------------------------- down --


async def test_no_daemon_is_down_and_nothing_to_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Down is a state, not a refusal: `status` and `resume` say so and exit 1,
    and `stop` finds the state it was asked for and exits 0."""
    private_runtime(tmp_path, monkeypatch)
    for command in (["service", "status"], ["service", "resume"]):
        result = await run_phern(*command)
        assert result.exit_code == 1, f"{command}: {result.output}"
        assert "no daemon socket" in result.output
    stopped = await run_phern("service", "stop")
    assert stopped.exit_code == 0, stopped.output
    assert "nothing to stop" in stopped.output


# ------------------------------------------------------------------- status --


async def test_status_is_ready_with_nothing_owed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with serving(tmp_path, monkeypatch) as daemon:
        result = await run_phern("service", "status")
        assert result.exit_code == 0, result.output
        assert "ready" in result.output
        assert f"pid {daemon.running.status().pid}" in result.output
        assert "no root owes work" in result.output, "the owed table, even when empty"


async def test_a_root_the_daemon_could_not_write_is_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Readiness is the probe's answer, and its reason is printed as the probe gave
    it (P13-08): the one state an orchestrator reads off a 503 is the one a person
    reads off `status`."""
    async with serving(tmp_path, monkeypatch) as daemon:
        daemon.running.supervisor.unwritten.add("torn")
        result = await run_phern("service", "status")
        assert result.exit_code == 2, result.output
        assert "degraded" in result.output
        assert "could not be written: torn" in result.output


async def test_a_stored_root_that_owes_work_is_degraded_until_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log a daemon left mid-turn is work nobody is doing. `status` says so and
    exits 2; `resume` mounts it, prints the pass, and the look after it is ready."""
    await left_mid_turn(tmp_path, monkeypatch, "left")
    async with serving(tmp_path, monkeypatch) as daemon:
        before = await run_phern("service", "status")
        assert before.exit_code == 2, before.output
        assert "not mounted: left" in before.output
        assert "phern service resume" in before.output, "the fix, named"

        resumed = await run_phern("service", "resume")
        assert resumed.exit_code == 0, resumed.output
        assert "left" in resumed.output and "resumed" in resumed.output
        assert daemon.holds("left")

        after = await run_phern("service", "status")
        assert after.exit_code == 0, after.output


async def test_a_root_parked_on_a_person_is_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mounted root waiting on a credential, with a prompt waiting in its inbox
    for it, is not down and not ready: nothing this daemon can do moves it, and the
    verdict says whose turn it is (T5). The daemon says so on the row (`held`), from
    `Root.held`, rather than leaving the client to read it off `status`."""
    route_needs(monkeypatch, "PH_SERVICE_KEY")
    async with serving(tmp_path, monkeypatch) as daemon:
        root = await daemon.root("keyed")
        assert root.status == "needs-credential"
        await daemon.running.supervisor.prompt("keyed", "go")
        await daemon.running.supervisor._flush(root)
        [row] = (await daemon.running.supervisor.owed()).sessions
        assert row.held and row.mounted
        result = await run_phern("service", "status")
        assert result.exit_code == 2, result.output
        assert "held on a person: keyed (needs-credential)" in result.output


# --------------------------------------------------------- start, stop, watch --


async def test_start_spawns_a_daemon_with_the_options_it_was_handed_and_stop_ends_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point, against a real process: the options after `start` reach
    `phern daemon` as typed, the command returns once the daemon is ready and has
    been asked to resume, a second `start` joins rather than spawns, and `stop`
    waits for the daemon to go so the next `start` finds a clear door."""
    private_runtime(tmp_path, monkeypatch)
    socket = resolve_roots().daemon_socket()
    started = await run_phern(
        "service", "start", "--profile", "headless", "--passivate-after", "off"
    )
    try:
        assert started.exit_code == 0, started.output
        assert f"started a daemon on {socket}" in started.output
        assert "ready" in started.output
        again = await run_phern("service", "start", "--profile", "headless")
        assert again.exit_code == 0, again.output
        assert "already listening" in again.output
    finally:
        stopped = await run_phern("service", "stop")
    assert stopped.exit_code == 0, stopped.output
    assert "daemon stopped" in stopped.output
    assert not socket.exists()


async def test_a_refused_spawn_is_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The launcher's refusal — here, a daemon that exited before listening, which
    `test_daemon_launch.py` runs for real — is the verdict `down`, said as the
    launcher said it."""
    private_runtime(tmp_path, monkeypatch)

    async def refused(**_: object) -> None:
        raise DaemonAbsent("the daemon exited with status 2 before listening")

    monkeypatch.setattr(service, "ensure_daemon", refused)
    result = await run_phern("service", "start", "--no-such-option")
    assert result.exit_code == EXIT["down"], result.output
    assert "exited with status 2 before listening" in result.output


async def test_watch_once_starts_a_daemon_that_is_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pass of the loop, for a cron slot: down is acted on, not reported, and
    the options after `watch` are what the start is handed. The start itself is the
    test above's."""
    private_runtime(tmp_path, monkeypatch)
    asked: list[Sequence[str]] = []

    async def start(_roots: PathRoots, options: Sequence[str]) -> service.Verdict:
        asked.append(options)
        return "ready"

    monkeypatch.setattr(service, "_start", start)
    result = await run_phern("service", "watch", "--once", "--profile", "headless")
    assert result.exit_code == 0, result.output
    assert "down:" in result.output and "starting one" in result.output
    assert asked == [["--profile", "headless"]]


async def test_watch_once_resumes_what_is_owed_and_reports_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Against a daemon that is up: a stored root that owes work is resumed in the
    same pass, and a root the daemon could not write is reported, with the exit code
    an orchestrator branches on."""
    await left_mid_turn(tmp_path, monkeypatch, "left")
    async with serving(tmp_path, monkeypatch) as daemon:
        passed = await run_phern("service", "watch", "--once")
        assert passed.exit_code == 0, passed.output
        assert "owed: left; resuming" in passed.output
        assert daemon.holds("left")

        daemon.running.supervisor.unwritten.add("torn")
        degraded = await run_phern("service", "watch", "--once")
        assert degraded.exit_code == 2, degraded.output
        assert "could not be written: torn" in degraded.output
