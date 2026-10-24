"""P13-01, P13-02, P13-08, P13-12 — what a daemon says about the work it holds.

A root that dies busy must be findable by the next daemon without reading every
log: the supervisor notes each root's holds on the live-roots index at every
transition that can move them, keeps a busy root's entry across its own shutdown,
and answers `sessions/owed` from the file — or from the logs when the file cannot
vouch for itself. A write that fails is loud: the root is listed as unwritten and
readiness is no until a later write succeeds. And the three orchestrator questions
are decided in one place and answered over HTTP with the status code a probe reads.
"""

from __future__ import annotations

import socket
from collections.abc import Mapping
from pathlib import Path

import pytest
from daemon_helpers import PROFILE, http, running, set_phase, stale, supervised, unplugged

from ph.keys import SESSIONS
from ph.paths import resolve_roots
from ph.seams.live_index import LiveEntry, LiveIndex
from ph.session import Session
from ph.testing import log_event
from ph_app.daemon import supervisor as supervisor_module
from ph_app.daemon.notify import notify, watchdog_interval
from ph_app.daemon.probes import parse_bind
from ph_app.daemon.server import DaemonUnavailable, serve
from ph_app.daemon.supervisor import Supervisor
from ph_app.payloads import OwedRoot

pytestmark = pytest.mark.anyio


def _index() -> LiveIndex:
    return LiveIndex(resolve_roots().home)


async def _settled(supervisor: Supervisor) -> None:
    writer = supervisor._live
    assert writer is not None
    await writer.settle()


# ------------------------------------------------------------------ the index --


async def test_a_busy_root_is_on_the_index_and_a_quiet_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("root")
        await _settled(supervisor)
        assert "root" not in _index().read(), "an idle root owes nothing"
        set_phase(root, "running")
        # A flush is the moment the index is owed news (`written` in `_start`).
        await supervisor._flush(root)
        await _settled(supervisor)
        entry = _index().read()["root"]
        assert entry.holds == ("turn",)
        assert entry.cwd == (root.session.header.cwd or "")
        set_phase(root, "idle")
        await supervisor._flush(root)
        await _settled(supervisor)
        assert "root" not in _index().read()


async def test_a_flush_that_moved_no_hold_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The writer remembers what it wrote, so re-stating an entry at every flush —
    which the host does — costs no thread hop, no read and no rewrite."""
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("root")
        set_phase(root, "running")
        await supervisor._flush(root)
        await _settled(supervisor)
        writes: list[object] = []
        real = LiveIndex.record_all

        def counted(index: LiveIndex, changes: Mapping[str, LiveEntry | None]) -> bool:
            writes.append(changes)
            return real(index, changes)

        monkeypatch.setattr(LiveIndex, "record_all", counted)
        for _ in range(3):
            await supervisor._flush(root)
        await _settled(supervisor)
        assert writes == []
        set_phase(root, "idle")


async def test_shutdown_leaves_a_busy_root_on_the_index_for_the_next_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An orderly stop mid-turn is still a root that owes work: the entry is the
    last word of this daemon, and the next one's reason to mount it."""
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("root")
        set_phase(root, "running")
        await supervisor.aclose()
    assert _index().read()["root"].holds == ("turn",)


async def test_owed_lists_what_the_index_holds_and_whether_it_is_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("root")
        set_phase(root, "running")
        await supervisor._flush(root)
        await _settled(supervisor)
        listing = await supervisor.owed()
        [row] = listing.sessions
        assert row.session_id == "root"
        assert row.holds == ["turn"]
        assert row.mounted and row.status == "running"
        # A file no rebuild has vouched for is rebuilt on the first ask, and a
        # mounted root's entry survives that: its log lags what the process knows.
        assert listing.rebuilt and listing.trusted
        again = await supervisor.owed()
        assert not again.rebuilt and again.sessions == listing.sessions
        set_phase(root, "idle")


async def test_owed_rebuilds_from_the_logs_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A log that ends inside a turn is a root that owes one, whatever the file says
    — and with no root mounted the rebuild reads the daemon's own sessions as JSONL."""
    async with supervised(tmp_path, monkeypatch) as first:
        root = await first.start("root")
        log_event(root.session, "turn/start", {"turn": 1})
        await first._flush(root)
    _index().path.unlink(missing_ok=True)
    async with supervised(tmp_path, monkeypatch) as second:
        listing = await second.owed(rebuild=True)
        assert listing.rebuilt
        [row] = listing.sessions
        assert row.session_id == "root"
        assert row.holds == ["turn"]
        assert not row.mounted and row.status == "stored"


# ---------------------------------------------------------- loud write failures --


async def test_an_unwritable_root_makes_readiness_no_until_a_write_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with unplugged(tmp_path, monkeypatch) as server:
        root = await server.supervisor.start("root")
        server.serving = True
        assert server.probe("ready").ok

        async def refused(_ctx: object, _session: Session) -> bool:
            return False

        monkeypatch.setattr(supervisor_module, "session_written", refused)
        await server.supervisor._flush(root)
        reply = server.probe("ready")
        assert not reply.ok and any("root" in reason for reason in reply.reasons)
        assert server.status().unwritten == ["root"]
        monkeypatch.undo()
        # Any write of the root clears it — here the store's own flush, which the
        # ladder's `_drive` calls directly — not only the doors that set the mark.
        await root.ctx.require(SESSIONS).flush(root.session)
        assert server.probe("ready").ok
        assert server.status().unwritten == []


# ------------------------------------------------------------------ the probes --


async def test_the_three_probes_are_decided_in_one_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with unplugged(tmp_path, monkeypatch) as server:
        assert not server.probe("startup").ok, "nothing is accepting yet"
        assert not server.probe("ready").ok
        assert server.probe("live").ok, "the loop is running even before the door opens"
        server.serving = True
        assert server.probe("startup").ok
        assert server.probe("ready").ok
        stale(server)
        reply = server.probe("live")
        assert not reply.ok and "event loop" in reply.reasons[0]
        server.stop.set()
        assert all(not server.probe(name).ok for name in ("startup", "ready", "live"))


def test_parse_bind_reads_the_spellings_a_unit_file_uses() -> None:
    assert parse_bind("127.0.0.1:9000") == ("127.0.0.1", 9000)
    assert parse_bind(":9000") == ("127.0.0.1", 9000)
    assert parse_bind("9000") == ("127.0.0.1", 9000)
    assert parse_bind("[::1]:9000") == ("::1", 9000)
    with pytest.raises(ValueError, match="host:port"):
        parse_bind("nine")


async def test_the_probes_answer_over_tcp_with_the_code_an_orchestrator_reads(
    tmp_path: Path,
) -> None:
    async with running(tmp_path, probe="127.0.0.1:0") as daemon:
        address = daemon.running.probe_address
        assert address.startswith("127.0.0.1:")
        assert daemon.running.status().probe == address
        code, body = await http(address, "GET /ready HTTP/1.1\r\nHost: probe\r\n\r\n")
        assert code == 200 and body["ok"] is True and body["name"] == "ready"
        code, body = await http(address, "GET /status HTTP/1.1\r\n\r\n")
        assert code == 200 and body["probe"] == address
        code, body = await http(address, "GET /nope HTTP/1.1\r\n\r\n")
        assert code == 404 and body["probes"] == ["startup", "ready", "live"]
        code, _body = await http(address, "POST /live HTTP/1.1\r\n\r\n")
        assert code == 405
        daemon.running.supervisor.unwritten.add("ghost")
        code, body = await http(address, "GET /ready HTTP/1.1\r\n\r\n")
        assert code == 503 and "ghost" in body["reasons"][0]


async def test_a_probe_bind_that_is_not_an_address_is_the_daemons_refusal(
    tmp_path: Path,
) -> None:
    """Named like a socket the kernel will not bind, and before anything is built."""
    socket_path = tmp_path / "daemon.sock"
    with pytest.raises(DaemonUnavailable, match="probes"):
        await serve(PROFILE, path=socket_path, probe="127.0.0.1:nine")
    assert not socket_path.exists(), "the unix socket was let go with the refusal"


# ------------------------------------------------------------------ sd_notify --


def test_notify_speaks_to_the_socket_systemd_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "notify.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as listener:
        listener.bind(str(path))
        listener.settimeout(2)
        monkeypatch.setenv("NOTIFY_SOCKET", str(path))
        assert notify("READY=1")
        assert listener.recv(64) == b"READY=1"
    monkeypatch.delenv("NOTIFY_SOCKET")
    assert not notify("READY=1"), "nothing to tell is not a failure"


def test_the_watchdog_interval_is_half_of_what_systemd_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert watchdog_interval() is None
    monkeypatch.setenv("WATCHDOG_USEC", "10000000")
    assert watchdog_interval() == 5.0
    monkeypatch.setenv("WATCHDOG_PID", "1")
    assert watchdog_interval() is None, "another process was asked to feed it"


def test_an_owed_row_carries_the_entry_it_was_built_from() -> None:
    """`OwedRoot.of` is the one place an index entry reaches the wire."""
    entry = LiveEntry(
        "root",
        ("turn", "goal"),
        updated=7,
        family="f",
        cwd="/w",
        profile="work",
        failed=True,
        damaged=("root-bad",),
    )
    row = OwedRoot.of(entry, mounted=False, status="stored")
    assert row.session_id == "root"
    assert row.holds == ["turn", "goal"]
    assert (row.updated, row.cwd, row.profile, row.failed) == (7, "/w", "work", True)
    assert row.damaged == ["root-bad"]
    assert (row.mounted, row.status) == (False, "stored")
