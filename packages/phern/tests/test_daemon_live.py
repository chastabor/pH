"""P13-01, P13-02, P13-08, P13-12 — what a daemon says about the work it holds.

A root that dies busy must be findable by the next daemon without reading every
log: the supervisor notes each root's holds on the live-roots index at every
transition that can move them, keeps a busy root's entry across its own shutdown,
and answers `sessions/owed` from the file — or from the logs when the file cannot
vouch for itself. A write that fails, of the root's log or a child's, is loud: the
root is listed as unwritten and readiness is no until a later write of that log
succeeds. And the three orchestrator questions
are decided in one place and answered over HTTP with the status code a probe reads.
"""

from __future__ import annotations

import socket
from collections.abc import Mapping
from pathlib import Path

import anyio
import pytest
from daemon_helpers import (
    PROFILE,
    hold_the_provider,
    http,
    logged_types,
    running,
    set_phase,
    spawned,
    stale,
    supervised,
    unplugged,
    until,
)

from ph.keys import SESSIONS
from ph.paths import resolve_roots
from ph.persistence import JsonlSessionStore
from ph.seams.live_index import LiveEntry, LiveIndex
from ph.session import now_ms
from ph.testing import log_event, raising, stored_types
from ph_app.daemon.notify import notify, watchdog_interval
from ph_app.daemon.probes import parse_bind
from ph_app.daemon.server import DaemonUnavailable, serve
from ph_app.daemon.supervisor import Supervisor
from ph_app.payloads import OwedRoot

pytestmark = pytest.mark.anyio


def _index() -> LiveIndex:
    return LiveIndex(resolve_roots().home)


def _writes(monkeypatch: pytest.MonkeyPatch) -> list[set[str]]:
    """Every batch the index is handed from here on, by the roots in it."""
    batches: list[set[str]] = []
    record_all = LiveIndex.record_all

    def counted(index: LiveIndex, changes: Mapping[str, LiveEntry | None]) -> bool:
        batches.append(set(changes))
        return record_all(index, changes)

    monkeypatch.setattr(LiveIndex, "record_all", counted)
    return batches


def _said(caplog: pytest.LogCaptureFixture) -> list[str]:
    """What the daemon's own log said."""
    return [one.getMessage() for one in caplog.records if one.name == "ph_app.daemon"]


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
        # A hold moves with a record, as a turn's start and end are, and the flush
        # that writes it is the moment the index is owed news (`written` in
        # `_start`). A flush that writes nothing announces nothing.
        set_phase(root, "running")
        log_event(root.session, "turn/start", {"turn": 1})
        await supervisor._flush(root)
        await _settled(supervisor)
        entry = _index().read()["root"]
        assert entry.holds == ("turn",)
        assert entry.cwd == (root.session.header.cwd or "")
        set_phase(root, "idle")
        log_event(root.session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
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
        writes = _writes(monkeypatch)
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


async def test_a_daemon_cancelled_rather_than_stopped_leaves_its_roots_as_a_stop_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` cancelled — an embedding host's teardown, or this suite's — is the stop
    a `shutdown` frame asks for: the turn is left cut for the next daemon's resumer,
    and the index names the root, exactly as after an orderly stop.

    Sabotage: enter `serve`'s task group without `_cancel_is_a_stop`, and the turn
    ends `aborted` with the root off the index.
    """
    hold_the_provider(monkeypatch)
    async with running(tmp_path) as daemon:
        root = await daemon.root("root")
        await daemon.running.supervisor.prompt(root.id, "go")
        await until(
            lambda: "turn/start" in stored_types(root.ctx, root.id),
            what="the turn's first barrier",
        )
    assert "turn/end" not in logged_types(root.id), "nobody ended the turn"
    assert _index().read()["root"].holds == ("turn",)


async def test_a_cancelled_serve_still_ends_cancelled(tmp_path: Path) -> None:
    """The stop a cancel asks for is still a cancel to whoever made it: `serve` runs
    its teardown and then raises, rather than returning a value to a caller that
    abandoned the call."""
    ended: list[str] = []
    ready = anyio.Event()

    async def served() -> None:
        try:
            await serve(PROFILE, path=tmp_path / "daemon.sock", ready=ready, passivate_after=None)
            ended.append("returned")
        except anyio.get_cancelled_exc_class():
            ended.append("cancelled")
            raise

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(served)
        await ready.wait()
        tasks.cancel_scope.cancel()
    assert ended == ["cancelled"]


async def test_shutdown_tells_the_index_about_every_root_in_one_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every root's last word is noted before any is released, so they land in one
    write of the file rather than one per root, each a lock, a read and a rewrite.

    Sabotage: note each root just before its own release, and the first write holds
    only the first root.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        for name in ("first", "second", "third"):
            set_phase(await supervisor.start(name), "running")
        await _settled(supervisor)
        batches = _writes(monkeypatch)
        await supervisor.aclose()
    assert batches[0] == {"first", "second", "third"}, batches
    assert {name: entry.holds for name, entry in _index().read().items()} == dict.fromkeys(
        ("first", "second", "third"), ("turn",)
    )


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Any write of the root that fails marks it, through a door the daemon does not
    own as much as its own: the store says so (`session/unwritten`), and the daemon's
    log says why. Here nothing was owed since the last write, so the mark is cleared
    by an announcement the next flush makes only because the one before it failed.

    Sabotage: drop `session/unwritten` from `SessionStore.flush`, and the root is
    never marked; forget the failure there, and the mark stays.
    """
    async with unplugged(tmp_path, monkeypatch) as server:
        root = await server.supervisor.start("root")
        server.serving = True
        assert server.probe("ready").ok

        refusing = root.ctx.on("session/flush", raising(OSError("no space left on device")))
        # The store's own door, as a barrier before a model request goes through.
        assert not await root.ctx.require(SESSIONS).written(root.session)
        reply = server.probe("ready")
        assert not reply.ok and any("root" in reason for reason in reply.reasons)
        assert server.status().unwritten == ["root"]
        said = _said(caplog)
        assert any("no space left on device" in line for line in said), "the log says why"
        refusing()
        # Any write of the root clears it — here the store's own flush, which the
        # ladder's `_drive` calls directly — not only the doors that set the mark.
        await root.ctx.require(SESSIONS).flush(root.session)
        assert server.probe("ready").ok
        assert server.status().unwritten == []


async def test_a_childs_unwritten_log_counts_against_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A root resumes into whatever its family lost, so a child's log that could not
    be written marks the root as its own would, and the daemon's log names the child.
    The mark is the child's: a write of the root leaves it, and a write of the child
    clears it.

    Sabotage: hear only the root's own log, and the root is never marked. Clear a
    root on any write of it, and the root's own write clears the child's mark.
    """
    async with unplugged(tmp_path, monkeypatch) as server:
        root = await server.supervisor.start("root")
        server.serving = True
        sessions = root.ctx.require(SESSIONS)
        child = spawned(root, "r1")
        assert server.probe("ready").ok

        refusing = root.ctx.on("session/flush", raising(OSError("no space left on device")))
        assert not await sessions.written(child)
        assert server.status().unwritten == ["root"]
        said = _said(caplog)
        assert any(child.id in line and "no space left" in line for line in said), said
        refusing()
        # Something to write, so the root's flush is announced.
        log_event(root.session, "turn/start", {"turn": 1})
        await sessions.flush(root.session)
        assert not server.probe("ready").ok, "the root's own write left the child's mark"
        await sessions.flush(child)
        assert server.probe("ready").ok
        assert server.status().unwritten == []


async def test_what_the_last_write_could_not_write_counts_against_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The mount's last write runs after every listener has gone, so the store cannot
    tell the daemon then. What it could not write is read once the root is down, and
    counts as any failed write does: readiness is no after a passivation (P13-08).
    Only the last write fails here; the host's flush and the checkpoint barrier go
    through the listener the store registered before the patch.

    Sabotage: drop the read from `Supervisor._release`, and the root is never marked.
    """
    async with unplugged(tmp_path, monkeypatch) as server:
        root = await server.supervisor.start("root")
        server.serving = True
        monkeypatch.setattr(JsonlSessionStore, "flush", raising(OSError("no space left on device")))
        await server.supervisor.passivate(root, now=now_ms())

        assert server.supervisor.unwritten == {"root": "root"}
        assert not server.probe("ready").ok
        said = _said(caplog)
        assert any("session root of root root" in line for line in said), said


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
        daemon.running.supervisor.unwritten["ghost"] = "ghost"
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
