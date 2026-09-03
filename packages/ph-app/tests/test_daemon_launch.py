"""P7-08 — a UI gets itself a daemon, and two UIs get one between them.

The harness lives in the daemon, so an interactive front end needs one to exist.
Making a person start it first would put a second command in front of the
ordinary case, and it would be the kind nobody remembers until the first one
fails — so a UI starts one when the socket is absent, ephemeral, and says
nothing about it.

Two claims carry the file, and they pull in opposite directions.

**Absent means start one; unresponsive means do not.** A path nothing answers on
is the aftermath of a crash — or of a logout that reaped the door out from under
a process still holding every session lease it took (P5-11) — and a second
supervisor there is exactly the wrong move. `ensure_daemon` stops at "something
is listening" and leaves that diagnosis to the connect behind it, which already
has the paragraph.

**Two UIs opening together start one daemon.** Without the lock both find no
socket, both spawn, and the loser's `serve` refuses a live socket — so the person
who opened two terminals gets a traceback in the second. The re-check *inside*
the lock is what turns the loser into a client.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import running, shut_down

from ph_app.cli import spawn_command
from ph_app.daemon.launch import DaemonAbsent, ensure_daemon

pytestmark = pytest.mark.anyio

ARGV = spawn_command(profile="headless", provider="fake", model="fake-1")
"""What a UI would spawn. `ensure_daemon` takes the argv and knows nothing else."""


@pytest.fixture(autouse=True)
def _runtime(tmp_path: Path, monkeypatch: Any) -> None:
    """A `$PH_RUNTIME` of this test's own, so the socket path is ours.

    `ensure_daemon` resolves the socket rather than taking one, deliberately — a
    UI that could be pointed at a socket of its own construction is a UI that can
    disagree with `ph agents` about where the daemon is.
    """
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "run"))


# ------------------------------------------------------------------- argv --


def test_the_spawned_daemon_is_this_python_and_is_ephemeral() -> None:
    """Two properties of the command, and both are correctness rather than taste.

    `sys.executable -m ph_app` because a bare `ph` resolves through `PATH` and may
    find a different install, an older version, or nothing at all — a checkout
    with no `pip install -e`, a virtualenv the shell has not activated. A daemon
    composed by a different pH than its client is one whose profile, event
    vocabulary and wire version nobody chose.

    `--ephemeral` because this daemon was nobody's decision. Sabotage: drop it,
    and every TUI a person ever opens leaves a supervisor behind.
    """
    argv = spawn_command(profile="tui", provider="fake", model="fake-1")

    assert argv[:3] == [sys.executable, "-m", "ph_app"]
    assert argv[3] == "daemon"
    assert "--ephemeral" in argv
    assert argv[argv.index("--profile") + 1] == "tui"


# ------------------------------------------------------- when one is there --


async def test_a_daemon_that_is_already_listening_is_left_alone(tmp_path: Path) -> None:
    """The common case, and it must stay cheap: a second terminal, a browser tab.

    Sabotage: spawn unconditionally, and the second UI's daemon refuses the first
    one's socket — which is a crash in the UI, not a message.
    """
    async with running(tmp_path, path=Path(tmp_path / "run" / "daemon.sock")) as daemon:
        started = await ensure_daemon(argv=ARGV)

        assert started.path == daemon.path
        assert not started.spawned


async def test_no_spawn_refuses_and_names_the_command() -> None:
    """`--no-spawn` is a posture, not a failure.

    For a deployment running `ph daemon` under an init system, where a UI-started
    daemon would be a second supervisor competing for the same session leases.
    The refusal names what to type, because a person who chose this flag is
    somebody who wants to run the command themselves.
    """
    with pytest.raises(DaemonAbsent) as raised:
        await ensure_daemon(argv=ARGV, spawn=False)

    assert "ph daemon" in str(raised.value)


# ------------------------------------------------------------ the spawning --


async def test_a_ui_with_no_daemon_starts_one_and_waits_for_the_door() -> None:
    """The whole point, against a real process on a real socket.

    Waits on the *connect* rather than on the file, which is the difference this
    module is built on: the path exists before `serve()` is listening, and a
    `path.exists()` poll returns during exactly that window. A UI that believed
    it would hand its user a connect error on a daemon that was starting fine.

    Torn down through the socket, because that is the only handle a spawned
    daemon leaves — which is itself the thing being asserted.
    """
    started = await ensure_daemon(argv=ARGV)

    assert started.spawned
    assert started.path.exists()

    await shut_down(started.path)


async def test_two_uis_starting_at_once_start_one_daemon() -> None:
    """The race, run for real — two `ensure_daemon` calls with no ordering.

    Exactly one may report `spawned`. Both must come back pointing at the same
    socket, and the daemon behind it must be alive: the failure the lock prevents
    is not "two daemons" — `_clear_stale` refuses the second — it is the *loser
    crashing*, so a person who opened two terminals loses the second one.

    Sabotage: drop the lock, or move the re-check outside it, and this either
    reports two spawns or raises out of one of the two tasks.
    """
    outcomes: list[Any] = []

    async with anyio.create_task_group() as tasks:

        async def launch() -> None:
            outcomes.append(await ensure_daemon(argv=ARGV))

        tasks.start_soon(launch)
        tasks.start_soon(launch)

    assert len(outcomes) == 2
    assert {one.path for one in outcomes} == {outcomes[0].path}
    assert [one.spawned for one in outcomes].count(True) == 1, "one started it; one joined it"

    await shut_down(outcomes[0].path)


async def test_a_socket_that_exists_but_refuses_is_not_a_reason_to_spawn_twice(
    tmp_path: Path,
) -> None:
    """A stale path is cleared by the daemon on its way up, not adopted here.

    The socket file is left behind by a crash; nothing answers on it. This is the
    case where "does the file exist" and "is a daemon there" disagree, and the
    honest reading is the second one — so a daemon *is* started, and `serve`'s own
    `_clear_stale` removes the corpse. What must not happen is `ensure_daemon`
    seeing a file and reporting success on a socket with nothing behind it.
    """
    path = tmp_path / "run" / "daemon.sock"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # a file at the socket's path, answering nothing

    started = await ensure_daemon(argv=ARGV)

    assert started.spawned, "a file is not a daemon"

    await shut_down(started.path)
