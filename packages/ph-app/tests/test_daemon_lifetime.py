"""P7-08 — an auto-started daemon leaves; a service daemon stays.

**The lifetime is decided by who started it, and nothing else.** `ph daemon`
typed at a prompt is a service: somebody chose to run a supervisor, and one that
exits when idle is one that is not there when the next client arrives. A daemon a
UI spawned because the socket was absent was nobody's decision, and a process
left resident after the thing that started it closed is the kind of accretion
nobody attributes to the right cause a week later.

Nothing here is a new mechanism, which is the reason it is a predicate on an
existing cadence rather than a fourth timer. P5-05's sweep already asks "is
anything still using this root"; `spent()` asks the same question one level up,
about the process. What is new is only the *four claimants* — a connection, a
root, an appointment, and the person who typed the command — and that each of
them can keep a daemon up alone.

The exit deliberately has a **window of its own** rather than waiting for the
passivation sweep to empty `roots`, and that is asserted too: coupling them would
make `--passivate-after off` pin an ephemeral daemon forever and
`--passivate-after 90` hold it for ninety minutes, neither of which is what
either flag says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import running, until

from ph.paths import resolve_roots
from ph.seams.schedule_index import ScheduleIndex
from ph.session import now_ms
from ph.testing import StubAgent
from ph_app.daemon.recovery import EPHEMERAL_QUIET, PASSIVATE_AFTER

pytestmark = pytest.mark.anyio


def _appointment(session_id: str = "later", *, at: int = 4_000_000_000_000) -> None:
    """Put one appointment on the books, through the index's own writer.

    The index rather than `schedule/create`, because the claimant under test is
    *the file* — what a daemon reads at boot with no roots at all — and going
    through a root would mean a mounted root existing, which is a different
    condition of the same predicate.
    """
    ScheduleIndex(resolve_roots().home).record(session_id, next_at=at, now=at)


# --------------------------------------------------------------- the window --


def test_the_two_windows_differ_by_intent_not_by_tuning() -> None:
    """Ninety minutes against one, and the interaction that decides the design.

    Asserted as a relation rather than as two numbers, because what matters is
    that the ephemeral window is *far* shorter: left at `PASSIVATE_AFTER`, an
    ephemeral daemon could not reach an empty `roots` — and so could not satisfy
    its own exit predicate — until ninety minutes after the last turn, which
    would make "ephemeral" a word with no behaviour behind it for an hour and a
    half.
    """
    assert EPHEMERAL_QUIET < PASSIVATE_AFTER / 10


async def test_the_exit_does_not_wait_on_the_passivation_window(tmp_path: Path) -> None:
    """Two settings, two questions — and passivation turned *off* is not a pin.

    `--passivate-after off` says "keep the roots", which is a statement about
    roots. It cannot also mean "stay resident after everyone has gone", and a
    predicate that waited for `roots` to empty would have made it mean exactly
    that: an ephemeral daemon that could never leave.

    Sabotage: make condition 3 `not supervisor.roots`, and this hangs on a root
    nothing will ever release.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None) as daemon:
        root = await daemon.server.supervisor.start("kept")
        assert root.id in daemon.server.supervisor.roots

        # An hour on, rather than a sleep: quiet is measured from the log — or,
        # for a session with no events yet, from when the root was mounted — so
        # `now` is the honest way to ask "and once this root has been idle a
        # while?". The root is still mounted, because nothing swept it.
        assert daemon.server.spent(now=now_ms() + 3_600_000), "nothing else wants it"
        assert root.id in daemon.server.supervisor.roots, "and it was not released to get there"


# ------------------------------------------------------------ the predicate --


async def test_an_explicitly_started_daemon_is_never_spent(tmp_path: Path) -> None:
    """A service daemon with nothing to do is still a service.

    The first condition, and the one that cannot be derived from the others: an
    idle service daemon looks identical to a spent ephemeral one from the inside.
    Sabotage: drop the `ephemeral` term, and `ph daemon` exits a minute after the
    last client disconnects — which is the failure mode this whole row is a
    reaction to.
    """
    async with running(tmp_path) as daemon:
        assert not daemon.server.supervisor.roots
        assert not daemon.server.spent(), "somebody chose to run this"


async def test_a_connected_client_keeps_an_ephemeral_daemon_up(tmp_path: Path) -> None:
    """*Connected*, not attached — the distinction the count exists for.

    A client that has opened the socket and asked nothing yet has no
    subscription and no root, so a predicate reading attachments would find
    nothing and stop the daemon out from under a call in flight. `ph agents
    doctor` is exactly that shape: connect, ask, print, leave.

    Sabotage: count attached roots instead of open connections.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        assert daemon.server.spent(), "nothing has connected yet"

        await daemon.client()
        await until(lambda: bool(daemon.server.open_connections), what="the connection to count")

        assert not daemon.server.spent(), "a client is on the socket"


async def test_a_root_somebody_is_watching_keeps_an_ephemeral_daemon_up(
    tmp_path: Path,
) -> None:
    """A subscriber is its own claim, which is P5-05's rule and not a new one.

    Condition 3 is `passivatable`, so every reason a root is still wanted is
    already written down once — a watcher, a live child, an appointment of its
    own, a turn in flight. This exercises the reuse through the case a client
    creates: attach, and the root you are watching cannot be swept out from under
    you, so the daemon behind it stays too.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        root = await daemon.server.supervisor.start("watched")
        client = await daemon.client()
        await client.call("session/attach", sessionId=root.id)
        await until(lambda: bool(root.subscribers), what="the attach to land")

        assert not daemon.server.spent()


async def test_an_appointment_keeps_an_ephemeral_daemon_up(tmp_path: Path) -> None:
    """Any appointment, however far off, because nothing else will fire it.

    The fourth claimant and the least visible: there is no client, no root and
    nothing in memory saying this daemon is wanted — only a file. A daemon that
    exited at 23:59 with a 00:05 appointment indexed would lose the run and the
    schedule would take the blame.

    Sabotage: drop condition 4, and this passes as spent.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        assert daemon.server.spent(), "nothing on the books yet"

        _appointment()

        assert not daemon.server.spent(), "somebody has an appointment with this daemon"


# ----------------------------------------------------------------- the exit --


async def test_the_sweep_that_finds_nothing_left_ends_the_daemon(tmp_path: Path) -> None:
    """One pass: release what is quiet, then leave if nobody needs the process.

    Driven by calling `server.sweep()` rather than by waiting out `SWEEP_EVERY`,
    which is the same reason `spent` is a predicate: a test should assert the
    rule, not the clock. The sweep still runs first — not because the exit
    depends on it, but because releasing a root on the ordinary path is what
    flushes its log and drops its lease outside teardown's shielded window.

    Sabotage: never consult `spent()`, and the daemon serves an empty supervisor
    until somebody kills it.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=0.0) as daemon:
        root = await daemon.server.supervisor.start("done")
        # Nobody watching: a subscriber is its own claim on a root's life, and
        # `start` leaves none.
        assert not root.subscribers

        released = await daemon.server.sweep()

        assert released == ["done"]
        assert daemon.server.stop.is_set(), "the pass that released the last root also ended it"


async def test_a_service_daemon_sweeps_and_stays(tmp_path: Path) -> None:
    """The same pass, the same empty supervisor, the opposite outcome."""
    async with running(tmp_path, passivate_after=0.0) as daemon:
        await daemon.server.supervisor.start("done")

        released = await daemon.server.sweep()

        assert released == ["done"]
        assert not daemon.server.stop.is_set()


async def test_a_root_parked_on_a_person_does_not_keep_an_ephemeral_daemon_alive(
    tmp_path: Path,
) -> None:
    """`waiting` is releasable, and this is why that mattered.

    A turn suspended on an approval reports `running` from the agent, because it
    genuinely is mid-turn — but the only thing it waits for is a human, and
    calling that busy holds a whole process for somebody who closed their laptop.
    So `waiting` joins `idle` in `passivatable`, and the sweep that releases such
    a root is the pass that lets this daemon leave.

    What it costs is stated in `NON_GUARANTEES`: the ask was in memory, so
    stopping loses it. The log keeps the question.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=0.0) as daemon:
        root = await daemon.server.supervisor.start("parked")
        outcome: list[Any] = []

        async with anyio.create_task_group() as tasks:

            async def ask() -> None:
                outcome.append(
                    await root.ctx.approval.request(
                        agent=StubAgent(ctx=root.ctx, session=root.session),
                        tool_name="write",
                        call_id="c1",
                    )
                )

            tasks.start_soon(ask)
            await anyio.sleep(0.05)
            assert root.status == "waiting", "parked on a human, by the desk's own reckoning"

            await daemon.server.sweep()

            assert daemon.server.stop.is_set()
            tasks.cancel_scope.cancel()


async def test_the_socket_is_gone_once_an_ephemeral_daemon_has_left(tmp_path: Path) -> None:
    """So the next client starts one rather than hitting a crash diagnosis.

    Teardown already unlinks — this asserts the consequence, because the two
    failure shapes read completely differently to a person: an *absent* socket
    means "no daemon, start one", and a *present but refusing* one is the
    aftermath of a crash and says something else entirely. An ephemeral daemon
    that left its socket behind would make every ordinary exit look like a crash.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=0.0) as daemon:
        path = daemon.path
        await daemon.server.supervisor.start("done")

        await daemon.server.sweep()
        await until(lambda: not path.exists(), what="the socket to be unlinked")

    assert not path.exists()


async def test_a_session_created_and_never_used_does_not_pin_the_daemon(
    tmp_path: Path,
) -> None:
    """The hole P7-08 made visible, closed where it was: `Root.idle_for`.

    Quiet is measured from the log, which is right — it survives a restart and
    cannot drift from the transcript. But a `session/new` whose client then
    vanished leaves a log with *no events*, and reading that as "zero
    milliseconds idle" made the root permanently unpassivatable: a mounted
    profile held for the life of the daemon, under a predicate designed to
    release it. Now it measures from the header's `created_at` — still the log.

    Sabotage: return `0` for an empty log, and this hangs on a root nothing can
    ever release.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        root = await daemon.server.supervisor.start("never-used")
        assert root.session.last_event is None, "the case under test: nothing has happened"

        assert root.idle_for(now_ms() + 3_600_000) >= 3_600_000
        assert daemon.server.spent(now=now_ms() + 3_600_000)
