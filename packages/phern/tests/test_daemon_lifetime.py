"""P7-08 and P9-06 — an auto-started daemon leaves; a service daemon stays.

**The lifetime is decided by who started it, and nothing else.** `phern daemon`
typed at a prompt is a service: somebody chose to run a supervisor, and one that
exits when idle is one that is not there when the next client arrives. A daemon a
UI spawned because the socket was absent was nobody's decision, and a process
left resident after the thing that started it closed is the kind of accretion
nobody attributes to the right cause a week later.

Nothing here is a new mechanism, which is the reason it is a predicate rather
than a fourth timer. P5-05's sweep already asks "is anything still using this
root"; `holds()` asks the same question one level up, about the process. What is
new is only the *claimants* — a connection, a turn in flight, an appointment, a
keep-alive somebody asked for, and the person who typed the command — and that each
of them can keep a daemon up alone.

P9-06 moved the exit off the cadence and onto the transitions. Closing the last
terminal ends an auto-started daemon *then*, not up to a sweep later, so the
gates come in two halves: what `holds()` says, and when `check_lifetime()` acts
on it. The exit no longer has a quiet window of its own either — it asks what is
happening now, where the sweep asks how long a root has been quiet — and the
fifth claimant, a `--keep-alive` the client asked for, is the only one that ends on
a clock rather than on an event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import running, until

from ph.keys import APPROVAL
from ph.paths import resolve_roots
from ph.seams.schedule_index import ScheduleIndex
from ph.session import SurfaceIntent, now_ms
from ph.testing import StubAgent, user_payload
from ph_app.daemon.launch import SPAWN_TIMEOUT

pytestmark = pytest.mark.anyio


def _appointment(session_id: str = "later", *, at: int = 4_000_000_000_000) -> None:
    """Put one appointment on the books, through the index's own writer.

    The index rather than `schedule/create`, because the claimant under test is
    *the file* — what a daemon reads at boot with no roots at all — and going
    through a root would mean a mounted root existing, which is a different
    condition of the same predicate.
    """
    ScheduleIndex(resolve_roots().home).record(session_id, next_at=at, now=at)


# -------------------------------------------- what the exit does not wait for --


async def test_the_exit_no_longer_waits_on_the_root_quiet_window(tmp_path: Path) -> None:
    """A root that was busy a second ago does not hold the process for a minute.

    The exit used to ask P5-05's `passivatable` whole, and so inherited a
    sixty-second window measured from the log's last event. That window is right
    for releasing a *root* — somebody who stepped away comes back to a mounted
    profile rather than to a rehydrate — and wrong for a process nobody is
    connected to: it put a minute between closing the last terminal and the
    daemon leaving, with nothing a person could see to account for it.

    So the two halves ask different questions now, which is why `EPHEMERAL_QUIET`
    is gone rather than retuned: a second quiet window would have been a second
    answer to when a root is finished.

    Sabotage: put a quiet term back into `spent` — `passivatable(root, now=...,
    after=60.0)` — and this fails on a log written a moment ago.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None) as daemon:
        daemon.aged()
        root = await daemon.running.supervisor.start("just-finished")
        root.session.append("user/message", user_payload("the last thing"), SurfaceIntent("append"))
        assert root.idle_for(now_ms()) < 1_000, "the log was written just now"

        assert daemon.running.spent(), "idle is idle, however recently"
        assert not daemon.running.holds(), "and nothing else wants it either"


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
        root = await daemon.running.supervisor.start("kept")
        assert root.id in daemon.running.supervisor.roots

        # An hour on, rather than a sleep: quiet is measured from the log — or,
        # for a session with no events yet, from when the root was mounted — so
        # `now` is the honest way to ask "and once this root has been idle a
        # while?". The root is still mounted, because nothing swept it.
        assert daemon.running.spent(now=now_ms() + 3_600_000), "nothing else wants it"
        assert root.id in daemon.running.supervisor.roots, "and it was not released to get there"


# ------------------------------------------------------------ the predicate --


async def test_an_explicitly_started_daemon_is_never_spent(tmp_path: Path) -> None:
    """A service daemon with nothing to do is still a service.

    The first condition, and the one that cannot be derived from the others: an
    idle service daemon looks identical to a spent ephemeral one from the inside.
    Sabotage: drop the `ephemeral` term, and `phern daemon` exits a minute after the
    last client disconnects — which is the failure mode this whole row is a
    reaction to.
    """
    async with running(tmp_path) as daemon:
        assert not daemon.running.supervisor.roots
        assert not daemon.running.spent(), "somebody chose to run this"


async def test_a_connected_client_keeps_an_ephemeral_daemon_up(tmp_path: Path) -> None:
    """*Connected*, not attached — the distinction the count exists for.

    A client that has opened the socket and asked nothing yet has no
    subscription and no root, so a predicate reading attachments would find
    nothing and stop the daemon out from under a call in flight. `phern agents
    doctor` is exactly that shape: connect, ask, print, leave.

    Sabotage: count attached roots instead of open connections.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        daemon.aged()
        assert daemon.running.spent(), "nothing has connected yet"

        await daemon.client("asks")
        await until(lambda: bool(daemon.running.connections), what="the connection to register")

        assert not daemon.running.spent(), "a client is on the socket"
        assert daemon.running.holds() == ["client"]


async def test_a_running_turn_holds_an_ephemeral_daemon_past_the_last_detach(
    tmp_path: Path,
) -> None:
    """The one claim a connection cannot make, because the work outlives it.

    `phern -p` prompts and detaches; a daemon that left with the last connection
    would take the turn with it, halfway through a tool call, leaving a log that
    shows a prompt and no answer.

    **A watcher is no longer a claimant of its own**, and that is a deletion
    rather than an oversight: a subscription is registered by a connection and
    dropped in that connection's own teardown — before `_handle`'s `finally`
    asks — so "somebody is watching" cannot outlive "somebody is connected", and
    keeping both would be one fact with two answers. What `busy` adds over the
    connection is the opposite case: work with nobody watching it.

    Sabotage: drop the `task` term from `holds`, and a detached prompt is stopped
    mid-turn by the disconnect that started it.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        await daemon.busy_root("working")

        client = await daemon.client("asks")
        await until(lambda: bool(daemon.running.connections), what="the connection to register")
        await client.aclose()
        await until(lambda: not daemon.running.connections, what="the disconnect to land")

        assert daemon.running.holds() == ["task"], "nobody is here; something is running"
        assert not daemon.running.stop.is_set()


async def test_an_appointment_keeps_an_ephemeral_daemon_up(tmp_path: Path) -> None:
    """Any appointment, however far off, because nothing else will fire it.

    The fourth claimant and the least visible: there is no client, no root and
    nothing in memory saying this daemon is wanted — only a file. A daemon that
    exited at 23:59 with a 00:05 appointment indexed would lose the run and the
    schedule would take the blame.

    Sabotage: drop condition 4, and this passes as spent.
    """
    async with running(tmp_path, ephemeral=True) as daemon:
        daemon.aged()
        assert daemon.running.spent(), "nothing on the books yet"

        _appointment()

        assert not daemon.running.spent(), "somebody has an appointment with this daemon"
        assert daemon.running.holds() == ["schedule"], "and it can say which claim that is"


async def test_a_keep_alive_holds_it_for_exactly_as_long_as_it_says(tmp_path: Path) -> None:
    """Armed on the last disconnect, cleared on the next connect, and it expires.

    The window is "since you left", which is the only reading under which
    `--keep-alive 30s` means what the person typing it means: a daemon that started
    the window at boot would be gone before the second terminal opened, and one
    that never cleared it would re-arm a window it was already inside.

    Asserted against the deadline rather than by sleeping thirty seconds —
    `holds` takes a `now` for the reason `passivatable` does.

    Sabotage: arm it at start rather than on the last disconnect.
    """
    async with running(tmp_path, ephemeral=True, keep_alive=30.0) as daemon:
        assert daemon.running.keep_alive_until is None, "nobody has left yet"

        first = await daemon.client("asks")
        await until(lambda: bool(daemon.running.connections), what="the connection to register")
        await first.aclose()
        await until(lambda: not daemon.running.connections, what="the disconnect to land")

        armed = daemon.running.keep_alive_until
        assert armed is not None, "the last one out arms it"
        assert daemon.running.holds(now=armed - 1) == ["keep-alive"]
        assert not daemon.running.holds(now=armed), "the deadline is the end of the window"
        assert not daemon.running.stop.is_set(), "and it was still open when they left"

        await daemon.client("asks")
        await until(lambda: bool(daemon.running.connections), what="the second client to arrive")

        assert daemon.running.keep_alive_until is None, "a window is for a daemon nobody is using"


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
        daemon.aged()
        root = await daemon.running.supervisor.start("done")
        # Nobody watching: a subscriber is its own claim on a root's life, and
        # `start` leaves none.
        assert not root.subscribers

        released = await daemon.running.sweep()

        assert released == ["done"]
        assert daemon.running.stop.is_set(), "the pass that released the last root also ended it"


async def test_a_service_daemon_sweeps_and_stays(tmp_path: Path) -> None:
    """The same pass, the same empty supervisor, the opposite outcome."""
    async with running(tmp_path, passivate_after=0.0) as daemon:
        await daemon.running.supervisor.start("done")

        released = await daemon.running.sweep()

        assert released == ["done"]
        assert not daemon.running.stop.is_set()


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
        daemon.aged()
        root = await daemon.running.supervisor.start("parked")
        outcome: list[Any] = []

        async with anyio.create_task_group() as tasks:

            async def ask() -> None:
                outcome.append(
                    await root.ctx.require(APPROVAL).request(
                        agent=StubAgent(ctx=root.ctx, session=root.session),
                        tool_name="write",
                        call_id="c1",
                    )
                )

            tasks.start_soon(ask)
            await anyio.sleep(0.05)
            assert root.status == "waiting", "parked on a human, by the desk's own reckoning"

            await daemon.running.sweep()

            assert daemon.running.stop.is_set()
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
        daemon.aged()
        path = daemon.path
        await daemon.running.supervisor.start("done")

        await daemon.running.sweep()
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
        root = await daemon.running.supervisor.start("never-used")
        assert root.session.last_event is None, "the case under test: nothing has happened"

        assert root.idle_for(now_ms() + 3_600_000) >= 3_600_000
        assert daemon.running.spent(now=now_ms() + 3_600_000)


async def test_an_auto_started_daemon_exits_when_the_last_connection_closes(
    tmp_path: Path,
) -> None:
    """At once, which is the row's whole point, and not one sweep later.

    The exit rode the passivation sweep alone, so closing the last terminal left
    a daemon resident for the rest of a `SWEEP_EVERY` window with nothing to do —
    the gap a person reads as "it did not work". A connection closing is an
    *event*, and the object that owns the lifetime is the one that sees it.

    Two clients, so the assertion is about the **last** one: a daemon that left
    when any connection closed would take down the terminal beside it. The sweep
    is pushed out to ten minutes rather than turned off, because a cadence that
    could still cover for the missing call would make this gate prove nothing.

    Sabotage: leave the check on the sweep alone, and this hangs for ten minutes.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None, sweep_every=600.0) as daemon:
        first = await daemon.client("asks")
        second = await daemon.client("asks")
        await until(lambda: len(daemon.running.connections) == 2, what="both clients to register")

        await first.aclose()
        await until(lambda: len(daemon.running.connections) == 1, what="the first client to go")
        assert not daemon.running.stop.is_set(), "somebody is still on the socket"

        await second.aclose()
        await until(lambda: daemon.running.stop.is_set(), what="the daemon to stop")


async def test_a_knock_on_the_socket_does_not_end_the_daemon_it_was_checking_for(
    tmp_path: Path,
) -> None:
    """The window between the spawn and the UI's first connect, closed.

    `launch.listening()` is a connect and an immediate close — the only test that
    tells "a socket file exists" from "a daemon is behind it", and every spawn
    runs it in a poll until the door opens. Reading that close as "the last
    client left" stopped the daemon the poll had just declared ready, so the UI
    that started it connected to nothing: `test_daemon_launch.py` saw it as three
    `DaemonGone`s, which points nowhere near this line.

    `listening`'s own two steps, paused between them: a connect and a close, with
    a wait in the middle so the daemon is *observed* accepting and dropping it.
    Calling `listening` straight through cannot be asserted on — it returns
    before the accept loop has built the connection, so a poll for "the knock is
    over" can be satisfied by a knock that has not happened yet, and the gate
    passes under its own sabotage.

    Note what this daemon deliberately is *not*: `aged()`. The window is the
    subject, so the gate sits on the launcher's side of it.

    Sabotage: drop the `served` term from `spent`, and a UI can no longer start a
    daemon at all.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None) as daemon:
        knock = await anyio.connect_unix(str(daemon.path))
        await until(lambda: bool(daemon.running.connections), what="the knock to be accepted")

        await knock.aclose()
        await until(lambda: not daemon.running.connections, what="the knock to end")

        assert not daemon.running.stop.is_set(), "nobody has been served yet"


async def test_an_explicitly_started_daemon_still_never_exits(tmp_path: Path) -> None:
    """The same transition, the same empty daemon, the opposite outcome.

    `check_lifetime` runs on the connection path now, and that is a path a
    *service* daemon takes too — `phern agents doctor` connects, asks, prints and
    leaves. So the flag has to be read there as well; a check that only asked
    "is anything connected" would end `phern daemon` on the first such call.
    """
    async with running(tmp_path, passivate_after=None) as daemon:
        client = await daemon.client("asks")
        await until(lambda: bool(daemon.running.connections), what="the connection to register")

        await client.aclose()
        await until(lambda: not daemon.running.connections, what="the disconnect to land")

        assert not daemon.running.stop.is_set(), "somebody chose to run this"


async def test_the_keep_alive_expires_without_a_client_to_notice(tmp_path: Path) -> None:
    """The backstop, and the reason the sweep still asks after P9-06.

    Every other hold ends on an event — a connection closes, a turn ends, a
    schedule is canceled — and `check_lifetime` runs on each. A keep-alive ends on a
    *clock*, with nobody left in the process to notice, so without the sweep's
    call an ephemeral daemon with `--keep-alive` would outlive its own window and sit
    there until something unrelated happened to it.

    Driven with no connection at all and the deadline already behind us, because
    the subject is the pass rather than the wait.

    Sabotage: drop `check_lifetime()` from `sweep`, and this hangs.
    """
    async with running(tmp_path, ephemeral=True, keep_alive=30.0, passivate_after=None) as daemon:
        daemon.aged()
        daemon.running.keep_alive_until = now_ms() - 1
        assert not daemon.running.holds(), "the window is behind us"

        await daemon.running.sweep()

        assert daemon.running.stop.is_set(), "the pass that noticed is the one that ends it"


async def test_a_daemon_nobody_spoke_to_gives_up_when_its_launcher_would_have(
    tmp_path: Path,
) -> None:
    """The far edge of the spawn window, which is what keeps it from being a pin.

    `served` protects a daemon from leaving before the UI that spawned it can
    find it. Latched, that protection would be permanent: a UI that crashed
    between the spawn and its first frame would leave a supervisor resident for
    good, which is the accretion this whole row exists to prevent — the failure
    arriving from the other side.

    `SPAWN_TIMEOUT` bounds it, and is the launcher's own number rather than a
    second one beside it: past that point `_await_socket` has already given up,
    so there is nobody left to protect.

    Sabotage: make `served` the only term, and this hangs — forever, on a real
    machine.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None) as daemon:
        assert not daemon.running.served, "the case under test: nobody has spoken"
        assert not daemon.running.spent(), "the launcher may still be on its way"
        assert not daemon.running.holds(), "and nothing is claiming it, either"

        assert daemon.running.spent(now=now_ms() + int(SPAWN_TIMEOUT * 1000)), (
            "past the point the launcher waits, there is nobody to wait for"
        )


async def test_a_knock_outliving_the_last_client_does_not_strand_the_daemon(
    tmp_path: Path,
) -> None:
    """The hole the per-connection guard left, closed by moving it to the process.

    A probe open while the last real client leaves *holds* the daemon — it is a
    connection, and `holds` counts connections, which is the conservative
    direction. Then it closes without ever having spoken, and while the exit was
    guarded per-connection that close asked nothing: the daemon sat there until
    something unrelated happened to it, up to a whole `sweep_every` later.

    Now the check on every teardown is unconditional, because `spent()` knows
    about the spawn window itself — and this daemon has been served, so there is
    no window left to protect.

    Sabotage: guard the teardown's `check_lifetime()` on `connection.spoke`
    again, and this waits for a sweep that is ten minutes out.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None, sweep_every=600.0) as daemon:
        client = await daemon.client("asks")
        await until(lambda: daemon.running.served, what="the client's first frame")

        knock = await anyio.connect_unix(str(daemon.path))
        await until(lambda: len(daemon.running.connections) == 2, what="the knock to be accepted")

        await client.aclose()
        await until(lambda: len(daemon.running.connections) == 1, what="the client to go")
        assert not daemon.running.stop.is_set(), "a connection is a connection"

        await knock.aclose()
        await until(lambda: daemon.running.stop.is_set(), what="the daemon to stop")


async def test_a_turn_finishing_with_nobody_watching_ends_the_daemon(tmp_path: Path) -> None:
    """The transition nothing on the socket can see, and the one it exists for.

    A detached `phern -p` prompts and hangs up: the daemon it spawned has no
    connection, no watcher and no appointment, and the only thing holding it is
    the turn. Nothing arrives on the socket when that turn ends — the client is
    long gone — so the *agent's* status change is the event, which is why
    `Supervisor.recheck_lifetime` is registered as a listener of its own rather
    than living inside `announce`, whose whole body is guarded on there being
    watchers to announce to.

    `sweep_every` is pushed out to ten minutes so the cadence cannot cover for
    the missing registration: what ends this daemon has to be the turn.

    Sabotage: drop `ctx.on("agent/status", lifetime)`, or fold it back inside
    `announce`'s `subscribers` guard, and a one-shot run leaves a supervisor
    behind every time.
    """
    async with running(tmp_path, ephemeral=True, passivate_after=None, sweep_every=600.0) as daemon:
        daemon.aged()
        root = await daemon.running.supervisor.start("detached")
        assert not root.subscribers, "the case under test: nobody is listening"

        await daemon.running.supervisor.prompt("detached", "do the thing")

        await until(lambda: daemon.running.stop.is_set(), what="the finished turn to end it")
