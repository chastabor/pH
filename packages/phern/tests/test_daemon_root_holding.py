"""A root is one holding: in the table only while whole, out of it through one door.

The Holdings rules (DESIGN.md §2.5) applied to the daemon's own entry, a `Root` in
`Supervisor.roots`. Everybody who reaches a root reaches it by id from that table —
`start`'s fast path, a connection's attach and part, the release pass, a restart —
so the table is what has to be true: a root is put there when its build is done,
and taken out at the moment its ending is decided, before anything is awaited. The
door that ends it (`_unmount`) also ends its task, after the release, so the log a
release left is the log the next mount reads, and a task that outlived its root
can neither write nor publish.

Each test here failed before the change it names (`plans/Holdings_Followups_Todo.md`,
D1 to D9).

@module tests.test_daemon_root_holding
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import (
    hold_the_provider,
    logged_types,
    running,
    set_phase,
    supervised,
    until,
)

from ph.session import now_ms
from ph_app.daemon import supervisor as supervisor_module
from ph_app.daemon.supervisor import RootStartAbandoned, Supervisor
from ph_app.payloads import SessionStatusNotice
from ph_app.runtime import read_start

pytestmark = pytest.mark.anyio


def _gated_mount(monkeypatch: pytest.MonkeyPatch) -> tuple[anyio.Event, anyio.Event]:
    """Hold the next mount at its first step: `(reached, release)`.

    `read_start` is the first thing `_start` awaits, so a mount held there has
    unmounted whatever a restart is replacing and built nothing yet.
    """
    reached, release = anyio.Event(), anyio.Event()

    async def held(root_id: str | None) -> Any:  # noqa: ANN401
        reached.set()
        await release.wait()
        return await read_start(root_id)

    monkeypatch.setattr(supervisor_module, "read_start", held)
    return reached, release


# ------------------------------------------------------------ D1: the task --


async def test_a_root_released_mid_turn_leaves_no_task_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1. The release ends the root's task, after the log is written.

    The wake channel closing ends a task waiting for work, and nothing ended one
    inside a turn: it ran on against a disposed mount for the life of the daemon,
    a model request still open on the provider. **After** the release, so the
    turn stays open on disk, as a shutdown leaves it for the next daemon's resumer
    (`test_a_daemon_cancelled_rather_than_stopped_leaves_its_roots_as_a_stop_does`).

    Sabotage: drop the cancel from `_unmount`, and the agent still reads running.
    """
    hold_the_provider(monkeypatch)
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("r")
        await supervisor.prompt("r", "go")
        await until(lambda: root.agent.status == "running", what="the turn to start")
        await until(
            lambda: "request/context" in [event.type for event in root.session.events],
            what="the model request",
        )
        await supervisor.passivate(root, now=now_ms())
        assert root.agent.status == "idle", "the released root's turn is still running"
    assert "turn/end" not in logged_types("r"), "the release must leave the turn cut, not end it"


async def test_a_turn_that_fails_after_its_root_left_takes_no_ladder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1. A root that is not the table's is not retried, and says nothing.

    The ladder appends `supervisor/retry` and publishes `retrying` to the root's
    watchers — after a restart, the same watchers as its successor's. A drive that
    finds its root gone has nothing to retry: what it held is released.

    Sabotage: drop the identity check in `_drive`, and `retrying` is heard.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("r")
        heard: list[Any] = []
        root.subscribe(lambda _method, payload: heard.append(payload.get("status")))

        async def released_then_raises() -> None:
            # What a release under the turn leaves `run` to find on its way out.
            supervisor.roots.pop("r")
            raise RuntimeError("the store went with the mount")

        monkeypatch.setattr(root.agent, "run", released_then_raises)
        await supervisor._drive(root)
        supervisor.roots["r"] = root
    assert "retrying" not in heard and "failed" not in heard, heard
    assert "supervisor/retry" not in [event.type for event in root.session.events]


async def test_what_a_replaced_root_still_says_reaches_nobody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1, D8. What the old root still says after a restart reaches nobody.

    The watchers were copied to the successor and left on the old root, so anything
    the old root published afterwards — a ladder's `retrying`, a `failed` — reached
    the watchers of the live root under the same id. They are the id's now
    (`Audience`), and a root leaves its id's audience as its unmount ends.

    Sabotage: leave the unmounted root its audience, and its notice is heard.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        old = await supervisor.start("r")
        heard: list[Any] = []
        old.subscribe(lambda _method, payload: heard.append(payload.get("status")))
        back = await supervisor.restart(old)
        assert back is not None and back is not old
        assert len(back.subscribers) == 1, "whoever was attached is watching the new root"
        old.publish(SessionStatusNotice(session_id="r", status="failed"))
    assert "failed" not in heard, "the old root still reaches its successor's watchers"


# ------------------------------------------------- D2: a restart is a mount --


async def test_a_restart_its_caller_abandons_finishes_on_the_supervisors_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2. The client that asked for a restart leaving does not leave the root gone.

    `restart` ran on the connection's own task, and a client disconnecting cancels
    every handler it has in flight: the cancel landed inside the mount, after the
    unmount, so the root was out of the table with nothing said and its watchers
    were left on a dead root. A mount is the supervisor's work (`start`), and so is
    the remount a restart is.

    Sabotage: run the remount inline in `restart`, and no root is mounted.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        old = await supervisor.start("r")
        watcher: list[Any] = []
        old.subscribe(lambda _method, payload: watcher.append(payload))
        reached, release = _gated_mount(monkeypatch)
        try:
            async with anyio.create_task_group() as caller:
                caller.start_soon(supervisor.restart, old)
                await reached.wait()
                caller.cancel_scope.cancel()
        finally:
            release.set()
        await until(lambda: "r" in supervisor.roots, what="the restart to finish")
        assert supervisor.roots["r"] is not old
        assert supervisor.roots["r"].subscribers, "the watcher moved to the new root"


# ------------------------------------------------------ D3, D8: attachment --


async def test_attaching_again_after_a_release_watches_the_new_root(tmp_path: Path) -> None:
    """D3. A connection's attachment follows the id to whichever root holds it.

    `_join` subscribed only when the id was not yet in the connection's `attached`
    set, and a root released and mounted again is a new object: the connection
    that attached to the first was never subscribed to the second, and a TUI
    waited for frames that never came.

    Sabotage: guard the subscribe on `attached` again, and the new root has no
    watcher.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/attach", sessionId="r")
        first = daemon.held("r")
        await daemon.running.supervisor.passivate(first, now=now_ms())
        await client.call("session/attach", sessionId="r")
        assert daemon.held("r") is not first
        assert daemon.held("r").subscribers, "attached, and watching nothing"


async def test_a_watcher_that_leaves_during_a_restart_is_not_moved_to_the_new_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D8. A detach during a restart lands, though no root is under the id.

    `_part` looked the root up by id, and a restart has taken it out of the table
    until the new one is whole: the detach found nothing, and the restart then
    moved the departed connection to the new root, which it held mounted for a
    watcher that had left. A connection parts from its id's audience now
    (`Supervisor.part`), which is there whether a root is or not.

    Sabotage: have `part` do nothing unless a root is mounted, and the new root
    has a watcher.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/attach", sessionId="r")
        old = daemon.held("r")
        reached, release = _gated_mount(monkeypatch)
        supervisor = daemon.running.supervisor
        daemon.tasks.start_soon(supervisor.restart, old)
        try:
            await reached.wait()
            await client.call("session/detach", sessionId="r")
        finally:
            release.set()
        await until(lambda: daemon.holds("r") and daemon.held("r") is not old, what="the restart")
        assert not daemon.held("r").subscribers, "the departed watcher was moved"


# ------------------------------------------- the audience: the id's, not a root's --


async def test_a_restarted_root_has_what_its_watchers_had(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tray and the ask count are the id's, as the watchers are.

    A restart used to copy each onto the new root by hand, so a field added to one
    side was a field to remember on the other. The new root reads its id's
    audience, and there is nothing to carry.

    Sabotage: hand each `Root` an `Audience()` of its own in `_start`, and the new
    root has none of it.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        old = await supervisor.start("r")
        heard: list[Any] = []

        def watcher(_method: str, payload: dict[str, Any]) -> None:
            heard.append(payload)

        old.subscribe(watcher)
        tray = old.staged
        old.audience.asked = 3

        back = await supervisor.restart(old)

        assert back is not None and back is not old
        assert back.subscribers == {watcher}
        assert back.staged is tray
        assert back.audience.asked == 3


async def test_watchers_of_a_restart_that_failed_hear_the_next_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart that does not come back tells whoever is attached, and leaves them
    attached to the id: the next mount of it is theirs to hear.

    They used to be cleared off the dead root, while each connection still held the
    id as attached, so a prompt that mounted the root again ran in front of nobody.

    Sabotage: clear the audience's subscribers where the restart fails, and the new
    root has no watcher.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        old = await supervisor.start("r")
        heard: list[Any] = []

        def watcher(_method: str, payload: dict[str, Any]) -> None:
            heard.append(payload.get("status"))

        old.subscribe(watcher)
        starting = Supervisor._start

        async def refused(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("the profile would not mount")

        monkeypatch.setattr(Supervisor, "_start", refused)
        assert await supervisor.restart(old) is None
        assert "failed" in heard
        assert "r" not in supervisor.roots and "r" in supervisor.audiences

        monkeypatch.setattr(Supervisor, "_start", starting)
        again = await supervisor.start("r")

        assert again.subscribers == {watcher}


async def test_an_audience_goes_with_the_last_of_what_held_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Supervisor.audiences` is by id and outlives a root, so each thing that can be
    the last to hold one lets it go: a root unmounted with nobody attached, a mount
    that failed, and the last watcher of an id nothing is mounted under.

    Sabotage: make `_unattended` do nothing, and all three are left in the table.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        quiet = await supervisor.start("quiet")
        assert "quiet" in supervisor.audiences
        await supervisor.passivate(quiet, now=now_ms())
        assert "quiet" not in supervisor.audiences, "released with nobody attached"

        checking = Supervisor._check_credential

        async def refused(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("the build failed after the root was made")

        monkeypatch.setattr(Supervisor, "_check_credential", refused)
        with pytest.raises(RuntimeError, match="the build failed"):
            await supervisor.start("never")
        assert "never" not in supervisor.audiences, "a mount that failed"
        monkeypatch.setattr(Supervisor, "_check_credential", checking)

        watched = await supervisor.start("watched")

        def watcher(_method: str, _payload: dict[str, Any]) -> None:
            return None

        front_end: Any = object()
        watched.subscribe(watcher)
        watched.audience.front_ends.add(front_end)
        starting = Supervisor._start

        async def will_not(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("the profile would not mount")

        monkeypatch.setattr(Supervisor, "_start", will_not)
        assert await supervisor.restart(watched) is None
        monkeypatch.setattr(Supervisor, "_start", starting)
        assert "watched" in supervisor.audiences, "still attached, to an id with no root"
        supervisor.part("watched", watcher, front_end)
        assert "watched" not in supervisor.audiences, "the last watcher left"
        assert supervisor.audiences == {}


# -------------------------------------------------- D4: release, decided --


async def test_a_root_prompted_while_its_release_waited_is_not_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D4. The release pass asks again once it holds the lock.

    `sweep` decided a root was quiet, then waited for `_starting`, then checked only
    that the root was the same object: a prompt in that wait was recorded as
    `passivated` and cut short.

    Sabotage: drop the second `passivatable` from the release, and the busy root is
    released.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("r")
        released: list[list[str]] = []

        async def sweep() -> None:
            released.append(await supervisor.sweep(after=0.0))

        async with anyio.create_task_group() as tasks:
            await supervisor._starting.acquire()
            try:
                tasks.start_soon(sweep)
                await anyio.sleep(0.05)
                set_phase(root, "running")
            finally:
                supervisor._starting.release()
        assert released == [[]], "a root that started working while the pass waited"
        assert supervisor.roots.get("r") is root
        set_phase(root, "idle")


# ---------------------------------------------- D5: in the table when whole --


async def test_a_root_is_not_handed_out_before_its_build_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D5. `start`'s fast path finds only whole roots.

    The root went into the table before its children were readmitted and before the
    relay that sends its events was registered, so a second caller was handed a root
    whose events it would miss — or, if the build then failed, a success reply for
    a root that never ran.

    Sabotage: put the root in the table before `_resume_children` again, and the
    second start returns while the first is still building.
    """
    reached, release = anyio.Event(), anyio.Event()

    async def resuming(self: Supervisor, root: Any) -> None:  # noqa: ANN401
        reached.set()
        await release.wait()

    monkeypatch.setattr(Supervisor, "_resume_children", resuming)
    async with supervised(tmp_path, monkeypatch) as supervisor:
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(supervisor.start, "r")
            try:
                await reached.wait()
                assert "r" not in supervisor.roots, "in the table half-built"
                second: list[object] = []
                with anyio.move_on_after(0.1):
                    second.append(await supervisor.start("r"))
                assert second == [], "handed out while its build was still running"
            finally:
                release.set()
        assert "r" in supervisor.roots


# ------------------------------------------------- D6: a ring nobody hears --


async def test_ringing_a_root_whose_task_has_gone_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D6. A ring after the wake closed is a ring for nobody, as a full channel's is.

    `ring` suppressed `WouldBlock` alone, so a closed wake raised out of a listener
    during shutdown and out of `credentials/store` for a credential that had been
    stored.

    Sabotage: suppress `WouldBlock` alone again, and this raises.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("r")
        await root.wake.aclose()
        root.ring()


# ---------------------------------------------------- D7: during shutdown --


async def test_no_root_is_handed_out_once_shutdown_has_begun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7. The `_closing` check comes before the fast path.

    `aclose` releases roots one at a time, so a mounted root could be handed to a
    handler between the start of shutdown and its own release, and then released
    under it.

    Sabotage: check `_closing` after the fast path again, and this returns a root.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        await supervisor.start("r")
        supervisor._closing = True
        with pytest.raises(RootStartAbandoned):
            await supervisor.start("r")
        supervisor._closing = False


async def test_shutdown_takes_each_root_out_of_the_table_as_it_releases_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D7. A root being released is not in the table any more.

    The table was cleared after every root had been released, so for the whole of
    one root's unwind it was still listed, and still found by id.

    Sabotage: clear the table at the end of `aclose` instead, and the root is found
    while it unwinds.
    """
    seen: list[bool] = []
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("r")
        root.exits.callback(lambda: seen.append("r" in supervisor.roots))
    assert seen == [False]
