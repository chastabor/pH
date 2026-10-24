"""Phase 13's SIGKILL gates: a daemon killed mid-work, and the next one on its `$PH_HOME`.

`test_mass_restart.py` kills a process holding a mounted profile and restarts with
the core's own resume, which is what a daemon's mount does for one root. What it
cannot reach is the rest of a daemon: the live-roots index the next one reads to
find what is owed, the resume pass that mounts it, the continuation of a cut-short
turn, the ring a finishing child gives an idle parent, the scheduler that wakes a
root for its appointment, and the probes an orchestrator polls meanwhile. Here the
process killed is a daemon (`daemon_restart_host.py`), and what comes up after it
is a second daemon — in this process, so the test can watch its roots, or a second
host when what is under test is a kill of the one that resumed.

What each gate pins is in its docstring; the sabotage named there is the change
that turns it red.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import (
    answered,
    continuations,
    daemon_socket,
    http,
    logged,
    logged_types,
    private_runtime,
    run_phern,
    serving,
    stale,
    supervised,
    taken_from,
    until,
)
from daemon_restart_host import GOAL, PROFILE, ROOT, Mode
from killed_host import killed_when_ready

from ph.json import as_obj
from ph.keys import GOALS, SUBAGENTS
from ph.llm.fake import FakeAdapter
from ph.llm.types import GenerateOptions, StreamChunk
from ph.paths import resolve_roots
from ph.persistence import Resumption, interrupted_turn_closers, resumption_of
from ph.seams.subagents import ChildState
from ph.session import SessionEvent, now_ms
from ph.testing import logged_events, not_none, stored_log
from ph_app import verbs
from ph_app.daemon.supervisor import Root, Supervisor
from ph_app.params import CancelScheduleParams, CreateScheduleParams

pytestmark = pytest.mark.anyio

HOST = Path(__file__).with_name("daemon_restart_host.py")
SETTLE_SECONDS = 30.0


def _killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Mode) -> dict[str, Any]:
    """A daemon in `mode`, killed once its work is on disk. The runtime directory is
    pinned first, so the socket the host leaves is the one the next daemon finds."""
    private_runtime(tmp_path, monkeypatch)
    return killed_when_ready(HOST, mode)


def _children(root: Root) -> dict[str, ChildState]:
    """The root's children by run id, each as its own log tells it."""
    return root.ctx.require(SUBAGENTS).children(root.id)


@asynccontextmanager
async def _resumed_by_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Root]:
    """The next daemon, told `phern agents resume --all` from outside, and the root
    that pass mounted."""
    async with serving(tmp_path, monkeypatch, profile=PROFILE) as daemon:
        asked = await run_phern("agents", "resume", "--all")
        assert asked.exit_code == 0, asked.output
        yield daemon.held(ROOT)


def _completed(root: Root) -> list[SessionEvent]:
    """Every turn of the root's that ran to an answer."""
    return [
        event
        for event in root.session.events
        if event.type == "turn/end" and as_obj(event.data.get("reason")).get("kind") == "completed"
    ]


# ------------------------------------------------- P13-03, P13-04: the goal --


async def test_an_open_goal_settles_under_the_next_daemon_with_no_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed mid-turn on an open goal, a child at the model. `phern agents resume
    --all` against the next daemon finds the root on the index the killed one kept,
    and the mount is the whole of the resume: the cut turn is closed and continued
    because the goal is open, the child is readmitted on its second start and
    finishes, and the goal settles — with nothing attached to the root, since the
    command that asked returned at once.

    Sabotage: drop the goal term from `_continue_interrupted`, and no continuation is
    sent; drop `_resume_children` from `_start`, and the child never finishes.
    """
    crashed = _killed(tmp_path, monkeypatch, "goal")
    run = crashed["child"]["run"]
    async with _resumed_by_cli(tmp_path, monkeypatch) as root:
        goals = root.ctx.require(GOALS)

        def finished() -> bool:
            state = goals.states(root.session).get(GOAL.id)
            child = _children(root).get(run)
            done = child is not None and child.status == "done"
            return state is not None and state.settled and done

        await until(finished, what="the goal and the child to end", seconds=SETTLE_SECONDS)
        assert goals.states(root.session)[GOAL.id].outcome == "achieved"
        assert _children(root)[run].starts == 2, "one start before the kill, one after"
        assert len(continuations(root.session)) == 1, "the cut turn was continued once"
        assert not root.subscribers, "nobody was attached to see it through"
        await until(
            lambda: "goal/settled" in logged_types(ROOT),
            what="the settlement to reach disk",
            seconds=SETTLE_SECONDS,
        )


# --------------------------------------------- P13-05: a child's ending wakes --


async def test_a_readmitted_child_that_finishes_wakes_its_idle_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed with the root idle and its child at the model. The next daemon readmits
    the child and the root stays idle — there is nothing of its own to continue —
    until the child finishes; then the child's ending notice rings the root, and a
    turn of the root's takes that very notice, by the id the child's own log gave it.

    Sabotage: deliver the notice with `inject` in `_deliver_notice`, and the root sits
    on it for good.
    """
    crashed = _killed(tmp_path, monkeypatch, "wake")
    run, child_id = crashed["child"]["run"], crashed["child"]["session"]
    release = anyio.Event()
    answer = FakeAdapter.stream

    async def gated(self: FakeAdapter, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        """The child answers only once the test has seen the root idle."""
        if options.session_id == child_id:
            await release.wait()
        async for chunk in answer(self, options):
            yield chunk

    monkeypatch.setattr(FakeAdapter, "stream", gated)
    async with _resumed_by_cli(tmp_path, monkeypatch) as root:
        await until(lambda: _children(root)[run].status == "running", what="the readmission")
        assert root.status == "idle" and not root.agent.inbox.has_pending
        mark = root.session.seq
        release.set()

        await until(
            lambda: answered(root) and taken_from(root.session, "ph.seams.subagents"),
            what="the root to wake and read the notice",
            seconds=SETTLE_SECONDS,
        )
        notice = not_none(_children(root)[run].notice, "the child's ending notice")
        [read] = taken_from(root.session, "ph.seams.subagents")
        assert read.seq > mark, "read by a turn the restart ran, not one before it"
        assert read.data["id"] == notice.id


# ------------------------------------------------- P13-07: the durable tick --


@pytest.mark.parametrize("where", ["tick-claim", "tick-write"])
async def test_a_kill_inside_a_schedule_tick_costs_exactly_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: Mode
) -> None:
    """Killed inside `fire_due`: after the claim and before the delivery, with the tick
    only in the buffer, or after the write and before the index was told, with the
    tick and its prompt on disk. Either way the next daemon's scheduler wakes the
    root from the index alone, and the schedule runs exactly once: claimed once,
    taken once, answered once. A later pass finds nothing due, and the index lets
    the spent schedule go.

    Sabotage: write the log before `deliver` in `fire_due`, and a kill after the
    claim leaves the tick on disk with no prompt behind it, which no pass runs.
    """
    _killed(tmp_path, monkeypatch, where)
    left = logged_types(ROOT)
    if where == "tick-claim":
        assert "schedule/tick" not in left, "the claim was the buffer's when the kill came"
    else:
        assert left.count("schedule/tick") == 1
        assert "agent/inbox/spliced" in left[left.index("schedule/tick") :]

    async with serving(tmp_path, monkeypatch, profile=PROFILE) as daemon:
        await until(
            lambda: daemon.holds(ROOT) and answered(daemon.held(ROOT)),
            what="the scheduler to wake the root and run its schedule",
            seconds=SETTLE_SECONDS,
        )
        root = daemon.held(ROOT)
        supervisor = daemon.running.supervisor
        assert await supervisor.wake_and_tick() == [], "nothing is due again"
        assert logged(root.session).count("schedule/tick") == 1
        assert len(taken_from(root.session, "ph.seams.schedule")) == 1, "the prompt was taken once"
        assert len(_completed(root)) == 1, "and answered once"
        # The index is written behind the loop (`IndexWriter`), on the root's own scope.
        with anyio.fail_after(SETTLE_SECONDS):
            await root.ctx.settled()
        assert ROOT not in supervisor.appointments(), "the index lets the spent schedule go"


# -------------------------------------------------------- P13-12: the probes --


async def test_the_probes_tell_each_state_over_tcp_after_a_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What an orchestrator polls on the daemon it started after a kill: up at once,
    over the socket file the killed one left; not ready while the resume pass is
    running, and ready once it ends; not ready while a root's log refuses a write,
    and ready again once one lands; not live once the loop stops stamping.
    """
    _killed(tmp_path, monkeypatch, "turn")
    assert daemon_socket().exists(), "a kill leaves the socket's file behind"
    passing = anyio.Event()
    resume_owed = Supervisor.resume_owed

    async def held(self: Supervisor, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        await passing.wait()
        return await resume_owed(self, *args, **kwargs)

    monkeypatch.setattr(Supervisor, "resume_owed", held)
    async with serving(
        tmp_path, monkeypatch, profile=PROFILE, probe="127.0.0.1:0", resume_owed=True
    ) as daemon:
        server = daemon.running

        async def probe(name: str) -> tuple[int, list[str]]:
            code, body = await http(server.probe_address, f"GET /{name} HTTP/1.1\r\n\r\n")
            return code, [str(reason) for reason in body.get("reasons", [])]

        assert await probe("startup") == (200, [])
        code, reasons = await probe("ready")
        assert code == 503 and any("resume pass" in reason for reason in reasons), reasons
        assert await probe("live") == (200, [])
        passing.set()
        await until(lambda: server.resumed, what="the resume pass to end")
        assert ROOT in server.supervisor.roots, "the pass mounted what the killed one owed"
        assert await probe("ready") == (200, [])

        log = stored_log(resolve_roots().sessions_dir(), ROOT)
        log.chmod(0o444)
        client = await daemon.client()
        later = str(now_ms() + 3_600_000)
        await client.call(
            verbs.SCHEDULE_CREATE,
            CreateScheduleParams(
                session_id=ROOT, schedule_id="later", kind="once", spec=later, prompt="later"
            ),
        )
        code, reasons = await probe("ready")
        assert code == 503 and any(ROOT in reason for reason in reasons), reasons
        _code, status = await http(server.probe_address, "GET /status HTTP/1.1\r\n\r\n")
        assert status["unwritten"] == [ROOT]
        log.chmod(0o644)
        await client.call(
            verbs.SCHEDULE_CANCEL, CancelScheduleParams(session_id=ROOT, schedule_id="later")
        )
        assert await probe("ready") == (200, []), "a write that lands clears it"

        stale(server)
        code, reasons = await probe("live")
        assert code == 503 and "event loop" in reasons[0], reasons


# ---------------------------------------------- P13-10: the second kill --


async def test_a_second_kill_right_after_a_resume_adds_no_closers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Killed mid-turn on a person's prompt; a second daemon resumes it with
    `--resume-owed` and is killed the moment its pass ends, before anything else
    can write. The repair the resume made is on disk regardless, so a third mount
    closes nothing and adds only its own record of the reopen.

    Sabotage: drop the flush after a repairing resume from `_start`, and the second
    kill takes the closers with it, so the third mount repairs the same turn again.
    """
    _killed(tmp_path, monkeypatch, "turn")
    second = killed_when_ready(HOST, "resume")
    assert second["resumed"] == [ROOT]

    events = logged_events(ROOT)
    assert interrupted_turn_closers(events) == [], "the repair reached disk before the kill"
    [*_, last] = [event for event in events if event.type == "session/resumed"]
    repaired = Resumption.of(last)
    assert repaired.interrupted and repaired.closed > 0
    before = [event.type for event in events]

    async with supervised(tmp_path, monkeypatch, profile=PROFILE) as third:
        root = await third.start(ROOT)
        again = not_none(resumption_of(root.session), "the third reopen")
        assert (again.interrupted, again.closed) == (False, 0)
        await third._flush(root)
    assert logged_types(ROOT)[len(before) :] == ["session/resumed"]
