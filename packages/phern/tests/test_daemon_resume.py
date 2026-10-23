"""P13-03 to P13-07 and P13-10 — a daemon resumes what the last one owed.

A fresh daemon mounts the roots the live-roots index says owe work (`sessions/resume`,
`--resume-owed`), and each mount is the whole of a resume: the repair that closed
the crashed turn is on disk before any work, a turn nobody will send again is
continued up to a bound, a child's ending wakes the idle parent it reports to, a
root held for a credential holds its children with it, and a schedule fires through
the seam's door, which puts the index after the log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import (
    A_PERSON,
    cut_turn,
    left_mid_turn,
    left_with_a_damaged_child,
    route_needs,
    running,
    set_phase,
    supervised,
    unplugged,
    until,
)

from ph.json import as_obj, as_seq
from ph.keys import GOALS, INTENTS, SCHEDULE, SUBAGENTS
from ph.persistence import resumption_of
from ph.seams import schedule as schedule_module
from ph.seams.goals import Goal
from ph.seams.schedule import Schedule, ScheduleService, scheduled_source
from ph.seams.subagents import ChildNotice, SubagentService, record_ended
from ph.session import Session
from ph.session.kinds import SHELL_COMMAND
from ph.testing import admitted_child, log_event, stored_types
from ph_app.daemon.recovery import ROOT_RESUME_LIMIT, interrupted_resumes
from ph_app.daemon.supervisor import Root
from ph_app.wire import source_of

pytestmark = pytest.mark.anyio

A_SCHEDULE = scheduled_source(Schedule(id="nightly", kind="interval", spec="60000", prompt="go"))
KEY = "PH_P13_ROOT_KEY"


def _spliced(session: Session) -> list[object]:
    """Every message spliced into this log's inbox, oldest first."""
    return [
        message
        for event in session.events_from(0)
        if event.type == "agent/inbox/spliced"
        for message in as_seq(event.data.get("inserted"))
    ]


def _continuations(session: Session) -> list[str]:
    """The text of every continuation the supervisor put in this root's inbox."""
    return [
        str(as_obj(as_seq(as_obj(message).get("content"))[0]).get("text"))
        for message in _spliced(session)
        if source_of(message)[1] == "ph_app.daemon.supervisor"
    ]


def _answered(root: Root) -> bool:
    """Whether the root's last turn ran to an answer."""
    return root.last_turn == "completed"


# ------------------------------------------------- P13-04: the cut-short turn --


async def test_a_turn_a_schedule_sent_is_continued_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn nobody will send again — a schedule's — is continued by the mount
    itself: the resume closes it, the supervisor tells the root the turn above was
    cut short and which attempt this is, and the ring runs it with no client.

    Sabotage: drop `_continue_interrupted` from `_start`, and the root sits idle.
    """
    await left_mid_turn(tmp_path, monkeypatch, "sched", source=A_SCHEDULE)
    async with supervised(tmp_path, monkeypatch) as second:
        root = await second.start("sched")
        await until(lambda: _answered(root), what="the continued turn to run")
        [said] = _continuations(root.session)
        assert f"attempt 1 of {ROOT_RESUME_LIMIT}" in said
        assert root.status == "idle"
        assert interrupted_resumes(root.session) == 0, "the answer forgave the restart"


async def test_a_persons_interrupted_turn_waits_for_them_unless_the_daemon_is_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A person whose prompt was cut short is at a keyboard somewhere and sends it
    again; a turn run unasked on their behalf is one they did not ask for.
    `--continue-interrupted` widens it for a deployment where nobody is.

    And the repair is on disk before anything else (P13-10): a second daemon finds
    the turn closed and closes nothing, where before the closers rode in the buffer
    until the next barrier — which a root that ran no turn never reached, so every
    boot repaired the same turn again.
    """
    await left_mid_turn(tmp_path, monkeypatch, "person", source=A_PERSON)
    async with supervised(tmp_path, monkeypatch) as second:
        root = await second.start("person")
        assert not root.agent.inbox.has_pending and root.status == "idle"
        assert _continuations(root.session) == []
        resumed = resumption_of(root.session)
        assert resumed is not None and resumed.interrupted and resumed.turn == 1
        on_disk = stored_types(root.ctx, "person")
        assert "session/resumed" in on_disk and on_disk[-1] == "session/resumed"
        assert "turn/end" in on_disk, "the closers reached disk with the resume record"
    async with supervised(tmp_path, monkeypatch) as third:
        root = await third.start("person")
        again = resumption_of(root.session)
        assert again is not None and not again.interrupted and again.closed == 0
    await left_mid_turn(tmp_path, monkeypatch, "person-told", source=A_PERSON)
    async with supervised(tmp_path, monkeypatch) as told:
        told.continue_interrupted = True
        root = await told.start("person-told")
        assert root.agent.inbox.has_pending, "told to, the daemon continues a person's turn"


async def test_an_open_goal_is_reason_enough_to_continue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/autonomous` runs only on `agent/turn-stopping`, which a killed turn never
    reached: the goal was open, its gates unrun, and nothing would run another turn.
    With a goal open the person's own cut-short prompt is continued too — they asked
    for the work to carry on until the gates pass."""
    await left_mid_turn(tmp_path, monkeypatch, "goal", source=A_PERSON, before=_open_goal)
    async with supervised(tmp_path, monkeypatch) as second:
        root = await second.start("goal")
        await until(lambda: _answered(root), what="the continued turn to run")
        assert len(_continuations(root.session)) == 1


def _open_goal(root: Root) -> None:
    root.ctx.require(GOALS).set(root.session, Goal(id="g1", objective="finish", gates=[]))


async def test_a_resume_that_only_settled_an_intent_continues_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair that settled an intent left open *between* turns is `interrupted`
    and cut no turn (`Resumption.turn`). The scheduled turn before it ran to its
    end, so nothing is owed — and a resumer that walked back to the last
    `turn/start` it could find ran that finished turn again.

    Sabotage: decide on `resumed.interrupted` alone, and the finished turn runs twice.
    """
    async with supervised(tmp_path, monkeypatch) as first:
        root = await first.start("between")
        cut_turn(root, source=A_SCHEDULE)
        log_event(root.session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
        await root.ctx.require(INTENTS).open(
            root.session, SHELL_COMMAND, {"command": "make", "surface": False}
        )
    async with supervised(tmp_path, monkeypatch) as second:
        root = await second.start("between")
        resumed = resumption_of(root.session)
        assert resumed is not None and resumed.interrupted and resumed.turn is None
        assert _continuations(root.session) == []
        assert not root.agent.inbox.has_pending


async def test_the_root_ladder_gives_up_past_its_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A turn cut short `ROOT_RESUME_LIMIT` times in a row with no answer between is
    in front of something that keeps stopping: the root is failed in its own log,
    flushed, and reads `failed` — so a resumer that runs every minute does not spend
    one more attempt per restart, and `sessions/resume` leaves it on disk unless
    forced. A goal keeps it on the index, which is how the resumer sees it at all.

    Sabotage: count the resumes from the ladder's `attempts` rather than the log,
    and every boot is attempt one.
    """

    def cut_before(root: Root) -> None:
        _open_goal(root)
        for _ in range(ROOT_RESUME_LIMIT):
            log_event(
                root.session,
                "session/resumed",
                {"events": 1, "interrupted": True, "closed": 1, "turn": 1},
            )

    await left_mid_turn(tmp_path, monkeypatch, "looping", source=A_SCHEDULE, before=cut_before)
    async with supervised(tmp_path, monkeypatch) as second:
        root = await second.start("looping")
        assert root.status == "failed" and root.recovery.failed
        assert _continuations(root.session) == []
        assert "supervisor/failed" in stored_types(root.ctx, "looping")
    async with supervised(tmp_path, monkeypatch) as third:
        report = await third.resume_owed(all_owed=True)
        [row] = report.sessions
        assert (row.session_id, row.outcome) == ("looping", "spent")
        assert "looping" not in third.roots
        forced = await third.resume_owed(["looping"], force=True)
        [row] = forced.sessions
        assert (row.outcome, row.status) == ("resumed", "failed")


# ------------------------------------------------------- P13-03: the pass --


async def test_resume_owed_mounts_what_is_owed_and_says_what_became_of_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One row per root: the owed one is mounted and continued, one named by id is
    mounted whether or not it owes anything, a second pass finds them mounted, and
    an id that is not a root is refused with the reason rather than failing the pass.
    """
    await left_mid_turn(tmp_path, monkeypatch, "owed", source=A_SCHEDULE)
    async with supervised(tmp_path, monkeypatch) as first:
        await first.start("quiet")
        family = await first.start("family")
        child = admitted_child(family.ctx, family.session, "r1")
        await first._flush(family)
        child_id = child.id
    async with supervised(tmp_path, monkeypatch) as second:
        report = await second.resume_owed(["quiet", child_id], all_owed=True)
        rows = {row.session_id: row for row in report.sessions}
        # `family` owes its child's work, which is on the index; `quiet` owes
        # nothing and is mounted because it was named.
        assert set(rows) == {"owed", "family", "quiet", child_id}
        assert rows["owed"].outcome == rows["family"].outcome == "resumed"
        assert rows["quiet"].outcome == "resumed" and rows["quiet"].status == "idle"
        assert rows[child_id].outcome == "refused" and "root" in rows[child_id].detail
        assert set(second.roots) == {"owed", "family", "quiet"}
        await until(lambda: _answered(second.roots["owed"]), what="the owed turn")
        again = await second.resume_owed(all_owed=True)
        assert {row.outcome for row in again.sessions} <= {"mounted"}


async def test_readiness_waits_for_the_first_resume_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--resume-owed` runs the pass once the door is open, and `ready` says no until
    it ends: an orchestrator that sent work to a daemon still mounting what the last
    one owed would race the mounts."""
    async with unplugged(tmp_path, monkeypatch) as server:
        server.serving = True
        assert server.probe("ready").ok
        server.resumed = False
        reply = server.probe("ready")
        assert not reply.ok and any("resume pass" in reason for reason in reply.reasons)
    await left_mid_turn(tmp_path, monkeypatch, "booted", source=A_SCHEDULE)
    async with running(tmp_path, resume_owed=True) as daemon:
        await until(lambda: daemon.running.resumed, what="the resume pass to finish")
        assert daemon.running.probe("ready").ok
        assert "booted" in daemon.running.supervisor.roots, "mounted by the daemon itself"


# ------------------------------------------- P13-05: a child's ending wakes --


async def test_a_childs_ending_wakes_the_idle_parent_it_reports_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent resumed past the turn that was waiting is idle when its readmitted
    child finishes. The notice used to be injected without a wake, so the parent sat
    "running" on a pending message for the life of the daemon; it rings now.

    Sabotage: deliver the notice with `inject`, or drop the `agent/woken` listener
    from `_start`, and the turn never runs.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("parent")
        child = admitted_child(root.ctx, root.session, "r1")
        notice = ChildNotice(text="[child r1 finished: all good]", summary="r1 finished")

        await record_ended(root.ctx, child, "done", notice=notice)

        await until(lambda: _answered(root), what="the parent to read the notice")
        assert root.status == "idle"


async def test_a_ring_with_nothing_pending_runs_no_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wake whose message a running turn already took is a wake for nothing, and
    `run()` over an empty inbox opens and closes an empty turn. The root's task
    checks the inbox before it drives."""
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("rung")
        before = len(root.session.events_from(0))
        for _ in range(3):
            root.ring()
        await supervisor._flush(root)
        assert [e.type for e in root.session.events_from(before)] == []


# ------------------------------------------- P13-06: the hold covers children --


async def test_a_credential_hold_covers_the_roots_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root held for a credential does not readmit its children: readmitted under
    a root that cannot run, they would spend their ladders on a key a person could
    supply in a second. The key that arrives readmits them.

    Sabotage: drop the credential term from `_start`'s guard, and the sweep runs
    while the root is held.
    """
    swept: list[str] = []
    original = SubagentService._resume

    async def spy(self: Any, parent: Any, *, retry_limit: int) -> Any:  # noqa: ANN401
        swept.append(parent.id)
        return await original(self, parent, retry_limit=retry_limit)

    monkeypatch.setattr(SubagentService, "_resume", spy)
    route_needs(monkeypatch, KEY)
    async with running(tmp_path) as daemon:
        root = await daemon.root("held")
        assert root.status == "needs-credential"
        assert swept == [], "children were swept under a held root"
        client = await daemon.client()
        await client.call("credentials/store", sessionId=root.id, name=KEY, value="shh")
        await until(lambda: root.needs_credential is None, what="the hold to lift")
        assert swept == [root.agent.id]


# ------------------------------------------------- P13-07: the durable tick --


async def test_the_index_moves_only_once_the_tick_and_its_prompt_are_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fire_due`: claim, deliver, write, then tell the index. The index used to be
    told at the claim, while the tick was still in a buffer: a kill there left the
    file saying "later" over a log saying "due", and the run was lost. A write that
    fails leaves the index where it was and the root unwritten.

    Sabotage: reindex inside `claim`, and the index moves with nothing on disk.
    """
    told: list[list[str]] = []
    reindex = ScheduleService.reindex

    def spying(
        self: ScheduleService, session: Session, *, now: int | None = None, new: bool = False
    ) -> None:
        if now is not None:
            told.append(stored_types(supervisor.roots[session.id].ctx, session.id))
        reindex(self, session, now=now, new=new)

    monkeypatch.setattr(ScheduleService, "reindex", spying)
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("cron")
        seam = root.ctx.require(SCHEDULE)
        seam.create(root.session, Schedule(id="s", kind="interval", spec="60000", prompt="go"))
        made = seam.states(root.session)["s"].created_at

        assert await supervisor.tick(now=made + 90_000) == ["s"]

        # In that order, and both: the woken turn may already have started behind
        # them while the write was in flight.
        [on_disk] = told
        assert "schedule/tick" in on_disk and "agent/inbox/spliced" in on_disk, on_disk
        assert on_disk.index("schedule/tick") < on_disk.index("agent/inbox/spliced")
        delivered = [source_of(message)[1] for message in _spliced(root.session)]
        assert delivered == ["ph.seams.schedule"], "a scheduled prompt is the schedule's"
        await until(lambda: _answered(root), what="the scheduled turn")
        set_phase(root, "idle")

        async def refused(_ctx: object, _session: Session) -> bool:
            return False

        monkeypatch.setattr(schedule_module, "session_written", refused)
        assert await supervisor.tick(now=made + 150_000) == ["s"]
        assert len(told) == 1, "the index moved over a log that did not take the tick"
        assert "cron" in supervisor.unwritten


# ------------------------------------------------------- P13-09: damaged logs --


async def test_a_damaged_child_log_lists_its_root_damaged_and_a_pass_leaves_it_to_the_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One child log that will not read used to refuse the whole family. The root is
    listed for the damage with the child named and no holds — what the child owes is
    unknown — `resume --all` leaves it on disk and says what moves it, a root named by
    id still mounts, and nothing is written into the file that will not read."""
    path = await left_with_a_damaged_child(tmp_path, monkeypatch)
    child_id, before = path.stem, path.read_bytes()
    async with supervised(tmp_path, monkeypatch) as second:
        listing = await second.owed(rebuild=True)
        [row] = listing.sessions
        assert (row.session_id, row.holds, row.damaged) == ("family", [], [child_id])
        passed = await second.resume_owed(all_owed=True)
        [outcome] = passed.sessions
        assert outcome.outcome == "damaged"
        assert "phern doctor sessions --repair" in outcome.detail
        assert not second.roots, "nothing to resume, so nothing mounted"
        [named] = (await second.resume_owed(["family"])).sessions
        assert named.outcome == "resumed"
        mounted = second.roots["family"]
        assert mounted.ctx.require(SUBAGENTS).unreadable("family") == (child_id,)
        [row] = (await second.owed()).sessions
        assert row.mounted and row.damaged == [child_id], "the mounted entry says so too"
        [again] = (await second.resume_owed(all_owed=True)).sessions
        assert again.outcome == "mounted", "mounted is asked before damaged"
    assert path.read_bytes() == before, "no tombstone into a log that will not read"
