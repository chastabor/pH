"""The orphan journal (F5): strays nothing else can clean up.

Every other cleanup path in pH is structural, and none of them runs under
`SIGKILL`. This is the one that does, and the property that matters most is the
one about *restraint*: a pid whose start token no longer matches is a different
process, and killing it would be far worse than leaving a stray behind.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ph.cordis import Context
from ph.keys import SUBPROCESS
from ph.orphans import OrphanJournal, argv_digest, process_start_token
from ph.seams.subprocess import SubprocessSpawnSpec
from ph.testing import MountProfile

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="the start token is read from /proc"
    ),
]


def _journal(tmp_path: Path) -> OrphanJournal:
    return OrphanJournal(path=tmp_path / "processes.jsonl")


def _dead_pid() -> int:
    """A pid that is certainly not running — a process we started and reaped."""
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    return gone.pid


def _orphaned(journal: OrphanJournal, pid: int) -> None:
    """Re-write the journal as if a *previous*, now-dead pH had recorded `pid`.

    The sweep leaves a live owner's children alone, so a stray in a test has to
    come from a run that is over — which is the only way a real one ever does.
    """
    records = [json.loads(line) for line in journal.path.read_text().splitlines()]
    for record in records:
        if record.get("pid") == pid:
            record["owner"] = _dead_pid()
            record["ownerToken"] = None
    journal.path.write_text("".join(json.dumps(one) + "\n" for one in records), encoding="utf-8")


def test_a_spawn_is_recorded_with_a_start_token(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.record(pid=os.getpid(), argv=["python", "-m", "ph_runtime"], label="a1")
    [record] = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert record["op"] == "spawn"
    assert record["pid"] == os.getpid()
    assert record["startToken"] == process_start_token(os.getpid())
    assert record["argv"] == argv_digest(["python", "-m", "ph_runtime"])


def test_a_reaped_child_is_not_swept(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    journal.record(pid=child.pid, argv=["x"], label=None)
    journal.forget(child.pid)
    report = journal.sweep()
    assert report.killed == ()
    assert journal.path.read_text().strip() == ""


def test_a_live_stray_is_killed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        journal.record(pid=stray.pid, argv=["stray"], label="a1")
        _orphaned(journal, stray.pid)
        report = journal.sweep()
        assert stray.pid in report.killed
        assert stray.wait(timeout=10) != 0
    finally:
        if stray.poll() is None:  # pragma: no cover
            stray.kill()
            stray.wait()


def test_a_live_owners_children_are_left_alone(tmp_path: Path) -> None:
    """One journal per user per boot, so it holds other runs' *live* children.

    A daemon's `git` child and a hard-killed run's stray sit in the same file,
    and "is the pid alive" cannot tell them apart — a sweep that used only that
    would kill the daemon's work, which is the same mistake the start token
    exists to prevent one level down. A record whose owner is still running
    belongs to a process that will clean it up itself.

    The record stays in the journal rather than being compacted away: the sweep
    that runs after that owner finally dies is the one that has to find it.
    """
    journal = _journal(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        # Recorded by *this* process, which is alive — the ordinary case.
        journal.record(pid=child.pid, argv=["held"], label=None)

        report = journal.sweep()

        assert report.killed == () and report.held == (child.pid,)
        assert child.poll() is None, "somebody else's live child was not touched"
        assert str(child.pid) in journal.path.read_text(), "and it is still on the record"
    finally:
        child.kill()
        child.wait()


def test_a_reused_pid_is_spared(tmp_path: Path) -> None:
    """The whole reason a token is recorded at all.

    A journalled pid that now belongs to something else must be left alone: this
    process is a live pid with a token that will not match a forged record, and
    if the sweep killed it the test would not finish.
    """
    journal = _journal(tmp_path)
    journal.path.write_text(
        json.dumps(
            {"op": "spawn", "pid": os.getpid(), "startToken": "0", "argv": "x", "namespace": "a"}
        )
        + "\n"
    )
    report = journal.sweep()
    assert report.killed == ()
    assert os.getpid() in report.stale


def test_the_journal_is_compacted_to_what_is_outstanding(tmp_path: Path) -> None:
    """Swept at every start, so left uncompacted it would grow forever."""
    journal = _journal(tmp_path)
    for _ in range(20):
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        journal.record(pid=child.pid, argv=["x"], label=None)
        journal.forget(child.pid)
    assert len(journal.path.read_text().splitlines()) == 40
    journal.sweep()
    assert journal.path.read_text().strip() == ""


def test_a_missing_journal_sweeps_to_nothing(tmp_path: Path) -> None:
    assert _journal(tmp_path / "nowhere").sweep().killed == ()


def test_a_corrupt_line_does_not_stop_the_sweep(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.path.write_text('not json\n{"op": "spawn"}\n{"op": "spawn", "pid": "x"}\n')
    assert journal.sweep().killed == ()


async def test_the_subprocess_seam_journals_a_spawn_and_forgets_a_reap(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The wiring, which is the half a journal with no caller does not have.

    `ph_rlm.kernel` has journalled its guest since F5; every *other* child pH
    spawns — git, jj, agentfs, the sandbox backend, `bash`, `!` — went through
    `ctx.subprocess` and was recorded nowhere. One seam spawns them all, so one
    place closes it.

    Asserted through the real seam rather than by calling `record`: what was
    missing was never the journal.
    """
    ctx: Context = await mount()
    seam = ctx.require(SUBPROCESS)
    journal = OrphanJournal(path=tmp_path / "processes.jsonl")
    seam.journal = journal

    scope = ctx.scope("spawner")
    child = await seam.spawn(
        SubprocessSpawnSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(60)"), cwd=tmp_path
        ),
        scope=scope,
    )
    spawned = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [one["op"] for one in spawned] == ["spawn"]
    assert spawned[0]["pid"] == child.pid
    assert spawned[0]["owner"] == os.getpid(), "so another run's sweep leaves it alone"

    await scope.dispose()

    settled = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert [one["op"] for one in settled] == ["spawn", "reap"], "the pair closes on disposal"
    assert journal.sweep().killed == (), "so a later sweep has nothing to do"


async def test_a_seam_with_nowhere_to_journal_still_spawns(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A diagnostic must not be able to stop the harness working.

    A read-only `$PH_RUNTIME` is a deployment fact, and refusing to run `git`
    over it would trade a working harness for a tidier one. The hole is open
    again in that deployment, which `phern doctor` says out loud — see `_describe`.
    """
    ctx: Context = await mount()
    seam = ctx.require(SUBPROCESS)
    seam.journal = None

    outcome = await seam.run(
        SubprocessSpawnSpec(argv=(sys.executable, "-c", "print('ok')"), cwd=tmp_path)
    )

    assert outcome.stdout.strip() == "ok"


async def test_run_closes_the_record_without_waiting_for_its_scope(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The reap is known when the child exits, not when its scope unwinds.

    `run` spawns, waits and reaps — but it used to leave the *effect* registered,
    and the journal's `forget` lives in that effect's disposer. So the record
    stayed open until the owning scope went away, which for a daemon root is
    hours: `processes.jsonl` grew a live `spawn` line per command, `SweepReport.held`
    became every command the process had ever run, and `_effects` grew beside it.

    Asserted without disposing the scope, which is the whole point — the previous
    version passed if you disposed first.
    """
    ctx: Context = await mount()
    seam = ctx.require(SUBPROCESS)
    journal = OrphanJournal(path=tmp_path / "processes.jsonl")
    seam.journal = journal
    scope = ctx.scope("runner")

    for _ in range(3):
        await seam.run(
            SubprocessSpawnSpec(argv=(sys.executable, "-c", "pass"), cwd=tmp_path), scope=scope
        )

    ops = [json.loads(line)["op"] for line in journal.path.read_text().splitlines()]
    assert ops.count("spawn") == 3 and ops.count("reap") == 3, ops
    assert journal.sweep().held == (), "nothing is still outstanding"
