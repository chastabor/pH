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

from ph_rlm.kernel.journal import OrphanJournal, argv_digest, process_start_token

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the start token is read from /proc"
)


def _journal(tmp_path: Path) -> OrphanJournal:
    return OrphanJournal(path=tmp_path / "processes.jsonl")


def test_a_spawn_is_recorded_with_a_start_token(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal.record(pid=os.getpid(), argv=["python", "-m", "ph_runtime"], namespace="a1")
    [record] = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert record["op"] == "spawn"
    assert record["pid"] == os.getpid()
    assert record["startToken"] == process_start_token(os.getpid())
    assert record["argv"] == argv_digest(["python", "-m", "ph_runtime"])


def test_a_reaped_child_is_not_swept(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    journal.record(pid=child.pid, argv=["x"], namespace=None)
    journal.forget(child.pid)
    report = journal.sweep()
    assert report.killed == ()
    assert journal.path.read_text().strip() == ""


def test_a_live_stray_is_killed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    stray = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        journal.record(pid=stray.pid, argv=["stray"], namespace="a1")
        report = journal.sweep()
        assert stray.pid in report.killed
        assert stray.wait(timeout=10) != 0
    finally:
        if stray.poll() is None:  # pragma: no cover
            stray.kill()
            stray.wait()


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
        journal.record(pid=child.pid, argv=["x"], namespace=None)
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
