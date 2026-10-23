"""P13-09 and P13-11 at the doctor: `phern doctor sessions [--repair]`, and the lease
mechanism `phern doctor` prints.

The gates are the operator's: a damaged log is named with its line and the exit code
says so; `--repair` cuts it back to what reads, keeps the whole file beside it, and
leaves a log a running process holds; and the lease section says `flock` and proves
it with a lock taken where the leases live.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from daemon_helpers import left_with_a_damaged_child, private_runtime, run_phern

from ph.paths import resolve_roots
from ph.persistence import damage_in
from ph.testing import hold_session

pytestmark = pytest.mark.anyio


async def test_doctor_names_the_lease_mechanism_and_probes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P13-11: the lease is `flock` or nothing, said where a person reads it and
    proved in the directory the leases live in, before any profile mounts."""
    private_runtime(tmp_path, monkeypatch)
    result = await run_phern("doctor", "--profile", "headless")
    assert result.exit_code == 0, result.output
    assert "session lease" in result.output
    assert "flock(2)" in result.output
    assert "a lock was taken and released" in result.output


async def test_doctor_sessions_names_the_line_a_log_stops_reading_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = await left_with_a_damaged_child(tmp_path, monkeypatch)
    result = await run_phern("doctor", "sessions")
    assert result.exit_code == 1, result.output
    assert path.stem in result.output and "Expecting" in result.output
    assert "2 stored log(s)" in result.output
    assert "--repair" in result.output, "the fix, named"
    assert damage_in(path) is not None, "a look changes nothing"


async def test_doctor_sessions_repair_cuts_the_log_back_and_keeps_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = await left_with_a_damaged_child(tmp_path, monkeypatch)
    whole = path.read_bytes()
    result = await run_phern("doctor", "sessions", "--repair")
    assert result.exit_code == 0, result.output
    assert "cut at line 3" in result.output
    assert damage_in(path) is None
    assert path.with_name(f"{path.name}.damaged").read_bytes() == whole
    again = await run_phern("doctor", "sessions")
    assert again.exit_code == 0 and "every log reads" in again.output


async def test_doctor_sessions_leaves_a_log_another_process_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer appending behind the cut would put `seq` backwards, so a leased log is
    reported and left; the exit code says a damaged log remains."""
    path = await left_with_a_damaged_child(tmp_path, monkeypatch)
    whole = path.read_bytes()
    with hold_session(resolve_roots().sessions_dir(), path.stem):
        result = await run_phern("doctor", "sessions", "--repair")
    assert result.exit_code == 1, result.output
    assert "active in another process" in result.output
    assert path.read_bytes() == whole
