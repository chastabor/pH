"""`ph.workspace_caches` — the indexes a gone workspace left behind, and only those.

A worktree is a workspace of its own, so a profile that runs children in worktrees
left a code graph and a text index per tree, keyed by a digest nothing could turn
back into a path. The sweep is the rule that lets them go, and these tests are its
refusals: a live workspace's entry stays, a recently used one stays (a drive
unplugged for now), one with no record stays (one being made), and a cache swept
today is not walked again.
"""

from __future__ import annotations

import os
from pathlib import Path
from time import time

from ph.workspace_caches import STALE_AFTER, SWEEP_EVERY, WORKSPACE_FILE, digest, use


def _later() -> float:
    """A clock past the age guard for everything used now. Asked when a test runs,
    not when the module loads: a suite's first minutes would eat the margin."""
    return time() + STALE_AFTER + 60.0


def test_only_the_entry_of_a_workspace_that_is_gone_is_removed(tmp_path: Path) -> None:
    """Sabotage: drop the existence test, and the live workspace's index goes too."""
    base, live, here = tmp_path / "cache", tmp_path / "live", tmp_path / "here"
    live.mkdir()
    here.mkdir()
    kept = use(base, live)
    gone = use(base, tmp_path / "gone")
    unrecorded = base / digest(str(tmp_path / "making"))
    unrecorded.mkdir()

    use(base, here, now=_later())

    assert (kept / WORKSPACE_FILE).read_text(encoding="utf-8") == str(live)
    assert unrecorded.exists() and not gone.exists()


def test_a_recently_used_entry_outlives_its_workspace_for_a_while(tmp_path: Path) -> None:
    """A tree absent for now — an unmounted drive, a checkout being moved — is not a
    tree gone for good, and taking its index costs a full re-index when it is back.

    Sabotage: drop the age guard, and the entry is removed the moment its tree is.
    """
    base = tmp_path / "cache"
    unplugged = use(base, tmp_path / "unplugged")

    use(base, tmp_path, now=time() + SWEEP_EVERY + 60.0)

    assert unplugged.exists()


def test_a_cache_swept_today_is_not_walked_again(tmp_path: Path) -> None:
    """Every index call goes through `use`, and a walk of every entry each time is
    work a daily sweep against a week's guard does not need.

    Sabotage: sweep on every call, and the stale entry goes a day early.
    """
    base = tmp_path / "cache"
    gone = use(base, tmp_path / "gone")
    later = _later()
    os.utime(base / ".swept", (later - 60.0, later - 60.0))

    use(base, tmp_path, now=later)

    assert gone.exists()
