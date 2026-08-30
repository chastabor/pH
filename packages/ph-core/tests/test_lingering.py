"""P5-11 — lingering detection (I-6).

Gate: *simulated session end → clear diagnostic, not a silent failure.*

Half of that gate lives in `ph-app`, where a daemon has a socket to lose. This
half is the probe behind every sentence it prints: which environment variable
`$PH_RUNTIME` came from, whether logind reaps what is put there, and whether
this user is lingering — read from the marker directory `enable-linger`
maintains, so the answer costs a `stat` rather than a subprocess and survives a
host with no `loginctl` on `PATH`.

The marker directory is root-owned in reality, which is why `LINGER_DIR` is a
module constant: a test that had to write to `/var/lib/systemd/linger` could not
run at all, and one that skipped itself on the strength of the real host's
answer would prove whatever that host happened to say. The `reaped_host` fixture
in the repo-root `conftest.py` is where that redirection lives, once — reachable
from both packages' suites, which `packages/ph-app/tests/daemon_helpers.py` is
not.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from ph.lingering import lifetime, linger_state, socket_identity

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="a logind property; Windows reaps no runtime dir"
)

ReapedHost = Callable[..., Path]
"""The repo-root `reaped_host` fixture, spelled where it is read — structurally
rather than by `from conftest import …`, which resolves to the nearest conftest
on `sys.path` rather than to the root one the fixture lives in."""


# ---------------------------------------------------------------- the probe --


def test_linger_is_read_from_the_marker_the_command_creates(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """`enable-linger` writes a file; this reads that file, and nothing else.

    The equivalence is the whole departure from the row's wording: it asked for
    "`loginctl` state" and logind's own check is the presence of this path, so
    reading it directly is the same answer without a D-Bus round trip on a
    command a person runs *because* something is already broken.
    """
    reaped_host(linger=False)
    assert linger_state() == "off"

    (tmp_path / "linger" / "someone").touch()
    assert linger_state() == "on"


def test_a_host_with_no_marker_directory_is_unknown_rather_than_off(
    reaped_host: ReapedHost,
) -> None:
    """The reading that must never be flattened into "off".

    "Off" ends in an instruction; "unknown" ends in a question, and a host that
    keeps this state somewhere else — or runs no logind — deserves the question.
    Reporting it as off would tell a person to run a command that fixes nothing
    and then blame them when the daemon still went away.
    """
    reaped_host(linger=None)
    life = lifetime()
    assert life.linger == "unknown"
    assert life.survives_logout is None
    assert "show-user" in life.advice, "a question, not an instruction"
    # And the sentence still says the one thing that *is* known about this host:
    # the path is inside the tree logind reaps. Flattening `unknown` and
    # `outside the reaped tree` into one `None` produced "the socket outlives
    # logout" here, which is the only claim nobody could make about it.
    assert "which logind removes at logout" in life.explanation
    assert "the socket outlives logout" not in life.explanation


def test_the_tier_names_the_variable_it_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaped_host: ReapedHost
) -> None:
    """Derivation is reported, not re-derived: the tier decides which name to print."""
    reaped_host()
    assert dict(lifetime().describe())["derivation"].startswith("$PH_RUNTIME")

    # The same host with no override: `resolve_roots` falls to tier 1, and the
    # name printed follows the tier rather than a table in another module.
    monkeypatch.delenv("PH_RUNTIME")
    assert dict(lifetime().describe())["derivation"].startswith("$XDG_RUNTIME_DIR")

    # And tier 3, which is the one with no variable behind it at all.
    monkeypatch.delenv("XDG_RUNTIME_DIR")
    assert "no runtime dir in the env" in dict(lifetime().describe())["derivation"]


# --------------------------------------------------------- what survives what --


def test_a_reaped_runtime_dir_without_lingering_does_not_survive_logout(
    reaped_host: ReapedHost,
) -> None:
    """I-6's case, and the one sentence the whole row exists to produce."""
    reaped_host(linger=False)
    life = lifetime()
    assert life.reaped_at_logout
    assert life.survives_logout is False
    assert life.advice == "loginctl enable-linger someone"
    rows = dict(life.describe())
    assert "logind removes" in rows["survives logout"]
    assert rows["enable it"] == "loginctl enable-linger someone"


def test_lingering_is_what_makes_it_survive(reaped_host: ReapedHost) -> None:
    """And when it does, there is nothing left to advise."""
    reaped_host(linger=True)
    life = lifetime()
    assert life.survives_logout is True
    assert life.advice == ""
    assert "enable it" not in dict(life.describe())


def test_an_override_inside_the_runtime_dir_is_reaped_just_the_same(
    reaped_host: ReapedHost,
) -> None:
    """The reason this is a containment test rather than `tier == "xdg-runtime"`.

    `PH_RUNTIME=/run/user/1000/ph-dev` is tier `override` and lives exactly
    where logind is about to delete: the tier says where the answer came from,
    not where it points, and a check on the tier alone would call this one safe.
    """
    reaped_host(linger=False)
    life = lifetime()
    assert life.tier == "override"
    assert life.reaped_at_logout and life.survives_logout is False


def test_a_runtime_dir_outside_the_reaped_tree_is_undecided_not_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaped_host: ReapedHost
) -> None:
    """`/tmp/ph-$UID` keeps the socket; whether it keeps the *process* is logind's.

    Reported as unknown rather than yes, because `KillUserProcesses=yes` is
    systemd's own default and signals the daemon at logout wherever its socket
    happens to live. Lingering exempts the user from both, which is why one
    piece of advice covers two failures.
    """
    reaped_host()
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "elsewhere"))
    life = lifetime()
    assert not life.reaped_at_logout
    assert life.survives_logout is None
    assert "KillUserProcesses" in life.verdict()


def test_a_daemon_asks_about_its_own_socket_not_the_derived_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reaped_host: ReapedHost
) -> None:
    """`serve()` takes an explicit path, so the two are not always the same file."""
    runtime = reaped_host()
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "elsewhere"))
    inside = lifetime(runtime / "daemon.sock")
    assert inside.reaped_at_logout and inside.survives_logout is False
    assert lifetime().survives_logout is None, "the derived root is somewhere else entirely"


# ------------------------------------------------------------- the socket's id --


def test_socket_identity_tells_removed_apart_from_replaced(tmp_path: Path) -> None:
    """Why the watch compares an inode rather than asking whether a file exists.

    Logout removes the socket; logging back in and starting a second daemon
    puts a *different* socket at the same path. An existence check calls the
    second one a recovery, when it is two supervisors believing they own this
    user's roots — I-5's hazard, arriving by a door P5-03 does not cover.

    **The successor is created while the original is still there.** This test
    got the ordering wrong twice, and both times the mistake was the same one:
    an inode number is free for reuse the instant its last name goes, and ext4
    hands it straight back. Creating the replacement *after* the unlink —
    whether at the same path or at another one moved into place — lets it
    inherit the very number under test, so the assertion compares a value with
    itself. It cannot fail on ZFS, which is what this repository is developed on
    and which issues object ids monotonically; it fails every time on ext4,
    which is what CI runs.

    Two names that exist at the same moment cannot share an inode, because that
    is what a hard link is and `touch` does not make one. So the coexistence is
    the guarantee, and it is **asserted rather than assumed** below — a premise
    this test has now been wrong about twice should fail where it is stated, not
    three lines later where it is used.
    """
    path = tmp_path / "daemon.sock"
    path.touch()
    bound = socket_identity(path)
    assert bound is not None
    assert socket_identity(path) == bound, "unchanged between two reads"

    successor = tmp_path / "second.sock"
    successor.touch()
    theirs = socket_identity(successor)
    assert theirs != bound, "two files that coexist cannot share an inode"

    path.unlink()
    assert socket_identity(path) is None, "removed"

    os.replace(successor, path)
    assert socket_identity(path) == theirs
    assert socket_identity(path) != bound, "replaced, and not mistaken for recovered"


def test_socket_identity_does_not_follow_a_symlink_to_someone_elses(tmp_path: Path) -> None:
    """`lstat`, so a path swapped for a link reads as the link, not its target.

    Held against a symlink and its target that exist *together*, which is what
    makes the two identities necessarily different — the same correction the
    test above records.
    """
    real = tmp_path / "real.sock"
    real.touch()
    link = tmp_path / "ours.sock"
    link.symlink_to(real)

    assert socket_identity(link) is not None
    assert socket_identity(link) != socket_identity(real), "the link, not what it points at"
