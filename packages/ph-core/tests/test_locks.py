"""`ph.locks` — and the gate that keeps it the only way in.

Why the invariant matters is `ph.locks`' own docstring; what makes it want a
*gate* is that breaking it is invisible at the call site. Four of five modules
remembered `thread_local=False`; `ph_rlm.harness.service` did not, and was
correct only because the one method that took the lock never crossed a thread.
A reviewer cannot see that and a parse can.
"""

from __future__ import annotations

import ast
import errno
import fcntl
import pathlib

import pytest
from workspace_layout import workspace_modules

from ph.locks import LockBusy, LockUnsupported, acquire_file_lock, describe_lock, file_lock

_ALLOWED = {"ph/locks.py"}
"""The one shipped module that may reach for `filelock`.

Package-relative, the name `workspace_modules` reports, so it survives a layout
move. Tests are outside that walk entirely — a test standing in for another
process is entitled to the real thing."""


def test_no_shipped_module_imports_filelock_but_ph_locks() -> None:
    """One place owns `thread_local=False`, so no site can omit it.

    **The import, not the call.** Matching `FileLock(` would miss
    `from filelock import FileLock as Lock`, and miss `SoftFileLock`,
    `UnixFileLock` and `BaseFileLock`, which share the same thread-local
    default. "Nobody but `ph.locks` talks to `filelock`" is the actual rule and
    is one node type simpler to check.

    Parsed rather than grepped, because `builders.py` explains in prose that two
    suites once spelled `FileLock(...)` by hand and a line scan cannot tell that
    sentence from the thing it warns about. The substring pre-gate only skips
    files that provably cannot import it, which takes the walk from 332 ms to
    single digits — `workspace_modules` is 268 files and four other gates
    already parse them.

    Sabotage: import anything from `filelock` in a shipped module.
    """
    offenders: list[str] = []
    for name, path in workspace_modules():
        if name in _ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        if "filelock" not in text:
            continue
        for node in ast.walk(ast.parse(text, filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                reached = [node.module or ""]
            elif isinstance(node, ast.Import):
                reached = [alias.name for alias in node.names]
            else:
                continue
            if any(one.split(".")[0] == "filelock" for one in reached):
                offenders.append(f"{name}:{node.lineno}")

    assert offenders == [], f"reach filelock through ph.locks: {offenders}"


def test_a_held_lock_is_refused_rather_than_waited_on_forever(tmp_path: pathlib.Path) -> None:
    """The second acquire is refused rather than waited on, and says what was
    locked.

    A message and nothing else — `LockBusy` deliberately carries no fields,
    because every shipped caller already holds the path and raises its own
    domain error from this one.
    """
    path = tmp_path / "nested" / "thing.lock"
    release = acquire_file_lock(path, timeout=0, what="the thing")
    try:
        with pytest.raises(LockBusy) as raised:
            acquire_file_lock(path, timeout=0, what="the thing")
    finally:
        release()

    assert "the thing" in str(raised.value)
    # filelock makes the parent on every acquire, which is why `ph.locks` does
    # not: a lock beside a file in a directory that does not exist yet is the
    # ordinary case, and it is handled a layer down.
    assert path.parent.is_dir()


def test_the_lock_is_released_when_the_body_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "thing.lock"
    with pytest.raises(RuntimeError), file_lock(path, timeout=0, what="the thing"):
        raise RuntimeError("boom")

    # Free again, which a `finally`-less implementation would not manage.
    acquire_file_lock(path, timeout=0, what="the thing")()


# ------------------------------------------------------- P13-11: flock or nothing --


def test_a_filesystem_without_flock_is_refused_rather_than_softened(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The library's unix lock switches to an *existence* lock when `flock` answers
    "not supported" — a file whose presence is the claim, which a killed process
    leaves behind and the next daemon is refused by, for every session it held.
    `ph.locks` pins that switch off: the refusal is `LockUnsupported`, naming the
    path and the reason, and not `LockBusy`, since nobody holds anything.

    Sabotage: drop `fallback_to_soft=False`, and this test sees the lock taken.
    """

    def no_flock(_fd: int, _operation: int) -> None:
        raise OSError(errno.ENOSYS, "flock is not implemented here")

    # The filesystem's answer, at the one call the library makes for it.
    monkeypatch.setattr(fcntl, "flock", no_flock)
    path = tmp_path / "thing.lock"
    with pytest.raises(LockUnsupported) as refused:
        acquire_file_lock(path, timeout=0, what="the thing")
    assert str(path) in str(refused.value) and "flock" in str(refused.value)
    assert not isinstance(refused.value, LockBusy)


def test_doctor_describes_the_lease_mechanism_by_probing_it(tmp_path: pathlib.Path) -> None:
    """What `phern doctor` prints: the mechanism by name, the directory, its
    filesystem, and a lock taken and let go there for real — so the answer is the
    filesystem's and not this module's opinion of it."""
    rows = dict(describe_lock(tmp_path / "leases"))
    assert "flock" in rows["mechanism"]
    assert rows["directory"] == str(tmp_path / "leases")
    assert rows["filesystem"]
    assert rows["probe"] == "a lock was taken and released"
    # The probe left the directory a lease would make, and nothing held in it.
    acquire_file_lock(tmp_path / "leases" / "any.lock", timeout=0, what="a lease")()
