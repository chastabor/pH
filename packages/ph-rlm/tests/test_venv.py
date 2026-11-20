"""Which interpreter runs model code (D8).

These tests do not build the managed venv. A build shells out to `uv` and
reaches the network, and that is not a property to assert in every test run.
The tests assert every *decision* around the build: staleness, refusals, and the
fact that deleting `$PH_CACHE` costs a rebuild and nothing else.

## Why the guest-import probe is cached

`PythonCodeRuntime.environment()` already memoizes its own resolution, so a
deployment pays for the probe one time in any case. The cache removes the cost
for anything that resolves repeatedly in one process: **79 subprocess spawns and
1.45 s across the test suite**, which answered one question about one path again
and again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ph_rlm.kernel.protocol import PROTOCOL_VERSION
from ph_rlm.kernel.venv import (
    INTERPRETER_ENV,
    MARKER_NAME,
    VENV_DIR,
    RuntimeVenvError,
    guest_project_dir,
    resolve_interpreter,
)


def test_the_host_interpreter_is_usable_in_a_checkout(tmp_path: Path) -> None:
    environment = resolve_interpreter(cache=tmp_path, mode="host")
    assert environment.kind == "host"
    assert environment.python == Path(sys.executable)


def test_an_override_that_cannot_import_the_guest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal comes at resolve time, and not as a `boot` timeout in a user's
    session."""
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv(INTERPRETER_ENV, str(fake))
    with pytest.raises(RuntimeVenvError, match="cannot import ph_runtime"):
        resolve_interpreter(cache=tmp_path, mode="host")


def test_an_override_that_does_not_exist_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INTERPRETER_ENV, str(tmp_path / "nope"))
    with pytest.raises(RuntimeVenvError, match="does not exist"):
        resolve_interpreter(cache=tmp_path, mode="managed")


def test_a_current_marker_avoids_a_rebuild(tmp_path: Path) -> None:
    """A marker decides staleness, not a heuristic.

    Without the marker, a guest that is one protocol version behind would be
    found as a refused `boot` at the first cell of a user's session (D7).
    """
    from importlib.metadata import version

    root = tmp_path / VENV_DIR
    binary = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    (root / MARKER_NAME).write_text(
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "guest": version("ph-runtime-guest"),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "skills": "e3b0c44298fc1c14",
            }
        )
    )
    environment = resolve_interpreter(cache=tmp_path, mode="managed")
    assert environment.kind == "managed"
    assert environment.rebuilt is False


def test_a_marker_from_an_older_protocol_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / VENV_DIR
    binary = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    (root / MARKER_NAME).write_text(json.dumps({"protocol": PROTOCOL_VERSION - 1}))

    built: list[Path] = []
    monkeypatch.setattr("ph_rlm.kernel.venv._build", lambda root, skills: built.append(root))
    environment = resolve_interpreter(cache=tmp_path, mode="managed")
    assert built == [root]
    assert environment.rebuilt is True


def _records_builds(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace `_build` with a stand-in that records that it ran and leaves a
    usable binary.

    Every test here is about *whether* a build happened, and never about what a
    build produces. The stand-in therefore writes the binary that `_current`
    looks for, and records the root that it was asked for.
    """
    built: list[Path] = []

    def fake_build(root: Path, skills: object) -> None:
        built.append(root)
        binary = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("")

    monkeypatch.setattr("ph_rlm.kernel.venv._build", fake_build)
    return built


def test_a_changed_skill_set_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = _records_builds(monkeypatch)
    resolve_interpreter(cache=tmp_path, mode="managed", skills=("skill-a",))
    assert len(built) == 1
    # The same skill set reuses the venv that is there. A different set rebuilds.
    resolve_interpreter(cache=tmp_path, mode="managed", skills=("skill-a",))
    assert len(built) == 1
    resolve_interpreter(cache=tmp_path, mode="managed", skills=("skill-a", "skill-b"))
    assert len(built) == 2


def test_the_guest_project_is_found_from_the_hosts_own_copy() -> None:
    """A managed venv is built from the same source that the mirror test just checked."""
    project = guest_project_dir()
    assert project is not None
    assert (project / "pyproject.toml").exists()
    assert project.name == "ph-runtime-guest"


def test_a_second_process_waits_rather_than_rebuilding_over_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: `$PH_CACHE` is shared, and `_build` starts by deleting the tree.

    `PythonCodeRuntime.environment` serializes this *process*. The cache is
    shared by every pH on the machine: a second host, a `phern` in another
    terminal, the daemon of a subagent. With two of them in `_build`, the
    `shutil.rmtree` of one runs against the install of the other. The loser gets
    a venv that is missing whatever had not been written, and then writes a
    marker that says the venv is current. Every later run uses that venv, and
    nothing ever reports why.

    The lock is a *file* lock for that reason, and the check is repeated inside
    it: the process that loses the race wants the venv that the winner just
    built, and not a second build of it. The test simulates this by holding the
    lock itself. From here, that looks the same as another process that holds
    it.
    """
    from filelock import FileLock

    root = tmp_path / VENV_DIR
    built = _records_builds(monkeypatch)
    monkeypatch.setattr("ph_rlm.kernel.venv._LOCK_TIMEOUT", 0.1)

    held = FileLock(f"{root}.lock", thread_local=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    held.acquire()
    try:
        with pytest.raises(RuntimeVenvError, match="has been building"):
            resolve_interpreter(cache=tmp_path, mode="managed")
    finally:
        held.release()

    assert built == [], "it built over a venv another process was installing into"
    # With no holder, the build goes ahead as before.
    resolve_interpreter(cache=tmp_path, mode="managed")
    assert built == [root]


def test_a_venv_another_process_just_built_is_not_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the F5 lock: the waiter checks again and does not rebuild.

    Building is the expensive step, so a process that waited for a build wants
    what the winner produced. Here `_build` stands in for the process that held
    the lock, and it leaves a current venv behind. The test asserts that the
    second call neither builds nor claims that it built.
    """
    from ph_rlm.kernel.venv import _marker

    root = tmp_path / VENV_DIR
    built = _records_builds(monkeypatch)
    first = resolve_interpreter(cache=tmp_path, mode="managed")
    assert first.rebuilt is True
    assert _marker(()) == json.loads((root / MARKER_NAME).read_text())

    second = resolve_interpreter(cache=tmp_path, mode="managed")

    assert built == [root], "the current venv was rebuilt"
    assert second.rebuilt is False


def test_a_guest_from_a_checkout_is_installed_editable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5: the marker records the *version* of the guest, not its source.

    A version does not move during development. A non-editable install therefore
    kept serving the bytes that it had at build time through every later edit of
    the guest. A developer who worked on the guest had to delete the venv by
    hand to see a change, or did not notice that the change was missing. The
    skills beside it already take `--editable` for exactly this reason, and the
    comment at that line has said so since it was written.
    """
    from ph_rlm.kernel.venv import _build, guest_project_dir

    project = guest_project_dir()
    assert project is not None, "this suite runs from a checkout"

    argv: list[list[str]] = []
    monkeypatch.setattr("ph_rlm.kernel.venv.shutil.which", lambda _name: "/usr/bin/uv")
    monkeypatch.setattr("ph_rlm.kernel.venv._run", lambda command: argv.append(command))

    _build(tmp_path / VENV_DIR, ())

    install = next(one for one in argv if "install" in one)
    assert "--editable" in install, "a checkout's guest was installed frozen at build time"
    assert str(project) in install
