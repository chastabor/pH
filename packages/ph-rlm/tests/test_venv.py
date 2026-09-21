"""Which interpreter runs model code (D8).

The managed venv is not built here: doing so shells out to `uv` and reaches the
network, which is not a property worth asserting in every test run. What is
asserted is every *decision* around it — staleness, refusals, and the fact that
deleting `$PH_CACHE` costs a rebuild and nothing else.

## Why the guest-import probe is cached

`PythonCodeRuntime.environment()` already memoizes its own resolution, so a
deployment pays the probe once regardless. What the cache removes is the cost to
anything that resolves repeatedly in one process: **79 subprocess spawns and
1.45 s across the test suite**, re-answering one question about one path.
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
    """A refusal at resolve time, not a `boot` timeout in someone's session."""
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
    """The staleness check is a marker, not a heuristic.

    A guest one protocol behind would otherwise be found as a refused `boot` at
    the first cell of somebody's session (D7).
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
    """`_build` replaced by one that says it ran and leaves a usable binary.

    Every test here is about *whether* a build happened, never about what one
    produces — so the stand-in writes the one file `_current` looks for and
    records the root it was asked for.
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
    # The same skill set reuses what is there; a different one rebuilds.
    resolve_interpreter(cache=tmp_path, mode="managed", skills=("skill-a",))
    assert len(built) == 1
    resolve_interpreter(cache=tmp_path, mode="managed", skills=("skill-a", "skill-b"))
    assert len(built) == 2


def test_the_guest_project_is_found_from_the_hosts_own_copy() -> None:
    """A managed venv is built from the same source the mirror test just checked."""
    project = guest_project_dir()
    assert project is not None
    assert (project / "pyproject.toml").exists()
    assert project.name == "ph-runtime-guest"


def test_a_second_process_waits_rather_than_rebuilding_over_the_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5 — `$PH_CACHE` is shared and `_build` opens by deleting the tree.

    `PythonCodeRuntime.environment` serializes this *process*; the cache is
    shared by every pH on the machine — a second host, a `phern` in another
    terminal, a subagent's daemon. Two of them in `_build` is one `shutil.rmtree`
    against the other's install: the loser gets a venv missing whatever had not
    been written, and then writes a marker saying it is current, so every later
    run uses it and nothing ever reports why.

    The lock is a *file* lock for that reason, and the check is repeated inside
    it: the process that loses the race wants the venv the winner just built, not
    a second build of it. Simulated with the lock held by this test, which is
    what another process holding it looks like from here.
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
    # And with nobody holding it, the build goes ahead as before.
    resolve_interpreter(cache=tmp_path, mode="managed")
    assert built == [root]


def test_a_venv_another_process_just_built_is_not_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of F5's lock: the waiter re-checks rather than rebuilding.

    Building is the expensive thing, so a process that waited out a build wants
    what the winner produced. Modelled by having `_build` — which here stands in
    for the process that held the lock — leave a current venv behind, and
    asserting the second call neither builds nor claims to have.
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
    """F5 — the marker digests the guest's *version*, not its source.

    A version does not move during development, so a non-editable install kept
    serving the bytes it had at build time through every later edit of the guest:
    working on it meant deleting the venv by hand to see a change, or not
    noticing that you had not. The skills beside it already take `--editable` for
    exactly this argument, spelled out at that line since it was written.
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
