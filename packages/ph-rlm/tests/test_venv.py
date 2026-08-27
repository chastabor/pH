"""Which interpreter runs model code (D8).

The managed venv is not built here: doing so shells out to `uv` and reaches the
network, which is not a property worth asserting in every test run. What is
asserted is every *decision* around it — staleness, refusals, and the fact that
deleting `$PH_CACHE` costs a rebuild and nothing else.
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


def test_a_changed_skill_set_is_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[Path] = []

    def fake_build(root: Path, skills: object) -> None:
        built.append(root)
        binary = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("")

    monkeypatch.setattr("ph_rlm.kernel.venv._build", fake_build)
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
