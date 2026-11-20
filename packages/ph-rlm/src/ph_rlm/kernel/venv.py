"""Which interpreter runs the code that the model writes (D8).

There are three answers. They differ in what model code can reach:

* **`managed`** (the default) is a venv at `$PH_CACHE/runtime-venv`. It holds
  `ph-runtime-guest`, `dill` and the Python skills, and nothing else. It is in
  `$PH_CACHE` and not in `$PH_HOME` because it is *rebuildable and large*. A
  user who backs up the harness state must not have to copy a venv, and deleting
  the venv must cost a rebuild and nothing else (Q1).
* **`host`** is the interpreter that pH itself runs on. It is fast and needs no
  `uv` and no network, so the test suite uses it. It also puts `ph-core`,
  pydantic and Textual on the `sys.path` of the child, so model code can import
  the modules of the harness. That reaches no live objects, because a different
  process shares nothing. But it is a wider surface than the managed venv, and
  that is why `host` is not the default.
* **`override`** is `$PH_RUNTIME_PYTHON` or a configured path, for a deployment
  whose skills need a particular interpreter.

A marker file decides staleness, not a heuristic. The marker records the
protocol version, the version of the guest package, the Python version of the
host and the skill set. Any difference causes a rebuild. Without the marker, a
guest that is one protocol version behind would be found as a refused `boot` at
the first cell of a user's session (D7).

@module ph_rlm.kernel.venv
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, TypeAlias

from ph.characters import head
from ph.locks import LockBusy, file_lock
from ph.orphans import argv_digest

from .protocol import PROTOCOL_VERSION

__all__ = [
    "INTERPRETER_ENV",
    "MARKER_NAME",
    "VENV_DIR",
    "InterpreterMode",
    "RuntimeEnvironment",
    "RuntimeVenvError",
    "guest_project_dir",
    "resolve_interpreter",
]

log = logging.getLogger("ph_rlm.kernel.venv")

INTERPRETER_ENV = "PH_RUNTIME_PYTHON"
VENV_DIR = "runtime-venv"
MARKER_NAME = ".ph-runtime.json"

_LOCK_TIMEOUT = 900.0
"""How long to wait for another process that is building the same venv (F5).

The value is generous on purpose. It is the opposite trade from
`ph.seams._json_index.LOCK_TIMEOUT`. That timeout limits a stall on the event
loop, where giving up costs a late run and waiting costs the daemon. This wait
runs one time, in a worker thread. It waits for a `uv pip install` of the guest
and every configured skill, which takes minutes on a cold wheel cache. Giving up
early would mean building a second copy over the first, and that is the
collision that the lock exists to prevent.
"""

InterpreterMode: TypeAlias = Literal["managed", "host"]


class RuntimeVenvError(RuntimeError):
    """The runtime interpreter could not be resolved or built."""


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """The interpreter that will run cells, and where it came from."""

    python: Path
    kind: Literal["managed", "host", "override"]
    root: Path | None = None
    rebuilt: bool = False

    def describe(self) -> str:
        return f"{self.kind}: {self.python}"


def guest_project_dir() -> Path | None:
    """The `ph-runtime-guest` source project, when pH runs from a checkout.

    The project is found from the copy of the package that the *host* has. A
    managed venv is therefore built from the same source that the mirror test
    just checked. In an installed wheel there is no project directory above the
    package, and this function returns `None`. That is the signal to install by
    requirement name instead.
    """
    spec = importlib.util.find_spec("ph_runtime")
    if spec is None or not spec.origin:
        return None
    candidate = Path(spec.origin).resolve().parent.parent.parent
    return candidate if (candidate / "pyproject.toml").exists() else None


@cache
def _host_can_import_guest(python: Path) -> bool:
    """Whether an interpreter can import the guest. Asked one time for each path.

    The answer is cached because it is a property of an interpreter, and getting
    it means spawning one. Both callers *refuse* on `False`. A stale negative
    therefore cannot strand a process that would otherwise have recovered,
    because there is no recovery path. A stale positive means that someone
    uninstalled the guest from a live interpreter in the middle of a run, and
    the next spawn reports that anyway. The cache limits the cost for anything
    that resolves repeatedly in one process, because each answer is a subprocess
    spawn.
    """
    try:
        completed = subprocess.run(
            [str(python), "-c", "import ph_runtime, sys; sys.exit(0)"],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _marker(skills: Sequence[str]) -> dict[str, object]:
    try:
        guest = version("ph-runtime-guest")
    except PackageNotFoundError:  # pragma: no cover — a checkout without install
        guest = "unknown"
    return {
        "protocol": PROTOCOL_VERSION,
        "guest": guest,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "skills": argv_digest(sorted(skills)),
    }


def resolve_interpreter(
    *,
    cache: Path,
    mode: InterpreterMode = "managed",
    skills: Sequence[str] = (),
    override: str | None = None,
) -> RuntimeEnvironment:
    """The interpreter to spawn. Builds the managed venv if it is stale."""
    configured = override or os.environ.get(INTERPRETER_ENV)
    if configured:
        python = Path(configured)
        if not python.exists():
            raise RuntimeVenvError(f"{INTERPRETER_ENV} points at {python}, which does not exist")
        if not _host_can_import_guest(python):
            raise RuntimeVenvError(
                f"{python} cannot import ph_runtime; install ph-runtime-guest into it "
                "or unset the override to use the managed venv"
            )
        return RuntimeEnvironment(python=python, kind="override")

    if mode == "host":
        python = Path(sys.executable)
        if not _host_can_import_guest(python):
            raise RuntimeVenvError(
                "the host interpreter cannot import ph_runtime; install ph-runtime-guest "
                'or use python: "managed"'
            )
        return RuntimeEnvironment(python=python, kind="host")

    return _managed(cache / VENV_DIR, skills)


def _managed(root: Path, skills: Sequence[str]) -> RuntimeEnvironment:
    """The managed venv, built if the venv on disk is not the one that is wanted.

    **The build runs under a file lock, not a process lock** (F5).
    `PythonCodeRuntime.environment` serializes *this* process, but `$PH_CACHE`
    is shared by every pH on the machine: a second host, a `phern` in another
    terminal, the daemon of a subagent. `_build` starts with
    `shutil.rmtree(root)`. With two builders, one deletes the tree that the
    other is installing into. The loser gets a venv that is missing whatever had
    not been written yet, and the marker that it then writes says that the venv
    is current. Every later run uses that venv.

    The check is made twice, before the lock and again inside it. Building is
    the expensive step and waiting is the common case: the process that loses
    the race wants the venv that the winner just built, and not a second build
    of it. `_LOCK_TIMEOUT` is generous because the wait is for a
    `uv pip install`.
    """
    python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    marker_path = root / MARKER_NAME
    wanted = _marker(skills)
    if _current(python, marker_path, wanted):
        return RuntimeEnvironment(python=python, kind="managed", root=root)
    try:
        with file_lock(f"{root}.lock", timeout=_LOCK_TIMEOUT, what="the runtime venv"):
            if _current(python, marker_path, wanted):
                # The process that held the lock built it. It is reported as
                # not rebuilt, because this process did not rebuild it.
                return RuntimeEnvironment(python=python, kind="managed", root=root)
            _build(root, skills)
            marker_path.write_text(json.dumps(wanted, indent=2) + "\n", encoding="utf-8")
    except LockBusy as busy:
        # **The message must not say "remove the lock file".** It used to give
        # that advice, and that is the one instruction that can cause the
        # damage that the lock prevents. `flock` is held by the open descriptor
        # (`ph.persistence.lease`). A dead process has therefore already
        # released the lock, and a leftover `.lock` file is not a lock. A *live*
        # process is still installing, and deleting the file would let this
        # process `rmtree` the tree while the other process installs into it.
        raise RuntimeVenvError(
            f"another process has been building the runtime venv at {root} for over "
            f"{_LOCK_TIMEOUT:.0f}s; wait for it, or stop that process and try again"
        ) from busy
    return RuntimeEnvironment(python=python, kind="managed", root=root, rebuilt=True)


def _current(python: Path, marker_path: Path, wanted: dict[str, object]) -> bool:
    """Whether the venv on disk is the one that `wanted` describes."""
    if not (python.exists() and marker_path.exists()):
        return False
    try:
        return bool(json.loads(marker_path.read_text(encoding="utf-8")) == wanted)
    except (OSError, ValueError):
        return False


def _build(root: Path, skills: Sequence[str]) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeVenvError(
            "uv is not on PATH, so the managed runtime venv cannot be built; install uv, "
            f"or set {INTERPRETER_ENV} to an interpreter that already has ph-runtime-guest"
        )
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.parent.mkdir(parents=True, exist_ok=True)
    log.info("ph_rlm.kernel: building the runtime venv at %s", root)
    _run([uv, "venv", "--seed", str(root)])

    project = guest_project_dir()
    # **The guest is installed editable when it comes from a checkout** (F5).
    # This is the same rule that the skills below state, for the same reason.
    # `_marker` records the *version* of the guest and not its source, and the
    # version does not move during development. A non-editable install
    # therefore kept serving the bytes that it had at build time through every
    # later edit. A developer who worked on the guest had to delete the venv by
    # hand to see a change, or did not notice that the change was missing.
    requirements = ["--editable", str(project)] if project is not None else ["ph-runtime-guest"]
    for spec in skills:
        # A local directory is installed **editable** (P3-18). The staleness
        # marker digests the specs and not their contents, so a non-editable
        # local skill would keep serving the bytes that it had at build time
        # after every later edit. A published requirement is installed normally.
        local = Path(spec).expanduser()
        if local.is_dir():
            requirements.extend(["--editable", str(local)])
        else:
            requirements.append(spec)
    _run([uv, "pip", "install", "--python", str(root), *requirements])


def _run(argv: list[str]) -> None:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeVenvError(f"{argv[0]} failed: {error}") from error
    if completed.returncode != 0:
        raise RuntimeVenvError(
            f"{' '.join(argv[:3])} failed with status {completed.returncode}:\n"
            f"{head(completed.stderr.strip(), 2000)}"
        )
