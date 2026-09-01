"""Which interpreter runs the model's code (D8).

Three answers, and the difference between them is what model code can reach:

* **`managed`** (the default) — a venv at `$PH_CACHE/runtime-venv` holding
  `ph-runtime-guest`, `dill` and the Python skills, and nothing else. It is in
  `$PH_CACHE` rather than `$PH_HOME` because it is *rebuildable and large*: a
  user backing up their harness state should not be copying a venv, and deleting
  it should cost a rebuild and nothing else (Q1).
* **`host`** — the interpreter pH itself is running on. Fast, needs no `uv` and
  no network, and therefore what the test suite uses. It also puts `ph-core`,
  pydantic and Textual on the child's `sys.path`, so model code can import the
  harness's own modules. That reaches no live objects — a different process
  shares nothing — but it is a wider surface than the managed venv, and it is
  the reason this is not the default.
* **`override`** — `$PH_RUNTIME_PYTHON` or a configured path, for a deployment
  whose skills need a particular interpreter.

Staleness is a marker file, not a heuristic. It records the protocol version,
the guest package's version and the skill set; any difference rebuilds. A guest
one protocol behind would otherwise be discovered as a refused `boot` at the
first cell of somebody's session (D7).

@module ph_rlm.kernel.venv
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, TypeAlias

from .journal import argv_digest
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
    """The `ph-runtime-guest` source project, when pH is running from a checkout.

    Found from the *host's* own copy of the package, so a managed venv is built
    from the same source the mirror test just checked. In an installed wheel
    there is no project directory above the package and this returns `None`,
    which is the signal to install by requirement name instead.
    """
    import importlib.util

    spec = importlib.util.find_spec("ph_runtime")
    if spec is None or not spec.origin:
        return None
    candidate = Path(spec.origin).resolve().parent.parent.parent
    return candidate if (candidate / "pyproject.toml").exists() else None


@cache
def _host_can_import_guest(python: Path) -> bool:
    """Whether an interpreter can import the guest — asked once per path.

    Cached because it is a property of an interpreter, answered by spawning one, and
    both callers *refuse* on `False` — so a stale negative cannot strand a process
    that would otherwise have recovered, because there is no recovery path. A stale
    positive means someone uninstalled the guest from a live interpreter mid-run,
    which the next spawn reports anyway. The cache bounds the cost to anything that
    resolves repeatedly in one process, since each answer is a subprocess spawn.
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
    from importlib.metadata import PackageNotFoundError, version

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
    """The interpreter to spawn, building the managed venv if it is stale."""
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
    python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    marker_path = root / MARKER_NAME
    wanted = _marker(skills)
    if python.exists() and marker_path.exists():
        try:
            if json.loads(marker_path.read_text(encoding="utf-8")) == wanted:
                return RuntimeEnvironment(python=python, kind="managed", root=root)
        except (OSError, ValueError):
            pass
    _build(root, skills)
    marker_path.write_text(json.dumps(wanted, indent=2) + "\n", encoding="utf-8")
    return RuntimeEnvironment(python=python, kind="managed", root=root, rebuilt=True)


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
    requirements = [str(project) if project is not None else "ph-runtime-guest"]
    for spec in skills:
        # A local directory is installed **editable** (P3-18): the staleness
        # marker digests the specs, not their contents, so a non-editable local
        # skill would keep serving the bytes it had at build time after every
        # subsequent edit. A published requirement is installed normally.
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
            f"{completed.stderr.strip()[:2000]}"
        )
