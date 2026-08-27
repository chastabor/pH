"""P1-20 — §4.9 resource ownership.

Gates: *`SIGTERM` unwinds within the grace period; the lint catches a raw
`Popen`.*

The lint is the load-bearing half. Invariant I2 says cleanup is structural
rather than remembered, and that property survives exactly as long as nobody
acquires an artifact outside the seam. A convention would hold until the next
plugin author who has not read §4.9; a test holds indefinitely.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ph.cordis import Context
from ph.resources import temporary_directory

pytestmark = pytest.mark.anyio

# Dotted call targets that acquire an artifact directly, and the seam that
# should hand it out instead. Matched on the full dotted path rather than the
# bare name, so `sessions.fork(...)` is not mistaken for `os.fork()`.
_FORBIDDEN_CALLS = {
    "subprocess.Popen": (
        "spawn through ctx.subprocess, so the child is terminated and reaped with its scope"
    ),
    "subprocess.run": "spawn through ctx.subprocess",
    "subprocess.call": "spawn through ctx.subprocess",
    "subprocess.check_output": "spawn through ctx.subprocess",
    "anyio.open_process": "spawn through ctx.subprocess",
    "asyncio.create_subprocess_exec": "spawn through ctx.subprocess",
    "asyncio.create_subprocess_shell": "spawn through ctx.subprocess",
    "os.fork": "pH does not fork; spawn through ctx.subprocess",
    "os.system": "spawn through ctx.subprocess",
    "tempfile.mkdtemp": (
        "use ph.resources.temporary_directory, so the path is removed with its scope"
    ),
    "tempfile.mkstemp": "use ph.resources.temporary_directory",
    "tempfile.TemporaryDirectory": (
        "use ph.resources.temporary_directory: TemporaryDirectory cleans up on a "
        "weakref.finalize, which is GC-timed rather than scope-timed"
    ),
}

# The seams themselves. Each *is* the module that hands the artifact out.
_SEAM_OWNERS = {
    "ph/resources.py": {"tempfile.mkdtemp"},
    "ph/seams/subprocess.py": {"anyio.open_process"},
}


def _dotted(node: ast.expr) -> str | None:
    parts: list[str] = []
    current: ast.expr | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _core_files() -> list[Path]:
    import ph

    return sorted(Path(ph.__path__[0]).rglob("*.py"))


def test_artifacts_are_acquired_only_through_their_seam() -> None:
    import ph

    root = Path(ph.__path__[0]).parent
    offenders: list[str] = []
    for path in _core_files():
        relative = path.relative_to(root).as_posix()
        allowed = _SEAM_OWNERS.get(relative, set())
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _dotted(node.func)
            if target is None or target in allowed:
                continue
            reason = _FORBIDDEN_CALLS.get(target)
            if reason is not None:
                offenders.append(f"{relative}:{node.lineno} calls {target} - {reason}")
    assert offenders == [], "\n".join(offenders)


async def test_a_temporary_directory_disposes_with_its_scope() -> None:
    root = Context()
    scope = root.scope("agent")
    path = await temporary_directory(scope)
    assert path.is_dir()
    # 0700 and unguessable: a world-readable scratch directory is a leak of
    # whatever the agent put in it.
    assert oct(path.stat().st_mode)[-3:] == "700"

    await scope.dispose()
    # Gone when the scope went, not when the garbage collector got round to it.
    assert not path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_sigterm_unwinds_the_scope_within_the_grace_period(tmp_path: Path) -> None:
    marker = tmp_path / "disposed.txt"
    program = textwrap.dedent(
        f"""
        import os, signal, threading, anyio
        from ph.cordis import Context
        from ph.resources import install_lifecycle

        root = Context()
        scope = root.scope("agent")
        scope.add_disposer(lambda: open({str(marker)!r}, "w").write("disposed"))
        install_lifecycle(root, grace_seconds=5.0)

        # Signal ourselves from another thread so the handler runs on the main one.
        threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        anyio.run(anyio.sleep, 3)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, timeout=30, check=False
    )
    assert marker.exists(), (
        f"SIGTERM did not unwind the scope; stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert marker.read_text() == "disposed"
    # And it left for real rather than hanging: a shutdown path that can hang is
    # a shutdown path that will.
    assert completed.returncode != 0


async def test_effects_release_in_reverse_even_when_one_fails() -> None:
    root = Context()
    released: list[str] = []

    def broken() -> None:
        released.append("broken")
        raise RuntimeError("teardown failed")

    root.add_disposer(lambda: released.append("first"))
    root.add_disposer(broken)
    root.add_disposer(lambda: released.append("last"))

    await root.dispose()
    # A failing disposer is logged and the unwind continues: one plugin's bad
    # teardown must not strand every artifact registered before it.
    assert released == ["last", "broken", "first"]
