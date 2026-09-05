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
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph import resources
from ph.cordis import Context
from ph.resources import install_lifecycle, temporary_directory

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


# ------------------------------------------------- installing and removing --
#
# The signal *path* is proved above, in a subprocess, because a test that let
# `_leave` run would kill the runner. What that cannot show is what installing
# leaves behind, and it is the half an embedded host depends on: `ph` is also a
# library, and a harness that permanently captured `SIGINT` from the process
# that mounted it would take the interrupt away from its owner.


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_installing_and_releasing_leaves_the_handlers_as_they_were() -> None:
    """`install_lifecycle` returns a disposer, and it has to actually restore.

    Asserted by identity against the handlers in place before, for both signals:
    a release that dropped `SIGINT` back to `SIG_DFL` rather than to whatever was
    there would silently disarm a host's own Ctrl-C after pH had been mounted once.
    """
    before = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}

    release = install_lifecycle(Context())
    installed = {number: signal.getsignal(number) for number in before}
    assert all(installed[number] is not before[number] for number in before)
    assert len(set(installed.values())) == 1, "one handler serves both signals"

    release()

    assert {number: signal.getsignal(number) for number in before} == before


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_a_signal_with_no_loop_running_disposes_where_it_stands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `atexit`-shaped path: a signal that arrives before the loop exists.

    There is nowhere to schedule a teardown, so it runs synchronously through
    `anyio.run` rather than being dropped — which is what would happen if the
    handler assumed a loop. `_leave` is patched because its whole job is to make
    the process die with the signal's own exit code, and that is not something a
    test can survive.
    """
    left: list[int] = []
    monkeypatch.setattr(resources, "_leave", left.append)
    root = Context()
    disposed: list[str] = []
    root.add_disposer(lambda: disposed.append("root"))
    release = install_lifecycle(root)

    try:
        signal.raise_signal(signal.SIGTERM)
    finally:
        release()

    assert disposed == ["root"]
    assert left == [signal.SIGTERM], "and it left with the signal it was given"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_the_root_is_disposed_once_however_many_signals_arrive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`finished` is what stops a second `SIGTERM` re-entering a teardown.

    A disposer that runs twice is a disposer that can fail the second time — a
    directory already removed, a process already reaped — and a shutdown that
    raises is one that does not finish. The same flag is why `atexit` is a no-op
    after a signal has already unwound the root.
    """
    monkeypatch.setattr(resources, "_leave", lambda _signum: None)
    root = Context()
    disposed: list[str] = []
    root.add_disposer(lambda: disposed.append("root"))
    release = install_lifecycle(root)

    try:
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)
    finally:
        release()

    assert disposed == ["root"], "the second signal unwound the root again"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_a_host_is_told_before_the_teardown_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """`on_signal` runs first, and that ordering is the point of the hook.

    A front end uses it to stop drawing before the scopes it is drawing *from*
    go away; called after the unwind it would be told about a teardown it had
    already rendered the wreckage of.
    """
    monkeypatch.setattr(resources, "_leave", lambda _signum: None)
    order: list[str] = []
    root = Context()
    root.add_disposer(lambda: order.append("disposed"))
    release = install_lifecycle(root, on_signal=lambda number: order.append(f"told:{number}"))

    try:
        signal.raise_signal(signal.SIGTERM)
    finally:
        release()

    assert order == [f"told:{int(signal.SIGTERM)}", "disposed"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
async def test_a_signal_inside_the_loop_schedules_rather_than_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordinary case, and the deadlock it is written against.

    A handler runs on the main thread, *interrupting* the event loop — so one
    that awaited the teardown there would be waiting on the loop it had just
    stopped. It schedules a task instead and returns, and the task is kept
    referenced because one with no reference can be collected mid-flight,
    abandoning the very teardown this exists to run.
    """
    left: list[int] = []
    monkeypatch.setattr(resources, "_leave", left.append)
    root = Context()
    disposed: list[str] = []
    root.add_disposer(lambda: disposed.append("root"))
    release = install_lifecycle(root)

    try:
        signal.raise_signal(signal.SIGTERM)
        assert disposed == [], "the handler blocked the loop it needed"
        # Yield until the scheduled unwind has run; it is a task on this loop.
        for _ in range(50):
            await anyio.sleep(0)
            if disposed:
                break
    finally:
        release()

    assert disposed == ["root"]
    assert left == [signal.SIGTERM]


def _explode() -> None:
    raise RuntimeError("teardown failed")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_a_teardown_that_raises_still_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shutdown that cannot fail to finish is the whole point of the grace path.

    One plugin's bad disposer must not turn `SIGTERM` into a process that stays
    up: the failure is logged and the signal is re-raised anyway. Both routes
    into the teardown say this — the no-loop one here, the scheduled one below —
    because the handler picks between them on whether a loop happens to be
    running, which is not something the operator chose.
    """
    left: list[int] = []
    monkeypatch.setattr(resources, "_leave", left.append)
    root = Context()
    root.add_disposer(_explode)
    release = install_lifecycle(root)

    try:
        signal.raise_signal(signal.SIGTERM)
    finally:
        release()

    assert left == [signal.SIGTERM]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
async def test_a_teardown_that_raises_inside_the_loop_still_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled half of the claim above."""
    left: list[int] = []
    monkeypatch.setattr(resources, "_leave", left.append)
    root = Context()
    root.add_disposer(_explode)
    release = install_lifecycle(root)

    try:
        signal.raise_signal(signal.SIGTERM)
        for _ in range(50):
            await anyio.sleep(0)
            if left:
                break
    finally:
        release()

    assert left == [signal.SIGTERM]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_leaving_re_raises_the_signal_under_the_default_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**The exit code is the thing being protected.**

    A harness that caught `SIGTERM`, cleaned up and then exited 0 would tell
    everything watching it — a supervisor, a shell, CI — that it finished its
    work. `_leave` restores the default disposition and re-raises, so the process
    dies of the signal it was sent and the code says so.

    `os.kill` is patched: the real call is the one thing in this module a test
    cannot survive.
    """
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(resources.os, "kill", lambda pid, number: killed.append((pid, number)))
    monkeypatch.setattr(signal, "signal", lambda number, handler: dispositions.append(handler))
    dispositions: list[Any] = []

    resources._leave(signal.SIGTERM)

    assert dispositions == [signal.SIG_DFL], "it left its own handler in place"
    assert killed == [(os.getpid(), signal.SIGTERM)]
