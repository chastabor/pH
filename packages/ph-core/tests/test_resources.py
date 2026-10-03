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
import threading
import time
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph import resources
from ph.cordis import Context
from ph.orphans import process_alive
from ph.resources import (
    HARD_STOP_THREAD,
    Stopped,
    install_lifecycle,
    temporary_directory,
    until_signaled,
)

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
# `leave_on` run would kill the runner. What that cannot show is what installing
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
    handler assumed a loop. `leave_on` is patched because its whole job is to make
    the process die with the signal's own exit code, and that is not something a
    test can survive.
    """
    left: list[int] = []
    monkeypatch.setattr(resources, "leave_on", left.append)
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
    monkeypatch.setattr(resources, "leave_on", lambda _signum: None)
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
    monkeypatch.setattr(resources, "leave_on", lambda _signum: None)
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
    monkeypatch.setattr(resources, "leave_on", left.append)
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
    monkeypatch.setattr(resources, "leave_on", left.append)
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
    monkeypatch.setattr(resources, "leave_on", left.append)
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
    work. `leave_on` restores the default disposition and re-raises, so the process
    dies of the signal it was sent and the code says so.

    `os.kill` is patched: the real call is the one thing in this module a test
    cannot survive.
    """
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, number: killed.append((pid, number)))
    monkeypatch.setattr(signal, "signal", lambda number, handler: dispositions.append(handler))
    dispositions: list[Any] = []

    resources.leave_on(signal.SIGTERM)

    assert dispositions == [signal.SIG_DFL], "it left its own handler in place"
    assert killed == [(os.getpid(), signal.SIGTERM)]


# ----------------------------------------------------- stopping on a signal --
#
# The hard stop ends the process it runs in, so the two tests of it run a host in
# a subprocess. The graceful path, which returns, is tested in this one.

_STUCK_HOST = textwrap.dedent(
    """
    import subprocess, sys
    from functools import partial
    import anyio
    from ph.orphans import host_journal
    from ph.resources import until_signaled

    async def stuck() -> None:
        stray = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        journal = host_journal()
        assert journal is not None
        journal.record(pid=stray.pid, argv=["stray"], label=None)
        print(stray.pid, flush=True)
        # An unwind that never ends: the signal cancels the run, and the shield is
        # what a disposer stuck past every budget looks like from here.
        with anyio.CancelScope(shield=True):
            await anyio.sleep(60)

    anyio.run(partial(until_signaled, stuck, within=float(sys.argv[1])))
    """
)


def _stuck_host(tmp_path: Path, *, within: float) -> tuple[subprocess.Popen[str], int]:
    """A host whose unwind will not end, holding one journaled child. Returns the
    host and the child's pid, once the child is on the record."""
    env = {
        **os.environ,
        "PH_HOME": str(tmp_path / "home"),
        "PH_RUNTIME": str(tmp_path / "runtime"),
    }
    host = subprocess.Popen(
        [sys.executable, "-c", _STUCK_HOST, str(within)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert host.stdout is not None
    return host, int(host.stdout.readline())


def _gone_within(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return not process_alive(pid)


def _leave_nothing(host: subprocess.Popen[str], stray: int) -> None:
    if host.poll() is None:
        host.kill()
        host.wait()
    if process_alive(stray):
        os.killpg(stray, signal.SIGKILL)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_a_stop_that_overruns_its_bound_kills_what_the_host_owns_and_leaves(
    tmp_path: Path,
) -> None:
    """The bound on a graceful stop, and what it leaves behind: nothing.

    A host told to stop by `SIGTERM` unwinds, and its unwind is what stops each
    child it started. One whose unwind does not end is stopped from a thread, and a
    shell command it started must not outlive it, since nothing on Linux kills an
    unconfined one and nothing on macOS kills any. So the hard stop kills what the
    process still owns in the orphan journal, then exits `128 + signum`.

    Sabotage: drop `kill_owned` from `_leave_now`, and the stray outlives its host.
    """
    host, stray = _stuck_host(tmp_path, within=0.5)
    try:
        host.send_signal(signal.SIGTERM)

        assert host.wait(timeout=20) == 128 + signal.SIGTERM
        assert _gone_within(stray, 5), "the host's child outlived its hard stop"
    finally:
        _leave_nothing(host, stray)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
def test_a_second_signal_leaves_at_once_and_takes_the_hosts_children(tmp_path: Path) -> None:
    """A person who asks twice is not made to wait out the bound.

    The second signal is the hard stop's own act, with no wait: what the process
    owns is killed, and it leaves through the signal's default action, so the code
    is the signal's rather than one this process chose.

    Sabotage: drop the second-signal branch of `stop_on_signals`, and the host
    waits out its sixty-second bound.
    """
    host, stray = _stuck_host(tmp_path, within=60)
    try:
        host.send_signal(signal.SIGTERM)
        # Time for the first to be taken as the stop; a second before that is the
        # default action, which is the test failing the other way.
        time.sleep(0.5)
        host.send_signal(signal.SIGTERM)

        assert host.wait(timeout=20) == -signal.SIGTERM
        assert _gone_within(stray, 5), "the host's child outlived its second signal"
    finally:
        _leave_nothing(host, stray)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
async def test_a_signal_stops_a_one_shot_run_through_its_own_unwind() -> None:
    """The graceful path: the run is canceled, its unwind runs, and the bound is
    lifted once that unwind is over.

    A hard stop left armed after the stop it bounds would end a process that had
    already finished stopping cleanly, partway through whatever it did next.

    Sabotage: drop the `hard.cancel()` in `stop_on_signals`, and the thread is still
    armed when the run comes back.
    """
    unwound: list[str] = []

    async def run() -> str:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            await anyio.sleep(30)
        finally:
            unwound.append("unwound")
        return "finished"

    assert await until_signaled(run) == Stopped(signal.SIGTERM)
    assert unwound == ["unwound"]

    def armed() -> bool:
        return any(thread.name == HARD_STOP_THREAD for thread in threading.enumerate())

    deadline = time.monotonic() + 2
    while armed() and time.monotonic() < deadline:
        await anyio.sleep(0.01)
    assert not armed(), "the hard stop is still armed after the stop it bounds"


async def test_a_run_no_signal_reaches_returns_its_own_value() -> None:
    async def run() -> str:
        return "finished"

    assert await until_signaled(run) == "finished"


async def test_a_runs_own_failure_comes_back_as_itself() -> None:
    """Not in an `ExceptionGroup`, which is how the run's task group would raise it.

    The CLI names the refusals it turns into a sentence (`SessionBusy`, a model it
    cannot route), and a group matches none of them, so a session another process
    held came back as a traceback.

    Sabotage: let the run raise through the group, and this is an `ExceptionGroup`.
    """

    async def run() -> str:
        raise LookupError("no such session")

    with pytest.raises(LookupError, match="no such session"):
        await until_signaled(run)
