"""P1-20 — §4.9 resource ownership.

Gates: *a signal unwinds a one-shot run, bounded by the hard stop; the lint
catches a raw `Popen`.*

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
    run_until_signaled,
    temporary_directory,
    until_signaled,
)
from ph.testing import settled

pytestmark = pytest.mark.anyio

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")

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


@posix_only
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

    with pytest.raises(SystemExit):
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
        stray = subprocess.Popen(["sleep", "60"], start_new_session=True)
        journal = host_journal()
        assert journal is not None
        journal.record(pid=stray.pid, argv=["stray"], label=None)
        print(stray.pid, flush=True)
        try:
            await anyio.sleep_forever()
        finally:
            # An unwind that never ends: the signal cancels the run, and the shield
            # is what a disposer stuck past every budget looks like from here.
            print("stopping", flush=True)
            with anyio.CancelScope(shield=True):
                await anyio.sleep(60)

    anyio.run(partial(until_signaled, stuck, within=float(sys.argv[1])))
    """
)


def _host(tmp_path: Path, program: str, *args: str) -> subprocess.Popen[str]:
    """`program` run as a host process of its own, with its own `$PH_RUNTIME`."""
    return subprocess.Popen(
        [sys.executable, "-c", program, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PH_RUNTIME": str(tmp_path / "runtime")},
    )


def _stuck_host(tmp_path: Path, *, within: float) -> tuple[subprocess.Popen[str], int]:
    """A host whose unwind will not end, holding one journaled child. Returns the
    host and the child's pid, once the child is on the record."""
    host = _host(tmp_path, _STUCK_HOST, str(within))
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


@posix_only
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
    host, stray = _stuck_host(tmp_path, within=0.1)
    try:
        host.send_signal(signal.SIGTERM)

        assert host.wait(timeout=20) == 128 + signal.SIGTERM
        assert _gone_within(stray, 5), "the host's child outlived its hard stop"
    finally:
        _leave_nothing(host, stray)


@posix_only
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
        # The first taken as the stop; a second before that is the default action,
        # which is the test failing the other way.
        assert host.stdout is not None
        assert host.stdout.readline().strip() == "stopping"
        host.send_signal(signal.SIGTERM)

        assert host.wait(timeout=20) == -signal.SIGTERM
        assert _gone_within(stray, 5), "the host's child outlived its second signal"
    finally:
        _leave_nothing(host, stray)


@posix_only
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

    await settled(lambda: not armed(), "the hard stop to be disarmed after the stop it bounds")


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


_ONE_SHOT_HOST = textwrap.dedent(
    """
    import sys
    import anyio
    from ph.resources import run_until_signaled

    async def run() -> str:
        print("ready", flush=True)
        try:
            await anyio.sleep_forever()
        finally:
            print("unwound")
        return "finished"

    run_until_signaled(run, on_stop=lambda signum: print("stopped", signum, file=sys.stderr))
    print("returned", flush=True)
    """
)


@posix_only
def test_a_one_shot_host_stopped_by_a_signal_leaves_through_it(tmp_path: Path) -> None:
    """What every one-shot host does once a signal has stopped its run, said once.

    The run unwinds, the host is told, what the unwind wrote reaches the pipe, and
    the process dies of the signal it was sent rather than returning: a host that
    returned would exit 0, and `--mode rpc` would wait at exit for the thread still
    blocked on stdin. `unwound` is printed unflushed to a pipe, so it is only seen
    if the flush happened before the default action.

    Sabotage: drop the `sys.stdout.flush()` in `run_until_signaled`, and `unwound`
    is lost; return instead of `leave_on`, and `returned` is printed.
    """
    host = _host(tmp_path, _ONE_SHOT_HOST)
    try:
        assert host.stdout is not None
        assert host.stdout.readline().strip() == "ready"
        host.send_signal(signal.SIGTERM)
        out, err = host.communicate(timeout=20)
    finally:
        if host.poll() is None:
            host.kill()
            host.wait()

    assert host.returncode == -signal.SIGTERM
    assert out.split() == ["unwound"], f"stdout was {out!r}"
    assert f"stopped {int(signal.SIGTERM)}" in err


def test_a_one_shot_run_no_signal_reaches_returns_its_value() -> None:
    async def run() -> str:
        return "finished"

    assert run_until_signaled(run) == "finished"
