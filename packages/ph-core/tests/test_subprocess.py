"""`ctx.subprocess` — stopping a child, and everything it started (J4).

The seam had no test module of its own: what it does is covered incidentally by
the shell seam and `tool-bash`, and both of those spawn a command that exits on
its own. Nothing exercised the path that matters here — the one where the child
does *not* cooperate — which is why two ways of failing to stop one survived.

**A command is usually more than one process.** `sh -c 'a | b'` is a shell and
two children, and `Popen.terminate` signals the shell. The pipeline kept the CPU
and the pipes, outliving the timeout that was meant to end it; a caller reading
`timed_out` was told the right thing about a command that was still running.

**And the escalation has to survive the cancellation that asked for it.** A
timeout arrives as a cancel, and only the reap was shielded — so the `kill()`
after the grace was skipped exactly when it was needed, and a child ignoring
`SIGTERM` held the disposer, and its scope, for as long as it liked.

POSIX only. `killpg`, process groups and `trap` are the mechanisms under test,
and on Windows the seam falls back to signalling the child alone — which is what
it always did.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import anyio
import pytest

from ph.cancel import CancelToken
from ph.cordis import Context, maybe_await
from ph.orphans import process_alive, signal_child
from ph.seams.subprocess import SubprocessHandle, SubprocessService, SubprocessSpawnSpec
from ph.session import Session
from ph.testing import StubAgent, run_tool, settled, tool_runtime
from ph.tools import TOOL_ABORTED

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(sys.platform == "win32", reason="process groups are POSIX"),
]


def _service() -> tuple[Context, SubprocessService]:
    """A scope that owns the children, and the seam that spawns them."""
    root = Context()
    return root, SubprocessService(ctx=root)


async def _pid_written(marker: Path, what: str) -> int:
    """The pid a test's command wrote to `marker`, once it has."""
    await settled(lambda: marker.exists() and marker.read_text().strip(), what)
    return int(marker.read_text().strip())


async def test_a_pipeline_does_not_survive_the_command_that_started_it(
    tmp_path: Path,
) -> None:
    """The whole group goes, not just the shell.

    The inner `sleep` writes its own pid out before it parks, so the assertion
    is about a process this test can name rather than about a count of
    survivors — a pipeline whose members are only implied is one a passing test
    can be wrong about.
    """
    root, service = _service()
    marker = tmp_path / "inner.pid"
    child = await service.spawn(
        SubprocessSpawnSpec(
            argv=("sh", "-c", f"(sleep 30 & echo $! > {marker}; wait) | cat"),
            cwd=tmp_path,
            grace_ms=200,
        ),
        scope=root,
    )
    inner = await _pid_written(marker, "the pipeline to start")
    assert process_alive(inner), "the inner process never ran"

    await child.terminate()

    await settled(lambda: not process_alive(inner), "the inner process to be signalled")


async def test_a_member_that_ignores_the_asking_does_not_outlive_its_leader(
    tmp_path: Path,
) -> None:
    """The kill is of the group, whether or not the child is still there.

    The shell heeds `SIGTERM` and exits; the process it started ignores it. The
    kill was sent only while the shell lived, so the inner one ran on with no
    parent and nothing that knew of it — the same gap a sandbox wrapper fell
    through. Named by the pid it writes, as the pipeline above is.
    """
    root, service = _service()
    marker = tmp_path / "inner.pid"
    # A file, so `$$` is the inner shell's own pid and not the one that quoted it.
    script = tmp_path / "inner.sh"
    script.write_text(f"trap '' TERM\necho $$ > {marker}\nwhile true; do sleep 0.05; done\n")
    child = await service.spawn(
        SubprocessSpawnSpec(argv=("sh", "-c", f"sh {script} & wait"), cwd=tmp_path, grace_ms=5_000),
        scope=root,
    )
    inner = await _pid_written(marker, "the inner one to start")
    assert process_alive(inner), "the inner process never ran"
    assert inner != child.pid, "the pid written is the leader's own"

    await child.terminate()

    await settled(
        lambda: not process_alive(inner), "the member that ignored the asking to be killed"
    )


async def test_a_command_stopped_after_its_leader_exited_takes_its_job_with_it(
    tmp_path: Path,
) -> None:
    """Whether a child was stopped is what its caller meant, not whether its leader
    had exited. Here the shell exits at once and the job it started holds stdout,
    so the run is canceled with its leader gone: that is a stop, and the job goes
    with it — where reading the leader's exit as "ended by itself" left it
    running, and forgotten by the journal a restart sweeps by."""
    root, service = _service()
    marker = tmp_path / "job.pid"
    signal = CancelToken()
    job: list[int] = []

    async def stop_once_started() -> None:
        job.append(await _pid_written(marker, "the job to start"))
        signal.cancel("user")

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(stop_once_started)
        with anyio.fail_after(10):
            result = await service.run(
                SubprocessSpawnSpec(
                    argv=("sh", "-c", f"sleep 30 & echo $! > {marker}; echo started"),
                    cwd=tmp_path,
                ),
                scope=root,
                signal=signal,
            )

    assert result.canceled
    await settled(lambda: not process_alive(job[0]), "the job a stopped command left to be killed")


@pytest.mark.parametrize("through", ["run", "spawn"])
async def test_a_command_seen_through_keeps_what_it_left_running(
    tmp_path: Path, through: str
) -> None:
    """The other half: only a child that is *stopped* has its group killed. Every
    child is released through `terminate`, and one seen through — its output read
    to the end, its exit awaited — may have left a process running on purpose, a
    server it started. By `run`, and by a caller that drives `spawn` itself."""
    root, service = _service()
    marker = tmp_path / "left.pid"
    spec = SubprocessSpawnSpec(
        argv=("sh", "-c", f"sleep 30 >/dev/null 2>&1 & echo $! > {marker}"), cwd=tmp_path
    )
    if through == "run":
        exit_code = (await service.run(spec, scope=root)).exit_code
    else:
        child = await service.spawn(spec, scope=root)
        await child.pump()
        exit_code = await child.wait()
        assert child.finished and child.dispose is not None
        await maybe_await(child.dispose())
    left = int(marker.read_text().strip())
    try:
        assert exit_code == 0
        assert process_alive(left), "a process the command left running was killed with it"
    finally:
        os.kill(left, signal.SIGKILL)


async def _stubborn(tmp_path: Path, grace_ms: int) -> tuple[Context, SubprocessHandle, int]:
    """A running child that has installed `trap '' TERM`, and its pid.

    The incantation is the thing under test, so it is written once: stated twice,
    a change to it lands in one test and not the other.
    """
    root, service = _service()
    ready = tmp_path / "ready"
    child = await service.spawn(
        SubprocessSpawnSpec(
            argv=("sh", "-c", f"trap '' TERM; touch {ready}; while true; do sleep 0.05; done"),
            cwd=tmp_path,
            grace_ms=grace_ms,
        ),
        scope=root,
    )
    await settled(ready.exists, "the child to install its trap")
    pid = child.pid
    assert pid is not None
    return root, child, pid


async def test_a_child_that_ignores_term_is_killed_within_the_grace(tmp_path: Path) -> None:
    """`trap '' TERM` is a program declining to stop, which is allowed.

    What is not optional is that the harness stops it anyway. The ladder is
    ask, wait, insist — and `SIGKILL` is the rung a process cannot decline.

    **This one held before J4 as well**, and is here as the contract the two
    tests around it build on: the ladder itself was never the broken part. What
    was broken is who it reaches (the test above) and whether it finishes (the
    test below).
    """
    _root, child, pid = await _stubborn(tmp_path, grace_ms=200)

    with anyio.fail_after(5):
        await child.terminate()

    assert child.returncode is not None, "the child outlived the ladder"
    assert not process_alive(pid)


async def test_the_kill_still_happens_when_the_caller_is_canceled(tmp_path: Path) -> None:
    """A timeout *is* a cancellation, so the escalation has to outlive one.

    Only the reap used to be shielded: a caller cancelled during the grace
    unwound straight past the `kill()`, and a child ignoring `SIGTERM` went on
    running with nothing left that would ever stop it. The cancel here lands
    while `terminate` is waiting out its grace, which is the window that was
    open.

    **Without the fix this hangs rather than fails**, and the hang is the defect
    rather than a shortcoming of the test: the unwind reached the shielded reap,
    `aclose()` waited for a child that had declined to exit, and nothing
    remained that would ever kill it. A shielded wait cannot be interrupted from
    outside, so no timeout here could turn it into a failure.
    """
    _root, child, pid = await _stubborn(tmp_path, grace_ms=400)

    async with anyio.create_task_group() as tasks:

        async def stop() -> None:
            await child.terminate()

        tasks.start_soon(stop)
        # Inside the grace, so the cancel lands while `terminate` is waiting.
        await anyio.sleep(0.1)
        tasks.cancel_scope.cancel()

    assert not process_alive(pid), "the cancel skipped the kill"


async def test_a_child_is_a_group_leader_so_a_group_can_be_signalled(
    tmp_path: Path,
) -> None:
    """The property the other three rest on, asserted directly.

    `start_new_session=True` is what makes `killpg` mean "this command" rather
    than "this shell and whatever else shares pH's group" — which, without it,
    would have included pH itself.
    """
    root, service = _service()
    child = await service.spawn(
        SubprocessSpawnSpec(argv=("sleep", "5"), cwd=tmp_path, grace_ms=200), scope=root
    )
    pid = child.pid
    assert pid is not None

    assert os.getpgid(pid) == pid, "the child shares a group with the harness"
    assert os.getpgid(pid) != os.getpgid(os.getpid())

    await child.terminate()


async def test_a_cancel_token_stops_a_running_child(tmp_path: Path) -> None:
    """The bound a *person* can reach, which there was not one of (C7).

    `timeout_ms` is a number chosen before the command ran. Escape is the
    decision made while watching it, and nothing between the tool pipeline and
    the child was observing one — so `bash sleep 3600` with no timeout could not
    be interrupted at all: the cancel reached the pipeline, the pipeline
    discarded the result, and the child ran on.

    No `timeout_ms` here on purpose, so the token is the only thing that can end
    this. Under `fail_after` because without the fix it does not fail, it waits
    out the sleep.
    """
    root, service = _service()
    signal = CancelToken()
    ready = tmp_path / "ready"

    async with anyio.create_task_group() as tasks:

        async def stop_shortly() -> None:
            await settled(ready.exists, "the command to start")
            signal.cancel("user")

        tasks.start_soon(stop_shortly)
        with anyio.fail_after(10):
            result = await service.run(
                SubprocessSpawnSpec(
                    argv=("sh", "-c", f"touch {ready}; sleep 30"), cwd=tmp_path, grace_ms=200
                ),
                scope=root,
                signal=signal,
            )

    assert result.exit_code != 0, "the child was not stopped"
    # Not a timeout: nothing timed out, somebody asked. A caller that reported
    # `timed_out` for a cancel would be telling the model the wrong story — and
    # for a while there was no third answer to give instead, so a caller could
    # only tell this apart from an ordinary non-zero exit by not asking (N3).
    assert not result.timed_out
    assert result.canceled, "a cancel was indistinguishable from a command that failed"


class _Recorded:
    """A process with no group, that records what it was sent and dies of either."""

    pid = -1

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.sent: list[str] = []

    def terminate(self) -> None:
        self.sent.append("terminate")
        self.returncode = -15

    def kill(self) -> None:
        self.sent.append("kill")
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0

    async def aclose(self) -> None:
        return None


def _handle(
    tmp_path: Path, process: _Recorded, *, forwards_signals: bool = True
) -> SubprocessHandle:
    return SubprocessHandle(
        spec=SubprocessSpawnSpec(argv=("true",), cwd=tmp_path, forwards_signals=forwards_signals),
        process=process,  # type: ignore[arg-type]
    )


def test_the_handle_signals_the_group_before_the_process(tmp_path: Path) -> None:
    """And falls back, because both reasons it cannot are ordinary.

    `killpg` does not exist on Windows, and a child that has already been reaped
    has no group — in both cases the direct child is the honest best effort, and
    it is what this did before there were groups at all.
    """
    process = _Recorded()

    signal_child(process, kill=False)

    assert process.sent == ["terminate"], "no group, and the child was not signalled either"


async def test_a_child_whose_wrapper_passes_no_signal_on_is_killed_and_not_asked(
    tmp_path: Path,
) -> None:
    """`terminate` asks, waits and then insists — of a child that can hear the
    asking (`SubprocessSpawnSpec.forwards_signals`). `test_sandbox_local` holds
    the same with a real wrapper."""

    async def ended(*, forwards_signals: bool) -> list[str]:
        process = _Recorded()
        await _handle(tmp_path, process, forwards_signals=forwards_signals).terminate()
        return process.sent

    assert await ended(forwards_signals=True) == ["terminate"]
    assert await ended(forwards_signals=False) == ["kill"]


async def test_a_canceled_command_is_an_abort_rather_than_an_exit_code() -> None:
    """N3 — the cancel reached the child and stopped at the seam's vocabulary.

    `run` applies three bounds and reported two, so a child killed because
    somebody pressed stop came back as an ordinary `ShellResult` with
    `timed_out=False` and whatever exit code the kill produced. The tool then
    returned a *value*: `dispatch` has no post-hoc signal check — the only one is
    before the body runs — so the pipeline recorded a successful call whose
    rendered text was `[exit -15]`, and a model reading that sees a command that
    failed on its own rather than a turn the person ended.

    Driven through a stub shell rather than a real cancel race, because the claim
    is the translation, not the killing: `test_the_child_is_stopped_when_the_caller_cancels`
    one file up already pins that the signal reaches the child.

    Sabotage: drop the `result.canceled` branch from `bash_tool` and the call
    comes back `is_error=False` carrying the kill's exit code.
    """
    from ph.keys import SHELL
    from ph.seams.shell import ShellResult
    from ph.tools.builtin import bash_tool

    root, _runtime = tool_runtime()

    class _Stopped:
        async def run(self, command: str, **_: object) -> ShellResult:
            # What the seam now answers for a child the signal ended: a
            # terminated child's status, and the field that says why.
            return ShellResult(
                exit_code=-15, stdout="", stderr="", argv=("sh", "-c", command), canceled=True
            )

    root.provide(SHELL, _Stopped())
    await bash_tool.apply(root, None)
    agent = StubAgent(ctx=root, session=Session("bash-cancel"))

    result = await run_tool(root, "bash", {"command": "sleep 30"}, agent=agent)

    assert result.is_error, "a canceled command was reported as one that ran"
    assert result.error is not None
    assert result.error.kind == "aborted", (
        f"a person's interrupt was reported to the model as {result.error.kind}"
    )
    # After dispatch, not before: the child ran, so the call is not safe to retry
    # blind — which is the distinction `aborted_result` keeps two codes for.
    assert (result.error.info or {}).get("code") == TOOL_ABORTED
