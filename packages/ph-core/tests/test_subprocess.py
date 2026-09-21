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
import sys
from pathlib import Path

import anyio
import pytest

from ph.cancel import CancelToken
from ph.cordis import Context
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


def _alive(pid: int) -> bool:
    """Whether this pid is still there, without reaping anything."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
    await settled(lambda: marker.exists() and marker.read_text().strip(), "the pipeline to start")
    inner = int(marker.read_text().strip())
    assert _alive(inner), "the inner process never ran"

    await child.terminate()

    await settled(lambda: not _alive(inner), "the inner process to be signalled")


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
    assert not _alive(pid)


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

    assert not _alive(pid), "the cancel skipped the kill"


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


def test_the_handle_signals_the_group_before_the_process(tmp_path: Path) -> None:
    """And falls back, because both reasons it cannot are ordinary.

    `killpg` does not exist on Windows, and a child that has already been reaped
    has no group — in both cases the direct child is the honest best effort, and
    it is what this did before there were groups at all.
    """

    class _Gone:
        """A process whose group is no longer there."""

        returncode: int | None = None
        pid = -1
        terminated = False

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:  # pragma: no cover — not reached here
            self.terminated = True

    process = _Gone()
    handle = SubprocessHandle(
        spec=SubprocessSpawnSpec(argv=("true",), cwd=tmp_path),
        process=process,  # type: ignore[arg-type]
    )

    handle._signal(kill=False)

    assert process.terminated, "no group, and the child was not signalled either"


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
