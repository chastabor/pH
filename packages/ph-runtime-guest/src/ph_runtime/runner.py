"""The run loop of the guest: one process, many cells, one namespace.

The important structure: **one task reads the channel, and a cell runs in another
task.** A cell that awaits `tools.read(...)` is blocked on a `reply` frame. If
the task that reads frames were the task that runs the cell, the reply could
never arrive. This is the classic control-channel deadlock. It is the reason why
prime-agent needed a workaround for interrupts, and why fd 3 is not the channel
that the run occupies (D5).

**Cancellation, and a correction to the plan.** The design was "the `cancel`
frame plus `SIGINT`". `SIGINT` was for a cell that spins in Python and never
yields to the scheduler. But the *default* `SIGINT` handler of Python raises
`KeyboardInterrupt` into whatever frame is executing. When a cell is `await`ing,
that frame belongs to `asyncio`, so the signal killed the whole guest and not
the cell. That is the common case, not the rare one.

`SIGINT` is therefore installed with `loop.add_signal_handler`. That delivers it
as an ordinary loop callback, which can never land inside library internals.
Three mechanisms then cover three different situations:

* The cell awaits a `reply` or a `sleep`. The `cancel` frame, or the `SIGINT`
  callback, cancels the task of the cell.
* The cell spins in Python. The loop is starved, so neither of those arrives.
  `SIGXCPU` from the per-run CPU budget does land in the frame of the cell. The
  cell is executing bytecode, which is the reason why the other two mechanisms
  cannot reach it.
* Neither works in time. The host escalates to `SIGKILL` and restarts the guest.
  The namespace is lost and the model is told so, which is better than a kernel
  that is stuck.

Runs are serialized: one namespace, one program at a time.

@module ph_runtime.runner
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import os
import reprlib
import signal
import sys
import time
import traceback
from contextvars import ContextVar
from functools import partial
from types import FrameType
from typing import Any, NoReturn

from . import snapshot as snapshot_module
from ._characters import head, head_bytes, utf8_size
from .cell import CELL_FILENAME, CELL_FUNCTION, compile_cell
from .channel import Channel, jsonable
from .errors import RunStopped, ToolFailed
from .lifecycle import die_with_parent
from .limits import (
    CPU_BUDGET_MESSAGE,
    CpuBudgetExceeded,
    apply_limits,
    arm_cpu_budget,
    relax_cpu_budget,
)
from .protocol import PROTOCOL_VERSION, as_str, truncation_marker
from .proxies import build_namespaces
from .skill import UnavailableSkill, wrap_skill_module

__all__ = ["Runner", "main"]


_RUN: ContextVar[int] = ContextVar("ph_runtime.run", default=0)
"""The program that the code running now belongs to (F4).

**This is a `ContextVar` and not a field, and that is the whole fix.** A field is
read at the moment that a call goes out. The guest is persistent, so a task that
a cell left behind can still be running. Such a task stamped its calls with
whatever run was open *at that moment*, and the host accepted them as calls of
that run. With a field, the guard, the required wire field, and the protocol
bump would have covered only the case where no run is open.

`asyncio.create_task` copies the current context. A task that the cell starts
therefore carries the run that started it for as long as the task lives. The
next `_execute` sets this variable in its own task and cannot relabel the older
task. `0` is the value outside any run, and no run uses it.
"""

FLUSH_BYTES = 8 * 1024
FLUSH_SECONDS = 0.05
"""When a coalesced output buffer is sent: when it is full enough, or old enough.

Both conditions are needed to keep *both* properties. With size alone, the first
line of a slow cell would be held until the cell produced 8 KiB. With time alone,
a fast cell would send one frame for each write. With both, a cell that prints a
lot sends kilobytes per frame, and a slow cell still shows progress within 50 ms.
"""


_RUNAWAY_EXIT = 3
"""The exit status of a runtime that ended itself because of a runaway (M1).

It is different from 0 and from the signal-derived statuses that a kill
produces. No code branches on it, because the host only asks whether
`returncode` is `None`. The status is for a person who reads a process table or
a core-dump report. The model and the transcript get the sentence on stderr."""


class _CappedStream(io.TextIOBase):
    """The `sys.stdout` of the cell: coalesced, streamed to the host, and capped
    (D4).

    **Output is coalesced because one frame for each `write` is quadratic, in two
    ways.** `print` issues two writes (the text and the newline). If each write
    becomes a `channel.send`, then on CPython 3.12 the
    `_SelectorSocketTransport.write` of `asyncio` calls
    `get_write_buffer_size()`, which is `sum(map(len, self._buffer))` over the
    pending deque. A cell that never awaits never lets the transport drain, so
    that sum grows with everything written so far. The host then also pays its
    own work for each frame.

    Output is capped because unlimited stdout is unlimited context. The marker is
    the one that the host would have written (D4).
    """

    def __init__(self, runner: Runner, stream: str, cap: int) -> None:
        self._runner = runner
        self._stream = stream
        self._cap = cap
        self._pending: list[str] = []
        self._pending_bytes = 0
        self._flushed_at = time.monotonic()
        self._mark_truncated = False
        self.written = 0
        self.dropped = 0
        """The bytes that this stream discarded, counted to the end of the run.
        The count includes every write after the cap, as well as the write that
        crossed it. It is sent on the `done` frame of the run, which is the first
        moment that the whole count is known."""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        # Measure the size in the way that the cut does (`utf8_size`). ASCII text
        # then needs no encode.
        size = utf8_size(text)
        room = self._cap - self.written
        if size <= room:
            self.written += size
            self._buffer(text)
            return len(text)
        if room <= 0:
            # Past the cap, the text is only counted. A cell that prints megabytes
            # pays this for each write, so it is the minimum work possible.
            self.dropped += size
            return len(text)
        # This is the write that crosses the cap. It is cut in bytes, between
        # characters and never through one (`_characters`, the rule of ph-core).
        # The marker is put where the text was cut, with no number, because the
        # amount that follows is not known yet.
        kept = head_bytes(text, room)
        if kept:
            self._buffer(kept)
        self.dropped += size - utf8_size(kept)
        self._mark_truncated = True
        self._buffer(truncation_marker(None, self._cap))
        self.written = self._cap
        return len(text)

    def flush(self) -> None:
        """Send the buffered text. Called at settle, and by `print(flush=True)`."""
        if not self._pending:
            return
        frame: dict[str, Any] = {
            "type": "log",
            "stream": self._stream,
            "text": "".join(self._pending),
        }
        self._pending.clear()
        self._pending_bytes = 0
        self._flushed_at = time.monotonic()
        if self._mark_truncated:
            frame["truncated"] = True
            self._mark_truncated = False
        self._runner.channel.send(frame)

    def _buffer(self, text: str) -> None:
        self._pending.append(text)
        self._pending_bytes += len(text)
        if self._pending_bytes >= FLUSH_BYTES or (
            time.monotonic() - self._flushed_at >= FLUSH_SECONDS
        ):
            self.flush()


class Runner:
    """One persistent namespace and the frames that drive it."""

    def __init__(self, channel: Channel, boot: dict[str, Any]) -> None:
        self.channel = channel
        self.namespace_id = boot.get("namespaceId")
        self.max_log_bytes = int(boot["maxLogBytes"])
        self.max_value_bytes = int(boot["maxValueBytes"])
        self.max_snapshot_bytes = int(boot["maxSnapshotBytes"])
        self.cpu_seconds = int(boot["cpuSeconds"])
        self.idle_cpu_seconds = int(boot["idleCpuSeconds"])
        self.globals: dict[str, Any] = {
            "__name__": "__ph_cell__",
            "__builtins__": __builtins__,
            # `RunStopped` is deliberately absent. The cell gets no name with
            # which to catch a refusal (C3).
            "ToolFailed": ToolFailed,
        }
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_call_id = 0
        self._run: asyncio.Task[None] | None = None
        self._restored: dict[str, list[str]] = {}
        """What the batches of the open restore have reported so far (O3).

        It is cleared when the answer is sent, so a second restore starts empty."""
        self._owed: int | None = None
        """The run whose `done` is not yet sent, or `None` (F7).

        The host has no wall clock for a run. It waits for that frame. Every path
        out of `_execute` is therefore written to send one, and there are two
        fallbacks for the ways in which settling itself can fail. A task that is
        cancelled *before its first line* runs none of them. `create_task` only
        schedules. When `run` and `cancel` arrive in one read chunk, the cancel
        reaches a coroutine that never entered its own `try`.

        `_send_done` clears this field. `_answer_if_owed` reads it, and the task
        itself calls that function on the way out. The completion of the run
        therefore checks this fact, and each exit path does not have to remember
        it."""
        self._cpu_exceeded = False
        """Set by the `SIGXCPU` handler, and read by the cancellation branch of
        `_execute`.

        The cancel route loses the reason. A task that was cancelled for a budget
        and a task that the host cancelled both arrive as `CancelledError`. If a
        cell that used up its CPU were told "the run was canceled", the model
        would look for a user action that did not happen."""
        self._snapshotter = snapshot_module.NamespaceSnapshotter()
        """Owns the per-variable memo, so that an unchanged variable is not
        serialized again and not sent again (D17)."""
        self._install_namespaces(boot.get("namespaces") or [])
        self.install_skills(boot.get("skills") or [])
        # Everything that the bootstrap put in globals is the surface of the
        # harness, not the state of the cell. The snapshot skips it (see
        # `snapshot`).
        self._protected = set(self.globals)

    # ----------------------------------------------------------- bootstrap --

    def install_signal_handlers(self) -> None:
        """Route `SIGINT` through the loop, and route `SIGXCPU` to the place where
        the cell is."""
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            # Not available on Windows. There the Job Object of the host and the
            # `cancel` frame are the mechanisms that apply.
            loop.add_signal_handler(signal.SIGINT, self._on_interrupt)
        with contextlib.suppress(AttributeError, OSError, ValueError):
            # **Not** `add_signal_handler`, unlike `SIGINT` above. A loop callback
            # cannot reach a cell that spins in Python, and such a cell is the one
            # thing that the CPU budget exists to stop. The loop is starved, so a
            # callback that is queued on it never runs. This handler must land in
            # the running frame, which is what `signal.signal` does.
            signal.signal(signal.SIGXCPU, self._on_cpu_budget)

    def _on_interrupt(self) -> None:
        run = self._run
        if run is not None and not run.done():
            run.cancel()

    def _on_cpu_budget(self, _signum: int, frame: FrameType | None) -> None:
        """The cell used its CPU budget. Raise into the cell, or cancel it, but
        never both.

        **A raise is safe only where the cell is.** This handler runs on the main
        thread at the next bytecode boundary, wherever that thread is, and the
        raise unwinds from *there*. When the cell is executing, that is correct,
        and it is the only route that reaches a spinning cell. In other cases the
        frame that gets the exception belongs to `asyncio`: the cell is
        `await`ing, or it already finished, or it used its CPU on a `to_thread`
        worker while the main thread waited in `select()`. The guest then dies
        from a budget breach in one cell. The whole process, with the namespace,
        is lost for a limit that should cost one run.

        The stack therefore decides. If the filename of the cell is on the stack,
        the handler raises. If not, it cancels the run task and lets `_execute`
        report the breach. Both paths end in the same `cpu` error. They differ
        only in what they are allowed to interrupt.
        """
        relax_cpu_budget()
        if _executing_cell(frame):
            raise CpuBudgetExceeded(CPU_BUDGET_MESSAGE)
        if self._owed is None:
            # **No run is active and something still uses CPU** (M1). No cell
            # owns this CPU, so there is no run to cancel and no frame to blame.
            # A worker or a task that a finished cell left behind used the
            # budget, and no cooperative route can reach it. The only way left to
            # stop it is to end the process.
            self._end_runaway()
        # Set the flag only on the route that loses the reason. The raise carries
        # its own reason and is caught by name.
        self._cpu_exceeded = True
        self._on_interrupt()

    def _end_runaway(self) -> NoReturn:
        """End this process, because a finished cell left work that uses CPU (M1).

        **The exit is `os._exit`, because of what is being escaped.** A runaway by
        definition does not yield, so a clean shutdown would wait on it. A
        non-daemon thread blocks interpreter exit, and `sys.exit` from a signal
        handler unwinds only the main thread. `lifecycle._watch_parent` uses the
        same exit for the same reason.

        **The namespace is lost, and that is the real cost.** The namespace
        already holds a thread that nobody can stop. The host finds a dead child
        on the next cell and tells the model that the namespace was lost. Every
        other way of dying already takes that path.

        The reason is written to stderr first, because stderr is the only channel
        left that a person reads. The host does not read frames between runs. A
        `fault` frame would stay in a buffer that nobody drains, waiting for a
        cell that never runs.
        """
        sys.stderr.write(
            "ph: a finished cell left work burning CPU with no run to charge it to; "
            "ending this runtime so it cannot spend a core until the session does\n"
        )
        sys.stderr.flush()
        os._exit(_RUNAWAY_EXIT)

    def _install_namespaces(self, declared: list[dict[str, Any]]) -> None:
        namespaces = build_namespaces(declared, self._dispatch)
        self.globals.update(namespaces)

    def install_skills(self, names: list[str]) -> None:
        """Import each Python skill and bind it as a callable (the ported
        convention of §6.8).

        The bootstrap calls this, so the names are in `_protected` with the rest
        of the surface of the harness. A skill is a capability that the
        deployment installed, not cell state. A snapshot of it would try to
        pickle a module.
        """
        for name in names:
            try:
                module = importlib.import_module(name)
            except Exception as error:
                self.globals[name] = UnavailableSkill(name, head(str(error), 200))
            else:
                self.globals[name] = wrap_skill_module(module)

    # -------------------------------------------------------------- frames --

    async def serve(self) -> None:
        """Read frames until the host stops or sends `shutdown`."""
        while True:
            frame = await self.channel.receive()
            if frame is None:
                return
            kind = frame.get("type")
            if kind == "shutdown":
                await self._abort_run()
                return
            if kind == "run":
                self._begin(frame)
            elif kind == "reply":
                self._resolve(frame)
            elif kind == "cancel":
                await self._abort_run()
            elif kind == "restore":
                self._restore(frame)
            elif kind == "ping":
                self._pong(frame)

    def _pong(self, frame: dict[str, Any]) -> None:
        """Answer the probe of the host, and settle the run that it asks about if
        nothing else will.

        **The answer is sent from here, and that is the measurement** (M2). This
        method runs on the event loop, in the same reader task that takes a
        `cancel` frame. The round trip that the host times is therefore exactly
        how far behind that loop is. A cell that burns CPU in straight-line
        Python starves the loop, and the answer is late or never comes. For the
        same reason, both routes of `_interrupt` fail on such a cell. A late
        answer is a *load* reading, and the host treats it as one.

        **This is a repair, not a report.** The host has no wall clock for a run
        and waits for `done`. It therefore names the run that it waits for, and
        this method answers the one question that settles it: will anything here
        still send that frame? `_owed` says that the terminal frame is
        outstanding, and `self._run` says that the task is alive. If neither is
        true, the run ended and nobody told the host. `_send_done` then tells it,
        through the path that `_answer_if_owed` already uses for the same
        condition. The first design reported `_owed` to the host and let the
        host infer. The only remedy of the host was then a kill: it could
        conclude but not repair, and the cost of the conclusion was the
        namespace.

        The send is synchronous and buffered, like every other send here. A lock
        or an await would put the answer behind whatever is already stuck, and
        an answer that waits for the stuck code measures nothing.
        """
        probe, asked = frame.get("id"), frame.get("run")
        if not isinstance(probe, int):
            return
        self.channel.send({"type": "pong", "id": probe})
        if not isinstance(asked, int) or self._owed == asked:
            return
        if self._run is not None and not self._run.done():
            # A task is still running for a run that this guest already answered
            # for. That is not the concern of this function. `_owed` is what
            # tracks the terminal frame.
            return
        self._send_done(
            asked, {"error": {"kind": "aborted", "message": "the run ended without settling"}}
        )

    def _begin(self, frame: dict[str, Any]) -> None:
        run_id = frame.get("id")
        if not isinstance(run_id, int):
            return
        if self._run is not None and not self._run.done():
            self.channel.send(
                {
                    "type": "done",
                    "id": run_id,
                    "error": {"kind": "busy", "message": "a program is already running"},
                }
            )
            return
        self._owed = run_id
        task = asyncio.get_running_loop().create_task(
            self._execute(run_id, as_str(frame.get("program")))
        )
        # **This one line covers every way in which this task can stop.** The task
        # can be cancelled before its first statement, or cancelled during the
        # settle. A `BaseException` can end it above the `try` of `_execute`:
        # `arm_cpu_budget`, `_CappedStream`, and `_RUN.set` are all outside that
        # `try`. A later change can add an exit path. The alternative, which this
        # replaces, was a check at the one caller that knew about the problem.
        # `_abort_run` could not even reach that check for a task that had
        # already finished without settling.
        task.add_done_callback(partial(self._answer_if_owed, run_id))
        self._run = task

    def _resolve(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("id")
        future = self._pending.pop(call_id, None) if isinstance(call_id, int) else None
        if future is None or future.done():
            return
        if frame.get("ok"):
            future.set_result(frame.get("value"))
            return
        message = as_str(frame.get("message"), "the call was refused")
        if frame.get("fatal"):
            future.set_exception(RunStopped(message))
        else:
            future.set_exception(ToolFailed(as_str(frame.get("name"), "the call"), message))

    def _restore(self, frame: dict[str, Any]) -> None:
        """Handle one batch of a restore. The answer is sent after the last batch
        lands (O3).

        The host sends one frame for each variable. A frame that carries the whole
        namespace has no size limit, while each *value* is limited by
        `maxSnapshotBytes`. The reader on this side is sized for one value, from
        the `frame_cap` of the host, before any frame arrives (F8). A frame that
        holds more would kill the guest, where a failed restore is the correct
        outcome.

        The batches are accumulated and not answered one by one. One `restore` is
        one `done` on the side of the host. A reply for each batch would settle
        the pump of the host on the first reply, and the other batches would
        arrive with no run open.
        """
        variables = frame.get("variables")
        outcome = snapshot_module.restore(
            self.globals, variables if isinstance(variables, list) else []
        )
        for key, names in outcome.items():
            self._restored.setdefault(key, []).extend(names)
        if frame.get("more"):
            return
        answer, self._restored = self._restored, {}
        self.channel.send({"type": "done", "id": frame.get("id"), "value": answer})

    async def _abort_run(self) -> None:
        run = self._run
        if run is None or run.done():
            return
        owed = self._owed
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        # **The run is also answered here, and that does not settle it twice.**
        # The callback is scheduled through `call_soon`. If this method left the
        # answer to the callback, it would return with the run *not* yet settled.
        # This method used to guarantee a settled run on return, and the host
        # depends on that. `_answer_if_owed` is idempotent, because `_send_done`
        # clears `_owed`, so the second call does nothing. The callback stays as
        # the answer for every path that does not come through this method.
        if owed is not None:
            self._answer_if_owed(owed, run)

    def _answer_if_owed(self, run_id: int, task: asyncio.Task[None]) -> None:
        """Settle a run that ended without settling itself (F7).

        **The callback is bound to the run, not to the current value of `_owed`.**
        The guard in `_begin` is `self._run.done()`. That is already true for a
        task whose callback has not fired yet, because `add_done_callback`
        schedules through `call_soon`. A task that finished without settling
        could therefore be followed by a new `run` frame. The *stale* callback
        would then send `done` for the run that had just started. That run would
        be reported as finished before its first statement, and could never be
        settled afterward. With the id, the callback answers only for its own
        run.

        The host waits for `done` with no wall clock of its own. A run that stops
        and sends no `done` blocks the kernel until a person cancels it.
        `_execute` sends the frame on every path that it controls. This method
        covers the paths that `_execute` does not control. One is a cancellation
        that lands before the first statement: `create_task` schedules and does
        not start, so `run` and `cancel` in one read chunk cancel a coroutine
        whose body never ran. The other is anything that is raised outside the
        `try` of `_execute`.

        In the ordinary case this method does nothing, because `_send_done` has
        already cleared `_owed`.
        """
        if self._owed != run_id:
            return
        if task.cancelled():
            error = {"kind": "aborted", "message": "the run was canceled"}
        elif (raised := task.exception()) is None:
            # `exception()` is safe here, because the cancelled case is handled above.
            error = {"kind": "aborted", "message": "the run ended without settling"}
        else:
            error = {
                "kind": type(raised).__name__,
                "message": f"the run ended without settling: {raised!r}",
            }
        self._send_done(run_id, {"error": error})

    # ------------------------------------------------------------ dispatch --

    async def _dispatch(
        self,
        namespace: str,
        name: str,
        arguments: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Send one binding call to the host and wait for the answer."""
        self._next_call_id += 1
        call_id = self._next_call_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        self.channel.send(
            {
                "type": "call",
                "id": call_id,
                "run": _RUN.get(),
                "global": namespace,
                "name": name,
                "args": _plain(arguments),
            }
        )
        try:
            return await future
        finally:
            self._pending.pop(call_id, None)

    # ----------------------------------------------------------- execution --

    async def _execute(self, run_id: int, program: str) -> None:
        _RUN.set(run_id)
        self._cpu_exceeded = False
        # Record every task that is alive before the cell runs. What the cell
        # leaves behind is the difference, which is exact. The alternative,
        # "cancel everything except the two tasks that I know about", is a rule
        # that silently kills the first task that anything else in this module
        # starts.
        before = asyncio.all_tasks()
        arm_cpu_budget(self.cpu_seconds)
        out = _CappedStream(self, "stdout", self.max_log_bytes)
        err = _CappedStream(self, "stderr", self.max_log_bytes)
        error: dict[str, Any] | None = None
        value: Any = None
        try:
            code = compile_cell(program)
        except SyntaxError as syntax_error:
            # Send the message, not a traceback. A traceback here shows the frames
            # of the *guest*, and the model needs the sentence about its own
            # program.
            self._settle(
                run_id, None, {"kind": "SyntaxError", "message": str(syntax_error)}, out, err
            )
            return
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(code, self.globals)
                cell = self.globals.pop(CELL_FUNCTION)
                value = await cell()
        except asyncio.CancelledError:
            # This is the other route of the budget (see `_on_cpu_budget`). It
            # looks the same as a cancel from the host unless the flag is set.
            error = (
                {"kind": "cpu", "message": CPU_BUDGET_MESSAGE}
                if self._cpu_exceeded
                else {"kind": "aborted", "message": "the run was canceled"}
            )
        except KeyboardInterrupt:
            error = {"kind": "aborted", "message": "the run was interrupted"}
        except CpuBudgetExceeded as exceeded:
            error = {"kind": "cpu", "message": str(exceeded)}
        except RunStopped as stopped:
            # A denial or a budget. It is reported as its own kind, so that the
            # host settles the whole run and does not return the text as a result
            # (C3).
            error = {"kind": "stopped", "message": str(stopped)}
        except BaseException as raised:
            error = {"kind": type(raised).__name__, "message": _cell_traceback(raised)}
        finally:
            self.globals.pop(CELL_FUNCTION, None)
            for pending in self._pending.values():
                pending.cancel()
            self._pending.clear()
            _cancel_all(asyncio.all_tasks() - before)
        # The snapshot is taken *before* the run settles, so `done` is the last
        # frame of a run and nothing arrives after it. The host can then read the
        # frames of one run inline, without a reader task whose lifetime crosses
        # calls. A task group that is entered in one task and exited in another
        # is exactly the bug that the other order caused.
        # **Nothing between here and `done` may fail to settle.** The host has no
        # wall clock for a run. It waits for this frame, and a run that never
        # sends one blocks the kernel until a person cancels it. The snapshot is
        # the step that can raise, because it serializes whatever the cell left
        # behind. A failure there therefore costs the namespace and is reported,
        # and does not cost the session.
        try:
            self._snapshot(run_id)
        except BaseException as raised:
            err.write(f"the namespace could not be snapshotted: {raised!r}\n")
        try:
            self._settle(run_id, value, error, out, err)
        except BaseException as raised:
            # **This is the last resort, and the comment above is the reason for
            # it.** When only the snapshot was guarded, `_settle` was outside the
            # promise that the comment states. `_encode_value` catches `TypeError`
            # and `ValueError`. But a container that a thread left by the cell is
            # still mutating raises `RuntimeError` from the encoder. A run that
            # cannot report its result must still report that fact, or the host
            # waits for a frame that nothing will send.
            self._send_done(
                run_id,
                {
                    "error": {
                        "kind": type(raised).__name__,
                        "message": f"the run could not be settled: {raised!r}",
                    }
                },
            )

    def _settle(
        self,
        run_id: int,
        value: object,
        error: dict[str, Any] | None,
        out: _CappedStream,
        err: _CappedStream,
    ) -> None:
        # Send buffered output before `done`, so that the host has every log frame
        # of a run before the frame that settles the run.
        out.flush()
        err.flush()
        body: dict[str, Any] = {}
        if error is not None:
            body["error"] = error
        else:
            encoded, degraded = _encode_value(value, self.max_value_bytes)
            if encoded is not None:
                body["value"] = encoded
            if degraded:
                body["truncated"] = True
        dropped = out.dropped + err.dropped
        if dropped:
            body["truncated"] = True
            body["dropped"] = dropped
        self._send_done(run_id, body)

    def _send_done(self, run_id: int, body: dict[str, Any]) -> None:
        """Send the terminal frame of one *started* run, and clear what the run
        owed.

        The three ways in which a started run ends all go through this method:
        the ordinary settle, the last-resort catch in `_execute`, and the answer
        of `_abort_run` for a run that never reached its own `finally`. `_owed`
        therefore cannot say that a frame is outstanding after it was sent, or
        the reverse.

        Not every `done` on the wire goes through this method. The busy refusal
        in `_begin` and the answer of `_restore` are about a run that this run
        did not start. If either went through this method, it would clear
        `_owed` for a run that is still in progress.

        The envelope is built here and not at each caller. A terminal frame
        without its `type` or `id` is a frame that the host never pairs with the
        run, which has the same effect as no frame.
        """
        self._owed = None
        self.channel.send({"type": "done", "id": run_id, **body})
        # **The time that nothing else limits** (M1). `relax_cpu_budget` turned
        # the limit off at the first breach. It must, or the second delivery
        # lands in the teardown above and costs this frame. The next
        # `arm_cpu_budget` does not come until the next cell. Between the two, a
        # `to_thread` worker or a detached task that spins in pure Python uses
        # CPU freely. A cancel reaches neither. The run reported `cpu`, and the
        # thread continued with nothing left to notice it.
        #
        # The budget is armed here and not in the `finally` of `_execute`,
        # because every started run passes through this point. That includes the
        # two last-resort paths, which exist because the ordinary path failed.
        #
        # **The arm comes after the send, and it must not raise.** The whole
        # contract of this function is that the terminal frame goes out, because
        # the host waits for it with no clock of its own. Anything added below
        # the send is a new way for the one path that must complete to stop
        # halfway. `arm_cpu_budget` already suppresses what `setrlimit` raises,
        # so an exception here is something unforeseen. Its cost is the standing
        # budget, which is the behavior that this replaced, and not a lost run.
        #
        # `arm_cpu_budget` itself decides whether it is safe to arm at all. See
        # `_uncaught_budget`.
        with contextlib.suppress(Exception):
            arm_cpu_budget(self.idle_cpu_seconds)

    def _snapshot(self, run_id: int) -> None:
        """Send one frame for each changed variable (F3).

        **The cap that limits a value is per variable, but one frame used to
        carry all of the variables.** `changed` applies `max_snapshot_bytes` to
        each value that it encodes. A cell that leaves three variables just under
        the cap therefore produced a frame three times that size. The host
        refuses a frame over its `frame_cap`, on the reasonable ground that a
        peer that writes megabytes with no newline is hostile. A legitimate
        namespace was therefore read as an attack. The channel closed, and the
        model was told that the runtime had exited. The namespace was gone, and
        nothing named the real cause.

        With one frame for each variable, the limit on the wire and the limit on
        a value are the same limit. The host already records each variable
        separately: `SnapshotPolicy.record` appends one `kernel/snapshot` event
        for each. Nothing downstream can see the difference.
        """
        for record in self._snapshotter.changed(
            self.globals, protected=self._protected, max_value_bytes=self.max_snapshot_bytes
        ):
            self.channel.send({"type": "snapshot", "id": run_id, "variables": [record]})


def _plain(value: object) -> object:
    """Round-trip the value through JSON, so that a proxy object cannot travel
    inside `args`."""
    try:
        return json.loads(json.dumps(value, default=jsonable))
    except (TypeError, ValueError):  # pragma: no cover
        return {}


def _encode_value(value: object, cap: int) -> tuple[object, bool]:
    """Return the value of the cell as JSON if it fits. If not, return a bounded
    `repr`.

    **Both halves stop at the cap and do not build the whole output to measure
    it.** Models often end a cell with `df`, so this case is common. `iterencode`
    stops as soon as the prefix is over the cap, and `reprlib` never builds more
    than it needs.
    """
    if value is None:
        return None, False
    size = 0
    try:
        for chunk in json.JSONEncoder(default=jsonable).iterencode(value):
            size += len(chunk)
            if size > cap:
                return _bounded_repr(value, cap), True
    except (TypeError, ValueError):
        return _bounded_repr(value, cap), True
    return value, False


def _bounded_repr(value: object, cap: int) -> str:
    """`repr(value)` without building a repr larger than `cap`."""
    printer = reprlib.Repr()
    printer.maxstring = printer.maxother = cap
    printer.maxlist = printer.maxtuple = printer.maxdict = printer.maxset = 64
    printer.maxlevel = 6
    try:
        return head_bytes(printer.repr(value), cap)
    except Exception:  # a __repr__ may raise anything at all
        return f"<unprintable {type(value).__name__}>"


def _cell_traceback(raised: BaseException) -> str:
    """Return the traceback as the *cell* sees it, with the frames of the runner
    removed.

    A model that reads a failure must see its own program, not
    `ph_runtime/runner.py`. Those frames are the implementation of pH. They are
    the same on every failure, and they invite the model to debug the harness and
    not its cell.
    """
    frames = list(traceback.extract_tb(raised.__traceback__))
    cell_frames: list[traceback.FrameSummary] = []
    for index, entry in enumerate(frames):
        if entry.filename == CELL_FILENAME:
            cell_frames = frames[index:]
            break
    lines = ["Traceback (most recent call last):\n"] if cell_frames else []
    lines += traceback.format_list(cell_frames)
    lines += traceback.format_exception_only(type(raised), raised)
    return "".join(lines)


def _cancel_all(tasks: set[asyncio.Task[Any]]) -> None:
    """Cancel the tasks that a run started. **The namespace persists. Tasks do
    not.**

    A task made with `asyncio.create_task` in a cell outlives that cell, and the
    loop keeps running between programs because the guest is persistent. A task
    that was left behind therefore continued to call bindings after its run had
    settled. The host served those calls against whichever program was open
    next: its bindings, its call budget, and its parent dispatch id. The work of
    one cell was recorded as the work of another, and the approval decisions
    were made for the wrong turn.

    The tasks are cancelled at settle and not refused later. The refusal is the
    backstop, and this is the fix: a task that is gone sends nothing.
    """
    current = asyncio.current_task()
    for task in tasks:
        if task is not current:
            task.cancel()


def _executing_cell(frame: FrameType | None) -> bool:
    """Whether the code of the cell is on this stack now.

    The `SIGXCPU` handler needs this distinction, and it is exact, not a
    heuristic. A cell that is running has its frame on the stack of the main
    thread. A cell that is suspended at an `await` does not: its frame belongs to
    a coroutine that the loop holds, and the stack holds `asyncio`. This function
    therefore answers "may an exception be raised here" with the one question
    that decides it.

    **The walk has no depth limit, on purpose.** A depth limit looks prudent in a
    signal handler, but here it is the opposite. An `f_back` chain is finite and
    has no cycle, because it ends at the outermost frame of the interpreter.
    There is nothing to guard against. With a limit, a cell that recursed past
    the limit would get `False` and the *cancel* route. That is the one route
    that cannot reach a cell that spins in Python, which is the case that the
    budget exists for. The limit would have silently disabled the fix for deep
    cells.
    """
    while frame is not None:
        if frame.f_code.co_filename == CELL_FILENAME:
            return True
        frame = frame.f_back
    return False


def main() -> int:
    """The entry point: `python -m ph_runtime`, started by the host with fd 3 attached."""
    return asyncio.run(_serve())


async def _serve() -> int:
    channel = await Channel.open()
    boot = await channel.receive()
    if boot is None or boot.get("type") != "boot":
        return 1
    if boot.get("protocol") != PROTOCOL_VERSION:
        # Refuse the pairing and do not serve it. A guest that misreads one frame
        # at a time is worse than a guest that does not start (D7).
        channel.send(
            {
                "type": "fault",
                "message": (
                    f"protocol {boot.get('protocol')} is not {PROTOCOL_VERSION}; "
                    "the runtime venv is stale — delete $PH_CACHE/runtime-venv"
                ),
            }
        )
        await channel.aclose()
        return 2
    mechanism = die_with_parent()
    applied = apply_limits(address_space_bytes=int(boot["addressSpaceBytes"]))
    runner = Runner(channel, boot)
    runner.install_signal_handlers()
    channel.send(
        {
            "type": "boot-ack",
            "protocol": PROTOCOL_VERSION,
            "python": sys.version.split()[0],
            "limits": {**applied, "dieWithParent": mechanism},
        }
    )
    await runner.serve()
    await channel.aclose()
    return 0
