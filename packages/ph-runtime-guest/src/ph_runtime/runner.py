"""The guest's run loop: one process, many cells, one namespace.

The shape that matters: **the channel is read by one task, and a cell runs in
another.** A cell that awaits `tools.read(...)` is blocked on a `reply` frame,
so if the loop that reads frames were the loop running the cell, the reply could
never arrive — the classic control-channel deadlock, which is exactly why
prime-agent needed a workaround for interrupts and why fd 3 is not the channel
the run occupies (D5).

**Cancellation, and a correction to the plan.** The design was "the `cancel`
frame plus `SIGINT`", the second being for a cell spinning in Python that never
yields to the scheduler. But Python's *default* `SIGINT` handler raises
`KeyboardInterrupt` into whatever frame is executing — and when a cell is
`await`ing, that frame is `asyncio`'s own, so the signal killed the entire guest
instead of the cell. That is the common case, not the rare one.

So `SIGINT` is installed with `loop.add_signal_handler`, which delivers it as an
ordinary loop callback and can never land inside library internals. Three
mechanisms then cover three genuinely different situations:

* awaiting a `reply` or a `sleep` → the `cancel` frame, or the `SIGINT`
  callback, cancels the cell's task;
* spinning in Python → the loop is starved, so neither arrives. `SIGXCPU` from
  the per-run CPU budget does land in the cell's own frame (it is executing
  bytecode, which is the whole reason it is unreachable by the others);
* neither works in time → the host escalates to `SIGKILL` and restarts. The
  namespace is lost and the model is told so, which beats a wedged kernel.

Runs are serialized: one namespace, one program at a time.

@module ph_runtime.runner
"""

from __future__ import annotations

import asyncio
import contextlib
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
"""Which program the code running right now belongs to (F4).

**A `ContextVar`, not a field, and that is the whole of the fix.** A field is
read at the moment the call goes out, so a task the cell left behind — still
running, because the guest is persistent — stamped whatever run was open *then*
and the host waved it through as its own. The guard, the required wire field and
the protocol bump would all have bought only the case where no run is open.

`asyncio.create_task` copies the current context, so a task the cell starts
carries the run that started it for as long as it lives, and the next `_execute`
setting this in its own task cannot relabel it. `0` is the value outside any
run, which no run uses.
"""

FLUSH_BYTES = 8 * 1024
FLUSH_SECONDS = 0.05
"""When a coalesced output buffer goes out: full enough, or old enough.

The pair is what keeps *both* properties. Size alone would hold a slow cell's
first line until it had produced 8 KiB; time alone would send a frame per write
for a fast one. Together, a chatty cell sends kilobytes per frame and a slow one
still shows progress within 50 ms.
"""


_RUNAWAY_EXIT = 3
"""Exit status for a runtime that ended itself over a runaway (M1).

Distinct from 0 and from the signal-derived statuses a kill produces. Nothing
branches on it — the host only ever asks whether `returncode` is `None` — so
this is for a person reading a process table or a core-dump report, and the
sentence on stderr is what the model and the transcript get."""


class _CappedStream(io.TextIOBase):
    """The cell's `sys.stdout`, coalesced, streamed to the host, and capped (D4).

    **Coalesced because a frame per `write` is quadratic, twice over.** `print` issues
    two writes (the text and the newline), and each one becoming a `channel.send`
    means: on CPython 3.12 `asyncio`'s `_SelectorSocketTransport.write` calls
    `get_write_buffer_size()`, which is `sum(map(len, self._buffer))` over the pending
    deque — and a cell that never awaits never lets the transport drain, so that sum
    grows with everything written so far. The host then pays its own per-frame work.

    Capped because unbounded stdout is unbounded context, and the marker is the one
    the host would have written (D4).
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
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        # ASCII is the common case and its character count *is* its byte count,
        # so the encode is only paid for text that needs it.
        size = len(text) if text.isascii() else len(text.encode("utf-8", "replace"))
        room = max(self._cap - self.written, 0)
        if size <= room:
            self.written += size
            self._buffer(text)
            return len(text)
        if room:
            self._buffer(text[:room])
        if not self.truncated:
            self.truncated = True
            self._mark_truncated = True
            self._buffer(truncation_marker(size - room, self._cap))
        self.written = self._cap
        return len(text)

    def flush(self) -> None:
        """Send whatever is buffered. Called at settle, and by `print(flush=True)`."""
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
            # `RunStopped` is deliberately absent: the cell is given no name to
            # catch a refusal by (C3).
            "ToolFailed": ToolFailed,
        }
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_call_id = 0
        self._run: asyncio.Task[None] | None = None
        self._owed: int | None = None
        """The run whose `done` has not been sent yet, or `None` (F7).

        The host has no wall clock on a run — it waits for that frame — so every
        path out of `_execute` is written to send one, and there are two
        fallbacks for the ways settling can itself fail. A task cancelled
        *before its first line* runs none of them: `create_task` only schedules,
        so `run` and `cancel` arriving in one read chunk cancels a coroutine
        that never entered its own `try`.

        Cleared by `_send_done` and read by `_answer_if_owed`, which the task
        itself calls on the way out — so this is a fact the run's completion
        checks, not one each exit path has to remember."""
        self._cpu_exceeded = False
        """Set by the `SIGXCPU` handler, read by `_execute`'s cancellation arm.

        The cancel route loses the reason — a task cancelled for a budget and one
        cancelled by the host both arrive as `CancelledError` — and "the run was
        canceled" for a cell that burned its CPU sends the model looking for a
        user who did not do it."""
        self._snapshotter = snapshot_module.NamespaceSnapshotter()
        """Owns the per-variable memo, so an unchanged variable is neither
        re-serialized nor re-sent (D17)."""
        self._install_namespaces(boot.get("namespaces") or [])
        self.install_skills(boot.get("skills") or [])
        # Everything the bootstrap put in globals is the harness's surface, not
        # the cell's state, so the snapshot skips it (see `snapshot`).
        self._protected = set(self.globals)

    # ----------------------------------------------------------- bootstrap --

    def install_signal_handlers(self) -> None:
        """Route `SIGINT` through the loop, and `SIGXCPU` to wherever the cell is."""
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            # Not available on Windows, where the host's Job Object and the
            # `cancel` frame are the mechanisms that apply.
            loop.add_signal_handler(signal.SIGINT, self._on_interrupt)
        with contextlib.suppress(AttributeError, OSError, ValueError):
            # **Not** `add_signal_handler`, unlike `SIGINT` above. A loop callback
            # cannot reach a cell spinning in Python, and a cell spinning in
            # Python is the one thing the CPU budget exists to stop — the loop is
            # starved, so a callback queued on it never runs. This one has to
            # land in the running frame, which is what `signal.signal` does.
            signal.signal(signal.SIGXCPU, self._on_cpu_budget)

    def _on_interrupt(self) -> None:
        run = self._run
        if run is not None and not run.done():
            run.cancel()

    def _on_cpu_budget(self, _signum: int, frame: FrameType | None) -> None:
        """The cell spent its CPU budget. Raise into it, or cancel it — never both.

        **Raising is only safe where the cell is.** This handler runs on the main
        thread at the next bytecode boundary, wherever that thread happens to be,
        and the raise unwinds from *there*. When the cell is executing that is
        exactly right and it is the only route that reaches a spin. When the cell
        is `await`ing — or has already finished, or burned its CPU on a
        `to_thread` worker while the main thread sat in `select()` — the frame
        that gets the exception belongs to `asyncio`, and the guest dies of a
        budget breach in one cell. The whole process, namespace included, for a
        limit that is supposed to cost one run.

        So the stack decides: the cell's own filename on it means raise, and
        anything else means cancel the run task and let `_execute` report it.
        Both paths end in the same `cpu` error; they differ only in what they are
        allowed to interrupt.
        """
        relax_cpu_budget()
        if _executing_cell(frame):
            raise CpuBudgetExceeded(CPU_BUDGET_MESSAGE)
        if self._owed is None:
            # **Nothing is running and something is still burning** (M1). No
            # cell owns this CPU, so there is no run to cancel and no frame to
            # blame — what spent the budget is a worker or a task a finished
            # cell left behind, and it is unreachable by every cooperative
            # route. Ending the process is the only thing left that stops it.
            self._end_runaway()
        # Set only on the route that loses the reason: the raise carries its own
        # and is caught by name.
        self._cpu_exceeded = True
        self._on_interrupt()

    def _end_runaway(self) -> NoReturn:
        """End this process: a finished cell left something burning (M1).

        **`os._exit`, and the reason is the thing being escaped.** A runaway is
        by definition not yielding, so a clean shutdown would wait on it: a
        non-daemon thread blocks interpreter exit, and `sys.exit` from a signal
        handler only unwinds the main thread. `lifecycle._watch_parent` takes
        the same exit for the same reason.

        **The namespace goes, and that is the honest price.** It is already
        holding a thread nobody can stop; the host discovers a dead child on the
        next cell and tells the model the namespace was lost, which is the path
        every other way of dying already takes.

        Said on stderr first, because that is the only channel left that a
        person reads — the host is not reading frames between runs, so a `fault`
        would sit in a buffer nobody drains until a cell that will never run.
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
        """Import each Python skill and bind it callable (§6.8's ported convention).

        Called from the bootstrap, so the names are `_protected` with the rest of
        the harness's surface: a skill is capability the deployment installed,
        not cell state, and snapshotting it would try to pickle a module.
        """
        import importlib

        for name in names:
            try:
                module = importlib.import_module(name)
            except Exception as error:
                self.globals[name] = UnavailableSkill(name, str(error)[:200])
            else:
                self.globals[name] = wrap_skill_module(module)

    # -------------------------------------------------------------- frames --

    async def serve(self) -> None:
        """Read frames until the host stops or says `shutdown`."""
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
        """Answer the host's probe, and settle the run it asks about if nobody will.

        **Answered from here, which is the measurement** (M2). This runs on the
        event loop, in the same reader task that takes a `cancel` frame — so the
        round trip the host times is exactly how far behind that loop is. A cell
        burning CPU in straight-line Python starves it and the answer is late or
        never, which is the same reason `_interrupt`'s two routes both fail on
        such a cell. That is a *load* reading, and the host treats it as one.

        **The repair, not a report.** The host has no wall clock on a run and
        waits for `done`, so it names the run it is waiting for and this answers
        the only question that settles it: is anyone here still going to send
        that frame? `_owed` says the terminal frame is outstanding and
        `self._run` says the task is alive; with neither, the run ended without
        anyone telling the host, and `_send_done` says so through the path
        `_answer_if_owed` already takes for the same condition. Reporting
        `_owed` outward and letting the host infer was the first shape, and it
        made the host's remedy a kill: it could only conclude, not repair, and
        the price of concluding was the namespace.

        Synchronous and buffered, like every other send here: taking a lock or
        awaiting would put the answer behind whatever is already wedged, and an
        answer that waits for the wedge measures nothing.
        """
        probe, asked = frame.get("id"), frame.get("run")
        if not isinstance(probe, int):
            return
        self.channel.send({"type": "pong", "id": probe})
        if not isinstance(asked, int) or self._owed == asked:
            return
        if self._run is not None and not self._run.done():
            # A task still going for a run this guest has already answered for
            # is not this function's business; `_owed` is the frame's clock.
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
        # **Every way this task can stop, covered by one line.** Cancelled before
        # its first statement, cancelled mid-settle, killed by a `BaseException`
        # above `_execute`'s own `try` — `arm_cpu_budget`, `_CappedStream`,
        # `_RUN.set` are all outside it — or stopped by an exit path added later.
        # The alternative is what this replaces: a check at the one caller that
        # happened to know, which `_abort_run` could not even reach for a task
        # that had already finished without settling.
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
        variables = frame.get("variables")
        outcome = snapshot_module.restore(
            self.globals, variables if isinstance(variables, list) else []
        )
        self.channel.send({"type": "done", "id": frame.get("id"), "value": outcome})

    async def _abort_run(self) -> None:
        run = self._run
        if run is None or run.done():
            return
        owed = self._owed
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run
        # **Answered here too, and that is not a double-settle.** The callback is
        # scheduled through `call_soon`, so leaving it to fire would make this
        # method return with the run *not* yet settled — a contract it used to
        # keep, and one the host relies on. `_answer_if_owed` is idempotent
        # because `_send_done` clears `_owed`, so whichever runs second does
        # nothing. The callback stays as the answer for every path that does not
        # come through here at all.
        if owed is not None:
            self._answer_if_owed(owed, run)

    def _answer_if_owed(self, run_id: int, task: asyncio.Task[None]) -> None:
        """Settle a run that ended without settling itself (F7).

        **Bound to the run, not to whatever `_owed` says now.** `_begin`'s guard
        is `self._run.done()`, which is already true for a task whose callback
        has not fired — `add_done_callback` schedules through `call_soon`. So a
        task that finished without settling could be followed by a new `run`
        frame, and the *stale* callback would then send `done` for the run that
        had just started: reported finished before its first statement, and
        never settleable after. The id makes the callback answer only for its
        own run.

        The host waits for `done` with no wall clock of its own, so a run that
        stops without sending one wedges the kernel until a person cancels it.
        `_execute` sends the frame on every path it controls; this covers the
        ones it does not — a cancellation that lands before its first statement
        (`create_task` schedules rather than starts, so `run` and `cancel` in one
        read chunk cancels a coroutine whose body never ran) and anything raised
        outside its own `try`.

        A no-op in the ordinary case, because `_send_done` has already cleared
        `_owed`.
        """
        if self._owed != run_id:
            return
        if task.cancelled():
            error = {"kind": "aborted", "message": "the run was canceled"}
        elif (raised := task.exception()) is None:
            # `exception()` is safe here: the cancelled case returned above.
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
        """Marshal one binding call and wait for the host's answer."""
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
        # Every task alive before the cell ran. What the cell leaves behind is
        # the difference, which is exact — the alternative, "cancel everything
        # except the two I know about", is a rule that silently kills the first
        # task anything else in this module ever starts.
        before = asyncio.all_tasks()
        arm_cpu_budget(self.cpu_seconds)
        out = _CappedStream(self, "stdout", self.max_log_bytes)
        err = _CappedStream(self, "stderr", self.max_log_bytes)
        error: dict[str, Any] | None = None
        value: Any = None
        try:
            code = compile_cell(program)
        except SyntaxError as syntax_error:
            # The message, not a traceback: a traceback here is the *guest's*
            # frames, and the model needs the sentence about its own program.
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
            # The budget's other route (see `_on_cpu_budget`), which arrives here
            # indistinguishable from the host's cancel unless the flag says so.
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
            # A denial or a budget. Reported as its own kind so the host settles
            # the whole run rather than handing the text back as a result (C3).
            error = {"kind": "stopped", "message": str(stopped)}
        except BaseException as raised:
            error = {"kind": type(raised).__name__, "message": _cell_traceback(raised)}
        finally:
            self.globals.pop(CELL_FUNCTION, None)
            for pending in self._pending.values():
                pending.cancel()
            self._pending.clear()
            _cancel_all(asyncio.all_tasks() - before)
        # Snapshotted *before* the run settles, so `done` is the last frame of a
        # run and nothing arrives after it. That lets the host read one run's
        # frames inline instead of keeping a reader task whose lifetime crosses
        # calls — and a task group entered in one task and exited in another is
        # exactly the bug that ordering caused.
        # **Nothing between here and `done` may fail to settle.** The host has no
        # wall clock on a run: it waits for this frame, and a run that never
        # sends one wedges the kernel until a human cancels. Snapshotting is the
        # step that can raise — it serializes whatever the cell left behind — so
        # a failure there costs the namespace and says so, rather than costing
        # the session.
        try:
            self._snapshot(run_id)
        except BaseException as raised:
            err.write(f"the namespace could not be snapshotted: {raised!r}\n")
        try:
            self._settle(run_id, value, error, out, err)
        except BaseException as raised:
            # **The last resort, and the reason it exists is the comment above.**
            # Guarding the snapshot alone left `_settle` outside the promise it
            # states: `_encode_value` catches `TypeError` and `ValueError`, and a
            # container being mutated by a thread the cell left running raises
            # `RuntimeError` out of the encoder instead. A run that cannot report
            # its result must still report that, or the host waits for a frame
            # nobody is going to send.
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
        # Buffered output goes out before `done`, so the host has every log
        # frame of a run before the frame that settles it.
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
        if out.truncated or err.truncated:
            body["truncated"] = True
        self._send_done(run_id, body)

    def _send_done(self, run_id: int, body: dict[str, Any]) -> None:
        """Send one *started* run's terminal frame, and clear what it owed.

        The three ways a started run ends go through here — the ordinary settle,
        `_execute`'s last-resort catch, and `_abort_run`'s answer for a run that
        never reached its own `finally` — so `_owed` cannot say a frame is
        outstanding when it was sent, or the reverse.

        Not every `done` on the wire: `_begin`'s busy refusal and `_restore`
        answer about a run this one did not start, and routing either through
        here would clear `_owed` for a run that is still going.

        The envelope is built here rather than at each caller because a terminal
        frame missing its `type` or `id` is one the host never pairs to the run,
        which is indistinguishable from never sending it.
        """
        self._owed = None
        self.channel.send({"type": "done", "id": run_id, **body})
        # **The window nothing else bounds** (M1). `relax_cpu_budget` switched
        # the limit off at the first breach — it has to, or the second delivery
        # lands in the teardown above and costs this very frame — and the next
        # `arm_cpu_budget` is not until the next cell. Between the two, a
        # `to_thread` worker or a detached task spinning in pure Python burns
        # freely: a cancel reaches neither, so the run reported `cpu` and the
        # thread carried on with nothing left to notice.
        #
        # Armed here rather than in `_execute`'s `finally` because this is the
        # one point every started run passes through, including the two
        # last-resort paths that exist precisely because the ordinary one
        # failed.
        #
        # **After the send, and it may not raise.** This function's whole
        # contract is that the terminal frame goes out — the host waits for it
        # with no clock of its own — so anything added below the send is a new
        # way for the one path that must complete to stop half way.
        # `arm_cpu_budget` already suppresses what `setrlimit` throws, so
        # reaching this is something unforeseen, and what it costs is the
        # standing budget: the behavior this replaced, not a lost run.
        #
        # Whether it is safe to arm at all is `arm_cpu_budget`'s own question —
        # see `_uncaught_budget`.
        with contextlib.suppress(Exception):
            arm_cpu_budget(self.idle_cpu_seconds)

    def _snapshot(self, run_id: int) -> None:
        """One frame per changed variable (F3).

        **The cap that bounds a value is per variable; the frame carried all of
        them.** `max_snapshot_bytes` is applied by `changed` to each value it
        encodes, so a cell that leaves three variables just under it produces a
        frame three times the size — and the host refuses a frame over
        the host's `frame_cap` on the reasonable ground that a peer writing megabytes
        with no newline is hostile. A legitimate namespace was therefore read as
        an attack: the channel closed and the model was told the runtime had
        exited, with the namespace gone and nothing naming the real cause.

        Sending one frame each makes the wire bound and the value bound the same
        bound. The host already records per variable — `SnapshotPolicy.record`
        appends one `kernel/snapshot` event each — so nothing downstream can tell
        the difference.
        """
        for record in self._snapshotter.changed(
            self.globals, protected=self._protected, max_value_bytes=self.max_snapshot_bytes
        ):
            self.channel.send({"type": "snapshot", "id": run_id, "variables": [record]})


def _plain(value: object) -> object:
    """Round-trip through JSON so a proxy object cannot ride along in `args`."""
    try:
        return json.loads(json.dumps(value, default=jsonable))
    except (TypeError, ValueError):  # pragma: no cover
        return {}


def _encode_value(value: object, cap: int) -> tuple[object, bool]:
    """The cell's value as JSON if it fits, else a bounded `repr`.

    **Both halves stop at the cap rather than building the whole thing to measure
    it** — and "end the cell with `df`" is exactly how models write cells.
    `iterencode` stops as soon as the prefix is over the cap, and `reprlib` never
    builds more than it needs.
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
        return printer.repr(value)[:cap]
    except Exception:  # a __repr__ may raise anything at all
        return f"<unprintable {type(value).__name__}>"


def _cell_traceback(raised: BaseException) -> str:
    """The traceback as the *cell* sees it, with the runner's frames removed.

    A model reading a failure should see its own program, not
    `ph_runtime/runner.py` — those frames are pH's implementation, they are
    identical on every failure, and they invite the model to debug the harness
    instead of its cell.
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
    """A run owns the tasks it started. **The namespace persists; tasks do not.**

    `asyncio.create_task` in a cell outlives the cell that made it, and the loop
    keeps running between programs because the guest is persistent — so a task
    left behind went on calling bindings after its run had settled. The host
    served those calls against whichever program was open next: its bindings, its
    call budget, its parent dispatch id. One cell's work was recorded as
    another's, and the approval decisions were taken for the wrong turn.

    Cancelled at settle rather than refused later, because the refusal is the
    backstop and this is the fix: a task that is gone issues nothing.
    """
    current = asyncio.current_task()
    for task in tasks:
        if task is not current:
            task.cancel()


def _executing_cell(frame: FrameType | None) -> bool:
    """Whether the cell's own code is on this stack, right now.

    The discrimination the `SIGXCPU` handler needs, and it is exact rather than a
    heuristic: a cell that is running has its frame on the main thread's stack,
    and a cell that is suspended at an `await` does not — its frame belongs to a
    coroutine the loop is holding, and what is on the stack is `asyncio`. So this
    answers "may an exception be raised here" by asking the only question that
    settles it.

    **Unbounded on purpose.** A depth cap looks like the prudent thing to write
    in a signal handler, and here it is the opposite: an `f_back` chain is finite
    and acyclic — it ends at the interpreter's outermost frame — so there is
    nothing to guard against, and a cell that recursed past the cap would be
    answered `False` and get the *cancel* route. That is the one route that
    cannot reach a cell spinning in Python, which is the case the budget exists
    for. The cap would have quietly switched the fix off for deep cells.
    """
    while frame is not None:
        if frame.f_code.co_filename == CELL_FILENAME:
            return True
        frame = frame.f_back
    return False


def main() -> int:
    """Entry point. `python -m ph_runtime`, spawned by the host with fd 3 attached."""
    return asyncio.run(_serve())


async def _serve() -> int:
    channel = await Channel.open()
    boot = await channel.receive()
    if boot is None or boot.get("type") != "boot":
        return 1
    if boot.get("protocol") != PROTOCOL_VERSION:
        # Refused rather than served: a guest that misreads one frame at a time
        # is worse than a guest that will not start (D7).
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
