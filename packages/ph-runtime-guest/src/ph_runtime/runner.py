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
import reprlib
import signal
import sys
import time
import traceback
from typing import Any

from . import snapshot as snapshot_module
from .cell import CELL_FILENAME, CELL_FUNCTION, compile_cell
from .channel import Channel
from .errors import RunStopped, ToolFailed
from .lifecycle import die_with_parent
from .limits import CpuBudgetExceeded, apply_limits, arm_cpu_budget
from .protocol import PROTOCOL_VERSION, truncation_marker
from .proxies import build_namespaces
from .skill import UnavailableSkill, wrap_skill_module

__all__ = ["Runner", "main"]


FLUSH_BYTES = 8 * 1024
FLUSH_SECONDS = 0.05
"""When a coalesced output buffer goes out: full enough, or old enough.

The pair is what keeps *both* properties. Size alone would hold a slow cell's
first line until it had produced 8 KiB; time alone would send a frame per write
for a fast one. Together, a chatty cell sends kilobytes per frame and a slow one
still shows progress within 50 ms.
"""


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
        """Route `SIGINT` through the loop rather than into the running frame."""
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            # Not available on Windows, where the host's Job Object and the
            # `cancel` frame are the mechanisms that apply.
            loop.add_signal_handler(signal.SIGINT, self._on_interrupt)

    def _on_interrupt(self) -> None:
        run = self._run
        if run is not None and not run.done():
            run.cancel()

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
        self._run = asyncio.get_running_loop().create_task(
            self._execute(run_id, str(frame.get("program", "")))
        )

    def _resolve(self, frame: dict[str, Any]) -> None:
        call_id = frame.get("id")
        future = self._pending.pop(call_id, None) if isinstance(call_id, int) else None
        if future is None or future.done():
            return
        if frame.get("ok"):
            future.set_result(frame.get("value"))
            return
        message = str(frame.get("message") or "the call was refused")
        if frame.get("fatal"):
            future.set_exception(RunStopped(message))
        else:
            future.set_exception(ToolFailed(str(frame.get("name") or "the call"), message))

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
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run

    # ------------------------------------------------------------ dispatch --

    async def _dispatch(self, namespace: str, name: str, arguments: dict[str, Any]) -> Any:
        """Marshal one binding call and wait for the host's answer."""
        self._next_call_id += 1
        call_id = self._next_call_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        self.channel.send(
            {
                "type": "call",
                "id": call_id,
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
            error = {"kind": "aborted", "message": "the run was cancelled"}
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
        # Snapshotted *before* the run settles, so `done` is the last frame of a
        # run and nothing arrives after it. That lets the host read one run's
        # frames inline instead of keeping a reader task whose lifetime crosses
        # calls — and a task group entered in one task and exited in another is
        # exactly the bug that ordering caused.
        self._snapshot(run_id)
        self._settle(run_id, value, error, out, err)

    def _settle(
        self,
        run_id: int,
        value: Any,
        error: dict[str, Any] | None,
        out: _CappedStream,
        err: _CappedStream,
    ) -> None:
        # Buffered output goes out before `done`, so the host has every log
        # frame of a run before the frame that settles it.
        out.flush()
        err.flush()
        frame: dict[str, Any] = {"type": "done", "id": run_id}
        if error is not None:
            frame["error"] = error
        else:
            encoded, degraded = _encode_value(value, self.max_value_bytes)
            if encoded is not None:
                frame["value"] = encoded
            if degraded:
                frame["truncated"] = True
        if out.truncated or err.truncated:
            frame["truncated"] = True
        self.channel.send(frame)

    def _snapshot(self, run_id: int) -> None:
        records = self._snapshotter.changed(
            self.globals, protected=self._protected, max_value_bytes=self.max_snapshot_bytes
        )
        if records:
            self.channel.send({"type": "snapshot", "id": run_id, "variables": records})


def _plain(value: Any) -> Any:
    """Round-trip through JSON so a proxy object cannot ride along in `args`."""
    try:
        return json.loads(json.dumps(value, default=repr))
    except (TypeError, ValueError):  # pragma: no cover
        return {}


def _encode_value(value: Any, cap: int) -> tuple[Any, bool]:
    """The cell's value as JSON if it fits, else a bounded `repr`.

    Both halves stop at the cap rather than building the whole thing to measure
    it. `json.dumps` of a 1M-element list took 40 ms and `repr` another 37 ms —
    77 ms to produce 64 KiB — and "end the cell with `df`" is exactly how models
    write cells. `iterencode` stops as soon as the prefix is over the cap, and
    `reprlib` never builds more than it needs.
    """
    if value is None:
        return None, False
    size = 0
    try:
        for chunk in json.JSONEncoder().iterencode(value):
            size += len(chunk)
            if size > cap:
                return _bounded_repr(value, cap), True
    except (TypeError, ValueError):
        return _bounded_repr(value, cap), True
    return value, False


def _bounded_repr(value: Any, cap: int) -> str:
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
