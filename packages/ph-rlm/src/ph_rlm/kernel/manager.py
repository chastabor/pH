"""`code-runtime-python` — one CPython child per agent, governed at the frame (D1).

This is the provider dsh withheld. Its README says why: *"`run_code` state is
fresh per run — a persistent REPL-style kernel is rejected for the MVP
(cross-call state would be invisible to the log)"*. The objection is right, and
what answers it is not this file but the obligation the seam checks at
registration — `persistence: "namespace"` requires `kernel/snapshot` emission —
so the state a cell leaves behind is in the log with everything else.

Three properties are worth reading before changing anything here.

**A run reads its own frames, and owns the tasks that serve them.** The obvious
design is a long-lived reader task per kernel, and it was the first one here —
but a background task needs a task group, and a group entered when the kernel
starts is exited when the kernel closes, which is a *different task*. anyio
refuses that, correctly, and the symptom was a `ClosedResourceError` surfacing
from a cancelled drain instead of the failure being reported.

So `run()` opens a task group for the duration of one program: it reads frames
inline until `done`, starts one task per concurrent binding call, and drains the
child's stdout and stderr in the same group. Everything is entered and exited by
one task. `done` is therefore the last frame of a run — which is why the guest
snapshots *before* settling.

None of this reintroduces the deadlock that made fd 3 a separate channel (D5): a
cell awaiting a `reply` is not blocking the host's read loop, because the host's
read loop is what delivers the reply.

**Every inbound frame is rebuilt, never trusted.** The child executes
model-written code and holds the descriptor, so it is a hostile peer (C10). See
`codec.decode`.

**A child that dies is replaced, and the model is told.** The namespace is gone —
that is not recoverable — but the session is not, so the next run gets a fresh
kernel prefixed with a reset notice rather than a dead harness.

@module ph_rlm.kernel.manager
"""

from __future__ import annotations

import logging
import signal
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import anyio
import anyio.abc

from ph.cancel import CancelToken, is_cancelled
from ph.cordis import Context, Disposer, plugin
from ph.paths import resolve_roots
from ph.seams.code_runtime import (
    CodeBinding,
    CodeBindingNamespace,
    CodeRunRequest,
    CodeRunResult,
)
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.subprocess import scrub_env
from ph.seams.workspace import workspace_of
from ph.session.json import thaw_json
from ph.tools.code_mode import CodeRunFailure, ToolCallError
from ph.tools.errors import error_message
from ph.wire import WireModel

from .codec import decode, encode
from .journal import JOURNAL_NAME, OrphanJournal
from .protocol import (
    FD_ENV,
    PROTOCOL_VERSION,
    BootFrame,
    CancelFrame,
    ReplyFrame,
    RestoreFrame,
    RunFrame,
    ShutdownFrame,
    truncation_marker,
)
from .venv import InterpreterMode, RuntimeEnvironment, resolve_interpreter

__all__ = ["RESET_NOTICE", "Config", "Kernel", "KernelLimits", "PythonCodeRuntime", "apply"]

log = logging.getLogger("ph_rlm.kernel.manager")

RESET_NOTICE = "<runtime_reset>"
"""Prefixed to the first result after a kernel died.

Named rather than described: the namespace is gone and every variable with it, so
a model that reads this and re-derives what it needs is behaving correctly. Left
unsaid, it would re-run one cell at a time discovering `NameError`s."""

MAX_FRAME_BYTES = 64 * 1024 * 1024
"""The largest inbound frame the host will assemble.

The guest caps its own reads at the same number; this is the host's side of it,
and the reason it exists is that the child can write whatever it likes onto
fd 3 (C10). Sized to hold a `maxSnapshotBytes` payload with base64 and JSON
overhead."""

_CANCEL_POLL_SECONDS = 0.05
"""`CancelToken` is a polled flag, not an awaitable — it has to answer "was this
cancelled" at points where nothing is pending, which a cancel scope cannot do. So
the run waits on the settle event in short hops and checks the token between."""


class KernelLimits(WireModel):
    """What one child may consume. Every value is sent; the guest has no defaults.

    A `WireModel` rather than a dataclass so `Config` can extend it: a limit
    added here reaches the row config and the `boot` frame without being typed
    again in either. It was three declarations and two hand-written copies, which
    is four edits before the protocol is even touched.
    """

    cpu_seconds: int = 30
    address_space_bytes: int = 2 * 1024**3
    max_log_bytes: int = 65_536
    max_value_bytes: int = 65_536
    max_snapshot_bytes: int = 16 * 1024 * 1024

    def to_boot(
        self,
        *,
        namespaces: list[dict[str, Any]],
        namespace_id: str | None,
        skills: Sequence[str],
    ) -> BootFrame:
        return BootFrame(
            **{name: getattr(self, name) for name in KernelLimits.model_fields},
            namespaces=namespaces,
            namespace_id=namespace_id,
            skills=list(skills),
        )


@dataclass(slots=True)
class _ActiveRun:
    """Everything one program produces, collected as its frames arrive."""

    run_id: int
    bindings: dict[tuple[str, str], CodeBinding]
    settled: bool = False
    logs: list[str] = field(default_factory=list)
    displays: list[dict[str, Any]] = field(default_factory=list)
    value: Any = None
    error: str | None = None
    truncated: bool = False
    failure: CodeRunFailure | None = None
    """A refusal or a budget, raised out of `run()` once the program unwinds (C3)."""
    aborting_since: float | None = None
    """When this run was asked to stop, by either route — the caller cancelling,
    or a dispatch being refused. Held here rather than in `_pump` because
    `_serve_call` starts the abort from its own task, and the escalation clock
    the pump runs has to be the same clock."""


class SnapshotPolicy(Protocol):
    """What the runtime needs of whoever keeps the namespace in the log (D17).

    Two responsibilities, deliberately apart: the provider knows about processes
    and frames, the policy knows about events and blobs. Neither has to know
    both, and `ph_rlm.snapshot` can fold a stored log with no runtime running.
    """

    async def record(self, namespace: str, run_id: int, variables: list[dict[str, Any]]) -> None:
        """Persist the changed variables of one settled run."""
        ...

    async def materialize(self, namespace: str) -> list[dict[str, Any]]:
        """The payloads a freshly started kernel should be given back."""
        ...

    async def restored(self, namespace: str, outcome: dict[str, Any]) -> None:
        """Record which of them came back and which did not."""
        ...


class KernelDied(RuntimeError):
    """The child is gone. The namespace with it; the session is not."""


@dataclass(slots=True)
class Kernel:
    """One child process, one persistent namespace."""

    namespace: str
    environment: RuntimeEnvironment
    limits: KernelLimits
    journal: OrphanJournal
    cwd: Path | None = None
    """The child's working directory — `workspace.root` for this agent (D21).

    **This is what makes the `worktree` tier bound authored code rather than
    merely observe it.** A cell's `open("notes.txt", "w")` reaches no policy
    waterfall by construction (N1), but it does resolve against this directory,
    so a relative write lands in the agent's own checkout. An absolute path still
    escapes, and only the `sandbox` tier refuses that (§4.8, E13)."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for the child, from `workspace.env`.

    The build-tool redirection (E12): `TMPDIR`, `PYTEST_ADDOPTS` and friends
    pointed inside the agent's scratch, so a cell that runs the project's tests
    does not dirty the worktree with `.pytest_cache/` — which would flip the
    disposal policy from "remove a clean tree" to "keep everything"."""
    snapshots: SnapshotPolicy | None = None
    skills: tuple[str, ...] = ()
    """Import names bound callable at boot (P3-18), from `rlm-skills-python`."""
    boot_timeout: float = 30.0
    shutdown_grace: float = 5.0
    cancel_grace: float = 2.0
    """How long a cancelled cell has to unwind before the child is killed.

    **What the abort ladder can and cannot do**, stated here because this field is
    the ladder's only tuning point. `cancel` frame → `SIGINT` → `SIGKILL` at
    expiry stops a cell at its next yield point, or kills it if it never reaches
    one. It cannot preempt straight-line synchronous Python: statements after a
    caught refusal run before the guest's loop regains control, so C3's bound is
    "about one cell" rather than exactly one. The residue is narrower than the
    raw-`pathlib` non-goal (§11, Q10) and distinct from it: this one is about
    *time*, and a deployment widens it by raising this number."""

    _process: anyio.abc.Process | None = None
    _sock: socket.socket | None = None
    _run_seq: int = 0
    _buffer: bytearray = field(default_factory=bytearray)
    _scanned: int = 0
    """How much of `_buffer` has already been searched for a frame boundary, so
    a large frame is scanned once rather than once per chunk."""
    _alive: bool = False
    _reset_notice: bool = False
    _lock: anyio.Lock = field(default_factory=anyio.Lock)
    _send_lock: anyio.Lock = field(default_factory=anyio.Lock)
    """Serializes writes. `wait_writable` refuses two waiters on one socket, and
    there are genuinely three writers: the run loop sending `run`/`cancel`, and a
    `_serve_call` task per concurrent binding call answering with `reply`."""

    # ---------------------------------------------------------------- start --

    async def start(self, namespaces: list[dict[str, Any]]) -> None:
        """Spawn the child, hand it fd 3, and wait for it to report ready.

        Also the *restart* path, so the read buffer is cleared here: a leftover
        half-frame would otherwise prepend one incarnation's bytes to the next
        one's `boot-ack`.
        """
        await self._teardown()
        self._buffer.clear()
        self._scanned = 0
        host_end, child_end = socket.socketpair()
        argv = [str(self.environment.python), "-m", "ph_runtime"]
        child_fd = child_end.fileno()
        # `scrub_env`, not `os.environ`: this is the child the seam's own
        # docstring describes — "a child runs code the model wrote, so it does
        # not inherit `*KEY*`". A cell that can read `os.environ` can print the
        # provider credential into its own output, which is then logged.
        environ = scrub_env(
            extra={
                **self.env,
                "NO_COLOR": "1",
                "PYTHONUNBUFFERED": "1",
                FD_ENV: str(child_fd),
            }
        )
        try:
            self._process = await anyio.open_process(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environ,
                cwd=str(self.cwd) if self.cwd is not None else None,
                pass_fds=(child_fd,),
            )
        finally:
            # Closed in the parent either way: held open, the parent would never
            # see the child's EOF, and a failed spawn would leak a descriptor.
            child_end.close()

        host_end.setblocking(False)
        self._sock = host_end
        self._alive = True
        pid = self._process.pid
        if pid is not None:
            self.journal.record(pid=pid, argv=argv, namespace=self.namespace)

        await self._send(
            self.limits.to_boot(
                namespaces=namespaces, namespace_id=self.namespace, skills=self.skills
            )
        )
        try:
            with anyio.fail_after(self.boot_timeout):
                fault = await self._await_boot_ack()
        except TimeoutError as timeout:
            await self.aclose()
            raise KernelDied(
                f"the runtime did not report ready within {self.boot_timeout}s "
                f"({self.environment.describe()})"
            ) from timeout
        if fault is not None:
            await self.aclose()
            raise KernelDied(fault)

    async def _rehydrate(self) -> None:
        """Hand a freshly started kernel the namespace the log remembers (D17).

        Not on a restart after a crash: those payloads describe a namespace the
        model already knows is gone, and silently reconstituting half of it would
        be worse than the empty namespace the reset notice announces.
        """
        if self.snapshots is None or self._reset_notice:
            return
        variables = await self.snapshots.materialize(self.namespace)
        if not variables:
            return
        outcome = await self._restore(variables)
        await self.snapshots.restored(self.namespace, outcome)

    async def _await_boot_ack(self) -> str | None:
        """Wait for the child to report ready. Returns a fault message, or `None`.

        What it applied is logged rather than stored: the three die-with-parent
        mechanisms carry genuinely different guarantees — a session that ran under
        `getppid-poll` had a one-second window in which a hard-killed host could
        leave a stray — and that belongs in the record, not in a field nothing
        reads.
        """
        while True:
            line = await self._recv_line()
            if line is None:
                return (
                    "the runtime exited before reporting ready; "
                    f"{self.environment.describe()} could not start ph_runtime"
                )
            frame = decode(line)
            if frame is None:
                continue
            if frame["type"] == "boot-ack":
                if frame["protocol"] != PROTOCOL_VERSION:
                    return f"the runtime speaks protocol {frame['protocol']}"
                log.info(
                    "ph_rlm.kernel: %s ready on python %s, limits %s",
                    self.namespace,
                    frame["python"],
                    frame["limits"],
                )
                return None
            if frame["type"] == "fault":
                return str(frame["message"])

    # ----------------------------------------------------------------- runs --

    async def run(
        self,
        program: str,
        namespaces: Sequence[CodeBindingNamespace],
        token: CancelToken | None,
    ) -> CodeRunResult:
        """Run one program to settlement. Serialized: one namespace, one program."""
        async with self._lock:
            declaration = [_declare(namespace) for namespace in namespaces]
            restarted = False
            if not self._alive:
                await self.start(declaration)
                await self._rehydrate()
                # Consumed by the run that *gets* the fresh namespace, not by the
                # run that lost the old one: the cell that called `os._exit` knows
                # what it did, and the next one is the one facing an empty
                # namespace with no idea why.
                restarted, self._reset_notice = self._reset_notice, False
            self._run_seq += 1
            active = _ActiveRun(
                run_id=self._run_seq,
                bindings={
                    (namespace.name, binding.name): binding
                    for namespace in namespaces
                    for binding in namespace.bindings
                },
            )
            async with anyio.create_task_group() as tasks:
                process = self._process
                tasks.start_soon(self._drain, process.stdout if process else None, active)
                tasks.start_soon(self._drain, process.stderr if process else None, active)
                await self._send(RunFrame(id=active.run_id, program=program))
                await self._pump(active, tasks, token)
                # The drains and any in-flight call tasks belong to this run.
                tasks.cancel_scope.cancel()

            if active.failure is not None:
                # C3: the refusal ends the tool call, not just the program, so it
                # reaches the model as the reason the cell produced nothing.
                raise active.failure
            logs = "".join(active.logs)
            if restarted:
                logs = f"{RESET_NOTICE} the runtime restarted; the namespace is empty.\n{logs}"
            return CodeRunResult(
                logs=logs,
                value=active.value,
                error=active.error,
                truncated=active.truncated,
                reset=restarted,
                displays=tuple(active.displays),
            )

    async def _pump(
        self, active: _ActiveRun, tasks: anyio.abc.TaskGroup, token: CancelToken | None
    ) -> None:
        """Read this run's frames until it settles, checking cancellation between."""
        while not active.settled:
            line: bytes | None = None
            with anyio.move_on_after(_CANCEL_POLL_SECONDS):
                line = await self._recv_line()
                if line is None:
                    self._on_closed(active)
                    return
            if line is not None:
                frame = decode(line)
                if frame is not None:
                    # A forged or garbled frame is dropped. Raising here would let
                    # the child crash the host on demand (C10).
                    await self._handle(frame, active, tasks)
                continue
            if active.aborting_since is None and is_cancelled(token):
                await self._begin_abort(active)
            elif (
                active.aborting_since is not None
                and anyio.current_time() - active.aborting_since > self.cancel_grace
            ):
                # Neither the frame nor the signal reached it, which means the
                # cell is spinning in Python and the guest's loop is starved.
                # Killing costs the namespace; leaving it costs the session.
                await self._kill_unresponsive(active)
                return

    async def _begin_abort(self, active: _ActiveRun) -> None:
        """Start the stop ladder and the clock that escalates it."""
        active.aborting_since = anyio.current_time()
        await self._interrupt(active.run_id)

    async def _interrupt(self, run_id: int) -> None:
        """Ask twice, by two routes that fail in different ways (D5).

        The frame is read by the guest's reader task; `SIGINT` arrives as a loop
        callback. Both are cooperative and both need the guest's loop to be
        running, so a cell spinning in Python answers neither — that is what the
        grace period and `_kill_unresponsive` are for.
        """
        await self._send(CancelFrame(id=run_id))
        process = self._process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError, OSError):
                process.send_signal(signal.SIGINT)

    async def _kill_unresponsive(self, active: _ActiveRun) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError, OSError):
                process.kill()
        await self._teardown()
        self._reset_notice = True
        active.error = (
            "the program did not stop when cancelled and the runtime was killed; "
            "the namespace is gone"
        )
        active.settled = True

    async def _restore(self, variables: list[dict[str, Any]]) -> dict[str, Any]:
        """Put snapshotted variables back before the next run (D17).

        Called only from `_rehydrate`, which runs *inside* `run()` — so the run
        lock is already held and must not be taken again: `anyio.Lock` is not
        reentrant, and going through a locking wrapper here raised instead of
        waiting, surfacing as a restored namespace becoming a failed cell.
        """
        if not self._alive:
            return {"restored": [], "failed": [record.get("var") for record in variables]}
        self._run_seq += 1
        active = _ActiveRun(run_id=self._run_seq, bindings={})
        async with anyio.create_task_group() as tasks:
            await self._send(RestoreFrame(id=active.run_id, variables=variables))
            with anyio.move_on_after(self.boot_timeout):
                await self._pump(active, tasks, None)
            tasks.cancel_scope.cancel()
        return active.value if isinstance(active.value, dict) else {}

    # --------------------------------------------------------------- frames --

    async def _handle(
        self, frame: dict[str, Any], active: _ActiveRun, tasks: anyio.abc.TaskGroup
    ) -> None:
        kind = frame["type"]
        if kind == "call":
            tasks.start_soon(self._serve_call, frame, active)
        elif kind == "log":
            active.logs.append(frame["text"])
            if frame.get("truncated"):
                active.truncated = True
        elif kind == "display":
            active.displays.append(frame)
        elif kind == "snapshot":
            # Awaited, not spawned: the guest sends this *before* `done`, so the
            # namespace is durable before the model is told the cell finished.
            # The same rule as the checkpoint barriers (A4) — a side effect whose
            # record could not be written is worse than one that did not happen.
            if frame["id"] == active.run_id and self.snapshots is not None:
                await self.snapshots.record(self.namespace, frame["id"], frame["variables"])
        elif kind == "done":
            self._settle(frame, active)

    def _settle(self, frame: dict[str, Any], active: _ActiveRun) -> None:
        if frame["id"] != active.run_id:
            # A `done` for another run is a forged frame — the guest sends one
            # per run and only for the open one — so it settles nothing (C10).
            return
        error = frame.get("error")
        if isinstance(error, dict):
            active.error = str(error.get("message") or error.get("kind") or "the program failed")
        else:
            active.value = frame.get("value")
        if frame.get("truncated"):
            active.truncated = True
        active.settled = True

    async def _serve_call(self, frame: dict[str, Any], active: _ActiveRun) -> None:
        """One binding call, back through the full tool pipeline (C1)."""
        key = (frame["global"], frame["name"])
        binding = active.bindings.get(key)
        if binding is None or binding.dispatch is None:
            await self._reply(frame["id"], ok=False, message=f"{key[0]}.{key[1]} is not available")
            return
        try:
            value = await binding.dispatch(**frame["args"])
        except CodeRunFailure as failure:
            # Recorded, answered, *and* aborted — all three, because only the
            # third actually enforces C3.
            #
            # The reply makes a well-behaved cell raise `RunStopped` and unwind.
            # But a cell is not obliged to behave: `except BaseException: pass`
            # followed by `Path(...).write_text(...)` completed the write, and
            # the run then "failed" afterwards — the tool call reported a refusal
            # the program had already routed around. Raw Python is not reachable
            # by any waterfall, so the only thing that can stop it is ending the
            # process's turn: the same frame-then-signal-then-kill ladder user
            # cancellation uses. "Run-scoped abort fires" is the plan's wording
            # and this is it.
            active.failure = failure
            await self._reply(frame["id"], ok=False, message=failure.message, fatal=True)
            if active.aborting_since is None:
                await self._begin_abort(active)
        except ToolCallError as error:
            await self._reply(frame["id"], ok=False, message=error.message)
        except Exception as error:
            await self._reply(frame["id"], ok=False, message=error_message(error))
        else:
            await self._reply(frame["id"], ok=True, value=_json_safe(value))

    async def _reply(
        self,
        call_id: int,
        *,
        ok: bool,
        value: Any = None,
        message: str | None = None,
        fatal: bool | None = None,
    ) -> None:
        await self._send(ReplyFrame(id=call_id, ok=ok, value=value, message=message, fatal=fatal))

    # ----------------------------------------------------------------- pipes --

    async def _drain(self, stream: Any, active: _ActiveRun) -> None:
        """Collect the *process's* own fd 1/2 — a grandchild's output, mainly.

        A `print` in the cell arrives as a `log` frame instead, because the guest
        redirects `sys.stdout`. What reaches these pipes is what a subprocess the
        cell spawned wrote, and it is capped for the same reason (D4).

        Both streams append into the run's one ordered log, so the result text
        reads in the order things actually happened — which is what prime-agent's
        `stdout + stderr + result` concatenation was approximating.
        """
        if stream is None:
            return
        written = 0
        cap = self.limits.max_log_bytes
        # `ClosedResourceError` is the ordinary end of this task: the run is over
        # and the group cancelled it, or the child exited. Neither is a failure
        # to report, and letting it escape would mask the real outcome.
        with suppress(anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream):
            async for chunk in stream:
                if written >= cap:
                    continue
                text = chunk.decode("utf-8", "replace")
                room = cap - written
                if len(text) > room:
                    active.logs.append(text[:room])
                    active.logs.append(truncation_marker(len(text) - room, cap))
                    active.truncated = True
                    written = cap
                else:
                    active.logs.append(text)
                    written += len(text)

    async def _recv_line(self) -> bytes | None:
        """The next frame's bytes, or `None` when the child is gone or hostile.

        Two things here are not incidental.

        **The buffer is a `bytearray` and only its tail is scanned.** With `bytes`
        and `+=`, a multi-megabyte frame copies the whole buffer per 64 KiB chunk
        and re-scans it for a newline: a 16 MiB snapshot spent **834 ms of
        1160 ms** doing exactly that. Appending to a `bytearray` and searching
        from the previous length makes both linear — measured 1160 ms → 221 ms.

        **The buffer is capped.** The child holds this descriptor and runs
        model-written code, so `os.write(3, b"x" * 10**10)` is a thing it can do;
        without a cap the host grows the buffer until it is killed. A frame past
        the cap is not a frame — it is treated as the channel being unusable
        (C10).
        """
        sock = self._sock
        if sock is None:
            return None
        buffer = self._buffer
        while True:
            index = buffer.find(b"\n", self._scanned)
            if index >= 0:
                line = bytes(buffer[:index])
                # `del` on the front of a bytearray is amortized O(1) in CPython.
                del buffer[: index + 1]
                self._scanned = 0
                return line
            self._scanned = len(buffer)
            if self._scanned > MAX_FRAME_BYTES:
                log.warning(
                    "ph_rlm.kernel: the runtime sent %d bytes with no frame boundary; "
                    "closing the channel",
                    self._scanned,
                )
                return None
            try:
                await anyio.wait_readable(sock)
                chunk = sock.recv(65536)
            except (OSError, anyio.ClosedResourceError):
                return None
            if not chunk:
                return None
            buffer += chunk

    async def _send(self, frame: WireModel) -> None:
        sock = self._sock
        if sock is None:
            return
        view = memoryview(encode(frame))
        async with self._send_lock:
            while view:
                try:
                    await anyio.wait_writable(sock)
                    sent = sock.send(view)
                except (OSError, anyio.ClosedResourceError):
                    # Only a real closure gets here now. Contention used to land
                    # in this branch as `BusyResourceError` and was reported as
                    # the child having exited — eight concurrent replies were
                    # enough to "kill" a perfectly healthy kernel.
                    self._on_closed()
                    return
                view = view[sent:]

    def _on_closed(self, active: _ActiveRun | None = None) -> None:
        """The child is gone. Fail whatever was waiting; do not lose the session."""
        if self._alive:
            self._alive = False
            self._reset_notice = True
        if active is not None and not active.settled:
            active.error = "the runtime exited before the program finished"
            active.settled = True

    # ---------------------------------------------------------------- close --

    async def aclose(self) -> None:
        """Ask, then wait, then kill — and reap on every path (F4)."""
        await self._teardown()

    async def _teardown(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            await self._send(ShutdownFrame())
            with anyio.move_on_after(self.shutdown_grace):
                await process.wait()
            if process.returncode is None:
                with suppress(ProcessLookupError, OSError):
                    process.kill()
        self._alive = False
        if process is not None:
            # A child that exited while the parent lives and is never reaped is a
            # zombie; this is the `finally` that prevents one (F4).
            with suppress(Exception):
                await process.wait()
            if process.pid is not None:
                self.journal.forget(process.pid)
        if self._sock is not None:
            with suppress(OSError):
                self._sock.close()
            self._sock = None


@dataclass(slots=True)
class PythonCodeRuntime:
    """The `ctx.code_runtime` provider: one kernel per namespace, spawned lazily."""

    language: ClassVar[str] = "python"
    isolation: ClassVar[Literal["process"]] = "process"
    persistence: ClassVar[Literal["namespace"]] = "namespace"
    declares_kernel_snapshots: ClassVar[bool] = True
    """The promise the seam checks at registration (D6). `rlm-kernel-snapshot`
    turns the frames this provider surfaces into `kernel/snapshot` events."""

    limits: KernelLimits
    journal: OrphanJournal
    cache: Path
    interpreter_mode: InterpreterMode = "managed"
    interpreter_override: str | None = None
    skills: tuple[str, ...] = ()
    """Requirement specs installed into the managed venv. A local directory is
    installed editable, so editing a skill does not need a venv rebuild."""
    skill_modules: tuple[str, ...] = ()
    """The import names those specs provide, handed to each kernel at boot.

    Two lists rather than one because they answer different questions and can
    legitimately differ: a distribution named `acme-websearch` imports as
    `acme_websearch`, and `rlm-skills-python` is what knows both."""
    workspaces: Callable[[str], Any] | None = None
    """Agent id → its `Workspace`, set by the row (D21). See `workspace_for`."""
    boot_timeout: float = 30.0
    shutdown_grace: float = 5.0
    cancel_grace: float = 2.0
    snapshots: SnapshotPolicy | None = None
    """Set by the `rlm-kernel-snapshot` row. Absent, the runtime still runs — but
    `persistence: "namespace"` would then be a promise nothing keeps, which is
    why the bundle mounts both rows together."""
    _kernels: dict[str, Kernel] = field(default_factory=dict)
    _scopes: dict[str, Context] = field(default_factory=dict)
    """Agent id → its scope, so a kernel is released *structurally* (F1).

    The alternative was closing on the `agent/disposed` event, but `emit`
    schedules an async listener without awaiting it — so release would happen
    eventually rather than as part of unwinding. Registering the disposer on the
    agent's own scope makes the child process an artifact of that scope, which
    is what every other acquired resource in pH already is."""
    _environment: RuntimeEnvironment | None = None
    _resolve_lock: anyio.Lock = field(default_factory=anyio.Lock)

    def describe(self) -> list[tuple[str, str]]:
        """What `ph doctor` prints about the workers that run model code (I-2).

        **Nothing here resolves the interpreter.** `environment()` shells out to
        `uv` to build the managed venv, and a diagnostic that took thirty
        seconds and a network the first time someone ran it would be a
        diagnostic people stop running. So an unresolved interpreter is reported
        as unresolved, which is also the more useful fact: it says the venv has
        not been built yet.
        """
        environment = self._environment
        interpreter = (
            environment.describe()
            if environment is not None
            else f"{self.interpreter_mode} — not resolved yet, built on the first cell"
        )
        gib = self.limits.address_space_bytes / 1024**3
        return [
            ("workers", "one CPython child per agent namespace, spawned lazily"),
            ("on a dead child", "replaced, and the model is told the namespace was lost"),
            ("interpreter", interpreter),
            ("per-child limits", f"{self.limits.cpu_seconds}s CPU, {gib:.3g} GiB address space"),
            ("live kernels", str(len(self._kernels))),
        ]

    async def environment(self) -> RuntimeEnvironment:
        """Resolve the interpreter once, on first use.

        Lazily, and in a worker thread: building the managed venv shells out to
        `uv`, and neither `ph --dump-config` nor a session that runs no cells
        should pay for it.
        """
        async with self._resolve_lock:
            if self._environment is None:
                self._environment = await anyio.to_thread.run_sync(
                    lambda: resolve_interpreter(
                        cache=self.cache,
                        mode=self.interpreter_mode,
                        skills=self.skills,
                        override=self.interpreter_override,
                    )
                )
            return self._environment

    def workspace_for(self, agent_id: str) -> Any:
        """This agent's workspace, or `None` before one is acquired.

        Resolved per kernel rather than held as one runtime-wide `cwd`, because
        the namespace *is* the agent id: a parent and its children share this
        runtime and must not share a checkout, which is the collision the
        `worktree` tier exists to prevent.
        """
        if self.workspaces is None:
            return None
        try:
            return self.workspaces(agent_id)
        except Exception:
            log.warning("ph_rlm.kernel: workspace lookup failed for %s", agent_id, exc_info=True)
            return None

    def remember_scope(self, agent: Any) -> None:
        """Note an agent's scope, so its kernel can be owned by it."""
        agent_id = getattr(agent, "id", None)
        scope = getattr(agent, "ctx", None)
        if isinstance(agent_id, str) and scope is not None:
            self._scopes[agent_id] = scope

    async def run(self, request: CodeRunRequest) -> CodeRunResult:
        namespace = request.namespace or "default"
        kernel = self._kernels.get(namespace)
        if kernel is None:
            kernel = await self._acquire(namespace)
        token = request.cancel_scope if isinstance(request.cancel_scope, CancelToken) else None
        return await kernel.run(request.program, request.bindings, token)

    async def _acquire(self, namespace: str) -> Kernel:
        workspace = self.workspace_for(namespace)
        kernel = Kernel(
            namespace=namespace,
            environment=await self.environment(),
            limits=self.limits,
            journal=self.journal,
            cwd=None if workspace is None else workspace.root,
            env={} if workspace is None else workspace.env,
            snapshots=self.snapshots,
            skills=self.skill_modules,
            boot_timeout=self.boot_timeout,
            shutdown_grace=self.shutdown_grace,
            cancel_grace=self.cancel_grace,
        )
        self._kernels[namespace] = kernel
        scope = self._scopes.get(namespace)
        if scope is not None:

            async def enter() -> Disposer:
                return partial(self.close_namespace, namespace)

            await scope.effect(enter, label=f"code-runtime:{namespace}")
        return kernel

    async def close_namespace(self, namespace: str) -> None:
        """Shut one agent's kernel down when the agent goes."""
        kernel = self._kernels.pop(namespace, None)
        if kernel is not None:
            await kernel.aclose()

    async def aclose(self) -> None:
        for namespace in list(self._kernels):
            await self.close_namespace(namespace)


def _declare(namespace: CodeBindingNamespace) -> dict[str, Any]:
    """What the guest needs to build a proxy: names, not dispatch closures."""
    return {
        "name": namespace.name,
        "description": namespace.description,
        "bindings": [
            {"name": binding.name, "description": binding.description}
            for binding in namespace.bindings
        ],
    }


def _json_safe(value: Any) -> Any:
    """A tool's result in a form the reply frame can carry.

    `thaw_json` is the whole job: a value that came back through the log is
    frozen — a `MappingProxyType` over tuples — and `json.dumps` will not
    serialize that. There is no round trip through JSON here, because `encode`
    is about to serialize the frame anyway; doing it twice cost 2.8 ms against
    1.2 ms for a 1 MiB result. Anything `json` still cannot represent is handled
    by `encode`'s `default`.
    """
    return thaw_json(value)


class Config(KernelLimits):
    """Row config for the Python runtime: the limits, plus how to reach a child.

    Extends `KernelLimits` so the YAML stays flat (`cpuSeconds: 30`) and the
    limits have one declaration.
    """

    python: InterpreterMode = "managed"
    """`managed` builds `$PH_CACHE/runtime-venv`; `host` reuses pH's own
    interpreter, which is faster and wider — see `venv`."""
    interpreter: str | None = None
    boot_timeout_seconds: float = 30.0
    shutdown_grace_seconds: float = 5.0
    cancel_grace_seconds: float = 2.0
    """How long a cell asked to stop has before it is killed.

    Both directions cost something: longer widens the window in which a cell that
    swallowed a refusal can finish synchronous work (see `Kernel.cancel_grace`);
    shorter kills a cell that is legitimately slow to unwind, and killing costs
    the namespace."""
    skills: tuple[str, ...] = ()
    sweep_orphans: bool = True


@plugin("code-runtime-python", config=Config, inject=["code_runtime"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the runtime, and sweep strays from a run that was hard-killed."""
    roots = resolve_roots()
    journal = OrphanJournal(path=roots.runtime / JOURNAL_NAME)
    if config.sweep_orphans:
        # At every start, because a session nobody reopens would never reconcile
        # its own strays (F5).
        report = await anyio.to_thread.run_sync(journal.sweep)
        if report.killed or report.unverifiable:
            log.info(
                "ph_rlm.kernel: killed %s, could not verify %s",
                list(report.killed),
                list(report.unverifiable),
            )

    runtime = PythonCodeRuntime(
        # `Config` *is* a `KernelLimits`, so the limits need no copying.
        limits=config,
        journal=journal,
        cache=roots.cache,
        interpreter_mode=config.python,
        interpreter_override=config.interpreter,
        skills=config.skills,
        boot_timeout=config.boot_timeout_seconds,
        shutdown_grace=config.shutdown_grace_seconds,
        cancel_grace=config.cancel_grace_seconds,
    )

    async def enter() -> Disposer:
        return runtime.aclose

    await ctx.effect(enter, label="code-runtime-python")
    ctx.provide("python_runtime", runtime)
    ctx.code_runtime.register(runtime)

    # The namespace *is* the agent id, so a kernel is scoped exactly like the
    # agent's tools, its inbox and its log — and released by the same unwinding.
    ctx.on("agent/created", runtime.remember_scope)

    # Asked at kernel start rather than captured here, because a workspace is
    # acquired after the agent exists (P4-08) and a value read at mount would be
    # `None` for every kernel this runtime ever spawns.
    runtime.workspaces = partial(workspace_of, ctx)

    contribute(
        ctx, Diagnostic(id="code-runtime", title="Code runtime", read=runtime.describe, order=40)
    )
