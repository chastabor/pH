"""`code-runtime-python` — one CPython child per agent, governed at the frame (D1).

This is the provider dsh withheld, on the grounds that *"cross-call state would be
invisible to the log"*. What answers that is not this file but the obligation the
seam checks at registration — `persistence: "namespace"` requires
`kernel/snapshot` emission — so the state a cell leaves behind is in the log with
everything else.

Three properties are worth reading before changing anything here.

**A run reads its own frames, and owns the tasks that serve them.** `run()` opens
a task group for the duration of one program: it reads frames inline until `done`,
starts one task per concurrent binding call, and drains the child's stdout and
stderr in the same group, so everything is entered and exited by **one** task. A
long-lived reader task per kernel — the obvious design, and the first one here —
needs a group entered when the kernel starts and exited when it closes, which is a
*different* task; anyio refuses that, and the symptom was a `ClosedResourceError`
from a cancelled drain instead of the real failure being reported. `done` is
therefore the last frame of a run, which is why the guest snapshots *before*
settling.

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
from ph.seams.sandbox import ConfinedArgv, SandboxPolicy
from ph.seams.subprocess import first_line, scrub_env
from ph.seams.workspace import workspace_of, workspace_policy
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

STDERR_GRACE = 2.0
"""How long `_stderr_so_far` waits for a child that has not closed its stderr.

Named because it is reachable: the paths that quote a failed child's words include
one where the child is still running (see `_stderr_so_far`), and an unnamed literal
in a `move_on_after` reads as though nobody expected to wait at all."""

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
    so a relative write lands in the agent's own checkout. An absolute path escapes
    *this* mechanism, and only confinement refuses it (§4.8, E13) — which is
    `confine` below, when a backend is there to do it."""
    confine: Callable[[tuple[str, ...]], ConfinedArgv] | None = None
    """How to bound this child at the kernel, or `None` where nothing can.

    **This is what makes `sandbox` mean something for authored code.** `cwd` above
    bounds a *relative* write by putting the child in the agent's tree; only this
    refuses `open("/etc/passwd", "w")` from inside a cell, which §4.8 names as the
    one thing no tier below `sandbox` can do. Until it existed, `permissions-fs`
    told operators that "a sandbox provider bounds what a code cell can reach
    directly" while the kernel was spawned unconfined — the sentence is true now.

    Supplied by the runtime rather than resolved here, and resolved per *kernel*,
    because the writable set is the agent's own workspace and scratch. `None` when
    no backend is mounted or the agent has no workspace, which is the same
    condition `ctx.shell` declines on and for the same reason: the seam refuses
    rather than passing through, so asking for confinement that cannot be given
    would turn every cell into a `SANDBOX_UNAVAILABLE` denial."""
    applied_limits: dict[str, Any] = field(default_factory=dict)
    """What the guest reported it could actually apply, from `boot-ack`.

    **Stored because `ph doctor` reports it, and it is not always the request.**
    `KernelLimits` is what the host *asked* for; macOS refuses `RLIMIT_AS`
    outright, so a limit named in the request can simply not be in force. Printing
    the request there would claim a bound nothing enforces, which is E1's failure
    one layer down from the tier table — so `PythonCodeRuntime.describe` reads this
    and says "not applied" where the guest said `None`. Empty until `boot-ack`."""
    confined: ConfinedArgv | None = None
    """What `confine` produced, once this kernel has started.

    One field rather than a fact copied out of it per reader: `_interrupt` needs
    `forwards_signals` — which is *not* the same question as "was I confined at
    all", since `bwrap` puts a wrapper and a PID namespace in the way while
    `sandbox-exec` execs its target — and reporting a refusal needs the effective
    `policy` to know whether the network was even in play. Deriving either from
    "confinement happened" would drop a working cancel route on macOS and misread
    an outage as a denial."""
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
        argv = (str(self.environment.python), "-m", "ph_runtime")
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
        # **fd 3 crosses the boundary**, which is the property that makes confining
        # this child possible at all: `bwrap` passes an inherited descriptor
        # through to the command it execs, and so does the `sh -c … exec "$@"` the
        # egress shim wraps around it. Measured on both, with the socket live at
        # the far end — see `test_kernel_confinement.py`.
        spawn: tuple[str, ...] = argv
        if self.confine is not None:
            confined = self.confine(argv)
            spawn = confined.argv
            self.confined = confined
        try:
            self._process = await anyio.open_process(
                list(spawn),
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
            # What is actually running, wrapper included: the journal verifies an
            # orphan is ours before killing it, and the pid it holds is the
            # wrapper's.
            self.journal.record(pid=pid, argv=spawn, namespace=self.namespace)

        await self._send(
            self.limits.to_boot(
                namespaces=namespaces, namespace_id=self.namespace, skills=self.skills
            )
        )
        try:
            with anyio.fail_after(self.boot_timeout):
                fault = await self._await_boot_ack()
        except TimeoutError as timeout:
            # Quoted here too, and this is the branch that catches a wrapper which
            # *hangs* rather than exits — the egress shim waiting on its readiness
            # poll, a bind that never completes. Read before `aclose`, which drops
            # the process this reads from.
            said = first_line(await self._stderr_so_far())
            await self.aclose()
            raise KernelDied(
                f"the runtime did not report ready within {self.boot_timeout}s "
                f"({self.environment.describe()})" + (f": {said}" if said else "")
            ) from timeout
        if fault is not None:
            await self.aclose()
            raise KernelDied(fault)

    async def _stderr_so_far(self) -> str:
        """One read of whatever the child has already written to stderr.

        Bounded twice, because neither bound is redundant. `STDERR_GRACE` covers a
        child that is still *running* — the boot timeout reached this, and so does
        a frame flood, where `_recv_line` gives up on a child that is alive and
        writing — and the cap is the same one `_drain` applies to a stream the
        model can influence. `EndOfStream` on the ordinary "child died quietly"
        path is a normal outcome, not a failure, which is why everything is
        suppressed: this runs while reporting another failure and must not replace
        it with its own.
        """
        process = self._process
        if process is None or process.stderr is None:
            return ""
        with suppress(Exception), anyio.move_on_after(STDERR_GRACE):
            return (await process.stderr.receive(self.limits.max_log_bytes)).decode(
                "utf-8", "replace"
            )
        return ""

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
        """Read the child's first frame. Returns a fault message, or `None`.

        **Until `boot-ack`, the only frames that exist are `boot-ack` and `fault`,
        and anything else is a fault to report.** The guest sends `boot-ack` before
        `Runner.serve` and before any skill import or cell, so nothing
        model-written has written to fd 3 yet and C10's tolerance — junk is skipped,
        the peer is hostile — is not owed here. It is actively wrong here: every
        frame this loop declines to understand is one it would wait `boot_timeout`
        for in silence with nothing to quote, which is the failure this function was
        rewritten to end. So the phase is stated once, over both ways a guest can
        violate it (a line that will not decode, and a decodable frame that is not
        one of the two), rather than over only the first.

        The die-with-parent mechanism it applied is logged rather than stored: the
        three carry genuinely different guarantees — a session that ran under
        `getppid-poll` had a one-second window in which a hard-killed host could
        leave a stray — and that belongs in the record. What it *could not apply*
        is stored, because `describe()` reports it: see `applied_limits`.
        """
        while True:
            line = await self._recv_line()
            if line is None:
                # **The child's own words**, which this used to throw away. A guest
                # that cannot start says why on stderr — a missing module, an
                # interpreter that will not run, `bwrap` refusing a bind source —
                # and without it every one of those reads as the same sentence.
                # The same argument `probe_sandbox` makes about quoting a backend:
                # "the runtime did not start" is true and useless.
                said = first_line(await self._stderr_so_far())
                because = f": {said}" if said else ""
                return (
                    "the runtime exited before reporting ready; "
                    f"{self.environment.describe()} could not start ph_runtime{because}"
                )
            frame = decode(line)
            if frame is None:
                # Not C10's "junk → skip": nothing model-written has run yet, so a
                # line this host cannot read is the guest speaking a shape it does
                # not accept (D7), and skipping it means waiting out `boot_timeout`
                # in silence with nothing to quote. Measured on macOS: the guest
                # reported `addressSpaceBytes` as RLIM_INFINITY, the codec's
                # lossless-integer rule refused the frame, and every start ended
                # 60 s later as "did not report ready".
                return (
                    "the runtime's first frame could not be read as protocol "
                    f"{PROTOCOL_VERSION} ({self.environment.describe()}): "
                    f"{line[:200].decode('utf-8', 'replace')!r}"
                )
            if frame["type"] == "boot-ack":
                if frame["protocol"] != PROTOCOL_VERSION:
                    return f"the runtime speaks protocol {frame['protocol']}"
                limits = frame["limits"]
                log.info(
                    "ph_rlm.kernel: %s ready on python %s, limits %s",
                    self.namespace,
                    frame["python"],
                    limits,
                )
                self.applied_limits = limits if isinstance(limits, dict) else {}
                return None
            if frame["type"] == "fault":
                return str(frame["message"])
            # Readable, and still not one of the two frames that exist before the
            # guest is ready. Looping here is how a `log` or `done` arriving early
            # became the same silent `boot_timeout` an undecodable line did.
            return (
                f"the runtime sent {frame['type']!r} before reporting ready "
                f"({self.environment.describe()})"
            )

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

        **The signal route is dropped when it would not reach the guest, and that
        is a refusal to make cancellation destructive.** Under `bwrap`,
        `self._process` is the wrapper, and it does not forward signals — measured,
        it dies of `SIGINT` itself (`rc=-2`), taking the PID namespace and the
        agent's whole namespace with it. So sending it would turn every "stop this
        cell" into "lose everything the cell had", which is the outcome the ladder's
        last rung exists to avoid rather than to reach first.

        Asked of `ConfinedArgv.forwards_signals` rather than of "am I confined":
        `sandbox-exec` execs its target and signals land, so a backend that does
        forward keeps both routes.

        What is lost is narrower than it looks. The guest installs `SIGINT` through
        `loop.add_signal_handler`, so it arrives as an ordinary loop callback —
        the same loop the `cancel` frame's reader runs on — and the two therefore
        cover the *same* situation. The signal's independent value is as a backup
        for a wedged channel with a live loop, and `_kill_unresponsive` still
        covers that after the grace period.
        """
        await self._send(CancelFrame(id=run_id))
        if self.confined is not None and not self.confined.forwards_signals:
            return
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
            # third actually enforces C3. The reply makes a well-behaved cell
            # raise `RunStopped` and unwind, but a cell is not obliged to behave
            # and raw Python is not reachable by any waterfall — so the only
            # thing that can stop it is ending the process's turn, via the same
            # frame-then-signal-then-kill ladder user cancellation uses.
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

        **The buffer is a `bytearray` and only its tail is scanned.** With `bytes` and
        `+=`, a multi-megabyte frame copies the whole buffer per 64 KiB chunk and
        re-scans it for a newline, which is quadratic in the frame size. Appending to a
        `bytearray` and searching from the previous length makes both linear.

        **The buffer is capped.** The child holds this descriptor and runs model-written
        code, so `os.write(3, b"x" * 10**10)` is a thing it can do; without a cap the host
        grows the buffer until it is killed. A frame past the cap is not a frame — it is
        treated as the channel being unusable (C10).
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
                    # Only a real closure gets here: `_send_lock` is what keeps
                    # contention from arriving as `BusyResourceError` and being
                    # reported as the child having exited.
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
    sandbox: Callable[[], Any] | None = None
    """The `ctx.sandbox` seam, asked for **when a kernel starts** rather than held.

    A resolver rather than the seam itself, for the reason `workspaces` is one: a
    value read at mount would be `None` for every kernel this runtime ever spawns
    if the backend row is layered after this one, and `FsPermissions.ctx` is the
    same shape for the same reason. Absent entirely — a profile with no sandbox
    seam at all — leaves cells bounded by `cwd` and nothing else, which is what
    they were before."""
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
        return [
            ("workers", "one CPython child per agent namespace, spawned lazily"),
            ("on a dead child", "replaced, and the model is told the namespace was lost"),
            ("interpreter", interpreter),
            ("per-child limits", self._limits()),
            ("live kernels", str(len(self._kernels))),
            ("cells confined by", self._confinement()),
        ]

    def _limits(self) -> str:
        """The per-child limits, as the live kernels actually got them.

        **Read from the guests where there are any**, which is `_confinement`'s rule
        and for `_confinement`'s reason: the request is what the host asked for, and
        on a platform that refuses `RLIMIT_AS` — macOS does, with `ValueError:
        current limit exceeds maximum limit` — the number in force is not the number
        configured. Printing the request would be `ph doctor` claiming a bound
        nothing holds, which is the shape E1 forbids one level up in the tier table.

        The weakest live kernel wins, again as `_confinement` does: one child that
        could not take the limit means the row must not say every child has it.
        """
        gib = self.limits.address_space_bytes / 1024**3
        asked = f"{self.limits.cpu_seconds}s CPU, {gib:.3g} GiB address space"
        reports = [kernel.applied_limits for kernel in self._kernels.values()]
        started = [report for report in reports if report]
        if not started:
            return f"{asked} (requested; no kernel has started to apply them yet)"
        if any(report.get("addressSpaceBytes") is None for report in started):
            return (
                f"{self.limits.cpu_seconds}s CPU; "
                "address space not applied — this platform refused it"
            )
        return asked

    def _confinement(self) -> str:
        """Whether authored code is bounded at the kernel, and by what.

        Its own row in `ph doctor` because it is the difference between a cell that
        can write `/etc` and one that cannot, and because it is *conditional* — on a
        backend, and on the agent having a workspace. Read from the kernels that are
        actually running where there are any, so this reports what is true rather
        than what would be true.

        **It reports the weakest live kernel, not the strongest.** A first draft
        joined the set of backends that had confined something, which read as
        "bwrap — every live kernel's writes are bounded" in the very case that
        sentence is false: one agent with a workspace and one without. There is only
        ever one provider, so there was never a plurality to join — only a way to
        overstate.
        """
        seam = None if self.sandbox is None else self.sandbox()
        if seam is None or seam.provider is None:
            return "nothing — no sandbox backend, so a cell's raw open() is bounded by cwd only"
        if not self._kernels:
            return "a backend is mounted; the next kernel to start will be bounded by it"
        if any(kernel.confine is None for kernel in self._kernels.values()):
            return "nothing — these kernels started without a workspace to bound them to"
        return f"{seam.provider.backend} — every live kernel's writes are bounded at the kernel"

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
        result = await kernel.run(request.program, request.bindings, token)
        self._note_denial(kernel, namespace, result)
        return result

    def _note_denial(self, kernel: Kernel, namespace: str, result: CodeRunResult) -> None:
        """Record a boundary the cell hit, the way `ctx.shell` records one.

        **Here rather than in `Kernel`**, which holds a confiner and no seam: this is
        where both are in hand. The seam owns what counts as a refusal
        (`report_denial`), so a cell that writes `/etc/passwd` now leaves the same
        `sandbox/denied` record — and the same `/sandbox allow path` line — that the
        identical refusal from `tool-bash` leaves. Without it the kernel had adopted
        the boundary and not the rule, and the asymmetry was visible inside one
        change: a *network* refusal was already recorded, because the proxy is told
        which agent it is refusing.

        Only for a run that failed, which is `report_denial`'s own requirement: the
        error is where `OSError: [Errno 30] Read-only file system` lands, and a cell
        that printed those words and succeeded was refused nothing. The namespace is
        the agent id, so the record lands in the transcript whoever wrote the cell
        is reading.
        """
        confined = kernel.confined
        if confined is None or result.error is None:
            return
        seam = None if self.sandbox is None else self.sandbox()
        if seam is not None:
            seam.report_denial(confined, (result.error, result.logs), namespace)

    def confiner(
        self, namespace: str, workspace: Any = None
    ) -> Callable[[tuple[str, ...]], ConfinedArgv] | None:
        """How to bound this agent's kernel, or `None` where nothing can.

        **The same condition `ctx.shell` applies, deliberately.** A command the
        harness runs for an agent and a cell the agent writes are the same kind of
        thing — somebody else's code, in that agent's workspace — so they get the
        same boundary from the same `workspace_policy`, and a deployment cannot end
        up with `tool-bash` confined and `run_code` not. It is *not* gated on the
        containment tier for the same reason: `ctx.shell` confines at every rung
        where a backend exists, and `ph doctor`'s containment section already says
        so in the sentence about commands the harness wraps.

        `None` when there is no backend, none that enforces, or no workspace to be
        the writable root — never a passthrough, because the seam refuses rather
        than pretending and a caller must not mistake absence for confinement.

        `workspace` is the one `_acquire` has already resolved; omitted, it is looked
        up. Two lookups for one spawn also logged the failure warning twice.
        """
        seam = None if self.sandbox is None else self.sandbox()
        if workspace is None:
            workspace = self.workspace_for(namespace)
        if seam is None or not seam.available or workspace is None:
            return None
        policy: SandboxPolicy = workspace_policy(workspace)
        # `agent=` so a host the egress proxy refuses is recorded in *this* agent's
        # session, which is the transcript whoever wrote the cell is reading.
        return partial(seam.confine, policy=policy, agent=namespace)

    async def _acquire(self, namespace: str) -> Kernel:
        workspace = self.workspace_for(namespace)
        kernel = Kernel(
            namespace=namespace,
            environment=await self.environment(),
            limits=self.limits,
            journal=self.journal,
            cwd=None if workspace is None else workspace.root,
            env={} if workspace is None else workspace.env,
            confine=self.confiner(namespace, workspace),
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

    `thaw_json` is the whole job: a value that came back through the log is frozen —
    a `MappingProxyType` over tuples — and `json.dumps` will not serialize that. There
    is deliberately **no round trip through JSON here**, because `encode` is about to
    serialize the frame anyway. Anything `json` still cannot represent is handled by
    `encode`'s `default`.
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
    # Asked when a kernel starts, not now: a profile may layer its backend after
    # this row, and a seam read here would be the wrong answer for exactly that
    # profile — `workspace-readonly-scratch` and `permissions-fs` both say so.
    runtime.sandbox = partial(ctx.get, "sandbox")

    contribute(
        ctx, Diagnostic(id="code-runtime", title="Code runtime", read=runtime.describe, order=40)
    )
