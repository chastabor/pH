"""`code-runtime-python`: one CPython child for each agent, governed at the frame (D1).

dsh withheld this provider, on the grounds that *"cross-call state would be
invisible to the log"*. This file does not answer that objection. The answer is
the obligation that the seam checks at registration: `persistence: "namespace"`
requires that the provider emits `kernel/snapshot` events. The state that a cell
leaves behind is therefore in the log with everything else.

Read these three properties before you change anything here.

**A run reads its own frames, and owns the tasks that serve them.** `run()` opens
a task group for the duration of one program. It reads frames inline until
`done`, starts one task for each concurrent binding call, and drains the stdout
and stderr of the child in the same group. **One** task therefore enters and
exits everything. The obvious design, and the first one here, was a long-lived
reader task for each kernel. That design needs a group that is entered when the
kernel starts and exited when it closes, and a *different* task does each.
anyio refuses that. The symptom was a `ClosedResourceError` from a canceled
drain, and the real failure was not reported. `done` is therefore the last frame
of a run, which is the reason why the guest snapshots *before* it settles.

None of this brings back the deadlock that made fd 3 a separate channel (D5). A
cell that awaits a `reply` does not block the read loop of the host, because
that read loop is what delivers the reply.

**Every inbound frame is rebuilt, never trusted.** The child executes
model-written code and holds the descriptor, so it is a hostile peer (C10). See
`codec.decode`.

**A child that dies is replaced, and the model is told.** The namespace is gone
and cannot be recovered. The session is not gone. The next run gets a fresh
kernel, and its result starts with a reset notice. The model does not get a dead
harness.

@module ph_rlm.kernel.manager
"""

from __future__ import annotations

import logging
import math
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
from pydantic import PositiveFloat

from ph.cancel import CancelToken, first_of, is_canceled
from ph.characters import capped_text, head_bytes, stream_decoder, utf8_size
from ph.cordis import Context, Disposer, InactiveScopeError, Releaser, plugin, releasing
from ph.json import as_int, as_str, thaw_json
from ph.keys import AGENTS, CODE_RUNTIME
from ph.orphans import OrphanJournal, host_journal, signal_child
from ph.paths import resolve_roots
from ph.seams.code_runtime import (
    CodeBinding,
    CodeBindingNamespace,
    CodeRunRequest,
    CodeRunResult,
)
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.sandbox import ConfinedArgv, SandboxPolicy, SandboxSeam
from ph.seams.subprocess import first_line, scrub_env
from ph.seams.workspace import Workspace, confined_env, workspace_of, workspace_policy
from ph.text import count_of
from ph.tools.code_mode import CodeRunFailure, ToolCallError
from ph.tools.errors import error_message
from ph.tools.sdk import parameter_names
from ph.wire import WireModel

from ..keys import PYTHON_RUNTIME
from .codec import decode, encode
from .protocol import (
    FD_ENV,
    FRAME_BYTES_ENV,
    PROTOCOL_VERSION,
    BootFrame,
    CallFrame,
    CancelFrame,
    DoneFrame,
    InboundFrame,
    PingFrame,
    ReplyFrame,
    RestoreFrame,
    RunFrame,
    ShutdownFrame,
    truncation_marker,
)
from .venv import InterpreterMode, RuntimeEnvironment, resolve_interpreter

__all__ = ["RESET_NOTICE", "Config", "Kernel", "KernelLimits", "PythonCodeRuntime", "apply"]

log = logging.getLogger("ph_rlm.kernel.manager")

_STREAM_ENDED = (anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)
"""What the pipe of a child raises when it has no more data.

The three exceptions are one outcome reported in three ways: the group canceled
the reader, the child exited, or the stream closed under the reader. None of them
is a failure to report. A reader that let one escape would replace the outcome
that it was called to explain with its own outcome. That is the bug that
`_drain` and `_collect_boot_noise` both guard against. The tuple has a name for
the same reason that `_CHANNEL_GONE` has one."""

RESET_NOTICE = "<runtime_reset>"
"""The prefix of the first result after a kernel died.

The notice names the event and does not describe it. The namespace is gone, and
every variable with it, so a model that reads this and derives again what it
needs behaves correctly. Without the notice, the model would run one cell at a
time again and discover each `NameError`."""

MAX_FRAME_BYTES = 64 * 1024 * 1024
"""The lowest value of `frame_cap`. A cap exists at all because the child can
write whatever it likes onto fd 3 (C10).

**The guest reads with the same derived number** (F8). The guest opens its
connection before it has read a `boot` frame. `start` therefore passes
`frame_cap` to it through `FRAME_BYTES_ENV`, in the same way that the descriptor
itself crosses. `ph_runtime.channel` keeps its own 64 MiB only for a host that
names no limit."""


class KernelLimits(WireModel):
    """What one child may consume. The host sends every value, because the guest
    has no defaults.

    This is a `WireModel` and not a dataclass, so that `Config` can extend it. A
    limit that is added here reaches the row config and the `boot` frame, and is
    not typed again in either. Earlier there were three declarations and two
    hand-written copies, which is four edits before the protocol is even touched.
    """

    cpu_seconds: int = 30
    idle_cpu_seconds: int = 2
    """The CPU seconds that a namespace may use while **no cell is running** (M1).

    This is the time that nothing else limits. `cpu_seconds` is armed at the
    start of a run, and the first `SIGXCPU` disarms it. It is disarmed and not
    lowered, because a second delivery lands in the teardown of the guest and
    costs the `done` frame. What remains is a `to_thread` worker or a detached
    task that spins in pure Python. A cancel reaches neither. The run reports
    `cpu`, the thread continues to use CPU, and with the budget off no second
    signal ever comes.

    The value is small, because nothing legitimate uses it. A background thread
    that a cell left on purpose is *waiting*, and waiting costs no CPU. Only a
    runaway reaches this limit, and the guest ends its own process when it does.
    The namespace is already lost, and the alternative is a core that is used
    until the session ends."""
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


def frame_cap(limits: KernelLimits) -> int:
    """The largest inbound frame that *this* kernel will assemble (M3).

    **The value is derived, because the design depends on the ratio.** The fix
    for F3 made the guest send one frame for each changed variable, so a frame
    holds one `maxSnapshotBytes` payload. The constant above used to assert, with
    nothing to support it, that 64 MiB was enough for that. It is enough for the
    shipped 16 MiB. It stops being enough when a deployment raises the limit. The
    cost is then not a rejected frame but a lost namespace: the host refuses the
    frame and closes the channel, and the model is told that the runtime exited.
    The old docstring claimed the sizing, but nothing enforced it. The only check
    was a test that monkeypatched the constant.

    The result is twice the payload plus one megabyte. base64 uses four bytes for
    each three, and the JSON envelope and the name of the variable take at most a
    few hundred. The slack costs nothing, because this is a ceiling on junk and
    not an allocation.
    """
    return max(MAX_FRAME_BYTES, limits.max_snapshot_bytes * 2 + 1024 * 1024)


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
    dropped: int = 0
    """The total bytes of output that the caps of this run discarded: what the
    print streams of the guest report on `done`, plus what the pipe drains here
    skipped."""
    failure: CodeRunFailure | None = None
    """A refusal or a budget. `run()` raises it after the program unwinds (C3)."""
    probe_id: int = 0
    """The id of the last probe that was sent. It is compared with the `id` of
    the answer.

    A pong with a different id is unsolicited. It is a late duplicate, or a
    forged frame from fd 3, which this module assumes is possible everywhere
    else (C10). Such a pong clears nothing and times nothing."""
    probed_at: float = 0.0
    """The time when the last probe was sent, whether it was answered or not.
    Before the first probe, it is the time when the run opened, which `run` sets.
    The first probe is therefore due `probe_seconds` into the run, and a cell
    that is shorter than that is never probed. The default is a plain value and
    not a factory that reads the clock, so that a run can be built with no loop
    running, as a test of `settle` does.

    **This value never changes while a probe is outstanding.** That makes it a
    measurement and not only a schedule: the round trip that is finally recorded
    is the whole time that the loop took to return to its reader, including the
    stall. The value also sets the schedule. A probe is due `probe_seconds`
    after the last one was *sent*, not after the last one was answered. If only
    the answer decided, the next probe would be sent at the moment the last
    answer arrived."""
    probe: Literal["idle", "waiting", "stalled"] = "idle"
    """The state of the outstanding probe. `stalled` is `waiting` past its due
    time. The state is held, so that one starved period is counted one time."""
    aborting_since: float | None = None
    """The time when this run was asked to stop, by either route: the caller
    canceled, or a dispatch was refused. The field is here and not in `_pump`,
    because `_serve_call` starts the abort from its own task, and the escalation
    clock that `_watch` runs must be the same clock."""
    moved: anyio.Event | None = None
    """Set when a deadline that `_watch` sleeps toward may have changed: an abort
    that another task started (`_serve_call`), or a probe that was answered after
    it stalled. `_watch` makes a new event each time it plans. The value is
    `None` before the first plan."""

    def _move(self) -> None:
        if self.moved is not None:
            self.moved.set()

    def begin_abort(self) -> bool:
        """Start the escalation clock. **The first caller wins.** Returns whether
        this call was the first.

        This is separate from *asking* the guest to stop, and the separation is
        the purpose. The ask is a socket write, and a write blocks when the guest
        has stopped reading. That is exactly the run that the escalation exists
        for. Both callers used to start the clock through a helper that sent the
        frame first. The clock for a blocked channel therefore started only
        after the blocked channel had accepted a frame. `_watch` waited inside
        its own `_send` and never returned to the loop that times the grace, so
        the abort ladder stopped one step before the kill. The clock now starts
        here, in a method that cannot block, and the ask follows it.
        """
        if self.aborting_since is not None:
            return False
        self.aborting_since = anyio.current_time()
        # The grace deadline exists now, and `_watch` may be asleep until a probe
        # is due.
        self._move()
        return True

    def answered(self, probe: int, at: float) -> float | None:
        """Record one `pong`. Returns the round trip, or `None` if the pong does
        not answer the outstanding probe.

        There is one reading, because the repair arrives as a `done` frame and
        not in the answer. `PingFrame.run` gives the reason. What remains is the
        loop gauge, which `phern doctor` prints and which nothing acts on.

        The id is compared and not trusted. A frame from fd 3 can say anything,
        and an unsolicited pong would otherwise record a round trip against
        whichever probe was outstanding.
        """
        if self.probe != "waiting" and self.probe != "stalled":
            return None
        if probe != self.probe_id:
            return None
        if self.probe == "stalled":
            # A stalled probe has no deadline, so `_watch` sleeps until this
            # answer. The next probe has been due for some time.
            self._move()
        self.probe = "idle"
        return at - self.probed_at

    def settle(self, *, error: str | None = None, value: object = None) -> bool:
        """Record how this run ended. **The first writer wins.** Returns whether
        this call was the first.

        Three places end a run: a `done` frame, the close of the channel, and the
        kill that follows an abort that got no response. They race by
        construction. `_watch` can kill while `_pump` is blocked in the middle of
        a frame, and the socket close in `_teardown` is itself what wakes
        `_pump`. Each place used to assign `error` and `settled` directly, and
        only `_on_closed` checked first. The outcome therefore depended on three
        call sites that had to run in the correct order relative to each other.
        The design depended on that order, and nothing showed it. A `done` frame
        that was already buffered when the kill landed would replace "the
        runtime was killed" with the value of the program, and nothing reported
        that.

        The method is idempotent here, so the call sites need no order. The
        account that the caller sees is decided by which stop happened first,
        and not by which task reached the assignment last.
        """
        if self.settled:
            return False
        self.error = error
        self.value = value
        self.settled = True
        return True


class SnapshotPolicy(Protocol):
    """What the runtime needs from the component that keeps the namespace in the
    log (D17).

    The two responsibilities are separate on purpose. The provider knows about
    processes and frames, and the policy knows about events and blobs. Neither
    needs to know both, and `ph_rlm.snapshot` can fold a stored log with no
    runtime running.
    """

    async def record(self, namespace: str, run_id: int, variables: list[dict[str, Any]]) -> None:
        """Persist the changed variables of one settled run."""
        ...

    async def materialize(self, namespace: str) -> tuple[list[dict[str, Any]], list[str]]:
        """The payloads to give back to a kernel that just started, and the names
        whose payload could not be read. A restore reports those names as
        failed."""
        ...

    async def restored(self, namespace: str, outcome: dict[str, Any]) -> None:
        """Record which variables came back and which did not."""
        ...


class KernelDied(RuntimeError):
    """The child is gone, and the namespace with it. The session is not gone."""


@dataclass(slots=True)
class Kernel:
    """One child process, one persistent namespace."""

    namespace: str
    environment: RuntimeEnvironment
    limits: KernelLimits
    journal: OrphanJournal | None
    cwd: Path | None = None
    """The working directory of the child: `workspace.root` for this agent (D21).

    **Because of this field, the `worktree` tier limits authored code and does
    not only observe it.** By construction, the `open("notes.txt", "w")` of a
    cell reaches no policy waterfall (N1). But it does resolve against this
    directory, so a relative write lands in the checkout of the agent itself. An
    absolute path escapes *this* mechanism, and only confinement refuses it
    (§4.8, E13). Confinement is `confine` below, when a backend exists to do
    it."""
    confine: Callable[[tuple[str, ...]], ConfinedArgv] | None = None
    """How to confine this child at the kernel, or `None` where nothing can.

    **This field gives `sandbox` a meaning for authored code.** `cwd` above
    limits a *relative* write, because it puts the child in the tree of the
    agent. Only this field refuses `open("/etc/passwd", "w")` from inside a cell,
    and §4.8 names that as the one thing that no tier below `sandbox` can do.
    Before this field existed, `permissions-fs` told operators that "a sandbox
    provider bounds what a code cell can reach directly" while the kernel was
    spawned unconfined. The sentence is true now.

    The runtime supplies the value, and it is not resolved here. It is resolved
    for each *kernel*, because the writable set is the workspace and the scratch
    directory of the agent itself. The value is `None` when no backend is mounted
    or the agent has no workspace. `ctx.shell` declines on the same condition,
    for the same reason. The seam refuses and does not pass a command through,
    so a request for confinement that cannot be given would turn every cell into
    a `SANDBOX_UNAVAILABLE` denial."""
    applied_limits: dict[str, Any] = field(default_factory=dict)
    """What the guest reported that it could actually apply, from `boot-ack`.

    **This is stored because `phern doctor` reports it, and it is not always the
    request.** `KernelLimits` is what the host *asked* for. macOS refuses
    `RLIMIT_AS` outright, so a limit that the request names can be not in force.
    Printing the request there would claim a limit that nothing enforces. That is
    the failure of E1, one layer down from the tier table.
    `PythonCodeRuntime.describe` therefore reads this field, through `_limits`,
    and says "not applied" where the guest said `None`. The field is empty until
    `boot-ack`."""
    confined: ConfinedArgv | None = None
    """What `confine` produced, after this kernel has started.

    This is one field, and not one fact copied out of it for each reader.
    `_interrupt` needs `forwards_signals`. That is *not* the same question as
    "was I confined at all": `bwrap` puts a wrapper and a PID namespace between
    the host and the guest, while `sandbox-exec` execs its target. The report of
    a refusal needs the effective `policy`, to know whether the network was
    involved at all. If either were derived from "confinement happened", macOS
    would lose a working cancel route, and an outage would be read as a
    denial."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for the child: `workspace.env`, or `confined_env` for a
    confined kernel.

    This is the build-tool redirection (E12). `TMPDIR`, `PYTEST_ADDOPTS` and
    similar variables point inside the scratch directory of the agent. A cell
    that runs the tests of the project then does not leave `.pytest_cache/` in
    the worktree. That directory would change the disposal policy from "remove a
    clean tree" to "keep everything"."""
    snapshots: SnapshotPolicy | None = None
    skills: tuple[str, ...] = ()
    """The import names that are bound as callables at boot (P3-18). They come
    from `rlm-skills-python`."""
    boot_timeout: float = 30.0
    shutdown_grace: float = 5.0
    cancel_grace: float = 2.0
    """How long a canceled cell has to unwind before the child is killed.

    **What the abort ladder can and cannot do** is stated here, because this
    field is the only setting of the ladder. The abort ladder is the `cancel`
    frame, then `SIGINT`, then `SIGKILL` when this time expires. It stops a cell
    at its next yield point, or kills it if it never reaches one. It cannot
    preempt straight-line synchronous Python. Statements after a caught refusal
    run before the loop of the guest gets control again, so the limit of C3 is
    "about one cell" and not exactly one. What remains is narrower than the
    raw-`pathlib` non-goal (§11, Q10) and different from it. This one is about
    *time*, and a deployment widens it when it raises this number."""

    probe_seconds: float | None = 1.0
    """How often a run is probed while it is open (M2), or `None` for never.

    A probe is one frame each way, answered from the reader task of the guest. It
    is cheap enough to run continuously, and that is the purpose. A gauge that is
    read only when something already looks wrong tells nothing about what normal
    was. `Config.probe_seconds` says what turning it off costs."""
    _process: anyio.abc.Process | None = None
    _sock: socket.socket | None = None
    loop_worst: float | None = None
    """The worst probe round trip that this kernel has seen, or `None` before the
    first (M2). The worst value is the number to read. A loop that is usually
    prompt and sometimes seconds behind is a cell that uses CPU in bursts, and
    that is what a person who looks at a slow agent wants to see.

    This is a plain field, like `applied_limits` above.
    `PythonCodeRuntime._loops` reads it from outside the class in either case, so
    a property enforced nothing and put the same sentence in two places."""
    loop_stalls: int = 0
    """The number of probes that went a whole `probe_seconds` with no answer. One
    is counted for each starved period, not one for each tick."""
    _run_seq: int = 0
    _buffer: bytearray = field(default_factory=bytearray)
    _scanned: int = 0
    """How much of `_buffer` was already searched for a frame boundary. A large
    frame is then scanned one time, and not one time for each chunk."""
    _boot_noise: bytearray = field(default_factory=bytearray)
    """Whatever the child wrote to fd 1 and fd 2 before it sent `boot-ack` (F7).
    It is read while boot waits, so that a guest cannot block on a full pipe.
    `_boot_said` quotes it when boot fails, which is the one moment when it is
    useful."""
    _alive: bool = False
    _closed: bool = False
    """Closed permanently (`aclose`), as opposed to stopped for a restart
    (`_teardown`). A run that already held this kernel when its namespace was
    closed is refused. It does not start a guest that no table holds and no close
    reaches (K1)."""
    _reset_notice: bool = False
    _lock: anyio.Lock = field(default_factory=anyio.Lock)
    _send_lock: anyio.Lock = field(default_factory=anyio.Lock)
    """Serializes writes. `wait_writable` refuses two waiters on one socket, and
    several tasks write during one run: `run` sends the `run` frame, tasks that
    `_watch` starts send `cancel` and `ping`, and one `_serve_call` task for each
    concurrent binding call answers with `reply`."""

    # ---------------------------------------------------------------- start --

    async def start(self, namespaces: list[dict[str, Any]]) -> None:
        """Spawn the child, give it fd 3, and wait for it to report ready.

        This is also the *restart* path, so the read buffer is cleared here. A
        half-frame that was left over would otherwise put the bytes of one guest
        process before the `boot-ack` of the next.
        """
        await self._teardown()
        self._buffer.clear()
        self._scanned = 0
        host_end, child_end = socket.socketpair()
        argv = (str(self.environment.python), "-m", "ph_runtime")
        child_fd = child_end.fileno()
        # Use `scrub_env`, not `os.environ`. This is the child that the docstring
        # of the seam describes: "a child runs code the model wrote, so it does
        # not inherit `*KEY*`". A cell that can read `os.environ` can print the
        # provider credential into its own output, and the output is then logged.
        environ = scrub_env(
            extra={
                **self.env,
                "NO_COLOR": "1",
                "PYTHONUNBUFFERED": "1",
                FD_ENV: str(child_fd),
                # One expression gives the limit to both readers. A variable at
                # the snapshot limit must also fit one frame going *in* (F8).
                FRAME_BYTES_ENV: str(frame_cap(self.limits)),
            }
        )
        # **fd 3 crosses the boundary.** That property makes it possible to
        # confine this child at all. `bwrap` passes an inherited descriptor
        # through to the command that it execs, and so does the
        # `sh -c … exec "$@"` that the egress shim wraps around it. Both were
        # measured, with the socket live at the far end. See
        # `test_kernel_confinement.py`.
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
                # **A session of its own, whatever confines it** (J9). This is
                # the one pH child that does not go through `ph.seams.subprocess`,
                # which sets the same flag for everything that it launches. It
                # is also the child that most needs the flag. It executes
                # model-authored cells, so a cell could `open("/dev/tty")` and
                # reach the terminal of the user directly, or push characters
                # back into it with `TIOCSTI`. Both require the *controlling*
                # terminal, which `setsid()` takes away.
                #
                # The flag is set here, and not only as the `--new-session` of
                # bwrap, because that argv element closes the hole on exactly
                # one of three configurations. Seatbelt has no equivalent, and a
                # deployment with no sandbox backend got nothing at all. The
                # bwrap flag stays as a second protection for a `ConfinedArgv`
                # that other code spawns.
                start_new_session=True,
            )
        finally:
            # Close the descriptor in the parent in both cases. If it stayed
            # open, the parent would never see the EOF of the child, and a failed
            # spawn would leak a descriptor.
            child_end.close()

        host_end.setblocking(False)
        self._sock = host_end
        self._alive = True
        pid = self._process.pid
        if pid is not None and self.journal is not None:
            # Record what is actually running, including the wrapper. The journal
            # verifies that an orphan is ours before it kills the orphan, and the
            # pid that it holds is the pid of the wrapper.
            self.journal.record(pid=pid, argv=spawn, label=self.namespace)
        try:
            await self._boot(namespaces)
        except BaseException:
            # **A boot that does not finish stops its guest** (K2), however the
            # boot ends. `_alive` is set before `boot-ack`. A stop between the two
            # used to leave a guest that the next run treated as ready: no start,
            # no restore, and its cell sent to a guest that was still booting.
            # `_teardown` shields itself.
            await self._teardown(ask=False)
            raise

    async def _boot(self, namespaces: list[dict[str, Any]]) -> None:
        """Send the boot frame to a spawned guest and wait for it to report ready."""
        assert self._process is not None
        await self._send(
            self.limits.to_boot(
                namespaces=namespaces, namespace_id=self.namespace, skills=self.skills
            )
        )
        self._boot_noise.clear()
        try:
            with anyio.fail_after(self.boot_timeout):
                # **The pipes are read while the host waits for `boot-ack`**
                # (F7). `_drain` runs only for the duration of a *run*, so
                # during boot nothing consumed fd 1 or fd 2. A guest that writes
                # more than the pipe buffer before `boot-ack` blocks on the
                # write and never sends it. The host then reports "did not
                # report ready within Ns" for a child that was ready and
                # blocked, and quotes whatever fits in one late read.
                #
                # The group is entered and exited inside this function, which is
                # the rule that the module docstring states. The long-lived
                # reader task is what anyio refuses, and what reported a
                # canceled drain when the real failure was something else.
                async with anyio.create_task_group() as booting:
                    booting.start_soon(self._collect_boot_noise, self._process.stdout)
                    booting.start_soon(self._collect_boot_noise, self._process.stderr)
                    fault = await self._await_boot_ack()
                    booting.cancel_scope.cancel()
        except TimeoutError as timeout:
            # The output is quoted here too. This branch catches a wrapper that
            # *hangs* and does not exit: the egress shim that waits on its
            # readiness poll, or a bind that never completes. The output is read
            # here, before `start` stops the process that it comes from.
            said = first_line(self._boot_said())
            raise KernelDied(
                f"the runtime did not report ready within {self.boot_timeout}s "
                f"({self.environment.describe()})" + (f": {said}" if said else "")
            ) from timeout
        if fault is not None:
            raise KernelDied(fault)

    async def _collect_boot_noise(self, stream: Any) -> None:  # noqa: ANN401
        """Read one pipe into `_boot_noise` until boot settles or the pipe ends.

        The read is capped like the read of `_drain`, and for the same reason. A
        wrapper can write to this stream without limit, and so can the
        environment through a `sitecustomize`. The host holds the text in memory
        to quote it. Every ending is ordinary here (the group cancels this task,
        or the child exits), so nothing escapes to replace the failure that is
        being reported.
        """
        if stream is None:
            return
        cap = self.limits.max_log_bytes
        with suppress(*_STREAM_ENDED):
            async for chunk in stream:
                if len(self._boot_noise) < cap:
                    self._boot_noise.extend(chunk[: cap - len(self._boot_noise)])

    def _boot_said(self) -> str:
        """Whatever the child wrote to its pipes before boot settled.

        **Both pipes, not only stderr.** `_collect_boot_noise` is started on both
        fd 1 and fd 2, so this is the two streams interleaved in arrival order.
        That is the correct answer for the callers, which quote a child that
        failed to start. A wrapper that explains itself on stdout names the same
        failure, and to drop that text would drop the sentence that gives the
        reason.

        The implementation only reads the buffer. A second read of the pipe would
        be wrong, not only redundant. Both callers run during boot, while
        `_collect_boot_noise` iterates those streams, and a concurrent `receive`
        on one of them raises `BusyResourceError`. The collector already applied
        the cap.

        An empty result is an ordinary answer: the child died and wrote nothing.
        The callers then leave the quote out and do not invent one.
        """
        return capped_text(self._boot_noise, self.limits.max_log_bytes)

    async def _rehydrate(self) -> None:
        """Give a kernel that just started the namespace that the log holds (D17).

        This does not happen on a restart after a crash. Those payloads describe
        a namespace that the model already knows is gone. To rebuild half of it
        silently would be worse than the empty namespace that the reset notice
        announces.
        """
        if self.snapshots is None or self._reset_notice:
            return
        variables, unreadable = await self.snapshots.materialize(self.namespace)
        if not variables and not unreadable:
            return
        outcome = await self._restore(variables)
        # The outcome is recorded even when nothing came back (S11). A namespace
        # whose every blob was unreadable is exactly the case that the model most
        # needs to be told about.
        failed = [*outcome.get("failed", []), *unreadable]
        await self.snapshots.restored(self.namespace, {**outcome, "failed": failed})

    async def _await_boot_ack(self) -> str | None:
        """Read the first frame of the child. Returns a fault message, or `None`.

        **Until `boot-ack`, the only frames that exist are `boot-ack` and
        `fault`, and anything else is a fault to report.** The guest sends
        `boot-ack` before `Runner.serve` and before any cell, so nothing
        model-written has written to fd 3 yet. (The guest does import its skills
        before `boot-ack`: `Runner.__init__` calls `install_skills`.) The
        tolerance of C10 (junk is skipped, because the peer is hostile) is
        therefore not needed here. It is actively wrong here. For every frame
        that this loop declines to understand, it would wait `boot_timeout` in
        silence with nothing to quote, and this function was rewritten to end
        that failure. The rule for this phase is therefore stated one time, for
        both ways in which a guest can break it: a line that does not decode,
        and a decodable frame that is not one of the two. It does not cover only
        the first.

        The die-with-parent mechanism that the guest applied is logged and not
        stored. The mechanisms give different guarantees, and a log that names
        one says what was in force. (Before P12-07, a macOS guest ran under
        `getppid-poll`, with a window of one second in which a host that was
        killed hard could leave a stray process. `kqueue-exit` closed that
        window.) What the guest *could not apply* is stored, because
        `describe()` reports it. See `applied_limits`.
        """
        while True:
            line = await self._recv_line()
            if line is None:
                # **The words of the child itself**, which this code used to
                # discard. A guest that cannot start gives the reason on stderr:
                # a missing module, an interpreter that does not run, or `bwrap`
                # refusing a bind source. Without that text, every one of those
                # reads as the same sentence. `probe_sandbox` makes the same
                # argument about quoting a backend: "the runtime did not start"
                # is true and useless.
                said = first_line(self._boot_said())
                because = f": {said}" if said else ""
                return (
                    "the runtime exited before reporting ready; "
                    f"{self.environment.describe()} could not start ph_runtime{because}"
                )
            frame = decode(line)
            if frame is None:
                # This is not the "junk → skip" of C10. Nothing model-written has
                # run yet, so a line that this host cannot read is the guest
                # sending a shape that the host does not accept (D7). To skip it
                # means to wait for `boot_timeout` in silence with nothing to
                # quote. This was found on macOS. The guest reported
                # `addressSpaceBytes` as RLIM_INFINITY, the codec of that time
                # refused the whole frame because of it, and every start ended
                # 60 s later as "did not report ready". The codec is fixed. This
                # branch would have reported the cause in a minute, where
                # nothing ever did.
                return (
                    "the runtime's first frame could not be read as protocol "
                    f"{PROTOCOL_VERSION} ({self.environment.describe()}): "
                    f"{head_bytes(line[:800].decode('utf-8', 'replace'), 200)!r}"
                )
            if frame["type"] == "boot-ack":
                if frame["protocol"] != PROTOCOL_VERSION:
                    return f"the runtime speaks protocol {frame['protocol']}"
                log.info(
                    "ph_rlm.kernel: %s ready on python %s, limits %s",
                    self.namespace,
                    frame["python"],
                    frame["limits"],
                )
                # No `isinstance` is needed. The codec coerced `limits` to its
                # declared `"obj"` or refused the frame, and the type now says
                # so.
                self.applied_limits = frame["limits"]
                return None
            if frame["type"] == "fault":
                return as_str(frame["message"])
            # The frame is readable, but it is not one of the two frames that
            # exist before the guest is ready. When this code looped here, a
            # `log` or `done` that arrived early became the same silent
            # `boot_timeout` as a line that does not decode.
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
        """Run one program until it settles. Runs are serialized: one namespace,
        one program at a time."""
        async with self._lock:
            if self._closed:
                raise KernelDied(f"the runtime for {self.namespace} was closed; its agent is gone")
            restarted = False
            if not self._alive:
                await self.start([_declare(namespace) for namespace in namespaces])
                await self._rehydrate()
                # The notice goes to the run that *gets* the fresh namespace, not
                # to the run that lost the old one. The cell that called
                # `os._exit` knows what it did. The next cell is the one that
                # meets an empty namespace and does not know why.
                restarted, self._reset_notice = self._reset_notice, False
            self._run_seq += 1
            active = _ActiveRun(
                run_id=self._run_seq,
                bindings={
                    (namespace.name, binding.name): binding
                    for namespace in namespaces
                    for binding in namespace.bindings
                },
                probed_at=anyio.current_time(),
            )
            try:
                async with anyio.create_task_group() as tasks:
                    process = self._process
                    tasks.start_soon(self._drain, process.stdout if process else None, active)
                    tasks.start_soon(self._drain, process.stderr if process else None, active)
                    # The clock of the abort ladder runs as a sibling task, and
                    # does not interrupt the read. `_pump` gives the reason.
                    tasks.start_soon(self._watch, active, token, tasks)
                    await self._send(RunFrame(id=active.run_id, program=program))
                    await self._pump(active, tasks)
                    # The drains and any in-flight call tasks belong to this run.
                    tasks.cancel_scope.cancel()
            except BaseException:
                # **A run that does not settle stops its guest** (K2). A stop
                # through `token` goes through the abort ladder and settles. A
                # stop from outside (the task of the tool call was canceled)
                # used to skip the ladder. The guest then continued the cell and
                # kept the next run waiting behind it. The next run now starts a
                # guest and restores what the log holds of the namespace.
                await self._teardown(ask=False)
                raise

            if active.failure is not None:
                # C3: the refusal ends the tool call, not only the program. It
                # reaches the model as the reason why the cell produced nothing.
                raise active.failure
            logs = "".join(active.logs)
            if restarted:
                logs = f"{RESET_NOTICE} the runtime restarted; the namespace is empty.\n{logs}"
            if active.dropped:
                # The total that the run dropped is reported one time, now that
                # every stream has ended. The markers in the logs could say only
                # where each stream was cut. The sentence is the one that
                # `tool-bash` uses (D4), so every renderer of the logs passes it
                # on.
                logs += truncation_marker(active.dropped, self.limits.max_log_bytes)
            return CodeRunResult(
                logs=logs,
                value=active.value,
                error=active.error,
                truncated=active.truncated,
                dropped=active.dropped,
                reset=restarted,
                displays=tuple(active.displays),
            )

    async def _pump(self, active: _ActiveRun, tasks: anyio.abc.TaskGroup) -> None:
        """Read the frames of this run until it settles. Nothing interrupts the
        read.

        **The scope that this loop used to run in cost time for each frame, not
        for each read.** The body was wrapped in a 50 ms `move_on_after`, so that
        the loop could ask two questions between reads: has the caller canceled,
        and has the abort grace expired. Neither question is about the socket.
        But `_recv_line` returns directly from its buffer whenever a frame is
        already there, and a 64 KiB read from a cell that prints a lot carries
        many frames. The scope was entered and exited for every `log` frame, not
        for every syscall. This was measured on a cell that emitted 3,000 of
        them: **37.5 ms against 13.8 ms**. That is 2.7 times, or 7.9 µs for each
        frame, for a deadline that was almost never reached.

        The clock therefore moved to `_watch`, a sibling task that sleeps and
        does not hold a socket. **What wakes this loop now** is the thing that it
        waits for: a frame that settles the run, or the close of the channel.
        `_teardown` calls `notify_closing` (in `_close_channel`) before it
        closes the socket, so a kill from `_watch` arrives here as `_recv_line`
        returning `None` on the next turn.

        **About issue 58, which this change was first written to fix: it does
        not fix it, and the attribution was wrong.** The claim was that canceling
        the wait leaves the fd registered. The `_wait_until_readable` of anyio
        registers `f.set_result` as the reader callback, and removes the reader
        in a done-callback one loop iteration later. A queued readiness callback
        can therefore call `set_result(None)` on a canceled future and raise
        `InvalidStateError` from a bare loop callback. That mechanism is real,
        and the captured `<Handle Future.set_result(None)>` is its signature.
        But it cannot be reached from here. `_recv_line` calls the *free*
        `anyio.wait_readable`, which is `AsyncIOBackend.wait_readable`. That is
        a different implementation: it wraps `fut.set_result` in
        `except InvalidStateError: pass` and removes the reader synchronously
        inside the callback. Canceling it is safe. The defective code belongs to
        `_RawSocketMixin`, that is, to `UNIXSocketStream`,
        `UNIXSocketListener.accept` and `connect_unix`. That is the **daemon**
        socket, where the intermittent failure was observed both times. Issue 58
        is still open, and its scope is now that socket. This change is
        justified by the measurement above.
        """
        while not active.settled:
            line = await self._recv_line()
            if line is None:
                self._on_closed(active)
                # **The guest is stopped, not only disowned.** A channel that
                # ended because the guest exited leaves a child to reap. A
                # channel that ended because the host refused an oversized frame
                # leaves a child that is *still running*. That child still holds
                # the memory of the namespace, and nothing can reach it:
                # `_alive` is false, so the next run spawns a second process
                # beside it, and the first leaks until the session ends. In both
                # cases this channel is finished, and the process at the other
                # end has no further use.
                await self._teardown()
                return
            frame = decode(line)
            if frame is not None:
                # A forged or garbled frame is dropped. Raising here would let
                # the child crash the host on demand (C10).
                await self._handle(frame, active, tasks)

    async def _watch(
        self, active: _ActiveRun, token: CancelToken | None, tasks: anyio.abc.TaskGroup
    ) -> None:
        """The clock of the abort ladder, outside the read path (C3).

        This task asks the two questions that `_pump` used to interrupt itself to
        ask, and is otherwise asleep. **It sleeps until something is due**
        (P12-03), and does not wake on a 50 ms tick. It plans the next moment
        when it has a question (`_next_deadline`). It then sleeps until that
        moment, until the caller cancels (`token.wait()`), or until
        `active.moved` says that a deadline changed. A run that is never
        canceled wakes this task one time for each `probe_seconds`, for the
        probe, and never when the probe is off.

        Concurrency with `_pump` is not new. `_serve_call` has always started an
        abort from its own task, which is the reason why `aborting_since` is on
        `_ActiveRun` and not in `_pump`. This task is a second caller of the
        same kind, and `begin_abort` sets `moved` so that the grace deadline is
        planned at once.

        The loop is `while True` and not `while not active.settled`. The loop
        starts before the first frame can arrive, so the entry test could never
        be false. The check that matters is the one *after* the wait: a run that
        settled while this task slept must not get a false interrupt and must
        not be killed. If the check were written as an entry condition, it would
        read as the termination test, and a reader could delete the real check
        as redundant.
        """
        while True:
            active.moved = moved = anyio.Event()
            with anyio.CancelScope(deadline=self._next_deadline(active)):
                if token is not None and active.aborting_since is None:
                    await first_of(moved, token)
                else:
                    await moved.wait()
            if active.settled:
                return
            if active.aborting_since is None:
                # **The clock that the host never had** (M2). Both steps below
                # start only because somebody pressed stop. A run that is never
                # canceled and never settles therefore had nothing watching it:
                # the host waits for `done` with no wall clock of its own.
                #
                # What this asks is narrow. It is stated exactly here, because
                # this text once claimed more. The probe catches a guest that
                # reached `_send_done`, cleared what it owed, and left the host
                # with no terminal frame. The guest-side failures that the guard
                # of `_snapshot` and `relax_cpu_budget` cover all leave the run
                # still *owed*. `_answer_if_owed` answers those, and this probe
                # never sees them. Those guards are still necessary.
                #
                # No step follows the probe, because the guest settles the run
                # itself.
                self._probe(active, tasks)
                if is_canceled(token):
                    # **The clock starts first, and the ask is started, not
                    # awaited.** `_interrupt` writes a frame, and a write waits
                    # for `_send_lock`. A reply that the guest does not read can
                    # hold that lock indefinitely. If this code awaited the ask,
                    # the clock would wait inside the thing that it times. The
                    # grace below then never expired and the kill never came.
                    # When the ask is only started, it can take any amount of
                    # time.
                    active.begin_abort()
                    tasks.start_soon(self._interrupt, active.run_id)
            elif anyio.current_time() - active.aborting_since >= self.cancel_grace:
                # Neither the frame nor the signal reached the cell. That means
                # the cell spins in Python and the loop of the guest is starved.
                # A kill costs the namespace. No kill costs the session.
                await self._kill_unresponsive(active)
                return

    def _next_deadline(self, active: _ActiveRun) -> float:
        """The time when `_watch` next has a question to ask, on the clock of the
        loop, or `inf`.

        **The clock is monotonic on purpose**, unlike the deadlines of the
        daemon (`ph.wall_clock`). These are intervals inside one running cell,
        and a cell that was suspended with the machine was suspended too.

        During an abort, only the grace matters. Otherwise the probe decides. A
        probe is due `probe_seconds` after the last one was sent (or after the
        run opened). While one is outstanding, it is due to be called stalled
        after the same interval. A stalled probe has no deadline, and neither
        does a run with the probe off. The answer to a stalled probe arrives
        through `active.moved`. `_probe` asks this function too, so there is one
        rule.
        """
        if active.aborting_since is not None:
            return active.aborting_since + self.cancel_grace
        if self.probe_seconds is None or active.probe == "stalled":
            return math.inf
        return active.probed_at + self.probe_seconds

    def _probe(self, active: _ActiveRun, tasks: anyio.abc.TaskGroup) -> None:
        """Keep the loop gauge supplied, and give the guest a chance to settle
        (M2).

        **The send is started and never awaited**, for the same reason as the
        cancel ask that follows it in `_watch`. `_send` waits for `_send_lock`,
        which a reply that the guest does not read can hold indefinitely. A
        clock that waits inside the thing that it times is not a clock. One
        probe is outstanding at a time, so a starved loop costs one unanswered
        frame and not one for each interval.

        `_watch` also calls this when it woke for something else (a cancel, or a
        deadline that changed). This method therefore acts only when
        `_next_deadline` says that the probe is due: an idle probe is sent, and a
        waiting probe is called stalled. With the probe off, or with one
        stalled, that moment never comes.

        **No verdict is made here.** The frame names the run that the host waits
        for. A guest that is not running that run and does not owe it a `done`
        answers by sending one. The repair therefore arrives as an ordinary
        terminal frame through `_pump`, and no step needs to be added. What
        remains on this side is the measurement: how late the answer was, and
        whether it came.
        """
        now = anyio.current_time()
        if now < self._next_deadline(active):
            return
        if active.probe == "idle":
            active.probe_id += 1
            active.probed_at = now
            active.probe = "waiting"
            tasks.start_soon(self._ping, active.run_id, active.probe_id)
        elif active.probe == "waiting":
            # Counted one time for each starved period. The probe keeps its own
            # timestamp, because the value to measure is how long the loop took
            # to return. To reset the clock here would discard that value.
            active.probe = "stalled"
            self.loop_stalls += 1

    async def _ping(self, run_id: int, probe: int) -> None:
        await self._send(PingFrame(id=probe, run=run_id))

    async def _interrupt(self, run_id: int) -> None:
        """Ask twice, by two routes that fail in different ways (D5).

        The reader task of the guest reads the frame, and `SIGINT` arrives as a
        loop callback. Both are cooperative, and both need the loop of the guest
        to be running. A cell that spins in Python answers neither. The grace
        period and `_kill_unresponsive` exist for that case.

        **The signal route is not used when the signal would not reach the
        guest. This is a refusal to make cancellation destructive.** Under
        `bwrap`, `self._process` is the wrapper, and the wrapper does not
        forward signals. This was measured: the wrapper itself dies of `SIGINT`
        (`rc=-2`), and takes with it the PID namespace and the whole namespace
        of the agent. To send the signal would turn every "stop this cell" into
        "lose everything that the cell had". That is the outcome of the last
        step of the abort ladder, and the ladder exists to avoid that outcome,
        not to reach it first.

        The question is put to `ConfinedArgv.forwards_signals`, and is not "am I
        confined". `sandbox-exec` execs its target and signals land, so a
        backend that does forward keeps both routes.

        Less is lost than it seems. The guest installs `SIGINT` through
        `loop.add_signal_handler`, so the signal arrives as an ordinary loop
        callback, on the same loop that the reader of the `cancel` frame runs
        on. The two routes therefore cover the *same* situation. The separate
        value of the signal is as a backup for a blocked channel with a live
        loop, and `_kill_unresponsive` still covers that case after the grace
        period.
        """
        await self._send(CancelFrame(id=run_id))
        if self.confined is not None and not self.confined.forwards_signals:
            return
        process = self._process
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError, OSError):
                process.send_signal(signal.SIGINT)

    def _kill_group(self, process: anyio.abc.Process) -> None:
        """Kill the guest **and whatever it started** (J9), before or after the
        guest is reaped (`ph.orphans.signal_child`). `start_new_session` alone
        leaves the subprocess of a cell in a session that nothing signals."""
        with suppress(OSError):
            signal_child(process, kill=True)

    async def _kill_unresponsive(self, active: _ActiveRun) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            self._kill_group(process)
        # The run is settled before the teardown, because the teardown is what
        # wakes `_pump`: `_teardown` closes the channel, and `_recv_line` returns
        # `None`. `_on_closed` would otherwise be the first writer, with a vaguer
        # account of the same death. `settle` decides the rest: after this call
        # has claimed the outcome, nothing can overwrite it.
        active.settle(
            error=(
                "the program did not stop when canceled and the runtime was killed; "
                "the namespace is gone"
            )
        )
        await self._teardown()
        self._reset_notice = True

    async def _restore(self, variables: list[dict[str, Any]]) -> dict[str, Any]:
        """Put snapshotted variables back before the next run (D17).

        Only `_rehydrate` calls this, and `_rehydrate` runs *inside* `run()`. The
        run lock is therefore already held and must not be taken again.
        `anyio.Lock` is not reentrant. When this went through a wrapper that
        took the lock, the wrapper raised and did not wait, and a restored
        namespace appeared as a failed cell.
        """
        if not self._alive:
            return {"restored": [], "failed": [record.get("var") for record in variables]}
        if not variables:
            # This is reached when every blob that a namespace named was
            # unreadable (S11). `_rehydrate` still reports those, with nothing to
            # send. The loop below sends one frame for each variable, so it
            # sends none here. The guest answers a restore by *counting* frames
            # up to the one with `more` unset. A restore of nothing would
            # therefore wait for `boot_timeout` for an answer that nothing asked
            # for.
            return {"restored": [], "failed": []}
        self._run_seq += 1
        active = _ActiveRun(run_id=self._run_seq, bindings={})
        # **One frame for each variable, in the way that `snapshot` arrives**
        # (O3). `maxSnapshotBytes` limits each *value*, and a frame that carried
        # the whole namespace had no limit. The defect of F3 therefore remained,
        # mirrored, in the one direction that its fix did not reach. It is worse
        # here than there. An over-limit line is a `ValueError` in the reader of
        # the guest, `receive` returns `None`, and the guest exits. That is the
        # reason why that reader is also sized from `frame_cap` (F8). In that
        # case the namespace is lost on the way back in, although to keep it is
        # the purpose of a restore.
        async with anyio.create_task_group() as tasks:
            for index, record in enumerate(variables):
                await self._send(
                    RestoreFrame(
                        id=active.run_id,
                        variables=[record],
                        more=index < len(variables) - 1,
                    )
                )
            # There is no `_watch` here. A restore has no bindings, so no
            # dispatch can be refused and nothing can start an abort. The clock
            # would have nothing to read. `boot_timeout` is the limit here. It
            # is one cancellation at the end, and not a poll throughout.
            with anyio.move_on_after(self.boot_timeout):
                await self._pump(active, tasks)
            tasks.cancel_scope.cancel()
        return active.value if isinstance(active.value, dict) else {}

    # --------------------------------------------------------------- frames --

    async def _handle(
        self, frame: InboundFrame, active: _ActiveRun, tasks: anyio.abc.TaskGroup
    ) -> None:
        # `frame["type"]` is compared in place and not bound to a name first. The
        # union narrows on the comparison, and mypy does not carry the narrowing
        # through an intermediate variable. The docstring of `InboundFrame` says
        # so.
        if frame["type"] == "call":
            tasks.start_soon(self._serve_call, frame, active)
        elif frame["type"] == "log":
            active.logs.append(frame["text"])
            if frame.get("truncated"):
                active.truncated = True
        elif frame["type"] == "display":
            # The frame is converted here, and the two ph-core types that it
            # flows into are not widened. A `TypedDict` is not a subtype of
            # `dict`, and a frame that has no producer yet is a poor reason to
            # loosen a wire-facing model in another package. On this line the
            # checker still sees `frame` as a `DisplayFrame`, so nothing is
            # unchecked.
            active.displays.append(dict(frame))
        elif frame["type"] == "snapshot":
            # This is awaited, not spawned. The guest sends the frame *before*
            # `done`. The namespace is therefore in the log, with its blobs
            # already on disk, before the result of the cell is. The barrier of
            # the next request writes both before the model reads the result
            # (S11).
            if frame["id"] == active.run_id and self.snapshots is not None:
                await self.snapshots.record(self.namespace, frame["id"], frame["variables"])
        elif frame["type"] == "pong":
            latency = active.answered(frame["id"], anyio.current_time())
            if latency is not None:
                self.loop_worst = max(self.loop_worst or 0.0, latency)
        elif frame["type"] == "done":
            self._settle(frame, active)

    def _settle(self, frame: DoneFrame, active: _ActiveRun) -> None:
        if frame["id"] != active.run_id:
            # A `done` for another run is a forged frame, because the guest
            # sends one for each run and only for the open run. It settles
            # nothing (C10).
            return
        error = frame.get("error")
        if isinstance(error, dict):
            settled = active.settle(
                error=as_str(error.get("message") or error.get("kind"), "the program failed")
            )
        else:
            settled = active.settle(value=frame.get("value"))
        if settled:
            # These are taken only together with an outcome that this frame
            # supplied. A `done` that arrives after a kill describes a program
            # that the caller is not told about, and the same is true of its
            # truncation flag and its total.
            if frame.get("truncated"):
                active.truncated = True
            # The total of the guest, which is known only now that its streams
            # have ended.
            active.dropped += max(as_int(frame.get("dropped")), 0)

    async def _serve_call(self, frame: CallFrame, active: _ActiveRun) -> None:
        """Serve one binding call through the full tool pipeline (C1)."""
        if frame["run"] != active.run_id:
            # This call comes from a program that already settled. A task that
            # the cell left behind made it, and the task still runs because the
            # guest is persistent. To serve the call would charge the budget of
            # this run for the work of another run, and record it under the
            # dispatch id of this run (F4). The call is answered and not
            # dropped, because the caller awaits a reply, and silence would hold
            # a task open for the life of the kernel. The guest cancels such
            # tasks at settle, so a call that reaches this point comes from a
            # task that survived cancellation.
            await self._reply(
                frame["id"], ok=False, message="the program that made this call has finished"
            )
            return
        key = (frame["global"], frame["name"])
        binding = active.bindings.get(key)
        if binding is None or binding.dispatch is None:
            await self._reply(frame["id"], ok=False, message=f"{key[0]}.{key[1]} is not available")
            return
        try:
            value = await binding.dispatch(**frame["args"])
        except CodeRunFailure as failure:
            # The failure is recorded, answered, *and* aborted. All three are
            # needed, because only the third enforces C3. The reply makes a
            # well-behaved cell raise `RunStopped` and unwind. But a cell is not
            # obliged to behave, and no waterfall can reach raw Python. The only
            # thing that can stop the cell is to end the turn of the process,
            # with the same abort ladder that user cancellation uses: the
            # frame, then the signal, then the kill.
            active.failure = failure
            # The clock is first, the reply is second, and the ask is third. The
            # reply must precede the `cancel` frame: a well-behaved cell raises
            # `RunStopped` and unwinds, which is the cleaner of the two stops.
            # But the reply is also a write that can block behind a guest that
            # has stopped reading. An abort ladder that starts only after the
            # reply went out never starts for the cell that has stopped reading.
            asking = active.begin_abort()
            await self._reply(frame["id"], ok=False, message=failure.message, fatal=True)
            if asking:
                await self._interrupt(active.run_id)
        except ToolCallError as error:
            await self._reply(frame["id"], ok=False, message=error.message, name=frame["name"])
        except Exception as error:
            await self._reply(
                frame["id"], ok=False, message=error_message(error), name=frame["name"]
            )
        else:
            # `thaw_json` does the whole job. A value that came back through the
            # log is frozen (a `MappingProxyType` over tuples), and `json.dumps`
            # does not serialize that. There is deliberately **no round trip
            # through JSON here**, because `encode` serializes the frame next.
            # The `default` of `encode` handles anything that `json` still cannot
            # represent.
            await self._reply(frame["id"], ok=True, value=thaw_json(value))

    async def _reply(
        self,
        call_id: int,
        *,
        ok: bool,
        value: object = None,
        message: str | None = None,
        name: str | None = None,
        fatal: bool | None = None,
    ) -> None:
        await self._send(
            ReplyFrame(id=call_id, ok=ok, value=value, message=message, name=name, fatal=fatal)
        )

    # ----------------------------------------------------------------- pipes --

    async def _drain(self, stream: Any, active: _ActiveRun) -> None:  # noqa: ANN401
        """Collect fd 1 and fd 2 of the *process* itself. This is mainly the
        output of a grandchild.

        A `print` in the cell arrives as a `log` frame, because the guest
        redirects `sys.stdout`. What reaches these pipes is what a subprocess
        that the cell spawned wrote. It is capped for the same reason (D4).

        Both streams append to the one ordered log of the run, so the result
        text reads in the order in which things happened. The
        `stdout + stderr + result` concatenation of prime-agent approximated
        that order.
        """
        if stream is None:
            return
        written = 0
        cap = self.limits.max_log_bytes
        # One decoder serves the whole stream, so a character that is split
        # between two reads is joined and not shown as two U+FFFDs. The cap is
        # counted in bytes, which is the unit that it names, and the cut falls
        # between characters.
        decoder = stream_decoder()

        def take(text: str) -> None:
            """Keep `text`, or as much of it as the cap has room for, and mark
            where it was cut. This one rule serves a read and the remainder of
            the stream."""
            nonlocal written
            size = utf8_size(text)
            room = cap - written
            if size <= room:
                active.logs.append(text)
                written += size
                return
            kept = head_bytes(text, room)
            active.logs.append(kept)
            # The marker shows where the cut is. The amount that followed is in
            # the total of the run.
            active.logs.append(truncation_marker(None, cap))
            active.dropped += size - utf8_size(kept)
            active.truncated = True
            written = cap

        # An exception in `_STREAM_ENDED` is the ordinary end of this task: the
        # run is over and the group canceled the task, or the child exited.
        # Neither is a failure to report, and an escaped exception would hide the
        # real outcome.
        with suppress(*_STREAM_ENDED):
            async for chunk in stream:
                if written >= cap:
                    # The chunk is read, so that the child never blocks on a
                    # full pipe. It is counted, so that the total of the run is
                    # complete.
                    active.dropped += len(chunk)
                else:
                    take(decoder.decode(chunk))
        # If the stream ended inside a character, the decoder returns one U+FFFD
        # for it. That character is reported and not lost.
        rest = decoder.decode(b"", final=True)
        if rest and written >= cap:
            active.dropped += utf8_size(rest)
        elif rest:
            take(rest)

    async def _recv_line(self) -> bytes | None:
        """The bytes of the next frame, or `None` when the child is gone or
        hostile.

        **The buffer is a `bytearray`, and only its tail is scanned.** With
        `bytes` and `+=`, a frame of several megabytes copies the whole buffer
        for each 64 KiB chunk and scans it again for a newline. That is
        quadratic in the frame size. Appending to a `bytearray` and searching
        from the previous length makes both linear.

        **The buffer is capped.** The child holds this descriptor and runs
        model-written code, so it can do `os.write(3, b"x" * 10**10)`. Without a
        cap, the host grows the buffer until the host is killed. A frame past
        the cap is not a frame. The channel is then treated as unusable (C10).
        """
        sock = self._sock
        if sock is None:
            return None
        buffer = self._buffer
        # The value is read one time here, like `max_log_bytes` in `_drain` and
        # `_collect_boot_noise`. `limits` is frozen and set at construction, so
        # this is one value for the life of the kernel. The lookup measured as
        # 0.29% of a chunk, so this is a note about tidiness and not a fix.
        cap = frame_cap(self.limits)
        while True:
            index = buffer.find(b"\n", self._scanned)
            if index >= 0:
                line = bytes(buffer[:index])
                # `del` on the front of a bytearray is amortized O(1) in CPython.
                del buffer[: index + 1]
                self._scanned = 0
                return line
            self._scanned = len(buffer)
            if self._scanned > cap:
                log.warning(
                    "ph_rlm.kernel: the runtime sent %d bytes with no frame boundary; "
                    "closing the channel",
                    self._scanned,
                )
                return None
            try:
                await anyio.wait_readable(sock)
                chunk = sock.recv(65536)
            except _CHANNEL_GONE:
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
                except _CHANNEL_GONE:
                    # Only a real closure reaches this branch. `_send_lock` keeps
                    # contention from arriving as `BusyResourceError` and being
                    # reported as an exit of the child.
                    self._on_closed()
                    return
                view = view[sent:]

    def _on_closed(self, active: _ActiveRun | None = None) -> None:
        """The child is gone. Fail whatever was waiting, and do not lose the
        session."""
        if self._alive:
            self._alive = False
            self._reset_notice = True
        if active is not None:
            active.settle(error="the runtime exited before the program finished")

    # ---------------------------------------------------------------- close --

    async def aclose(self) -> None:
        """Ask, then wait, then kill, and reap on every path (F4). The close is
        permanent: from now on, a run that held this kernel is refused (K1)."""
        self._closed = True
        await self._teardown()

    async def _teardown(self, *, ask: bool = True) -> None:
        """Stop the guest for a restart, a close, or a failed boot, even when a
        stop lands in this method itself.

        **The work is shielded, and its last steps are in a `finally`** (K2). The
        process is detached from the kernel first. A stop that landed in the
        polite shutdown therefore used to skip the kill, `_alive = False`, the
        reap, and the journal entry, and nothing would come back for the guest.
        A journal entry is dropped only after the process is reaped. A guest
        that a shortened close leaves running therefore still belongs to the
        orphan sweep.

        `ask=False` kills without asking first. It is for a guest that is
        discarded during boot or during a cell. Such a guest usually cannot read
        the ask, because a cell blocks its loop, so the ask only used up
        `shutdown_grace` before the kill.
        """
        process, self._process = self._process, None
        try:
            with releasing():
                await self._stop(process, ask=ask)
        finally:
            self._alive = False
            if process is not None:
                if process.returncode is None:
                    self._kill_group(process)
                elif process.pid is not None and self.journal is not None:
                    self.journal.forget(process.pid)
            self._close_channel()

    async def _stop(self, process: anyio.abc.Process | None, *, ask: bool) -> None:
        """Ask the guest to leave, wait, kill it if it will not, and reap it."""
        if process is not None and process.returncode is None and not ask:
            self._kill_group(process)
        if process is not None and process.returncode is None:
            # **The ask has a time limit, because it is a write and the peer may
            # not be reading.** A guest that spins in Python drains nothing. The
            # socket buffer fills, and this send waits on `_send_lock` forever.
            # `aclose` then waits with it, and the shutdown hangs on exactly the
            # kernel that most needs to be shut down. A frame that the guest
            # never reads deserves no longer wait than the exit itself. The kill
            # below is what actually ends the guest, and a half-written frame on
            # a channel that is about to be closed costs nothing.
            #
            # **One instant serves both scopes.** The teardown in the `serve` of
            # the daemon states the same rule. Two scopes that each get
            # `shutdown_grace` spend it twice. A blocked guest then cost ten
            # seconds under a constant that says five, and the number no longer
            # meant what it says.
            until = anyio.current_time() + self.shutdown_grace
            with anyio.CancelScope(deadline=until):
                await self._send(ShutdownFrame())
            with anyio.CancelScope(deadline=until):
                await process.wait()
            if process.returncode is None:
                self._kill_group(process)
        if process is not None:
            # A child that exited while the parent lives, and that is never
            # reaped, is a zombie. This wait prevents one (F4).
            with suppress(Exception):
                await process.wait()
            # **Whatever the guest started is killed on this path too** (J9). A
            # clean shutdown orphans the subprocess of a cell as completely as a
            # kill does: the guest exits, its session outlives it, and nothing
            # else looks for it. After the reap, the group is named by the pid
            # of the guest, because the reap removed the answer of `getpgid`.
            self._kill_group(process)

    def _close_channel(self) -> None:
        """Close the host end of the socket of the guest, and wake a read that
        waits on it."""
        if self._sock is not None:
            # `notify_closing` comes before the close, because the read of
            # `_pump` no longer has a timeout of its own, and this call is what
            # ends it. To close an fd with a `wait_readable` pending is not
            # enough on epoll. The closed fd leaves the interest list silently,
            # no event is ever delivered, and the future never resolves. `_pump`
            # would then block forever on a channel that no longer exists.
            # Today the EOF of the child, from the kill in `_kill_unresponsive`,
            # usually arrives first. That is an accidental rescue of the kind
            # that stops being true after an unrelated edit. anyio provides this
            # call for the case and documents the order.
            with suppress(OSError, ValueError):
                anyio.notify_closing(self._sock)
            with suppress(OSError):
                self._sock.close()
            self._sock = None


_CHANNEL_GONE = (OSError, ValueError, anyio.ClosedResourceError)
"""What this socket raises when it was closed under a caller.

There are three exceptions, because the answer depends on *where* the close
lands, and the third was missing. `sock.send` and `sock.recv` on a closed socket
raise `OSError` (EBADF). `anyio.notify_closing` raises `ClosedResourceError` into
a waiter that already waits. But `wait_readable` and `wait_writable` that are
**entered** with a closed socket raise
`ValueError("Invalid file descriptor: -1")`: `socket.close()` sets the fileno to
`-1`, and the event loop refuses to register it.

The third case needs a second pass through the wait, which is why a partial
write finds it. A frame that is larger than the socket buffer goes through
`wait_writable` and `send` many times. When `_teardown` closes the channel
between two of those turns, the close lands exactly there.
`test_a_cell_blocked_behind_a_large_reply_is_still_killed` reaches it: an 8 MB
reply is being written while the abort ladder kills the guest. It was observed
on macOS while Linux passed, because which of the three arrives first depends on
buffer sizes and wakeup order, and not on anything that this code decides. The
module already reads `ValueError` as "this fd is gone" where it calls
`notify_closing`. This is the same reading at the two places that wait.
"""


@dataclass(slots=True)
class PythonCodeRuntime:
    """The `ctx.code_runtime` provider: one kernel per namespace, spawned lazily."""

    language: ClassVar[str] = "python"
    isolation: ClassVar[Literal["process"]] = "process"
    persistence: ClassVar[Literal["namespace"]] = "namespace"
    declares_kernel_snapshots: ClassVar[bool] = True
    """The promise that the seam checks at registration (D6). `rlm-kernel-snapshot`
    turns the frames that this provider passes on into `kernel/snapshot` events."""

    limits: KernelLimits
    journal: OrphanJournal | None
    cache: Path
    interpreter_mode: InterpreterMode = "managed"
    interpreter_override: str | None = None
    skills: tuple[str, ...] = ()
    """Requirement specs installed into the managed venv. A local directory is
    installed editable, so editing a skill does not need a venv rebuild."""
    skill_modules: tuple[str, ...] = ()
    """The import names that those specs provide. Each kernel gets them at boot.

    There are two lists and not one, because they answer different questions and
    can correctly differ. A distribution named `acme-websearch` imports as
    `acme_websearch`, and `rlm-skills-python` is the row that knows both."""
    workspaces: Callable[[str], Workspace | None] | None = None
    """Agent id → its `Workspace`, set by the row (D21). See `workspace_for`."""
    sandbox: Callable[[], SandboxSeam | None] | None = None
    """The `ctx.sandbox` seam, which is requested **when a kernel starts** and is
    not held.

    This is a resolver and not the seam itself, for the same reason as
    `workspaces`. If the backend row is layered after this row, a value that is
    read at mount would be `None` for every kernel that this runtime ever spawns.
    `FsPermissions.ctx` has the same form for the same reason. When the resolver
    is absent (a profile with no sandbox seam at all), cells are limited by `cwd`
    and nothing else, which is how they were limited before."""
    boot_timeout: float = 30.0
    shutdown_grace: float = 5.0
    cancel_grace: float = 2.0
    probe_seconds: float | None = 1.0
    snapshots: SnapshotPolicy | None = None
    """Set by the `rlm-kernel-snapshot` row. Without it, the runtime still runs,
    but `persistence: "namespace"` would then be a promise that nothing keeps.
    For that reason the bundle mounts both rows together."""
    _kernels: dict[str, Kernel] = field(default_factory=dict)
    _closers: dict[str, Releaser] = field(default_factory=dict)
    """The close of each kernel, registered on the scope of its agent. An entry is
    dropped without being run when the kernel is closed in another way
    (`close_namespace`)."""
    _acquire_lock: anyio.Lock = field(default_factory=anyio.Lock)
    """Held across the whole of `_acquire`, so one namespace gets one kernel.

    The lock is runtime-wide and not per namespace, and that costs nothing. The
    suspension that it covers is `environment()`, which already serializes on its
    own `_resolve_lock`. There is no concurrency here to keep.
    """
    scopes: Callable[[str], Context | None] | None = None
    """Agent id → its live scope, so that a kernel is released *structurally*
    (F1).

    The alternative was to close on the `agent/disposed` event. But `emit`
    schedules an async listener and does not await it, so the release would
    happen at some later time and not as part of unwinding. A disposer that is
    registered on the scope of the agent itself makes the child process an
    artifact of that scope, like every other acquired resource in pH.

    The agent registry is asked when a kernel is built
    (`AgentRegistry.scope_of`), and the answer for an agent that is gone is a
    refusal (`apply`). `None`, for a runtime that no registry is connected to,
    builds a kernel that only the unmount of this row closes. This used to be a
    separate table, filled on `agent/created` and never emptied. It held the
    scope of every agent that the runtime had ever heard of."""
    _environment: RuntimeEnvironment | None = None
    _resolve_lock: anyio.Lock = field(default_factory=anyio.Lock)

    def _loops(self) -> str:
        """How far behind the event loops of the guests run (M2).

        **This is a load reading, and it is printed whether or not anything is
        wrong.** Each kernel is probed every `probe_seconds` while a cell is
        open. The guest answers from the reader task, which shares the loop with
        the cell. The round trip is therefore the delay that anything
        cooperative would meet: a `cancel` frame, a `SIGINT` callback, or a
        binding reply. The worst value is the number to read. A loop that is
        usually prompt and sometimes seconds behind is a cell that uses CPU in
        bursts, and that is what a person who looks at a slow agent wants to
        see.

        A stall is a probe that went a whole `probe_seconds` with no answer,
        which means a cell that did not yield for that time (one second by
        default). It is counted and not judged. That is ordinary for compute,
        and the per-run CPU budget (`cpu_seconds`) is what limits it.
        """
        # A kernel whose every probe was starved has a stall count and no round
        # trip. A filter on the round trip alone dropped it from the total *and*
        # from the sum. The line could then report nothing measured while stalls
        # were recorded, which is the case that most needs to be seen.
        measured = [
            one for one in self._kernels.values() if one.loop_worst is not None or one.loop_stalls
        ]
        if self.probe_seconds is None:
            return "probe off (`probeSeconds: null`), so nothing is measured"
        if not measured:
            return "no cell has run yet, so nothing has been measured"
        stalls = sum(one.loop_stalls for one in measured)
        worst = max((one.loop_worst or 0.0) for one in measured)
        return (
            f"{len(measured)} probed; worst round trip {worst * 1000:.0f} ms, "
            f"{count_of(stalls, 'stall')} "
            f"(a cell that did not yield for {self.probe_seconds:g}s)"
        )

    def describe(self) -> list[tuple[str, str]]:
        """What `phern doctor` prints about the workers that run model code (I-2).

        **Nothing here resolves the interpreter.** `environment()` shells out to
        `uv` to build the managed venv. A diagnostic that took thirty seconds
        and a network the first time someone ran it would be a diagnostic that
        people stop running. An unresolved interpreter is therefore reported as
        unresolved. That is also the more useful fact: it says that the venv is
        not built yet.
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
            ("guest loops", self._loops()),
            ("cells confined by", self._confinement()),
        ]

    def _limits(self) -> str:
        """The per-child limits, as the live kernels actually got them.

        **The limits are read from the guests where there are any.** This is the
        rule of `_confinement`, for the same reason. The request is what the
        host asked for. On a platform that refuses `RLIMIT_AS`, the number in
        force is not the number configured. macOS refuses it, with `ValueError:
        current limit exceeds maximum limit`. To print the request would make
        `phern doctor` claim a limit that nothing holds. E1 forbids that kind of
        claim one level up, in the tier table.

        The weakest live kernel decides, as in `_confinement`. If one child could
        not take the limit, the row must not say that every child has it.
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
        """Whether authored code is confined at the kernel, and by what.

        This has its own row in `phern doctor`, because it is the difference
        between a cell that can write `/etc` and a cell that cannot. It is also
        *conditional*: on a backend, and on the agent having a workspace. Where
        kernels are running, the answer is read from them, so this reports what
        is true and not what would be true.

        **It reports the weakest live kernel, not the strongest.** A first draft
        joined the set of backends that had confined something. That read as
        "bwrap — every live kernel's writes are bounded" in the one case where
        the sentence is false: one agent with a workspace and one without. There
        is only ever one provider, so there was never more than one name to
        join. The join was only a way to overstate.
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
        """Resolve the interpreter one time, on first use.

        The work is lazy and runs in a worker thread. Building the managed venv
        shells out to `uv`, and neither `phern --dump-config` nor a session that
        runs no cells should pay for it.
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

    def workspace_for(self, agent_id: str) -> Workspace | None:
        """The workspace of this agent, or `None` before one is acquired.

        It is resolved for each kernel and not held as one runtime-wide `cwd`,
        because the namespace *is* the agent id. A parent and its children share
        this runtime and must not share a checkout. The `worktree` tier exists
        to prevent that collision.
        """
        if self.workspaces is None:
            return None
        try:
            return self.workspaces(agent_id)
        except Exception:
            log.warning("ph_rlm.kernel: workspace lookup failed for %s", agent_id, exc_info=True)
            return None

    async def run(self, request: CodeRunRequest) -> CodeRunResult:
        """Run one program in its namespace, within the boundary of the agent in
        either case (`CodeRunRequest.agent`, S21). The namespace is the one of
        the agent itself unless the request names another."""
        namespace = request.namespace or request.agent
        kernel = self._kernels.get(namespace)
        if kernel is None:
            kernel = await self._acquire(namespace, request.agent)
        result = await kernel.run(request.program, request.bindings, request.token)
        self._note_denial(kernel, request.agent, result)
        return result

    def _note_denial(self, kernel: Kernel, agent_id: str, result: CodeRunResult) -> None:
        """Record a boundary that the cell hit, in the way that `ctx.shell`
        records one.

        **This is here and not in `Kernel`.** `Kernel` holds a confiner and no
        seam, and both are available here. The seam owns the decision of what
        counts as a refusal (`report_denial`). A cell that writes `/etc/passwd`
        therefore now leaves the same `sandbox/denied` record, and the same
        `/sandbox allow path` line, as the identical refusal from `tool-bash`.
        Without this, the kernel had adopted the boundary and not the rule. The
        asymmetry was visible inside one change: a *network* refusal was already
        recorded, because the proxy is told which agent it refuses.

        This runs only for a run that failed, which `report_denial` itself
        requires. The error is where `OSError: [Errno 30] Read-only file system`
        lands, and a cell that printed those words and succeeded was refused
        nothing. The record is made for the agent that the run was for, so it
        lands in the transcript that the author of the cell reads.
        """
        confined = kernel.confined
        if confined is None or result.error is None:
            return
        seam = None if self.sandbox is None else self.sandbox()
        if seam is not None:
            seam.report_denial(confined, (result.error, result.logs), agent_id)

    def confiner(
        self,
        agent_id: str,
        workspace: Workspace | None = None,
    ) -> Callable[[tuple[str, ...]], ConfinedArgv] | None:
        """How to confine the kernel of this agent, or `None` where nothing can.

        **This is the same condition that `ctx.shell` applies, on purpose.** A
        command that the harness runs for an agent and a cell that the agent
        writes are the same kind of thing: code from somebody else, in the
        workspace of that agent. They therefore get the same boundary from the
        same `workspace_policy`, and a deployment cannot have `tool-bash`
        confined and `run_code` not confined. For the same reason, this is *not*
        gated on the containment tier. `ctx.shell` confines at every tier where
        a backend exists, and the containment section of `phern doctor` already
        says so in the sentence about commands that the harness wraps.

        The result is `None` when there is no backend, no backend that enforces,
        or no workspace to be the writable root. It is never a passthrough. The
        seam refuses and does not pretend, and a caller must not mistake absence
        for confinement.

        `workspace` is the one that `_build` already resolved. When it is
        omitted, it is looked up. Two lookups for one spawn also logged the
        failure warning twice.
        """
        seam = None if self.sandbox is None else self.sandbox()
        if workspace is None:
            workspace = self.workspace_for(agent_id)
        if seam is None or not seam.available or workspace is None:
            return None
        policy: SandboxPolicy = workspace_policy(workspace)
        # `agent=` is passed so that a host that the egress proxy refuses is
        # recorded in the session of *this* agent. That is the transcript that
        # the author of the cell reads.
        return partial(seam.confine, policy=policy, agent=agent_id)

    async def _acquire(self, namespace: str, agent_id: str) -> Kernel:
        """The one kernel for this namespace, built one time however many callers
        ask (F6).

        **The check is made twice, because `environment()` suspends.** `run`
        reads `_kernels` and finds nothing, then awaits the interpreter resolve.
        On a cold cache the resolve builds a venv and takes seconds. Two first
        runs on one namespace both got past the read, and both built a `Kernel`.
        The second overwrote the first in the dict. Both then spawned a guest on
        their first cell, and `close_namespace` could reach only the one that
        was still in `_kernels`. The other ran until the process exited, and
        held a socket, a pid, and whatever the cell had open.

        Two first runs on one namespace are ordinary. The namespace of a model
        *is* its agent id, and a fan-out of parallel tool calls on one agent is
        what Code Mode is for.
        """
        async with self._acquire_lock:
            settled = self._kernels.get(namespace)
            if settled is not None:
                return settled
            return await self._build(namespace, agent_id)

    async def _build(self, namespace: str, agent_id: str) -> Kernel:
        """A kernel for `namespace`, within the boundary of `agent_id`: the
        workspace of the agent as the directory and environment, its
        confinement, and its scope as the owner that closes the kernel. This is
        the same whether the namespace is the one of the agent itself or another
        that it runs beside, such as the namespace of the H1 probe."""
        workspace = self.workspace_for(agent_id)
        confine = self.confiner(agent_id, workspace)
        env: Mapping[str, str] = {}
        if workspace is not None:
            # This is what `ctx.shell` gives a confined command. A confined
            # kernel has no `/tmp` of its own (`confined_env`).
            env = confined_env(workspace) if confine is not None else workspace.env
        kernel = Kernel(
            namespace=namespace,
            environment=await self.environment(),
            limits=self.limits,
            journal=self.journal,
            cwd=None if workspace is None else workspace.root,
            env=env,
            confine=confine,
            snapshots=self.snapshots,
            skills=self.skill_modules,
            boot_timeout=self.boot_timeout,
            shutdown_grace=self.shutdown_grace,
            cancel_grace=self.cancel_grace,
            probe_seconds=self.probe_seconds,
        )
        scope = None if self.scopes is None else self.scopes(agent_id)
        if scope is not None:
            # The close is registered before the entry is made, and nothing
            # suspends between them. A kernel in the table therefore always has
            # the close that removes it.
            self._closers[namespace] = scope.add_disposer(
                partial(self.close_namespace, namespace), label=f"code-runtime:{namespace}"
            )
        self._kernels[namespace] = kernel
        return kernel

    async def close_namespace(self, namespace: str) -> None:
        """Shut down the kernel of one agent. This happens when the agent goes,
        or before it goes. In the second case the close is removed from the scope
        of the agent."""
        closer = self._closers.pop(namespace, None)
        if closer is not None:
            closer.drop()
        kernel = self._kernels.pop(namespace, None)
        if kernel is not None:
            await kernel.aclose()

    async def aclose(self) -> None:
        for namespace in list(self._kernels):
            await self.close_namespace(namespace)


def _declare(namespace: CodeBindingNamespace) -> dict[str, Any]:
    """What the guest needs to build a proxy: names, not dispatch closures. It
    also gets the parameter names of each binding, in the order that the SDK
    listing shows them, so that a call written from the listing binds as it reads
    (`ph_runtime.proxies`)."""
    return {
        "name": namespace.name,
        "description": namespace.description,
        "bindings": [
            {
                "name": binding.name,
                "description": binding.description,
                "parameters": parameter_names(binding),
            }
            for binding in namespace.bindings
        ],
    }


class Config(KernelLimits):
    """The row config for the Python runtime: the limits, plus how to reach a
    child.

    It extends `KernelLimits`, so that the YAML stays flat (`cpuSeconds: 30`) and
    the limits have one declaration.
    """

    python: InterpreterMode = "managed"
    """Which interpreter runs the cells. `managed` builds
    `$PH_CACHE/runtime-venv`. `host` reuses the interpreter of pH itself, which
    is faster and is a wider surface. See `venv`."""
    interpreter: str | None = None
    boot_timeout_seconds: float = 30.0
    shutdown_grace_seconds: float = 5.0
    cancel_grace_seconds: float = 2.0
    """How long a cell that was asked to stop has before it is killed.

    Both directions have a cost. A longer time widens the window in which a cell
    that swallowed a refusal can finish synchronous work (see
    `Kernel.cancel_grace`). A shorter time kills a cell that is legitimately slow
    to unwind, and a kill costs the namespace."""
    probe_seconds: PositiveFloat | None = 1.0
    """How often the guest loop of an open run is probed (M2), or `null` for
    never.

    A probe is one frame each way, answered by the reader task of the guest. The
    round trip is what `phern doctor` reports as how far behind that loop runs. A
    lower value samples a bursty cell more finely. A higher value is cheaper on a
    host with many namespaces. It is a measurement interval, not a deadline.
    Zero or less is refused, and is not read as "as often as possible".

    **What `null` gives up** (P12-03). The loop gauge of the doctor has nothing
    to report. The probe is also the only thing that repairs a run whose `done`
    frame was lost (`PingFrame.run`), because no event says that a frame was
    never sent. With the probe off, such a run waits until somebody cancels it,
    and the abort ladder ends it from there. An unattended run has nobody to do
    that."""
    skills: tuple[str, ...] = ()


@plugin("code-runtime-python", affects="environment", config=Config, inject=[CODE_RUNTIME])
async def apply(ctx: Context, config: Config) -> None:
    """Register the runtime. The journal is shared, and `subprocess-local` sweeps
    it.

    **This row used to sweep too, and both rows swept the same file.** There is
    one journal per user per boot, so an `rlm` profile parsed and rewrote it
    twice at every start. The second pass did nothing to the output of the first,
    and each pass was behind its own thread hop. `subprocess-local` is in
    `base.yaml` and is therefore always mounted, so it owns the sweep. This row
    keeps its own handle, because the guest cannot be spawned through that seam
    at all. `spawn` passes `stdin=None`, which means *inherit*. To give the
    process that runs model-authored cells a descriptor on the terminal of the
    user is the opposite of the purpose of J9. (The cause is not `pass_fds`,
    which an earlier version of this sentence blamed. `anyio.open_process` takes
    it and the spec could forward it, so that reading invited exactly the change
    that this rules out.) The kernel writes to the journal through `ph.orphans`
    directly, and shares the ledger and the sweep.
    """
    roots = resolve_roots()
    journal = host_journal()

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
        probe_seconds=config.probe_seconds,
    )

    async def enter() -> Disposer:
        return runtime.aclose

    await ctx.effect(enter, label="code-runtime-python")
    ctx.provide(PYTHON_RUNTIME, runtime)
    ctx.require(CODE_RUNTIME).register(runtime)

    # The namespace *is* the agent id, so a kernel is scoped exactly like the
    # tools, the inbox and the log of the agent, and the same unwinding releases
    # it. The registry is asked when a kernel is built, because the registry
    # knows which agents are live.
    def scope_of(agent_id: str) -> Context | None:
        agents = ctx.get(AGENTS)
        if agents is None:
            return None
        scope = agents.scope_of(agent_id)
        if scope is None:
            # The agent went away while the interpreter was being resolved, which
            # takes seconds on a cold cache. When the kernel was built anyway, it
            # had no owner. Its guest ran until this row unmounted, and an agent
            # that was brought back under the same id ran its cells in it.
            raise InactiveScopeError(
                f"agent {agent_id} is gone; no kernel is built for an agent that cannot own it"
            )
        return scope

    runtime.scopes = scope_of

    # The workspace is asked for at kernel start and not captured here. A
    # workspace is acquired after the agent exists (P4-08), and a value read at
    # mount would be `None` for every kernel that this runtime ever spawns.
    runtime.workspaces = partial(workspace_of, ctx)
    # The seam is asked for when a kernel starts, not now. A profile may layer
    # its backend after this row, and a seam read here would be the wrong answer
    # for exactly that profile. `workspace-readonly-scratch` and `permissions-fs`
    # both say so.
    runtime.sandbox = partial(ctx.get, "sandbox")

    contribute(
        ctx, Diagnostic(id="code-runtime", title="Code runtime", read=runtime.describe, order=40)
    )
