"""`ctx.subprocess`: spawning, with nothing implicit.

Two properties matter more than convenience here.

**The spec is fully explicit.** `SubprocessSpawnSpec` names argv, cwd, stdio and
the termination grace. There are no hidden defaults. A caller that did not
think about the working directory of a process that the *model* asked for must
still state one. A default inherited from the harness process is the wrong
answer, because `ctx.workspace` gives each agent its own root (Phase 4).

**The environment is scrubbed.** A child runs code the model wrote, so it does
not inherit `*KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*PASSWD*` or
`*CREDENTIAL*` (I-4, `DEFAULT_SCRUB`). A credential reaches an adapter as a
`CredentialRef` and is resolved at the edge (I-3). A child never needs the
value, and a child that does must be given it explicitly.

**Both bounds are in this seam, and that is the purpose of P7-13.** A child is
the program of somebody else. It decides how much it prints and how long it
runs. Every caller that trusted it could lose the whole process to one runaway
command. `max_output` bounds the memory and `timeout_ms` bounds the time. Both
are in the seam, so `!!`, `tool-bash` and every future caller get them, and no
caller has to remember them. What a caller then clips for a log or a card is a
*display* choice, on top of a bound that already held.

@module ph.seams.subprocess
"""

from __future__ import annotations

import logging
import os
import subprocess as _sp
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import anyio
import anyio.abc
from pydantic import Field

from ..cancel import Cancellation, raced
from ..characters import capped_text
from ..cordis import Context, Disposer, maybe_await, plugin
from ..keys import SUBPROCESS
from ..orphans import OrphanJournal, host_journal, signal_child
from ..wire import WireModel
from .diagnostics import Diagnostic, contribute

__all__ = [
    "COHERENT",
    "DEFAULT_SCRUB",
    "LOCATION",
    "MAX_OUTPUT",
    "EnvScrub",
    "SubprocessHandle",
    "SubprocessResult",
    "SubprocessService",
    "SubprocessSpawnSpec",
    "apply",
    "first_line",
    "scrub_env",
]

log = logging.getLogger("ph.seams.subprocess")

Stdio: TypeAlias = Literal["pipe", "inherit", "null"]
Stream: TypeAlias = Literal["stdout", "stderr"]

MAX_OUTPUT = 8 * 1024 * 1024
"""How much of each stream of the output of a child is *kept*, for each spawn
(P7-13).

The value is generous, because this is not a display limit. `ph.seams.shell`
clips 64 KiB into the log (`SHELL_OUTPUT`), and `tool-result-offload` spills at
80 000 characters. Both of those run on a string that already exists. This
value is the ceiling that stops that string from existing. A `find /` or a
build log on a bad day is tens of megabytes. The drain used to hold all of it
and then decode a second copy to `str`.

**The value is per stream, so the real ceiling is twice this**: the `bytearray`
and then the decoded `str`, for each of two streams. It is per stream and not a
shared budget, because a chatty stdout must not starve the stderr that says why
the command failed. That is the half that a reader needs most.

This is the *default* and not the only value. `subprocess-local` takes a
`max_output_bytes`, so an operator whose gate is a 40 MiB build log has a place
to say so. There is no way to write *unbounded*. A caller that forgets the
bound is exactly the caller that this protects.
"""

DEFAULT_SCRUB = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD", "CREDENTIAL")
"""The substrings that mark an environment name as a credential, by default
(I-4).

The match is by substring and is deliberately broad. `MY_API_KEY_2` and
`GH_TOKEN_FILE` are both worth losing. A child that really needs a secret
should be given it explicitly, and not by inheritance.

This is a **default** and not the rule. `subprocess-local` takes `scrub` and
`keep`, because a match this broad is wrong for some deployment. See
`EnvScrub.keep`.
"""

COHERENT = (("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"),)
"""Families of environment variables that must be dropped or kept **together**.

**A broad scrub is acceptable. An *incoherent* scrub is not.** git configures
itself from `GIT_CONFIG_COUNT=n` plus `GIT_CONFIG_KEY_0…n-1` and
`GIT_CONFIG_VALUE_0…n-1`, and `KEY` matches those key names. So the default
scrub removed the keys and kept the count. git then read a configuration that
declares two entries and supplies none:

    error: missing config key GIT_CONFIG_KEY_0
    fatal: unable to parse command-line config

That failure hits **every** git and jj invocation that pH makes, on any host
that uses this entirely standard mechanism: a sandbox, a CI runner, tooling
that injects `http.extraheader`. It was found when a sandbox set the variables
here. The bug was never in the sandbox.

To drop the whole family is also the *safer* reading of I-4, and not a
relaxation of it. `GIT_CONFIG_VALUE_n` is where the value of an injected auth
header would be, and `VALUE` matches none of the marks. To keep a family whole
would keep a credential. This seam drops the family whole.
"""


LOCATION = frozenset({"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"})
"""Variables that outrank `cwd` for the tool that reads them (J10).

This takes the argument of `COHERENT` one step further. The test that this seam
applies is not "is this a secret", because `GIT_CONFIG_COUNT` is not a secret.
The test is **would a child misbehave if it inherited this**. By that test,
these three are the worst names that pH can be started with. git resolves them
before `cwd`. So a pH that is launched from a `git rebase --exec`, a pre-commit
hook or a `git bisect run` gives every child an answer to "which repository"
that has nothing to do with the workspace that the child was given. The
worktree of an agent is staged into the index of the user, checkpoints are
written to the object store of the user, and a `git commit` that the *model*
typed lands on the branch of the user.

The names are dropped here and not at each spawner, because three spawners
needed the rule and the third was missed: `workspace_git.git`,
`workspace_jj.jj` and `shell.run`. `shell.run` is the one that a model drives
directly. A caller that means to set one of these names passes it in `extra`.
`extra` is applied after the drop, so the value survives: `tree_hash` keeps its
`GIT_INDEX_FILE`. That is the only way around the drop. `keep` does the same
for a credential-shaped name that an operator wants back.
"""


def _family(name: str) -> tuple[str, ...] | None:
    """The coherent family `name` belongs to, if any."""
    for family in COHERENT:
        if any(name == member or name.startswith(member) for member in family):
            return family
    return None


@dataclass(frozen=True, slots=True)
class EnvScrub:
    """Which environment names a spawned child never inherits (I-4).

    This is a value and not a module constant, because the breadth is deliberate
    and is therefore sometimes wrong. `GIT_CONFIG_KEY_n` is a *configuration*
    name that matches `KEY`. A deployment that wants git configured through it
    needs a way to say so without editing pH. `subprocess-local` carries both
    fields.

    The default is unchanged, and the default is the safe direction. A
    deployment that configures nothing scrubs exactly what it scrubbed before.
    """

    marks: tuple[str, ...] = DEFAULT_SCRUB
    """Substrings that mark a name as a credential, matched case-insensitively."""
    keep: frozenset[str] = frozenset()
    """Names that are kept even though they match. This is the override, and it
    matches by exact name on purpose. A pattern here would be a second scrub
    language to get wrong, and what an operator actually knows is the name of
    the variable."""

    def drops(self, name: str) -> bool:
        """Whether a child is denied this name."""
        if name in self.keep:
            return False
        upper = name.upper()
        return any(mark.upper() in upper for mark in self.marks)

    def apply(
        self, base: Mapping[str, str] | None = None, *, extra: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """The parent environment minus the credentials, the broken families and
        the inherited tool locations."""
        source = os.environ if base is None else base
        dropped = {name for name in source if self.drops(name)}
        # A family loses every member as soon as it loses one, so a child never
        # sees a half-configured tool. See `COHERENT`.
        broken = {_family(name) for name in dropped} - {None}
        kept = {
            name: value
            for name, value in source.items()
            # `LOCATION` does not go through `drops`. That predicate answers
            # "is this a credential", and these names are not credentials. See
            # `LOCATION`.
            if name not in dropped and name not in LOCATION and _family(name) not in broken
        }
        if extra:
            kept.update(extra)
        return kept


def first_line(text: str) -> str:
    """The first non-blank line of the output of a tool, or `""`.

    It is here and not in each seam that reads the output of a subprocess,
    because two seams had their own copy and a third was about to. A decline
    that quotes the backend says "setting up uid map: Permission denied" where
    it would otherwise say "the sandbox refused a write". Only the quoted line
    tells an operator what to do.
    """
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def scrub_env(
    base: Mapping[str, str] | None = None, *, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The parent environment minus anything that looks like a credential.

    This is the **default** scrub, for a caller with no `ctx` to ask. A caller
    that has a `ctx` calls `ctx.subprocess.env()`. That is the same scrub under
    the `scrub` and `keep` of the deployment, and it is the reason that those
    two are config.
    """
    return EnvScrub().apply(base, extra=extra)


@dataclass(frozen=True, slots=True)
class SubprocessSpawnSpec:
    """Everything about one spawn, stated."""

    argv: tuple[str, ...]
    cwd: Path
    stdio: Stdio = "pipe"
    grace_ms: int = 5_000
    """How long a terminated child may take to exit before it is killed.

    This is the *termination* grace and not a run limit. `timeout_ms` is the
    run limit. An earlier version confused the two: `ctx.shell` passed the
    `timeout_ms` of a caller here. A tool whose description promised "kill the
    command after this long" then only widened the time that a child that was
    already dying had to finish (P7-13)."""
    timeout_ms: int | None = None
    """How long the child may run before `run()` terminates it. `None` waits.

    It is on the spec beside `grace_ms` because the two are one policy about
    the lifetime of a child, and a caller that reads this class should see
    both."""
    env: dict[str, str] | None = None
    """`None` scrubs and inherits. A dict replaces the whole environment, so
    `{}` is "nothing at all"."""
    max_output: int | None = None
    """How much of each stream to keep, or `None` for the value of the
    deployment.

    See `MAX_OUTPUT`. A spec that names no value is still capped, and that is
    the purpose."""
    on_stderr_line: Callable[[str], None] | None = None
    """Called with each line that the child writes to stderr, as it is written.

    It is for a child whose stderr is a report of how far it has got, such as a
    tool run with `--progress` (`ph.toolcli`). Such a stream is **delivered and
    not kept**. `pump` keeps the head of a stream up to `max_output`, and the
    line of a progress stream that matters most is its last. So with a listener
    the `stderr` of the handle stays empty and every line reaches the listener,
    however many there are. An exception that a listener raises is logged, and
    the drain goes on: a mistake of a reader must not leave the child blocked
    on a full pipe."""
    forwards_signals: bool = True
    """Whether a signal sent to this argv reaches the command that it runs
    (`ConfinedArgv.forwards_signals`). It is `False` for a wrapper that keeps
    the command in a PID namespace, as `bwrap` does. Such a child is killed
    outright: `terminate` does not ask it to stop first.

    **Asking leaves it running.** The wrapper passes no `SIGTERM` on, so the
    command never had the grace. For its first milliseconds, the init of the
    namespace ignores `SIGTERM`, as an init does, and it has not yet armed
    `--die-with-parent`. The wrapper dies of the signal, this seam sees that
    its child is gone and stops, and the command starts with no parent.
    Measured 2026-11-11, with `bwrap … --unshare-pid sleep` and its group
    signaled 0 to 50 ms after the spawn: `SIGTERM` left 30 of 88 running, all
    between 1 and 6 ms, and `SIGKILL` left none. A stop has to leave nothing
    behind, so that a restart begins clean. A polite stop is only worth having
    where the command hears it."""


@dataclass(slots=True)
class SubprocessHandle:
    """A live child, with offset-addressable output."""

    spec: SubprocessSpawnSpec
    process: anyio.abc.Process
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    cap: int = MAX_OUTPUT
    """How much of each stream the drain of this child keeps. `spawn` resolves
    it from the spec and the config of the row, so `pump` reads one number."""
    dropped: int = 0
    """Bytes that this child printed past `cap` and nobody kept, across both
    streams.

    It is a count and not a flag. "Output was truncated" tells a model less
    than "37 MB was dropped", and a caller that decides whether to run the
    command again with a filter wants the size. It is one sum, because every
    reader of it (the shell seam, `!!`, `tool-bash`) asks only how much went
    missing."""
    dispose: Disposer | None = None
    """Releases the effect of this child: terminate, reap, forget, deregister.

    `spawn` sets it. `spawn` used to discard what `ctx.effect` returned. A
    caller that was finished with a child then had no way to say so, and the
    effect lived as long as its scope did. `run` is the caller that is always
    finished."""
    _drained: bool = False
    _waited: bool = False

    @property
    def finished(self) -> bool:
        """Whether the child was seen through: its output was read to the end
        (`pump`) and its exit was awaited (`wait`). The release of such a child
        leaves its group alone, because whatever the child left running it left
        on purpose. Any other release is a stop (`terminate`). The exit alone
        is not enough. A leader that exited while a job that it started holds
        the pipe was not seen through, and a timeout or a cancel that stops it
        then is a stop."""
        return self._drained and self._waited

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)

    @property
    def returncode(self) -> int | None:
        code: int | None = self.process.returncode
        return code

    async def pump(self) -> None:
        """Drain both pipes until the child closes them, keeping at most `max_output`.

        **Reading never stops. Only keeping stops.** A drain that stopped
        reading at the cap would leave the child blocked on a full pipe. The
        block would be permanent, because nothing else empties the pipe. A
        bound that deadlocks the thing that it protects is worse than no bound.
        So every chunk is read, and the overflow is counted and not stored.

        The *head* is what survives. A command that fails usually says why in
        its first lines. To keep the tail would serve a progress bar and lose
        the error.
        """
        cap = self.cap

        async def drain(stream: Stream) -> None:
            source = self.process.stdout if stream == "stdout" else self.process.stderr
            if source is None:
                return
            sink = self.stdout if stream == "stdout" else self.stderr
            dropped = 0
            listener = self.spec.on_stderr_line if stream == "stderr" else None
            if listener is not None:
                self.dropped += await _delivered(source, listener, cap)
                return
            try:
                async for chunk in source:
                    room = cap - len(sink)
                    if len(chunk) <= room:
                        # The whole chunk fits, so it is added directly and not
                        # sliced. Every ordinary command takes this path.
                        sink.extend(chunk)
                        continue
                    if room > 0:
                        sink.extend(chunk[:room])
                    dropped += len(chunk) - max(room, 0)
            finally:
                # Added one time and not per chunk. It is in a `finally` so
                # that a canceled drain still reports what it had already
                # discarded.
                self.dropped += dropped

        async with anyio.create_task_group() as scope:
            scope.start_soon(drain, "stdout")
            scope.start_soon(drain, "stderr")
        self._drained = True

    async def wait(self) -> int:
        """Await exit and reap. Idempotent."""
        code = int(await self.process.wait())
        self._waited = True
        return code

    async def terminate(self) -> None:
        """Stop the child, and always reap it. The steps are: send `SIGTERM` to
        its group, wait for at most `grace_ms`, then send `SIGKILL` to its
        group. A child whose wrapper passes no signal on gets no `SIGTERM`: it
        is killed at once (`SubprocessSpawnSpec.forwards_signals`).

        A stop kills the whole group (J4), whether the child itself is gone or
        not. A child that was seen through (`finished`) is only reaped.

        The `aclose()` in the `finally` keeps zombies from accumulating (F4). A
        child that exited and was never awaited stays in the process table for
        as long as the parent lives.

        **The whole method is shielded, and bounded** (J4). Only the reap used
        to be shielded. A caller that was canceled during the grace then
        unwound past the `kill()`, and a cancel during the grace is the
        ordinary way that a timeout arrives. A child that ignores `SIGTERM`
        went on holding the disposer, and its scope, until it chose to exit.
        The escalation is the purpose of this method, so it has to survive the
        cancellation that asked for it. The grace keeps that from being an
        unbounded wait.
        """
        with anyio.CancelScope(shield=True):
            try:
                if self.process.returncode is None and self.spec.forwards_signals:
                    signal_child(self.process, kill=False)
                    with anyio.move_on_after(self.spec.grace_ms / 1000):
                        await self.process.wait()
                if not self.finished:
                    signal_child(self.process, kill=True)
            except ProcessLookupError:  # pragma: no cover - already gone
                pass
            finally:
                try:
                    # `aclose` and not `wait`: it closes the pipes of the child
                    # *and* reaps. A timeout cancels `pump` in the middle of a
                    # read, so the drains let go of streams that nobody then
                    # closed. The transport was then finalized after the loop
                    # had gone. asyncio reports that as an unraisable
                    # `Event loop is closed` from a `__del__`, in whatever test
                    # ran next.
                    await self.process.aclose()
                except Exception:
                    log.debug("ph.seams.subprocess: reaping pid %s failed", self.pid)


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    """What one finished child produced, and what was lost in getting it.

    This is a record and not the `(code, out, err)` tuple that it used to be.
    With a cap and a clock in the seam, a result can now be *incomplete* in two
    ways, and a tuple has no place to say so. Every caller would then render a
    truncated log as if it were the whole log.
    """

    exit_code: int
    stdout: str
    stderr: str
    dropped: int = 0
    """Bytes past the cap that nobody kept, across both streams."""
    timed_out: bool = False
    canceled: bool = False
    """Whether `signal` ended the child before the command finished (N3).

    **A third bound needs a third answer.** `run` applies three bounds: the
    timeout, a disposal, and the cancel of the user. It reported only two. So a
    child that was killed because somebody pressed stop came back as an
    ordinary result, with `timed_out=False` and whatever exit code the kill
    produced. That is indistinguishable from a command that exited on its own.
    A caller must not read it that way: a cancel is not a result."""


class Config(WireModel):
    """Row config for `subprocess-local`."""

    sweep_orphans: bool = True
    """Whether mounting this row kills the children that a hard-killed run left
    behind.

    It is on by default, because a re-parented child is using CPU now. A
    deployment that shares `$PH_RUNTIME` with something that pH does not own
    would turn it off."""
    scrub: list[str] = Field(default_factory=lambda: list(DEFAULT_SCRUB))
    """Substrings that mark an environment variable as a credential that a child
    never inherits (I-4). The match is case-insensitive and anywhere in the
    name.

    This is config and not a constant, because the breadth is deliberate and is
    therefore sometimes wrong. A deployment that narrows it also has to say so
    in a profile, where the change can be reviewed, and not by editing pH."""
    keep: list[str] = Field(default_factory=list)
    """Names that are kept even though they match `scrub`. This is the exception,
    by exact name.

    `GIT_CONFIG_KEY_0` is the case that this exists for: a *configuration* name
    that matches `KEY`. It needs no entry here today, because `COHERENT` keeps
    the numbered-config family of git whole either way. This setting is for the
    deployment that really wants such a variable passed through. Without it,
    that deployment could only fork pH."""
    max_output_bytes: int = MAX_OUTPUT
    """How much of each stream of the output of a child is kept. See
    `MAX_OUTPUT`.

    This is config and not a constant, for the reason that the `ignore` of
    `fs-local` is (P6-19): the value is wrong for some deployment. `ctx.shell`
    takes no `max_output`. So without this setting the 8 MiB default would be
    the only ceiling that `!!`, `tool-bash` and an autonomous gate could ever
    run under, and an operator whose gate is a 40 MiB build log would have to
    edit source."""


@dataclass(slots=True)
class SubprocessService:
    """The service published as `ctx.subprocess`."""

    ctx: Context
    journal: OrphanJournal | None = None
    """Where a spawn is recorded, so that a hard kill leaves a trail (F5).

    **It is optional, because a deployment may have nowhere to write it**: a
    read-only `$PH_RUNTIME`, or a test with no roots. A seam that refused to
    spawn because of a diagnostic would trade a working harness for a tidier
    one. `None` means that the gap that this closes is open again.
    `phern doctor` reports that plainly, and the seam does not pretend
    otherwise.
    """
    max_output: int = MAX_OUTPUT
    """The deployment's ceiling, applied to any spec that names none."""
    scrub: EnvScrub = field(default_factory=EnvScrub)
    """The credential scrub of the deployment, applied to every child that this
    seam spawns.

    It is on the seam because the alternative is a module constant that no
    profile can reach. "What a spawned child inherits" is exactly what a
    deployment has to be able to state. See `EnvScrub` and `COHERENT`.
    """

    def env(self, *, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        """The environment that a child of this deployment gets. **This is the one
        function that builds it.**

        Every seam that spawns a tool (git, jj, agentfs, the sandbox backend,
        the kernel) used to call the module-level `scrub_env` itself. The
        `scrub` and `keep` of the deployment then reached none of them: the row
        could be configured, and every actual child was still built from the
        default. The config has an effect only where the spawner asks the seam.
        The kernel manager of `ph-rlm` does not spawn through this seam. It
        reads `scrub` from the seam and applies it to the guest itself.
        """
        return self.scrub.apply(extra=extra)

    async def spawn(
        self, spec: SubprocessSpawnSpec, *, scope: Context | None = None
    ) -> SubprocessHandle:
        """Spawn a child owned by `scope`, terminated and reaped on disposal."""
        owner = self.ctx.owner_for(scope)
        env = self.env() if spec.env is None else spec.env
        cap = self.max_output if spec.max_output is None else spec.max_output
        handle: dict[str, SubprocessHandle] = {}

        async def enter() -> Disposer:
            process = await anyio.open_process(
                list(spec.argv),
                cwd=str(spec.cwd),
                env=env,
                stdout=_stdio(spec.stdio),
                stderr=_stdio(spec.stdio),
                stdin=None,
                # **The child gets its own process group, so that the whole
                # command can be stopped** (J4). `terminate` used to signal the
                # direct child. For `sh -c 'build | tee log'` that child is the
                # shell. The other members of the pipeline kept the CPU, kept
                # the pipes and outlived the timeout that was supposed to end
                # them. A child that leads its own group is what makes `killpg`
                # mean "this command".
                #
                # It also detaches the child from the controlling terminal, and
                # that is wanted. A Ctrl-C in the shell of the user is for pH.
                # A child that pH manages should be ended by the escalation in
                # `terminate`, and not by a signal that reaches it from the
                # side.
                start_new_session=True,
            )
            child = SubprocessHandle(spec=spec, process=process, cap=cap)
            handle["child"] = child
            # **The spawn is journaled here, between the spawn and the
            # disposer.** Every cleanup path of pH is structural: this effect,
            # the scope that owns it, and the reap in the `finally` of
            # `terminate`. None of them runs under `SIGKILL`. POSIX answers
            # `SIGKILL` by re-parenting the child to init, and does not kill
            # the child. A later start sweeps by this record.
            #
            # The write is inline and does not go through `to_thread`. Without
            # the `fsync` it is one buffered write (~20 µs), and the hop cost
            # more than the work.
            journal = self.journal
            if journal is not None:
                journal.record(pid=process.pid, argv=spec.argv, label=spec.argv[0])

            async def release() -> None:
                try:
                    await child.terminate()
                finally:
                    # This is in the `finally` for the same reason as the reap
                    # in `terminate`: it must run even when the stop fails.
                    # Otherwise the record of the child stays open, and the next
                    # sweep looks for something that is already gone.
                    if journal is not None:
                        journal.forget(process.pid)

            return release

        # The disposer is assigned after `effect` returns, because `enter` is
        # what puts the handle in the dict in the first place.
        release = await owner.effect(enter, label=f"subprocess({spec.argv[0]})")
        child = handle["child"]
        child.dispose = release
        return child

    async def run(
        self,
        spec: SubprocessSpawnSpec,
        *,
        scope: Context | None = None,
        signal: Cancellation | None = None,
    ) -> SubprocessResult:
        """Spawn, drain, and await: the common case as one call, with both bounds
        applied.

        `timeout_ms` is enforced here and not by each caller, for the same
        reason as `max_output`: most callers forget a bound that every caller
        has to remember. A child that outstays the timeout is terminated
        through the same path that a disposal uses, so the grace and the reap
        have one implementation. What the child managed to print is still
        returned, because a command that hung after it said something useful
        should not lose the useful part.

        **`signal` is the third bound, and the only one that a user can
        reach** (C7). A timeout is a number chosen in advance. Escape is a
        decision made while watching. Without `signal`, `bash sleep 3600` with
        no `timeout_ms` could not be interrupted at all. The cancel reached the
        tool pipeline and stopped there, because nothing between the pipeline
        and the child was watching. A canceled child is terminated by the same
        path as a timeout, so there is one description of how a child ends.
        """
        child = await self.spawn(spec, scope=scope)
        seconds = None if spec.timeout_ms is None else spec.timeout_ms / 1000
        canceled = False
        try:

            async def drain() -> int:
                await child.pump()
                return await child.wait()

            with anyio.move_on_after(seconds) as bound:
                # `raced` answers `None` for "the signal won", and `drain`
                # answers an exit code. So `None` here is unambiguous. A
                # timeout takes the other path: `move_on_after` cancels the
                # body, this line never finishes, and `cancelled_caught`
                # reports it.
                canceled = await raced(signal, drain) is None
        finally:
            # **The child is released, and not only reaped.** The reap is in a
            # `finally` on purpose (F4): an exception between spawn and wait
            # must not leave the child alive. The disposer does that reap. It
            # also closes the journal record and drops the effect. When this
            # only reaped, both stayed open until the owning scope unwound,
            # and for a daemon root that is hours. The journal grew a live
            # `spawn` line for each command that ran, and `_effects` grew an
            # entry for each command beside it.
            await maybe_await(child.dispose() if child.dispose is not None else None)
        # This reads `returncode` and not the value of `wait()`. Both paths out
        # of the block leave it set, and one read means that the timeout path
        # needs no second form. It is `-1` only when a child left no status at
        # all.
        return SubprocessResult(
            exit_code=-1 if child.returncode is None else child.returncode,
            stdout=capped_text(child.stdout, child.cap),
            stderr=capped_text(child.stderr, child.cap),
            dropped=child.dropped,
            timed_out=bound.cancelled_caught,
            canceled=canceled,
        )


async def _delivered(
    source: anyio.abc.ByteReceiveStream, listener: Callable[[str], None], cap: int
) -> int:
    """Hand `source` to `listener` one line at a time until the child closes it.
    Returns how many bytes were dropped. Only a line longer than `cap` loses
    bytes. It is cut at `cap`, so that one line cannot fill the memory of this
    process."""

    def deliver(line: bytes) -> None:
        try:
            listener(capped_text(line, cap))
        except Exception:
            log.warning("ph.seams.subprocess: a stderr listener failed", exc_info=True)

    pending = bytearray()
    dropped = 0
    async for chunk in source:
        *lines, rest = chunk.split(b"\n")
        for line in lines:
            deliver(bytes(pending) + line)
            pending.clear()
        kept = rest[: max(cap - len(pending), 0)]
        pending.extend(kept)
        dropped += len(rest) - len(kept)
    if pending:
        # A last line that the child did not end: the last words of a crash.
        deliver(bytes(pending))
    return dropped


def _stdio(mode: Stdio) -> int | None:
    if mode == "pipe":
        return _sp.PIPE
    if mode == "null":
        return _sp.DEVNULL
    return None


@plugin("subprocess-local", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local subprocess provider, and sweep what a hard kill left (F5).

    **The sweep runs at every start**, which is the rule of the journal itself.
    A session that nobody reopens would never reconcile its strays, and a child
    that was re-parented to init is using CPU now and is not waiting to be
    noticed. The sweep is safe to run from any pH process, because the journal
    records the *owner* of each child. The children of a live owner are held
    and never killed.

    The sweep runs off the loop, because it checks `/proc` for each record and
    may send `SIGKILL`. It runs after `provide`, so a deployment whose sweep
    fails still has a working seam.
    """
    journal = host_journal()
    ctx.provide(
        SUBPROCESS,
        SubprocessService(
            ctx=ctx,
            journal=journal,
            max_output=config.max_output_bytes,
            scrub=EnvScrub(marks=tuple(config.scrub), keep=frozenset(config.keep)),
        ),
    )
    contribute(
        ctx,
        Diagnostic(
            id="subprocess",
            title="Child processes",
            read=lambda: _describe(journal),
            order=21,
        ),
    )
    if journal is not None and config.sweep_orphans:
        # `sweep` already logs what it did, under the same condition. A second
        # sentence here for the same event is what `ph.text` exists to prevent.
        await anyio.to_thread.run_sync(journal.sweep)


def _describe(journal: OrphanJournal | None) -> list[tuple[str, str]]:
    if journal is None:
        return [("orphan journal", "unavailable — a hard kill will leave strays unswept")]
    return [("orphan journal", str(journal.path))]


def platform_shell() -> Sequence[str]:
    """The shell argv prefix for this platform."""
    if sys.platform == "win32":
        return ("cmd.exe", "/c")
    return ("bash", "-lc")
