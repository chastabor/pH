"""`ctx.subprocess` — spawning, with nothing implicit.

Two properties matter more than convenience here.

**The spec is fully explicit.** `SubprocessSpawnSpec` names argv, cwd, stdio and
the termination grace. No hidden defaults: a caller that did not think about the
working directory of a process the *model* asked for should have to say so, and
a default inherited from the harness process is exactly the wrong answer once
`ctx.workspace` starts handing out per-agent roots (Phase 4).

**The environment is scrubbed.** A child runs code the model wrote, so it does
not inherit `*KEY*`, `*SECRET*`, `*TOKEN*` or `*PASSWORD*` (I-4). Credentials
reach an adapter as a `CredentialRef` and are resolved at the edge (I-3); a
child never needs the value, and the one that does is asking for it.

Readers are **offset-based** rather than streams: a tool that reads a long
build log needs to say "give me from byte N", and an oversized buffer spills to
`ctx.spill_store` instead of into the model's context.

**Both bounds live here, and that is the point of P7-13.** A child is somebody
else's program: it decides how much it prints and how long it runs, and every
caller that trusted it was one runaway command away from taking the process down
with it. `max_output` bounds the memory and `timeout_ms` bounds the clock, in the
seam, so `!!`, `tool-bash` and every future caller inherit them instead of each
remembering. What a caller then clips for a log or a card is a *display* choice
on top of a bound that already held.

@module ph.seams.subprocess
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import anyio
import anyio.abc

from ..cordis import Context, plugin
from ..wire import WireModel

__all__ = [
    "MAX_OUTPUT",
    "SECRET_PATTERN",
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
"""How much of each stream a child's output is *kept*, per spawn (P7-13).

Generous, because this is not a display limit: `ph_app.shell` clips 64 KiB into
the log and `tool-result-offload` spills at 80 000 characters, and both of those
run on a string that already exists. This is the ceiling that stops the string
existing — a `find /` or a build log on a bad day is tens of megabytes, and the
drain used to hold all of it and then decode a second copy to `str`.

**Per stream, so the real ceiling is twice this**: the `bytearray` and then the
decoded `str`, for each of two streams. Per stream rather than a shared budget
because a chatty stdout must not starve the stderr that says why the command
failed — which is the half a reader needs most.

The *default*, not the only value: `subprocess-local` takes a `max_output_bytes`
so an operator whose gate is a 40 MiB build log has somewhere to say so. What
there is no spelling for is *unbounded* — a caller who forgets is exactly the
caller this protects.
"""

SECRET_PATTERN = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
"""Environment names a model-run child never inherits.

Substring matching, deliberately broad: `MY_API_KEY_2` and `GH_TOKEN_FILE` are
both worth losing, and a child that genuinely needs a secret should be given it
explicitly rather than by inheritance."""


def first_line(text: str) -> str:
    """The first non-blank line of a tool's output, or `""`.

    Here rather than in each seam that reads a subprocess's words, because two
    of them had it and a third was about to: a decline that quotes the backend
    is the difference between "the sandbox refused a write" and "setting up uid
    map: Permission denied", and only the second tells an operator what to do.
    """
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def scrub_env(
    base: Mapping[str, str] | None = None, *, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The parent environment minus anything that looks like a credential."""
    source = os.environ if base is None else base
    scrubbed = {key: value for key, value in source.items() if not SECRET_PATTERN.search(key)}
    if extra:
        scrubbed.update(extra)
    return scrubbed


@dataclass(frozen=True, slots=True)
class SubprocessSpawnSpec:
    """Everything about one spawn, stated."""

    argv: tuple[str, ...]
    cwd: Path
    stdio: Stdio = "pipe"
    grace_ms: int = 5_000
    """How long a terminated child may take to exit before it is killed.

    The *termination* grace, not a run limit — `timeout_ms` is that. They were
    conflated once: `ctx.shell` passed a caller's `timeout_ms` here, so a tool
    whose description promised "kill the command after this long" only widened
    the window a already-dying child had to finish (P7-13)."""
    timeout_ms: int | None = None
    """How long the child may run before `run()` terminates it. `None` waits.

    On the spec beside `grace_ms` because the two are one policy about a child's
    lifetime, and a caller reading this class should see both."""
    env: dict[str, str] | None = None
    """`None` scrubs and inherits; a dict replaces the environment wholesale
    (so `{}` is "nothing at all")."""
    max_output: int | None = None
    """How much of each stream to keep, or `None` for the deployment's own.

    See `MAX_OUTPUT` — a spec that says nothing still caps, which is the point."""


@dataclass(slots=True)
class SubprocessHandle:
    """A live child, with offset-addressable output."""

    spec: SubprocessSpawnSpec
    process: Any
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)
    cap: int = MAX_OUTPUT
    """How much of each stream this child's drain keeps. Resolved by `spawn`
    from the spec and the row's config, so `pump` reads one number."""
    dropped: int = 0
    """Bytes this child printed past `cap` and nobody kept, across both streams.

    Counted rather than flagged: "output was truncated" is a worse thing to tell
    a model than "37 MB was dropped", and a caller deciding whether to re-run the
    command with a filter wants the size. Summed, because every reader of it —
    the shell seam, `!!`, `tool-bash` — asks only how much went missing."""

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)

    @property
    def returncode(self) -> int | None:
        code: int | None = self.process.returncode
        return code

    def read_from(self, offset: int, *, stream: Stream = "stdout") -> bytes:
        """Bytes from `offset` on — the shape a tool polling a long run needs."""
        buffer = self.stdout if stream == "stdout" else self.stderr
        return bytes(buffer[offset:])

    async def pump(self) -> None:
        """Drain both pipes until the child closes them, keeping at most `max_output`.

        **Reading never stops; only keeping does.** A drain that stopped reading
        at the cap would leave the child blocked on a full pipe — forever, since
        nothing else is going to empty it — and a bound that deadlocks the thing
        it was protecting is worse than no bound. So every chunk is read and the
        overflow is counted instead of stored.

        The *head* is what survives. A command that fails usually says why in its
        first lines, and a reader with an offset (`read_from`) is reading forward
        from the start; keeping the tail would serve a progress bar and lose the
        error.
        """
        cap = self.cap

        async def drain(stream: Stream) -> None:
            source = self.process.stdout if stream == "stdout" else self.process.stderr
            if source is None:
                return
            sink = self.stdout if stream == "stdout" else self.stderr
            dropped = 0
            try:
                async for chunk in source:
                    room = cap - len(sink)
                    if len(chunk) <= room:
                        # The whole chunk fits: extend it directly rather than
                        # slicing, which is the case every ordinary command takes.
                        sink.extend(chunk)
                        continue
                    if room > 0:
                        sink.extend(chunk[:room])
                    dropped += len(chunk) - max(room, 0)
            finally:
                # Once, not per chunk, and in a `finally` so a cancelled drain
                # still reports what it had already thrown away.
                self.dropped += dropped

        async with anyio.create_task_group() as scope:
            scope.start_soon(drain, "stdout")
            scope.start_soon(drain, "stderr")

    async def wait(self) -> int:
        """Await exit and reap. Idempotent."""
        return int(await self.process.wait())

    async def terminate(self) -> None:
        """Ask, wait out the grace, then insist — and always reap.

        The `wait()` in the finally is what keeps a zombie from accumulating
        (F4): a child that exited but was never awaited stays in the table for
        as long as the parent lives.
        """
        try:
            if self.process.returncode is None:
                self.process.terminate()
                with anyio.move_on_after(self.spec.grace_ms / 1000):
                    await self.process.wait()
            if self.process.returncode is None:
                self.process.kill()
        except ProcessLookupError:  # pragma: no cover - already gone
            pass
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    # `aclose`, not `wait`: it closes the child's pipes *and*
                    # reaps. A timeout cancels `pump` mid-read, so the drains let
                    # go of streams nobody then closed — and the transport was
                    # finalised after the loop had gone, which asyncio reports as
                    # an unraisable `Event loop is closed` from a `__del__` in
                    # whatever test ran next.
                    await self.process.aclose()
                except Exception:
                    log.debug("ph.seams.subprocess: reaping pid %s failed", self.pid)


@dataclass(frozen=True, slots=True)
class SubprocessResult:
    """What one finished child produced, and what was lost getting it.

    A record rather than the `(code, out, err)` tuple this used to be: with a cap
    and a clock in the seam there are now two ways a result can be *incomplete*,
    and a tuple has nowhere to say so — which would leave every caller rendering
    a truncated log as if it were the whole thing.
    """

    exit_code: int
    stdout: str
    stderr: str
    dropped: int = 0
    """Bytes past the cap that nobody kept, across both streams."""
    timed_out: bool = False

    @property
    def truncated(self) -> bool:
        """Whether what this holds is a prefix of what the child printed."""
        return self.dropped > 0


class Config(WireModel):
    """Row config for `subprocess-local`."""

    max_output_bytes: int = MAX_OUTPUT
    """How much of each stream a child's output is kept. See `MAX_OUTPUT`.

    Config rather than a constant, for the reason `fs-local`'s `ignore` is
    (P6-19): the value has a wrong answer for somebody. `ctx.shell` takes no
    `max_output`, so without this the 8 MiB default is the only ceiling `!!`,
    `tool-bash` and an autonomous gate can ever run under — and an operator whose
    gate is a 40 MiB build log would have to edit source."""


@dataclass(slots=True)
class SubprocessService:
    """The service published as `ctx.subprocess`."""

    ctx: Context
    max_output: int = MAX_OUTPUT
    """The deployment's ceiling, applied to any spec that names none."""

    async def spawn(
        self, spec: SubprocessSpawnSpec, *, scope: Context | None = None
    ) -> SubprocessHandle:
        """Spawn a child owned by `scope`, terminated and reaped on disposal."""
        owner = self.ctx.owner_for(scope)
        env = scrub_env() if spec.env is None else spec.env
        cap = self.max_output if spec.max_output is None else spec.max_output
        handle: dict[str, SubprocessHandle] = {}

        async def enter() -> Any:
            process = await anyio.open_process(
                list(spec.argv),
                cwd=str(spec.cwd),
                env=env,
                stdout=_stdio(spec.stdio),
                stderr=_stdio(spec.stdio),
                stdin=None,
            )
            child = SubprocessHandle(spec=spec, process=process, cap=cap)
            handle["child"] = child

            def release() -> Any:
                return child.terminate()

            return release

        await owner.effect(enter, label=f"subprocess({spec.argv[0]})")
        return handle["child"]

    async def run(
        self, spec: SubprocessSpawnSpec, *, scope: Context | None = None
    ) -> SubprocessResult:
        """Spawn, drain, and await — the common case as one call, both bounds applied.

        `timeout_ms` is enforced here rather than by each caller, for the reason
        `max_output` is: a bound every caller has to remember is a bound most of
        them will not. A child that outstays it is terminated through the same
        path a disposal uses, so the grace and the reap are not a second story —
        and what it managed to print is still returned, because a command that
        hung after saying something useful should not lose the useful part.
        """
        child = await self.spawn(spec, scope=scope)
        seconds = None if spec.timeout_ms is None else spec.timeout_ms / 1000
        try:
            with anyio.move_on_after(seconds) as bound:
                await child.pump()
                await child.wait()
        finally:
            # Reaping in a finally is the whole point: an exception between
            # spawn and wait must not leave the child unreaped (F4).
            if child.returncode is None:
                await child.terminate()
        # `returncode` rather than `wait()`'s value: both paths out of the block
        # leave it set, and reading it once means the timeout path has no second
        # spelling. `-1` only when a child left no status at all.
        return SubprocessResult(
            exit_code=-1 if child.returncode is None else child.returncode,
            stdout=child.stdout.decode("utf-8", errors="replace"),
            stderr=child.stderr.decode("utf-8", errors="replace"),
            dropped=child.dropped,
            timed_out=bound.cancelled_caught,
        )


def _stdio(mode: Stdio) -> Any:
    import subprocess as _sp

    if mode == "pipe":
        return _sp.PIPE
    if mode == "null":
        return _sp.DEVNULL
    return None


@plugin("subprocess-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local subprocess provider."""
    ctx.provide("subprocess", SubprocessService(ctx=ctx, max_output=config.max_output_bytes))


def platform_shell() -> Sequence[str]:
    """The shell argv prefix for this platform."""
    if sys.platform == "win32":
        return ("cmd.exe", "/c")
    return ("bash", "-lc")
