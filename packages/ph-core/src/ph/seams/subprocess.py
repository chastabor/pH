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

@module ph.seams.subprocess
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import anyio
import anyio.abc

from ..cordis import Context, plugin

__all__ = [
    "SECRET_PATTERN",
    "SubprocessHandle",
    "SubprocessService",
    "SubprocessSpawnSpec",
    "apply",
    "scrub_env",
]

log = logging.getLogger("ph.seams.subprocess")

Stdio: TypeAlias = Literal["pipe", "inherit", "null"]

SECRET_PATTERN = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
"""Environment names a model-run child never inherits.

Substring matching, deliberately broad: `MY_API_KEY_2` and `GH_TOKEN_FILE` are
both worth losing, and a child that genuinely needs a secret should be given it
explicitly rather than by inheritance."""


def scrub_env(
    base: dict[str, str] | None = None, *, extra: dict[str, str] | None = None
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
    """How long a terminated child may take to exit before it is killed."""
    env: dict[str, str] | None = None
    """`None` scrubs and inherits; a dict replaces the environment wholesale
    (so `{}` is "nothing at all")."""


@dataclass(slots=True)
class SubprocessHandle:
    """A live child, with offset-addressable output."""

    spec: SubprocessSpawnSpec
    process: Any
    stdout: bytearray = field(default_factory=bytearray)
    stderr: bytearray = field(default_factory=bytearray)

    @property
    def pid(self) -> int | None:
        return getattr(self.process, "pid", None)

    @property
    def returncode(self) -> int | None:
        code: int | None = self.process.returncode
        return code

    def read_from(self, offset: int, *, stream: Literal["stdout", "stderr"] = "stdout") -> bytes:
        """Bytes from `offset` on — the shape a tool polling a long run needs."""
        buffer = self.stdout if stream == "stdout" else self.stderr
        return bytes(buffer[offset:])

    async def pump(self) -> None:
        """Drain both pipes until the child closes them."""

        async def drain(source: Any, sink: bytearray) -> None:
            if source is None:
                return
            async for chunk in source:
                sink.extend(chunk)

        async with anyio.create_task_group() as scope:
            scope.start_soon(drain, self.process.stdout, self.stdout)
            scope.start_soon(drain, self.process.stderr, self.stderr)

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
                    await self.process.wait()
                except Exception:
                    log.debug("ph.seams.subprocess: reaping pid %s failed", self.pid)


@dataclass(slots=True)
class SubprocessService:
    """The service published as `ctx.subprocess`."""

    ctx: Context

    async def spawn(
        self, spec: SubprocessSpawnSpec, *, scope: Context | None = None
    ) -> SubprocessHandle:
        """Spawn a child owned by `scope`, terminated and reaped on disposal."""
        owner = scope or self.ctx
        env = scrub_env() if spec.env is None else spec.env
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
            child = SubprocessHandle(spec=spec, process=process)
            handle["child"] = child

            def release() -> Any:
                return child.terminate()

            return release

        await owner.effect(enter, label=f"subprocess({spec.argv[0]})")
        return handle["child"]

    async def run(
        self, spec: SubprocessSpawnSpec, *, scope: Context | None = None
    ) -> tuple[int, str, str]:
        """Spawn, drain, and await — the common case as one call."""
        child = await self.spawn(spec, scope=scope)
        try:
            await child.pump()
            code = await child.wait()
        finally:
            # Reaping in a finally is the whole point: an exception between
            # spawn and wait must not leave the child unreaped (F4).
            if child.returncode is None:
                await child.terminate()
        return (
            code,
            child.stdout.decode("utf-8", errors="replace"),
            child.stderr.decode("utf-8", errors="replace"),
        )


def _stdio(mode: Stdio) -> Any:
    import subprocess as _sp

    if mode == "pipe":
        return _sp.PIPE
    if mode == "null":
        return _sp.DEVNULL
    return None


@plugin("subprocess-local")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the local subprocess provider."""
    ctx.provide("subprocess", SubprocessService(ctx=ctx))


def platform_shell() -> Sequence[str]:
    """The shell argv prefix for this platform."""
    if sys.platform == "win32":
        return ("cmd.exe", "/c")
    return ("bash", "-lc")
