"""`ctx.fs` — filesystem access with an interception point before every mutation.

Every write and every edit passes through a waterfall (`fs/write-intent`,
`fs/edit-intent`) *before* it touches the disk. That ordering is the whole
value: a policy plugin that ran after the write would be a reporter, not a
gate — which is precisely the failure the feature map records in prime-agent's
`edit` skill, where the diff is emitted **after** the file changed, so "there is
no point at which anything can say no".

`fs/observed` records reads, which is what lets read-before-edit be a policy
row rather than a hard-coded rule.

The honest scope: this bounds *tool-mediated* access. Model-authored
`open(path, "w")` inside a code cell is unreachable from here by construction
(N1) — a deny-list needs a registered name. That is what the containment ladder
is for, and no wording in this module should suggest otherwise.

@module ph.seams.fs
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, events, plugin
from ..session import Session
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel

__all__ = [
    "EditIntent",
    "FileSlice",
    "FsService",
    "GrepMatch",
    "WriteIntent",
    "apply",
    "read_before_edit",
]

log = logging.getLogger("ph.seams.fs")

events.declare(
    "fs/write-intent",
    "waterfall",
    owner="ph.seams.fs",
    doc="Before a whole-file write reaches disk. A listener that vetoes prevents it.",
)
events.declare(
    "fs/edit-intent",
    "waterfall",
    owner="ph.seams.fs",
    doc="Before an in-place edit reaches disk. A listener that vetoes prevents it.",
)
events.declare(
    "fs/changed",
    "emit",
    owner="ph.seams.fs",
    doc="A path was written or edited; consumers refresh their view.",
)


class FileSlice(WireModel):
    """A window onto a file, with enough context to ask for the next one."""

    path: str
    text: str
    offset: int = 0
    """First line index included (0-based)."""
    lines: int = 0
    total_lines: int = 0
    truncated: bool = False


class GrepMatch(WireModel):
    path: str
    line: int
    text: str


@dataclass(frozen=True, slots=True)
class WriteIntent:
    """A whole-file write awaiting policy."""

    path: Path
    content: str
    creating: bool
    agent: Any = None


@dataclass(frozen=True, slots=True)
class EditIntent:
    """An in-place replacement awaiting policy."""

    path: Path
    old_text: str
    new_text: str
    replace_all: bool
    agent: Any = None


class FsDenied(HarnessError, PermissionError):
    """A write or edit was refused by policy.

    A `HarnessError` as well as a `PermissionError`, so the refusal survives the
    trip to a consumer as a *denial*. Without that it reached `ToolFailure.kind`
    as an ordinary `failed`, which is the same hole C3 closes for
    `tools/pre-execute`: a Code Mode program could `except` a policy veto from
    this seam and route around it, and the durable record said the tool merely
    failed. `SandboxError` next door already had this; this seam, whose whole
    argument is being a gate rather than a report, did not.

    Still a `PermissionError`, so every existing catcher keeps working.
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, message: str) -> None:
        super().__init__(message, "FS_DENIED")


@dataclass(slots=True)
class FsService:
    """The service published as `ctx.fs`."""

    ctx: Context
    root: Path
    _observed: dict[Path, float] = field(default_factory=dict)
    """Last-read mtime per path — the state read-before-edit consults."""

    def resolve(self, path: str | Path) -> Path:
        """Resolve against the workspace root.

        A relative path is the agent's business; an absolute one is passed
        through, because refusing it here would be a confinement claim this
        layer cannot make (N2).
        """
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate)

    # ------------------------------------------------------------------ read --

    async def read(
        self,
        path: str | Path,
        *,
        offset: int = 0,
        limit: int | None = 2_000,
        session: Session | None = None,
    ) -> FileSlice:
        """Read a line window and record the observation."""
        target = self.resolve(path)
        text = await anyio.to_thread.run_sync(
            lambda: target.read_text(encoding="utf-8", errors="replace")
        )
        all_lines = text.splitlines()
        window = all_lines[offset:] if limit is None else all_lines[offset : offset + limit]
        self._observe(target, session)
        return FileSlice(
            path=str(target),
            text="\n".join(window),
            offset=offset,
            lines=len(window),
            total_lines=len(all_lines),
            truncated=limit is not None and offset + len(window) < len(all_lines),
        )

    def _observe(self, target: Path, session: Session | None) -> None:
        try:
            self._observed[target] = target.stat().st_mtime
        except OSError:  # pragma: no cover - raced deletion
            return
        if session is not None:
            session.append("fs/observed", {"path": str(target)})

    def observed_mtime(self, path: str | Path) -> float | None:
        return self._observed.get(self.resolve(path))

    # ----------------------------------------------------------------- write --

    async def write(
        self, path: str | Path, content: str, *, agent: Any = None, session: Session | None = None
    ) -> Path:
        """Write a whole file, after `fs/write-intent` allows it."""
        target = self.resolve(path)
        intent = WriteIntent(
            path=target, content=content, creating=not target.exists(), agent=agent
        )
        await self._gate("fs/write-intent", intent)
        await anyio.to_thread.run_sync(_write_text, target, content)
        self._observe(target, session)
        self.ctx.emit("fs/changed", target, contained=True)
        return target

    async def edit(
        self,
        path: str | Path,
        old_text: str,
        new_text: str,
        *,
        replace_all: bool = False,
        agent: Any = None,
        session: Session | None = None,
    ) -> int:
        """Replace text in place, after `fs/edit-intent` allows it.

        Returns the replacement count. A unique-match requirement is the tool's
        to enforce; this layer reports what it did.
        """
        target = self.resolve(path)
        intent = EditIntent(
            path=target,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            agent=agent,
        )
        await self._gate("fs/edit-intent", intent)
        original = await anyio.to_thread.run_sync(lambda: target.read_text(encoding="utf-8"))
        count = original.count(old_text)
        if count == 0:
            raise ValueError(f"no occurrence of the target text in {target}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"{count} occurrences of the target text in {target}; pass replace_all "
                "or include more surrounding context to make it unique"
            )
        updated = original.replace(old_text, new_text, -1 if replace_all else 1)
        await anyio.to_thread.run_sync(_write_text, target, updated)
        self._observe(target, session)
        self.ctx.emit("fs/changed", target, contained=True)
        return count if replace_all else 1

    async def _gate(self, event: str, intent: Any) -> None:
        async def inner(_intent: Any) -> str | None:
            return None

        reason = await self.ctx.waterfall(event, intent, inner=inner)
        if reason is not None:
            raise FsDenied(str(reason))

    # ------------------------------------------------------------------ find --

    async def glob(
        self, pattern: str, *, root: str | Path | None = None, limit: int = 1_000
    ) -> list[str]:
        base = self.resolve(root) if root is not None else self.root
        return await anyio.to_thread.run_sync(
            lambda: [str(path) for path in _walk(base, pattern, limit)]
        )

    async def grep(
        self,
        pattern: str,
        *,
        root: str | Path | None = None,
        glob: str = "**/*",
        limit: int = 200,
    ) -> list[GrepMatch]:
        base = self.resolve(root) if root is not None else self.root
        try:
            expression = re.compile(pattern)
        except re.error as error:
            raise ValueError(f"invalid regular expression: {error}") from error

        def scan() -> list[GrepMatch]:
            matches: list[GrepMatch] = []
            for candidate in _walk(base, glob, None):
                if not _greppable(candidate):
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                for number, line in enumerate(text.splitlines(), start=1):
                    if expression.search(line):
                        matches.append(GrepMatch(path=str(candidate), line=number, text=line[:500]))
                        if len(matches) >= limit:
                            return matches
            return matches

        return await anyio.to_thread.run_sync(scan)


_IGNORED_PARTS = frozenset(
    {".git", "node_modules", "__pycache__", ".venv", ".mypy_cache", ".ruff_cache", "dist"}
)

GREP_MAX_BYTES = 2 * 1024 * 1024
"""Files above this are skipped by `grep`: a build artifact that happens to
match `**/*` must not be read whole into memory line by line."""


def _walk(base: Path, pattern: str, limit: int | None) -> Iterator[Path]:
    """Matching files under `base`, pruning ignored directories *before* descent.

    `Path.glob` would materialize `node_modules` in full and then let `_ignored`
    discard it; pruning `dirs` in place means an ignored tree is never entered.
    Directories and files are visited in sorted order so results are stable.
    """
    yielded = 0
    for current, dirs, files in os.walk(base):
        dirs[:] = sorted(name for name in dirs if name not in _IGNORED_PARTS)
        directory = Path(current)
        for name in sorted(files):
            path = directory / name
            relative = path.relative_to(base).as_posix()
            if not _matches(relative, pattern):
                continue
            yield path
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _matches(relative: str, pattern: str) -> bool:
    """Glob semantics close to `Path.glob`: `**/` matches any depth, including none."""
    if pattern.startswith("**/"):
        tail = pattern[3:]
        return bool(
            fnmatch.fnmatchcase(relative, tail) or fnmatch.fnmatchcase(relative, f"*/{tail}")
        )
    if "**" in pattern:
        return bool(fnmatch.fnmatchcase(relative, pattern.replace("**", "*")))
    return bool(fnmatch.fnmatchcase(relative, pattern))


def _greppable(path: Path) -> bool:
    """Text, and small enough to read whole."""
    try:
        if path.stat().st_size > GREP_MAX_BYTES:
            return False
        with path.open("rb") as handle:
            return b"\0" not in handle.read(8192)
    except OSError:
        return False


def _write_text(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


class Config(WireModel):
    """Row config for the local filesystem provider."""

    root: str | None = None


@plugin("fs-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local filesystem provider."""
    root = Path(config.root).expanduser() if config.root else Path.cwd()
    ctx.provide("fs", FsService(ctx=ctx, root=root))


@plugin("fs-read-before-edit", inject=["fs"])
async def read_before_edit(ctx: Context, config: Any) -> None:
    """Refuse an edit to a file this session has not read since it last changed.

    Its own row, not a rule inside `edit`, for two reasons: a deployment can
    drop it, and it applies to *every* editing tool rather than the one that
    remembered to check.
    """
    fs: FsService = ctx.fs

    async def gate(intent: EditIntent, next_: Callable[[], Any]) -> Any:
        observed = fs.observed_mtime(intent.path)
        if observed is None:
            return (
                f"read {intent.path} before editing it, so the edit is based on its "
                "current contents"
            )
        try:
            current = intent.path.stat().st_mtime
        except OSError:
            return f"{intent.path} no longer exists"
        if current > observed:
            return f"{intent.path} changed on disk since it was read; read it again before editing"
        return await next_()

    ctx.on("fs/edit-intent", gate)
