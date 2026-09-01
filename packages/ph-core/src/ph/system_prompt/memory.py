"""`memory-agents-md` — what the agent was told to remember, placed after the cache (G8).

The content is what `agent-instructions` found in Phase 1: `AGENTS.md` walking up
from the agent's root, nearest first, then the user's `$PH_HOME/AGENTS.md`. What
this row changes is **where it lands**, and that is two properties rather than one.

**It is a `context()`, not a `section()`.** A section is part of the cached prefix,
so an edit to `AGENTS.md` would change the prefix and invalidate every cached
token before it — on a file whose whole purpose is to be edited. A `context()`
materializes as a durable snapshot *after* retained history and only when its text
changed, which is exactly the shape a mutable instruction file needs.

**The same move makes memory live.** A provider is asked on every assembly, so
what the user wrote a minute ago is in the next turn — where a row that read the
files once at mount did nothing at all until the process restarted. Cheap and live
turn out to be the same change; the cache is the reason it is *safe*, not the
reason it is worth doing.

**Per agent, not per process.** The provider is handed the request, so the root is
`ctx.fs.root_for(agent)`: a child working in a worktree (D21) reads the `AGENTS.md`
of the tree it is actually in. A root read once at mount would have handed every
child the parent's instructions and been wrong in exactly the deployment the tier
exists for.

**Locating and reading are two passes, in that order.** `assemble` runs once per
model *step*, so "memory is live" must not mean "every step reads 64 KiB from
disk": re-reading is a `stat` per candidate directory per turn, and the bytes are
read only when a file's mtime or size moved.

@module ph.system_prompt.memory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from typing import Any, TypeAlias

from ..cordis import Context, plugin
from ..paths import resolve_roots
from .assembly import AssembleContext, PromptContext

__all__ = ["FILENAME", "MAX_BYTES", "MemoryFiles", "apply", "discover", "locate", "render"]

log = logging.getLogger("ph.system_prompt.memory")

Signature: TypeAlias = "tuple[tuple[str, tuple[int, int]], ...]"
"""What the files looked like: path, mtime and size, in discovery order."""

CACHE_MAX = 32
"""How many agent roots to remember rendered text for. See `MemoryFiles._cache`."""

FILENAME = "AGENTS.md"
MAX_BYTES = 64 * 1024
"""Per file. A memory file larger than this is being used as a corpus, and a
corpus belongs behind a retrieval seam rather than in every request."""

ORDER_MEMORY = -10
"""Ahead of the session facts and the harness note, because instructions the
user wrote outrank anything the harness has to say about its own state."""


@dataclass(frozen=True, slots=True)
class DiscoveredInstructions:
    path: Path
    text: str
    scope: str
    """`"user"` for `$PH_HOME`, `"project"` for anything under the workspace."""


def locate(root: Path, *, home: Path) -> list[tuple[Path, str, tuple[int, int]]]:
    """Where the instruction files are and what state they are in — no reads.

    Nearest-first because a subdirectory's instructions are more specific than
    the repository's, and the model reads a prompt top to bottom. Every level is
    kept rather than the nearest one winning: `AGENTS.md` is additive by
    convention, and a repository root's rules still apply inside a package.

    Returns the `stat` facts alongside each path so a caller can decide whether
    anything changed before paying for the content. One `stat` per candidate
    directory, which is what the walk costs anyway — `is_file()` is a `stat`.
    """
    found: list[tuple[Path, str, tuple[int, int]]] = []
    seen: set[Path] = set()
    for directory in [root, *root.parents, home]:
        candidate = directory / FILENAME
        if candidate in seen:
            continue
        seen.add(candidate)
        state = _state(candidate)
        if state is not None:
            # `home` is last, and labelled for what it is: the same file reached
            # both ways is one instruction, not two.
            found.append((candidate, "user" if directory == home else "project", state))
    return found


def _state(path: Path) -> tuple[int, int] | None:
    """`(mtime, size)` for a regular file, or `None` for anything else."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size) if S_ISREG(stat.st_mode) else None


def discover(root: Path, *, home: Path | None = None) -> list[DiscoveredInstructions]:
    """Find instruction files, nearest first, user-level last, and read them."""
    user = home if home is not None else resolve_roots().home
    return [_read(path, scope) for path, scope, _state in locate(root, home=user)]


def _read(path: Path, scope: str) -> DiscoveredInstructions:
    """One file, capped at the read rather than after it.

    `MAX_BYTES` bounds what a *request* can carry, so reading 5 MiB in order to
    keep the first 64 KiB of it is the cost this cap exists to avoid — the same
    `handle.read(limit)` the skills reader uses on `SKILL.md`.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            text = handle.read(MAX_BYTES)
    except OSError as error:
        log.warning("ph.system_prompt.memory: %s could not be read: %s", path, error)
        text = ""
    return DiscoveredInstructions(path=path, text=text, scope=scope)


def render(found: list[DiscoveredInstructions]) -> str:
    """The snapshot text, or `""` when there is nothing to say.

    Empty is how a `context()` opts out for this assembly, so a deployment with
    no `AGENTS.md` anywhere contributes no snapshot at all rather than a heading
    with nothing under it.
    """
    if not found:
        return ""
    body = "\n\n".join(f"<!-- {item.scope}: {item.path} -->\n{item.text.strip()}" for item in found)
    return (
        "The following instructions come from AGENTS.md files in this workspace "
        "and from the user's own configuration. Follow them.\n\n"
        f"{body}"
    )


@dataclass(slots=True)
class MemoryFiles:
    """The provider, holding only what it reads.

    A class rather than a closure because it outlives the `apply` that built it
    and would otherwise keep that whole frame alive — and because the cache
    needs somewhere to live that is not a mutable default.
    """

    ctx: Context
    home: Path
    _cache: dict[Path, tuple[Signature, str]] = field(default_factory=dict)
    """Rendered text per agent root, keyed by what the files looked like.

    Bounded, because the key is a *workspace root* and a long-running daemon
    fans out one per child: unbounded, it would retain a rendered snapshot for
    every worktree the process ever saw. Dropping the whole map on overflow
    rather than evicting one entry — the next assembly re-reads and the cost is
    a few hundred microseconds, which is not worth an LRU.
    """

    def root(self, agent: Any) -> Path:
        fs = self.ctx.get("fs")
        if fs is None:
            return Path.cwd()
        root: Path = fs.root_for(agent)
        return root

    def text(self, request: AssembleContext) -> str:
        """This agent's memory, reading only what changed since last time."""
        root = self.root(request.agent)
        located = locate(root, home=self.home)
        signature = tuple((str(path), state) for path, _scope, state in located)
        cached = self._cache.get(root)
        if cached is not None and cached[0] == signature:
            return cached[1]
        rendered = render([_read(path, scope) for path, scope, _state in located])
        if len(self._cache) >= CACHE_MAX:
            self._cache.clear()
        self._cache[root] = (signature, rendered)
        return rendered


@plugin("memory-agents-md", inject=["system_prompt"])
async def apply(ctx: Context, config: Any) -> None:
    """Contribute discovered `AGENTS.md` files as a post-cache snapshot."""
    # Resolved once: `$PH_HOME` is a process constant, and asking for it per
    # assembly costs a `stat` or two on a path that cannot have moved.
    memory = MemoryFiles(ctx=ctx, home=resolve_roots().home)
    ctx.system_prompt.context(
        PromptContext(name="memory", order=ORDER_MEMORY, text=memory.text), scope=ctx
    )
