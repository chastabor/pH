"""`memory-agents-md`: what the agent was told to remember, placed after the cache (G8).

The content is the `AGENTS.md` files that the earlier `agent-instructions` row
found: a walk up from the root of the agent, nearest first, and then the
`$PH_HOME/AGENTS.md` of the user. This row changes **where the content lands**,
and that is two properties and not one.

**It is a `context()`, not a `section()`.** A section is part of the cached
prefix. An edit to `AGENTS.md` would change the prefix and invalidate every
cached token before it, on a file whose purpose is to be edited. A `context()`
materializes as a durable snapshot *after* retained history, and only when its
text changed. That is the shape that a mutable instruction file needs.

**The same change makes memory live.** A provider is asked on every assembly, so
what the user wrote a minute ago is in the next turn. A row that read the files
one time at mount did nothing until the process restarted. Cheap and live are
the same change. The cache is the reason that the change is *safe*, not the
reason that it is worth doing.

**For each agent, not for each process.** The provider is given the request, so
the root is `ctx.fs.root_for(agent)`. A child that works in a worktree (D21)
reads the `AGENTS.md` of the tree that it is in. A root that was read one time
at mount would have given every child the instructions of the parent. It would
have been wrong in exactly the deployment that the tier exists for.

**Locating and reading are two passes, in that order.** `assemble` runs one time
for each model *step*, so "memory is live" must not mean "every step reads
64 KiB from disk". A repeat read is a `stat` for each candidate directory on
each assembly, and the bytes are read only when the mtime or the size of a file
moved.

@module ph.system_prompt.memory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from typing import TypeAlias

from ..agent.types import AgentHandle
from ..characters import head_bytes
from ..cordis import Context, plugin
from ..keys import FS, SYSTEM_PROMPT
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
"""The limit for each file. A memory file that is larger than this is used as a
corpus, and a corpus belongs behind a retrieval seam and not in every request."""

ORDER_MEMORY = -10
"""Memory comes ahead of the session facts and the harness note, because
instructions that the user wrote outrank anything that the harness says about
its own state."""


@dataclass(frozen=True, slots=True)
class DiscoveredInstructions:
    path: Path
    text: str
    scope: str
    """`"user"` for `$PH_HOME`, `"project"` for the root of the agent and every
    directory above it."""


def locate(root: Path, *, home: Path) -> list[tuple[Path, str, tuple[int, int]]]:
    """Where the instruction files are and what state they are in, without a read.

    The order is nearest first, because the instructions of a subdirectory are
    more specific than those of the repository, and the model reads a prompt
    from top to bottom. Every level is kept, and the nearest does not win
    alone: `AGENTS.md` is additive by convention, and the rules of a repository
    root still apply inside a package.

    Each path is returned with its `stat` facts, so that a caller can decide
    whether anything changed before it pays for the content. The cost is one
    `stat` for each candidate directory, which the walk costs anyway, because
    `is_file()` is a `stat`.
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
            # `home` is last, and it is labeled as what it is. The same file
            # reached both ways is one instruction, not two.
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
    """One file, capped at the read and not after it.

    `MAX_BYTES` bounds what a *request* can carry. To read 5 MiB and keep the
    first 64 KiB of it is the cost that this cap exists to avoid. The skills
    reader uses the same `handle.read(limit)` on `SKILL.md`. A text read counts
    characters, so the text is cut again to the bytes that the cap names,
    between characters (`ph.characters.head_bytes`).
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            text = head_bytes(handle.read(MAX_BYTES), MAX_BYTES)
    except OSError as error:
        log.warning("ph.system_prompt.memory: %s could not be read: %s", path, error)
        text = ""
    return DiscoveredInstructions(path=path, text=text, scope=scope)


def render(found: list[DiscoveredInstructions]) -> str:
    """The snapshot text, or `""` when there is nothing to say.

    An empty string is how a `context()` opts out of this assembly. A
    deployment with no `AGENTS.md` anywhere therefore contributes no snapshot,
    and not a heading with nothing under it.
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
    """The provider, which holds only what it reads.

    It is a class and not a closure for two reasons. It outlives the `apply`
    that built it, and a closure would keep that whole frame alive. The cache
    also needs a place to live that is not a mutable default.
    """

    ctx: Context
    home: Path
    _cache: dict[Path, tuple[Signature, str]] = field(default_factory=dict)
    """The rendered text for each agent root, with the signature of the files
    that it was rendered from.

    The cache is bounded, because the key is a *workspace root* and a
    long-running daemon fans out one root for each child. Without a bound it
    would retain a rendered snapshot for every worktree that the process ever
    saw. On overflow the whole map is dropped, and no single entry is evicted.
    The next assembly reads again, and the cost is a few hundred microseconds,
    which is not worth an LRU.
    """

    def root(self, agent: AgentHandle | None) -> Path:
        fs = self.ctx.get(FS)
        if fs is None:
            return Path.cwd()
        root: Path = fs.root_for(agent)
        return root

    def text(self, request: AssembleContext) -> str:
        """The memory of this agent. The files are read only when one of them
        changed since the last call."""
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


@plugin("memory-agents-md", affects="environment", inject=[SYSTEM_PROMPT])
async def apply(ctx: Context, config: None) -> None:
    """Contribute discovered `AGENTS.md` files as a post-cache snapshot."""
    # Resolved one time: `$PH_HOME` is a process constant, and a request for it
    # on each assembly costs a `stat` or two on a path that cannot have moved.
    memory = MemoryFiles(ctx=ctx, home=resolve_roots().home)
    ctx.require(SYSTEM_PROMPT).context(
        PromptContext(name="memory", order=ORDER_MEMORY, text=memory.text), scope=ctx
    )
