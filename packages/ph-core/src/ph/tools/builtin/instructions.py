"""`agent-instructions` — AGENTS.md discovery.

Project instructions are found by walking from the workspace root upward, then
the user's `$PH_HOME/AGENTS.md`. Nearest-first, because a subdirectory's
instructions are more specific than the repository's.

Placement is the load-bearing detail, and it lands properly in Phase 4 (G8):
discovered instructions belong in the cached *prefix*, so editing them does not
invalidate every later turn's cache. Phase 1 registers them as an ordinary
static section, which has that property already.

@module ph.tools.builtin.instructions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...cordis import Context, plugin
from ...paths import resolve_roots
from ...system_prompt.assembly import ORDER_DEPLOYMENT_PERSONA, PromptSection

__all__ = ["DiscoveredInstructions", "apply", "discover"]

FILENAME = "AGENTS.md"
MAX_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class DiscoveredInstructions:
    path: Path
    text: str
    scope: str
    """`"user"` for `$PH_HOME`, `"project"` for anything under the workspace."""


def discover(root: Path, *, home: Path | None = None) -> list[DiscoveredInstructions]:
    """Find instruction files, nearest first, user-level last."""
    found: list[DiscoveredInstructions] = []
    seen: set[Path] = set()
    for directory in [root, *root.parents]:
        candidate = directory / FILENAME
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        found.append(
            DiscoveredInstructions(
                path=candidate,
                text=candidate.read_text(encoding="utf-8", errors="replace")[:MAX_BYTES],
                scope="project",
            )
        )
    user_home = home if home is not None else resolve_roots().home
    user_file = user_home / FILENAME
    if user_file.is_file() and user_file not in seen:
        found.append(
            DiscoveredInstructions(
                path=user_file,
                text=user_file.read_text(encoding="utf-8", errors="replace")[:MAX_BYTES],
                scope="user",
            )
        )
    return found


@plugin("agent-instructions", inject=["system_prompt"])
async def apply(ctx: Context, config: Any) -> None:
    """Contribute discovered AGENTS.md files as a static prompt section."""
    fs = ctx.get("fs")
    root = Path(getattr(fs, "root", Path.cwd()))
    discovered = discover(root)
    if not discovered:
        return
    body = "\n\n".join(
        f"<!-- {item.scope}: {item.path} -->\n{item.text.strip()}" for item in discovered
    )
    ctx.system_prompt.section(
        PromptSection(
            name="agent-instructions",
            order=ORDER_DEPLOYMENT_PERSONA + 10,
            text=(
                "The following instructions come from AGENTS.md files in this "
                "workspace and from the user's own configuration. Follow them.\n\n"
                f"{body}"
            ),
        )
    )
