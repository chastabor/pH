"""A `SKILL.md` on disk, written the same way in both packages.

The format is `ph.seams.skills`', and two packages test against it: the seam's
own reader in ph-core, and `rlm-skills-python`'s package half in ph-rlm. The
builder was written out in the second one first; this is it where both can
reach it, before a third copy decides frontmatter looks slightly different.

@module ph.testing.skills
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["write_skill"]


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "search the web for a query",
    front_name: str | None = None,
    body: str = "Instructions here.",
) -> Path:
    """One skill directory, returning it. `front_name` writes a disagreeing name."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {front_name if front_name is not None else name}\n"
        f"description: {description}\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return directory
