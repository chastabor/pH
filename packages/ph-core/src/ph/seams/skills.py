"""`ctx.skills` — the capability layer, and the boundary it must not cross.

A skill is a package a *distribution or a user* installed. The model cannot
install one, and `/refine` (Phase 3) cannot mint one: that is invariant I7 and
the reason skills and the Continual Harness share a word but not a mechanism —
the knowledge layer writes *procedure*, never *capability* (Q13).

Progressive disclosure (the catalog in the prompt, the body on demand) is G9 in
Phase 4; this is the seam it will attach to.

@module ph.seams.skills
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cordis import Context, Disposer, plugin
from ..wire import WireModel
from ._registry import claim_key

__all__ = ["NAME_PATTERN", "Skill", "SkillService", "apply"]

NAME_MAX = 64
DESCRIPTION_MAX = 1_024
NAME_PATTERN = re.compile(rf"^[a-z0-9-]{{1,{NAME_MAX}}}$")
"""The one format of a skill name, owned here because two readers of `SKILL.md`
share it (`rlm-skills-python` now, `skills-progressive` in Phase 4) and a
catalog whose names another reader refuses would be two definitions of "valid
skill" drifting apart."""


class Skill(WireModel):
    """One installed skill: a name, a one-line description, and a body on disk."""

    name: str
    description: str
    path: str | None = None
    source: str = "installed"


@dataclass(slots=True)
class SkillService:
    """The service published as `ctx.skills`."""

    ctx: Context
    _skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill, *, scope: Context | None = None) -> Disposer:
        """Install a skill. Bounds are enforced here so a catalog stays a catalog."""
        if NAME_PATTERN.match(skill.name) is None:
            raise ValueError(
                f'"{skill.name}" is not a skill name: 1..{NAME_MAX} of lowercase [a-z0-9-]'
            )
        if len(skill.description) > DESCRIPTION_MAX:
            raise ValueError(f"a skill description must be at most {DESCRIPTION_MAX} characters")
        return claim_key(scope or self.ctx, self._skills, skill.name, skill, label="skill")

    def list(self) -> list[Skill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def body(self, name: str) -> str | None:
        """The skill's full text, read only when asked for (G9)."""
        skill = self._skills.get(name)
        if skill is None or skill.path is None:
            return None
        return Path(skill.path).read_text(encoding="utf-8")


@plugin("skills")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the skills seam."""
    ctx.provide("skills", SkillService(ctx=ctx))
