"""`rlm-skills-python` — the half of a skill that only Code Mode can use (P3-18).

A skill is a directory with a `SKILL.md`, and `ph.seams.skills` owns that:
the format, the limits, the registry, the catalog, and the `skill` tool that
hands the model a body. This row owns the part that is only true here — **a
skill directory that is also an installable package** gets installed into the
kernel venv and its import name handed to every kernel at boot, so `await
name(...)` works inside a cell.

Which is why the catalog entry is a `hint` on the registered skill rather than a
second prompt section: the model should be told about a skill once, in one list,
with the ways it can actually be used attached to it.

**Two lists, deliberately.** `skills` is what to install and `skill_modules` is
what to import — a distribution named `acme-websearch` imports as
`acme_websearch`, so one list could not answer both.

@module ph_rlm.skills
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio
from pydantic import Field

from ph.cordis import Context, plugin
from ph.seams.skills import Skill, discover_skills
from ph.wire import WireModel

__all__ = ["Config", "PythonSkill", "apply"]

log = logging.getLogger("ph_rlm.skills")


class Config(WireModel):
    """Row config for `rlm-skills-python`."""

    paths: list[str] = Field(default_factory=list)
    """Directories to scan, earliest first. Each holds `<name>/SKILL.md`.

    Ordered because the last source wins: a project directory listed after a
    user one shadows it by name, which is the layering `skills-progressive`
    adopts (Deep Agents' base → user → project → team)."""


@dataclass(frozen=True, slots=True)
class PythonSkill:
    """A discovered skill, plus how to reach its code from a cell."""

    skill: Skill
    module: str | None = None
    """The import name, when the directory is an installable package."""

    @property
    def directory(self) -> Path:
        """Derived, so a wrapper cannot hold a directory that disagrees with the
        skill it wraps — `python_half` is the only constructor and computed it
        from the same field anyway."""
        return Path(self.skill.path or ".").parent

    @property
    def requirement(self) -> str | None:
        """What to install. `None` for a documentation-only skill."""
        return str(self.directory) if self.module is not None else None

    @property
    def hint(self) -> str:
        """What the shared catalog should add for this skill, if anything."""
        return "" if self.module is None else f"Callable in a cell as `await {self.module}(...)`."


def python_half(skill: Skill) -> PythonSkill:
    """Pair a discovered skill with its package facts.

    Derived from the skill's own path rather than re-scanned: the seam already
    found the file and validated it, and a second walk of the same directories
    is a second answer to "which skills are installed" waiting to disagree.
    """
    return PythonSkill(skill=skill, module=_module_name(Path(skill.path or ".").parent))


def _module_name(directory: Path) -> str | None:
    """The import name a skill directory provides, or `None`.

    Read from `pyproject.toml` rather than guessed from the directory: a package
    named `acme-websearch` imports as `acme_websearch`, and a `[project] name`
    that disagrees with the folder is the normal case, not the odd one.
    """
    manifest = directory / "pyproject.toml"
    if not manifest.is_file():
        return None
    try:
        parsed = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        log.warning("ph_rlm.skills: %s is unreadable: %s", manifest, error)
        return None
    name = (parsed.get("project") or {}).get("name")
    return str(name).replace("-", "_") if isinstance(name, str) and name else None


@plugin("rlm-skills-python", config=Config, inject=["skills", "python_runtime"])
async def apply(ctx: Context, config: Config) -> None:
    """Discover skills, register them, and make their packages importable.

    Injects `python_runtime` rather than reading a config knob on it: the venv
    and the boot frame are the runtime's to own, and this row is the one that
    knows which packages a skill directory provides. Contributed at mount, which
    is before the interpreter is resolved (it is lazy) and before any kernel
    boots — so nothing has to be rebuilt or restarted for a skill to be there.
    """
    if not config.paths:
        return
    discovered = await anyio.to_thread.run_sync(
        partial(discover_skills, config.paths, source="rlm-skills-python")
    )
    found = [python_half(skill) for skill in discovered]
    if not found:
        return

    for one in found:
        ctx.skills.register(one.skill.model_copy(update={"hint": one.hint}), scope=ctx)

    runtime = ctx.python_runtime
    requirements = tuple(one.requirement for one in found if one.requirement is not None)
    modules = tuple(one.module for one in found if one.module is not None)
    runtime.skills = (*runtime.skills, *requirements)
    runtime.skill_modules = (*runtime.skill_modules, *modules)
    log.info(
        "ph_rlm.skills: %s skill(s), %s importable: %s",
        len(found),
        len(modules),
        ", ".join(one.skill.name for one in found),
    )
