"""`rlm-skills-python`: the half of a skill that only Code Mode can use (P3-18).

A skill is a directory with a `SKILL.md`. `ph.seams.skills` owns that part: the
format, the limits, the registry, the catalog, and the `skill` tool that gives
the model a body. This row owns the part that is true only here. **A skill
directory that is also an installable package** is installed into the kernel
venv, and its import name is given to every kernel at boot. `await name(...)`
therefore works inside a cell.

That is why the catalog entry is a `hint` on the registered skill and not a
second prompt section. The model must be told about a skill one time, in one
list, with the ways that the skill can be used attached to it.

**There are two lists, deliberately.** `skills` is what to install, and
`skill_modules` is what to import. A distribution named `acme-websearch` imports
as `acme_websearch`, so one list could not answer both questions.

@module ph_rlm.skills
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

import anyio
from pydantic import Field

from ph.cordis import Context, plugin
from ph.keys import SKILLS
from ph.seams.skills import Skill, discover_skills
from ph.wire import WireModel

from .keys import PYTHON_RUNTIME

__all__ = ["Config", "PythonSkill", "apply", "give_to_runtime"]

log = logging.getLogger("ph_rlm.skills")


class Config(WireModel):
    """The row config for `rlm-skills-python`."""

    paths: list[str] = Field(default_factory=list)
    """The directories to scan, earliest first. Each one holds `<name>/SKILL.md`.

    The order matters because the last source wins. A project directory that is
    listed after a user directory shadows it by name. That is the layering that
    `skills-progressive` adopts (base → user → project → team, from Deep
    Agents)."""


@dataclass(frozen=True, slots=True)
class PythonSkill:
    """A discovered skill, plus how to reach its code from a cell."""

    skill: Skill
    module: str | None = None
    """The import name, when the directory is an installable package."""

    @property
    def directory(self) -> Path:
        """The directory of the skill. It is derived, so a wrapper cannot hold a
        directory that disagrees with the skill that it wraps. `python_half` is
        the only constructor, and it computed the directory from the same field
        anyway."""
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

    The facts are derived from the path of the skill itself, and the directories
    are not scanned again. The seam already found the file and validated it. A
    second walk of the same directories would be a second answer to "which skills
    are installed", and the two answers could disagree.
    """
    return PythonSkill(skill=skill, module=_module_name(Path(skill.path or ".").parent))


def _module_name(directory: Path) -> str | None:
    """The import name that a skill directory provides, or `None`.

    The name is read from `pyproject.toml` and not guessed from the directory. A
    package named `acme-websearch` imports as `acme_websearch`, and a
    `[project] name` that disagrees with the folder is the normal case, not the
    unusual one.
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


class _SkillHolder(Protocol):
    """What `give_to_runtime` writes: the two skill lists of `PythonCodeRuntime`."""

    skills: tuple[str, ...]
    skill_modules: tuple[str, ...]


def give_to_runtime(
    ctx: Context, runtime: _SkillHolder, requirements: tuple[str, ...], modules: tuple[str, ...]
) -> None:
    """Add the skills of a row to the lists of the runtime, and take them back
    when the row goes.

    **The skills are taken back, one occurrence each** (Holdings rules 2 and 4).
    The row used to append and keep nothing to undo the append with. A row that
    was mounted again then added its packages a second time, and a row that was
    let go left its packages for every later kernel to install and import. The
    entries of another row are left where they are, including an entry that is
    equal.
    """

    def take_back() -> None:
        runtime.skills = _without(runtime.skills, requirements)
        runtime.skill_modules = _without(runtime.skill_modules, modules)

    # The take-back is registered before the write that it undoes (Holdings rule 1).
    ctx.add_disposer(take_back, label="rlm-skills-python(runtime)")
    runtime.skills = (*runtime.skills, *requirements)
    runtime.skill_modules = (*runtime.skill_modules, *modules)


def _without(held: tuple[str, ...], gone: tuple[str, ...]) -> tuple[str, ...]:
    """`held` without one occurrence of each item of `gone`, in order."""
    left = list(held)
    for one in gone:
        if one in left:
            left.remove(one)
    return tuple(left)


@plugin("rlm-skills-python", affects="environment", config=Config, inject=[SKILLS, PYTHON_RUNTIME])
async def apply(ctx: Context, config: Config) -> None:
    """Discover skills, register them, and make their packages importable.

    The row injects `python_runtime` and does not read a config setting on it.
    The venv and the boot frame belong to the runtime, and this row is the one
    that knows which packages a skill directory provides. The skills are
    contributed at mount. That is before the interpreter is resolved (it is lazy)
    and before any kernel boots, so nothing must be rebuilt or restarted for a
    skill to be there.
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
        ctx.require(SKILLS).register(one.skill.model_copy(update={"hint": one.hint}), scope=ctx)

    runtime = ctx.require(PYTHON_RUNTIME)
    requirements = tuple(one.requirement for one in found if one.requirement is not None)
    modules = tuple(one.module for one in found if one.module is not None)
    give_to_runtime(ctx, runtime, requirements, modules)
    log.info(
        "ph_rlm.skills: %s skill(s), %s importable: %s",
        len(found),
        len(modules),
        ", ".join(one.skill.name for one in found),
    )
