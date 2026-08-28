"""`rlm-skills-python` — capability-layer skills, discovered and importable (P3-18).

A skill is a package a **distribution or a user** installed. The model cannot
install one and `/refine` cannot mint one: that is I7, and it is why skills and
the Continual Harness share a word but not a mechanism (Q13). This row is the
capability half — the knowledge half is `rlm-harness`.

Three parts, and the middle one is what makes this a *Python* skills row rather
than a second reader of the same Markdown:

* **Discovery.** `<dir>/<name>/SKILL.md` with YAML frontmatter, validated to the
  format `skills-progressive` (Phase 4) will read: `name` must be 1-64 lowercase
  `[a-z0-9-]` and equal to its directory, `description` at most 1024 characters,
  the file at most 10 MiB. Later sources win, so a project skill shadows a user
  one of the same name.
* **Installation.** A skill directory holding a `pyproject.toml` is installed
  **editable** into the runtime venv, and its import name is handed to the kernel
  at boot, where `ph_runtime.skill.wrap_skill_module` binds it callable — so a
  cell writes `await websearch(query=...)`, the convention §6.8 ports from
  prime-agent. A skill that fails to import binds a stub that says so, because a
  model finding a name undefined reads it as its own mistake.
* **The catalog.** Names and descriptions in the prompt, never bodies.
  `ctx.skills.body(name)` is what Phase 4's progressive disclosure will read.

**The two lists are deliberately two.** What to *install* is a requirement spec
and happens once per venv build; what to *import* is a module name and happens at
every kernel boot. `acme-websearch` installs and imports as `acme_websearch`, so
one list could not answer both questions.

@module ph_rlm.skills
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import anyio
from pydantic import Field

from ph.cordis import Context, LoaderError, plugin, safe_yaml_load
from ph.seams.skills import DESCRIPTION_MAX, NAME_PATTERN, Skill
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, PromptSection
from ph.wire import WireModel

__all__ = ["Config", "PythonSkill", "apply", "discover", "render_catalog"]

log = logging.getLogger("ph_rlm.skills")

SKILL_FILE = "SKILL.md"
MAX_SKILL_BYTES = 10 * 1024 * 1024
FRONTMATTER_MAX = 64 * 1024
"""How much of a `SKILL.md` validation reads. The fields it needs are capped at
about a kilobyte, and the body — up to 10 MiB — stays on disk by design (G9),
so reading it here would pull in exactly the bytes this row exists to defer."""

ORDER_SKILLS = ORDER_TOOL_GUIDANCE + 50

FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---", re.S)


class Config(WireModel):
    """Row config for `rlm-skills-python`."""

    paths: list[str] = Field(default_factory=list)
    """Directories to scan, earliest first. Each holds `<name>/SKILL.md`.

    Ordered because the last source wins: a project directory listed after a
    user one shadows it by name, which is the layering `skills-progressive`
    adopts (Deep Agents' base → user → project → team)."""


@dataclass(frozen=True, slots=True)
class PythonSkill:
    """One discovered skill: the catalog entry, and how to reach its code."""

    name: str
    description: str
    path: Path
    directory: Path
    module: str | None = None
    """The import name, when the directory is an installable package."""

    @property
    def requirement(self) -> str | None:
        """What to install. `None` for a documentation-only skill."""
        return str(self.directory) if self.module is not None else None


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


def _read(path: Path) -> PythonSkill | None:
    """One `SKILL.md`, validated. `None` — with a reason logged — if it is not one.

    Refused rather than half-accepted: a skill whose name does not match its
    directory would be addressed by one string in the catalog and found under
    another on disk, which is a bug report from a confused model later.
    """
    directory = path.parent
    try:
        if path.stat().st_size > MAX_SKILL_BYTES:
            log.warning("ph_rlm.skills: %s is larger than %s bytes", path, MAX_SKILL_BYTES)
            return None
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(FRONTMATTER_MAX)
    except OSError as error:
        log.warning("ph_rlm.skills: %s could not be read: %s", path, error)
        return None

    matched = FRONTMATTER.match(head)
    if matched is None:
        log.warning("ph_rlm.skills: %s has no YAML frontmatter", path)
        return None
    try:
        # `safe_yaml_load` is the codebase's one YAML policy: no tags, and no
        # implicit date coercion — `description: 2024-01-01` stays the string
        # the author wrote instead of becoming a datetime `str()` reshapes.
        front = safe_yaml_load(matched.group(1), origin=str(path)) or {}
    except LoaderError as error:
        log.warning("ph_rlm.skills: %s has invalid frontmatter: %s", path, error)
        return None
    if not isinstance(front, dict):
        log.warning("ph_rlm.skills: %s frontmatter is not a mapping", path)
        return None

    name = str(front.get("name") or "")
    description = str(front.get("description") or "")
    if NAME_PATTERN.match(name) is None:
        log.warning("ph_rlm.skills: %s has an invalid name %r", path, name)
        return None
    if name != directory.name:
        log.warning("ph_rlm.skills: %s is named %r but lives in %r", path, name, directory.name)
        return None
    if not description or len(description) > DESCRIPTION_MAX:
        log.warning("ph_rlm.skills: %s needs a description of 1..%s chars", path, DESCRIPTION_MAX)
        return None
    return PythonSkill(
        name=name,
        description=description,
        path=path,
        directory=directory,
        module=_module_name(directory),
    )


def discover(paths: list[str]) -> list[PythonSkill]:
    """Every valid skill under `paths`, later sources shadowing earlier ones."""
    found: dict[str, PythonSkill] = {}
    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            log.debug("ph_rlm.skills: %s is not a directory; skipped", root)
            continue
        for candidate in sorted(root.glob(f"*/{SKILL_FILE}")):
            skill = _read(candidate)
            if skill is not None:
                found[skill.name] = skill
    return [found[name] for name in sorted(found)]


def render_catalog(skills: list[PythonSkill]) -> str:
    """Names and descriptions. Bodies stay on disk until something asks (G9)."""
    if not skills:
        return ""
    lines = [
        "# Skills",
        "",
        "Capabilities this deployment installed. Recognize when one applies, read its "
        "full instructions before using it, then follow them.",
        "",
    ]
    for skill in skills:
        call = (
            f" Callable in a cell as `await {skill.module}(...)`."
            if skill.module is not None
            else ""
        )
        # The path makes "read the instructions first" followable — a catalog
        # that names a file it does not locate leaves the model guessing paths.
        lines.append(f"- **{skill.name}** — {skill.description}{call}")
        lines.append(f"  Full instructions: `{skill.path}`.")
    lines.extend(["", "Read a skill's full instructions (path above) before relying on it."])
    return "\n".join(lines)


@plugin("rlm-skills-python", config=Config, inject=["skills", "system_prompt", "python_runtime"])
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
    found = await anyio.to_thread.run_sync(discover, config.paths)
    if not found:
        return

    for skill in found:
        ctx.skills.register(
            Skill(
                name=skill.name,
                description=skill.description,
                path=str(skill.path),
                source="rlm-skills-python",
            )
        )

    runtime = ctx.python_runtime
    requirements = tuple(skill.requirement for skill in found if skill.requirement is not None)
    modules = tuple(skill.module for skill in found if skill.module is not None)
    runtime.skills = (*runtime.skills, *requirements)
    runtime.skill_modules = (*runtime.skill_modules, *modules)
    log.info(
        "ph_rlm.skills: %s skill(s), %s importable: %s",
        len(found),
        len(modules),
        ", ".join(skill.name for skill in found),
    )

    catalog = render_catalog(found)
    if catalog:
        ctx.system_prompt.section(
            PromptSection(name="rlm:skills", order=ORDER_SKILLS, text=catalog)
        )
