"""`ctx.skills` — the capability layer, and the boundary it must not cross.

A skill is a package a *distribution or a user* installed. The model cannot
install one, and `/refine` (Phase 3) cannot mint one: that is invariant I7 and
the reason skills and the Continual Harness share a word but not a mechanism —
the knowledge layer writes *procedure*, never *capability* (Q13).

Progressive disclosure is G9, and it lives here too, as the `skills-progressive`
row at the bottom of this file: the catalog goes in the prompt, the body stays on
disk until the model asks for it by name. One module, because "what a skill is"
— the format on disk, the limits, the registry, and what the model is told about
it — is one question, and the row that reads a `SKILL.md` and the seam that
bounds its name would otherwise state the same rule twice.

**Nothing is scanned by default.** A deployment names its directories; a
well-known path scanned at every start would make "install a skill" mean "drop a
file in a directory", which is the capability boundary I7 draws — a skill is
something a *distribution or a user* installed, deliberately.

@module ph.seams.skills
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from pydantic import Field

from ..cordis import Context, Disposer, LoaderError, plugin, safe_yaml_load
from ..system_prompt.assembly import ORDER_TOOL_GUIDANCE, PromptSection
from ..tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..tools.presentation import simple_views
from ..tools.registry import register_when_composed
from ..wire import WireModel
from ._names import require_slug, slug_pattern
from ._registry import claim_key

__all__ = [
    "FRONTMATTER_MAX",
    "HINT_MAX",
    "MAX_SKILL_BYTES",
    "NAME_PATTERN",
    "SKILL_FILE",
    "Skill",
    "SkillService",
    "apply",
    "discover_skills",
    "progressive",
    "read_skill",
    "render_catalog",
]

log = logging.getLogger("ph.seams.skills")

NAME_MAX = 64
DESCRIPTION_MAX = 1_024
HINT_MAX = 256
"""A hint is one clause after a description, and it reaches every prompt the
same way — so it is bounded for the reason the description is, at the length a
clause actually needs."""
SKILL_FILE = "SKILL.md"
MAX_SKILL_BYTES = 10 * 1024 * 1024
"""G9's third limit, enforced in two places for two different reasons: the
scanner refuses a file this large so a catalog cannot be poisoned by one, and
`SkillService.body` refuses to *return* one so a skill that grew after discovery
cannot arrive in a context window."""
FRONTMATTER_MAX = 64 * 1024
"""How much of a `SKILL.md` validation reads. The fields it needs are capped at
about a kilobyte, and the body — up to 10 MiB — stays on disk by design, so
reading it here would pull in exactly the bytes this row exists to defer."""

ORDER_SKILLS = ORDER_TOOL_GUIDANCE + 50

FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---", re.S)
NAME_PATTERN = slug_pattern(NAME_MAX)
"""The one format of a skill name, at this seam's own bound.

Exported because two readers of `SKILL.md` share it (`rlm-skills-python` now,
`skills-progressive` in Phase 4) and a catalog whose names another reader
refuses would be two definitions of "valid skill" drifting apart — and because
one of them *tests* rather than raises, warning past a bad frontmatter block
instead of refusing the whole scan. The format itself is `_names`', shared with
the screen ids that are the same kind of token."""


class Skill(WireModel):
    """One installed skill: a name, a one-line description, and a body on disk."""

    name: str
    description: str
    path: str | None = None
    source: str = "installed"
    hint: str = ""
    """One extra clause the catalog prints after the description.

    For what a *particular registrar* knows and the catalog cannot derive:
    `rlm-skills-python` installs a skill's package into the kernel venv, so for
    those skills there is a second way to use one — `await name(...)` in a cell —
    and a model told only "read the file" would never find it. A field rather
    than a second catalog, because two catalogs is how a deployment ends up
    telling the model about the same skill twice in one prompt."""


@dataclass(slots=True)
class SkillService:
    """The service published as `ctx.skills`."""

    ctx: Context
    _skills: dict[str, Skill] = field(default_factory=dict)

    def register(self, skill: Skill, *, scope: Context | None = None) -> Disposer:
        """Install a skill. Bounds are enforced here so a catalog stays a catalog."""
        require_slug(skill.name, maximum=NAME_MAX, kind="skill name")
        if len(skill.description) > DESCRIPTION_MAX:
            raise ValueError(f"a skill description must be at most {DESCRIPTION_MAX} characters")
        if len(skill.hint) > HINT_MAX:
            raise ValueError(f"a skill hint must be at most {HINT_MAX} characters")
        return claim_key(scope or self.ctx, self._skills, skill.name, skill, label="skill")

    def list(self) -> list[Skill]:
        return [self._skills[name] for name in sorted(self._skills)]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def body(self, name: str) -> str | None:
        """The skill's full text, read only when asked for (G9).

        Capped again here rather than trusting discovery: a file is validated
        once at mount and read whenever the model asks, and the window between
        those two is exactly where a skill directory that is a checkout gets
        `git pull`ed. `None` for an unknown name or a body that no longer fits,
        because both mean "there is nothing to hand you".
        """
        skill = self._skills.get(name)
        if skill is None or skill.path is None:
            return None
        path = Path(skill.path)
        try:
            if path.stat().st_size > MAX_SKILL_BYTES:
                log.warning("ph.seams.skills: %s grew past %s bytes", path, MAX_SKILL_BYTES)
                return None
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            log.warning("ph.seams.skills: %s could not be read: %s", path, error)
            return None


@plugin("skills")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the skills seam."""
    ctx.provide("skills", SkillService(ctx=ctx))


# ------------------------------------------------------- the format on disk --


def read_skill(path: Path, *, source: str = "skills-progressive") -> Skill | None:
    """One `SKILL.md`, validated. `None` — with a reason logged — if it is not one.

    **Refused rather than half-accepted**: a skill whose name does not match its
    directory would be addressed by one string in the catalog and found under
    another on disk, which is a bug report from a confused model later. A bad
    skill is skipped rather than fatal, because one malformed directory must not
    cost a deployment every other skill it installed.
    """
    directory = path.parent
    try:
        if path.stat().st_size > MAX_SKILL_BYTES:
            log.warning("ph.seams.skills: %s is larger than %s bytes", path, MAX_SKILL_BYTES)
            return None
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(FRONTMATTER_MAX)
    except OSError as error:
        log.warning("ph.seams.skills: %s could not be read: %s", path, error)
        return None

    matched = FRONTMATTER.match(head)
    if matched is None:
        log.warning("ph.seams.skills: %s has no YAML frontmatter", path)
        return None
    try:
        # `safe_yaml_load` is the codebase's one YAML policy: no tags, and no
        # implicit date coercion — `description: 2024-01-01` stays the string
        # the author wrote instead of becoming a datetime `str()` reshapes.
        front = safe_yaml_load(matched.group(1), origin=str(path)) or {}
    except LoaderError as error:
        log.warning("ph.seams.skills: %s has invalid frontmatter: %s", path, error)
        return None
    if not isinstance(front, dict):
        log.warning("ph.seams.skills: %s frontmatter is not a mapping", path)
        return None

    name = str(front.get("name") or "")
    description = str(front.get("description") or "")
    if NAME_PATTERN.match(name) is None:
        log.warning("ph.seams.skills: %s has an invalid name %r", path, name)
        return None
    if name != directory.name:
        log.warning("ph.seams.skills: %s is named %r but lives in %r", path, name, directory.name)
        return None
    if not description or len(description) > DESCRIPTION_MAX:
        log.warning("ph.seams.skills: %s needs a description of 1..%s chars", path, DESCRIPTION_MAX)
        return None
    return Skill(name=name, description=description, path=str(path), source=source)


def discover_skills(paths: Sequence[str], *, source: str = "skills-progressive") -> list[Skill]:
    """Every valid skill under `paths`, later sources shadowing earlier ones.

    Shadowing rather than colliding, because the list is a precedence order: a
    user's own `code-review` is meant to win over the one their distribution
    ships, and a refusal would make overriding a skill impossible.
    """
    found: dict[str, Skill] = {}
    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            log.debug("ph.seams.skills: %s is not a directory; skipped", root)
            continue
        for candidate in sorted(root.glob(f"*/{SKILL_FILE}")):
            skill = read_skill(candidate, source=source)
            if skill is not None:
                found[skill.name] = skill
    return [found[name] for name in sorted(found)]


# ------------------------------------------------ the catalog and the body --


def render_catalog(skills: list[Skill], *, tool: bool = True) -> str:
    """Names and descriptions. Bodies stay on disk until something asks (G9).

    `tool=False` drops the sentence naming the `skill` tool, for the deployment
    where the catalog is rendered and the tool is not registered — telling a
    model to call something that is not in its schema is worse than saying
    nothing, because it looks like the model's mistake.
    """
    if not skills:
        return ""
    how = " with the `skill` tool" if tool else ""
    lines = [
        "# Skills",
        "",
        f"Capabilities this deployment installed. Recognize when one applies, read its "
        f"full instructions{how} before using it, then follow them.",
        "",
    ]
    for skill in skills:
        hint = f" {skill.hint}" if skill.hint else ""
        lines.append(f"- **{skill.name}** — {skill.description}{hint}")
    return "\n".join(lines)


class SkillArgs(ToolModel):
    name: str = Field(description="The skill's name, exactly as the catalog spells it.")


class SkillValue(ToolModel):
    name: str
    path: str | None = None
    instructions: str


class Config(WireModel):
    """Row config for `skills-progressive`."""

    paths: list[str] = Field(default_factory=list)
    """Directories to scan, earliest first. Each holds `<name>/SKILL.md`.

    Empty by default and on purpose: see the module docstring. A path is
    `~`-expanded, so `~/.ph/skills` is a deployment's choice to write down
    rather than this row's to assume."""


@plugin("skills-progressive", config=Config, inject=["skills", "system_prompt", "tools"])
async def progressive(ctx: Context, config: Config) -> None:
    """Catalog in the prompt, body on demand (G9).

    **The catalog is a provider, not a fixed string.** It reads the registry at
    assembly time, so a skill registered by *another* row — `rlm-skills-python`
    is the one that exists — appears in this catalog rather than in a second one
    of its own, and no row has to be mounted before another. The cost is that
    registering a skill mid-session moves the cached prefix, which is honest:
    the model was genuinely told something new.
    """
    if config.paths:
        # Threaded because a scan is `glob` plus an open per candidate, and mount
        # is on the event loop that a TUI is about to draw on.
        for skill in await anyio.to_thread.run_sync(discover_skills, config.paths):
            ctx.skills.register(skill, scope=ctx)

    def catalog(_request: Any) -> str:
        # Asked per assembly, so a skill registered by a later row is in the
        # catalog — and the `skill` tool it names is checked the same way,
        # because the two can legitimately disagree for one deployment.
        return render_catalog(ctx.skills.list(), tool=ctx.tools.get("skill") is not None)

    ctx.system_prompt.section(
        PromptSection(name="skills", order=ORDER_SKILLS, text=catalog), scope=ctx
    )

    async def read_body(args: SkillArgs, _run: ToolRunContext) -> Any:
        skill = ctx.skills.get(args.name)
        body = ctx.skills.body(args.name)
        if skill is None or body is None:
            # Named, because "unknown skill" and "this skill has no readable
            # body" are different problems for the model: one is a typo it can
            # fix from the catalog, the other is a deployment fault it cannot.
            known = ", ".join(one.name for one in ctx.skills.list()) or "none"
            raise ValueError(f"no readable skill named {args.name!r}; installed: {known}")
        return {"name": skill.name, "path": skill.path, "instructions": body}

    def build_tool() -> Any:
        """The tool, or `None` where there is nothing to read.

        See `register_when_composed`: a `skill` tool in a deployment that
        installed no skills is a prompt-sized advertisement for a capability
        that is not there.
        """
        if not ctx.skills.list():
            return None
        return define_tool(
            "skill",
            "Read one installed skill's full instructions before using it. The catalog "
            "in the system prompt lists what is installed; this returns the whole body.",
            parameters=SkillArgs,
            output=ToolOutput(
                schema=SkillValue, render=lambda _a, v: text_content(v["instructions"])
            ),
            execute=read_body,
            is_concurrency_safe=True,
            **simple_views("read", "Read skill", "name"),
        )

    register_when_composed(ctx, build_tool)
