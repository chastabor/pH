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
from ..system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ..tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..tools.presentation import simple_views
from ..tools.registry import register_when_composed
from ..wire import WireModel
from ._names import require_slug, slug_pattern
from ._registry import claim_entry, claim_key
from ._restriction import NameFilter

__all__ = [
    "FRONTMATTER_MAX",
    "HINT_MAX",
    "MAX_SKILL_BYTES",
    "NAME_PATTERN",
    "SKILL_FILE",
    "Skill",
    "SkillRestriction",
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


SkillRestriction = NameFilter
"""A per-scope filter over installed skills. Restrictions intersect.

The same `NameFilter` `ctx.tools` narrows with — one rule, so a change to what
`allow` and `deny` mean cannot reach one registry and miss the other.
**Intersecting is the load-bearing word**: a restriction can only subtract, so a
scope cannot grant itself a skill by adding one, and a child under a restricted
parent cannot climb back out (P4-13b).
"""


@dataclass(slots=True)
class SkillService:
    """The service published as `ctx.skills`.

    **Installation is global; only *reach* is scoped.** `register`'s `scope=` is
    a lifetime, not a visibility — it says which row's unloading takes the skill
    away, and the skill is in the catalog of every agent until then. Narrowing
    is a separate verb (`restrict`) for the reason P4-13b states as its whole
    security content: a child may hold a subset of its parent and never more, so
    the registry offers subtraction and no addition. There is deliberately no
    way to give one agent a skill another cannot see — to give a child a skill,
    install it, which gives it to the parent too.
    """

    ctx: Context
    _skills: dict[str, Skill] = field(default_factory=dict)
    _restrictions: dict[Context | None, list[SkillRestriction]] = field(default_factory=dict)
    """Filters keyed by the scope they belong to, as `ctx.tools` keys its layers.

    A flat list scanned per read was the first shape and it made **every agent
    pay for every other agent's narrowing**: a fan-out of sixteen children put
    sixteen filters in front of the parent's own catalog, none of which could
    change what the parent sees, and the parent's prompt assembly measured 3.9 times
    slower for it. Keyed, a read walks `isolation_chain()` — the filters that
    can possibly apply and no others.
    """
    _reach: dict[tuple[Context | None, ...], tuple[int, frozenset[str]]] = field(
        default_factory=dict
    )
    """What each chain may reach, memoized against `_generation`.

    `ToolRuntime.view`'s arrangement for the same reason: the answer is read once
    per skill per prompt assembly and changes only when something registers or
    unloads.
    """
    _generation: int = 0

    def register(self, skill: Skill, *, scope: Context | None = None) -> Disposer:
        """Install a skill. Bounds are enforced here so a catalog stays a catalog."""
        require_slug(skill.name, maximum=NAME_MAX, kind="skill name")
        if len(skill.description) > DESCRIPTION_MAX:
            raise ValueError(f"a skill description must be at most {DESCRIPTION_MAX} characters")
        if len(skill.hint) > HINT_MAX:
            raise ValueError(f"a skill hint must be at most {HINT_MAX} characters")
        released = claim_key(
            self.ctx.owner_for(scope), self._skills, skill.name, skill, label="skill"
        )
        self._changed()

        def release() -> None:
            released()
            self._changed()

        return release

    def restrict(self, restriction: SkillRestriction, *, scope: Context | None = None) -> Disposer:
        """Narrow what one scope may see. Filters along the chain intersect.

        Pass `scope=` — a filter owned by this service's context applies to every
        agent, which is a deployment-wide policy and almost never what a caller
        narrowing one child meant.
        """
        # Two questions, two answers (P6-12): the bucket is *which scope this
        # narrows*, and must stay where the caller put it; the disposer is *when
        # it lifts*, which is the row that asked. One value answering both is
        # what this row found in five registries.
        bucket = self._restrictions.setdefault(self.ctx.layer_for(scope).isolation, [])
        released = claim_entry(
            self.ctx.owner_for(scope), bucket, restriction, label="skill-restriction"
        )

        self._changed()

        def release() -> None:
            released()
            self._changed()

        return release

    def _changed(self) -> None:
        self._generation += 1

    def reach(self, scope: Context | None = None) -> frozenset[str]:
        """Every skill name this scope may use. The one place filters compose."""
        target = scope or self.ctx
        chain = tuple(target.isolation_chain())
        cached = self._reach.get(chain)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        names = frozenset(
            name
            for name in self._skills
            if all(one.admits(name) for key in chain for one in self._restrictions.get(key, ()))
        )
        self._reach[chain] = (self._generation, names)
        return names

    def admits(self, name: str, scope: Context | None = None) -> bool:
        """Whether `scope` may reach this skill at all."""
        return name in self.reach(scope)

    def list(self, scope: Context | None = None) -> list[Skill]:
        """What this scope may use, sorted. The catalog and every gate read this."""
        reachable = self.reach(scope)
        return [self._skills[name] for name in sorted(reachable)]

    def get(self, name: str, scope: Context | None = None) -> Skill | None:
        return self._skills.get(name) if self.admits(name, scope) else None

    def body(self, name: str, scope: Context | None = None) -> str | None:
        """The skill's full text, read only when asked for (G9).

        Capped again here rather than trusting discovery: a file is validated
        once at mount and read whenever the model asks, and the window between
        those two is exactly where a skill directory that is a checkout gets
        `git pull`ed. `None` for an unknown name or a body that no longer fits,
        because both mean "there is nothing to hand you".
        """
        skill = self.get(name, scope)
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

    def catalog(request: AssembleContext) -> str:
        # Per assembly *and* per agent: a child narrowed at spawn (P4-13b) must
        # not be told about skills it cannot read, or the catalog is a list of
        # things to try and fail at. The `skill` tool is checked the same way,
        # because the two can legitimately disagree for one deployment.
        return render_catalog(
            ctx.skills.list(request.scope),
            # Scoped too: a child narrowed away from the `skill` tool must not be
            # told in prose to call it. Telling a model to use something absent
            # from its schema reads as the model's mistake, not the profile's.
            tool=ctx.tools.get("skill", scope=request.scope) is not None,
        )

    ctx.system_prompt.section(
        PromptSection(name="skills", order=ORDER_SKILLS, text=catalog), scope=ctx
    )

    async def read_body(args: SkillArgs, run: ToolRunContext) -> Any:
        # The *agent's* scope, so a narrowed child is refused a skill it can see
        # named nowhere — the catalog and the gate answer the same question.
        scope = getattr(run.agent, "ctx", None)
        skill = ctx.skills.get(args.name, scope)
        body = ctx.skills.body(args.name, scope)
        if skill is None or body is None:
            # Named, because "unknown skill" and "this skill has no readable
            # body" are different problems for the model: one is a typo it can
            # fix from the catalog, the other is a deployment fault it cannot.
            known = ", ".join(one.name for one in ctx.skills.list(scope)) or "none"
            raise ValueError(f"no readable skill named {args.name!r}; available: {known}")
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
