"""`ctx.skills` — the capability layer, and the boundary it must not cross.

A skill is a package a *distribution or a user* installed. The model cannot
install one, and `/refine` cannot mint one: that is invariant I7, and the reason
skills and the Continual Harness share a word but not a mechanism — the knowledge
layer writes *procedure*, never *capability* (Q13).

Progressive disclosure is G9 and lives here too, as the `skills-progressive` row
at the bottom of this file: the catalog goes in the prompt, the body stays on disk
until the model asks for it by name. One module, because "what a skill is" — the
format on disk, the limits, the registry, and what the model is told about it — is
one question.

**Nothing is scanned by default.** A deployment names its directories; a
well-known path scanned at every start would make "install a skill" mean "drop a
file in a directory", which is the capability boundary I7 draws.

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

from ..cordis import (
    DEPLOYMENT,
    Boundary,
    Context,
    Disposer,
    LoaderError,
    boundary_of,
    chain_label,
    drop_dead_chains,
    plugin,
    safe_yaml_load,
)
from ..system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ..tools.definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..tools.presentation import simple_views
from ..tools.registry import register_when_composed
from ..wire import WireModel
from ._names import require_slug, slug_pattern
from ._registry import claim_entry, claim_key
from ._restriction import NameFilter

__all__ = [
    "ARGUMENT_HINT_MAX",
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

MAX_ALLOWED_TOOLS = 32
"""How many tools one skill may name. Bounded for the reason every other
model-adjacent list is: it reaches a prompt, and a skill naming two hundred is
one that has not decided what it does."""

ARGUMENT_HINT_MAX = 200
"""How long `argument-hint` may be. It rides the catalog, which is in the prompt
every turn — a paragraph here is paid for by every request in the session.

Its own name, not `HINT_MAX`: that one is 40 lines up and bounds `Skill.hint`,
the clause a *registrar* adds. Redefining it here silently retightened an
unrelated rule — `SkillService.register` began refusing a 201-character hint it
had always accepted — which is what a second constant with one name buys."""

MAX_PARAMETERS = 16
"""How many inputs one skill may declare. A procedure needing more than this has
not decided what it is for, and every one of them is a name the model must get
right from one line of `argument-hint`."""

PLACEHOLDER = re.compile(r"\{\{\s*parameters\.([A-Za-z0-9_-]+)\s*\}\}")
"""What an interpolated input looks like in a body: `{{parameters.version}}`.

Deliberately narrow. A skill body is prose *and* code samples, and code is full
of braces — so only this one prefix is touched, and anything else the author
wrote survives verbatim. The alternative, a general template language, would make
a `SKILL.md` unable to contain an example of itself."""

PARAMETER_TYPES = {"string": "string", "number": "number", "boolean": "boolean"}
"""What an author may declare, mapped to JSON Schema's own words.

Three, because a skill's input is something a model types into a tool call: a
list or an object is a shape the `argument-hint` cannot describe in one line, and
an author who needs one is describing a procedure, not an input."""

VERSION_PATTERN = re.compile(r"\A[0-9A-Za-z][0-9A-Za-z.+_-]{0,31}\Z")
"""What a `version` may look like. Deliberately looser than semver: pH does not
compare versions, it *reports* the one that won a shadowing contest, and refusing
`2024-01-a` would be this seam having an opinion it cannot act on."""

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
    version: str = ""
    """What the author called this revision. Empty when the file says nothing.

    Reported, never compared: `discover_skills` shadows by *precedence*, so a
    user's `code-review` wins over their distribution's whatever the versions
    say — and the question a person then has is which one they are running. It
    rides the `skill` tool's result rather than the catalog, because the catalog
    is in the prompt every turn and this is only interesting once."""
    argument_hint: str = ""
    """How the skill is invoked, in the shape `CommandDefinition.argument_hint`
    uses. This one *is* in the catalog: a skill the model cannot tell how to
    start is a skill it reads and then guesses at."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """The inputs this skill takes, as a JSON Schema object (P7-18).

    Authored in the friendlier per-input form — `type`, `required`, `default`,
    `enum`, `hint` — and converted here, so the validator is the one
    `validate_json_schema_value` already is rather than a second checker written
    for this seam.

    **Not in the catalog**, and that is the G9 trade made deliberately: a
    parameter list per skill in every prompt is paid for on every request, where
    a refusal naming the missing input is paid for only by the call that got it
    wrong. `argument-hint` is the author's one line for the common case; the
    schema is what makes the refusal exact."""
    allowed_tools: list[str] = Field(default_factory=list)
    """The tools this skill expects to use. A declaration, not an enforcement.

    pH already has the enforcing mechanism — `ctx.tools.restrict` — and
    deliberately does not wire it here: it is a *scope* boundary with a disposer,
    restrictions intersect, and a turn is not a scope, so two skills read in one
    turn could narrow to nothing with no moment to widen back. P7-18's executed
    skill is that scope.

    What it buys instead is `SkillValue.missing_tools`: the reader resolves this
    list against what the caller can actually see, so the model learns a tool is
    absent when it reads the skill rather than halfway through following it."""
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

    A flat list scanned per read makes **every agent pay for every other agent's
    narrowing**: a fan-out of sixteen children puts sixteen filters in front of the
    parent's own catalog, none of which can change what the parent sees. Keyed, a read
    walks `isolation_chain()` — the filters that can possibly apply and no others.
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
        # what this row found in five registries. Asked as a pair (P6-29) rather
        # than as two calls, so a disposed activation warns once rather than
        # twice — `owner_for` is the half that logs.
        # Through the *pair*, not `by.owner` (P6-29): the bucket is keyed by
        # the layer's isolation, so a restriction registered by a body running
        # for an agent would otherwise sit under that agent's key with its
        # disposer on the row — and stay there until the next `_changed()`
        # happened to sweep it. `Running.add_disposer` releases on whichever
        # scope ends first, which is the same reason `ToolRuntime._claim` uses
        # it for `_layers`.
        by = self.ctx.running_for(scope)
        bucket = self._restrictions.setdefault(by.layer.isolation, [])
        released = claim_entry(by, bucket, restriction, label="skill-restriction")

        self._changed()

        def release() -> None:
            released()
            self._changed()

        return release

    def _changed(self) -> None:
        """Bump the generation **and drop the caches**, as `ToolRuntime` does.

        Bumping alone made every entry stale without removing any, so `_reach`
        and `_restrictions` grew an entry per scope ever seen and never shrank —
        both keyed by `Context`, so a settled agent stayed reachable through its
        own cache line. Since P6-27 a child's `_reach` key is `(child, parent,
        None)` rather than `(child, None)`, which made a stale child entry pin
        its parent's context too.
        """
        self._generation += 1
        self._reach.clear()
        self._restrictions = {
            scope: value
            for scope, value in self._restrictions.items()
            if scope is None or scope.active
        }

    def reach(self, scope: Boundary) -> frozenset[str]:
        """Every skill name this boundary may use. The one place filters compose.

        **Stated, with no default** (P6-32). This resolved `scope or self.ctx`,
        and `self.ctx` is the mount — the *unrestricted* set — so an unstated
        boundary was not "no skills", it was all of them. P6-31 left that
        deliberately, because with `Context | None` there was no way for the
        callers who legitimately mean the deployment (the prompt catalog, a
        `ph doctor` probe) to say so, and making the seam refuse would have
        refused them too. `DEPLOYMENT` is that way, so the ambiguity is gone and
        the default can come off: a caller that states nothing now fails mypy.
        """
        target = boundary_of(scope, self.ctx)
        chain = tuple(target.isolation_chain())
        cached = self._reach.get(chain)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        # On the miss, before the table grows — `_changed` already sweeps dead
        # scopes out of `_restrictions`, and this is the same sweep for the memo
        # keyed by those scopes (I2).
        drop_dead_chains(self._reach)
        names = self._resolve(chain)
        self._reach[chain] = (self._generation, names)
        return names

    def _resolve(self, chain: tuple[Context | None, ...]) -> frozenset[str]:
        """What one isolation chain may reach: a pure function of the chain and the tables.

        Split out of `reach` so the memo and the check that audits it cannot hold
        two definitions of one fold — a second copy would let `stale_reach`
        confirm a set the pipeline never serves, which is the one failure an
        audit must not have. Purity is the property that makes it checkable at
        all: given the same chain, `_skills` and `_restrictions`, this returns the
        same set, so a cached answer that differs is drift and nothing else.

        **The filters are gathered once, not per name.** Written inside the
        per-name `all(...)` this did one dict lookup per skill per chain key to
        answer a question with no filters in it at all — 5.44 µs against 0.36 µs
        for twenty skills on an unnarrowed chain, which is every deployment-wide
        resolve and every agent nobody narrowed.
        """
        filters = [one for key in chain for one in self._restrictions.get(key, ())]
        if not filters:
            return frozenset(self._skills)
        return frozenset(name for name in self._skills if all(one.admits(name) for one in filters))

    def stale_reach(self) -> list[str]:
        """Cached reach sets that no longer equal a rebuild (I6, P6-01).

        Here rather than in the invariant row that declares it, for
        `ToolRuntime.stale_views`' reason: the cache is this class's own secret,
        and a check written against `_reach` from outside is one a rename
        disables without anybody noticing.

        **What a stale entry costs is the ceiling, not a listing.** `reach` is
        what decides which skills an agent may use, so an entry that outlived a
        `_changed()` keeps a skill reachable after the row owning it unloaded, or
        hands a narrowed child the set its parent holds — the widening P4-13b
        calls the whole security content of that row, and the one direction this
        registry is built to make impossible.

        Every cached entry is at the current generation, because `_changed`
        clears the table when it bumps the counter — so every one is an answer
        `reach` would serve, and every one is compared.
        """
        found: list[str] = []
        entries = list(self._reach.items())
        for chain, (_, cached) in entries:
            fresh = self._resolve(chain)
            if fresh != cached:
                found.append(
                    f"the cached skill reach for {chain_label(chain)} serves {sorted(cached)} "
                    f"where a rebuild gives {sorted(fresh)}"
                )
        # **And the ceiling itself, which comparing a fold against itself cannot
        # see.** Both sides above go through `_resolve`, so a fold that composed
        # restrictions *wrongly* would be confirmed, not caught — and the wrong
        # composition is the one this registry exists to prevent. The tool
        # registry shipped exactly that bug: "a child inherited its parent's
        # scoped tools and could not be narrowed out of them, so a grant could
        # widen a child but never bound it."
        #
        # Checkable here because two things hold for skills and not for tools:
        # `isolation_chain` is suffix-ordered, so an ancestor's chain is a
        # suffix of its descendant's; and a restriction only ever intersects
        # (`NameFilter` neither field widens). A deeper chain therefore applies a
        # superset of the filters, so its reach must be a subset. A tool view
        # has no such rule — a grandchild may legitimately hold what its
        # granting parent cannot see — which is why this clause is skills' own
        # and not copied from next door.
        for chain, (_, narrow) in entries:
            for ancestor, (_, wide) in entries:
                if (
                    len(ancestor) < len(chain)
                    and chain[-len(ancestor) :] == ancestor
                    and (extra := sorted(narrow - wide))
                ):
                    found.append(
                        f"{chain_label(chain)} reaches {extra}, which its ancestor "
                        f"{chain_label(ancestor)} does not — a narrowing widened"
                    )
        return found

    def admits(self, name: str, scope: Boundary) -> bool:
        """Whether `scope` may reach this skill at all."""
        return name in self.reach(scope)

    def list(self, scope: Boundary) -> list[Skill]:
        """What this boundary may use, sorted. The catalog and every gate read this."""
        reachable = self.reach(scope)
        return [self._skills[name] for name in sorted(reachable)]

    def get(self, name: str, scope: Boundary) -> Skill | None:
        return self._skills.get(name) if self.admits(name, scope) else None

    def body(self, name: str, scope: Boundary) -> str | None:
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
    # After the required fields, so a skill with a bad name is refused for its
    # name rather than for whatever the optional half noticed first.
    optional = _optional_front(front, path)
    if optional is None:
        return None
    return Skill(name=name, description=description, path=str(path), source=source, **optional)


def _optional_front(front: dict[str, Any], path: Path) -> dict[str, Any] | None:
    """The fields a `SKILL.md` may declare, or `None` with a reason logged.

    **Refused rather than dropped**, like the required fields above it: a skill
    whose `allowed-tools` is a string rather than a list would otherwise be
    installed with that field silently empty, and the author would have no way to
    tell the difference between "pH ignored it" and "pH does not support it".

    Kebab-case on the wire and snake_case on the model, which is the convention
    the format already follows for every multi-word key an author writes.
    """
    version = str(front.get("version") or "")
    if version and VERSION_PATTERN.match(version) is None:
        log.warning("ph.seams.skills: %s has an unusable version %r", path, version)
        return None

    hint = str(front.get("argument-hint") or "")
    if len(hint) > ARGUMENT_HINT_MAX:
        log.warning(
            "ph.seams.skills: %s has an argument-hint over %s chars", path, ARGUMENT_HINT_MAX
        )
        return None

    schema = _parameter_schema(front.get("parameters"), path)
    if schema is None:
        return None

    raw = front.get("allowed-tools")
    if raw is None:
        tools: list[str] = []
    elif isinstance(raw, list) and all(isinstance(one, str) for one in raw):
        tools = [one.strip() for one in raw if one.strip()]
    else:
        log.warning("ph.seams.skills: %s allowed-tools is not a list of names", path)
        return None
    if len(tools) > MAX_ALLOWED_TOOLS:
        log.warning("ph.seams.skills: %s names more than %s tools", path, MAX_ALLOWED_TOOLS)
        return None

    return {
        "version": version,
        "argument_hint": hint,
        "allowed_tools": tools,
        "parameters": schema,
    }


def _parameter_schema(declared: Any, path: Path) -> dict[str, Any] | None:
    """An author's `parameters:` block as a JSON Schema, or `None` if it is not one.

    Playbook's shape on the outside — one entry per input with `type`, `required`,
    `default`, `enum` — and JSON Schema on the inside, because pH already has a
    validator for that and a second one written here would be a second answer to
    "is this input acceptable".

    `additionalProperties: False` is the point of declaring at all: an argument
    the skill never heard of is a typo the model can fix, and silently ignoring
    it is how a `--dry-run` that was spelled `--dryrun` runs for real.
    """
    if declared is None:
        return {}
    if not isinstance(declared, dict):
        log.warning("ph.seams.skills: %s parameters is not a mapping", path)
        return None
    if len(declared) > MAX_PARAMETERS:
        log.warning("ph.seams.skills: %s declares more than %s parameters", path, MAX_PARAMETERS)
        return None

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, spec in declared.items():
        if NAME_PATTERN.match(str(name)) is None:
            log.warning("ph.seams.skills: %s has an invalid parameter name %r", path, name)
            return None
        if not isinstance(spec, dict):
            log.warning("ph.seams.skills: %s parameter %r is not a mapping", path, name)
            return None
        kind = PARAMETER_TYPES.get(str(spec.get("type") or "string"))
        if kind is None:
            log.warning(
                "ph.seams.skills: %s parameter %r has an unsupported type %r",
                path,
                name,
                spec.get("type"),
            )
            return None
        entry: dict[str, Any] = {"type": kind}
        if spec.get("hint"):
            entry["description"] = str(spec["hint"])
        choices = spec.get("enum")
        if choices is not None:
            if not isinstance(choices, list) or not choices:
                log.warning("ph.seams.skills: %s parameter %r has an unusable enum", path, name)
                return None
            entry["enum"] = list(choices)
        if "default" in spec:
            entry["default"] = spec["default"]
        elif spec.get("required"):
            required.append(str(name))
        properties[str(name)] = entry

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def rendered_body(body: str, skill: Skill, arguments: dict[str, Any]) -> str:
    """The skill's instructions with its inputs filled in (P7-18).

    Raises `ValueError` — which the `skill` tool's body turns into a result the
    model reads — when the arguments do not satisfy the declaration, or when the
    *body* names an input the frontmatter never declared. That second one is an
    author's typo, and it surfaces here rather than at load because G9 keeps the
    body on disk until something asks for it: there is no earlier moment that has
    both halves in hand.

    Defaults are applied before validation so a declared default satisfies a
    `required`-looking read, and after it the body sees a value for every
    declared name — an author writing `{{parameters.tag}}` never has to think
    about whether the caller passed one.
    """
    from ..tools.json_schema import validate_json_schema_value

    declared = dict(skill.parameters.get("properties") or {})
    if not declared and not arguments:
        return body
    values = {
        name: spec["default"]
        for name, spec in declared.items()
        if isinstance(spec, dict) and "default" in spec
    }
    values.update(arguments)
    violations = validate_json_schema_value(skill.parameters, values)
    if violations:
        # One validator, not two. An unknown argument is `additionalProperties:
        # False` doing its job — an explicit membership check beside it was a
        # second answer to one question, and the one that ran first, so the
        # shared validator's rule was unreachable and untested.
        raise ValueError(
            f"skill {skill.name!r} arguments: {'; '.join(violations)}. "
            f"It declares: {', '.join(sorted(declared)) or 'none'}"
        )

    missing: list[str] = []

    def fill(found: re.Match[str]) -> str:
        name = found.group(1)
        if name not in declared:
            missing.append(name)
            return found.group(0)
        return str(values.get(name, ""))

    filled = PLACEHOLDER.sub(fill, body)
    if missing:
        raise ValueError(
            f"skill {skill.name!r} refers to undeclared parameter(s) "
            f"{', '.join(sorted(set(missing)))} in its body"
        )
    return filled


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
        usage = f" Usage: `{skill.name} {skill.argument_hint}`." if skill.argument_hint else ""
        lines.append(f"- **{skill.name}** — {skill.description}{hint}{usage}")
    return "\n".join(lines)


class SkillArgs(ToolModel):
    name: str = Field(description="The skill's name, exactly as the catalog spells it.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Inputs the skill declared, if any. A skill that takes none ignores this; "
            "one that does refuses an unknown name, a missing required input, or a "
            "value outside its declared choices, and names what it wanted."
        ),
    )


class SkillValue(ToolModel):
    name: str
    path: str | None = None
    instructions: str
    version: str = ""
    """Which revision these instructions came from, when the file names one."""
    allowed_tools: list[str] = Field(default_factory=list)
    """What the skill says it needs, told at the moment the model starts it."""
    parameters: list[str] = Field(default_factory=list)
    """The inputs this skill declares, so a caller that guessed wrong the first
    time can get it right on the second without opening the file."""
    missing_tools: list[str] = Field(default_factory=list)
    """Which of those this caller cannot reach — resolved, not just echoed.

    An echoed list leaves the model to diff it against its own catalog. Resolved
    against `run.scope`, it answers the question the declaration was for: can I
    actually follow these instructions from here?"""


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
        # The *stated* boundary, so a narrowed child is refused a skill it can
        # see named nowhere — the catalog and the gate answer the same question,
        # and until P6-24 they answered it in two: the catalog above reads
        # `request.scope` while this read `getattr(run.agent, "ctx", None)`, the
        # approval-routing target. `run.scope` is a non-optional `Context` and
        # was sitting two lines away, which is the shape `fs_tools` had.
        scope = run.scope
        skill = ctx.skills.get(args.name, scope)
        body = ctx.skills.body(args.name, scope)
        if skill is None or body is None:
            # Named, because "unknown skill" and "this skill has no readable
            # body" are different problems for the model: one is a typo it can
            # fix from the catalog, the other is a deployment fault it cannot.
            known = ", ".join(one.name for one in ctx.skills.list(scope)) or "none"
            raise ValueError(f"no readable skill named {args.name!r}; available: {known}")
        # Rendered, not raw: a declared input is filled in before the model reads
        # the instruction that uses it, and arguments that do not satisfy the
        # declaration are refused here rather than discovered three steps into a
        # procedure that has already changed something.
        instructions = rendered_body(body, skill, dict(args.arguments))
        return {
            "name": skill.name,
            "path": skill.path,
            "instructions": instructions,
            "parameters": sorted(skill.parameters.get("properties") or {}),
            "version": skill.version,
            "allowed_tools": list(skill.allowed_tools),
            # Resolved against the same scope the body was fetched with, so the
            # catalog, the gate and this answer one question in one voice.
            "missing_tools": [
                name for name in skill.allowed_tools if ctx.tools.get(name, scope=scope) is None
            ],
        }

    def build_tool() -> Any:
        """The tool, or `None` where there is nothing to read.

        See `register_when_composed`: a `skill` tool in a deployment that
        installed no skills is a prompt-sized advertisement for a capability
        that is not there.
        """
        # `DEPLOYMENT`, and it is the right answer rather than the convenient one
        # (P6-32): the question is whether this *deployment* installed any skills
        # at all, asked once at mount to decide whether the tool exists. It is
        # not a per-agent question and there is no agent yet to ask it for.
        if not ctx.skills.list(DEPLOYMENT):
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
