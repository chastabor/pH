"""`ctx.skills`: the capability layer, and the boundary that it must not cross.

A skill is a package that a *distribution or a user* installed. The model cannot
install one, and `/refine` cannot create one. That is invariant I7. It is also
the reason that skills and the Continual Harness share a word but not a
mechanism: the knowledge layer writes *procedure*, never *capability* (Q13).

Progressive disclosure is G9, and it is implemented here too, as the
`skills-progressive` row at the bottom of this file. The catalog goes in the
prompt. The body stays on disk until the model asks for it by name. All of this
is one module because "what a skill is" is one question: the format on disk,
the limits, the registry, and what the model is told about the skill.

**`skills-progressive` scans nothing by default.** A deployment names its
directories. A well-known path that was scanned at every start would make
"install a skill" mean "drop a file in a directory", and I7 draws the capability
boundary between those two. A row that ships skills with its package installs
them on purpose, with `offer_skills`.

@module ph.seams.skills
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import anyio
from pydantic import Field

from ..cordis import (
    DEPLOYMENT,
    Boundary,
    ChildLimit,
    ChildReach,
    Context,
    Disposer,
    LoaderError,
    NarrowingRefused,
    ScopedEntries,
    boundary_of,
    chain_label,
    drop_dead_chains,
    events,
    plugin,
    safe_yaml_load,
)
from ..json import JsonValue, as_str, dumps
from ..keys import SKILLS, SYSTEM_PROMPT, TOOLS
from ..llm.structured import structural_warning
from ..session import Session
from ..session.writers import log_writer
from ..system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ..tools.definition import (
    ToolDefinition,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    define_tool,
    text_content,
)
from ..tools.json_schema import checked_object_schema, validate_json_schema_value
from ..tools.presentation import simple_views
from ..tools.registry import register_when_composed
from ..wire import WireModel
from ._names import require_slug, slug_pattern
from ._registry import claim_key, contribute_item
from ._restriction import NameFilter
from .models import KEY_MAX

__all__ = [
    "ARGUMENT_HINT_MAX",
    "FRONTMATTER_MAX",
    "HINT_MAX",
    "MAX_SKILL_BYTES",
    "NAME_PATTERN",
    "READ",
    "SKILL_FILE",
    "Skill",
    "SkillReadVia",
    "SkillRestriction",
    "SkillService",
    "apply",
    "discover_skills",
    "offer_skills",
    "progressive",
    "read_skill",
    "read_summary",
    "record_read",
    "render_catalog",
    "rendered_skill",
]

log = logging.getLogger("ph.seams.skills")

_LOG = log_writer(__name__)

READ = "skill/read"
"""The body of a skill was read at runtime. The event exists for the audit
(session profiles, S8, decision 12). It records the skill, its version, and the
hash of exactly what was read. `via` says how the body was read: by the `skill`
tool, or into the prompt of a child at a spawn that names the skill. The event
is ignorable. It never blocks a task and never makes a restart ask, because
skills are meant to keep being refined. The base of a session records only the
directories that skills are found in."""


def read_summary(data: Mapping[str, JsonValue]) -> str:
    """One `skill/read` as an audit reads it: `review 2.0 sha256:0a1b2c3d4e5f (tool)`."""
    version = as_str(data.get("version")) or "no version"
    digest = as_str(data.get("sha256"))[:12]
    return f"{as_str(data.get('name'))} {version} sha256:{digest} ({as_str(data.get('via'))})"


SkillReadVia: TypeAlias = Literal["tool", "brief"]
"""How the body of a skill reached an agent: the `skill` tool, or the brief of a
spawn."""


def record_read(session: Session | None, skill: Skill, body: str, *, via: SkillReadVia) -> None:
    """Record that `body` reached an agent. `body` is the body of `skill` as it
    was read just now.

    The body is hashed as read. Two runs of a session can then tell whether
    they followed the same text, even when the author never changed `version`.
    Nothing is recorded for an agent that has no session to log to.
    """
    if session is None:
        return
    _LOG.append(
        session,
        READ,
        {
            "name": skill.name,
            "version": skill.version,
            "path": skill.path or "",
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "via": via,
        },
    )


NAME_MAX = 64
DESCRIPTION_MAX = 1_024
HINT_MAX = 256
"""A hint is one clause after a description, and it reaches every prompt in the
same way as the description. So it is bounded for the same reason, at the length
that a clause needs."""
SKILL_FILE = "SKILL.md"
MAX_SKILL_BYTES = 10 * 1024 * 1024
"""The third limit of G9. Two places enforce it, for two different reasons. The
scanner refuses a file this large, so that one file cannot poison a catalog.
`SkillService.body` refuses to *return* one, so that a skill that grew after
discovery cannot arrive in a context window."""
FRONTMATTER_MAX = 64 * 1024
"""How much of a `SKILL.md` validation reads. The frontmatter that validation
needs is small: `name` and `description` are capped at about a kilobyte
together, and each list has a cap on its entries. The body, up to 10 MiB, stays
on disk by design, so reading it here would pull in exactly the bytes that this
row exists to defer."""

MAX_ALLOWED_TOOLS = 32
"""How many tools one skill may name. The list is bounded for the same reason as
every other list that reaches the model: it reaches a prompt. A skill that names
two hundred tools has not decided what it does."""

ARGUMENT_HINT_MAX = 200
"""How long `argument-hint` may be. The hint is part of the catalog, and the
catalog is in the prompt every turn. So every request in the session pays for a
paragraph here.

The constant has its own name and does not reuse `HINT_MAX`. `HINT_MAX` is
defined above and bounds `Skill.hint`, the clause that a *registrar* adds. An
earlier version redefined `HINT_MAX` here, and that silently tightened an
unrelated rule: `SkillService.register` began to refuse a 201-character hint
that it had always accepted. That is the cost of a second constant with the same
name."""

MAX_STEPS = 32
"""How many steps one skill may declare. A procedure longer than this is two
procedures. Every step is also an entry that the model cannot delete, so the cap
also bounds how much of someone's plan a skill may take over."""

MAX_ACCEPTS = 16
"""The largest number of media patterns that one skill may state it hands to its
child (`accepts`)."""

MEDIA_PATTERN = re.compile(r"[a-z]+/(\*|[a-z0-9][a-z0-9.+-]*)")
"""A media type that a skill states (`image/png`), or a whole family of them
(`image/*`)."""

MAX_PARAMETERS = 16
"""How many inputs one skill may declare. A procedure that needs more than this
has not decided what it is for. The model must also get the name of every input
right from one line of `argument-hint`."""

PLACEHOLDER = re.compile(r"\{\{\s*parameters\.([A-Za-z0-9_-]+)\s*\}\}")
"""The form of an interpolated input in a body: `{{parameters.version}}`.

The pattern is narrow on purpose. A skill body is prose *and* code samples, and
code is full of braces. So only this one prefix is replaced, and everything else
that the author wrote stays verbatim. The alternative is a general template
language, and with one a `SKILL.md` could not contain an example of itself."""

PARAMETER_TYPES = {"string": "string", "number": "number", "boolean": "boolean"}
"""The types that an author may declare, mapped to the words of JSON Schema.

There are three, because the input of a skill is something that a model types
into a tool call. The `argument-hint` cannot describe a list or an object in one
line, and an author who needs one is describing a procedure, not an input."""

VERSION_PATTERN = re.compile(r"\A[0-9A-Za-z][0-9A-Za-z.+_-]{0,31}\Z")
"""The form that a `version` may have. It is looser than semver on purpose. pH
does not compare versions. It *reports* the version of the skill that won a
shadowing contest. If this seam refused `2024-01-a`, it would hold an opinion
that it cannot act on."""

events.declare(
    "skills/read",
    "emit",
    owner="ph.seams.skills",
    doc="A skill's body was read. Carries the skill, its steps with the "
    "caller's inputs already filled in, its own nudge budget (or None), and the "
    "session — so a row that turns a declared procedure into work can do so "
    "without this seam knowing what a todo list is, and without a listener "
    "needing the arguments to render them.",
)

ORDER_SKILLS = ORDER_TOOL_GUIDANCE + 50

FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---", re.S)
NAME_PATTERN = slug_pattern(NAME_MAX)
"""The one format of a skill name, at the bound of this seam.

`read_skill` *tests* a name against the pattern and does not raise. For a bad
frontmatter block it logs a warning and skips that skill, and the scan
continues. `SkillService.register` raises for the same name (`require_slug`).
Every reader of `SKILL.md` goes through `read_skill`: `skills-progressive`,
`rlm-skills-python` and `offer_skills`, all through `discover_skills`. So the
pattern is a second holder of the rule, and a test holds it equal to what
`register` enforces. If the two differed, a catalog would hold names that
another reader refuses, and there would be two definitions of "valid skill" that
drift apart. The format itself belongs to `_names`, which shares it with the
screen ids, the same kind of token."""

MODEL_KEY = slug_pattern(KEY_MAX)
"""What a `model:` may name: a key of the `models` list of a profile, by the rule
of that list (`ph.seams.models.Config`). A key that no list could hold is
therefore refused at discovery, and not at the spawn that reads it. An `effort:`
is held to the same rule, because a route names its `efforts` by it
(`ModelRoute`)."""


class Skill(WireModel):
    """One installed skill: a name, a one-line description, and a body on disk."""

    name: str
    description: str
    path: str | None = None
    source: str = "installed"
    version: str = ""
    """The name that the author gave this revision. Empty when the file says
    nothing.

    The version is reported and never compared. `discover_skills` shadows by
    *precedence*, so the `code-review` of a user wins over the one of their
    distribution, whatever the versions say. The user then wants to know which
    one is running. The version is in the result of the `skill` tool and not in
    the catalog, because the catalog is in the prompt every turn and the version
    is interesting only one time."""
    argument_hint: str = ""
    """How the skill is invoked, in the form that
    `CommandDefinition.argument_hint` uses. This field *is* in the catalog. When
    the model cannot tell how to start a skill, it reads the skill and then
    guesses."""
    steps: list[str] = Field(default_factory=list)
    """An ordered procedure that this skill is, as opposed to advice that it
    gives (P7-18).

    The steps are seeded into the todo list when the skill is read. The model
    may mark these entries done and may not delete them. That makes a *skill* a
    procedure and not a suggestion, and it is what a listener on
    `agent/turn-stopping` then steers against.

    **The steps are implicitly sequential.** Each step waits on the step before
    it, through the `requires` field that `tool-todo` already has. A step list
    whose order does not matter is a checklist, and an author who needs a real
    graph writes the entries by hand. This field is for the common case: "do
    these in order".

    The field is declared here and consumed in `ph-stabilize`. This seam cannot
    cross that package boundary. So reading a skill emits `skills/read` and does
    not reach for a todo list, which ph-core knows nothing about."""
    model: str = ""
    """The listed model that a child directed by this skill runs on, by key
    (session profiles, S7b). An example is `model: classify` for a skill that
    sorts and does not write. The key is read when a spawn names this skill. The
    profile of the parent holds the key. A key that the parent cannot run
    degrades: the child runs on the model of its parent, and the parent is told
    (`ModelFallback`). Empty says nothing."""
    effort: str = ""
    """The reasoning effort that a child directed by this skill runs at: a name
    that the `efforts` of its route define, or `default` (`DEFAULT_EFFORT`). It
    takes precedence over the effort that the route sets, and an effort that the
    spawn names takes precedence over it (`child_options`). Empty says
    nothing."""
    accepts: list[str] = Field(default_factory=list)
    """The media that this skill hands to a child it directs, as media types or
    families (`image/*`). The model of the child must accept them for the child
    to see them (A3). The list is stated apart from `model`, which names the
    job, so a cheap classifier that takes images is one key and one list. A
    child whose route cannot take the media runs on the route of its parent if
    that route can. Otherwise the child sees a text placeholder for each one,
    and its parent is told. Empty says nothing."""
    max_nudges: int | None = None
    """How many times `skill-steps` may steer toward the steps of this skill
    without the plan moving, or `None` for the `maxNudges` of the row (D16).

    The author sets it, because the author knows what a partial run is worth. A
    release checklist abandoned at step three is worse than one that spent a
    few more model calls to finish. A survey that stops early has still produced
    something: `0` says so, and it seeds the steps without ever steering. The
    field is declared here and read in `ph-stabilize`, the same package crossing
    that `steps` makes."""
    output: dict[str, Any] = Field(default_factory=dict)
    """The value that a child directed by this skill returns, as a JSON Schema
    object. An example is the `{"type": "object", "properties": {"kind": ...}}`
    of a categorizing skill. A child that is given the schema finishes by
    returning a value that matches, and fails if it cannot
    (`ph.seams.subagents.settle_value`). Empty says nothing.

    The schema is written as JSON Schema itself and not in the friendlier form
    of `parameters`, because a value is a structure (lists, nested objects) and
    an input is one typed word. It is held to what pH checks: the root is an
    object, and it has no keyword that the validator would skip
    (`object_schema_problems`)."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    """The inputs that this skill takes, as a JSON Schema object (P7-18).

    The author writes them in the friendlier per-input form: `type`, `required`,
    `default`, `enum`, `hint`. They are converted here. The validator is then
    `validate_json_schema_value`, which exists already, and not a second checker
    written for this seam.

    **The parameters are not in the catalog.** That is the G9 trade, made on
    purpose. Every request pays for a parameter list per skill in every prompt.
    Only the call that got an input wrong pays for a refusal that names the
    missing input. `argument-hint` is the one line that the author has for the
    common case. The schema makes the refusal exact."""
    allowed_tools: list[str] = Field(default_factory=list)
    """The tools that this skill expects to use. This is a declaration, not an
    enforcement.

    pH already has the mechanism that enforces, `ctx.tools.restrict`, and does
    not wire it here on purpose. A restriction is a *scope* boundary with a
    disposer, restrictions intersect, and a turn is not a scope. So two skills
    read in one turn could narrow to nothing, with no moment to widen back. The
    executed skill of P7-18 is that scope.

    The benefit of the declaration is `SkillValue.missing_tools`. The reader
    resolves this list against what the caller can see. The model then learns
    that a tool is absent when it reads the skill, and not halfway through
    following it."""
    hint: str = ""
    """One extra clause that the catalog prints after the description.

    It holds what a *particular registrar* knows and the catalog cannot derive.
    `rlm-skills-python` installs the package of a skill into the kernel venv.
    For those skills there is a second way to use one: `await name(...)` in a
    cell. A model that is told only "read the file" would never find it. This is
    a field and not a second catalog, because with two catalogs a deployment
    tells the model about the same skill twice in one prompt."""


SkillRestriction = NameFilter
"""A per-scope filter over installed skills. Restrictions intersect.

It is the same `NameFilter` that `ctx.tools` narrows with. With one rule, a
change to the meaning of `allow` and `deny` cannot reach one registry and miss
the other. **The design depends on the word "intersect".** A restriction can
only subtract. So a scope cannot grant itself a skill by adding one, and a child
under a restricted parent cannot climb back out (P4-13b).
"""


@dataclass(slots=True)
class SkillService:
    """The service published as `ctx.skills`.

    **Installation is global. Only *reach* is scoped.** The `scope=` of
    `register` is a lifetime, not a visibility. It says which row takes the
    skill away when that row unloads, and until then the skill is in the catalog
    of every agent. Narrowing is a separate verb (`restrict`). P4-13b states the
    reason as its whole security content: a child may hold a subset of its
    parent and never more. So the registry offers subtraction and no addition.
    There is no way to give one agent a skill that another cannot see, and that
    is deliberate. To give a child a skill, install it, and the parent gets it
    too.
    """

    ctx: Context
    _skills: dict[str, Skill] = field(default_factory=dict)
    _registrants: dict[str, str] = field(default_factory=dict)
    """The profile row of each installed skill (`Context.row_id`), as
    `ToolRuntime` keeps the row of a tool. It says which row gave the skill, for
    a narrowing that drops that row (S7b)."""
    _restrictions: ScopedEntries[SkillRestriction] = field(default_factory=ScopedEntries)
    """Filters, keyed by the scope that they belong to (`ScopedEntries`), as
    `ctx.tools` keeps its layers.

    With a flat list scanned per read, **every agent pays for the narrowing of
    every other agent**. A fan-out of sixteen children puts sixteen filters in
    front of the catalog of the parent, and none of them can change what the
    parent sees. With keys, a read walks `isolation_chain()`: the filters that
    can possibly apply, and no others.
    """
    _reach: dict[tuple[Context | None, ...], tuple[int, frozenset[str]]] = field(
        default_factory=dict
    )
    """What each chain may reach, memoized against `_generation`.

    This is the arrangement of `ToolRuntime.view`, for the same reason. The
    answer is read one time per skill per prompt assembly, and it changes only
    when something registers or unloads.
    """
    _generation: int = 0

    def register(self, skill: Skill, *, scope: Context | None = None) -> Disposer:
        """Install a skill. Bounds are enforced here so a catalog stays a catalog."""
        require_slug(skill.name, maximum=NAME_MAX, kind="skill name")
        if len(skill.description) > DESCRIPTION_MAX:
            raise ValueError(f"a skill description must be at most {DESCRIPTION_MAX} characters")
        if len(skill.hint) > HINT_MAX:
            raise ValueError(f"a skill hint must be at most {HINT_MAX} characters")
        owner = self.ctx.owner_for(scope)

        def released() -> None:
            # This runs inside the disposer that the scope holds (the `then` of
            # `claim_key`). So a row that unwinds removes its registrant and
            # also tells the catalog.
            self._registrants.pop(skill.name, None)
            self._changed()

        release = claim_key(owner, self._skills, skill.name, skill, label="skill", then=released)
        if owner.row_id:
            self._registrants[skill.name] = owner.row_id
        self._changed()
        return release

    def registrants(self) -> dict[str, str]:
        """Each installed skill by name, with the profile row that it lives
        with. That is the row that registered it, or the row of this seam for a
        skill registered with no scope outside the `apply` of any row."""
        return dict(self._registrants)

    def restrict(self, restriction: SkillRestriction, *, scope: Context | None = None) -> Disposer:
        """Narrow what one scope may see. Filters along the chain intersect.

        Pass `scope=`. A filter that the context of this service owns applies to
        every agent. That is a deployment-wide policy, and almost never what a
        caller that narrows one child meant.
        """
        # There are two questions with two answers (P6-12). The bucket is
        # *which scope this narrows*, and it must stay where the caller put it.
        # The disposer is *when the narrowing lifts*, and that is the row that
        # asked. P6-12 found one value answering both questions in five
        # registries. The two are asked as a pair (P6-29) and not as two calls,
        # so a disposed activation warns one time and not two times:
        # `owner_for` is the half that logs.
        # The release goes through the *pair*, not `by.owner` (P6-29). The
        # bucket is keyed by the isolation of the layer. A restriction that a
        # body running for an agent registers would otherwise sit under the key
        # of that agent with its disposer on the row, and stay there.
        # `Running.add_disposer` releases on whichever scope ends first, which
        # is the reason that `ToolRuntime._claim` uses it for `_layers`. The
        # release also announces itself (`then`), so a child that unwinds takes
        # its narrowing out of the memo too.
        by = self.ctx.running_for(scope)
        release = self._restrictions.add(
            by, restriction, label="skill-restriction", then=self._changed
        )
        self._changed()
        return release

    def _changed(self) -> None:
        """Bump the generation **and drop the caches**, as `ToolRuntime` does.

        Bumping alone made every entry stale and removed none. `_reach` then
        grew by one entry per scope ever seen and never shrank. The table is
        keyed by `Context`, so a settled agent stayed reachable through its own
        cache entry. Since P6-27 the `_reach` key of a child is
        `(child, parent, None)` and not `(child, None)`, so a stale child entry
        also kept the context of its parent alive. `_restrictions` empties
        itself: a scope goes when its last filter goes.
        """
        self._generation += 1
        self._reach.clear()

    def reach(self, scope: Boundary) -> frozenset[str]:
        """Every skill name that this boundary may use. Filters compose only
        here.

        **The boundary is stated, with no default** (P6-32). This method used to
        resolve `scope or self.ctx`. `self.ctx` is the mount, which is the
        *unrestricted* set. So an unstated boundary did not mean "no skills": it
        meant all of them. P6-31 left that on purpose. With `Context | None`,
        the callers that legitimately meant the deployment (the prompt catalog,
        a `phern doctor` probe) had no way to say so, and a seam that refused
        would have refused them too. `DEPLOYMENT` is that way. The ambiguity is
        gone and the default is removed: a caller that states nothing now fails
        mypy.
        """
        target = boundary_of(scope, self.ctx)
        chain = tuple(target.isolation_chain())
        cached = self._reach.get(chain)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        # This runs on the miss, before the table grows. `_restrictions` drops a
        # scope with its last filter, and this gives the same care to the memo
        # that is keyed by those scopes (I2).
        drop_dead_chains(self._reach)
        names = self._resolve(chain)
        self._reach[chain] = (self._generation, names)
        return names

    def _resolve(self, chain: tuple[Context | None, ...]) -> frozenset[str]:
        """What one isolation chain may reach: a pure function of the chain and the tables.

        It is split out of `reach` so that the memo and the check that audits it
        cannot hold two definitions of one fold. With a second copy,
        `stale_reach` could confirm a set that the pipeline never serves, and
        that is the one failure that an audit must not have. Purity makes the
        check possible. Given the same chain, `_skills` and `_restrictions`,
        this returns the same set. So a cached answer that differs is drift and
        nothing else.

        **The filters are gathered one time, not per name.** An earlier version
        gathered them inside the per-name `all(...)`. It did one dict lookup per
        skill per chain key to answer a question that had no filters in it at
        all. That took 5.44 µs against 0.36 µs for twenty skills on an
        unnarrowed chain, and an unnarrowed chain is every deployment-wide
        resolve and every agent that nobody narrowed.
        """
        filters = self._restrictions.gathered(chain)
        if not filters:
            return frozenset(self._skills)
        return frozenset(name for name in self._skills if all(one.admits(name) for one in filters))

    def stale_reach(self) -> list[str]:
        """Cached reach sets that no longer equal a rebuild (I6, P6-01).

        The check is here and not in the invariant row that declares it, for the
        reason that `ToolRuntime.stale_views` gives. The cache is private to
        this class, and a rename would silently disable a check written against
        `_reach` from outside.

        **The cost of a stale entry is the ceiling, not a listing.** `reach`
        decides which skills an agent may use. An entry that outlived a
        `_changed()` keeps a skill reachable after the row that owns it
        unloaded. Or it hands a narrowed child the set that its parent holds.
        P4-13b calls that widening the whole security content of that row, and
        it is the one direction that this registry is built to make impossible.

        Every cached entry is at the current generation, because `_changed`
        clears the table when it bumps the counter. So every entry is an answer
        that `reach` would serve, and every entry is compared.
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
        # **The ceiling itself is also checked, because a fold compared against
        # itself cannot see it.** Both sides above go through `_resolve`. A fold
        # that composed restrictions *wrongly* would therefore be confirmed, not
        # caught, and the wrong composition is the one that this registry exists
        # to prevent. The tool registry shipped exactly that bug: a child
        # inherited the scoped tools of its parent and could not be narrowed out
        # of them, so a grant could widen a child but never bound it.
        #
        # The check is possible here because two things hold for skills and not
        # for tools. `isolation_chain` is suffix-ordered, so the chain of an
        # ancestor is a suffix of the chain of its descendant. And a restriction
        # only ever intersects (neither field of `NameFilter` widens). A deeper
        # chain therefore applies a superset of the filters, so its reach must
        # be a subset. A tool view has no such rule: a grandchild may
        # legitimately hold what its granting parent cannot see. That is why
        # this clause belongs to skills alone and is not copied from the tool
        # registry.
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
        """The full text of the skill, read only when it is asked for (G9).

        The size is capped again here, and discovery is not trusted. A file is
        validated one time at mount and read whenever the model asks. A skill
        directory that is a checkout can be updated with `git pull` between
        those two moments. The result is `None` for an unknown name or for a
        body that no longer fits, because both mean "there is nothing to hand
        you".

        **This is a read, not a record of one** (S8). A `skill/read` says that a
        body reached an agent, and only the reader knows that: the `skill` tool
        when the arguments render, a spawn when its child is admitted. So each
        reader records with `record_read` when it hands the body over.
        """
        skill = self.get(name, scope)
        if skill is None or skill.path is None:
            return None
        path = Path(skill.path)
        try:
            if path.stat().st_size > MAX_SKILL_BYTES:
                log.warning("ph.seams.skills: %s grew past %s bytes", path, MAX_SKILL_BYTES)
                return None
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            log.warning("ph.seams.skills: %s could not be read: %s", path, error)
            return None
        return text


@plugin("skills", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the skills seam."""
    ctx.provide(SKILLS, SkillService(ctx=ctx))


# ------------------------------------------------------- the format on disk --


def offer_skills(ctx: Context, root: Path, *, source: str) -> None:
    """Install the `SKILL.md` files under `root` as the skills of `source`, if a
    skills registry is mounted. This is how a package ships skills.

    `discover_skills` globs `<root>/*/SKILL.md` and validates each file. It
    skips a malformed one and logs the reason. Each skill arrives through
    `contribute_item`, so it arrives exactly when the row that offers it does,
    and it leaves with that row. A row offers its skills on purpose, and that
    keeps the empty `paths` of `skills-progressive` meaningful: installing a
    skill is something that a distribution does deliberately (I7). Only the
    one-line description is in the prompt every turn. The body stays on disk
    until the model asks for it by name (G9).
    """
    for skill in discover_skills([str(root)], source=source):
        contribute_item(ctx, SKILLS, skill, label=f"skill({skill.name})")


def read_skill(path: Path, *, source: str = "skills-progressive") -> Skill | None:
    """One `SKILL.md`, validated. `None`, with a reason logged, if the file is
    not a valid one.

    **A bad skill is refused and not half-accepted.** A skill whose name does
    not match its directory would be addressed by one string in the catalog and
    found under another on disk, and a confused model would report that as a bug
    later. A bad skill is skipped and is not fatal, because one malformed
    directory must not cost a deployment every other skill that it installed.
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
        # `safe_yaml_load` is the one YAML policy of the codebase: no tags, and
        # no implicit date coercion. `description: 2024-01-01` stays the string
        # that the author wrote and does not become a datetime that `str()`
        # reshapes.
        front = safe_yaml_load(matched.group(1), origin=str(path)) or {}
    except LoaderError as error:
        log.warning("ph.seams.skills: %s has invalid frontmatter: %s", path, error)
        return None
    if not isinstance(front, dict):
        log.warning("ph.seams.skills: %s frontmatter is not a mapping", path)
        return None

    name = as_str(front.get("name"))
    description = as_str(front.get("description"))
    if NAME_PATTERN.match(name) is None:
        log.warning("ph.seams.skills: %s has an invalid name %r", path, name)
        return None
    if name != directory.name:
        log.warning("ph.seams.skills: %s is named %r but lives in %r", path, name, directory.name)
        return None
    if not description or len(description) > DESCRIPTION_MAX:
        log.warning("ph.seams.skills: %s needs a description of 1..%s chars", path, DESCRIPTION_MAX)
        return None
    # This comes after the required fields, so a skill with a bad name is
    # refused for its name and not for what the optional half noticed first.
    optional = _optional_front(front, path)
    if optional is None:
        return None
    return Skill(name=name, description=description, path=str(path), source=source, **optional)


def _optional_front(front: dict[str, Any], path: Path) -> dict[str, Any] | None:
    """The fields that a `SKILL.md` may declare, or `None` with a reason logged.

    **A bad field is refused and not dropped**, as with the required fields in
    `read_skill`. Otherwise a skill whose `allowed-tools` is a string and not a
    list would be installed with that field silently empty. The author could not
    tell "pH ignored it" from "pH does not support it".

    The keys are kebab-case on the wire and snake_case on the model. The format
    already follows that convention for every multi-word key that an author
    writes.
    """
    version = as_str(front.get("version"))
    if version and VERSION_PATTERN.match(version) is None:
        log.warning("ph.seams.skills: %s has an unusable version %r", path, version)
        return None

    hint = as_str(front.get("argument-hint"))
    if len(hint) > ARGUMENT_HINT_MAX:
        log.warning(
            "ph.seams.skills: %s has an argument-hint over %s chars", path, ARGUMENT_HINT_MAX
        )
        return None

    schema = _parameter_schema(front.get("parameters"), path)
    if schema is None:
        return None

    output = _output_schema(front.get("output"), path)
    if output is None:
        return None

    steps = _string_list(front, "steps", path, cap=MAX_STEPS)
    if steps is None:
        return None
    if len(set(steps)) != len(steps):
        # `requires` names entries by their content, and a seeded step is an
        # entry. Two identical steps would be one entry that the plan waits on
        # two times.
        log.warning("ph.seams.skills: %s repeats a step", path)
        return None

    tools = _string_list(front, "allowed-tools", path, cap=MAX_ALLOWED_TOOLS)
    if tools is None:
        return None

    nudges = front.get("max-nudges")
    # `bool` is checked first. The `true` of YAML is an `int` to `isinstance`,
    # and a budget of `true` nudges is a typo and not a number.
    if nudges is not None and (
        isinstance(nudges, bool) or not isinstance(nudges, int) or nudges < 0
    ):
        log.warning("ph.seams.skills: %s max-nudges is not a non-negative integer", path)
        return None

    model = _key(front, "model", path)
    if model is None:
        return None
    effort = _key(front, "effort", path)
    if effort is None:
        return None

    accepts = _string_list(front, "accepts", path, cap=MAX_ACCEPTS)
    if accepts is None:
        return None
    unknown = [one for one in accepts if not MEDIA_PATTERN.fullmatch(one)]
    if unknown:
        log.warning("ph.seams.skills: %s accepts what is not a media type: %s", path, unknown)
        return None

    return {
        "version": version,
        "model": model,
        "effort": effort,
        "accepts": accepts,
        "argument_hint": hint,
        "allowed_tools": tools,
        "parameters": schema,
        "output": output,
        "steps": steps,
        "max_nudges": nudges,
    }


def _key(front: dict[str, Any], key: str, path: Path) -> str | None:
    """A name that a profile defines, held to the rule of its `models` list
    (`MODEL_KEY`): a `model`, or an `effort` that the `efforts` of a route
    define. Absent is `""`."""
    value = as_str(front.get(key))
    if value and not MODEL_KEY.match(value):
        log.warning("ph.seams.skills: %s %s %r is not a key", path, key, value)
        return None
    return value


def _string_list(front: dict[str, Any], key: str, path: Path, *, cap: int) -> list[str] | None:
    """How a `SKILL.md` declares a list of short strings. One function serves
    `steps`, `allowed-tools` and `accepts`.

    The rule is the same for each key: a list of strings, trimmed and capped, or
    else the skill is refused. `steps` and `allowed-tools` were written out
    twice, and so they came to disagree about a blank entry: one refused the
    skill and the other dropped the entry silently. Refusing is the rule that
    the rest of the block already follows, so it is the rule that was kept. An
    empty step is a plan entry with no content, and an empty tool name matches
    nothing.

    Absent is not malformed: a key that nobody wrote is an empty list.
    """
    raw = front.get(key)
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(one, str) and one.strip() for one in raw):
        log.warning("ph.seams.skills: %s %s is not a list of non-empty strings", path, key)
        return None
    values = [one.strip() for one in raw]
    if len(values) > cap:
        log.warning("ph.seams.skills: %s declares more than %s entries in %s", path, cap, key)
        return None
    return values


def _parameter_schema(declared: object, path: Path) -> dict[str, Any] | None:
    """The `parameters:` block of an author as a JSON Schema, or `None` if it is
    not a valid block.

    The outside has the shape of Playbook: one entry per input with `type`,
    `required`, `default`, `enum`. The inside is JSON Schema, because pH already
    has a validator for that. A second validator written here would be a second
    answer to "is this input acceptable".

    `additionalProperties: False` is the purpose of declaring at all. An
    argument that the skill never heard of is a typo that the model can fix. If
    it were silently ignored, a `--dry-run` that was spelled `--dryrun` would
    run for real.
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
        kind = PARAMETER_TYPES.get(as_str(spec.get("type"), "string"))
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
            entry["description"] = as_str(spec["hint"])
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


def _output_schema(declared: object, path: Path) -> dict[str, Any] | None:
    """The `output:` block of an author, or `None` with the reason logged if pH
    cannot hold a value to it. Absent is `{}`. A schema that may not constrain
    the reply is kept, with a warning (`structural_warning`)."""
    if declared is None:
        return {}
    try:
        output = checked_object_schema(declared)
    except ValueError as problems:
        log.warning("ph.seams.skills: %s output cannot be checked: %s", path, problems)
        return None
    warning = structural_warning(output)
    if warning is not None:
        log.warning("ph.seams.skills: %s output: %s", path, warning)
    return output


def _rendered_value(value: object) -> str:
    """One declared argument, as the text of the skill should read it (D1).

    **`as_str` is the wrong narrowing here**, and its own contract says so. It
    answers `""` for anything that is not a string. That is right where a
    non-string is a *malformed* value and wrong where it is a declared one. A
    skill may declare `{"type": "number"}` or `{"type": "boolean"}`, and every
    such parameter rendered as nothing. That happened silently, in the
    instructions that the model reads and in every step seeded into its plan.
    The `Retry {{parameters.attempts}} times` of an author became
    `Retry  times`.

    A boolean is spelled as the schema that declared it spells it: `true` and
    `false`, not the `True` of Python. An author who writes
    `--strict={{parameters.strict}}` means the JSON value that they declared. A
    model that reads `True` in a shell flag has been handed a bug to find.

    `None` renders empty and not as `"None"`. That is what an author means by a
    declared-but-null value, and it is what `as_str` did for it before.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    # Everything else is rendered in its JSON spelling. That is `dumps` and not
    # four branches. `dumps` already renders `true`/`false` and not the `True`
    # of Python, numbers without the surprises of a repr, and a list or an
    # object as the author declared it and not with the single quotes of Python.
    return dumps(value)


def rendered_skill(body: str, skill: Skill, arguments: dict[str, Any]) -> tuple[str, list[str]]:
    """The instructions of the skill *and* its steps, with its inputs filled in
    (P7-18).

    One call returns both, because they are one document. `steps:` is the
    procedure that the body explains. An author who writes
    `Run {{parameters.gate}}` in a step means the same thing as in a paragraph.
    Rendering only the body would seed the *literal* `{{parameters.gate}}` into
    someone's todo list. There, unlike in a paragraph, the model cannot go back
    and read what it stood for.

    Raises `ValueError` when the arguments do not satisfy the declaration, or
    when the text names an input that the frontmatter never declared. The model
    reads the error as the result of the `skill` tool. The second case is a typo
    of the author. It surfaces here and not at load, because G9 keeps the body
    on disk until something asks for it: no earlier moment has both halves in
    hand.

    **`body` is the whole file, frontmatter included** (`SkillService.body`). So
    the scan for undeclared names already covers a `steps:` block without
    looking at one. That is why there is no separate gate for that case and no
    separate message that names it. The step *list* must still be filled in
    here, because it was parsed at load and is a second copy of the same
    sentences. Unrendered, the todo list and the instructions would spell one
    procedure in two ways.

    **There is no shortcut for a skill that declares nothing.** There was one:
    an early return handed such a body straight back. It made the scan depend on
    something that the author cannot see. A skill with no `parameters:` block
    was checked when the model happened to pass arguments and not otherwise, and
    a skill with only `steps:` seeded a literal `{{parameters.gate}}` into the
    plan. Prose with braces in it is unaffected either way, because
    `PLACEHOLDER` matches only this one prefix. That keeps a skill able to
    contain an example of itself, and no other exemption is needed.

    Defaults are applied before validation, so a declared default satisfies a
    read that looks `required`. After validation the text sees a value for every
    declared name. An author who writes `{{parameters.tag}}` never has to think
    about whether the caller passed one.
    """
    declared = dict(skill.parameters.get("properties") or {})
    values = {
        name: spec["default"]
        for name, spec in declared.items()
        if isinstance(spec, dict) and "default" in spec
    }
    values.update(arguments)
    violations = validate_json_schema_value(skill.parameters, values)
    if violations:
        # There is one validator, not two. `additionalProperties: False` refuses
        # an unknown argument, as intended. An explicit membership check beside
        # it was a second answer to one question. It also ran first, so the rule
        # of the shared validator was unreachable and untested.
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
        return _rendered_value(values.get(name))

    filled = PLACEHOLDER.sub(fill, body)
    steps = [PLACEHOLDER.sub(fill, step) for step in skill.steps]
    if missing:
        # There is one refusal for both halves, and it does not say which half.
        # An author who reads it opens the file, and a model that reads it
        # cannot fix either one. The name of the parameter is what makes the
        # refusal actionable.
        raise ValueError(
            f"skill {skill.name!r} refers to undeclared parameter(s) "
            f"{', '.join(sorted(set(missing)))} in its body"
        )
    return filled, steps


def discover_skills(paths: Sequence[str], *, source: str = "skills-progressive") -> list[Skill]:
    """Every valid skill under `paths`. A later source shadows an earlier one.

    The sources shadow and do not collide, because the list is a precedence
    order. The `code-review` of a user is meant to win over the one that their
    distribution ships, and a refusal would make it impossible to override a
    skill.
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

    `tool=False` drops the sentence that names the `skill` tool. It is for the
    deployment where the catalog is rendered and the tool is not registered.
    Telling a model to call something that is not in its schema is worse than
    saying nothing, because it looks like a mistake of the model.
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
    """Which of those tools this caller cannot reach. The list is resolved, not
    only echoed.

    An echoed list leaves the model to diff it against its own catalog. Resolved
    against `run.scope`, it answers the question that the declaration was for:
    can I actually follow these instructions from here?"""


class Config(WireModel):
    """Row config for `skills-progressive`."""

    paths: list[str] = Field(default_factory=list)
    """The directories to scan, earliest first. Each one holds `<name>/SKILL.md`.

    The list is empty by default on purpose: see the module docstring. A path is
    `~`-expanded, so a deployment that wants `~/.ph/skills` writes it down, and
    this row does not assume it."""


def narrows(mounted: Config, asked: Config, reach: ChildReach) -> ChildLimit:
    """The `skills-progressive` row of a child may scan fewer of the directories
    of its parent, and never others. The child holds only the skills found under
    the directories that it keeps (S7b)."""
    if asked.paths == mounted.paths:
        return ChildLimit()
    theirs, mine = _scanned(mounted), _scanned(asked)
    if mine == theirs:
        return ChildLimit()
    extra = [path for path in mine if path not in theirs]
    if extra:
        raise NarrowingRefused(
            f"its skills-progressive reads {', '.join(map(str, extra))}, which its parent's "
            "does not"
        )
    service = reach.ctx.get(SKILLS)
    if service is None:
        return ChildLimit()
    mine_by_row = service.registrants()

    def found_elsewhere(name: str) -> bool:
        # Only the skills of this row count, by the row that installed each one.
        # A skill of another row, or of a second `skills-progressive` row, does
        # not belong to this row, so this row does not withhold it.
        skill = service.get(name, reach.boundary)
        if mine_by_row.get(name) != reach.row or skill is None or skill.path is None:
            return False
        where = Path(skill.path).resolve()
        return not any(where.is_relative_to(path) for path in mine)

    return ChildLimit(withheld_skills=frozenset(filter(found_elsewhere, reach.skills)))


def _scanned(config: Config) -> list[Path]:
    """The directories a `skills-progressive` config scans, resolved."""
    return [Path(one).expanduser().resolve() for one in config.paths]


@plugin(
    "skills-progressive",
    affects="environment",
    config=Config,
    inject=[SKILLS, SYSTEM_PROMPT, TOOLS],
    narrows=narrows,
)
async def progressive(ctx: Context, config: Config) -> None:
    """Catalog in the prompt, body on demand (G9).

    **The catalog is a provider, not a fixed string.** It reads the registry at
    assembly time. A skill that *another* row registers therefore appears in
    this catalog and not in a second catalog of its own, and no row has to be
    mounted before another. `rlm-skills-python` is such a row, and so is a row
    that ships skills with `offer_skills`. The cost is that registering a skill
    mid-session moves the cached prefix. That is the correct behavior: the model
    really was told something new.
    """
    if config.paths:
        # The scan runs in a thread. A scan is `glob` plus an open per
        # candidate, and mount is on the event loop that a TUI is about to draw
        # on.
        registry = ctx.require(SKILLS)
        for skill in await anyio.to_thread.run_sync(discover_skills, config.paths):
            registry.register(skill, scope=ctx)

    def catalog(request: AssembleContext) -> str:
        # The catalog is built per assembly *and* per agent. A child narrowed at
        # spawn (P4-13b) must not be told about skills that it cannot read, or
        # the catalog is a list of things to try and fail at. The `skill` tool
        # is checked in the same way, because the two can legitimately disagree
        # for one deployment.
        return render_catalog(
            ctx.require(SKILLS).list(request.scope),
            # This is scoped too. A child narrowed away from the `skill` tool
            # must not be told in prose to call it. Telling a model to use
            # something that is absent from its schema looks like a mistake of
            # the model, not of the profile.
            tool=ctx.require(TOOLS).get("skill", scope=request.scope) is not None,
        )

    ctx.require(SYSTEM_PROMPT).section(
        PromptSection(name="skills", order=ORDER_SKILLS, text=catalog), scope=ctx
    )

    async def read_body(args: SkillArgs, run: ToolRunContext) -> dict[str, Any]:
        # This uses the *stated* boundary, so a narrowed child is refused a
        # skill that it sees named nowhere. The catalog and the gate answer the
        # same question. Until P6-24 they answered it in two ways: the catalog
        # above reads `request.scope`, and this read
        # `getattr(run.agent, "ctx", None)`, the approval-routing target.
        # `run.scope` is a non-optional `Context` and was two lines away.
        # `fs_tools` had the same shape.
        skills = ctx.require(SKILLS)
        tools = ctx.require(TOOLS)
        scope = run.scope
        skill = skills.get(args.name, scope)
        body = skills.body(args.name, scope)
        if skill is None or body is None:
            # The refusal names the available skills, because "unknown skill"
            # and "this skill has no readable body" are different problems for
            # the model. One is a typo that it can fix from the catalog. The
            # other is a deployment fault that it cannot fix.
            known = ", ".join(one.name for one in skills.list(scope)) or "none"
            raise ValueError(f"no readable skill named {args.name!r}; available: {known}")
        # The body is rendered, not raw. A declared input is filled in before
        # the model reads the instruction that uses it. Arguments that do not
        # satisfy the declaration are refused here, and are not discovered three
        # steps into a procedure that has already changed something.
        instructions, steps = rendered_skill(body, skill, dict(args.arguments))
        # The read is recorded only now. Arguments that the declaration refuses
        # end the call before the body is handed over, and a `skill/read` would
        # say that it had been handed over.
        record_read(run.session, skill, body, via="tool")
        # **The read is announced, not acted on.** A skill that declares `steps`
        # is a procedure. Turning a procedure into entries that the model must
        # work through belongs to `tool-todo`. That row is in ph-stabilize, a
        # package that this seam cannot import, and a deployment may not have
        # mounted it. The event carries the declaration across that boundary, so
        # ph-core needs no opinion about plans (P7-18). It is an `emit`, so a
        # raise inside `skill-steps` is logged and does not undo the read of a
        # body that is rendered and about to be returned.
        ctx.emit(
            "skills/read",
            {
                "skill": skill,
                "steps": steps,
                "max_nudges": skill.max_nudges,
                "session": run.session,
            },
        )
        return {
            "name": skill.name,
            "path": skill.path,
            "instructions": instructions,
            "parameters": sorted(skill.parameters.get("properties") or {}),
            "version": skill.version,
            "allowed_tools": list(skill.allowed_tools),
            # Resolved against the same scope that the body was fetched with, so
            # the catalog, the gate and this list give one answer to one
            # question.
            "missing_tools": [
                name for name in skill.allowed_tools if tools.get(name, scope=scope) is None
            ],
        }

    def build_tool() -> ToolDefinition | None:
        """The tool, or `None` where there is nothing to read.

        See `register_when_composed`. A `skill` tool in a deployment that
        installed no skills is a prompt-sized advertisement for a capability
        that is not there.
        """
        # `DEPLOYMENT` is the correct answer here, and not only the convenient
        # one (P6-32). The question is whether this *deployment* installed any
        # skills at all. It is asked one time at mount, to decide whether the
        # tool exists. It is not a per-agent question, and there is no agent yet
        # to ask it for.
        if not ctx.require(SKILLS).list(DEPLOYMENT):
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
