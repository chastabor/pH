"""`ctx.subagents` — delegation to a child agent, and the handle it returns.

The seam definition only; the provider ships with a profile (`rlm-child` in
Phase 3). Two things about the contract are load-bearing enough to state here
rather than leave to a provider's docstring:

**The handle returns before the child answers.** `start()` resolves once the
child is *admitted* — session created, admission logged, task detached — not
once it has finished. That is the non-blocking fan-out an RLM parent depends on:
it spawns eight children, keeps working, and their replies arrive as ordinary
inbox messages on later turns. A contract where `start()` awaited completion
would make the parent's control loop serial and force it to poll.

**Completion is available but separate.** `SubagentRun.result()` awaits
quiescence for the caller that genuinely wants to block — a generic `task` tool
returning the child's last text. Both callers use one provider, which is why the
answer is reachable but never the thing `start()` gives back.

**`access` defaults to read (E4).** A child asks for the workspace guarantee it
needs; the default is the conservative one, so a delegation that never mentions
`access` cannot silently receive a writable repo. The provider resolves the
request against whatever tier is actually available and reports what was granted
— a child told nothing about its workspace attempts writes and reads the
failures as bugs.

@module ph.seams.subagents
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import Field

from ..cordis import Context, Disposer, Running, plugin, running
from ..session import Session, SessionFoldCache
from ..system_prompt.assembly import PromptSection
from ..tools.registry import ToolRestriction
from ..wire import WireModel
from ._registry import claim_key
from .skills import ORDER_SKILLS, SkillRestriction

__all__ = [
    "ADMITTED",
    "DELETED",
    "SETTLED_STATUSES",
    "STATUS",
    "USAGE",
    "Access",
    "FamilyRole",
    "RehydratableProvider",
    "StatusCause",
    "SubagentPreset",
    "SubagentPresetService",
    "SubagentProvider",
    "SubagentRequest",
    "SubagentResult",
    "SubagentRun",
    "SubagentService",
    "SubagentSpawnError",
    "SubagentStatus",
    "apply",
    "child_is_live",
    "default_child_name",
    "descendants",
    "downgrade_text",
    "family_reach",
    "fold_subagent_event",
    "reachable_family",
    "roster_name",
    "subagent_roster",
]

log = logging.getLogger("ph.seams.subagents")

ORDER_BRIEF = ORDER_SKILLS + 10
"""After the skills catalog, because the brief is the *assignment* and the
catalog is the menu — a reader who has just been told what to do should meet the
list of other things last."""

ADMITTED = "subagent/admitted"
DELETED = "subagent/deleted"
STATUS = "subagent/status"
USAGE = "subagent/usage-attributed"
"""The four durable records a delegation leaves in its *parent's* log.

Named here rather than spelled at each `append` site, because the fold below and
every producer have to agree on them exactly."""

_ROSTER_TYPES = frozenset({ADMITTED, DELETED, STATUS})

Access: TypeAlias = Literal["read", "write"]
"""What a child asks of the parent's workspace. `read` is the default (E4)."""

SubagentStatus: TypeAlias = Literal["queued", "running", "done", "error", "cancelled"]

SETTLED_STATUSES: frozenset[str] = frozenset({"done", "error", "cancelled"})
"""The statuses that mean a child has stopped. Beside the vocabulary it reads.

Here rather than in the consumer, for the reason the four event names are here:
the fold and every producer have to agree exactly. P5-05's sweeper wrote its own
copy — `{"completed", "failed", "cancelled", "deleted"}` — and it was wrong in
three of four members. `completed` and `failed` are names no producer emits (the
writer says `done` and `error`), `deleted` is not a status at all, and the two
that actually mean settled were missing. The effect was that a root which had
ever run a child to completion could never be released: every settled child read
as live, forever, which is most of what passivation exists to do.
"""


def child_is_live(row: Mapping[str, Any]) -> bool:
    """Whether this roster row is still working.

    Deletion is a tombstone rather than a status — `fold_subagent_event` sets
    `deleted` and leaves `status` alone — so both have to be read, which is the
    other half a hand-written copy got wrong.

    An unrecognised status counts as **live**, deliberately: a caller that
    releases a parent on the strength of this must fail towards keeping one
    alive. Getting it backwards abandons a running child; getting it this way
    costs memory until the child settles.
    """
    if row.get("deleted"):
        return False
    return str(row.get("status", "queued")) not in SETTLED_STATUSES


"""A child's lifecycle, as the parent's roster and the TUI panel see it.

Lifecycle only. *Why* a child is live — woken to answer a question rather than
still on its first task — is a separate `cause` on the same record, because the
roster folds status last-write-wins: a `rehydrated` member would have meant a
woken child that is actively working reads as not-running to every consumer that
branches on `"running"`."""

StatusCause: TypeAlias = Literal["rehydrated"]
"""Why a child entered its current status, when it is not simply "it started"."""

DowngradeReason: TypeAlias = Literal["workspace-not-mounted"]
"""Why a granted access is narrower than the one requested."""

_DOWNGRADE_TEXT: dict[str, str] = {
    "workspace-not-mounted": (
        "granted read rather than write: no workspace tier is mounted to enforce a "
        "writable repo, so one cannot be promised"
    )
}


def downgrade_text(reason: DowngradeReason | str) -> str:
    """The sentence for one downgrade code, rendered in exactly one place.

    The model, the card and the transcript all need this text; three copies of it
    is three things to edit when the tier lands, and nothing that notices when
    only two were.
    """
    return _DOWNGRADE_TEXT.get(reason, f"access was narrowed ({reason})")


class SubagentSpawnError(Exception):
    """A delegation was refused before the child existed.

    Distinct from a child that ran and failed: this one produced no session, no
    log and no artifacts, so a caller may retry it with different arguments.
    """


@dataclass(frozen=True, slots=True)
class SubagentRequest:
    """One delegation, as the caller describes it.

    `parent` is the agent handle delegating, not an id: the provider needs its
    session to log the admission and its inbox to deliver the reply, and looking
    both up from an id would let a caller name an agent it does not hold.
    """

    prompt: str
    parent: Any
    scope: Context | None = None
    """The boundary this delegation is made **from** (P6-31).

    Not the child's — that does not exist yet when the ceiling is computed, and when
    it does it is `SubagentRun.scope`, which `_enforce` checks is inside this one.

    The same value and the same argument as `ToolExecutionInput.scope`: the caller
    states the boundary, the seam does not guess it. Optional only because a
    `SubagentRequest` is built by callers and by providers rather than only inside
    this seam; `_delegating_boundary` resolves it.
    """
    name: str | None = None
    """A stable label for the roster. Defaulted by the provider when omitted."""
    provider: str | None = None
    model: str | None = None
    """Exact selector. A provider must not silently fall back to another model —
    a child that answered on a cheaper model than the parent asked for is a
    result the parent cannot interpret."""
    reasoning_effort: str | None = None
    access: Access = "read"
    preset: str | None = None
    """A named kind of child the deployment configured (`subagent-presets`).

    Resolved into the fields below before the ceiling is checked, so a preset is
    a set of defaults and never a way past it.
    """
    skills: tuple[str, ...] | None = None
    """Which skills the child gets. `None` inherits the parent's whole set.

    A **subset of the parent's, always** — naming one the parent does not hold
    is refused rather than granted, because a spawn that could widen would make
    delegation the privilege escalation I7 exists to prevent (P4-13b). To give a
    child a skill, install it, which gives it to the parent too.

    `()` is a real answer and not the same as `None`: a child that should read
    no skill at all. Naming one is also *direction* — the named skills' bodies
    are put in the child's own prompt, because a child spawned to follow a
    procedure should not have to spend a turn fetching it.
    """
    tools: tuple[str, ...] | None = None
    """Which tools the child gets. `None` inherits the parent's whole set.

    Same ceiling, same reason. Applied as a `ToolRestriction`, which can only
    subtract — the Code Mode transport stays reachable regardless, because it is
    unrestrictable by construction and a child with no way to call anything is
    not a narrower child, it is a broken one.
    """


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """What a finished child produced, for a caller that waited.

    A plain dataclass, not a `WireModel`: it never crosses a JSON boundary. What
    is durable about a child's outcome is the `subagent/status` record; this is
    the in-process answer handed to whoever awaited `result()`.
    """

    status: SubagentStatus
    answer: str = ""
    """The child's last non-empty assistant text. Empty when it never spoke."""
    error: str | None = None


@dataclass(slots=True)
class SubagentRun:
    """A live delegation. Returned at admission, before the child answers.

    The fields are the admission facts a parent can act on immediately: what to
    call it, where its log is, and which guarantees it actually got. `granted`
    may differ from what was asked when the available tier cannot honour the
    request, and it is the value the roster and the child's own prompt report.
    """

    id: str
    name: str
    session_id: str
    parent_id: str
    model_provider: str
    """Which LLM provider the child runs on. Named for what it is, because the
    *subagent* provider is a different thing on the same handle and one field
    called `provider` for both is how `rehydrate` looked up the wrong one."""
    model: str
    requested_access: Access
    granted_access: Access
    owner: str = ""
    """Which `ctx.subagents` provider owns this run, stamped by the service.

    Not `provider` — `SubagentRequest.provider` is the *LLM* provider, and one
    word for both is how `rehydrate` looked up the wrong one. Not on the wire
    either: the provider appends the admission from inside its own `start()`,
    before the service could stamp this, so a serialized copy would read `""` in
    every log. It is a routing stamp, not a fact about the child."""
    downgrade_reason: DowngradeReason | None = None
    """Why `granted` is narrower than `requested`, as a code rather than prose.

    A durable event carrying an English sentence is unparseable by the consumer
    that has to branch on it, and it goes stale silently: the reason a `write`
    was refused today stops being true when the workspace tier lands, in every
    log already written. `downgrade_text()` renders the sentence from the code,
    once."""
    result: Callable[[], Awaitable[SubagentResult]] | None = None
    """Awaits quiescence and reports the outcome. `None` from a provider whose
    children cannot be waited on."""
    dispose: Callable[[], Any] | None = None
    """Releases the child early. Registered as an effect of the parent's scope by
    the provider, so a disposed parent unwinds its children (I2)."""
    grant: Grant | None = None
    """What this child was bounded to, stamped by the seam once it has applied it.

    On the run because the run is what a provider keeps: a rehydration builds a
    *new* scope for a settled child and the filters that bounded the old one
    went with it, so replaying the ceiling needs the grant to have outlived the
    scope — and the parent it was computed from."""
    scope: Context | None = None
    """The child's own scope, set by the provider so the **seam** can bound it.

    The ceiling was a documented obligation on providers before this field, and
    the second call site had already missed it: `rehydrate` builds a fresh scope
    for a settled child and narrowed nothing, so a child that outlived its own
    restriction came back holding the deployment-wide set. Handing the scope
    back makes the enforcement the seam's, on both paths, rather than a rule a
    provider is trusted to remember."""

    def to_wire(self) -> dict[str, Any]:
        """The admission facts, for an event or a roster row.

        No `status`: a child's state is the fold over `subagent/status`, and a
        second copy on the handle would be a value frozen at whatever the last
        in-process update left."""
        wire: dict[str, Any] = {
            "runId": self.id,
            "name": self.name,
            "sessionId": self.session_id,
            "parentId": self.parent_id,
            "modelProvider": self.model_provider,
            "model": self.model,
            "requestedAccess": self.requested_access,
            "grantedAccess": self.granted_access,
        }
        if self.downgrade_reason is not None:
            wire["downgradeReason"] = self.downgrade_reason
        return wire


@runtime_checkable
class RehydratableProvider(Protocol):
    """A provider whose settled children can be given a runtime again (P3-13).

    A second Protocol rather than a method on `SubagentProvider`, because not
    every way of running a child can resume one — and rather than a `getattr`
    probe, because a provider whose method is misnamed or has the wrong arity
    would then fail silently as "cannot rehydrate", which is the failure mode
    that already cost this package a day.
    """

    async def rehydrate(self, run_id: str) -> bool: ...


@runtime_checkable
class SubagentProvider(Protocol):
    """A way of running a child agent.

    One method. A `capabilities` set was drafted here for a consumer to branch
    on — whether children survive the parent, whether they can be messaged — and
    removed again: with one provider there is nothing to branch on, and the
    vocabulary a second provider needs is not guessable from the first.
    """

    async def start(self, request: SubagentRequest) -> SubagentRun: ...


@dataclass(frozen=True, slots=True)
class _Registered:
    """A delegation provider and who registered it (P6-29).

    The sixth of these, and one of the two that survived P6-29 unbound: a
    provider is an *object satisfying a Protocol*, so `dict[str, SubagentProvider]`
    names no callable for `_row_bodies` to find, and this claims its slot with
    `claim_key` rather than `claim_slot`, so the source match that used to stand
    in for "is this a provider" did not look here either. `_provider_fields`
    discriminates on the Protocol now, which is true of a provider however it was
    registered."""

    provider: SubagentProvider
    by: Running


@dataclass(slots=True)
class SubagentService:
    """The service published as `ctx.subagents`.

    Named providers, unlike `ctx.code_runtime`'s single slot: "run a child" has
    genuinely different answers in one deployment — an RLM child, an ephemeral
    research task — and the caller names which it wants.
    """

    ctx: Context
    _providers: dict[str, _Registered] = field(default_factory=dict)
    _runs: dict[str, SubagentRun] = field(default_factory=dict)
    _rosters: SessionFoldCache[dict[str, dict[str, Any]]] = field(
        default_factory=lambda: SessionFoldCache(subagent_roster)
    )
    """The roster fold, cached per session. The prompt, the model's roster tool
    and every name lookup ask for the same fold several times per model step.

    The *fold* stays a pure function of a log — `subagent_roster` has to keep
    working on a fork slice and on a stored log, which is what a cache attached
    to `Session` could not have done."""

    def register_provider(
        self, name: str, provider: SubagentProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim one delegation strategy under `name`."""
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._providers, name, _Registered(provider, by), label="subagent-provider"
        )

    def provider_names(self) -> list[str]:
        return sorted(self._providers)

    def resolve(self, name: str | None) -> str | None:
        """Which provider a consumer should use, or `None` when there is no answer.

        The policy is here because it is a property of *this* table: empty means
        the one that is mounted, two mounted with no name chosen is a question
        this seam refuses to answer for a caller, and a configured name is
        checked against what is actually mounted rather than trusted.

        That last clause is the one worth stating. A row that trusts its own
        `provider:` setting advertises a capability whenever the provider row
        was renamed or removed — the phantom arriving by the one route that
        looks like a deliberate choice.
        """
        names = self.provider_names()
        if name is not None:
            if name in names:
                return name
            log.warning(
                "ph.seams.subagents: no provider named %r is mounted (mounted: %s)",
                name,
                ", ".join(names) or "none",
            )
            return None
        if len(names) == 1:
            return names[0]
        if names:
            log.warning(
                "ph.seams.subagents: %s providers are mounted (%s); a consumer must name one",
                len(names),
                ", ".join(names),
            )
        return None

    def require(self, name: str) -> _Registered:
        entry = self._providers.get(name)
        if entry is None:
            offered = ", ".join(self.provider_names()) or "none"
            raise SubagentSpawnError(
                f'no subagent provider named "{name}" is registered (registered: {offered})'
            )
        return entry

    def held_by(
        self, request: SubagentRequest, boundary: Context | None = None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """What the parent holds, in the two registries a grant covers.

        One reader for both halves of the ceiling — the refusal, the
        materialization and the "did this narrow anything" check all need the
        same two sets, and computing them three times in three spellings is how
        the three come to disagree about what "holds" means.

        `boundary` is threaded for the same reason `held` is on `check_grant`
        and `grant_for`: `start` needs the resolution three times (the ceiling,
        the brief, the containment check), and passing the one answer down makes
        the three agree by construction rather than by re-derivation.
        """
        parent_scope = boundary if boundary is not None else self._delegating_boundary(request)
        skills = self.ctx.get("skills")
        tools = self.ctx.get("tools")
        return (
            tuple(sorted(skills.reach(parent_scope))) if skills is not None else (),
            tuple(tools.names(scope=parent_scope)) if tools is not None else (),
        )

    def _delegating_boundary(self, request: SubagentRequest) -> Context:
        """The boundary a spawn's ceiling is computed in — stated, not guessed (P6-31).

        Four cases:

        * **a stated `request.scope`** wins, as it does everywhere else — "a stated scope
          wins before any handle is consulted" is the security property here;
        * **no scope and no parent** is the mount: a spawn with no parent is a root
          delegation, and the deployment-wide set is what it legitimately holds;
        * **a parent whose `.ctx` cannot be read** is **refused**. `None` is not "no
          ceiling" — `SkillService.reach` and `ToolRuntime.names` both resolve it to the
          *mount*, the unrestricted set — so an unreadable parent did not narrow a child,
          it handed it everything the deployment holds. There is no narrower default to
          pick, and this is the one path whose stake is that a spawn which could widen
          would make delegation a privilege escalation (I7);
        * **a parent with a usable `.ctx`** is that scope.

        Resolved here, at the entry, by the code that knows a parent was meant — never in
        the downstream registries, which since P6-32 require a stated `Boundary` and so
        cannot guess either.
        """
        if request.scope is not None:
            return request.scope
        if request.parent is None:
            return self.ctx
        scope = getattr(request.parent, "ctx", None)
        if not isinstance(scope, Context):
            raise SubagentSpawnError(
                f"{type(request.parent).__name__} was passed as `parent` but exposes no "
                "`ctx: Context`, so the ceiling this child inherits is unknowable; pass "
                "`scope=` beside it (P6-31)"
            )
        return scope

    def check_grant(
        self, request: SubagentRequest, held: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    ) -> None:
        """Refuse a spawn that asks for more than the parent holds (P4-13b).

        **Here rather than in each provider**, because this is the one path every
        delegation takes and a ceiling one provider forgot would not be one. The seam
        applies the grant too, through `SubagentRun.scope`, because `rehydrate` builds a
        fresh scope for a settled child and would otherwise hand it the deployment-wide
        set.

        Refused rather than silently intersected: a child that came back with something
        other than what was asked for is a result the parent cannot interpret — a
        `reviewer` child missing its review skill does the job wrong and reports success.
        """
        held_skills, held_tools = held if held is not None else self.held_by(request)
        for kind, asked, holds in (
            ("skill", request.skills, held_skills),
            ("tool", request.tools, held_tools),
        ):
            if asked is None:
                continue
            missing = sorted(name for name in asked if name not in holds)
            if missing:
                raise SubagentSpawnError(
                    f"a child cannot be granted {kind}s its parent does not hold: "
                    f"{', '.join(missing)}. Grant it to the parent first "
                    f"(the parent holds: {', '.join(sorted(holds)) or 'none'})."
                )

    def resolve_preset(self, request: SubagentRequest) -> SubagentRequest:
        """Fill what a named preset supplies and the caller left unsaid.

        Defaults, not a ceiling: a caller may still narrow further, and cannot
        widen past the parent whatever it names, because `check_grant` runs on
        the *resolved* request. An unknown name is refused rather than ignored —
        a spawn that asked for a `reviewer` and silently got a generic child is
        the failure `_resolve_model` refuses for the same reason one field over.
        """
        if request.preset is None:
            return request
        service = self.ctx.get("subagent_presets")
        preset = service.get(request.preset) if service is not None else None
        if preset is None:
            offered = ", ".join(service.names()) if service is not None else ""
            raise SubagentSpawnError(
                f'no subagent preset named "{request.preset}" is configured '
                f"(configured: {offered or 'none'})"
            )
        return replace(
            request,
            skills=request.skills if request.skills is not None else preset.skills,
            tools=request.tools if request.tools is not None else preset.tools,
        )

    def grant_for(
        self,
        request: SubagentRequest,
        held: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
        *,
        boundary: Context | None = None,
    ) -> Grant:
        """Materialize what this child may reach, from a parent that still exists.

        `None` means "everything the parent holds" and is written out as that
        explicit list. That used to be load-bearing for a reason P6-27 removed —
        agents were siblings, so a parent's filter did not reach its child and
        applying nothing handed the child of a narrowed parent the
        deployment-wide set. The chain does that job now. What the list is for is
        the ruling: **a child's capability is fixed at admission**, so this
        records what the parent held *then* rather than deferring to what it
        holds whenever the child is next asked. See `Grant`.
        """
        held_skills, held_tools = held if held is not None else self.held_by(request, boundary)
        named = request.skills
        skills = self.ctx.get("skills")
        target = boundary if boundary is not None else self._delegating_boundary(request)
        return Grant(
            skills=named if named is not None else held_skills,
            tools=request.tools if request.tools is not None else held_tools,
            brief=(_brief_text(skills, named, target) if named and skills is not None else ""),
        )

    def _enforce(
        self,
        grant: Grant,
        run: SubagentRun,
        held: tuple[tuple[str, ...], tuple[str, ...]],
        boundary: Context,
    ) -> None:
        """Bound the child, or refuse the spawn if this provider cannot be bounded.

        Fail-closed, and narrowly: a provider that does not hand back a scope is
        unbounded, which is only *unsafe* when the grant actually narrows something. So a
        deployment where nothing is restricted keeps working with any provider, and the
        moment a spawn means to narrow, a provider that cannot deliver that is refused
        rather than silently ignored.

        **The containment check has no off switch** (P6-31). `boundary` is resolved by
        `_delegating_boundary`, which refuses rather than yielding `None`, so by here it
        is always a `Context` and the check always runs.
        """
        if run.scope is not None:
            # **A provider's scope must be inside the parent's** (P6-27).
            # Containment is the isolation chain now, so a scope built anywhere
            # else silently opts out of it — and opts out *invisibly*, because
            # the child still gets the admission grant and therefore still
            # passes every ceiling assertion; all it loses is the inherited
            # narrowing. Checked at the one place every delegation passes
            # through, for `check_grant`'s reason: "a ceiling one provider forgot
            # would not be one." A provider that predates nesting, or a future
            # one that forgets `parent=`, fails here instead of at nothing.
            if not boundary.reaches(run.scope):
                raise SubagentSpawnError(
                    f'the "{run.owner or "subagent"}" provider built the child a scope outside '
                    "its parent's, so the child would not inherit the parent's ceiling; a "
                    "child's scope must be created with `agents.create(..., parent=…)`"
                )
            run.grant = grant
            grant.apply(self.ctx, run.scope)
            return
        if (grant.skills, grant.tools) != held or grant.brief:
            raise SubagentSpawnError(
                f'the "{run.owner or "subagent"}" provider does not expose a child scope, '
                "so a narrowed child cannot be bounded; it can only run children that "
                "inherit everything their parent holds"
            )

    async def start(self, name: str, request: SubagentRequest) -> SubagentRun:
        """Admit a child and return its handle. Does not wait for an answer."""
        request = self.resolve_preset(request)
        # Once, then threaded — the ceiling, the brief and the containment
        # check must be answers to the *same* boundary, and one resolution
        # makes that true by construction (the `held` argument one line down
        # exists for the identical reason).
        boundary = self._delegating_boundary(request)
        held = self.held_by(request, boundary)
        self.check_grant(request, held)
        grant = self.grant_for(request, held, boundary=boundary)
        entry = self.require(name)
        # As the row that registered the provider (P6-29). A provider's `start`
        # is row code this registry invokes — the same category as a tool's
        # `execute` — and it ran unbound, so anything it registered landed on the
        # seam and outlived its row. The layer is the registration's own, for the
        # reason `CompactionSeam.engine_by` states: the target is in hand here
        # (`request.parent`) but reading it means another copy of P6-24's
        # `getattr(agent, "ctx", None)`, and the child's own containment is
        # `Grant`'s subject rather than this binding's.
        with running(entry.by):
            run = await entry.provider.start(request)
        # Stamped here rather than trusted from the provider: the service is what
        # knows which name the caller asked for, and `rehydrate` has to be able
        # to find its way back to the same provider.
        run.owner = name
        self._enforce(grant, run, held, boundary)
        self._runs[run.id] = run
        return run

    def roster(self, session: Session) -> dict[str, dict[str, Any]]:
        """`subagent_roster(session)`, folded at most once per appended event.

        The read every consumer should use when it has a `ctx`.
        """
        return self._rosters.read(session)

    def forget_session(self, session_id: str) -> None:
        """Drop what this service cached about one session."""
        self._rosters.forget(session_id)

    def name_of(self, sessions: Iterable[Session], agent_id: str) -> str:
        """`roster_name`, through the cached fold. The live path."""
        by_id = {session.id: session for session in sessions}
        session = by_id.get(agent_id)
        parent = by_id.get(session.header.parent_session or "") if session else None
        return _name_in(self.roster(parent), agent_id) if parent is not None else agent_id

    def get(self, run_id: str) -> SubagentRun | None:
        return self._runs.get(run_id)

    def list(self, *, parent_id: str | None = None) -> list[SubagentRun]:
        """Live runs, optionally only one parent's, in admission order."""
        runs = list(self._runs.values())
        if parent_id is None:
            return runs
        return [run for run in runs if run.parent_id == parent_id]

    async def ensure_addressable(self, session_id: str) -> bool:
        """Make the agent behind `session_id` reachable, waking it if it settled.

        The one place the session-id → run → provider → wake path is stated. A
        caller that wants to address an agent asks this rather than composing the
        three hops itself, which is how the second caller forgets the
        revoked-child refusal.
        """
        if self.ctx.agents.get(session_id) is not None:
            return True
        run = next((one for one in self._runs.values() if one.session_id == session_id), None)
        return bool(run is not None and await self.rehydrate(run.id))

    async def rehydrate(self, run_id: str) -> bool:
        """Make a settled child addressable again (P3-13).

        A child that finished had its agent released, so it has no inbox to steer
        into — but its session, its log and its roster row are all still there.
        Rehydration is the provider re-attaching a runtime to that state; a
        provider that cannot do it says so by not implementing the method, and
        the caller gets `False` rather than an exception it has to interpret.
        """
        run = self._runs.get(run_id)
        if run is None:
            return False
        entry = self._providers.get(run.owner)
        if entry is None or not isinstance(entry.provider, RehydratableProvider):
            return False
        with running(entry.by):
            return bool(await entry.provider.rehydrate(run_id))

    def forget(self, run_id: str) -> SubagentRun | None:
        """Drop a run from the live table. The log keeps the tombstone."""
        return self._runs.pop(run_id, None)


class SubagentPreset(WireModel):
    """A named kind of child a deployment is willing to spawn.

    **No prompt field, deliberately.** The obvious design gives a preset its own
    standing instructions, and then a directing skill and a preset are two
    channels saying what a child is for — competing where they disagree and
    duplicated where they do not. A skill body already *is* a standing
    instruction (P4-13b), so a preset binds a name to the capability and lets the
    skill do the directing: `reviewer` is `skills: [code-review]`, and what a
    reviewer does is written once, in the skill, where a human edits it.
    """

    skills: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None


class PresetConfig(WireModel):
    """Row config for `subagent-presets`."""

    presets: dict[str, SubagentPreset] = Field(default_factory=dict)
    """Named by the deployment, selected by a parent.

    A **menu, never a grant**: naming a preset whose entries the parent does not
    hold is refused like any other spawn, because a preset that widened whatever
    selected it would put the escalation one indirection away and under the
    model's control (P4-13b). Presets exist so a deployment writes "what a
    reviewer is" once, not so it can hand out more than the parent has.
    """


@dataclass(slots=True)
class SubagentPresetService:
    """The service published as `ctx.subagent_presets`.

    Config-only, with no `register(..., scope=)` — the one table under
    `ph.seams` without one, and deliberately: a preset is what a *deployment* is
    willing to spawn, so a package that could ship one would be widening what a
    profile allows without the profile saying so. A package ships the skill; a
    profile decides which children may hold it.
    """

    presets: dict[str, SubagentPreset] = field(default_factory=dict)

    def get(self, name: str) -> SubagentPreset | None:
        return self.presets.get(name)

    def names(self) -> list[str]:
        return sorted(self.presets)


@plugin("subagent-presets", config=PresetConfig)
async def presets(ctx: Context, config: PresetConfig) -> None:
    """Publish the deployment's named child kinds. None ship in `ph-base`."""
    ctx.provide("subagent_presets", SubagentPresetService(presets=dict(config.presets)))


@plugin("subagents", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the subagent seam definition. No provider ships in ph-base."""
    service = SubagentService(ctx=ctx)
    ctx.provide("subagents", service)
    # A disposed session's last projection is a value nobody can reach; the cache
    # is bounded either way, but holding it is holding it for nothing.
    ctx.on("session/disposed", lambda session: service.forget_session(session.id))


@dataclass(frozen=True, slots=True)
class Grant:
    """What one child may reach, resolved to names and fixed at admission.

    **A child's capability is fixed at the moment it is admitted.** A parent that
    gains a tool afterwards does not widen a child already running: the child was
    spawned for a job, with a ceiling its prompt and its brief were written against,
    and silently growing that mid-flight would make "what could this child do"
    unanswerable from the admission record. A parent that needs a child with more
    spawns a *new* child, whose admission says so.

    That is why the allow-list stays even though the isolation chain would bound the
    child anyway: the chain answers "no more than the parent **holds**", and this
    answers "no more than the parent held **then**". Only the second is stable enough
    to read a transcript against.

    `brief` is rendered here rather than read per assembly: it is the cached prompt
    prefix, and a `PromptSection` that hits the filesystem on every model step is
    neither static nor free.
    """

    skills: tuple[str, ...]
    tools: tuple[str, ...]
    brief: str = ""

    def apply(self, ctx: Context, scope: Context) -> None:
        """Bound a child's scope to this grant.

        **Narrowing is by restriction, never by registration.** A scope's own
        registration is unmaskable by its own filter — a filter reaches everything
        *outside* the scope that wrote it and nothing inside — so registering on a child
        is a way to hand it something its parent cannot see, which is the opposite of a
        ceiling. Filters only intersect, so they are the only instrument a spawn may use.

        An *ancestor's* registration is maskable (P6-27), and must be, or a child could
        never be narrowed below a parent that registered on its own scope. What still
        holds is that a scope cannot filter itself.
        """
        skills = ctx.get("skills")
        if skills is not None:
            skills.restrict(SkillRestriction(allow=frozenset(self.skills)), scope=scope)
        tools = ctx.get("tools")
        if tools is not None:
            tools.restrict(ToolRestriction(allow=frozenset(self.tools)), scope=scope)
        prompt = ctx.get("system_prompt")
        if self.brief and prompt is not None:
            prompt.section(
                PromptSection(name="subagent:brief", text=self.brief, order=ORDER_BRIEF),
                scope=scope,
            )


def _brief_text(skills: Any, named: Sequence[str], scope: Context) -> str:
    """The named skills' instructions, read once.

    **A named skill is direction, not a lookup.** G9 keeps bodies out of the
    prompt because a catalog of twenty is twenty bodies the model probably will
    not need; a child spawned *for* one will need that one, certainly. Making it
    spend a turn fetching what it was created to do is a wasted call and a real
    chance it never fetches at all — so the named bodies go in its prompt, and
    everything else it can still reach stays catalog-only.
    """
    parts = []
    for name in named:
        body = skills.body(name, scope)
        if body:
            parts.append(f"## {name}\n\n{body.strip()}")
    if not parts:
        return ""
    return (
        "You were delegated this task to follow the instructions below. "
        "They are not background reading.\n\n" + "\n\n".join(parts)
    )


def subagent_roster(session: Session) -> dict[str, dict[str, Any]]:
    """One parent's children, folded from its own log (A11, P3-13).

    Admission creates a row, status updates it, deletion tombstones it. A deleted
    child stays visible as a tombstone: its transcript is still on disk, and a
    parent asking what happened to the one it revoked deserves an answer other
    than silence.

    In the seam rather than in the bundle that produces the events, because the
    two consumers live in different packages — the model's roster tool in the
    RLM bundle, the subagent panel in the app, which cannot import the bundle. A
    second copy is exactly the "two projections of one fold that disagree" that
    A11 exists to forbid.
    """
    roster: dict[str, dict[str, Any]] = {}
    for event in session.events:
        fold_subagent_event(roster, event)
    return roster


def fold_subagent_event(roster: dict[str, dict[str, Any]], event: Any) -> None:
    """Fold one event into a roster, in place. The rules, in one place.

    Exported because there is a second consumer with a different *shape* — the
    TUI's panel folds incrementally, one event at a time, and cannot call the
    whole-log version. Sharing the step rather than the loop is what stops the
    two from being two implementations of one fold (A11): the app's panel and
    the model's roster now cannot disagree about `cause`, seeding or tombstones,
    because there is nothing for them to disagree with.
    """
    # The type test first: a long parent log is mostly `assistant/chunk`, and
    # reading `data["runId"]` off every one of them to discover it is absent
    # costs a mapping get and a string per event.
    if event.type not in _ROSTER_TYPES:
        return
    run_id = str(event.data.get("runId"))
    if event.type == ADMITTED:
        # `queued` by default: `to_wire()` deliberately omits status, and the
        # first `subagent/status` comes from a detached job — so without this
        # a reader between the two sees a child with no status at all.
        roster[run_id] = {"status": "queued", **event.data}
        return
    row = roster.get(run_id)
    if row is None:
        return
    if event.type == STATUS:
        row.update({key: value for key, value in event.data.items() if key != "runId"})
    else:
        row["deleted"] = True
        row["deletedReason"] = event.data.get("reason")


FamilyRole: TypeAlias = Literal["self", "parent", "sibling", "child"]
"""How one agent stands relative to another, within reach."""


def reachable_family(sessions: Iterable[Session], agent_id: str) -> dict[str, FamilyRole]:
    """Every agent `agent_id` may address, mapped to how it is related (C7).

    The enumeration half of `family_reach`, and derived *from* it rather than
    restating the rule — the guard that refuses a send and the roster that tells
    the model who it may address must not be able to disagree.

    An agent's id is its session's id, so the parent link is
    `SessionHeader.parent_session` and nothing needs a side index. Sessions are
    passed in rather than read from a store, so this answers on a resumed log
    with no live agents as readily as in a running process.
    """
    parents = {session.id: session.header.parent_session for session in sessions}
    if agent_id not in parents:
        return {}
    mine = parents[agent_id]
    reach: dict[str, FamilyRole] = {}
    for other, other_parent in parents.items():
        if not family_reach(
            sender_parent=mine, sender_id=agent_id, target_parent=other_parent, target_id=other
        ):
            continue
        if other == agent_id:
            reach[other] = "self"
        elif other == mine:
            reach[other] = "parent"
        elif other_parent == agent_id:
            reach[other] = "child"
        else:
            reach[other] = "sibling"
    return reach


def descendants(lineage: Iterable[tuple[str, str | None]], agent_id: str) -> list[str]:
    """`agent_id` and everything spawned beneath it, transitively (P6-28).

    **Not `reachable_family`, and the difference is the point.** That answers "who
    may this agent *address*" — the C7 nuclear family, including siblings and the
    parent — which is the right rule for a message. This answers "whose leftovers are
    mine to account for": a sibling's worktree is not this agent's to enumerate,
    still less to collect, and borrowing the messaging rule would widen a filesystem
    question with an answer computed for a different one (I7).

    Transitive where `reachable_family` is one hop: a grandchild that failed is
    evidence its grandparent is the only live party left to look at, because the
    child that spawned it settled too.

    An agent's id is its session's id, so the links are `parent_session` and nothing
    needs a side index. **`(id, parent)` pairs rather than `Session` objects**, which
    is what lets the collector answer this from a *listing* — a family is narrowed
    before a single log is read rather than after all of them are.

    Breadth-first, and cycle-safe by construction: `seen` is tested before descent,
    so a log claiming its own ancestor as a child costs a wasted lookup rather than a
    hang.
    """
    children: dict[str, list[str]] = {}
    known = set()
    for session_id, parent in lineage:
        known.add(session_id)
        if parent:
            children.setdefault(parent, []).append(session_id)
    if agent_id not in known:
        return []
    found = [agent_id]
    seen = {agent_id}
    for current in found:
        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                found.append(child)
    return found


def roster_name(sessions: Iterable[Session], agent_id: str) -> str:
    """What an agent is called, or its id when nobody named it.

    The name is a fact the *parent* recorded at admission, so it is only knowable
    from the parent's roster fold — which is why it lives beside that fold rather
    than in whichever module needed it first.

    **Prefer `SubagentService.name_of`**, which asks the same question through
    the cached fold; every live caller does. This is the variant for a reader
    holding stored sessions and no `ctx` — the trajectory view (P3-24) is the
    one that will want it. Both delegate to `_name_in`, so the two cannot answer
    differently: two copies of one lookup is how a prompt names one agent while
    a send delivers to another.
    """
    by_id = {session.id: session for session in sessions}
    session = by_id.get(agent_id)
    parent = by_id.get(session.header.parent_session or "") if session else None
    if parent is None:
        return agent_id
    return _name_in(subagent_roster(parent), agent_id)


def _name_in(roster: dict[str, dict[str, Any]], agent_id: str) -> str:
    """The name a folded roster gives one session, or the id."""
    for row in roster.values():
        if row.get("sessionId") == agent_id:
            return str(row.get("name") or agent_id)
    return agent_id


def family_reach(
    *, sender_parent: str | None, sender_id: str, target_parent: str | None, target_id: str
) -> bool:
    """Whether `sender` may address `target` under the nuclear-family rule (C7).

    A sender reaches its parent, its direct children, and same-depth siblings
    sharing a parent — roots counting as siblings of each other. Placed in the
    seam rather than in the RLM bundle because it is the reach rule for *any*
    provider's children, and the guard that enforces it (P3-12) must not be able
    to disagree with the roster that displays it.
    """
    if sender_id == target_id:
        return True
    if target_id == sender_parent:  # the parent
        return True
    if target_parent == sender_id:  # a direct child
        return True
    return sender_parent == target_parent  # a sibling, roots included


def default_child_name(prompt: str, run_id: str, *, taken: Sequence[str] = ()) -> str:
    """`subagent-<prompt-slug>-<id8>`, unique among `taken`.

    A name the parent can read back matters more than it looks: the roster, the
    terminal notice and every `agent_message` address use it, and a child called
    `child-4` tells the parent nothing about which delegation it was.
    """
    words = [
        word for word in "".join(c if c.isalnum() else " " for c in prompt).split()[:4] if word
    ]
    slug = "-".join(words).lower()[:32] or "task"
    candidate = f"subagent-{slug}-{run_id[:8]}"
    if candidate not in taken:
        return candidate
    for suffix in range(2, 100):
        alternative = f"{candidate}-{suffix}"
        if alternative not in taken:
            return alternative
    return f"{candidate}-{len(taken)}"
