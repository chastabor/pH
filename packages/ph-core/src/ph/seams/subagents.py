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

from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from ..cordis import Context, Disposer, plugin
from ..session import Session, SessionFoldCache
from ._registry import claim_key

__all__ = [
    "ADMITTED",
    "DELETED",
    "STATUS",
    "USAGE",
    "Access",
    "FamilyRole",
    "RehydratableProvider",
    "StatusCause",
    "SubagentProvider",
    "SubagentRequest",
    "SubagentResult",
    "SubagentRun",
    "SubagentService",
    "SubagentSpawnError",
    "SubagentStatus",
    "apply",
    "default_child_name",
    "downgrade_text",
    "family_reach",
    "reachable_family",
    "roster_name",
    "subagent_roster",
]

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
    name: str | None = None
    """A stable label for the roster. Defaulted by the provider when omitted."""
    provider: str | None = None
    model: str | None = None
    """Exact selector. A provider must not silently fall back to another model —
    a child that answered on a cheaper model than the parent asked for is a
    result the parent cannot interpret."""
    reasoning_effort: str | None = None
    access: Access = "read"


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


@dataclass(slots=True)
class SubagentService:
    """The service published as `ctx.subagents`.

    Named providers, unlike `ctx.code_runtime`'s single slot: "run a child" has
    genuinely different answers in one deployment — an RLM child, an ephemeral
    research task — and the caller names which it wants.
    """

    ctx: Context
    _providers: dict[str, SubagentProvider] = field(default_factory=dict)
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
        return claim_key(
            scope or self.ctx, self._providers, name, provider, label="subagent-provider"
        )

    def provider_names(self) -> list[str]:
        return sorted(self._providers)

    def require(self, name: str) -> SubagentProvider:
        provider = self._providers.get(name)
        if provider is None:
            offered = ", ".join(self.provider_names()) or "none"
            raise SubagentSpawnError(
                f'no subagent provider named "{name}" is registered (registered: {offered})'
            )
        return provider

    async def start(self, name: str, request: SubagentRequest) -> SubagentRun:
        """Admit a child and return its handle. Does not wait for an answer."""
        run = await self.require(name).start(request)
        # Stamped here rather than trusted from the provider: the service is what
        # knows which name the caller asked for, and `rehydrate` has to be able
        # to find its way back to the same provider.
        run.owner = name
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
        """`roster_name`, through the cached fold."""
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
        provider = self._providers.get(run.owner)
        if not isinstance(provider, RehydratableProvider):
            return False
        return bool(await provider.rehydrate(run_id))

    def forget(self, run_id: str) -> SubagentRun | None:
        """Drop a run from the live table. The log keeps the tombstone."""
        return self._runs.pop(run_id, None)


@plugin("subagents", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Mount the subagent seam definition. No provider ships in ph-base."""
    service = SubagentService(ctx=ctx)
    ctx.provide("subagents", service)
    # A disposed session's last projection is a value nobody can reach; the cache
    # is bounded either way, but holding it is holding it for nothing.
    ctx.on("session/disposed", lambda session: service.forget_session(session.id))


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
        # The type test first: a long parent log is mostly `assistant/chunk`,
        # and reading `data["runId"]` off every one of them to discover it is
        # absent costs a mapping get and a string per event.
        if event.type not in _ROSTER_TYPES:
            continue
        run_id = str(event.data.get("runId"))
        if event.type == ADMITTED:
            # `queued` by default: `to_wire()` deliberately omits status, and the
            # first `subagent/status` comes from a detached job — so without this
            # a reader between the two sees a child with no status at all.
            roster[run_id] = {"status": "queued", **event.data}
            continue
        row = roster.get(run_id)
        if row is None:
            continue
        if event.type == STATUS:
            row.update({key: value for key, value in event.data.items() if key != "runId"})
        else:
            row["deleted"] = True
            row["deletedReason"] = event.data.get("reason")
    return roster


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


def roster_name(sessions: Iterable[Session], agent_id: str) -> str:
    """What an agent is called, or its id when nobody named it.

    The name is a fact the *parent* recorded at admission, so it is only knowable
    from the parent's roster fold — which is why it lives beside that fold rather
    than in whichever module needed it first. Two modules and the P3-19 panel
    need it, and two copies of one lookup is how a prompt names one agent while a
    send delivers to another. Prefer `SubagentService.name_of`, which folds
    through the cache; this is for a reader holding stored sessions and no `ctx`.
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
