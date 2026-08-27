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

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from ..cordis import Context, Disposer, plugin
from ..session import Session
from ._registry import claim_key

__all__ = [
    "ADMITTED",
    "DELETED",
    "STATUS",
    "USAGE",
    "Access",
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
"""A child's lifecycle, as the parent's roster and the TUI panel see it."""

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
    provider_name: str
    model: str
    requested_access: Access
    granted_access: Access
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
            "provider": self.provider_name,
            "model": self.model,
            "requestedAccess": self.requested_access,
            "grantedAccess": self.granted_access,
        }
        if self.downgrade_reason is not None:
            wire["downgradeReason"] = self.downgrade_reason
        return wire


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
        self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> SubagentRun | None:
        return self._runs.get(run_id)

    def list(self, *, parent_id: str | None = None) -> list[SubagentRun]:
        """Live runs, optionally only one parent's, in admission order."""
        runs = list(self._runs.values())
        if parent_id is None:
            return runs
        return [run for run in runs if run.parent_id == parent_id]

    def forget(self, run_id: str) -> SubagentRun | None:
        """Drop a run from the live table. The log keeps the tombstone."""
        return self._runs.pop(run_id, None)


@plugin("subagents")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the subagent seam definition. No provider ships in ph-base."""
    ctx.provide("subagents", SubagentService(ctx=ctx))


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
            roster[run_id] = dict(event.data)
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
