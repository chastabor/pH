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
import secrets
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, TypeAlias, get_args, runtime_checkable

import anyio
from pydantic import Field, ValidationError

from ..agent.types import AgentDriver, AgentHandle
from ..cordis import (
    ChildReach,
    Context,
    Disposer,
    InactiveScopeError,
    LoaderError,
    NarrowingRefused,
    Running,
    maybe_await,
    plugin,
    releasing,
    running,
)
from ..json import JsonValue, as_obj, as_seq, as_str, thaw_json
from ..keys import (
    AGENTS,
    GOALS,
    NAMED_PROFILES,
    SANDBOX,
    SESSION_PERSISTENCE,
    SESSIONS,
    SKILLS,
    SUBAGENT_PRESETS,
    SUBAGENTS,
    SYSTEM_PROMPT,
    TOOLS,
)
from ..llm.types import (
    CONTEXT_SUMMARY_MAX_CHARS,
    PluginSource,
    create_user_message,
    new_message_id,
)
from ..persistence.opening import open_session, stored_session
from ..persistence.protocol import SessionPersistence
from ..session import (
    Session,
    SessionBatch,
    SessionEvent,
    SessionFoldCache,
    SessionHeader,
    child_session_id,
    is_child_id,
    session_written,
)
from ..session.kinds import CREDENTIAL_WAIT, SESSION_HOLDER, hold_of
from ..session.writers import log_writer
from ..system_prompt.assembly import PromptSection
from ..tools.definition import Deny, call_id_of
from ..tools.errors import (
    SPAWN_REFUSED,
    TOOL_BUDGET_SPENT,
    TOOL_DENIED,
    FailureKind,
    HarnessError,
)
from ..tools.registry import ToolRestriction
from ..wire import WireForm, WireModel, literal_lookup
from ._registry import claim_entry, claim_key
from .credentials import hold_for_credential, missing_credential
from .invariants import contribute_fold_cache
from .models import ModelChoice, ModelChoiceError, choose
from .skills import ORDER_SKILLS, SkillRestriction, SkillService
from .subagent_profiles import narrowing
from .token_meter import SPENDING_TYPES, event_tokens

_LOG = log_writer(__name__)

__all__ = [
    "ACCESS_LEVELS",
    "ADMITTED",
    "CHILD_EVENT_TYPES",
    "DELETED",
    "INTERRUPTED_DETAIL",
    "MAX_NAME_CHARS",
    "PARENT_TEARDOWN",
    "SETTLED_STATUSES",
    "STATUS",
    "SUSPENDED_DETAIL",
    "UNRECOVERABLE_DETAIL",
    "Access",
    "Admission",
    "ChildCounts",
    "ChildNotice",
    "ChildState",
    "FamilyRole",
    "ReadmittingProvider",
    "RehydratableProvider",
    "RevokingProvider",
    "SettledStatus",
    "SpawnGuard",
    "StatusCause",
    "SubagentAwaiter",
    "SubagentPreset",
    "SubagentPresetService",
    "SubagentProvider",
    "SubagentRequest",
    "SubagentResult",
    "SubagentRun",
    "SubagentService",
    "SubagentSpawnError",
    "SubagentStatus",
    "admission_payload",
    "admitted_by",
    "apply",
    "child_is_live",
    "child_model_key",
    "child_route",
    "child_state",
    "child_state_of",
    "default_child_name",
    "downgrade_text",
    "exhausted_detail",
    "extend_child_state",
    "family_reach",
    "fold_child_event",
    "open_child_log",
    "parent_went_away",
    "reachable_family",
    "record_admitted",
    "record_deleted",
    "record_ended",
    "record_started",
    "record_waiting",
    "restarts_since_progress",
]

log = logging.getLogger("ph.seams.subagents")

ORDER_BRIEF = ORDER_SKILLS + 10
"""After the skills catalog, because the brief is the *assignment* and the
catalog is the menu — a reader who has just been told what to do should meet the
list of other things last."""

ADMITTED = "subagent/admitted"
DELETED = "subagent/deleted"
STATUS = "subagent/status"
"""The three records a delegation leaves — in the **child's own** log (Phase 11).

Each session owns its log: a child records what it was asked, each start, each
wait, its ending and its deletion in its own log, and nothing about it is written
to its parent's. A parent, the resume sweep, budgets, caps and front ends read a
child's state from the child's log (`child_state`). The parent's copy this
replaces was a second account of the child that every durability rule — the
admission's flush (S2), a restart's (S10), the child's log before the parent's
`done` (F1), the usage catch-up after a crash (L5) — existed to keep in step.

Named here rather than spelled at each `append` site, because the fold below and
every producer have to agree on them exactly."""

_ANSWER = "assistant/message"
CHILD_EVENT_TYPES = frozenset(
    {ADMITTED, DELETED, STATUS, *SPENDING_TYPES, CREDENTIAL_WAIT.opened, CREDENTIAL_WAIT.settled}
)
"""What a child's state is folded from (`fold_child_event`): the three records; the ones
that spend tokens (`SPENDING_TYPES`) — an answer among them, which is also what forgives
the ladder; and a credential hold's pair. Any other record leaves a child's state as it
was, so a reader of a stored log parses only these, and a watcher skips the rest."""

Access: TypeAlias = Literal["read", "write"]
"""What a child asks of the parent's workspace. `read` is the default (E4)."""

ACCESS_LEVELS: Mapping[str, Access] = literal_lookup(Access)
"""Every `Access` by its own spelling — the read-side check for a value that
arrived as a `str`. See `literal_lookup`.

The value it guards comes off the **log**: `_readmit_one` rebuilds a child's
request from `subagent/admitted`, and `admission_payload` records the narrowing
precisely because "a restart re-derives the ceiling from this record and nothing
else". A `cast` stood here, which asserts rather than checks — and what this one
feeds is `_take_workspace`'s `access`, the difference between a child getting a
writable checkout of the project and a read-only one."""

SubagentStatus: TypeAlias = Literal["queued", "running", "done", "error", "canceled"]
"""A child's lifecycle, as its own log records it and its parent and the TUI panel
read it.

Lifecycle only. *Why* a child is live — woken to answer a question rather than
still on its first task — is a separate `cause` on the same record, because a
child's state folds status last-write-wins: a `rehydrated` member would have meant a
woken child that is actively working reads as not-running to every consumer that
branches on `"running"`."""

UNRECOVERABLE_DETAIL = (
    "the harness stopped while this child was running, and no provider here can "
    "start it again; its transcript is on disk"
)
"""Interrupted, with nothing mounted that could resume it.

Its own sentence because the answer a parent needs differs: the ladder was not
spent, and a deployment that mounted the provider again could have carried on —
so "it ran out of attempts" would be false. Settled all the same: a row left
`queued` for a provider that will never come holds the root out of passivation
for good."""

INTERRUPTED_DETAIL = "the harness stopped while this child was running; it did not finish"
"""Why a child that was mid-turn at shutdown did not finish.

A sentence rather than a code, because its reader is a person or a model looking
at a child's state and asking what happened to a child that never answered."""


PARENT_TEARDOWN = "parent-teardown"
"""Why a child was revoked when its parent ended or went away (I2): its parent's scope
unwound under it, or — found on a restart — its parent had ended and left it
unfinished. One reason for both, because they are one event: the second is the first,
completed by the next process where the crash cut it short."""


SUSPENDED_DETAIL = (
    "the harness stopped while this child was working; it is started again when its parent is"
)
"""Why a child is `queued` after its whole mount unwound under it (S1).

Written by a provider in place of a tombstone: a daemon stopping, or a root remounted
on a new profile, is not a revocation, so the row stays live and the resume sweep
readmits it. `queued` rather than left `running`, because nothing crashed: the sweep
readmits a `queued` row without asking the ladder whether it may. Its next start is
still a restart (`cause: resumed`), since it was one, so a child that then crashes
without having answered carries that start onto the ladder."""


def exhausted_detail(limit: int) -> str:
    """The ladder spent, naming the bound **actually in force**.

    Built from `INTERRUPTED_DETAIL` rather than restating it, so the two cannot
    come to describe one interruption differently.
    """
    return (
        f"{INTERRUPTED_DETAIL} — and it has now been interrupted "
        f"{limit} times, so it will not be started again"
    )


SettledStatus: TypeAlias = Literal["done", "error", "canceled"]
"""A status that means a child has stopped."""

SETTLED_STATUSES: frozenset[str] = frozenset(get_args(SettledStatus))
"""The statuses that mean a child has stopped. Beside the vocabulary it reads.

Here rather than in the consumer, for the reason the four event names are here:
the fold and every producer have to agree exactly. P5-05's sweeper wrote its own
copy — `{"completed", "failed", "canceled", "deleted"}` — and it was wrong in
three of four members. `completed` and `failed` are names no producer emits (the
writer says `done` and `error`), `deleted` is not a status at all, and the two
that actually mean settled were missing. The effect was that a root which had
ever run a child to completion could never be released: every settled child read
as live, forever, which is most of what passivation exists to do.
"""


def child_route(request: SubagentRequest) -> tuple[str, str]:
    """The route a child runs on: its request's selector, else its parent's.

    **No fallback on an explicit selector**: a child asked for a model runs on that
    model or not at all, since falling back would answer the parent's question on a
    model it did not choose. Stated here, once, for the provider that builds the
    child and for the resume check that asks whether its credential is here (T5).
    Empty where neither says; a provider refuses that.
    """
    options = request.parent.options
    return request.provider or options.provider or "", request.model or options.model or ""


def child_model_key(request: SubagentRequest) -> str:
    """The listed name of the route `child_route` gives a child (S7b): the key its
    spawn resolved, none for a route named whole, else its parent's own."""
    if request.model_key is not None:
        return request.model_key
    if request.provider or request.model:
        return ""
    return request.parent.options.model_key


class _Standing(Protocol):
    """What `child_is_live` reads: a child's state, or a front end's row of it."""

    @property
    def deleted(self) -> bool: ...

    @property
    def status(self) -> SubagentStatus: ...


def child_is_live(state: _Standing) -> bool:
    """Whether this child is still working.

    Deletion is a tombstone rather than a status — `fold_child_event` sets
    `deleted` and leaves `status` alone — so both have to be read, which is the
    other half a hand-written copy got wrong.

    A status the fold does not recognize leaves the one before it standing
    (`fold_child_event`), and a child admitted with none yet reads `queued`, so
    the failure direction is kept: a caller that releases a parent on the strength
    of this must fail towards keeping one alive. Getting it backwards abandons a
    running child; getting it this way costs memory until the child settles.
    """
    return not state.deleted and state.status not in SETTLED_STATUSES


StatusCause: TypeAlias = Literal["rehydrated", "resumed"]
"""Why a child entered its current status, when it is not simply "it started".

`resumed` is a child put back on its feet after its harness stopped mid-turn —
the ladder below, and the reason a reader of its log can tell that run from a
first attempt."""

DowngradeReason: TypeAlias = Literal["workspace-not-mounted"]
"""Why a granted access is narrower than the one requested."""

_DOWNGRADE_REASONS: Mapping[str, DowngradeReason] = literal_lookup(DowngradeReason)

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


def _refused(reason: Deny | str) -> SubagentSpawnError:
    """One guard's answer, as the error the pipeline already knows how to read.

    **The code is derived from the reading, exactly as `registry._gate` derives
    it** (D15). The tool side picks `budget_result` or `denied_result` off
    `Deny.failure_kind` and carries no code on the decision at all; carrying one
    here meant the same code arrived with two readings — `SPAWN_REFUSED` as a
    retryable misconfiguration from a direct raise, and as a policy denial from
    the ceiling — which is the one distinction `TOOL_DENIED`'s own docstring
    exists to keep.

    `Deny` rather than a second refusal type for the same reason: the child
    ceiling and the tool ceiling are the same decision, and the row that governs
    both now writes it once. A bare string is still a guard's answer and means
    the plain refusal every guard but the ceiling wants — `ToolRuntime.guard` is
    posture-less for that reason, and this keeps the two seams' guards the same
    shape.
    """
    if isinstance(reason, str):
        return SubagentSpawnError(reason)
    return SubagentSpawnError(
        reason.reason,
        TOOL_BUDGET_SPENT if reason.failure_kind == "failed" else TOOL_DENIED,
        failure_kind=reason.failure_kind,
        concludes_turn=reason.concludes_turn,
    )


class SubagentSpawnError(HarnessError):
    """A delegation was refused before the child existed.

    Distinct from a child that ran and failed: this one produced no session, no
    log and no artifacts, so a caller may retry it with different arguments.

    **A `HarnessError`, so the refusal keeps its shape on the way out** (D15).
    It was a bare `Exception`, which meant it carried no code and no
    `failure_kind` and arrived at the model as a generic failure with the turn
    carrying on — the posture D7 replaced for tool calls, surviving here because
    nothing had given a spawn refusal a way to say more. The two tool bodies
    that catch it also flattened it into a `ValueError`/`ToolCallError`, so even
    a richer error would not have reached the pipeline; they now let it through.

    The defaults are the old behaviour exactly — `SPAWN_REFUSED`, a failure the
    model may act on, the turn continuing — because most refusals here are
    misconfiguration a retry *can* fix: no such provider, no such preset, a
    grant the parent does not hold. Only a spent ceiling says otherwise, and it
    says so by passing them.
    """

    def __init__(
        self,
        message: str,
        code: str = SPAWN_REFUSED,
        *,
        failure_kind: FailureKind = "failed",
        concludes_turn: bool = False,
    ) -> None:
        super().__init__(message, code)
        self.failure_kind = failure_kind
        self.concludes_turn = concludes_turn


@dataclass(frozen=True, slots=True)
class SubagentRequest:
    """One delegation, as the caller describes it.

    `parent` is the agent handle delegating, not an id: the provider needs its
    session to log the admission and its inbox to deliver the reply, and looking
    both up from an id would let a caller name an agent it does not hold.
    """

    prompt: str
    parent: AgentHandle
    """Who is delegating. **Required for a spawn, and tolerated as `None` by
    `held_by` alone.**

    `_boundary_for` has an explicit `if request.parent is None` branch — a
    grant computed with no parent inherits the mount's own ceiling — but every
    provider that goes on to *spawn* reads the parent's session, scope and
    options, so declaring it optional here made eight reads in
    `ph_rlm.subagents` type errors for a case they never receive. The
    declaration is the spawn contract; the one caller that asks only for a
    grant passes `None` past it deliberately.

    The handle, not the driver: a provider reads this parent and never steers
    it. The one notice that does reach a parent's inbox (`_inject`) looks the
    driver up by session id instead, because a parent disposed while its child
    ran has no inbox to deliver to — so the wider type bought nothing and let a
    provider reach `cancel` on the agent that asked it for a child.
    """
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
    """A stable label among its siblings, which is how they are addressed. Made from
    the task when omitted, and unique either way: the seam names every child before
    its provider is asked (`SubagentService.start`)."""
    provider: str | None = None
    model: str | None = None
    """Exact selector. A provider must not silently fall back to another model —
    a child that answered on a cheaper model than the parent asked for is a
    result the parent cannot interpret."""
    model_key: str | None = None
    """A model the parent's profile lists, by key (session profiles, S7b) — what a
    spawn names, where `provider` and `model` are the route it resolves to. A key the
    list does not hold is refused, so the models an agent can reach are the ones its
    profile says (`SubagentService.resolve_model`). `None` with no route runs the
    child on its parent's."""
    reasoning_effort: str | None = None
    access: Access = "read"
    preset: str | None = None
    """A named kind of child the deployment configured (`subagent-presets`).

    Resolved into the fields below before the ceiling is checked, so a preset is
    a set of defaults and never a way past it.
    """
    profile: str | None = None
    """A named profile the parent assigns its child (session profiles, S7b), read as a
    narrowing on the parent's own mount (`ph.seams.subagent_profiles`): the tools and
    skills of what it runs, its default model, and a read-only sandbox. Resolved into
    the fields around it before the ceiling is checked; one that would widen anything
    is refused, naming the row."""
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
    paths: tuple[str, ...] | None = None
    """The extra directories the child's sandbox binds writable, canonical. `None`
    binds the parent's. Set by an assigned profile's `sandbox-allow` row
    (`resolve_profile`) and applied as a limit (`SandboxSeam.restrict_paths`), which
    can only subtract.
    """
    call_id: str | None = None
    """The tool call that asked for this child: a `task` call, or the dispatch of an
    `rlm.run` inside a cell (S2).

    Recorded on the admission so the call can be answered from the child's own log
    after a restart (`admitted_by`) — the admission is on disk before the child runs, so a
    call with no admission started no child, and one with an admission started this
    one. Without it, a delegating call cut short by a crash could only be answered
    "unknown", and the model delegated the same task again beside the child the
    resume had already put back to work.
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


SubagentAwaiter: TypeAlias = Callable[[], Awaitable[SubagentResult]]
"""How a caller waits for one delegation to settle.

Named because a provider *builds* one and `SubagentRun.result` *holds* one, in
different packages — `ph_rlm.subagents._awaiter` spelled this union out by hand
so that the two copies had nothing linking them.
"""


@dataclass(slots=True)
class SubagentRun(WireForm):
    """A live delegation. Returned at admission, before the child answers.

    `WireForm`, not `WireDataclass`: it serializes itself, but it cannot take the
    alias-derived body. That body emits every non-`None` field, and six of these
    do not travel — `owner` (which defaults to `""`, so it would always ride),
    plus `result`, `dispose`, `grant`, `scope` and `ready`, a `Grant`, a `Context`
    and an event among them. Only `id` → `runId` is a spelling the alias function would miss.

    The fields are the admission facts a parent can act on immediately: what to
    call it, where its log is, and which guarantees it actually got. `granted`
    may differ from what was asked when the available tier cannot honor the
    request, and it is the value its admission and the child's own prompt report.
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
    word for both is how `rehydrate` looked up the wrong one. Not in `to_wire`,
    which is the handle a caller is given; the service writes it onto the
    admission (`record_admitted`), because readmission after a restart has to find
    the provider that owns the child. It was missing from the log once, and a
    readmit then worked only while exactly one provider was mounted."""
    downgrade_reason: DowngradeReason | None = None
    """Why `granted` is narrower than `requested`, as a code rather than prose.

    A durable event carrying an English sentence is unparseable by the consumer
    that has to branch on it, and it goes stale silently: the reason a `write`
    was refused today stops being true when the workspace tier lands, in every
    log already written. `downgrade_text()` renders the sentence from the code,
    once."""
    result: SubagentAwaiter | None = None
    """Awaits quiescence and reports the outcome. `None` from a provider whose
    children cannot be waited on."""
    dispose: Disposer | None = None
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

    The ceiling was a documented obligation on providers before this field, and the
    second call site had already missed it: `rehydrate` builds a fresh scope for a
    settled child. Handing the scope back makes the enforcement the seam's, on both
    paths, rather than a rule a provider is trusted to remember.
    """
    ready: anyio.Event = field(default_factory=anyio.Event, repr=False, compare=False)
    """Set by the seam when the child may take its first step, and awaited by the
    provider's drive before it does.

    A provider starts the drive before it hands the run back, and only then can the
    seam bound it (`_enforce` needs `scope`), so without this a child ran ahead of
    its own ceiling. Set once the child is bounded; for a readmitted child, once its
    own children have been swept as well (L5b), so the children its first prompt
    describes are the settled ones. A child released before this is set — refused,
    or its admission unwritten — has it set by the release, and its drive, let
    through, finds itself released and runs nothing."""

    def to_wire(self) -> dict[str, JsonValue]:
        """The admission facts, for a caller holding the handle.

        No `status`: a child's state is the fold over its own `subagent/status`
        records, and a second copy on the handle would be a value frozen at whatever
        the last in-process update left."""
        wire: dict[str, JsonValue] = {
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


class ChildNotice(WireModel):
    """What a child's ending tells its parent: a message into the parent's inbox,
    carried on the ending itself (`record_ended`).

    On the child's record because the child is what ended: its own log says what its
    parent is owed, and the id says whether the parent's log has it. So a crash
    between the child's ending and the parent's write costs the parent nothing — the
    sweep that brings the parent back delivers what its log lacks
    (`SubagentService._resume`).
    """

    text: str
    summary: str
    """The line a transcript shows for it (`PluginSource.summary`)."""
    id: str = Field(default_factory=new_message_id)
    """The message's own id, which the parent's inbox records it by."""


class Admission(WireModel):
    """A child's `subagent/admitted` record: the run, the task, **the narrowing**, and
    what the seam stamps beside them.

    One model for the writer and every reader — `ChildState`, the readmission that
    rebuilds a request from it — so a key one side spells and the other looks for
    cannot come apart. `CodeDispatchRef` states the same argument for the same
    reason.

    **No session id and no parent id**: the record is in the child's own log, whose
    id is the one and whose header names the other (`delegating_parent`).

    **The narrowing is logged because a restart re-derives the ceiling from this
    record and nothing else.** `preset`, `skills` and `tools` are what `grant_for`
    resolves a child's reach from, and an admission that recorded only the run would
    come back after a restart with the parent's *whole* set — a child quietly wider
    than the one that was admitted, which is §6.5 broken by a power cut. `None`
    means "inherited everything", as it does on the request.
    """

    run_id: str
    name: str = ""
    owner: str = ""
    """The provider row that runs it, so a readmission asks that one (P11-04)."""
    prompt: str = ""
    model_provider: str = ""
    model: str = ""
    model_key: str | None = None
    """The key a spawn named beside the route it resolved to, for the audit (item 8)."""
    reasoning_effort: str | None = None
    requested_access: Access = "read"
    granted_access: Access = "read"
    downgrade_reason: DowngradeReason | None = None
    preset: str | None = None
    profile: str | None = None
    skills: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None
    paths: tuple[str, ...] | None = None
    call_id: str | None = None
    """The call that asked for it, which a cut-short call is answered by (S2)."""
    parent_turn: int | None = None
    """The seq of the parent's latest `turn/start` at admission: the turn a spawn cap
    counts it in."""
    goal_id: str | None = None
    """The parent's goal open at admission: the one its spend is charged to."""


def admission_payload(
    run: SubagentRun,
    request: SubagentRequest,
    *,
    owner: str = "",
    parent_turn: int | None = None,
    goal_id: str | None = None,
) -> dict[str, JsonValue]:
    """The `subagent/admitted` payload for `run`, admitted on `request` — an `Admission`,
    in the log's spelling."""
    admission = Admission(
        run_id=run.id,
        name=run.name,
        owner=owner,
        prompt=request.prompt.strip(),
        model_provider=run.model_provider,
        model=run.model,
        model_key=request.model_key,
        reasoning_effort=request.reasoning_effort,
        requested_access=run.requested_access,
        granted_access=run.granted_access,
        downgrade_reason=run.downgrade_reason,
        preset=request.preset,
        profile=request.profile,
        skills=request.skills,
        tools=request.tools,
        paths=request.paths,
        call_id=request.call_id,
        parent_turn=parent_turn,
        goal_id=goal_id,
    )
    # Lists rather than the model's tuples: the log's own spelling of a sequence.
    thawed = thaw_json(admission.to_wire())
    assert isinstance(thawed, dict)
    return dict[str, JsonValue](thawed)


# ------------------------------------------- a child's records, in its own log --


def record_admitted(
    ctx: Context,
    child: Session,
    run: SubagentRun,
    request: SubagentRequest,
    *,
    owner: str,
) -> SessionEvent:
    """A child's admission, in its own log — the record a resume finds it by (S2).

    Written by the seam (`SubagentService._admit`), not by a provider: the seam
    holds the resolved request and the provider's run both, and it is the one that
    knows `owner`. Beside the payload it stamps what a reader of the child needs
    from its parent's side *at this moment*: the parent's latest `turn/start`
    (`parentTurn`, which the spawn caps count a turn by) and the parent's open goal
    (`goalId`, which the child's spend is charged to). Made durable by the seam
    before the child's gate opens; a child whose admission cannot be written is
    refused.
    """
    parent = request.parent.session
    goals = ctx.get(GOALS)
    goal = goals.open(parent) if goals is not None and parent is not None else None
    payload = admission_payload(
        run,
        request,
        owner=owner,
        parent_turn=_current_turn(parent) if parent is not None else None,
        goal_id=goal.goal.id if goal is not None else None,
    )
    return _LOG.append(child, ADMITTED, payload)


def _current_turn(parent: Session) -> int | None:
    """The seq of `parent`'s latest `turn/start`: the turn a child admitted now is
    stamped with (`parentTurn`), and the one the spawn caps count."""
    opened = parent.latest("turn/start")
    return opened.seq if opened is not None else None


def record_waiting(child: Session, /, **extra: JsonValue) -> None:
    """`queued`, for a child that is waiting: for a slot (`slots`), or stopped with
    its mount (`SUSPENDED_DETAIL`).

    No barrier: a wait decides nothing a crash could get wrong. The statuses whose
    place on disk does decide something have their own doors — `record_started` for
    a restart the ladder counts, `record_ended` for an ending its parent is about to
    be told, `record_deleted` for a revocation.

    **The one module that writes `subagent/status`**, so the shape the fold reads
    has one spelling, and so the writers-of-record table can hold a provider to it:
    a provider that appends a status of its own fails `test_log_writers`, which is
    what keeps these doors from being optional.
    """
    _append_status(child, "queued", **extra)


def _append_status(
    log: Session | SessionBatch, status: SubagentStatus, /, **extra: JsonValue
) -> SessionEvent:
    return _LOG.append(log, STATUS, {"status": status, **extra})


async def record_started(
    ctx: Context, child: Session, /, *, cause: StatusCause | None = None
) -> SessionEvent:
    """`running`, for a child about to take its turn — **on its own disk first when it
    is a restart** (S10). The caller starts the attempt after this returns.

    The ladder counts `running` records with `cause: "resumed"`
    (`restarts_since_progress`), and a restart that reached only memory before the
    child took the daemon down again was never counted: a crash loop never advanced
    the count on disk, so `CHILD_RETRY_LIMIT` never tripped. A first start counts
    nothing, and neither does a woken (`rehydrated`) one, so those ride the child's
    next flush — its first model request, which is before anything they describe.

    Best effort: a log that cannot be written has bigger problems than this count,
    and refusing the child its turn would not write the record either.
    """
    event = _append_status(child, "running", **({} if cause is None else {"cause": cause}))
    if cause == "resumed":
        await session_written(ctx, child)
    return event


async def record_ended(
    ctx: Context,
    child: Session,
    status: SettledStatus,
    /,
    *,
    notice: ChildNotice | None = None,
    **extra: JsonValue,
) -> SessionEvent:
    """A child's ending, in its own log — **on its disk before the caller hands the
    result to the parent** (F1).

    Nothing else writes the child's log at that point: its last barrier was *before*
    its last model request. A parent told "done, here is the answer" by a child whose
    log ended before the answer is one repair would call interrupted on the next
    open, and readmit to do the work again. The caller delivers the result after
    this returns. Best effort, for `record_started`'s reason: an ending that cannot
    be written is still the ending.

    **What it left unfinished beneath it ends with it** (`_end_beneath`). **And what
    it tells its parent** (`notice`) rides on the ending, on the child's disk first,
    and is then delivered into the parent's inbox (`_deliver_notice`).
    """
    fields: dict[str, JsonValue] = dict(extra)
    if notice is not None:
        fields["notice"] = notice.to_wire()
    event = _append_status(child, status, **fields)
    await session_written(ctx, child)
    await _end_beneath(ctx, child)
    if notice is not None:
        agents = ctx.get(AGENTS)
        parent_id = child.header.delegating_parent
        parent = agents.get(parent_id) if agents is not None and parent_id else None
        if parent is not None:
            _deliver_notice(parent, notice)
    return event


def _deliver_notice(parent: AgentDriver, notice: ChildNotice) -> None:
    """A child's ending notice into `parent`'s inbox, without waking it.

    **Not flushed here**: a child never writes its parent's log, and it need not. The
    notice is on the child's ending, on the child's disk, so one the parent's log lost
    to a crash before its own next barrier — or one whose parent was not live to take
    it — is delivered by the sweep that brings the parent back, which finds its id
    missing. And a parent that read it had written it first, at the barrier before the
    request that read it, so it is never delivered twice.
    """
    parent.inject(
        create_user_message(
            content=[{"type": "text", "text": notice.text}],
            source=PluginSource(
                plugin="ph.seams.subagents",
                form="notice",
                summary=notice.summary[:CONTEXT_SUMMARY_MAX_CHARS],
            ),
            message_id=notice.id,
        )
    )


def _inbox_ids(log: Session) -> set[str]:
    """The id of every message delivered into `log`'s inbox."""
    return {
        as_str(as_obj(message).get("id"))
        for event in log.events
        if event.type == "agent/inbox/spliced"
        for message in as_seq(event.data.get("inserted"))
    }


async def record_deleted(ctx: Context, child: Session, /, reason: str) -> None:
    """A child's tombstone — with the `canceled` that ends it, for a child that had not
    ended — **in one batch** (S14), in its own log, and on its disk before the caller
    releases anything.

    Apart, a flush between them left a child `canceled` and not deleted. A child
    that already settled is not settled again: a `canceled` over its `done` turned a
    finished child into a revoked one. Whether it ended is read from its own log,
    not from a caller's flag, so the rule has one reader — through the seam's cached
    fold when there is one, since a long log refolded per revocation is a parent's
    teardown paying for every child it had. A child tombstoned already is left as it
    is, and **what it left unfinished beneath it ends with it** (`_end_beneath`).
    """
    await _tombstone(ctx, child, reason)
    await _end_beneath(ctx, child)


async def _tombstone(ctx: Context, child: Session, reason: str) -> None:
    """`record_deleted`'s own write, without taking anything beneath the child with
    it: what `_revoke_beneath` writes on each descendant, since its walk already
    reaches every level."""
    service = ctx.get(SUBAGENTS)
    known = service.state(child.id) if service is not None else None
    state = known if known is not None else child_state(child)
    if state.deleted:
        return
    with child.batch() as batch:
        if state.status not in SETTLED_STATUSES:
            _append_status(batch, "canceled", reason=reason)
        _LOG.append(batch, DELETED, {"reason": reason})
    await session_written(ctx, child)


async def _end_beneath(ctx: Context, child: Session) -> None:
    """**A child that ends takes what it left unfinished beneath it** — in the doors
    that end one, so it holds whichever provider or seam path ended it
    (`SubagentService._revoke_beneath`)."""
    service = ctx.get(SUBAGENTS)
    if service is not None:
        await service._revoke_beneath(child.id)


async def open_child_log(
    ctx: Context, parent: Session, run_id: str, /, **meta: JsonValue
) -> Session:
    """A child's own log, opened — resumed when one survived, else created — named and
    filed the way its parent finds it by (Phase 11).

    The id is its parent's with its run's after (`child_session_id`), and the header
    names the parent and says it is a sub-agent, with its parent's family, since the
    store files a child with its parent only while the parent is live in it. Every
    provider opens its children here, so none can name one a restart would not find;
    `meta` adds what a provider says of its own (`agentPreset`).
    """
    return await open_session(
        ctx,
        child_session_id(parent.id, run_id),
        meta={
            "parentSession": parent.id,
            "family": parent.header.family,
            "origin": "subagent",
            "delegationDepth": (parent.header.delegation_depth or 0) + 1,
            **meta,
        },
    )


def parent_went_away(name: str) -> SubagentSpawnError:
    """A spawn refused because its parent's scope was disposed mid-admission."""
    return SubagentSpawnError(
        f"subagent {name}: its parent went away while it was being admitted, so it was not started"
    )


def admitted_by(children: Mapping[str, ChildState], call: SessionEvent) -> ChildState | None:
    """The child a call record admitted, or `None` if it admitted none.

    For a delegating tool's `reconcile` (S2): `call` is the record a crash left
    unanswered — a `tool/call`, or a Code Mode dispatch's
    `tool/code-dispatch-start` — and `children` are the parent's, read from their
    own logs. The admission reaches the child's disk before the child takes a step
    (`SubagentService._admit`), so no child means none ran, and the call is safe to
    make again.
    """
    call_id = call_id_of(call)
    if not call_id:
        return None
    # One at most: a readmit writes no second admission.
    return next((state for state in children.values() if state.call_id == call_id), None)


@runtime_checkable
class ReadmittingProvider(Protocol):
    """A provider that can take an admitted child back from the log alone (P5-04).

    A daemon that stopped between a child's admission and its first turn left the
    work described in the child's own log and running nowhere. `readmit` is how the
    next daemon puts it back: the run id and the session id come from that log, so
    the child keeps its identity, its name and its log rather than arriving as a
    second child the parent never asked for.

    Its own Protocol, and not a method on `SubagentProvider`, for
    `RehydratableProvider`'s reason exactly — resuming an *un-run* child is not
    something every way of running one can do, and a `getattr` probe would report
    a provider whose method is misnamed as one that cannot do it.

    Returning `None` declines this one child without failing the sweep: the next
    daemon is starting a root, and one child it cannot rebuild must not stop the
    others from coming back.
    """

    async def readmit(
        self, request: SubagentRequest, *, run_id: str, session_id: str, restarts: int = 0
    ) -> SubagentRun | None: ...


@runtime_checkable
class RevokingProvider(Protocol):
    """A provider that can stop a child it is running and tombstone it (Phase 11).

    `SubagentService.delete` is the door a revocation goes through: a child this
    process is running is its provider's to stop — its job, its agent — and the
    provider writes the tombstone into the child's log as it lets go
    (`record_deleted`). A child nothing is running here, settled by an earlier
    process, is tombstoned by the seam itself. `False` means the provider holds no
    such child.
    """

    async def revoke(self, run_id: str, reason: str) -> bool: ...


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

    One of the two row-supplied objects that `_row_bodies` cannot find by shape: a
    provider is an *object satisfying a Protocol*, so `dict[str, SubagentProvider]`
    names no callable, and this claims its slot with `claim_key` rather than
    `claim_slot`. `_provider_fields` discriminates on the Protocol, which is true of a
    provider however it was registered.
    """

    provider: SubagentProvider
    by: Running


SpawnGuard: TypeAlias = Callable[[SubagentRequest], "Deny | str | None"]
"""A policy asked before a child exists: a reason to refuse, or `None` to allow.

Deny-only and asked before the provider is, so a refusal produces no session, no
log and no artifact — `SubagentSpawnError`'s contract. The limits row's child caps
are the first registrant (P4-04)."""


@dataclass(frozen=True, slots=True)
class _SpawnGuard:
    check: SpawnGuard
    by: Running


@dataclass(frozen=True, slots=True)
class ChildCounts:
    """A parent's children, as the spawn caps count them (`SubagentService.child_counts`)."""

    turn: int = 0
    """Those admitted in the parent's current turn, and those on their way."""
    session: int = 0
    """All of them, a deleted one included, and those on their way."""


@dataclass(frozen=True, slots=True)
class _Readmitter:
    """The provider a child's admission names, able to readmit it: its name, its row,
    and itself as a `ReadmittingProvider` — asked once per child, and carried."""

    name: str
    by: Running
    provider: ReadmittingProvider


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
    _guards: list[_SpawnGuard] = field(default_factory=list)
    _folds: SessionFoldCache[ChildState] = field(
        default_factory=lambda: SessionFoldCache(child_state, extend=extend_child_state)
    )
    """Each live child's state, folded from its own log and cached per session. The
    prompt, the model's roster tool, the spawn caps and every name lookup read it
    several times per model step.

    The *fold* stays a pure function of one log — `child_state_of` has to work on a
    stored log too — so drift is checked per child (I6), with no key spanning logs."""
    _live: dict[str, set[str]] = field(default_factory=dict)
    """The live children of each parent, by session id: kept as sessions are
    published and let go (`_published`, `_let_go`), since a child's parent link
    never changes. A parent's children cost its own children, not every session the
    mount holds."""
    _stored: dict[str, dict[str, ChildState]] = field(default_factory=dict)
    """Children this process is not running, by parent id and then session id. A
    parent is here once its stored children were read (`load_children`), and a child
    of it that is let go leaves its last state here (`_let_go`)."""
    _spawning: dict[str, set[str]] = field(default_factory=dict)
    """The names of each parent's spawns past the guards and not yet admitted, by
    parent id (`child_counts`)."""

    def register_provider(
        self, name: str, provider: SubagentProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim one delegation strategy under `name`."""
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._providers, name, _Registered(provider, by), label="subagent-provider"
        )

    def guard(self, check: SpawnGuard, *, scope: Context | None = None) -> Disposer:
        """Register a deny-only policy asked before every admission (P4-04).

        The shape `ToolRuntime.guard` has, for the same reasons: monotonic — a guard
        can refuse and never widen — and asked *before* the provider is, so a refused
        spawn produces nothing to clean up. A count of children is the seam's business
        to *ask* and a policy row's to *decide*, which is why the child caps are a
        registration here rather than a field on this seam's config.
        """
        by = self.ctx.running_for(scope)
        return claim_entry(by.owner, self._guards, _SpawnGuard(check, by), label="subagent-guard")

    def child_counts(self, parent: Session) -> ChildCounts:
        """How many children `parent` has made, this turn and in all: what a spawn cap
        counts, read from the children's own logs when a spawn is judged.

        **A spawn on its way counts as a child.** Its guards run before its provider
        builds the child, and the child is in `children` only once its admission is
        written. With nothing else counted, two spawns from one step (the driver runs a
        step's tool calls side by side) each saw the other missing, and both passed a
        cap one of them crossed. A spawn is on its parent's list from its last guard
        until its admission is written or it is refused, with no await at either
        hand-off, so it is counted once, in this turn: the one its admission stamps.

        **The turn is the parent's latest `turn/start`, by seq** (`_current_turn`). A
        child admitted before the parent had a turn counts in the turn only while the
        parent still has none. Deleted children count, and a fork starts with none,
        since no child names it.
        """
        children = self.children(parent.id).values()
        spawning = len(self._spawning.get(parent.id, ()))
        turn = _current_turn(parent)
        return ChildCounts(
            turn=spawning + sum(1 for child in children if child.parent_turn == turn),
            session=spawning + len(children),
        )

    def _named(self, request: SubagentRequest, parent: Session | None) -> str:
        """The name a spawn's child is addressed by, unique among its siblings.

        **Named by the seam, before its provider is asked**, for `child_counts`'
        reason: a provider that named a child from the admitted ones let two spawns
        from one step take one name. `start` holds the name on the parent's list of
        spawns on their way. A name asked for that a sibling has is refused, since
        names address children (`agent_message`, the roster); one left out is made
        from the task (`default_child_name`).
        """
        taken: set[str] = set()
        if parent is not None:
            taken.update(state.name for state in self.children(parent.id).values())
            taken.update(self._spawning.get(parent.id, ()))
        if request.name is None:
            return default_child_name(request.prompt, secrets.token_hex(4), taken=taken)
        name = request.name.strip()
        if not name or len(name) > MAX_NAME_CHARS:
            raise SubagentSpawnError(f"a subagent name must be 1..{MAX_NAME_CHARS} characters")
        if name in taken:
            raise SubagentSpawnError(
                f'a sibling is already named "{name}"; names address children, so they '
                "must be unique among siblings"
            )
        return name

    def _landed(self, parent: Session | None, name: str) -> None:
        """Take a spawn off its parent's list: admitted, and a child from here, or
        refused. A second call does nothing."""
        if parent is None or (names := self._spawning.get(parent.id)) is None:
            return
        names.discard(name)
        if not names:
            del self._spawning[parent.id]

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
        skills = self.ctx.get(SKILLS)
        tools = self.ctx.get(TOOLS)
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
        service = self.ctx.get(SUBAGENT_PRESETS)
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

    def resolve_model(self, request: SubagentRequest) -> SubagentRequest:
        """The route a spawn's model key names, from the parent's own list (S7b, item 8).

        The key is the request's, or the one a skill it names asks for in its front
        matter (`model: classify`) — the skill directs the child, so it may say what
        the child should think with. Resolved by the mount's `models` list, which is
        the parent's profile as it runs, `/model` included: a key it does not hold is
        refused, naming what it does. A request that already carries a route — a
        readmitted child, fixed at its admission — is left as it is.
        """
        if request.provider or request.model:
            return request
        key = request.model_key or self._skill_model(request)
        if not key:
            return request
        try:
            entry = choose(self.ctx, ModelChoice(key=key))
        except ModelChoiceError as error:
            raise SubagentSpawnError(f"a child cannot run on {key!r}: {error}") from error
        return replace(
            request,
            model_key=key,
            provider=entry.route.provider,
            model=entry.route.model,
            reasoning_effort=request.reasoning_effort or entry.route.reasoning_effort,
        )

    def _skill_model(self, request: SubagentRequest) -> str:
        """The model the skills a spawn names ask for, or `""` when none does."""
        service = self.ctx.get(SKILLS)
        if not request.skills or service is None:
            return ""
        boundary = self._delegating_boundary(request)
        asked = {
            name: skill.model
            for name in request.skills
            if (skill := service.get(name, boundary)) is not None and skill.model
        }
        if len(set(asked.values())) > 1:
            said = ", ".join(f"{name} names {key}" for name, key in sorted(asked.items()))
            raise SubagentSpawnError(
                f"the skills this child is given name different models ({said}); "
                "name the one it should run on with `model=`"
            )
        return next(iter(asked.values()), "")

    async def resolve_profile(self, request: SubagentRequest) -> SubagentRequest:
        """A profile the parent assigns, as the narrowing it is (S7b, item 9).

        Composed by the host (`ctx.named_profiles`) and read against the parent's
        mount (`ph.seams.subagent_profiles.narrowing`), then written into the request
        as the tools, skills, model and access it resolves to — so the ceiling below
        checks what it gives, and the admission records it: a child's reach is fixed
        when it is admitted. **A ceiling, not defaults**: what a spawn names beside
        it may narrow further, and naming more than it gives is refused.
        """
        name = request.profile
        if name is None:
            return request
        composer = self.ctx.get(NAMED_PROFILES)
        if composer is None:
            raise SubagentSpawnError(
                f'this deployment cannot compose a named profile, so "{name}" cannot be '
                "assigned to a child"
            )
        try:
            # On a worker thread: composing reads and parses every layer file, and a
            # spawn runs on the event loop every root of a daemon shares.
            composed = await anyio.to_thread.run_sync(composer.compose, name)
        except (LoaderError, OSError, ValueError) as error:
            raise SubagentSpawnError(f'profile "{name}" does not compose: {error}') from error
        boundary = self._delegating_boundary(request)
        held_skills, held_tools = self.held_by(request, boundary)
        reach = ChildReach(
            ctx=self.ctx, boundary=boundary, agent=request.parent.id, skills=held_skills
        )
        try:
            narrowed = narrowing(composed, reach, held_tools=held_tools)
        except NarrowingRefused as refusal:
            raise SubagentSpawnError(f'profile "{name}": {refusal}') from refusal
        for kind, asked, gives in (
            ("tool", request.tools, narrowed.tools),
            ("skill", request.skills, narrowed.skills),
        ):
            beyond = sorted(set(asked or ()) - set(gives))
            if beyond:
                raise SubagentSpawnError(
                    f'profile "{name}" does not give its child the {kind}s {", ".join(beyond)}'
                )
        if narrowed.limit.read_only and request.access == "write":
            raise SubagentSpawnError(f'profile "{name}" is read-only, so its child cannot write')
        return replace(
            request,
            tools=request.tools if request.tools is not None else narrowed.tools,
            skills=request.skills if request.skills is not None else narrowed.skills,
            model_key=request.model_key or narrowed.limit.model_key,
            paths=narrowed.limit.writable_paths,
        )

    def grant_for(
        self,
        request: SubagentRequest,
        held: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
        *,
        boundary: Context | None = None,
    ) -> Grant:
        """Materialize what this child may reach, from a parent that still exists.

        `None` means "everything the parent holds" and is written out as that explicit
        list. The isolation chain already bounds the child; what the list is for is the
        ruling — **a child's capability is fixed at admission**, so this records what the
        parent held *then* rather than deferring to what it holds whenever the child is
        next asked. See `Grant`.
        """
        held_skills, held_tools = held if held is not None else self.held_by(request, boundary)
        named = request.skills
        skills = self.ctx.get(SKILLS)
        target = boundary if boundary is not None else self._delegating_boundary(request)
        return Grant(
            skills=named if named is not None else held_skills,
            tools=request.tools if request.tools is not None else held_tools,
            paths=request.paths,
            brief=(
                _brief_text(skills, named, target, request.parent.session)
                if named and skills is not None
                else ""
            ),
        )

    async def _end_child(self, session_id: str, detail: str) -> None:
        """End, in its own log, a child the seam is giving up on.

        **A child with an admission and no ending is one nothing can release.**
        Its log says it was admitted, so `child_is_live` reads it as working, and
        its parent is held out of passivation by a child that no longer exists
        (K3, K4). It has to *end*, and only its log can end it.

        Best effort, and never raised: this is how the seam gives up on a child —
        refused, unresumable, spent — and a root coming back must not be held
        hostage by a log it could not write. A child whose spawn was refused before
        it had a log leaves nothing to end.
        """
        try:
            await self._write_child(
                session_id, lambda child: record_ended(self.ctx, child, "error", detail=detail)
            )
        except Exception:
            log.exception("ph.seams.subagents: %s could not be ended in its own log", session_id)

    async def _write_child(
        self, session_id: str, write: Callable[[Session], Awaitable[object]]
    ) -> None:
        """Write to a child's own log, whether or not this process is running it.

        A live child's session is written as it is. One this process does not hold —
        settled by an earlier process, or not readmitted yet — is claimed and loaded
        without being resumed (`stored_session`), written, flushed, and let go: the
        sweep ends a child it will not readmit this way, and records a hold on one
        waiting for a credential, and neither is a reason to rebuild its runtime.
        Nothing to write to — no live session, and none stored — is a no-op.
        """
        sessions = self.ctx.require(SESSIONS)
        live = sessions.get(session_id)
        if live is not None:
            await write(live)
            return
        store = self.ctx.get(SESSION_PERSISTENCE)
        if store is None or not store.exists(session_id):
            return
        child = await stored_session(self.ctx, session_id)
        try:
            await write(child)
            await session_written(self.ctx, child)
        finally:
            sessions.dispose(session_id)

    async def _abandon(self, run: SubagentRun) -> None:
        """Release a child this seam has decided not to admit (K3).

        Best-effort: the refusal is what the caller has to see, and a disposer
        that throws must not replace it with its own exception. `releasing`
        rather than a bare shield for the reason it gives — a disposer that hangs
        must not make the refusal unkillable.
        """
        if run.dispose is None:
            return
        with releasing(), suppress(Exception):
            await maybe_await(run.dispose())

    async def _admit(
        self,
        run: SubagentRun,
        *,
        owner: str,
        grant: Grant,
        held: tuple[tuple[str, ...], tuple[str, ...]],
        boundary: Context,
        request: SubagentRequest | None = None,
    ) -> SubagentRun:
        """Bind a child a provider has already started, or release it (K3, L6).

        **The sequence, in one place, because both spawners need all of it.**
        `_enforce` reads `run.scope`, which only the provider can produce, so the
        check cannot move ahead of the spawn — and by the time it refuses there
        is a child driving. Three things then have to happen and no two of them
        are optional: the child is released, its admission is *ended* in the log,
        and the refusal reaches the caller.

        `start` had all three and `_readmit_one` had none of them: it called
        `_enforce` bare, so a `check_grant` refusal after a profile edit left the
        child running unbounded and undisposed while `resume_children`'s
        `except` ended its record — the K3 shape arriving through the path
        K4 had already been written for. Two spawners with one admission each
        was the defect; one admission both call is the fix.

        `request` is `start`'s, and with it **the seam writes the child's admission,
        into the child's own log** (S2, Phase 11) — the one record a resume finds a
        child by. The seam rather than the provider, because it holds both halves:
        the resolved request and the provider's run, and `owner`, which is how a
        readmission finds the provider again. It reaches the child's disk before the
        gate opens, fail-closed as a durable intent is: one that cannot be written is
        a refusal like the ceiling's. A provider that opened no log for its child is
        refused too, since that log is the child's only record. A readmit writes no
        admission — the one it was rebuilt from is already there — and passes none.

        `start`'s spawn comes off its parent's list in the step that writes its
        admission, which is the step it starts counting as a child in (`child_counts`).

        The caller opens the child's gate (`SubagentRun.ready`) once this returns:
        `start` at once, a readmission after its own children are swept.
        """
        # Stamped here rather than trusted from the provider: the service is what
        # knows which name the caller asked for, and `rehydrate` has to be able
        # to find its way back to the same provider.
        run.owner = owner
        child = self.ctx.require(SESSIONS).get(run.session_id)
        # A parent with no log of its own has nothing a child could be found by, and
        # nothing is recorded for its child: the one case a spawn is not durable.
        parent = request.parent.session if request is not None else None
        try:
            if request is not None and parent is not None:
                if child is None or not _names_parent(child, parent.id):
                    # Its log is how its parent finds it after a restart
                    # (`SessionArchive.descendants_of`), so a child with no log, or one
                    # that does not name its parent, is one nothing could bring back.
                    raise SubagentSpawnError(
                        f'subagent {run.name}: the "{owner}" provider opened no log for it '
                        "that names its parent, and a child's own log is its only record, "
                        "so it was not started"
                    )
                # Before the ceiling: a child refused below reads as admitted and then
                # ended, rather than as a log with no account of what it was.
                record_admitted(self.ctx, child, run, request, owner=owner)
                self._landed(parent, run.name)
            try:
                self._enforce(grant, run, held, boundary)
            except InactiveScopeError as gone:
                # The ceiling registers on the child's scope, which a parent that went
                # away took with it: the same refusal a provider's own builds get.
                raise parent_went_away(run.name) from gone
            if (
                parent is not None
                and child is not None
                and not await session_written(self.ctx, child)
            ):
                raise SubagentSpawnError(
                    f"subagent {run.name}: its admission could not be written to its "
                    "own log, so it was not started"
                )
        except SubagentSpawnError as refused:
            await self._abandon(run)
            await self._end_child(run.session_id, str(refused))
            raise
        self._runs[run.id] = run
        return run

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
        request = self.resolve_model(self.resolve_preset(await self.resolve_profile(request)))
        parent = request.parent.session
        if parent is not None:
            # Every child the parent has, on disk as well as in memory, before anything
            # counts them — a spawn cap, the names its children have taken. Once
            # per mount (`load_children`); a parent resumed through the sweep has it.
            await self.load_children(parent.id, parent.header.family)
        # Guards first: a refusal here has nothing to unwind, which is the
        # contract `SubagentSpawnError` states. Bound to the registering row, as
        # every registry-invoked body is (P6-29).
        for guard in list(self._guards):
            with running(guard.by):
                reason = guard.check(request)
            if reason is not None:
                raise _refused(reason)
        child_name = self._named(request, parent)
        request = replace(request, name=child_name)
        if parent is not None:
            # On its way from here, with no await since its guards (`child_counts`).
            self._spawning.setdefault(parent.id, set()).add(child_name)
        try:
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
            try:
                with running(entry.by):
                    run = await entry.provider.start(request)
            except InactiveScopeError as gone:
                # **A spawn whose parent's scope died under it is a refusal**, whichever
                # provider was building it: every registration a child needs — its
                # workspace, its release, its job — is on a scope inside the parent's.
                # The provider releases what it built and lets this through; the caller
                # is owed the spawn's own answer, with a code, not a raw scope error.
                raise parent_went_away(child_name) from gone
            # The seam's to give, and stamped as `_admit` stamps the owner.
            run.name = child_name
            admitted = await self._admit(
                run, owner=name, grant=grant, held=held, boundary=boundary, request=request
            )
        finally:
            self._landed(parent, child_name)
        admitted.ready.set()
        return admitted

    def children(self, parent_id: str) -> dict[str, ChildState]:
        """One parent's children by run id, as their own logs tell it (Phase 11).

        The read every consumer should use when it has a `ctx`. A child this process
        runs through its own cached fold; one it does not, from what
        `load_children` read off the store or what its session held when it was let
        go — a live child's state winning over a stored copy of it. A log with no
        admission is not a child. In admission order.
        """
        found = dict(self._stored.get(parent_id, {}))
        sessions = self.ctx.get(SESSIONS)
        for session_id in self._live.get(parent_id, ()):
            session = sessions.get(session_id) if sessions is not None else None
            if session is not None:
                found[session_id] = self._folds.read(session)
        admitted = sorted(
            (state for state in found.values() if state.admitted),
            key=lambda state: (state.admitted_at, state.session_id),
        )
        return {state.run_id: state for state in admitted}

    def family(self, parent_id: str) -> list[ChildState]:
        """Everything beneath `parent_id` — its children, theirs, and so on — each
        before its own children, siblings in admission order.

        The one walk of a delegation tree: a goal's spend, the daemon's panel, what
        holds a root, what waits for a credential. Cycle-safe for `descendants`'
        reason — a log claiming an ancestor as its child costs a wasted lookup, not a
        hang. Every level, once the tree's root was read (`load_children`), which a
        resumed session's is as it opens.
        """
        found: list[ChildState] = []
        seen = {parent_id}
        stack = list(reversed(self.children(parent_id).values()))
        while stack:
            state = stack.pop()
            if state.session_id in seen:
                continue
            seen.add(state.session_id)
            found.append(state)
            stack.extend(reversed(self.children(state.session_id).values()))
        return found

    async def load_children(self, parent_id: str, family: str) -> dict[str, ChildState]:
        """`children`, with everything beneath `parent_id` on disk that this process is
        not running read in — **the whole tree, not one level**.

        One read of the store (`SessionArchive.descendants_of`) finds every level,
        since a tree is filed in one family under its root's id; each state is filed
        under the parent its own header names, and every parent in the tree counts as
        read. So a restart sees a grandchild beneath a child it did not readmit —
        which one level at a time left unread, and so uncounted, unlisted and never
        ended.

        Once per tree per mount: a stored child changes only when this process opens
        it, and then it is live — or when its session is let go, which hands its state
        back (`_let_go`). A resumed session's tree is read as it opens
        (`open_session`); the sweep, a spawn, a revocation and the crash checks ask
        again, which then costs a lookup. On a worker thread, and only the records a
        child's state is folded from (`CHILD_EVENT_TYPES`): a child's log is mostly
        streamed chunks, and a mount runs on the loop every root of a daemon shares.

        `family` is the parent's (`SessionHeader.family`) — every descendant's too,
        since a tree is filed with its root — and is where the store looks.
        """
        if parent_id not in self._stored:
            store = self.ctx.get(SESSION_PERSISTENCE)
            sessions = self.ctx.get(SESSIONS)
            live = {one.id for one in sessions.list()} if sessions is not None else set()
            tree = (
                {}
                if store is None
                else await anyio.to_thread.run_sync(
                    lambda: _stored_tree(store, parent_id, family, live)
                )
            )
            self._stored.setdefault(parent_id, {})
            for owner, states in tree.items():
                # What is already known of a child wins over this read of it.
                self._stored[owner] = states | self._stored.get(owner, {})
        return self.children(parent_id)

    def delegated_tokens(self, parent_id: str, goal_id: str | None = None) -> int:
        """What `parent_id`'s children spent — only those admitted under `goal_id`, when
        one is named — and everything beneath them, from their own logs.

        **Every level**: a goal is a budget for a delegation tree, and a grandchild's
        answers are part of what the goal asked for. The roster this replaces charged
        one level only — a child's own answers, mirrored into its parent's log — so a
        grandchild's spend, and every child's compactions, reached no goal at all.
        """
        return sum(
            state.tokens + sum(below.tokens for below in self.family(state.session_id))
            for state in self.children(parent_id).values()
            if goal_id is None or state.goal_id == goal_id
        )

    def state(self, session_id: str) -> ChildState | None:
        """One child's state by its session id, live or stored, or `None`."""
        sessions = self.ctx.get(SESSIONS)
        live = sessions.get(session_id) if sessions is not None else None
        if live is not None and live.header.delegating_parent:
            state = self._folds.read(live)
            return state if state.admitted else None
        for stored in self._stored.values():
            if session_id in stored:
                return stored[session_id]
        return None

    def _published(self, session: Session) -> None:
        """A session was published: index it under its parent when it is a child."""
        parent_id = session.header.delegating_parent
        if parent_id:
            self._live.setdefault(parent_id, set()).add(session.id)

    def _let_go(self, session: Session) -> None:
        """A session was let go: forget its fold, and keep a child's last state as its
        stored copy — the state its log now holds, which nothing will change until
        this process opens it again. Kept only for a parent whose stored children
        were read; for any other, the store answers when they are."""
        parent_id = session.header.delegating_parent
        if parent_id:
            self._live.get(parent_id, set()).discard(session.id)
            stored = self._stored.get(parent_id)
            state = self._folds.read(session)
            if stored is not None and state.admitted:
                stored[session.id] = state
        self._folds.forget(session.id)

    def forget_session(self, session_id: str) -> None:
        """Drop what this service cached about one session: its own fold, and the
        stored tree it read for it, every level of which `load_children` filed."""
        self._folds.forget(session_id)
        for state in self.family(session_id):
            self._stored.pop(state.session_id, None)
        self._stored.pop(session_id, None)

    def stale_folds(self, sessions: Iterable[Session]) -> list[str]:
        """Cached child states that no longer equal the fold of their log (I6).

        Asked of the cache rather than reconstructed here. A child's state is what
        the prompt tells its parent about it and what the interruption ladder counts
        starts against, so a drifted one is a parent reasoning about a child its log
        does not describe.
        """
        return self._folds.stale(sessions)

    def name_of(self, agent_id: str) -> str:
        """What an agent is called — the name its own admission records — or its id."""
        state = self.state(agent_id)
        return (state.name or agent_id) if state is not None else agent_id

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
        if self.ctx.require(AGENTS).get(session_id) is not None:
            return True
        run = next((one for one in self._runs.values() if one.session_id == session_id), None)
        return bool(run is not None and await self.rehydrate(run.id))

    async def resume_children(self, parent: AgentDriver, *, retry_limit: int) -> Sequence[str]:
        """What a resumed root owes its unfinished children (P5-04). Returns the revived.

        Two opposite answers to two states, which is why this exists rather than
        one sweep over "everything unsettled":

        * **queued** — admitted and never run. Re-driven, because nothing was
          claimed, spent or written and the work is the same work.
        * **running** — interrupted mid-turn. Put back on the *ladder*: its turn
          is closed on resume and its task is presented again, up to
          `retry_limit` times, after which it is failed and says so.

        **The ladder is what makes re-running an interrupted child sound**, and it
        is the piece this row shipped without at first. A child that started a
        turn has *claimed* its task from its inbox, so simply driving it again
        finds nothing pending and ends at step zero reporting `completed` — the
        empty turn P5-04 found at the root, and a parent told its child succeeded
        when it did not. Re-presenting the task is what makes the next attempt a
        real one, and the count is what stops it being infinite.

        Doing nothing was never the third option: `child_is_live` counts an
        unsettled child, so a row nothing will ever move keeps its whole root out
        of passivation for the life of the process while the parent waits on a
        reply nobody is writing.

        **Every decision is written in the log it is about** (Phase 11). The ladder
        reads a child's starts and answers from the child's own log, which is where
        they are, so nothing has to be caught up first. A child it gives up on is
        ended in its own log; one it puts back is readmitted, and its drive writes
        the restart on its own disk before the attempt (S10). Nothing orders one
        child's records against another's or against the parent's, so a crash
        mid-sweep leaves each child decided or undecided, and the next start decides
        the undecided the same way.

        `retry_limit` has **no default**, and that is P6-32's rule rather than an
        inconvenience: how many attempts work is worth is the host's policy, and
        a seam that answered it for a caller who said nothing would be choosing
        one. The daemon states it beside the root's own ladder
        (`ph_app.daemon.recovery.CHILD_RETRY_LIMIT`), which is where somebody
        tuning restart behavior will already be looking.
        """
        session = parent.session
        if session is None:
            return []
        await self.load_children(session.id, session.header.family)
        return await self._resume(parent, retry_limit=retry_limit)

    async def readmit_waiting(self, parent: AgentDriver, *, retry_limit: int) -> Sequence[str]:
        """Put back to work the children of `parent` held for a credential that has
        since arrived (T5), and theirs. Returns the revived.

        The same sweep a resume runs, asked again: a child still missing its
        credential stays held and appends nothing, and one whose name is here now has
        its hold settled and is readmitted. For whoever hands a deployment a
        credential — the daemon's `credentials/store`.

        **Every level** (L5b): a child that is running may hold children of its own
        waiting for the same name. So each live child of `parent` is asked too.
        `retry_limit` is the host's, for the sweep a readmitted child's own children
        get.
        """
        session = parent.session
        if session is None:
            return []
        await self.load_children(session.id, session.header.family)
        revived = list(await self._resume(parent, retry_limit=retry_limit))
        for state in self.children(session.id).values():
            child = self.ctx.require(AGENTS).get(state.session_id)
            if child is not None:
                revived.extend(await self.readmit_waiting(child, retry_limit=retry_limit))
        return revived

    async def _resume(self, parent: AgentDriver, *, retry_limit: int) -> Sequence[str]:
        """Decide, child by child, what each of `parent`'s unfinished children becomes:
        ended, held, or running again. Returns the run ids running again.

        **Only children this process is not running**, `queued` or interrupted
        `running`. For each, in its own log:

        * **Nothing can readmit it** — ended. Left live, it holds its parent out of
          passivation for a reply nobody is writing.
        * **Interrupted, and its ladder spent** — ended, saying how many times.
          Re-presenting the task is what makes a restart real, and the count is what
          stops it being infinite; an answer forgives the restarts before it
          (`restarts_since_progress`).
        * **Its route names a credential this deployment cannot supply** (T5) — held,
          not readmitted: readmitted, it would spend its ladder failing on a key a
          person could supply in a second. It stays live, no start is counted, and
          its own log says which name it waits for. `readmit_waiting` asks again.
        * **Otherwise readmitted**, with the ceiling re-derived from its admission —
          `check_grant`, `grant_for`, `_enforce` — so a child comes back from a power
          cut no wider than it was admitted (§6.5). The spawn guards do not run: they
          gate new work, and a cap would count the child against itself. Its own
          children are swept before its first step (L5b, `_sweep_readmitted`).

        One child that cannot be rebuilt is logged and ended rather than failing the
        sweep (K4): a root coming back must not be held hostage by the least
        recoverable thing it holds, and **a provider that declines is an ending too**
        (L6).

        **An ended child's notice reaches its parent** (`ChildNotice`): one its parent's
        log lacks — a crash came between the child's ending and the parent's write — is
        delivered now, and one the log has is not delivered again.

        **An ended child's unfinished descendants are revoked** (`_revoke_beneath`):
        one given up on here takes them with it as its ending is written
        (`record_ended`), and one that ended in an earlier process left them to a
        crash that came before they were.
        """
        session = parent.session
        if session is None:
            return []
        revived: list[str] = []
        delivered: set[str] | None = None
        for state in self.children(session.id).values():
            if state.run_id in self._runs:
                continue
            if not child_is_live(state):
                await self._revoke_beneath(state.session_id)
                if state.notice is not None and not state.deleted:
                    delivered = _inbox_ids(session) if delivered is None else delivered
                    if state.notice.id not in delivered:
                        _deliver_notice(parent, state.notice)
                continue
            readmitter = self._readmitter(state)
            interrupted = state.status == "running"
            spent = interrupted and restarts_since_progress(state) >= retry_limit
            if readmitter is None or spent:
                await self._end_child(
                    state.session_id,
                    exhausted_detail(retry_limit)
                    if spent
                    else UNRECOVERABLE_DETAIL
                    if interrupted
                    else "no provider here can start this child; its transcript is on disk",
                )
                continue
            try:
                request = self._request_of(parent, state)
                if await self._held_for_credential(state, request):
                    continue
                run = await self._readmit_one(state, request, readmitter)
            except Exception as error:
                log.exception("ph.seams.subagents: %s could not be readmitted", state.run_id)
                await self._end_child(state.session_id, f"this child could not be resumed: {error}")
                continue
            if run is None:
                await self._end_child(
                    state.session_id, "the provider that owns this child could not resume it"
                )
                continue
            await self._sweep_readmitted(run, retry_limit=retry_limit)
            revived.append(state.run_id)
        return revived

    async def _revoke_beneath(self, session_id: str) -> None:
        """Tombstone (`PARENT_TEARDOWN`), each in its own log, every unfinished
        descendant of a child that ended, that this process is not running.

        **A running one is its provider's**: a child's own children are effects of its
        scope, revoked as it unwinds (I2). What that teardown never reaches is a child
        this process is not running — held for a credential, left beneath a child a
        restart readmitted, or reached on disk by a delete — and what a crash between a
        child's ending and its children's left behind. Left alone, such a descendant
        reads as working for good: nothing readmits a child beneath one that ended, so
        it held its root out of passivation and asked for a credential it would never
        use. Every level in one walk (`family`), one log at a time, best effort.
        """
        for state in self.family(session_id):
            if not child_is_live(state) or state.run_id in self._runs:
                continue
            try:
                await self._write_child(
                    state.session_id,
                    lambda child: _tombstone(self.ctx, child, PARENT_TEARDOWN),
                )
            except Exception:
                log.exception(
                    "ph.seams.subagents: %s could not be revoked beneath its ended parent",
                    state.session_id,
                )

    async def _sweep_readmitted(self, run: SubagentRun, *, retry_limit: int) -> None:
        """A readmitted child's own children, swept as its parent's were (L5b), and
        then its gate opened (`SubagentRun.ready`).

        Logged and left rather than raised: the child itself is readmitted, and one
        level that cannot be swept must not undo the level above it. The gate opens
        whatever happened, since the child was bounded before this ran.
        """
        try:
            child = self.ctx.require(AGENTS).get(run.session_id)
            if child is not None:
                await self.resume_children(child, retry_limit=retry_limit)
        except Exception:
            log.exception("ph.seams.subagents: %s's own children could not be resumed", run.id)
        finally:
            run.ready.set()

    def _request_of(self, parent: AgentDriver, state: ChildState) -> SubagentRequest:
        """The request a child was admitted with, rebuilt from its own admission."""
        admission = state.admission or Admission(run_id=state.run_id)
        return self.resolve_preset(
            SubagentRequest(
                prompt=admission.prompt,
                parent=parent,
                name=admission.name or None,
                provider=admission.model_provider or None,
                model=admission.model or None,
                model_key=admission.model_key,
                reasoning_effort=admission.reasoning_effort,
                access=admission.requested_access,
                preset=admission.preset,
                skills=admission.skills,
                tools=admission.tools,
                paths=admission.paths,
                call_id=admission.call_id,
            )
        )

    async def _held_for_credential(self, state: ChildState, request: SubagentRequest) -> bool:
        """Whether this child waits for a credential, with its own log made to say so.

        Asked of the route first, without opening anything: a child whose log already
        says what is true — waiting for this name, or for nothing — needs nothing
        written, and most readmitted children are that. Only a hold that starts or
        ends opens the child's log to record it. A held child that was interrupted
        mid-turn is marked `queued` beside the hold, since it is not running.
        """
        name = missing_credential(self.ctx, *child_route(request))
        if name == state.awaiting:
            return name is not None

        async def record(child: Session) -> None:
            await hold_for_credential(self.ctx, child, SESSION_HOLDER, *child_route(request))
            if name is not None and state.status == "running":
                record_waiting(child, detail=INTERRUPTED_DETAIL)

        await self._write_child(state.session_id, record)
        return name is not None

    async def _readmit_one(
        self, state: ChildState, request: SubagentRequest, readmitter: _Readmitter
    ) -> SubagentRun | None:
        """One child, through the admission path it originally took."""
        boundary = self._delegating_boundary(request)
        held = self.held_by(request, boundary)
        self.check_grant(request, held)
        grant = self.grant_for(request, held, boundary=boundary)
        with running(readmitter.by):
            run = await readmitter.provider.readmit(
                request,
                run_id=state.run_id,
                session_id=state.session_id,
                # How many times this child has *already* been started, which is
                # what tells a restart from a first run — and it is `starts`,
                # never `restarts_since_progress`: an answer forgives the ladder,
                # so a child that got somewhere and was then stopped again would
                # otherwise be readmitted as though it had never run, its restart
                # go unrecorded, and the ladder never count it again.
                restarts=state.starts,
            )
        if run is None:
            return None
        return await self._admit(
            run, owner=readmitter.name, grant=grant, held=held, boundary=boundary
        )

    def _readmitter(self, state: ChildState) -> _Readmitter | None:
        """The provider that could put this child back, or `None` if none can.

        By the `owner` its admission records, so it is the provider that ran it — not
        whichever one happens to be mounted alone.
        """
        name = self.resolve(state.owner or None)
        entry = self._providers.get(name or "")
        if name is None or entry is None or not isinstance(entry.provider, ReadmittingProvider):
            return None
        return _Readmitter(name=name, by=entry.by, provider=entry.provider)

    async def delete(self, parent: Session, run_id: str, *, reason: str) -> bool:
        """Revoke one of `parent`'s children, with a tombstone in its own log. Its
        transcript stays on disk. `False` when there is no such child, or it was
        revoked already.

        A tombstone rather than a removal, because the child's log and artifacts
        outlive it: a parent looking for what a revoked child did should find the
        revocation, not a gap. **Any child, live or not** (Phase 11): one this process
        runs is its provider's to stop (`RevokingProvider`), which writes the
        tombstone as it lets go; one settled by an earlier process is tombstoned
        here. A revocation used to reach only children this process held in memory.
        Either way what it left unfinished beneath it goes with it (`record_deleted`).
        """
        state = (await self.load_children(parent.id, parent.header.family)).get(run_id)
        if state is None or state.deleted:
            return False
        run = self._runs.get(run_id)
        entry = self._providers.get(run.owner) if run is not None else None
        if entry is not None and isinstance(entry.provider, RevokingProvider):
            with running(entry.by):
                if await entry.provider.revoke(run_id, reason):
                    return True
        await self._write_child(
            state.session_id, lambda child: record_deleted(self.ctx, child, reason)
        )
        self.forget(run_id)
        return True

    async def rehydrate(self, run_id: str) -> bool:
        """Make a settled child addressable again (P3-13).

        A child that finished had its agent released, so it has no inbox to steer
        into — but its session and its log are both still there.
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


@plugin("subagent-presets", affects="environment", config=PresetConfig)
async def presets(ctx: Context, config: PresetConfig) -> None:
    """Publish the deployment's named child kinds. None ship in `ph-base`."""
    ctx.provide(SUBAGENT_PRESETS, SubagentPresetService(presets=dict(config.presets)))


@plugin("subagents", affects="environment", inject=[SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Mount the subagent seam definition. No provider ships in ph-base."""
    service = SubagentService(ctx=ctx)
    ctx.provide(SUBAGENTS, service)
    # A child let go keeps its last state as its stored copy, and its fold goes: a
    # disposed session's cached projection is a value nobody can reach.
    ctx.on("session/created", service._published)
    ctx.on("session/disposed", service._let_go)
    contribute_fold_cache(
        ctx, id="subagent-fold-cache", subject="subagent state", stale=service.stale_folds
    )


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
    paths: tuple[str, ...] | None = None
    """The extra writable directories, where an assigned profile narrowed them."""
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
        skills = ctx.get(SKILLS)
        if skills is not None:
            skills.restrict(SkillRestriction(allow=frozenset(self.skills)), scope=scope)
        tools = ctx.get(TOOLS)
        if tools is not None:
            tools.restrict(ToolRestriction(allow=frozenset(self.tools)), scope=scope)
        sandbox = ctx.get(SANDBOX)
        if self.paths is not None and sandbox is not None:
            sandbox.restrict_paths(self.paths, scope=scope)
        prompt = ctx.get(SYSTEM_PROMPT)
        if self.brief and prompt is not None:
            prompt.section(
                PromptSection(name="subagent:brief", text=self.brief, order=ORDER_BRIEF),
                scope=scope,
            )


def _brief_text(
    skills: SkillService, named: Sequence[str], scope: Context, session: Session | None
) -> str:
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
        # Read into a child's prompt is read at runtime, for the audit (S8).
        body = skills.body(name, scope, session=session, via="brief")
        if body:
            parts.append(f"## {name}\n\n{body.strip()}")
    if not parts:
        return ""
    return (
        "You were delegated this task to follow the instructions below. "
        "They are not background reading.\n\n" + "\n\n".join(parts)
    )


_UNADMITTED = Admission(run_id="")
"""What a `ChildState` with no admission reads as — never handed out by a reader."""

_STATUSES: Mapping[str, SubagentStatus] = literal_lookup(SubagentStatus)
_CAUSES: Mapping[str, StatusCause] = literal_lookup(StatusCause)


@dataclass(frozen=True, slots=True)
class ChildState:
    """One child, as its own log tells it (Phase 11).

    The fold of a child's own log — its admission, its statuses, its tombstone, its
    answers, its credential hold — and of nothing else. No parent keeps a copy, so
    there is no second account of the child to disagree with this one: every rule
    that used to keep a parent's roster in step with its child's log (S2, S10, F1,
    L5) is a rule about one log now.

    `admission` is `None` for a log that has none — or one no `Admission` parses —
    which is not a child: a workspace can reach the disk before the admission does,
    and a spawn refused before it was admitted leaves such a log behind. Every reader
    skips it; the properties below read an admitted child's record.

    `starts`, `resumes` and `resumes_at_last_answer` are the interruption ladder's
    whole state, all folded and all totals: the log already records one `running`
    per drive and one `assistant/message` per answer, so "how many times has this
    been started" and "how many restarts had there been at its last answer" are
    questions about records that are already there. What the ladder *makes* of them
    is `restarts_since_progress`, beside the sweep that decides (P3).
    """

    session_id: str
    parent_id: str
    admission: Admission | None = None
    admitted_at: float = 0.0
    status: SubagentStatus = "queued"
    """`queued` until its first status: admitted, and not yet started."""
    cause: StatusCause | None = None
    detail: str | None = None
    answer_preview: str | None = None
    notice: ChildNotice | None = None
    """What its ending told its parent (`record_ended`), or `None`."""
    deleted: bool = False
    deleted_reason: str | None = None
    awaiting: str | None = None
    """The credential this child is held for (T5), by name, or `None`."""
    starts: int = 0
    resumes: int = 0
    resumes_at_last_answer: int = 0
    tokens: int = 0
    """What it spent — its answers and its compactions — by `TokenUsage.total`."""

    @property
    def admitted(self) -> bool:
        return self.admission is not None

    @property
    def _record(self) -> Admission:
        return self.admission or _UNADMITTED

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def owner(self) -> str:
        return self._record.owner

    @property
    def call_id(self) -> str | None:
        return self._record.call_id

    @property
    def goal_id(self) -> str | None:
        return self._record.goal_id

    @property
    def parent_turn(self) -> int | None:
        return self._record.parent_turn

    @property
    def model(self) -> str:
        return self._record.model

    @property
    def granted_access(self) -> Access:
        return self._record.granted_access

    @property
    def downgrade_reason(self) -> DowngradeReason | None:
        return self._record.downgrade_reason

    def run(self) -> SubagentRun:
        """The admission facts as the handle a spawn returned — for a caller that
        answers from the log what the live path answers from the handle: a
        delegating tool's `reconcile` builds the value its `execute` would have."""
        record = self._record
        return SubagentRun(
            id=record.run_id,
            name=record.name,
            session_id=self.session_id,
            parent_id=self.parent_id,
            model_provider=record.model_provider,
            model=record.model,
            requested_access=record.requested_access,
            granted_access=record.granted_access,
            owner=record.owner,
            downgrade_reason=record.downgrade_reason,
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """The child as a row: its admission, where it is, and how it stands — for
        the model's roster tool and a front end's panel."""
        admission = thaw_json(self.admission.to_wire()) if self.admission is not None else {}
        assert isinstance(admission, dict)
        row: dict[str, JsonValue] = {
            **admission,
            "sessionId": self.session_id,
            "parentId": self.parent_id,
            "status": self.status,
            "starts": self.starts,
            "resumes": self.resumes,
            "tokens": self.tokens,
        }
        optional: dict[str, JsonValue | None] = {
            "cause": self.cause,
            "detail": self.detail,
            "answerPreview": self.answer_preview,
            "deletedReason": self.deleted_reason,
            "awaiting": self.awaiting,
        }
        row.update({key: value for key, value in optional.items() if value is not None})
        if self.deleted:
            row["deleted"] = True
        return row


def fold_child_event(state: ChildState, event: SessionEvent) -> ChildState:
    """One event of a child's own log folded into its state. The rules, in one place.

    **Each field is set by its own record, and replaced whole.** A status carries its
    `cause` and `detail` or it carries none, so a later status never keeps an
    earlier one's reason: the roster this replaces merged each status into its row,
    and a child woken after a restart went on saying `resumed`. A status this fold
    does not know leaves the one before it standing — a child is never read as
    settled by a record nobody can read, which is `child_is_live`'s direction.

    An `assistant/message` is an answer: it forgives the restarts before it, and it
    and a compaction are what the child spent.
    """
    kind = event.type
    # The type test first: a child's log is mostly `assistant/chunk`.
    if kind not in CHILD_EVENT_TYPES:
        return state
    data = event.data
    if kind == ADMITTED:
        try:
            admission = Admission.model_validate(data)
        except ValidationError:
            return state
        return replace(state, admission=admission, admitted_at=float(event.time))
    if kind == STATUS:
        status = _STATUSES.get(as_str(data.get("status")))
        if status is None:
            return state
        cause = _CAUSES.get(as_str(data.get("cause")))
        return replace(
            state,
            status=status,
            cause=cause,
            detail=as_str(data.get("detail")) or None,
            answer_preview=as_str(data.get("answerPreview")) or None,
            notice=_notice_of(data.get("notice")),
            starts=state.starts + (status == "running"),
            resumes=state.resumes + (cause == "resumed"),
        )
    if kind == DELETED:
        return replace(state, deleted=True, deleted_reason=as_str(data.get("reason")) or None)
    if kind == _ANSWER:
        return replace(
            state, resumes_at_last_answer=state.resumes, tokens=state.tokens + event_tokens(event)
        )
    if kind in SPENDING_TYPES:
        return replace(state, tokens=state.tokens + event_tokens(event))
    holder, name = hold_of(event)
    if holder != SESSION_HOLDER:
        return state
    if kind == CREDENTIAL_WAIT.opened:
        return replace(state, awaiting=name)
    return replace(state, awaiting=None) if state.awaiting == name else state


def _notice_of(recorded: JsonValue) -> ChildNotice | None:
    """An ending's notice, or `None` when it carries none or one this build cannot read."""
    if recorded is None:
        return None
    try:
        return ChildNotice.model_validate(thaw_json(recorded))
    except ValidationError:
        return None


def child_state(log: Session) -> ChildState:
    """A child's state, folded from its own log. See `ChildState`."""
    return child_state_of(log.id, log.header, log.events)


def child_state_of(
    session_id: str, header: SessionHeader | None, events: Iterable[SessionEvent]
) -> ChildState:
    """`child_state` over a header and events rather than a session: a log read from
    the store, which a resume holds before any `Session` is built for it."""
    parent = header.delegating_parent if header is not None else None
    state = ChildState(session_id=session_id, parent_id=parent or "")
    for event in events:
        state = fold_child_event(state, event)
    return state


def extend_child_state(previous: ChildState, log: Session, start: int) -> ChildState:
    """`child_state` continued from a prefix — what `SubagentService`'s cache folds a
    new slice with. Extending the fold of a prefix equals folding the whole log."""
    state = previous
    for event in log.events_from(start):
        state = fold_child_event(state, event)
    return state


def _stored_tree(
    store: SessionPersistence, parent_id: str, family: str, live: set[str]
) -> dict[str, dict[str, ChildState]]:
    """Everything beneath `parent_id` on disk that this process is not running, each
    folded from its stored log and filed under the parent its header names — with an
    entry, empty or not, for every session in the tree, since each one's children
    were in this read. On a worker thread (`load_children`)."""
    tree: dict[str, dict[str, ChildState]] = {}
    for row in store.descendants_of(parent_id, family):
        tree.setdefault(row.session_id, {})
        if row.session_id in live:
            continue
        # A child's log starts at seq 0, so its own file is the whole of it — read by
        # its family, which the listing already found, rather than searched for, and
        # only for what its state is folded from.
        header, events = store.read_own(row.session_id, family=row.family, types=CHILD_EVENT_TYPES)
        state = child_state_of(row.session_id, header, events)
        if state.admitted:
            tree.setdefault(state.parent_id, {})[row.session_id] = state
    return tree


def _names_parent(child: Session, parent_id: str) -> bool:
    """Whether a child's log names `parent_id` the way the store finds it by.

    The header's parent link decides (`delegating_parent`); the id's `<parent>-`
    prefix is how `SessionArchive.descendants_of` narrows the family directory before it
    reads a header, so a child without it is one the listing never reaches.
    """
    return child.header.delegating_parent == parent_id and is_child_id(parent_id, child.id)


def restarts_since_progress(state: ChildState) -> int:
    """How many times a child has been restarted since it last answered (P3).

    **The ladder's rule, stated where the ladder is decided** rather than folded
    into the state: a child stopped, working an hour, then stopped again met two
    incidents, not a lifetime's, so an answer forgives the restarts before it. An
    `assistant/message` in its own log is the answer — a model reply, which a turn
    that did nothing cannot produce. The state keeps both facts as totals
    (`resumes`, `resumes_at_last_answer`), so a different rule is a change here and
    not to the fold.
    """
    return state.resumes - state.resumes_at_last_answer


FamilyRole: TypeAlias = Literal["self", "parent", "sibling", "child"]
"""How one agent stands relative to another, within reach."""


class _Parented(Protocol):
    """A session's id and its parent link, and nothing else.

    All `reachable_family` reads — its docstring already says sessions are
    "passed in rather than read from a store, so this answers on a resumed log
    with no live agents as readily as in a running process", which is a
    structural contract described in prose. Written as one, a caller can hand it
    a family tree it built for the purpose: the messaging test does exactly
    that, and its two-field double satisfied this all along while failing the
    declared `Session` (issue 32).
    """

    @property
    def id(self) -> str: ...

    @property
    def header(self) -> _HasParent: ...


class _HasParent(Protocol):
    @property
    def delegating_parent(self) -> str | None: ...


def reachable_family(sessions: Iterable[_Parented], agent_id: str) -> dict[str, FamilyRole]:
    """Every agent `agent_id` may address, mapped to how it is related (C7).

    The enumeration half of `family_reach`, and derived *from* it rather than
    restating the rule — the guard that refuses a send and the roster that tells
    the model who it may address must not be able to disagree.

    An agent's id is its session's id, so the parent link is
    `SessionHeader.delegating_parent` and nothing needs a side index — the agent
    that spawned it, which a fork's `parent_session` is not: a forked root is a
    root. Sessions are passed in rather than read from a store, so this answers on
    a resumed log with no live agents as readily as in a running process.
    """
    parents = {session.id: session.header.delegating_parent for session in sessions}
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


MAX_NAME_CHARS = 64
"""The longest name a spawn may give its child."""


def default_child_name(prompt: str, tag: str, *, taken: Collection[str] = ()) -> str:
    """`subagent-<prompt-slug>-<tag>`, unique among `taken`. `tag` tells apart two
    children given the same task; the seam passes eight random hex digits.

    A name the parent can read back matters more than it looks: the roster, the
    terminal notice and every `agent_message` address use it, and a child called
    `child-4` tells the parent nothing about which delegation it was.
    """
    words = [
        word for word in "".join(c if c.isalnum() else " " for c in prompt).split()[:4] if word
    ]
    slug = "-".join(words).lower()[:32] or "task"
    candidate = f"subagent-{slug}-{tag}"
    if candidate not in taken:
        return candidate
    for suffix in range(2, 100):
        alternative = f"{candidate}-{suffix}"
        if alternative not in taken:
            return alternative
    return f"{candidate}-{len(taken)}"
