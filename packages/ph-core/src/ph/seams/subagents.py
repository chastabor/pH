"""`ctx.subagents`: delegation to a child agent, and the handle that it returns.

This module is the seam definition only. A row that a profile mounts registers
the provider: `ph-rlm` registers `rlm-child`. Three rules of the contract are
important enough to state here and not in the docstring of a provider.

**The handle returns before the child answers.** `start()` returns when the
child is *admitted*: its session is created, its admission is logged, and its
task is detached. It does not wait for the child to finish. An RLM parent
depends on this fan-out, which does not block. The parent spawns eight children
and keeps working, and their replies arrive as ordinary inbox messages on later
turns. If `start()` awaited completion, the control loop of the parent would be
serial, and the parent would have to poll.

**Completion is available, but separate.** `SubagentRun.result()` awaits
quiescence for a caller that does want to block. One such caller is a generic
`task` tool that returns the last text of the child. Both kinds of caller use
one provider. For that reason the answer is reachable, but `start()` never
returns it.

**`access` defaults to read (E4).** A child asks for the workspace guarantee
that it needs. The default is the conservative one, so a delegation that never
mentions `access` cannot silently receive a writable repo. The provider resolves
the request against the tier that is actually available, and reports what it
granted. A child that is told nothing about its workspace attempts writes and
reads the failures as bugs.

@module ph.seams.subagents
"""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, assert_never, get_args, runtime_checkable

import anyio
from pydantic import Field, ValidationError

from ..agent.types import AgentDriver, AgentHandle, AgentOptions
from ..cancel import reason_of
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
from ..json import JsonObject, JsonValue, as_obj, as_seq, as_str, thaw_json
from ..keys import (
    AGENTS,
    GOALS,
    LLM,
    MODELS,
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
from ..llm.media import untaken
from ..llm.structured import SchemaViolation, structural_warning, validated_shape
from ..llm.types import (
    DEFAULT_EFFORT,
    Message,
    PluginSource,
    create_user_message,
    new_message_id,
    text_of,
    unmapped_effort,
)
from ..locks import KeyedLocks
from ..persistence.jsonl import LogDamaged
from ..persistence.opening import open_session, stored_session
from ..persistence.protocol import SessionArchive
from ..session import (
    Session,
    SessionBatch,
    SessionEvent,
    SessionFoldCache,
    SessionHeader,
    child_session_id,
    derive_event_message,
    is_child_id,
    session_written,
)
from ..session.kinds import CREDENTIAL_WAIT, SESSION_HOLDER, hold_of
from ..session.writers import log_writer
from ..system_prompt.assembly import PromptSection
from ..tools.definition import (
    Deny,
    ToolDefinition,
    ToolRunContext,
    call_id_of,
    define_tool,
)
from ..tools.errors import (
    SPAWN_REFUSED,
    TOOL_BUDGET_SPENT,
    TOOL_DENIED,
    FailureKind,
    HarnessError,
)
from ..tools.json_schema import checked_object_schema, validate_json_schema_value
from ..tools.registry import ToolRestriction
from ..wire import WireForm, WireModel, literal_lookup
from ._registry import claim_entry, claim_key
from .credentials import hold_for_credential, missing_credential
from .invariants import contribute_fold_cache
from .models import (
    ModelChoice,
    ModelEntry,
    ModelList,
    ModelRoute,
    UnrunnableChoice,
    UnrunnableReason,
    choose,
)
from .skills import (
    ORDER_SKILLS,
    Skill,
    SkillRestriction,
    SkillService,
    offer_skills,
    record_read,
)
from .subagent_profiles import narrowing
from .token_meter import SPENDING_TYPES, event_tokens

_LOG = log_writer(__name__)

__all__ = [
    "ADMITTED",
    "CHILD_EVENT_TYPES",
    "DELETED",
    "INTERRUPTED_DETAIL",
    "MAX_NAME_CHARS",
    "PARENT_TEARDOWN",
    "RESULT_TOOL",
    "RETURNED",
    "REVOKING_AT_ONCE",
    "SETTLED_STATUSES",
    "SHIPPED_SKILLS",
    "STATUS",
    "SUSPENDED_DETAIL",
    "UNRECOVERABLE_DETAIL",
    "Access",
    "Admission",
    "ChildCounts",
    "ChildNotice",
    "ChildState",
    "EffortUnsent",
    "FallbackReason",
    "FamilyRole",
    "MediaShortfall",
    "ModelFallback",
    "ReadmittingProvider",
    "RehydratableProvider",
    "Returned",
    "ReturnedVia",
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
    "admission_notes",
    "admission_payload",
    "admitted_by",
    "apply",
    "child_is_live",
    "child_options",
    "child_state",
    "child_state_of",
    "default_child_name",
    "ended_text",
    "exhausted_detail",
    "extend_child_state",
    "family_reach",
    "fold_child_event",
    "last_answer",
    "offer_shipped_skills",
    "open_child_log",
    "parent_went_away",
    "reachable_family",
    "record_admitted",
    "record_deleted",
    "record_ended",
    "record_returned",
    "record_started",
    "record_waiting",
    "restarts_since_progress",
    "said_text",
    "settle_value",
    "stored_tree",
]

log = logging.getLogger("ph.seams.subagents")

ORDER_BRIEF = ORDER_SKILLS + 10
"""The brief comes after the skills catalog, because the brief is the *assignment*
and the catalog is the menu. The assignment is what the reader meets last, so
no list of other things follows it."""

ORDER_OUTPUT = ORDER_BRIEF + 1
"""Right after the brief: what to hand back is the end of the assignment."""

RESULT_TOOL = "submit_result"
"""The tool that a child with an `output` returns its value through. It is
registered on the scope of that child, so no other agent has it
(`SubagentService._give_output`)."""

ADMITTED = "subagent/admitted"
DELETED = "subagent/deleted"
STATUS = "subagent/status"
RETURNED = "subagent/returned"
"""The records that a delegation leaves. They are in the log of the **child
itself** (Phase 11). `returned` is the value of a child that has an `output`
(`record_returned`).

Each session owns its log. A child records in its own log what it was asked,
each start, each wait, its ending and its deletion. Nothing about the child is
written to the log of its parent. A parent, the resume sweep, budgets, caps and
front ends read the state of a child from the log of the child (`child_state`).
Before this, the parent kept a copy, which was a second account of the child.
Every durability rule existed to keep the two accounts in step: the flush of the
admission (S2), the flush of a restart (S10), the log of the child before the
`done` of the parent (F1), and the usage catch-up after a crash (L5).

The names are defined here and not spelled at each `append` site, because the
fold below and every producer must agree on them exactly."""

_ANSWER = "assistant/message"
CHILD_EVENT_TYPES = frozenset(
    {
        ADMITTED,
        DELETED,
        STATUS,
        RETURNED,
        *SPENDING_TYPES,
        CREDENTIAL_WAIT.opened,
        CREDENTIAL_WAIT.settled,
    }
)
"""The record types that the state of a child is folded from (`fold_child_event`):

- The four records above.
- The records that spend tokens (`SPENDING_TYPES`). An answer is one of them, and
  an answer is also what forgives the ladder.
- The two records of a credential hold.

Any other record leaves the state of a child as it was. So a reader of a stored
log parses only these types, and a watcher skips the rest."""

Access: TypeAlias = Literal["read", "write"]
"""What a child asks of the parent's workspace. `read` is the default (E4)."""

SubagentStatus: TypeAlias = Literal["queued", "running", "done", "error", "canceled"]
"""The lifecycle of a child, as its own log records it and as its parent and the
TUI panel read it.

This is the lifecycle only. *Why* a child is live is a separate `cause` on the
same record: for example, the child was woken to answer a question and is not
still on its first task. The reason is the fold. The state of a child takes the
last status that was written. With a `rehydrated` member here, a woken child
that is actively working would read as not running to every consumer that
branches on `"running"`."""

UNRECOVERABLE_DETAIL = (
    "the harness stopped while this child was running, and no provider here can "
    "start it again; its transcript is on disk"
)
"""The child was interrupted, and nothing that is mounted could resume it.

This case has its own sentence because the parent needs a different answer. The
ladder was not spent, and a deployment that mounted the provider again could
have carried on. So "it ran out of attempts" would be false. The child is
settled all the same. A child that is left `queued` for a provider that will
never come holds the root out of passivation permanently."""

INTERRUPTED_DETAIL = "the harness stopped while this child was running; it did not finish"
"""Why a child that was in the middle of a turn at shutdown did not finish.

It is a sentence and not a code, because its reader is a person or a model that
looks at the state of a child and asks what happened to a child that never
answered."""


PARENT_TEARDOWN = "parent-teardown"
"""Why a child was revoked when its parent ended or went away (I2). Either the scope
of its parent unwound under it, or a restart found that its parent had ended and
left it unfinished. Both cases have one reason, because they are one event. In
the second case a crash cut the first short, and the next process completed it."""


REVOKING_AT_ONCE = 8
"""How many logs of descendants `_revoke_beneath` opens at one time.

The number is bounded for two reasons. Each open log is held whole in memory
until its tombstone is written. Each also takes a worker thread to read it,
sweep it and flush it, and that thread comes from the limiter that every root in
the process shares. So an unbounded walk of a family would queue the file reads
and flushes of other roots behind it. Eight overlaps the waits that are worth
overlapping (a `git` reconcile, an `fsync`) without either cost."""


SUSPENDED_DETAIL = (
    "the harness stopped while this child was working; it is started again when its parent is"
)
"""Why a child is `queued` after its whole mount unwound under it (S1).

A provider writes it in place of a tombstone. A daemon that stops, or a root
that is remounted on a new profile, is not a revocation. So the child stays
live, and the resume sweep readmits it. The status is `queued` and is not left
as `running`, because nothing crashed: the sweep readmits a `queued` child
without asking the ladder whether it may. The next start of the child is still a
restart (`cause: resumed`), because it was one. So a child that then crashes
before it has answered carries that start onto the ladder."""


def exhausted_detail(limit: int) -> str:
    """The sentence for a spent ladder. It names the bound that is **actually in
    force**.

    It is built from `INTERRUPTED_DETAIL` and does not restate it, so the two
    cannot come to describe one interruption differently.
    """
    return (
        f"{INTERRUPTED_DETAIL} — and it has now been interrupted "
        f"{limit} times, so it will not be started again"
    )


SettledStatus: TypeAlias = Literal["done", "error", "canceled"]
"""A status that means a child has stopped."""

SETTLED_STATUSES: frozenset[str] = frozenset(get_args(SettledStatus))
"""The statuses that mean a child has stopped. The set is defined beside the
vocabulary that it reads.

It is here and not in the consumer for the same reason that the four event names
are here: the fold and every producer must agree exactly. The sweeper of P5-05
wrote its own copy, `{"completed", "failed", "canceled", "deleted"}`, and three
of its four members were wrong. No producer emits `completed` or `failed` (the
writer says `done` and `error`), and `deleted` is not a status at all. The two
names that do mean settled were missing. The effect was that a root that had
ever run a child to completion could never be released. Every settled child read
as live permanently, and releasing such a root is most of what passivation
exists to do.
"""


def child_options(request: SubagentRequest) -> AgentOptions:
    """The options that the agent of a child runs with.

    **A child on a resolved route runs with the settings of that route**, as a
    root that is started on the route does (`ModelEntry.options`). It takes none
    of the settings of its parent: a temperature that was tuned for the model of
    the parent is not tuned for this one. A child on the route of its parent runs
    with the options of its parent. In both cases the reasoning effort is the
    one that the spawn asked for, else the one that its skills name, else the one
    of its route. `default` in any of them asks for the effort of the provider
    itself. The rule is stated here one time, for every provider that builds a
    child.
    """
    resolved = request.entry.options() if request.entry is not None else request.parent.options
    effort = request.reasoning_effort or request.skill_effort or resolved.reasoning_effort
    return replace(resolved, reasoning_effort=effort)


class _Standing(Protocol):
    """What `child_is_live` reads: a child's state, or a front end's row of it."""

    @property
    def deleted(self) -> bool: ...

    @property
    def status(self) -> SubagentStatus: ...


def child_is_live(state: _Standing) -> bool:
    """Whether this child is still working.

    A deletion is a tombstone and not a status: `fold_child_event` sets `deleted`
    and leaves `status` unchanged. So both must be read. That is the other half
    that a hand-written copy got wrong.

    A status that the fold does not recognize leaves the previous status in place
    (`fold_child_event`), and a child that was admitted and has no status yet
    reads `queued`. So the direction of failure is kept: a caller that releases a
    parent because of this answer must fail toward keeping the parent alive. The
    wrong direction abandons a running child. This direction costs memory until
    the child settles.
    """
    return not state.deleted and state.status not in SETTLED_STATUSES


StatusCause: TypeAlias = Literal["rehydrated", "resumed"]
"""Why a child entered its current status, when the reason is not simply "it
started".

`resumed` marks a child that was started again after its harness stopped in the
middle of a turn. The ladder counts these starts (`restarts_since_progress`),
and the cause lets a reader of the log tell that run from a first attempt."""

DowngradeReason: TypeAlias = Literal["workspace-not-mounted"]
"""Why a granted access is narrower than the one requested."""

_DOWNGRADE_TEXT: dict[str, str] = {
    "workspace-not-mounted": (
        "granted read rather than write: no workspace tier is mounted to enforce a "
        "writable repo, so one cannot be promised"
    )
}


def _downgrade_text(reason: DowngradeReason | str) -> str:
    """The sentence for one downgrade code, for `admission_notes`. There is one copy,
    so there is one place to edit when the wording changes."""
    return _DOWNGRADE_TEXT.get(reason, f"access was narrowed ({reason})")


FallbackReason: TypeAlias = UnrunnableReason | Literal["contested", "cannot-accept"]
"""Why a child runs on the route of its parent and not on the model that it was
asked onto:

- `unlisted`: the list does not hold the key.
- `unserved`: no adapter here serves its provider.
- `contested`: the skills that it was given name different models.
- `cannot-accept`: its route cannot take the media that those skills hand it."""


class ModelFallback(WireModel):
    """A model that a child was asked onto and that this mount cannot run, so the
    child runs on the route of its parent (`SubagentService.resolve_model`). At a
    spawn it is a model that a skill names. At a readmission it can be any model.

    It holds what was asked and a code, not a sentence, for the same reasons as
    `downgrade_reason`: the admission is durable, and the profile that a sentence
    would describe changes after the admission. `admission_notes` renders the
    sentence."""

    asked: tuple[str, ...] = Field(min_length=1)
    """The model that was asked for: a key, or `provider/model` for a whole route.
    For skills that disagree, it holds each key that they name."""
    reason: FallbackReason
    media: tuple[str, ...] | None = None
    """For `cannot-accept`, the media that its skills hand it and that the route
    cannot take. Absent otherwise."""


class MediaShortfall(WireModel):
    """Media that the skills of a child hand it and that neither its route nor the
    route of its parent can take (A3). The child runs and sees a text placeholder
    for each one, and its parent is told. This is a record and not a refusal, for
    the reason behind graceful degradation: the child can still do what words can
    do, and the user learns what to set up."""

    media: tuple[str, ...] = Field(min_length=1)
    route: str
    """The route it runs on, `provider/model`."""


def _shortfall_text(shortfall: MediaShortfall) -> str:
    """The sentence for media a child cannot see, for `admission_notes`."""
    media = ", ".join(shortfall.media)
    return (
        f"runs on {shortfall.route}, which cannot take {media}, which its skills hand it, "
        "and neither can yours: it sees a placeholder for each, so ask of it only what "
        "it can answer without seeing them, until the person running you sets up a "
        "model that takes them"
    )


class EffortUnsent(WireModel):
    """A reasoning effort that a child runs without, because the profile of its
    route does not say how to send it (`ModelRoute.efforts`). The request of the
    child itself withholds the effort by the same rule (`unmapped_effort`). So the
    child thinks at the default of its provider, and its parent is told. This is
    a record and not a refusal, as `MediaShortfall` is: the child runs on the
    model that it was asked onto, and only how hard it thinks differs."""

    effort: str
    route: str
    """The route it runs on, `provider/model`."""
    defined: tuple[str, ...] = ()
    """The efforts that route's profile defines, which a parent can name instead."""


def _unsent_text(unsent: EffortUnsent) -> str:
    """The sentence for an effort a child runs without, for `admission_notes`."""
    named = ", ".join(unsent.defined)
    can = f"{named} or {DEFAULT_EFFORT}" if named else f"only {DEFAULT_EFFORT}"
    return (
        f"thinks at its provider's default rather than {unsent.effort!r}, which the "
        f"profile does not say how to send to {unsent.route}; it can run at {can} there"
    )


def _fallback_text(fallback: ModelFallback) -> str:
    """The sentence for a model fallback, for `admission_notes`."""
    asked = ", ".join(f'"{model}"' for model in fallback.asked)
    match fallback.reason:
        case "unlisted":
            why = f"rather than {asked}, which this profile does not list"
        case "unserved":
            why = (
                f"rather than {asked}, which this profile lists on a provider no adapter "
                "here serves"
            )
        case "contested":
            why = f"because the skills it was given name different models ({asked})"
        case "cannot-accept":
            media = ", ".join(fallback.media or ())
            why = f"rather than {asked}, which cannot take {media}, which its skills hand it"
        case unknown:
            assert_never(unknown)
    return (
        f"runs on your own model {why}; it may do the work less well until the person "
        "running you sets up the model it should run on"
    )


def ended_text(name: str, result: SubagentResult) -> str:
    """Why a child that its caller waited on gave the caller nothing to use: how the
    child ended, why, and its last text, which may still hold what the caller
    needs. Every tool that waits uses this one wording."""
    why = f": {result.error}" if result.error else ""
    said = f"\n\nIts last text:\n{result.answer}" if result.answer else ""
    return f"{name} ended as {result.status}{why}{said}"


def admission_notes(run: SubagentRun) -> list[str]:
    """What a parent is told about how its child was admitted, in addition to the
    facts: why the access of the child is narrower than the request, why the child
    runs on another model than its skill asked for, what the child cannot see, and
    an effort that it runs without. There is one list, so the delegating tools
    cannot tell a parent different things."""
    notes: list[str] = []
    if run.downgrade_reason is not None:
        notes.append(_downgrade_text(run.downgrade_reason))
    if run.model_fallback is not None:
        notes.append(_fallback_text(run.model_fallback))
    if run.media_shortfall is not None:
        notes.append(_shortfall_text(run.media_shortfall))
    if run.effort_unsent is not None:
        notes.append(_unsent_text(run.effort_unsent))
    return notes


def _refused(reason: Deny | str) -> SubagentSpawnError:
    """The answer of one guard, as the error that the pipeline already knows how to
    read.

    **The code is derived from the reading, exactly as `registry._gate` derives
    it** (D15). The tool side picks `budget_result` or `denied_result` from
    `Deny.failure_kind`, and its decision carries no code at all. When the
    decision carried a code here, the same code arrived with two readings.
    `SPAWN_REFUSED` meant a misconfiguration that a retry can fix when it came
    from a direct raise, and a policy denial when it came from the ceiling. That
    is the one distinction that the docstring of `TOOL_DENIED` exists to keep.

    The answer is a `Deny` and not a second refusal type for the same reason. The
    child ceiling and the tool ceiling are the same decision, and the row that
    governs both now writes it one time. A bare string is still the answer of a
    guard. It means the plain refusal that every guard except the ceiling wants.
    `ToolRuntime.guard` has no posture for that reason, and this keeps the guards
    of the two seams the same shape.
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

    This is distinct from a child that ran and failed. A refusal produced no
    session, no log and no artifacts, so a caller may retry it with different
    arguments.

    **It is a `HarnessError`, so the refusal keeps its shape on the way out**
    (D15). It was a bare `Exception` before. It then carried no code and no
    `failure_kind`, and it arrived at the model as a generic failure while the
    turn carried on. That is the posture that D7 replaced for tool calls. It
    survived here because nothing had given a spawn refusal a way to say more.
    The two tool bodies that caught it also flattened it into a `ValueError` or a
    `ToolCallError`, so even a richer error would not have reached the pipeline.
    They now let it through.

    The defaults are exactly the old behavior: `SPAWN_REFUSED`, a failure that
    the model may act on, and a turn that continues. Most refusals here are a
    misconfiguration that a retry *can* fix: no such provider, no such preset, a
    grant that the parent does not hold. Only a spent ceiling says otherwise, and
    it says so by passing other values.
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

    `parent` is the handle of the agent that delegates, not an id. The seam and
    the provider read its session, its scope and its options, and a lookup of
    those from an id would let a caller name an agent that it does not hold.
    """

    prompt: str
    parent: AgentHandle
    """The agent that delegates. **Required for a spawn, and tolerated as `None` by
    `held_by` alone.**

    `_delegating_boundary` has an explicit `if request.parent is None` branch: a
    grant that is computed with no parent inherits the ceiling of the mount. But
    every provider that goes on to *spawn* reads the session, the scope and the
    options of the parent. When this field was declared optional, eight reads in
    `ph_rlm.subagents` became type errors for a case that they never receive.
    The declaration is the contract of a spawn. The one caller that asks only for
    a grant passes `None` past the declaration on purpose.

    The type is the handle and not the driver, because a provider reads this
    parent and never steers it. The one notice that does reach the inbox of a
    parent (`_deliver_notice`) goes to a driver that is looked up by session id,
    because a parent that was disposed while its child ran has no inbox to
    deliver to. So the wider type had no benefit, and it let a provider reach
    `cancel` on the agent that asked it for a child.
    """
    scope: Context | None = None
    """The boundary that this delegation is made **from** (P6-31).

    It is not the scope of the child. That scope does not exist yet when the
    ceiling is computed. When it exists, it is `SubagentRun.scope`, and `_enforce`
    checks that it is inside this one.

    This is the same value and the same argument as `ToolExecutionInput.scope`:
    the caller states the boundary, and the seam does not guess it. It is optional
    only because callers and providers build a `SubagentRequest`, and not only
    this seam. `_delegating_boundary` resolves it.
    """
    name: str | None = None
    """A stable label among its siblings, which is how they are addressed. When it
    is omitted, the name is made from the task. It is unique in both cases,
    because the seam names every child before it asks the provider
    (`SubagentService.start`)."""
    choice: ModelChoice = field(default_factory=ModelChoice)
    """The model that the spawn names (session profiles, S7b). It is a key that the
    profile of the parent lists, or a whole route. The key is the one that the
    spawn gives or the one that its assigned profile gives. A choice of neither
    leaves the model to the skills that the child is given, and else to the route
    of its parent. The admission records the choice, and a readmission resolves
    it again (`SubagentService.resolve_model`)."""
    reasoning_effort: str | None = None
    """An effort that the spawn names. It takes precedence over the effort of its
    route (`child_options`)."""
    entry: ModelEntry | None = None
    """What `resolve_model` resolved the spawn to: the listed entry, or a whole
    route that no entry matches, under no key. A child on it runs with its
    settings (`child_options`). `None` runs the child on the route of its parent,
    with the options of its parent."""
    model_fallback: ModelFallback | None = None
    """The model that the child was asked onto and that this mount could not run, so
    the child runs on the route of its parent (`resolve_model`)."""
    media_shortfall: MediaShortfall | None = None
    """Media that its skills hand it and that neither its route nor the route of its
    parent can take (`resolve_model`)."""
    effort_unsent: EffortUnsent | None = None
    """The effort that the child will run at, when the profile of its route does not
    say how to send it (`resolve_model`)."""
    skill_effort: str | None = None
    """The reasoning effort that the skills it is given name (`Skill.effort`), when
    they agree (`resolve_model`). It takes precedence over the effort of its
    route, and the effort of the spawn takes precedence over it
    (`child_options`)."""
    output: dict[str, Any] | None = None
    """The JSON Schema object that the value of the child must match. It is the
    schema of the spawn, or else the one that the skills it is given declare
    (`resolve_output`). A child with an output is given `submit_result` and owes a
    value. `None` asks only for an answer."""
    access: Access = "read"
    preset: str | None = None
    """A named kind of child that the deployment configured (`subagent-presets`).

    It is resolved into the fields below before the ceiling is checked. So a
    preset is a set of defaults and never a way past the ceiling.
    """
    profile: str | None = None
    """A named profile that the parent assigns to its child (session profiles, S7b).
    It is read as a narrowing on the mount of the parent
    (`ph.seams.subagent_profiles`): the tools and skills of what it runs, its
    default model, and a read-only sandbox. It is resolved into the fields around
    it before the ceiling is checked. A profile that would widen anything is
    refused, and the refusal names the row."""
    skills: tuple[str, ...] | None = None
    """Which skills the child gets. `None` inherits the whole set of the parent.

    The set is **always a subset of the set of the parent**. A spawn that names a
    skill that the parent does not hold is refused and not granted. A spawn that
    could widen would make delegation the privilege escalation that I7 exists to
    prevent (P4-13b). To give a child a skill, install it, which gives it to the
    parent too.

    `()` is a real answer and is not the same as `None`: it is a child that should
    read no skill at all. Naming a skill is also *direction*. The bodies of the
    named skills are put in the prompt of the child, because a child that is
    spawned to follow a procedure should not have to spend a turn fetching it.
    """
    tools: tuple[str, ...] | None = None
    """Which tools the child gets. `None` inherits the whole set of the parent.

    The ceiling and the reason are the same as for `skills`. The set is applied as
    a `ToolRestriction`, which can only subtract. The Code Mode transport stays
    reachable in every case, because by construction it cannot be restricted. A
    child with no way to call anything is not a narrower child. It is a broken
    one.
    """
    paths: tuple[str, ...] | None = None
    """The extra directories that the sandbox of the child binds writable, as
    canonical paths. `None` binds the directories of the parent. The
    `sandbox-allow` row of an assigned profile sets it (`resolve_profile`). It is
    applied as a limit (`SandboxSeam.restrict_paths`), which can only subtract.
    """
    call_id: str | None = None
    """The tool call that asked for this child: a `task` call, or the dispatch of an
    `rlm.run` inside a cell (S2).

    It is recorded on the admission, so that the call can be answered from the
    log of the child after a restart (`admitted_by`). The admission is on disk
    before the child runs. So a call with no admission started no child, and a
    call with an admission started this one. Without the id, a delegating call
    that a crash cut short could only be answered "unknown". The model then
    delegated the same task again, beside the child that the resume had already
    put back to work.
    """


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """What a finished child produced, for a caller that waited.

    It is a plain dataclass and not a `WireModel`, because it never crosses a JSON
    boundary. The durable part of the outcome of a child is the `subagent/status`
    record. This is the in-process answer that is handed to the caller that
    awaited `result()`.
    """

    status: SubagentStatus
    answer: str = ""
    """The child's last non-empty assistant text. Empty when it never spoke."""
    error: str | None = None
    value: JsonObject | None = None
    """What a child with an `output` returned, checked against it (`settle_value`)."""


SubagentAwaiter: TypeAlias = Callable[[], Awaitable[SubagentResult]]
"""How a caller waits for one delegation to settle.

The type has a name because a provider *builds* one and `SubagentRun.result`
*holds* one, in different packages. `ph_rlm.subagents._awaiter` spelled the type
out by hand, so nothing linked the two copies.
"""


@dataclass(slots=True)
class SubagentRun(WireForm):
    """A live delegation. It is returned at admission, before the child answers.

    It is a `WireForm` and not a `WireDataclass`: it serializes itself, but it
    cannot take the body that is derived from the aliases. That body emits every
    field that is not `None`, and some of these fields do not travel. They are
    `owner` (which defaults to `""`, so it would always be emitted), and `output`,
    `result`, `dispose`, `grant`, `scope` and `ready`, which include a `Grant`, a
    `Context` and an event. Only `id` → `runId` is a spelling that the alias
    function would miss.

    The fields are the admission facts that a parent can act on immediately: what
    to call the child, where its log is, and which guarantees it actually got.
    `granted_access` may differ from the request when the available tier cannot
    honor the request. It is the value that the admission and the prompt of the
    child report.
    """

    id: str
    name: str
    session_id: str
    parent_id: str
    model_provider: str
    """Which LLM provider the child runs on. The name says what it is, because the
    *subagent* provider is a different thing on the same handle. When one field
    named `provider` served both, `rehydrate` looked up the wrong one."""
    model: str
    requested_access: Access
    granted_access: Access
    owner: str = ""
    """Which `ctx.subagents` provider owns this run. The service stamps it.

    It is not named `provider`. The *LLM* provider is `model_provider` on this
    handle, and when one word served both, `rehydrate` looked up the wrong one.
    It is not in `to_wire`, which is the handle that a caller is given. The
    service writes it onto the admission (`record_admitted`), because a
    readmission after a restart must find the provider that owns the child. At
    one time it was missing from the log, and a readmit then worked only while
    exactly one provider was mounted."""
    downgrade_reason: DowngradeReason | None = None
    """Why `granted_access` is narrower than `requested_access`, as a code and not
    as prose.

    A consumer that must branch on a durable event cannot parse an English
    sentence in it. The sentence also goes stale silently: the reason that a
    `write` was refused stops being true when a workspace tier is mounted, in
    every log that is already written. `admission_notes()` renders the sentence
    from the code, in one place."""
    model_fallback: ModelFallback | None = None
    """The model that this child was asked onto and runs without. The seam stamps it
    from the resolved request, at a spawn and at a readmission."""
    media_shortfall: MediaShortfall | None = None
    """Media that the skills of this child hand it and that it cannot see. It is
    stamped as `model_fallback` is."""
    effort_unsent: EffortUnsent | None = None
    """The effort this child runs without, stamped as `model_fallback` is."""
    output: dict[str, Any] | None = None
    """The shape that this child owes its value in. The seam stamps it from the
    request, for the drive of the provider to settle (`settle_value`). It is not
    in `to_wire`, because a parent holds the schema that it asked for."""
    result: SubagentAwaiter | None = None
    """Awaits quiescence and reports the outcome. `None` from a provider whose
    children cannot be waited on."""
    dispose: Disposer | None = None
    """Stops and lets go of a child that the seam would not admit, **without
    revoking it**. It writes no tombstone, because a refusal is not a deletion. It
    writes nothing at all, so its caller ends the child first. The seam does so
    (`SubagentService._abandon`). The `task` tool revokes a child whose wait was
    canceled (`SubagentService.delete`) before it calls this. The teardown of a
    parent, and a delete, reach the children of a provider by other paths, and
    those paths revoke."""
    grant: Grant | None = None
    """What this child was bounded to. The seam stamps it when it applies it.

    It is on the run because the run is what a provider keeps. A rehydration
    builds a *new* scope for a settled child, and the filters that bounded the old
    scope went with that scope. So to apply the ceiling again, the grant must
    have outlived the scope, and the parent that it was computed from."""
    scope: Context | None = None
    """The scope of the child itself. The provider sets it so that the **seam** can
    bound it.

    Before this field, the ceiling was a documented obligation on providers, and
    the second call site had already missed it: `rehydrate` builds a fresh scope
    for a settled child. Because the provider hands the scope back, the seam
    enforces the ceiling on both paths, and the ceiling is not a rule that a
    provider is trusted to remember.
    """
    ready: anyio.Event = field(default_factory=anyio.Event, repr=False, compare=False)
    """The seam sets it when the child may take its first step. The drive of the
    provider awaits it before the child takes that step.

    A provider starts the drive before it hands the run back, and only then can
    the seam bound the child (`_enforce` needs `scope`). So without this event a
    child ran ahead of its own ceiling. The event is set when the child is
    bounded. For a readmitted child it is set when the children of that child
    have been swept as well (L5b), so the children that its first prompt
    describes are the settled ones. A child can be released before the event is
    set, because it was refused or its admission was not written. The release
    then sets the event. The drive is let through, finds that the child is
    released, and runs nothing."""

    def to_wire(self) -> dict[str, JsonValue]:
        """The admission facts, for a caller that holds the handle.

        There is no `status`. The state of a child is the fold over its own
        `subagent/status` records, and a second copy on the handle would be a
        value frozen at whatever the last in-process update left."""
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
        if self.model_fallback is not None:
            wire["modelFallback"] = self.model_fallback.to_wire()
        if self.media_shortfall is not None:
            wire["mediaShortfall"] = self.media_shortfall.to_wire()
        if self.effort_unsent is not None:
            wire["effortUnsent"] = self.effort_unsent.to_wire()
        return wire


class ChildNotice(WireModel):
    """What the ending of a child tells its parent: a message into the inbox of the
    parent. The ending record itself carries it (`record_ended`).

    It is on the record of the child because the child is what ended. The log of
    the child says what its parent is owed, and the id says whether the log of
    the parent has it. So a crash between the ending of the child and the write
    of the parent costs the parent nothing. The sweep that brings the parent back
    delivers what the log of the parent lacks (`SubagentService._resume`).
    """

    text: str
    summary: str
    """The line a transcript shows for it (`PluginSource.summary`)."""
    id: str = Field(default_factory=new_message_id)
    """The message's own id, which the parent's inbox records it by."""


class Admission(WireModel):
    """The `subagent/admitted` record of a child: the run, the task, **the
    narrowing**, and what the seam stamps beside them.

    The writer and every reader use this one model. The readers include
    `ChildState` and the readmission that rebuilds a request from the record. So
    a key that one side spells and the other side looks for cannot come apart.
    `CodeDispatchRef` states the same argument for the same reason.

    **It holds no session id and no parent id.** The record is in the log of the
    child itself. The id of that log is the session id, and the header of that
    log names the parent (`delegating_parent`).

    **The narrowing is logged because a restart derives the ceiling again from
    this record and from nothing else.** `preset`, `skills`, `tools` and `paths`
    are what the reach of a child is resolved from (`grant_for`). An admission
    that recorded only the run would come back after a restart with the *whole*
    set of the parent. The child would then be quietly wider than the one that
    was admitted, and a power cut would have broken §6.5. `None` means
    "inherited everything", as it does on the request.
    """

    run_id: str
    name: str = ""
    owner: str = ""
    """The name of the provider that runs it, so that a readmission asks that
    provider (P11-04)."""
    prompt: str = ""
    model_provider: str = ""
    model: str = ""
    choice: ModelChoice | None = None
    """The model that the spawn named (`SubagentRequest.choice`), or `None` for
    none. A readmission resolves the route of the child from this, `skills` and
    `reasoning_effort` alone, against the list of its parent as it is at that
    time (`_request_of`)."""
    reasoning_effort: str | None = None
    """The effort the spawn named, `None` for none."""
    model_key: str | None = None
    """The key the spawn resolved to, for the audit (item 8)."""
    route: ModelRoute | None = None
    """The route that the spawn resolved to, with its settings, for the audit.
    `None` for a child on the route of its parent. A readmission works the route
    out again and does not read it back."""
    model_fallback: ModelFallback | None = None
    """The model that the child was asked onto and ran without, for the audit. It is
    what the user who runs the parent must set up for the child to run there."""
    media_shortfall: MediaShortfall | None = None
    """Media its skills handed it that it could not see, for the audit, as
    `model_fallback` is."""
    effort_unsent: EffortUnsent | None = None
    """The effort it ran without, for the audit, as `model_fallback` is."""
    output: dict[str, Any] | None = None
    """The shape that the child owes its value in. A readmission reads it back as it
    was and does not work it out again. The parent was promised this shape, so a
    readmitted child still owes it."""
    requested_access: Access = "read"
    granted_access: Access = "read"
    downgrade_reason: DowngradeReason | None = None
    preset: str | None = None
    profile: str | None = None
    skills: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None
    paths: tuple[str, ...] | None = None
    call_id: str | None = None
    """The call that asked for it. A call that was cut short is answered by it
    (S2)."""
    parent_turn: int | None = None
    """The seq of the latest `turn/start` of the parent at admission. It is the turn
    that a spawn cap counts the child in."""
    goal_id: str | None = None
    """The goal of the parent that was open at admission. The spend of the child is
    charged to it."""


def admission_payload(
    run: SubagentRun,
    request: SubagentRequest,
    *,
    owner: str = "",
    parent_turn: int | None = None,
    goal_id: str | None = None,
) -> dict[str, JsonValue]:
    """The `subagent/admitted` payload for `run`, which was admitted on `request`:
    an `Admission`, in the spelling of the log."""
    entry = request.entry
    admission = Admission(
        run_id=run.id,
        name=run.name,
        owner=owner,
        prompt=request.prompt.strip(),
        model_provider=run.model_provider,
        model=run.model,
        choice=None if request.choice.is_default else request.choice,
        reasoning_effort=request.reasoning_effort,
        model_key=(entry.key or None) if entry is not None else None,
        route=entry.route if entry is not None else None,
        model_fallback=run.model_fallback,
        media_shortfall=run.media_shortfall,
        effort_unsent=run.effort_unsent,
        output=request.output,
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
    # Lists and not the tuples of the model: a list is how the log spells a sequence.
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
    """Write the admission of a child in its own log. A resume finds the child by
    this record (S2).

    The seam writes it (`SubagentService._admit`), and a provider does not. The
    seam holds both the resolved request and the run of the provider, and only
    the seam knows `owner`. Beside the payload, it stamps what a reader of the
    child needs from the side of its parent *at this moment*:

    - The latest `turn/start` of the parent (`parentTurn`), which the spawn caps
      count a turn by.
    - The open goal of the parent (`goalId`), which the spend of the child is
      charged to.

    The seam makes the record durable before the gate of the child opens. A child
    whose admission cannot be written is refused.
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
    """The seq of the latest `turn/start` of `parent`. A child that is admitted now
    is stamped with this turn (`parentTurn`), and the spawn caps count it."""
    opened = parent.latest("turn/start")
    return opened.seq if opened is not None else None


def record_waiting(child: Session, /, **extra: JsonValue) -> None:
    """Write `queued` for a child that is waiting. The child waits for a slot
    (`slots`), or it was stopped with its mount (`SUSPENDED_DETAIL`), or it was
    interrupted and is now held for a credential (`_held_for_credential`).

    There is no barrier, because a wait decides nothing that a crash could get
    wrong. The statuses whose place on disk does decide something have their own
    functions:

    - `record_started`, for a restart that the ladder counts.
    - `record_ended`, for an ending that the parent is about to be told.
    - `record_deleted`, for a revocation.

    **This is the one module that writes `subagent/status`.** So the shape that
    the fold reads has one spelling, and the writers-of-record table can hold a
    provider to it. A provider that appends a status of its own fails
    `test_log_writers`, and that is what keeps these functions from being
    optional.

    Nothing is written for a tombstoned child (`_tombstoned`).
    """
    if _tombstoned(child, "queued"):
        return
    _append_status(child, "queued", **extra)


def _tombstoned(child: Session, status: SubagentStatus) -> bool:
    """Whether `child` is tombstoned, and so takes no status after the tombstone.

    A revocation writes the ending and the tombstone and then waits for the disk
    (`record_deleted`). The drive of a provider can come back from its own wait
    during that wait: past its gate, or out of a run that the revocation
    canceled. The fold takes the last status that it reads. So a `running` or a
    `done` that was written then turned a revoked child into one that ran and
    finished, and the notice of its ending told the parent so. The tombstone is
    in the log before that wait starts. So the functions that write a status hold
    the rule, whichever provider is driving.

    `record_started` and `record_waiting` ask only this, and not whether the child
    ended, because a finished child that is woken writes `running` over its
    `done`.
    """
    if child.latest(DELETED) is None:
        return False
    log.debug("ph.seams.subagents: %s is tombstoned; `%s` not written", child.id, status)
    return True


def _append_status(
    log: Session | SessionBatch, status: SubagentStatus, /, **extra: JsonValue
) -> SessionEvent:
    return _LOG.append(log, STATUS, {"status": status, **extra})


async def record_started(
    ctx: Context, child: Session, /, *, cause: StatusCause | None = None
) -> SessionEvent | None:
    """Write `running` for a child that is about to take its turn. **When the start
    is a restart, the record reaches the disk of the child first** (S10). The
    caller starts the attempt after this returns.

    The ladder counts `running` records with `cause: "resumed"`
    (`restarts_since_progress`). A restart that reached only memory before the
    child took the daemon down again was never counted. A crash loop never
    advanced the count on disk, so `CHILD_RETRY_LIMIT` was never reached. A first
    start counts nothing, and neither does a woken (`rehydrated`) start. So those
    records are written with the next flush of the child. That flush is its first
    model request, which is before anything that the records describe.

    This is best effort. A log that cannot be written has bigger problems than
    this count, and refusing the child its turn would not write the record
    either.

    It returns `None`, with nothing written, for a child that is already
    tombstoned (`_tombstoned`). The child was revoked while its drive waited, and
    the caller does not start the attempt.
    """
    if _tombstoned(child, "running"):
        return None
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
) -> SessionEvent | None:
    """Write the ending of a child in its own log. **The record reaches the disk of
    the child before the caller hands the result to the parent** (F1).

    Nothing else writes the log of the child at that point: its last barrier was
    *before* its last model request. Suppose that a parent is told "done, here is
    the answer" by a child whose log ended before the answer. On the next open,
    repair would call that child interrupted and readmit it to do the work
    again. The caller delivers the result after this returns. This is best
    effort, for the same reason as in `record_started`: an ending that cannot be
    written is still the ending.

    **What the child left unfinished beneath it ends with it** (`_end_beneath`).
    **What the child tells its parent** (`notice`) is appended to the ending
    record. It reaches the disk of the child first, and is then delivered into
    the inbox of the parent (`_deliver_notice`).

    **A child that owes a value cannot end `done` without one**, whichever
    provider ends it. This function settles the value (`settle_value`). It ends a
    child that has no value as `error` and says why. The notice is then one that
    the seam writes, in place of the notice of the provider, which was worded for
    an answer.

    **A child is ended one time**, as `record_deleted` settles a child one time.
    The function returns `None`, with nothing written and nobody told, for a
    child that is no longer live. Such a child is tombstoned or has already
    ended. The revocation of a tombstoned child wrote its ending, and its parent
    is owed no notice of one. The parent of a child that had already ended was
    told then. Each provider path and seam path that ends a child relies on this
    and does not ask first.
    """
    if not child_is_live(_standing(ctx, child)):
        log.debug("ph.seams.subagents: %s has ended; `%s` not written", child.id, status)
        return None
    fields: dict[str, JsonValue] = dict(extra)
    problem = settle_value(child).problem if status == "done" else ""
    failed = _failed_notice(child, problem, summary="returned no value") if problem else None
    if failed is not None:
        status = "error"
        fields["detail"] = problem
        notice = failed
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


def said_text(answer: str) -> str:
    """What a notice says about the last text of a child, or nothing when the child
    said nothing. Every notice that carries this text uses this one wording,
    including the notices of a provider."""
    return f" Last assistant text: {answer}" if answer else ""


def _failed_notice(child: Session, why: str, *, summary: str) -> ChildNotice | None:
    """What a child that failed tells its parent: which child, why, and what it last
    said. All three are read from the log of the child itself, which holds its
    name and its run id. `summary` follows the name. The result is `None` for a
    log that no admission names, because such a log has no parent to tell.

    Two endings use this one wording: the failure that `record_ended` writes for
    a child that owed a value, and the ending of a child that the sweep gives up
    on (`SubagentService._end_child`)."""
    admission = child.projection(ADMITTED, _admission_of)
    if admission is None:
        return None
    said = said_text(last_answer(child))
    return ChildNotice(
        text=f"[child {admission.name} ({admission.run_id}) failed: {why}.{said}]",
        summary=f"{admission.name} {summary}",
    )


def _deliver_notice(parent: AgentDriver, notice: ChildNotice) -> None:
    """Put the ending notice of a child into the inbox of `parent`, and wake the
    parent (P13-05).

    **The parent is woken, and the notice is not only injected.** A parent that
    is in the middle of a turn, blocked in the tool that waits for the child,
    takes the notice at its next step in both cases. This function is for the
    parent that is *idle*. A restart closed the turn that was waiting, the
    parent was resumed past that turn, and the child was readmitted and finished
    later. `inject` left the result in the inbox of that parent with nothing to
    run it, and the parent stayed "running" on a pending message for the life of
    the daemon. `steer` emits `agent/woken`, and a host that drives the agent on
    demand listens for that event.

    **Nothing is flushed here.** A child never writes the log of its parent, and
    it does not need to. The notice is on the ending record of the child, on the
    disk of the child. A crash can take the notice from the log of the parent
    before the next barrier of the parent, or the parent can be absent when the
    notice arrives. In both cases the sweep that brings the parent back finds
    that the id of the notice is missing, and delivers it. A parent that read
    the notice had written it first, at the barrier before the request that read
    it. So the notice is never delivered twice.
    """
    parent.steer(_notice_message(notice.text, notice.summary, notice.id))


ReturnedVia: TypeAlias = Literal["tool", "text"]
"""How a child handed its value back: through `submit_result`, or as a reply that was
the JSON object itself."""

REMINDED = "reminded to return its result"
"""The summary that the one reminder of a child carries. The end of a later turn
reads the summary to tell that the reminder was sent (`_reminded`). The summary
identifies the reminder, because the other messages that the seam puts into an
inbox are ending notices."""


def record_returned(child: Session, /, value: JsonObject, *, via: ReturnedVia) -> SessionEvent:
    """Write the checked value of a child in its own log. `settle_value` reads the
    record back, and the fold carries the value to the parent
    (`ChildState.value`).

    The record has no barrier of its own. The ending of the child comes after it
    and flushes it (`record_ended`). When a crash takes the value before that
    flush, the readmitted run asks for the value again.
    """
    return _LOG.append(child, RETURNED, {"value": value, "via": via})


def _result_tool(output: dict[str, Any]) -> ToolDefinition:
    """`submit_result` for one child. Its arguments are the value, and they are
    checked against `output`. A value that does not match is an error that names
    each violation. The model fixes it on its next step, as it would fix the
    arguments of any tool. A value that matches is recorded in the log of the
    child itself, and the turn of the child ends."""

    # `Any`, because `define_tool` asks for it on a body with a raw schema. The
    # check below does the work of the type.
    def execute(arguments: Any, run: ToolRunContext) -> str:  # noqa: ANN401
        violations = validate_json_schema_value(output, arguments)
        if violations:
            raise ValueError(
                f"this value is not the shape asked for: {'; '.join(violations)}. "
                f"Call {RESULT_TOOL} again with it corrected."
            )
        if run.session is None:
            raise ValueError(f"{RESULT_TOOL} returns a delegated child's value into its log")
        record_returned(run.session, as_obj(arguments), via="tool")
        run.conclude_turn()
        return "Your result is recorded; you are done."

    return define_tool(
        RESULT_TOOL,
        "Return your result, which is how this task ends. Call it once you are done, "
        "with your result as the arguments, in the shape they declare. A value that "
        "does not match comes back as an error saying what to fix.",
        parameters=output,
        output={"type": "string"},
        execute=execute,
        effect_free=True,
    )


def _output_note(output: dict[str, Any]) -> str:
    """The section of the prompt of a child which says that the child owes a value,
    and in what shape. The schema of its tool says the shape too. This section
    says it where a child that works in code, or that makes no tool calls, reads
    it."""
    return (
        f"You owe your result in a set shape. When you are done, call `{RESULT_TOOL}` "
        f"with it (from code, `await tools.{RESULT_TOOL}(...)`); that is your answer, and "
        "your task is not finished until a call succeeds. If you cannot call it, end with "
        "a reply that is only the JSON object. The shape, as JSON Schema:"
        f"\n\n```json\n{json.dumps(output, indent=2)}\n```"
    )


@dataclass(frozen=True, slots=True)
class Returned:
    """What a child with an `output` handed back (`settle_value`): its value, or why
    there is none."""

    value: JsonObject | None = None
    problem: str = ""


def settle_value(child: Session) -> Returned:
    """What a child that has run to quiescence returned. It is the value that the
    child returned in this start, through `submit_result` or as its last reply
    (which is then recorded as the value), or the reason why there is none.

    A provider calls this to word its own ending. `record_ended` holds every
    ending to the same answer, and the result of the waiter follows the log
    (`_settle_waiter`).

    The shape that the child owes is the one in its admission. So a child whose
    parent has no log, and which therefore wrote no admission, is held to
    nothing here. A child that was woken to be asked something else
    (`rehydrated`) owes nothing a second time: its value went to its parent when
    it first ended.
    """
    output = _owed(child)
    if output is None:
        return Returned()
    returned = _value_of(child, output)
    if returned.problem:
        return Returned(problem=f"it returned no value in the shape asked for: {returned.problem}")
    return returned


async def _remind_owed(agent: AgentDriver, _turn: int) -> None:
    """Remind a child that is about to stop, owes a value and has not returned it.
    The child is reminded one time in each start. The reminder objects to the
    stop as any `agent/turn-stopping` listener does: it steers one message in,
    so the same turn goes on.

    Who owes what is read from the log of the child itself, from the `output` of
    its admission. So a readmitted child is held to it too. The count of
    reminders is read from that log as well, for the same reason as in
    `nudges_since` (`ph_stabilize.skill_steps`): a counter on a listener starts
    at zero after a restart.
    """
    child = agent.session
    # A root owes nothing, and is not made to read its whole log back to say so.
    output = _owed(child) if child is not None and child.header.delegating_parent else None
    if child is None or output is None:
        return
    returned = _value_of(child, output)
    if returned.value is None and not _reminded(child):
        agent.steer(_notice_message(_reminder(returned.problem), REMINDED))


def _value_of(child: Session, output: dict[str, Any]) -> Returned:
    """The value of this start. It is the one that `submit_result` recorded since the
    latest `running` of the child. Otherwise it is the last reply of the child,
    read as the JSON object, and recorded when it passes the check. When that
    reply does not pass, the result holds what was wrong with it, as the
    problem."""
    since = _since(child)
    returned = child.latest(RETURNED)
    if returned is not None and returned.seq >= since:
        return Returned(value=thaw_json(as_obj(returned.data.get("value"))))
    answer = child.projection(_ANSWER, _answer_of)
    try:
        value = validated_shape(answer[1] if answer and answer[0] >= since else "", output)
    except SchemaViolation as violation:
        return Returned(problem=str(violation))
    record_returned(child, value, via="text")
    return Returned(value=value)


def _since(child: Session) -> int:
    """The seq at which the records of this start begin. That is one past the
    latest status of the child, which is its `running` while it runs. A child
    with no status yet has the whole of its log."""
    status = child.latest(STATUS)
    return 0 if status is None else status.seq + 1


def _woken(child: Session) -> bool:
    """Whether the child's latest start woke it to be asked something else."""
    status = child.latest(STATUS)
    return status is not None and status.data.get("cause") == "rehydrated"


def _reminded(child: Session) -> bool:
    """Whether the child was already reminded in this start. The answer is read from
    its log."""
    events = (child.at(index) for index in range(_since(child), child.seq))
    return any(
        event is not None
        and event.type == "user/message"
        and as_str(as_obj(event.data.get("source")).get("summary")) == REMINDED
        for event in events
    )


def _owed(child: Session) -> dict[str, Any] | None:
    """The shape that a child owes its value in for this start, by its own
    admission. It is `None` for a child that owes no value, or that was woken to
    be asked something else."""
    admission = child.projection(ADMITTED, _admission_of)
    if admission is None or admission.output is None or _woken(child):
        return None
    return admission.output


def _admission_of(event: SessionEvent) -> Admission | None:
    """One `subagent/admitted` record as an `Admission`, or `None` for a record that
    does not parse. The fold reads an admission through this function. The
    reminder (`_remind_owed`) and `record_ended` read it as a projection of the
    session. A projection parses the record one time for a session, and not at
    the end of every turn."""
    try:
        return Admission.model_validate(event.data)
    except ValidationError:
        return None


def _answer_of(event: SessionEvent) -> tuple[int, str] | None:
    """The assistant text of one event with its seq, or `None` when the event
    carries none. So the projection scans back to the last turn that said
    something (`last_answer`)."""
    message = derive_event_message(event)
    if message is None or message.role != "assistant":
        return None
    text = text_of(message.content).strip()
    return (event.seq, text) if text else None


def last_answer(child: Session | None) -> str:
    """The last non-empty assistant text of a child, which by convention is its
    answer. It is an incremental fold and not a walk of the log, because the log
    is longest at the time when this is asked."""
    answer = child.projection(_ANSWER, _answer_of) if child is not None else None
    return answer[1] if answer is not None else ""


def _reminder(problem: str) -> str:
    """The one correction that is sent to a child that has not returned its value."""
    return (
        f"You have not returned your result in the shape asked for ({problem}). Call "
        f"`{RESULT_TOOL}` with it now, or reply with only the JSON object."
    )


def _notice_message(text: str, summary: str, message_id: str | None = None) -> Message:
    """A message from this seam into the inbox of an agent: the ending of a child to
    its parent, or a reminder to a child. Both have one shape, so a reader of
    either knows that it came from this seam."""
    return create_user_message(
        content=[{"type": "text", "text": text}],
        source=PluginSource(
            plugin="ph.seams.subagents",
            form="notice",
            summary=summary,
        ),
        message_id=message_id,
    )


def _inbox_ids(log: Session) -> set[str]:
    """The id of every message that was delivered into the inbox of `log`."""
    return {
        as_str(as_obj(message).get("id"))
        for event in log.events
        if event.type == "agent/inbox/spliced"
        for message in as_seq(event.data.get("inserted"))
    }


async def record_deleted(ctx: Context, child: Session, /, reason: str) -> None:
    """Write the tombstone of a child in its own log. For a child that had not ended,
    the `canceled` that ends it is written with the tombstone **in one batch**
    (S14). Both are on the disk of the child before the caller releases
    anything.

    When the two records were written apart, a flush between them left a child
    `canceled` and not deleted. A child that has already settled is not settled
    again: a `canceled` over its `done` turned a finished child into a revoked
    one. Whether the child ended is read from its own log, and not from a flag
    of the caller (`_standing`). A child that is already tombstoned is left as
    it is. **What the child left unfinished beneath it ends with it**
    (`_end_beneath`).
    """
    await _tombstone(ctx, child, reason)
    await _end_beneath(ctx, child)


def _standing(ctx: Context, child: Session) -> ChildState:
    """The state of `child`, which says whether it ended or was revoked. The two
    functions that will not do either twice read it: `record_ended`, and
    `record_deleted` through `_tombstone`.

    The state comes from the cached fold of the seam when there is one. Without
    the cache, each revocation folds a long log again, and the teardown of a
    parent pays that cost for every child that it had. The state comes from the
    log itself when there is no cached fold, or when the child has no admission
    for the seam to know it by."""
    service = ctx.get(SUBAGENTS)
    known = service.state(child.id) if service is not None else None
    return known if known is not None else child_state(child)


async def _tombstone(ctx: Context, child: Session, reason: str) -> None:
    """The write of `record_deleted` itself. It ends nothing beneath the child.
    `_revoke_beneath` writes this on each descendant, because its walk already
    reaches every level."""
    state = _standing(ctx, child)
    if state.deleted:
        return
    with child.batch() as batch:
        if state.status not in SETTLED_STATUSES:
            _append_status(batch, "canceled", reason=reason)
        _LOG.append(batch, DELETED, {"reason": reason})
    await session_written(ctx, child)


async def _end_beneath(ctx: Context, child: Session) -> None:
    """**A child that ends takes with it what it left unfinished beneath it.** The two
    functions that end a child call this (`record_ended` and `record_deleted`).
    So the rule holds whichever provider path or seam path ended the child
    (`SubagentService._revoke_beneath`)."""
    service = ctx.get(SUBAGENTS)
    if service is not None:
        await service._revoke_beneath(child.id)


async def open_child_log(
    ctx: Context, parent: Session, run_id: str, /, **meta: JsonValue
) -> Session:
    """Open the log of a child itself. A log that survived is resumed, and otherwise
    a new one is created. The log is named and filed in the way that its parent
    finds it by (Phase 11).

    The id is the id of its parent followed by the id of its run
    (`child_session_id`). The header names the parent and says that the session
    is a sub-agent. The family of the child is the family of its parent. It is
    given and not inherited, because the store files a child with its parent
    only while the parent is live in the store. It is given to `open_session`,
    which then looks for a surviving log in that family and not across the
    store. Every provider opens its children here, so no provider can name a
    child in a way that a restart would not find. `meta` adds what a provider
    says of its own (`agentPreset`).
    """
    return await open_session(
        ctx,
        child_session_id(parent.id, run_id),
        family=parent.header.family,
        meta={
            "parentSession": parent.id,
            "origin": "subagent",
            "delegationDepth": (parent.header.delegation_depth or 0) + 1,
            **meta,
        },
    )


def parent_went_away(name: str) -> SubagentSpawnError:
    """The refusal of a spawn whose parent had its scope disposed during the
    admission."""
    return SubagentSpawnError(
        f"subagent {name}: its parent went away while it was being admitted, so it was not started"
    )


def admitted_by(children: Mapping[str, ChildState], call: SessionEvent) -> ChildState | None:
    """The child that a call record admitted, or `None` if the call admitted none.

    This is for the `reconcile` of a delegating tool (S2). `call` is the record
    that a crash left unanswered: a `tool/call`, or the
    `tool/code-dispatch-start` of a Code Mode dispatch. `children` are the
    children of the parent, read from their own logs. The admission reaches the
    disk of the child before the child takes a step (`SubagentService._admit`).
    So no child means that none ran, and the call is safe to make again.
    """
    call_id = call_id_of(call)
    if not call_id:
        return None
    # One at most: a readmit writes no second admission.
    return next((state for state in children.values() if state.call_id == call_id), None)


@runtime_checkable
class ReadmittingProvider(Protocol):
    """A provider that can take an admitted child back from the log alone (P5-04).

    A daemon that stopped between the admission of a child and its first turn
    left the work described in the log of the child and running nowhere.
    `readmit` is how the next daemon puts the child back. The run id and the
    session id come from that log. So the child keeps its identity, its name and
    its log, and does not arrive as a second child that the parent never asked
    for.

    It is a Protocol of its own, and not a method on `SubagentProvider`, for
    exactly the reason that `RehydratableProvider` gives. Not every way of
    running a child can resume a child that has *not run*. A `getattr` probe
    would report a provider whose method is misnamed as one that cannot do it.

    A return of `None` declines this one child and does not fail the sweep. The
    next daemon is starting a root, and one child that it cannot rebuild must
    not stop the others from coming back.
    """

    async def readmit(
        self, request: SubagentRequest, *, run_id: str, session_id: str, restarts: int = 0
    ) -> SubagentRun | None: ...


@runtime_checkable
class RevokingProvider(Protocol):
    """A provider that can stop a child that it is running, and tombstone it
    (Phase 11).

    Every revocation goes through `SubagentService.delete`. A child that this
    process is running belongs to its provider, with its job and its agent, and
    the provider stops it. The provider writes the tombstone into the log of the
    child as it lets go (`record_deleted`). A child that nothing is running
    here, because an earlier process settled it, is tombstoned by the seam
    itself. `False` means that the provider holds no such child.
    """

    async def revoke(self, run_id: str, reason: str) -> bool: ...


@runtime_checkable
class RehydratableProvider(Protocol):
    """A provider whose settled children can be given a runtime again (P3-13).

    It is a second Protocol and not a method on `SubagentProvider`, because not
    every way of running a child can resume one. It is not a `getattr` probe,
    because a provider whose method is misnamed or has the wrong arity would
    then fail silently as "cannot rehydrate". That failure mode has already cost
    this package a day.
    """

    async def rehydrate(self, run_id: str) -> bool: ...


@runtime_checkable
class SubagentProvider(Protocol):
    """A way of running a child agent.

    It has one method. A `capabilities` set was drafted here for a consumer to
    branch on: whether children survive the parent, and whether they can be
    messaged. It was removed again. With one provider there is nothing to branch
    on, and the vocabulary that a second provider needs cannot be guessed from
    the first.
    """

    async def start(self, request: SubagentRequest) -> SubagentRun: ...


@dataclass(frozen=True, slots=True)
class _Registered:
    """A delegation provider and the row that registered it (P6-29).

    This is one of the two row-supplied objects that `_row_bodies` cannot find
    by shape. A provider is an *object that satisfies a Protocol*, so
    `dict[str, SubagentProvider]` names no callable. The service also claims the
    entry with `claim_key` and not with `claim_slot`. `_provider_fields`
    discriminates on the Protocol, and that holds for a provider however it was
    registered.
    """

    provider: SubagentProvider
    by: Running


SpawnGuard: TypeAlias = Callable[[SubagentRequest], "Deny | str | None"]
"""A policy that is asked before a child exists. It returns a reason to refuse, or
`None` to allow.

A guard can only deny, and it is asked before the provider is. So a refusal
produces no session, no log and no artifact, which is the contract of
`SubagentSpawnError`. The child caps of the limits row are the first registrant
(P4-04)."""


@dataclass(frozen=True, slots=True)
class _SpawnGuard:
    check: SpawnGuard
    by: Running


@dataclass(frozen=True, slots=True)
class ChildCounts:
    """The children of a parent, as the spawn caps count them
    (`SubagentService.child_counts`)."""

    turn: int = 0
    """The children admitted in the current turn of the parent, and the spawns that
    are in progress."""
    session: int = 0
    """All the children, including a deleted one, and the spawns that are in
    progress."""


@dataclass(frozen=True, slots=True)
class _Readmitter:
    """The provider that the admission of a child names, when it can readmit the
    child: its name, the row that registered it, and the provider itself as a
    `ReadmittingProvider`. It is looked up one time for each child and then
    passed on."""

    name: str
    by: Running
    provider: ReadmittingProvider


@dataclass(slots=True)
class SubagentService:
    """The service published as `ctx.subagents`.

    It holds named providers, where `ctx.code_runtime` has a single slot. "Run a
    child" has genuinely different answers in one deployment, such as an RLM
    child and an ephemeral research task, and the caller names the one that it
    wants.
    """

    ctx: Context
    _providers: dict[str, _Registered] = field(default_factory=dict)
    _runs: dict[str, SubagentRun] = field(default_factory=dict)
    _guards: list[_SpawnGuard] = field(default_factory=list)
    _folds: SessionFoldCache[ChildState] = field(
        default_factory=lambda: SessionFoldCache(child_state, extend=extend_child_state)
    )
    """The state of each live child, folded from its own log and cached for each
    session. The prompt, the roster tool of the model, the spawn caps and every
    name lookup read it several times in each model step.

    The *fold* stays a pure function of one log, because `child_state_of` must
    work on a stored log too. So drift is checked for each child (I6), and no
    key spans logs."""
    _live: dict[str, set[str]] = field(default_factory=dict)
    """The live children of each parent, by session id. The table is kept up to date
    as sessions are published and let go (`_published`, `_let_go`). That is
    enough, because the parent link of a child never changes. So reading the
    children of a parent costs a lookup of those children, and not a pass over
    every session that the mount holds."""
    _stored: dict[str, dict[str, ChildState]] = field(default_factory=dict)
    """The children that this process is not running, by parent id and then by
    session id. A parent is in the table after its stored children were read
    (`load_children`). When a child of such a parent is let go, its last state
    stays here (`_let_go`)."""
    _unreadable: dict[str, set[str]] = field(default_factory=dict)
    """The children whose logs cannot be read, by parent id and then by session id
    (P13-09). Either the tree read found the damage (`stored_tree`), or the
    strict read of a readmission did (`_isolate`). Such a child is neither live
    nor ended, so no sweep waits on it or writes into it."""
    _spawning: dict[str, set[str]] = field(default_factory=dict)
    """The names of the spawns of each parent that have passed the guards and are not
    yet admitted, by parent id (`child_counts`)."""
    _sweeps: KeyedLocks[str] = field(default_factory=KeyedLocks)
    """One sweep of the children of a parent runs at a time, by parent id (S1). A
    second sweep waits.

    `_resume` asks `state.run_id in self._runs` and then awaits, while a hold is
    written or a log is opened, before `_admit` writes the entry. So two sweeps
    of one parent that overlapped both took the same child to readmit. The write
    or the open of the second sweep failed on the log that the first held. Its
    `except Exception` then ended the child that the first was putting back, and
    told the parent that the child had failed. Two `credentials/store` requests
    in a row are two such sweeps.

    **The second sweep waits and does not skip.** It reads the state that the
    first left, so a credential that arrived while the first ran is not missed.
    The lock is for each parent. So the sweep of the children of a readmitted
    child (`_sweep_readmitted`) and the descent in `readmit_waiting` each take
    the lock of their own parent, and not this one."""

    def register_provider(
        self, name: str, provider: SubagentProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim one delegation strategy under `name`."""
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._providers, name, _Registered(provider, by), label="subagent-provider"
        )

    def guard(self, check: SpawnGuard, *, scope: Context | None = None) -> Disposer:
        """Register a deny-only policy that is asked before every admission (P4-04).

        It has the shape of `ToolRuntime.guard`, for the same reasons. It is
        monotonic: a guard can refuse and can never widen. It is asked *before*
        the provider is, so a refused spawn produces nothing to clean up. The
        seam *asks* for a count of children, and a policy row *decides* on it.
        For that reason the child caps are a registration here, and not a field
        on the config of this seam.
        """
        by = self.ctx.running_for(scope)
        return claim_entry(by.owner, self._guards, _SpawnGuard(check, by), label="subagent-guard")

    def child_counts(self, parent: Session) -> ChildCounts:
        """How many children `parent` has made, in this turn and in all. This is what a
        spawn cap counts. It is read from the logs of the children themselves
        when a spawn is judged.

        **A spawn that is in progress counts as a child.** Its guards run before
        its provider builds the child, and the child is in `children` only after
        its admission is written. When nothing else was counted, two spawns from
        one step each saw the other missing, because the driver runs the tool
        calls of a step side by side. Both then passed a cap that one of them
        crossed. A spawn is on the list of its parent from its last guard until
        its admission is written or it is refused, and neither hand-off has an
        await. So it is counted one time, in this turn, which is the turn that
        its admission stamps.

        **The turn is the latest `turn/start` of the parent, by seq**
        (`_current_turn`). A child that was admitted before the parent had a
        turn counts in the turn only while the parent still has none. Deleted
        children count. A fork starts with none, because no child names it.
        """
        children = self.children(parent.id).values()
        spawning = len(self._spawning.get(parent.id, ()))
        turn = _current_turn(parent)
        return ChildCounts(
            turn=spawning + sum(1 for child in children if child.parent_turn == turn),
            session=spawning + len(children),
        )

    def _named(self, request: SubagentRequest, parent: Session | None) -> str:
        """The name that the child of a spawn is addressed by. It is unique among its
        siblings.

        **The seam names the child before it asks the provider**, for the same
        reason as in `child_counts`: when a provider named a child from the
        admitted ones, two spawns from one step could take one name. `start`
        holds the name on the list of the spawns of the parent that are in
        progress. A requested name that a sibling already has is refused,
        because names address children (`agent_message`, the roster). A name
        that is left out is made from the task (`default_child_name`).
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
        """Take a spawn off the list of its parent. Either it was admitted, and is a
        child from now on, or it was refused. A second call does nothing."""
        if parent is None or (names := self._spawning.get(parent.id)) is None:
            return
        names.discard(name)
        if not names:
            del self._spawning[parent.id]

    def provider_names(self) -> list[str]:
        return sorted(self._providers)

    def resolve(self, name: str | None) -> str | None:
        """Which provider a consumer should use, or `None` when there is no answer.

        The policy is here because it is a property of *this* table:

        - No name means the one provider that is mounted.
        - Two mounted providers with no name chosen is a question that this seam
          refuses to answer for a caller.
        - A configured name is checked against what is actually mounted, and is
          not trusted.

        The last rule is the important one. A row that trusts its own
        `provider:` setting advertises a capability whenever the provider row
        was renamed or removed. The capability that does not exist then arrives
        by the one route that looks like a deliberate choice.
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
        """What the parent holds, in the two registries that a grant covers.

        Both halves of the ceiling have one reader. The refusal, the
        materialization and the check for "did this narrow anything" all need
        the same two sets. If they were computed three times in three spellings,
        the three would come to disagree about what "holds" means.

        `boundary` is passed in for the same reason that `held` is a parameter
        of `check_grant` and `grant_for`. `start` needs the resolution three
        times: for the ceiling, for the brief and for the containment check.
        Passing the one answer down makes the three agree by construction, and
        not because each derived it again.

        The `submit_result` of a parent is not among what it holds. It is how
        that parent answers its own parent. A child that was handed it would
        return a value into a contract that nobody has.
        """
        parent_scope = boundary if boundary is not None else self._delegating_boundary(request)
        skills = self.ctx.get(SKILLS)
        tools = self.ctx.get(TOOLS)
        return (
            tuple(sorted(skills.reach(parent_scope))) if skills is not None else (),
            tuple(name for name in tools.names(scope=parent_scope) if name != RESULT_TOOL)
            if tools is not None
            else (),
        )

    def _delegating_boundary(self, request: SubagentRequest) -> Context:
        """The boundary that the ceiling of a spawn is computed in. It is stated, not
        guessed (P6-31).

        There are four cases:

        - **A stated `request.scope`** wins, as it does everywhere else. A
          stated scope wins before any handle is consulted, and that is the
          security property here.
        - **No scope and no parent** is the mount. A spawn with no parent is a
          root delegation, and the deployment-wide set is what it legitimately
          holds.
        - **A parent whose `.ctx` cannot be read** is **refused**. `None` was
          never "no ceiling". Before P6-32, `SkillService.reach` and
          `ToolRuntime.names` both resolved it to the *mount*, which is the
          unrestricted set. So an unreadable parent did not narrow a child: it
          handed the child everything that the deployment holds. There is no
          narrower default to pick. The stake on this path is that a spawn which
          could widen would make delegation a privilege escalation (I7).
        - **A parent with a usable `.ctx`** is that scope.

        The boundary is resolved here, at the entry, by the code that knows that
        a parent was meant. It is never resolved in the downstream registries.
        Since P6-32 they require a stated `Boundary`, so they cannot guess
        either.
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

        **The check is here and not in each provider**, because this is the one
        path that every delegation takes. A ceiling that one provider forgot
        would not be a ceiling. The seam applies the grant too, through
        `SubagentRun.scope`. `rehydrate` builds a fresh scope for a settled
        child, and would otherwise hand that child the deployment-wide set.

        The spawn is refused and not silently intersected. A child that came
        back with something other than what was asked for is a result that the
        parent cannot interpret. A `reviewer` child that is missing its review
        skill does the job wrong and reports success.
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
        """Fill in what a named preset supplies and the caller did not state.

        These are defaults and not a ceiling. A caller may still narrow further.
        It cannot widen past the parent whatever it names, because `check_grant`
        runs on the *resolved* request. An unknown name is refused and not
        ignored. A spawn that asked for a `reviewer` and silently got a generic
        child is the failure that `resolve_model` refuses, for the same reason,
        for the model field.
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

    def resolve_model(
        self, request: SubagentRequest, *, readmitting: bool = False
    ) -> SubagentRequest:
        """Resolve the route that a child runs on, by the list of its parent (S7b, item
        8). The route is resolved at the spawn, and again at each readmission.

        The list is the `models` of the mount. That is the profile of the parent
        as it runs, with `/model` included. A child that is resolved onto an
        entry runs with the settings of that entry (`child_options`). A route is
        named in one of three ways:

        - **By the spawn** (`SubagentRequest.choice`). The choice is a key, or a
          whole route. The key is the one that the spawn gives or the one that
          its assigned profile gives. A whole route runs as the entry that it
          matches. A route that matches no entry runs as given with no settings,
          as a root that is started on it does (`ModelList.resolve`).
        - **By the skills that the child is given**, when the spawn names none
          (`model: classify` in their front matter).
        - **By neither.** The child runs on the route of its parent, with the
          options of its parent. It is refused when the parent has no route
          (`_require_parents_route`). That holds at a readmission too, because
          there is nothing to fall back to.

        **A spawn is refused only where whoever chose can choose again at
        once.** A route that the spawn named and that this mount cannot run is
        refused, and the refusal names why. The parent is right there, and
        running the child elsewhere would answer the parent on a model that it
        did not pick. Skills that name different models are refused too, and the
        parent settles them with `model=`. A model that a skill names and that
        this mount cannot run degrades instead, because the author of the skill
        wrote the key without knowing this profile. The child runs on the route
        of its parent. The request carries a `ModelFallback`, the admission
        records it, and the answer of the spawn reports it. So the parent and
        the user both learn what to set up.

        **A readmission degrades all of them** (`readmitting`). Nobody is there
        to choose again, and a refusal would end the child. The route is worked
        out again and not read back from the admission. So a model that the user
        has set up since then reaches a readmitted child as it reaches a new
        one, and so does a skill that now names another model.

        **Then the media that its skills hand it** (`Skill.accepts`, A3). The
        route that the child resolved to is asked whether it can take them
        (`_with_media`). A route that cannot take them gives way to the route of
        the parent where that route can. A route that the spawn named itself is
        refused there instead, by the rule above. Where neither route can take
        them, the child runs and sees a placeholder for each, and its parent is
        told (`MediaShortfall`).

        **Then the effort that its skills name** (`Skill.effort`).
        `child_options` puts it under the effort of the spawn and over the
        effort of the route (`_with_effort`). When the profile of the route does
        not say how to send the effort that wins, the child runs without it and
        its parent is told (`EffortUnsent`). This is never a refusal: in both
        cases the child is on the model that it was asked onto.
        """
        routed = self._with_media(
            self._resolve_route(request, readmitting=readmitting), readmitting=readmitting
        )
        return self._with_effort(routed)

    def _with_effort(self, request: SubagentRequest) -> SubagentRequest:
        """`request` with the effort that its skills name, where they agree, and with
        the effort that it runs at where its route cannot send that effort.
        Skills that name different efforts name none, so the child runs at the
        effort of its route. An effort is a setting to degrade around. It is not
        a contract that the parent must settle."""
        named = self._named_by_skills(request, lambda skill: skill.effort)
        efforts = set(named.values())
        if len(efforts) > 1:
            log.warning(
                "a child's skills name different efforts (%s), so it runs at its route's",
                ", ".join(f"{name}: {effort}" for name, effort in sorted(named.items())),
            )
        request = replace(request, skill_effort=efforts.pop() if len(efforts) == 1 else None)
        return replace(request, effort_unsent=self._unsent_effort(request))

    def _unsent_effort(self, request: SubagentRequest) -> EffortUnsent | None:
        """The effort that the child of `request` runs at, when the profile of its route
        does not say how to send it. The question is asked by the same rule that
        the request of the child itself withholds the effort by
        (`unmapped_effort`). It is asked of the list that the spawn resolved its
        route from."""
        options = child_options(request)
        listed = self.ctx.get(MODELS) or ModelList()
        defined = listed.efforts_for(options.provider, options.model, key=options.model_key)
        if not unmapped_effort(options.reasoning_effort, defined):
            return None
        route = f"{options.provider}/{options.model}"
        return EffortUnsent(effort=options.reasoning_effort, route=route, defined=tuple(defined))

    def _resolve_route(self, request: SubagentRequest, *, readmitting: bool) -> SubagentRequest:
        """The route of `resolve_model`, before the question about media is asked."""
        choice = request.choice
        by_skills = choice.is_default
        named = self._named_by_skills(request, lambda skill: skill.model) if by_skills else {}
        keys = tuple(sorted(set(named.values())))
        if len(keys) > 1:
            said = ", ".join(f"{name} names {key}" for name, key in sorted(named.items()))
            fallback = ModelFallback(asked=keys, reason="contested")
            why = f"the skills this child is given name different models ({said})"
            refusal = f"{why}; name the one it should run on with `model=`"
        else:
            if keys:
                choice = ModelChoice(key=keys[0])
            elif by_skills:
                self._require_parents_route(request)
                return request
            try:
                return replace(request, entry=choose(self.ctx, choice))
            except UnrunnableChoice as error:
                fallback = ModelFallback(asked=(choice.spelled,), reason=error.reason)
                why = f"{choice.spelled!r} cannot run here: {error}"
                refusal = f"a child cannot run on {choice.spelled!r}: {error}"
        # Refusable where whoever chose can choose again at once: a spawn, about its own
        # choice or about skills that disagree.
        refusable = fallback.reason == "contested" or not by_skills
        return self._degraded(request, fallback, why, refusal, refusable, readmitting)

    def _degraded(
        self,
        request: SubagentRequest,
        fallback: ModelFallback,
        why: str,
        refusal: str,
        refusable: bool,
        readmitting: bool,
    ) -> SubagentRequest:
        """`request` on the route of its parent, with the fallback reported in the log
        and on the request. Where whoever chose can choose again at once, and
        this is not a readmission, the spawn is refused instead."""
        if refusable and not readmitting:
            raise SubagentSpawnError(refusal)
        self._require_parents_route(request)
        log.warning("a child runs on its parent's route, since %s", why)
        return replace(request, entry=None, model_fallback=fallback)

    def _with_media(self, request: SubagentRequest, *, readmitting: bool) -> SubagentRequest:
        """`request` on a route that can take the media that its skills hand the child,
        where there is one (`resolve_model`)."""
        needed = sorted(
            {media for skill in self._given_skills(request).values() for media in skill.accepts}
        )
        if not needed:
            return request
        llm = self.ctx.require(LLM)

        def missing(provider: str, model: str) -> tuple[str, ...]:
            return untaken(needed, llm.resolve_model(provider, model).accepts)

        parent, entry = request.parent.options, request.entry
        route = entry.route if entry is not None else None
        unseen = missing(
            *((route.provider, route.model) if route else (parent.provider, parent.model))
        )
        if not unseen:
            return request
        if entry is not None and not missing(parent.provider, parent.model):
            asked = entry.key or entry.route.label
            said = f"{asked!r} cannot take {', '.join(unseen)}, which its skills hand it"
            refusal = (
                f"a child cannot run on {asked!r}: {said}; name a model that takes them, or "
                "leave `model=` unset to run it on yours"
            )
            fallback = ModelFallback(asked=(asked,), reason="cannot-accept", media=unseen)
            refusable = not request.choice.is_default
            return self._degraded(request, fallback, said, refusal, refusable, readmitting)
        label = route.label if route else f"{parent.provider}/{parent.model}"
        log.warning(
            "a child on %s sees placeholders for %s, which its skills hand it and neither "
            "its route nor its parent's takes",
            label,
            ", ".join(unseen),
        )
        return replace(request, media_shortfall=MediaShortfall(media=unseen, route=label))

    @staticmethod
    def _require_parents_route(request: SubagentRequest) -> None:
        """Refuse a child that is left on the route of its parent when the parent has
        none."""
        options = request.parent.options
        if not (options.provider and options.model):
            raise SubagentSpawnError(
                "a subagent needs a provider and a model; the parent has none to inherit"
            )

    def resolve_output(self, request: SubagentRequest) -> SubagentRequest:
        """Resolve the shape that the child of a spawn owes its value in. It is the
        `output` of the spawn itself, else the one that the skills it is given
        declare, else none.

        A schema that pH cannot hold a value to is refused
        (`object_schema_problems`). Skills that declare different outputs are
        refused too, and the parent settles them by passing `output`. These are
        refused and not degraded: the parent asked for a structure that it means
        to read, and a child that runs without that contract would answer some
        other question. The shape is not resolved again at a readmission. The
        admission holds the shape that the parent was promised (`_request_of`).
        """
        if request.output is not None:
            # A plain dict from here on. It is the type that the request declares,
            # and it is sent to the model of the child as the schema of its tool.
            try:
                output = checked_object_schema(request.output)
            except ValueError as problems:
                raise SubagentSpawnError(
                    f"the output a child was asked for cannot be checked: {problems}"
                ) from problems
            warning = structural_warning(output)
            if warning is not None:
                log.warning("a child's output: %s", warning)
            return replace(request, output=output)
        declared = self._named_by_skills(request, lambda skill: skill.output)
        outputs = list(declared.values())
        if any(output != outputs[0] for output in outputs[1:]):
            raise SubagentSpawnError(
                f"the skills this child is given declare different outputs "
                f"({', '.join(sorted(declared))}); pass the one it should return as `output`"
            )
        return replace(request, output=outputs[0] if outputs else None)

    def _named_by_skills[T](
        self, request: SubagentRequest, read: Callable[[Skill], T]
    ) -> dict[str, T]:
        """What each skill that a spawn names asks for, as `read` returns it, keyed by
        skill name. Skills that ask for nothing are left out."""
        return {
            name: value
            for name, skill in self._given_skills(request).items()
            if (value := read(skill))
        }

    def _given_skills(self, request: SubagentRequest) -> dict[str, Skill]:
        """The skills that a spawn names, by name, as the parent holds them."""
        service = self.ctx.get(SKILLS)
        if not request.skills or service is None:
            return {}
        boundary = self._delegating_boundary(request)
        return {
            name: skill
            for name in request.skills
            if (skill := service.get(name, boundary)) is not None
        }

    async def resolve_profile(self, request: SubagentRequest) -> SubagentRequest:
        """Resolve a profile that the parent assigns, as the narrowing that it is (S7b,
        item 9).

        The host composes the profile (`ctx.named_profiles`), and it is read
        against the mount of the parent
        (`ph.seams.subagent_profiles.narrowing`). It is then written into the
        request as the tools, the skills, the model and the writable paths that
        it resolves to. So the ceiling check that follows checks what the
        profile gives, and the admission records it: the reach of a child is
        fixed when it is admitted. **The profile is a ceiling, not a set of
        defaults.** What a spawn names beside it may narrow further, and a spawn
        that names more than the profile gives is refused.
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
            # On a worker thread. Composing reads and parses every layer file, and a
            # spawn runs on the event loop that every root of a daemon shares.
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
            choice=request.choice
            if not request.choice.is_default
            else ModelChoice(key=narrowed.limit.model_key or ""),
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

        `None` means "everything that the parent holds", and it is written out
        as that explicit list. The isolation chain already bounds the child. The
        list is for the ruling that **the capability of a child is fixed at
        admission**. So it records what the parent held *then*, and does not
        defer to what the parent holds whenever the child is next asked. See
        `Grant`.
        """
        held_skills, held_tools = held if held is not None else self.held_by(request, boundary)
        named = request.skills
        skills = self.ctx.get(SKILLS)
        target = boundary if boundary is not None else self._delegating_boundary(request)
        brief, briefed = _brief(skills, named, target) if named and skills is not None else ("", ())
        return Grant(
            skills=named if named is not None else held_skills,
            tools=request.tools if request.tools is not None else held_tools,
            paths=request.paths,
            brief=brief,
            briefed=briefed,
        )

    async def _end_child(self, session_id: str, detail: str, *, tell_parent: bool = False) -> None:
        """End a child that the seam is giving up on, in the log of the child itself,
        and tell its parent.

        **Nothing can release a child that has an admission and no ending.** Its
        log says that it was admitted, so `child_is_live` reads it as working,
        and a child that no longer exists holds its parent out of passivation
        (K3, K4). The child has to *end*, and only its log can end it.

        **The parent was promised a report** (P13-05). The delegating tool told
        the parent that the child "reports back by message". A child that the
        sweep gives up on is spent, unresumable or without a provider. Such a
        child used to end in silence, and the parent waited for a reply that
        nobody would write. With `tell_parent`, the ending carries a notice that
        says which child, why, and what it last said (`_failed_notice`). The
        notice is delivered in the way that the notice of any ending is. A spawn
        that is refused at admission gets no notice, because the refusal is
        already the answer to the caller.

        A child that has already ended is left as it ended (`record_ended`).
        `_abandon` ends a readmission that `_admit` refused, and the sweep that
        catches the refusal then comes here for the same child.

        This is best effort and never raises. It is how the seam gives up on a
        child that is refused, unresumable or spent, and a root that is coming
        back must not be blocked by a log that it could not write. A child whose
        spawn was refused before it had a log leaves nothing to end.
        """

        async def end(child: Session) -> None:
            notice = (
                _failed_notice(child, detail, summary="was given up on") if tell_parent else None
            )
            await record_ended(self.ctx, child, "error", detail=detail, notice=notice)

        try:
            await self._write_child(session_id, end)
        except Exception:
            log.exception("ph.seams.subagents: %s could not be ended in its own log", session_id)

    async def _write_child(
        self, session_id: str, write: Callable[[Session], Awaitable[object]]
    ) -> None:
        """Write to the log of a child itself, whether or not this process is running
        the child.

        The session of a live child is written as it is. A child that this
        process does not hold was settled by an earlier process, or is not yet
        readmitted. Its log is claimed and loaded without being resumed
        (`stored_session`), then written, flushed and let go. The sweep ends a
        child that it will not readmit in this way, and records a hold on a
        child that waits for a credential. Neither is a reason to rebuild the
        runtime of the child. When there is nothing to write to, because there
        is no live session and none is stored, the call does nothing.

        **The log is read by path when the state of the child says where**
        (`ChildState.family`). Every child that a sweep writes to was read from
        the store with its header (`load_children`), so its log is opened where
        that header files it. A lookup by id instead listed every family
        directory in the store, on the event loop, one time for each child. A
        child that this service knows nothing of is looked for first, so no
        lease is taken on an id that has no log.
        """
        sessions = self.ctx.require(SESSIONS)
        live = sessions.get(session_id)
        if live is not None:
            await write(live)
            return
        store = self.ctx.get(SESSION_PERSISTENCE)
        state = self.state(session_id)
        family = state.family if state is not None else None
        if store is None or (
            family is None and not await anyio.to_thread.run_sync(store.exists, session_id)
        ):
            return
        try:
            child = await stored_session(self.ctx, session_id, family=family)
        except LookupError:
            return
        try:
            await write(child)
            await session_written(self.ctx, child)
        finally:
            sessions.dispose(session_id)

    async def _abandon(self, run: SubagentRun, why: str, *, tell_parent: bool) -> None:
        """End and release a child that this seam has decided not to admit (K3).

        **The child is ended first, with `why`, and then released.** So its
        account is on its disk before anything is let go, as the account of a
        revocation is (`record_deleted`). The release revokes nothing
        (`SubagentRun.dispose`), so a refused child reads `error`, not deleted.

        The release is best effort. The refusal is what the caller has to see,
        and a disposer that throws must not replace it with its own exception.
        The release uses `releasing` and not a bare shield, for the reason that
        `releasing` gives: a disposer that hangs must not make the refusal
        unkillable.
        """
        await self._end_child(run.session_id, why, tell_parent=tell_parent)
        if run.dispose is None:
            return
        with releasing(), suppress(Exception):
            await maybe_await(run.dispose())

    async def _admit(
        self,
        run: SubagentRun,
        request: SubagentRequest,
        *,
        record: bool,
        owner: str,
        grant: Grant,
        held: tuple[tuple[str, ...], tuple[str, ...]],
        boundary: Context,
    ) -> SubagentRun:
        """Bind a child that a provider has already started on `request`, or release it
        (K3, L6).

        **The sequence is in one place, because both spawners need all of it.**
        `_enforce` reads `run.scope`, which only the provider can produce. So
        the check cannot move ahead of the spawn, and when it refuses, a child
        is already driving. Three things must then happen, and none of them is
        optional. The admission is *ended* in the log, the child is released
        (`_abandon` does both, in that order), and the refusal reaches the
        caller.

        `start` had all three and `_readmit_one` had none of them. It called
        `_enforce` bare. So a `check_grant` refusal after a profile edit left
        the child running unbounded and undisposed, while the `except` of the
        sweep (`_resume`) ended its record. That was the failure of K3, arriving
        through the path that K4 had already been written for. The defect was
        two spawners with one admission each. The fix is one admission that both
        call.

        With `record`, which `start` passes, **the seam writes the admission of
        the child into the log of the child itself** (S2, Phase 11). It is the
        one record that a resume finds a child by. The seam writes it and the
        provider does not, because the seam holds both halves: the resolved
        request and the run of the provider. It also holds `owner`, which is how
        a readmission finds the provider again. The admission reaches the disk
        of the child before the gate opens. It fails closed, as a durable intent
        does: an admission that cannot be written is a refusal, like the refusal
        of the ceiling. A provider that opened no log for its child is refused
        too, because that log is the only record of the child. A readmission
        writes no admission, because the one that it was rebuilt from is already
        there.

        The spawn of `start` comes off the list of its parent in the step that
        writes its admission. That is the step in which it starts to count as a
        child (`child_counts`).

        The caller opens the gate of the child (`SubagentRun.ready`) after this
        returns. `start` opens it at once. A readmission opens it after the
        children of the child are swept.
        """
        # The owner is stamped here and not trusted from the provider. The service
        # knows which name the caller asked for, and `rehydrate` must be able to
        # find the same provider again. The fallback is what this start resolved
        # (`resolve_model`), at a readmission as at a spawn. The output is what
        # the child owes its parent.
        run.owner = owner
        run.model_fallback = request.model_fallback
        run.media_shortfall = request.media_shortfall
        run.effort_unsent = request.effort_unsent
        run.output = request.output
        self._settle_waiter(run)
        child = self.ctx.require(SESSIONS).get(run.session_id)
        # A parent with no log of its own has nothing that a child could be found
        # by, and nothing is recorded for its child. This is the one case in which
        # a spawn is not durable.
        parent = request.parent.session if record else None
        try:
            if parent is not None:
                if child is None or not _names_parent(child, parent.id):
                    # The parent finds the child after a restart by the log of the
                    # child (`SessionArchive.descendants_of`). So nothing could bring
                    # back a child with no log, or with a log that does not name its
                    # parent.
                    raise SubagentSpawnError(
                        f'subagent {run.name}: the "{owner}" provider opened no log for it '
                        "that names its parent, and a child's own log is its only record, "
                        "so it was not started"
                    )
                # The admission is written before the ceiling is checked. So a child
                # that is refused below reads as admitted and then ended, and not as
                # a log with no account of what it was.
                record_admitted(self.ctx, child, run, request, owner=owner)
                self._landed(parent, run.name)
            try:
                self._enforce(grant, run, held, boundary)
                self._give_output(run)
            except InactiveScopeError as gone:
                # The ceiling registers on the scope of the child, and a parent that
                # went away took that scope with it. This is the same refusal that
                # the builds of a provider itself get.
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
        except BaseException as stopped:
            # **Any way out ends what the provider started** (S2, Holdings rule 2),
            # and not only a refusal. A cancel at the flush of the admission, or
            # anything else raised after `provider.start`, left a child driving
            # toward a gate that nobody would open, an admission with no ending,
            # and nothing in `_runs` for a delete to reach. The ending is shielded,
            # because a cancel is one of the ways out. The refusal of a spawn is
            # the answer to its caller. A readmission has no caller, so its parent
            # is told here, on the one ending that the child gets.
            why = (
                str(stopped)
                if isinstance(stopped, SubagentSpawnError)
                else f"subagent {run.name} was not admitted: {reason_of(stopped)}"
            )
            with releasing():
                await self._abandon(run, why, tell_parent=not record)
            raise
        self._runs[run.id] = run
        return run

    def _settle_waiter(self, run: SubagentRun) -> None:
        """Make the result that a waiter is handed agree with the log of the child
        itself, for a child that owes a value. This holds whichever provider ran
        the child.

        `record_ended` may have ended as `error` a child that its provider
        called `done`, and the value is the one that the log holds. So `task`
        answers from the log, as its reconcile does after a crash. A provider
        that relies on `record_ended` gets the right result with no code of its
        own."""
        awaited = run.result
        if run.output is None or awaited is None:
            return

        async def settled() -> SubagentResult:
            result = await awaited()
            state = self.state(run.session_id)
            if result.status != "done" or state is None:
                return result
            if state.status == "error":
                return replace(result, status="error", value=None, error=state.detail)
            if result.value is None and state.value is not None:
                return replace(result, value=thaw_json(state.value))
            return result

        run.result = settled

    def _give_output(self, run: SubagentRun) -> None:
        """Give a child that owes a value the way to return it: `submit_result`, and a
        note in its prompt which says that it must.

        **The tool is registered on the scope of the child itself, on purpose.**
        `Grant.apply` narrows only by restriction, because the filter of a scope
        cannot mask a registration of that same scope. A return channel needs
        exactly that. The child has the tool whatever its grant names, and no
        other agent has it. `held_by` keeps it from the children of the child
        too. The tool grants no reach. It is only a way to answer.
        """
        if run.output is None:
            return
        tools = self.ctx.get(TOOLS)
        if run.scope is None or tools is None:
            raise SubagentSpawnError(
                f'the "{run.owner or "subagent"}" provider gives its child no scope of its '
                "own, so the child cannot be given the tool it returns a value through"
            )
        tools.register(_result_tool(run.output), scope=run.scope)
        prompt = self.ctx.get(SYSTEM_PROMPT)
        if prompt is not None:
            note = PromptSection(
                name="subagent:output", text=_output_note(run.output), order=ORDER_OUTPUT
            )
            prompt.section(note, scope=run.scope)

    def _enforce(
        self,
        grant: Grant,
        run: SubagentRun,
        held: tuple[tuple[str, ...], tuple[str, ...]],
        boundary: Context,
    ) -> None:
        """Bound the child, or refuse the spawn if this provider cannot be bounded.

        The check fails closed, and narrowly. A provider that does not hand back
        a scope is unbounded. That is only *unsafe* when the grant actually
        narrows something. So a deployment in which nothing is restricted keeps
        working with any provider. When a spawn means to narrow, a provider that
        cannot deliver the narrowing is refused and not silently ignored.

        **The containment check has no off switch** (P6-31).
        `_delegating_boundary` resolves `boundary`, and it refuses where it
        would otherwise yield `None`. So `boundary` is always a `Context` here,
        and the check always runs.
        """
        if run.scope is not None:
            # **The scope that a provider builds must be inside the scope of the
            # parent** (P6-27). Containment is now the isolation chain, so a
            # scope that is built anywhere else silently opts out of it. It opts
            # out *invisibly*: the child still gets the admission grant, so it
            # still passes every ceiling assertion. It loses only the inherited
            # narrowing. The check is at the one place that every delegation
            # passes through, for the reason that `check_grant` gives: a ceiling
            # that one provider forgot would not be a ceiling. A provider that
            # predates nesting, or a future provider that forgets `parent=`,
            # fails here. Without this check it would fail nowhere.
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
        resolved = self.resolve_preset(await self.resolve_profile(request))
        request = self.resolve_output(self.resolve_model(resolved))
        parent = request.parent.session
        if parent is not None:
            # Read every child that the parent has, on disk as well as in memory,
            # before anything counts them: a spawn cap, or the names that its
            # children have taken. The read happens one time for each mount
            # (`load_children`). A parent that was resumed through the sweep
            # already has it.
            await self.load_children(parent.id, parent.header.family)
        # The guards run first. A refusal here has nothing to unwind, which is the
        # contract that `SubagentSpawnError` states. Each guard runs bound to the
        # row that registered it, as every body that a registry invokes does
        # (P6-29).
        for guard in list(self._guards):
            with running(guard.by):
                reason = guard.check(request)
            if reason is not None:
                raise _refused(reason)
        child_name = self._named(request, parent)
        request = replace(request, name=child_name)
        if parent is not None:
            # The spawn is in progress from here, with no await since its guards
            # (`child_counts`).
            self._spawning.setdefault(parent.id, set()).add(child_name)
        try:
            # The boundary is resolved one time and then passed down. The
            # ceiling, the brief and the containment check must be answers to the
            # *same* boundary, and one resolution makes that true by
            # construction. The `held` argument one line down exists for the
            # same reason.
            boundary = self._delegating_boundary(request)
            held = self.held_by(request, boundary)
            self.check_grant(request, held)
            grant = self.grant_for(request, held, boundary=boundary)
            entry = self.require(name)
            # The provider runs as the row that registered it (P6-29). The
            # `start` of a provider is row code that this registry invokes, in
            # the same category as the `execute` of a tool. It used to run
            # unbound, so anything that it registered landed on the seam and
            # outlived its row. The layer is the layer of the registration, for
            # the reason that `CompactionSeam.engine_by` states. The target is in
            # hand here (`request.parent`), but reading it means another copy of
            # the `getattr(agent, "ctx", None)` of P6-24. The containment of the
            # child itself is the subject of `Grant`, and not of this binding.
            try:
                with running(entry.by):
                    run = await entry.provider.start(request)
            except InactiveScopeError as gone:
                # **A spawn is refused when the scope of its parent was disposed
                # under it**, whichever provider was building it. Every
                # registration that a child needs (its workspace, its release,
                # its job) is on a scope inside the scope of the parent. The
                # provider releases what it built and lets this error through.
                # The caller is owed the answer of the spawn itself, with a
                # code, and not a raw scope error.
                raise parent_went_away(child_name) from gone
            # The seam gives the name, and stamps it as `_admit` stamps the owner.
            run.name = child_name
            admitted = await self._admit(
                run, request, record=True, owner=name, grant=grant, held=held, boundary=boundary
            )
        finally:
            self._landed(parent, child_name)
        # Record what was read into the prompt of the child, now that the child
        # has one (S8).
        grant.record_brief(request.parent.session)
        admitted.ready.set()
        return admitted

    def children(self, parent_id: str) -> dict[str, ChildState]:
        """The children of one parent by run id, as their own logs describe them
        (Phase 11).

        Every consumer that has a `ctx` should use this read. The state of a
        child that this process runs comes from its own cached fold. The state
        of any other child comes from what `load_children` read from the store,
        or from what its session held when it was let go. The state of a live
        child takes precedence over a stored copy of it. A log with no admission
        is not a child. The children are in admission order.
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
        """Everything beneath `parent_id`: its children, their children, and so on.
        Each child comes before its own children, and siblings are in admission
        order.

        This is the one walk of a delegation tree. Its readers are the spend of
        a goal, the panel of the daemon, the check of what holds a root, and the
        list of what waits for a credential. The walk is cycle-safe, for the
        reason that `descendants` gives (`ph.persistence.families`): a log that
        claims an ancestor as its child costs a wasted lookup, and not a hang.
        The walk reaches every level after the root of the tree was read
        (`load_children`). The tree of a resumed session is read as the session
        opens.
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
        """`children`, after a read of everything beneath `parent_id` on disk that
        this process is not running. The read covers **the whole tree, not one
        level**.

        One read of the store (`SessionArchive.descendants_of`) finds every
        level, because a tree is filed in one family under the id of its root.
        Each state is filed under the parent that its own header names, and
        every parent in the tree counts as read. So a restart sees a grandchild
        beneath a child that it did not readmit. A read of one level at a time
        left that grandchild unread, and so uncounted, unlisted and never ended.

        The read happens one time for each tree in each mount. A stored child
        changes in two cases only. Either this process opens it, and then it is
        live. Or its session is let go, and that hands its state back
        (`_let_go`). The tree of a resumed session is read as the session opens
        (`open_session`). The sweep, a spawn, a revocation and the reconcile of
        a call that a crash cut short ask again, and a repeated call costs only
        a lookup. The read runs on a worker thread, and it reads only the
        records that the state of a child is folded from (`CHILD_EVENT_TYPES`).
        The log of a child is mostly streamed chunks, and a mount runs on the
        event loop that every root of a daemon shares.

        `family` is the family of the parent (`SessionHeader.family`), and it is
        where the store looks. It is the family of every descendant too, because
        a tree is filed with its root.
        """
        if parent_id not in self._stored:
            store = self.ctx.get(SESSION_PERSISTENCE)
            sessions = self.ctx.get(SESSIONS)
            live = {one.id for one in sessions.list()} if sessions is not None else set()
            tree = (
                StoredTree()
                if store is None
                else await anyio.to_thread.run_sync(
                    lambda: stored_tree(store, parent_id, family, live)
                )
            )
            self._stored.setdefault(parent_id, {})
            for owner, states in tree.states.items():
                # What is already known of a child takes precedence over this
                # read of it.
                self._stored[owner] = states | self._stored.get(owner, {})
            for child, owner in tree.unreadable.items():
                self._unreadable.setdefault(owner, set()).add(child)
        return self.children(parent_id)

    def unreadable(self, root_id: str) -> tuple[str, ...]:
        """The children beneath `root_id`, at every level, whose logs cannot be read
        (P13-09). The live-roots index carries them as `damaged`. The result is
        sorted, so an unchanged answer compares equal. The logs are read again
        when the tree is read again, which is at the next mount of the root. So
        a repair shows then."""
        if not self._unreadable:
            return ()
        owners = [root_id, *(state.session_id for state in self.family(root_id))]
        return tuple(sorted(one for owner in owners for one in self._unreadable.get(owner, ())))

    def _isolate(self, state: ChildState, damaged: LogDamaged) -> None:
        """Set aside a child whose log the strict read refused (P13-09).

        The tree read folds only the records that the state of a child is made
        of. So damage in a record of another type passes the tree read, and a
        readmission finds it when it opens the whole log. To end the child at
        that point would write into a file that cannot be read. To leave the
        child live would hold its root permanently. So the child is filed with
        the refusals of the tree read itself, and it is taken out of the
        children of its parent.
        """
        log.warning(
            "ph.seams.subagents: child %s will not read and is left alone: %s",
            state.session_id,
            damaged,
        )
        self._stored.get(state.parent_id, {}).pop(state.session_id, None)
        self._unreadable.setdefault(state.parent_id, set()).add(state.session_id)

    def delegated_tokens(self, parent_id: str, goal_id: str | None = None) -> int:
        """What the children of `parent_id` spent, with everything beneath them, read
        from their own logs. When `goal_id` is named, only the children that
        were admitted under that goal count.

        **Every level counts.** A goal is a budget for a delegation tree, and
        the answers of a grandchild are part of what the goal asked for. The
        roster that this replaces charged one level only: the answers of a child
        itself, mirrored into the log of its parent. So the spend of a
        grandchild, and the compactions of every child, reached no goal at all.
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
        """A session was let go. Keep the last state of a child as its stored copy.

        That is the state that its log now holds, and nothing changes it until
        this process opens the log again. The copy is kept only for a parent
        whose stored children were read. For any other parent, the store answers
        when its children are read. The fold of the child goes with the session
        after this function has read it (`SessionTable`)."""
        parent_id = session.header.delegating_parent
        if parent_id:
            live = self._live.get(parent_id)
            if live is not None:
                live.discard(session.id)
                # The entry goes with its last child. Otherwise the table kept
                # one empty set for each parent that ever delegated.
                if not live:
                    del self._live[parent_id]
            stored = self._stored.get(parent_id)
            state = self._folds.read(session)
            if stored is not None and state.admitted:
                stored[session.id] = state

    def forget_session(self, session_id: str) -> None:
        """Drop the stored tree that this service read for one session.
        `load_children` filed every level of it. The fold of the session itself
        goes with the session."""
        for owner in [*(state.session_id for state in self.family(session_id)), session_id]:
            self._stored.pop(owner, None)
            self._unreadable.pop(owner, None)

    def name_of(self, agent_id: str) -> str:
        """What an agent is called: the name that its own admission records, or else
        its id."""
        state = self.state(agent_id)
        return (state.name or agent_id) if state is not None else agent_id

    def get(self, run_id: str) -> SubagentRun | None:
        return self._runs.get(run_id)

    def waiter(self, parent_id: str, run_id: str) -> SubagentAwaiter:
        """How a parent waits for one of its own children.

        For a live run it is the waiter of that run. For a child that this
        process no longer holds, it is the ending that the log of the child
        records, and the answer is then the preview that the ending kept. A tool
        that waits for a child by its run id uses this one reading, so the
        ownership check and the account of the log cannot differ between tools.

        :raises LookupError: for a child that is not a child of the parent, or
            for a child that is still working with nothing here to wait on.
        """
        state = self.children(parent_id).get(run_id)
        if state is None:
            raise LookupError(f"you have no child {run_id!r}")
        live = self._runs.get(run_id)
        if live is not None and live.result is not None:
            return live.result
        if child_is_live(state):
            raise LookupError(f"{state.name} is still working, and cannot be waited on")

        async def logged() -> SubagentResult:
            return SubagentResult(
                status=state.status,
                answer=state.answer_preview or "",
                error=state.detail,
                value=state.value,
            )

        return logged

    def list(self, *, parent_id: str | None = None) -> list[SubagentRun]:
        """The live runs, or only those of one parent, in admission order."""
        runs = list(self._runs.values())
        if parent_id is None:
            return runs
        return [run for run in runs if run.parent_id == parent_id]

    async def ensure_addressable(self, session_id: str) -> bool:
        """Make the agent behind `session_id` reachable, and wake it if it settled.

        This is the one place that states the path from a session id to a run,
        to a provider, to a wake. A caller that wants to address an agent asks
        this function and does not compose the three hops itself. A second
        caller that composes them forgets the refusal of a revoked child.
        """
        if self.ctx.require(AGENTS).get(session_id) is not None:
            return True
        run = next((one for one in self._runs.values() if one.session_id == session_id), None)
        return bool(run is not None and await self.rehydrate(run.id))

    async def resume_children(self, parent: AgentDriver, *, retry_limit: int) -> Sequence[str]:
        """Give a resumed root what it owes its unfinished children (P5-04). It
        returns the run ids of the children that run again.

        There are two states, and they need opposite answers. That is why this
        is not one sweep over "everything unsettled":

        - **queued**: the child was admitted and never ran. It is driven again,
          because nothing was claimed, spent or written, and the work is the
          same work.
        - **running**: the child was interrupted in the middle of a turn. It is
          put back on the *ladder*. Its turn is closed on resume and its task is
          presented again, up to `retry_limit` times. After that it is failed,
          and its ending says so.

        **The ladder is what makes it sound to run an interrupted child again.**
        This row first shipped without it. A child that started a turn has
        *claimed* its task from its inbox. So a plain second drive finds nothing
        pending and ends at step zero reporting `completed`. That is the empty
        turn that P5-04 found at the root, and the parent is told that its child
        succeeded when it did not. Presenting the task again makes the next
        attempt a real one, and the count stops the attempts from being
        infinite.

        To do nothing was never a third option. `child_is_live` counts an
        unsettled child. So a child that nothing will ever move keeps its whole
        root out of passivation for the life of the process, while the parent
        waits for a reply that nobody is writing.

        **Every decision is written in the log that it is about** (Phase 11).
        The ladder reads the starts and the answers of a child from the log of
        the child itself, which is where they are. So nothing has to be caught
        up first. A child that the sweep gives up on is ended in its own log. A
        child that the sweep puts back is readmitted, and its drive writes the
        restart on its own disk before the attempt (S10). Nothing orders the
        records of one child against the records of another child or of the
        parent. So a crash in the middle of a sweep leaves each child decided or
        undecided, and the next start decides the undecided children in the same
        way.

        `retry_limit` has **no default**. That is the rule of P6-32, and not an
        inconvenience. How many attempts the work is worth is the policy of the
        host, and a seam that answered for a caller who said nothing would be
        choosing a policy. The daemon states the limit beside the ladder of the
        root itself (`ph_app.daemon.recovery.CHILD_RETRY_LIMIT`), which is where
        somebody who tunes restart behavior already looks. `--mode rpc` and the
        one-shot modes read the same bound.
        """
        session = parent.session
        if session is None:
            return []
        async with self._sweeps.hold(session.id):
            await self.load_children(session.id, session.header.family)
            return await self._resume(parent, retry_limit=retry_limit)

    async def readmit_waiting(self, parent: AgentDriver, *, retry_limit: int) -> Sequence[str]:
        """Put back to work the children of `parent` that were held for a credential
        which has since arrived (T5), and their children. It returns the run ids
        of the children that run again.

        This is the sweep that a resume runs, asked again. A child that still
        lacks its credential stays held and appends nothing. A child whose
        credential is here now has its hold settled and is readmitted. The
        caller is whoever hands a deployment a credential: the
        `credentials/store` of the daemon.

        **Every level is asked** (L5b). A child that is running may hold
        children of its own that wait for the same name. So each live child of
        `parent` is asked too. `retry_limit` is the limit of the host, for the
        sweep that the children of a readmitted child get.
        """
        session = parent.session
        if session is None:
            return []
        async with self._sweeps.hold(session.id):
            await self.load_children(session.id, session.header.family)
            revived = list(await self._resume(parent, retry_limit=retry_limit))
        for state in self.children(session.id).values():
            child = self.ctx.require(AGENTS).get(state.session_id)
            if child is not None:
                revived.extend(await self.readmit_waiting(child, retry_limit=retry_limit))
        return revived

    async def _resume(self, parent: AgentDriver, *, retry_limit: int) -> Sequence[str]:
        """Decide, child by child, what each unfinished child of `parent` becomes:
        ended, held, or running again. It returns the run ids of the children
        that run again.

        **Only the children that this process is not running are decided**: a
        `queued` child, or a `running` child that was interrupted. The decision
        for each child is written in its own log:

        - **Nothing can readmit it.** The child is ended. A child that stayed
          live would hold its parent out of passivation for a reply that nobody
          is writing.
        - **It was interrupted, and its ladder is spent.** The child is ended,
          and the ending says how many times. Presenting the task again makes a
          restart real, and the count stops the restarts from being infinite.
          An answer forgives the restarts before it
          (`restarts_since_progress`).
        - **Its route names a credential that this deployment cannot supply**
          (T5). The child is held and not readmitted. A readmitted child would
          spend its ladder failing on a key that a person could supply in a
          second. The child stays live, no start is counted, and its own log
          says which name it waits for. `readmit_waiting` asks again.
        - **Otherwise the child is readmitted.** Its ceiling is derived again
          from its admission (`check_grant`, `grant_for`, `_enforce`). So a
          child comes back from a power cut no wider than it was admitted
          (§6.5). The spawn guards do not run. They gate new work, and a cap
          would count the child against itself. The children of the child are
          swept before its first step (L5b, `_sweep_readmitted`).

        One child that cannot be rebuilt is logged and ended, and it does not
        fail the sweep (K4). A root that is coming back must not be blocked by
        the least recoverable thing that it holds. **A provider that declines is
        an ending too** (L6).

        **The notice of an ended child reaches its parent** (`ChildNotice`). A
        notice that the log of the parent lacks is delivered now. The log lacks
        it when a crash came between the ending of the child and the write of
        the parent. A notice that the log has is not delivered again.

        **The unfinished descendants of an ended child are revoked**
        (`_revoke_beneath`). A child that is given up on here takes them with it
        as its ending is written (`record_ended`). A child that ended in an
        earlier process left them behind if a crash came before they were
        revoked. The descendants of every ended child are revoked in one call,
        before the rest of the sweep.
        """
        session = parent.session
        if session is None:
            return []
        revived: list[str] = []
        delivered: set[str] | None = None
        children = self.children(session.id)
        # The subtrees of these children are disjoint from one another and from
        # the subtree of every live child. So nothing below has to be ordered
        # against them.
        await self._revoke_beneath(
            *(
                state.session_id
                for state in children.values()
                if state.run_id not in self._runs and not child_is_live(state)
            )
        )
        for state in children.values():
            if state.run_id in self._runs:
                continue
            if not child_is_live(state):
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
                    tell_parent=True,
                )
                continue
            try:
                request = self._request_of(parent, state)
                if await self._held_for_credential(state, request):
                    continue
                run = await self._readmit_one(state, request, readmitter)
            except LogDamaged as damaged:
                self._isolate(state, damaged)
                continue
            except Exception as error:
                log.exception("ph.seams.subagents: %s could not be readmitted", state.run_id)
                await self._end_child(
                    state.session_id,
                    f"this child could not be resumed: {error}",
                    tell_parent=True,
                )
                continue
            if run is None:
                await self._end_child(
                    state.session_id,
                    "the provider that owns this child could not resume it",
                    tell_parent=True,
                )
                continue
            await self._sweep_readmitted(run, retry_limit=retry_limit)
            revived.append(state.run_id)
        return revived

    async def _revoke_beneath(self, *session_ids: str) -> None:
        """Tombstone every unfinished descendant of each ended child in `session_ids`
        that this process is not running. Each tombstone is written in the log
        of that descendant, with the reason `PARENT_TEARDOWN`.

        **A running descendant belongs to its provider.** The children of a
        child are effects of its scope, and they are revoked as that scope
        unwinds (I2). That teardown never reaches a child that this process is
        not running: a child that is held for a credential, a child that was
        left beneath a child that a restart readmitted, or a child that a delete
        reached on disk. It also never reaches what a crash left behind between
        the ending of a child and the endings of its children. Such a
        descendant, left alone, reads as working permanently, because nothing
        readmits a child beneath a child that ended. It held its root out of
        passivation, and it asked for a credential that it would never use. One
        walk covers every level of every subtree (`family`). The walk is best
        effort.

        **The logs are written concurrently, for every subtree together.** A
        resume hands over every ended child in one call. Each log is opened
        through `stored_session`, which settles what a read of the log owes
        before the write (`SessionStore.loaded`): a spill sweep and a workspace
        reconcile, which runs `git` for each leaked tree. When the logs were
        opened one at a time, a restart paid that cost back to back for every
        descendant. Nothing orders the tombstone of one log against the
        tombstone of another. A crash in the middle leaves each descendant
        revoked or not revoked, and the next start finishes the rest in the
        same way.
        """
        pending = [
            state
            for session_id in session_ids
            for state in self.family(session_id)
            if child_is_live(state) and state.run_id not in self._runs
        ]

        limiter = anyio.CapacityLimiter(REVOKING_AT_ONCE)

        async def revoke(state: ChildState) -> None:
            try:
                async with limiter:
                    await self._write_child(
                        state.session_id,
                        lambda child: _tombstone(self.ctx, child, PARENT_TEARDOWN),
                    )
            except Exception:
                log.exception(
                    "ph.seams.subagents: %s could not be revoked beneath its ended parent",
                    state.session_id,
                )

        async with anyio.create_task_group() as group:
            for state in pending:
                group.start_soon(revoke, state)

    async def _sweep_readmitted(self, run: SubagentRun, *, retry_limit: int) -> None:
        """Sweep the children of a readmitted child, as the children of its parent
        were swept (L5b), and then open its gate (`SubagentRun.ready`).

        A failure is logged and not raised. The child itself is readmitted, and
        one level that cannot be swept must not undo the level above it. The
        gate opens whatever happened, because the child was bounded before this
        ran.
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
        """The request that a child was admitted with, rebuilt from its own
        admission. Its route is worked out again from what the spawn asked for
        (`resolve_model`)."""
        admission = state.admission or Admission(run_id=state.run_id)
        request = SubagentRequest(
            prompt=admission.prompt,
            parent=parent,
            name=admission.name or None,
            choice=admission.choice or ModelChoice(),
            reasoning_effort=admission.reasoning_effort,
            output=admission.output,
            access=admission.requested_access,
            preset=admission.preset,
            skills=admission.skills,
            tools=admission.tools,
            paths=admission.paths,
            call_id=admission.call_id,
        )
        return self.resolve_model(self.resolve_preset(request), readmitting=True)

    async def _held_for_credential(self, state: ChildState, request: SubagentRequest) -> bool:
        """Whether this child waits for a credential. The function also makes the log
        of the child say so.

        The route is asked first, and nothing is opened for that. A child whose
        log already says what is true (it waits for this name, or for nothing)
        needs nothing written, and most readmitted children are in that case.
        The log of the child is opened only to record a hold that starts or
        ends. A held child that was interrupted in the middle of a turn is
        marked `queued` beside the hold, because it is not running.
        """
        options = child_options(request)
        route = (options.provider, options.model)
        name = missing_credential(self.ctx, *route)
        if name == state.awaiting:
            return name is not None

        async def record(child: Session) -> None:
            await hold_for_credential(self.ctx, child, SESSION_HOLDER, *route)
            if name is not None and state.status == "running":
                record_waiting(child, detail=INTERRUPTED_DETAIL)

        await self._write_child(state.session_id, record)
        return name is not None

    async def _readmit_one(
        self, state: ChildState, request: SubagentRequest, readmitter: _Readmitter
    ) -> SubagentRun | None:
        """Readmit one child, through the admission path that it first took."""
        boundary = self._delegating_boundary(request)
        held = self.held_by(request, boundary)
        self.check_grant(request, held)
        grant = self.grant_for(request, held, boundary=boundary)
        with running(readmitter.by):
            run = await readmitter.provider.readmit(
                request,
                run_id=state.run_id,
                session_id=state.session_id,
                # How many times this child has *already* been started. That
                # count tells a restart from a first run. It is `starts` and
                # never `restarts_since_progress`, because an answer forgives
                # the ladder. With the other count, a child that made progress
                # and was then stopped again would be readmitted as though it
                # had never run. Its restart would go unrecorded, and the ladder
                # would never count it again.
                restarts=state.starts,
            )
        if run is None:
            return None
        admitted = await self._admit(
            run,
            request,
            record=False,
            owner=readmitter.name,
            grant=grant,
            held=held,
            boundary=boundary,
        )
        grant.record_brief(request.parent.session)
        return admitted

    def _readmitter(self, state: ChildState) -> _Readmitter | None:
        """The provider that could put this child back, or `None` if none can.

        The provider is found by the `owner` that the admission of the child
        records. So it is the provider that ran the child, and not whichever
        provider happens to be the only one mounted.
        """
        name = self.resolve(state.owner or None)
        entry = self._providers.get(name or "")
        if name is None or entry is None or not isinstance(entry.provider, ReadmittingProvider):
            return None
        return _Readmitter(name=name, by=entry.by, provider=entry.provider)

    async def delete(self, parent: Session, run_id: str, *, reason: str) -> bool:
        """Revoke one child of `parent`, with a tombstone in the log of the child
        itself. Its transcript stays on disk. The result is `False` when there
        is no such child, or when the child was already revoked.

        The child gets a tombstone and is not removed, because its log and its
        artifacts outlive it. A parent that looks for what a revoked child did
        should find the revocation, and not a gap. **Any child can be revoked,
        live or not** (Phase 11). A child that this process runs is stopped by
        its provider (`RevokingProvider`), which writes the tombstone as it lets
        go. The seam tombstones any other child here: a child that this process
        is not running, such as one that an earlier process settled, or a child
        whose provider cannot revoke it. A revocation used to reach only the
        children that this process held in memory. In both cases, what the child
        left unfinished beneath it goes with it (`record_deleted`).
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

        The agent of a child that finished was released, so the child has no
        inbox to steer into. Its session and its log are both still there. In a
        rehydration, the provider attaches a runtime to that state again. A
        provider that cannot do this says so by not implementing the method. The
        caller then gets `False`, and not an exception that it has to interpret.
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
    """A named kind of child that a deployment is willing to spawn.

    **A preset has no prompt field, on purpose.** The obvious design gives a
    preset its own standing instructions. A directing skill and a preset are
    then two channels that say what a child is for. They compete where they
    disagree, and they are duplicated where they agree. The body of a skill
    already *is* a standing instruction (P4-13b). So a preset binds a name to
    the capability and lets the skill do the directing. For example, `reviewer`
    is `skills: [code-review]`, and what a reviewer does is written one time, in
    the skill, where a human edits it.
    """

    skills: tuple[str, ...] | None = None
    tools: tuple[str, ...] | None = None


class PresetConfig(WireModel):
    """Row config for `subagent-presets`."""

    presets: dict[str, SubagentPreset] = Field(default_factory=dict)
    """The presets, which the deployment names and a parent selects.

    A preset is a **menu, never a grant**. A spawn that names a preset whose
    entries the parent does not hold is refused like any other spawn. A preset
    that widened whatever selected it would put the escalation one indirection
    away, and under the control of the model (P4-13b). Presets exist so that a
    deployment writes "what a reviewer is" one time. They do not exist so that
    a deployment can hand out more than the parent has.
    """


@dataclass(slots=True)
class SubagentPresetService:
    """The service published as `ctx.subagent_presets`.

    The table is filled from config only, and it has no
    `register(..., scope=)`. It is the one table under `ph.seams` without one,
    and that is on purpose. A preset is what a *deployment* is willing to
    spawn. A package that could ship a preset would widen what a profile allows
    without the profile saying so. A package ships the skill, and a profile
    decides which children may hold it.
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


SHIPPED_SKILLS = Path(__file__).parent.parent / "shipped_skills"
"""The skills that ph-core ships for handing to a child: `categorize`, which asks
for the `classify` kind and owes a checked value (A2)."""


def offer_shipped_skills(ctx: Context) -> None:
    """Install the shipped delegation skills, if a skills registry is mounted.

    A row that registers a provider calls this, because a provider is what
    starts a child. In `ph-rlm` that row is `rlm-subagent-provider`, which
    registers `rlm-child`. The row of this seam does not call it. A skill for
    handing to a child is noise in a profile that cannot start a child. Only the
    one-line description of the skill is added to the prompt (G9).
    """
    offer_skills(ctx, SHIPPED_SKILLS, source="subagents")


@plugin("subagents", affects="environment", inject=[SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Mount the subagent seam definition. No provider ships in ph-base."""
    service = SubagentService(ctx=ctx)
    ctx.provide(SUBAGENTS, service)
    # A child that is let go keeps its last state as its stored copy. Its fold
    # goes with the session, after this listener has read it
    # (`SessionStore.dispose`).
    ctx.on("session/created", service._published)
    ctx.on("session/disposed", service._let_go)
    ctx.on("agent/turn-stopping", _remind_owed)
    # The state of a child is what the prompt tells its parent about it, and
    # what the interruption ladder counts starts against. With a drifted state,
    # a parent reasons about a child that the log of the child does not
    # describe.
    contribute_fold_cache(
        ctx, id="subagent-fold-cache", subject="subagent state", stale=service._folds.stale
    )


@dataclass(frozen=True, slots=True)
class Grant:
    """What one child may reach, resolved to names and fixed at admission.

    **The capability of a child is fixed at the moment that it is admitted.** A
    parent that gains a tool afterwards does not widen a child that is already
    running. The child was spawned for a job, with a ceiling that its prompt
    and its brief were written against. If the ceiling grew silently while the
    child ran, "what could this child do" could not be answered from the
    admission record. A parent that needs a child with more spawns a *new*
    child, whose admission says so.

    That is why the allow-list stays although the isolation chain would bound
    the child anyway. The chain answers "no more than the parent **holds**",
    and the allow-list answers "no more than the parent held **then**". Only
    the second answer is stable enough to read a transcript against.

    `brief` is rendered here and not read at each assembly. It is the cached
    prefix of the prompt, and a `PromptSection` that reads the filesystem on
    every model step is neither static nor free.
    """

    skills: tuple[str, ...]
    tools: tuple[str, ...]
    paths: tuple[str, ...] | None = None
    """The extra writable directories, where an assigned profile narrowed them."""
    brief: str = ""
    briefed: tuple[tuple[Skill, str], ...] = ()
    """The skills that `brief` holds, and exactly what was read of each. The reads
    are recorded as `skill/read` (`via="brief"`) when the child is admitted
    (`record_brief`), and not when the skills are read. A spawn that was
    refused before the admission put nothing in the prompt of any child."""

    def record_brief(self, session: Session | None) -> None:
        """Record in `session` what this brief put in the prompt of the child, for
        each skill."""
        for skill, text in self.briefed:
            record_read(session, skill, text, via="brief")

    def apply(self, ctx: Context, scope: Context) -> None:
        """Bound the scope of a child to this grant.

        **Narrowing is by restriction, never by registration.** The filter of a
        scope cannot mask a registration of that same scope. A filter reaches
        everything *outside* the scope that wrote it, and nothing inside. So a
        registration on a child is a way to hand the child something that its
        parent cannot see, which is the opposite of a ceiling. Filters only
        intersect, so they are the only instrument that a spawn may use.

        The registration of an *ancestor* can be masked (P6-27). It must be, or
        a child could never be narrowed below a parent that registered on its
        own scope. What still holds is that a scope cannot filter itself.
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


def _brief(
    skills: SkillService, named: Sequence[str], scope: Context
) -> tuple[str, tuple[tuple[Skill, str], ...]]:
    """The instructions of the named skills, read one time, and what was read of
    each.

    **A named skill is direction, not a lookup.** G9 keeps bodies out of the
    prompt, because a catalog of twenty skills is twenty bodies that the model
    probably does not need. A child that was spawned *for* one skill certainly
    needs that one. A child that must spend a turn fetching what it was created
    to do wastes a call, and there is a real chance that it never fetches the
    body at all. So the named bodies go in its prompt, and everything else that
    it can still reach stays in the catalog only.
    """
    parts: list[str] = []
    read: list[tuple[Skill, str]] = []
    for name in named:
        skill, body = skills.get(name, scope), skills.body(name, scope)
        if skill is not None and body:
            parts.append(f"## {name}\n\n{body.strip()}")
            read.append((skill, body))
    if not parts:
        return "", ()
    return (
        "You were delegated this task to follow the instructions below. "
        "They are not background reading.\n\n" + "\n\n".join(parts)
    ), tuple(read)


_UNADMITTED = Admission(run_id="")
"""What a `ChildState` with no admission reads as. No reader hands it out."""

_STATUSES: Mapping[str, SubagentStatus] = literal_lookup(SubagentStatus)
_CAUSES: Mapping[str, StatusCause] = literal_lookup(StatusCause)


@dataclass(frozen=True, slots=True)
class ChildState:
    """One child, as its own log describes it (Phase 11).

    The state is the fold of the log of the child itself, and of nothing else:
    its admission, its statuses, its tombstone, the value that it returned, the
    records that spend tokens (an answer is one of them) and its credential
    hold. No parent keeps a copy, so there is no second account of the child
    that could disagree with this one. Every rule that used to keep the roster
    of a parent in step with the log of its child (S2, S10, F1, L5) is now a
    rule about one log.

    `admission` is `None` for a log that has no admission, or whose admission
    does not parse as an `Admission`. Such a log is not a child. A workspace
    can reach the disk before the admission does, and a spawn that was refused
    before it was admitted leaves such a log behind. Every reader skips it. The
    properties below read the record of an admitted child.

    `starts`, `resumes` and `resumes_at_last_answer` are the whole state of the
    interruption ladder. All three are folded, and all three are totals. The
    log already records one `running` for each drive and one
    `assistant/message` for each answer. So "how many times has this been
    started" and "how many restarts had there been at its last answer" are
    questions about records that are already there. What the ladder *makes* of
    the totals is `restarts_since_progress`, beside the sweep that decides
    (P3).
    """

    session_id: str
    parent_id: str
    family: str | None = None
    """The family directory that its log is filed in, as its own header says. A
    write to a child that this process is not running reads that path and does
    not search the store for the log (`SubagentService._write_child`). `None`
    for a log with no header."""
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
    """What it spent, by `TokenUsage.total`. The spend is the sum over the records
    in `SPENDING_TYPES`: its answers, its compactions, and the questions that a
    decision model answered."""
    value: JsonObject | None = None
    """What it returned, for a child with an `output` (`record_returned`), or `None`."""

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
        """The admission facts as the handle that a spawn returned. It is for a caller
        that answers from the log what the live path answers from the handle.
        The `reconcile` of a delegating tool builds from it the value that its
        `execute` would have returned."""
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
            model_fallback=record.model_fallback,
            media_shortfall=record.media_shortfall,
            effort_unsent=record.effort_unsent,
        )

    def to_wire(self) -> dict[str, JsonValue]:
        """The child as a row: its admission, where it is, and its current state. The
        roster tool of the model and the panel of a front end read it."""
        admission = thaw_json(self.admission.to_wire()) if self.admission is not None else {}
        assert isinstance(admission, dict)
        # The reader of a row asked for the schema and already holds it. The
        # value is what is new.
        admission.pop("output", None)
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
            "value": thaw_json(self.value) if self.value is not None else None,
        }
        row.update({key: value for key, value in optional.items() if value is not None})
        if self.deleted:
            row["deleted"] = True
        return row


def fold_child_event(state: ChildState, event: SessionEvent) -> ChildState:
    """Fold one event of the log of a child into the state of that child. The
    rules are in this one place.

    **Each field is set by its own record, and replaced whole.** A status
    carries its `cause` and `detail`, or it carries none. So a later status
    never keeps the reason of an earlier one. The roster that this replaces
    merged each status into its row, and a child that was woken after a restart
    went on saying `resumed`. A status that this fold does not know leaves the
    previous status in place. So a child is never read as settled because of a
    record that nobody can read, which is the direction that `child_is_live`
    keeps.

    An `assistant/message` is an answer, and it forgives the restarts before
    it. An answer and the other records in `SPENDING_TYPES` are what the child
    spent.
    """
    kind = event.type
    # The type test comes first, because the log of a child is mostly
    # `assistant/chunk`.
    if kind not in CHILD_EVENT_TYPES:
        return state
    data = event.data
    if kind == ADMITTED:
        admission = _admission_of(event)
        if admission is None:
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
    if kind == RETURNED:
        return replace(state, value=as_obj(data.get("value")))
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
    """The notice of an ending. `None` when the ending carries no notice, or one
    that this build cannot read."""
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
    """`child_state` over a header and events, and not over a session. It is for a
    log that was read from the store. A resume holds such a log before any
    `Session` is built for it."""
    parent = header.delegating_parent if header is not None else None
    state = ChildState(
        session_id=session_id,
        parent_id=parent or "",
        family=header.family if header is not None else None,
    )
    for event in events:
        state = fold_child_event(state, event)
    return state


def extend_child_state(previous: ChildState, log: Session, start: int) -> ChildState:
    """`child_state` continued from a prefix. The cache of `SubagentService` folds
    a new slice of the log with it. Extending the fold of a prefix gives the
    same state as folding the whole log."""
    state = previous
    for event in log.events_from(start):
        state = fold_child_event(state, event)
    return state


@dataclass(frozen=True, slots=True)
class StoredTree:
    """What `stored_tree` read of one family."""

    states: dict[str, dict[str, ChildState]] = field(default_factory=dict)
    """The children of each session by session id, filed under the parent that the
    header of each child names. Every session in the tree has an entry, empty
    or not, because the children of each one were in this read."""
    unreadable: dict[str, str] = field(default_factory=dict)
    """The logs that the read refused, by session id. Each is stored with the
    parent that its listing row names."""


def stored_tree(store: SessionArchive, parent_id: str, family: str, live: set[str]) -> StoredTree:
    """Everything beneath `parent_id` on disk that this process is not running,
    each folded from its stored log. The function blocks. It runs on a worker
    thread (`load_children`), or on the thread of a rebuild
    (`ph.seams.live_index`), which asks the same question of the same family.

    **One log that cannot be read costs its own row and not the tree**
    (P13-09). The read used to raise through this function, so one damaged
    child file refused its root. The root could not be mounted, its siblings
    were never readmitted, and nothing could repair the file because nothing
    could open the family. Now the child is reported as `unreadable`, and it is
    among the children of no parent. It is not live, so no sweep waits on it or
    readmits it. It is not ended, so no sweep writes a tombstone into a file
    that it cannot read. `phern doctor sessions` names the line, and its
    `--repair` cuts the log back to what can be read.

    **Only a refusal of the content of the log is caught.** That is a
    `ValueError` from every backend: the `LogDamaged` of the JSONL reader, or a
    row that another backend cannot validate. A log that has gone missing or
    cannot be opened is not damage, and it still raises as before.
    """
    tree = StoredTree()
    for row in store.descendants_of(parent_id, family):
        tree.states.setdefault(row.session_id, {})
        if row.session_id in live:
            continue
        # The log of a child starts at seq 0, so its own file is the whole log.
        # The file is read by its family, which the listing already found, and
        # is not searched for. Only the records that its state is folded from
        # are read.
        try:
            header, events = store.read_own(
                row.session_id, family=row.family, types=CHILD_EVENT_TYPES
            )
        except ValueError as refused:
            log.warning(
                "ph.seams.subagents: child %s of %s will not read and is left alone: %s",
                row.session_id,
                parent_id,
                refused,
            )
            tree.unreadable[row.session_id] = row.delegating_parent or parent_id
            continue
        state = child_state_of(row.session_id, header, events)
        if state.admitted:
            tree.states.setdefault(state.parent_id, {})[row.session_id] = state
    return tree


def _names_parent(child: Session, parent_id: str) -> bool:
    """Whether the log of a child names `parent_id` in the way that the store finds
    the child by.

    The parent link of the header decides (`delegating_parent`). The
    `<parent>-` prefix of the id is how `SessionArchive.descendants_of` narrows
    the family directory before it reads a header. So the listing never reaches
    a child without that prefix.
    """
    return child.header.delegating_parent == parent_id and is_child_id(parent_id, child.id)


def restarts_since_progress(state: ChildState) -> int:
    """How many times a child has been restarted since it last answered (P3).

    **The rule of the ladder is stated where the ladder is decided**, and it is
    not folded into the state. A child that was stopped, worked for an hour,
    and was then stopped again met two incidents, and not the incidents of a
    lifetime. So an answer forgives the restarts before it. The answer is an
    `assistant/message` in the log of the child itself. It is a model reply,
    which a turn that did nothing cannot produce. The state keeps both facts as
    totals (`resumes`, `resumes_at_last_answer`). So a different rule is a
    change here, and not a change to the fold.
    """
    return state.resumes - state.resumes_at_last_answer


FamilyRole: TypeAlias = Literal["self", "parent", "sibling", "child"]
"""How one agent is related to another agent that it can reach."""


class _Parented(Protocol):
    """The id of a session and its parent link, and nothing else.

    This is all that `reachable_family` reads. The docstring of that function
    already says that the sessions are passed in and are not read from a store,
    so that the function answers on a resumed log with no live agents as well
    as in a running process. That is a structural contract described in prose.
    With the contract written as a Protocol, a caller can hand the function a
    family tree that it built for the purpose. The messaging test does exactly
    that. Its two-field double satisfied this Protocol all along, and it failed
    the declared `Session` (issue 32).
    """

    @property
    def id(self) -> str: ...

    @property
    def header(self) -> _HasParent: ...


class _HasParent(Protocol):
    @property
    def delegating_parent(self) -> str | None: ...


def reachable_family(sessions: Iterable[_Parented], agent_id: str) -> dict[str, FamilyRole]:
    """Every agent that `agent_id` may address, mapped to how it is related (C7).

    This is the enumeration half of `family_reach`. It is derived *from*
    `family_reach` and does not restate the rule. The guard that refuses a send
    and the roster that tells the model who it may address must not be able to
    disagree.

    The id of an agent is the id of its session. So the parent link is
    `SessionHeader.delegating_parent`, and nothing needs a side index. That
    link names the agent that spawned it. The `parent_session` of a fork does
    not: a forked root is a root. The sessions are passed in and are not read
    from a store. So this function answers on a resumed log with no live agents
    as well as in a running process.
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

    A sender reaches its parent, its direct children, and the siblings at its
    own depth that share its parent. Roots count as siblings of each other. The
    rule is in the seam and not in the RLM bundle, because it is the reach rule
    for the children of *any* provider. The guard that enforces it (P3-12) must
    not be able to disagree with the roster that displays it.
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
    children that were given the same task. The seam passes eight random hex
    digits.

    A name that the parent can read back matters more than it seems to. The
    roster, the ending notice and every `agent_message` address use it. A child
    called `child-4` tells the parent nothing about which delegation it was.
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
