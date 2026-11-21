"""What a tool is, and what one call produces.

The main decision in this module is that `output` is **mandatory** (P1-01). A
tool returns a canonical JSON *value*. The declared `render` of the tool
projects that value into the content that the model sees. Two consequences
follow, and both are reasons why dsh made `output` mandatory:

* The durable record is the projection of the value. A UI and a replay
  therefore render the same card from the log alone, and nothing depends on the
  live object.
* A tool cannot silently give the model prose that the schema of the tool does
  not describe, because the content is derived from a validated value.

`ToolRunContext` is the view that the body has of its own call. It holds the
accepted execution and the definition that will render the value. It also
offers the two things that a body may do to the turn other than return: defer a
context message, and conclude the turn.

@module ph.tools.definition
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from ..agent.types import AgentHandle
from ..cancel import CancelToken
from ..cordis import Boundary, Context, Running, maybe_await
from ..json import JsonObject, JsonValue, as_obj, as_str, thaw_json
from ..llm.types import ContentBlock, Message, TextBlock, ToolSchema
from ..session import Session, SessionEvent

if TYPE_CHECKING:
    from ..seams.approval import ApprovalDecisionName
from .errors import (
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    TOOL_BUDGET_SPENT,
    TOOL_DENIED,
    TOOL_TURN_CONCLUDED,
    FailureKind,
    ToolOutputError,
)
from .json_schema import parse_arguments, schema_of, validate_json_schema_value
from .presentation import ToolCallView, ToolResultView

__all__ = [
    "Accept",
    "Allow",
    "Ask",
    "Block",
    "Deny",
    "Done",
    "ExecutionMode",
    "FailureKind",
    "NotDone",
    "PostToolDecision",
    "PreToolDecision",
    "Reconciled",
    "Respond",
    "ToolDefinition",
    "ToolExecution",
    "ToolExecutionInput",
    "ToolExecutionResult",
    "ToolFailure",
    "ToolModel",
    "ToolOutput",
    "ToolResult",
    "ToolRunContext",
    "TransportPresentation",
    "Unknown",
    "WorkspaceConfinement",
    "aborted_result",
    "budget_result",
    "concluded_result",
    "define_tool",
    "denied_result",
    "error_result",
    "text_content",
]


class ToolModel(BaseModel):
    """Base for a tool's argument and output schemas — **snake_case on purpose**.

    The one deliberate exception to the camelCase wire rule (Q2). A tool's
    parameter names are not a pH-internal boundary: they are what the *model*
    types. Under Code Mode it writes `await tools.edit(old_text=...)` in Python,
    so a parameter renamed to `oldText` by an alias generator would change the
    tool's public API to satisfy a convention that exists for pH's own frames.

    Declared rather than inferred, so the exemption is visible at every
    declaration site and the wire test can check that nothing else claims it.
    """

    model_config = ConfigDict(extra="forbid")


SchemaDeclaration: TypeAlias = "type[BaseModel] | dict[str, Any]"


def text_content(text: str) -> list[ContentBlock]:
    """The one-block content most tools return."""
    return [TextBlock(text=text)]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """The canonical output declaration of a tool.

    `render` is a pure projection from the arguments and the value to content.
    `presentation_meta` is the private durable payload of the tool. It is
    computed only for top-level calls, because a nested Code Mode dispatch has
    no card of its own.

    **The arguments are not the model that the body received.** The registry
    calls `render` with `execution.arguments`. So a renderer reads
    `args.get("path")` where the body, two lines above it, read `args.path`. If
    the arguments were typed as the *model*, the type would check only the few
    renderers that read `args` at all. `JsonObject` is what the arguments are.
    `ToolDefinition.render` narrows to `JsonObject` one time, so the annotation
    is true at run time and not only in the source.

    **The value stays `Any` on purpose.** `ToolDefinition.render` validates the
    value against `output.schema` and raises `ToolOutputError` before it reaches
    this callback. That validation guarantees the *required* fields of a schema.
    It guarantees nothing for a `{"type": "object"}` output, and nothing for a
    field that has a default, so a renderer that reads either of those uses
    `.get`. Typing the value as `JsonValue` was measured. Every read would need
    a narrowing, which is 125 lines across nine files: for example
    `as_int(data["offset"]) + 1`, for arithmetic that a declared `int` already
    promised. The type would also not be enforced. Parameters are contravariant,
    so a named renderer that declares `value: Any` satisfies the slot, and only
    inline lambdas would have to narrow. The same field would then be read in
    two ways, depending on how its renderer was written. The value becomes the
    output *model* when the kernel codec can carry one (plan P8-02, issues 1 and
    2).
    """

    schema: SchemaDeclaration
    render: Callable[[JsonObject, Any], Sequence[ContentBlock]]
    presentation_meta: Callable[[JsonObject, Any], JsonObject | None] | None = None


@dataclass(frozen=True, slots=True)
class ToolFailure:
    """The canonical detail of a failure.

    `kind` is a `FailureKind`. `ph.tools.errors` declares that type, because it
    is what an error says about itself. `info` is the routable identity, for the
    failures that have one.
    """

    message: str
    kind: FailureKind = "failed"
    info: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The settled outcome handed to `present_result`."""

    content: tuple[ContentBlock, ...]
    is_error: bool
    meta: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """The outcome of one call, local to the execution.

    The durable event deliberately has no `value`. The log carries the
    projection that the model saw. A consumer that needs the structured value
    asks the tool to render it again, and does not trust a second copy.
    """

    is_error: bool
    content: tuple[ContentBlock, ...]
    value: Any = None
    error: ToolFailure | None = None
    meta: JsonObject | None = None
    additional_contexts: tuple[Message, ...] = ()
    concludes_turn: bool = False


def error_result(
    message: str, info: dict[str, str] | None = None, *, kind: FailureKind = "failed"
) -> ToolExecutionResult:
    """A failure result carrying the Native `Error: ` envelope the model expects."""
    return ToolExecutionResult(
        is_error=True,
        content=(TextBlock(text=f"Error: {message}"),),
        error=ToolFailure(message=message, kind=kind, info=info),
    )


def denied_result(reason: str, *, concludes_turn: bool = False) -> ToolExecutionResult:
    """Policy refused the call. The result is routable as a denial, so Code Mode
    can end the run (C3)."""
    return replace(
        error_result(reason, {"name": "ToolDenied", "code": TOOL_DENIED}, kind="denied"),
        concludes_turn=concludes_turn,
    )


def budget_result(reason: str, *, concludes_turn: bool) -> ToolExecutionResult:
    """A configured ceiling stopped the call, and in the usual case the turn too (D7).

    This function is the counterpart of `denied_result`. It is separate for the
    reason that `TOOL_BUDGET_SPENT` gives: nothing judged this call, so `kind`
    is `failed`. `concludes_turn` has no default. The two facts are independent,
    and the caller knows both. A budget that does not refill within the turn
    should end the turn. A per-call ceiling that the next call could pass should
    not. A default would make the more common case silent.
    """
    return replace(
        error_result(reason, {"name": "ToolBudgetSpent", "code": TOOL_BUDGET_SPENT}, kind="failed"),
        concludes_turn=concludes_turn,
    )


def aborted_result(*, started: bool) -> ToolExecutionResult:
    """A cancellation, before or after the body ran.

    The two codes exist for this difference. A call that was aborted before
    dispatch had no effect and is safe to retry. A call that was aborted after
    dispatch may have completed.
    """
    if started:
        return error_result(
            "tool call aborted", {"name": "Canceled", "code": TOOL_ABORTED}, kind="aborted"
        )
    return error_result(
        "tool call aborted before dispatch",
        {"name": "Canceled", "code": TOOL_ABORTED_BEFORE_DISPATCH},
        kind="aborted",
    )


def concluded_result() -> ToolExecutionResult:
    """A call that the turn ended before the call ran (C13).

    The batch stops dispatching when a result concludes the turn. Every call
    that the model asked for still owes a `tool/result`, because the next
    request is rejected when a `tool_use` block has no answer. The kind is
    `aborted`, so the result reads as "no effect, safe to retry", like the
    cancellation that it resembles. It has its own code, so that it is not
    mistaken for a cancellation.
    """
    return error_result(
        "the turn ended before this call ran",
        {"name": "TurnConcluded", "code": TOOL_TURN_CONCLUDED},
        kind="aborted",
    )


@dataclass(frozen=True, slots=True)
class ExecutionMode:
    """How one pending call may overlap its siblings."""

    kind: Literal["parallel", "exclusive"]


@dataclass(frozen=True, slots=True)
class ToolExecutionInput:
    """The description that a caller gives of one call, before the registry accepts it.

    `scope` selects which guards, restrictions and presentation apply. It is the
    per-agent policy boundary. `session` is where the call is recorded. The
    caller states both: the loop knows the shape of its agent, and the registry
    must not guess it. `session` defaults to no log. `scope` has had no default
    since P6-32. See the field below.
    """

    call_id: str
    name: str
    arguments: JsonValue
    scope: Boundary
    """The per-agent policy boundary. **Required since P6-32.**

    The class docstring already said that the caller states this field and that
    the registry must not guess it. The code could not enforce that. The field
    defaulted to `None`, and `_execution` resolved `None` as
    `call.scope or self.ctx`. That is the mount, whose isolation is the *global*
    layer. So a caller that stated nothing did not get "no tools". It got every
    tool that the deployment holds, and the registry guessed after all.

    A caller that means the wide answer on purpose passes `DEPLOYMENT`. Because
    both forms are available, the default was removed, and the statement in the
    class docstring is now enforced."""
    session: Session | None = None
    agent: AgentHandle | None = None
    """The agent to route an approval prompt to, when there is one."""
    root_call_id: str | None = None
    """The call that the model requested and that owns this execution tree. The
    default is `call_id`."""
    parent: object | None = None
    """The opaque token of the enclosing transport, for a Code Mode sub-dispatch.

    The presence of a parent also distinguishes a sub-dispatch from a call that
    the model made directly. Under `mode: code`, only a call WITH a parent may
    name a native tool (C6)."""
    cancel: CancelToken | None = None


@dataclass(frozen=True, slots=True)
class Done:
    """The call happened, and this is the value it produced (P10-13)."""

    value: Any


@dataclass(frozen=True, slots=True)
class NotDone:
    """The call did not happen and its effect is not there, so running it again
    is safe."""


@dataclass(frozen=True, slots=True)
class Unknown:
    """The tool cannot tell. This is the correct answer whenever the check would
    be a guess."""


Reconciled: TypeAlias = Done | NotDone | Unknown
"""What a tool says about a call that a crash left unresolved
(`ToolDefinition.reconcile`)."""

WorkspaceConfinement: TypeAlias = Callable[[JsonValue, AgentHandle | None], bool]
"""Whether every effect of one call is a file in the workspace of the agent.

The function takes the arguments of the call and the agent that makes the call.
The arguments are plain JSON, as they will run or as the log recorded them. This
is the per-call form of `ToolDefinition.effects_confined_to_workspace`."""


def call_id_of(record: SessionEvent) -> str:
    """The id of the call that a record names, for a record that is passed to
    `ToolDefinition.reconcile`.

    The id is the `callId` of a `tool/call`, or the `subCallId` of a Code Mode
    dispatch. It is the id that the body of the call saw as
    `ToolRunContext.call_id`. The result is empty when the record names
    neither."""
    return as_str(record.data.get("subCallId")) or as_str(record.data.get("callId"))


def recorded_arguments(record: SessionEvent) -> JsonValue:
    """The arguments of a call record, as the call ran with them.

    The arguments of a `tool/call` are the text from the model, parsed as the
    batch parses it. The arguments of a Code Mode dispatch are the object that
    its program passed."""
    raw = record.data.get("arguments")
    return parse_arguments(raw) if isinstance(raw, str) else thaw_json(raw)


@dataclass(slots=True)
class ToolExecution:
    """One accepted call inside the pipeline.

    The arguments have crossed a lossless-JSON boundary and are frozen. The
    registry assigns the `token`. Nested calls share it only as their opaque
    `parent`.
    """

    call_id: str
    root_call_id: str
    name: str
    arguments: JsonValue
    token: object
    scope: Context
    session: Session | None = None
    agent: AgentHandle | None = None
    parent: object | None = None
    signal: CancelToken | None = None
    """The live view of the cancellation. A `tools/execute` wrapper may replace
    it for its delegated lifetime, but only with a child token, so the wrapper
    can narrow the cancellation and never widen it. A wrapper cannot remove
    it."""
    substituted: bool = False
    """Whether `arguments` holds what approval put there and not what the model
    sent (`Edited`, P4-05). The flag is on the execution because it qualifies
    `arguments`, and every stage that holds an execution can read it. The
    write-ahead record reads it to log what will run. See `Allow.arguments`."""
    restore_covered: bool = False
    """Whether a workspace restore takes back everything that this call does
    (`ToolRuntime.restore_covers`), for a call nested in a Code Mode run.
    `ToolRuntime.prepare` asks one time, after the gate has decided (S4). The
    record of the dispatch carries the answer for `/revert`, and the barrier of
    the checkpoint policy skips on it. Two asks could disagree, because the
    answer follows links that a sibling dispatch may move between them. The
    value is `False` for any other call, where nothing reads it."""


@dataclass(slots=True)
class ToolRunContext:
    """The view that the body has of its own execution."""

    execution: ToolExecution
    definition: ToolDefinition
    """The tool that will render the value of this call. The run carries it, so
    the pipeline never has to look it up again, and cannot lose it on a path
    that ends early."""
    by: Running
    """Who registered that tool, and where (P6-29).

    `create_execution` resolves it one time, from the same view that produced
    `definition`. So it cannot disagree with `definition`, and it cannot be
    absent while the definition is present. A new lookup at each stage would be
    wrong. A tool that unregisters itself during its own `execute` is gone from
    the rebuilt view by the time of `finish`. `finalize_content` would then run
    under a fallback, and not as the row that wrote it.
    """
    _deferred: list[Message] = field(default_factory=list)
    _concluded: bool = False
    _presented: tuple[object, tuple[ContentBlock, ...]] | None = None
    """A value, and the content the body says it renders as (`present`)."""
    _cut_off: bool = False
    """Whether the body was entered and then left by a cancellation, and not by
    a return or a raise: its budget ran out, or a cancel reached it. The effect
    of the call may have happened, so a keyed call is settled as unknown
    (`ToolRuntime._settle_effect`). The registry sets the flag where it calls
    the body, which is the one place that knows that the body was entered."""

    @property
    def call_id(self) -> str:
        return self.execution.call_id

    @property
    def root_call_id(self) -> str:
        return self.execution.root_call_id

    @property
    def name(self) -> str:
        return self.execution.name

    @property
    def agent(self) -> AgentHandle | None:
        return self.execution.agent

    @property
    def session(self) -> Session | None:
        return self.execution.session

    @property
    def scope(self) -> Context:
        return self.execution.scope

    @property
    def token(self) -> object:
        return self.execution.token

    @property
    def signal(self) -> CancelToken | None:
        return self.execution.signal

    @property
    def idempotency_key(self) -> str:
        """A key for **the retries that this tool makes itself** against a far
        side that takes one.

        Examples are an HTTP `Idempotency-Key` and an SMTP `Message-ID`. The key
        is stable for this call and unique to it: `{session}/{call_id}`. For a
        Code Mode dispatch it is `{session}/{call_id}:code:{n}`, which is already
        deterministic. The key has no other purpose. It does **not** survive a
        call that the model issues again, because that call gets a new call id.
        Recognizing that repeat is the job of `ToolDefinition.idempotency_key`.
        """
        session = self.execution.session
        return self.call_id if session is None else f"{session.id}/{self.call_id}"

    def defer_context(self, context: Message) -> None:
        """Attach a context message to the result of this call.

        The loop appends the message only *after* the `tool/result`, so the call
        and the result stay adjacent. A context that was inserted between them
        would break the pairing that every provider relies on.
        """
        self._deferred.append(context)

    def conclude_turn(self) -> None:
        """Mark a successful result as terminal for this turn."""
        self._concluded = True

    def present(self, value: object, content: Sequence[ContentBlock]) -> None:
        """State what `value` renders as. `value` is the one that this body is
        about to return.

        **This is for a tool whose rendering was made where the tool ran**: a
        UTCP command that printed its own `text` beside its value, or a remote
        tool that answered in prose. That text is a pure projection of the
        arguments and the value. The `render` of the tool made it in another
        process. This process cannot make it again, because the information is
        not in the value.

        The registry uses the content wherever it would have rendered *this*
        value: at the dispatch, for the `projected` measurement of a
        post-execute row, and for the new render after a row passes the value
        through. Every policy row therefore sees and edits the same content that
        the model will see. A row that **replaces** the value gets the rendering
        that the definition makes of the replacement, because the text no longer
        describes the value. The match is by identity, not by equality. In both
        cases the value is still validated against the declared output.
        """
        self._presented = (value, tuple(content))

    def presented_for(self, value: object) -> tuple[ContentBlock, ...] | None:
        """The content that `present` gave for this exact value, or `None`."""
        presented = self._presented
        return presented[1] if presented is not None and presented[0] is value else None

    def raise_if_canceled(self) -> None:
        """Stop a long body when the call was canceled."""
        if self.execution.signal is not None:
            self.execution.signal.raise_if_canceled()


@dataclass(frozen=True, slots=True)
class Allow:
    arguments: JsonValue = None
    """Run the call with these arguments and not with the ones that the model sent.

    `None` is the ordinary case and leaves the call unchanged. A substitution
    arrives here and `prepare()` applies it, because the execution belongs to
    the pipeline. A listener that wrote `execution.arguments` itself would write
    the one field that the owner of the run has already passed to every listener
    before it.

    **Three records, three authors, and none of them is rewritten.** The
    assistant message holds what the model asked for. `approval/decided` carries
    the substitution that the human made. `tool/call`, which is written after
    the gate (P7-15), records what ran. A reader sees the request, the
    correction, and the act, each attributed to its author. One record that was
    overwritten in place would have attributed the arguments of the human to the
    model.
    """
    has_arguments: bool = False
    """Whether `arguments` is a substitution. This is separate from `None`,
    because `None` is a value that a tool may legitimately be called with."""
    kind: Literal["allow"] = "allow"


@dataclass(frozen=True, slots=True)
class Deny:
    reason: str
    concludes_turn: bool = False
    """Whether this refusal also ends the turn.

    This is the flag that a *successful* result carries
    (`ToolRunContext.conclude_turn`), set from the other side. `commit_ready`
    reads `concludes_turn` on every committed result, and that includes a denial
    from `prepare()`. So a policy row that must stop the conversation says so
    here, and does not need a second way to end a turn. The batch that is
    already in flight still settles. What ends is what comes after it."""
    failure_kind: Literal["denied", "failed"] = "denied"
    """How the model reads this: as a policy refusal, or as a spent budget (D7).

    This is separate from `concludes_turn`, which decides what the *loop* does,
    because the two questions have different answers. A budget ceiling ends the
    turn and is not a policy refusal. The `end` exit of `ph_stabilize.limits`
    refuses and ends the turn. `fs.FsDenied` and `fs.FileTooLarge` differ in the
    same way, under the same field name, and `code_mode.CodeRunFailure`
    separates `denied` from `budget`. With this field a policy row says which
    one it means, and the pipeline does not guess from the type of the decision.

    This is not the discriminator `kind` below. `kind` names the decision and
    not how the model reads it, and the pipeline handles a closed set of
    decisions."""
    kind: Literal["deny"] = "deny"


@dataclass(frozen=True, slots=True)
class Respond:
    """Do not run the call. Give the model this result instead.

    The result is *successful*, because the answerer answered the question that
    the call asked. A denial that the model would have to interpret is the wrong
    shape for "you don't need to run that, here is the answer".

    This is a member of the vocabulary of the pipeline, and not something that
    only the approval seam can reach. The reason is the same as for
    `Deny.concludes_turn`. A cache row, a replay row, or a "disabled here, do
    this instead" row each want to answer a call without running it. A
    capability that is wired to one consumer is a capability that the next
    consumer must build again.
    """

    message: str
    kind: Literal["respond"] = "respond"


@dataclass(frozen=True, slots=True)
class Ask:
    reason: str | None = None
    allowed_decisions: tuple[ApprovalDecisionName, ...] = ()
    """Which of `approve | edit | reject | respond` a front end should offer.

    **This is the vocabulary of the approval seam.** The name is imported under
    `TYPE_CHECKING`, so this module imports nothing from the approval seam at
    run time, and borrowing the name costs nothing. The field was
    `tuple[str, ...]` until `ctx.require(APPROVAL)` gave the call a type. Before
    that, a policy row could name a fifth decision and learn of the mistake only
    at the first ask (plan P8-05, issue 18).

    An empty tuple means all of them. The value is carried from the row that
    asked to the answerer that prompts, because *what may be decided* is a
    policy question. A front end that guessed would offer `edit` for a tool
    whose arguments must not be written by hand."""
    kind: Literal["ask"] = "ask"


PreToolDecision: TypeAlias = "Allow | Deny | Ask | Respond"


@dataclass(frozen=True, slots=True)
class Accept:
    """Keep the call successful, and optionally replace either projection.

    **Two projections, two consumers** (D10). `value` is what the program
    receives under Code Mode: `bridge.call` returns it, and a cell subscripts
    it. `content` is what reaches the transcript of the model. Ordinarily the
    content is rendered from the value. That is why a row that replaces the
    value leaves `content` unset and lets the registry render again.

    A row that supplies *both* states explicitly that they differ. Offload is
    that case: a result that is too large for the context is spilled to a file.
    The model gets a pointer, and the program keeps the whole object in its
    variable. That combination used to raise, on the reasoning that content is
    derived from the value and two sources would disagree. That reasoning is
    true of a row that means them to agree. It is exactly wrong for the one row
    whose job is to make them differ.
    """

    content: Sequence[ContentBlock] | None = None
    """What the model sees. `None` leaves the choice to the registry: the
    content of the body itself, or the render of a replaced `value`."""
    value: Any = None
    has_value: bool = False
    """Whether `value` is a replacement. This is separate from
    `value is not None`, because `None` is a value that a tool may legitimately
    return."""
    additional_contexts: tuple[Message, ...] = ()
    kind: Literal["accept"] = "accept"


@dataclass(frozen=True, slots=True)
class Block:
    """Turn the result into an error whose content is corrective feedback."""

    feedback: Sequence[ContentBlock]
    additional_contexts: tuple[Message, ...] = ()
    kind: Literal["block"] = "block"


PostToolDecision: TypeAlias = "Accept | Block"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A registered tool: what the model sees, and what one call does."""

    name: str
    description: str
    parameters: dict[str, Any]
    """JSON Schema for the arguments, exactly as the model receives it."""
    output: ToolOutput
    execute: Callable[[Any, ToolRunContext], Awaitable[Any]]
    """The argument model that `define_tool` checked cannot be recovered from
    here. A definition lives in a registry beside every other tool, so the type
    is erased at the table. The check happens at the constructor, where the body
    and its declared `parameters` meet. `PluginSpec.apply` has the same shape.

    Always awaited. `define_tool` wraps every body in an `async def` that
    runs it through `maybe_await`. So the question of sync or async is settled
    before a body reaches this field. The `await` on it in `registry.py` is
    therefore a checked await, and not one that passes because `Any` awaits."""
    finalize_content: (
        Callable[[ToolExecution, ToolExecutionResult], Sequence[ContentBlock] | None] | None
    ) = None
    """A synchronous final transform of the content. It is called exactly one
    time for every normalized outcome, including the pipeline failures that
    bypass `tools/post-execute`. It must be total and must not raise."""
    timeout_ms: int | None = None
    """A cooperative budget. The `tools/execute` wrapper of the `tools-timeout`
    row enforces it. It is never sent to the model: `schemas()` passes only the
    name, the description and the parameters."""
    self_limits: bool = False
    """The tool bounds its own output and offers the model a way to page.

    The tool declares this, and a policy row does not match it by name. `read`,
    `grep` and similar tools are *registered plugins*: a deployment renames
    them, and an MCP server adds its own. A list of names in another package
    cannot know them. The offload row of `ph-stabilize` reads the flag (G2) and
    leaves a self-limiting result inline. A model that reached the cap of this
    tool already knows how to ask for the next page, so a spill file that tells
    it again teaches nothing.
    """
    reads_paths: bool = False
    """Given an absolute path, this tool returns the contents of the file, and
    it can page through a large file.

    The tool declares this, and nothing matches it by name, for the same reason
    as `self_limits` one field up. A deployment renames its tools and an MCP
    server adds its own, so a list of names in another package cannot know
    them. The offload row of `ph-stabilize` reads the flag. The replacement text
    of that row must tell the model *how* to reach a spilled result. The
    upstream wording says `read_file`, which is not a tool that pH has. pH used
    to append a paragraph that corrected the name, and it now names the tool
    that exists.
    """
    searches_paths: bool = False
    """Given a path, this tool searches inside it and does not return it whole.

    This is the other half of what the offload replacement text must say. A
    spilled result is an ordinary file, and a search of a file is usually better
    than paging through it. But only a tool that takes a path as its root can
    search it.
    """
    arguments_disposable: bool = False
    """The model does not need to read the arguments of this call again.

    A tool sets this when its arguments are a *payload that it already
    delivered*, and not an instruction that the model refers back to. Examples
    are a file body passed to `write` and the replacement text passed to `edit`.
    Retained history may then elide the arguments under context pressure (the
    truncation pass of `compaction-summarize`, G4).

    The tool declares this, and a policy row does not match it by name, for the
    same reason as `self_limits`. These are *registered plugins*, so a
    deployment renames them and an MCP server adds its own, and a list of names
    in another package cannot know them. The distinction is real and narrow. A
    long `run_code` cell is exactly the kind of argument that a model reads
    again, so it stays.
    """
    effects_confined_to_workspace: bool | WorkspaceConfinement = False
    """Every effect of this call is a file inside the workspace of the agent (E7, N3).

    The field is named for what the *tool* knows, and not for what a consumer
    does with it. `/revert` derives "a workspace restore undoes this" from it,
    but that is an inference that the `/revert` row makes.

    The value is true for a tool whose entire effect is a file inside the
    workspace. It is false, the default, for anything that can reach outside the
    tree: a shell command that published a package, a spawn, a message. **Git
    restores the tree, not the world.** A tool with no effect at all declares
    `effect_free` instead.

    **Per call, where a tool can say** (S4). `ctx.fs.resolve` passes an absolute
    path through unchanged, and a link or an allowance can lead out of the tree.
    A flat `True` on `write` and `edit` therefore reported a write outside the
    workspace as covered, and the checkpoint policy skipped the barrier that a
    crash needed. A `WorkspaceConfinement` is called with the arguments of the
    call and the agent that makes the call. `confined` is the only reader. A
    predicate that raises counts as not confined.

    The tool declares this, and the command does not match it by name. These are
    *registered plugins*, so a deployment renames them and an MCP server adds
    its own. The default is the safe direction: a tool that declares nothing is
    reported as *not* undone.
    """
    effect_free: bool = False
    """This call changes nothing: no file, no process, no message (S4).

    So a crash in the middle of the call can leave nothing half done, and a
    restore has nothing to take back. The checkpoint policy lets a nested
    dispatch of the call skip its barrier, whether or not the run took a restore
    point, and `/revert` never lists the call. This differs from
    `effects_confined_to_workspace`, whose effects are real and are covered only
    by a restore that exists. A `read` must not pay for a barrier on a profile
    whose workspace takes no restore points, where every `write` must pay.
    """
    is_concurrency_safe: Callable[[Any], bool] | None = None
    """Only `True` puts a call in a parallel group. An omitted predicate, a
    raise, and any return that is not `True` all mean exclusive. That is the
    safe default: a tool that has not considered overlap must not be assumed to
    tolerate it."""
    is_irreversible: Callable[[Any], bool] | None = None
    """This call may do something that the harness cannot take back (P6-16).

    This is a **per-call predicate over the arguments** and not a flag, because
    the tool that most needs it cannot answer in general. One `run_code` cell
    may read a file or drop a table, and `bash` is a different command every
    time. It has the same shape as `is_concurrency_safe` for the same reason:
    the producer is the only party that knows.

    It is roughly the complement of `effects_confined_to_workspace`, but it is a
    separate field and is not derived from it. A call may reach outside the tree
    and still be easy to take back, such as a URL read or a query. A gate that
    asks too often teaches its user to stop reading it. `hitl` decides whether a
    person is actually asked.
    """
    idempotency_key: Callable[[Any], str | None] | None = None
    """What the **effect** of this call is, as a key: the same key means the same
    effect (P10-12).

    DESIGN I2: outside state has to be idempotent, because a session may repeat
    an action. After `TOOL_OUTCOME_UNKNOWN` the model issues the call again
    under a **new** call id, so nothing that is keyed on the call can recognize
    the repeat. Only the effect can, and only the producer knows what its effect
    is: a message id, an order number, a path and a digest. The function is
    called with the arguments that will run, as plain JSON, after any approval
    substitution. A return of `None` means that this call has no key. A function
    that raises is treated in the same way, and the raise is logged.

    A keyed call is recorded as a `TOOL_EFFECT` intent. A call whose key settled
    before is answered from the log and marked `repeated`, and does not run. A
    call whose first attempt failed runs again. A call whose outcome is unknown
    runs again with a note that the earlier call may have happened, or
    `reconcile` is asked about it (P10-13).

    **No shipped tool declares one**, and that is deliberate. None has an
    external effect whose repeat is worth suppressing (`bash` is a different
    command every time), and suppressing a legitimate repeat is the failure that
    this field could cause.
    """
    reconcile: Callable[[Any, SessionEvent, Session], Awaitable[Reconciled]] | None = None
    """Did this call happen? Asked on resume, about a call that a crash left
    unresolved (P10-13).

    The `TOOL_OUTCOME_UNKNOWN` text of repair asks the *model* to decide, from
    the semantics of the tool, whether to retry. It asks the model because
    nothing asked the tool, and the tool knows. The function is given the
    arguments of the call (plain JSON, as recorded), its `tool/call` record, and
    the session that is being resumed. It answers `Done(value)` when the effect
    is there, and the value is rendered as the result of the call. It answers
    `NotDone()` when the effect is not there and running the call again is safe.
    It answers `Unknown()` whenever the check would be a guess. A raise counts
    as `Unknown`.

    The function is asked in two places. The first is on resume, about a
    `tool/call` that a crash left unresolved, or about a Code Mode dispatch that
    was left open. For the dispatch it is given the `tool/code-dispatch-start`
    record, and the arguments are then the object that the program passed (L6b).
    The second is in the pipeline, about a repeat of an effect whose first
    attempt nobody saw finish (`idempotency_key`). There it is given the
    `tool/effect` record instead.

    Both are asked through `ToolRuntime`, bound as the row that registered the
    tool.

    **Not enforced: an agent-scoped tool is never asked on resume.** Resume
    looks tools up at `DEPLOYMENT` scope, before any agent exists. A tool that
    is registered on the scope of an agent is therefore not visible there, and
    its call keeps the `TOOL_OUTCOME_UNKNOWN` text.
    """
    present_call: Callable[[JsonObject], ToolCallView | None] | None = None
    present_result: Callable[[JsonObject, ToolResult], ToolResultView | None] | None = None

    def schema(self) -> ToolSchema:
        """The model-facing schema. Nothing else about the tool reaches the wire."""
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    def render(
        self,
        args: JsonValue,
        value: object,
        *,
        presented: Sequence[ContentBlock] | None = None,
    ) -> tuple[ContentBlock, ...]:
        """Project a validated value into the content that the model sees.

        `args` arrives as whatever `parse_arguments` made of the JSON from the
        model, which is the raw *string* when the JSON was malformed. `as_obj`
        therefore makes the `JsonObject` of a renderer true in every case. It
        guards a path that no shipped tool reaches today: every renderer that
        reads `args` belongs to a pydantic-declared tool, and a malformed call
        to such a tool fails validation first. The same narrowing in
        `presentation.render_call_view` is the one that has a live case.
        """
        violations = validate_json_schema_value(self.output.schema, value)
        if violations:
            raise ToolOutputError(self.name, violations)
        if presented is not None:
            # The content was rendered where the tool ran
            # (`ToolRunContext.present`). The value above is still held to the
            # declared output.
            return tuple(presented)
        try:
            rendered = self.output.render(as_obj(args), value)
        except Exception as error:
            raise ToolOutputError(self.name, [f"output.render failed: {error}"]) from error
        return tuple(rendered)

    def project_meta(self, args: JsonValue, value: object) -> JsonObject | None:
        if self.output.presentation_meta is None:
            return None
        try:
            return self.output.presentation_meta(as_obj(args), value)
        except Exception as error:
            raise ToolOutputError(
                self.name, [f"output.presentation_meta failed: {error}"]
            ) from error

    def classify(self, args: object) -> ExecutionMode:
        """The live overlap classification for one call's arguments."""
        parallel = _answers_true(self.is_concurrency_safe, args)
        return ExecutionMode(kind="parallel" if parallel else "exclusive")

    def irreversible(self, args: object) -> bool:
        """Whether this call declares itself hard to take back (P6-16)."""
        return _answers_true(self.is_irreversible, args)

    def confined(self, args: JsonValue, agent: AgentHandle | None) -> bool:
        """Whether every effect of this call is a file in `agent`'s workspace (S4)."""
        declared = self.effects_confined_to_workspace
        return declared if isinstance(declared, bool) else _answers_true(declared, args, agent)


@dataclass(frozen=True, slots=True)
class TransportPresentation:
    """The parts of the Code Mode transport that a *profile* may restate.

    The name of the transport is reserved, so that nothing can occupy it and
    misdirect a model that was told to call it (P1-04, C6). But a profile still
    needs to present the transport under its own name and description. The RLM
    profile calls it `python`, and describes the interpreter of pH in its own
    words.

    Only the presentation is restated. `parameters` and `execute` are absent by
    design. The argument schema and the governed body are what make the
    transport the transport. A profile that could replace them would have
    replaced Code Mode and not renamed it.
    """

    name: str
    description: str
    output: ToolOutput | None = None
    present_call: Callable[[JsonObject], ToolCallView | None] | None = None
    present_result: Callable[[JsonObject, ToolResult], ToolResultView | None] | None = None

    def rename(self, transport: ToolDefinition) -> ToolDefinition:
        """The transport as this profile presents it."""
        return replace(
            transport,
            name=self.name,
            description=self.description,
            output=self.output if self.output is not None else transport.output,
            present_call=self.present_call or transport.present_call,
            present_result=self.present_result or transport.present_result,
        )


def define_tool[A: BaseModel](
    name: str,
    description: str,
    *,
    parameters: type[A] | dict[str, Any],
    output: ToolOutput | SchemaDeclaration,
    execute: Callable[[A, ToolRunContext], Any],
    render: Callable[[JsonObject, Any], Sequence[ContentBlock]] | None = None,
    presentation_meta: Callable[[JsonObject, Any], JsonObject | None] | None = None,
    finalize_content: Callable[..., Sequence[ContentBlock] | None] | None = None,
    timeout_ms: int | None = None,
    self_limits: bool = False,
    reads_paths: bool = False,
    searches_paths: bool = False,
    arguments_disposable: bool = False,
    effects_confined_to_workspace: bool | WorkspaceConfinement = False,
    effect_free: bool = False,
    is_concurrency_safe: Callable[[Any], bool] | bool | None = None,
    is_irreversible: Callable[[Any], bool] | bool | None = None,
    idempotency_key: Callable[[Any], str | None] | None = None,
    reconcile: Callable[[Any, SessionEvent, Session], Awaitable[Reconciled]] | None = None,
    present_call: Callable[[JsonObject], ToolCallView | None] | None = None,
    present_result: Callable[[JsonObject, ToolResult], ToolResultView | None] | None = None,
) -> ToolDefinition:
    """Build a `ToolDefinition` that validates the arguments before the body sees them.

    When `parameters` is a pydantic model, the body receives a **validated model
    instance**, so a tool never checks its own input by hand. A raw schema dict
    passes the parsed value through, and the tool owns the validation.

    **The function is generic in `A`, and that is all of the checking.** `A` is
    inferred from `parameters` when it is a model. A body that is annotated with
    a different model from the one declared beside it is therefore an error
    here, at the declaration, and not a validation failure at the first call.
    Only the body sees `A`. `ToolOutput` says why the renderers cannot. A
    raw-schema tool leaves `A` to be inferred from its body, where it is `Any`.
    Such a tool is checked exactly as loosely as it was declared.

    The value side is deliberately not parameterized, and `ToolOutput` gives the
    reasons. That is also why `execute` cannot say "sync or async" in the way
    that `CommandBody` does. `MaybeAwaitable[Any]` *is* `Any`, so the
    `Awaitable[Any] | Any` that was here before stated nothing that a type
    checker could use. `run` awaits through `maybe_await`.
    """
    resolved_output = (
        output
        if isinstance(output, ToolOutput)
        else ToolOutput(
            schema=output,
            render=render or (lambda _args, value: text_content(_default_render(value))),
            presentation_meta=presentation_meta,
        )
    )
    typed = (
        parameters if isinstance(parameters, type) and issubclass(parameters, BaseModel) else None
    )

    async def run(raw_args: object, run_ctx: ToolRunContext) -> Any:  # noqa: ANN401
        args: Any = raw_args
        if typed is not None:
            args = typed.model_validate(raw_args if raw_args is not None else {})
        return await maybe_await(execute(args, run_ctx))

    return ToolDefinition(
        name=name,
        description=description,
        parameters=schema_of(parameters),
        output=resolved_output,
        execute=run,
        finalize_content=finalize_content,
        timeout_ms=timeout_ms,
        self_limits=self_limits,
        reads_paths=reads_paths,
        searches_paths=searches_paths,
        arguments_disposable=arguments_disposable,
        effects_confined_to_workspace=effects_confined_to_workspace,
        effect_free=effect_free,
        is_concurrency_safe=_classifier(is_concurrency_safe),
        is_irreversible=_classifier(is_irreversible),
        idempotency_key=idempotency_key,
        reconcile=reconcile,
        present_call=present_call,
        present_result=present_result,
    )


def _classifier(flag: Callable[[Any], bool] | bool | None) -> Callable[[Any], bool] | None:
    """`True` means always, a callable means itself, and anything else means not
    declared."""
    if flag is True:
        return lambda _args: True
    return flag if callable(flag) else None


def _answers_true(classifier: Callable[..., bool] | None, *args: object) -> bool:
    """Only `True` counts. An omitted classifier, a return that is not `True`
    and a **raise** all mean "not declared". A predicate that raises did not
    answer. If silence became a claim, one buggy classifier would make that
    claim on every call."""
    if classifier is None:
        return False
    try:
        return classifier(*args) is True
    except Exception:
        return False


def _default_render(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "(no output)"
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)
