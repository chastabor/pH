"""What a tool *is*, and what one call produces.

The load-bearing decision here is that `output` is **mandatory** (P1-01). A tool
returns a canonical JSON *value*; its declared `render` projects that value into
the content the model sees. Two consequences follow, and both are why dsh made
it mandatory:

* the durable record is the value's projection, so a UI and a replay render the
  same card from the log alone — nothing depends on the live object;
* a tool cannot quietly hand the model prose that its own schema does not
  describe, because content is derived from a validated value.

`ToolRunContext` is the body's view: the accepted execution, the definition that
will render its value, and the two things a body may do to the turn beyond
returning — defer a context message, and conclude the turn.

@module ph.tools.definition
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from ..cancel import CancelToken
from ..cordis import Context
from ..llm.types import ContentBlock, Message, TextBlock, ToolSchema
from ..session import Session
from .errors import TOOL_ABORTED, TOOL_ABORTED_BEFORE_DISPATCH, TOOL_DENIED, ToolOutputError
from .json_schema import schema_of, validate_json_schema_value
from .presentation import ToolCallView, ToolResultView

__all__ = [
    "Accept",
    "Allow",
    "Ask",
    "Block",
    "Deny",
    "ExecutionMode",
    "FailureKind",
    "PostToolDecision",
    "PreToolDecision",
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
    "aborted_result",
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
FailureKind: TypeAlias = Literal["denied", "failed", "aborted"]


def text_content(text: str) -> list[Any]:
    """The one-block content most tools return."""
    return [TextBlock(text=text)]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """A tool's canonical output declaration.

    `render` is a pure projection from validated arguments and value to content.
    `presentation_meta` is the tool's private durable payload, computed only for
    top-level calls (a nested Code Mode dispatch has no card of its own).
    """

    schema: SchemaDeclaration
    render: Callable[[Any, Any], Sequence[Any]]
    presentation_meta: Callable[[Any, Any], Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolFailure:
    """Canonical failure detail.

    `kind` is the fact every consumer branches on and none may infer: policy
    **denied** the call, the tool **failed**, or cancellation **aborted** it.
    Code Mode ends the run on the first, lets the program handle the second, and
    a UI card colours each differently. `info` is the routable identity for the
    subset of failures that have one.
    """

    message: str
    kind: FailureKind = "failed"
    info: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The settled outcome handed to `present_result`."""

    content: tuple[Any, ...]
    is_error: bool
    meta: Any = None


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """One call's execution-local outcome.

    `value` is deliberately absent from the durable event: the log carries the
    projection the model saw, and a consumer that needs the structured value
    asks the tool to render it again rather than trusting a second copy.
    """

    is_error: bool
    content: tuple[Any, ...]
    value: Any = None
    error: ToolFailure | None = None
    meta: Any = None
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


def denied_result(reason: str) -> ToolExecutionResult:
    """Policy said no. Routable as a denial, so Code Mode can end the run (C3)."""
    return error_result(reason, {"name": "ToolDenied", "code": TOOL_DENIED}, kind="denied")


def aborted_result(*, started: bool) -> ToolExecutionResult:
    """Cancellation, before or after the body ran.

    The distinction is the whole reason there are two codes: a call aborted
    before dispatch had no effect and is safe to retry; one aborted after may
    have completed.
    """
    if started:
        return error_result(
            "tool call aborted", {"name": "Cancelled", "code": TOOL_ABORTED}, kind="aborted"
        )
    return error_result(
        "tool call aborted before dispatch",
        {"name": "Cancelled", "code": TOOL_ABORTED_BEFORE_DISPATCH},
        kind="aborted",
    )


@dataclass(frozen=True, slots=True)
class ExecutionMode:
    """How one pending call may overlap its siblings."""

    kind: Literal["parallel", "exclusive"]


@dataclass(frozen=True, slots=True)
class ToolExecutionInput:
    """A caller's description of one call, before the registry accepts it.

    `scope` selects which guards, restrictions and presentation apply — the
    per-agent policy boundary — and `session` is where the call is recorded.
    Both are stated by the caller (the loop knows its agent's shape; the
    registry must not guess it) and default to the global context and no log.
    """

    call_id: str
    name: str
    arguments: Any
    scope: Context | None = None
    session: Session | None = None
    agent: Any = None
    """The agent to route an approval prompt to, when there is one."""
    root_call_id: str | None = None
    """The model-requested call owning this execution tree; defaults to `call_id`."""
    parent: object | None = None
    """The enclosing transport's opaque token, for a Code Mode sub-dispatch.

    Its presence is also what distinguishes a sub-dispatch from a model-direct
    call: under `mode: code` only a call WITH a parent may name a native tool
    (C6)."""
    cancel: CancelToken | None = None


@dataclass(slots=True)
class ToolExecution:
    """One accepted call inside the pipeline.

    Arguments have crossed a lossless-JSON boundary and are frozen. The `token`
    is registry-assigned and is shared with nested calls only as their opaque
    `parent`.
    """

    call_id: str
    root_call_id: str
    name: str
    arguments: Any
    token: object
    scope: Context
    session: Session | None = None
    agent: Any = None
    parent: object | None = None
    signal: CancelToken | None = None
    """The live cancellation view. A `tools/execute` wrapper may replace it for
    its delegated lifetime — with a child token, so it can narrow but never
    widen — but cannot remove it."""


@dataclass(slots=True)
class ToolRunContext:
    """The body's view of its own execution."""

    execution: ToolExecution
    definition: ToolDefinition
    """The tool that will render this call's value. Carried on the run so the
    pipeline never has to look it up again — or lose it on a path that ends
    early."""
    _deferred: list[Message] = field(default_factory=list)
    _concluded: bool = False

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
    def agent(self) -> Any:
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

    def defer_context(self, context: Message) -> None:
        """Attach a context message to this call's own result.

        The loop appends it only *after* the `tool/result`, so call/result
        adjacency survives — a context spliced between them would break the
        pairing every provider relies on.
        """
        self._deferred.append(context)

    def conclude_turn(self) -> None:
        """Mark a successful result as terminal for this turn."""
        self._concluded = True

    def raise_if_cancelled(self) -> None:
        """Bail out of a long body when the call was cancelled."""
        if self.execution.signal is not None:
            self.execution.signal.raise_if_cancelled()


@dataclass(frozen=True, slots=True)
class Allow:
    kind: Literal["allow"] = "allow"


@dataclass(frozen=True, slots=True)
class Deny:
    reason: str
    kind: Literal["deny"] = "deny"


@dataclass(frozen=True, slots=True)
class Ask:
    reason: str | None = None
    kind: Literal["ask"] = "ask"


PreToolDecision: TypeAlias = "Allow | Deny | Ask"


@dataclass(frozen=True, slots=True)
class Accept:
    """Keep the call successful, optionally replacing one projection."""

    content: Sequence[Any] | None = None
    value: Any = None
    has_value: bool = False
    additional_contexts: tuple[Message, ...] = ()
    kind: Literal["accept"] = "accept"


@dataclass(frozen=True, slots=True)
class Block:
    """Turn the result into an error whose content is corrective feedback."""

    feedback: Sequence[Any]
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
    execute: Callable[[Any, ToolRunContext], Any]
    finalize_content: (
        Callable[[ToolExecution, ToolExecutionResult], Sequence[Any] | None] | None
    ) = None
    """Synchronous last-mile content transform, invoked exactly once for every
    normalized outcome — pipeline failures that bypass `tools/post-execute`
    included. Must be total and must not raise."""
    timeout_ms: int | None = None
    """Cooperative budget, enforced by the `tools-timeout` row's `tools/execute`
    wrapper. Never sent to the model: `schemas()` whitelists
    name/description/parameters only."""
    is_concurrency_safe: Callable[[Any], bool] | None = None
    """Only `True` opts a call into a parallel group. Omission, a raise, and any
    non-`True` return are all exclusive — the safe default, since a tool that
    has not thought about overlap must not be assumed to tolerate it."""
    present_call: Callable[[Any], ToolCallView | None] | None = None
    present_result: Callable[[Any, ToolResult], ToolResultView | None] | None = None

    def schema(self) -> ToolSchema:
        """The model-facing schema. Nothing else about the tool reaches the wire."""
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    def render(self, args: Any, value: Any) -> tuple[Any, ...]:
        """Project a validated value into model-facing content."""
        violations = validate_json_schema_value(self.output.schema, value)
        if violations:
            raise ToolOutputError(self.name, violations)
        try:
            rendered = self.output.render(args, value)
        except Exception as error:
            raise ToolOutputError(self.name, [f"output.render failed: {error}"]) from error
        return tuple(rendered)

    def project_meta(self, args: Any, value: Any) -> Any:
        if self.output.presentation_meta is None:
            return None
        try:
            return self.output.presentation_meta(args, value)
        except Exception as error:
            raise ToolOutputError(
                self.name, [f"output.presentation_meta failed: {error}"]
            ) from error

    def classify(self, args: Any) -> ExecutionMode:
        """The live overlap classification for one call's arguments."""
        classifier = self.is_concurrency_safe
        if classifier is None:
            return ExecutionMode(kind="exclusive")
        try:
            return ExecutionMode(kind="parallel" if classifier(args) is True else "exclusive")
        except Exception:
            return ExecutionMode(kind="exclusive")


@dataclass(frozen=True, slots=True)
class TransportPresentation:
    """The parts of the Code Mode transport a *profile* is allowed to restate.

    The transport's name is reserved so nothing can occupy it and misdirect a
    model told to call it (P1-04, C6) — but a profile still needs to present it
    under its own name and description. The RLM profile calls it `ipython` and
    ports prime-agent's wording verbatim, so a model that knows that surface
    finds the surface it knows.

    Only the presentation is restated. `parameters` and `execute` are absent by
    design: the argument schema and the governed body are what make the transport
    the transport, and a profile that could replace them would have replaced
    Code Mode rather than renamed it.
    """

    name: str
    description: str
    output: ToolOutput | None = None
    present_call: Callable[[Any], ToolCallView | None] | None = None
    present_result: Callable[[Any, ToolResult], ToolResultView | None] | None = None

    def rename(self, transport: ToolDefinition) -> ToolDefinition:
        """The transport as this profile presents it."""
        return dataclasses_replace(
            transport,
            name=self.name,
            description=self.description,
            output=self.output if self.output is not None else transport.output,
            present_call=self.present_call or transport.present_call,
            present_result=self.present_result or transport.present_result,
        )


def define_tool(
    name: str,
    description: str,
    *,
    parameters: SchemaDeclaration,
    output: ToolOutput | SchemaDeclaration,
    execute: Callable[..., Any],
    render: Callable[[Any, Any], Sequence[Any]] | None = None,
    presentation_meta: Callable[[Any, Any], Any] | None = None,
    finalize_content: Callable[..., Sequence[Any] | None] | None = None,
    timeout_ms: int | None = None,
    is_concurrency_safe: Callable[[Any], bool] | bool | None = None,
    present_call: Callable[[Any], ToolCallView | None] | None = None,
    present_result: Callable[[Any, ToolResult], ToolResultView | None] | None = None,
) -> ToolDefinition:
    """Build a `ToolDefinition`, validating arguments before the body sees them.

    When `parameters` is a pydantic model the body receives a **validated model
    instance**, so a tool never hand-checks its own input; a raw schema dict
    passes the parsed value through and the tool owns validation.
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

    async def run(raw_args: Any, run_ctx: ToolRunContext) -> Any:
        args: Any = raw_args
        if typed is not None:
            args = typed.model_validate(raw_args if raw_args is not None else {})
        result = execute(args, run_ctx)
        return await result if inspect.isawaitable(result) else result

    safe: Callable[[Any], bool] | None
    if is_concurrency_safe is True:
        safe = lambda _args: True  # noqa: E731 - a one-expression classifier
    elif callable(is_concurrency_safe):
        safe = is_concurrency_safe
    else:
        safe = None

    return ToolDefinition(
        name=name,
        description=description,
        parameters=schema_of(parameters),
        output=resolved_output,
        execute=run,
        finalize_content=finalize_content,
        timeout_ms=timeout_ms,
        is_concurrency_safe=safe,
        present_call=present_call,
        present_result=present_result,
    )


def _default_render(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "(no output)"
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _content_blocks(blocks: Sequence[Any]) -> list[ContentBlock]:  # pragma: no cover - typing aid
    return list(blocks)
