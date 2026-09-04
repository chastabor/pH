"""Code Mode: one transport, and every binding call governed (C1-C3).

This is the file the whole containment argument rests on. A `run_code` cell is
*one* tool call, but a cell that writes forty files must not be *one* governance
evaluation — that is exactly the collapse the feature map records in
prime-agent, where `ToolName = "ipython"` reduces permission, approval, call
limits, offload and `fs/write-intent` to a single decision per cell.

So the bridge re-enters the **complete** pipeline per binding call:

* each `await tools.<name>(...)` is a sub-call with the outer execution's opaque
  token as `parent`, dispatched through `tools/pre-execute` → approval → guards
  → `tools/execute` → body → `tools/post-execute` (C1);
* each one logs `tool/code-dispatch-start` at entry and `tool/code-dispatch` at
  settle, so forty writes are forty durable records rather than one stdout blob
  (C2). Both are log-only: `derive_messages()` ignores them, so sub-calls never
  re-enter model context;
* a **denial fails the whole run** (C3, Q9). This is pH's one deliberate
  divergence from dsh, and the reason is that a program which can `except` a
  refusal can route around it — retry with a different path, fall back to
  `subprocess`. Failing the run puts the refusal in the model's context and
  bounds partial state to *about* one cell: the bound is best-effort in time,
  because the abort is a signal ladder that stops the program at its next yield
  point (see `Kernel.cancel_grace` in `ph_rlm.kernel.manager`). A *failed* call
  (timeout, bad arguments) keeps dsh's `ToolCallError` semantics, because that is
  the model's to handle.

Budgets (C4) are enforced here rather than by the runtime: one approved cell
must not be able to issue unbounded governed calls, and the bridge is the only
place that sees them all.

@module ph.tools.code_mode
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, ClassVar, Literal

import anyio
from pydantic import Field

from ..cancel import CancelToken
from ..cordis import Context, events, maybe_await, plugin
from ..seams.code_runtime import CodeBinding, CodeBindingNamespace, CodeRunRequest
from ..session import Session
from ..session.json import freeze_json_value, thaw_json
from ..system_prompt.assembly import ORDER_TOOL_GUIDANCE, PromptSection
from ..wire import WireModel
from .definition import (
    ToolExecutionInput,
    ToolExecutionResult,
    ToolOutput,
    ToolRunContext,
    define_tool,
    text_content,
)
from .errors import FailureKind, HarnessError
from .registry import RUN_CODE, PreparedCall, ToolRuntime
from .sdk import code_only_rule, render_python_sdk, render_typescript_sdk

__all__ = [
    "MAX_DISPATCHES_PER_RUN",
    "MAX_SUBAGENT_SPAWNS_PER_RUN",
    "CodeBindingsRequest",
    "CodeRunFailure",
    "DispatchBridge",
    "ToolCallError",
    "apply",
    "governed_binding",
]

log = logging.getLogger("ph.tools.code_mode")

MAX_DISPATCHES_PER_RUN = 256
"""How many governed calls one cell may issue (C4).

A cell the human approved is still one decision; without a cap it could issue
unbounded governed calls on the strength of it."""

MAX_SUBAGENT_SPAWNS_PER_RUN = 32
"""How many children one cell may spawn (C4). Counted by bindings that declare
`counts_as_spawn`."""

events.declare(
    "tools/code-dispatch-log",
    "waterfall",
    owner="ph.tools.code_mode",
    doc="One settled sub-dispatch about to be logged; the spill policy reshapes it here.",
)


class ToolCallError(HarnessError):
    """A sub-call *failed*. Raised inside the program, for the model to handle."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"tools.{name} failed: {message}", "TOOL_CALL_FAILED")
        self.tool_name = name


class CodeRunFailure(HarnessError):
    """The whole run is over.

    `kind: "denied"` is the C3 case: a refusal the program is not allowed to
    catch, because catching it is how a program routes around policy.

    All three kinds end the run; they do not mean the same thing to a consumer,
    so each declares its own `failure_kind`. A budget is the tool failing (the
    program asked for too much), a denial is policy, an abort is cancellation.
    """

    _FAILURE_KINDS: ClassVar[dict[str, FailureKind]] = {
        "denied": "denied",
        "budget": "failed",
        "aborted": "aborted",
    }

    def __init__(self, kind: Literal["denied", "budget", "aborted"], message: str) -> None:
        super().__init__(message, f"CODE_RUN_{kind.upper()}")
        self.kind = kind
        self.failure_kind = self._FAILURE_KINDS[kind]


class CodeDispatchRef(WireModel):
    """Which dispatch this is — the identity its two records share.

    A dispatch writes twice: `tool/code-dispatch-start` before the binding's body
    runs (B4, P7-15) and `tool/code-dispatch` once it settles. **Readers pair
    them by these fields**, and by nothing else — the TUI adapter looks up the
    parent card by `parentCallId` and the dispatch card by `subCallId`, and
    `/revert` reads `parentCallId` and `name` off the start record to list what a
    run did outside the tree it restored.

    One type, so the pairing cannot drift. The start record used to be a
    hand-written camelCase dict while the settle went through `to_wire()`: four
    fields spelled twice, where renaming one — or changing `wire_alias` — moves
    the settle record and silently leaves the start record's literals behind,
    unpairing every reader at once with nothing failing.
    """

    root_call_id: str
    parent_call_id: str
    sub_call_id: str
    name: str


class CodeDispatchLog(CodeDispatchRef):
    """One settled sub-dispatch, as the log waterfall sees it."""

    is_error: bool


@dataclass(frozen=True, slots=True)
class CodeBindingsRequest:
    """What a namespace factory is told about the run it is binding for.

    `bridge is None` means the namespace is being *described* rather than
    bound — the SDK prompt section asks the factories the same question the run
    does, so the block cannot list a namespace the program could not reach, or
    omit one it can. Two fields only: the bridge already carries the execution
    and the session, and a second copy of either would be one more thing the two
    construction sites could disagree about.
    """

    scope: Context
    bridge: DispatchBridge | None = None

    @property
    def describing(self) -> bool:
        return self.bridge is None


@dataclass(slots=True)
class DispatchBridge:
    """Turns binding calls inside one program into governed sub-calls."""

    tools: ToolRuntime
    ctx: Context
    execution: Any
    session: Session | None
    token: CancelToken
    max_parallel: int = 10
    max_dispatches: int = MAX_DISPATCHES_PER_RUN
    max_spawns: int = MAX_SUBAGENT_SPAWNS_PER_RUN
    _dispatched: int = 0
    _spawned: int = 0
    _failure: CodeRunFailure | None = None
    _limiter: anyio.CapacityLimiter = field(init=False)

    def __post_init__(self) -> None:
        self._limiter = anyio.CapacityLimiter(max(1, self.max_parallel))

    @property
    def dispatch_count(self) -> int:
        return self._dispatched

    def _settle(self, kind: Literal["denied", "budget", "aborted"], message: str) -> CodeRunFailure:
        # Once the run is settled, later awaits stop immediately rather than
        # racing to do more work the outcome has already discarded.
        self._failure = CodeRunFailure(kind, message)
        return self._failure

    async def call(self, binding: CodeBinding, arguments: Any) -> Any:
        """Dispatch one binding call through the full pipeline.

        :raises ToolCallError: the call failed; the program may handle it.
        :raises CodeRunFailure: the call was denied or a budget was reached;
            the program may not.
        """
        if self._failure is not None:
            raise self._failure
        if self._dispatched >= self.max_dispatches:
            raise self._settle(
                "budget",
                f"this program reached max_dispatches_per_run={self.max_dispatches}; "
                "split the work across cells",
            )
        if binding.counts_as_spawn:
            self._spawned += 1
            if self._spawned > self.max_spawns:
                raise self._settle(
                    "budget", f"this program reached max_subagent_spawns_per_run={self.max_spawns}"
                )

        # Deterministic and ordered, so a reader can pair a start with its settle
        # and see submission order without a timestamp.
        sub_call_id = f"{self.execution.call_id}:code:{self._dispatched}"
        self._dispatched += 1
        # One lossless snapshot proves the arguments are JSON; `create_execution`
        # accepts the frozen form directly, so it is not thawed again for the call.
        snapshot = freeze_json_value(arguments, frozen_input=True)

        ref = CodeDispatchRef(
            root_call_id=self.execution.root_call_id,
            parent_call_id=self.execution.call_id,
            sub_call_id=sub_call_id,
            name=binding.name,
        )

        async with self._limiter:
            result = await self.tools.execute(
                ToolExecutionInput(
                    call_id=sub_call_id,
                    root_call_id=self.execution.root_call_id,
                    name=binding.name,
                    arguments=snapshot,
                    scope=self.execution.scope,
                    session=self.session,
                    agent=self.execution.agent,
                    parent=self.execution.token,
                    cancel=self.token,
                ),
                write_ahead=partial(self._log_start, ref),
            )
            await self._log_settle(ref, result)

        if result.is_error:
            failure = result.error
            reason = failure.message if failure is not None else "denied"
            if failure is not None and failure.kind == "denied":
                raise self._settle(
                    "denied",
                    f"tools.{binding.name} was refused: {reason}. The program was stopped; "
                    "re-plan with this refusal in mind.",
                )
            raise ToolCallError(binding.name, reason)
        return result.value

    def _log_start(self, ref: CodeDispatchRef, prepared: PreparedCall) -> None:
        """The dispatch, recorded once the pipeline has decided and before it runs.

        The same point `execute_tool_calls` writes `tool/call` (B4, P7-15): a
        dispatch parked on an approval is not yet something the program *did*,
        and `/revert` reads these records as the list of what it did.

        `execution.arguments` unconditionally: `create_execution` sets it from the
        snapshot this dispatch passed in, and the only thing that ever replaces it
        is approval's substitution — which is exactly what this should record.
        (`batch.py`'s ternary is *not* removable for the same shape: there the
        unsubstituted branch is the model's original bytes, a different object.)
        """
        if self.session is None:
            return
        self.session.append(
            "tool/code-dispatch-start",
            {**ref.to_wire(), "arguments": thaw_json(prepared.run.execution.arguments)},
        )

    async def _log_settle(self, ref: CodeDispatchRef, result: ToolExecutionResult) -> None:
        async def inner(record: CodeDispatchLog, blocks: Sequence[Any]) -> Sequence[Any]:
            return blocks

        record = CodeDispatchLog(**ref.__dict__, is_error=result.is_error)
        # The spill policy replaces an oversized dispatch content *individually*
        # here (C5), so one large read does not melt its siblings into a blob.
        shaped = await self.ctx.waterfall(
            "tools/code-dispatch-log", record, result.content, inner=inner
        )
        if self.session is not None:
            self.session.append(
                "tool/code-dispatch",
                {**record.to_wire(), "content": [block.to_wire() for block in shaped]},
            )


class Config(WireModel):
    """Row config for the Code Mode transport."""

    max_dispatches_per_run: int = MAX_DISPATCHES_PER_RUN
    max_subagent_spawns_per_run: int = MAX_SUBAGENT_SPAWNS_PER_RUN
    max_parallel_sub_calls: int = 10


@plugin("tools-code-mode", config=Config, inject=["tools", "code_runtime", "system_prompt"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the reserved transport, the shipped SDK renderers, and the prompt section."""
    tools: ToolRuntime = ctx.tools
    ctx.code_runtime.register_sdk_renderer("python", render_python_sdk)
    ctx.code_runtime.register_sdk_renderer("typescript", render_typescript_sdk)

    async def run_code(args: Any, run: ToolRunContext) -> Any:
        # `Mapping`, not `dict`: accepted arguments are frozen into a
        # `MappingProxyType`, which is a Mapping but not a dict instance.
        program = args.get("program") if isinstance(args, Mapping) else None
        if not isinstance(program, str) or not program.strip():
            raise ToolCallError(run.name, "program must be a non-empty string")
        runtime = ctx.code_runtime.require()
        bridge = DispatchBridge(
            tools=tools,
            ctx=ctx,
            execution=run.execution,
            session=run.session,
            token=(run.signal or CancelToken()).child(),
            max_parallel=config.max_parallel_sub_calls,
            max_dispatches=config.max_dispatches_per_run,
            max_spawns=config.max_subagent_spawns_per_run,
        )
        namespaces = await _namespaces(tools, CodeBindingsRequest(scope=run.scope, bridge=bridge))
        outcome = await runtime.run(
            CodeRunRequest(
                program=program,
                bindings=namespaces,
                namespace=getattr(run.agent, "id", None),
                cancel_scope=bridge.token,
            )
        )
        return CodeCellValue(
            logs=outcome.logs,
            value=outcome.value,
            error=outcome.error,
            dispatches=bridge.dispatch_count,
            truncated=outcome.truncated,
            reset=outcome.reset,
            displays=list(outcome.displays),
        ).to_wire()

    tools.register_transport(
        define_tool(
            RUN_CODE,
            "Run a program. Every capability is reached from inside it; see the SDK below.",
            parameters={
                "type": "object",
                "properties": {"program": {"type": "string"}},
                "required": ["program"],
            },
            output=ToolOutput(
                schema=CodeCellValue,
                render=lambda _args, value: text_content(_render_run(value)),
            ),
            execute=run_code,
            # A cell is model-authored raw Python, and this is the tool a profile
            # renames — so a name-keyed gate is inert exactly here (P6-16).
            is_irreversible=True,
        )
    )

    async def sdk_section(request: Any) -> str:
        scope: Context = request.scope
        runtime = ctx.code_runtime.provider
        if runtime is None:
            return ""
        language = getattr(runtime, "language", "python")
        renderer = ctx.code_runtime.sdk_renderer(language)
        if renderer is None:
            raise RuntimeError(
                f'no tools:sdk renderer for language "{language}"; Code Mode cannot describe '
                "its own surface, and a listing in the wrong syntax would make the model "
                "write code that cannot run"
            )
        transport = tools.view(scope).transport_name
        described = await _namespaces(tools, CodeBindingsRequest(scope=scope))
        return f"{code_only_rule(transport)}\n\n{renderer(described)}"

    ctx.system_prompt.section(
        PromptSection(name="tools:sdk", order=ORDER_TOOL_GUIDANCE, text=sdk_section)
    )


class CodeCellValue(WireModel):
    """The transport's return value, typed so a field cannot be silently dropped.

    Six untyped dict keys read back through `.get()` with defaults is how
    `truncated` and `displays` went missing without a failing test: an omitted
    key is indistinguishable from its default. Constructed *from* the
    `CodeRunResult`, so a field added to the result but not carried here is a
    visible type-level decision — and the transport's `ToolOutput.schema` is
    this model, so `render()` actually validates the value it renders.
    """

    logs: str = ""
    value: Any = None
    error: str | None = None
    dispatches: int = 0
    truncated: bool = False
    reset: bool = False
    displays: list[dict[str, Any]] = Field(default_factory=list)


async def _namespaces(
    tools: ToolRuntime, request: CodeBindingsRequest
) -> tuple[CodeBindingNamespace, ...]:
    """Every namespace this program may reach, `tools` first.

    `tools` is built here rather than registered because it is not optional:
    Code Mode without the registry as a namespace is a transport to nowhere. The
    rest come from `register_code_namespace` claims, read off the same view the
    tool surface is — name conflicts already failed at registration.
    """
    view = tools.view(request.scope)
    # Contributed namespaces first, so `tools` can be built knowing which tools
    # already have a namespaced face. The order the *program* sees is still
    # `tools` first — it is the one namespace that is never optional.
    contributed: list[CodeBindingNamespace] = []
    presented: set[str] = set()
    for name in sorted(view.code_namespaces):
        namespace = await maybe_await(view.code_namespaces[name](request))
        if namespace.name != name:
            raise RuntimeError(
                f'the factory registered for code binding namespace "{name}" returned one '
                f'named "{namespace.name}"; the SDK block and the dispatch table would disagree'
            )
        contributed.append(namespace)
        presented.update(
            binding.presents for binding in namespace.bindings if binding.presents is not None
        )
    return (_tools_namespace(tools, request.scope, request.bridge, presented), *contributed)


def _tools_namespace(
    tools: ToolRuntime,
    scope: Context,
    bridge: DispatchBridge | None,
    presented: set[str] = frozenset(),  # type: ignore[assignment]
) -> CodeBindingNamespace:
    """The `tools` namespace: every visible tool, as a governed binding.

    With no bridge the namespace is descriptive — for rendering the SDK — and
    carries no dispatch closures.
    """
    view = tools.view(scope)
    bindings: list[CodeBinding] = []
    for name in sorted(view.visible):
        # `transport_name`, not `RUN_CODE`: under a profile that renames it the
        # transport is visible under the new name, and binding it into `tools`
        # would hand the program a way to re-enter itself.
        if name == view.transport_name:
            continue
        # A tool another namespace already presents (`rlm.run` for `rlm_run`),
        # as declared by that namespace's own bindings. Still dispatchable and
        # still addressable by policy — just not offered twice, under two names,
        # in one SDK block.
        if name in presented:
            continue
        definition = view.visible[name]
        binding = CodeBinding(
            name=name, description=definition.description, parameters=definition.parameters
        )
        if bridge is not None:
            binding = CodeBinding(
                name=name,
                description=definition.description,
                parameters=definition.parameters,
                dispatch=partial(_dispatch, bridge, binding),
            )
        bindings.append(binding)
    return CodeBindingNamespace(
        name="tools",
        description="governed capabilities; every call is recorded and may be refused",
        bindings=tuple(bindings),
    )


async def _dispatch(bridge: DispatchBridge, binding: CodeBinding, **arguments: Any) -> Any:
    return await bridge.call(binding, arguments)


def governed_binding(
    request: CodeBindingsRequest,
    public_name: str,
    definition: Any,
    *,
    counts_as_spawn: bool = False,
) -> CodeBinding:
    """One registered tool, as a namespaced binding a program may await.

    The one place that knows the protocol every namespace has to follow: the
    binding the *program* writes may be named differently from the tool it
    dispatches (`rlm.run` for `rlm_run`), the dispatch closure is omitted when
    the namespace is being described rather than bound, and the binding handed to
    the bridge carries the *tool's* name so `tool/code-dispatch-start` records
    the governed capability rather than the alias.
    """
    governed = CodeBinding(
        name=definition.name,
        description=definition.description,
        parameters=definition.parameters,
        counts_as_spawn=counts_as_spawn,
    )
    bridge = request.bridge
    dispatch = None if bridge is None else partial(_dispatch, bridge, governed)
    return replace(governed, name=public_name, dispatch=dispatch, presents=definition.name)


def _render_run(value: Any) -> str:
    parts: list[str] = []
    logs = value.get("logs") or ""
    if logs:
        parts.append(logs.rstrip())
    if value.get("error"):
        parts.append(f"[error]\n{value['error']}")
    result = value.get("value")
    if result is not None:
        parts.append(f"[result] {result!r}")
    return "\n".join(parts) if parts else "(no output)"
