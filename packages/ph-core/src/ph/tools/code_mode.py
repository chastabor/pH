"""Code Mode: one transport, and every binding call governed (C1-C3).

The whole containment argument depends on this file. A `run_code` cell is *one*
tool call. But a cell that writes forty files must not be *one* governance
evaluation. The feature map records that collapse in prime-agent, where
`ToolName = "ipython"` reduces permission, approval, call limits, offload and
`fs/write-intent` to a single decision per cell.

So the bridge enters the **complete** pipeline again for each binding call:

* Each `await tools.<name>(...)` is a sub-call. Its `parent` is the opaque
  token of the outer execution. It is dispatched through `tools/pre-execute` →
  approval → guards → `tools/execute` → `tools/body` → body →
  `tools/post-execute` (C1).
* Each sub-call logs `tool/code-dispatch-start` at entry and
  `tool/code-dispatch` at settle. So forty writes are forty durable records and
  not one stdout blob (C2). Both records are log-only. `derive_messages()`
  ignores them, so sub-calls never enter the model context again.
* A **denial fails the whole run** (C3, Q9). This is the one deliberate
  divergence of pH from dsh. A program that can `except` a refusal can route
  around it: it can retry with a different path, or fall back to `subprocess`.
  Failing the run puts the refusal in the context of the model, and limits the
  partial state to *about* one cell. The limit is best-effort in time, because
  the abort is a signal ladder that stops the program at its next yield point
  (see `Kernel.cancel_grace` in `ph_rlm.kernel.manager`). A *failed* call
  (timeout, bad arguments) keeps the `ToolCallError` semantics of dsh, because
  the model is the one that handles that failure.

Budgets (C4) are enforced here and not by the runtime. One approved cell must
not be able to issue unbounded governed calls, and the bridge is the only place
that sees all of them.

@module ph.tools.code_mode
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, ClassVar, Literal

import anyio
from pydantic import Field

from ..cancel import CancelToken
from ..cordis import Context, events, maybe_await, plugin
from ..json import thaw_json
from ..keys import CODE_RUNTIME, SYSTEM_PROMPT, TOOLS
from ..llm.types import ContentBlock, ToolSchema
from ..seams.code_runtime import CodeBinding, CodeBindingNamespace, CodeRunRequest
from ..session import Claim, IntentJournal, Session, intents_of
from ..session.json import freeze_json_value
from ..session.kinds import TOOL_DISPATCH
from ..system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ..wire import WireModel
from .definition import (
    ToolExecution,
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

A cell that the human approved is still one decision. Without a cap, that one
approval would let the cell issue unbounded governed calls."""

MAX_SUBAGENT_SPAWNS_PER_RUN = 32
"""How many children one cell may spawn (C4). Each call of a binding that
declares `counts_as_spawn` counts as one spawn."""

events.declare(
    "tools/code-dispatch-log",
    "waterfall",
    owner="ph.tools.code_mode",
    doc="One settled sub-dispatch about to be logged; a listener may reshape its content here.",
)


class ToolCallError(HarnessError):
    """A sub-call *failed*. Raised inside the program, for the model to handle."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"tools.{name} failed: {message}", "TOOL_CALL_FAILED")
        self.tool_name = name


class CodeRunFailure(HarnessError):
    """The whole run is over.

    `kind: "denied"` is the C3 case. The program is not allowed to catch this
    refusal, because a program that catches a refusal can route around policy.

    All three kinds end the run. They do not mean the same thing to a consumer,
    so each kind declares its own `failure_kind`. A budget is the tool failing
    (the program asked for too much), a denial is policy, and an abort is
    cancellation.
    """

    _FAILURE_KINDS: ClassVar[dict[str, FailureKind]] = {
        "denied": "denied",
        "budget": "failed",
        "aborted": "aborted",
    }

    def __init__(
        self,
        kind: Literal["denied", "budget", "aborted"],
        message: str,
        *,
        concludes_turn: bool = False,
    ) -> None:
        super().__init__(message, f"CODE_RUN_{kind.upper()}")
        self.kind = kind
        self.failure_kind = self._FAILURE_KINDS[kind]
        self.concludes_turn = concludes_turn
        """Set when the sub-result that stopped the run ended the turn (C13).

        No `kind` implies it. `max_dispatches` and `max_spawns` also settle as
        `budget`, and they end only the program. That is the difference between
        the ceiling of a cell and the ceiling of the turn."""


class CodeDispatchRef(WireModel):
    """Which dispatch this is: the identity that its two records share.

    A dispatch writes two records: `tool/code-dispatch-start` before the body
    of the binding runs (B4, P7-15), and `tool/code-dispatch` when it settles.
    **Readers pair the two records by these fields**, and by nothing else. The
    TUI adapter looks up the parent card by `parentCallId` and the dispatch card
    by `subCallId`. `/revert` reads `parentCallId` and `name` from the start
    record, to list what a run did outside the tree that it restored.

    Both records use this one type, so the pairing cannot drift. The start
    record was at first a hand-written camelCase dict, and the settle went
    through `to_wire()`. Four fields were spelled twice. A rename of one field,
    or a change to `wire_alias`, moved the settle record and left the literals
    of the start record behind. Every reader then lost the pairing at the same
    time, and nothing failed.
    """

    root_call_id: str
    parent_call_id: str
    sub_call_id: str
    name: str


class CodeDispatchLog(CodeDispatchRef):
    """One settled sub-dispatch, as the log waterfall sees it."""

    is_error: bool


# `TOOL_DISPATCH` is the pair that the two records of a dispatch make. It is
# declared in `ph.session.kinds` with the other kinds of ph-core (T4), together
# with `DISPATCH_INTERRUPTED`, the body that its repaired settle carries. That
# leaf module spells the wire keys of `CodeDispatchRef` (`DISPATCH_REF_KEYS`),
# because it cannot import this model. A test holds the two equal.


@dataclass(frozen=True, slots=True)
class CodeBindingsRequest:
    """What a namespace factory is told about the run that it binds for.

    `bridge is None` means that the namespace is being *described* and not
    bound. The SDK prompt section asks the factories the same question that the
    run asks. So the block cannot list a namespace that the program could not
    reach, or omit one that it can reach. There are only two fields. The bridge
    already carries the execution and the session, and a second copy of either
    would be one more thing on which the two construction sites could disagree.
    """

    scope: Context
    bridge: DispatchBridge | None = None


@dataclass(slots=True)
class DispatchBridge:
    """Turns binding calls inside one program into governed sub-calls."""

    tools: ToolRuntime
    ctx: Context
    execution: ToolExecution
    session: Session | None
    token: CancelToken
    max_parallel: int = 10
    max_dispatches: int = MAX_DISPATCHES_PER_RUN
    max_spawns: int = MAX_SUBAGENT_SPAWNS_PER_RUN
    _dispatched: int = 0
    _spawned: int = 0
    _failure: CodeRunFailure | None = None
    _concluded_turn: bool = False
    """Whether a sub-result ended the *outer* turn, and not only this program (C13).

    That the program stopped and that the turn ended are two facts, and at first
    only the first fact reached the loop. `run_code` returns an ordinary
    successful cell value. So a ceiling that ended the turn inside a program
    left the loop running, and the model spent another step to discover it.
    `run_code` reads this flag. It is the only code that holds both this bridge
    and the `ToolRunContext` that can report the fact."""
    _limiter: anyio.CapacityLimiter = field(init=False)
    _journal: IntentJournal = field(init=False)
    _started: dict[str, Claim] = field(init=False, default_factory=dict)
    """The intent of each dispatch, by sub-call id, from its start to its settle."""

    def __post_init__(self) -> None:
        self._limiter = anyio.CapacityLimiter(max(1, self.max_parallel))
        self._journal = intents_of(self.ctx)

    @property
    def dispatch_count(self) -> int:
        return self._dispatched

    @property
    def concluded_turn(self) -> bool:
        """Whether a sub-call's result ended the turn this program runs in."""
        return self._concluded_turn

    def _settle(
        self,
        kind: Literal["denied", "budget", "aborted"],
        message: str,
        *,
        concludes_turn: bool = False,
    ) -> CodeRunFailure:
        # After the run is settled, later awaits stop immediately. They do not
        # race to do more work that the outcome has already discarded.
        self._failure = CodeRunFailure(kind, message, concludes_turn=concludes_turn)
        return self._failure

    async def call(self, binding: CodeBinding, arguments: object) -> Any:  # noqa: ANN401
        """Dispatch one binding call through the full pipeline.

        :raises ToolCallError: the call failed. The program may handle it.
        :raises CodeRunFailure: the call was denied or a budget was reached.
            The program may not handle it.
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

        # The id is deterministic and ordered. So a reader can pair a start with
        # its settle, and can see the submission order without a timestamp.
        sub_call_id = f"{self.execution.call_id}:code:{self._dispatched}"
        self._dispatched += 1
        # One lossless snapshot proves that the arguments are JSON.
        # `create_execution` accepts the frozen form directly, so the snapshot
        # is not thawed again for the call.
        snapshot = freeze_json_value(arguments, frozen_input=True)

        ref = CodeDispatchRef(
            root_call_id=self.execution.root_call_id,
            parent_call_id=self.execution.call_id,
            sub_call_id=sub_call_id,
            name=binding.name,
        )

        async with self._limiter:
            try:
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
            except BaseException:
                self._abandon(ref)
                raise

        failure = result.error
        # **A result that ends the turn ends the program** (D7). This check
        # comes first because it is the broadest of the three. A ceiling that
        # stops the turn at first arrived as an ordinary `ToolCallError`, which
        # a generated program can `except`. So the posture that was meant to
        # report a breach most loudly was the one that a program could
        # suppress. The program then spent the rest of the turn on calls that
        # could only fail.
        #
        # The check reads `concludes_turn` and not a second failure kind,
        # because the loop itself stops on that flag. Whatever ends the turn
        # outside ends the run in here, and this code needs no list of the rows
        # that can do it. The check is above `is_error` for the same reason.
        # `conclude_turn` is public on `ToolRunContext`, and a *successful*
        # "you are done here" result is exactly what a tool will use it for.
        if result.concludes_turn:
            self._concluded_turn = True
            stopped = failure.message if failure is not None else "the turn was concluded"
            raise self._settle(
                "budget",
                f"tools.{binding.name} stopped the turn: {stopped}. The program was "
                "stopped; nothing it called after this could have succeeded.",
                concludes_turn=True,
            )
        if result.is_error:
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
        """Record the dispatch after the pipeline has decided and before it runs.

        This is the same point at which `execute_tool_calls` writes `tool/call`
        (B4, P7-15). A dispatch that is parked on an approval is not yet
        something that the program *did*, and `/revert` reads these records as
        the list of what the program did.

        The record takes `execution.arguments` unconditionally. `create_execution`
        sets it from the snapshot that this dispatch passed in. Only the
        substitution of approval ever replaces it, and that is exactly what this
        record must hold. (The ternary in `batch.py` has the same shape but
        cannot be removed. There the unsubstituted branch is the original bytes
        of the model, which is a different object.)

        **The record also says whether a workspace restore takes the call back**
        (`restoreCovered`, S4). The value is the answer that `prepare` gave for
        the call that is about to run. `/revert` reads this value and does not
        ask again after its restore. At that time, a link that the run made and
        the restore removed would hide a write that left the tree.
        """
        if self.session is None:
            return
        execution = prepared.run.execution
        self._started[ref.sub_call_id] = self._journal.record(
            self.session,
            TOOL_DISPATCH,
            {
                **ref.to_wire(),
                "arguments": thaw_json(execution.arguments),
                "restoreCovered": execution.restore_covered,
            },
        )

    def _abandon(self, ref: CodeDispatchRef) -> None:
        """Settle as `outcome-unknown` a dispatch whose own settle will not come,
        in the way that `journal.claim` settles its own intent (S15). The causes
        are a cancel, or a raise in the pipeline or in the log waterfall.

        Left open, the dispatch read as running in the live log, and so did its
        card, until the repair of a restart closed it. The method is synchronous,
        so it completes inside a canceled scope. It does nothing when the start
        was never recorded or is already settled.
        """
        started = self._started.pop(ref.sub_call_id, None)
        if started is not None and self.session is not None:
            self._journal.abandon(self.session, started, "outcome-unknown")

    async def _log_settle(self, ref: CodeDispatchRef, result: ToolExecutionResult) -> None:
        async def inner(
            record: CodeDispatchLog, blocks: Sequence[ContentBlock]
        ) -> Sequence[ContentBlock]:
            return blocks

        record = CodeDispatchLog(**ref.__dict__, is_error=result.is_error)
        # A spill policy can replace an oversized dispatch content *individually*
        # here (C5), so that one large read is not merged with its siblings
        # into one blob.
        shaped = await self.ctx.waterfall(
            "tools/code-dispatch-log", record, result.content, inner=inner
        )
        if self.session is None:
            return
        data = {**record.to_wire(), "content": [block.to_wire() for block in shaped]}
        # The intent is always open here. The pipeline writes the start for
        # every call that its gate decides, a refusal included
        # (`ToolRuntime.prepare`, P7-15). So a settle always has its intent. A
        # settle without one would be a bug to report, and not a record to
        # write on its own.
        started = self._started.pop(ref.sub_call_id)
        self._journal.settle(self.session, started, data)


class Config(WireModel):
    """The row config for the Code Mode transport."""

    max_dispatches_per_run: int = MAX_DISPATCHES_PER_RUN
    max_subagent_spawns_per_run: int = MAX_SUBAGENT_SPAWNS_PER_RUN
    max_parallel_sub_calls: int = 10


@plugin(
    "tools-code-mode",
    affects="environment",
    config=Config,
    inject=[TOOLS, CODE_RUNTIME, SYSTEM_PROMPT],
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the reserved transport, the shipped SDK renderers, and the prompt section."""
    tools: ToolRuntime = ctx.require(TOOLS)
    ctx.require(CODE_RUNTIME).register_sdk_renderer("python", render_python_sdk)
    ctx.require(CODE_RUNTIME).register_sdk_renderer("typescript", render_typescript_sdk)

    async def run_code(args: object, run: ToolRunContext) -> dict[str, Any]:
        # `Mapping`, not `dict`: accepted arguments are frozen into a
        # `MappingProxyType`, which is a Mapping but not a dict instance.
        program = args.get("program") if isinstance(args, Mapping) else None
        if not isinstance(program, str) or not program.strip():
            raise ToolCallError(run.name, "program must be a non-empty string")
        # The namespace of the run is the id of the calling agent. So a call
        # with no agent has nowhere to put its bindings.
        if run.agent is None:
            raise ToolCallError(run.name, "this tool has to be called by an agent")
        runtime = ctx.require(CODE_RUNTIME).require()
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
                agent=run.agent.id,
                bindings=namespaces,
                namespace=run.agent.id,
                token=bridge.token,
            )
        )
        if bridge.concluded_turn:
            # **The other way out** (C13). A `CodeRunFailure` that escapes this
            # function carries `concludes_turn` on itself, and
            # `registry._failure` reads it. That is the path when the bridge
            # raises into the caller. Over the kernel wire the failure does not
            # escape. The dispatch fails, the program dies in the guest, and the
            # runtime *returns* an outcome that holds the error. Then only this
            # bridge still holds the fact, and `ToolRunContext.conclude_turn`
            # is the way that the loop provides to mark a successful result as
            # terminal.
            run.conclude_turn()
        return CodeCellValue(
            logs=outcome.logs,
            value=outcome.value,
            error=outcome.error,
            dispatches=bridge.dispatch_count,
            truncated=outcome.truncated,
            dropped=outcome.dropped,
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
            # A cell is raw Python that the model wrote, and this is the tool
            # that a profile renames. So a gate that is keyed by name has no
            # effect exactly here (P6-16).
            is_irreversible=True,
        )
    )

    async def sdk_section(request: AssembleContext) -> str:
        scope: Context = request.scope
        runtime = ctx.require(CODE_RUNTIME).provider
        if runtime is None:
            return ""
        language = runtime.language
        renderer = ctx.require(CODE_RUNTIME).sdk_renderer(language)
        if renderer is None:
            raise RuntimeError(
                f'no tools:sdk renderer for language "{language}"; Code Mode cannot describe '
                "its own surface, and a listing in the wrong syntax would make the model "
                "write code that cannot run"
            )
        transport = tools.view(scope).transport_name
        described = await _namespaces(tools, CodeBindingsRequest(scope=scope))
        return f"{code_only_rule(transport)}\n\n{renderer(described)}"

    ctx.require(SYSTEM_PROMPT).section(
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
    dropped: int = 0
    reset: bool = False
    displays: list[dict[str, Any]] = Field(default_factory=list)


async def _namespaces(
    tools: ToolRuntime, request: CodeBindingsRequest
) -> tuple[CodeBindingNamespace, ...]:
    """Every namespace that this program may reach, with `tools` first.

    `tools` is built here and not registered, because it is not optional. Code
    Mode without the registry as a namespace is a transport to nowhere. The
    other namespaces come from `register_code_namespace` claims. They are read
    from the same view as the tool surface, and name conflicts already failed
    at registration.
    """
    view = tools.view(request.scope)
    # The contributed namespaces are built first, so that `tools` can be built
    # with the knowledge of which tools another namespace already presents. The
    # *program* still sees `tools` first, because it is the one namespace that
    # is never optional.
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
    presented: AbstractSet[str] = frozenset(),
) -> CodeBindingNamespace:
    """The `tools` namespace: every visible tool, as a governed binding.

    With no bridge, the namespace only describes the tools, for rendering the
    SDK. It carries no dispatch closures.
    """
    view = tools.view(scope)
    bindings: list[CodeBinding] = []
    for name in sorted(view.visible):
        # Compare with `transport_name`, not `RUN_CODE`. Under a profile that
        # renames the transport, it is visible under the new name. A binding
        # for it in `tools` would give the program a way to enter itself again.
        if name == view.transport_name:
            continue
        # Skip a tool that another namespace already presents (`rlm.run` for
        # `rlm_run`), as the bindings of that namespace declare. The tool can
        # still be dispatched, and policy can still address it. It is only not
        # offered two times, under two names, in one SDK block. The parameter
        # is an `AbstractSet` and not a `Container` because this is one
        # membership test per visible tool. A `list` would satisfy `Container`
        # and would make the loop O(n·m) without a warning.
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


async def _dispatch(
    bridge: DispatchBridge,
    binding: CodeBinding,
    **arguments: object,
) -> Any:  # noqa: ANN401
    return await bridge.call(binding, arguments)


def governed_binding(
    request: CodeBindingsRequest,
    public_name: str,
    definition: ToolSchema,
    *,
    counts_as_spawn: bool = False,
) -> CodeBinding:
    """One registered tool, as a namespaced binding that a program may await.

    This is the one place that knows the protocol that every namespace must
    follow:

    * The binding that the *program* writes may have a different name from the
      tool that it dispatches (`rlm.run` for `rlm_run`).
    * The dispatch closure is omitted when the namespace is being described and
      not bound.
    * The binding that is passed to the bridge carries the name of the *tool*,
      so that `tool/code-dispatch-start` records the governed capability and
      not the alias.
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


def _render_run(value: Any) -> str:  # noqa: ANN401
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
