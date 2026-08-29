"""`ctx.tools` — the registry, the visibility rules, and the pipeline.

Three separable things live here because they share one traversal:

* **registration** — global or per-agent, with scoped shadowing by name (B7);
* **visibility** — restrictions intersect over *global* names only, so a
  restriction can never silence a tool an agent registered for itself;
* **the pipeline** — `tools/pre-execute` → approval on `ask` → monotonic guards
  → `tools/execute` (around) → body → `tools/post-execute` → normalize →
  `finalize_content` → `tools/result` (B1-B5).

**Ordering note.** dsh runs guards *after* approval
(`docs/tool-execution-pipeline.md`, and `prepareExecution` in
`packages/core/tools/src/index.ts`), and pH follows it. The pH plans' summary
tables list guards before approval; that is a transcription slip, and the
difference matters: a guard is deny-only and runs last, so it is the final word
even over a human's explicit approval. That is what "monotonic" buys — policy
that must not be reorderable stays a guard rather than a listener.

@module ph.tools.registry
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..cancel import Cancelled, is_cancelled
from ..cordis import Context, Disposer, events, plugin
from ..llm.types import ToolSchema, text_of
from ..seams.approval import Edited, Responded, denial_reason
from ..seams.code_runtime import CodeBindingNamespace, validate_binding_name
from ..session.json import freeze_json_value, thaw_json
from ..wire import WireModel
from .definition import (
    Accept,
    Allow,
    Ask,
    Block,
    Deny,
    ExecutionMode,
    FailureKind,
    PostToolDecision,
    Respond,
    ToolDefinition,
    ToolExecution,
    ToolExecutionInput,
    ToolExecutionResult,
    ToolFailure,
    ToolOutput,
    ToolRunContext,
    TransportPresentation,
    aborted_result,
    denied_result,
    error_result,
    text_content,
)
from .errors import HarnessError, ToolNotFoundError, error_info, error_message

__all__ = [
    "RUN_CODE",
    "CodeNamespaceFactory",
    "PreparedCall",
    "ToolGuard",
    "ToolRestriction",
    "ToolRuntime",
    "apply",
]

log = logging.getLogger("ph.tools")

RUN_CODE = "run_code"
"""The reserved Code Mode transport name.

Reserved in *every* mode and unregisterable, unshadowable and unrestrictable:
if a deployment could occupy the name, a model told to call `run_code` would
reach something else (P1-04). Only `register_transport` may claim it."""

PresentationMode = Literal["native", "code", "both"]

ToolGuard = Callable[[ToolExecution], str | None]
"""A monotonic guard: a reason denies, `None` abstains."""

CodeNamespaceFactory = Callable[[Any], "CodeBindingNamespace | Awaitable[CodeBindingNamespace]"]
"""Builds one binding namespace for one question — a live run, or the SDK block.

The argument is a `CodeBindingsRequest` (`ph.tools.code_mode`), typed `Any` here
because the registry must not import the module that consumes it."""


"""A monotonic guard: a reason denies, `None` abstains.

There is no allow result *by construction*, which is what makes listener order
unable to turn a denial back into permission (B2)."""

events.declare(
    "tools/pre-execute",
    "waterfall",
    owner="ph.tools",
    doc="Pre-dispatch policy: allow, deny or ask. Hooks, permissions, sandbox.",
)
events.declare(
    "tools/execute",
    "waterfall",
    owner="ph.tools",
    doc="Around the tool body: timeouts, retries, metrics. May replace the signal.",
)
events.declare(
    "tools/post-execute",
    "waterfall",
    owner="ph.tools",
    doc="Post-dispatch policy: accept, replace a projection, or block with feedback.",
)
events.declare(
    "tools/result", "emit", owner="ph.tools", doc="The frozen authoritative outcome of one call."
)
events.declare(
    "tools/change",
    "emit",
    owner="ph.tools",
    doc="The visible tool set changed; consumers re-read schemas().",
)


# One reason per non-grant, distinct on purpose: a model must be able to tell a
# human's "no" from a missing channel, because only one is worth re-planning
# around and the other is a misconfiguration to report.
@dataclass(frozen=True, slots=True)
class ToolRestriction:
    """A per-scope filter over global tools. Restrictions intersect."""

    allow: frozenset[str] | None = None
    deny: frozenset[str] | None = None

    def admits(self, name: str) -> bool:
        if self.allow is not None and name not in self.allow:
            return False
        return not (self.deny is not None and name in self.deny)


@dataclass(slots=True)
class _Layer:
    """One scope's registry contribution."""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    restrictions: list[ToolRestriction] = field(default_factory=list)
    guards: list[ToolGuard] = field(default_factory=list)
    mode: PresentationMode | None = None
    """The presentation this scope declared for itself.

    One cell, not a list: two answers to "which form does the model see" is a
    contradiction rather than something to merge."""
    transport: TransportPresentation | None = None
    """How this scope names and describes the transport. One cell for the same
    reason `mode` is: the model is offered exactly one callable."""
    code_namespaces: dict[str, CodeNamespaceFactory] = field(default_factory=dict)
    """Binding namespaces this scope contributes to Code Mode runs (P3-10)."""

    def empty(self) -> bool:
        return (
            not (self.tools or self.restrictions or self.guards or self.code_namespaces)
            and self.mode is None
            and self.transport is None
        )

    def admits(self, name: str) -> bool:
        return all(restriction.admits(name) for restriction in self.restrictions)


@dataclass(frozen=True, slots=True)
class _View:
    """One scope's resolved registry, from a single layer traversal."""

    visible: dict[str, ToolDefinition]
    mode: PresentationMode
    schemas: tuple[ToolSchema, ...]
    """The model-facing schemas, or empty under Code Mode — where the model is
    offered one callable and reaches the rest through the SDK (P1-04)."""
    transport_name: str
    """What this scope's model calls the transport.

    Every check that used to compare against `RUN_CODE` compares against this
    instead — the C6 refusal, the route-back text, the `tools` namespace that
    must not bind the transport to itself, the code-only rule in the prompt.
    Reading it from the view keeps those four in agreement by construction."""
    code_namespaces: dict[str, CodeNamespaceFactory]
    """The contributed binding namespaces this scope's programs may reach,
    shadowed by name like tools. Resolved with the view so the run and the SDK
    block read one answer."""


@dataclass(frozen=True, slots=True)
class PreparedCall:
    """One call between pipeline stages.

    `result is None` means the body has not run and `dispatch()` is next.
    Otherwise `needs_post` says whether `tools/post-execute` still applies: a
    denial is a result policy may still shape, while a pipeline *failure* bypasses
    it — a listener that raised must not get to reshape the error its own raise
    produced.
    """

    run: ToolRunContext
    result: ToolExecutionResult | None = None
    needs_post: bool = True


class Config(WireModel):
    """Row config for `ph.tools`.

    A `WireModel` like every other JSON boundary: a row is authored as YAML in
    camelCase and read as snake_case in Python, under the one alias rule (Q2).
    `extra="forbid"` turns a mistyped key into a startup error rather than a
    silently ignored setting.
    """

    mode: PresentationMode = "native"


@dataclass(slots=True)
class ToolRuntime:
    """The service published as `ctx.tools`."""

    ctx: Context
    default_mode: PresentationMode = "native"
    _layers: dict[Context | None, _Layer] = field(default_factory=dict)
    _generation: int = 0
    """Bumped on every mutation. The view cache is valid for one generation."""
    _views: dict[tuple[Context | None, ...], tuple[int, _View]] = field(default_factory=dict)

    # ---------------------------------------------------------- registration --

    def _layer(self, key: Context | None) -> _Layer:
        layer = self._layers.get(key)
        if layer is None:
            layer = self._layers[key] = _Layer()
        return layer

    def _changed(self) -> None:
        self._generation += 1
        self._views.clear()
        self.ctx.emit("tools/change")

    def _release(self, key: Context | None, undo: Callable[[], Any]) -> None:
        undo()
        layer = self._layers.get(key)
        if layer is not None and layer.empty():
            del self._layers[key]
        self._changed()

    def _claim(
        self,
        owner: Context,
        mutate: Callable[[_Layer], None],
        undo: Callable[[_Layer], Any],
        label: str,
    ) -> Disposer:
        """Mutate the owner's layer and hand back the disposer that undoes it."""
        key = owner.isolation
        layer = self._layer(key)
        mutate(layer)
        self._changed()
        return owner.add_disposer(lambda: self._release(key, lambda: undo(layer)), label=label)

    def register(self, definition: ToolDefinition, *, scope: Context | None = None) -> Disposer:
        """Register a tool globally, or on an agent's scope to shadow by name."""
        if definition.name == RUN_CODE:
            raise ValueError(
                f'"{RUN_CODE}" is the reserved Code Mode transport and cannot be registered'
            )
        # A chain walk over the layer cells, not `view()`: registration
        # invalidates the view cache one line later, so building one here made
        # every mount O(N^2) in throwaway schema construction.
        if definition.name == self._presented_name(scope or self.ctx):
            raise ValueError(
                f'"{definition.name}" is how this profile presents the Code Mode transport '
                "and cannot be registered"
            )
        return self._register(definition, scope)

    def register_transport(
        self, definition: ToolDefinition, *, scope: Context | None = None
    ) -> Disposer:
        """Claim the reserved transport name. The one caller is the Code Mode row.

        Same registration path as every other tool — shadowing rules, the
        `tools/change` emit, layer cleanup — minus the reservation check that
        exists to keep everyone *else* off the name.
        """
        if definition.name != RUN_CODE:
            raise ValueError(f'register_transport is for "{RUN_CODE}", not "{definition.name}"')
        return self._register(definition, scope)

    def _register(self, definition: ToolDefinition, scope: Context | None) -> Disposer:
        owner = scope or self.ctx
        key = owner.isolation

        def add(layer: _Layer) -> None:
            if definition.name in layer.tools:
                where = (
                    "already registered globally (for a per-agent variant, register "
                    "through that agent's scope instead)"
                    if key is None
                    else "already registered in this scope"
                )
                raise ValueError(f'tool "{definition.name}" is {where}')
            layer.tools[definition.name] = definition

        return self._claim(
            owner,
            add,
            lambda layer: layer.tools.pop(definition.name, None),
            f"tool({definition.name})",
        )

    def restrict(self, restriction: ToolRestriction, *, scope: Context | None = None) -> Disposer:
        """Mask global tools for one scope. Restrictions intersect."""
        return self._claim(
            scope or self.ctx,
            lambda layer: layer.restrictions.append(restriction),
            lambda layer: _discard(layer.restrictions, restriction),
            "tools.restrict",
        )

    def guard(self, guard: ToolGuard, *, scope: Context | None = None) -> Disposer:
        """Register a monotonic deny-only guard. It runs last, and it is final."""
        return self._claim(
            scope or self.ctx,
            lambda layer: layer.guards.append(guard),
            lambda layer: _discard(layer.guards, guard),
            "tools.guard",
        )

    def present_as(self, mode: PresentationMode, *, scope: Context | None = None) -> Disposer:
        """Declare the presentation for one agent, shadowing the deployment default."""

        def clear(layer: _Layer) -> None:
            layer.mode = None

        return self._claim(
            scope or self.ctx, lambda layer: setattr(layer, "mode", mode), clear, "tools.present_as"
        )

    def present_transport(
        self, presentation: TransportPresentation, *, scope: Context | None = None
    ) -> Disposer:
        """Name and describe the transport for one scope (P3-09).

        Refused when the name is already a visible tool: the whole point of the
        reservation is that a model told to call this name reaches the transport,
        and a silent shadow either way would defeat it. The mirror check lives in
        `register`, so the two orders fail the same way.
        """
        target = scope or self.ctx
        occupant = self.view(target).visible.get(presentation.name)
        if occupant is not None:
            raise ValueError(
                f'cannot present the Code Mode transport as "{presentation.name}": a tool of '
                "that name is already visible here, and the transport name must be "
                "unshadowable (C6)"
            )

        def add(layer: _Layer) -> None:
            if layer.transport is not None:
                # A real claim, unlike a silent overwrite: with two presentations
                # on one cell, the first disposal would clear what the second
                # still wants, and the survivor would depend on disposal order.
                raise ValueError(
                    "this scope already presents the Code Mode transport as "
                    f'"{layer.transport.name}"'
                )
            layer.transport = presentation

        def clear(layer: _Layer) -> None:
            if layer.transport is presentation:
                layer.transport = None

        return self._claim(target, add, clear, "tools.present_transport")

    def register_code_namespace(
        self, name: str, factory: CodeNamespaceFactory, *, scope: Context | None = None
    ) -> Disposer:
        """Claim one Code Mode binding namespace for this scope (P3-10).

        The factory answers one question, asked twice with the same request: by
        the run (with a live bridge) and by the SDK prompt section (with none) —
        so the block cannot describe a namespace a program could not reach. A
        keyed claim rather than a listener, so two rows wanting one name fail
        *here*, at mount, instead of on every cell of a booted deployment; a
        scoped claim shadows a global one by name, like a tool's.

        A namespace that re-presents a registered tool says so on the *binding*
        (`CodeBinding.presents`), not here: the binding list is then the only
        statement of the fact, so it cannot drift from a second list, and a
        namespace cannot suppress a tool it does not actually offer.
        """
        validate_binding_name(name)
        if name == "tools":
            raise ValueError('"tools" is contributed by Code Mode itself and cannot be registered')

        def add(layer: _Layer) -> None:
            if name in layer.code_namespaces:
                raise ValueError(
                    f'code binding namespace "{name}" is already registered in this scope'
                )
            layer.code_namespaces[name] = factory

        return self._claim(
            scope or self.ctx,
            add,
            lambda layer: layer.code_namespaces.pop(name, None),
            f"tools.code_namespace({name})",
        )

    # ------------------------------------------------------------ visibility --

    def _presented_name(self, target: Context) -> str:
        """The transport's presented name, from the layer cells alone."""
        presentation = next(
            (layer.transport for _key, layer in self._chain(target) if layer.transport is not None),
            None,
        )
        return presentation.name if presentation is not None else RUN_CODE

    def _chain(self, target: Context) -> Iterator[tuple[Context | None, _Layer]]:
        for key in target.isolation_chain():
            layer = self._layers.get(key)
            if layer is not None:
                yield key, layer

    def view(self, scope: Context | None = None) -> _View:
        """Resolve what one scope sees, most-specific-first.

        Memoized per isolation chain until the next registry change. The view is
        read several times per tool call and per prompt assembly, and changes
        only when a row registers or disposes something.
        """
        target = scope or self.ctx
        cache_key = tuple(target.isolation_chain())
        cached = self._views.get(cache_key)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        view = self._build_view(target)
        self._views[cache_key] = (self._generation, view)
        return view

    def _build_view(self, target: Context) -> _View:
        visible: dict[str, ToolDefinition] = {}
        layers = list(self._chain(target))
        mode = next((layer.mode for _key, layer in layers if layer.mode is not None), None)
        for index, (key, layer) in enumerate(layers):
            for name, definition in layer.tools.items():
                if name in visible:
                    # A nearer scope already answered this name; scoped
                    # registrations shadow globals rather than merging.
                    continue
                if key is None and not all(outer.admits(name) for _k, outer in layers[:index]):
                    # A restriction filters GLOBAL names only, so an agent's own
                    # registration cannot be masked out from under it.
                    continue
                visible[name] = definition
        resolved_mode = mode if mode is not None else self.default_mode
        presentation = next(
            (layer.transport for _key, layer in layers if layer.transport is not None), None
        )
        transport_name = RUN_CODE
        transport = visible.get(RUN_CODE)
        if presentation is not None and transport is not None:
            if presentation.name != RUN_CODE and presentation.name in visible:
                # The claim-time checks in `register` and `present_transport` are
                # scope-local snapshots; a scoped presentation plus a later
                # parent-scope registration slips past both. This is the one
                # place that resolves every view, so the contradiction fails
                # loudly here instead of silently clobbering either side.
                raise ValueError(
                    f'the Code Mode transport is presented as "{presentation.name}", but a '
                    "tool of that name is also visible in this scope; the transport name "
                    "must be unshadowable (C6)"
                )
            # Renamed in place, so `visible` never holds the transport twice and
            # nothing downstream has to know which name it was registered under.
            del visible[RUN_CODE]
            visible[presentation.name] = presentation.rename(transport)
            transport_name = presentation.name
        code_namespaces: dict[str, CodeNamespaceFactory] = {}
        for _key, layer in layers:
            for name, factory in layer.code_namespaces.items():
                if name not in code_namespaces:
                    code_namespaces[name] = factory
        schemas = (
            ()
            if resolved_mode == "code"
            else tuple(visible[name].schema() for name in sorted(visible))
        )
        return _View(
            visible=visible,
            mode=resolved_mode,
            schemas=schemas,
            transport_name=transport_name,
            code_namespaces=code_namespaces,
        )

    def mode_for(self, scope: Context | None = None) -> PresentationMode:
        return self.view(scope).mode

    def get(self, name: str, *, scope: Context | None = None) -> ToolDefinition | None:
        return self.view(scope).visible.get(name)

    def names(self, *, scope: Context | None = None) -> list[str]:
        return sorted(self.view(scope).visible)

    def schemas(self, *, scope: Context | None = None) -> list[ToolSchema]:
        """The model-facing schemas for one scope.

        Whitelists name/description/parameters, so no internal field — a
        timeout budget, a concurrency classifier — can leak into a request.
        """
        return list(self.view(scope).schemas)

    def guard_reason(self, execution: ToolExecution) -> str | None:
        """The first monotonic denial from every layer this call can see."""
        for _key, layer in self._chain(execution.scope):
            for guard in layer.guards:
                reason = guard(execution)
                if reason is not None:
                    return reason
        return None

    def execution_mode(self, call: ToolExecutionInput) -> ExecutionMode:
        """The live overlap classification, re-read before every start."""
        definition = self.get(call.name, scope=call.scope)
        if definition is None:
            return ExecutionMode(kind="exclusive")
        return definition.classify(call.arguments)

    # --------------------------------------------------------------- pipeline --

    async def execute(self, call: ToolExecutionInput) -> ToolExecutionResult:
        """Run one call through the whole pipeline."""
        prepared = await self.prepare(call)
        if prepared.result is None:
            prepared = await self.dispatch(prepared.run)
        assert prepared.result is not None
        if prepared.needs_post:
            return await self.finalize(prepared.run, prepared.result)
        return self.finish(prepared.run, prepared.result)

    def _execution(self, call: ToolExecutionInput, arguments: Any) -> ToolExecution:
        return ToolExecution(
            call_id=call.call_id,
            root_call_id=call.root_call_id or call.call_id,
            name=call.name,
            arguments=arguments,
            token=object(),
            scope=call.scope or self.ctx,
            session=call.session,
            agent=call.agent,
            parent=call.parent,
            signal=call.cancel,
        )

    def create_execution(self, call: ToolExecutionInput) -> ToolRunContext:
        """Accept a call: resolve the tool, snapshot arguments, mint the token.

        `UNKNOWN_TOOL` is decided here — *before* any policy listener runs — so a
        name the prompt never offered cannot be observed, logged as policy, or
        allowed by a permissive row (C6).
        """
        view = self.view(call.scope)
        definition = view.visible.get(call.name)
        if definition is None:
            if call.name == RUN_CODE and view.transport_name != RUN_CODE:
                # "Not enabled" would be false — it is enabled, under the name
                # this profile presents; say which, so the model can correct.
                raise ToolNotFoundError(
                    call.name,
                    f'the Code Mode transport is presented as "{view.transport_name}" here',
                )
            if call.name in {RUN_CODE, view.transport_name}:
                raise ToolNotFoundError(call.name, "Code Mode is not enabled for this agent")
            raise ToolNotFoundError(call.name)
        if view.mode == "code" and call.parent is None and call.name != view.transport_name:
            raise ToolNotFoundError(
                call.name,
                f"call it from inside {view.transport_name} as `await tools.{call.name}(...)`",
            )
        execution = self._execution(call, freeze_json_value(call.arguments, frozen_input=True))
        return ToolRunContext(execution=execution, definition=definition)

    async def prepare(self, call: ToolExecutionInput) -> PreparedCall:
        """Ordered pre-policy: pre-execute, approval, guards.

        Everything here is awaited in model order by the batch scheduler; only
        `dispatch` overlaps. That is what keeps policy decisions deterministic
        while still letting two slow tools run at once.
        """
        try:
            run = self.create_execution(call)
        except Exception as error:
            placeholder = ToolRunContext(
                execution=self._execution(call, None), definition=_UNRESOLVED
            )
            return PreparedCall(run=placeholder, result=_failure(error), needs_post=False)

        execution = run.execution
        if is_cancelled(execution.signal):
            return PreparedCall(run=run, result=aborted_result(started=False))

        try:

            async def inner(_exec: ToolExecution) -> Allow:
                return Allow()

            gate = await self.ctx.waterfall(
                "tools/pre-execute", execution, inner=inner, scope=execution.scope
            )
            if isinstance(gate, Ask):
                gate = await self._service_ask(execution, gate)
            if isinstance(gate, Respond):
                # Answered in the tool's own voice. A *successful* result,
                # because the model asked a question and got one — a denial it
                # would have had to interpret is the wrong shape for "here is the
                # answer, no need to run it".
                return PreparedCall(
                    run=run,
                    result=ToolExecutionResult(
                        is_error=False, content=tuple(text_content(gate.message))
                    ),
                    needs_post=False,
                )
            if isinstance(gate, Allow) and gate.has_arguments:
                # The substitution lands here, in the one place that owns the
                # execution — after every `tools/pre-execute` listener has seen
                # the call the model actually made.
                execution.arguments = freeze_json_value(gate.arguments, frozen_input=True)
            if is_cancelled(execution.signal):
                return PreparedCall(run=run, result=aborted_result(started=False))
            # Guards run last and only on an allow: a denial already decided,
            # and asking a guard to confirm it would invite a re-permit.
            reason = self.guard_reason(execution) if isinstance(gate, Allow) else gate.reason
            if reason is not None:
                concludes = isinstance(gate, Deny) and gate.concludes_turn
                return PreparedCall(run=run, result=denied_result(reason, concludes_turn=concludes))
            return PreparedCall(run=run)
        except Exception as error:
            return PreparedCall(run=run, result=_failure(error), needs_post=False)

    async def _service_ask(self, execution: ToolExecution, ask: Ask) -> Allow | Deny | Respond:
        """Resolve an `ask` through the approval seam. Fail closed (B3).

        The seam is consumed opportunistically: a deployment that mounts no
        approval service, and an agent-less call with nowhere to route the
        prompt, both deny.

        The two answers that carry data are translated into the pipeline's own
        vocabulary on the way back — `Edited` into an `Allow` that substitutes,
        `Responded` into a `Respond` — so `prepare()`'s switch stays closed over
        the pipeline's own decisions and neither capability is reachable only
        through approval.
        """
        approval = self.ctx.get("approval")
        name = execution.name
        if approval is None:
            return Deny(
                reason=ask.reason or f'tool "{name}" requires approval, which is not available'
            )
        if execution.agent is None:
            return Deny(
                reason=f'tool "{name}" requires approval, but the call has no agent '
                "to route it through"
            )
        outcome = await approval.request(
            agent=execution.agent,
            tool_name=name,
            call_id=execution.call_id,
            reason=ask.reason,
            cancel=execution.signal,
            allowed_decisions=ask.allowed_decisions,
            arguments=thaw_json(execution.arguments),
        )
        if outcome == "allowed-once":
            return Allow()
        if isinstance(outcome, Edited):
            # The human corrected the call rather than refusing it. `tool/call`
            # already recorded what the model asked for and `approval/decided`
            # carries the substitution, so the log holds both and attributes each
            # to whoever made it.
            return Allow(arguments=outcome.arguments, has_arguments=True)
        if isinstance(outcome, Responded):
            return Respond(message=outcome.message)
        return Deny(reason=denial_reason(outcome, f'tool "{name}"'))

    async def dispatch(self, run: ToolRunContext) -> PreparedCall:
        """The around-dispatch stage: `tools/execute` wrappers, then the body."""
        execution = run.execution
        definition = run.definition
        try:

            async def body(dispatch_exec: ToolExecution) -> ToolExecutionResult:
                # A wrapper may have replaced the signal for its delegated
                # lifetime; the body sees whatever reached it.
                run.execution = dispatch_exec
                value = await definition.execute(dispatch_exec.arguments, run)
                content = definition.render(dispatch_exec.arguments, value)
                meta = (
                    None
                    if dispatch_exec.parent is not None
                    else definition.project_meta(dispatch_exec.arguments, value)
                )
                return ToolExecutionResult(
                    is_error=False,
                    content=content,
                    value=value,
                    meta=meta,
                    additional_contexts=tuple(run._deferred),
                    concludes_turn=run._concluded,
                )

            result = await self.ctx.waterfall(
                "tools/execute", execution, inner=body, scope=execution.scope
            )
            if not isinstance(result, ToolExecutionResult):
                raise TypeError("tools/execute must resolve to a ToolExecutionResult")
            return PreparedCall(run=run, result=result)
        except Exception as error:
            return PreparedCall(run=run, result=_failure(error, started=True))

    async def finalize(
        self, run: ToolRunContext, result: ToolExecutionResult
    ) -> ToolExecutionResult:
        """Post-execute, then definition-owned finalization, then notify."""
        try:
            decided = await self._post_execute(run, result)
        except Exception as error:
            decided = _failure(error, started=True)
        return self.finish(run, decided)

    async def _post_execute(
        self, run: ToolRunContext, result: ToolExecutionResult
    ) -> ToolExecutionResult:
        execution = run.execution

        async def inner(_exec: ToolExecution, _result: ToolExecutionResult) -> PostToolDecision:
            return Accept()

        decision = await self.ctx.waterfall(
            "tools/post-execute", execution, result, inner=inner, scope=execution.scope
        )
        if isinstance(decision, Block):
            # A block exposes only the context its own decision supplied:
            # context the body deferred belonged to an outcome that no longer
            # stands.
            return ToolExecutionResult(
                is_error=True,
                content=tuple(decision.feedback),
                error=ToolFailure(message=_message_from_content(decision.feedback)),
                additional_contexts=decision.additional_contexts,
            )
        if not isinstance(decision, Accept):
            raise TypeError("tools/post-execute must resolve to an Accept or Block")
        if decision.content is not None and decision.has_value:
            raise TypeError("tools/post-execute accept cannot replace both value and content")
        changes: dict[str, Any] = {
            "additional_contexts": (*result.additional_contexts, *decision.additional_contexts)
        }
        if decision.has_value:
            changes["value"] = decision.value
            changes["content"] = run.definition.render(execution.arguments, decision.value)
        elif decision.content is not None:
            changes["content"] = tuple(decision.content)
        return replace(result, **changes)

    def finish(self, run: ToolRunContext, result: ToolExecutionResult) -> ToolExecutionResult:
        """Definition-owned content finalization, then `tools/result`.

        Runs for *every* normalized outcome, pipeline failures included — which
        is the point: a tool whose content needs a last-mile transform must not
        have to trust that policy let its result through.
        """
        finalize = run.definition.finalize_content
        if finalize is not None:
            try:
                replacement = finalize(run.execution, result)
            except Exception:
                log.exception("ph.tools: %s.finalize_content raised; content preserved", run.name)
                replacement = None
            if replacement is not None:
                result = replace(result, content=tuple(replacement))
        self.ctx.emit("tools/result", run.execution, result, scope=run.scope, contained=True)
        return result


def _discard(items: list[Any], target: Any) -> None:
    with suppress(ValueError):  # already released
        items.remove(target)


def _failure(error: object, *, started: bool = False) -> ToolExecutionResult:
    """Normalize any raised value into a structured `is_error` result (B5).

    A tool that raises must not be able to take the loop down with it, and the
    model needs the failure in the same shape as every other result. The
    failure's `kind` comes from the error class that knows it.
    """
    if isinstance(error, Cancelled):
        return aborted_result(started=started)
    # The error's own answer, not a mapping this function keeps: a boolean here
    # collapsed three kinds into two and silently reported an abort as a failure.
    kind: FailureKind = error.failure_kind if isinstance(error, HarnessError) else "failed"
    return error_result(error_message(error), error_info(error), kind=kind)


def _message_from_content(content: Sequence[Any]) -> str:
    text = text_of(content, placeholder=lambda kind: f"[{kind} content]")
    return text or "tool result blocked by post-execute policy"


def _unresolved(_args: Any, _run: ToolRunContext) -> Any:  # pragma: no cover - never dispatched
    raise ToolNotFoundError("<unresolved>")


_UNRESOLVED = ToolDefinition(
    name="<unresolved>",
    description="placeholder for a call that failed before its tool resolved",
    parameters={},
    output=ToolOutput(schema={}, render=lambda _a, _v: ()),
    execute=_unresolved,
)
"""The definition a run carries when `create_execution` itself failed.

A run always has a definition, so `finish` never has to ask whether one exists;
this one renders nothing and is never dispatched."""


@plugin("tools", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the tool registry."""
    ctx.provide("tools", ToolRuntime(ctx=ctx, default_mode=config.mode))
