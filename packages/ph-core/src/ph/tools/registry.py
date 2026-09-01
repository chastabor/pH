"""`ctx.tools` — the registry, the visibility rules, and the pipeline.

Three separable things live here because they share one traversal:

* **registration** — global or per-agent, with scoped shadowing by name (B7);
* **visibility** — restrictions intersect over *global* names only, so a
  restriction can never silence a tool an agent registered for itself;
* **the pipeline** — `tools/pre-execute` → approval on `ask` → monotonic guards
  → `tools/execute` (around) → body → `tools/post-execute` → normalize →
  `finalize_content` → `tools/result` (B1-B5).

**Guards run after approval**, following dsh. A guard is deny-only and runs last,
so it is the final word even over a human's explicit approval — that is what
"monotonic" buys, and it is why policy that must not be reorderable stays a guard
rather than a listener. (The pH plans' summary tables list the two the other way
round; that is a transcription slip, not a second design.)

@module ph.tools.registry
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ..cancel import Cancelled, is_cancelled
from ..cordis import (
    DEPLOYMENT,
    Boundary,
    Context,
    Disposer,
    Running,
    boundary_of,
    events,
    plugin,
    running,
)
from ..llm.types import ToolSchema, text_of
from ..seams._restriction import NameFilter
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
    "register_when_composed",
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
ToolRestriction = NameFilter
"""A per-scope filter over global tools. Restrictions intersect.

The rule itself is `ph.seams._restriction.NameFilter`, shared with `ctx.skills`,
which asks the same question of a different table. The name stays because it is
what every call site and every profile-facing docstring says.
"""


@dataclass(slots=True)
class _Layer:
    """One scope's registry contribution."""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    by: dict[str, Running] = field(default_factory=dict)
    """Who registered each tool, and which scope it landed on (P6-29).

    Kept beside `tools` rather than on `ToolDefinition`, which is frozen row data
    a plugin author builds and hands over — a `Context` on it would make the
    value a registration record and let the same definition be registered twice
    with one of them lying. Written and popped only by `_claim`, and only *after*
    the mutation it describes has been accepted: a parallel dict is the one shape
    the five sibling registries rejected in favour of a `_Registered(value, by)`
    record, and this is the hazard they were avoiding. It survives here because
    `_claim`'s `mutate`/`undo` closures are built before the pair is known, so
    folding the pair into `tools` would mean threading it through all six call
    sites to serve one."""
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
    by: dict[str, Running]
    """Who registered each visible tool, resolved by the same shadowing walk.

    Beside `visible` rather than derived later because one traversal decides
    both: the layer that *won* a name is the layer whose registration owns it,
    and asking again afterwards would be a second walk free to disagree with the
    first the moment shadowing changed."""
    mode: PresentationMode
    schemas: tuple[ToolSchema, ...]
    """The model-facing schemas, or empty under Code Mode — where the model is
    offered one callable and reaches the rest through the SDK (P1-04)."""
    transport_name: str
    """What this scope's model calls the transport.

    Four checks compare against this rather than against `RUN_CODE` — the C6
    refusal, the route-back text, the `tools` namespace that must not bind the
    transport to itself, and the code-only rule in the prompt. Reading it from the
    view keeps them in agreement by construction."""
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
        scope: Context | None,
        mutate: Callable[[_Layer], None],
        undo: Callable[[_Layer], Any],
        label: str,
        *,
        record: str = "",
    ) -> Disposer:
        """Mutate a layer and hand back the disposer that undoes it.

        **Two questions, from one `scope`.** `owner.isolation` chooses *which layer* a
        registration lands in — tool visibility, B7's subject — while lifetime chooses
        *when it goes away*. Both are derived here rather than passed in, so the
        relationship is stated once instead of at each call site where a caller could pair
        them wrongly.

        `isolation` is what keeps the two answers apart even though both read the running
        binding (P6-26). An activation scope is not isolated, so a row's tool still lands
        on the global layer; an agent's scope *is* its own isolation, so a tool registered
        from a body running for that agent lands on that agent's layer — the case that let
        a body inside a contained child install a tool the whole deployment could see.

        **The lifetime is the *intersection* of the two, not a choice between them**
        (P6-29): the two scopes are unrelated branches and either can end first. Owning it
        by the row alone strands the agent's `_Layer` under a disposed key; owning it by
        the agent alone lets a registration outlive the row whose code made it, which is
        I2 verbatim. So the release goes on both scopes and the first to fire wins,
        through `Running.add_disposer` — where that rule lives, because `ph.seams.skills`
        keys `_restrictions` by the layer too.
        """
        by = self.ctx.running_for(scope)
        key = by.layer.isolation
        layer = self._layer(key)
        mutate(layer)
        if record:
            # Only a registration this registry will later *invoke* needs the
            # pair kept — a tool, not a guard or a restriction, which are called
            # as policy rather than as the row's body. Here rather than in
            # `_register` so it is written where the pair is derived: computing
            # it twice would ask `owner_for` twice and warn twice for a disposed
            # activation, which is the one branch that is meant to be audible.
            # Named `record` after `claim_slot`'s parameter, which does the same
            # job for the at-most-one shape — one idiom, one word.
            #
            # **After `mutate`, not before.** `_register`'s `add` refuses a
            # duplicate name by raising, and writing the pair first meant a
            # *rejected* registration overwrote the surviving one's — so the
            # tool that stayed ran as the row whose registration had just been
            # refused. Invisible until the next `_changed()` rebuilt the view
            # from the corrupted cell, which is the worst kind of visible.
            layer.by[record] = by
        self._changed()

        def finish() -> None:
            if record:
                layer.by.pop(record, None)
            self._release(key, lambda: undo(layer))

        return by.add_disposer(finish, label=label)

    def register(self, definition: ToolDefinition, *, scope: Context | None = None) -> Disposer:
        """Register a tool globally, or on an agent's scope to shadow by name."""
        if definition.name == RUN_CODE:
            raise ValueError(
                f'"{RUN_CODE}" is the reserved Code Mode transport and cannot be registered'
            )
        # A chain walk over the layer cells, not `view()`: registration
        # invalidates the view cache one line later, so building one here made
        # every mount O(N^2) in throwaway schema construction.
        #
        # `scope or self.ctx` survives here and in `present_transport`, stated
        # per §5 rule 6 — and the honest reason is not that converting the read
        # would make one `None` mean two things. It already does: this line
        # resolves the *mount*, while `_register` two lines down resolves
        # `layer_for(scope)`, which since P6-26 is the running body's layer. So
        # the reserved-name check and the registration it guards can disagree,
        # and the check takes the wider view — a tool body registering under
        # `running(...)` can occupy the transport name C6 says is unshadowable.
        # Latent (every `ctx.tools.register` in the tree is a row's `apply`), and
        # the fix is not to convert this half: both halves are asking
        # `layer_for`'s question, so the *pair* converts together or neither
        # does. P6-12's mechanism owns that, and P6-32 defers to it.
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
        key = self.ctx.layer_for(scope).isolation

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
            scope,
            add,
            lambda layer: layer.tools.pop(definition.name, None),
            f"tool({definition.name})",
            record=definition.name,
        )

    def restrict(self, restriction: ToolRestriction, *, scope: Context | None = None) -> Disposer:
        """Mask global tools for one scope. Restrictions intersect."""
        return self._claim(
            scope,
            lambda layer: layer.restrictions.append(restriction),
            lambda layer: _discard(layer.restrictions, restriction),
            "tools.restrict",
        )

    def guard(self, guard: ToolGuard, *, scope: Context | None = None) -> Disposer:
        """Register a monotonic deny-only guard. It runs last, and it is final."""
        return self._claim(
            scope,
            lambda layer: layer.guards.append(guard),
            lambda layer: _discard(layer.guards, guard),
            "tools.guard",
        )

    def present_as(self, mode: PresentationMode, *, scope: Context | None = None) -> Disposer:
        """Declare the presentation for one agent, shadowing the deployment default."""

        def clear(layer: _Layer) -> None:
            layer.mode = None

        return self._claim(
            scope,
            lambda layer: setattr(layer, "mode", mode),
            clear,
            "tools.present_as",
        )

    def present_transport(
        self, presentation: TransportPresentation, *, scope: Context | None = None
    ) -> Disposer:
        """Name and describe the transport for one scope (P3-09).

        Refused when the name is already a visible tool: the whole point of the
        reservation is that a model told to call this name reaches the transport,
        and a silent shadow either way would defeat it. The mirror check lives in
        `register`, so the two orders fail the same way — and `register`'s
        comment is where `scope or self.ctx` surviving in both is argued.
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

        return self._claim(scope, add, clear, "tools.present_transport")

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
            scope,
            add,
            lambda layer: layer.code_namespaces.pop(name, None),
            f"tools.code_namespace({name})",
        )

    # ------------------------------------------------------------ visibility --

    def _presented_name(self, target: Context) -> str:
        """The transport's presented name, from the layer cells alone."""
        presentation = next(
            (layer.transport for layer in self._chain(target) if layer.transport is not None),
            None,
        )
        return presentation.name if presentation is not None else RUN_CODE

    def _chain(self, target: Context) -> Iterator[_Layer]:
        """This scope's layers, most-specific-first.

        Layers alone: the key had one consumer, the `key is None` guard that
        P6-27 deleted, and every remaining call site was unpacking it into a
        throwaway — in two different spellings, which is how a reader comes to
        wonder whether it once mattered. It is still `target.isolation_chain()`
        for anything that needs it back.
        """
        for key in target.isolation_chain():
            layer = self._layers.get(key)
            if layer is not None:
                yield layer

    def view(self, scope: Boundary) -> _View:
        """Resolve what one boundary sees, most-specific-first.

        Memoized per isolation chain until the next registry change: the view is read
        several times per tool call and per prompt assembly, and changes only when a row
        registers or disposes something.

        **Stated, with no default** (P6-32). This resolved `scope or self.ctx`, and
        `self.ctx` is the mount — whose isolation is `None`, the *global* layer — so an
        unstated boundary was not "no tools", it was every tool the deployment holds.
        `DEPLOYMENT` now says the wide thing deliberately, so a caller that says nothing
        fails rather than being handed the widest answer.
        """
        target = boundary_of(scope, self.ctx)
        cache_key = tuple(target.isolation_chain())
        cached = self._views.get(cache_key)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        view = self._build_view(target)
        self._views[cache_key] = (self._generation, view)
        return view

    def _build_view(self, target: Context) -> _View:
        visible: dict[str, ToolDefinition] = {}
        by: dict[str, Running] = {}
        layers = list(self._chain(target))
        mode = next((layer.mode for layer in layers if layer.mode is not None), None)
        # Carried rather than re-sliced. `layers[:index]` allocated a list and a
        # generator *per tool name per layer* to say "the strictly nearer
        # scopes"; accumulating them says the same thing once, and says it in the
        # shape the rule is stated in.
        nearer: list[_Layer] = []
        for layer in layers:
            for name, definition in layer.tools.items():
                if name in visible:
                    # A nearer scope already answered this name; scoped
                    # registrations shadow globals rather than merging.
                    continue
                if not all(outer.admits(name) for outer in nearer):
                    # **A restriction reaches everything *outside* the scope that
                    # wrote it, and nothing inside** (P6-27). `nearer` excludes
                    # this layer, so a layer is never filtered by its own
                    # restriction — an agent's own registration cannot be masked
                    # out from under it — while an ancestor's is, which is what
                    # "a child holds a subset of its parent" means.
                    #
                    # This read `key is None and not all(...)`, filtering
                    # **global** names only: the same answer for a flat tree,
                    # where a chain is [self, global] and there is no ancestor to
                    # inherit from, and the wrong one once agents nest — a child
                    # inherited its parent's scoped tools and could not be
                    # narrowed out of them, so a grant could widen a child but
                    # never bound it.
                    #
                    # **"Inside" means this layer, not this subtree.** A tool
                    # registered on a *descendant's* scope is not reachable by an
                    # ancestor's filter either, so a grandchild can hold what its
                    # granting parent cannot see. Deliberate, and the reason a
                    # spawn may only ever `restrict`: see `Grant.apply`, which
                    # states that registering on a child is a way to hand it
                    # something its parent lacks, and is therefore not an
                    # instrument containment may use.
                    continue
                visible[name] = definition
                registered = layer.by.get(name)
                if registered is not None:
                    by[name] = registered
            nearer.append(layer)
        resolved_mode = mode if mode is not None else self.default_mode
        presentation = next(
            (layer.transport for layer in layers if layer.transport is not None), None
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
        for layer in layers:
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
            by=by,
            mode=resolved_mode,
            schemas=schemas,
            transport_name=transport_name,
            code_namespaces=code_namespaces,
        )

    def mode_for(self, scope: Boundary) -> PresentationMode:
        return self.view(scope).mode

    def get(self, name: str, *, scope: Boundary) -> ToolDefinition | None:
        return self.view(scope).visible.get(name)

    def names(self, *, scope: Boundary) -> list[str]:
        return sorted(self.view(scope).visible)

    def schemas(self, *, scope: Boundary) -> list[ToolSchema]:
        """The model-facing schemas for one scope.

        Whitelists name/description/parameters, so no internal field — a
        timeout budget, a concurrency classifier — can leak into a request.
        """
        return list(self.view(scope).schemas)

    def guard_reason(self, execution: ToolExecution) -> str | None:
        """The first monotonic denial from every layer this call can see."""
        for layer in self._chain(execution.scope):
            for guard in layer.guards:
                reason = guard(execution)
                if reason is not None:
                    return reason
        return None

    def execution_mode(self, call: ToolExecutionInput) -> ExecutionMode:
        """The live overlap classification, re-read before every start.

        **The last definition-owned body, bound like the other four** (P6-29): a
        classifier that registers something unwinds with the row that wrote it instead of
        landing wherever the scheduler happened to be called from.

        The layer bound around the call is `boundary_of(call.scope, ...)`, and for
        `DEPLOYMENT` that is the mount. The equivalence is load-bearing: only globally
        registered tools are visible under `DEPLOYMENT`, and a global registration's own
        layer *is* the mount's, so binding the mount here binds the pair's recorded layer.
        """
        view = self.view(call.scope)
        definition = view.visible.get(call.name)
        if definition is None:
            return ExecutionMode(kind="exclusive")
        with running(view.by.get(call.name), boundary_of(call.scope, self.ctx)):
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
            # The one narrowing point onto the pipeline's *persistent record*:
            # `ToolExecution.scope` is non-optional, so every stage downstream
            # reads a `Context` (P6-32). `execution_mode` narrows the same field
            # once more, transiently, because the scheduler classifies calls
            # that have no execution yet.
            scope=boundary_of(call.scope, self.ctx),
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
        # **Who the tool's own code runs as, resolved here and carried** (P6-29).
        # The two halves come from different places on purpose: the *owner* is
        # what registration recorded, so a registration made from a tool body
        # unwinds with the row that wrote the tool (I2); the *layer* is the live
        # execution's scope at each stage, so what it registers is visible to the
        # agent it ran for and to nobody else (B7).
        #
        # From the same `view` that produced `definition`, so one traversal fills
        # `visible` and `by` together and the pair cannot be absent while the
        # definition is present. Asking again per stage would be wrong: a tool
        # that unregisters itself during its own `execute` is gone from the
        # rebuilt view by `finish`.
        by = view.by.get(call.name) or Running(execution.scope, execution.scope)
        return ToolRunContext(execution=execution, definition=definition, by=by)

    async def prepare(self, call: ToolExecutionInput) -> PreparedCall:
        """Ordered pre-policy: pre-execute, approval, guards.

        Everything here is awaited in model order by the batch scheduler; only
        `dispatch` overlaps. That is what keeps policy decisions deterministic
        while still letting two slow tools run at once.
        """
        try:
            run = self.create_execution(call)
        except Exception as error:
            unresolved = self._execution(call, None)
            placeholder = ToolRunContext(
                execution=unresolved,
                by=Running(unresolved.scope, unresolved.scope),
                definition=_UNRESOLVED,
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
                # **The definition's code runs as the agent it was invoked
                # for** (P6-26), all three callbacks and not only `execute` —
                # `render` and `project_meta` are the same row's code reached
                # through the same registry, and a rule that covered one of the
                # three is one the next reader has to reconstruct.
                # Nothing bound it before: `owner_for` fell through to the seam,
                # so a registration made here outlived the run, and `layer_for`
                # fell through to the *global* layer, so a body inside a
                # contained child installed a tool the whole deployment could
                # see — the last way to escape the ceiling P6-27 made
                # structural. The execution already carries the right scope; it
                # was simply never made current.
                with running(run.by, dispatch_exec.scope):
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
            # As the agent, like `dispatch`'s call to the same callable (P6-26).
            # Reached through a second waterfall, so without this it would run
            # bound to whichever `tools/post-execute` wrapper resolved last —
            # which is a stranger's row, chosen by the mounted profile.
            with running(run.by, execution.scope):
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
                # The last definition-owned body, and the one that runs for
                # failures too — so it is the likeliest of the five to register
                # a follow-up, and the least acceptable one to leave ambient.
                with running(run.by, run.scope):
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


def register_when_composed(ctx: Context, build: Callable[[], ToolDefinition | None]) -> None:
    """Register a tool once the profile is whole, if `build` says there is one.

    For the tool whose *existence* depends on something another row supplies: a
    `skill` tool needs an installed skill, a `task` tool needs a subagent provider.

    **A tool that cannot work must not be registered**, rather than registered and
    refused: its schema and description are in the prompt of every request, so a tool
    the deployment cannot perform spends the context window teaching a capability
    that is not there — and the model spends a turn discovering that.

    **`profile/mounted`, not `ctx.inject`.** `inject` waits on a *service key*, and
    neither condition is one: "the skills registry is non-empty" and "some subagent
    provider is registered" are facts about a service's contents. `profile/mounted`
    is the one moment the profile is known to be whole.

    `build` returning `None` means "not in this deployment". A name already
    registered is left alone, so a second dispatch is a no-op rather than a
    duplicate-name `ValueError` — which, on this event, would abort the process.
    """

    def once() -> None:
        definition = build()
        # `DEPLOYMENT` is the mount's chain — the view `scope=None` always
        # read, named (P6-32). It is NOT a union over agents: a tool registered
        # on one agent's scope is invisible here, verified, so this checks
        # "already claimed at the deployment layer" and nothing more. That is
        # the check this always made; a true is-it-taken-*anywhere* audit is a
        # per-layer question no single boundary answers.
        if definition is not None and ctx.tools.get(definition.name, scope=DEPLOYMENT) is None:
            # No `scope=`. It carried a `scope=ctx` until P6-25, and that was
            # the tell: a listener firing after `apply` had returned saw no
            # activation, so the registration would have landed on the tool
            # registry and outlived this row — and the only thing stopping it
            # was that somebody remembered. Dispatch now runs a listener as an
            # effect of the scope that registered it, so the ordinary call is
            # the correct one here as everywhere else.
            ctx.tools.register(definition)

    ctx.on("profile/mounted", once)


@plugin("tools", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the tool registry."""
    ctx.provide("tools", ToolRuntime(ctx=ctx, default_mode=config.mode))
