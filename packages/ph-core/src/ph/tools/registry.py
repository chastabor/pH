"""`ctx.tools`: the registry, the visibility rules, and the pipeline.

Three separate things are in this module because they share one traversal:

* **Registration.** A tool is registered globally or for one agent. A scoped
  registration shadows a global one by name (B7).
* **Visibility.** Restrictions intersect. A restriction filters only the tools
  that are registered outside the scope that wrote it: the global tools and
  the tools of an ancestor scope (P6-27). So a restriction can never hide a
  tool that an agent registered for itself.
* **The pipeline.** `tools/pre-execute` → approval on `ask` → monotonic guards
  → `tools/execute` (around) → `tools/body` → body → `tools/post-execute` →
  normalize → `finalize_content` → `tools/result` (B1-B5).

**Guards run after approval**, as in dsh. A guard can only deny, and it runs
last. So a guard has the final word, even over the explicit approval of a
human. That is the benefit of "monotonic". It is also the reason why a policy
that must not be reorderable stays a guard and does not become a listener. (The
summary tables of the pH plans list the two in the opposite order. That is a
transcription error and not a second design.)

@module ph.tools.registry
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Never, NoReturn

from ..agent.types import AgentHandle
from ..cancel import Canceled, is_canceled
from ..cordis import (
    DEPLOYMENT,
    Boundary,
    Context,
    Disposer,
    MaybeAwaitable,
    Running,
    ScopedTable,
    boundary_of,
    chain_label,
    drop_dead_chains,
    events,
    plugin,
    running,
    settled,
)
from ..json import JsonObject, JsonValue, as_bool, as_str, thaw_json
from ..keys import APPROVAL, TOOLS
from ..llm.types import ContentBlock, TextBlock, ToolSchema, content_from_wire, text_of
from ..seams._restriction import NameFilter
from ..seams.approval import Edited, Responded, denial_reason
from ..seams.code_runtime import CodeBindingNamespace, validate_binding_name
from ..session import Claim, IntentJournal, Prior, Session, SessionEvent, intents_of
from ..session.json import freeze_json_value
from ..session.kinds import TOOL_EFFECT, effect_settle
from ..wire import WireModel
from .definition import (
    Accept,
    Allow,
    Ask,
    Block,
    Deny,
    Done,
    ExecutionMode,
    FailureKind,
    NotDone,
    PostToolDecision,
    PreToolDecision,
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
    budget_result,
    denied_result,
    error_result,
    recorded_arguments,
    text_content,
)
from .errors import (
    TOOL_EFFECT_IN_FLIGHT,
    HarnessError,
    ToolNotFoundError,
    error_info,
    error_message,
)

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


EFFECT_IN_FLIGHT = (
    "Not run: another call with this same effect (call {call_id}) is still running. "
    "Wait for it; asking again once it has returned gets its result."
)
"""What a keyed call says when the same effect is already being carried out here."""

EFFECT_MAY_HAVE_HAPPENED = (
    "Note: an earlier call with this same effect (call {call_id}) was not seen to "
    "finish, so it may already have happened. Check before relying on this having "
    "been the first time."
)
"""What a keyed call's result says when it ran after an unknown outcome."""

RUN_CODE = "run_code"
"""The reserved name of the Code Mode transport.

The name is reserved in *every* mode. It cannot be registered, shadowed or
restricted. If a deployment could occupy the name, a model that was told to
call `run_code` would reach something else (P1-04). Only `register_transport`
may claim it."""

PresentationMode = Literal["native", "code", "both"]

ToolGuard = Callable[[ToolExecution], str | None]
"""A monotonic guard: a reason denies, and `None` abstains.

There is no allow result *by construction*. For that reason, the order of the
listeners cannot turn a denial back into a permission (B2)."""

CodeNamespaceFactory = Callable[[Any], "MaybeAwaitable[CodeBindingNamespace]"]
"""Builds one binding namespace for one caller: a live run, or the SDK block.

The argument is a `CodeBindingsRequest` (`ph.tools.code_mode`). It is typed
`Any` here because the registry must not import the module that consumes it.

**A `TYPE_CHECKING` forward reference is not an option either.** It is the
obvious fix, and it breaks a gate. This alias is a runtime value, and
`test_registration_ownership` calls `get_type_hints(_Layer)` on every declared
dataclass. That call raises `NameError` on a name that exists only for the type
checker. The walk suppresses the error and drops `code_namespaces` from the
surface that it audits."""


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
    "tools/body",
    "serial",
    owner="ph.tools",
    doc="A call about to run its body, once every `tools/execute` listener has had its say. "
    "A barrier that must cover what they appended waits here.",
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


ToolRestriction = NameFilter
"""A per-scope filter over the tools that the scope inherits. Restrictions
intersect.

The inherited tools are the global tools and the tools of an ancestor scope
(P6-27). The rule itself is `ph.seams._restriction.NameFilter`. `ctx.skills`
shares it and asks the same question of a different table. The name
`ToolRestriction` stays because every call site and every profile-facing
docstring uses it.
"""


@dataclass(slots=True)
class _Layer:
    """One scope's registry contribution."""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    by: dict[str, Running] = field(default_factory=dict)
    """Who registered each tool, and which scope the registration landed on (P6-29).

    The pairs are kept beside `tools` and not on `ToolDefinition`. A definition
    is frozen row data that a plugin author builds and hands over. A `Context`
    on it would make the value a registration record, and the same definition
    could then be registered twice with one of the two records wrong. Only
    `_claim` writes and pops this dict, and only *after* the mutation that the
    entry describes has been accepted. A parallel dict is the one shape that the
    six sibling registries rejected in favor of a `_Registered(value, by)`
    record, and this is the hazard that they avoided. The dict stays here
    because the `mutate` and `undo` closures that `_claim` takes are built
    before the pair is known. To fold the pair into `tools` would mean passing
    it through all six call sites for the benefit of one."""
    restrictions: list[ToolRestriction] = field(default_factory=list)
    guards: list[ToolGuard] = field(default_factory=list)
    mode: PresentationMode | None = None
    """The presentation that this scope declared for itself.

    This is one cell and not a list. Two answers to "which form does the model
    see" are a contradiction, and not something to merge."""
    transport: TransportPresentation | None = None
    """How this scope names and describes the transport. It is one cell for the
    same reason as `mode`: the model is offered exactly one callable."""
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

    It is stored beside `visible` and not derived later, because one traversal
    decides both. The layer that *won* a name is the layer whose registration
    owns it. A second walk afterwards could disagree with the first as soon as
    the shadowing changed."""
    mode: PresentationMode
    schemas: tuple[ToolSchema, ...]
    """The schemas that the model sees. Under Code Mode this is only the schema
    of the transport. There the model is offered one callable and reaches the
    rest through the SDK (P1-04)."""
    transport_name: str
    """The name by which the model of this scope calls the transport.

    Four checks compare against this name and not against `RUN_CODE`:

    * the C6 refusal
    * the route-back text
    * the `tools` namespace, which must not bind the transport to itself
    * the code-only rule in the prompt

    All four read the name from the view, so they agree by construction."""
    code_namespaces: dict[str, CodeNamespaceFactory]
    """The contributed binding namespaces that the programs of this scope may
    reach. A nearer scope shadows by name, as with tools. They are resolved with
    the view so that the run and the SDK block read one answer."""


@dataclass(frozen=True, slots=True)
class _Effect:
    """The open intent of a keyed call, from before its wrappers to after its body."""

    journal: IntentJournal
    session: Session
    claim: Claim
    unsure_of: str | None = None
    """The call id of the earlier attempt that nobody saw finish, when this call
    retries one."""

    def unknown(self) -> None:
        """Settle the intent as `outcome-unknown`. The effect may have happened.
        So `reconcile` decides a repeat, or the repeat runs with the note, as
        after a crash."""
        self.journal.abandon(self.session, self.claim, "outcome-unknown")


def _effect_settle(
    key: str, call_id: str, *, is_error: bool, content: Sequence[ContentBlock]
) -> JsonObject:
    """A `tool/effect-settled` payload, built from the blocks of the result. The
    builder is `effect_settle`, the builder of the `TOOL_EFFECT` kind, which the
    closer of repair also calls."""
    return effect_settle(
        key, call_id, is_error=is_error, content=[block.to_wire() for block in content]
    )


def _repeated(prior: Prior, settled: SessionEvent) -> ToolExecutionResult:
    """The result that the first call with this effect returned, given back to the
    repeat."""
    meta: JsonObject = {"repeated": True, "repeatOf": as_str(prior.opened.data.get("callId"))}
    return ToolExecutionResult(
        is_error=as_bool(settled.data.get("isError")),
        content=tuple(content_from_wire(settled.data.get("content"))),
        meta=meta,
    )


@dataclass(frozen=True, slots=True)
class PreparedCall:
    """One call between pipeline stages.

    `result is None` means that the body has not run and `dispatch()` is next.
    Otherwise `needs_post` says whether `tools/post-execute` still applies. A
    denial is a result that policy may still shape. A pipeline *failure*
    bypasses the stage: a listener that raised must not reshape the error that
    its own raise produced.
    """

    run: ToolRunContext
    result: ToolExecutionResult | None = None
    needs_post: bool = True


class Config(WireModel):
    """The row config for `ph.tools`.

    The config is a `WireModel`, like every other JSON boundary. A row is
    written as YAML in camelCase and read as snake_case in Python, under the one
    alias rule (Q2). `extra="forbid"` turns a mistyped key into a startup error,
    so the setting is not silently ignored.
    """

    mode: PresentationMode = "native"


def _refuse_decision(decision: Never, *, hook: str, expected: str) -> NoReturn:
    """A `waterfall` hook resolved to something outside its decision union.

    `Never` gives the build-time check of `assert_never`: a variant that is
    added without an arm fails the type check. The raise is a `TypeError` and
    not the `AssertionError` of `assert_never`. At run time this is not an
    unreachable branch. `waterfall` calls listeners that a plugin registered,
    so the value is wrong and not impossible. A `TypeError` is the project
    convention for that case (`ph.cordis.context` raises the same for an effect
    that returns no disposer).
    """
    raise TypeError(f"{hook} must resolve to {expected}")


def _projection_key(execution: ToolExecution) -> tuple[str, str]:
    """Whose call this is, and the id of the call. The id of an agent is the id
    of its session, and a call with neither belongs to the deployment."""
    if execution.agent is not None:
        return execution.agent.id, execution.call_id
    return (execution.session.id if execution.session is not None else ""), execution.call_id


@dataclass(slots=True)
class ToolRuntime:
    """The service published as `ctx.tools`."""

    ctx: Context
    default_mode: PresentationMode = "native"
    _layers: ScopedTable[_Layer] = field(default_factory=lambda: ScopedTable(_Layer, _Layer.empty))
    """The contribution of each scope (`ScopedTable`). It is made on the first
    registration of the scope and dropped with the last."""
    _generation: int = 0
    """Incremented on every mutation. The view cache is valid for one generation."""
    _views: dict[tuple[Context | None, ...], tuple[int, _View]] = field(default_factory=dict)
    _projecting: dict[tuple[str, str], ToolRunContext] = field(default_factory=dict)
    """The runs that are inside `tools/post-execute` now. `projected` renders
    against them. The key says whose call it is and gives the id of the call
    (`_projection_key`).

    The key includes the call because a policy row that asks for a projection
    asks about *its* call, and several calls can be in the waterfall at the
    same time. **It also includes whose call it is** (T2). An id that is minted
    for a call that its provider did not name is the same for two agents at the
    same turn, step and index. Keyed by id alone, the entry of the second call
    replaced the entry of the first, and the `finally` of the second call then
    removed it while the first call still needed it."""

    # ---------------------------------------------------------- registration --

    def _changed(self) -> None:
        self._generation += 1
        self._views.clear()
        self.ctx.emit("tools/change")

    def _claim(
        self,
        scope: Context | None,
        mutate: Callable[[_Layer], None],
        undo: Callable[[_Layer], Any],
        label: str,
        *,
        record: str = "",
    ) -> Disposer:
        """Mutate a layer and return the disposer that undoes the mutation.

        **One `scope` answers two questions.** `layer.isolation` chooses *which
        layer* a registration lands in. That is tool visibility, the subject of
        B7. The lifetime chooses *when the registration goes away*. Both answers
        come from the one `Running` pair that is requested here. They are not
        passed in, so a caller cannot pair them wrongly.

        `isolation` keeps the two answers apart, although both read the running
        binding (P6-26). An activation scope is not isolated, so the tool of a
        row still lands on the global layer. The scope of an agent *is* its own
        isolation. So a tool that is registered from a body that runs for that
        agent lands on the layer of that agent. Before this rule, a body inside
        a contained child could install a tool that the whole deployment could
        see.

        **The lifetime is the *intersection* of the two, and not a choice
        between them** (P6-29). The two scopes are unrelated branches, and
        either can end first. If the row alone owns the registration, the
        `_Layer` of the agent stays under a disposed key. If the agent alone
        owns it, the registration can outlive the row whose code made it, which
        is I2 verbatim. So the release goes on both scopes, and the first one to
        fire wins. `ScopedTable.claim` does this. It is the same table shape
        that holds the restrictions of the skill registry and the path limits of
        the sandbox.
        """
        by = self.ctx.running_for(scope)

        def apply(layer: _Layer) -> None:
            mutate(layer)
            if record:
                # The pair is kept only for a registration that this registry
                # will later *invoke*. That is a tool. A guard or a restriction
                # is called as policy and not as the body of the row. The pair
                # is written here and not in `_register`, so that it is written
                # where it is derived. To compute it twice would ask `owner_for`
                # twice and warn twice for a disposed activation, and that is
                # the one branch that is meant to be noticed. The parameter is
                # named `record` after the name that `claim_slot` uses for the
                # same job in the at-most-one shape: one idiom, one word.
                #
                # **After `mutate`, not before.** The `add` of `_register`
                # refuses a duplicate name by raising. When the pair was written
                # first, a *rejected* registration overwrote the pair of the
                # surviving one. The tool that stayed then ran as the row whose
                # registration had just been refused. The error was invisible
                # until the next `_changed()` rebuilt the view from the
                # corrupted cell.
                layer.by[record] = by

        def revert(layer: _Layer) -> None:
            if record:
                layer.by.pop(record, None)
            undo(layer)

        released = self._layers.claim(by, apply, revert, label=label, then=self._changed)
        self._changed()
        return released

    def register(self, definition: ToolDefinition, *, scope: Context | None = None) -> Disposer:
        """Register a tool globally, or on the scope of an agent to shadow by name."""
        if definition.name == RUN_CODE:
            raise ValueError(
                f'"{RUN_CODE}" is the reserved Code Mode transport and cannot be registered'
            )
        # This is a chain walk over the layer cells and not `view()`.
        # Registration invalidates the view cache one line later, so a view
        # built here made every mount O(N^2) in schema construction that was
        # thrown away.
        #
        # `scope or self.ctx` stays here and in `present_transport`, and §5
        # rule 6 requires the reason to be stated. The reason is not that
        # converting the read would make one `None` mean two things. It already
        # does. This line resolves the *mount*. `_register`, a few lines down,
        # resolves `layer_for(scope)`, which since P6-26 is the layer of the
        # running body. So the reserved-name check and the registration that it
        # guards can disagree, and the check takes the wider view. A tool body
        # that registers under `running(...)` can occupy the transport name that
        # C6 says cannot be shadowed. The problem is latent, because every
        # `ctx.tools.register` in the tree is the `apply` of a row. The fix is
        # not to convert this half. Both halves ask the question of `layer_for`,
        # so the *pair* converts together or neither does. The mechanism of
        # P6-12 owns that, and P6-32 defers to it.
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

        The registration path is the same as for every other tool: the shadowing
        rules, the `tools/change` emit, the layer cleanup. Only the reservation
        check is skipped, because that check exists to keep every *other* caller
        off the name.
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

    def registrants(self) -> dict[str, str]:
        """Each deployment-wide tool by name, with the profile row that registered
        it (`Context.row_id`). It is for a caller that must say which tools a row
        gives (the narrowing of an assigned profile, S7b). A tool that no row
        registered is absent."""
        layer = self._layers.get(None)
        if layer is None:
            return {}
        return {name: by.owner.row_id for name, by in layer.by.items() if by.owner.row_id}

    def restrict(self, restriction: ToolRestriction, *, scope: Context | None = None) -> Disposer:
        """Mask the tools that one scope inherits. Restrictions intersect."""
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
        """Declare the presentation for one agent. It shadows the deployment default.

        A scope may hold one, as it may hold one transport (`present_transport`).
        Before this rule, a second declaration overwrote the first, and either
        release cleared the field whatever it then held. So what survived
        depended on the order in which the two were released.
        """

        def claim(layer: _Layer) -> None:
            if layer.mode is not None:
                raise ValueError(f'this scope already presents its tools as "{layer.mode}"')
            layer.mode = mode

        def clear(layer: _Layer) -> None:
            layer.mode = None

        return self._claim(scope, claim, clear, "tools.present_as")

    def present_transport(
        self, presentation: TransportPresentation, *, scope: Context | None = None
    ) -> Disposer:
        """Name and describe the transport for one scope (P3-09).

        The call is refused when the name is already a visible tool. The purpose
        of the reservation is that a model that is told to call this name
        reaches the transport, and a silent shadow in either direction would
        defeat that. The mirror check is in `register`, so the two orders fail
        in the same way. The comment in `register` explains why
        `scope or self.ctx` stays in both.
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
                # This is a real claim and not a silent overwrite. With two
                # presentations on one cell, the first disposal would clear what
                # the second still needs, and the survivor would depend on the
                # disposal order.
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

        The factory answers one question, and it is asked twice with the same
        request: by the run (with a live bridge) and by the SDK prompt section
        (with none). So the block cannot describe a namespace that a program
        could not reach. This is a keyed claim and not a listener. So two rows
        that want one name fail *here*, at mount, and not on every cell of a
        booted deployment. A scoped claim shadows a global one by name, as a
        scoped tool does.

        A namespace that presents a registered tool again says so on the
        *binding* (`CodeBinding.presents`), and not here. The binding list is
        then the only statement of the fact, so it cannot drift from a second
        list. A namespace also cannot suppress a tool that it does not actually
        offer.
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
            (
                layer.transport
                for layer in self._layers.along(target.isolation_chain())
                if layer.transport is not None
            ),
            None,
        )
        return presentation.name if presentation is not None else RUN_CODE

    def stale_views(self) -> list[str]:
        """Cached views that no longer equal what a rebuild would produce (I6, P6-01).

        The check is here and not in the invariant row that declares it, because
        the cache is private to this class. A rename would silently disable a
        check that was written against `_views` from outside. An invariant that
        can stop being checked without anybody noticing is worse than no
        invariant.

        Every cached entry is at the current generation, because `_changed`
        clears the table when it increments the counter. So every entry is an
        answer that `view` would serve, and every entry is compared.
        """
        found: list[str] = []
        for chain, (_, cached) in self._views.items():
            fresh = self._build_view(self._layers.along(chain))
            differing = sorted(
                name
                for name in fresh.visible.keys() | cached.visible.keys()
                if fresh.visible.get(name) is not cached.visible.get(name)
            )
            if differing:
                found.append(
                    f"the cached view for {chain_label(chain)} differs from a rebuild "
                    f"on {differing}"
                )
        return found

    def view(self, scope: Boundary) -> _View:
        """Resolve what one boundary sees, most specific first.

        The view is memoized per isolation chain until the next registry change.
        It is read several times per tool call and per prompt assembly, and it
        changes only when a row registers or disposes something.

        **The boundary is stated, and there is no default** (P6-32). This method
        at first resolved `scope or self.ctx`. `self.ctx` is the mount, whose
        isolation is `None`, the *global* layer. So an unstated boundary did not
        mean "no tools". It meant every tool that the deployment holds.
        `DEPLOYMENT` now states the wide answer deliberately. A caller that
        states nothing fails and does not get the widest answer.
        """
        target = boundary_of(scope, self.ctx)
        cache_key = tuple(target.isolation_chain())
        cached = self._views.get(cache_key)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        # This runs on the miss, before the table grows. The `Context` of a
        # disposed agent is retained by its own key until something invalidates
        # the cache, and a deployment that has stopped registering never does
        # that (I2).
        drop_dead_chains(self._views)
        # The view is built from the **cache key**, and not from the context
        # that it was first built for. The key is `target.isolation_chain()`,
        # and the layers of the key are all that `_build_view` needs of the
        # target. That makes the view a pure function of the key, so
        # `stale_views` can check it.
        view = self._build_view(self._layers.along(cache_key))
        self._views[cache_key] = (self._generation, view)
        return view

    def _build_view(self, layers: Sequence[_Layer]) -> _View:
        visible: dict[str, ToolDefinition] = {}
        by: dict[str, Running] = {}
        mode = next((layer.mode for layer in layers if layer.mode is not None), None)
        # The nearer layers are carried and not sliced again. `layers[:index]`
        # allocated a list and a generator *per tool name per layer* to say
        # "the strictly nearer scopes". An accumulated list says the same thing
        # one time, in the form in which the rule is stated.
        nearer: list[_Layer] = []
        for layer in layers:
            for name, definition in layer.tools.items():
                if name in visible:
                    # A nearer scope already answered this name. Scoped
                    # registrations shadow globals and do not merge with them.
                    continue
                if name != RUN_CODE and not all(outer.admits(name) for outer in nearer):
                    # **The transport is never restricted** (`RUN_CODE`). A
                    # grant lists the names that its parent sees, and there the
                    # transport has the name that a profile presents it as.
                    # Filtering it here under its registered name left every
                    # child in Code Mode with nothing to call. What a cell may
                    # *do* is restricted where it matters: at each binding that
                    # it awaits.
                    #
                    # **A restriction reaches everything *outside* the scope
                    # that wrote it, and nothing inside** (P6-27). `nearer`
                    # excludes this layer, so a layer is never filtered by its
                    # own restriction, and the registration of an agent cannot
                    # be masked away from that agent. The registration of an
                    # ancestor is filtered. That is the meaning of "a child
                    # holds a subset of its parent".
                    #
                    # This check at first read `key is None and not all(...)`,
                    # which filtered **global** names only. That gives the same
                    # answer for a flat tree, where a chain is [self, global]
                    # and there is no ancestor to inherit from. It gives the
                    # wrong answer when agents nest. A child inherited the
                    # scoped tools of its parent and could not be narrowed out
                    # of them, so a grant could widen a child but never limit
                    # it.
                    #
                    # **"Inside" means this layer, not this subtree.** The
                    # filter of an ancestor also does not reach a tool that is
                    # registered on the scope of a *descendant*. So a
                    # grandchild can hold what its granting parent cannot see.
                    # This is deliberate, and it is the reason why a spawn may
                    # only ever `restrict`. See `Grant.apply`, which states
                    # that registering on a child is a way to give the child
                    # something that its parent lacks, and is therefore not an
                    # instrument that containment may use.
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
                # The claim-time checks in `register` and `present_transport`
                # are snapshots of one scope. A scoped presentation followed by
                # a later registration on a parent scope passes both. This is
                # the one place that resolves every view, so the contradiction
                # fails loudly here and does not silently overwrite either side.
                raise ValueError(
                    f'the Code Mode transport is presented as "{presentation.name}", but a '
                    "tool of that name is also visible in this scope; the transport name "
                    "must be unshadowable (C6)"
                )
            # The transport is renamed in place. So `visible` never holds it
            # twice, and nothing downstream needs to know the name under which
            # it was registered.
            del visible[RUN_CODE]
            visible[presentation.name] = presentation.rename(transport)
            transport_name = presentation.name
        code_namespaces: dict[str, CodeNamespaceFactory] = {}
        for layer in layers:
            for name, factory in layer.code_namespaces.items():
                if name not in code_namespaces:
                    code_namespaces[name] = factory
        if resolved_mode == "code":
            # One callable: the transport, under the name that this scope
            # presents it as. Everything else is reached through the SDK that
            # the prompt lists, so the other schemas stay out. The schema of the
            # transport must be offered. Otherwise a real provider gets a
            # request with no tool to call, and the model can run nothing. The
            # fake adapter never asked for a schema, which is why this went
            # unnoticed.
            offered = visible.get(transport_name)
            schemas: tuple[ToolSchema, ...] = () if offered is None else (offered.schema(),)
        else:
            schemas = tuple(visible[name].schema() for name in sorted(visible))
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

    def restore_covers(
        self, name: str, arguments: JsonValue, *, scope: Boundary, agent: AgentHandle | None
    ) -> bool:
        """Whether restoring the workspace takes back everything this call to `name` does.

        The answer is the `effects_confined_to_workspace` of the tool itself,
        read through the view of this scope. A shadowed registration is a
        different tool, and an unscoped lookup erred in the *unsafe* direction:
        a row that shadowed `write` resolved to the `True` of the global
        builtin. An unknown tool is not covered. The question is asked of
        **this call** (S4), with its arguments and the agent that makes it,
        because a restore does not take back a write whose path leaves the
        tree. A call that changes nothing (`effect_free`) is covered, because
        there is nothing to take back.

        `prepare` asks one time per nested call (`ToolExecution.restore_covered`),
        for two readers that must not drift apart:

        * `/revert`, which lists what a restore did not undo, from the record of
          each dispatch.
        * The barrier of the checkpoint policy before a nested Code Mode
          dispatch. The barrier exists so that a crash cannot empty that list.
        """
        definition = self.get(name, scope=scope)
        return definition is not None and (
            definition.effect_free or definition.confined(arguments, agent)
        )

    def effect_free(self, name: str, *, scope: Boundary) -> bool:
        """Whether a call to `name` changes nothing, read through the view of this
        scope (S4). An unknown tool is not effect-free."""
        definition = self.get(name, scope=scope)
        return definition is not None and definition.effect_free

    def names(self, *, scope: Boundary) -> list[str]:
        return sorted(self.view(scope).visible)

    def path_tools(self, *, scope: Boundary) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The visible reader and searcher tools, by what they declare (G8/P4-02).

        **The method is here and not at the row that needed it**, because the
        question is about the registry: which visible tools declare
        `reads_paths` and `searches_paths`. A policy row that matched names
        could not know. A deployment renames its tools and an MCP server adds
        its own. `ToolDefinition.self_limits` in `definition.py` makes the same
        argument.

        **Both sets are returned whole, with no tiebreak.** An earlier version
        returned the first reader. That put a *wording* decision (which of
        several names to put in a sentence) into the tool table, and it showed
        as an asymmetric return. The caller picks one. The next question that
        the caller asks ("name two", "prefer the one that paginates") then
        needs no change here. The names are sorted, so a deployment with two
        readers is deterministic and does not depend on registration order.
        """
        visible = self.view(scope).visible
        reads = tuple(sorted(name for name, one in visible.items() if one.reads_paths))
        searches = tuple(sorted(name for name, one in visible.items() if one.searches_paths))
        return (reads, searches)

    def schemas(self, *, scope: Boundary) -> list[ToolSchema]:
        """The schemas that the model sees for one scope.

        Each schema is a whitelist of name, description and parameters. So no
        internal field, such as a timeout budget or a concurrency classifier,
        can leak into a request.
        """
        return list(self.view(scope).schemas)

    def guard_reason(self, execution: ToolExecution) -> str | None:
        """The first monotonic denial from every layer this call can see."""
        for layer in self._layers.along(execution.scope.isolation_chain()):
            for guard in layer.guards:
                reason = guard(execution)
                if reason is not None:
                    return reason
        return None

    def execution_mode(self, call: ToolExecutionInput) -> ExecutionMode:
        """The live overlap classification, read again before every start.

        **The classifier is a definition-owned body, and it is bound like the
        others** (P6-29). A classifier that registers something unwinds with the
        row that wrote it. The registration does not land wherever the scheduler
        was called from.

        The layer that is bound around the call is `boundary_of(call.scope, ...)`,
        and for `DEPLOYMENT` that is the mount. The code depends on this
        equivalence. Only globally registered tools are visible under
        `DEPLOYMENT`, and the layer of a global registration *is* the layer of
        the mount. So binding the mount here binds the recorded layer of the
        pair.
        """
        view = self.view(call.scope)
        definition = view.visible.get(call.name)
        if definition is None:
            return ExecutionMode(kind="exclusive")
        with running(view.by.get(call.name), boundary_of(call.scope, self.ctx)):
            return definition.classify(call.arguments)

    # --------------------------------------------------------------- pipeline --

    async def execute(
        self,
        call: ToolExecutionInput,
        *,
        write_ahead: Callable[[PreparedCall], None] | None = None,
    ) -> ToolExecutionResult:
        """Run one call through the whole pipeline. `prepare` says what `write_ahead` is."""
        prepared = await self.prepare(call, write_ahead=write_ahead)
        if prepared.result is None:
            prepared = await self.dispatch(prepared.run)
        assert prepared.result is not None
        if prepared.needs_post:
            return await self.finalize(prepared.run, prepared.result)
        return self.finish(prepared.run, prepared.result)

    def _execution(self, call: ToolExecutionInput, arguments: JsonValue) -> ToolExecution:
        return ToolExecution(
            call_id=call.call_id,
            root_call_id=call.root_call_id or call.call_id,
            name=call.name,
            arguments=arguments,
            token=object(),
            # This is the one narrowing point onto the *persistent record* of
            # the pipeline. `ToolExecution.scope` is non-optional, so every
            # stage downstream reads a `Context` (P6-32). `execution_mode`
            # narrows the same field one more time, transiently, because the
            # scheduler classifies calls that have no execution yet.
            scope=boundary_of(call.scope, self.ctx),
            session=call.session,
            agent=call.agent,
            parent=call.parent,
            signal=call.cancel,
        )

    def create_execution(self, call: ToolExecutionInput) -> ToolRunContext:
        """Accept a call: resolve the tool, snapshot the arguments, mint the token.

        `UNKNOWN_TOOL` is decided here, *before* any policy listener runs. So a
        name that the prompt never offered cannot be observed, logged as policy,
        or allowed by a permissive row (C6).
        """
        view = self.view(call.scope)
        definition = view.visible.get(call.name)
        if definition is None:
            if call.name == RUN_CODE and view.transport_name != RUN_CODE:
                # "Not enabled" would be false. The transport is enabled, under
                # the name that this profile presents. The message says which
                # name, so that the model can correct the call.
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
        # **Who the code of the tool runs as is resolved here and carried**
        # (P6-29). The two halves come from different places on purpose. The
        # *owner* is what registration recorded, so a registration that a tool
        # body makes unwinds with the row that wrote the tool (I2). The *layer*
        # is the scope of the live execution at each stage, so what the body
        # registers is visible to the agent that it ran for and to nobody else
        # (B7).
        #
        # The pair comes from the same `view` that produced `definition`. One
        # traversal fills `visible` and `by` together, so the pair cannot be
        # absent while the definition is present. To ask again per stage would
        # be wrong: a tool that unregisters itself during its own `execute` is
        # gone from the rebuilt view by `finish`.
        by = view.by.get(call.name) or Running(execution.scope, execution.scope)
        return ToolRunContext(execution=execution, definition=definition, by=by)

    def _render(self, run: ToolRunContext, value: object) -> tuple[ContentBlock, ...]:
        """The renderer of the tool, bound as the row that registered the tool (P6-26).

        There is one way to call it, because two would be two answers to "who is
        running". The renderer is row code that is reached through a second
        waterfall. Without a binding of its own, it runs bound to whichever
        `tools/post-execute` wrapper resolved last. That is the row of a
        stranger, chosen by the mounted profile.
        """
        with running(run.by, run.execution.scope):
            return run.definition.render(
                run.execution.arguments, value, presented=run.presented_for(value)
            )

    def projected(self, execution: ToolExecution, value: object) -> tuple[ContentBlock, ...] | None:
        """What `value` would render to for this call, or `None` if nothing can answer.

        **This is for a policy row that must measure a value before the registry
        renders it** (D10). `tools/post-execute` runs *before* the render. So a
        row that replaces the value cannot see the content that its replacement
        will become. `tool-result-offload` returned early for exactly that
        reason, and so it never offloaded a structured result, however large.
        The row now asks here.

        **The answer comes from the live run, and not from the table.** The
        first version looked the tool up again (`view(scope)`, `visible`, `by`).
        `ToolRunContext.by` says not to do that. The pair is resolved one time
        in `create_execution`, from the same view that produced `definition`,
        so it cannot disagree with the definition. A tool that unregisters
        itself during its own `execute` is gone from the rebuilt view by the
        time this method is asked. For such a tool, that version answered
        `None`, and offload silently stopped guarding. Or it rendered under a
        synthesized `Running(scope, scope)`. That is the unbound-row-code
        regression that `test_registration_ownership` exists to catch, and the
        method whose docstring cites the test reintroduced it.

        `None` means that no run of this call is in `tools/post-execute`. That
        is the correct answer for a caller that decides whether something is
        too big. For that caller, a guess at a projection would be worse than
        no measurement.

        In the ordinary case this is not a second render. A caller that measures
        in this way supplies the content back, and `_post_execute` then renders
        nothing.
        """
        run = self._projecting.get(_projection_key(execution))
        return None if run is None else self._render(run, value)

    def projected_content(
        self, execution: ToolExecution, decision: Accept, result: ToolExecutionResult
    ) -> Sequence[ContentBlock] | None:
        """The content that an `Accept` in `tools/post-execute` will show the model.

        There are three cases, and the middle one is D10:

        * A decision that replaced the *content* states the content directly.
        * A decision that replaced the **value** states neither. `_post_execute`
          renders again from the value after the waterfall, so the content that
          will be sent does not exist yet. It is requested from `projected`. A
          `None` result comes from that method: there is no live run to render
          against, and a guess would be worse than no answer.
        * A decision that replaced neither leaves the content of the body.

        This is for a row that measures or extends what the model will read:
        offload, which measures the size of a result, and ph-clm, which appends
        its receipt. Like `projected`, it stays an approximation of the final
        content, because `finish` runs the `finalize_content` of a definition
        afterwards.
        """
        if decision.content is not None:
            return decision.content
        if decision.has_value:
            return self.projected(execution, decision.value)
        return result.content

    async def prepare(
        self,
        call: ToolExecutionInput,
        *,
        write_ahead: Callable[[PreparedCall], None] | None = None,
    ) -> PreparedCall:
        """Ordered pre-policy (pre-execute, approval, guards), then the record.

        The batch scheduler *starts* the prepares in model order, but the
        prepares of a parallel group overlap. Each slot awaits its own gate, and
        a slot that is parked on a human does not hold the others. What the
        model sees stays in model order: `commit_ready` appends results in slot
        order, whatever order they settled in.

        `write_ahead` runs after the gate has decided and before anything acts.
        This is **the** place where a durable record of the act belongs (B4,
        P7-15):

        * Not earlier. A call that is parked on a human approval has done
          nothing yet. A record written there says that an act was attempted
          when it was only asked about.
        * Not later. By then the effect of the body has escaped, and a crash
          inside the body would leave no evidence that the action was tried.

        Here, the record is true whichever way the run ends. It is also written
        for a denial, so that the call/result pair that every reader relies on
        stays a pair. The hook is on `prepare` and not on `execute` because "the
        gate decided" is a fact of this method, and a caller that drives the
        stages itself needs the same hook.
        """
        prepared = await self._gate(call)
        execution = prepared.run.execution
        if execution.parent is not None:
            # Asked one time, of the call as it will run
            # (`ToolExecution.restore_covered`).
            execution.restore_covered = self.restore_covers(
                execution.name, execution.arguments, scope=execution.scope, agent=execution.agent
            )
        if write_ahead is not None:
            write_ahead(prepared)
        return prepared

    async def _gate(self, call: ToolExecutionInput) -> PreparedCall:
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
        if is_canceled(execution.signal):
            return PreparedCall(run=run, result=aborted_result(started=False))

        try:
            # The type is `PreToolDecision` and not `Allow`. The chain is the
            # four-way union, and a listener returns any member of it.
            # `waterfall` reads its type from here, so the type of the default
            # alone would narrow every caller.
            async def inner(_exec: ToolExecution) -> PreToolDecision:
                return Allow()

            gate = await self.ctx.waterfall(
                "tools/pre-execute", execution, inner=inner, scope=execution.scope
            )
            if isinstance(gate, Ask):
                gate = await self._service_ask(execution, gate)
            if isinstance(gate, Respond):
                # The answer is returned as the result of the tool itself. It
                # is a *successful* result, because the model asked a question
                # and got an answer. A denial that the model would have to
                # interpret is the wrong shape for "here is the answer, no need
                # to run it".
                return PreparedCall(
                    run=run,
                    result=ToolExecutionResult(
                        is_error=False, content=tuple(text_content(gate.message))
                    ),
                    needs_post=False,
                )
            if not isinstance(gate, (Allow, Deny)):
                # A fifth decision must not reach `gate.reason` below and silently deny.
                _refuse_decision(gate, hook="tools/pre-execute", expected="an Allow or Deny")
            if isinstance(gate, Allow) and gate.has_arguments:
                # The substitution is applied here, in the one place that owns
                # the execution, after every `tools/pre-execute` listener has
                # seen the call that the model actually made.
                execution.arguments = freeze_json_value(gate.arguments, frozen_input=True)
                execution.substituted = True
            if is_canceled(execution.signal):
                return PreparedCall(run=run, result=aborted_result(started=False))
            # Guards run last and only on an allow. A denial is already decided,
            # and asking a guard to confirm it would invite a re-permit.
            reason = self.guard_reason(execution) if isinstance(gate, Allow) else gate.reason
            if reason is not None:
                # **A refusal is not always policy refusing** (D7). A spent
                # budget ends the call in the same way and means something else
                # to every consumer. So the row says which it is, and this code
                # does not infer it from the decision type. The reason of a
                # guard takes neither flag. `guard_reason` only answers for an
                # `Allow`, and a monotonic guard is a plain denial that does not
                # conclude the turn.
                if isinstance(gate, Deny):
                    settle = budget_result if gate.failure_kind == "failed" else denied_result
                    return PreparedCall(
                        run=run,
                        result=settle(reason, concludes_turn=gate.concludes_turn),
                    )
                return PreparedCall(run=run, result=denied_result(reason))
            return PreparedCall(run=run)
        except Exception as error:
            return PreparedCall(run=run, result=_failure(error), needs_post=False)

    async def _service_ask(self, execution: ToolExecution, ask: Ask) -> Allow | Deny | Respond:
        """Resolve an `ask` through the approval seam. Fail closed (B3).

        The seam is used only if it is present. A deployment that mounts no
        approval service denies, and so does a call with no agent, which has
        nowhere to route the prompt.

        The two answers that carry data are translated into the decisions of
        the pipeline on the way back: `Edited` into an `Allow` that substitutes,
        and `Responded` into a `Respond`. So the switch in `_gate` stays closed
        over the decisions of the pipeline, and neither capability is reachable
        only through approval.
        """
        approval = self.ctx.get(APPROVAL)
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
            # The human corrected the call and did not refuse it. See
            # `Allow.arguments` for where each version of the call is recorded.
            return Allow(arguments=outcome.arguments, has_arguments=True)
        if isinstance(outcome, Responded):
            return Respond(message=outcome.message)
        return Deny(reason=denial_reason(outcome, f'tool "{name}"'))

    async def dispatch(self, run: ToolRunContext) -> PreparedCall:
        """The around-dispatch stage: the record of the effect, the `tools/execute`
        wrappers, then the body.

        **A call that names its effect is answered from the log when that is
        possible** (P10-12). The record opens here, before the wrappers, so the
        barrier of the checkpoint policy carries it to disk before the body
        runs. The settle holds what the call returned. So when the same effect
        is requested again, under the new call id that a re-issued call always
        has, the call gets that result back, marked `repeated`, and no second
        effect happens.
        """
        effect = await self._open_effect(run)
        if isinstance(effect, ToolExecutionResult):
            return PreparedCall(run=run, result=effect)
        try:
            prepared = await self._dispatch(run)
        except BaseException:
            # The call was canceled after the body was entered, so the effect
            # may have happened. If the intent were left open, it would read as
            # still running in this process and refuse every repeat until a
            # restart.
            if effect is not None:
                effect.unknown()
            raise
        if effect is not None:
            prepared = self._settle_effect(run, effect, prepared)
        return prepared

    async def _open_effect(self, run: ToolRunContext) -> _Effect | ToolExecutionResult | None:
        """The effect's intent, or the recorded result of a repeat, or `None`."""
        session = run.execution.session
        key = self._key_of(run)
        if session is None or key is None:
            return None
        journal = intents_of(self.ctx)
        data = {"key": f"{run.name}:{key}", "tool": run.name, "callId": run.call_id}
        opened = await journal.open_once(session, TOOL_EFFECT, data)
        if isinstance(opened, Claim):
            # New, or reopened because the prior attempt failed or never started.
            return _Effect(journal=journal, session=session, claim=opened)
        earlier = as_str(opened.opened.data.get("callId"))
        if opened.running_here:
            # **Still running, here.** A second intent opened under the first
            # would displace the settle of the first call, and running the
            # effect again is the repeat that the key exists to prevent (T0).
            return error_result(
                EFFECT_IN_FLIGHT.format(call_id=earlier or "?"),
                {"name": "ToolEffectInFlight", "code": TOOL_EFFECT_IN_FLIGHT},
            )
        if opened.outcome == "done" and opened.settled is not None:
            return _repeated(opened, opened.settled)
        # Nobody saw the first attempt finish. It was settled `unknown`, or a
        # process whose resume had no kind for it left it orphaned. So the tool
        # is asked whether the effect happened (P10-13) before a new attempt is
        # decided for it.
        again = journal.record(session, TOOL_EFFECT, data)
        # A cancel while the tool is asked leaves the first attempt as unknown
        # as it was. When this intent was left open instead, it read as running
        # here and refused every repeat.
        async with journal.settling_on_failure(session, again):
            said = await self._answer(
                run.definition,
                run.by,
                run.execution.scope,
                thaw_json(run.execution.arguments),
                opened.opened,
                session,
            )
        if isinstance(said, tuple):
            journal.settle(
                session,
                again,
                _effect_settle(again.key, run.call_id, is_error=False, content=said),
            )
            meta: JsonObject = {"reconciled": True, "repeatOf": earlier}
            return ToolExecutionResult(is_error=False, content=said, meta=meta)
        return _Effect(
            journal=journal,
            session=session,
            claim=again,
            unsure_of=None if isinstance(said, NotDone) else earlier,
        )

    async def reconciled(
        self, record: SessionEvent, session: Session, *, scope: Boundary
    ) -> tuple[ContentBlock, ...] | NotDone | None:
        """Ask the tool that a call record names whether that call happened.

        This is for `resume_session` (P10-13), which has no run. It is the same
        answer that the pipeline gets for a repeat of an unknown effect. The
        tool is looked up in the view of `scope` and bound as the row that
        registered it. The record is read here: the name of its tool, and its
        arguments, parsed as the batch parses them. So the caller passes the
        record that it holds and needs nothing of this package but the types
        (T4). `None` is the `Unknown` of the tool.

        The record is either of the two that a call leaves:

        * a `tool/call`, whose arguments are the string from the model
        * the `tool/code-dispatch-start` of a Code Mode dispatch (L6b), whose
          arguments are the object that the program passed
        """
        name = as_str(record.data.get("name"))
        view = self.view(scope)
        definition = view.visible.get(name)
        if definition is None:
            return None
        bound = boundary_of(scope, self.ctx)
        by = view.by.get(name) or Running(bound, bound)
        return await self._answer(
            definition, by, bound, recorded_arguments(record), record, session
        )

    async def _answer(
        self,
        definition: ToolDefinition,
        by: Running,
        scope: Context,
        arguments: JsonValue,
        record: SessionEvent,
        session: Session,
    ) -> tuple[ContentBlock, ...] | NotDone | None:
        """What the tool says about a call: `Done`, rendered as the result of the
        call, `NotDone`, or `None` for `Unknown`. A tool with no `reconcile` and a
        tool that raises also answer `None`. The check is bound as the
        registering row (P6-26), and so is the render."""
        check = definition.reconcile
        if check is None:
            return None
        try:
            with running(by, scope):
                said = await check(arguments, record, session)
                if isinstance(said, Done):
                    return tuple(definition.render(arguments, said.value))
                return said if isinstance(said, NotDone) else None
        except Exception:
            log.warning(
                "ph.tools: %s could not reconcile; treating as unknown",
                definition.name,
                exc_info=True,
            )
            return None

    def _key_of(self, run: ToolRunContext) -> str | None:
        read = run.definition.idempotency_key
        if read is None:
            return None
        try:
            key = read(thaw_json(run.execution.arguments))
        except Exception:
            log.warning(
                "ph.tools: %s's idempotency_key raised; running unkeyed", run.name, exc_info=True
            )
            return None
        return key if isinstance(key, str) and key else None

    def _settle_effect(
        self, run: ToolRunContext, effect: _Effect, prepared: PreparedCall
    ) -> PreparedCall:
        result = prepared.result
        assert result is not None
        if run._cut_off:
            # The body was entered and did not finish. If the effect were settled
            # `failed`, as its result says, a repeat would run it again and ask
            # nothing.
            effect.unknown()
            return prepared
        effect.journal.settle(
            effect.session,
            effect.claim,
            _effect_settle(
                effect.claim.key, run.call_id, is_error=result.is_error, content=result.content
            ),
        )
        if effect.unsure_of is None:
            return prepared
        note = EFFECT_MAY_HAVE_HAPPENED.format(call_id=effect.unsure_of or "?")
        return replace(
            prepared, result=replace(result, content=(*result.content, TextBlock(text=note)))
        )

    async def _dispatch(self, run: ToolRunContext) -> PreparedCall:
        execution = run.execution
        definition = run.definition
        try:

            async def body(dispatch_exec: ToolExecution) -> ToolExecutionResult:
                # A wrapper may have replaced the signal for its delegated
                # lifetime. The body sees whatever reached it.
                run.execution = dispatch_exec
                await self.ctx.serial("tools/body", dispatch_exec, scope=dispatch_exec.scope)
                # Never start a canceled body. The check is last, after every
                # wrapper and the barrier, because a cancellation lands while a
                # slow one of them runs.
                if is_canceled(dispatch_exec.signal):
                    return aborted_result(started=False)
                # **The code of the definition runs as the agent that it was
                # invoked for** (P6-26). This holds for all three callbacks and
                # not only `execute`. `render` and `project_meta` are code of
                # the same row, reached through the same registry. A rule that
                # covered one of the three is a rule that the next reader must
                # reconstruct. Before this, nothing bound the code. `owner_for`
                # fell through to the seam, so a registration made here
                # outlived the run. `layer_for` fell through to the *global*
                # layer, so a body inside a contained child installed a tool
                # that the whole deployment could see. That was the last way to
                # escape the ceiling that P6-27 made structural. The execution
                # already carried the right scope, and nothing made it current.
                with running(run.by, dispatch_exec.scope):
                    try:
                        value = await definition.execute(dispatch_exec.arguments, run)
                    except BaseException as left:
                        run._cut_off = not isinstance(left, Exception)
                        raise
                    content = definition.render(
                        dispatch_exec.arguments, value, presented=run.presented_for(value)
                    )
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

            answered = await self.ctx.waterfall(
                "tools/execute", execution, inner=body, scope=execution.scope
            )
            result = settled("tools/execute", answered, ToolExecutionResult)
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

        # The run is published for `projected` only while the waterfall runs.
        # A row may ask what a value that it proposes would render to, and the
        # answer must come from the binding of this run and not from a fresh
        # lookup. The entry is popped in a `finally`, so a row that raises
        # cannot leave the call behind to be answered about later.
        key = _projection_key(execution)
        self._projecting[key] = run
        try:
            decision = await self.ctx.waterfall(
                "tools/post-execute", execution, result, inner=inner, scope=execution.scope
            )
        finally:
            # Remove only the entry of this run (Holdings rule 4). Another run
            # with the same key would be a different call, and the end of this
            # run is not the end of that one.
            if self._projecting.get(key) is run:
                del self._projecting[key]
        if isinstance(decision, Block):
            # A block exposes only the context that its own decision supplied.
            # The context that the body deferred belonged to an outcome that no
            # longer stands.
            return ToolExecutionResult(
                is_error=True,
                content=tuple(decision.feedback),
                error=ToolFailure(message=_message_from_content(decision.feedback)),
                additional_contexts=decision.additional_contexts,
            )
        if not isinstance(decision, Accept):
            _refuse_decision(decision, hook="tools/post-execute", expected="an Accept or Block")
        changes: dict[str, Any] = {
            "additional_contexts": (*result.additional_contexts, *decision.additional_contexts)
        }
        if decision.has_value:
            changes["value"] = decision.value
        if decision.content is not None:
            # **Content and a replaced value together are allowed,
            # deliberately** (D10). This at first raised, on the reading that
            # content is derived from the value and so two sources would
            # disagree. They are two *consumers*. See `Accept`, which owns the
            # rule.
            changes["content"] = tuple(decision.content)
        elif decision.has_value:
            # No content still means "render it". A row that means value and
            # content to agree says exactly that.
            changes["content"] = self._render(run, decision.value)
        return replace(result, **changes)

    def finish(self, run: ToolRunContext, result: ToolExecutionResult) -> ToolExecutionResult:
        """Definition-owned content finalization, then `tools/result`.

        This runs for *every* normalized outcome, pipeline failures included,
        and that is the purpose. A tool whose content needs a final transform
        must not have to trust that policy let its result through.
        """
        finalize = run.definition.finalize_content
        if finalize is not None:
            try:
                # This is the last definition-owned body of the pipeline, and
                # it also runs for failures. So it is the likeliest of these
                # bodies to register a follow-up, and the least acceptable one
                # to leave unbound.
                with running(run.by, run.scope):
                    replacement = finalize(run.execution, result)
            except Exception:
                log.exception("ph.tools: %s.finalize_content raised; content preserved", run.name)
                replacement = None
            if replacement is not None:
                result = replace(result, content=tuple(replacement))
        self.ctx.emit("tools/result", run.execution, result, scope=run.scope)
        return result


def _discard(items: list[Any], target: object) -> None:
    with suppress(ValueError):  # already released
        items.remove(target)


def _failure(error: object, *, started: bool = False) -> ToolExecutionResult:
    """Normalize any raised value into a structured `is_error` result (B5).

    A tool that raises must not be able to stop the loop, and the model needs
    the failure in the same shape as every other result. The `kind` of the
    failure comes from the error class that knows it.
    """
    if isinstance(error, Canceled):
        return aborted_result(started=started)
    # The kind is the answer of the error itself, and not a mapping that this
    # function keeps. A boolean here collapsed three kinds into two and
    # silently reported an abort as a failure.
    kind: FailureKind = error.failure_kind if isinstance(error, HarnessError) else "failed"
    settled = error_result(error_message(error), error_info(error), kind=kind)
    # **Whether the failure ended the turn also comes from the error** (C13),
    # from the same class and for the same reason. Only the raiser knows that
    # a raise concludes a turn. A result that is built from the raise and
    # drops that fact leaves the loop running.
    if isinstance(error, HarnessError) and error.concludes_turn:
        return replace(settled, concludes_turn=True)
    return settled


def _message_from_content(content: Sequence[ContentBlock]) -> str:
    text = text_of(content, placeholder=lambda kind: f"[{kind} content]")
    return text or "tool result blocked by post-execute policy"


def _unresolved(
    _args: object,
    _run: ToolRunContext,
) -> NoReturn:  # pragma: no cover - never dispatched
    raise ToolNotFoundError("<unresolved>")


_UNRESOLVED = ToolDefinition(
    name="<unresolved>",
    description="placeholder for a call that failed before its tool resolved",
    parameters={},
    output=ToolOutput(schema={}, render=lambda _a, _v: ()),
    execute=_unresolved,
)
"""The definition that a run carries when `create_execution` itself failed.

A run always has a definition, so `finish` never needs to ask whether one
exists. This one renders nothing and is never dispatched."""


def register_when_composed(ctx: Context, build: Callable[[], ToolDefinition | None]) -> None:
    """Register a tool when the profile is whole, if `build` returns one.

    This is for a tool whose *existence* depends on something that another row
    supplies. A `skill` tool needs an installed skill, and a `task` tool needs a
    subagent provider.

    **A tool that cannot work must not be registered.** It must not be
    registered and then refused. Its schema and description are in the prompt
    of every request. So a tool that the deployment cannot perform spends the
    context window on a capability that is not there, and the model spends a
    turn to discover that.

    **The trigger is `profile/mounted`, not `ctx.inject`.** `inject` waits on a
    *service key*, and neither condition is one. "The skills registry is
    non-empty" and "some subagent provider is registered" are facts about the
    contents of a service. `profile/mounted` is the one moment at which the
    profile is known to be whole.

    `build` returns `None` to mean "not in this deployment". A name that is
    already registered is left alone. So a second dispatch does nothing and does
    not raise a duplicate-name `ValueError`, which on this event would abort
    the process.
    """

    def once() -> None:
        definition = build()
        # `DEPLOYMENT` is the chain of the mount. It names the view that
        # `scope=None` always read (P6-32). It is NOT a union over agents. A
        # tool that is registered on the scope of one agent is invisible here
        # (verified). So this checks "already claimed at the deployment layer"
        # and nothing more. That is the check that this code always made. A
        # true audit of "is it taken *anywhere*" is a question per layer, which
        # no single boundary answers.
        if (
            definition is not None
            and ctx.require(TOOLS).get(definition.name, scope=DEPLOYMENT) is None
        ):
            # No `scope=`. The call carried `scope=ctx` until P6-25, and that
            # was the sign of the problem. A listener that fired after `apply`
            # had returned saw no activation. So the registration would have
            # landed on the tool registry and outlived this row, and the only
            # protection was that somebody remembered the argument. Dispatch
            # now runs a listener as an effect of the scope that registered it,
            # so the ordinary call is correct here as it is everywhere else.
            ctx.require(TOOLS).register(definition)

    ctx.on("profile/mounted", once)


@plugin("tools", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the tool registry."""
    ctx.provide(TOOLS, ToolRuntime(ctx=ctx, default_mode=config.mode))
