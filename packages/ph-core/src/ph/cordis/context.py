"""Context, services, effects, scopes and the four dispatch modes.

The Python subset of Cordis that pH needs (D1). A `Context` is three things at
once, exactly as in dsh:

* a **repository of services** — a plugin claims `ctx.<key>` with `provide()`
  and every other plugin finds it by key rather than by import;
* an **event bus** — `emit` / `parallel` / `serial` / `bail` / `waterfall`,
  with the dispatch mode fixed by the declaration (see `ph.cordis.events`);
* a **disposal scope** — every registration and every acquired artifact is an
  effect that unwinds when the scope disposes (invariant I2).

There are three kinds of context, and the difference is load-bearing:

| kind | built by | provides into | its listeners reach |
|---|---|---|---|
| root | `Context()` | itself | everything |
| activation scope | `reconcile()`, for a plugin | the realm the row was mounted in | everything |
| isolated scope | `ctx.scope()`, for an agent | itself | that agent alone |

A row's service is therefore visible to every sibling row, while an agent's
registration shadows the global one for that agent only. One rule,
`reaches()`, decides visibility for event dispatch and for every scoped registry.

Two deliberate departures from the TypeScript original, both because Python has
no synchronous-await:

* activation is driven by an explicit ``await ctx.reconcile()`` rather than by
  a microtask, so a test or a loader knows exactly when the plugin tree has
  settled;
* ``effect()`` is a coroutine (an artifact may need awaiting to acquire), while
  ``add_disposer()`` is the synchronous path used by `on()` and `provide()`.

@module ph.cordis.context
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import anyio

from .errors import InactiveScopeError, ServiceConflictError, ServiceNotFoundError
from .events import events as event_registry
from .plugin import PluginSpec, normalize_plugin

__all__ = [
    "Context",
    "Disposer",
    "ForkScope",
    "Hook",
    "Listener",
    "is_bailed",
    "maybe_await",
]

log = logging.getLogger("ph.cordis")

Disposer: TypeAlias = Callable[[], Any]
"""A teardown callable. It may return an awaitable; `dispose()` awaits it."""

Listener: TypeAlias = Callable[..., Any]

_MAX_RECONCILE_ROUNDS = 64
_MISSING: Any = object()


async def maybe_await(value: Any) -> Any:
    """Await `value` when it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def is_bailed(value: Any) -> bool:
    """Whether a listener's return value stops a `serial`/`bail` dispatch.

    Ported verbatim from cordis: anything but `None` and `False` bails. `0` and
    `""` bail, deliberately — a decision object is never falsy by accident, and
    treating a legitimate zero as "no answer" is the bug this rule prevents.
    """
    return value is not None and value is not False


@dataclass(slots=True)
class Hook:
    """One registered listener record."""

    ctx: Context
    callback: Listener
    prepend: bool = False
    global_: bool = False


def _invoke(hook: Hook, *args: Any) -> Any:
    """Call one listener as an effect of the scope that registered it (P6-25).

    **The one place ownership is established for a dispatch**, and the reason it
    is a function rather than a line in each loop: there are five dispatch modes,
    and "remember to bind" written five times is the shape of rule this row
    exists to stop relying on. A sixth mode cannot omit what it never writes.

    Sync and async are one path here. A listener returning an awaitable gets it
    back wrapped in `_as_owner`, so whoever awaits or spawns it receives a
    coroutine that binds itself; a listener returning a value has already run
    inside the binding below. Callers need no idea which kind they have, which is
    what lets `emit` spawn and `serial` await the same result.

    **The binding is spelled inline rather than through `running`**, and that is
    a measurement rather than a preference: `@contextmanager` costs a generator,
    an `__enter__`, an `__exit__` and a `StopIteration`, and `emit` fires once
    per streamed chunk. Cross-process A/B on a 2 000-chunk turn, headless with
    two listeners: **8.2 ms** before this row, **14.3 ms** through a
    `@contextmanager`, **10.5 ms** inline. `running` keeps the
    once-per-activation path, where the difference is unmeasurable.

    An earlier draft of this paragraph priced the inline form at 120 ns per
    listener and reported the row as free. Both were wrong, and the review that
    caught it is the reason for the `None` branch below: that figure counted only the
    `set`/`reset` pair (**93 ns** measured) and missed the `inspect.isawaitable`
    this same function introduced — **172 ns**, run twice per listener, which
    made the awaitability question cost more than the ownership it was serving.
    """
    # Inline, and through the scope's memoized self-pair. Building a
    # `Running` here costs **206 ns** — a frozen dataclass sets its fields
    # through `object.__setattr__` — against **22 ns** for this load-and-branch,
    # and it runs once per listener per chunk: 1.76 ms became 2.84 ms on the
    # 2 000-emit bench below before the memo, which is the same 55% this
    # function's inline binding exists to have avoided in the first place.
    owner = hook.ctx
    pair = owner._running_self
    if pair is None:
        pair = owner._running_self = Running(owner, owner)
    token = _ACTIVATING.set(pair)
    try:
        result = hook.callback(*args)
        # `None` first, and it is most of the win. A listener that returns
        # nothing is the overwhelming case, `inspect.isawaitable` costs **172 ns**
        # against **16 ns** for this identity check, and it runs a second time in
        # `emit` on the value this just returned — so the awaitability question
        # was costing more per listener than the binding it was added to serve.
        if result is None:
            return None
        return _as_owner(hook.ctx, result) if inspect.isawaitable(result) else result
    finally:
        _ACTIVATING.reset(token)


@dataclass(slots=True)
class _Effect:
    dispose: Disposer
    label: str
    done: bool = False


@dataclass(slots=True)
class _Provision:
    value: object
    owner: Context


@dataclass(slots=True)
class _Dependent:
    """A registration waiting for services: a mounted plugin or an `inject()`.

    Owns the one deactivation sequence. Every path that tears an activation
    down — a service disappearing, an unmount, a failed `apply` — goes through
    `deactivate()`, so the protocol lives in one place.
    """

    ctx: Context
    keys: tuple[str, ...]
    activate: Callable[[Context], Any]
    label: str
    active: bool = False
    scope: Context | None = None
    disposed: bool = False

    def ready(self) -> bool:
        return self.ctx.active and all(self.ctx.has(key) for key in self.keys)

    def deactivate(self) -> Awaitable[None] | None:
        """Drop the activation scope, unwinding everything it registered."""
        scope, self.scope, self.active = self.scope, None, False
        self.ctx._runtime.dirty = True
        return scope.dispose() if scope is not None else None

    def retire(self) -> Awaitable[None] | None:
        """Deactivate and leave the tree for good."""
        self.disposed = True
        return self.deactivate()


@dataclass(slots=True)
class _Runtime:
    """State shared by every context in one tree, held by reference."""

    hooks: dict[str, list[Hook]] = field(default_factory=dict)
    dependents: list[_Dependent] = field(default_factory=list)
    dirty: bool = True
    background: set[asyncio.Future[Any]] = field(default_factory=set)


class ForkScope:
    """A mounted plugin: its config, and the scope its activation owns.

    A fork survives deactivation. When an injected service disappears the
    activation scope is disposed — unwinding everything the plugin registered —
    but the fork stays mounted, so re-providing the service re-activates it.
    """

    __slots__ = ("_config", "_dependent", "_parent", "_spec", "_unmount")

    def __init__(self, parent: Context, spec: PluginSpec, config: Any) -> None:
        self._parent = parent
        self._spec = spec
        self._config = config
        self._dependent, self._unmount = parent._register_dependent(
            spec.inject, self._apply, label=f"plugin({spec.name})"
        )

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def active(self) -> bool:
        return self._dependent.active

    @property
    def ctx(self) -> Context | None:
        """The activation scope while the plugin is active, else `None`."""
        return self._dependent.scope

    @property
    def config(self) -> Any:
        return self._config

    async def _apply(self, ctx: Context) -> None:
        await maybe_await(self._spec.apply(ctx, self._spec.resolve_config(self._config)))

    async def dispose(self) -> None:
        """Unmount the plugin: deactivate it and drop it from the tree."""
        await maybe_await(self._unmount())
        await self._parent.reconcile()


@dataclass(frozen=True, slots=True)
class Deployment:
    """Everything this deployment holds — the widest boundary, said out loud.

    **"Widest" along the restriction axis only, and that is worth being exact
    about**: a reader given `DEPLOYMENT` resolves the *mount's* isolation chain,
    which no restriction narrows — but an **agent-scoped registration is not on
    it**. `view(DEPLOYMENT)` does not see a tool registered on one agent's scope,
    verified; it is the view `scope=None` always read, named. A caller asking
    "is this available to *anyone*" (an audit, a collision check across agents)
    is asking a per-layer question this sentinel does not answer, and making
    `boundary_of` iterate layers to answer it would turn a boundary into a
    search.

    **The point is that it has a name** (P6-32). A policy reader takes a scope to
    answer for, and `scope: Context | None = None` made two different things one
    spelling: *"I did not state a boundary"*, which is an absence of information,
    and *"I mean the deployment"*, which is a legitimate answer that `ph doctor`,
    the prompt catalog and a root spawn all need. With one spelling every reader
    has to pick a default for the ambiguous case, and the convenient default is
    `scope or self.ctx` — the mount, which is the widest boundary there is.

    That default is one root cause wearing four faces, each found and fixed
    separately as though it were local: `owner_for(None)` falling through to the
    seam so a registration outlived its row (P6-12); `_scope_of(agent)` returning
    `None` so an agent-scoped screen applied to nobody (P6-24); `held_by`'s
    `None` reaching `reach`/`names` as the *unrestricted* set, so an unreadable
    parent handed a child everything the deployment holds (P6-31); and
    `_enforce` skipping its containment check on the same `None`.

    So "everything" gets a name and the default comes off. `DEPLOYMENT` is
    greppable — "who asked for the widest boundary" is one search — where passing
    the mount `Context` explicitly would not be, since `ctx` is in scope
    everywhere and reaches for itself.

    **Not `None` meaning "nothing", which was the other candidate.** That trades a
    silent-wide failure for a silent-narrow one: an empty tool set or an empty
    prompt degrades a model quietly rather than erroring, and nobody notices. The
    half that does the work is removing the *default*, so a call site that states
    no boundary fails mypy rather than any runtime rule.
    """


DEPLOYMENT = Deployment()
"""The one instance. There is nothing to construct and nothing to configure."""

Boundary: TypeAlias = "Context | Deployment"
"""What a policy reader takes: a scope, or `DEPLOYMENT` for all of them.

Deliberately **not** `| None`. A reader whose parameter has no default cannot be
called without answering the question, and mypy is what asks — at build time,
against every call site, rather than a runtime rule that only fires on the paths
a test happens to drive.

**A *registration* takes a `Context`, never this**, and so do the two
dispatch-time resolvers — `Context.owner_for` and `Context.layer_for` — which
read the running binding rather than a stated value. `ctx.fs`'s `screen`,
`ToolRuntime`'s `register`/`restrict`/`guard`/`present_as` and every `claim_*`
call hand their scope to `add_disposer` and to `reaches` — it is what the
registration is an *effect of* and what it is visible *to*, P6-12's two
questions. `DEPLOYMENT` is neither: there is nothing for it to unwind with. So a
mechanical `Context | None → Boundary` sweep must stop at the readers, and the
registration methods P6-32 defers are deferred to P6-12's mechanism rather than
merely unconverted."""


def boundary_of(scope: Boundary, mount: Context) -> Context:
    """Resolve a stated boundary to the scope that answers it.

    One narrowing site rather than one per seam, and mypy is the reason it has to
    be: `scope is DEPLOYMENT` reads as the obvious spelling and does **not**
    narrow the union — identity against a value is not a type guard — so every
    seam writing it by hand would either repeat an `isinstance` or reach for a
    cast. Written once, the union is discharged here and a reader gets a
    `Context`.
    """
    return mount if isinstance(scope, Deployment) else scope


@dataclass(frozen=True, slots=True)
class Running:
    """Who is running — as the *two* questions it has to answer (P6-29).

    Public because it is also **what a registry records at registration time**.
    Every one of them invokes a row's body later — a tool, a command, a
    compaction note, a prompt section, a status field, a diagnostic, a telemetry
    sink, and each of the five single-slot providers — and to enter the right
    binding then, they have to have kept both answers now. `Context.running_for`
    hands them exactly this object, so "what registration recorded" and "what the
    invoker binds" are one type rather than two fields a registry pairs up by
    hand and can pair up wrongly.

    P6-12 established that a registration answers two questions with one value,
    and split the readers into `Context.owner_for` (whose lifetime) and
    `Context.layer_for` (who may see it). It did not split what they read, and
    for two rows it did not have to: an `apply` and a listener both run *as one
    scope*, so the owner and the layer are the same object and one `Context` in
    the var said both truthfully.

    A body a **registry** invokes is the case where they come apart. A tool's
    `execute` belongs to the row that registered the tool — that is whose code it
    is, and I2 says a registration made from it unwinds with that row — while it
    is visible to the *agent* it was invoked for, which is B7's answer and a
    different scope entirely. P6-26 bound `execution.scope` for both, which made
    containment right and quietly re-answered lifetime as "the agent": a tool
    registered from a tool body outlived the row that registered the tool,
    verified.

    Two fields rather than a second `ContextVar` because they are one fact — *who
    is running* — read two ways, and two variables can be set out of step. The
    only way to bind one is `running()`, which takes both.

    **What this makes possible is the rest of the rule.** `CompactionSeam.render`
    and `SystemPromptService.assemble` each hold the target scope in a local and
    invoke a row's body two lines later; they could not join P6-26 because the
    target is a *layer* while the owner is still the registering row, and one
    `Context` cannot say both. The same shape covers `StatusField.read`,
    `Diagnostic.read` and a `claim_slot` provider, where there is no target at
    all and both halves are simply what registration recorded — which is why "a
    provider has no agent" stopped being an objection: nothing needs an agent for
    the *owner* half.
    """

    owner: Context
    """Whose lifetime a registration made now joins — `Context.owner_for`."""
    layer: Context
    """Which scope it is visible to — `Context.layer_for`."""

    def add_disposer(self, dispose: Disposer, *, label: str = "") -> Disposer:
        """Release when **either** scope ends — the pair's lifetime, not a half's.

        Before P6-29 the two could not diverge, so "when does this go away" had
        one answer and `Context.add_disposer` was the whole of it. Once a body a
        registry invokes registers as its row *for* an agent, they are unrelated
        branches and either can end first — and picking one is wrong in a
        different direction each way. Owning it by the row alone leaves whatever
        the layer keyed on a disposed scope: `ToolRuntime._layers` went 2 → 2
        across an agent's disposal where it goes 2 → 1. Owning it by the agent
        alone lets a registration outlive the row whose code made it, which is I2
        verbatim. So it is neither: the registration is meaningful only while
        both are alive.

        **Here rather than in each registry**, because a registry keyed on the
        layer has to ask this question and there is now more than one — the tools
        registry keys `_layers` by `layer.isolation`, and `ph.seams.skills` keys
        `_restrictions` the same way. A rule with two sites and one
        implementation is the shape this whole row exists to delete. It lives on
        `Running` and not on `Context` because `Context.add_disposer(..., also=)`
        would ask one scope to know about a pair it does not hold; the pair is
        the object that holds both.

        The once-guard is `Context.add_disposer`'s: the releaser it returns flips
        `_Effect.done` and drops the effect before calling through, so `finish`
        calling the layer's releaser re-enters exactly one level and stops at a
        guard that already exists. Whichever scope ends first removes the sibling
        effect from the other, so nothing is left behind on the one that did not.
        `claim_key`, `claim_entry` and `claim_slot` take `Context | Running`, so a
        seam opts in by handing them the pair instead of `by.owner`.
        """
        drop_layer: Disposer | None = None

        def finish() -> None:
            if drop_layer is not None:
                drop_layer()
            dispose()

        drop = self.owner.add_disposer(finish, label=label)
        if self.layer is not self.owner:
            drop_layer = self.layer.add_disposer(drop, label=f"{label}@visible")
        return drop


_ACTIVATING: ContextVar[Running | None] = ContextVar("ph.cordis.activating", default=None)
"""Whose code is running: an `apply`'s activation scope, a listener's owner, or
the (row, target) pair a registry enters around a body it invokes.

See `Context.current_owner` and `Context.current_layer`, and `_invoke` for the
listener half."""


class running:
    """Bind "who is running" for the duration of a block.

    Public since P6-26, because the fourth consumer is in another package's
    layer: a registry that invokes a body — a tool's `execute`, a command's
    `run` — has to make that body's scope current, and reaching into a private
    of `ph.cordis.context` to do it would be the seam layer depending on an
    underscore.

    **It takes the `Running` itself**, which is how every registry calls it: the
    pair exists so a registry stops keeping two fields it can pair up wrongly,
    and taking it apart into two positional arguments at nine call sites would
    have handed that exact mistake back — `running(a.owner, b.layer)` is silently
    valid. `layer=` beside a pair *overrides* the visibility half, which is the
    one thing a caller legitimately knows better: `ph.seams.fs` binds the agent
    whose path is being resolved rather than the scope the resolver registered on.

    **`None` binds nothing**, for the empty half of an at-most-one slot. Five
    seams hold a provider that may not be registered, and each was spelling
    `running(by.owner, by.layer) if by is not None else nullcontext()` — the same
    conditional five times, plus two private `_as_provider()` helpers that existed
    only because the expression was too long to inline. The state is real, so it
    belongs in the one place that knows what binding means.

    **`layer` defaults to `owner`, which is the whole of the ordinary case.**
    An `apply` and a listener each run as one scope, so `running(scope)` still
    means what it always did and every existing call site is unchanged. Passing
    the second argument is the registry case (P6-29): the body belongs to the row
    that registered it and is visible to the target it was invoked for, and those
    are two scopes. Spelled as one call with a default rather than two functions,
    because a caller choosing between `running` and `running_as` is a caller who
    can pick the wrong one — here the *shape of the call* says whether the two
    questions have one answer, and the record it binds cannot be half-set.

    **A class rather than a `@contextmanager`, and that is a measurement.** The
    generator form costs a frame, a `StopIteration` and two `send`s: **709 ns**
    per entry against **316 ns** for byte-identical branch logic. This stopped
    being a once-per-activation helper in P6-26 and P6-29 — it is now entered
    once per tool call, once per slash command, once per prompt row per
    `assemble`, and once per telemetry sink per record, at twenty-one sites. In
    situ against the pre-row build on base+headless, a **tool call is at parity:
    26.67 µs against 26.74** — the cheaper form pays for the whole pair
    mechanism. `_invoke` still spells the two calls inline for the same reason
    one scale down: `emit` fires per streamed chunk, where even 316 ns is too
    much.

    **What it does not pay for is `assemble`, and that is worth stating**
    (§5 rule 6). Prompt assembly enters this once per *contributing row*, so the
    cost scales in the thing plugins add: **55.06 µs against 49.85** with the
    nine rows base+headless contributes, ~1.2 µs per row beyond that. Once per
    turn, against a model round-trip, so it is affordable — but a deployment with
    forty prompt rows pays forty bindings, and the honest place to notice that is
    here rather than in a profiler later.

    **Single-use, like every other context manager built at its `with`.** The
    token lives on the instance, so re-entering one object would lose the outer
    token; all twenty-one call sites construct inline, which is the only shape
    that makes sense for a binding named after the block it wraps.
    """

    __slots__ = ("_pair", "_token")

    def __init__(self, owner: Running | Context | None, layer: Context | None = None) -> None:
        pair: Running | None
        if owner is None:
            pair = None
        elif isinstance(owner, Running):
            pair = owner if layer is None or layer is owner.layer else Running(owner.owner, layer)
        elif layer is None or layer is owner:
            # The coinciding case, which is every `apply` and every listener: the
            # scope keeps one record rather than building one per entry. Cleared
            # in `dispose`, because the pair holds `self` twice and a disposed
            # context is meant to become collectable the moment its parent drops
            # it.
            pair = owner._running_self
            if pair is None:
                pair = owner._running_self = Running(owner, owner)
        else:
            pair = Running(owner, layer)
        self._pair = pair
        self._token: Token[Running | None] | None = None

    def __enter__(self) -> None:
        if self._pair is not None:
            self._token = _ACTIVATING.set(self._pair)

    def __exit__(self, *_exc: Any) -> None:
        if self._token is not None:
            _ACTIVATING.reset(self._token)
            self._token = None


async def _as_owner(scope: Context, awaitable: Any) -> Any:
    """Await something as an effect of `scope`, binding when the body *runs*.

    The half of P6-25 its first version got backwards. An `async def` listener
    called by `emit` only *builds* a coroutine — the body runs later, on a task
    `_spawn` creates — and a task copies the context at creation, which is after
    a binding around the *call* has already been reset. So the coroutine carried
    the emitter's binding and never the listener's, and both gaps this row exists
    to close survived verbatim for the most ordinary listener shape in the
    harness: `current_owner()` read `None` inside the body, and anything it
    registered landed on the seam and outlived its row.

    Wrapping the awaitable moves the binding *inside* the body, where it is the
    coroutine's own first act rather than something its creator did before
    handing it over.
    """
    # Through `running` rather than the inline pair `_invoke` uses: this is
    # already an extra coroutine frame around an `await`, so a generator's
    # `__enter__`/`__exit__` is noise beside a scheduler hop — and it leaves the
    # module with two spellings of the binding instead of three.
    with running(scope):
        return await awaitable


class Context:
    """One node of the plugin tree: services, listeners and effects."""

    __slots__ = (
        "__weakref__",
        "_active",
        "_children",
        "_effects",
        "_isolation",
        "_label",
        "_module",
        "_parent",
        "_provide_to",
        "_running_self",
        "_runtime",
        "_services",
    )

    # Declared for the type checker; `__slots__` still owns the storage.
    _parent: Context | None
    _label: str
    _module: str
    _children: list[Context]
    _effects: list[_Effect]
    _services: dict[str, _Provision]
    _active: bool
    _isolation: Context | None
    _runtime: _Runtime
    _provide_to: Context
    _running_self: Running | None

    def __init__(
        self,
        parent: Context | None = None,
        *,
        label: str = "root",
        provide_to: Context | None = None,
        module: str = "",
        isolated: bool = False,
    ) -> None:
        self._parent = parent
        self._label = label
        self._module = module
        self._children = []
        self._effects = []
        self._services = {}
        self._active = True
        self._running_self = None
        self._runtime = parent._runtime if parent is not None else _Runtime()
        self._provide_to = provide_to if provide_to is not None else self
        # Fixed at construction, so `reaches()` is a lookup rather than a walk.
        self._isolation = self if isolated else (parent._isolation if parent is not None else None)
        if parent is not None:
            parent._children.append(self)

    @classmethod
    def _activation_scope(cls, dependent: _Dependent) -> Context:
        """The transparent scope a plugin's `apply` runs in."""
        owner = dependent.ctx
        return cls(owner, label=dependent.label, provide_to=owner._provide_to, module=owner._module)

    def owner_for(self, scope: Context | None = None) -> Context:
        """Whose lifetime a registration made *now* belongs to (I2, P6-12).

        Called as `self.ctx.owner_for(scope)` from a seam, where `self.ctx` is
        the seam's own context and is therefore the fallback rather than the
        answer. Three cases, in order:

        * **an explicit `scope=`** — the caller said so, and the only thing that
          overrides the rest. It now means what it always read as, *"register on
          someone else's lifetime"* (an agent's, so the registration shadows a
          global one for that agent alone), instead of being a rule twenty
          modules had to remember to get the *ordinary* case right;
        * **the activation scope**, when a row's `apply` is what is running —
          the scope cordis disposes when the row unmounts, so the registration
          goes with it. This is the fix;
        * **this context**, outside any activation: today's behaviour, kept so
          the change is strictly additive for callers that are not rows at all
          (a test standing a service up by hand, a mode wiring one directly).

        **This does not take a `Boundary`, and P6-32 settled that deliberately**
        rather than leaving it unconverted. That row deletes a `None` which
        resolves *wider* than any stated value. `None` here resolves through
        `_ACTIVATING` to whoever is running — the row, verified — which is the
        **narrowest** correct answer and the whole of P6-12. It is not "give me
        everything", it is "I am not registering on someone else's behalf".

        The third branch does fall back to the seam, and that is a widening in
        I2's sense — but it is the one P6-12 diagnosed and it warns where it
        happens (below). See `Boundary`.

        **A disposed activation scope declines, and says so.** Contextvars
        propagate into tasks spawned from `apply`, so a background task
        registering after its row unmounted would otherwise hit
        `add_disposer`'s `InactiveScopeError` where today it leaks quietly.
        Raising is a behaviour change this row did not sign up for — but a
        silent fallback would make the one path that still outlives its owner
        both invisible and unmeasurable, so it warns. That is the only branch
        here that fails open, and §5 rule 6 wants it visible where it happens.
        """
        if scope is not None:
            return scope
        activating = _ACTIVATING.get()
        if activating is None:
            return self
        owner = activating.owner
        if owner.active:
            return owner
        log.warning(
            "ph.cordis: %s registered after its activation scope %s was disposed; "
            "the registration will outlive the row that made it (I2)",
            self.path,
            owner.path,
        )
        return self

    def layer_for(self, scope: Context | None = None) -> Context:
        """Which scope a registration is *visible to* — the other question (P6-12).

        **This does not take a `Boundary` either, and for `owner_for`'s reason
        rather than its own** (P6-32). That one is a lifetime question, so
        `DEPLOYMENT` is obviously not an answer; this one is a *visibility*
        question whose result feeds `reaches`, which is exactly the shape a
        boundary has — so the lifetime argument does not transfer and the
        exemption needs its own.

        It is that both are **dispatch-time resolvers**: they read `_ACTIVATING`
        to answer "who is running", where a `Boundary` parameter is a caller
        *stating* something. `None` here means "ask the binding", and inside a
        row it resolves to that row — the narrowest answer, verified. A caller
        with a boundary to state passes `scope=`; there is no third thing
        `DEPLOYMENT` could mean.

        Separate from `owner_for` because they are different questions that had
        the same answer, and the review of this row found the conflation in five
        registries: a tool layer key, a prompt section's `reaches` target, a
        skill restriction's bucket, a compaction note's `reaches` target, and an
        fs screen's. Visibility is `scope or this context` — unchanged, and
        deliberately *not* the activating row, because moving it would change
        what an agent can see (B7) rather than when a registration goes away.

        Spelled out so a call site says which question it is asking. A registry
        that needs both writes `self.ctx.owner_for(scope)` and
        `self.ctx.layer_for(scope)` side by side, and a reader can see that it
        asked both rather than reusing one answer for two purposes.

        **It follows the running binding too, and the isolation rule is why that is
        safe** (P6-26). This deliberately did not, on the grounds that moving
        visibility would change what an agent can see — but the two cases the
        objection conflates are told apart by `isolation`, for free. A row's
        activation scope is *not* isolated, so it inherits the mount's isolation
        and a root row's registration still lands on the global layer, unchanged.
        An agent's scope *is* its own isolation, so a body running for an agent
        lands on that agent's layer — which is the containment P6-27 made
        structural and this was the last way to escape: a tool body inside a
        contained child was installing **globally visible** tools.

        **The disposed branch fails open, and deliberately does not warn.**
        Falling back to `self` means the seam's own context, whose isolation is
        `None` — the *global* layer — so a body that outlived the scope it ran
        for registers something the whole deployment can see. That is the mirror
        of `owner_for`'s fail-open, in B7's direction rather than I2's, and it is
        reachable the same way: contextvars propagate into tasks, so one spawned
        from a tool body and still running after its agent went away takes this
        branch. Silent because it cannot happen alone — all five call sites ask
        `owner_for` about the same `scope` a line or two later
        (`tools/registry._claim`, `system_prompt/assembly.section`,
        `seams/skills.restrict`, `seams/compaction.note`), and that is where it
        is logged. Two lines for one registration makes the honest one easier to
        miss.
        """
        if scope is not None:
            return scope
        activating = _ACTIVATING.get()
        if activating is None:
            return self
        layer = activating.layer
        return layer if layer.active else self

    def running_for(self, scope: Context | None = None) -> Running:
        """Both answers at once, for a registry that has to *record* them (P6-29).

        **Every site that needs both should ask once**, and the reason is the
        warning: `owner_for` logs when the activation scope it would return has
        already been disposed, and that branch is meant to be audible. Two calls
        for one registration make it audible twice, which is how a reader learns
        to skim it.

        It matters most for a registry that will invoke the registered body
        *later*, because that one must **keep** both until then. Two fields on
        its own record is two things to hold in step, and the failure mode is
        silent — `ph.seams.compaction` shipped a `_NoteRegistration.owner` that
        held `layer_for`'s answer, the conflation this row is about wearing the
        other question's name. One object cannot be half-updated, and `running()`
        takes it whole so no call site has to take it apart again.
        """
        return Running(self.owner_for(scope), self.layer_for(scope))

    @staticmethod
    def current_owner() -> Context | None:
        """Whose *lifetime* the running code belongs to, or `None` (P6-12).

        Was `current_scope`, and the rename is P6-29's point rather than tidying:
        "the current scope" was one name for the two questions `owner_for` and
        `layer_for` were split apart to keep separate, so the public reader was
        still spelling the conflation the private readers had stopped making.
        There are two now, named after the two they serve.

        **The owner a seam registration should default to.** Cordis already
        builds exactly the right scope in `_activation_scope` and hands it to
        `apply` — it is the thing disposed when the row unmounts — but a seam
        only ever saw `ctx`, and `ctx.commands.register(...)` gave the registry
        no way to know who was calling. So every seam defaulted the owner to its
        *own* context, and a registration made by a row became an effect of the
        **seam**, outliving the row that made it: I2 held only where a caller
        remembered `scope=`, which was 1 of 38 call sites in the tree.

        A `ContextVar` rather than a parameter threaded through forty
        signatures, because the answer is a property of *who is running*, not of
        what they are asking for — and because the seam methods are the API
        rows use, which cannot grow a mandatory argument without breaking every
        one of them.

        **It answers "whose code is running", across both ways cordis runs any**
        (P6-25). An `apply` binds its activation scope; a dispatch binds the
        scope that registered the listener. So a registration belongs to the row
        that made it whether that row is being mounted, is handling an event
        fired by another row, or is handling one fired long after every `apply`
        returned.

        It was briefly only the first of those, and the two gaps either side were
        what P6-25 closed: a listener dispatched from another row's `apply` had
        its registration attributed to the *emitting* row, and one made after
        `apply` returned saw `None` and fell back to the seam — outliving its row
        exactly as before the mechanism existed.

        **What P6-26 bound**: every definition-owned body the *tools* registry
        invokes — `execute`, `render`, `project_meta`, `finalize_content` — and a
        command's `run`, each to the agent it runs *for*. Not one statement but
        all of them, because they are one row's code reached through one
        registry, and `render` proved what a partial rule costs: with only
        `execute` wrapped it ran bound to *whichever `tools/execute` wrapper
        called the inner* — measured as `plugin(session-checkpoint-policy)` on
        the headless profile, a stranger's row picked by what happens to be
        mounted.

        **What P6-29 finished**, once the binding could hold both answers rather
        than one: the four row bodies the *other* registries invoke — a
        compaction note's `text`, a prompt section's and a prompt context's, a
        status field's `read`, a diagnostic's — plus the variable and tool
        providers beside them, every telemetry sink, and all five single-slot
        providers (`compaction`, `code_runtime`, `sandbox`, `workspace`, and
        `fs`'s rebase resolver) — and `ToolDefinition.classify`, the last of the
        definition's own bodies. "A provider has no agent" had been the reason to
        defer most of those, and it was only ever an objection to the *layer*
        half; the owner half needs no agent at all, so `claim_slot` records the
        pair and the scheduler binds a classifier from the same view that
        resolved the tool.

        **What still runs unbound**, and this list is meant to be exhaustive —
        `test_registration_ownership.py` enumerates the surface by introspection
        and fails on anything not classified, so it is checked rather than
        believed:

        * a `waterfall`'s `inner` — the *producer's* body, which nothing
          registered, so it inherits its caller's binding. `tools/execute`'s
          inner is the exception that binds itself, per the paragraph above;
        * `CodeRuntimeSeam`'s SDK renderers, which the seam *hands out* rather
          than invokes — `sdk_renderer(language)` returns the row's callable and
          its caller calls it. Binding that means returning a wrapper instead of
          the row's own function, which changes what a caller holds in order to
          fix where it runs. A different shape from every entry above, and one to
          decide on rather than to slip in;
        * teardown callbacks — a disposer, a workspace release — which run *as*
          a scope unwinds. Binding one would offer a lifetime to register on at
          the moment that lifetime is ending, and `add_disposer` refuses an
          inactive scope for the same reason.

        Contextvars propagate into tasks, so a coroutine spawned from a bound
        callback carries the binding; `Context.owner_for` declines a *disposed*
        scope and warns rather than raising.
        """
        activating = _ACTIVATING.get()
        return activating.owner if activating is not None else None

    @staticmethod
    def current_layer() -> Context | None:
        """Which scope the running code is *visible to*, or `None` (P6-29).

        The other half of `current_owner` above, and equal to it for everything
        cordis itself runs: an `apply` and a listener each run as one scope, so
        both readers answer the same object and the distinction costs nothing to
        carry. It is a registry-invoked body that separates them — a tool's
        `execute` is the registering row's code (`current_owner`) run for one
        agent (`current_layer`) — and `Context.layer_for` is where that answer is
        actually consumed. This exists so a *test* can ask the question a body
        can otherwise only ask by registering something and seeing where it went.
        """
        activating = _ACTIVATING.get()
        return activating.layer if activating is not None else None

    # ------------------------------------------------------------- identity --

    def __repr__(self) -> str:
        return f"<Context {self.path}>"

    @property
    def label(self) -> str:
        return self._label

    @property
    def parent(self) -> Context | None:
        return self._parent

    @property
    def root(self) -> Context:
        *_, last = self._chain()
        return last

    @property
    def path(self) -> str:
        return "/".join(reversed([node._label for node in self._chain()]))

    @property
    def active(self) -> bool:
        return self._active

    def _chain(self) -> Iterator[Context]:
        node: Context | None = self
        while node is not None:
            yield node
            node = node._parent

    def is_ancestor_of(self, other: Context) -> bool:
        """Whether `other` is this context or a descendant of it."""
        return any(node is self for node in other._chain())

    @property
    def isolation(self) -> Context | None:
        """The scope a registration made here belongs to; `None` for global.

        The key every scoped registry (tools, prompt sections) files a
        registration under, so "who can see this" is one question with one
        answer. A plugin's activation scope is transparent and answers `None`.
        """
        return self._isolation

    def isolation_chain(self) -> list[Context | None]:
        """This context's isolation scopes, most specific first, ending in `None`.

        A scoped registry walks this to resolve a name: the innermost scope that
        registered one wins, and the global layer is consulted last.
        """
        chain: list[Context | None] = []
        for node in self._chain():
            key = node._isolation
            if key is not None and key not in chain:
                chain.append(key)
        chain.append(None)
        return chain

    def reaches(self, target: Context) -> bool:
        """Whether a registration made here applies to work happening in `target`.

        The one visibility rule, shared by event dispatch and by every scoped
        registry (tools, prompt sections): a global registration reaches
        everything, an agent-scoped one reaches that agent alone.
        """
        return self._isolation is None or self._isolation.is_ancestor_of(target)

    def _assert_active(self) -> None:
        if not self._active:
            raise InactiveScopeError(f"scope {self.path} is disposed")

    # ------------------------------------------------------------- services --

    def provide(self, key: str, service: object) -> Disposer:
        """Claim `ctx.<key>` for `service` in this context's provisioning realm.

        Returns a disposer registered as an effect of the calling scope, so the
        service unregisters when its plugin unloads (invariant I2).
        """
        self._assert_active()
        target = self._provide_to
        existing = target._services.get(key)
        if existing is not None:
            raise ServiceConflictError(
                f'service "{key}" is already provided in realm {target.path} '
                f"by {existing.owner.path}"
            )
        target._services[key] = _Provision(value=service, owner=self)
        self._runtime.dirty = True

        def unprovide() -> None:
            current = target._services.get(key)
            if current is not None and current.value is service:
                del target._services[key]
                self._runtime.dirty = True

        return self.add_disposer(unprovide, label=f"provide({key})")

    def _provision(self, key: str) -> Any:
        """Resolve `key` most-specific-first up the scope chain."""
        for node in self._chain():
            provision = node._services.get(key)
            if provision is not None:
                return provision.value
        return _MISSING

    def get(self, key: str, default: Any = None) -> Any:
        value = self._provision(key)
        return default if value is _MISSING else value

    def has(self, key: str) -> bool:
        return self._provision(key) is not _MISSING

    def __getattr__(self, key: str) -> Any:
        # Never intercept private/dunder lookups: doing so turns a missing
        # attribute during __init__ into unbounded recursion.
        if key.startswith("_"):
            raise AttributeError(key)
        value = self._provision(key)
        if value is _MISSING:
            raise ServiceNotFoundError(f'no service "{key}" is provided at or above {self.path}')
        return value

    # -------------------------------------------------------------- effects --

    def add_disposer(self, dispose: Disposer, *, label: str = "") -> Disposer:
        """Register an already-acquired teardown as an effect of this scope."""
        self._assert_active()
        effect = _Effect(dispose=dispose, label=label)
        self._effects.append(effect)

        def release() -> Any:
            if effect.done:
                return None
            effect.done = True
            with suppress(ValueError):  # already removed by dispose()
                self._effects.remove(effect)
            return effect.dispose()

        return release

    async def effect(
        self, enter: Callable[[], Disposer | Awaitable[Disposer]], *, label: str = ""
    ) -> Disposer:
        """Acquire an artifact and register its release as an effect.

        Every external artifact an agent takes — a child process, a worktree, a
        temp path, a lock — is acquired through here, so cleanup is structural
        rather than remembered (§4.9, invariant I2).
        """
        self._assert_active()
        dispose = await maybe_await(enter())
        if not callable(dispose):
            raise TypeError(f"effect {label or enter!r} did not return a disposer")
        return self.add_disposer(dispose, label=label)

    # --------------------------------------------------------------- scopes --

    def scope(self, label: str = "scope", *, module: str = "") -> Context:
        """Create an isolated child scope that owns its own registrations.

        Used for `agent.ctx`: a registration made on the child shadows the
        global one for that agent alone, and its listeners hear only that agent.
        """
        self._assert_active()
        # `Context.__init__` appends to `self._children`, and `dispose` already
        # cascades over those *before* its own effects — so an `add_disposer`
        # here would be a second copy of the same teardown, and one `dispose`
        # never releases: the effect stays on the parent with `done=False` for
        # the parent's whole life, pinning every child that ever finished.
        # Measured at 1.0 KB per settled child, unbounded, and since P6-27 the
        # parent it pins is a live agent rather than the registry.
        child = Context(self, label=label, module=module or self._module, isolated=True)
        return child

    def plugin(self, plugin: Any, config: Any = None) -> ForkScope:
        """Mount `plugin` as a child fork of this context.

        The fork's `apply` runs only once every key in its `inject` list
        resolves from this context — the load order is expressed through
        service requirements, never through file order.
        """
        self._assert_active()
        return ForkScope(self, normalize_plugin(plugin), config)

    def inject(
        self, keys: Sequence[str], fn: Callable[[Context], Any], *, label: str = "inject"
    ) -> Disposer:
        """Run `fn(scope)` once every key in `keys` is available.

        `fn` receives a fresh child scope; when any key disappears that scope is
        disposed, and it is re-created when the key returns.
        """
        _, release = self._register_dependent(keys, fn, label=label)
        return release

    def _register_dependent(
        self, keys: Sequence[str], activate: Callable[[Context], Any], *, label: str
    ) -> tuple[_Dependent, Disposer]:
        """The one registration path for plugins and injections alike."""
        self._assert_active()
        dependent = _Dependent(ctx=self, keys=tuple(keys), activate=activate, label=label)
        self._runtime.dependents.append(dependent)
        self._runtime.dirty = True
        return dependent, self.add_disposer(dependent.retire, label=label)

    async def reconcile(self) -> None:
        """Settle the plugin tree: activate what is ready, deactivate what is not.

        Runs to a fixpoint, because activating one plugin may provide the
        service another was waiting on.
        """
        runtime = self._runtime
        for _ in range(_MAX_RECONCILE_ROUNDS):
            if not runtime.dirty:
                return
            runtime.dirty = False
            for dependent in list(runtime.dependents):
                if dependent.disposed:
                    runtime.dependents.remove(dependent)
                    continue
                ready = dependent.ready()
                if ready and not dependent.active:
                    scope = Context._activation_scope(dependent)
                    dependent.scope, dependent.active = scope, True
                    runtime.dirty = True
                    # Bound around the activation, and released on the way out
                    # rather than left to the task ending: `reconcile` activates
                    # every ready dependent in one loop on one task, so a token
                    # left behind would make the next row's registrations land on
                    # the previous row's scope (P6-12).
                    with running(scope):
                        try:
                            await maybe_await(dependent.activate(scope))
                        except BaseException:
                            await maybe_await(dependent.deactivate())
                            raise
                elif not ready and dependent.active:
                    # The fork survives: re-providing the missing service
                    # reactivates it on a later reconcile.
                    await maybe_await(dependent.deactivate())
        raise RuntimeError(
            "ph.cordis: plugin activation did not settle; a provide/dispose cycle "
            f"is oscillating after {_MAX_RECONCILE_ROUNDS} rounds"
        )

    async def dispose(self) -> None:
        """Unwind this scope: children first, then own effects LIFO.

        The order is load-bearing for anything that registers an effect *about* a
        child — a subagent's tombstone, a supervisor's bookkeeping — because
        such an effect runs after that child's scope is already gone. See
        `ph_rlm.subagents._release`, which is exactly that shape.
        """
        if not self._active:
            return
        self._active = False
        for child in reversed(list(self._children)):
            await child.dispose()
        self._children.clear()
        while self._effects:
            effect = self._effects.pop()
            if effect.done:
                continue
            effect.done = True
            try:
                await maybe_await(effect.dispose())
            except Exception:
                log.exception("ph.cordis: effect %r failed to dispose", effect.label)
        self._services.clear()
        # Breaks the `self -> Running -> self` cycle the memo makes, in the
        # same breath as the parent/child one below: a context that has been
        # disposed and dropped by its parent must not need a gc pass.
        self._running_self = None
        parent = self._parent
        if parent is not None and self in parent._children:
            parent._children.remove(self)
        self._runtime.dirty = True

    # ------------------------------------------------------------- dispatch --

    def on(
        self, event: str, listener: Listener, *, prepend: bool = False, global_: bool = False
    ) -> Disposer:
        """Register a listener owned by this scope.

        `prepend` places the listener before existing ones — reserve it for a
        listener that must run before ordinary registrations. `global_` opts out
        of scope filtering.
        """
        self._assert_active()
        event_registry.require(event)
        event_registry.note_consumer(event, self._module)
        hooks = self._runtime.hooks.setdefault(event, [])
        hook = Hook(ctx=self, callback=listener, prepend=prepend, global_=global_)
        if prepend:
            hooks.insert(0, hook)
        else:
            hooks.append(hook)

        def off() -> None:
            with suppress(ValueError):
                hooks.remove(hook)

        return self.add_disposer(off, label=f"on({event})")

    def _hooks(self, event: str, *, scope: Context | None = None) -> list[Hook]:
        """The hook *records* one dispatch would reach, in registration order.

        Records rather than bare callables, because a `Hook` carries the scope
        that registered it and `_invoke` needs it: a listener runs as an effect
        of the row that wrote it (P6-25).

        It replaced a public `collect` that returned callables and had no callers
        left once the five dispatch loops moved here — and keeping that would
        have been worse than a dead method: hand-rolling `for cb in ctx.collect(e)`
        is a dispatch with no binding, which is exactly the defect this row
        closes, offered as the only public route.

        **Deliberately a list, not a generator.** `bail` and `serial` return out
        of the loop early, so a generator holding a binding across its `yield`
        would be closed by the collector on some other task and release the token
        in the wrong context.
        """
        hooks = self._runtime.hooks.get(event)
        if not hooks:
            return []
        target = scope if scope is not None else self
        return [hook for hook in hooks if hook.global_ or hook.ctx.reaches(target)]

    def emit(
        self, event: str, *args: Any, scope: Context | None = None, contained: bool = False
    ) -> None:
        """Dispatch synchronously, ignoring listener return values.

        A listener that returns a coroutine is scheduled and not awaited, which
        is cordis's behaviour. A listener that raises stops the dispatch unless
        `contained=True`, which logs the failure and continues — the mode a
        producer uses when the event records something that already happened
        and no listener may un-happen it.
        """
        event_registry.check(event, "emit")
        for hook in self._hooks(event, scope=scope):
            try:
                result = _invoke(hook, *args)
            except Exception:
                if not contained:
                    raise
                log.exception("ph.cordis: %s listener failed", event)
                continue
            if result is not None and inspect.isawaitable(result):
                # `None` first for `_invoke`'s reason: this is the second of the
                # two awaitability checks per listener, on the per-chunk path.
                # `_invoke` already wrapped it, so the task binds when the body
                # runs rather than inheriting whatever was current at `_spawn`.
                self._spawn(result, event)

    def bail(self, event: str, *args: Any, scope: Context | None = None) -> Any:
        """Dispatch synchronously until a listener returns a bail value."""
        event_registry.check(event, "bail")
        for hook in self._hooks(event, scope=scope):
            result = _invoke(hook, *args)
            if is_bailed(result):
                return result
        return None

    async def serial(self, event: str, *args: Any, scope: Context | None = None) -> Any:
        """Await listeners in registration order until one bails."""
        event_registry.check(event, "serial")
        for hook in self._hooks(event, scope=scope):
            result = await maybe_await(_invoke(hook, *args))
            if is_bailed(result):
                return result
        return None

    async def parallel(self, event: str, *args: Any, scope: Context | None = None) -> None:
        """Run every listener concurrently and await all of them.

        Every listener runs even if one fails; the failures are collected and
        raised together, which is `Promise.allSettled` + `AggregateError`.
        """
        event_registry.check(event, "parallel")
        hooks = self._hooks(event, scope=scope)
        if not hooks:
            return
        failures: list[Exception] = []

        async def run(hook: Hook) -> None:
            try:
                await maybe_await(_invoke(hook, *args))
            except Exception as error:
                failures.append(error)

        async with anyio.create_task_group() as group:
            for hook in hooks:
                group.start_soon(run, hook)
        if failures:
            raise ExceptionGroup(f'listeners failed for "{event}"', failures)

    async def waterfall(
        self, event: str, *args: Any, inner: Callable[..., Any], scope: Context | None = None
    ) -> Any:
        """Around-middleware: each listener wraps the rest of the chain.

        Listeners run outermost-first and receive `(*args, next)`. Calling
        `next()` delegates; returning without calling it vetoes the rest of the
        chain, `inner` included — that veto is how a policy plugin replaces
        built-in behaviour without the built-in knowing.

        `next(*replacement)` additionally hands the rest of the chain different
        arguments. Cordis expects a listener to mutate a shared payload instead,
        which is not available here: pH's payloads are frozen values, and a
        rewrite that has to be explicit is a rewrite a reader can see.
        """
        event_registry.check(event, "waterfall")
        hooks = self._hooks(event, scope=scope)
        state: list[Any] = list(args)
        index = 0

        async def next_(*replacement: Any) -> Any:
            nonlocal index
            if replacement:
                state[:] = replacement
            if index < len(hooks):
                hook = hooks[index]
                index += 1
                return await maybe_await(_invoke(hook, *state, next_))
            # `inner` is the *producer's* body rather than a listener — nothing
            # registered it, so it runs under whatever binding the caller of
            # `waterfall` already had. That is still true after P6-26, which
            # bound the bodies a *registry* owns: `tools/execute`'s inner is one
            # of them and binds itself from the inside, before calling on. The
            # other thirteen are a seam's own fallback, which is the row's code
            # and wants the row's binding — exactly what it inherits here.
            return await maybe_await(inner(*state))

        return await next_()

    def detach(self, coro: Any, *, label: str) -> None:
        """Run `coro` outside the caller's lifetime, tracked and drained.

        For work that must outlive the call that started it and must not be
        awaited by it: an async `emit` listener, a subagent running while its
        parent keeps working. `drain()` is what makes this honest — the task is
        in a pool the host awaits at shutdown, so "fire and forget" does not mean
        "lost on exit". A failure is logged at its own boundary, because there is
        no caller left to raise into.

        With no running loop the coroutine is closed rather than leaked, and the
        drop is logged: a sync host that cannot run it should not silently hold a
        never-awaited coroutine either.
        """
        try:
            task = asyncio.ensure_future(coro)
        except RuntimeError:  # pragma: no cover - no running loop
            coro.close()
            log.warning("ph.cordis: dropped detached %s (no event loop)", label)
            return
        self._runtime.background.add(task)

        def done(finished: asyncio.Future[Any]) -> None:
            self._runtime.background.discard(finished)
            if not finished.cancelled() and finished.exception() is not None:
                log.error("ph.cordis: detached %s failed", label, exc_info=finished.exception())

        task.add_done_callback(done)

    def _spawn(self, coro: Any, event: str) -> None:
        """Track a fire-and-forget coroutine returned by an `emit` listener."""
        self.detach(coro, label=f"listener for {event}")

    async def drain(self) -> None:
        """Await every detached coroutine: async `emit` listeners, and `detach()`."""
        while self._runtime.background:
            for task in list(self._runtime.background):
                # Already logged by the done callback; awaiting here is only
                # about knowing the task has settled.
                with suppress(Exception):
                    await task
                self._runtime.background.discard(task)
