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

    **The one place ownership is established for a dispatch.** A function rather
    than a line in each loop: there are five dispatch modes, and a sixth cannot omit
    what it never writes.

    Sync and async are one path. A listener returning an awaitable gets it back
    wrapped in `_as_owner`, so whoever awaits or spawns it receives a coroutine that
    binds itself; one returning a value has already run inside the binding. Callers
    need no idea which kind they have, which is what lets `emit` spawn and `serial`
    await the same result.

    The binding is spelled inline rather than through `running` because `emit` fires
    once per streamed chunk, where even a context manager's own frame is too much.
    """
    # Inline, and through the scope's memoized self-pair: building a `Running`
    # per call means a frozen dataclass setting its fields through
    # `object.__setattr__`, once per listener per chunk.
    owner = hook.ctx
    pair = owner._running_self
    if pair is None:
        pair = owner._running_self = Running(owner, owner)
    token = _ACTIVATING.set(pair)
    try:
        result = hook.callback(*args)
        # `None` first, and it is most of the win. A listener that returns nothing
        # is the overwhelming case, and `inspect.isawaitable` runs a second time in
        # `emit` on the value this just returned — so the awaitability question
        # costs more per listener than the binding it was added to serve.
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
    """Everything this deployment holds — the widest boundary, said out loud (P6-32).

    **"Widest" along the restriction axis only.** A reader given `DEPLOYMENT`
    resolves the *mount's* isolation chain, which no restriction narrows — but an
    **agent-scoped registration is not on it**: `view(DEPLOYMENT)` does not see a
    tool registered on one agent's scope. It is the view `scope=None` always read,
    named. "Is this available to *anyone*" is a per-layer question this sentinel does
    not answer.

    **The point is that it has a name.** `scope: Context | None = None` made two
    different things one spelling — "I did not state a boundary", an absence of
    information, and "I mean the deployment", a legitimate answer. With one spelling
    every reader has to default the ambiguous case, and the convenient default is
    the mount: the widest boundary there is.

    Not `None` meaning "nothing", which trades a silent-wide failure for a
    silent-narrow one. The half that does the work is removing the *default*, so a
    call site stating no boundary fails mypy rather than a runtime rule.
    """


DEPLOYMENT = Deployment()
"""The one instance. There is nothing to construct and nothing to configure."""

Boundary: TypeAlias = "Context | Deployment"
"""What a policy reader takes: a scope, or `DEPLOYMENT` for all of them.

Deliberately **not** `| None`. A reader whose parameter has no default cannot be
called without answering the question, and mypy asks at build time against every
call site rather than only the paths a test drives.

**A *registration* takes a `Context`, never this**, and so do the two
dispatch-time resolvers, `Context.owner_for` and `Context.layer_for`, which read
the running binding rather than a stated value: a scope is what a registration is
an *effect of* and what it is visible *to* (P6-12's two questions). `DEPLOYMENT`
is neither — there is nothing for it to unwind with.
"""


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
    Every registry invokes a row's body later — a tool, a command, a compaction
    note, a prompt section, a status field, a diagnostic, a telemetry sink, each of
    the five single-slot providers — and to enter the right binding then it must have
    kept both answers now. `Context.running_for` hands them exactly this object.

    The two come apart precisely when a **registry** invokes the body: a tool's
    `execute` belongs to the row that registered the tool (I2 — whose code it is)
    while it is visible to the *agent* it was invoked for (B7 — a different scope).

    Two fields rather than a second `ContextVar` because they are one fact — *who is
    running* — read two ways, and two variables can be set out of step. The only way
    to bind one is `running()`, which takes both.
    """

    owner: Context
    """Whose lifetime a registration made now joins — `Context.owner_for`."""
    layer: Context
    """Which scope it is visible to — `Context.layer_for`."""

    def add_disposer(self, dispose: Disposer, *, label: str = "") -> Disposer:
        """Release when **either** scope ends — the pair's lifetime, not a half's.

        Once a body a registry invokes registers as its row *for* an agent, the two are
        unrelated branches and either can end first. Picking one is wrong in a different
        direction each way: owning it by the row alone leaves whatever the layer keyed on
        a disposed scope, and owning it by the agent alone lets a registration outlive
        the row whose code made it, which is I2 verbatim. The registration is meaningful
        only while both are alive.

        **Here rather than in each registry**, because more than one registry keys on the
        layer (`ToolRuntime._layers`, `ph.seams.skills._restrictions`). On `Running` and
        not on `Context` because `Context.add_disposer(..., also=)` would ask one scope
        to know about a pair it does not hold.

        The once-guard is `Context.add_disposer`'s: the releaser flips `_Effect.done` and
        drops the effect before calling through, so `finish` re-enters exactly one level
        and stops. Whichever scope ends first removes the sibling effect from the other.
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

    **It takes the `Running` itself**, which is how every registry calls it: the pair
    exists so a registry stops keeping two fields it can pair up wrongly, and
    `running(a.owner, b.layer)` at nine call sites would be silently valid. `layer=`
    beside a pair *overrides* the visibility half — the one thing a caller
    legitimately knows better (`ph.seams.fs` binds the agent whose path is being
    resolved, not the scope the resolver registered on).

    **`None` binds nothing**, for the empty half of an at-most-one slot: five seams
    hold a provider that may not be registered.

    **`layer` defaults to `owner`**, which is the whole of the ordinary case — an
    `apply` and a listener each run as one scope, so `running(scope)` means what it
    always did. One call with a default rather than two functions, because the
    *shape of the call* then says whether the two questions have one answer, and the
    record it binds cannot be half-set.

    **Single-use**, like every context manager built at its `with`: the token lives
    on the instance, so re-entering one object would lose the outer token.

    A class rather than a `@contextmanager`, because the generator form costs a
    frame, a `StopIteration` and two `send`s at each of the twenty-one call sites —
    this is entered once per tool call, once per slash command, once per prompt row
    per `assemble`, and once per telemetry sink per record.
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

    An `async def` listener called by `emit` only *builds* a coroutine — the body
    runs later on a task, and a task copies the context at creation, which is after a
    binding around the *call* has been reset. Wrapping the awaitable moves the
    binding inside the body, where it is the coroutine's own first act rather than
    something its creator did before handing it over.
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

        Called as `self.ctx.owner_for(scope)` from a seam, where `self.ctx` is the seam's
        own context and is therefore the fallback rather than the answer. Three cases, in
        order:

        * **an explicit `scope=`** — the caller said so, and the only thing that
          overrides the rest. It means *"register on someone else's lifetime"*;
        * **the activation scope**, when a row's `apply` is what is running — the scope
          cordis disposes when the row unmounts, so the registration goes with it;
        * **this context**, outside any activation: for callers that are not rows at all
          (a test standing a service up by hand, a mode wiring one directly).

        **This does not take a `Boundary`** (P6-32): `None` here resolves through
        `_ACTIVATING` to whoever is running, which is the *narrowest* correct answer, not
        "give me everything".

        **A disposed activation scope declines, and says so.** Contextvars propagate into
        tasks spawned from `apply`, so a background task registering after its row
        unmounted would otherwise leak quietly onto the seam. That is the only branch
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

        Separate from `owner_for` because they are different questions that used to have
        the same answer. Visibility is `scope or this context`, and deliberately *not*
        the activating row: moving it would change what an agent can see (B7) rather than
        when a registration goes away.

        **It does not take a `Boundary` either** (P6-32), for `owner_for`'s reason: both
        are dispatch-time *resolvers* reading `_ACTIVATING`, where a `Boundary` parameter
        is a caller **stating** something. `None` means "ask the binding".

        **It follows the running binding, and `isolation` is why that is safe** (P6-26).
        A row's activation scope is not isolated, so it inherits the mount's and a root
        row's registration still lands on the global layer. An agent's scope *is* its own
        isolation, so a body running for an agent lands on that agent's layer — which is
        the containment P6-27 made structural, and the last way to escape it was a tool
        body inside a contained child installing globally visible tools.

        **The disposed branch fails open and deliberately does not warn.** Falling back
        to `self` means the global layer, so a body that outlived its scope registers
        something the whole deployment can see — the mirror of `owner_for`'s fail-open,
        in B7's direction. Silent because all five call sites ask `owner_for` about the
        same `scope` a line or two later, and that is where it is logged; two lines for
        one registration makes the honest one easier to miss.
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

        **Every site that needs both should ask once**, because `owner_for` logs when the
        activation scope it would return has already been disposed, and two calls for one
        registration make that warning audible twice.

        It matters most for a registry that will invoke the body *later*, which must keep
        both until then: two fields on its own record is two things to hold in step, and
        the failure mode is silent. One object cannot be half-updated.
        """
        return Running(self.owner_for(scope), self.layer_for(scope))

    @staticmethod
    def current_owner() -> Context | None:
        """Whose *lifetime* the running code belongs to, or `None` (P6-12).

        **The owner a seam registration defaults to.** Cordis builds the right scope in
        `_activation_scope` and hands it to `apply`, but a seam only ever saw `ctx`, so
        `ctx.commands.register(...)` gave the registry no way to know who was calling and
        every seam defaulted the owner to its *own* context — making a row's
        registration an effect of the **seam**, outliving the row that made it.

        A `ContextVar` rather than a parameter threaded through forty signatures, because
        the answer is a property of *who is running*, not of what they are asking for —
        and the seam methods are the API rows use, which cannot grow a mandatory
        argument.

        **It answers "whose code is running", across both ways cordis runs any** (P6-25):
        an `apply` binds its activation scope, and a dispatch binds the scope that
        registered the listener. So a registration belongs to the row that made it
        whether that row is being mounted, is handling another row's event, or is
        handling one fired long after every `apply` returned.

        Contextvars propagate into tasks, so a coroutine spawned from a bound callback
        carries the binding; `Context.owner_for` declines a *disposed* scope and warns.

        **What is bound and what still runs unbound is enumerated by introspection**,
        and the enumeration fails on anything unclassified — so that list is checked
        rather than believed, and this docstring does not keep a second copy of it.
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

        Records rather than bare callables, because a `Hook` carries the scope that
        registered it and `_invoke` needs it: a listener runs as an effect of the row
        that wrote it (P6-25). There is deliberately no public route that returns bare
        callables — hand-rolling `for cb in ctx.collect(e)` is a dispatch with no
        binding.

        **Deliberately a list, not a generator.** `bail` and `serial` return out of the
        loop early, so a generator holding a binding across its `yield` would be closed
        by the collector on some other task and release the token in the wrong context.
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
