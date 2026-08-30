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
from contextvars import ContextVar
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


_ACTIVATING: ContextVar[Context | None] = ContextVar("ph.cordis.activating", default=None)
"""The activation scope of the `apply` currently running. See `Context.current_scope`."""


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
        if activating.active:
            return activating
        log.warning(
            "ph.cordis: %s registered after its activation scope %s was disposed; "
            "the registration will outlive the row that made it (I2)",
            self.path,
            activating.path,
        )
        return self

    def layer_for(self, scope: Context | None = None) -> Context:
        """Which scope a registration is *visible to* — the other question (P6-12).

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
        """
        return scope if scope is not None else self

    @staticmethod
    def current_scope() -> Context | None:
        """The activation scope `apply` is running in, or `None` outside one (P6-12).

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

        **It answers "which `apply` is on the stack", not "whose callback is
        running", and the two come apart in one place.** The variable is live
        for the whole dynamic extent of the activation `await`, so a listener
        belonging to row B, dispatched from row A's `apply`, sees *A* — and a
        registration B makes there unwinds when A unmounts. No shipped row emits
        during `apply`, so this is latent rather than live, but it is one
        `ctx.emit` away. The mirror gap is that a registration made *after*
        `apply` returns — from a `profile/mounted` listener, a tool body, a turn
        hook — sees `None` and lands on the seam, which is why
        `register_when_composed` passes `scope=ctx` by hand.

        Closing both means keying on the owner of the *callback* rather than the
        activation: `Hook` already carries `ctx`, so dispatch could set this the
        same way. That is a change to how every listener is invoked and wants
        its own row (P6-25) rather than a line here.

        It also propagates into tasks spawned from `apply`, which is the third
        edge: a registration made from a background task after its scope was
        disposed would raise. `Context.owner_for` declines a disposed scope and
        warns instead.
        """
        return _ACTIVATING.get()

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
        child = Context(self, label=label, module=module or self._module, isolated=True)
        self.add_disposer(child.dispose, label=f"scope({label})")
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
                    # Set around the activation, and reset in `finally` rather
                    # than left to the task ending: `reconcile` activates every
                    # ready dependent in one loop on one task, so a token left
                    # behind would make the next row's registrations land on the
                    # previous row's scope (P6-12).
                    token = _ACTIVATING.set(scope)
                    try:
                        await maybe_await(dependent.activate(scope))
                    except BaseException:
                        await maybe_await(dependent.deactivate())
                        raise
                    finally:
                        _ACTIVATING.reset(token)
                elif not ready and dependent.active:
                    # The fork survives: re-providing the missing service
                    # reactivates it on a later reconcile.
                    await maybe_await(dependent.deactivate())
        raise RuntimeError(
            "ph.cordis: plugin activation did not settle; a provide/dispose cycle "
            f"is oscillating after {_MAX_RECONCILE_ROUNDS} rounds"
        )

    async def dispose(self) -> None:
        """Unwind this scope: children first, then own effects LIFO."""
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

    def collect(self, event: str, *, scope: Context | None = None) -> list[Listener]:
        """The listeners one dispatch would reach, in registration order."""
        hooks = self._runtime.hooks.get(event)
        if not hooks:
            return []
        target = scope if scope is not None else self
        return [hook.callback for hook in hooks if hook.global_ or hook.ctx.reaches(target)]

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
        for callback in self.collect(event, scope=scope):
            try:
                result = callback(*args)
            except Exception:
                if not contained:
                    raise
                log.exception("ph.cordis: %s listener failed", event)
                continue
            if inspect.isawaitable(result):
                self._spawn(result, event)

    def bail(self, event: str, *args: Any, scope: Context | None = None) -> Any:
        """Dispatch synchronously until a listener returns a bail value."""
        event_registry.check(event, "bail")
        for callback in self.collect(event, scope=scope):
            result = callback(*args)
            if is_bailed(result):
                return result
        return None

    async def serial(self, event: str, *args: Any, scope: Context | None = None) -> Any:
        """Await listeners in registration order until one bails."""
        event_registry.check(event, "serial")
        for callback in self.collect(event, scope=scope):
            result = await maybe_await(callback(*args))
            if is_bailed(result):
                return result
        return None

    async def parallel(self, event: str, *args: Any, scope: Context | None = None) -> None:
        """Run every listener concurrently and await all of them.

        Every listener runs even if one fails; the failures are collected and
        raised together, which is `Promise.allSettled` + `AggregateError`.
        """
        event_registry.check(event, "parallel")
        callbacks = self.collect(event, scope=scope)
        if not callbacks:
            return
        failures: list[Exception] = []

        async def run(callback: Listener) -> None:
            try:
                await maybe_await(callback(*args))
            except Exception as error:
                failures.append(error)

        async with anyio.create_task_group() as group:
            for callback in callbacks:
                group.start_soon(run, callback)
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
        callbacks = self.collect(event, scope=scope)
        state: list[Any] = list(args)
        index = 0

        async def next_(*replacement: Any) -> Any:
            nonlocal index
            if replacement:
                state[:] = replacement
            if index < len(callbacks):
                callback = callbacks[index]
                index += 1
                return await maybe_await(callback(*state, next_))
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
