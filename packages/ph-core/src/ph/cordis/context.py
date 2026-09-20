"""Context, services, effects, scopes and the four dispatch modes.

The Python subset of Cordis that pH needs (D1). A `Context` is three things at
once, exactly as in dsh:

* a **repository of services** — a plugin claims `ctx.<key>` with `provide()`
  and every other plugin finds it by key rather than by import;
* an **event bus** — `emit` / `parallel` / `serial` / `waterfall`,
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
from collections.abc import Awaitable, Callable, Iterator, MutableMapping, Sequence
from contextlib import suppress
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias, cast, overload
from weakref import ref

import anyio

from .errors import InactiveScopeError, ServiceConflictError, ServiceNotFoundError
from .events import events as event_registry
from .key import ServiceKey, service_name, service_names
from .plugin import PluginSpec, normalize_plugin

__all__ = [
    "Context",
    "Disposer",
    "ForkScope",
    "Hook",
    "Listener",
    "Next",
    "is_bailed",
    "maybe_await",
    "settled",
    "settled_or_none",
]

log = logging.getLogger("ph.cordis")

Disposer: TypeAlias = Callable[[], object]
"""A teardown callable. It may return an awaitable; `dispose()` awaits it.

`object` rather than `Any` because nothing here inspects what a disposer hands
back — `dispose()` awaits it and drops it — and `Any` invited a caller to.
Narrower would be wrong: a disposer is often an existing call that happens to
return something, and `None | Awaitable[None]` would make every one of those
spell a discard.
"""

Listener: TypeAlias = Callable[..., object]
"""One registered listener. The arguments stay open — fourteen chains across the
tree take different ones — but the return is `object` for `Disposer`'s reason:
`_invoke` hands it to `is_bailed` and `maybe_await`, both of which take `object`,
and every chain's answer is made good by `settled`."""

type MaybeAwaitable[T] = T | Awaitable[T]
"""A `T`, or a `T` that has to be awaited first — what `maybe_await` takes.

The seam between a row's ordinary code and this runtime's. A body, a listener or
a prompt section is written by whoever mounted the row; requiring `async def` of
all of them would make every synchronous one spell a coroutine it has no use for,
so the runtime accepts either and `maybe_await` settles it. `Disposer` above is
the deliberate exception — it is return-agnostic rather than maybe-awaited, and
`object` already subsumes an awaitable.

Use it where the mixture is real. It is an *inference* trap in a parameter that
has to solve `T` from a body's declared return, which is why the tool pipeline
overloads on the body kind instead of spelling this union.

Named because every declaration that takes a body spelled it by hand, and read
differently at each: `T | Awaitable[T]`, `Awaitable[T] | T`, and
`str | Awaitable[str | None] | None` are one type wearing three faces. The last
is what `RUF036` leaves you with once `T` is itself optional, which is the point
at which the hand-spelling stops being readable at all."""

type Next[T] = Callable[..., Awaitable[T]]
"""The rest of a `waterfall` chain, as the listener wrapping it sees.

`T` is the chain's own type, which `waterfall` reads off the producer's `inner`
— so a listener on `agent/pre-step` takes a `Next[PreStepDecision]` and returns
one. Named because twenty listeners across twelve files spell this shape, and
ten of them imported `Callable` and `Awaitable` for nothing else.

It states a convention, not a check: `on` takes a `Listener`, so a listener's own
annotation is what it claims rather than what anything verifies."""

_MAX_RECONCILE_ROUNDS = 64
_MISSING: Final = object()


async def maybe_await[T](value: MaybeAwaitable[T]) -> T:
    """Await `value` when it is awaitable, otherwise return it unchanged.

    Generic for the reason `waterfall` is: the caller knows what it handed in,
    and an `Any` here erased it again at every call site — `effect` got back an
    untyped `dispose`, `code_mode` an untyped namespace. A caller whose own
    argument is `Any` still gets `Any`, which is the honest answer: the erasure
    is that caller's to fix, not this one's.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def settled[T](
    event: str, value: object, kind: type[T], *, refusal: type[Exception] = TypeError
) -> T:
    """What the chain returned, checked — `waterfall`'s last step is a `cast`.

    A listener arrives through an entry point and `on` takes a `Listener`, so no
    checker between the row and here sees what one returns. `waterfall` states
    the chain's type; this is where that claim is made good, once per producer.

    Here rather than hand-written at each: of nine producers, three raised a
    `TypeError`, three quietly substituted a default, two coerced with `str()`
    and one never looked — so a single broken row ended a turn in one chain and
    went unnoticed in another. The message names the event because that is what
    a person goes and looks at.

    **A seam that owns a refusal vocabulary names it.** `refusal=` exists for
    `ctx.fs`, whose `FsDenied` is a `HarnessError` on purpose: a veto has to
    reach a consumer as a *denial*, because a Code Mode program can `except` a
    `failed` and route around it. A wrong answer from a policy listener is still
    a refusal, so it must not arrive wearing the other kind.

    Two limits, and they are separate. Mechanically this needs a type
    `isinstance` can test, so the union chains — `tools/pre-execute`,
    `tools/post-execute` — narrow by arm instead. Separately, `approval/request`
    keeps its own lookup *and denies rather than raising*, which is a policy
    choice: on that path refusing is the safe answer and an exception is not.
    """
    if not isinstance(value, kind):
        raise refusal(f"{event} must resolve to a {kind.__name__}, not {value!r}")
    return value


def settled_or_none[T](
    event: str, value: object, kind: type[T], *, refusal: type[Exception] = TypeError
) -> T | None:
    """`settled` for a chain whose `None` is an answer rather than an absence."""
    return None if value is None else settled(event, value, kind, refusal=refusal)


def is_bailed(value: object) -> bool:
    """Whether a listener's return value stops a `serial` dispatch.

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


def _invoke(hook: Hook, *args: object) -> object:
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


@dataclass(slots=True, weakref_slot=True)
class _Effect:
    """One registered teardown, and two facts about it that are not the same.

    `done` means **claimed** — somebody has taken responsibility for running
    this, so nobody else may. `ran` means **completed** — nothing is owed. One
    flag served both until the unwind gained a deadline, and then the difference
    became the whole story: at the moment the budget expires the loop marks each
    remaining effect claimed and never runs it, so a single flag reported a tree
    that had torn down nothing as a tree that owed nothing.
    """

    dispose: Disposer
    label: str
    done: bool = False
    ran: bool = False


GRACE_SECONDS = 10.0
"""How long one scope tree's unwind may take before it stops waiting on itself.

**Here rather than in `ph.resources`, because this is what now bounds it.** The
number was `resources.GRACE_SECONDS`, where the process-level shutdown path
wrapped `ctx.dispose()` in a shielded `move_on_after` — a caller hand-rolling a
guarantee the primitive should own, while every other caller had none. `dispose`
applies it itself now, and `ph.resources` re-exports the name so the shutdown
path reads unchanged.

Ten seconds because it is a budget for *finishing*, not a timeout for one
operation: a worktree to unlink, a child to reap, a proxy to close. The
`server.py` shutdown says the other half — a root that will not unwind must not
become a process that will not exit — which is why this is a deadline and not a
bare shield.
"""


def releasing() -> anyio.CancelScope:
    """A cleanup scope that survives the cancel that sent it, but not for ever.

    The `finally` that hands a kernel resource back — an overlay unmount, a
    worktree deregistration, a child's disposer — is reachable under raw
    cancellation, and unshielded its `await` is simply skipped: a live FUSE mount
    nobody unmounted, a registration that makes `git branch -D` refuse the branch
    an export just produced, a subagent still spending tokens for a turn that was
    abandoned.

    **Shielded with a deadline, not shielded.** An unmount that hangs because the
    backend is wedged is the other way to lose a process, and a shield with no
    bound makes it unkillable — which is the failure `dispose` already argues
    against. `GRACE_SECONDS` is what the rest of the tree unwinds in, so a
    cleanup does not get to be slower than the runtime that owns it; if it
    expires the cleanup is abandoned, which is where this started, the difference
    being that it has been tried.

    Here rather than in the seam that first needed it, because the pattern was
    already written out inline at six sites across the tree and the seventh
    author would have written a seventh. It does **not** consult
    `_Runtime.unwind_deadline`: a cleanup reached during an unwind takes a fresh
    budget on top of the tree's, which is the "ten roots, ten budgets" shape
    `unwind_by` exists to prevent. Folding the two is the right next change and
    needs its own test; the sites using this today are cleanups that run inside
    an *operator's* call rather than inside a dispose.
    """
    return anyio.CancelScope(deadline=anyio.current_time() + GRACE_SECONDS, shield=True)


DRAIN_SECONDS = GRACE_SECONDS / 2
"""How long `drain` waits on detached work before the unwind behind it starts.

**Derived from `GRACE_SECONDS` rather than declared beside it**, because the two
are spent back to back by a host tearing a mount down and the relation is the
whole point: a listener still running here has already been told its scope is
going, while the effects behind it are worktrees, leases and child processes
that outlive the process if nobody hands them back. When both cannot be
afforded, the unwind is the one that must run. Written as a literal, that
sentence would stop being true the first time somebody raised the grace period.
"""

_UNWINDING: ContextVar[bool] = ContextVar("ph.cordis.unwinding", default=False)
"""Whether the caller is already inside a shielded unwind.

**Per task rather than per tree, and the distinction is the whole of it.** This
was a field on `_Runtime`, which every scope in one tree shares, and it answered
the nested case correctly: `dispose` recurses into children, and a second shield
per layer costs 232% on a 341-scope tree for no guarantee the outer one is not
already giving. What it could not tell apart was a *sibling* — two scopes of one
tree disposed concurrently on two tasks — which found the flag set by the other
and ran its whole unwind bare, so the first one's cancellation stranded the
second one's effects. That is the shape a daemon reaches whenever a passivation
overlaps a shutdown.

A `ContextVar` answers the question that was being asked all along: is there a
shield above *me*. A nested `dispose` inherits the value on the same task; a
sibling on another task does not, and raises its own.
"""

ABANDONED_LEDGER = 64
"""How many cut-short unwinds one tree remembers.

Bounded because the ledger holds the abandoned disposers alive — deliberately,
since they are still callable and each stands for a resource nobody has freed —
and an unbounded list of them in a process that keeps abandoning would be a leak
about a leak. 64 is far past the point where a reader has stopped counting and
started asking what is wrong with the deployment; the count of what fell off is
kept, so the report never quietly under-states.
"""


@dataclass(frozen=True, slots=True)
class Abandoned:
    """One scope whose unwind was cut short, and the teardown it left holding.

    **`outstanding` is derived, not recorded.** The entry watches the live
    effects rather than snapshotting their names, so one released afterwards
    through the closure `add_disposer` handed out drops off the report by
    itself. A name-snapshot would keep accusing a deployment of a lease somebody
    had already returned — and `dispose()` cannot be that somebody, since its
    early return makes a second call a no-op, so those closures are the only way
    back and the ledger has to notice them being used.

    **Weakly, because the alternative is a leak about a leak.** A strong
    reference pins far more than the disposer: `provide`'s `unprovide` captures
    the `Context` and the service, and `on`'s `off` captures a `Hook` holding the
    same context — so any scope that ever called either kept its whole subtree
    and every service in it alive for the life of the tree, none of which
    `outstanding` ever mentioned. Real disposers here are bound methods on live
    objects (an httpx client and its pool, a subprocess handle), which CPython
    reclaims on the last reference and cannot while a diagnostic holds one.

    The label is cached beside the reference because it outlives it, and the two
    cases are different answers rather than one: a live reference not yet done is
    **outstanding** — somebody still holds the closure and can release it — while
    a dead one is `unreclaimable`, since nothing can ever call it again.
    """

    path: str
    watched: tuple[tuple[str, ref[_Effect]], ...]
    """Each undisposed effect's label and a weak reference to it. Read through
    `outstanding` and `unreclaimable`; never dereferenced elsewhere."""

    @property
    def outstanding(self) -> tuple[str, ...]:
        """Effects nobody has claimed, that a holder could still run.

        Unclaimed *and* still referenced. `release` refuses an effect the unwind
        already claimed — that guard is what stops a disposer running twice — so
        a claimed one is not something anybody can act on, however incomplete.
        """
        return tuple(
            label for label, watch in self.watched if (one := watch()) is not None and not one.done
        )

    @property
    def unreclaimable(self) -> tuple[str, ...]:
        """Effects nothing will ever run, for either of the two reasons.

        The last holder is gone, so no closure survives to call — or the unwind
        claimed it and never finished it, which `release` then refuses on the
        `done` guard. The second is the case a deadline creates: the disposer cut
        off mid-unmount is the one most likely to have left something half-done,
        and it is also the one nobody can retry.

        Kept apart from `outstanding` because they ask different things of a
        reader. One names work somebody can still do; this names a resource that
        is stranded until the process ends, and telling an operator to go and act
        on it would be telling them to do something impossible.
        """
        return tuple(
            label
            for label, watch in self.watched
            if (one := watch()) is None or (one.done and not one.ran)
        )


ForkState: TypeAlias = Literal["unmounted", "failed", "active", "activating", "unwound", "waiting"]
"""Where a mounted plugin stands. See `ForkScope.state`, which is the only producer."""


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
    module: str = ""
    """The module whose code runs in the activation scope, when one is known.

    A plugin's, stamped by `ForkScope` from its `apply`. It reaches
    `Context._module`, which is what `ctx.on` reports to `note_consumer` — so
    this field is the whole reason `phern events` can name who listens to an event
    rather than printing an empty column. Blank for an `inject` callback, which
    is code the calling row already owns and which therefore inherits it."""
    active: bool = False
    ever_active: bool = False
    """Whether this has ever activated — the bit that tells `waiting on fs` for a
    fiber that never came up from the same words for one that came up and was
    unwound when `fs` went away. Only the second is a provider swap, which is what
    a live reader of the topology is asking about."""
    scope: Context | None = None
    disposed: bool = False
    failure: str | None = None
    """Why this dependent's activation raised, or `None`. Set once; never retried.

    **What stops the next `reconcile` running a failed `apply` again.**
    `deactivate()` marks the tree dirty, so a dependent left `ready()` after its
    `apply` raised was re-activated on the very next round — the same `apply`,
    raising the same way, ahead of every row still waiting its turn in the list.
    A profile with one broken row therefore mounted nothing behind it and re-ran
    whatever that row had already done, once per `reconcile`, for the life of the
    process.

    A reason rather than a flag, for `Abandoned`'s reason: the state is reported
    to a person — `phern doctor` prints it beside `waiting on` — and "failed" with
    no sentence is a report that sends the reader to the logs. `None` rather than
    an empty string so the test below is a comparison and not a truthiness that
    happens to hold because the message is never blank.
    """

    def ready(self) -> bool:
        return self.ctx.active and self.failure is None and not self.missing()

    def missing(self) -> list[str]:
        """The inject keys not yet provided at this dependent's own scope.

        One predicate for `ready`, for the loader's refusal, and for `phern doctor`'s
        `waiting on …` — asked against `self.ctx`, which for a private copy in a
        realm is the realm. Two callers had spelled it against two different
        scopes."""
        return [key for key in self.keys if not self.ctx.has(key)]

    def deactivate(self) -> MaybeAwaitable[None]:
        """Drop the activation scope, unwinding everything it registered."""
        scope, self.scope, self.active = self.scope, None, False
        self.ctx._runtime.dirty = True
        return scope.dispose() if scope is not None else None

    def retire(self) -> MaybeAwaitable[None]:
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
    unwind_deadline: float | None = None
    """The instant this tree's unwind must finish by, or `None` when none is set.

    **One budget per tree, not one per scope.** `dispose` recurses into children,
    so a deadline computed fresh at each level would multiply down the depth of
    the tree — `GRACE_SECONDS` at every layer, which is exactly the failure
    `daemon/server.py` names when it insists on one shared number.

    Read by `drain` and by `dispose`, which spend it in that order. Set by
    whoever starts the unwind, or *before* it by `Context.unwind_by` —
    which is how a caller holding several trees spends one budget across all of
    them. Cleared when that unwind ends.
    """
    abandoned: list[Abandoned] = field(default_factory=list)
    """Unwinds this tree could not finish, newest last, capped at
    `ABANDONED_LEDGER`. On the runtime rather than on the scope because the scope
    that could not finish is, by the end of `_leave_tree`, unreachable from
    anything — which is precisely why this was invisible."""
    abandoned_dropped: int = 0
    """How many entries the cap discarded, so a report can say so rather than
    imply the oldest abandonment was the first."""


class ForkScope:
    """A mounted plugin: its config, and the scope its activation owns.

    A fork survives deactivation. When an injected service disappears the
    activation scope is disposed — unwinding everything the plugin registered —
    but the fork stays mounted, so re-providing the service re-activates it.
    """

    __slots__ = ("_config", "_dependent", "_parent", "_spec", "_unmount")

    def __init__(self, parent: Context, spec: PluginSpec, config: object) -> None:
        self._parent = parent
        self._spec = spec
        self._config = config
        self._dependent, self._unmount = parent._register_dependent(
            spec.inject,
            self._apply,
            label=f"plugin({spec.name})",
            # Where the plugin's code lives, which nothing else in the mount path
            # knows: the loader has a name and an entry point, and the *module* is
            # the thing `ctx.on` reports as a consumer. Without it every scope
            # inherited the root's empty string and `note_consumer` — whose guard
            # is `if module` — recorded nothing, for every event, in every
            # profile.
            module=getattr(spec.apply, "__module__", ""),
        )

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def active(self) -> bool:
        return self._dependent.active

    @property
    def ever_active(self) -> bool:
        """Whether it has activated at least once; with `active` false, it was unwound."""
        return self._dependent.ever_active

    @property
    def failure(self) -> str | None:
        """Why this plugin's `apply` raised, or `None`. See `state`."""
        return self._dependent.failure

    @property
    def state(self) -> ForkState:
        """Where this fork stands, as one of six names.

        **The set is closed, so it is a `Literal` and not a ladder each reader
        rebuilds.** `Mount.topology` publishes this through the `topology` seam,
        so it is a surface rather than a log line, and the printer that used to
        derive it from five booleans owned the *ordering* as well — that `failed`
        has to be answered before `waiting on`, because a row that raised may
        also be missing a key and the reason it raised is the more useful
        sentence. A second consumer deriving that for itself is a second chance
        to get it wrong.

        `failed` is the one state that does not resolve itself. A waiting row
        comes up when its key arrives; a failed one stays down however the tree
        settles, because `ready()` refuses to hand the same `apply` a second
        chance. `activating` means every key is met and `reconcile` has not
        reached it yet, which only a reader inside the fixpoint ever sees.
        """
        dependent = self._dependent
        if dependent.disposed:
            return "unmounted"
        if dependent.failure is not None:
            return "failed"
        if dependent.active:
            return "active"
        if not dependent.missing():
            return "activating"
        return "unwound" if dependent.ever_active else "waiting"

    @property
    def unmounted(self) -> bool:
        """Whether `dispose()` has run: retired from the tree, never to reactivate."""
        return self._dependent.disposed

    @property
    def injects(self) -> tuple[str, ...]:
        """The service keys this plugin waits on — what `Mount.topology` names."""
        return self._spec.inject

    @property
    def waiting_on(self) -> list[str]:
        """The inject keys still unmet, at the scope that would have to provide them."""
        return self._dependent.missing()

    @property
    def ctx(self) -> Context | None:
        """The activation scope while the plugin is active, else `None`."""
        return self._dependent.scope

    @property
    def config(self) -> object:
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


def drop_dead_chains(cache: MutableMapping[tuple[Context | None, ...], Any]) -> None:
    """Delete every memo entry whose innermost scope has been disposed (I2).

    A chain-keyed cache holds its keys **strongly**, so an entry outlives the
    scope it describes: an agent is disposed, its `Context` is retained by the
    key tuple, and nothing drops it until the registry's next invalidation —
    which for a deployment that has stopped registering is never. Measured at
    ~2 KiB per agent and unbounded: 500 agents through a daemon leave 500 entries
    in each of the two caches.

    **Only `chain[0]` is asked**, not every key: the innermost scope is the one
    that dies, and `dispose` unwinds children inside the parent's own dispose, so
    an ancestor cannot be dead while its descendant is live except for the
    moment mid-unwind — and that entry is caught on the next miss. Scanning the
    whole chain cost 4x more to answer the same question.

    **Swept rather than unwound, and the alternative is real.** `Context` has a
    `__weakref__` slot, so an eviction disposer per scope with a `WeakSet` of
    hooked scopes would retain nothing and need no re-registration — it is not
    ruled out by the mechanics. The sweep is chosen because it is smaller: no
    per-registry hook table, no disposer per scope, and `SkillService._changed`
    already drops dead scopes out of `_restrictions` the same way. A cache is a
    thing legitimately allowed to forget.

    Called on a **miss**, which keeps it O(live scopes): a miss is the only moment
    the table grows, and sweeping then keeps the count proportional to what is
    alive rather than to what has ever run.

    Here, beside `chain_label` and `isolation_chain`, because the key's shape is
    the only thing the tool-view and skill-reach caches share.
    """
    dead = [chain for chain in cache if chain and chain[0] is not None and not chain[0].active]
    for chain in dead:
        del cache[chain]


def chain_label(chain: Sequence[Context | None]) -> str:
    """How a report names a cache entry keyed by an isolation chain.

    `chain[0]` rather than a scan for the first non-`None`, and the difference is
    a property rather than a shortcut: `isolation_chain` appends only non-`None`
    keys and then exactly one trailing `None`, so the innermost key is the first
    and `None` can only be last. A scan would read as though an interior `None`
    were possible, which is the shape neither caller could handle anyway.

    Here rather than in either caller because both chain-keyed caches — the tool
    view and the skill reach — print this in the same `phern doctor` section, and
    two spellings of one ordering invariant is how the two rows come to disagree
    about which scope an entry belongs to.
    """
    return chain[0].path if chain and chain[0] is not None else "the deployment"


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

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _ACTIVATING.reset(self._token)
            self._token = None


async def _as_owner(scope: Context, awaitable: Awaitable[object]) -> object:
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
        return cls(
            owner,
            label=dependent.label,
            provide_to=owner._provide_to,
            module=dependent.module or owner._module,
        )

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
    def children(self) -> tuple[Context, ...]:
        """The scopes this one will unwind, in creation order.

        A copy, not the list: `dispose` iterates `_children` and clears it, so a
        caller holding the live list would be mutating the unwind underneath it.
        The read-only counterpart of `parent`, and the other half of what a walk
        of the tree needs — `scope_invariant` is the caller, and it exists to
        find a scope this list should no longer contain.
        """
        return tuple(self._children)

    @property
    def root(self) -> Context:
        *_, last = self._chain()
        return last

    @property
    def abandoned(self) -> tuple[Abandoned, ...]:
        """Unwinds in this tree that were cut short, oldest first.

        A tree-wide fact read from any scope, because the record is on the shared
        runtime — the scope it describes is gone by the time the entry exists.
        Empty is the ordinary answer and the one a healthy process keeps.

        A copy, for `children`'s reason: the list is appended to from
        `_leave_tree`, which runs in a `finally` during cancellation, and a
        caller iterating the live list would be walking it as it grew.
        """
        return tuple(self._runtime.abandoned)

    @property
    def abandoned_dropped(self) -> int:
        """How many ledger entries the cap discarded. See `ABANDONED_LEDGER`."""
        return self._runtime.abandoned_dropped

    def descendants(self) -> Iterator[Context]:
        """Every scope beneath this one, itself first, with a cycle guard.

        The one tree walk, for `Mount.topology` and `scope_invariant` alike. The
        guard matters for the second caller: a tampered tree is exactly what a
        health check is asked to report on, and a walk that hung there would
        take `phern doctor` down with it.
        """
        seen: set[int] = set()
        pending = [self]
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            pending.extend(node.children)

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

    @overload
    def provide[T](self, key: ServiceKey[T], service: T) -> Disposer: ...
    @overload
    def provide(self, key: str, service: object) -> Disposer: ...
    def provide(self, key: str | ServiceKey[Any], service: object) -> Disposer:
        """Claim `ctx.<key>` for `service` in this context's provisioning realm.

        Returns a disposer registered as an effect of the calling scope, so the
        service unregisters when its plugin unloads (invariant I2).

        **Overloaded on the key, and the typed arm is where a seam's contract is
        checked.** `provide(LLM, adapter)` refuses anything that is not an
        `LlmRuntime` at type-check time; the `str` arm takes an `object`, as it
        always did, for the caller that has only a name.
        """
        self._assert_active()
        name = service_name(key)
        target = self._provide_to
        existing = target._services.get(name)
        if existing is not None:
            raise ServiceConflictError(
                f'service "{name}" is already provided in realm {target.path} '
                f"by {existing.owner.path}"
            )
        target._services[name] = _Provision(value=service, owner=self)
        self._runtime.dirty = True

        def unprovide() -> None:
            current = target._services.get(name)
            if current is not None and current.value is service:
                del target._services[name]
                self._runtime.dirty = True

        return self.add_disposer(unprovide, label=f"provide({name})")

    def _provision(self, key: str) -> object:
        """Resolve `key` most-specific-first up the scope chain."""
        for node in self._chain():
            provision = node._services.get(key)
            if provision is not None:
                return provision.value
        return _MISSING

    @overload
    def get[T](self, key: ServiceKey[T], default: T | None = None) -> T | None: ...
    @overload
    def get(self, key: str, default: object = None) -> Any: ...  # noqa: ANN401
    def get(self, key: str | ServiceKey[Any], default: object = None) -> Any:
        """The service under `key`, or `default` — the optional read.

        Typed through the key: `ctx.get(ATTACHMENTS)` is an `AttachmentStore |
        None`, so the `if store is None` a caller writes next is checked rather
        than habitual.
        """
        value = self._provision(service_name(key))
        return default if value is _MISSING else value

    def has(self, key: str | ServiceKey[Any]) -> bool:
        return self._provision(service_name(key)) is not _MISSING

    @overload
    def require[T](self, key: ServiceKey[T]) -> T: ...
    @overload
    def require(self, key: str) -> Any: ...  # noqa: ANN401
    def require(self, key: str | ServiceKey[Any]) -> Any:
        """The service under `key`, or `ServiceNotFoundError` — the required read.

        The typed spelling of `ctx.llm`: same lookup, same refusal, and the
        result is an `LlmRuntime` rather than `Any`. Named for what it does on
        absence — the seams' own `require()` methods set the precedent — so a
        reader can tell a call that may get `None` (`get`) from one that cannot.
        """
        name = service_name(key)
        value = self._provision(name)
        if value is _MISSING:
            raise ServiceNotFoundError(f'no service "{name}" is provided at or above {self.path}')
        return value

    # -------------------------------------------------------------- effects --

    def add_disposer(self, dispose: Disposer, *, label: str = "") -> Disposer:
        """Register an already-acquired teardown as an effect of this scope."""
        self._assert_active()
        effect = _Effect(dispose=dispose, label=label)
        self._effects.append(effect)

        def release() -> object:
            if effect.done:
                return None
            effect.done = effect.ran = True
            with suppress(ValueError):  # already removed by dispose()
                self._effects.remove(effect)
            return effect.dispose()

        return release

    async def effect(
        self, enter: Callable[[], MaybeAwaitable[Disposer]], *, label: str = ""
    ) -> Disposer:
        """Acquire an artifact and register its release as an effect.

        Every external artifact an agent takes — a child process, a worktree, a
        temp path, a lock — is acquired through here, so cleanup is structural
        rather than remembered (§4.9, invariant I2).

        **An artifact acquired into a scope that died mid-acquire is released
        here rather than leaked.** `enter()` is awaited, and a `dispose()` on
        another task can land inside that await, so the registration below would
        refuse a disposer for a thing that already exists — the one window where
        I2's "everything unwinds" turned on timing. Releasing it at once is the
        only answer left: this scope's effect list has already drained, so an
        entry added to it now is an entry nothing will ever run.

        **Shielded**, for `dispose`'s own reason one layer down. The cancellation
        that disposed the scope is usually still pending, so an unshielded
        release would strand precisely the artifact this branch exists to rescue.

        A release that itself raises is logged and the refusal still raised, as
        in `_unwind`: one situation answers with one exception, and the caller
        can act on neither the disposer's failure nor this one.

        The one property this cannot keep is LIFO order — the effects registered
        before it have already run — so an artifact whose release depends on one
        of them is released after that one is gone. Nothing better is available
        once the scope is down, and it is still strictly ahead of the leak.
        """
        self._assert_active()
        dispose = await maybe_await(enter())
        if not callable(dispose):
            raise TypeError(f"effect {label or enter!r} did not return a disposer")
        if not self._active:
            with anyio.CancelScope(shield=True):
                try:
                    await maybe_await(dispose())
                except Exception:
                    log.exception(
                        "ph.cordis: effect %r could not be released after its scope went away",
                        label or enter,
                    )
            raise InactiveScopeError(
                f"scope {self.path} was disposed while {label or enter!r} was being "
                "acquired; the artifact was released rather than registered"
            )
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

    def plugin(self, plugin: object, config: object = None) -> ForkScope:
        """Mount `plugin` as a child fork of this context.

        The fork's `apply` runs only once every key in its `inject` list
        resolves from this context — the load order is expressed through
        service requirements, never through file order.
        """
        self._assert_active()
        return ForkScope(self, normalize_plugin(plugin), config)

    def inject(
        self,
        keys: Sequence[str | ServiceKey[Any]],
        fn: Callable[[Context], Any],
        *,
        label: str = "inject",
    ) -> Disposer:
        """Run `fn(scope)` once every key in `keys` is available.

        `fn` receives a fresh child scope; when any key disappears that scope is
        disposed, and it is re-created when the key returns.
        """
        _, release = self._register_dependent(keys, fn, label=label)
        return release

    def _register_dependent(
        self,
        keys: Sequence[str | ServiceKey[Any]],
        activate: Callable[[Context], Any],
        *,
        label: str,
        module: str = "",
    ) -> tuple[_Dependent, Disposer]:
        """The one registration path for plugins and injections alike."""
        self._assert_active()
        dependent = _Dependent(
            ctx=self,
            keys=service_names(keys),
            activate=activate,
            label=label,
            module=module,
        )
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
                    dependent.ever_active = True
                    runtime.dirty = True
                    # Bound around the activation, and released on the way out
                    # rather than left to the task ending: `reconcile` activates
                    # every ready dependent in one loop on one task, so a token
                    # left behind would make the next row's registrations land on
                    # the previous row's scope (P6-12).
                    with running(scope):
                        try:
                            await maybe_await(dependent.activate(scope))
                        except BaseException as error:
                            # Recorded *before* the unwind, so a `deactivate`
                            # that fails in turn cannot lose the reason the
                            # activation failed — and so `ready()` is already
                            # false by the time `deactivate` marks the tree
                            # dirty, which is what keeps this from being retried.
                            dependent.failure = f"{type(error).__name__}: {error}"
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

    async def dispose(self, *, deadline: float | None = None) -> None:
        """Unwind this scope: children first, then own effects LIFO.

        The order is load-bearing for anything that registers an effect *about* a
        child — a subagent's tombstone, a supervisor's bookkeeping — because such
        an effect runs after that child's scope is already gone. See
        `ph_rlm.subagents._release`, which is exactly that shape.

        **Leaving the tree is guaranteed, whatever the unwind does** (I2). Every
        `await` below can be canceled, and `CancelledError` is a `BaseException`
        that the effect loop's `except Exception` deliberately does not catch — so
        without the `finally` a cancellation partway through left this scope
        `_active=False`, still in its parent's `_children`, still holding its
        services, and unretryable, because the early return above makes a second
        `dispose()` a no-op. That state is a leak of everything beneath it, and it
        is the one path a live process can reach it by.

        **Shielded, so an outside cancellation cannot strand the teardown it
        interrupted** — the larger half of the same failure. Until this was here,
        a Ctrl-C or a task group closing during unwind abandoned every effect
        after the one in flight: the worktree unlinked but the child unreaped, the
        lease held, and a `log.warning` nobody handled. `ph.resources` had
        hand-rolled exactly this around its own `ctx.dispose()`; every other
        caller — the agent registry, the loader, `ph_app.runtime.mounted`, and the
        recursive call below — had nothing.

        **A deadline and not a bare shield**, for `daemon/server.py`'s reason: a
        root that will not unwind must not become a process that will not exit. So
        a disposer that hangs still loses the effects behind it once the budget is
        spent — recorded by `_leave_tree`, which is the case the abandonment
        ledger genuinely exists for now that cancellation is handled here.

        **`deadline` is how a caller spends one budget over several trees.** A
        tree handed none takes `GRACE_SECONDS` from now, and nested scopes inherit
        it through `_Runtime.unwind_deadline`, so one unwind is one budget however
        deep it goes. What that cannot express is *several roots*, which are
        several runtimes: a daemon closing ten of them would otherwise spend ten
        budgets, so it passes one instant to each.
        """
        if not self._active:
            return
        self._active = False
        runtime = self._runtime
        if deadline is not None:
            runtime.unwind_deadline = deadline
        elif runtime.unwind_deadline is None:
            runtime.unwind_deadline = anyio.current_time() + GRACE_SECONDS
        budget = runtime.unwind_deadline
        owns_budget = not _UNWINDING.get()
        # Set unconditionally: `reset` restores whatever was there, so True over
        # True is a no-op and `owns_budget` stays the one name for the decision.
        token = _UNWINDING.set(True)
        try:
            if owns_budget:
                # **Entered once per unwind, not once per scope.** Every nested
                # `dispose` already runs inside this scope with the same
                # deadline, so its own would be a second shield against nothing —
                # measured at +232% on a 341-scope tree, which is the ordinary
                # shape of a mounted profile going away. `_UNWINDING` is what
                # makes "nested" mean nested rather than merely "somewhere in
                # this tree"; see its own note.
                #
                # `CancelScope(deadline=)` rather than `move_on_after(delay)`:
                # the budget is an instant, and a nested call re-deriving a delay
                # from it would drift by however long the layers above it took.
                with anyio.CancelScope(deadline=budget, shield=True):
                    await self._unwind()
            else:
                await self._unwind()
        finally:
            _UNWINDING.reset(token)
            if owns_budget:
                runtime.unwind_deadline = None
            self._leave_tree()

    def unwind_by(self, deadline: float) -> None:
        """Set the instant this tree's unwind must finish by, before it starts.

        For the caller that disposes several trees and wants **one** budget
        across them — a daemon closing ten roots, each its own `Context` with its
        own runtime, each reached through an `AsyncExitStack` that `dispose`'s
        own `deadline` argument cannot be threaded through. Without this each
        root took a fresh `GRACE_SECONDS` behind its own shield, so a bound the
        caller wrote as ten seconds was ten times that.

        Idempotent and harmless on a tree that is never disposed: the field is
        read only by `dispose`.
        """
        self._runtime.unwind_deadline = deadline

    async def _unwind(self) -> None:
        """Children first, then this scope's own effects, LIFO.

        Split out of `dispose` so the shielded and nested paths share one body
        rather than one being a copy of the other under a `with`.
        """
        for child in reversed(list(self._children)):
            await child.dispose()
        while self._effects:
            # Peeked, and popped only once it has actually run. Popping first
            # made the effect in flight when the budget expired invisible to
            # `_leave_tree` — the one most likely to be half-done, and the one a
            # reader most needs named.
            effect = self._effects[-1]
            if effect.done:
                self._effects.pop()
                continue
            effect.done = True
            try:
                await maybe_await(effect.dispose())
            except Exception:
                # A disposer that raised still ran; what it left behind is its
                # own business, and the traceback is the account of it.
                log.exception("ph.cordis: effect %r failed to dispose", effect.label)
            effect.ran = True
            self._effects.pop()

    def _leave_tree(self) -> None:
        """Unlink from the parent and drop what this scope served. Cannot fail.

        Nothing here awaits, which is the property that lets it run in a `finally`
        during cancellation: an `await` there would re-raise before finishing the
        job it is there to guarantee.

        Not `detach`, which on this class already means the opposite — running a
        coroutine *outside* the caller's lifetime.

        **What was left behind is named, never destroyed.** A non-empty `_effects`
        or `_children` here means a cancellation cut the unwind short; on the
        ordinary path both are already empty, since every child unlinks itself
        here and the effect loop drains as it goes. Those disposers were *already*
        unreachable through `dispose` before this method existed — the early
        return above makes a second call a no-op — so nothing that would otherwise
        have run is lost by arriving here. They stay callable through the
        `release` closure `add_disposer` handed out, which is why this reports
        them and leaves them alone.

        **Recorded as well as logged**, on the runtime rather than here, because
        by the last line of this method the scope that could not finish is
        unreachable from anything — unlinked from its parent, holding services it
        has dropped — which is exactly why a stranded lease or worktree used to
        be invisible to everything but a log nobody was reading. `scope_invariant`
        turns the ledger into a pollable invariant, so `phern doctor` and the
        daemon's own poll both report it without either learning what an effect
        is.

        """
        if self._effects or self._children:
            entry = Abandoned(
                path=self.path,
                watched=tuple((one.label or "unlabelled", ref(one)) for one in self._effects),
            )
            log.warning(
                "ph.cordis: %s was cut short while unwinding; %d child scope(s) and these "
                "effect(s) were never disposed: %s",
                self.path,
                len(self._children),
                list(entry.outstanding),
            )
            ledger = self._runtime.abandoned
            # Entries nothing is owed on go first: `outstanding` is derived, so an
            # entry whose effects were all released afterwards is a record of
            # history the log already carries, and evicting it before a live one
            # keeps the cap spent on what a person can still act on.
            if len(ledger) >= ABANDONED_LEDGER:
                ledger[:] = [one for one in ledger if one.outstanding or one.unreclaimable]
            ledger.append(entry)
            # One entry is appended per call and the cap is enforced on every
            # one, so the ledger can only ever be a single entry over it. A
            # general slice-and-count here defended a case that cannot arise, and
            # its comment claimed the opposite of the invariant.
            if len(ledger) > ABANDONED_LEDGER:
                del ledger[0]
                self._runtime.abandoned_dropped += 1
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

        **Deliberately a list, not a generator.** `serial` returns out of the loop
        early, so a generator holding a binding across its `yield` would be closed
        by the collector on some other task and release the token in the wrong context.
        """
        hooks = self._runtime.hooks.get(event)
        if not hooks:
            return []
        target = scope if scope is not None else self
        return [hook for hook in hooks if hook.global_ or hook.ctx.reaches(target)]

    def emit(
        self,
        event: str,
        *args: object,
        scope: Context | None = None,
        contained: bool = False,
    ) -> None:
        """Dispatch synchronously, ignoring listener return values.

        A listener that returns a coroutine is scheduled and not awaited, which
        is cordis's behavior. A listener that raises stops the dispatch unless
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

    async def serial(
        self,
        event: str,
        *args: object,
        scope: Context | None = None,
    ) -> object:
        """Await listeners in registration order until one bails."""
        event_registry.check(event, "serial")
        for hook in self._hooks(event, scope=scope):
            result = await maybe_await(_invoke(hook, *args))
            if is_bailed(result):
                return result
        return None

    async def parallel(
        self,
        event: str,
        *args: object,
        scope: Context | None = None,
    ) -> None:
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

    async def waterfall[T](
        self,
        event: str,
        *args: object,
        inner: Callable[..., Awaitable[T]],
        scope: Context | None = None,
    ) -> T:
        """Around-middleware: each listener wraps the rest of the chain.

        Listeners run outermost-first and receive `(*args, next)`. Calling
        `next()` delegates; returning without calling it vetoes the rest of the
        chain, `inner` included — that veto is how a policy plugin replaces
        built-in behavior without the built-in knowing.

        `next(*replacement)` additionally hands the rest of the chain different
        arguments. Cordis expects a listener to mutate a shared payload instead,
        which is not available here: pH's payloads are frozen values, and a
        rewrite that has to be explicit is a rewrite a reader can see.

        **The chain's type comes from `inner`.** Every waterfall settles on one
        type — `agent/pre-step` on a `PreStepDecision`, `approval/request` on an
        `ApprovalAnswer` — and that type was previously restated by hand at every
        listener, twenty of them across twelve files, with nothing checking any
        against the producer.

        What that buys is asymmetric, and worth being exact about. A *caller's*
        result is now checked: `T` flows from the `inner` it passed. A
        *listener's* `Next[T]` is not — `on` takes a `Listener`, so a listener's
        annotation is what it claims rather than what anything verifies. The
        convention is one declaration instead of twenty; the checking stops at
        the producer.

        So an `inner` must declare what the *chain* resolves to rather than what
        its own default happens to be: `tools/pre-execute` returns an `Allow`
        while the chain is a four-way `PreToolDecision`, and inferring `T` from
        the default there would be wrong for every other listener.
        """
        event_registry.check(event, "waterfall")
        hooks = self._hooks(event, scope=scope)
        state: list[object] = list(args)
        index = 0

        async def next_(*replacement: object) -> object:
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
            return await inner(*state)

        # The one place the chain's type is unverifiable: `on` takes a
        # `Listener`, rows load through entry points, so a listener's return is
        # never seen by this repo's checker. `settled` is where each producer
        # makes the claim good, and this cast is what lets them state it once
        # instead of at every listener.
        #
        # Not a `kind=` parameter here: of the fourteen chains only four could
        # pass a bare `type[T]` and five more a `T | None`, so it would be
        # omitted at ten and buy none of the can't-forget property that is its
        # whole point — while letting `T` be solved from two places, which is
        # what this signature exists to stop.
        return cast("T", await next_())

    def detach(self, coro: Any, *, label: str) -> None:  # noqa: ANN401
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

    def _spawn(self, coro: object, event: str) -> None:
        """Track a fire-and-forget coroutine returned by an `emit` listener."""
        self.detach(coro, label=f"listener for {event}")

    async def drain(self) -> None:
        """Await every detached coroutine: async `emit` listeners, and `detach()`.

        **Shielded and bounded, exactly as `dispose` is and for its reason.** The
        two are called as a pair by a host unwinding a mount, and `dispose`
        raises its own shield before its first await — so a bare drain left that
        pair with an unprotected first half, and a cancellation arriving during
        it took the whole unwind with it. The protection belongs here rather than
        around the call: `ph.resources` records what a caller's own scope buys
        against a self-shielding callee, which is that the outer deadline is
        inert, and the one host doing it by hand is how a rule gets two
        spellings.

        The bound is the other half of the same decision. Waiting on detached
        work is waiting on somebody else's checkpoint, so a listener that never
        reaches one would otherwise trade a skipped unwind for a shutdown that
        never finishes.

        **Its own share, or whatever is left of a budget somebody seeded for the
        whole teardown, whichever ends sooner.** A shield is opaque to a deadline
        outside it, so a drain that answered only to `DRAIN_SECONDS` would spend
        that much *per tree* inside a caller's total — ten roots closing under one
        ten-second bound could take fifty. `unwind_by` is how that total is
        declared, and reading it here is what makes the two halves of a teardown
        add up to it rather than to a multiple of it. The share still applies, so
        a drain cannot eat an unwind's time either.

        **`gather(return_exceptions=True)` for the distinction, not for
        concurrency**: `detach` hands every one of these to `ensure_future`, so
        they are already running and awaiting them in turn would not serialize
        anything (measured identical). What `gather` adds is that a member's
        failure or cancellation comes back as a *result* — awaiting each in turn
        re-raised it, so one listener canceled on its own account ended the drain
        for all of them. Failures are logged by the done callback `detach`
        attached, so nothing is swallowed twice.
        """
        if not self._runtime.background:
            # Nothing to wait on, so nothing to protect. The scope below costs a
            # clock read and about 3 µs to guarantee an empty loop, and this runs
            # once per unmount and some seventy times across the suite.
            return
        if _UNWINDING.get():
            # Already inside a teardown's shield. That is `dispose`'s rule and
            # this is its second reader: another scope here would be a shield
            # against nothing, at the cost `_UNWINDING` records.
            await self._settle()
            return
        share = anyio.current_time() + DRAIN_SECONDS
        whole = self._runtime.unwind_deadline
        token = _UNWINDING.set(True)
        try:
            budget = share if whole is None else min(share, whole)
            with anyio.CancelScope(deadline=budget, shield=True):
                await self._settle()
        finally:
            _UNWINDING.reset(token)

    async def _settle(self) -> None:
        """Await the detached pool until it is empty. `drain` is the only caller."""
        while self._runtime.background:
            settling = set(self._runtime.background)
            await asyncio.gather(*settling, return_exceptions=True)
            # Discarded here as well as by the done callback, so the loop ends on
            # what this call awaited rather than on when `call_soon` ran them.
            self._runtime.background -= settling
