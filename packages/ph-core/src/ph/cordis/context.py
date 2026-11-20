"""Context, services, effects, scopes and the four dispatch modes.

This module is the Python subset of Cordis that pH needs (D1). A `Context` is
three things at the same time, exactly as in dsh:

* A **repository of services**. A plugin claims `ctx.<key>` with `provide()`,
  and every other plugin finds the service by key and not by import.
* An **event bus**: `emit` / `parallel` / `serial` / `waterfall`. The declaration
  of an event fixes its dispatch mode (see `ph.cordis.events`).
* A **disposal scope**. Every registration and every acquired artifact is an
  effect that unwinds when the scope disposes (invariant I2).

There are three kinds of context, and the difference between them matters:

| kind | built by | provides into | its listeners reach |
|---|---|---|---|
| root | `Context()` | itself | everything |
| activation scope | `reconcile()`, for a plugin | the realm the row was mounted in | everything |
| isolated scope | `ctx.scope()`, for an agent | itself | that agent alone |

The service of a row is therefore visible to every sibling row. A registration
on the scope of an agent shadows the global one for that agent only. One rule,
`reaches()`, decides visibility for event dispatch and for every scoped registry.

This module departs from the TypeScript original in two places on purpose. Both
exist because Python has no synchronous await:

* An explicit ``await ctx.reconcile()`` drives activation, and not a microtask.
  A test or a loader therefore knows exactly when the plugin tree has settled.
* ``effect()`` is a coroutine, because an artifact can need an await to be
  acquired. ``add_disposer()`` is the synchronous path that `on()` and
  `provide()` use.

@module ph.cordis.context
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterator, MutableMapping, Sequence
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
    "Releaser",
    "is_bailed",
    "maybe_await",
    "settled",
    "settled_or_none",
]

log = logging.getLogger("ph.cordis")

Disposer: TypeAlias = Callable[[], object]
"""A teardown callable. It may return an awaitable, and `dispose()` awaits it.

The return type is `object` and not `Any`, because nothing here inspects what a
disposer returns. `dispose()` awaits the value and drops it, and `Any` invited a
caller to inspect it. A narrower type would be wrong. A disposer is often an
existing call that happens to return something, and `None | Awaitable[None]`
would make each of those calls discard its result explicitly.
"""

Listener: TypeAlias = Callable[..., object]
"""One registered listener. The arguments stay open, because the sixteen
`waterfall` chains across the tree take different ones. The return type is
`object` for the same reason as in `Disposer`. The dispatch methods pass what
`_invoke` returns to `is_bailed` and `maybe_await`, which both take `object`, and
`settled` checks the answer of every chain."""

AtDispose: TypeAlias = Literal["cancel", "finish", "leave"]
"""What a scope does at its dispose with work that `detach` started on it and
that still runs.

- `cancel` stops the work. Example: the task of an `emit` listener, which has no
  purpose after its row goes away.
- `finish` waits for the work inside the budget of the unwind. Example: a writer
  whose batch is a write that was already promised.
- `leave` does neither. It is for work that something else stops. Example: the
  owner of a job cancels the token of the job. A second, harder stop from the
  scope that only runs the job would arrive before the stop of that owner.
  `drain` does not wait for this work either, because that owner also decides
  how long the work gets."""

type MaybeAwaitable[T] = T | Awaitable[T]
"""A `T`, or a `T` that must be awaited first. This is what `maybe_await` takes.

This type is the seam between the ordinary code of a row and the code of this
runtime. Whoever mounted the row writes a body, a listener or a prompt section.
If the runtime required `async def` for all of them, every synchronous one would
have to be written as a coroutine that it has no use for. The runtime therefore
accepts either, and `maybe_await` resolves the value. `Disposer` above is the
deliberate exception. It does not care what is returned, and is not
maybe-awaited: `object` already includes an awaitable.

Use this type where the mixture is real. It is an *inference* trap in a
parameter that must solve `T` from the declared return of a body. For that
reason the tool pipeline overloads on the kind of body and does not spell this
union.

The type has a name because every declaration that takes a body spelled it by
hand, and each spelled it differently: `T | Awaitable[T]`, `Awaitable[T] | T`,
and `str | Awaitable[str | None] | None` are one type in three forms. The last
form is what `RUF036` leaves when `T` is itself optional, and at that point the
hand-written form is no longer readable."""

type Next[T] = Callable[..., Awaitable[T]]
"""The rest of a `waterfall` chain, as the listener that wraps it sees it.

`T` is the type of the chain, which `waterfall` reads from the `inner` of the
producer. A listener on `agent/pre-step` therefore takes a
`Next[PreStepDecision]` and returns a `PreStepDecision`. The type has a name
because twenty listeners across twelve files spell this shape, and ten of them
imported `Callable` and `Awaitable` for nothing else.

It states a convention and not a check. `on` takes a `Listener`, so the
annotation of a listener is a claim that nothing verifies."""

_MAX_RECONCILE_ROUNDS = 64
_MISSING: Final = object()


async def maybe_await[T](value: MaybeAwaitable[T]) -> T:
    """Await `value` when it is awaitable. Otherwise return it unchanged.

    The function is generic for the same reason as `waterfall`. The caller knows
    what it passed in, and an `Any` here erased that type again at every call
    site: `effect` got back an untyped `dispose`, and `code_mode` got an untyped
    namespace. A caller whose own argument is `Any` still gets `Any`, which is the
    correct answer. That caller must fix the erasure, and this function cannot.
    """
    if inspect.isawaitable(value):
        return await value
    return value


def settled[T](
    event: str, value: object, kind: type[T], *, refusal: type[Exception] = TypeError
) -> T:
    """Return what the chain returned, after a type check. The last step of
    `waterfall` is only a `cast`.

    A listener arrives through an entry point, and `on` takes a `Listener`. No
    checker between the row and this function therefore sees what a listener
    returns. `waterfall` states the type of the chain. This function checks that
    claim, one time for each producer.

    The check is here and not hand-written in each producer. Of nine producers,
    three raised a `TypeError`, three silently substituted a default, two coerced
    with `str()` and one never checked. A single broken row therefore ended a
    turn in one chain and went unnoticed in another. The message names the event
    because the event is what a person then looks at.

    **A seam that owns a refusal vocabulary names it.** `refusal=` exists for
    `ctx.fs`, whose `FsDenied` is a `HarnessError` on purpose. A veto must reach
    a consumer as a *denial*, because a Code Mode program can `except` a `failed`
    and route around it. A wrong answer from a policy listener is still a
    refusal, so it must not arrive as the other kind.

    There are two separate limits. First, this function needs a type that
    `isinstance` can test, so the union chains (`tools/pre-execute`,
    `tools/post-execute`) narrow by arm instead. Second, `approval/request` keeps
    its own lookup *and denies instead of raising*. That is a policy choice: on
    that path a refusal is the safe answer and an exception is not.
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
    """Whether the return value of a listener stops a `serial` dispatch.

    Ported verbatim from cordis: any value except `None` and `False` bails. `0`
    and `""` bail on purpose. A decision object is never falsy by accident, and
    this rule prevents the bug of treating a legitimate zero as "no answer".
    """
    return value is not None and value is not False


@dataclass(slots=True)
class Hook:
    """One registered listener record."""

    ctx: Context
    callback: Listener
    prepend: bool = False
    global_: bool = False


def _invoke(hook: Hook, *args: object, detach_for: str | None = None) -> object:
    """Call one listener as an effect of the scope that registered it (P6-25).

    **This is the one place that establishes ownership for a dispatch.** It is a
    function and not a line in each loop. There are four dispatch modes, and a
    fifth cannot omit a step that it never writes.

    Sync and async listeners use one path. When a listener returns an awaitable,
    the awaitable comes back wrapped in `_as_owner`, so whoever awaits it
    receives a coroutine that binds itself. A listener that returns a value has
    already run inside the binding. Callers do not need to know which kind they
    have.

    **`emit` spawns instead, inside the binding** (`detach_for`, the name of the
    event). A task copies the context in which it is created, so the task runs as
    the owner with no wrapper. A stop that arrives before its first step means
    that it never runs. With the wrapper, the listener started only at that step
    and was then canceled at its first `await` after its scope had gone. A stop
    that arrived sooner left the listener never awaited.

    The binding is written inline and does not go through `running`, because
    `emit` fires one time for each streamed chunk. At that rate even the frame of
    a context manager costs too much.
    """
    # Inline, and through the memoized self-pair of the scope. To build a
    # `Running` for each call means that a frozen dataclass sets its fields
    # through `object.__setattr__`, one time for each listener for each chunk.
    owner = hook.ctx
    pair = owner._running_self
    if pair is None:
        pair = owner._running_self = Running(owner, owner)
    token = _ACTIVATING.set(pair)
    try:
        result = hook.callback(*args)
        # `None` is tested first, and this gives most of the gain. A listener that
        # returns nothing is by far the most common case, and the awaitability
        # test costs more for each listener than the binding it was added to
        # serve.
        if result is None or not inspect.isawaitable(result):
            return result
        if detach_for is not None:
            owner.detach(result, label=f"listener for {detach_for}", at_dispose="cancel")
            return None
        return _as_owner(owner, result)
    finally:
        _ACTIVATING.reset(token)


@dataclass(slots=True, weakref_slot=True)
class _Effect:
    """One registered teardown, and two facts about it that are different.

    `done` means **claimed**: somebody has taken responsibility for running this
    teardown, so nobody else may run it. `ran` means **completed**: nothing is
    owed. One flag served both until the unwind got a deadline. After that the
    difference mattered. When the budget expires, the effect that is in flight
    is claimed and has not completed (`_unwind`). A single flag reported that
    effect as owing nothing.

    `homes` holds the effect list of every scope that the effect sits on. That is
    one list, or two for a teardown that goes when either scope of a pair goes
    (`Running.add_disposer`). A settled effect leaves all of them and keeps none.
    """

    dispose: Disposer
    label: str
    homes: tuple[list[_Effect], ...] = field(compare=False, repr=False)
    """Excluded from `==` and `repr`. Each home holds the other effects, so a
    comparison or a print of one effect would walk them all."""
    done: bool = False
    ran: bool = False

    def settle(self) -> bool:
        """Claim the effect as owing nothing, outside an unwind, and take it off
        its scopes. Return `False` if somebody already claimed it. The effect is
        marked before it is removed, so an unwind that arrives first finds it
        claimed."""
        if self.done:
            return False
        self.done = self.ran = True
        self.leave()
        return True

    def leave(self) -> None:
        """Take the effect off every scope that it sits on. Removal is by
        identity, because two registrations of one bound method with one label
        produce equal `_Effect`s (A8)."""
        for home in self.homes:
            remove_identical(home, self)
        self.homes = ()


def _register(dispose: Disposer, label: str, homes: tuple[list[_Effect], ...]) -> _Effect:
    """Register one teardown as the newest effect of every scope whose list is in
    `homes`.

    `homes` is a tuple that the caller already has, and not varargs. A scope
    keeps its own one-list tuple (`Context._homes`), so a registration builds no
    tuple."""
    effect = _Effect(dispose, label, homes)
    for home in homes:
        home.append(effect)
    return effect


class Releaser:
    """What `add_disposer` and `effect` return: a hold on one registered teardown.

    **A call runs the teardown now** and takes it off the scope. This happens one
    time: a second call does nothing, and a call after the unwind of the scope
    claimed the teardown also does nothing. A `Releaser` is therefore a
    `Disposer`, and can be registered as one.

    **`drop()` takes the teardown off the scope and does not run it.** It is for
    a holder that has already let the artifact go another way and so owes
    nothing. Without `drop()`, the only way to remove a teardown was to run it
    after it could find nothing to do, and correctness then depended on the order
    of two steps. A dropped teardown is settled like one that ran: `Abandoned`
    reports it as neither outstanding nor unreclaimable. A drop after a run, a
    run after a drop, and a drop that the unwind got ahead of all do nothing. In
    the last case the teardown is running, or is about to run.

    A teardown registered on two scopes (`Running.add_disposer`) is dropped from
    both.
    """

    __slots__ = ("_effect",)

    def __init__(self, effect: _Effect) -> None:
        self._effect = effect

    def __call__(self) -> object:
        effect = self._effect
        return effect.dispose() if effect.settle() else None

    def drop(self) -> None:
        self._effect.settle()


GRACE_SECONDS = 10.0
"""How long the unwind of one scope tree may take before it stops waiting for
itself.

**The number is here and not in `ph.resources`, because this module now
enforces it.** The number was `resources.GRACE_SECONDS`. There, the
process-level shutdown path wrapped `ctx.dispose()` in a shielded
`move_on_after`. That caller hand-wrote a guarantee that the primitive should
own, and every other caller had no guarantee. `dispose` now applies the number
itself, and `ph.resources` re-exports the name so that the shutdown path reads
unchanged.

The value is ten seconds because it is a budget for *finishing*, and not a
timeout for one operation: a worktree to unlink, a child to reap, a proxy to
close. The shutdown in `daemon/server.py` states the other half: a root that
will not unwind must not become a process that will not exit. For that reason
this is a deadline and not a bare shield.
"""


def releasing() -> anyio.CancelScope:
    """A cleanup scope that survives the cancel that sent it, but not for ever.

    A `finally` that returns a kernel resource (an overlay unmount, a worktree
    deregistration, the disposer of a child) can be reached under raw
    cancellation. Without a shield its `await` is skipped. The result is a live
    FUSE mount that nobody unmounted, a registration that makes `git branch -D`
    refuse the branch that an export just produced, or a subagent that still
    spends tokens for a turn that was abandoned.

    **It is shielded with a deadline, and not only shielded.** An unmount that
    hangs because the backend is stuck is the other way to lose a process. A
    shield with no bound makes that process unkillable, which is the failure that
    `dispose` already argues against. `GRACE_SECONDS` is the time in which the
    rest of the tree unwinds, so a cleanup may not be slower than the runtime
    that owns it. If the deadline expires, the cleanup is abandoned. That is the
    same outcome as with no shield, except that the cleanup has been tried.

    The function is here and not in the seam that first needed it. The pattern
    was already written inline at six sites across the tree, and the next author
    would have written a seventh.

    **Inside an unwind it takes the deadline of the tree, and not a fresh one**
    (L10). It used to take `GRACE_SECONDS` unconditionally. A cleanup reached
    *during* a dispose therefore opened a second budget inside the budget of the
    tree. That is "ten roots, ten budgets", the shape that `unwind_by` exists to
    prevent, one layer down. `_UNWIND_BUDGET` carries the instant at which the
    shield above this task expires, so the whole teardown still finishes by the
    number that its caller chose. Outside an unwind there is nothing to inherit
    and this is the first budget, which is the original behavior.
    """
    budget = _UNWIND_BUDGET.get()
    deadline = anyio.current_time() + GRACE_SECONDS if budget is None else budget
    return anyio.CancelScope(deadline=deadline, shield=True)


DRAIN_SECONDS = GRACE_SECONDS / 2
"""How long `drain` waits for detached work before the unwind behind it starts.

**The value is derived from `GRACE_SECONDS` and not declared beside it.** A host
that tears a mount down spends the two back to back, and the relation between
them is the purpose. A listener that still runs here is stopped by the unwind of
its own scope anyway. The effects behind it are worktrees, leases and child
processes, which outlive the process if nobody returns them. When there is not
time for both, the unwind is the one that must run. If the value were a literal,
that statement would become false the first time somebody raised the grace
period.
"""

_UNWIND_BUDGET: ContextVar[float | None] = ContextVar("ph.cordis.unwind-budget", default=None)
"""The instant at which the shield above this task expires, or `None` outside a
shield (L10).

This is the companion of `_UNWINDING`, and it is set in the same two places for
the same reason. `_UNWINDING` answers *whether* there is a shield above this
task. This variable answers *until when*. The two are separate and not folded
into one optional float, because `drain` shields to `min(share, whole)` while
`dispose` shields to the whole. Those are two different instants for one boolean
fact, and a reader that asks "am I nested" should not need to know which instant
it got.

Only `releasing` reads it. `releasing` is the cleanup that must fit inside
whatever is already running."""

_UNWINDING: ContextVar[bool] = ContextVar("ph.cordis.unwinding", default=False)
"""Whether the caller is already inside a shielded unwind.

**The value is per task and not per tree, and that difference is the reason for
this variable.** It was a field on `_Runtime`, which every scope in one tree
shares. That field answered the nested case correctly: `dispose` recurses into
children, and a second shield for each layer costs 232% on a 341-scope tree and
adds no guarantee that the outer shield does not already give. The field could
not distinguish a *sibling*: two scopes of one tree that are disposed
concurrently on two tasks. The second scope found the flag set by the first and
ran its whole unwind with no shield, so the cancellation of the first stranded
the effects of the second. A daemon reaches that shape whenever a passivation
overlaps a shutdown.

A `ContextVar` answers the question that was always meant: is there a shield
above *me*. A nested `dispose` inherits the value on the same task. A sibling on
another task does not inherit it, and raises its own shield.
"""

ABANDONED_LEDGER = 64
"""How many cut-short unwinds one tree remembers.

The ledger is bounded because a process that keeps abandoning would otherwise
grow it without limit. An entry holds only labels and weak references (see
`Abandoned`), but an unbounded list of entries would still be a leak caused by
the report of a leak. 64 is far more than a reader counts before they ask what
is wrong with the deployment. The number of entries that were discarded is kept,
so the report never silently understates.
"""


@dataclass(frozen=True, slots=True)
class Abandoned:
    """One scope whose unwind was cut short, and the teardowns that it still held.

    **`outstanding` is derived, not recorded.** The entry watches the live
    effects and does not snapshot their names. An effect that is released or
    dropped afterwards, through the `Releaser` that `add_disposer` returned,
    therefore leaves the report by itself. A snapshot of names would keep
    accusing a deployment of a lease that somebody had already returned.
    `dispose()` cannot be that somebody, because its early return makes a second
    call a no-op. Those releasers are the only way back, so the ledger must
    notice when they are used.

    **The references are weak, because a strong reference would be a leak caused
    by the report of a leak.** A strong reference keeps far more than the
    disposer alive. The `unprovide` of `provide` captures the `Context` and the
    service, and the `off` of `on` captures a `Hook` that holds the same context.
    Any scope that ever called either therefore kept its whole subtree and every
    service in it alive for the life of the tree, and `outstanding` mentioned
    none of that. Real disposers here are bound methods on live objects (an httpx
    client and its pool, a subprocess handle). CPython reclaims such an object on
    its last reference, and cannot while a diagnostic holds one.

    The label is cached beside the reference because the label outlives the
    reference. The two cases are different answers. A live reference that is not
    yet done is **outstanding**: somebody still holds the releaser and can call
    it. A dead reference is `unreclaimable`, because nothing can ever call it
    again.
    """

    path: str
    watched: tuple[tuple[str, ref[_Effect]], ...]
    """The label of each undisposed effect and a weak reference to the effect.
    Read it through `outstanding` and `unreclaimable`. Nothing else dereferences
    it."""

    @property
    def outstanding(self) -> tuple[str, ...]:
        """The effects that nobody has claimed and that a holder could still run.

        An effect here is unclaimed *and* still referenced. A `Releaser` refuses
        an effect that the unwind already claimed, and that guard stops a
        disposer from running twice. Nobody can therefore act on a claimed
        effect, however incomplete it is.
        """
        return tuple(
            label for label, watch in self.watched if (one := watch()) is not None and not one.done
        )

    @property
    def unreclaimable(self) -> tuple[str, ...]:
        """The effects that nothing will ever run, for either of two reasons.

        First, the last holder is gone, so no releaser remains to call. Second,
        the unwind claimed the effect and never finished it, and a `Releaser`
        then refuses it on the `done` guard. A deadline creates the second case.
        The disposer that was cut off in the middle of an unmount is the one most
        likely to have left something half-done, and it is also the one that
        nobody can retry.

        These are kept apart from `outstanding` because the two ask different
        things of a reader. `outstanding` names work that somebody can still do.
        This property names a resource that is stranded until the process ends.
        To tell an operator to act on it would be to ask for something
        impossible.
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
    """A registration that waits for services: a mounted plugin or an `inject()`.

    It owns the one deactivation sequence. Every path that tears an activation
    down (a service that disappears, an unmount, a failed `apply`) goes through
    `deactivate()`, so the protocol is in one place.
    """

    ctx: Context
    keys: tuple[str, ...]
    activate: Callable[[Context], Any]
    label: str
    transparent: bool = False
    """Whether the scope of this activation is transparent, as the scope of an
    ordinary row is.

    Only the loader sets it, for the row that asked for a realm (A9). `Context`
    already promises that "a plugin's activation scope is transparent and
    answers `None`". A realm mount inherits the isolation of the realm, so the
    promise was false there, and the row became invisible to `reaches`. It got no
    dispatch, and nobody could see a tool, prompt section, fs screen or skill
    restriction that it registered. Restoring the transparency fixes all of them
    at one time, with no separate setting for each registry."""
    module: str = ""
    """The module whose code runs in the activation scope, when one is known.

    For a plugin, `ForkScope` stamps it from the `apply` of the plugin. It
    reaches `Context._module`, which is what `ctx.on` reports to `note_consumer`.
    This field is therefore the reason that `phern events` can name who listens
    to an event and does not print an empty column. It is blank for an `inject`
    callback. That callback is code that the calling row already owns, so it
    inherits the module of that row."""
    plugin: str = ""
    """The name of the mounted plugin, stamped by `ForkScope`. It is blank for an
    `inject`, which runs as the row that asked. It reaches `Context.plugin_name`,
    which is how `waterfall_attributed` names the row that answered."""
    row: str = ""
    """The profile row that the plugin was mounted for, when the loader said. It
    reaches `Context.row_id`. It is blank for an `inject`, and for a plugin that
    was mounted by hand."""
    active: bool = False
    ever_active: bool = False
    """Whether this dependent has ever activated. `waiting on fs` can describe a
    fiber that never came up, and also one that came up and was unwound when `fs`
    went away. This bit tells them apart. Only the second is a provider swap,
    which is what a live reader of the topology asks about."""
    scope: Context | None = None
    disposed: bool = False
    failure: str | None = None
    """Why the activation of this dependent raised, or `None`. It is set one time,
    and the activation is never retried.

    **This field stops the next `reconcile` from running a failed `apply`
    again.** `deactivate()` marks the tree dirty. A dependent that was still
    `ready()` after its `apply` raised was therefore activated again on the next
    round: the same `apply`, which raised the same way, ahead of every row that
    still waited for its turn in the list. A profile with one broken row
    therefore mounted nothing behind that row. It also re-ran whatever that row
    had already done, one time for each `reconcile`, for the life of the process.

    The field is a reason and not a flag, for the same reason as in `Abandoned`.
    The state is reported to a person: `phern doctor` prints it beside
    `waiting on`, and "failed" with no sentence is a report that sends the reader
    to the logs. It is `None` and not an empty string, so that the test below is
    a comparison. A truthiness test would hold only because the message is never
    blank.
    """

    def ready(self) -> bool:
        return self.ctx.active and self.failure is None and not self.missing()

    def missing(self) -> list[str]:
        """The inject keys that are not yet provided at the scope of this
        dependent.

        This is one predicate for `ready`, for the refusal of the loader, and for
        the `waiting on …` of `phern doctor`. It is asked against `self.ctx`,
        which for a private copy in a realm is the realm. Two callers had written
        it against two different scopes."""
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
    """The detached tasks that `drain` waits for: every `detach` except one with
    `leave`."""
    elsewhere: set[asyncio.Future[Any]] = field(default_factory=set)
    """The detached tasks with `leave`, which `drain` does not wait for. They are
    held only so that the weak reference of the loop is not their last reference:
    asyncio keeps no strong reference of its own."""
    unwind_deadline: float | None = None
    """The instant by which the unwind of this tree must finish, or `None` when
    none is set.

    **There is one budget for each tree, and not one for each scope.** `dispose`
    recurses into children, so a deadline computed fresh at each level would
    multiply down the depth of the tree: `GRACE_SECONDS` at every layer. That is
    exactly the failure that `daemon/server.py` names when it insists on one
    shared number.

    `drain` and `dispose` read it, and spend it in that order. Whoever starts the
    unwind sets it, or `Context.unwind_by` sets it *before* the unwind. That is
    how a caller that holds several trees spends one budget across all of them.
    It is cleared when that unwind ends.
    """
    abandoned: list[Abandoned] = field(default_factory=list)
    """The unwinds that this tree could not finish, newest last, capped at
    `ABANDONED_LEDGER`. The list is on the runtime and not on the scope. By the
    end of `_leave_tree`, nothing can reach the scope that could not finish, and
    that is the reason why these unwinds were invisible."""
    abandoned_dropped: int = 0
    """How many entries the cap discarded. A report can then say so, and does not
    imply that the oldest abandonment in the list was the first."""


class ForkScope:
    """A mounted plugin: its config, and the scope that its activation owns.

    A fork survives deactivation. When an injected service disappears, the
    activation scope is disposed, which unwinds everything that the plugin
    registered. The fork stays mounted, so the plugin activates again when the
    service is provided again.
    """

    __slots__ = ("_config", "_dependent", "_parent", "_spec", "_unmount")

    def __init__(
        self,
        parent: Context,
        spec: PluginSpec,
        config: object,
        *,
        transparent: bool = False,
        row: str = "",
    ) -> None:
        self._parent = parent
        self._spec = spec
        self._config = config
        self._dependent, self._unmount = parent._register_dependent(
            spec.inject,
            self._apply,
            label=f"plugin({spec.name})",
            plugin=spec.name,
            row=row,
            transparent=transparent,
            # Where the code of the plugin lives. Nothing else in the mount path
            # knows it: the loader has a name and an entry point, and the
            # *module* is what `ctx.on` reports as a consumer. Without it, every
            # scope inherited the empty string of the root. `note_consumer` has
            # the guard `if module`, so it recorded nothing, for every event, in
            # every profile.
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
        """Whether it has activated at least one time. If `active` is also false,
        it was unwound."""
        return self._dependent.ever_active

    @property
    def failure(self) -> str | None:
        """Why this plugin's `apply` raised, or `None`. See `state`."""
        return self._dependent.failure

    @property
    def state(self) -> ForkState:
        """Where this fork stands, as one of six names.

        **The set is closed, so it is a `Literal` and not a ladder that each
        reader rebuilds.** `Mount.topology` publishes this through the `topology`
        seam, so it is a surface and not a log line. The printer that used to
        derive it from five booleans also owned the *ordering*: `failed` must be
        answered before `waiting on`, because a row that raised can also be
        missing a key, and the reason it raised is the more useful sentence. A
        second consumer that derives the state for itself has a second chance to
        get the ordering wrong.

        `failed` is the one state that does not resolve itself. A waiting row
        comes up when its key arrives. A failed row stays down however the tree
        settles, because `ready()` does not give the same `apply` a second
        chance. `activating` means that every key is met and `reconcile` has not
        reached the fork yet. Only a reader inside the fixpoint ever sees it.
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
        """Whether `dispose()` has run. The fork is then retired from the tree and
        never activates again."""
        return self._dependent.disposed

    @property
    def injects(self) -> tuple[str, ...]:
        """The service keys that this plugin waits for. `Mount.topology` names
        them."""
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

    @property
    def spec(self) -> PluginSpec:
        """The plugin that this fork mounted: its config model and its narrower."""
        return self._spec

    async def _apply(self, ctx: Context) -> None:
        await maybe_await(self._spec.apply(ctx, self._spec.resolve_config(self._config)))

    async def dispose(self) -> None:
        """Unmount the plugin: deactivate it and drop it from the tree."""
        await maybe_await(self._unmount())
        await self._parent.reconcile()


@dataclass(frozen=True, slots=True)
class Deployment:
    """Everything that this deployment holds: the widest boundary, with a name
    (P6-32).

    **It is "widest" along the restriction axis only.** A reader that is given
    `DEPLOYMENT` resolves the isolation chain of the *mount*, which no
    restriction narrows. An **agent-scoped registration is not on that chain**:
    `view(DEPLOYMENT)` does not see a tool registered on the scope of one agent.
    `DEPLOYMENT` is the view that `scope=None` always read, with a name. "Is this
    available to *anyone*" is a per-layer question that this sentinel does not
    answer.

    **The purpose is that it has a name.** `scope: Context | None = None` gave
    two different things one spelling. One is "I did not state a boundary", which
    is an absence of information. The other is "I mean the deployment", which is
    a legitimate answer. With one spelling, every reader must choose a default
    for the ambiguous case, and the convenient default is the mount: the widest
    boundary there is.

    It is not `None` with the meaning "nothing", which replaces a failure that is
    silently wide with one that is silently narrow. The part that does the work
    is the removal of the *default*. A call site that states no boundary then
    fails mypy, and not a runtime rule.
    """


DEPLOYMENT = Deployment()
"""The one instance. There is nothing to construct and nothing to configure."""

Boundary: TypeAlias = "Context | Deployment"
"""What a policy reader takes: a scope, or `DEPLOYMENT` for all of them.

It is deliberately **not** `| None`. A reader whose parameter has no default
cannot be called without an answer to the question. mypy asks at build time, at
every call site, and not only on the paths that a test drives.

**A *registration* takes a `Context`, never this.** The two dispatch-time
resolvers, `Context.owner_for` and `Context.layer_for`, also take a `Context`.
They read the running binding and not a stated value. A scope is what a
registration is an *effect of* and what it is visible *to* (the two questions of
P6-12). `DEPLOYMENT` is neither, because there is nothing for it to unwind with.
"""


def remove_identical[T](entries: list[T], value: T) -> None:
    """Remove **that object** from `entries`, and not an object equal to it. Do
    nothing if it is absent.

    `list.remove` compares with `==`, and two registrations of one value are
    equal by field. Disposing the second therefore deleted the first, and the
    disposer of the first then deleted the survivor, which reordered what was
    left. Every registry of values releases through this function: a hook
    (`Context.on`), a contribution (`claim_entry`), a per-scope filter
    (`ScopedEntries`), and the effects of a scope.

    **The search starts from the newest.** Registrations unwind newest first, so
    the newest occurrence is usually the one that goes: an effect that `_unwind`
    just ran, or the stop of a detached task. For one object that was claimed
    twice, the newest occurrence is the one that the later claim put there, so
    undoing that claim leaves the earlier one where it was. With a search from
    the front, the survivor moved behind everything that was claimed between the
    two.
    """
    for index in reversed(range(len(entries))):
        if entries[index] is value:
            del entries[index]
            return


def drop_dead_chains(cache: MutableMapping[tuple[Context | None, ...], Any]) -> None:
    """Delete every memo entry whose innermost scope has been disposed (I2).

    A chain-keyed cache holds its keys **strongly**, so an entry outlives the
    scope that it describes. An agent is disposed, the key tuple retains its
    `Context`, and nothing drops the entry until the next invalidation of the
    registry. For a deployment that has stopped registering, that never happens.
    The cost was measured at ~2 KiB for each agent, with no bound: 500 agents
    through a daemon leave 500 entries in each of the two caches.

    **Only `chain[0]` is tested**, and not every key. The innermost scope is the
    one that dies. `dispose` unwinds children inside the dispose of the parent,
    so an ancestor cannot be dead while its descendant is live, except for a
    moment in the middle of an unwind. The next miss catches that entry. A scan
    of the whole chain cost 4x more to answer the same question.

    **The cache is swept and not unwound, although the alternative is possible.**
    `Context` has a `__weakref__` slot. An eviction disposer for each scope, with
    a `WeakSet` of hooked scopes, would retain nothing and need no
    re-registration, so the mechanics do not rule it out. The sweep was chosen
    because it is smaller: no hook table for each registry and no disposer for
    each scope. A cache may legitimately forget. The table of a registry is
    different: it holds a disposer for each entry anyway, and drops a scope with
    its last entry (`ScopedTable`).

    It is called on a **miss**, which keeps it O(live scopes). A miss is the only
    moment at which the table grows. A sweep at that moment keeps the count
    proportional to what is alive, and not to what has ever run.

    The function is here, beside `chain_label` and `isolation_chain`, because the
    shape of the key is the only thing that the tool-view cache and the
    skill-reach cache share.
    """
    dead = [chain for chain in cache if chain and chain[0] is not None and not chain[0].active]
    for chain in dead:
        del cache[chain]


def chain_label(chain: Sequence[Context | None]) -> str:
    """The name that a report gives a cache entry keyed by an isolation chain.

    It reads `chain[0]` and does not scan for the first non-`None`. That is a
    property of the chain and not a shortcut: `isolation_chain` appends only
    non-`None` keys and then exactly one trailing `None`. The innermost key is
    therefore the first, and `None` can only be last. A scan would suggest that
    an interior `None` is possible, and neither caller could handle that shape
    anyway.

    The function is here and not in either caller, because both chain-keyed
    caches (the tool view and the skill reach) print this label in the same
    `phern doctor` section. With two spellings of one ordering invariant, the two
    rows could disagree about which scope an entry belongs to.
    """
    return chain[0].path if chain and chain[0] is not None else "the deployment"


def boundary_of(scope: Boundary, mount: Context) -> Context:
    """Resolve a stated boundary to the scope that answers it.

    There is one narrowing site and not one for each seam, and mypy is the
    reason. `scope is DEPLOYMENT` looks like the obvious spelling, but it does
    **not** narrow the union, because identity against a value is not a type
    guard. Every seam that wrote it by hand would either repeat an `isinstance`
    or use a cast. Written one time here, the union is resolved and a reader gets
    a `Context`.
    """
    return mount if isinstance(scope, Deployment) else scope


@dataclass(frozen=True, slots=True)
class Running:
    """Who is running, as the *two* questions that it must answer (P6-29).

    The class is public because it is also **what a registry records at
    registration time**. Every registry invokes the body of a row later: a tool,
    a command, a compaction note, a prompt section, a status field, a diagnostic,
    a telemetry sink, each of the five single-slot providers. To enter the right
    binding at that time, the registry must have kept both answers at
    registration. `Context.running_for` gives it exactly this object.

    The two answers differ exactly when a **registry** invokes the body. The
    `execute` of a tool belongs to the row that registered the tool (I2: whose
    code it is). It is visible to the *agent* that it was invoked for (B7: a
    different scope).

    There are two fields and not a second `ContextVar`, because they are one
    fact, *who is running*, read two ways. Two variables can be set out of step.
    The only way to bind one field is `running()`, which takes both.
    """

    owner: Context
    """The scope whose lifetime a registration made now joins. See
    `Context.owner_for`."""
    layer: Context
    """The scope that the registration is visible to. See `Context.layer_for`."""

    def add_disposer(self, dispose: Disposer, *, label: str = "") -> Releaser:
        """Release when **either** scope ends: the lifetime of the pair, and not
        of one half.

        When a body that a registry invokes registers as its row *for* an agent,
        the two scopes are unrelated branches and either can end first. Each
        single choice is wrong in a different direction. If the row alone owns
        the registration, the entry that a registry keys on the layer stays under
        a disposed scope. If the agent alone owns it, the registration can
        outlive the row whose code made it, which is I2 verbatim. The
        registration has meaning only while both are alive.

        **The method is here and not in each registry**, because more than one
        registry keys on the layer (`ToolRuntime._layers`,
        `ph.seams.skills._restrictions`). It is on `Running` and not on
        `Context`, because `Context.add_disposer(..., also=)` would ask one scope
        to know about a pair that it does not hold.

        **It is one effect on the lists of both scopes.** Whichever scope unwinds
        first runs the effect and takes it off the other, and the releaser runs
        or drops it on both. It used to be two effects: the effect of the layer
        ran the effect of the owner, and a closure and a link kept them in step.
        """
        if self.layer is self.owner:
            return self.owner.add_disposer(dispose, label=label)
        self.owner._assert_active()
        self.layer._assert_active()
        return Releaser(_register(dispose, label, (self.owner._effects, self.layer._effects)))


_ACTIVATING: ContextVar[Running | None] = ContextVar("ph.cordis.activating", default=None)
"""Whose code is running: the activation scope of an `apply`, the owner of a
listener, or the (row, target) pair that a registry enters around a body that it
invokes.

See `Context.current_owner` and `Context.current_layer`. See `_invoke` for the
listener half."""


class running:
    """Bind "who is running" for the duration of a block.

    **It takes the `Running` itself**, and every registry calls it that way. The
    pair exists so that a registry does not keep two fields that it can pair up
    wrongly, and `running(a.owner, b.layer)` at each of those call sites would be
    silently valid. `layer=` beside a pair *overrides* the visibility half. That
    is the one thing that a caller legitimately knows better: `ph.seams.fs` binds
    the agent whose path is being resolved, and not the scope that the resolver
    registered on.

    **`None` binds nothing.** This is for the empty half of an at-most-one slot:
    five seams hold a provider that might not be registered.

    **`layer` defaults to `owner`**, which covers the ordinary case. An `apply`
    and a listener each run as one scope, so `running(scope)` means what it
    always meant. There is one call with a default and not two functions,
    because the *shape of the call* then shows whether the two questions have one
    answer, and the record that it binds cannot be half-set.

    **It is single-use**, like every context manager that is built at its `with`.
    The token is on the instance, so entering one object a second time would lose
    the outer token.

    It is a class and not a `@contextmanager`, because the generator form costs a
    frame, a `StopIteration` and two `send`s at each of the forty call sites. It
    is entered one time for each tool call, one time for each slash command, one
    time for each prompt row in each `assemble`, and one time for each telemetry
    sink for each record.
    """

    __slots__ = ("_pair", "_token")

    def __init__(self, owner: Running | Context | None, layer: Context | None = None) -> None:
        pair: Running | None
        if owner is None:
            pair = None
        elif isinstance(owner, Running):
            pair = owner if layer is None or layer is owner.layer else Running(owner.owner, layer)
        elif layer is None or layer is owner:
            # The case where the two are the same scope, which is every `apply`
            # and every listener. The scope keeps one record and does not build
            # one for each entry. The record is cleared at dispose (in
            # `_leave_tree`), because the pair holds `self` twice, and a disposed
            # context must become collectable at the moment its parent drops it.
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
    """Await something as an effect of `scope`, and bind when the body *runs*.

    When `_invoke` calls an `async def` listener, the call only *builds* a
    coroutine. The body runs later, when `serial`, `parallel` or `waterfall`
    awaits it, and by then the binding around the *call* has been reset. Wrapping
    the awaitable moves the binding inside the body. The binding is then the
    first act of the coroutine itself, and not something that its creator did
    before it passed the coroutine on. `emit` needs no wrapper: it spawns inside
    the binding instead (`_invoke`).
    """
    # This goes through `running` and not the inline pair that `_invoke` uses.
    # There is already an extra coroutine frame around an `await` here, so the
    # cost of `__enter__` and `__exit__` is small beside a scheduler hop. The
    # module also then has two spellings of the binding, and not three.
    with running(scope):
        return await awaitable


class Context:
    """One node of the plugin tree: services, listeners and effects."""

    __slots__ = (
        "__weakref__",
        "_active",
        "_children",
        "_effects",
        "_homes",
        "_isolation",
        "_label",
        "_module",
        "_parent",
        "_plugin",
        "_provide_to",
        "_row",
        "_running_self",
        "_runtime",
        "_services",
    )

    # Declared for the type checker. `__slots__` still owns the storage.
    _parent: Context | None
    _label: str
    _module: str
    _plugin: str
    _row: str
    _children: list[Context]
    _effects: list[_Effect]
    _homes: tuple[list[_Effect]]
    """`(_effects,)`, built one time: the `homes` of an effect that is registered
    here."""
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
        # Inherited, so a scope that a plugin opens for itself still answers as
        # the plugin. `_activation_scope` is what sets it.
        self._plugin = parent._plugin if parent is not None else ""
        self._row = parent._row if parent is not None else ""
        self._children = []
        self._effects = []
        self._homes = (self._effects,)
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
        scope = cls(
            owner,
            label=dependent.label,
            provide_to=owner._provide_to,
            module=dependent.module or owner._module,
        )
        if dependent.plugin:
            scope._plugin = dependent.plugin
        if dependent.row:
            scope._row = dependent.row
        if dependent.transparent:
            # **A deployment row mounted into its own realm is still a
            # deployment row** (A9). `isolate:` narrows its *service lookup*,
            # and not who can see what it registers. But the scope inherits the
            # isolation of the realm, so `reaches` found that it was an ancestor
            # of no dispatch target, and every registry answered "nobody". This
            # restores the transparency that the activation scope of an
            # unisolated row already has. That fixes dispatch, tools, prompt
            # sections, fs screens and skills together, because they all ask
            # `reaches`.
            #
            # `provide` is unaffected: it goes through `_provide_to`, which is
            # still the realm. `isolation_chain()` walks *parents*, so a lookup
            # from here still finds the realm first.
            parent = owner._parent
            scope._isolation = parent._isolation if parent is not None else None
        return scope

    def owner_for(self, scope: Context | None = None) -> Context:
        """Whose lifetime a registration made *now* belongs to (I2, P6-12).

        A seam calls this as `self.ctx.owner_for(scope)`. `self.ctx` is the
        context of the seam itself, so it is the fallback and not the answer.
        There are three cases, in order:

        * **An explicit `scope=`.** The caller said so, and this is the only
          thing that overrides the rest. It means *"register on someone else's
          lifetime"*.
        * **The activation scope**, when the `apply` of a row is what is running.
          Cordis disposes that scope when the row unmounts, so the registration
          goes with it.
        * **This context**, outside any activation. This is for callers that are
          not rows at all: a test that sets a service up by hand, or a mode that
          wires one directly.

        **This does not take a `Boundary`** (P6-32). `None` here resolves through
        `_ACTIVATING` to whoever is running, which is the *narrowest* correct
        answer and not "give me everything".

        **A disposed activation scope declines, and logs a warning.** Contextvars
        propagate into tasks spawned from `apply`. A background task that
        registers after its row unmounted would otherwise leak silently onto the
        seam. That is the only branch here that fails open, and §5 rule 6
        requires that it is visible where it happens.
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
        """Which scope a registration is *visible to*: the other question (P6-12).

        It is separate from `owner_for` because the two are different questions
        that used to have the same answer. A change to this answer changes what
        an agent can see (B7), and not when a registration goes away. Visibility
        is the explicit `scope`, then the layer of the running binding, then this
        context.

        **It does not take a `Boundary` either** (P6-32), for the same reason as
        `owner_for`. Both are dispatch-time *resolvers* that read `_ACTIVATING`,
        and a `Boundary` parameter is a caller **stating** something. `None`
        means "ask the binding".

        **It follows the running binding, and `isolation` makes that safe**
        (P6-26). The activation scope of a row is not isolated, so it inherits
        the isolation of the mount, and the registration of a root row still
        lands on the global layer. The scope of an agent *is* its own isolation,
        so a body that runs for an agent lands on the layer of that agent. That
        is the containment that P6-27 made structural. The last way to escape it
        was a tool body inside a contained child that installed globally visible
        tools.

        **The disposed branch fails open and deliberately does not warn.** The
        fallback to `self` means the global layer, so a body that outlived its
        scope registers something that the whole deployment can see. This
        mirrors the fail-open of `owner_for`, in the direction of B7. It is
        silent because every call site also asks `owner_for` about the same
        `scope`, and `owner_for` logs the case. Two log lines for one
        registration make the line that matters easier to miss.
        """
        if scope is not None:
            return scope
        activating = _ACTIVATING.get()
        if activating is None:
            return self
        layer = activating.layer
        return layer if layer.active else self

    def running_for(self, scope: Context | None = None) -> Running:
        """Both answers in one call, for a registry that must *record* them
        (P6-29).

        **Every site that needs both should ask one time.** `owner_for` logs when
        the activation scope that it would return has already been disposed, and
        two calls for one registration log that warning twice.

        This matters most for a registry that will invoke the body *later*, which
        must keep both answers until then. Two fields on its own record are two
        things to hold in step, and the failure is silent. One object cannot be
        half-updated.
        """
        return Running(self.owner_for(scope), self.layer_for(scope))

    @staticmethod
    def current_owner() -> Context | None:
        """Whose *lifetime* the running code belongs to, or `None` (P6-12).

        **This is the owner that a seam registration defaults to.** Cordis builds
        the right scope in `_activation_scope` and passes it to `apply`, but a
        seam only ever saw `ctx`. `ctx.commands.register(...)` therefore gave the
        registry no way to know who was calling, and every seam defaulted the
        owner to its *own* context. That made the registration of a row an effect
        of the **seam**, and the registration outlived the row that made it.

        It is a `ContextVar` and not a parameter threaded through forty
        signatures. The answer is a property of *who is running*, and not of what
        they ask for. The seam methods are also the API that rows use, and that
        API cannot gain a mandatory argument.

        **It answers "whose code is running", for both ways in which cordis runs
        code** (P6-25). An `apply` binds its activation scope, and a dispatch
        binds the scope that registered the listener. A registration therefore
        belongs to the row that made it in all three cases: the row is being
        mounted, it is handling the event of another row, or it is handling an
        event fired long after every `apply` returned.

        Contextvars propagate into tasks, so a coroutine spawned from a bound
        callback carries the binding. `Context.owner_for` declines a *disposed*
        scope and warns.

        **Introspection enumerates what is bound and what still runs unbound**,
        and the enumeration fails on anything unclassified. That list is
        therefore checked and not only believed, and this docstring does not keep
        a second copy of it.
        """
        activating = _ACTIVATING.get()
        return activating.owner if activating is not None else None

    @staticmethod
    def current_layer() -> Context | None:
        """Which scope the running code is *visible to*, or `None` (P6-29).

        This is the other half of `current_owner` above. The two are equal for
        everything that cordis itself runs: an `apply` and a listener each run as
        one scope, so both readers return the same object and the distinction
        costs nothing to carry. A body that a registry invokes separates them:
        the `execute` of a tool is the code of the registering row
        (`current_owner`), run for one agent (`current_layer`).
        `Context.layer_for` is where that answer is used. This method exists so
        that a *test* can ask the question. A body can otherwise ask it only by
        registering something and looking at where it went.
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
    def plugin_name(self) -> str:
        """The mounted plugin that this scope belongs to, or `""` outside one.

        It is a field and not a parse of `label`. The label is prose for a person
        who reads a tree, and a name that a decision is keyed on must not depend
        on how that prose is spelled."""
        return self._plugin

    @property
    def row_id(self) -> str:
        """The profile row that this scope belongs to, or `""` outside one.

        The loader stamps it beside `plugin_name`, and it is inherited the same
        way. A registry can therefore say which row gave what it holds (the
        narrowing of an assigned child, S7b) and does not need to walk up to the
        mount. A private copy mounted in the realm of a row belongs to that row:
        it exists because the row isolates."""
        return self._row

    @property
    def parent(self) -> Context | None:
        return self._parent

    @property
    def children(self) -> tuple[Context, ...]:
        """The scopes that this one will unwind, in creation order.

        It is a copy and not the list. `dispose` iterates `_children` and clears
        it, so a caller that held the live list would mutate the unwind while it
        ran. This is the read-only counterpart of `parent`, and the other half of
        what a walk of the tree needs. `scope_invariant` is the caller, and it
        exists to find a scope that this list should no longer contain.
        """
        return tuple(self._children)

    @property
    def root(self) -> Context:
        *_, last = self._chain()
        return last

    @property
    def abandoned(self) -> tuple[Abandoned, ...]:
        """The unwinds in this tree that were cut short, oldest first.

        This is a fact about the whole tree that can be read from any scope. The
        record is on the shared runtime, because the scope that it describes is
        gone by the time the entry exists. An empty tuple is the ordinary answer,
        and a healthy process keeps it.

        It is a copy, for the same reason as `children`. `_leave_tree` appends to
        the list, and it runs in a `finally` during cancellation. A caller that
        iterated the live list would walk it while it grew.
        """
        return tuple(self._runtime.abandoned)

    @property
    def abandoned_dropped(self) -> int:
        """How many ledger entries the cap discarded. See `ABANDONED_LEDGER`."""
        return self._runtime.abandoned_dropped

    def descendants(self) -> Iterator[Context]:
        """Every scope beneath this one, with this one first, and with a cycle
        guard.

        This is the one tree walk, for both `Mount.topology` and
        `scope_invariant`. The guard matters for the second caller. A tampered
        tree is exactly what a health check must report on, and a walk that hung
        there would stop `phern doctor` too.
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
        """The scope that a registration made here belongs to, or `None` for
        global.

        Every scoped registry (tools, prompt sections) files a registration under
        this key, so "who can see this" is one question with one answer. A
        plugin's activation scope is transparent and answers `None`.
        """
        return self._isolation

    def isolation_chain(self) -> list[Context | None]:
        """The isolation scopes of this context, most specific first, ending in
        `None`.

        A scoped registry walks this chain to resolve a name. The innermost scope
        that registered the name wins, and the global layer is consulted last.
        """
        chain: list[Context | None] = []
        for node in self._chain():
            key = node._isolation
            if key is not None and key not in chain:
                chain.append(key)
        chain.append(None)
        return chain

    def reaches(self, target: Context) -> bool:
        """Whether a registration made here applies to work that happens in
        `target`.

        This is the one visibility rule, shared by event dispatch and by every
        scoped registry (tools, prompt sections). A global registration reaches
        everything, and an agent-scoped one reaches that agent alone.
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
        """Claim `ctx.<key>` for `service` in the provisioning realm of this
        context.

        It returns a disposer that is registered as an effect of the calling
        scope, so the service unregisters when its plugin unloads (invariant I2).

        **It is overloaded on the key, and the typed arm checks the contract of a
        seam.** `provide(LLM, adapter)` refuses anything that is not an
        `LlmRuntime` at type-check time. The `str` arm takes an `object`, as it
        always did, for a caller that has only a name.
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
        """The service under `key`, or `default`. This is the optional read.

        It is typed through the key: `ctx.get(ATTACHMENTS)` is an
        `AttachmentStore | None`. The `if store is None` that a caller writes
        next is therefore checked, and is not only a habit.
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
        """The service under `key`, or `ServiceNotFoundError`. This is the
        required read.

        It is the typed read of a seam such as `ctx.llm`: `require(LLM)` returns
        an `LlmRuntime` and not `Any`. It is named for what it does when the
        service is absent, and the `require()` methods of the seams set the
        precedent. A reader can therefore tell a call that can get `None` (`get`)
        from one that cannot.
        """
        name = service_name(key)
        value = self._provision(name)
        if value is _MISSING:
            raise ServiceNotFoundError(f'no service "{name}" is provided at or above {self.path}')
        return value

    # -------------------------------------------------------------- effects --

    def add_disposer(self, dispose: Disposer, *, label: str = "") -> Releaser:
        """Register a teardown that is already acquired as an effect of this
        scope.

        The `Releaser` that it returns runs the teardown early, or drops it
        without running it.

        **It is refused when the scope is down, even if the unwind of the scope
        still runs**, and callers depend on that. A subagent spawned under a
        parent that went away turns this refusal into its coded
        `parent_went_away`. If the registration were accepted in the middle of an
        unwind, the child would be built into the dying scope and torn down at
        once, and the spawn would report a run. `_hold`, which every held
        `detach` goes through, is the one registration that joins an unwind that
        is under way. Its docstring gives the reason.
        """
        self._assert_active()
        return Releaser(_register(dispose, label, self._homes))

    async def effect(
        self, enter: Callable[[], MaybeAwaitable[Disposer]], *, label: str = ""
    ) -> Releaser:
        """Acquire an artifact and register its release as an effect.

        Every external artifact that an agent takes (a child process, a worktree,
        a temp path, a lock) is acquired through here. Cleanup is therefore
        structural and not remembered (§4.9, invariant I2).

        **An artifact acquired into a scope that died during the acquire is
        released here and not leaked.** `enter()` is awaited, and a `dispose()`
        on another task can land inside that await. The registration below would
        then refuse a disposer for a thing that already exists. That was the one
        window in which the "everything unwinds" of I2 depended on timing. To
        release the artifact at once is the only answer left. The effect list of
        this scope has already drained, so nothing would ever run an entry that
        is added to it now.

        **The release is shielded**, for the same reason as in `dispose`, one
        layer down. The cancellation that disposed the scope is usually still
        pending, so an unshielded release would strand exactly the artifact that
        this branch exists to rescue.

        A release that itself raises is logged, and the refusal is still raised,
        as in `_unwind`. One situation gets one exception, and the caller can act
        on neither the failure of the disposer nor this one.

        The one property that this cannot keep is LIFO order, because the effects
        registered before it have already run. An artifact whose release depends
        on one of them is therefore released after that one is gone. Nothing
        better is available when the scope is down, and it is still strictly
        better than the leak.
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

        This is used for `agent.ctx`. A registration made on the child shadows
        the global one for that agent alone, and its listeners hear only that
        agent.
        """
        self._assert_active()
        # `Context.__init__` appends to `self._children`, and `dispose` already
        # cascades over the children *before* its own effects. An `add_disposer`
        # here would therefore be a second copy of the same teardown, and one
        # that `dispose` never releases. The effect would stay on the parent with
        # `done=False` for the whole life of the parent, and keep alive every
        # child that ever finished. This was measured at 1.0 KB for each settled
        # child, with no bound. Since P6-27 the parent it pins is a live agent
        # and not the registry.
        child = Context(self, label=label, module=module or self._module, isolated=True)
        return child

    def plugin(
        self, plugin: object, config: object = None, *, transparent: bool = False, row: str = ""
    ) -> ForkScope:
        """Mount `plugin` as a child fork of this context.

        The `apply` of the fork runs only when every key in its `inject` list
        resolves from this context. The load order is expressed through service
        requirements, and never through file order.

        `transparent` belongs to the loader, for an isolating row mounted into
        its own realm (A9). See `_Dependent.transparent`. It is **not** for the
        private copies mounted beside that row, whose invisibility is the
        isolation doing its work.
        """
        self._assert_active()
        return ForkScope(self, normalize_plugin(plugin), config, transparent=transparent, row=row)

    def inject(
        self,
        keys: Sequence[str | ServiceKey[Any]],
        fn: Callable[[Context], Any],
        *,
        label: str = "inject",
    ) -> Disposer:
        """Run `fn(scope)` when every key in `keys` is available.

        `fn` receives a fresh child scope. When any key disappears, that scope is
        disposed. It is created again when the key returns.
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
        plugin: str = "",
        row: str = "",
        transparent: bool = False,
    ) -> tuple[_Dependent, Disposer]:
        """The one registration path for plugins and injections alike."""
        self._assert_active()
        dependent = _Dependent(
            ctx=self,
            transparent=transparent,
            keys=service_names(keys),
            activate=activate,
            label=label,
            module=module,
            plugin=plugin,
            row=row,
        )
        self._runtime.dependents.append(dependent)
        self._runtime.dirty = True
        return dependent, self.add_disposer(dependent.retire, label=label)

    async def reconcile(self) -> None:
        """Settle the plugin tree: activate what is ready, deactivate what is not.

        It runs to a fixpoint, because the activation of one plugin can provide
        the service that another plugin was waiting for.
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
                    # and not when the task ends. `reconcile` activates every
                    # ready dependent in one loop on one task, so a token left
                    # behind would make the registrations of the next row land
                    # on the scope of the previous row (P6-12).
                    with running(scope):
                        try:
                            await maybe_await(dependent.activate(scope))
                        except BaseException as error:
                            # Recorded *before* the unwind, so a `deactivate`
                            # that also fails cannot lose the reason that the
                            # activation failed. `ready()` is then already false
                            # when `deactivate` marks the tree dirty, and that
                            # keeps this activation from being retried.
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
        """Unwind this scope: children first, then its own effects in LIFO order.

        The order matters for anything that registers an effect *about* a child
        (the tombstone of a subagent, the bookkeeping of a supervisor), because
        such an effect runs after the scope of that child is already gone. See
        `ph_rlm.subagents._release`, which has exactly that shape.

        **The scope always leaves the tree, whatever the unwind does** (I2).
        Every `await` below can be canceled, and `CancelledError` is a
        `BaseException` that the `except Exception` of the effect loop
        deliberately does not catch. Without the `finally`, a cancellation
        partway through left this scope with `_active=False`, still in the
        `_children` of its parent, and still holding its services. The scope also
        could not be retried, because the early return of `dispose` makes a
        second `dispose()` a no-op. That state leaks everything beneath the
        scope, and a cancellation is the one path by which a live process can
        reach it.

        **The unwind is shielded, so an outside cancellation cannot strand the
        teardown that it interrupted.** This is the larger half of the same
        failure. Before the shield was here, a Ctrl-C or a task group that closed
        during the unwind abandoned every effect after the one in flight: the
        worktree was unlinked but the child was not reaped, the lease was still
        held, and nobody handled the `log.warning`. `ph.resources` had
        hand-written exactly this shield around its own `ctx.dispose()`. Every
        other caller had nothing: the agent registry, the loader,
        `ph_app.runtime.mounted`, and the recursive call below.

        **It is a deadline and not a bare shield**, for the reason that
        `daemon/server.py` gives: a root that will not unwind must not become a
        process that will not exit. A disposer that hangs therefore still loses
        the effects behind it when the budget is spent. `_leave_tree` records
        them. Now that cancellation is handled here, that is the case that the
        abandonment ledger exists for.

        **`deadline` lets a caller spend one budget over several trees.** A tree
        that is given no deadline takes `GRACE_SECONDS` from now. Nested scopes
        inherit the deadline through `_Runtime.unwind_deadline`, so one unwind is
        one budget however deep it goes. That cannot express *several roots*,
        which are several runtimes. A daemon that closes ten of them would
        otherwise spend ten budgets, so it passes one instant to each.
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
        # Set unconditionally. `reset` restores whatever was there, so True over
        # True is a no-op and `owns_budget` stays the one name for the decision.
        token = _UNWINDING.set(True)
        budget_token = _UNWIND_BUDGET.set(budget)
        try:
            if owns_budget:
                # **Entered one time for each unwind, and not for each scope.**
                # Every nested `dispose` already runs inside this scope with the
                # same deadline, so its own scope would be a second shield
                # against nothing. That was measured at +232% on a 341-scope
                # tree, which is the ordinary shape of a mounted profile that
                # goes away. `_UNWINDING` makes "nested" mean nested, and not
                # only "somewhere in this tree". See its own note.
                #
                # `CancelScope(deadline=)` and not `move_on_after(delay)`: the
                # budget is an instant, and a nested call that derived a delay
                # from it again would drift by the time that the layers above it
                # took.
                with anyio.CancelScope(deadline=budget, shield=True):
                    await self._unwind()
            else:
                await self._unwind()
        finally:
            _UNWIND_BUDGET.reset(budget_token)
            _UNWINDING.reset(token)
            if owns_budget:
                runtime.unwind_deadline = None
            self._leave_tree()

    def unwind_by(self, deadline: float) -> None:
        """Set the instant by which the unwind of this tree must finish, before
        the unwind starts.

        This is for a caller that disposes several trees and wants **one** budget
        across them. Example: a daemon closes ten roots. Each root is its own
        `Context` with its own runtime, and each is reached through an
        `AsyncExitStack`, through which the `deadline` argument of `dispose`
        cannot be passed. Without this method each root took a fresh
        `GRACE_SECONDS` behind its own shield, so a bound that the caller wrote
        as ten seconds was ten times that.

        It is idempotent, and harmless on a tree that is never disposed: only
        `drain` and `dispose` read the field.
        """
        self._runtime.unwind_deadline = deadline

    async def _unwind(self) -> None:
        """Children first, then the effects of this scope, in LIFO order.

        This is split out of `dispose` so that the shielded path and the nested
        path share one body, and neither is a copy of the other under a `with`.
        """
        for child in reversed(list(self._children)):
            await child.dispose()
        while self._effects:
            # Peeked, and popped only after it has run. With a pop first, the
            # effect that was in flight when the budget expired was invisible to
            # `_leave_tree`. That effect is the one most likely to be half-done,
            # and the one whose name a reader needs most.
            effect = self._effects[-1]
            if effect.done:
                self._effects.pop()
                continue
            effect.done = True
            try:
                await maybe_await(effect.dispose())
            except Exception:
                # A disposer that raised still ran. What it left behind is its
                # own responsibility, and the traceback is the record of it.
                log.exception("ph.cordis: effect %r failed to dispose", effect.label)
            effect.ran = True
            # **That effect, and not the last one**, off every scope that it sits
            # on. A `detach` during the await above appends its stop here even
            # though this scope is down (`_hold`). A bare `pop()` took that new
            # effect, which had not run, in place of this one.
            effect.leave()

    def _leave_tree(self) -> None:
        """Unlink from the parent and drop what this scope served. Cannot fail.

        Nothing here awaits. That property lets it run in a `finally` during
        cancellation: an `await` there would re-raise before it finished the job
        that it is there to guarantee.

        The name is not `detach`, which on this class already means the opposite:
        to run a coroutine *outside* the lifetime of the caller.

        **What was left behind is named, and never destroyed.** A non-empty
        `_effects` or `_children` here means that a cancellation cut the unwind
        short. On the ordinary path both are already empty, because every child
        unlinks itself here and the effect loop drains as it goes. Those
        disposers were *already* unreachable through `dispose` before this method
        existed, because the early return in `dispose` makes a second call a
        no-op. Nothing that would otherwise have run is therefore lost by
        arriving here. The disposers stay callable through the `Releaser` that
        `add_disposer` returned, so this method reports them and does not touch
        them.

        **It is recorded as well as logged**, on the runtime and not here. By the
        last line of this method nothing can reach the scope that could not
        finish: it is unlinked from its parent and has dropped its services. That
        is the reason why a stranded lease or worktree used to be invisible to
        everything except a log that nobody read. `scope_invariant` turns the
        ledger into an invariant that can be polled, so `phern doctor` and the
        poll of the daemon both report it, and neither needs to know what an
        effect is.
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
            # Entries on which nothing is owed go first. `outstanding` is derived,
            # so an entry whose effects were all released afterwards is a record
            # of history that the log already has. Evicting it before a live
            # entry keeps the cap for what a person can still act on.
            if len(ledger) >= ABANDONED_LEDGER:
                ledger[:] = [one for one in ledger if one.outstanding or one.unreclaimable]
            ledger.append(entry)
            # One entry is appended in each call, and the cap is enforced in each
            # call, so the ledger can be at most one entry over the cap. A
            # general slice-and-count here defended a case that cannot occur, and
            # its comment claimed the opposite of the invariant.
            if len(ledger) > ABANDONED_LEDGER:
                del ledger[0]
                self._runtime.abandoned_dropped += 1
        self._services.clear()
        # This breaks the `self -> Running -> self` cycle that the memo makes, at
        # the same time as the parent/child cycle below. A context that has been
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

        `prepend` places the listener before the existing ones. Use it only for a
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
            """Remove **that hook**, and not a hook equal to it (A8).

            `list.remove` compares with `==`, and two registrations of one
            callback on one scope are equal by field. Disposing the second
            therefore deleted the first, and the disposer of the first then
            deleted the survivor. The damage is to the order. With a third
            listener between them, the survivor moves from in front of it to
            behind it, so a listener that was registered first starts to run
            last. Order is the one thing that `on` and `prepend` promise.

            It uses `remove_identical`, as `ph.seams._registry.claim_entry` does
            to release. Identity is used and not `eq=False` on `Hook`, which
            would answer this one question by silently changing what equality and
            hashing mean for the type.
            """
            remove_identical(hooks, hook)

        return self.add_disposer(off, label=f"on({event})")

    def _hooks(self, event: str, *, scope: Context | None = None) -> list[Hook]:
        """The hook *records* that one dispatch would reach, in registration
        order.

        These are records and not bare callables, because a `Hook` carries the
        scope that registered it, and `_invoke` needs that scope: a listener runs
        as an effect of the row that wrote it (P6-25). There is deliberately no
        public route that returns bare callables. A hand-written
        `for cb in ctx.collect(e)` is a dispatch with no binding.

        **It is deliberately a list and not a generator.** `serial` returns out
        of the loop early. A generator that held a binding across its `yield`
        would then be closed by the collector on some other task, and would
        release the token in the wrong context.
        """
        hooks = self._runtime.hooks.get(event)
        if not hooks:
            return []
        target = scope if scope is not None else self
        return [hook for hook in hooks if hook.global_ or hook.ctx.reaches(target)]

    def emit(self, event: str, *args: object, scope: Context | None = None) -> None:
        """Dispatch synchronously, and ignore the return values of listeners.

        A listener that returns a coroutine is scheduled and not awaited, which
        is the behavior of cordis. The coroutine is detached work of the scope
        that registered the listener: the dispose of that scope cancels it if it
        still runs, and the dispose of the emitter ends nothing. **A listener
        that raises is logged, and the dispatch continues.** An emitted event
        records something that already happened. No listener may undo it, or keep
        the listeners after it from hearing it. A listener that must be able to
        refuse belongs on a `serial` or `waterfall` event, which is where every
        refusal in the tree already is.

        **This is the rule of the mode, and not of each producer.** It was a
        `contained=True` that each producer passed, and about half of them never
        did. A raising listener on `agent/created` left `AgentRegistry.create`
        raising over an agent that it had already registered, and one on
        `session/created` did the same to a published session.
        """
        event_registry.check(event, "emit")
        for hook in self._hooks(event, scope=scope):
            try:
                # An async listener is spawned by `_invoke`, inside its binding.
                _invoke(hook, *args, detach_for=event)
            except Exception:
                log.exception("ph.cordis: %s listener failed", event)

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

        Every listener runs even if one fails. The failures are collected and
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

        Listeners run outermost first and receive `(*args, next)`. A call to
        `next()` delegates. A return without that call vetoes the rest of the
        chain, `inner` included. That veto is how a policy plugin replaces
        built-in behavior without the knowledge of the built-in code.

        `next(*replacement)` also gives the rest of the chain different
        arguments. Cordis expects a listener to mutate a shared payload instead,
        which is not possible here: the payloads of pH are frozen values. A
        rewrite that must be explicit is also a rewrite that a reader can see.

        **The type of the chain comes from `inner`.** Every waterfall settles on
        one type: `agent/pre-step` on a `PreStepDecision`, `approval/request` on
        an `ApprovalAnswer`. That type was previously restated by hand at every
        listener, twenty of them across twelve files, and nothing checked any of
        them against the producer.

        The benefit is asymmetric, and it is worth being exact about it. The
        result of a *caller* is now checked: `T` flows from the `inner` that it
        passed. The `Next[T]` of a *listener* is not checked: `on` takes a
        `Listener`, so the annotation of a listener is a claim that nothing
        verifies. The convention is one declaration in place of twenty, and the
        checking stops at the producer.

        An `inner` must therefore declare what the *chain* resolves to, and not
        what its own default happens to be. `tools/pre-execute` returns an
        `Allow` while the chain is a four-way `PreToolDecision`, and to infer `T`
        from the default there would be wrong for every other listener.
        """
        value, _ = await self._waterfall(event, args, inner, scope)
        return value

    async def waterfall_attributed[T](
        self,
        event: str,
        *args: object,
        inner: Callable[..., Awaitable[T]],
        scope: Context | None = None,
    ) -> tuple[T, str]:
        """`waterfall`, plus the plugin whose listener produced the answer.

        **The producer is the outermost listener that returned something other
        than what its `next_` gave it**, or that never called `next_`. A listener
        that returns the answer of the rest of the chain unchanged is a
        pass-through, however much it inspected on the way. A policy row that
        only *might* veto looks like that. The result is `""` when the answer is
        from `inner` itself, or from a listener registered outside any plugin.

        This is for a producer that must know *who* decided and not only *what*.
        The agent loop counts retries by the row that granted each one (G13). A
        row that named itself on its answer made a claim that the loop could not
        check, and was a second spelling of an id that cordis already holds.
        """
        return await self._waterfall(event, args, inner, scope)

    async def _waterfall[T](
        self,
        event: str,
        args: tuple[object, ...],
        inner: Callable[..., Awaitable[T]],
        scope: Context | None,
    ) -> tuple[T, str]:
        """The chain, and who produced its answer. See `waterfall_attributed`."""
        event_registry.check(event, "waterfall")
        hooks = self._hooks(event, scope=scope)
        # Appended innermost first as frames unwind, so the last entry is the
        # producer of the answer.
        producers: list[Hook] = []

        async def frame(position: int, *state: object) -> object:
            """The chain from `position` on. It has its own cursor on purpose (A6).

            A shared `index` that every `next_` advanced made re-entry silently
            wrong. A listener that awaited `next_()` twice, which is what a
            *retry* is, got the whole inner chain the first time and only `inner`
            the second time, because the cursor was already past everything. No
            shipped listener does this today (`media-degrade`, `llm-retry` and
            `permissions-fs` all call it in exclusive branches). For that reason
            the first listener that did it would have found the defect in
            production, as a policy row that stopped running under retry.

            Recursion gives each frame its own position. A second call to `next_`
            therefore runs the rest of the chain a second time, which is what a
            retry means. A replacement that is passed down is scoped to the call
            that passed it, and does not mutate a list that its siblings also
            read.
            """
            if position >= len(hooks):
                # `inner` is the body of the *producer* and not a listener.
                # Nothing registered it, so it runs under whatever binding the
                # caller of `waterfall` already had. That is still true after
                # P6-26, which bound the bodies that a *registry* owns. The
                # `inner` of `tools/execute` is one of them, and it binds itself
                # from the inside before it calls on. The others are the
                # fallback of a seam itself, which is the code of the row and
                # needs the binding of the row. That is exactly what it inherits
                # here.
                return await inner(*state)

            handed: list[object] = []

            async def next_(*replacement: object) -> object:
                answer = await frame(position + 1, *(replacement or state))
                handed.append(answer)
                return answer

            result = await maybe_await(_invoke(hooks[position], *state, next_))
            if not handed or result is not handed[-1]:
                producers.append(hooks[position])
            return result

        # This is the one place where the type of the chain cannot be verified.
        # `on` takes a `Listener` and rows load through entry points, so the
        # checker of this repo never sees what a listener returns. `settled` is
        # where each producer checks the claim, and this cast lets them state it
        # one time and not at every listener.
        #
        # There is no `kind=` parameter here. Of the sixteen chains only four
        # could pass a bare `type[T]` and seven more a `T | None`, so it would be
        # omitted at twelve. It would then give none of the can't-forget property
        # that is its purpose. It would also let `T` be solved from two places,
        # which is what this signature exists to stop.
        value = cast("T", await frame(0, *args))
        return value, producers[-1].ctx.plugin_name if producers else ""

    def detach(self, coro: Any, *, label: str, at_dispose: AtDispose) -> None:  # noqa: ANN401
        """Run `coro` outside the lifetime of the caller. The task is held, and it
        is stopped as `at_dispose` says.

        This is for work that must outlive the call that started it, and that the
        call must not await: an async `emit` listener, or a job that runs on
        after the step that started it. The task is held, so "fire and forget"
        does not mean that it is collected in mid-flight. A failure is logged at
        its own boundary, because there is no caller left to raise into.

        **This scope stops the work, unless `leave` names another owner.** With
        `cancel` or `finish`, the stop is an effect of this scope, registered
        now. It therefore unwinds at the position that its start earned: after
        what this scope took on later, and before what it already held. `drain()`
        also waits for that work, so the teardown of a host gives it a chance to
        finish first. `leave` registers no stop and is not drained: the owner of
        a job stops it through its token. The argument is required and has no
        default, because who stops the work is the one thing that a caller of
        this method cannot leave to chance.

        With no running loop the coroutine is closed and not leaked, and the drop
        is logged. A sync host that cannot run it should not silently hold a
        never-awaited coroutine either.
        """
        try:
            task = asyncio.ensure_future(coro)
        except RuntimeError:  # pragma: no cover - no running loop
            coro.close()
            log.warning("ph.cordis: dropped detached %s (no event loop)", label)
            return
        if at_dispose == "leave":
            pool, held = self._runtime.elsewhere, None
        else:
            pool, held = self._runtime.background, self._hold(task, at_dispose, label)
        pool.add(task)

        def done(finished: asyncio.Future[Any]) -> None:
            pool.discard(finished)
            if held is not None:
                # The task is over, so nothing is owed. It is settled and not
                # released, because a release would call the stop. For `finish`,
                # that stop is a coroutine that nobody awaits.
                held.settle()
            if not finished.cancelled() and finished.exception() is not None:
                log.error("ph.cordis: detached %s failed", label, exc_info=finished.exception())

        task.add_done_callback(done)

    def _hold(
        self, task: asyncio.Future[Any], at_dispose: Literal["cancel", "finish"], label: str
    ) -> _Effect:
        """Make the stop of `task` the newest effect of this scope, whether the
        scope is active or not.

        It does not use `add_disposer`, which refuses a scope that is down. The
        listeners of a scope still hear events while its children unwind, because
        their `off` runs with the effects after them. A refusal there raised out
        of `emit` into whatever was being disposed. Appended instead, the stop
        runs in the unwind that is already under way. After an unwind that its
        deadline cut short, nothing runs the stop, and the task continues as it
        always did.
        """
        stop: Disposer
        if at_dispose == "cancel":
            stop = task.cancel
        else:

            async def finished() -> None:
                await asyncio.wait({task})

            stop = finished
        return _register(stop, label, self._homes)

    async def drain(self) -> None:
        """Await the detached work that this tree stops itself: async `emit`
        listeners, and every `detach()` except one with `leave`.

        **It does not wait for `leave` work, because the stop by the owner loses
        nothing that a wait would save.** The owner of a job stops it in
        `dispose`. A child is suspended with its inbox and resumed later. A
        refinement writes nothing and is considered again at the next turn. A
        wait here was five seconds of every teardown with a job running. Those
        seconds were spent on work that rarely finishes in them (a refinement is
        a review call and a plan call, minutes on a local model) and that is
        stopped the same way afterwards. A listener is different: the cancel of
        its scope loses what it was doing, so the wait is its one chance to
        finish.

        **It is shielded and bounded, exactly as `dispose` is and for the same
        reason.** A host that unwinds a mount calls the two as a pair, and
        `dispose` raises its own shield before its first await. A bare drain
        therefore left that pair with an unprotected first half, and a
        cancellation that arrived during it took the whole unwind with it. The
        protection belongs here and not around the call. `daemon/server.py`
        records what a scope of the caller gives against a callee that shields
        itself: the outer deadline has no effect. When one host does it by hand,
        a rule gets two spellings.

        The bound is the other half of the same decision. To wait for detached
        work is to wait for the checkpoint of somebody else. A listener that
        never reaches one would otherwise replace a skipped unwind with a
        shutdown that never finishes.

        **The budget is its own share, or what is left of a budget that somebody
        set for the whole teardown, whichever ends sooner.** A shield is opaque
        to a deadline outside it. A drain that answered only to `DRAIN_SECONDS`
        would therefore spend that much *for each tree* inside the total of a
        caller: ten roots that close under one ten-second bound could take fifty
        seconds. `unwind_by` declares that total. Because `drain` reads it here,
        the two halves of a teardown add up to the total and not to a multiple of
        it. The share still applies, so a drain cannot use up the time of an
        unwind either.

        **It uses `asyncio.wait`, so the deadline ends the wait and nothing
        else.** `detach` passes every one of these to `ensure_future`, so they
        are already running, and awaiting them in turn would not serialize
        anything (measured identical). `wait` raises nothing that a member did,
        so one listener that is canceled on its own account does not end the
        drain for all of them. The done callback that `detach` attached logs
        failures, so nothing is swallowed twice. What still runs at the deadline
        is left to its owner, whose `dispose` stops it. It used to be `gather`.
        `gather` passes the cancellation of its waiter to every member and then
        raises one made from the last of them. A member that ran an anyio task
        group (`raced`) took that cancellation for its own and returned, the
        error carried no message, and the scope below let it through to the
        caller, past the `dispose` behind it.
        """
        if not self._runtime.background:
            # There is nothing to wait for, so there is nothing to protect. The
            # scope below costs a clock read and about 3 µs to guarantee an empty
            # loop, and this runs one time for each unmount and about seventy
            # times across the suite.
            return
        if _UNWINDING.get():
            # Already inside the shield of a teardown. That is the rule of
            # `dispose`, and this is its second reader. Another scope here would
            # be a shield against nothing, at the cost that `_UNWINDING` records.
            await self.settled()
            return
        share = anyio.current_time() + DRAIN_SECONDS
        whole = self._runtime.unwind_deadline
        token = _UNWINDING.set(True)
        budget = share if whole is None else min(share, whole)
        budget_token = _UNWIND_BUDGET.set(budget)
        try:
            with anyio.CancelScope(deadline=budget, shield=True):
                await self.settled()
        finally:
            _UNWIND_BUDGET.reset(budget_token)
            _UNWINDING.reset(token)

    async def settled(self) -> None:
        """Wait until the pool that `drain` waits for is empty, with no bound and
        no shield.

        `drain` is this method under a shield and a deadline, for a teardown. A
        caller that wants the wait itself, most of all a test, bounds it as it
        chooses. A deadline that the caller sets reaches in here. The shield of
        `drain` would make that deadline have no effect, and the deadline of
        `drain` itself would end the wait without a report.
        """
        while self._runtime.background:
            settling = set(self._runtime.background)
            await asyncio.wait(settling)
            # Discarded here as well as by the done callback, so the loop ends on
            # what this call awaited and not on when `call_soon` ran the
            # callbacks.
            self._runtime.background -= settling
