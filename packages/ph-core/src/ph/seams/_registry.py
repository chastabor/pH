"""Who owns a seam registration, and the one "release only if still mine" rule.

Every seam lets a plugin claim a slot (the sandbox provider, the code runtime) or
a key in a table (a command, a skill, a renderer), and returns a disposer. The
release step was written by hand six times, and the copies drifted: some checked
identity before they removed, and some did not. A disposer that removes whatever
*currently* occupies the slot would tear down a successor that was registered
after its own owner was replaced. So identity is checked here, in one place.

**A `Running` may be passed wherever a `Context` may** (P6-29). Both have
`add_disposer(release, label=)`. The pair releases when *either* of its two
scopes ends. A registry whose container is keyed by the visibility scope needs
that, and a registry that is keyed only by the owner does not. To opt in, pass
`by` and not `by.owner`. Nothing else changes.

**`Context.owner_for` says who owns a registration, and this module does not**
(P6-12). The function was in this module for a short time, and it does not
belong here. This module is about *tables*: release only what is still mine.
Ownership is about the lifetimes of a `Context` and touches nothing else. At that
time the function had three consumers, and two of them (`ph.tools.registry`,
`ph.system_prompt.assembly`) are not seams at all. With the function here, those
two imported an underscore module of a sibling package, and a latent import
cycle was one re-export away.

@module ph.seams._registry
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..cordis import Context

from ..cordis import Context, Disposer, Running, ServiceKey, remove_identical

__all__ = ["Registers", "claim_entry", "claim_key", "claim_slot", "contribute_item"]


def claim_key[T](
    owner: Context | Running,
    table: dict[str, T],
    key: str,
    value: T,
    *,
    label: str,
    then: Callable[[], None] | None = None,
) -> Disposer:
    """Put `value` under `key`. The disposer removes it only while it is still there.

    The function is generic, so the value and the table that it lands in must
    agree. `Any` here was the last hop of a chain that is otherwise checked end
    to end. A registry threads its element type through its own bucket and its
    `_register`. It then passed both to a helper that took either from anywhere.
    The type parameter costs nothing at a call site, because `T` is inferred
    from the table.

    `then` runs **after** the removal and **inside** the disposer that the owner
    holds. No other placement works. A registry that announces its own changes
    must also announce the change that an unwinding scope makes, and a scope
    unwinds by running what `add_disposer` was given. If a caller composes the
    pair outside, by wrapping the returned disposer, the scope holds the bare
    removal. The unload of a row then took its command away and told nobody.
    `ToolRuntime._claim` already has this shape for `tools/change`.
    """
    if key in table:
        raise ValueError(f"{label}: {key!r} is already registered")

    def release() -> None:
        if table.get(key) is value:
            del table[key]
        if then is not None:
            then()

    # The teardown is registered before the entry is written, here and in the
    # two functions below. An owner that is down refuses the teardown. If the
    # entry were written first, it would stay with nothing to remove it.
    releaser = owner.add_disposer(release, label=f"{label}({key})")
    table[key] = value
    return releaser


def claim_entry[T](owner: Context | Running, entries: list[T], value: T, *, label: str) -> Disposer:
    """Append `value`. The disposer removes **that object**, not one equal to it.

    The function is generic for the same reason as `claim_key` above. The type
    that a registry threaded through its bucket was no longer checked at this
    boundary.

    `list.remove` compares with `==`. For the closures and objects that the
    other contribution lists in this package hold, `==` is identity, so those
    lists got the right answer by accident. A registry of *values* does not. If
    two rows contribute an equal entry, the disposer of one would take the entry
    of the other. That is the failure that this module exists to prevent, in a
    list and not in a table.
    """
    releaser = owner.add_disposer(lambda: remove_identical(entries, value), label=label)
    entries.append(value)
    return releaser


def claim_slot(
    by: Running,
    holder: object,
    attribute: str,
    value: object,
    *,
    label: str,
) -> Disposer:
    """Set `holder.<attribute>`. The disposer clears it only while it still holds `value`.

    **It takes the pair and not the owner** (P6-29), and holds the pair in
    `<attribute>_by` for exactly as long as the slot itself. Most slots hold a
    provider, and a provider is a body that the seam invokes *later*. To enter
    the right binding at that time, the seam must have kept who registered the
    provider and where the registration landed. The record is set and cleared
    with the slot in one claim. The alternative was five seams that each wrote
    the same four lines around this call, which gave five chances for the record
    to outlive the value that it describes.

    **The name is derived and is not a parameter.** That is the difference
    between a rule and a reminder. When the name was a parameter, all five call
    sites passed `f"{attribute}_by"`, so the argument carried no information.
    The argument was also optional, so "forgot to record" was a silent state:
    `running(None)` binds nothing, and the P6-30 gate classifies the
    *registration method*, which is present in both cases. With a derived name,
    a new slot that omits the field fails at registration. Every one of these
    holders is `slots=True`, and `setattr` for an undeclared name raises
    immediately.

    This does not move the ownership decision here, which the module docstring
    above correctly refuses. The caller still decides, through
    `Context.running_for`, exactly as it decided through `owner_for` before.
    This function stores what it was told.
    """
    if getattr(holder, attribute) is not None:
        raise RuntimeError(f"{label}: a provider is already registered")
    record = f"{attribute}_by"

    def release() -> None:
        if getattr(holder, attribute) is value:
            setattr(holder, attribute, None)
            setattr(holder, record, None)

    releaser = by.owner.add_disposer(release, label=label)
    try:
        # The record is set first, so a holder that omits the field fails with
        # nothing set.
        setattr(holder, record, by)
        setattr(holder, attribute, value)
    except BaseException:
        releaser.drop()
        raise
    return releaser


class Registers[T](Protocol):
    """A service whose registration method is `register(item, *, scope=…)`.

    Ten services in this tree have that method: `diagnostics`, `invariants`,
    `commands`, `skills`, `tui_status`, `tui_screens`, `compaction`,
    `code_runtime`, `ph.tools` and `ph_text_index`. All ten have the signature
    `register(self, item, *, scope) -> Disposer`. Every seam already obeyed this
    Protocol before it had a name, and such a Protocol is worth a name. With
    the name, `contribute_item` reads the method from the key, and no caller has
    to restate it.

    `item` is positional-only, because the ten give it different *names*
    (`diagnostic`, `invariant`, `definition`, `skill`, `status_field`, `screen`,
    `engine`, `provider`) and no caller passes it by keyword.
    """

    def register(self, item: T, /, *, scope: Context | None = None) -> Disposer: ...


def contribute_item[T](ctx: Context, key: ServiceKey[Registers[T]], item: T, *, label: str) -> None:
    """Register `item` on the service at `key` when that service exists.

    This is the common case. It has the shape of `contribute_via` with the
    closure removed. Seven callers had written
    `lambda service, scope: service.register(item, scope=scope)` by hand, which
    restates the registration method at each call site. One of them then needed
    a *binder* exported from `ph.seams.skills`. A lambda that is built in a loop
    reads its free variable when the contribution finally runs, so every skill
    registered the last one. Passing the item makes that impossible: the item is
    bound at the call, so no free variable can be read late.

    `contribute_via` stays for a method that is not named `register`.
    `ctx.subagents` and `ctx.tools` both name theirs `guard`. That is one call
    site today.
    """
    contribute_via(
        ctx, key, lambda service, scope: service.register(item, scope=scope), label=label
    )


def contribute_via[S](
    ctx: Context, key: ServiceKey[S], contribute: Callable[[S, Context], object], *, label: str
) -> None:
    """Run `contribute(service, scope)` when the service at `key` exists, whether
    it is mounted yet or not.

    **It goes through `ctx.inject` and not through a `ctx.get` at `apply` time.**
    That is the difference between a contribution that works and one that works
    only if the rows happen to be in the right order. A `ctx.get` reads what has
    been provided *so far*. A contributor that is mounted above the seam
    therefore finds nothing, contributes nothing, and reports that nowhere.
    `inject` waits for the key through the reconcile fixpoint of the loader, so
    the two rows may sit in either order. `base.yaml` promises that about every
    other row in it.

    The dependency is still optional. A hard `inject=[key]` on the contributing
    row would make the *report* a precondition for the thing that is reported
    on.

    The unwind needs no extra work. The contribution is given a child scope that
    disposes when the key goes away or the row unloads. The contribution
    therefore leaves with the service that answers it, and no caller has to
    remember `scope=`.

    **The caller passes the call, not the name of a method.** This function used
    to take an `item` and a `method: str`, and it dispatched
    `getattr(service, method)(item)`. No checker follows that. A renamed
    `SkillService.register` therefore broke at mount for all eight callers,
    which is the failure that P8-05 exists to end one layer down. `S` comes from
    the key, so `contribute(skills, scope)` is an ordinary call that mypy
    resolves. The `method` parameter existed only because the method of
    `ctx.subagents` is named `guard`, and it is gone with the string.

    The function is here beside `claim_key` for the same reason that `claim_key`
    is here. This rule was written by hand in two seams, each with a paragraph
    of prose that explained why. A third seam that wanted the rule would copy
    whichever version it read first. A fourth then wrote it in a *downstream
    package*, where the next reader finds the copy before the rule.
    """

    def register(scope: Context) -> None:
        contribute(scope.require(key), scope)

    ctx.inject([key], register, label=label)
