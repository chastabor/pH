"""Runtime invariant: nothing disposed is still held by a live scope (P6-01, I2).

I2 says cleanup is *structural* — a scope goes and everything registered under it
goes with it, with no cleanup list anybody has to remember. The structure that
delivers it is the context tree: `Context.dispose` unwinds children first, then
its own effects, then removes itself from its parent's `_children`.

**The failure this polls for is the last of those three.** A disposed context
still listed by a live parent is a scope that ended and was not let go: its
services, its disposers and everything they close over stay reachable for as long
as the parent lives, which for a deployment scope is the process. It is the leak
shape that does not announce itself — nothing errors, nothing is served twice,
memory simply does not come back — and it is invisible to a test that disposes a
root and asserts on what ran, because *that* part works.

Walking the tree is O(contexts) and touches no I/O, but it is polled rather than
asserted in `dispose` for a reason worth stating: a check inside `dispose` would
run while the tree is mid-unwind, where a parent legitimately still lists a child
it is in the middle of disposing. The property is about what is true once the
unwind settles, and that is exactly what a poll sees.

**`Context.dispose` now guarantees this rather than merely usually having it**
(P6-02): leaving the parent's list happens in a `finally`, so a cancellation
partway through the unwind — the one path a live process could reach this state
by, since `CancelledError` is a `BaseException` the effect loop does not catch —
unlinks anyway.

Which means this poll no longer observes that failure *at all*, rather than
observing it less often: an abandoned scope is unlinked before anything could
walk to it, and a stranded grandchild points at a parent that no longer lists it,
so the walk cannot reach either. What is left here is tampering and a future bug
that reintroduces the state — a smaller subject, and an honest one.

**So this row carries two invariants, and the second is the one with content
now.** The state the first was written to observe is prevented rather than rare;
the state that replaced it — a cancelled unwind that left effects nobody will
run — is what `scope-teardown` reports, off the ledger `Context._leave_tree`
writes. That closes the half this module used to declare unenforced: a lease or a
worktree stranded by a cancelled teardown was reported by a `log.warning` no
shipped entry point installed a handler for, and is now a finding the daemon's
own poll records.

**Not `phern doctor`, and the distinction is the seam's own.** `phern doctor` mounts a
fresh profile in its own process — no scope disposed, so nothing abandoned —
which `ph.seams.invariants` already states as the limit of a poll. The ledger
lives on `_Runtime`, so only the process that did the abandoning can read it.
That process is the daemon, which polls itself; `phern doctor` will print this row
as holding, truthfully, about a deployment it has just built.

**Outstanding, not abandoned.** The ledger keeps the effects rather than their
names, so a disposer released afterwards through the closure `add_disposer`
handed out stops being reported. The invariant is about resources still held, not
about the history of unwinds that went badly — a deployment that recovered should
read as holding.

@module ph.seams.scope_invariant
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..text import count_of
from .invariants import Invariant, contribute

__all__ = ["abandoned", "apply", "violations"]


def violations(root: Context) -> list[str]:
    """Every retained-but-dead context reachable from `root`, and every broken link."""
    found: list[str] = []
    # `descendants` carries the cycle guard: a tampered tree is exactly what this
    # is asked to report on, and a walk that hung there would take the report down.
    for node in root.descendants():
        for child in node.children:
            if not child.active:
                found.append(f"{node.path} still holds disposed scope {child.path}")
            if child.parent is not node:
                # The other half of the same link. A child whose parent moved is
                # unreachable for disposal from the scope that lists it, so the
                # unwind would skip it while the list keeps it alive.
                found.append(f"{child.path} is listed by {node.path} but does not point back")
    return found


def abandoned(root: Context) -> list[str]:
    """Every cut-short unwind whose teardown still has not run.

    Entries with nothing owed are skipped rather than listed as healed: an
    invariant reports what is wrong *now*, and "this went badly once and was then
    put right" is history the log already carries.

    **Two kinds, because they need different things from a reader.**
    `outstanding` means somebody still holds the closure — the resource can be
    released, and naming it is actionable. `unreclaimable` means the last holder
    is gone, so nothing can ever call that disposer again; the resource is
    stranded until the process ends, and no amount of attention will change it.
    Reporting them as one sentence would tell an operator to go and do something
    about the half they cannot.

    **The dropped count rides the findings rather than being one.** It never
    decrements — nothing can prove the entries it counts were resolved after they
    fell off — so as a finding of its own it made `scope-teardown` permanently
    violated the moment a 65th abandonment happened, in a deployment that may
    have recovered completely. As a caveat on live findings it still says what it
    knows, in the report where it changes what the numbers mean.
    """
    found: list[str] = []
    for one in root.abandoned:
        if outstanding := one.outstanding:
            found.append(
                f"{one.path} left {count_of(len(outstanding), 'effect')} undisposed "
                f"after a cancelled unwind: {', '.join(outstanding)}"
            )
        if stranded := one.unreclaimable:
            found.append(
                f"{one.path} left {count_of(len(stranded), 'effect')} nothing can now "
                f"release: {', '.join(stranded)}"
            )
    if found and root.abandoned_dropped:
        found.append(
            f"and {count_of(root.abandoned_dropped, 'earlier abandonment')} "
            "fell off the ledger unread"
        )
    return found


@plugin("scope-invariant")
async def apply(ctx: Context, config: None) -> None:
    """Declare I2's structural half, pollable.

    Rooted at the *deployment* context rather than at this row's activation
    scope: the leak being looked for is one scope retaining another, and a row
    that only ever looked below itself would be blind to every scope above it —
    which is most of them.
    """
    root = ctx.root
    contribute(
        ctx,
        Invariant(
            id="scope-unwind",
            statement="no disposed scope is still held by a live one",
            check=lambda: violations(root),
            order=30,
        ),
    )
    contribute(
        ctx,
        Invariant(
            id="scope-teardown",
            statement="every effect a scope registered was disposed when it unwound",
            check=lambda: abandoned(root),
            # Beside `scope-unwind` and after it: they are the two halves of I2's
            # structural claim, and a reader meeting "nothing disposed is still
            # held" wants "and nothing held was left undisposed" next.
            order=31,
        ),
    )
