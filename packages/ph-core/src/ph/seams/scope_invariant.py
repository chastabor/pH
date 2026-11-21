"""Runtime invariant: nothing disposed is still held by a live scope (P6-01, I2).

I2 says that cleanup is *structural*. A scope goes and everything registered
under it goes with it, and nobody has to remember a cleanup list. The structure
that delivers this is the context tree. `Context.dispose` unwinds the children
first, then its own effects, and then removes itself from the `_children` of its
parent.

**This poll looks for a failure of the last of those three steps.** A disposed
context that a live parent still lists is a scope that ended and was not let go.
Its services, its disposers and everything that they close over stay reachable
for as long as the parent lives. For a deployment scope, that is the life of the
process. This kind of leak does not announce itself: nothing raises an error,
nothing is served twice, and the memory does not come back. A test that disposes
a root and asserts on what ran cannot see it, because *that* part works.

The walk of the tree is O(contexts) and does no I/O. It is still a poll and not
an assertion in `dispose`, for this reason. A check inside `dispose` would run
while the tree is in the middle of the unwind, where a parent legitimately still
lists a child that it is disposing. The property is about what is true when the
unwind has settled, and that is exactly what a poll sees.

**`Context.dispose` now guarantees this property** (P6-02). Before, the property
held only in most cases. The scope leaves the list of its parent in a `finally`,
so a cancellation partway through the unwind unlinks the scope anyway. A
cancellation was the one path by which a live process could reach this state,
because `CancelledError` is a `BaseException` that the effect loop does not
catch.

So this poll no longer observes that failure *at all*. It does not merely
observe it less often. An abandoned scope is unlinked before anything can walk
to it. A stranded grandchild points at a parent that no longer lists it. The
walk can reach neither. What is left for this check is tampering, and a future
bug that brings the state back. That is a smaller subject, and an accurate one.

**So this row carries two invariants, and the second is the one with content
now.** The state that the first was written to observe is prevented, and not
merely rare. The state that replaced it is a canceled unwind that left effects
that nobody will run. `scope-teardown` reports that state from the ledger that
`Context._leave_tree` writes. This closes the half that this module used to
declare as not enforced. A lease or a worktree that a canceled teardown stranded
was reported by a `log.warning`, and no shipped entry point installed a handler
for it. It is now a finding that the poll of the daemon records.

**`phern doctor` cannot report this finding, and the seam already says why.**
`phern doctor` mounts a fresh profile in its own process. No scope was disposed
there, so nothing was abandoned. `ph.seams.invariants` already states that as
the limit of a poll. The ledger is on `_Runtime`, so only the process that did
the abandoning can read it. That process is the daemon, which polls itself.
`phern doctor` prints this row as holding, and that is true of the deployment
that it has just built.

**Outstanding, not abandoned.** The ledger watches the effects themselves and
does not keep only their names. A teardown that is released or dropped
afterwards, through the `Releaser` that `add_disposer` returned, therefore stops
being reported. The invariant is about resources that are still held, and not
about the history of unwinds that went badly. A deployment that recovered must
read as holding.

@module ph.seams.scope_invariant
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..text import count_of
from .invariants import Invariant, contribute

__all__ = ["abandoned", "apply", "violations"]


def violations(root: Context) -> list[str]:
    """Every disposed context that is still held and reachable from `root`, and
    every broken link."""
    found: list[str] = []
    # `descendants` carries the cycle guard. A tampered tree is exactly what this
    # function must report on, and a walk that hung there would stop the report.
    for node in root.descendants():
        for child in node.children:
            if not child.active:
                found.append(f"{node.path} still holds disposed scope {child.path}")
            if child.parent is not node:
                # This is the other half of the same link. A child whose parent
                # moved cannot be reached for disposal from the scope that lists
                # it. The unwind would skip the child while the list keeps it
                # alive.
                found.append(f"{child.path} is listed by {node.path} but does not point back")
    return found


def abandoned(root: Context) -> list[str]:
    """Every unwind that was cut short and whose teardown has still not run.

    An entry on which nothing is owed is skipped, and is not listed as healed.
    An invariant reports what is wrong *now*. "This went badly one time and was
    then put right" is history that the log already carries.

    **There are two kinds, because they ask different things of a reader.**
    `outstanding` means that somebody still holds the releaser. The resource can
    be released, so a reader can act on its name. `unreclaimable` means that
    nothing can ever run that disposer again: the last holder is gone, or the
    unwind claimed the effect and did not finish it. The resource is stranded
    until the process ends, and no amount of attention changes that. One
    sentence for both kinds would tell an operator to act on the half that
    nobody can act on.

    **The dropped count is appended to the findings and is not a finding
    itself.** The count never decreases, because nothing can prove that the
    entries that it counts were resolved after they left the ledger. As a
    finding of its own, it made `scope-teardown` permanently violated from the
    65th abandonment on, in a deployment that may have recovered completely. As
    a caveat on live findings, it still says what it knows, in the report where
    it changes what the numbers mean.
    """
    found: list[str] = []
    for one in root.abandoned:
        if outstanding := one.outstanding:
            found.append(
                f"{one.path} left {count_of(len(outstanding), 'effect')} undisposed "
                f"after a canceled unwind: {', '.join(outstanding)}"
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


@plugin("scope-invariant", affects="deployment")
async def apply(ctx: Context, config: None) -> None:
    """Declare the structural half of I2, as pollable invariants.

    The walk is rooted at the *deployment* context and not at the activation
    scope of this row. The leak that it looks for is one scope that retains
    another. A row that looked only below itself would not see any scope above
    it, and most scopes are above it.
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
            # This is beside `scope-unwind` and after it. They are the two halves
            # of the structural claim of I2. A reader who meets "nothing disposed
            # is still held" wants "and nothing held was left undisposed" next.
            order=31,
        ),
    )
