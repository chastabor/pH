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

@module ph.seams.scope_invariant
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from .invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(root: Context) -> list[str]:
    """Every retained-but-dead context reachable from `root`, and every broken link."""
    found: list[str] = []
    seen: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            # A cycle would make this walk hang rather than report, which is the
            # one outcome a health check must not have.
            continue
        seen.add(id(node))
        for child in node.children:
            if not child.active:
                found.append(f"{node.path} still holds disposed scope {child.path}")
            if child.parent is not node:
                # The other half of the same link. A child whose parent moved is
                # unreachable for disposal from the scope that lists it, so the
                # unwind would skip it while the list keeps it alive.
                found.append(f"{child.path} is listed by {node.path} but does not point back")
            pending.append(child)
    return found


@plugin("scope-invariant")
async def apply(ctx: Context, _config: Any) -> None:
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
