"""Runtime invariant: a memoized tool view equals a freshly built one (P6-01, I6).

`ToolRegistry.view` caches per isolation chain and invalidates on a generation
counter that `_changed()` bumps. That makes the cache a **projection** in exactly
I6's sense — a stored answer that must equal the fold that produced it — and it
has the failure mode every cache has: a registration path that mutates a layer
without bumping the generation serves an answer that was true a moment ago.

**What that costs is not a stale listing.** The view is what decides which tools
an agent can call and, through `_build_view`'s shadowing rules, which
registration wins a name. A stale view hands a narrowed child the set its parent
had, or keeps a tool callable after the row that owns it unloaded — a capability
outliving its scope, which is I2's failure wearing a cache's clothes.

Rebuilding every cached view is O(cached chains * registrations), which is why
this is polled rather than asserted on read: checking on the read path would
cost exactly the memoization it is checking.

@module ph.tools.invariant
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from ..seams.invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every cached view that no longer equals what the registry would build.

    Asked of the registry rather than computed here: see `ToolRegistry.stale_views`
    for why the check lives with the cache it checks.
    """
    registry = ctx.get("tools")
    return [] if registry is None else list(registry.stale_views())


@plugin("tools-invariant")
async def apply(ctx: Context, _config: Any) -> None:
    """Declare the view cache's half of I6, pollable."""
    contribute(
        ctx,
        Invariant(
            id="tool-view-cache",
            statement="a memoized tool view equals the view the registry would build now",
            check=lambda: violations(ctx),
            order=20,
        ),
    )
