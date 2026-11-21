"""Runtime invariant: a memoized tool view equals a freshly built one (P6-01, I6).

`ToolRuntime.view` caches one view for each isolation chain. It invalidates the
cache on a generation counter that `_changed()` increments. The cache is
therefore a **projection** in exactly the sense of I6: a stored answer that must
equal the fold that produced it. It has the failure mode of every cache. A
registration path that mutates a layer and does not increment the generation
serves an answer that was true a moment ago.

**The cost of that failure is more than a stale listing.** The view decides
which tools an agent can call. Through the shadowing rules of `_build_view`, it
also decides which registration wins a name. A stale view gives a narrowed child
the set that its parent had, or keeps a tool callable after the row that owns it
unloaded. That is a capability that outlives its scope, which is the failure of
I2 in the form of a cache.

A rebuild of every cached view costs O(cached chains * registrations). So the
invariant is polled, and not asserted on read: a check on the read path would
cost exactly the memoization that it checks.

@module ph.tools.invariant
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..keys import TOOLS
from ..seams.invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every cached view that no longer equals what the registry would build.

    The registry is asked, and the check is not computed here. See
    `ToolRuntime.stale_views` for why the check is kept with the cache that it
    checks.
    """
    return list(ctx.require(TOOLS).stale_views())


@plugin("tools-invariant", affects="deployment", inject=[TOOLS])
async def apply(ctx: Context, config: None) -> None:
    """Declare the view-cache half of I6 as a pollable invariant.

    `inject=[TOOLS]` is here for the reason that `skills-invariant` states. A
    guard on `ctx.get(TOOLS) is None` would report this invariant as *holding*
    in a deployment that has no tool registry. That is a reassuring answer where
    it is least justified. With an unmet key the row never activates, so the
    invariant is absent from the report when its subject is absent.
    """
    contribute(
        ctx,
        Invariant(
            id="tool-view-cache",
            statement="a memoized tool view equals the view the registry would build now",
            check=lambda: violations(ctx),
            order=20,
        ),
    )
