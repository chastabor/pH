"""Runtime invariant: a memoized skill reach equals a freshly built one (P6-01, I6).

`SkillService.reach` memoizes per isolation chain and invalidates on a
generation counter. That is the arrangement of `ToolRuntime.view`, and it has
the same failure mode: a registration path that mutates the table without
bumping the counter serves an answer that was true a moment ago.

**The cost of a stale answer here is the capability ceiling.** `reach` is the
one place where skill filters compose, so it decides which skills an agent may
use. An entry that outlived its `_changed()` keeps a skill reachable after the
row that owns it unloaded. That is a capability that outlives its scope: the
failure of I2, in the form of a cache. Or the entry hands a narrowed child the
set that its parent holds. P4-13b calls that widening its whole security
content, and it is the one direction that the registry is built to make
impossible.

This check is separate from the check of the tool view and is not folded into
it. The two are different registries with different failure stories, and a
person can act on a report that names which cache drifted.

@module ph.seams.skills_invariant
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..keys import SKILLS
from .invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every cached reach set that no longer equals what the service would build.

    The service is asked, and nothing is computed here. See
    `SkillService.stale_reach` for the reason that the check lives with the
    cache that it checks.
    """
    return list(ctx.require(SKILLS).stale_reach())


@plugin("skills-invariant", affects="deployment", inject=[SKILLS])
async def apply(ctx: Context, config: None) -> None:
    """Declare the half of I6 that belongs to the reach cache, as a pollable
    invariant.

    `inject=[SKILLS]` is here because an unmet key means that the row never
    activates. A profile that drops the skills registry therefore drops this
    invariant with it. A guard on `ctx.get(SKILLS) is None` would instead report
    `holds` about a registry that is not there. That is a false reassurance, and
    the invariants seam separates an inline invariant from a pollable one to
    prevent exactly that.
    """
    contribute(
        ctx,
        Invariant(
            id="skill-reach-cache",
            statement="a memoized skill reach equals the set the registry would build now",
            check=lambda: violations(ctx),
            order=25,
        ),
    )
