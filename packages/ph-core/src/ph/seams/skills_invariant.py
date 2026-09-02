"""Runtime invariant: a memoized skill reach equals a freshly built one (P6-01, I6).

`SkillService.reach` memoizes per isolation chain and invalidates on a generation
counter, which is `ToolRegistry.view`'s arrangement and has `ToolRegistry.view`'s
failure mode: a registration path that mutates the table without bumping the
counter serves an answer that was true a moment ago.

**What a stale answer costs here is the capability ceiling.** `reach` is the one
place skill filters compose, so it decides which skills an agent may use. An
entry that outlived its `_changed()` keeps a skill reachable after the row owning
it unloaded — a capability outliving its scope, I2's failure wearing a cache's
clothes — or hands a narrowed child the set its parent holds, which is the
widening P4-13b calls its whole security content and the one direction the
registry is built to make impossible.

It is checked separately from the tool view rather than folded in with it,
because the two are different registries with different failure stories, and a
report naming which cache drifted is what a person can act on.

@module ph.seams.skills_invariant
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from .invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every cached reach set that no longer equals what the service would build.

    Asked of the service rather than computed here: see `SkillService.stale_reach`
    for why the check lives with the cache it checks.
    """
    return list(ctx.skills.stale_reach())


@plugin("skills-invariant", inject=["skills"])
async def apply(ctx: Context, _config: Any) -> None:
    """Declare the reach cache's half of I6, pollable.

    `inject=["skills"]` because an unmet key means the row never activates, so a
    profile that drops the skills registry drops this invariant with it. Guarding
    on `ctx.get("skills") is None` instead would report `holds` about a registry
    that is not there — "a lie shaped like reassurance", which is the one thing
    this seam's two-kinds distinction exists to prevent.
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
