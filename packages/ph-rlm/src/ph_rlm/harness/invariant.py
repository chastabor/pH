"""Runtime invariant: `harness_state.json` equals the `harness/*` fold (P6-01, I6).

The projection is written for humans and **nothing reads it back** — which is
exactly the arrangement that lets it drift without anyone noticing. Drift means
one of two things, and both are worth an alarm: either the fold changed and the
file was not rewritten, so a person is reading a stale account of what the agent
learned; or something started *writing* the file, in which case the deployment
now has two carriers for one fact (A11) and the log has quietly stopped being the
source of truth (I6).

This row declares the property; `HarnessService.stale_projections` is the check,
because the projection's layout is the service's own.

@module ph_rlm.harness.invariant
"""

from __future__ import annotations

from typing import Any

from ph.cordis import Context, plugin
from ph.seams.invariants import Invariant, contribute, contribute_fold_cache

from ..keys import HARNESS

__all__ = ["apply", "stale_folds", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every written projection that no longer equals the fold behind it."""
    harness = ctx.get(HARNESS)
    return [] if harness is None else list(harness.stale_projections())


def stale_folds(ctx: Context, sessions: Any) -> list[str]:
    """Every cached local state that no longer equals the fold behind it.

    The layer under `violations`: that one compares the *file* to the fold, this
    one the value the file is written from. Declared here beside its sibling so a
    reader meets the harness's two I6 claims together.
    """
    harness = ctx.get(HARNESS)
    return [] if harness is None else list(harness.stale_folds(sessions))


@plugin("harness-invariant", inject=[HARNESS])
async def apply(ctx: Context, config: None) -> None:
    """Declare I6's harness half, pollable.

    `inject=[HARNESS]` because, unlike the *declaration* seam, the thing being
    checked is a hard precondition: an invariant about a projection no row
    produces is not a weaker promise, it is a meaningless one.
    """
    contribute(
        ctx,
        Invariant(
            id="harness-projection",
            statement="harness_state.json equals the harness/refined fold",
            check=lambda: violations(ctx),
            order=40,
        ),
    )
    contribute_fold_cache(
        ctx,
        id="harness-fold-cache",
        subject="harness state",
        stale=lambda sessions: stale_folds(ctx, sessions),
    )
