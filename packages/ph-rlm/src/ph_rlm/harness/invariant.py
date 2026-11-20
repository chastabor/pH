"""Runtime invariant: `harness_state.json` equals the `harness/*` fold (P6-01, I6).

The projection is written for humans, and **nothing reads it back**. That is
exactly the arrangement that lets it drift without anyone noticing. Drift means
one of two things, and both need an alarm:

* The fold changed and the file was not rewritten. A person is then reading a
  stale account of what the agent learned.
* Something started to *write* the file. The deployment then has two carriers
  for one fact (A11), and the log has quietly stopped being the source of truth
  (I6).

This row declares the property. `HarnessService.stale_projections` is the check,
because the layout of the projection belongs to the service.

@module ph_rlm.harness.invariant
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ph.cordis import Context, plugin
from ph.seams.invariants import Invariant, contribute, contribute_fold_cache

from ..keys import HARNESS

if TYPE_CHECKING:
    from ph.session import Session


__all__ = ["apply", "stale_folds", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every written projection that no longer equals the fold behind it."""
    harness = ctx.get(HARNESS)
    return [] if harness is None else list(harness.stale_projections())


def stale_folds(ctx: Context, sessions: Iterable[Session]) -> list[str]:
    """Every cached local state that no longer equals the fold behind it.

    This is the layer under `violations`. That function compares the *file* to
    the fold. This function compares the value that the file is written from.
    It is declared here beside its sibling so that a reader finds the two I6
    claims of the harness together.
    """
    harness = ctx.get(HARNESS)
    return [] if harness is None else list(harness.stale_folds(sessions))


@plugin("harness-invariant", affects="deployment", inject=[HARNESS])
async def apply(ctx: Context, config: None) -> None:
    """Declare the harness half of I6, in a form that can be polled.

    `inject=[HARNESS]` is here because the thing that is checked is a hard
    precondition, unlike the *declaration* seam. An invariant about a projection
    that no row produces is not a weaker promise. It is a meaningless one.
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
