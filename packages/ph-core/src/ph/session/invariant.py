"""Runtime invariant: the projections of every session equal their folds (P6-01, I6).

`Session` keeps three incremental projections of its log: the events snapshot,
the surface, and the derived message history. `Session.stale` compares each to
the fold that produced it. This row declares the property. The check is with
the caches that it checks, for the reason that `ToolRuntime.stale_views` gives.

The fold is O(events) per session. That is why the check is polled and not run
on append, where it would make the cost of the log quadratic in its length.

@module ph.session.invariant
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..keys import SESSIONS
from ..seams.invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every live session whose projections disagree with its log."""
    return [detail for session in ctx.require(SESSIONS).list() for detail in session.stale()]


@plugin("session-invariant", affects="deployment", inject=[SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Declare the session half of I6, pollable.

    `inject=[SESSIONS]` is for the reason that `skills-invariant` states. A
    deployment with no session store has nothing for this to be true *of*. To
    report the invariant as holding would overstate the one thing that the log
    is supposed to be the source of.
    """
    contribute(
        ctx,
        Invariant(
            id="session-log",
            statement="every projection a session keeps of its log equals the fold of that log",
            check=lambda: violations(ctx),
            order=10,
        ),
    )
