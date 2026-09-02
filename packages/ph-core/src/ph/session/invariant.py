"""Runtime invariant: every session's projections equal their folds (P6-01, I6).

`Session` keeps three incremental projections of its log — the events snapshot,
the surface, and the derived message history — and `Session.stale` is where each
is compared to the fold that produced it. This row declares the property; the
check lives with the caches it checks, for the reason `ToolRuntime.stale_views`
gives.

The fold is O(events) per session, which is why this is polled rather than run
on append: doing it there would make the log's cost quadratic in its own length.

@module ph.session.invariant
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from ..seams.invariants import Invariant, contribute

__all__ = ["apply", "violations"]


def violations(ctx: Context) -> list[str]:
    """Every live session whose projections disagree with its log."""
    sessions = ctx.get("sessions")
    if sessions is None:
        return []
    return [detail for session in sessions.list() for detail in session.stale()]


@plugin("session-invariant")
async def apply(ctx: Context, _config: Any) -> None:
    """Declare the session half of I6, pollable."""
    contribute(
        ctx,
        Invariant(
            id="session-log",
            statement="every projection a session keeps of its log equals the fold of that log",
            check=lambda: violations(ctx),
            order=10,
        ),
    )
