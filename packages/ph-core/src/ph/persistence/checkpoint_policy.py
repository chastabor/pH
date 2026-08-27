"""`session-checkpoint-policy` — *when* the log reaches disk.

Persistence decides *how* to store; this row decides *when* to force it. They
are separate plugins for the reason dsh separated them: a deployment that wants
a different durability/latency trade makes it a config change, not a fork of the
backend.

Phase 0 places one barrier — before every model request. Phase 1 (A4) adds the
other two, before top-level tool dispatch and at step end, and the gate is a
crash injected after each barrier showing that everything before it survived.

@module ph.persistence.checkpoint_policy
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..cordis import Context, plugin
from ..llm.types import GenerateOptions

__all__ = ["apply"]


@plugin("session-checkpoint-policy", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Flush before each model request."""

    async def before_request(request: GenerateOptions, next_: Callable[[], Any]) -> Any:
        if request.session_id is not None:
            session = ctx.sessions.get(request.session_id)
            if session is not None:
                # Awaited, not scheduled: the point of the barrier is that the
                # request cannot be in flight while the events that motivated it
                # are still in a buffer.
                await ctx.sessions.flush(session)
        return await next_()

    ctx.on("llm/stream", before_request)
