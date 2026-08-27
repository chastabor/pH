"""`ph.agent_loop` — the ReAct driver, mounted as a row like anything else."""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from .driver import AgentCancelled, ReactLoopAgent

__all__ = ["AgentCancelled", "ReactLoopAgent", "apply"]


@plugin("agent-loop", inject=["agents", "llm", "sessions", "system_prompt"])
async def apply(ctx: Context, config: Any) -> None:
    """Register `ReactLoopAgent` as the driver `ctx.agents.create()` uses."""
    ctx.add_disposer(ctx.agents.register_driver(ReactLoopAgent), label="agent-loop")
