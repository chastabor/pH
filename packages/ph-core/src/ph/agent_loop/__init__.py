"""`ph.agent_loop` — the ReAct driver, mounted as a row like anything else."""

from __future__ import annotations

from functools import partial

from ..agent.registry import DriverFactory
from ..cordis import Context, plugin
from ..keys import AGENTS, LLM, SESSIONS, SYSTEM_PROMPT
from ..wire import WireModel
from .driver import AgentCancelled, ReactLoopAgent

__all__ = ["AgentCancelled", "Config", "ReactLoopAgent", "apply"]


class Config(WireModel):
    """Row config for the loop.

    `max_parallel_tool_calls` is the native batch pool width — an agent-loop
    setting, as in dsh (`agentLoop.config.maxParallelToolCalls`), distinct from
    the Code Mode row's `max_parallel_sub_calls` for one program's sub-calls.
    """

    max_parallel_tool_calls: int = 10


@plugin("agent-loop", config=Config, inject=[AGENTS, LLM, SESSIONS, SYSTEM_PROMPT])
async def apply(ctx: Context, config: Config) -> None:
    """Register `ReactLoopAgent` as the driver `ctx.agents.create()` uses."""
    # Annotated, so mypy holds the driver to `AgentDriver` here — `ctx.agents` is
    # untyped, so `register_driver` would take anything.
    factory: DriverFactory = partial(
        ReactLoopAgent, max_parallel_tool_calls=config.max_parallel_tool_calls
    )
    ctx.add_disposer(ctx.require(AGENTS).register_driver(factory), label="agent-loop")
