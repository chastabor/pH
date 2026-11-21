"""`ph.agent_loop`: the ReAct driver, mounted as a row like anything else."""

from __future__ import annotations

from functools import partial

from ..agent.registry import DriverFactory
from ..cordis import Context, plugin
from ..keys import AGENTS, LLM, SESSIONS, SYSTEM_PROMPT
from ..wire import WireModel
from .driver import ReactLoopAgent

__all__ = ["Config", "ReactLoopAgent", "apply"]


class Config(WireModel):
    """Row config for the loop.

    `max_parallel_tool_calls` is the width of the pool for a native tool batch.
    It is a setting of the agent loop, as in dsh
    (`agentLoop.config.maxParallelToolCalls`). It is distinct from the
    `max_parallel_sub_calls` of the Code Mode row, which is for the sub-calls of
    one program.
    """

    max_parallel_tool_calls: int = 10


@plugin(
    "agent-loop",
    affects="environment",
    config=Config,
    inject=[AGENTS, LLM, SESSIONS, SYSTEM_PROMPT],
)
async def apply(ctx: Context, config: Config) -> None:
    """Register `ReactLoopAgent` as the driver `ctx.agents.create()` uses."""
    # The annotation makes mypy hold the driver to `AgentDriver` on this line.
    factory: DriverFactory = partial(
        ReactLoopAgent, max_parallel_tool_calls=config.max_parallel_tool_calls
    )
    ctx.add_disposer(ctx.require(AGENTS).register_driver(factory), label="agent-loop")
