"""`tools-prompt`: the bridge that puts the tool schemas in the request.

This row is separate from `ph.tools` on purpose. The registry does not depend on
a system prompt, and a headless caller that assembles its own request can use
the registry without one. The bridge joins the two. It passes the **target
scope** through, so what the model is offered is a per-agent answer (B7): a
restriction or a scoped registration changes it.

@module ph.tools.prompt
"""

from __future__ import annotations

from ..cordis import Context, plugin
from ..keys import SYSTEM_PROMPT, TOOLS
from ..llm.types import ToolSchema

__all__ = ["apply"]


@plugin("tools-prompt", affects="environment", inject=[TOOLS, SYSTEM_PROMPT])
async def apply(ctx: Context, config: None) -> None:
    """Contribute the visible tool schemas for whichever scope is assembling."""

    def schemas(scope: Context) -> list[ToolSchema]:
        visible: list[ToolSchema] = ctx.require(TOOLS).schemas(scope=scope)
        return visible

    ctx.require(SYSTEM_PROMPT).tools(schemas)
