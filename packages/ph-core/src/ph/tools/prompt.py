"""`tools-prompt` — the bridge that puts tool schemas in the request.

A separate row from `ph.tools` on purpose: the registry does not depend on there
being a system prompt, and a headless caller assembling its own request can use
the registry without one. The bridge is what joins them, and it passes the
**target scope** through, so what the model is offered is a per-agent answer
(B7): a restriction or a scoped registration changes it.

@module ph.tools.prompt
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from ..llm.types import ToolSchema

__all__ = ["apply"]


@plugin("tools-prompt", inject=["tools", "system_prompt"])
async def apply(ctx: Context, config: Any) -> None:
    """Contribute the visible tool schemas for whichever scope is assembling."""

    def schemas(scope: Context) -> list[ToolSchema]:
        visible: list[ToolSchema] = ctx.tools.schemas(scope=scope)
        return visible

    ctx.system_prompt.tools(schemas)
