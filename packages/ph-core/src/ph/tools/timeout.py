"""`tools-timeout` — the `tools/execute` wrapper that makes `timeout_ms` true.

A `ToolDefinition` may declare a cooperative budget. Declaring it is a promise
the tool forwards `run.signal` and can reach quiescence when asked; this row is
what asks. A budget with nothing enforcing it would be worse than none — a tool
author who set `timeout_ms=5_000` and got no timeout has been told something
false by the type.

Its own row rather than registry code, as in dsh (`dsh-tool-call-timeout-policy`):
a deployment that wants a different timeout policy swaps the row.

@module ph.tools.timeout
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio

from ..cordis import Context, plugin
from .definition import ToolExecution, error_result

__all__ = ["apply"]


@plugin("tools-timeout", inject=["tools"])
async def apply(ctx: Context, config: Any) -> None:
    """Bound every dispatch whose tool declared `timeout_ms`."""

    async def bounded(execution: ToolExecution, next_: Callable[..., Any]) -> Any:
        definition = ctx.tools.get(execution.name, scope=execution.scope)
        budget = getattr(definition, "timeout_ms", None)
        if budget is None:
            return await next_()
        # The body observes the same cancellation the pipeline does, narrowed:
        # a child token can be cancelled by the timeout or by anything above it,
        # but cannot outlive its parent's cancellation.
        child = execution.signal.child() if execution.signal is not None else None
        with anyio.move_on_after(budget / 1000) as scope:
            result = await next_()
        if scope.cancelled_caught:
            if child is not None:
                child.cancel("timeout")
            return error_result(
                f'tool "{execution.name}" exceeded its {budget} ms budget',
                {"name": "Timeout", "code": "TIMEOUT"},
            )
        return result

    ctx.on("tools/execute", bounded)
