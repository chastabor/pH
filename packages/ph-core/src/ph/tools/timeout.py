"""`tools-timeout`: the `tools/execute` wrapper that enforces `timeout_ms`.

A `ToolDefinition` may declare a cooperative budget. By declaring it, the tool
promises that it forwards `run.signal` and can come to a stop when asked. This
row is what asks. A budget that nothing enforced would be worse than no budget:
a tool author who set `timeout_ms=5_000` and got no timeout was told something
false by the type.

This is a row of its own and not registry code, as in dsh
(`dsh-tool-call-timeout-policy`). A deployment that wants a different timeout
policy swaps the row.

@module ph.tools.timeout
"""

from __future__ import annotations

from dataclasses import replace

import anyio

from ..cordis import Context, Next, plugin
from ..keys import TOOLS
from .definition import ToolExecution, ToolExecutionResult, error_result
from .errors import TOOL_TIMEOUT

__all__ = ["apply"]


@plugin("tools-timeout", affects="environment", inject=[TOOLS])
async def apply(ctx: Context, config: None) -> None:
    """Bound every dispatch whose tool declared `timeout_ms`."""

    async def bounded(
        execution: ToolExecution,
        next_: Next[ToolExecutionResult],
    ) -> ToolExecutionResult:
        definition = ctx.require(TOOLS).get(execution.name, scope=execution.scope)
        # There are two different `None`s here: a tool that this scope cannot
        # see, and a tool that declared no budget. The `getattr` that this code
        # replaced gave one answer for both. Unbounded is the right answer to
        # each. But only the second is a decision. The first is a dispatch that
        # is about to fail anyway.
        if definition is None or definition.timeout_ms is None:
            return await next_()
        budget = definition.timeout_ms
        # The body observes the same cancellation as the pipeline, narrowed. The
        # timeout or anything above it can cancel a child token, but a child
        # token cannot outlive the cancellation of its parent.
        #
        # **And the body is given the child token** (C6). The child used to be
        # created and canceled and never passed on. The contract of
        # `ToolExecution.signal` says that a `tools/execute` wrapper may replace
        # the signal for its delegated lifetime, and that replacement did not
        # happen. A cooperative tool that polled `raise_if_canceled` saw the
        # *parent*, which the timeout never touches, so cancellation did
        # nothing. A thread-bound body ran to completion, and the scope above it
        # discarded its result. The budget then bounded the wait of the caller
        # and nothing else.
        child = execution.signal.child() if execution.signal is not None else None
        with anyio.move_on_after(budget / 1000) as scope:
            # `replace` is unconditional. With no parent there is no child, and
            # `signal=None` is the value of the field in both cases.
            result = await next_(replace(execution, signal=child))
        if scope.cancelled_caught:
            if child is not None:
                child.cancel("timeout")
            return error_result(
                f'tool "{execution.name}" exceeded its {budget} ms budget',
                {"name": "Timeout", "code": TOOL_TIMEOUT},
            )
        return result

    ctx.on("tools/execute", bounded)
