"""`session-checkpoint-policy` — *when* the log reaches disk (A4).

Persistence decides *how* to store; this row decides *when* to force it. They
are separate plugins for the reason dsh separated them: a deployment that wants
a different durability/latency trade makes it a config change, not a fork of the
backend.

Two barriers, and a third that turns out to be one of the first two:

1. **before each model request** (`llm/stream`) — the events that motivated the
   request are durable before it is in flight;
2. **before a top-level tool body** (`tools/execute`, `parent is None`) — the
   `tool/call` is durable before the side effect happens, which is what makes a
   crashed call recoverable as `TOOL_OUTCOME_UNKNOWN` rather than invisible.
   A nested Code Mode dispatch reuses the outer call's checkpoint;
3. **at step end** — on the request path this *is* barrier 1: the next
   request's flush covers everything the previous step committed, and a second
   fsync microseconds earlier would buy nothing. The only step end barrier 1
   never reaches is a pre-step **reject** (no request follows), so that is the
   one case flushed here.

Barriers 1 and 2 are **fail-closed**: if the flush raises, the adapter and the
tool body are not invoked. A side effect whose record could not be written is
worse than a side effect that did not happen.

@module ph.persistence.checkpoint_policy
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..agent.types import PreStepDecision, PreStepRequest
from ..cancel import is_cancelled
from ..cordis import Context, Next, plugin
from ..keys import SESSIONS
from ..llm.types import GenerateOptions, StreamChunk
from ..tools.definition import ToolExecution, ToolExecutionResult, aborted_result

__all__ = ["apply"]


@plugin("session-checkpoint-policy", inject=[SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Install the semantic checkpoints."""

    async def before_request(
        request: GenerateOptions,
        next_: Next[AsyncIterator[StreamChunk]],
    ) -> AsyncIterator[StreamChunk]:
        if request.session_id is not None:
            session = ctx.require(SESSIONS).get(request.session_id)
            if session is not None:
                # Awaited, not scheduled: the point of a barrier is that the
                # request cannot be in flight while the events that motivated it
                # are still in a buffer.
                await ctx.require(SESSIONS).flush(session)
        return await next_()

    async def before_tool_body(
        execution: ToolExecution, next_: Next[ToolExecutionResult]
    ) -> ToolExecutionResult:
        if execution.session is None or execution.parent is not None:
            # A nested dispatch is already covered by its outer call's barrier.
            return await next_()
        await ctx.require(SESSIONS).flush(execution.session)
        if is_cancelled(execution.signal):
            return aborted_result(started=False)
        return await next_()

    async def after_pre_step(
        request: PreStepRequest,
        next_: Next[PreStepDecision],
    ) -> PreStepDecision:
        decision = await next_()
        if decision.kind == "reject":
            # No request will follow to flush the previous step's results.
            await ctx.require(SESSIONS).flush(request.session)
        return decision

    ctx.on("llm/stream", before_request)
    ctx.on("tools/execute", before_tool_body)
    ctx.on("agent/pre-step", after_pre_step)
