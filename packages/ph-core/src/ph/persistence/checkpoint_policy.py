"""`session-checkpoint-policy` — *when* the log reaches disk (A4).

Persistence decides *how* to store; this row decides *when* to force it. They
are separate plugins for the reason dsh separated them: a deployment that wants
a different durability/latency trade makes it a config change, not a fork of the
backend.

Two barriers, and a third that turns out to be one of the first two:

1. **before each model request** (`llm/stream`) — the events that motivated the
   request are durable before it is in flight;
2. **before a tool body** (`tools/execute`) — the `tool/call` is durable before
   the side effect happens, which is what makes a crashed call recoverable as
   `TOOL_OUTCOME_UNKNOWN` rather than invisible. For a nested Code Mode
   dispatch the record is `tool/code-dispatch-start` — `TOOL_DISPATCH`, whose
   declared `tools-execute` barrier is this one, placed here because it must run
   after every pre-execute gate — and it is flushed **unless the dispatched
   call has nothing to recover** (F4, S4): it changes nothing (`effect_free`),
   or the run took a restore point (`has_restore_point`) and the call's every
   effect is a file in the workspace (`ToolRuntime.restore_covers`, the rule
   `/revert` lists by, asked of this call's own arguments). So an unknown tool
   is flushed, and so is a write whose path leaves the tree. One barrier per
   cell used to cover every dispatch in it, and under the `rlm` profile
   *every* tool the model calls is nested: a
   crash mid-cell left the outer call and none of what the cell did, so
   `/revert`'s list of what a restore does not undo came back empty;
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
from ..cancel import is_canceled
from ..cordis import Context, Next, plugin
from ..keys import SESSIONS, TOOLS
from ..llm.types import GenerateOptions, StreamChunk
from ..seams.workspace import has_restore_point
from ..tools.definition import ToolExecution, ToolExecutionResult, aborted_result

__all__ = ["apply"]


@plugin("session-checkpoint-policy", affects="deployment", inject=[SESSIONS])
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
        if execution.session is None:
            return await next_()
        tools = ctx.require(TOOLS)
        if execution.parent is not None and (
            tools.effect_free(execution.name, scope=execution.scope)
            or (
                has_restore_point(execution.session, execution.root_call_id)
                and tools.restore_covers(
                    execution.name,
                    execution.arguments,
                    scope=execution.scope,
                    agent=execution.agent,
                )
            )
        ):
            # Nothing a crash could leave half done: a read changes nothing, and an
            # edit in a run that took a restore point is one the restore takes back.
            # A cell of reads and edits should not pay one fsync per call — but
            # without the restore point, nothing would take the edits back.
            return await next_()
        await ctx.require(SESSIONS).flush(execution.session)
        if is_canceled(execution.signal):
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
