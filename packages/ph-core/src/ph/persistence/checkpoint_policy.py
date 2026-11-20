"""`session-checkpoint-policy`: *when* the log reaches disk (A4).

Persistence decides *how* to store. This row decides *when* to force a write.
They are separate plugins for the reason that dsh separated them: a deployment
that wants a different trade between durability and latency changes a config,
and does not fork the backend.

There are four barriers. On the request path the third is barrier 1, and the
fourth runs at the end:

1. **Before each model request** (`llm/send`). The events that motivated the
   request are durable before it is in flight. The barrier is on `llm/send` and
   not around `llm/stream`. A stream listener registered after this one wraps
   inside it, so what media-degrade appended went out with the request
   unflushed.
2. **Before a tool body** (`tools/body`). The `tool/call` is durable before the
   side effect happens. That makes a crashed call recoverable as
   `TOOL_OUTCOME_UNKNOWN` and not invisible. The barrier is on `tools/body` and
   not around `tools/execute`, for the reason of barrier 1: a wrapper
   registered after this one would run inside it, and what it appended would
   reach the body unflushed.

   For a nested Code Mode dispatch the record is `tool/code-dispatch-start`.
   Its kind is `TOOL_DISPATCH`, whose declared `tools-body` barrier is this
   one. The record is flushed **unless the dispatched call has nothing to
   recover** (F4, S4). That is the case when the call changes nothing
   (`effect_free`), or when the run took a restore point (`has_restore_point`)
   and every effect of the call is a file in the workspace
   (`ToolRuntime.restore_covers`, the rule that `/revert` lists by, asked of
   the arguments of this call). So an unknown tool is flushed, and so is a
   write whose path leaves the tree. Formerly one barrier per cell covered
   every dispatch in it, and under the `rlm` profile *every* tool that the
   model calls is nested. A crash in the middle of a cell left the outer call
   and none of what the cell did. So the list from `/revert` of what a restore
   does not undo came back empty.
3. **At step end.** On the request path this *is* barrier 1. The flush of the
   next request covers everything that the previous step committed, and a
   second fsync microseconds earlier would add nothing. The only step end that
   barrier 1 never reaches is a pre-step **reject**, because no request
   follows. So that is the one case flushed here. The flush is on
   `agent/step-rejected`, when the decision is final, because a row outside
   this one that rejected without calling on was never seen.
4. **After the agents have let go**, as this row unwinds. The row sits in the
   log layer beneath `agent`, and rows unwind in reverse. So this barrier runs
   after every agent scope has appended what it appends on the way out, such
   as `workspace/disposed`, the tombstone of a child or the `turn/end` of a
   canceled turn. It runs before persistence and telemetry go. So that tail is
   written through the store, which reports it on `session/durable`. The
   barrier then waits for the detached work that the write set off
   (`Context.drain`), which includes the telemetry ledger shipping the tail.
   The last write of the mount (`ph.persistence.protocol.write_on_unwind`)
   stays for a profile without this row. With this row it finds nothing owed.

Barriers 1 and 2 are **fail-closed**: if the flush raises, the adapter and the
tool body are not invoked. A side effect whose record could not be written is
worse than a side effect that did not happen.

@module ph.persistence.checkpoint_policy
"""

from __future__ import annotations

import anyio

from ..agent.types import PreStepRequest
from ..cordis import Context, plugin
from ..keys import SESSIONS, TOOLS
from ..llm.types import GenerateOptions
from ..seams.workspace import has_restore_point
from ..tools.definition import ToolExecution

__all__ = ["apply"]


@plugin(
    "session-checkpoint-policy", affects="deployment", inject=[SESSIONS], log_layer="checkpoint"
)
async def apply(ctx: Context, config: None) -> None:
    """Install the semantic checkpoints."""

    async def before_request(request: GenerateOptions) -> None:
        if request.session_id is not None:
            session = ctx.require(SESSIONS).get(request.session_id)
            if session is not None:
                # Awaited, not scheduled. The purpose of a barrier is that the
                # request cannot be in flight while the events that motivated it
                # are still in a buffer.
                await ctx.require(SESSIONS).flush(session)

    async def before_tool_body(execution: ToolExecution) -> None:
        if execution.session is None:
            return
        tools = ctx.require(TOOLS)
        if execution.parent is not None and (
            tools.effect_free(execution.name, scope=execution.scope)
            or (
                has_restore_point(execution.session, execution.root_call_id)
                # As the dispatch's record says (`ToolExecution.restore_covered`).
                and execution.restore_covered
            )
        ):
            # Nothing that a crash could leave half done. A read changes nothing,
            # and the restore takes back an edit in a run that took a restore
            # point. A cell of reads and edits should not pay one fsync for each
            # call. But without the restore point, nothing would take the edits
            # back.
            return
        await ctx.require(SESSIONS).flush(execution.session)

    async def after_reject(request: PreStepRequest) -> None:
        # No request will follow to flush the previous step's results.
        await ctx.require(SESSIONS).flush(request.session)

    async def after_the_agents() -> None:
        # Not fail-closed: nothing waits on it, and a log that could not be written
        # is for the last write of the mount to try again. The writes run all at
        # once. Each flush writes the ancestors of a session before it, and waits
        # on a write of them that is already under way, so a child still never
        # lands before its parent.
        sessions = ctx.require(SESSIONS)
        async with anyio.create_task_group() as writing:
            for session in sessions.list():
                writing.start_soon(sessions.written, session)
        # A `session/durable` reader acts off the flush path, in detached work
        # that its scope would cancel. The fan-out of the tail by the telemetry
        # ledger is such work. This is its one chance to finish while the rows
        # beneath are mounted.
        await ctx.drain()

    ctx.on("llm/send", before_request)
    ctx.on("tools/body", before_tool_body)
    ctx.on("agent/step-rejected", after_reject)
    ctx.add_disposer(after_the_agents, label="checkpoint.after-the-agents")
