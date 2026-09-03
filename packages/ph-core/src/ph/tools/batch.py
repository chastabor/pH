"""Scheduling one assistant step's tool calls.

Two properties have to hold at once, and they pull in opposite directions:

* **dispatch may overlap** — two slow reads should not run in series;
* **policy, results and result context stay in model order** — the transcript
  the model reads next must match the order it asked for, or a later turn
  reasons about a conversation that never happened.

The resolution is dsh's: `prepare` (pre-execute → approval → guards) is *started*
in model order and may overlap — see its docstring — and results commit through a window that
advances across *contiguous* settled slots. A call that finishes early waits its
turn.

`execution_mode` is re-read before every start, so a tool registered or
restricted mid-batch can still create a barrier for the calls after it.

Abort records a synthetic result for every call it skipped: replay must see a
result for each `tool/call`, or the pairing every provider requires is broken.

@module ph.tools.batch
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio

from ..cancel import CancelToken
from ..cordis import Context
from ..llm.types import Message, ToolCallBlock, new_message_id
from ..session import Session, SurfaceIntent, dumps
from .definition import ToolExecutionInput, ToolExecutionResult, aborted_result
from .json_schema import parse_arguments
from .registry import PreparedCall, ToolRuntime

__all__ = ["BatchOutcome", "execute_tool_calls", "parse_arguments"]

log = logging.getLogger("ph.tools.batch")


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    concluded: bool
    aborted: bool


@dataclass(frozen=True, slots=True)
class _Planned:
    block: ToolCallBlock
    call: ToolExecutionInput


@dataclass(frozen=True, slots=True)
class _GroupOutcome:
    consumed: int
    aborted: bool
    concluded: bool


async def execute_tool_calls(
    ctx: Context,
    agent: Any,
    turn: int,
    step: int,
    tool_calls: list[ToolCallBlock],
    token: CancelToken,
    accept_context: Callable[[Message], None],
    *,
    max_parallel: int = 10,
) -> BatchOutcome:
    """Run one step's calls, committing results in model order."""
    tools: ToolRuntime = ctx.tools
    session: Session = agent.session
    planned = [
        _Planned(
            block=block,
            call=ToolExecutionInput(
                call_id=block.id,
                name=block.name,
                arguments=parse_arguments(block.arguments),
                scope=agent.ctx,
                session=session,
                agent=agent,
                cancel=token,
            ),
        )
        for block in tool_calls
    ]

    index = 0
    concluded = False
    while index < len(planned):
        # Classified fresh each round, so a registry change between groups can
        # still turn a later call into a barrier.
        mode = tools.execution_mode(planned[index].call).kind
        group = planned[index:] if mode == "parallel" else [planned[index]]
        outcome = await _run_group(
            tools, session, turn, step, group, mode, token, accept_context, max_parallel
        )
        index += outcome.consumed
        concluded = concluded or outcome.concluded
        if outcome.aborted:
            for skipped in planned[index:]:
                _append_skipped(session, turn, step, skipped.block)
            return BatchOutcome(concluded=concluded, aborted=True)
    return BatchOutcome(concluded=concluded, aborted=False)


async def _run_group(
    tools: ToolRuntime,
    session: Session,
    turn: int,
    step: int,
    group: list[_Planned],
    mode: str,
    token: CancelToken,
    accept_context: Callable[[Message], None],
    max_parallel: int,
) -> _GroupOutcome:
    slots: list[PreparedCall | None] = [None] * len(group)
    call_seqs: list[int] = [-1] * len(group)
    committed = 0
    started = 0
    aborted = token.cancelled
    concluded = False
    in_flight: set[int] = set()
    failure: BaseException | None = None
    settled_send, settled_recv = anyio.create_memory_object_stream[int](
        max_buffer_size=max(len(group), 1)
    )

    async def commit_ready() -> None:
        # `committed` advances only across contiguous settled slots — the
        # model-order window (B6).
        nonlocal committed, concluded
        while committed < len(group):
            slot = slots[committed]
            if slot is None or slot.result is None:
                break
            result = (
                await tools.finalize(slot.run, slot.result)
                if slot.needs_post
                else tools.finish(slot.run, slot.result)
            )
            _append_result(
                session, turn, step, group[committed].block, result, call_seqs[committed]
            )
            for context in result.additional_contexts:
                accept_context(context)
            concluded = concluded or result.concludes_turn
            committed += 1

    async def run_one(index: int) -> None:
        nonlocal failure
        block = group[index].block

        def record(prepared: PreparedCall) -> None:
            # What will run: the model's own bytes unless approval substituted.
            ran = prepared.run.execution
            arguments = dumps(ran.arguments) if ran.substituted else block.arguments
            call_seqs[index] = _append_call(session, turn, step, block, arguments)

        try:
            prepared = await tools.prepare(group[index].call, write_ahead=record)
            if prepared.result is None:
                prepared = await tools.dispatch(prepared.run)
            slots[index] = prepared
        except BaseException as error:
            failure = failure or error
        finally:
            await settled_send.send(index)

    async def fill_pool(scope: anyio.abc.TaskGroup) -> None:
        nonlocal started
        while (
            not aborted
            and failure is None
            and started < len(group)
            and len(in_flight) < max_parallel
        ):
            if (
                started > 0
                and mode == "parallel"
                and tools.execution_mode(group[started].call).kind != "parallel"
            ):
                # An exclusive call ends this pool and opens the caller's next
                # barrier; it must not join a group already in flight.
                break
            in_flight.add(started)
            scope.start_soon(run_one, started)
            started += 1

    async with anyio.create_task_group() as scope:
        await fill_pool(scope)
        while in_flight:
            index = await settled_recv.receive()
            in_flight.discard(index)
            await commit_ready()
            aborted = aborted or token.cancelled
            if failure is not None:
                # Stop replenishing; the task group still drains what started.
                break
            await fill_pool(scope)

    settled_send.close()
    settled_recv.close()
    await commit_ready()
    if failure is not None:
        raise failure
    if aborted or token.cancelled:
        for skipped in group[started:]:
            _append_skipped(session, turn, step, skipped.block)
        return _GroupOutcome(consumed=len(group), aborted=True, concluded=concluded)
    return _GroupOutcome(consumed=started, aborted=False, concluded=concluded)


def _append_call(
    session: Session, turn: int, step: int, block: ToolCallBlock, arguments: str
) -> int:
    """Log the call once the gate has decided, and return the seq its result cites.

    `arguments` is the caller's to choose, and the choice is "what will run":
    the model's own bytes in the ordinary case — re-encoding every call would
    change the log for a formatting difference, and a malformed string the model
    sent is preserved as text so the tool can report it (`parse_arguments`) — or
    the substitution serialised the way the log serialises everything, since
    there is no model text for it.
    """
    event = session.append(
        "tool/call",
        {
            "turn": turn,
            "step": step,
            "callId": block.id,
            "name": block.name,
            "arguments": arguments,
        },
    )
    return event.seq


def _append_result(
    session: Session,
    turn: int,
    step: int,
    block: ToolCallBlock,
    result: ToolExecutionResult,
    call_seq: int,
) -> None:
    # Built as the wire dict directly: the content blocks are already validated
    # WireModels from `render()`, and `derive_messages()` validates the whole
    # message on the way back out, so an intermediate `Message` here would be a
    # third traversal that proves nothing new.
    message: dict[str, Any] = {
        "id": new_message_id(),
        "role": "user",
        "content": [
            {
                "type": "tool-result",
                "toolCallId": block.id,
                "content": [content.to_wire() for content in result.content],
                "isError": result.is_error,
            }
        ],
        "source": {"kind": "tool", "callId": block.id},
    }
    data: dict[str, Any] = {"turn": turn, "step": step, "message": message}
    if result.error is not None:
        data["failureKind"] = result.error.kind
        if result.error.info is not None:
            data["error"] = result.error.info
    if result.meta is not None:
        data["meta"] = result.meta
    session.append(
        "tool/result", data, SurfaceIntent("append", (call_seq,) if call_seq >= 0 else None)
    )


def _append_skipped(session: Session, turn: int, step: int, block: ToolCallBlock) -> None:
    """A call cancellation skipped still gets its durable call/result pair."""
    call_seq = _append_call(session, turn, step, block, block.arguments)
    _append_result(session, turn, step, block, aborted_result(started=False), call_seq)
