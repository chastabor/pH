"""Scheduling the tool calls of one assistant step.

Two properties must hold at the same time, and they conflict:

* **Dispatch may overlap.** Two slow reads must not run in series.
* **Policy, results and result context stay in model order.** The transcript
  that the model reads next must match the order that the model asked for.
  Otherwise a later turn reasons about a conversation that never happened.

The solution comes from dsh. `prepare` (pre-execute, then approval, then
guards) is *started* in model order, and the prepares may overlap. The
docstring of `ToolRuntime.prepare` gives the details. Results commit through a
window that advances across *contiguous* settled slots. A call that finishes
early waits for its turn.

`execution_mode` is read again before every start. So a tool that is registered
or restricted in the middle of a batch can still create a barrier for the calls
after it.

An abort records a synthetic result for every call that it skipped. Replay must
see a result for each `tool/call`. Without one, the pairing that every provider
requires is broken.

@module ph.tools.batch
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anyio

from ..agent.types import AgentHandle
from ..cancel import CancelToken
from ..cordis import Context
from ..json import dumps
from ..keys import TOOLS
from ..llm.types import Message, ToolCallBlock, new_message_id
from ..session import Session, SessionBatch, SurfaceIntent
from ..session.writers import log_writer
from .definition import (
    ToolExecutionInput,
    ToolExecutionResult,
    aborted_result,
    concluded_result,
)
from .json_schema import parse_arguments
from .registry import PreparedCall, ToolRuntime

_LOG = log_writer(__name__)

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
    agent: AgentHandle,
    turn: int,
    step: int,
    tool_calls: list[ToolCallBlock],
    token: CancelToken,
    accept_context: Callable[[Message], None],
    *,
    max_parallel: int = 10,
) -> BatchOutcome:
    """Run one step's calls, committing results in model order."""
    tools: ToolRuntime = ctx.require(TOOLS)
    session = agent.session
    if session is None:
        # The Protocol allows a handle with no session, because `StubAgent` is
        # one. Without a log, a batch has nowhere to record its calls.
        raise RuntimeError(f'agent "{agent.id}" has no session to record tool calls in')
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
    covered = 0
    """The index up to which the calls are answered if the code that runs now
    raises. A group answers its own calls on the way out (`_run_group`). So while
    a group runs, this index is the end of that group."""
    concluded = False
    try:
        while index < len(planned):
            covered = index
            # The mode is classified again in each round. So a registry change
            # between groups can still make a later call a barrier.
            mode = tools.execution_mode(planned[index].call).kind
            group = planned[index:] if mode == "parallel" else [planned[index]]
            covered = index + len(group)
            outcome = await _run_group(
                tools, session, turn, step, group, mode, token, accept_context, max_parallel
            )
            index += outcome.consumed
            concluded = concluded or outcome.concluded
            if outcome.aborted:
                for skipped in planned[index:]:
                    _append_skipped(session, turn, step, skipped.block)
                return BatchOutcome(concluded=concluded, aborted=True)
            if concluded:
                # **A group that ended the turn ends the step** (C13). Before
                # this check, `concluded` was folded here and the driver read
                # it, but the loop continued and opened the next barrier. So an
                # exclusive call that was queued behind a parallel run was
                # dispatched *after* the turn was over. The docstring of
                # `Deny.concludes_turn` says that the batch already in flight
                # still settles, and that what ends is what comes after it. A
                # group that has not started is what comes after it. The effect
                # is worst on a fan-out. There a ceiling is most likely to be
                # reached in the middle of a step, and the calls behind it
                # spawn children.
                for skipped in planned[index:]:
                    _append_skipped(session, turn, step, skipped.block, concluded_result())
                return BatchOutcome(concluded=True, aborted=False)
    except BaseException:
        # **The calls behind a group that raised are answered too** (T1). Before
        # this handler, a raise from a group, or a stop from outside the token,
        # left every call queued behind it as a `tool_use` with no
        # `tool_result`. The turn then closed, so repair had nothing open to
        # close, and the provider refused every later request. The appends are
        # synchronous, so the stop cannot cut them short.
        for skipped in planned[covered:]:
            _append_skipped(session, turn, step, skipped.block)
        raise
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
    aborted = token.canceled
    concluded = False
    in_flight: set[int] = set()
    failure: BaseException | None = None
    settled_send, settled_recv = anyio.create_memory_object_stream[int](
        max_buffer_size=max(len(group), 1)
    )

    async def commit_ready() -> None:
        # `committed` advances only across contiguous settled slots. This is
        # the model-order window (B6).
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
                # An exclusive call ends this pool and opens the next barrier of
                # the caller. It must not join a group that is already in flight.
                break
            in_flight.add(started)
            scope.start_soon(run_one, started)
            started += 1

    def answer_uncommitted() -> None:
        """Close every call that this group logged and did not answer (C8).

        **A raise is not a cancellation, and at first only the cancellation path
        closed these calls.** `commit_ready` advances across *contiguous*
        settled slots. That is the model-order window that B6 requires. So when
        one call raises, every call behind it in the group may have run and
        settled and still have no `tool/result`. The turn then ends with
        `tool_use` blocks that nothing answers, and the provider rejects them on
        the next request. The crash repair does not reach this case. Repair
        closes pairs in a turn that the log left *open*, and this turn closed
        normally.

        The aborted result has two forms, and they differ in what the model is
        told. A slot that holds a result ran to completion, and the batch did
        not commit it. It gets `aborted` after dispatch, which says "this may
        have happened". A slot that is still empty never dispatched, and that is
        safe to retry. A call past `started` was never logged, so it needs the
        pair that `_append_skipped` writes.
        """
        for index in range(committed, len(group)):
            block = group[index].block
            if index >= started or call_seqs[index] < 0:
                _append_skipped(session, turn, step, block)
                continue
            # A call that was still in flight when a stop arrived was dispatched,
            # and it may have done what it was asked to do. It does not get the
            # "safe to retry" result of a call that never reached its body.
            began = slots[index] is not None or index in in_flight
            _append_result(
                session, turn, step, block, aborted_result(started=began), call_seqs[index]
            )

    try:
        async with anyio.create_task_group() as scope:
            await fill_pool(scope)
            while in_flight:
                index = await settled_recv.receive()
                in_flight.discard(index)
                await commit_ready()
                aborted = aborted or token.canceled
                if failure is not None:
                    # Stop refilling the pool. The task group still drains the
                    # calls that started.
                    break
                await fill_pool(scope)
        await commit_ready()
    except BaseException:
        # **A stop from outside the token is answered too** (T1). Examples are
        # `SIGINT` on a one-shot run, and the task of a released root. Such a
        # stop reaches the group as a cancellation of the task group. Before
        # this handler, it skipped every answer that the code below writes. The
        # appends here are the same synchronous appends, so the stop cannot cut
        # them short.
        answer_uncommitted()
        raise
    finally:
        settled_send.close()
        settled_recv.close()
    if failure is not None:
        answer_uncommitted()
        raise failure
    if aborted or token.canceled:
        for skipped in group[started:]:
            _append_skipped(session, turn, step, skipped.block)
        return _GroupOutcome(consumed=len(group), aborted=True, concluded=concluded)
    return _GroupOutcome(consumed=started, aborted=False, concluded=concluded)


def _append_call(
    session: Session | SessionBatch, turn: int, step: int, block: ToolCallBlock, arguments: str
) -> int:
    """Log the call after the gate has decided, and return the seq that the result
    of the call cites.

    The caller chooses `arguments`, and it passes what will run. In the ordinary
    case that is the bytes that the model sent. Encoding every call again would
    change the log for a difference in formatting. A malformed string from the
    model is kept as text, so that the tool can report it (`parse_arguments`).
    When approval substituted the arguments, the caller passes the substitution,
    serialized in the way that the log serializes everything, because there is
    no model text for it.
    """
    event = _LOG.append(
        session,
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
    session: Session | SessionBatch,
    turn: int,
    step: int,
    block: ToolCallBlock,
    result: ToolExecutionResult,
    call_seq: int,
) -> None:
    # The message is built directly as the wire dict. The content blocks are
    # already validated WireModels from `render()`, and `derive_messages()`
    # validates the whole message when it is read back. So an intermediate
    # `Message` here would be a third traversal that proves nothing new.
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
    _LOG.append(
        session,
        "tool/result",
        data,
        SurfaceIntent("append", (call_seq,) if call_seq >= 0 else None),
    )


def _append_skipped(
    session: Session,
    turn: int,
    step: int,
    block: ToolCallBlock,
    result: ToolExecutionResult | None = None,
) -> None:
    """Write the durable call/result pair of a call that was never dispatched.

    The default result is a cancellation. Cancellation was the only reason to
    skip a call before C13 added the second reason. A turn that concluded passes
    its own result. Both results mean "this never ran", and only one of them
    means that somebody canceled the call.

    **The pair is written in one batch** (S14). Written separately, a torn tail
    could keep the call and lose its result. After repair, that call read as one
    that may have happened, although it never ran.
    """
    settled = aborted_result(started=False) if result is None else result
    with session.batch() as batch:
        call_seq = _append_call(batch, turn, step, block, block.arguments)
        _append_result(batch, turn, step, block, settled, call_seq)
