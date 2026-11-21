"""Assemble a stream of chunks into an assistant message.

This is the single assembly algorithm. The loop feeds it every chunk, and also
logs the raw chunks so that a replay is exact. When the stream ends, the loop
reads `blocks()`, `message()`, `usage` and `finish`. When a cancellation cut the
stream short, the loop reads `interrupted_blocks()` instead.

The assembler accepts a protocol that sends only deltas (no `block-start` and
no `block-end`). It ignores a delta that arrives for an index that a `block-end`
already closed. An adapter that misbehaves can therefore neither grow memory
without bound nor corrupt a completed block.

Ported from dsh `packages/llm/llm/src/assembler.ts`.

@module ph.llm.assembler
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Never, NoReturn

from .types import (
    BlockEnd,
    BlockStart,
    ContentBlock,
    Finish,
    FinishReason,
    Message,
    ReasoningBlock,
    ReasoningDelta,
    StreamChunk,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
    create_assistant_message,
)

__all__ = ["BlockAssembler", "named_call"]


def _call_id(turn: int, step: int, attempt: int, index: int) -> str:
    """The id of one block of one model call, built from its position in the run."""
    return f"call-{turn}-{step}-{attempt}-{index}"


def named_call(chunk: StreamChunk, turn: int, step: int, attempt: int) -> StreamChunk:
    """Give an id to a tool call that the provider did not name (G1, G10, G11).

    **Every wire needs this.** Google issues no call id at all. The other two
    wires omit one often enough that all three adapters had their own `call-<n>`
    fallback. Each fallback counted within a single stream. Every turn therefore
    restarted the numbering, and two calls in one conversation shared an id. pH
    pairs results to calls by id everywhere, so a shared id attributes a result
    to the wrong call, in the log and on the wire.

    **The caller is the place that already holds the coordinates.** G10 put the
    id creation in `BlockAssembler`. The assembler sees a stream and not the
    conversation, so a seed had to be passed in from three construction sites.
    Two of those sites could never create an id, and a fourth site would have
    collided without an error. G11 moved the id creation to the `llm/stream`
    seam. That was one layer better, but it was still conditional: the
    coordinates were two optional fields on `GenerateOptions`, and the caller
    had to remember to set them. `ReactLoopAgent._step` already holds the
    coordinates. It is also the only consumer in the repository that pairs a
    call to a result: `structured` and `compaction` read chunks and never pair.
    The argument for a lower layer was therefore an argument about a caller
    that does not exist, and its cost was two loop-only fields on the provider
    request type.

    **The id is built from `(turn, step, index)` and not from a counter over the
    history.** That makes this a pure function of the chunk. A seed that was
    read from the projected conversation moved *backwards* after a compaction
    shadowed a range, onto ids that were still in the session log. These
    coordinates are monotonic through resume and compaction, and a replay
    reproduces them and does not renumber.

    **`attempt` makes these the coordinates of a *call*** (G12). A retried step
    keeps its turn and step on purpose, because that is what makes `llm/retry`
    readable in the log. Without a third number, two attempts created the same
    ids. The driver counts the attempt, because every retry passes through the
    driver, whichever listener chose it. The attempt is the total of
    `RequestFailure.retries_by`, and the budget of one policy is only its own
    share of that total. Before G12 the ids were sound only by accident. An
    error finish skips `_append_assistant_message`, so the losing attempt never
    reached the transcript that pairing reads, but the raw `assistant/chunk`
    records held the repeated ids. `recorded_steps` inferred the same boundary
    from a trailing `Finish`, because it did not have this number.

    The loop applies this function to the chunk before it logs the chunk. The
    raw `assistant/chunk` and the assembled `assistant/message` therefore carry
    the same id. The two had drifted apart while only the assembler created ids.
    """
    if isinstance(chunk, ToolCallDelta) and not chunk.id:
        return replace(chunk, id=_call_id(turn, step, attempt, chunk.index))
    if (
        isinstance(chunk, BlockEnd)
        and isinstance(chunk.block, ToolCallBlock)
        and not chunk.block.id
    ):
        named = chunk.block.model_copy(update={"id": _call_id(turn, step, attempt, chunk.index)})
        return replace(chunk, block=named)
    return chunk


@dataclass(slots=True)
class _Partial:
    block_type: str
    text: str = ""
    tool_call_id: str = ""
    tool_call_name: str | None = None
    tool_call_arguments: str = ""
    block: ContentBlock | None = None
    """Set by `block-end`. It is authoritative, and it freezes the partial."""


def _refuse(chunk: Never) -> NoReturn:
    """Refuse a chunk of an unknown type, in the type check and at run time.

    The `Never` parameter is the signature of `assert_never`, so a new
    `StreamChunk` variant without an arm fails the type check. The raise is a
    `TypeError` and not the `AssertionError` of `assert_never`, because the
    runtime case is not an unreachable branch. A chunk arrives from an adapter,
    and an adapter is a plugin, so "you passed the wrong type" is what happened.
    `test_unknown_chunk_types_are_refused` checks that.
    """
    raise TypeError(f"BlockAssembler.push: unknown chunk {chunk!r}")


@dataclass(slots=True)
class BlockAssembler:
    """Incrementally assembles raw chunks into blocks and a final message."""

    _partials: dict[int, _Partial] = field(default_factory=dict)
    _order: list[int] = field(default_factory=list)
    _usage: TokenUsage | None = None
    _finish: FinishReason | None = None
    _replay_state: dict[str, Any] | None = None

    def push(self, chunk: StreamChunk) -> None:
        """Feed one chunk, in stream order."""
        # The order of the arms matters, and it is not the declaration order of
        # the union. A `match` costs ~30 ns on the arm that *hits*, and a miss
        # costs what an `isinstance` did, so the most frequent chunk goes first.
        # A stream is ~98% `TextDelta`. Behind `BlockStart`, every token paid
        # for the miss.
        match chunk:
            case TextDelta():
                partial = self._ensure(chunk.index, "text")
                if partial.block is None:  # closed by block-end; ignore stragglers
                    partial.text += chunk.text
            case ReasoningDelta():
                partial = self._ensure(chunk.index, "reasoning")
                if partial.block is None:
                    partial.text += chunk.text
            case ToolCallDelta():
                partial = self._ensure(chunk.index, "tool-call")
                if partial.block is None:
                    partial.tool_call_id = chunk.id
                    if chunk.name:
                        partial.tool_call_name = chunk.name
                    partial.tool_call_arguments += chunk.arguments_delta
            case BlockStart():
                if chunk.index not in self._partials:
                    self._order.append(chunk.index)
                    self._partials[chunk.index] = _Partial(block_type=chunk.block_type)
            case BlockEnd():
                partial = self._ensure(chunk.index, chunk.block.type)
                # The first close wins. A later close of the same block is
                # ignored, so that the streamed output and the final assembled
                # block agree.
                if partial.block is None:
                    partial.block = chunk.block
            case UsageChunk():
                self._usage = chunk.usage
            case Finish():
                # The first finish wins, as for `BlockEnd` above. A stream
                # states its outcome one time. An adapter that said "error" and
                # then closed with a "stop" overwrote the only record of the
                # failure, and the turn was logged as completed with half an
                # answer in it. Keeping the first finish makes that impossible
                # for every adapter, including adapters that this repository
                # does not own.
                if self._finish is None:
                    self._finish = chunk.reason
                    self._replay_state = chunk.replay_state
            case _ as unhandled:
                _refuse(unhandled)

    def _ensure(self, index: int, block_type: str) -> _Partial:
        partial = self._partials.get(index)
        if partial is None:
            partial = _Partial(block_type=block_type)
            self._partials[index] = partial
            self._order.append(index)
        return partial

    def _assemble(self, partial: _Partial, index: int) -> ContentBlock:
        if partial.block is not None:
            return partial.block
        if partial.block_type == "text":
            return TextBlock(text=partial.text)
        if partial.block_type == "reasoning":
            return ReasoningBlock(text=partial.text)
        if partial.block_type == "tool-call":
            # **No id is invented here** (G11). `named_call` gives an id to a
            # call that the provider did not name. `ReactLoopAgent._step`
            # applies it to the chunk before the loop logs the chunk, because
            # the step holds both the coordinates of the conversation and every
            # chunk. The assembler saw a stream and not the conversation. The
            # id that it created had to be seeded from three call sites. Two of
            # them could never create an id, and a fourth would have collided
            # without an error.
            return ToolCallBlock(
                id=partial.tool_call_id,
                name=partial.tool_call_name or "",
                arguments=partial.tool_call_arguments,
            )
        raise ValueError(f'cannot assemble incomplete block of type "{partial.block_type}"')

    def blocks(self) -> list[ContentBlock]:
        """Every block seen, in stream order.

        A `max-tokens` truncation drops tool calls. A call whose arguments were
        cut off cannot be executed safely, and a fabricated result for it would
        put a false record in the log.
        """
        assembled = [self._assemble(self._partials[index], index) for index in self._order]
        if self.finish.kind == "max-tokens":
            return [block for block in assembled if block.type != "tool-call"]
        return assembled

    def interrupted_blocks(self) -> list[ContentBlock]:
        """The prefix that an interrupted stream can safely finalize.

        The prefix is the text and reasoning blocks with non-whitespace content,
        in stream order. Tool calls are omitted because the interruption comes
        before dispatch. Keeping one would require a fabricated result.
        """
        kept: list[ContentBlock] = []
        for index in self._order:
            partial = self._partials[index]
            # **The kind is decided before the block is assembled, and that
            # order matters.** `_assemble` raises on a `block_type` outside the
            # three that it knows. `BlockStart.block_type` is a `str` that the
            # replay path passes through from the wire. Assembling first would
            # therefore turn an unrecognized block from an adapter into a raise
            # on the *cancel* path, where a raise costs most. It would also
            # build a model for each in-flight tool call and then discard it:
            # 38 µs against 2 µs at 32 calls.
            #
            # When the block is closed, its own `type` wins over the type of the
            # partial. The `getattr` that this line replaced did the same. This
            # is now a typed read, because `block` is `ContentBlock | None` and
            # not `Any`.
            kind = partial.block.type if partial.block is not None else partial.block_type
            if kind not in ("text", "reasoning"):
                continue
            block = self._assemble(partial, index)
            # This narrows for the checker what `kind` established for the
            # reader. `block_type` is a `str`, so mypy cannot prove that this
            # check is redundant.
            if isinstance(block, (TextBlock, ReasoningBlock)) and block.text.strip() != "":
                kept.append(block)
        return kept

    @property
    def usage(self) -> TokenUsage | None:
        """The usage from the `usage` chunk, or `None` until one arrives."""
        return self._usage

    @property
    def finish(self) -> FinishReason:
        """The finish reason, or `stop` when the stream ended without one."""
        return self._finish if self._finish is not None else FinishReason(kind="stop")

    @property
    def replay_state(self) -> dict[str, Any] | None:
        return self._replay_state

    def message(self, *, provider: str, model: str) -> Message:
        """The assembled assistant message."""
        return create_assistant_message(
            content=self.blocks(),
            provider=provider,
            model=model,
            replay_state=self._replay_state,
        )
