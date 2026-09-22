"""Chunk stream in, assistant message out.

The single canonical assembly algorithm: the loop feeds it every chunk while
logging the raw chunks for replay fidelity, then reads `blocks()` / `message()`
/ `usage` / `finish` once the stream ends — or `interrupted_blocks()` when
cancellation cut it short.

Tolerant of delta-only protocols (no `block-start`/`block-end`), and deltas
arriving for an index a `block-end` already closed are ignored, so a misbehaving
adapter can neither grow memory without bound nor corrupt a completed block.

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
    """One model call's block, named by where in the run it happened."""
    return f"call-{turn}-{step}-{attempt}-{index}"


def named_call(chunk: StreamChunk, turn: int, step: int, attempt: int) -> StreamChunk:
    """Give a tool call the provider did not name an id (G1, G10, G11).

    **Every wire needs this.** Google issues no call id at all, and the other
    two omit one often enough that all three had grown their own `call-<n>`
    fallback — each counting within a single stream, so every turn restarted the
    numbering and two calls in one conversation shared an id. pH pairs results
    to calls by id everywhere, so that is a result attributed to the wrong call
    in the log as well as on the wire.

    **Called where the coordinates already are.** G10 put the mint in
    `BlockAssembler`, which sees a stream and not the conversation, so a seed had
    to be threaded in from three construction sites — two of which could never
    mint anything, and a fourth would have collided in silence. G11 moved it to
    the `llm/stream` seam, which was one layer better and still conditional: the
    coordinates rode on `GenerateOptions` as two optional fields that the caller
    had to remember to set, so "nobody has to remember" had become "nobody has
    to remember to set turn and step". `ReactLoopAgent._step` holds them
    already, and is the only consumer in the repo that pairs a call to a result
    — `structured`, `compaction` and the kernel read chunks and never pair. So
    the argument for pushing it further down was an argument about a caller that
    does not exist, and the price was two loop-only fields on the provider
    request type.

    **`(turn, step, index)` rather than a counter over the history**, which is
    what makes it a pure function of the chunk. A seed read off the projected
    conversation walked *backwards* after a compaction shadowed a range, onto
    ids still in the session log. These coordinates are monotonic through resume
    and compaction, and replay reproduces them rather than renumbering.

    **`attempt` is what makes these the coordinates of a *call*** (G12). A
    retried step keeps its turn and step on purpose — `llm/retry` is legible in
    the log because of it — so without a third number two attempts minted the
    same ids. The driver counts it, because every retry passes through the
    driver whichever listener chose it — the total of `RequestFailure.retries_by`,
    where a policy's budget is only its own share. It was sound only by
    accident: an error finish skips `_append_assistant_message`, so the losing
    attempt never reached the transcript that pairing reads, while the raw
    `assistant/chunk` records held the repeat. `recorded_steps` was inferring
    the same boundary from a trailing `Finish` for want of the same number.

    Applied to the chunk, before the loop logs it, so the raw `assistant/chunk`
    and the assembled `assistant/message` carry the same id — they had drifted
    while only the assembler minted.
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
    """Set by `block-end` — authoritative, and freezes the partial."""


def _refuse(chunk: Never) -> NoReturn:
    """Both halves of "this cannot happen", in one call.

    The `Never` parameter is `assert_never`'s own signature, so adding a
    `StreamChunk` variant without an arm fails the type check. The raise is a
    `TypeError` rather than `assert_never`'s `AssertionError` because the runtime
    case is not an unreachable branch — a chunk arrives from an adapter, an
    adapter is a plugin, and "you passed the wrong type" is what happened.
    `test_unknown_chunk_types_are_refused` pins that.
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
        # Arm order is load-bearing, not the declaration order of the union: a
        # `match` pays ~30 ns on the arm that *hits* and a miss costs what an
        # `isinstance` did, so the most frequent chunk belongs first. A stream is
        # ~98% `TextDelta`, and behind `BlockStart` every token paid for the miss.
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
                # First close wins: ignoring re-close stragglers keeps streamed
                # output and the final assembled block in agreement.
                if partial.block is None:
                    partial.block = chunk.block
            case UsageChunk():
                self._usage = chunk.usage
            case Finish():
                # First finish wins, as `BlockEnd` above. A stream states its
                # outcome once; an adapter that says "error" and then tidies up
                # with a "stop" was overwriting the only record of the failure,
                # and the turn was logged as completed with half an answer in it.
                # Keeping the first makes that unrepresentable for every adapter,
                # including ones this repo does not own.
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
            # **No id is invented here** (G11). `LlmRuntime._normalized` names a
            # call the provider did not, on the chunk, before the loop logs it —
            # the one layer holding both the conversation's coordinates and
            # every chunk. This saw a stream and not the conversation, so the
            # id it minted had to be seeded from three call sites, two of which
            # could never mint and a fourth would have collided in silence.
            return ToolCallBlock(
                id=partial.tool_call_id,
                name=partial.tool_call_name or "",
                arguments=partial.tool_call_arguments,
            )
        raise ValueError(f'cannot assemble incomplete block of type "{partial.block_type}"')

    def blocks(self) -> list[ContentBlock]:
        """Every seen block, in stream order.

        Max-token truncation drops tool calls: a call whose arguments were cut
        off cannot be executed safely, and fabricating a result for it would put
        a lie in the log.
        """
        assembled = [self._assemble(self._partials[index], index) for index in self._order]
        if self.finish.kind == "max-tokens":
            return [block for block in assembled if block.type != "tool-call"]
        return assembled

    def interrupted_blocks(self) -> list[ContentBlock]:
        """The prefix an interrupted stream can safely finalize.

        Text and reasoning with non-whitespace content, in stream order. Tool
        calls are omitted because interruption precedes dispatch — keeping one
        would require a fabricated result.
        """
        kept: list[ContentBlock] = []
        for index in self._order:
            partial = self._partials[index]
            # **The kind is decided before assembling, and that ordering is
            # load-bearing.** `_assemble` raises on a `block_type` outside the
            # four it knows, and `BlockStart.block_type` is a `str` the replay
            # path passes through from the wire — so assembling first would turn
            # an unrecognized block from an adapter into a raise on the *cancel*
            # path, which is where a raise costs most. It also builds a model per
            # in-flight tool call to discard it: 38 µs against 2 µs at 32 calls.
            #
            # The closed block's own `type` wins over the partial's when there is
            # one, which is what the `getattr` this replaced was for — now a
            # typed read, because `block` is `ContentBlock | None` rather than
            # `Any`.
            kind = partial.block.type if partial.block is not None else partial.block_type
            if kind not in ("text", "reasoning"):
                continue
            block = self._assemble(partial, index)
            # Narrows for the checker what `kind` established for the reader:
            # `block_type` is a `str`, so mypy cannot prove this redundant.
            if isinstance(block, (TextBlock, ReasoningBlock)) and block.text.strip() != "":
                kept.append(block)
        return kept

    @property
    def usage(self) -> TokenUsage | None:
        """Usage from the `usage` chunk; `None` until one arrives."""
        return self._usage

    @property
    def finish(self) -> FinishReason:
        """The finish reason; `stop` when the stream ended without one."""
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
