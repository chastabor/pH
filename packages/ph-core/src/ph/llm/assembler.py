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

from dataclasses import dataclass, field
from typing import Any

from .types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    ReasoningBlock,
    ReasoningDelta,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
    create_assistant_message,
)

__all__ = ["BlockAssembler"]


@dataclass(slots=True)
class _Partial:
    block_type: str
    text: str = ""
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments: str = ""
    block: Any = None
    """Set by `block-end` — authoritative, and freezes the partial."""


@dataclass(slots=True)
class BlockAssembler:
    """Incrementally assembles raw chunks into blocks and a final message."""

    _partials: dict[int, _Partial] = field(default_factory=dict)
    _order: list[int] = field(default_factory=list)
    _usage: TokenUsage | None = None
    _finish: FinishReason | None = None
    _replay_state: dict[str, Any] | None = None

    def push(self, chunk: Any) -> None:
        """Feed one chunk, in stream order."""
        if isinstance(chunk, BlockStart):
            if chunk.index not in self._partials:
                self._order.append(chunk.index)
                self._partials[chunk.index] = _Partial(block_type=chunk.block_type)
            return
        if isinstance(chunk, (TextDelta, ReasoningDelta)):
            kind = "text" if isinstance(chunk, TextDelta) else "reasoning"
            partial = self._ensure(chunk.index, kind)
            if partial.block is not None:
                return  # closed by block-end; ignore stragglers
            partial.text += chunk.text
            return
        if isinstance(chunk, ToolCallDelta):
            partial = self._ensure(chunk.index, "tool-call")
            if partial.block is not None:
                return
            partial.tool_call_id = chunk.id
            if chunk.name:
                partial.tool_call_name = chunk.name
            partial.tool_call_arguments += chunk.arguments_delta
            return
        if isinstance(chunk, BlockEnd):
            partial = self._ensure(chunk.index, getattr(chunk.block, "type", "text"))
            # First close wins: ignoring re-close stragglers keeps streamed
            # output and the final assembled block in agreement.
            if partial.block is None:
                partial.block = chunk.block
            return
        if isinstance(chunk, UsageChunk):
            self._usage = chunk.usage
            return
        if isinstance(chunk, Finish):
            self._finish = chunk.reason
            self._replay_state = chunk.replay_state
            return
        raise TypeError(f"BlockAssembler.push: unknown chunk {chunk!r}")

    def _ensure(self, index: int, block_type: str) -> _Partial:
        partial = self._partials.get(index)
        if partial is None:
            partial = _Partial(block_type=block_type)
            self._partials[index] = partial
            self._order.append(index)
        return partial

    def _assemble(self, partial: _Partial, index: int) -> Any:
        if partial.block is not None:
            return partial.block
        if partial.block_type == "text":
            return TextBlock(text=partial.text)
        if partial.block_type == "reasoning":
            return ReasoningBlock(text=partial.text)
        if partial.block_type == "tool-call":
            return ToolCallBlock(
                id=partial.tool_call_id or f"call-{index}",
                name=partial.tool_call_name or "",
                arguments=partial.tool_call_arguments,
            )
        raise ValueError(f'cannot assemble incomplete block of type "{partial.block_type}"')

    def blocks(self) -> list[Any]:
        """Every seen block, in stream order.

        Max-token truncation drops tool calls: a call whose arguments were cut
        off cannot be executed safely, and fabricating a result for it would put
        a lie in the log.
        """
        assembled = [self._assemble(self._partials[index], index) for index in self._order]
        if self.finish.kind == "max-tokens":
            return [block for block in assembled if block.type != "tool-call"]
        return assembled

    def interrupted_blocks(self) -> list[Any]:
        """The prefix an interrupted stream can safely finalize.

        Text and reasoning with non-whitespace content, in stream order. Tool
        calls are omitted because interruption precedes dispatch — keeping one
        would require a fabricated result.
        """
        kept: list[Any] = []
        for index in self._order:
            partial = self._partials[index]
            kind = getattr(partial.block, "type", None) or partial.block_type
            if kind not in ("text", "reasoning"):
                continue
            block = self._assemble(partial, index)
            if block.text.strip() != "":
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

    def message(self, *, provider: str, model: str) -> Any:
        """The assembled assistant message."""
        return create_assistant_message(
            content=self.blocks(),
            provider=provider,
            model=model,
            replay_state=self._replay_state,
        )
