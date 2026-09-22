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

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
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

__all__ = ["BlockAssembler", "highest_minted_id"]

_MINT_PREFIX = "call-"
"""The spelling `BlockAssembler` mints, in one place.

Written once because three things have to agree on it: what `_minted` produces,
what `highest_minted_id` reads back, and the cheap test that decides whether a
block is worth the regex at all."""

_MINTED = re.compile(rf"^{re.escape(_MINT_PREFIX)}(\d+)$")
"""The shape `BlockAssembler` mints. Read back by `highest_minted_id`."""


def highest_minted_id(messages: Sequence[Message]) -> int:
    """The highest id pH has already minted in this conversation (G10).

    **A call id has to be unique across the conversation, not the request**, and
    nothing on a stream knows what came before it. pH pairs results to calls by
    id everywhere — `persistence.repair` keys its pending-call table on it — so
    two calls sharing one id is a result attributed to the wrong call in the log
    as well as on the wire, and the model is told a tool it did not call
    answered.

    Every wire needs this, which is why it is here. Google has no id at all;
    the other two have one the provider *usually* sends, and an
    OpenAI-compatible server that omits it (llama.cpp, vLLM, a gateway in front
    of either) or an Anthropic `content_block_start` without one falls back to
    the same mint. Filed as a Google defect (G1) and fixed there first, it was
    live on all three (G10).

    Read off the history rather than kept as adapter state: a resumed session's
    first request has no state to have kept. `max` rather than a count, so a
    compaction that shadowed an earlier turn cannot walk the counter backwards
    onto an id still in the transcript.

    **Only pH's own spelling counts.** A provider-issued id — `toolu_...`,
    `call_abc123` — does not match, so it neither raises the counter nor is
    continued; those are unique by the provider's own construction. Ids minted
    by *another* pH wire do match, deliberately: a session that changed models
    mid-conversation carries them, and the next mint has to clear them.
    """
    highest = 0
    for message in messages:
        for block in message.content:
            # `startswith` ahead of the regex, because this walks the whole
            # conversation on every model call and almost nothing in it is a
            # candidate: on the two wires where the provider issues ids, every
            # block fails this test and the regex never runs. Halves the walk on
            # a history that only grows.
            if not isinstance(block, ToolCallBlock) or not block.id.startswith(_MINT_PREFIX):
                continue
            if (found := _MINTED.match(block.id)) is not None:
                highest = max(highest, int(found.group(1)))
    return highest


@dataclass(slots=True)
class _Partial:
    block_type: str
    text: str = ""
    tool_call_id: str | None = None
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

    mint_from: int = 0
    """The highest call id already in this conversation (G10).

    Seeded by the caller from `highest_minted_id(request.messages)`, because the
    assembler sees one stream and the id has to be unique across the
    conversation. Zero — the default — is a stream with no history to clear,
    which is what an isolated unit test and the first turn of a session both
    are.
    """
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

    def _minted(self, index: int) -> str:
        """An id for a call the provider did not name (G10).

        `call-<n>` rather than a uuid: it is only ever paired within one
        conversation, and a stable spelling keeps a replayed session's ids
        identical to the live run's — the block index is fixed for a given
        stream and `mint_from` is read off the same history both times, so
        replay reproduces what was stored rather than renaming it.

        Off the block index, not a running count, so the two ways a call
        reaches here — a closed `BlockEnd` and a delta-only stream — agree on
        the id for the same block. The numbering may skip where a stream also
        carried text; unique and rising is the whole contract.
        """
        return f"{_MINT_PREFIX}{self.mint_from + index + 1}"

    def _assemble(self, partial: _Partial, index: int) -> ContentBlock:
        if partial.block is not None:
            # **A closed block is authoritative about everything but an absent
            # id** (G10). All three adapters send the finished `ToolCallBlock`
            # on `BlockEnd`, so returning it whole meant this fallback only ever
            # covered delta-only streams — and each adapter had written its own
            # mint for the path that actually runs. They were per-stream to a
            # one, which is the collision G1 names.
            if isinstance(partial.block, ToolCallBlock) and not partial.block.id:
                return partial.block.model_copy(update={"id": self._minted(index)})
            return partial.block
        if partial.block_type == "text":
            return TextBlock(text=partial.text)
        if partial.block_type == "reasoning":
            return ReasoningBlock(text=partial.text)
        if partial.block_type == "tool-call":
            return ToolCallBlock(
                id=partial.tool_call_id or self._minted(index),
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
