"""P0-12 — the vocabulary, and `BlockAssembler`.

Gate: *the assembler reconstructs a recorded stream.*

"Recorded" is the operative word. The loop logs every raw chunk, so replay
fidelity depends on the assembler producing the same message from the logged
chunks as it did live — otherwise a replayed session diverges from the one that
actually happened.
"""

from __future__ import annotations

import pytest

from ph.llm import BlockAssembler
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    LlmFailure,
    ReasoningDelta,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
    chunk_from_wire,
    is_token_delta,
)


def _recorded() -> list[object]:
    return [
        BlockStart(index=0, block_type="reasoning"),
        ReasoningDelta(index=0, text="let me "),
        ReasoningDelta(index=0, text="think"),
        BlockStart(index=1, block_type="text"),
        TextDelta(index=1, text="Hello, "),
        TextDelta(index=1, text="world"),
        BlockEnd(index=1, block=TextBlock(text="Hello, world")),
        BlockStart(index=2, block_type="tool-call"),
        ToolCallDelta(index=2, id="call-1", name="read", arguments_delta='{"pa'),
        ToolCallDelta(index=2, id="call-1", arguments_delta='th": "a"}'),
        UsageChunk(usage=TokenUsage(input_tokens=10, output_tokens=4)),
        Finish(reason=FinishReason(kind="tool-calls")),
    ]


def test_assembler_rebuilds_blocks_in_stream_order() -> None:
    assembler = BlockAssembler()
    for chunk in _recorded():
        assembler.push(chunk)
    blocks = assembler.blocks()
    assert [block.type for block in blocks] == ["reasoning", "text", "tool-call"]
    assert blocks[0].text == "let me think"
    assert blocks[1].text == "Hello, world"
    assert blocks[2].name == "read"
    # Tool arguments stay the raw JSON string the model produced, unparsed.
    assert blocks[2].arguments == '{"path": "a"}'
    assert assembler.usage == TokenUsage(input_tokens=10, output_tokens=4)
    assert assembler.finish.kind == "tool-calls"


def test_chunks_round_trip_through_the_log() -> None:
    for chunk in _recorded():
        assert chunk_from_wire(chunk.to_wire()) == chunk


def test_a_replayed_stream_produces_the_same_message() -> None:
    live = BlockAssembler()
    replayed = BlockAssembler()
    for chunk in _recorded():
        live.push(chunk)
        replayed.push(chunk_from_wire(chunk.to_wire()))
    left = live.message(provider="fake", model="m")
    right = replayed.message(provider="fake", model="m")
    assert [b.to_wire() for b in left.content] == [b.to_wire() for b in right.content]


def test_block_end_closes_a_block_and_later_deltas_are_ignored() -> None:
    assembler = BlockAssembler()
    assembler.push(BlockStart(index=0, block_type="text"))
    assembler.push(TextDelta(index=0, text="kept"))
    assembler.push(BlockEnd(index=0, block=TextBlock(text="kept")))
    # A misbehaving adapter must not be able to grow memory or corrupt a
    # completed block after it closed.
    assembler.push(TextDelta(index=0, text=" ignored"))
    assembler.push(BlockEnd(index=0, block=TextBlock(text="rewritten")))
    assert [block.text for block in assembler.blocks()] == ["kept"]


def test_delta_only_protocols_need_no_block_start() -> None:
    assembler = BlockAssembler()
    assembler.push(TextDelta(index=0, text="a"))
    assembler.push(TextDelta(index=0, text="b"))
    assert [block.text for block in assembler.blocks()] == ["ab"]


def test_max_tokens_drops_tool_calls() -> None:
    assembler = BlockAssembler()
    assembler.push(BlockStart(index=0, block_type="text"))
    assembler.push(TextDelta(index=0, text="partial"))
    assembler.push(BlockEnd(index=1, block=ToolCallBlock(id="c", name="read", arguments="{")))
    assembler.push(Finish(reason=FinishReason(kind="max-tokens")))
    # A call whose arguments were cut off cannot be executed safely, and
    # fabricating a result for it would put a lie in the log.
    assert [block.type for block in assembler.blocks()] == ["text"]


def test_interrupted_blocks_keep_only_visible_prefixes() -> None:
    assembler = BlockAssembler()
    assembler.push(BlockStart(index=0, block_type="text"))
    assembler.push(TextDelta(index=0, text="said this"))
    assembler.push(BlockStart(index=1, block_type="text"))
    assembler.push(TextDelta(index=1, text="   "))
    assembler.push(BlockStart(index=2, block_type="tool-call"))
    assembler.push(ToolCallDelta(index=2, id="c", name="read", arguments_delta="{}"))
    kept = assembler.interrupted_blocks()
    # Interruption precedes dispatch, so a retained tool call would need a
    # fabricated result. Whitespace-only blocks are noise.
    assert [block.text for block in kept] == ["said this"]


def test_missing_finish_defaults_to_stop() -> None:
    assert BlockAssembler().finish == FinishReason(kind="stop")


def test_finish_carries_structured_failures() -> None:
    failure = LlmFailure(message="rate limited", code="RATE_LIMIT", status=429)
    finish = Finish(reason=FinishReason(kind="error", failure=failure))
    restored = chunk_from_wire(finish.to_wire())
    assert restored.reason.failure == failure


def test_is_token_delta_ignores_empty_frames() -> None:
    assert is_token_delta(TextDelta(index=0, text="a"))
    assert not is_token_delta(TextDelta(index=0, text=""))
    assert not is_token_delta(ToolCallDelta(index=0, id="c", arguments_delta=""))
    assert is_token_delta(ToolCallDelta(index=0, id="c", name="read", arguments_delta=""))
    assert not is_token_delta(UsageChunk(usage=TokenUsage(input_tokens=1, output_tokens=1)))


def test_unknown_chunk_types_are_refused() -> None:
    with pytest.raises(TypeError):
        BlockAssembler().push(object())
    with pytest.raises(ValueError, match="unknown stream chunk"):
        chunk_from_wire({"type": "nonsense"})
