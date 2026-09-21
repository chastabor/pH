"""P0-12 — the vocabulary, and `BlockAssembler`.

Gate: *the assembler reconstructs a recorded stream.*

"Recorded" is the operative word. The loop logs every raw chunk, so replay
fidelity depends on the assembler producing the same message from the logged
chunks as it did live — otherwise a replayed session diverges from the one that
actually happened.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from ph.cordis import Context, running
from ph.json import JsonObject
from ph.keys import LLM
from ph.llm import BlockAssembler
from ph.llm.adapter import apply as llm_apply
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    LlmFailure,
    ReasoningDelta,
    StreamChunk,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
    chunk_from_wire,
    is_token_delta,
    user_text,
)
from ph.testing import as_kind, block_text, text_chunks

pytestmark = pytest.mark.anyio


def _recorded() -> list[StreamChunk]:
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
    assert block_text(blocks[0]) == "let me think"
    assert block_text(blocks[1]) == "Hello, world"
    assert as_kind(blocks[2], ToolCallBlock).name == "read"
    # Tool arguments stay the raw JSON string the model produced, unparsed.
    assert as_kind(blocks[2], ToolCallBlock).arguments == '{"path": "a"}'
    assert assembler.usage == TokenUsage(input_tokens=10, output_tokens=4)
    assert assembler.finish.kind == "tool-calls"


def test_chunks_round_trip_through_the_log() -> None:
    for chunk in _recorded():
        assert chunk_from_wire(chunk.to_wire()) == chunk


@pytest.mark.parametrize(
    ("wire", "path"),
    [
        ({"type": "block-start", "index": "x", "blockType": "text"}, "block-start.index"),
        ({"type": "block-start", "index": 0}, "block-start.blockType"),
        ({"type": "text-delta", "index": 0, "text": 42}, "text-delta.text"),
        (
            {"type": "tool-call-delta", "index": 0, "id": "c1", "argumentsDelta": []},
            "tool-call-delta.argumentsDelta",
        ),
        ({"type": "finish", "reason": "stop"}, "finish.reason"),
    ],
)
def test_a_mis_shaped_chunk_is_refused_by_field_path(wire: JsonObject, path: str) -> None:
    """A wrong *type* is as malformed as a missing key, and says where.

    Only the missing key used to be caught: the chunks are frozen dataclasses that
    validate nothing, so `{"index": "x"}` built a `BlockStart` whose index was a
    string and handed it to an assembler that indexes and concatenates with it.
    """
    with pytest.raises(ValueError, match=re.escape(path)):
        chunk_from_wire(wire)


def test_a_chunk_keeps_a_key_a_later_build_added() -> None:
    """A log is read by builds older than the one that wrote it."""
    wire = {**TextDelta(index=0, text="hi").to_wire(), "futureKey": 1}
    assert chunk_from_wire(wire) == TextDelta(index=0, text="hi")


def test_an_unknown_finish_kind_is_refused_rather_than_replayed() -> None:
    """The union validates the nested reason too, down to its `Literal` kind."""
    with pytest.raises(ValueError, match=re.escape("finish.reason.kind")):
        chunk_from_wire({"type": "finish", "reason": {"kind": "invented"}})


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
    assert [block_text(block) for block in assembler.blocks()] == ["kept"]


def test_delta_only_protocols_need_no_block_start() -> None:
    assembler = BlockAssembler()
    assembler.push(TextDelta(index=0, text="a"))
    assembler.push(TextDelta(index=0, text="b"))
    assert [block_text(block) for block in assembler.blocks()] == ["ab"]


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
    assert [block_text(block) for block in kept] == ["said this"]


def test_the_first_finish_wins() -> None:
    """An adapter that reports a failure and then tidies up must not erase it.

    Anthropic's wire sends an `error` event and pH's own reader then ran its
    end-of-stream cleanup, which appended a second `Finish` reading "stop". The
    last one landed, so a turn the provider had failed was recorded as completed
    with whatever text had arrived before the error — no retry, no report, and a
    truncated answer in the transcript.

    Fixed in the adapter too, but pinned here: this is the one place every
    adapter's stream passes through, including ones this repo does not own.
    """
    assembler = BlockAssembler()
    failure = LlmFailure(message="overloaded", code="PROVIDER_ERROR")
    assembler.push(Finish(reason=FinishReason(kind="error", failure=failure)))
    assembler.push(Finish(reason=FinishReason(kind="stop")))
    assert assembler.finish == FinishReason(kind="error", failure=failure)


def test_missing_finish_defaults_to_stop() -> None:
    assert BlockAssembler().finish == FinishReason(kind="stop")


def test_finish_carries_structured_failures() -> None:
    failure = LlmFailure(message="rate limited", code="RATE_LIMIT", status=429)
    finish = Finish(reason=FinishReason(kind="error", failure=failure))
    restored = chunk_from_wire(finish.to_wire())
    # `chunk_from_wire` answers with the whole `StreamChunk` union, and only
    # `Finish` carries a reason — the narrowing this test was asserting by
    # reading the field.
    assert isinstance(restored, Finish)
    assert restored.reason.failure == failure


def test_is_token_delta_ignores_empty_frames() -> None:
    assert is_token_delta(TextDelta(index=0, text="a"))
    assert not is_token_delta(TextDelta(index=0, text=""))
    assert not is_token_delta(ToolCallDelta(index=0, id="c", arguments_delta=""))
    assert is_token_delta(ToolCallDelta(index=0, id="c", name="read", arguments_delta=""))
    assert not is_token_delta(UsageChunk(usage=TokenUsage(input_tokens=1, output_tokens=1)))


def test_unknown_chunk_types_are_refused() -> None:
    with pytest.raises(TypeError):
        # `object()` is the point: `push` must refuse a chunk that is not one,
        # so the argument is deliberately outside the union it declares.
        BlockAssembler().push(object())  # type: ignore[arg-type]
    # The tagged union names every kind it knows, which is what a reader of a
    # log written by a newer build needs to see.
    with pytest.raises(ValueError, match=re.escape("Input tag 'nonsense'")):
        chunk_from_wire({"type": "nonsense"})


# ------------------------------------------------------------- the seam --


async def test_an_adapters_stream_body_runs_inside_its_rows_binding() -> None:
    """C10 — the `running(...)` wrapped generator *creation*, which runs nothing.

    `adapter.stream(request)` is an async-generator call and `_normalized(...)`
    is another, so the old line constructed two generators inside the binding and
    left both bodies to run later, on whoever consumed them — outside it. Every
    registration an adapter made while streaming therefore defaulted its owner to
    the seam and outlived the row that made it, which is exactly the P6-12 leak
    `current_owner` exists to close.

    Asserted at the first chunk rather than at construction, because "when the
    body runs" is the whole finding: a test that read the owner from
    `stream()`'s own frame would pass against the broken version.
    """
    root = Context()
    await llm_apply(root, None)
    runtime = root.require(LLM)
    row = root.scope("row:adapter")
    seen: list[Context | None] = []

    class Watching:
        def stream(self, options: GenerateOptions) -> Any:  # noqa: ANN401
            async def chunks() -> Any:  # noqa: ANN401
                seen.append(Context.current_owner())
                for chunk in text_chunks("hello"):
                    yield chunk

            return chunks()

    with running(row.running_for(row)):
        runtime.register_adapter(["watched"], Watching())

    options = GenerateOptions(
        provider="watched",
        model="m",
        messages=(user_text("hi"),),
    )
    async for _chunk in await runtime.stream(options):
        pass

    assert seen == [row], "the adapter's body ran outside its row's binding"
    await root.dispose()
