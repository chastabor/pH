"""P1-15 — the provider adapters, mapped and checked without a network.

The stream mapping is where an adapter quietly loses information, so these tests
pin the three places it would:

* **thinking is a separate block**, not text folded together — otherwise the
  transcript claims the model said what it was only considering;
* **tool arguments stream as deltas**, because `assistant/chunk` promises
  token-level replay fidelity and an adapter that only emitted completed calls
  would make that promise false;
* **usage counts are disjoint** — DeepSeek folds cache hits into
  `prompt_tokens`, so leaving them in bills every hit twice in pH's accounting.

The real-API smoke test is skipped without a key, per P1-15's gate.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.llm.adapter import LlmError
from ph.llm.assembler import BlockAssembler
from ph.llm.types import (
    GenerateOptions,
    MediaBlock,
    ToolSchema,
    create_tool_result_message,
    create_user_message,
)
from ph.seams.credentials import CredentialService
from ph_app.adapters._http import failure_from_status
from ph_app.adapters.anthropic import AnthropicAdapter
from ph_app.adapters.anthropic import Config as AnthropicConfig
from ph_app.adapters.openai_compatible import (
    OpenAiCompatibleAdapter,
    ProviderProfile,
    _StreamState,
    _to_openai,
    _to_usage,
)
from ph_app.adapters.sse import iter_sse

pytestmark = pytest.mark.anyio


class _Response:
    """A stub httpx streaming response over pre-baked SSE text."""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.status_code = 200

    async def aiter_text(self) -> Any:
        for chunk in self._chunks:
            yield chunk


async def test_sse_events_survive_a_boundary_mid_chunk() -> None:
    # The bug this framing exists to avoid: an event split across two network
    # reads must not be lost.
    response = _Response(['data: {"a"', ': 1}\n\ndata: {"b": 2}\n\n'])
    seen = [payload async for _event, payload in iter_sse(response)]
    assert seen == [{"a": 1}, {"b": 2}]


async def test_sse_stops_at_the_done_sentinel() -> None:
    response = _Response(['data: {"a": 1}\n\ndata: [DONE]\n\ndata: {"never": 1}\n\n'])
    seen = [payload async for _event, payload in iter_sse(response)]
    assert seen == [{"a": 1}]


def test_openai_thinking_maps_to_a_reasoning_block() -> None:
    state = _StreamState()
    chunks = [
        *state.consume({"choices": [{"delta": {"reasoning_content": "let me think"}}]}),
        *state.consume({"choices": [{"delta": {"content": "the answer"}}]}),
        *state.finish(),
    ]
    assembler = BlockAssembler()
    for chunk in chunks:
        assembler.push(chunk)
    blocks = assembler.blocks()
    # Two distinct blocks, in the order they streamed.
    assert [block.type for block in blocks] == ["reasoning", "text"]
    assert blocks[0].text == "let me think"
    assert blocks[1].text == "the answer"


def test_openai_tool_arguments_stream_as_deltas() -> None:
    state = _StreamState()
    chunks = [
        *state.consume(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "read", "arguments": '{"pa'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        *state.consume(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'th": "a"}'}}]
                        }
                    }
                ]
            }
        ),
        *state.consume({"choices": [{"finish_reason": "tool_calls"}]}),
        *state.finish(),
    ]
    deltas = [chunk for chunk in chunks if getattr(chunk, "type", "") == "tool-call-delta"]
    # Incremental, not one completed call: replay fidelity depends on it.
    assert [delta.arguments_delta for delta in deltas] == ['{"pa', 'th": "a"}']

    assembler = BlockAssembler()
    for chunk in chunks:
        assembler.push(chunk)
    (call,) = assembler.blocks()
    assert call.name == "read"
    assert json.loads(call.arguments) == {"path": "a"}
    assert assembler.finish.kind == "tool-calls"


def test_deepseek_cache_hits_are_subtracted_out() -> None:
    usage = _to_usage(
        {"prompt_tokens": 1_000, "prompt_cache_hit_tokens": 800, "completion_tokens": 20}
    )
    # Disjoint (D15): the window's occupancy is the sum, so double-counting the
    # cached prefix would over-report by 800 every turn.
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 800
    assert usage.input_tokens + (usage.cache_read_tokens or 0) == 1_000


def test_openai_cached_tokens_detail_shape_is_also_read() -> None:
    usage = _to_usage(
        {
            "prompt_tokens": 500,
            "prompt_tokens_details": {"cached_tokens": 100},
            "completion_tokens": 5,
        }
    )
    assert usage.input_tokens == 400
    assert usage.cache_read_tokens == 100


def test_a_length_finish_becomes_max_tokens() -> None:
    state = _StreamState()
    state.consume({"choices": [{"delta": {"content": "cut off"}, "finish_reason": "length"}]})
    finish = state.finish()[-1]
    assert finish.reason.kind == "max-tokens"


def test_tool_results_become_their_own_wire_role() -> None:
    message = create_tool_result_message(
        call_id="c1", content=[{"type": "text", "text": "output"}], is_error=False
    )
    (entry,) = _to_openai(message, {})
    # A tool result cannot be merged into a user message on this wire.
    assert entry["role"] == "tool"
    assert entry["tool_call_id"] == "c1"
    assert entry["content"] == "output"


def test_a_plain_user_message_stays_a_user_message() -> None:
    message = create_user_message(
        content=[{"type": "text", "text": "hello"}], source={"kind": "user"}
    )
    assert _to_openai(message, {}) == [{"role": "user", "content": "hello"}]


async def test_a_missing_credential_fails_before_any_request() -> None:
    root = Context()
    root.provide("credentials", CredentialService(ctx=root))
    adapter = OpenAiCompatibleAdapter(
        ctx=root,
        profile=ProviderProfile(provider="p", api_key_env="PH_TEST_DEFINITELY_ABSENT"),
    )
    with pytest.raises(LlmError) as caught:
        adapter._headers()
    assert caught.value.code == "MISSING_CREDENTIAL"
    # Named, so the operator knows which variable to set.
    assert "PH_TEST_DEFINITELY_ABSENT" in str(caught.value)


async def test_the_credential_is_read_only_at_the_edge() -> None:
    root = Context()
    credentials = CredentialService(ctx=root)
    credentials.provide_value("PH_TEST_EDGE_KEY", "sk-secret")
    root.provide("credentials", credentials)
    adapter = OpenAiCompatibleAdapter(
        ctx=root, profile=ProviderProfile(provider="p", api_key_env="PH_TEST_EDGE_KEY")
    )
    headers = adapter._headers()
    assert headers["Authorization"] == "Bearer sk-secret"
    # And the reference that travelled to get here carries no value.
    ref = credentials.reference("PH_TEST_EDGE_KEY")
    assert "sk-secret" not in json.dumps(ref.to_wire())


async def test_the_openai_request_body_carries_tools_and_the_system_slot() -> None:
    root = Context()
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))
    body = await adapter._body(
        GenerateOptions(
            provider="p",
            model="m",
            messages=(
                create_user_message(
                    content=[{"type": "text", "text": "hi"}], source={"kind": "user"}
                ),
            ),
            system="be brief",
            tools=(ToolSchema(name="read", description="Read.", parameters={"type": "object"}),),
            max_tokens=256,
        )
    )
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["tools"][0]["function"]["name"] == "read"
    assert body["max_tokens"] == 256
    assert body["stream"] is True
    # Usage is requested explicitly, because D15 makes it authoritative.
    assert body["stream_options"] == {"include_usage": True}


def test_status_classification_is_shared_and_overflow_is_per_wire() -> None:
    from ph_app.adapters.anthropic import _is_overflow as anthropic_overflow
    from ph_app.adapters.openai_compatible import _is_overflow as openai_overflow

    assert failure_from_status(429, "slow", is_overflow=lambda _b: False).code == "RATE_LIMIT"
    assert failure_from_status(529, "busy", is_overflow=lambda _b: False).code == "OVERLOADED"
    assert failure_from_status(503, "down", is_overflow=lambda _b: False).code == "SERVER_ERROR"
    assert failure_from_status(400, "bad", is_overflow=lambda _b: False).code == "REQUEST_FAILED"
    # Each wire phrases "too long" its own way; the classification is shared.
    assert openai_overflow("This model's maximum context length is 8192")
    assert anthropic_overflow("prompt is too long: 210000 tokens")
    # A bad-request body that merely mentions max_tokens is NOT an overflow —
    # treating it as one would compact a conversation that fit.
    assert not anthropic_overflow("max_tokens: must be greater than 0")


def test_anthropic_puts_tool_results_in_user_content() -> None:
    from ph_app.adapters.anthropic import _to_anthropic

    message = create_tool_result_message(
        call_id="c1", content=[{"type": "text", "text": "output"}], is_error=True
    )
    entry = _to_anthropic(message, {})
    # The main structural difference from the OpenAI wire.
    assert entry["role"] == "user"
    assert entry["content"][0]["type"] == "tool_result"
    assert entry["content"][0]["tool_use_id"] == "c1"
    assert entry["content"][0]["is_error"] is True


def test_anthropic_usage_needs_no_subtraction() -> None:
    from ph_app.adapters.anthropic import _to_usage as anthropic_usage

    usage = anthropic_usage(
        {
            "input_tokens": 200,
            "output_tokens": 20,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 50,
        }
    )
    # Already disjoint on this wire, unlike DeepSeek.
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 800
    assert usage.cache_write_tokens == 50


def test_anthropic_thinking_and_tool_blocks_map_across() -> None:
    from ph_app.adapters.anthropic import _StreamState as AnthropicState

    state = AnthropicState()
    chunks: list[Any] = []
    chunks += state.consume(
        "message_start",
        {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 0}}},
    )
    chunks += state.consume(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}},
    )
    chunks += state.consume(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
    )
    chunks += state.consume("content_block_stop", {"type": "content_block_stop", "index": 0})
    chunks += state.consume(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "tu_1", "name": "read"},
        },
    )
    chunks += state.consume(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"a"}'},
        },
    )
    chunks += state.consume("content_block_stop", {"type": "content_block_stop", "index": 1})
    chunks += state.consume(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 12},
        },
    )
    chunks += state.finish()

    assembler = BlockAssembler()
    for chunk in chunks:
        assembler.push(chunk)
    blocks = assembler.blocks()
    assert [block.type for block in blocks] == ["reasoning", "tool-call"]
    assert blocks[0].text == "hmm"
    assert blocks[1].name == "read"
    assert assembler.finish.kind == "tool-calls"
    assert assembler.usage is not None and assembler.usage.output_tokens == 12


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"), reason="no DEEPSEEK_API_KEY; smoke test skipped"
)
async def test_real_api_smoke(tmp_path: Any, monkeypatch: Any) -> None:  # pragma: no cover
    """P1-15's gate: one real round trip, skipped without a key."""
    from ph.agent.types import AgentOptions
    from ph_app.profiles import resolve_profile
    from ph_app.runtime import mounted

    monkeypatch.setenv("PH_HOME", str(tmp_path))
    async with mounted(resolve_profile("deepseek")) as run:
        session = run.ctx.sessions.create("smoke")
        agent = run.ctx.agents.create(
            session, AgentOptions(provider="deepseek", model="deepseek-chat", max_tokens=64)
        )
        await agent.prompt("Reply with the single word: ok")
        assert session.events[-1].data["reason"]["kind"] == "completed"
        assert any(event.type == "assistant/chunk" for event in session.events)


# ------------------------------------------------------------------ media --
# P7-01. Before this, both wire renderers built branches for the block kinds
# they knew and silently omitted the rest — so a message that was only an image
# reached the provider as an empty text block, and nobody would ever have known.


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"pixels" * 32


async def _with_attachment(root: Context, tmp_path: Path, mime: str) -> Any:
    """Mount a store on `root`, save one blob, and hand back its reference."""
    from ph.seams.attachments import AttachmentStore

    store = AttachmentStore(ctx=root, root=tmp_path / "attachments")
    root.provide("attachments", store)
    return await store.save_bytes(content=PNG_BYTES, mime=mime, name="shot.png")


def _media_message(ref: Any) -> Any:
    return create_user_message(
        content=[{"type": "text", "text": "what is this?"}, MediaBlock(attachment=ref)],
        source={"kind": "user"},
    )


async def test_anthropic_sends_an_image_as_a_base64_source(tmp_path: Path) -> None:
    """The capability the adapter now declares, actually exercised."""
    root = Context()
    ref = await _with_attachment(root, tmp_path, "image/png")
    adapter = AnthropicAdapter(ctx=root, config=AnthropicConfig())

    body = await adapter._body(
        GenerateOptions(provider="anthropic", model="m", messages=(_media_message(ref),))
    )

    (entry,) = body["messages"]
    image = next(block for block in entry["content"] if block["type"] == "image")
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == PNG_BYTES


async def test_anthropic_sends_a_pdf_as_a_document(tmp_path: Path) -> None:
    """One block keyed on MIME, two wire shapes — which is exactly why the
    branching lives in the adapter and not in the content-block union."""
    root = Context()
    ref = await _with_attachment(root, tmp_path, "application/pdf")
    adapter = AnthropicAdapter(ctx=root, config=AnthropicConfig())

    body = await adapter._body(
        GenerateOptions(provider="anthropic", model="m", messages=(_media_message(ref),))
    )

    (entry,) = body["messages"]
    assert any(block["type"] == "document" for block in entry["content"])


async def test_a_route_that_cannot_read_it_gets_a_pointer_not_a_silence(
    tmp_path: Path,
) -> None:
    """The bug, as a gate.

    An OpenAI-compatible chat route carries PDFs by `file_id` through the Files
    API (P7-03), not inline — so it declines this one. What must never happen is
    the block vanishing: the model is told, in words it can act on, that a file
    was not sent.
    """
    root = Context()
    ref = await _with_attachment(root, tmp_path, "application/pdf")
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))

    body = await adapter._body(
        GenerateOptions(provider="p", model="m", messages=(_media_message(ref),))
    )

    rendered = json.dumps(body["messages"])
    assert "application/pdf" in rendered and "shot.png" in rendered
    assert "was not sent" in rendered
    assert "input_audio" not in rendered and "image_url" not in rendered


async def test_a_blob_that_is_gone_degrades_rather_than_failing(tmp_path: Path) -> None:
    """A session copied without its attachments still opens and still runs."""
    root = Context()
    ref = await _with_attachment(root, tmp_path, "image/png")
    root.attachments.path_for(ref).unlink()
    adapter = AnthropicAdapter(ctx=root, config=AnthropicConfig())

    body = await adapter._body(
        GenerateOptions(provider="anthropic", model="m", messages=(_media_message(ref),))
    )

    assert "was not sent" in json.dumps(body["messages"])


async def test_openai_keeps_a_plain_user_message_a_string(tmp_path: Path) -> None:
    """A message with no media is byte-for-byte the request it always was.

    This wire takes both a string and a content list, and switching every user
    message to a list would have changed every existing prefix — which is what
    the cache is counting on (A12).
    """
    root = Context()
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))

    body = await adapter._body(
        GenerateOptions(
            provider="p",
            model="m",
            messages=(
                create_user_message(
                    content=[{"type": "text", "text": "hello"}], source={"kind": "user"}
                ),
            ),
        )
    )

    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_each_adapter_declares_what_it_accepts() -> None:
    """`accepts` empty means text-only, and text-only is the safe default —
    so an adapter that grows a media branch without declaring it would send
    nothing, rather than an adapter that declares one it cannot serialize."""
    anthropic = AnthropicAdapter(ctx=Context(), config=AnthropicConfig()).resolve_model("a", "m")
    openai = OpenAiCompatibleAdapter(
        ctx=Context(), profile=ProviderProfile(provider="p")
    ).resolve_model("p", "m")

    assert "image/png" in anthropic.accepts and "application/pdf" in anthropic.accepts
    assert "image/png" in openai.accepts
    assert "application/pdf" not in openai.accepts, "this wire needs the Files API (P7-03)"
    assert anthropic.max_attachment_bytes and openai.max_attachment_bytes
