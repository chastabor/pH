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

## Why the Anthropic block conversion has no per-kind branch list

The previous version built branches for four block kinds and **silently omitted
everything else**, so a message that was only an image reached the wire as an empty
text block. Nothing is dropped now: a block kind the converter does not recognise
still arrives.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph.cordis import Context
from ph.keys import AGENTS, ATTACHMENTS, SESSIONS
from ph.llm.adapter import LlmError, MediaRoute
from ph.llm.assembler import BlockAssembler
from ph.llm.types import (
    GenerateOptions,
    MediaBlock,
    ToolCallBlock,
    ToolSchema,
    create_tool_result_message,
    create_user_message,
)
from ph.seams.credentials import CredentialService
from ph.session.json import as_obj
from ph.testing import as_kind, block_text
from ph_app.adapters._http import failure_from_status
from ph_app.adapters.anthropic import (
    CACHE_BREAKPOINTS,
    CHECKPOINT_EVERY,
    AnthropicAdapter,
    _checkpoints,
)
from ph_app.adapters.anthropic import Config as AnthropicConfig
from ph_app.adapters.google import Config as GoogleConfig
from ph_app.adapters.google import GoogleAdapter
from ph_app.adapters.openai_compatible import (
    OpenAiCompatibleAdapter,
    ProviderProfile,
    WindowProbe,
    _StreamState,
    _to_openai,
    _to_usage,
    discover_window,
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


_PROBE = WindowProbe(path="/props", field=("default_generation_settings", "n_ctx"))
"""llama.cpp's, which is the one this ships configured for."""


_VLLM_PROBE = WindowProbe(path="/v1/models", field=("data", 0, "max_model_len"))
"""vLLM's, the second entry the shipped profile now carries."""


class _ServerStub:
    """A server that publishes some endpoints, 404s the rest, and can be slow.

    The one `HttpClient` double in this file. It began as a pair — one that
    answered every URL with the same payload, one that answered by URL — and the
    first was a strict subset: a probe that reads one field off one endpoint is
    `_ServerStub({url: payload})`, and "the server is not there" is
    `_ServerStub({})`, since `_ask` treats every exception alike.

    `slow` is what makes declared order testable. The probes are asked
    concurrently now, so a stub that replied instantly to both would pass
    whichever way the winner was chosen; delaying the *preferred* endpoint is
    what tells index selection apart from first-past-the-post.
    """

    def __init__(self, published: dict[str, Any], *, slow: frozenset[str] = frozenset()) -> None:
        self._published = published
        self._slow = slow
        self.asked: list[str] = []

    async def get_json(self, url: str, **_: Any) -> dict[str, Any]:
        self.asked.append(url)
        if url in self._slow:
            await anyio.sleep(0.05)
        answer = self._published.get(url)
        if answer is None:
            raise OSError(f"404 {url}")
        return dict(answer)


async def test_the_window_is_asked_of_the_server_when_the_route_says_to() -> None:
    """`contextWindowProbe` — the one number a person cannot keep right.

    llama.cpp divides `--ctx-size` by `--parallel` and reports the quotient, so
    a profile that copies the server's total budgets every agent against the
    whole KV cache. Asking removes the arithmetic rather than documenting it.
    """
    adapter = OpenAiCompatibleAdapter(
        ctx=None,  # type: ignore[arg-type]
        profile=ProviderProfile(
            provider="llama", base_url="http://server/v1", context_window_probes=(_PROBE,)
        ),
    )
    adapter.http = _ServerStub(  # type: ignore[assignment]
        {"http://server/props": {"default_generation_settings": {"n_ctx": 262_144}}}
    )

    assert await discover_window(adapter) == (262_144, "/props")
    # Beside `/v1`, not under it: `/props` is llama.cpp's own endpoint and the
    # OpenAI wire's prefix does not reach it.
    assert adapter.http.asked == ["http://server/props"]  # type: ignore[attr-defined]


async def test_a_probe_step_can_index_a_list() -> None:
    """`data[0].max_model_len` — the shape a bare key path could not reach.

    The first version walked objects only, so a server answering with an array
    was inexpressible and the second backend would have edited this file after
    all, which is the whole thing the declarative probe exists to avoid.
    Verified against a live `/v1/models` as well as here.
    """
    adapter = OpenAiCompatibleAdapter(
        ctx=None,  # type: ignore[arg-type]
        profile=ProviderProfile(
            provider="vllm",
            base_url="http://server/v1",
            context_window_probes=(
                WindowProbe(path="/v1/models", field=("data", 0, "max_model_len")),
            ),
        ),
    )
    adapter.http = _ServerStub(  # type: ignore[assignment]
        {"http://server/v1/models": {"data": [{"max_model_len": 131_072}]}}
    )

    assert await discover_window(adapter) == (131_072, "/v1/models")
    assert adapter.http.asked == ["http://server/v1/models"]  # type: ignore[attr-defined]


async def test_the_preferred_probe_wins_even_when_it_answers_last() -> None:
    """Order is the trust ordering, and it is decided by position, not by speed.

    llama.cpp publishes both of the shipped probes, and its `/v1/models` reports
    the server's **undivided** `n_ctx` — the number `/props` exists to correct.
    So a server answering both must be read at `/props` however the replies
    arrive; `/props` is delayed here precisely so that a first-past-the-post
    race would return the wrong window and fail this.

    Both are asked, which is the trade for asking them at once: sequentially a
    hung endpoint cost `PROBE_TIMEOUT` per probe out of a mount, and a mount is
    a person waiting for a TUI to open.
    """
    adapter = OpenAiCompatibleAdapter(
        ctx=None,  # type: ignore[arg-type]
        profile=ProviderProfile(
            provider="llama",
            base_url="http://server/v1",
            context_window_probes=(_PROBE, _VLLM_PROBE),
        ),
    )
    adapter.http = _ServerStub(  # type: ignore[assignment]
        {
            "http://server/props": {"default_generation_settings": {"n_ctx": 262_144}},
            "http://server/v1/models": {"data": [{"max_model_len": 1_572_864}]},
        },
        slow=frozenset({"http://server/props"}),
    )

    assert await discover_window(adapter) == (262_144, "/props")


async def test_a_probe_the_server_does_not_publish_falls_through_to_the_next() -> None:
    """The reason the field is a list: one `baseUrl` outlives one server.

    `LLAMA_BASE_URL` names a port. Swap llama.cpp for vLLM behind it and `/props`
    is a 404 — which, with a single probe, left the profile's 32768 fallback in
    force against a real window of 262144 and said nothing.
    """
    adapter = OpenAiCompatibleAdapter(
        ctx=None,  # type: ignore[arg-type]
        profile=ProviderProfile(
            provider="llama",
            base_url="http://server/v1",
            context_window=32_768,
            context_window_probes=(_PROBE, _VLLM_PROBE),
        ),
    )
    adapter.http = _ServerStub(  # type: ignore[assignment]
        {"http://server/v1/models": {"data": [{"max_model_len": 262_144}]}}
    )

    assert await discover_window(adapter) == (262_144, "/v1/models")
    assert sorted(adapter.http.asked) == [  # type: ignore[attr-defined]
        "http://server/props",
        "http://server/v1/models",
    ]


@pytest.mark.parametrize(
    "props",
    [
        {},
        {"default_generation_settings": {}},
        {"default_generation_settings": {"n_ctx": "many"}},
        {"default_generation_settings": {"n_ctx": 0}},
        {"default_generation_settings": []},
    ],
    ids=["no-settings", "no-n_ctx", "not-a-number", "zero", "wrong-shape"],
)
async def test_a_server_that_will_not_say_leaves_the_configured_window(props: Any) -> None:
    """Every way of not answering is one answer, because the caller does one
    thing with all of them: keep what the profile said.

    Zero among them — a window of nothing is not a window, and publishing it
    would make every request overflow before it was built."""
    adapter = OpenAiCompatibleAdapter(
        ctx=None,  # type: ignore[arg-type]
        profile=ProviderProfile(
            provider="p",
            base_url="http://server/v1",
            context_window=32_768,
            context_window_probes=(_PROBE,),
        ),
    )
    adapter.http = _ServerStub({"http://server/props": props})  # type: ignore[assignment]

    assert await discover_window(adapter) is None


async def test_a_server_that_is_not_there_is_not_an_error() -> None:
    """A mount must not fail over this. A window pH guessed low costs an early
    compaction; a mount that refused costs the person the session."""
    adapter = OpenAiCompatibleAdapter(
        ctx=None,  # type: ignore[arg-type]
        profile=ProviderProfile(
            provider="p", base_url="http://server/v1", context_window_probes=(_PROBE,)
        ),
    )
    adapter.http = _ServerStub({})  # type: ignore[assignment]

    assert await discover_window(adapter) is None


def test_discovery_and_its_fallback_are_two_fields() -> None:
    """Ask, and here is what to use when the answer does not come.

    Two facts, so two fields — a single value cannot carry both, which is what a
    `contextWindow: auto` keyword tried to do before it was removed: it said
    "discover" by *erasing* the fallback, and being adapter-local it made the
    same word a validation error on the two routes that share `MediaRoute`.
    """
    both = ProviderProfile.model_validate(
        {"provider": "p", "contextWindow": 32_768, "contextWindowProbes": [_PROBE.to_wire()]}
    )
    assert (both.context_window_probes, both.context_window) == ((_PROBE,), 32_768)

    # Discovery with no fallback is the probe and an unset window.
    alone = ProviderProfile.model_validate(
        {"provider": "p", "contextWindowProbes": [_PROBE.to_wire()]}
    )
    assert (alone.context_window_probes, alone.context_window) == ((_PROBE,), None)


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
    assert block_text(blocks[0]) == "let me think"
    assert block_text(blocks[1]) == "the answer"


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
    assert as_kind(call, ToolCallBlock).name == "read"
    assert json.loads(as_kind(call, ToolCallBlock).arguments) == {"path": "a"}
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


def test_vllm_cache_writes_are_subtracted_out_too() -> None:
    """A warm turn and a cold one, with the same arithmetic on each.

    vLLM folds both counts into `prompt_tokens`, so a write left inside it is
    added twice by `TokenUsage.total` — 784 of 1256 tokens on the measured turn,
    a 62% over-report of the window's occupancy, which is what decides when
    compaction fires.
    """
    cold = _to_usage(
        {
            "prompt_tokens": 1_256,
            "prompt_tokens_details": {"cached_tokens": 0, "created_cache_tokens": 784},
            "completion_tokens": 8,
        }
    )
    assert (cold.input_tokens, cold.cache_read_tokens, cold.cache_write_tokens) == (472, None, 784)
    assert cold.total == 1_264

    warm = _to_usage(
        {
            "prompt_tokens": 1_257,
            "prompt_tokens_details": {"cached_tokens": 784, "created_cache_tokens": 0},
            "completion_tokens": 8,
        }
    )
    assert (warm.input_tokens, warm.cache_read_tokens, warm.cache_write_tokens) == (473, 784, None)
    assert warm.total == 1_265


def test_a_deepseek_cache_miss_is_not_read_as_a_write() -> None:
    """`prompt_cache_miss_tokens` is the uncached remainder, which is what
    `input_tokens` already means — read as a write it would zero the input and
    call the entire prompt a cache write."""
    usage = _to_usage(
        {
            "prompt_tokens": 1_000,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
            "completion_tokens": 20,
        }
    )
    assert (usage.input_tokens, usage.cache_write_tokens) == (200, None)


def test_a_length_finish_becomes_max_tokens() -> None:
    state = _StreamState()
    state.consume({"choices": [{"delta": {"content": "cut off"}, "finish_reason": "length"}]})
    finish = state.finish()[-1]
    assert finish.reason.kind == "max-tokens"


def test_tool_results_become_their_own_wire_role() -> None:
    message = create_tool_result_message(
        call_id="c1", content=[{"type": "text", "text": "output"}], is_error=False
    )
    (entry,) = _to_openai(message, {}, {})
    # A tool result cannot be merged into a user message on this wire.
    assert entry["role"] == "tool"
    assert entry["tool_call_id"] == "c1"
    assert entry["content"] == "output"


def test_a_plain_user_message_stays_a_user_message() -> None:
    message = create_user_message(
        content=[{"type": "text", "text": "hello"}], source={"kind": "user"}
    )
    assert _to_openai(message, {}, {}) == [{"role": "user", "content": "hello"}]


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


async def test_a_response_schema_reaches_the_wire_as_a_constraint() -> None:
    """P7-17: `strict`, or the field is a hint the server may ignore.

    The shape is the provider's, so it is asserted exactly rather than by
    substring — a `json_schema` nested one level wrong is accepted by the API and
    silently constrains nothing, which is the failure the caller's own validation
    then has to catch on every reply forever.
    """
    root = Context()
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}

    body, _handles = await adapter._body(
        GenerateOptions(
            provider="p",
            model="m",
            messages=(
                create_user_message(
                    content=[{"type": "text", "text": "?"}], source={"kind": "user"}
                ),
            ),
            response_schema=schema,
        )
    )

    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "reply", "schema": schema, "strict": True},
    }
    assert adapter.resolve_model("p", "m").structured_output is True, "and the route says so"


def test_a_route_with_no_wire_support_says_so_rather_than_defaulting_quietly() -> None:
    """Anthropic has no `response_format`, so a caller there gets the instruction,
    the validation and the retry — and not the wire's guarantee (P7-17).

    Asserted rather than left to the default, because the whole point of the flag
    is that a caller can tell the two apart; a route that silently reported
    enforcement would send `ask_for_shape` down its shorter attempt budget and
    log an adapter bug that is not one.
    """
    adapter = AnthropicAdapter(ctx=Context(), config=AnthropicConfig())

    assert adapter.resolve_model("anthropic", "claude").structured_output is False


async def test_a_request_with_no_schema_asks_for_no_format() -> None:
    """The field is absent, not null: a provider that sees `response_format: null`
    may reject the call outright."""
    root = Context()
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))

    body, _handles = await adapter._body(GenerateOptions(provider="p", model="m", messages=()))

    assert "response_format" not in body


async def test_the_openai_request_body_carries_tools_and_the_system_slot() -> None:
    root = Context()
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))
    body, _handles = await adapter._body(
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
    assert block_text(blocks[0]) == "hmm"
    assert as_kind(blocks[1], ToolCallBlock).name == "read"
    assert assembler.finish.kind == "tool-calls"
    assert assembler.usage is not None and assembler.usage.output_tokens == 12


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"), reason="no DEEPSEEK_API_KEY; smoke test skipped"
)
async def test_real_api_smoke(tmp_path: Any, monkeypatch: Any) -> None:  # pragma: no cover
    """P1-15's gate: one real round trip, skipped without a key."""
    from ph.agent.types import AgentOptions
    from ph_app.profiles import compose_profile
    from ph_app.runtime import mounted

    monkeypatch.setenv("PH_HOME", str(tmp_path))
    async with mounted(compose_profile("deepseek")) as ctx:
        session = ctx.require(SESSIONS).create("smoke")
        agent = ctx.require(AGENTS).create(
            session, AgentOptions(provider="deepseek", model="deepseek-chat", max_tokens=64)
        )
        await agent.prompt("Reply with the single word: ok")
        assert as_obj(session.events[-1].data["reason"])["kind"] == "completed"
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

    body, _handles = await adapter._body(
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

    body, _handles = await adapter._body(
        GenerateOptions(provider="anthropic", model="m", messages=(_media_message(ref),))
    )

    (entry,) = body["messages"]
    assert any(block["type"] == "document" for block in entry["content"])


def test_every_route_capability_reaches_resolve_model() -> None:
    """One projection, so a capability cannot reach two adapters and miss the third.

    Six lines mapping a config field onto the identically-named `ResolvedModel`
    field had been written out in each adapter. The cost was not the lines: a
    seventh capability was three edits, and an adapter that missed one reports the
    *default* rather than the route's number — after which `media-degrade` applies
    the wrong rule for that provider alone, silently, because a default is a legal
    value.

    Driven through **`resolve_model`**, not through `resolved` — asserting on the
    projection alone would leave the thing it exists to prevent untested, since an
    adapter that regressed to a hand-rolled `ResolvedModel(...)` missing a field
    would still pass. All three are here because `google`'s had no test at all.

    The field list comes from `MediaRoute`'s own members rather than being written
    out here, which would have been the fourth copy of the thing that drifted.
    """
    carried = {name for name in vars(MediaRoute) if not name.startswith("_")}
    routes: list[tuple[Any, Any]] = [
        (AnthropicConfig(accepts=("image/png",), max_attachment_bytes=99), AnthropicAdapter),
        (ProviderProfile(provider="p", accepts=("image/png",), max_attachment_bytes=99), None),
        (GoogleConfig(accepts=("image/png",), max_attachment_bytes=99), GoogleAdapter),
    ]
    for route, adapter_type in routes:
        adapter = (
            OpenAiCompatibleAdapter(ctx=Context(), profile=route)
            if adapter_type is None
            else adapter_type(ctx=Context(), config=route)
        )
        model = adapter.resolve_model("p", "m")
        for name in carried:
            declared = getattr(route, name)
            assert getattr(model, name) == (
                frozenset(declared) if name == "accepts" else declared
            ), f"{type(route).__name__} declared {name} and it did not reach the resolved model"


async def test_a_renderer_with_no_shape_for_a_block_says_so(tmp_path: Path) -> None:
    """The renderer is total over its own wire vocabulary.

    Whether a route *may* send a video is `media-degrade`'s question, one layer
    up. This is the narrower one an adapter still owes: it has no `image_url` or
    `input_audio` meaning for one, so it must not dress it as either. What must
    never happen is the block vanishing.

    **This used to be asserted with a PDF, and P7-03's second half made that
    wrong.** The wire has a `file` part with two spellings — a `file_id` for an
    uploaded document and inline `file_data` for one that was not — so a PDF is
    now expressible here and rendering it as a pointer would be the silent
    downgrade, not the honest refusal. Video is the MIME this wire genuinely has
    no shape for, which is the claim the test was making all along.
    """
    root = Context()
    ref = await _with_attachment(root, tmp_path, "video/mp4")
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))

    body, _handles = await adapter._body(
        GenerateOptions(provider="p", model="m", messages=(_media_message(ref),))
    )

    rendered = json.dumps(body["messages"])
    assert "video/mp4" in rendered and "shot.png" in rendered
    assert "was not sent" in rendered
    assert "input_audio" not in rendered and "image_url" not in rendered


async def test_the_openai_wire_sends_a_pdf_inline_when_it_was_not_uploaded(
    tmp_path: Path,
) -> None:
    """What makes `load_handles`' fallback contract true on this wire (P7-03).

    An upload that fails for any reason leaves the id out and the attachment goes
    inline — that is the promise the seam makes, and it can only be kept by a
    renderer with an inline spelling. Without one, a route whose file API was
    momentarily down would degrade a document it can perfectly well send, which is
    a worse outcome than the one the upload exists to improve on.
    """
    root = Context()
    ref = await _with_attachment(root, tmp_path, "application/pdf")
    adapter = OpenAiCompatibleAdapter(ctx=root, profile=ProviderProfile(provider="p"))

    body, _handles = await adapter._body(
        GenerateOptions(provider="p", model="m", messages=(_media_message(ref),))
    )

    (entry,) = body["messages"]
    (part,) = [block for block in entry["content"] if block["type"] == "file"]
    assert part["file"]["filename"] == "shot.png"
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert "file_id" not in part["file"], "nothing uploaded it"


async def test_a_blob_that_is_gone_degrades_rather_than_failing(tmp_path: Path) -> None:
    """A session copied without its attachments still opens and still runs."""
    root = Context()
    ref = await _with_attachment(root, tmp_path, "image/png")
    root.require(ATTACHMENTS).path_for(ref).unlink()
    adapter = AnthropicAdapter(ctx=root, config=AnthropicConfig())

    body, _handles = await adapter._body(
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

    body, _handles = await adapter._body(
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


# ------------------------------------------------------- P6-13 cache markers --


def _markers(body: dict[str, Any]) -> list[str]:
    """Where this body carries `cache_control`, as readable labels."""
    found: list[str] = []
    for index, tool in enumerate(body.get("tools") or []):
        if "cache_control" in tool:
            found.append(f"tools[{index}]")
    system = body.get("system")
    if isinstance(system, list):
        found.extend(f"system[{i}]" for i, block in enumerate(system) if "cache_control" in block)
    for index, message in enumerate(body.get("messages") or []):
        for inner, block in enumerate(message["content"]):
            if "cache_control" in block:
                found.append(f"messages[{index}].content[{inner}]")
    return found


def _options(count: int, *, tools: int = 2, system: str | None = "sys") -> GenerateOptions:
    return GenerateOptions(
        provider="anthropic",
        model="m",
        messages=tuple(
            create_user_message(
                content=[{"type": "text", "text": f"turn {i}"}], source={"kind": "user"}
            )
            for i in range(count)
        ),
        system=system,
        tools=tuple(ToolSchema(name=f"t{i}", description="d", parameters={}) for i in range(tools)),
    )


async def test_cache_breakpoints_land_on_the_stable_boundaries() -> None:
    """P6-13. The wire order is tools → system → messages, and so is the budget.

    Anthropic's caching is opt-in: before this, every prefix-stability decision
    in the harness paid off on the routes that cache implicitly and nowhere here.
    The markers go on the *last* tool and the system block — one prefix each —
    and on two quantized message checkpoints.
    """
    adapter = AnthropicAdapter(ctx=Context(), config=AnthropicConfig())

    body, _handles = await adapter._body(_options(9))

    assert _markers(body) == [
        "tools[1]",  # the last tool: everything through the tool list
        "system[0]",
        "messages[4].content[0]",
        "messages[8].content[0]",
    ]
    # The system prompt has to be a block to carry one, so the string form goes.
    assert body["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]


async def test_the_breakpoint_budget_is_never_exceeded() -> None:
    """Four is Anthropic's limit, and a fifth marker is a request error.

    Asserted across the shapes a session actually sends — with and without
    tools, with and without a system prompt, at every length up to several turns
    — because the budget is spent by four independent decisions and nothing else
    would notice a fifth.
    """
    adapter = AnthropicAdapter(ctx=Context(), config=AnthropicConfig())

    for count in range(1, 24):
        for tools in (0, 3):
            for system in ("sys", None):
                body, _handles = await adapter._body(_options(count, tools=tools, system=system))
                found = _markers(body)
                assert len(found) <= CACHE_BREAKPOINTS, (count, tools, system, found)


def test_consecutive_requests_mark_a_checkpoint_in_common() -> None:
    """The property the whole placement exists for, and the one a naive one fails.

    A marker is *read* only where a request marks a prefix an earlier request
    also marked. Marking "the newest message" puts the marker at a different
    index every time, so it is written once and never read — the slot buys
    nothing. Quantizing makes consecutive requests agree.

    Checked for every way a conversation grows: a turn adds a user message and an
    assistant message, a tool round trip adds one at a time, so any step from one
    to `CHECKPOINT_EVERY` messages has to keep a checkpoint in common.
    """
    for count in range(1, 60):
        for step in range(1, CHECKPOINT_EVERY + 1):
            shared = set(_checkpoints(count)) & set(_checkpoints(count + step))
            assert shared, f"{count} → {count + step} shares no checkpoint"


def test_a_checkpoint_survives_the_request_that_advances_it() -> None:
    """Why there are two, not one.

    A single quantized marker goes dark exactly when it moves: the first request
    to mark index 8 would carry nothing at index 4, so the conversation it just
    cached would be re-read from scratch. The previous checkpoint is what makes
    the step cost one request's tail instead of the whole prefix.
    """
    # 9 messages is the first request whose latest checkpoint is 8.
    assert _checkpoints(8) == (0, 4)
    assert _checkpoints(9) == (4, 8)
    assert _checkpoints(1) == (0,), "a first request marks what it can"
    assert _checkpoints(0) == (), "and an empty conversation marks nothing"


async def test_caching_off_sends_the_shapes_the_route_always_sent() -> None:
    """A gateway that does not implement this rejects the field rather than ignoring it.

    So `cacheControl: false` is not "mark nothing" — it is the body from before
    P6-13, system string included.
    """
    adapter = AnthropicAdapter(ctx=Context(), config=AnthropicConfig(cache_control=False))

    body, _handles = await adapter._body(_options(9))

    assert _markers(body) == []
    assert body["system"] == "sys"
