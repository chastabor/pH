"""`llm-anthropic` — the Anthropic messages wire.

A separate adapter rather than a flag on the OpenAI one, because the shapes
differ in kind and not in detail: system is a top-level field, content is always
a block list, tool results live in *user* messages, and usage arrives split
across `message_start` and `message_delta`. Pretending one adapter fits both
would mean a tangle of conditionals in the path that is hardest to test. What
the two wires genuinely share — the client, the credential, the status→code
table — lives in `_http`.

The two mappings worth naming:

* **thinking blocks** map to pH's `reasoning`, not to text — same reason as
  DeepSeek's `reasoning_content`: the transcript must not claim the model said
  what it was only considering;
* **`cache_creation_input_tokens` / `cache_read_input_tokens`** are already
  disjoint from `input_tokens` here, so they map across directly (unlike
  DeepSeek, which needs subtraction).

@module ph_app.adapters.anthropic
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ph.cordis import Context, plugin
from ph.llm.adapter import ResolvedModel
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    LlmFailure,
    ReasoningBlock,
    ReasoningDelta,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
)
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret

__all__ = ["AnthropicAdapter", "apply"]

log = logging.getLogger("ph_app.adapters.anthropic")

API_VERSION = "2023-06-01"


class Config(WireModel):
    """Row config for the Anthropic route."""

    provider: str = "anthropic"
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    context_window: int | None = 200_000
    default_max_tokens: int = 8_192


def _is_overflow(body: str) -> bool:
    # Only the provider's own phrasing: `max_tokens` also appears in a plain
    # bad-request body, and treating that as an overflow would trigger a
    # compaction for a request that fit fine.
    return "prompt is too long" in body.lower()


@dataclass(slots=True)
class AnthropicAdapter:
    """Streams the Anthropic messages API."""

    ctx: Context
    config: Config
    http: HttpClient = field(default_factory=HttpClient)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": resolve_secret(self.ctx, self.config.api_key_env, self.config.provider),
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }

    def _body(self, options: GenerateOptions) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or self.config.default_max_tokens,
            "stream": True,
            "messages": [_to_anthropic(message) for message in options.messages],
        }
        if options.system:
            body["system"] = options.system
        if options.tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in options.tools
            ]
        if options.temperature is not None:
            body["temperature"] = options.temperature
        if options.stop:
            body["stop_sequences"] = list(options.stop)
        return body

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        state = _StreamState()
        async for event, payload in self.http.stream_sse(
            f"{self.config.base_url.rstrip('/')}/messages",
            headers=self._headers(),
            json=self._body(options),
            is_overflow=_is_overflow,
        ):
            for chunk in state.consume(event, payload):
                yield chunk
        for chunk in state.finish():
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(
            context_window=self.config.context_window,
            default_max_tokens=self.config.default_max_tokens,
        )


@dataclass(slots=True)
class _Open:
    kind: str
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    arguments: str = ""


@dataclass(slots=True)
class _StreamState:
    blocks: dict[int, _Open] = field(default_factory=dict)
    usage: TokenUsage | None = None
    stop_reason: str | None = None

    def consume(self, event: str, payload: dict[str, Any]) -> list[Any]:
        out: list[Any] = []
        kind = payload.get("type") or event
        if kind == "message_start":
            raw = (payload.get("message") or {}).get("usage")
            if isinstance(raw, dict):
                self.usage = _to_usage(raw)
        elif kind == "content_block_start":
            index = int(payload.get("index", 0))
            block = payload.get("content_block") or {}
            block_type = str(block.get("type", "text"))
            if block_type == "thinking":
                self.blocks[index] = _Open(kind="reasoning")
                out.append(BlockStart(index=index, block_type="reasoning"))
            elif block_type == "tool_use":
                self.blocks[index] = _Open(
                    kind="tool-call",
                    tool_id=str(block.get("id", "")),
                    tool_name=str(block.get("name", "")),
                )
                out.append(BlockStart(index=index, block_type="tool-call"))
            else:
                self.blocks[index] = _Open(kind="text")
                out.append(BlockStart(index=index, block_type="text"))
        elif kind == "content_block_delta":
            index = int(payload.get("index", 0))
            delta = payload.get("delta") or {}
            open_block = self.blocks.get(index)
            if open_block is None:
                return out
            if delta.get("type") == "thinking_delta":
                fragment = str(delta.get("thinking", ""))
                open_block.text += fragment
                out.append(ReasoningDelta(index=index, text=fragment))
            elif delta.get("type") == "input_json_delta":
                fragment = str(delta.get("partial_json", ""))
                open_block.arguments += fragment
                out.append(
                    ToolCallDelta(
                        index=index,
                        id=open_block.tool_id or f"call-{index}",
                        name=open_block.tool_name or None,
                        arguments_delta=fragment,
                    )
                )
            else:
                fragment = str(delta.get("text", ""))
                open_block.text += fragment
                out.append(TextDelta(index=index, text=fragment))
        elif kind == "content_block_stop":
            index = int(payload.get("index", 0))
            open_block = self.blocks.pop(index, None)
            if open_block is not None:
                out.append(BlockEnd(index=index, block=_close(open_block, index)))
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta.get("stop_reason"), str):
                self.stop_reason = delta["stop_reason"]
            raw = payload.get("usage")
            if isinstance(raw, dict):
                self.usage = _merge_usage(self.usage, raw)
        elif kind == "error":
            detail = (payload.get("error") or {}).get("message", "provider error")
            failure = LlmFailure(message=str(detail), code="PROVIDER_ERROR")
            out.append(Finish(reason=FinishReason(kind="error", failure=failure)))
        return out

    def finish(self) -> list[Any]:
        out: list[Any] = []
        for index, open_block in list(self.blocks.items()):
            out.append(BlockEnd(index=index, block=_close(open_block, index)))
        self.blocks.clear()
        if self.usage is not None:
            out.append(UsageChunk(usage=self.usage))
        out.append(Finish(reason=FinishReason(kind=_finish_kind(self.stop_reason))))
        return out


def _close(open_block: _Open, index: int) -> Any:
    if open_block.kind == "reasoning":
        return ReasoningBlock(text=open_block.text)
    if open_block.kind == "tool-call":
        return ToolCallBlock(
            id=open_block.tool_id or f"call-{index}",
            name=open_block.tool_name,
            arguments=open_block.arguments or "{}",
        )
    return TextBlock(text=open_block.text)


def _finish_kind(stop_reason: str | None) -> Any:
    if stop_reason == "tool_use":
        return "tool-calls"
    if stop_reason == "max_tokens":
        return "max-tokens"
    return "stop"


def _to_usage(raw: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        # Already disjoint on this wire, so no subtraction (unlike DeepSeek).
        input_tokens=int(raw.get("input_tokens") or 0),
        output_tokens=int(raw.get("output_tokens") or 0),
        cache_read_tokens=int(raw.get("cache_read_input_tokens") or 0) or None,
        cache_write_tokens=int(raw.get("cache_creation_input_tokens") or 0) or None,
    )


def _merge_usage(current: TokenUsage | None, raw: dict[str, Any]) -> TokenUsage:
    incoming = _to_usage(raw)
    if current is None:
        return incoming
    return TokenUsage(
        input_tokens=incoming.input_tokens or current.input_tokens,
        output_tokens=incoming.output_tokens or current.output_tokens,
        cache_read_tokens=incoming.cache_read_tokens or current.cache_read_tokens,
        cache_write_tokens=incoming.cache_write_tokens or current.cache_write_tokens,
    )


def _to_anthropic(message: Any) -> dict[str, Any]:
    """One pH message as an Anthropic message.

    Tool results are user-role content blocks here rather than their own role,
    which is the main structural difference from the OpenAI wire.
    """
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        kind = getattr(block, "type", "")
        if kind == "text":
            blocks.append({"type": "text", "text": block.text})
        elif kind == "reasoning":
            blocks.append({"type": "thinking", "thinking": block.text})
        elif kind == "tool-call":
            try:
                parsed = json.loads(block.arguments) if block.arguments else {}
            except json.JSONDecodeError:
                parsed = {}
            blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": parsed})
        elif kind == "tool-result":
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_call_id,
                    "content": [
                        {"type": "text", "text": inner.text}
                        for inner in block.content
                        if getattr(inner, "type", "") == "text"
                    ],
                    **({"is_error": True} if block.is_error else {}),
                }
            )
    role = "assistant" if message.role == "assistant" else "user"
    return {"role": role, "content": blocks or [{"type": "text", "text": ""}]}


@plugin("llm-anthropic", config=Config, inject=["llm", "credentials"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the Anthropic route."""
    adapter = AnthropicAdapter(ctx=ctx, config=config)
    handle = ctx.llm.register_adapter([config.provider], adapter)
    ctx.add_disposer(handle.dispose, label=f"llm({config.provider})")
    ctx.add_disposer(adapter.http.aclose, label=f"http({config.provider})")
