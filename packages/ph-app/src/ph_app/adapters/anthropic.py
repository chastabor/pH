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
  DeepSeek, which needs subtraction);
* **`cache_control` markers** — Anthropic's caching is opt-in, and every
  prefix-stability decision above this file (A12, P4-03's replayed summarize
  envelope) pays off on the OpenAI-compatible routes, which cache implicitly, and
  nowhere here until the markers are sent. `_checkpoints` is where they go and
  why.

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
from ._media import load_media, media_pointer

__all__ = ["AnthropicAdapter", "apply"]

log = logging.getLogger("ph_app.adapters.anthropic")

API_VERSION = "2023-06-01"


ACCEPTED_MEDIA: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/pdf",
)
"""The default for `Config.accepts` — what this provider's own models take.

A PDF rides a `document` block rather than an `image` one, which is the whole
reason `MediaBlock` is keyed on MIME and the branching lives in this file
instead of in the content-block union."""

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
"""The default per-attachment ceiling, declared so a caller can act on it."""

CACHE_BREAKPOINTS = 4
"""How many blocks one request may mark `cache_control`. Anthropic's limit, and
the whole design constraint: placing four markers well is the work.

The budget is spent `tools` (1) + `system` (1) + two message checkpoints (2),
which is all four. A fifth marker is a request error, so anything added here has
to take a slot from something else."""

CHECKPOINT_EVERY = 4
"""How often the moving message breakpoint advances, in messages.

**Why a breakpoint does not go on the last message.** A cache *read* happens
where a request marks a prefix that an earlier request also marked; a mark on
"the newest message" is at a different index in every request, so it is written
once and never read again — a slot spent on a prefix nobody asks for. Quantizing
makes consecutive requests agree: with a step of four, requests whose message
count is 5, 6, 7 and 8 all mark index 4, so the second and every later one read
what the first wrote.

Two of them, `latest` and the one before it, because a single quantized mark goes
dark exactly when it advances — the request that first marks index 8 would carry
no mark at 4 and re-read nothing. Keeping the previous checkpoint means the step
is paid once, not on every request that crosses a boundary.

Four is the smallest step that still spans an assistant turn (a user message, an
assistant message with a tool call, a tool result, the next assistant message),
so a checkpoint rarely lands mid-turn. The cost of a larger step is a longer
uncached tail; the cost of a smaller one is a shorter-lived cache entry."""

MIN_CACHEABLE_TOKENS = 1024
"""Anthropic's floor for a cache *write* (2048 on the smaller models).

Not enforced here, and worth saying why: a marker on a shorter prefix is ignored
rather than refused, so the cost of marking `tools` in a deployment with two
small tools is nothing, while a check would need a tokenizer this adapter does
not have and would disagree with the provider's."""

_WIRE_SHAPES: dict[str, str] = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "application/pdf": "document",
}
"""MIME → this wire's block type. A PDF is a `document`, not an `image`, which is
the branching that belongs in an adapter rather than in the block union."""


class Config(WireModel):
    """Row config for the Anthropic route."""

    provider: str = "anthropic"
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    context_window: int | None = 200_000
    default_max_tokens: int = 8_192
    accepts: tuple[str, ...] = ACCEPTED_MEDIA
    """MIME types this route takes as message content (P7-01).

    Row config beside `context_window`, not a module constant, for the same
    reason that one is: what a *route* can do is a property of the route, and
    this adapter serves whatever `base_url` a deployment points it at. A gateway
    in front of a text-only model sets `accepts: []` and gets honest pointers
    instead of requests the far end rejects."""
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES
    cache_control: bool = True
    """Send `cache_control` breakpoints (P6-13).

    Row config for `accepts`' reason: this adapter serves whatever `base_url`
    points at, and a gateway that does not implement Anthropic's caching rejects
    the field rather than ignoring it. Default on, because the shipped route is
    Anthropic's own and the markers are what make A12 worth anything there."""


def _breakpoint() -> dict[str, str]:
    """One `cache_control` marker. A fresh dict per call: these are handed to a
    JSON encoder inside a body a caller may edit, and a shared literal would make
    two markers one object."""
    return {"type": "ephemeral"}


def _checkpoints(count: int) -> tuple[int, ...]:
    """Which message indices carry a breakpoint, oldest first (P6-13).

    Quantized to `CHECKPOINT_EVERY` so that consecutive requests mark the *same*
    index — which is the only way a marker is ever read rather than merely
    written — and two of them so the cache does not go dark on the request where
    the checkpoint advances. `CHECKPOINT_EVERY` carries the reasoning.

    The empty tuple when there are no messages: nothing to mark beats an index
    guessed for a list that has none. Every shape that reaches this has at least
    one — compaction's `direct` sends a single instruction message — so the guard
    is for the caller this file does not know about.
    """
    if count <= 0:
        return ()
    latest = ((count - 1) // CHECKPOINT_EVERY) * CHECKPOINT_EVERY
    return tuple(index for index in (latest - CHECKPOINT_EVERY, latest) if index >= 0)


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

    async def _body(self, options: GenerateOptions) -> dict[str, Any]:
        media = await load_media(self.ctx.get("attachments"), options.messages)
        caching = self.config.cache_control
        messages = [_to_anthropic(message, media) for message in options.messages]
        if caching:
            for index in _checkpoints(len(messages)):
                # On the message's last block, which is what "cache everything
                # through here" means on this wire. `_to_anthropic` never returns
                # an empty content list, so there is always a block to mark.
                messages[index]["content"][-1]["cache_control"] = _breakpoint()
        body: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or self.config.default_max_tokens,
            "stream": True,
            "messages": messages,
        }
        if options.system:
            # A block list rather than the plain string, because only a block can
            # carry a marker. Left as a string when caching is off, so a gateway
            # that never wanted any of this sees the shape it always saw.
            body["system"] = (
                [{"type": "text", "text": options.system, "cache_control": _breakpoint()}]
                if caching
                else options.system
            )
        if options.tools:
            tools: list[dict[str, Any]] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters or {"type": "object", "properties": {}},
                }
                for tool in options.tools
            ]
            if caching:
                # Its own breakpoint rather than relying on `system`'s, and it
                # earns the slot: the wire order is tools → system → messages, so
                # a system prompt that changes mid-session — pH composes a `todos`
                # section into it whenever the list changes — invalidates every
                # marker after it. The tools survive that; without this they would
                # not.
                tools[-1]["cache_control"] = _breakpoint()
            body["tools"] = tools
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
            json=await self._body(options),
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
            accepts=frozenset(self.config.accepts),
            max_attachment_bytes=self.config.max_attachment_bytes,
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


def _to_anthropic(message: Any, media: dict[str, str]) -> dict[str, Any]:
    """One pH message as an Anthropic message.

    Tool results are user-role content blocks here rather than their own role, which
    is the main structural difference from the OpenAI wire.

    `media` holds base64 for the attachments this route will take; anything absent
    from it becomes a pointer. **Nothing is dropped** — a block kind this function
    does not recognise still reaches the wire, rather than a message that was only an
    image arriving as an empty text block.
    """
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        kind = getattr(block, "type", "")
        if kind == "text":
            blocks.append({"type": "text", "text": block.text})
        elif kind == "media":
            data = media.get(block.attachment.attachment_id)
            shape = _WIRE_SHAPES.get(block.attachment.mime)
            # Total over its own vocabulary, not a second copy of the accept
            # policy: `media-degrade` decides what may be sent, but this renderer
            # still has to be honest about what it can *express*. Without the
            # fallback, a MIME it has no shape for would be dressed as an image.
            if data is None or shape is None:
                blocks.append(media_pointer(block.attachment))
            else:
                blocks.append(
                    {
                        "type": shape,
                        "source": {
                            "type": "base64",
                            "media_type": block.attachment.mime,
                            "data": data,
                        },
                    }
                )
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
