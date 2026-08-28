"""`llm-openai-compatible` — the OpenAI chat-completions wire, DeepSeek included.

One adapter covers OpenAI, DeepSeek, and every gateway that speaks the same
shape, because the differences that matter are two fields:

* **`reasoning_content`** — DeepSeek streams thinking separately from text. pH
  has a distinct `reasoning` block for exactly this, so it is mapped rather than
  concatenated: folding thinking into visible text would make the transcript
  claim the model said things it was only considering.
* **`prompt_cache_hit_tokens`** — DeepSeek folds cache hits into
  `prompt_tokens`. pH's `TokenUsage` counts are **disjoint** (D15), so the
  cached part is subtracted out; leaving it in would bill every cache hit twice
  in pH's own accounting.

Tool calls stream as incremental `arguments` deltas, and that fidelity is kept:
`assistant/chunk` promises token-level replay, and an adapter that only emitted
completed calls would quietly make that promise false.

The credential is resolved **here and nowhere above** (I-3): the seam hands
around a `CredentialRef`, and only this function turns one into a header.

@module ph_app.adapters.openai_compatible
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ph.cordis import Context, plugin
from ph.llm.adapter import ResolvedModel
from ph.llm.types import (
    AttachmentRef,
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    ReasoningBlock,
    ReasoningDelta,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    UsageChunk,
    attachment_of,
    text_of,
)
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret
from ._media import load_media, media_pointer

__all__ = ["OpenAiCompatibleAdapter", "apply"]

log = logging.getLogger("ph_app.adapters.openai")


ACCEPTED_MEDIA: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "audio/wav",
    "audio/mpeg",
)
"""The default for `ProviderProfile.accepts` — the OpenAI baseline.

Images as `image_url` data URIs and audio as `input_audio`: two wire shapes for
two MIME families, which is the branching a per-medium block union would have
duplicated one layer up. PDFs are deliberately absent — this wire carries them
by `file_id` through the Files API (P7-03), and claiming them would produce a
request the provider rejects."""

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
"""The default per-attachment ceiling."""


class ProviderProfile(WireModel):
    """One route this adapter serves."""

    provider: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    context_window: int | None = None
    default_max_tokens: int | None = None
    accepts: tuple[str, ...] = ACCEPTED_MEDIA
    """MIME types this route takes as message content (P7-01).

    Per *profile*, which is the point: one `openai-compatible` row serves many
    routes — a hosted gateway, Groq, Together, a local llama.cpp — and they do
    not agree about media. A module constant made every one of them declare
    images and a 20 MB ceiling, so a text-only local model was promised
    capabilities it does not have. A route that takes none sets `accepts: []`."""
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES


class Config(WireModel):
    """Row config: the routes to register."""

    profiles: list[ProviderProfile] = field(default_factory=list)


def _is_overflow(body: str) -> bool:
    lowered = body.lower()
    return "context" in lowered and ("length" in lowered or "window" in lowered)


@dataclass(slots=True)
class OpenAiCompatibleAdapter:
    """Streams one OpenAI-compatible route."""

    ctx: Context
    profile: ProviderProfile
    http: HttpClient = field(default_factory=HttpClient)

    def _headers(self) -> dict[str, str]:
        secret = resolve_secret(self.ctx, self.profile.api_key_env, self.profile.provider)
        return {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}

    async def _body(self, options: GenerateOptions) -> dict[str, Any]:
        media = await load_media(self.ctx.get("attachments"), options.messages)
        messages: list[dict[str, Any]] = []
        if options.system:
            messages.append({"role": "system", "content": options.system})
        for message in options.messages:
            messages.extend(_to_openai(message, media))
        body: dict[str, Any] = {
            "model": options.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if options.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in options.tools
            ]
        if options.temperature is not None:
            body["temperature"] = options.temperature
        max_tokens = options.max_tokens or self.profile.default_max_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if options.stop:
            body["stop"] = list(options.stop)
        return body

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        state = _StreamState()
        async for _event, payload in self.http.stream_sse(
            f"{self.profile.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            json=await self._body(options),
            is_overflow=_is_overflow,
        ):
            for chunk in state.consume(payload):
                yield chunk
        for chunk in state.finish():
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(
            context_window=self.profile.context_window,
            default_max_tokens=self.profile.default_max_tokens,
            accepts=frozenset(self.profile.accepts),
            max_attachment_bytes=self.profile.max_attachment_bytes,
        )


@dataclass(slots=True)
class _StreamState:
    """Turns wire deltas into pH chunks, tracking block indexes."""

    text_index: int | None = None
    reasoning_index: int | None = None
    text: str = ""
    reasoning: str = ""
    tool_indexes: dict[int, int] = field(default_factory=dict)
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    next_index: int = 0
    usage: TokenUsage | None = None
    finish_reason: str | None = None

    def _claim(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index

    def consume(self, payload: dict[str, Any]) -> list[Any]:
        out: list[Any] = []
        raw_usage = payload.get("usage")
        if isinstance(raw_usage, dict):
            self.usage = _to_usage(raw_usage)
        for choice in payload.get("choices") or ():
            delta = choice.get("delta") or {}
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                self.finish_reason = reason

            thinking = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(thinking, str) and thinking:
                if self.reasoning_index is None:
                    self.reasoning_index = self._claim()
                    out.append(BlockStart(index=self.reasoning_index, block_type="reasoning"))
                self.reasoning += thinking
                out.append(ReasoningDelta(index=self.reasoning_index, text=thinking))

            content = delta.get("content")
            if isinstance(content, str) and content:
                if self.text_index is None:
                    self.text_index = self._claim()
                    out.append(BlockStart(index=self.text_index, block_type="text"))
                self.text += content
                out.append(TextDelta(index=self.text_index, text=content))

            for call in delta.get("tool_calls") or ():
                position = int(call.get("index", 0))
                if position not in self.tool_indexes:
                    self.tool_indexes[position] = self._claim()
                    self.tool_calls[position] = {"id": "", "name": "", "arguments": ""}
                    out.append(
                        BlockStart(index=self.tool_indexes[position], block_type="tool-call")
                    )
                record = self.tool_calls[position]
                if call.get("id"):
                    record["id"] = str(call["id"])
                function = call.get("function") or {}
                if function.get("name"):
                    record["name"] = str(function["name"])
                fragment = function.get("arguments") or ""
                record["arguments"] += fragment
                out.append(
                    ToolCallDelta(
                        index=self.tool_indexes[position],
                        id=record["id"] or f"call-{position}",
                        name=record["name"] or None,
                        arguments_delta=fragment,
                    )
                )
        return out

    def finish(self) -> list[Any]:
        out: list[Any] = []
        if self.reasoning_index is not None:
            out.append(
                BlockEnd(index=self.reasoning_index, block=ReasoningBlock(text=self.reasoning))
            )
        if self.text_index is not None:
            out.append(BlockEnd(index=self.text_index, block=TextBlock(text=self.text)))
        for position, index in self.tool_indexes.items():
            record = self.tool_calls[position]
            out.append(
                BlockEnd(
                    index=index,
                    block=ToolCallBlock(
                        id=record["id"] or f"call-{position}",
                        name=record["name"],
                        arguments=record["arguments"],
                    ),
                )
            )
        if self.usage is not None:
            # Usage precedes the terminal finish, and nothing follows finish.
            out.append(UsageChunk(usage=self.usage))
        out.append(Finish(reason=FinishReason(kind=_finish_kind(self.finish_reason))))
        return out


def _finish_kind(reason: str | None) -> Any:
    if reason == "tool_calls":
        return "tool-calls"
    if reason == "length":
        return "max-tokens"
    return "stop"


def _to_usage(raw: dict[str, Any]) -> TokenUsage:
    prompt = int(raw.get("prompt_tokens") or 0)
    cached = int(
        raw.get("prompt_cache_hit_tokens")
        or (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
        or 0
    )
    details = raw.get("completion_tokens_details") or {}
    return TokenUsage(
        # Disjoint counts (D15): DeepSeek folds cache hits into prompt_tokens,
        # so leaving them in would bill every hit twice in pH's accounting.
        input_tokens=max(0, prompt - cached),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_tokens=cached or None,
        reasoning_tokens=int(details.get("reasoning_tokens") or 0) or None,
    )


_AUDIO_FORMATS = {"audio/wav": "wav", "audio/mpeg": "mp3"}
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})


def _media_part(attachment: AttachmentRef, data: str) -> dict[str, Any] | None:
    """One attachment in this wire's shape, or `None` if it has none here.

    Total over its own vocabulary rather than a second copy of the accept policy:
    `media-degrade` decides what may be sent, but this renderer still has to be
    honest about what it can *express*. A PDF reaching here — this wire carries
    those by `file_id` (P7-03) — must not be dressed as an `image_url`.
    """
    audio = _AUDIO_FORMATS.get(attachment.mime)
    if audio is not None:
        return {"type": "input_audio", "input_audio": {"data": data, "format": audio}}
    if attachment.mime in _IMAGE_MIMES:
        return {"type": "image_url", "image_url": {"url": f"data:{attachment.mime};base64,{data}"}}
    return None


def _to_openai(message: Any, media: dict[str, str]) -> list[dict[str, Any]]:
    """One pH message as the wire's (sometimes several) messages.

    A user message becomes a *content list* as soon as it carries media, and a
    plain string otherwise — this wire accepts both, and keeping the string form
    for the overwhelmingly common case leaves every existing request byte-for-byte
    what it was, which is what the prefix cache is counting on (A12).
    """
    if message.role == "assistant":
        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": block.arguments},
            }
            for block in message.content
            if getattr(block, "type", "") == "tool-call"
        ]
        entry: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            entry["tool_calls"] = calls
        return [entry]
    results = [block for block in message.content if getattr(block, "type", "") == "tool-result"]
    if results:
        # A tool result is its own wire role, so it cannot be merged with text.
        return [
            {
                "role": "tool",
                "tool_call_id": block.tool_call_id,
                "content": text_of(block.content),
            }
            for block in results
        ]
    parts: list[dict[str, Any]] = []
    carries_media = False
    for block in message.content:
        attachment = attachment_of(block)
        if attachment is not None:
            carries_media = True
            data = media.get(attachment.attachment_id)
            part = None if data is None else _media_part(attachment, data)
            parts.append(part if part is not None else media_pointer(attachment))
        elif getattr(block, "type", "") == "text":
            parts.append({"type": "text", "text": block.text})
    if carries_media:
        # Keyed on the message *having* media, not on the rendered parts still
        # looking like media. A degraded attachment renders as a text part, so
        # asking "are any parts non-text" sent the list form only when the
        # attachment succeeded — and fell back to `text_of`, which reads
        # `TextBlock`s alone, for exactly the case the pointer exists to cover.
        # That is the silent drop this row is here to end, reintroduced one
        # branch lower down.
        return [{"role": "user", "content": parts}]
    return [{"role": "user", "content": text_of(message.content)}]


@plugin("llm-openai-compatible", config=Config, inject=["llm", "credentials"])
async def apply(ctx: Context, config: Config) -> None:
    """Register every configured OpenAI-compatible route."""
    for profile in config.profiles:
        adapter = OpenAiCompatibleAdapter(ctx=ctx, profile=profile)
        handle = ctx.llm.register_adapter([profile.provider], adapter)
        ctx.add_disposer(handle.dispose, label=f"llm({profile.provider})")
        ctx.add_disposer(adapter.http.aclose, label=f"http({profile.provider})")
