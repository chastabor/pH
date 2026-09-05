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
from ph.llm.adapter import LlmError, ResolvedModel
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
from ph.seams.uploads import FileHandle
from ph.session import now_ms
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret
from ._media import forget_named_handle, load_handles, load_media, media_pointer

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
duplicated one layer up. **PDFs are expressible and still not claimed here**: this
wire carries a document either by `file_id` or inline as a `file` part (P7-03), so
a route that takes one sets `accepts: [application/pdf]` — but one
`openai-compatible` row serves gateways, local llama.cpp and Groq, and most of
them do not, so a default that claimed it would produce requests the far end
rejects for the majority of routes."""

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
"""The default per-attachment ceiling."""

UPLOAD_PURPOSE = "user_data"
"""What this wire's Files API is told the bytes are for.

Row config below rather than a constant at the call site, because `purpose` is
this API's one required form field and gateways in front of it do not agree about
the vocabulary — `user_data` is what OpenAI's own chat completions read as an
input file, and a compatible server may want `assistants` or nothing at all."""


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
    uploads: tuple[str, ...] = ()
    """MIME types to send by *reference* through the Files API rather than inline (P7-03).

    Empty by default like the Anthropic row's, for a reason that is this wire's
    own rather than a beta header: a route here may be anything that speaks the
    shape, and most gateways implement `/chat/completions` without implementing
    `/files` at all. A declared MIME the wire cannot *reference* is intersected
    away rather than uploaded — see `_body`.

    Worth it where it applies: a 4 MB PDF is 5.5 MB of base64 on every step of a
    session, and the Files API is one upload and an id thereafter."""
    upload_purpose: str = UPLOAD_PURPOSE
    max_image_edge: int | None = None
    usable_image_edge: int | None = None
    """Pixel limits, when this route publishes them (P7-03).

    `None` on both by default, and that is the honest value: unlike `accepts`,
    where a wrong guess silently drops a block, an unknown pixel ceiling has no
    safe assumption — one route scales at 2048, another not at all, and a profile
    that invented a number would warn a person about an overpayment that is not
    happening. A deployment that knows its route sets them."""


class Config(WireModel):
    """Row config: the routes to register."""

    profiles: list[ProviderProfile] = field(default_factory=list)


def _is_overflow(body: str) -> bool:
    lowered = body.lower()
    return "context" in lowered and ("length" in lowered or "window" in lowered)


_MISSING_FILE_PHRASES = ("no such file", "file_not_found", "not_found", "not found")
"""How this provider says a referenced file is gone.

Its own tuple rather than a shared one, even though it currently agrees with the
Anthropic row's almost word for word: what a provider *says* is a fact about that
provider's prose, and the moment a third wire is added the shared version would
have to be the union — which is how a `not_found` about a model name comes to be
classified as a missing file. What is genuinely shared is what happens next
(`forget_named_handle`), and that is shared.

Matched alongside a check that the message names an id this request actually
sent, which is the half that makes the guess safe."""


def _is_missing_file(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in _MISSING_FILE_PHRASES)


@dataclass(slots=True)
class OpenAiCompatibleAdapter:
    """Streams one OpenAI-compatible route."""

    ctx: Context
    profile: ProviderProfile
    http: HttpClient = field(default_factory=HttpClient)

    def _headers(self) -> dict[str, str]:
        secret = resolve_secret(self.ctx, self.profile.api_key_env, self.profile.provider)
        return {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"}

    async def upload(self, ref: Any, content: bytes) -> FileHandle:
        """Hand the bytes to the Files API and keep the id it returns (P7-03).

        The `Uploader` half of `ctx.uploads` for this wire. `purpose` is the one
        form field the endpoint requires, which is why `post_multipart` grew a
        `data` argument rather than this building its own request.

        **The expiry is read when the provider states one**, unlike the Anthropic
        row which publishes none per file: `expires_at` is unix *seconds* here and
        `FileHandle` is in milliseconds, and getting that conversion wrong in the
        cheap direction would treat every handle as expired in 1970 and re-upload
        on every request. Absent means "no expiry announced" rather than "never
        expires" — the mid-turn invalidation is what covers a file that goes away,
        and it does not depend on anyone having told the truth.
        """
        reply = await self.http.post_multipart(
            f"{self.profile.base_url.rstrip('/')}/files",
            headers=self._headers(),
            field="file",
            filename=ref.name or "attachment",
            content=content,
            mime=ref.mime,
            data={"purpose": self.profile.upload_purpose},
            is_overflow=_is_overflow,
        )
        handle = str(reply.get("id") or "")
        if not handle:
            raise LlmError("the files API returned no id", "REQUEST_FAILED")
        expires = reply.get("expires_at")
        return FileHandle(
            provider=self.profile.provider,
            attachment_id=ref.attachment_id,
            handle=handle,
            uploaded_at=now_ms(),
            expires_at=int(expires) * 1000 if isinstance(expires, (int, float)) else None,
        )

    async def _body(self, options: GenerateOptions) -> tuple[dict[str, Any], dict[str, str]]:
        """The request, and the provider file ids it was built from — see `anthropic`."""
        handles = await load_handles(
            self.ctx.get("uploads"),
            options.messages,
            provider=self.profile.provider,
            # **Intersected with what this renderer can reference**, which is
            # where this row diverges from the Anthropic one and does so
            # deliberately. There, a declared MIME with no wire shape (video) is
            # uploaded and then degrades to a pointer, because Anthropic has no
            # video content block *at all* — the pointer was the outcome either
            # way and only the upload was wasted. Here every declarable MIME is
            # already expressible inline, so uploading one this wire cannot
            # reference would turn a picture that works today into a pointer,
            # because `load_media` skips the bytes of anything with a handle. A
            # configuration mistake must not be able to remove a capability.
            mimes=frozenset(self.profile.uploads) & _FILE_MIMES,
            session_id=options.session_id,
        )
        media = await load_media(self.ctx.get("attachments"), options.messages, skip=handles.keys())
        messages: list[dict[str, Any]] = []
        if options.system:
            messages.append({"role": "system", "content": options.system})
        for message in options.messages:
            messages.extend(_to_openai(message, media, handles))
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
        if options.response_schema is not None:
            # `strict` is what makes this a constraint rather than a hint: a
            # server that supports the field builds a grammar from the schema and
            # the reply cannot come back any other shape (P7-17). The caller
            # validates anyway — `structured_output` says whether this route
            # enforces it, and a caller must not have to know which one it got.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "reply",
                    "schema": options.response_schema,
                    "strict": True,
                },
            }
        return body, handles

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        state = _StreamState()
        body, handles = await self._body(options)
        referenced = list(handles.values())
        try:
            async for _event, payload in self.http.stream_sse(
                f"{self.profile.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=body,
                is_overflow=_is_overflow,
                is_missing_file=_is_missing_file,
            ):
                for chunk in state.consume(payload):
                    yield chunk
        except LlmError as error:
            # A handle this request referenced is gone — expired early, deleted
            # from another session, or never honoured. Forget it first, then let
            # it retry: `FILE_EXPIRED` is in `TRANSIENT_CODES` precisely because
            # the state that caused it is already cleared by the time the retry
            # runs, so the next attempt re-uploads rather than repeating a request
            # that cannot work.
            raise forget_named_handle(
                self.ctx, error, referenced, provider=self.profile.provider
            ) from error
        for chunk in state.finish():
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(
            context_window=self.profile.context_window,
            default_max_tokens=self.profile.default_max_tokens,
            accepts=frozenset(self.profile.accepts),
            max_attachment_bytes=self.profile.max_attachment_bytes,
            max_image_edge=self.profile.max_image_edge,
            usable_image_edge=self.profile.usable_image_edge,
            structured_output=True,
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
_FILE_MIMES = frozenset({"application/pdf"})
"""MIMEs this wire carries as a `file` part — by `file_id`, or inline as `file_data`.

The third shape beside `image_url` and `input_audio`, and the only one with two
spellings, which is why `uploads` is intersected with it: a `file` part is the
only place a `file_id` can go on this wire, so a MIME outside this set has no
reference form however it was configured."""


def _media_part(
    attachment: AttachmentRef, data: str | None, handle: str | None
) -> dict[str, Any] | None:
    """One attachment in this wire's shape, or `None` if it has none here.

    Total over its own vocabulary rather than a second copy of the accept policy:
    `media-degrade` decides what may be sent, but this renderer still has to be
    honest about what it can *express*. A MIME it has no shape for becomes a
    pointer rather than being dressed as an image.

    `handle` wins over `data` where both are offered, which is the whole point of
    uploading — and in practice they never both arrive, because `load_media` skips
    the bytes of anything `load_handles` claimed. `data` is therefore `None` for a
    referenced file, and the signature says so rather than the caller branching.
    """
    if handle is not None and attachment.mime in _FILE_MIMES:
        return {"type": "file", "file": {"file_id": handle}}
    if data is None:
        return None
    audio = _AUDIO_FORMATS.get(attachment.mime)
    if audio is not None:
        return {"type": "input_audio", "input_audio": {"data": data, "format": audio}}
    if attachment.mime in _IMAGE_MIMES:
        return {"type": "image_url", "image_url": {"url": f"data:{attachment.mime};base64,{data}"}}
    if attachment.mime in _FILE_MIMES:
        # The inline spelling, which is what makes `load_handles`' fallback
        # contract true here: an upload that fails for any reason leaves the id
        # out and the attachment goes inline. Without this a route whose file API
        # was down would degrade a document it can perfectly well send.
        return {
            "type": "file",
            "file": {
                "filename": attachment.name or "attachment.pdf",
                "file_data": f"data:{attachment.mime};base64,{data}",
            },
        }
    return None


def _to_openai(
    message: Any, media: dict[str, str], handles: dict[str, str]
) -> list[dict[str, Any]]:
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
            part = _media_part(
                attachment,
                media.get(attachment.attachment_id),
                handles.get(attachment.attachment_id),
            )
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
    uploads = ctx.get("uploads")
    for profile in config.profiles:
        adapter = OpenAiCompatibleAdapter(ctx=ctx, profile=profile)
        handle = ctx.llm.register_adapter([profile.provider], adapter)
        ctx.add_disposer(handle.dispose, label=f"llm({profile.provider})")
        if uploads is not None and profile.uploads:
            # Only for a route that references files. Most servers speaking this
            # shape implement `/chat/completions` and not `/files`, so an uploader
            # registered for every profile would put a file API behind a provider
            # name that has none — and the first attachment would spend a failed
            # round trip discovering it.
            ctx.add_disposer(
                uploads.register_uploader(profile.provider, adapter),
                label=f"uploader({profile.provider})",
            )
        ctx.add_disposer(adapter.http.aclose, label=f"http({profile.provider})")
