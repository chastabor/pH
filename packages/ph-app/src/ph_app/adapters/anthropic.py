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
from ph.llm.adapter import LlmError, ResolvedModel
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
from ph.seams.uploads import FileHandle
from ph.session import now_ms
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret
from ._media import forget_named_handle, load_handles, load_media, media_pointer

__all__ = ["AnthropicAdapter", "apply"]

log = logging.getLogger("ph_app.adapters.anthropic")

API_VERSION = "2023-06-01"

FILES_BETA = "files-api-2025-04-14"
"""The beta header the Files API is behind.

Configurable because a beta name is a date that moves, and a route pinned to an
older one must not need a new pH release."""


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

MAX_IMAGE_EDGE = 8_000
"""Longest edge this provider accepts. Over it the request is refused, so an
image over it degrades to a pointer here instead (P7-03)."""

USABLE_IMAGE_EDGE = 1_568
"""Longest edge this provider actually uses; larger images are scaled down before
the model sees them.

Row config rather than a rule in `ph.llm.media`, for `accepts`' reason: this is a
fact about a *route*, and a gateway in front of a different model scales
differently or not at all. Declared with the real number because the shipped
route is Anthropic's own — a `None` here would be pH declining to say what it
knows."""

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
    uploads: tuple[str, ...] = ()
    """MIME types to send by *reference* through the Files API rather than inline (P7-03).

    Empty by default, and that is not timidity: the Files API is a beta this
    account may not have, and a route that referenced a file without the beta
    header enabled would fail every request carrying one. A deployment that has
    it writes `uploads: [application/pdf]` — or a video type on a route whose
    provider takes one — and the handle cache does the rest.

    Worth it where it applies: a 4 MB PDF is 5.5 MB of base64 on *every step* of
    a session, and the Files API is one upload and an id thereafter."""
    files_beta: str = FILES_BETA
    max_image_edge: int = MAX_IMAGE_EDGE
    usable_image_edge: int = USABLE_IMAGE_EDGE
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


_MISSING_FILE_PHRASES = ("not_found", "not found", "no such file")
"""How this provider says a referenced file is gone.

Phrases rather than a status code: a 404 from a gateway in front of the API is
about the *route*, and retrying that as an expired upload would loop. Matched
alongside a check that the message names an id this request actually sent, which
is the half that makes the guess safe."""


def _is_missing_file(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in _MISSING_FILE_PHRASES)


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

    def _headers(self, *, files: bool = False) -> dict[str, str]:
        headers = {
            "x-api-key": resolve_secret(self.ctx, self.config.api_key_env, self.config.provider),
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        }
        if files:
            # Per *request*, not per route. Keying it on `config.uploads` made
            # the beta ride every request a configured route sent, so an account
            # without it would fail the plain text turns too — a capability the
            # deployment declared becoming an outage on requests that never used
            # it. The request's own handles are what answers that now: a turn
            # that referenced no file sends no beta.
            headers["anthropic-beta"] = self.config.files_beta
        return headers

    async def upload(self, ref: Any, content: bytes) -> FileHandle:
        """Hand the bytes to the Files API and keep the id it returns.

        The `Uploader` half of `ctx.uploads`. No expiry is recorded: this
        provider does not publish one per file, and inventing a TTL would be a
        prediction pH has no basis for — the mid-turn invalidation is what covers
        a file that goes away, and it does not depend on a guess.
        """
        reply = await self.http.post_multipart(
            f"{self.config.base_url.rstrip('/')}/files",
            headers=self._headers(files=True),
            field="file",
            filename=ref.name or "attachment",
            content=content,
            mime=ref.mime,
            is_overflow=_is_overflow,
        )
        handle = str(reply.get("id") or "")
        if not handle:
            raise LlmError("the files API returned no id", "REQUEST_FAILED")
        return FileHandle(
            provider=self.config.provider,
            attachment_id=ref.attachment_id,
            handle=handle,
            uploaded_at=now_ms(),
        )

    async def _body(self, options: GenerateOptions) -> tuple[dict[str, Any], dict[str, str]]:
        """The request, and the provider file ids it was built from.

        **The handles come back rather than being parsed out of the body again.**
        Each wire had grown a `_referenced_files` walker — an independent second
        model of where a file id can sit in a request — whose only input was the
        map `load_handles` had just returned. Add a place an id can appear and a
        walker silently misses it, `forget_named_handle` stops invalidating, and
        `FILE_EXPIRED` retries the same dead handle until the budget is gone.
        """
        handles = await load_handles(
            self.ctx.get("uploads"),
            options.messages,
            provider=self.config.provider,
            mimes=frozenset(self.config.uploads),
            session_id=options.session_id,
        )
        # After the handles and skipping them: a referenced file must not also be
        # read and encoded, which is the cost uploading exists to remove.
        media = await load_media(self.ctx.get("attachments"), options.messages, skip=handles.keys())
        caching = self.config.cache_control
        messages = [_to_anthropic(message, media, handles) for message in options.messages]
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
        return body, handles

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        state = _StreamState()
        body, handles = await self._body(options)
        referenced = list(handles.values())
        try:
            async for event, payload in self.http.stream_sse(
                f"{self.config.base_url.rstrip('/')}/messages",
                headers=self._headers(files=bool(referenced)),
                json=body,
                is_overflow=_is_overflow,
                is_missing_file=_is_missing_file,
            ):
                for chunk in state.consume(event, payload):
                    yield chunk
        except LlmError as error:
            # A handle this request referenced is gone — expired early, deleted
            # from another session, or never honoured. **Forget it first, then
            # let it retry**: `FILE_EXPIRED` is in `TRANSIENT_CODES` precisely
            # because the state that caused it is already cleared by the time the
            # retry runs, so the next attempt re-uploads rather than repeating a
            # request that cannot work. Failing the turn here would lose an
            # hour's conversation over a cache entry.
            raise forget_named_handle(
                self.ctx, error, referenced, provider=self.config.provider
            ) from error
        for chunk in state.finish():
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(
            context_window=self.config.context_window,
            default_max_tokens=self.config.default_max_tokens,
            accepts=frozenset(self.config.accepts),
            max_attachment_bytes=self.config.max_attachment_bytes,
            max_image_edge=self.config.max_image_edge,
            usable_image_edge=self.config.usable_image_edge,
            # `structured_output` stays False, and the silence is deliberate:
            # Anthropic has no `response_format` equivalent today, so a caller
            # here gets the instruction, the validation and the retry, and not
            # the wire's guarantee (P7-17). Declaring it rather than leaving the
            # default unremarked is what stops the next adapter copying an
            # omission it thought was an oversight.
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


def _to_anthropic(
    message: Any, media: dict[str, str], handles: dict[str, str] | None = None
) -> dict[str, Any]:
    """One pH message as an Anthropic message.

    `handles` wins over `media` where both are present: a file the provider
    already holds is referenced by id, which is the whole point of uploading it.

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
            handle = (handles or {}).get(block.attachment.attachment_id)
            data = media.get(block.attachment.attachment_id)
            shape = _WIRE_SHAPES.get(block.attachment.mime)
            if handle is not None and shape is not None:
                blocks.append({"type": shape, "source": {"type": "file", "file_id": handle}})
                continue
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
    uploads = ctx.get("uploads")
    if uploads is not None and config.uploads:
        # Only when this route references files. Registering an uploader a route
        # never uses would put a file API behind a provider name that has not
        # opted into the beta it needs.
        ctx.add_disposer(
            uploads.register_uploader(config.provider, adapter),
            label=f"uploader({config.provider})",
        )
    ctx.add_disposer(adapter.http.aclose, label=f"http({config.provider})")
