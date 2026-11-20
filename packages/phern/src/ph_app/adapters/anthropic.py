"""`llm-anthropic`: the Anthropic messages wire.

This is a separate adapter and not a flag on the OpenAI adapter, because the
shapes differ in kind and not in detail:

* The system prompt is a top-level field.
* Content is always a block list.
* Tool results are in *user* messages.
* Usage arrives in two parts, in `message_start` and in `message_delta`.

One adapter for both wires would need a tangle of conditionals in the path that
is hardest to test. What the adapters do share is in `_http`: the client, the
credential, and the table from status to code.

Three mappings are worth naming:

* **Thinking blocks** map to the `reasoning` of pH and not to text. The reason
  is the same as for the `reasoning_content` of DeepSeek: the transcript must
  not claim that the model said what it was only considering.
* **`cache_creation_input_tokens` and `cache_read_input_tokens`** are already
  disjoint from `input_tokens` on this wire, so they map across directly.
  DeepSeek differs: it needs a subtraction.
* **`cache_control` markers.** The caching of Anthropic is opt-in. Every
  decision above this file that keeps the prefix stable (A12, the replayed
  summarize envelope of P4-03) pays off on the OpenAI-compatible routes, which
  cache implicitly. It pays off nowhere here until the markers are sent.
  `_checkpoints` says where they go and why.

@module ph_app.adapters.anthropic
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, assert_never

from ph.cordis import Context, plugin
from ph.json import as_int, as_str
from ph.keys import ATTACHMENTS, CREDENTIALS, LLM, UPLOADS
from ph.llm.adapter import LlmError, ResolvedModel, resolved
from ph.llm.types import (
    AttachmentRef,
    BlockEnd,
    BlockStart,
    ContentBlock,
    Finish,
    FinishKind,
    FinishReason,
    GenerateOptions,
    MediaBlock,
    Message,
    ReasoningBlock,
    ReasoningDelta,
    SamplingSetting,
    StreamChunk,
    TextBlock,
    TextDelta,
    TokenUsage,
    ToolCallBlock,
    ToolCallDelta,
    ToolResultBlock,
    UsageChunk,
    is_error_finish,
    wire_effort,
)
from ph.seams.uploads import FileHandle
from ph.session import now_ms
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret, wire_error_finish
from ._media import forget_named_handle, load_handles, load_media, media_pointer

__all__ = ["AnthropicAdapter", "apply"]


SAMPLING_WIRE: dict[SamplingSetting, str] = {
    "temperature": "temperature",
    "topP": "top_p",
    "topK": "top_k",
}
"""Each sampling setting that this API takes, and its body field (A1). The API has
no `min_p` and no penalties. Any other setting that a route sets is dropped with a
warning (`ResolvedModel.sampling`)."""

_SIGNATURE_KEY = "signature"
"""The key under which this adapter keeps its attestation in
`ReasoningBlock.provider_state`."""

API_VERSION = "2023-06-01"

FILES_BETA = "files-api-2025-04-14"
"""The beta header that the Files API is behind.

It is configurable because a beta name is a date that moves. A route that must
stay on an older name must not need a new pH release."""


ACCEPTED_MEDIA: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "application/pdf",
)
"""The default for `Config.accepts`: what the models of this provider take.

A PDF goes in a `document` block and not in an `image` block. That is the reason
why `MediaBlock` is keyed on MIME, and why the branching is in this file and not
in the content-block union."""

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
"""The default per-attachment ceiling, declared so a caller can act on it."""

MAX_IMAGE_EDGE = 8_000
"""The longest edge that this provider accepts. The provider refuses a request
with a larger image, so such an image degrades to a pointer on this route
(P7-03)."""

USABLE_IMAGE_EDGE = 1_568
"""The longest edge that this provider uses. The provider scales a larger image
down before the model sees it.

This is row config and not a rule in `ph.llm.media`, for the same reason as
`accepts`: it is a fact about a *route*. A gateway in front of a different model
scales differently, or does not scale at all. The default is the real number
because the shipped route is the API of Anthropic itself. A `None` here would
mean that pH declines to say what it knows."""

CACHE_BREAKPOINTS = 4
"""How many blocks one request may mark with `cache_control`. This is the limit
of Anthropic, and it is the design constraint: the work is to place four markers
well.

The budget is spent on `tools` (1), `system` (1) and two message checkpoints (2).
That is all four. A fifth marker is a request error, so anything that is added
here must take a slot from something else."""

CHECKPOINT_EVERY = 4
"""How often the moving message breakpoint advances, in messages.

**Why a breakpoint does not go on the last message.** A cache *read* happens
where a request marks a prefix that an earlier request also marked. A mark on
"the newest message" is at a different index in every request. It is written
one time and never read again, so it spends a slot on a prefix that nobody asks
for. Quantizing makes consecutive requests agree. With a step of four, the
requests whose message count is 5, 6, 7 and 8 all mark index 4. The second
request and every later one read what the first request wrote.

There are two marks, `latest` and the one before it. A single quantized mark
stops being read at the moment when it advances: the request that first marks
index 8 would carry no mark at 4 and would read nothing back. Keeping the
previous checkpoint means that the step is paid for one time, and not on every
request that crosses a boundary.

Four is the smallest step that still spans an assistant turn: a user message, an
assistant message with a tool call, a tool result, and the next assistant
message. So a checkpoint rarely lands in the middle of a turn. A larger step
costs a longer uncached tail. A smaller step costs a cache entry that lives for
a shorter time."""

MIN_CACHEABLE_TOKENS = 1024
"""The floor of Anthropic for a cache *write* (2048 on the smaller models).

This adapter does not enforce the floor, for a reason. A marker on a shorter
prefix is ignored and not refused, so marking `tools` in a deployment with two
small tools costs nothing. A check would need a tokenizer that this adapter does
not have, and its count would disagree with the count of the provider."""

_WIRE_SHAPES: dict[str, str] = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "application/pdf": "document",
}
"""MIME type to the block type of this wire. A PDF is a `document` and not an
`image`. That branching belongs in an adapter and not in the block union."""


class Config(WireModel):
    """Row config for the Anthropic route."""

    provider: str = "anthropic"
    """The name that this route registers under. A request that names this
    provider goes to this adapter. The file handles of the route are kept under
    the same name."""
    base_url: str = "https://api.anthropic.com/v1"
    """The root address of the API that this route calls. The adapter appends
    `/messages` for a model request and `/files` for an upload. The default is
    the API of Anthropic itself, and a deployment may point the route at a
    gateway."""
    api_key_env: str = "ANTHROPIC_API_KEY"
    """The name of the environment variable that holds the API key. It is a
    name and never the key itself. The adapter resolves the name when it builds
    a request and sends the value in the `x-api-key` header. A request fails
    with `MISSING_CREDENTIAL` when the variable is not set."""
    context_window: int | None = 200_000
    """The window that pH budgets and compacts against, in tokens. `None` means
    that the route declares no window, and then nothing budgets or compacts
    against one."""
    default_max_tokens: int = 8_192
    """The `max_tokens` that a request carries when the request itself states
    none. The route also reports this number to the layers above
    (`ResolvedModel.default_max_tokens`)."""
    accepts: tuple[str, ...] = ACCEPTED_MEDIA
    """The MIME types that this route takes as message content (P7-01).

    This is row config beside `context_window` and not a module constant, for
    the same reason as that field: what a *route* can do is a property of the
    route, and this adapter serves whatever `base_url` a deployment points it
    at. A gateway in front of a text-only model sets `accepts: []`. Its
    attachments then go as pointers, and it sends no request that the far end
    rejects."""
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES
    """The largest attachment that this route sends, in bytes. The
    `media-degrade` row replaces a larger attachment with a text pointer before
    the request reaches this adapter. The default is 5 MiB."""
    uploads: tuple[str, ...] = ()
    """The MIME types that this route sends by *reference* through the Files API,
    and not inline (P7-03).

    The default is empty, and the reason is not caution. The Files API is a beta
    that this account may not have. A route that referenced a file without the
    beta header enabled would fail every request that carries one. A deployment
    that has the beta writes `uploads: [application/pdf]`, or a video type on a
    route whose provider takes one, and the handle cache does the rest.

    The saving is worth it where it applies. A 4 MB PDF is 5.5 MB of base64 on
    *every step* of a session. With the Files API it is one upload, and an id
    after that."""
    files_beta: str = FILES_BETA
    """The value of the `anthropic-beta` header that the Files API is behind.
    The adapter sends the header on an upload and on a request that references
    an uploaded file, and on no other request. It is configurable because a
    beta name is a date that moves. A route that must stay on an older name
    must not need a new pH release."""
    max_image_edge: int = MAX_IMAGE_EDGE
    """The longest edge of an image that this route accepts, in pixels. The
    `media-degrade` row replaces a larger image with a text pointer before the
    request reaches this adapter (P7-03). The default is the longest edge that
    this provider accepts. The provider refuses a request with a larger image."""
    usable_image_edge: int = USABLE_IMAGE_EDGE
    """The longest edge of an image that this route uses, in pixels. A larger
    image is still sent whole. The `media-degrade` row then records an
    `attachment/oversized` notice and logs a warning, because the extra pixels
    are uploaded on every request and discarded.

    The default is the longest edge that this provider uses. The provider
    scales a larger image down before the model sees it. The default is the
    real number because the shipped route is the API of Anthropic itself. A
    gateway in front of a different model scales differently, or does not scale
    at all."""
    cache_control: bool = True
    """Whether the route sends `cache_control` breakpoints (P6-13).

    This is row config for the same reason as `accepts`: this adapter serves
    whatever `base_url` points at, and a gateway that does not implement the
    caching of Anthropic rejects the field and does not ignore it. The default is
    on, because the shipped route is the API of Anthropic itself, and A12 has an
    effect there only because of the markers."""


def _breakpoint() -> dict[str, str]:
    """One `cache_control` marker. Each call returns a new dict. The markers go
    to a JSON encoder inside a body that a caller may edit, and a shared literal
    would make two markers one object."""
    return {"type": "ephemeral"}


def _checkpoints(count: int) -> tuple[int, ...]:
    """Which message indices carry a breakpoint, oldest first (P6-13).

    The indices are quantized to `CHECKPOINT_EVERY`, so that consecutive requests
    mark the *same* index. That is the only way that a marker is read and not
    only written. There are two indices, so that the cache is still read on the
    request where the checkpoint advances. `CHECKPOINT_EVERY` gives the
    reasoning.

    The result is the empty tuple when there are no messages: it is better to
    mark nothing than to guess an index for a list that has none. Every shape
    that reaches this function has at least one message. For example, the
    `direct` path of compaction sends a single instruction message. So the guard
    is for a caller that this file does not know about.
    """
    if count <= 0:
        return ()
    latest = ((count - 1) // CHECKPOINT_EVERY) * CHECKPOINT_EVERY
    return tuple(index for index in (latest - CHECKPOINT_EVERY, latest) if index >= 0)


_MISSING_FILE_PHRASES = ("not_found", "not found", "no such file")
"""How this provider says that a referenced file is gone.

These are phrases and not a status code. A 404 from a gateway in front of the
API is about the *route*, and a retry of that as an expired upload would loop.
The match is used together with a check that the message names an id that this
request sent. That check is the half that makes the guess safe."""


def _is_missing_file(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in _MISSING_FILE_PHRASES)


def _is_overflow(body: str) -> bool:
    # Only the phrasing of the provider itself counts. `max_tokens` also appears
    # in a plain bad-request body. To treat that as an overflow would trigger a
    # compaction for a request that fit.
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
            # The beta is sent for each *request* that needs it, and not for each
            # route. When it was keyed on `config.uploads`, the beta went on
            # every request that a configured route sent. An account without the
            # beta then failed the plain text turns too: a capability that the
            # deployment declared became an outage on requests that never used
            # it. The handles of the request now decide. A turn that referenced
            # no file sends no beta.
            headers["anthropic-beta"] = self.config.files_beta
        return headers

    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle:
        """Hand the bytes to the Files API and keep the id that it returns.

        This is the `Uploader` half of `ctx.uploads`. No expiry is recorded.
        This provider does not publish an expiry for each file, and an invented
        TTL would be a prediction that pH has no basis for. The invalidation in
        the middle of a turn covers a file that goes away, and it does not
        depend on a guess.
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
        handle = as_str(reply.get("id"))
        if not handle:
            raise LlmError("the files API returned no id", "REQUEST_FAILED")
        return FileHandle(
            provider=self.config.provider,
            attachment_id=ref.attachment_id,
            handle=handle,
            uploaded_at=now_ms(),
        )

    async def _body(self, options: GenerateOptions) -> tuple[dict[str, Any], dict[str, str]]:
        """The request, and the provider file ids that it was built from.

        **The handles are returned and are not parsed out of the body again.**
        Each wire had grown a `_referenced_files` walker: an independent second
        model of where a file id can be in a request. Its only input was the map
        that `load_handles` had just returned. If a new place for an id is
        added, a walker silently misses it. Then `forget_named_handle` stops
        invalidating, and `FILE_EXPIRED` retries the same dead handle until the
        budget is gone.
        """
        handles = await load_handles(
            self.ctx.get(UPLOADS),
            options.messages,
            provider=self.config.provider,
            mimes=frozenset(self.config.uploads),
            session_id=options.session_id,
        )
        # This runs after the handles and skips them. A referenced file must not
        # also be read and encoded, because that is the cost that uploading
        # removes.
        media = await load_media(self.ctx.get(ATTACHMENTS), options.messages, skip=handles.keys())
        caching = self.config.cache_control
        messages = [_to_anthropic(message, media, handles) for message in options.messages]
        if caching:
            for index in _checkpoints(len(messages)):
                # The marker goes on the last block of the message, which is what
                # "cache everything through here" means on this wire.
                # `_to_anthropic` never returns an empty content list, so there
                # is always a block to mark.
                messages[index]["content"][-1]["cache_control"] = _breakpoint()
        body: dict[str, Any] = {
            "model": options.model,
            "max_tokens": options.max_tokens or self.config.default_max_tokens,
            "stream": True,
            "messages": messages,
        }
        if options.system:
            # This is a block list and not the plain string, because only a block
            # can carry a marker. It stays a string when caching is off, so a
            # gateway that never wanted any of this sees the shape that it always
            # saw.
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
                # The tools have their own breakpoint and do not rely on the
                # breakpoint of `system`. The slot is justified. The wire order
                # is tools → system → messages, so a system prompt that changes
                # in the middle of a session invalidates every marker after it.
                # pH composes a `todos` section into the system prompt whenever
                # the list changes. The tools survive that change. Without this
                # marker they would not.
                tools[-1]["cache_control"] = _breakpoint()
            body["tools"] = tools
        body.update(options.sampling.on_wire(SAMPLING_WIRE))
        if options.stop:
            body["stop_sequences"] = list(options.stop)
        wire_effort(body, options)
        return body, handles

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
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
                    if is_error_finish(chunk):
                        # The wire has reported that the request failed, and this
                        # stream is over. The return here skips `finish()`.
                        # `finish()` would close the half-written blocks and
                        # append a second `Finish` that says "stop". That is how
                        # a failed turn came to be recorded as a completed turn
                        # with a truncated answer.
                        return
        except LlmError as error:
            # A handle that this request referenced is gone: it expired early, it
            # was deleted from another session, or it was never honored.
            # **Forget it first, then let the request retry.** `FILE_EXPIRED` is
            # in `TRANSIENT_CODES` because the state that caused it is already
            # cleared when the retry runs. The next attempt therefore uploads
            # again and does not repeat a request that cannot work. To fail the
            # turn here would lose an hour of conversation because of a cache
            # entry.
            raise forget_named_handle(
                self.ctx, error, referenced, provider=self.config.provider
            ) from error
        for chunk in state.finish():
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        # `structured_output` is False, and it is stated on purpose. Anthropic
        # has no equivalent of `response_format` today, so a caller here gets
        # the instruction, the validation and the retry, and not the guarantee
        # of the wire (P7-17). The argument is passed and not left as a default
        # without a remark, so that the next adapter does not copy an omission
        # that looked like an oversight.
        return resolved(
            self.config,
            structured_output=False,
            credential=self.config.api_key_env,
            sampling=frozenset(SAMPLING_WIRE),
            max_decision_tokens=None,
        )


@dataclass(slots=True)
class _Open:
    kind: str
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    arguments: str = ""
    signature: str = ""
    """The attestation of Anthropic, which arrives as its own delta after the
    thinking (G8)."""
    redacted: bool = False
    """Whether this block is `redacted_thinking`: ciphertext and not prose (G8)."""


@dataclass(slots=True)
class _StreamState:
    blocks: dict[int, _Open] = field(default_factory=dict)
    usage: TokenUsage | None = None
    stop_reason: str | None = None

    def consume(self, event: str, payload: dict[str, Any]) -> list[StreamChunk]:
        out: list[StreamChunk] = []
        kind = payload.get("type") or event
        if kind == "message_start":
            raw = (payload.get("message") or {}).get("usage")
            if isinstance(raw, dict):
                self.usage = _to_usage(raw)
        elif kind == "content_block_start":
            index = as_int(payload.get("index"))
            block = payload.get("content_block") or {}
            block_type = as_str(block.get("type"), "text")
            if block_type in ("thinking", "redacted_thinking"):
                # **`redacted_thinking` is reasoning too** (G8). It used to fall
                # to the `else` and open as *text*. So the opaque blob that
                # Anthropic encrypts when its safety systems withhold the
                # reasoning was shown to the user, and to the model, as something
                # that the assistant had said. Its payload is `data` and not
                # `text`. It must still be passed back, or the conversation
                # cannot continue. So it is kept, marked, and sent again as
                # itself.
                self.blocks[index] = _Open(
                    kind="reasoning",
                    # `data` is present on a redacted start and absent on a plain
                    # start. So this is the blob, or the empty string that a
                    # `thinking` block would have opened with anyway.
                    text=as_str(block.get("data")),
                    signature=as_str(block.get("signature")),
                    redacted=block_type == "redacted_thinking",
                )
                out.append(BlockStart(index=index, block_type="reasoning"))
            elif block_type == "tool_use":
                self.blocks[index] = _Open(
                    kind="tool-call",
                    tool_id=as_str(block.get("id")),
                    tool_name=as_str(block.get("name")),
                )
                out.append(BlockStart(index=index, block_type="tool-call"))
            else:
                self.blocks[index] = _Open(kind="text")
                out.append(BlockStart(index=index, block_type="text"))
        elif kind == "content_block_delta":
            index = as_int(payload.get("index"))
            delta = payload.get("delta") or {}
            open_block = self.blocks.get(index)
            if open_block is None:
                return out
            if delta.get("type") == "thinking_delta":
                fragment = as_str(delta.get("thinking"))
                open_block.text += fragment
                out.append(ReasoningDelta(index=index, text=fragment))
            elif delta.get("type") == "signature_delta":
                # **Captured, and nothing is emitted** (G8). This delta used to
                # arrive at the `else` below and become a `TextDelta` with `""`
                # in it: a text chunk on a *reasoning* block, for every thinking
                # block in every stream. The delta carries the attestation that
                # the next request must send back, and it was thrown away.
                open_block.signature += as_str(delta.get("signature"))
            elif delta.get("type") == "input_json_delta":
                fragment = as_str(delta.get("partial_json"))
                open_block.arguments += fragment
                out.append(
                    ToolCallDelta(
                        index=index,
                        id=open_block.tool_id,
                        name=open_block.tool_name or None,
                        arguments_delta=fragment,
                    )
                )
            else:
                fragment = as_str(delta.get("text"))
                open_block.text += fragment
                out.append(TextDelta(index=index, text=fragment))
        elif kind == "content_block_stop":
            index = as_int(payload.get("index"))
            open_block = self.blocks.pop(index, None)
            if open_block is not None:
                out.append(BlockEnd(index=index, block=_close(open_block)))
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            if isinstance(delta.get("stop_reason"), str):
                self.stop_reason = delta["stop_reason"]
            raw = payload.get("usage")
            if isinstance(raw, dict):
                self.usage = _merge_usage(self.usage, raw)
        elif kind == "error":
            wire_error = payload.get("error") or {}
            out.append(
                wire_error_finish(
                    wire_error,
                    kind=wire_error.get("type"),
                    is_overflow=_is_overflow,
                    is_missing_file=_is_missing_file,
                )
            )
        return out

    def finish(self) -> list[StreamChunk]:
        out: list[StreamChunk] = []
        for index, open_block in list(self.blocks.items()):
            out.append(BlockEnd(index=index, block=_close(open_block)))
        self.blocks.clear()
        if self.usage is not None:
            out.append(UsageChunk(usage=self.usage))
        out.append(Finish(reason=FinishReason(kind=_finish_kind(self.stop_reason))))
        return out


def _close(open_block: _Open) -> ContentBlock:
    if open_block.kind == "reasoning":
        return ReasoningBlock(
            text=open_block.text,
            provider_state=(
                {_SIGNATURE_KEY: open_block.signature} if open_block.signature else None
            ),
            redacted=open_block.redacted,
        )
    if open_block.kind == "tool-call":
        return ToolCallBlock(
            id=open_block.tool_id,
            name=open_block.tool_name,
            arguments=open_block.arguments or "{}",
        )
    return TextBlock(text=open_block.text)


def _finish_kind(stop_reason: str | None) -> FinishKind:
    if stop_reason == "tool_use":
        return "tool-calls"
    if stop_reason == "max_tokens":
        return "max-tokens"
    return "stop"


def _to_usage(raw: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        # Already disjoint on this wire, so no subtraction (unlike DeepSeek).
        input_tokens=as_int(raw.get("input_tokens")),
        output_tokens=as_int(raw.get("output_tokens")),
        cache_read_tokens=as_int(raw.get("cache_read_input_tokens")) or None,
        cache_write_tokens=as_int(raw.get("cache_creation_input_tokens")) or None,
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


def _signature_of(block: ReasoningBlock) -> str:
    """The attestation of this adapter, read from the opaque state of the block (G8).

    `provider_state` is deliberately untyped at the seam. It is whatever the
    producing adapter needs back, and only that adapter reads it. So this is the
    one place that knows the key. The result is empty for a block that another
    provider wrote, and the caller drops that block.
    """
    state = block.provider_state or {}
    return as_str(state.get(_SIGNATURE_KEY))


def _media_part(
    attachment: AttachmentRef, media: dict[str, str], handles: dict[str, str]
) -> dict[str, Any]:
    """One attachment as a `file` source or a `base64` source, or as a pointer if
    it has neither.

    This function is total over its own vocabulary. It is not a second copy of
    the accept policy: `media-degrade` decides what may be sent, but this
    renderer must still be correct about what it can *express*. Without the
    fallback, a MIME type that it has no shape for would be sent as an image.

    `handles` wins over `media`: a file that the provider already holds is
    referenced by id, which is the purpose of uploading it.
    """
    shape = _WIRE_SHAPES.get(attachment.mime)
    if shape is None:
        return media_pointer(attachment)
    handle = handles.get(attachment.attachment_id)
    if handle is not None:
        return {"type": shape, "source": {"type": "file", "file_id": handle}}
    data = media.get(attachment.attachment_id)
    if data is None:
        return media_pointer(attachment)
    return {
        "type": shape,
        "source": {"type": "base64", "media_type": attachment.mime, "data": data},
    }


def _to_anthropic(
    message: Message,
    media: dict[str, str],
    handles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One pH message as an Anthropic message.

    `handles` wins over `media` where both are present: a file that the provider
    already holds is referenced by id, which is the purpose of uploading it.

    Tool results are user-role content blocks here and do not have their own
    role. That is the main structural difference from the OpenAI wire.

    `media` holds base64 for the attachments that this route will take. An
    attachment that is absent from it becomes a pointer. **No attachment is
    dropped**, so a message that was only an image does not arrive as an empty
    text block. Every block kind has a case, and `assert_never` makes a new kind
    a type error. A reasoning block without a signature is left out on purpose,
    and the comment in that case says why.
    """
    blocks: list[dict[str, Any]] = []
    for block in message.content:
        match block:
            case TextBlock():
                blocks.append({"type": "text", "text": block.text})
            case ToolResultBlock():
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_call_id,
                        "content": [
                            {"type": "text", "text": inner.text}
                            for inner in block.content
                            if isinstance(inner, TextBlock)
                        ],
                        **({"is_error": True} if block.is_error else {}),
                    }
                )
            case ToolCallBlock():
                try:
                    parsed = json.loads(block.arguments) if block.arguments else {}
                except json.JSONDecodeError:
                    parsed = {}
                blocks.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": parsed}
                )
            case ReasoningBlock():
                # **Signed, or left out** (G8). Anthropic rejects a conversation
                # that sends a thinking block back without its signature. An
                # unsigned block is reasoning from another provider, or from a
                # log that was written before the signature was kept. It is
                # dropped, so that it does not make every later request in the
                # session fail. This is the same reading as the empty assistant
                # message of the OpenAI wire (G9). A block that this wire cannot
                # carry is not sent, and nothing depends on it, because reasoning
                # is the scratch work of the model and no result pairs with it.
                if block.redacted:
                    blocks.append({"type": "redacted_thinking", "data": block.text})
                elif signature := _signature_of(block):
                    blocks.append(
                        {"type": "thinking", "thinking": block.text, "signature": signature}
                    )
            case MediaBlock():
                blocks.append(_media_part(block.attachment, media, handles or {}))
            case _ as unhandled:
                assert_never(unhandled)
    role = "assistant" if message.role == "assistant" else "user"
    return {"role": role, "content": blocks or [{"type": "text", "text": ""}]}


@plugin("llm-anthropic", affects="environment", config=Config, inject=[LLM, CREDENTIALS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the Anthropic route."""
    adapter = AnthropicAdapter(ctx=ctx, config=config)
    handle = ctx.require(LLM).register_adapter([config.provider], adapter)
    ctx.add_disposer(handle.dispose, label=f"llm({config.provider})")
    uploads = ctx.get(UPLOADS)
    if uploads is not None and config.uploads:
        # Only when this route references files. To register an uploader that a
        # route never uses would put a file API behind a provider name that has
        # not opted into the beta that the API needs.
        ctx.add_disposer(
            uploads.register_uploader(config.provider, adapter),
            label=f"uploader({config.provider})",
        )
    ctx.add_disposer(adapter.http.aclose, label=f"http({config.provider})")
