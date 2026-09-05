"""`llm-google` — the Gemini `generateContent` wire, and the Files API video needs.

The third adapter, and it exists because of P7-03 rather than alongside it. That
row's gate is *a video reaches a route that accepts one*, and its own notes said
what stood in the way: **neither provider pH had an adapter for accepts video**,
so a route declaring it would have been a fiction and the transport was proven
against a shape that does not exist. This is the provider that takes one.

A separate adapter rather than a flag, for the reason `anthropic.py` gives about
itself and more so — almost nothing here has the same name as anywhere else.
Messages are `contents` and content is `parts`; the system prompt is
`systemInstruction`; tools are `functionDeclarations`; a tool result is a
`functionResponse` part addressed **by name** rather than by id; and usage is
`usageMetadata` with the thinking tokens counted *outside* the answer's.

Three mappings worth naming, because each is a place where a reasonable guess is
wrong:

* **`thoughtsTokenCount` is not inside `candidatesTokenCount`.** pH's
  `reasoning_tokens` is documented as a *subset* of `output_tokens` — `total`
  leaves it out for exactly that reason — so this adapter adds it in. Mapping it
  across directly would under-report every thinking turn's output.
* **A function call has no id on this wire.** pH's `ToolCallBlock` needs one and
  the pairing every consumer relies on is by id, so ids are minted here and the
  reverse map is rebuilt when a result goes back out as a `functionResponse`,
  which carries the *name*.
* **A thought does not go back.** There is no input shape for one, and rendering
  it as a text part would put the model's private reasoning into the conversation
  as something it said. The log still holds it; the request does not.

@module ph_app.adapters.google
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anyio

from ph.cordis import Context, plugin
from ph.llm.adapter import LlmError, ResolvedModel
from ph.llm.types import (
    AttachmentRef,
    BlockEnd,
    BlockStart,
    Finish,
    FinishKind,
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
)
from ph.seams.uploads import FileHandle
from ph.session import now_ms
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret
from ._media import forget_named_handle, load_handles, load_media, media_pointer

__all__ = ["GoogleAdapter", "apply"]

log = logging.getLogger("ph_app.adapters.google")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

ACCEPTED_MEDIA: tuple[str, ...] = (
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
    "audio/wav",
    "audio/mpeg",
    "audio/aac",
    "audio/ogg",
    "audio/flac",
    "video/mp4",
    "video/mpeg",
    "video/webm",
    "video/quicktime",
    "application/pdf",
)
"""The default for `Config.accepts` — what this provider's own models take.

The widest of the three, and video is the entry that matters: it is the format
that made upload-and-reference a row rather than an optimisation, and until this
adapter existed no route pH shipped could declare it honestly."""

UPLOADED_MEDIA: tuple[str, ...] = (
    "video/mp4",
    "video/mpeg",
    "video/webm",
    "video/quicktime",
)
"""The default for `Config.uploads` — video, by reference, always.

**Non-empty where both sibling rows default to nothing, and the difference is
about the provider rather than about confidence.** Anthropic's Files API is
behind a beta an account may not have; an OpenAI-compatible route may be a
gateway that never implemented `/files`. Neither caveat applies to this
provider's own service, where the Files API is the documented path and a request
carrying a video inline is bounded by a 20 MB total request size that most videos
are not under.

A gateway pretending to be Gemini sets `uploads: []` and gets inline bytes, which
is the same escape hatch the other two rows have in the other direction."""

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
"""The inline ceiling: this provider bounds the whole *request* at 20 MB.

Which is the argument for the Files API rather than a limit on what it will
accept — a 300 MB video is fine by reference and impossible inline, so this
number is what a block must fit under to go as `inlineData`."""

UPLOAD_POLL_MS = 500
UPLOAD_READY_MS = 120_000
"""How long an upload is waited on before the turn gives up on it.

Video is **processed** after it is stored, and a `fileUri` referenced before its
file reaches `ACTIVE` is refused — so this is the one uploader of the three that
is not done when the bytes have landed. Two minutes because that is the scale of
a few minutes of video being transcoded at the far end, and because the fallback
when it expires is not a failure: `load_handles` swallows it and the block goes
inline, which for a video over the inline cap becomes an honest pointer."""


class Config(WireModel):
    """Row config for the Google route."""

    provider: str = "google"
    base_url: str = BASE_URL
    api_key_env: str = "GEMINI_API_KEY"
    context_window: int | None = 1_048_576
    default_max_tokens: int | None = None
    accepts: tuple[str, ...] = ACCEPTED_MEDIA
    """MIME types this route takes as message content (P7-01)."""
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES
    uploads: tuple[str, ...] = UPLOADED_MEDIA
    """MIME types to send by reference through the Files API (P7-03)."""
    upload_ready_ms: int = UPLOAD_READY_MS
    max_image_edge: int | None = None
    usable_image_edge: int | None = None
    """Pixel limits, when a deployment knows them.

    `None` like the OpenAI row's rather than Anthropic's stated numbers: this
    provider tiles large images rather than publishing one edge it scales to, so a
    number invented here would warn a person about an overpayment that is not
    happening."""


_MISSING_FILE_PHRASES = ("is not in an active state", "was not found", "not found", "permission")
"""How this provider says a referenced file is gone or unusable.

Its own tuple, like each sibling's, because this is a fact about one provider's
prose — and this one differs in kind: a file here can be *present and not ready*,
which reads as a different sentence and needs the same remedy, since the handle
that was cached is one this request cannot use.

`permission` is in the list because a file deleted from another session answers
`PERMISSION_DENIED` rather than a not-found — matched, as everywhere, alongside
the check that the message names an id this request actually sent."""


def _is_missing_file(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in _MISSING_FILE_PHRASES)


def _is_overflow(body: str) -> bool:
    lowered = body.lower()
    return "token count" in lowered or "exceeds the maximum" in lowered


_FINISH_KINDS: dict[str, FinishKind] = {"MAX_TOKENS": "max-tokens", "STOP": "stop"}
"""This wire's finish reasons, and what pH calls them.

Everything not named here — `SAFETY`, `RECITATION`, `OTHER` — falls to `stop`,
which is the honest mapping rather than a lazy one: the model stopped, the
transcript holds what it produced, and pH's kinds carry no member meaning
"refused". Inventing `error` for a completed response would make the retry policy
re-run a request the provider will answer the same way."""


@dataclass(slots=True)
class GoogleAdapter:
    """Streams one Gemini route."""

    ctx: Context
    config: Config
    http: HttpClient = field(default_factory=HttpClient)

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": resolve_secret(
                self.ctx, self.config.api_key_env, self.config.provider
            ),
            "Content-Type": "application/json",
        }

    # ----------------------------------------------------------- uploading --

    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle:
        """Store the bytes through the Files API and wait until they are usable.

        **The only uploader of the three that is not finished when the bytes have
        landed.** A video is processed after it is stored and a `fileUri`
        referenced before its file reaches `ACTIVE` is refused, so an uploader
        that returned at the end of the transfer would hand back a handle whose
        first use fails — and be indistinguishable, from the cache's point of
        view, from a provider that expired it.

        The transfer itself is this provider's resumable protocol, whose two steps
        are why `post_raw` exists: the first is a JSON body answered with an empty
        one and a destination in `X-Goog-Upload-URL`, the second is the file's
        bytes with no form encoding at all.

        Failure of any kind here is caught by `load_handles` and the attachment
        goes inline — so the worst outcome of a slow provider is the behaviour
        every route had before the row.
        """
        base = self.config.base_url.rstrip("/")
        upload_base = base.replace("/v1beta", "/upload/v1beta")
        _body, headers = await self.http.post_raw(
            f"{upload_base}/files",
            headers={
                **self._headers(),
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(content)),
                "X-Goog-Upload-Header-Content-Type": ref.mime,
            },
            json={"file": {"display_name": ref.name or "attachment"}},
            is_overflow=_is_overflow,
        )
        destination = headers.get("x-goog-upload-url") or headers.get("X-Goog-Upload-URL")
        if not destination:
            raise LlmError("the files API started no upload", "REQUEST_FAILED")
        stored, _ = await self.http.post_raw(
            str(destination),
            headers={
                "Content-Length": str(len(content)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=content,
            is_overflow=_is_overflow,
        )
        record = _file_record(stored)
        uri = str(record.get("uri") or "")
        if not uri:
            raise LlmError("the files API returned no uri", "REQUEST_FAILED")
        record = await self._ready(record)
        return FileHandle(
            provider=self.config.provider,
            attachment_id=ref.attachment_id,
            handle=uri,
            uploaded_at=now_ms(),
            expires_at=_expiry(record.get("expirationTime")),
        )

    async def _ready(self, record: dict[str, Any]) -> dict[str, Any]:
        """Poll until the file is `ACTIVE`, or give up inside the budget.

        `FAILED` raises rather than waiting out the clock: the provider has said
        this file will never work, and the useful thing to do with that is fall
        back to the bytes now.
        """
        name = str(record.get("name") or "")
        waited = 0
        while str(record.get("state") or "ACTIVE") == "PROCESSING":
            if waited >= self.config.upload_ready_ms:
                raise LlmError(f"{name} was still processing after the upload budget", "TIMEOUT")
            await anyio.sleep(UPLOAD_POLL_MS / 1000)
            waited += UPLOAD_POLL_MS
            record = await self.http.get_json(
                f"{self.config.base_url.rstrip('/')}/{name}",
                headers=self._headers(),
                is_overflow=_is_overflow,
            )
            record = _file_record(record)
        if str(record.get("state") or "") == "FAILED":
            raise LlmError(f"{name} could not be processed", "REQUEST_FAILED")
        return record

    # ----------------------------------------------------------- streaming --

    async def _body(self, options: GenerateOptions) -> tuple[dict[str, Any], dict[str, str]]:
        """The request, and the provider file ids it was built from — see `anthropic`."""
        handles = await load_handles(
            self.ctx.get("uploads"),
            options.messages,
            provider=self.config.provider,
            mimes=frozenset(self.config.uploads),
            session_id=options.session_id,
        )
        media = await load_media(self.ctx.get("attachments"), options.messages, skip=handles.keys())
        names = _call_names(options.messages)
        contents = [
            content
            for message in options.messages
            if (content := _to_google(message, media, handles, names)) is not None
        ]
        body: dict[str, Any] = {"contents": contents}
        if options.system:
            body["systemInstruction"] = {"parts": [{"text": options.system}]}
        if options.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                        for tool in options.tools
                    ]
                }
            ]
        generation: dict[str, Any] = {}
        if options.temperature is not None:
            generation["temperature"] = options.temperature
        max_tokens = options.max_tokens or self.config.default_max_tokens
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        if options.stop:
            generation["stopSequences"] = list(options.stop)
        if options.response_schema is not None:
            # **The mime type and not the schema**, which is a decision rather
            # than an omission. This provider's `responseSchema` is an OpenAPI
            # subset that rejects keywords ordinary JSON Schema carries, so
            # forwarding pH's schema would fail requests for a capability
            # `resolve_model` says below it does not have. Asking for JSON is the
            # half that is true, and the caller validates either way — which is
            # exactly what `structured_output: False` exists to tell it.
            generation["responseMimeType"] = "application/json"
        if generation:
            body["generationConfig"] = generation
        return body, handles

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        state = _StreamState()
        body, handles = await self._body(options)
        referenced = list(handles.values())
        model = options.model.removeprefix("models/")
        try:
            async for _event, payload in self.http.stream_sse(
                f"{self.config.base_url.rstrip('/')}/models/{model}:streamGenerateContent?alt=sse",
                headers=self._headers(),
                json=body,
                is_overflow=_is_overflow,
                is_missing_file=_is_missing_file,
            ):
                for chunk in state.consume(payload):
                    yield chunk
        except LlmError as error:
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
            # See `_body`: this route is asked for JSON and not held to a schema,
            # so it does not enforce one. Claiming otherwise would make a caller
            # skip the validation that is actually doing the work.
            structured_output=False,
        )


def _file_record(payload: dict[str, Any]) -> dict[str, Any]:
    """The file's own fields, whether or not they arrived inside a `file` envelope.

    Both spellings are in the wild — the resumable finalize answers `{"file":
    {...}}` and a `GET` on the file answers the record itself — and a caller that
    guessed wrong would read every field as absent, which for `state` means
    "already ACTIVE" and hands back a handle whose first use fails.
    """
    inner = payload.get("file")
    return inner if isinstance(inner, dict) else payload


def _expiry(stated: Any) -> int | None:
    """`expirationTime` as epoch milliseconds, or `None` if it was not stated.

    RFC 3339 with a `Z`, which `fromisoformat` learned to read in 3.11 — and an
    unparseable stamp answers `None` rather than raising, because "no expiry
    announced" is a state `FileHandle` already has a meaning for and a failed turn
    over a date string is not.
    """
    if not isinstance(stated, str) or not stated:
        return None
    try:
        return int(datetime.fromisoformat(stated.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _call_names(messages: Any) -> dict[str, str]:
    """`tool_call_id → name`, so a result can be addressed the way this wire does.

    A `functionResponse` carries the function's **name**; pH's `ToolResultBlock`
    carries the call's **id**, because that is what every other wire pairs on. The
    map is rebuilt per request from the assistant messages already in history
    rather than kept as adapter state: a resumed session's first request has no
    state to have kept, and a map that was only right for calls this process saw
    would silently mis-address every result after a restart.
    """
    names: dict[str, str] = {}
    for message in messages:
        for block in message.content:
            if getattr(block, "type", "") == "tool-call":
                names[block.id] = block.name
    return names


def _to_google(
    message: Any, media: dict[str, str], handles: dict[str, str], names: dict[str, str]
) -> dict[str, Any] | None:
    """One pH message as one `contents` entry, or `None` if it says nothing here.

    `None` rather than an empty entry: a message whose only content was a
    reasoning block has nothing this wire can carry, and an entry with no parts is
    a request error rather than a no-op.
    """
    parts: list[dict[str, Any]] = []
    for block in message.content:
        kind = getattr(block, "type", "")
        attachment = attachment_of(block)
        if attachment is not None:
            parts.append(_media_part(attachment, media, handles))
        elif kind == "text":
            parts.append({"text": block.text})
        elif kind == "tool-call":
            try:
                arguments = json.loads(block.arguments) if block.arguments else {}
            except json.JSONDecodeError:
                arguments = {}
            parts.append({"functionCall": {"name": block.name, "args": arguments}})
        elif kind == "tool-result":
            parts.append(
                {
                    "functionResponse": {
                        "name": names.get(block.tool_call_id, block.tool_call_id),
                        "response": {"output": _result_text(block)},
                    }
                }
            )
        # A `reasoning` block falls through deliberately — see the module
        # docstring. There is no input shape for a thought here, and rendering one
        # as text would make the transcript claim the model said what it was only
        # considering.
    if not parts:
        return None
    return {"role": "model" if message.role == "assistant" else "user", "parts": parts}


def _result_text(block: Any) -> str:
    return "\n".join(inner.text for inner in block.content if getattr(inner, "type", "") == "text")


def _media_part(
    attachment: AttachmentRef, media: dict[str, str], handles: dict[str, str]
) -> dict[str, Any]:
    """One attachment as `fileData` or `inlineData`, or a pointer if neither.

    Total over its own vocabulary like every sibling renderer — but this wire's
    vocabulary is unusually total already: `inlineData` takes any MIME the route
    accepts, so the only way here is a blob that could not be read, which is the
    race `media_pointer` exists for.
    """
    handle = handles.get(attachment.attachment_id)
    if handle is not None:
        return {"fileData": {"mimeType": attachment.mime, "fileUri": handle}}
    data = media.get(attachment.attachment_id)
    if data is None:
        return media_pointer(attachment)
    return {"inlineData": {"mimeType": attachment.mime, "data": data}}


@dataclass(frozen=True, slots=True)
class _Call:
    """One function call this wire streamed, with the id pH gave it.

    Named fields rather than a 4-tuple because `finish` unpacks it a screen away
    from where it is built, and two of the four read identically in the wrong
    order. A dataclass rather than a `NamedTuple` because `index` is exactly the
    field name a tuple already uses for a method.
    """

    index: int
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class _StreamState:
    """Turns this wire's chunks into pH chunks, tracking block indexes.

    Text arrives as whole parts rather than as deltas here, which is not a reason
    to emit whole blocks: `assistant/chunk` promises token-level replay, so each
    part is emitted as a delta on the block it continues — the same fidelity the
    other two adapters keep, arriving in bigger pieces.
    """

    text_index: int | None = None
    reasoning_index: int | None = None
    text: str = ""
    reasoning: str = ""
    calls: list[_Call] = field(default_factory=list)
    next_index: int = 0
    usage: TokenUsage | None = None
    finish_reason: str | None = None

    def _claim(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index

    def consume(self, payload: dict[str, Any]) -> list[Any]:
        out: list[Any] = []
        raw_usage = payload.get("usageMetadata")
        if isinstance(raw_usage, dict):
            self.usage = _to_usage(raw_usage)
        for candidate in payload.get("candidates") or ():
            reason = candidate.get("finishReason")
            if isinstance(reason, str):
                self.finish_reason = reason
            content = candidate.get("content") or {}
            for part in content.get("parts") or ():
                out.extend(self._part(part))
        return out

    def _part(self, part: dict[str, Any]) -> list[Any]:
        out: list[Any] = []
        call = part.get("functionCall")
        if isinstance(call, dict):
            # An id is minted here because this wire has none. `call-<n>` per
            # request rather than a uuid: it is only ever paired within one
            # conversation, and a stable spelling keeps a replayed session's ids
            # identical to the live run's.
            streamed = _Call(
                index=self._claim(),
                id=f"call-{len(self.calls) + 1}",
                name=str(call.get("name") or ""),
                arguments=json.dumps(call.get("args") or {}),
            )
            self.calls.append(streamed)
            out.append(BlockStart(index=streamed.index, block_type="tool-call"))
            out.append(
                ToolCallDelta(
                    index=streamed.index,
                    id=streamed.id,
                    name=streamed.name,
                    arguments_delta=streamed.arguments,
                )
            )
            return out
        text = part.get("text")
        if not isinstance(text, str) or not text:
            return out
        if part.get("thought"):
            if self.reasoning_index is None:
                self.reasoning_index = self._claim()
                out.append(BlockStart(index=self.reasoning_index, block_type="reasoning"))
            self.reasoning += text
            out.append(ReasoningDelta(index=self.reasoning_index, text=text))
            return out
        if self.text_index is None:
            self.text_index = self._claim()
            out.append(BlockStart(index=self.text_index, block_type="text"))
        self.text += text
        out.append(TextDelta(index=self.text_index, text=text))
        return out

    def _kind(self) -> FinishKind:
        if self.calls:
            return "tool-calls"
        return _FINISH_KINDS.get(self.finish_reason or "", "stop")

    def finish(self) -> list[Any]:
        out: list[Any] = []
        if self.reasoning_index is not None:
            out.append(
                BlockEnd(index=self.reasoning_index, block=ReasoningBlock(text=self.reasoning))
            )
        if self.text_index is not None:
            out.append(BlockEnd(index=self.text_index, block=TextBlock(text=self.text)))
        for streamed in self.calls:
            out.append(
                BlockEnd(
                    index=streamed.index,
                    block=ToolCallBlock(
                        id=streamed.id, name=streamed.name, arguments=streamed.arguments
                    ),
                )
            )
        if self.usage is not None:
            out.append(UsageChunk(usage=self.usage))
        # A tool call is `STOP` on this wire, so the finish kind comes from what
        # was streamed rather than from what the provider called it — the same
        # fact `_finish_kind` reads off `finish_reason` elsewhere, derived here
        # because this provider does not report it.
        out.append(Finish(reason=FinishReason(kind=self._kind())))
        return out


def _to_usage(raw: dict[str, Any]) -> TokenUsage:
    """`usageMetadata` as pH's disjoint counts (D15).

    Two corrections, and both are the kind that report a smaller bill than the
    real one. Cached input is *inside* `promptTokenCount` here, so it is
    subtracted out the way DeepSeek's is. And `thoughtsTokenCount` is *outside*
    `candidatesTokenCount`, where pH's `reasoning_tokens` is documented as a
    subset of `output_tokens` — so it is added in, or every thinking turn's output
    is under-reported by the part that did the work.
    """
    prompt = int(raw.get("promptTokenCount") or 0)
    cached = int(raw.get("cachedContentTokenCount") or 0)
    thoughts = int(raw.get("thoughtsTokenCount") or 0)
    return TokenUsage(
        input_tokens=max(0, prompt - cached),
        output_tokens=int(raw.get("candidatesTokenCount") or 0) + thoughts,
        cache_read_tokens=cached or None,
        reasoning_tokens=thoughts or None,
    )


@plugin("llm-google", config=Config, inject=["llm", "credentials"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the Google route."""
    adapter = GoogleAdapter(ctx=ctx, config=config)
    handle = ctx.llm.register_adapter([config.provider], adapter)
    ctx.add_disposer(handle.dispose, label=f"llm({config.provider})")
    uploads = ctx.get("uploads")
    if uploads is not None and config.uploads:
        ctx.add_disposer(
            uploads.register_uploader(config.provider, adapter),
            label=f"uploader({config.provider})",
        )
    ctx.add_disposer(adapter.http.aclose, label=f"http({config.provider})")
