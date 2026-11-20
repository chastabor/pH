"""`llm-google`: the Gemini `generateContent` wire, and the Files API that video needs.

This is the third adapter. It exists because of P7-03, and not merely alongside
it. The gate of that row is *a video reaches a route that accepts one*. The notes
of the row said what stood in the way: **neither provider that pH had an adapter
for accepts video**. A route that declared video would have been a fiction, and
the transport was proven against a shape that does not exist. This is the
provider that takes a video.

It is a separate adapter and not a flag, for the reason that `anthropic.py` gives
about itself. The reason is stronger here, because almost nothing on this wire
has the same name as anywhere else:

* Messages are `contents`, and content is `parts`.
* The system prompt is `systemInstruction`.
* Tools are `functionDeclarations`.
* A tool result is a `functionResponse` part that is addressed **by name** and
  not by id.
* Usage is `usageMetadata`, and the thinking tokens are counted *outside* the
  tokens of the answer.

Three mappings are worth naming, because in each of them a reasonable guess is
wrong:

* **`thoughtsTokenCount` is not inside `candidatesTokenCount`.** The
  `reasoning_tokens` of pH is documented as a *subset* of `output_tokens`, and
  `total` leaves it out for that reason. So this adapter adds it in. A direct
  mapping would under-report the output of every thinking turn.
* **A function call has no id on this wire.** The `ToolCallBlock` of pH needs
  one, and every consumer pairs by id. So the call goes up with an empty id, and
  `named_call` (`ph.llm.assembler`) gives it one. The other two wires rely on
  the same function when a provider omits an id (G10). The id is built from the
  turn, the step, the attempt and the block index, so it cannot repeat. What
  stays here is the reverse map. It is rebuilt when a result goes back out as a
  `functionResponse`, which carries the *name* and not the id.
* **A thought does not go back.** There is no input shape for one. To render it
  as a text part would put the private reasoning of the model into the
  conversation as something that the model said. The log still holds the
  thought, and the request does not.

@module ph_app.adapters.google
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, assert_never

import anyio

from ph.cordis import Context, plugin
from ph.json import as_int, as_str
from ph.keys import ATTACHMENTS, CREDENTIALS, LLM, UPLOADS
from ph.llm.adapter import LlmError, ResolvedModel, resolved
from ph.llm.types import (
    TIMEOUT,
    AttachmentRef,
    BlockEnd,
    BlockStart,
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
    text_of,
    wire_effort,
)
from ph.seams.uploads import FileHandle
from ph.session import now_ms
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret, wire_error_finish
from ._media import forget_named_handle, load_handles, load_media, media_pointer

__all__ = ["GoogleAdapter", "apply"]

log = logging.getLogger("ph_app.adapters.google")

SAMPLING_WIRE: dict[SamplingSetting, str] = {
    "temperature": "temperature",
    "topP": "topP",
    "topK": "topK",
    "presencePenalty": "presencePenalty",
    "frequencyPenalty": "frequencyPenalty",
}
"""Each sampling setting that this API takes, and its `generationConfig` field (A1).
That is every setting except `minP`, which is dropped with a warning
(`ResolvedModel.sampling`)."""

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
"""The default for `Config.accepts`: what the models of this provider take.

This is the widest list of the three, and video is the entry that matters. Video
is the format that made upload-and-reference a row and not an optimization. Until
this adapter existed, no route that pH shipped could declare video truthfully."""

UPLOADED_MEDIA: tuple[str, ...] = (
    "video/mp4",
    "video/mpeg",
    "video/webm",
    "video/quicktime",
)
"""The default for `Config.uploads`: video, always by reference.

**This default is non-empty, and both sibling rows default to nothing. The
difference is about the provider and not about confidence.** The Files API of
Anthropic is behind a beta that an account may not have. An OpenAI-compatible
route may be a gateway that never implemented `/files`. Neither caveat applies to
the service of this provider itself. There the Files API is the documented path,
and a request that carries a video inline is bounded by a total request size of
20 MB, which most videos exceed.

A gateway that pretends to be Gemini sets `uploads: []` and gets inline bytes.
The other two rows have the same override in the other direction."""

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
"""The inline ceiling: this provider limits the whole *request* to 20 MB.

That limit is the argument for the Files API. It is not a limit on what the
provider accepts: a 300 MB video is fine by reference and impossible inline. So
this number is what a block must fit under to go as `inlineData`."""

UPLOAD_POLL_MS = 500
UPLOAD_READY_MS = 120_000
"""How long one attempt waits for an uploaded file to become usable.

Video is **processed** after it is stored, and a `fileUri` that is referenced
before its file reaches `ACTIVE` is refused. So this is the one uploader of the
three that is not done when the bytes have arrived. The value is two minutes
because that is the scale of a few minutes of video being transcoded at the far
end. Running out of time is also not a failure: `load_handles` catches the error
and the block goes inline, and a video over the inline cap becomes a pointer.

**Running out of time no longer discards the transfer.** It did before. Nothing
reached `_store`, so the next step of the same loop read the blob again, sent
every byte again, and waited the budget again. A 300 MB clip was transferred
again on each step, and each attempt left another copy against the Files quota of
the account. `_pending` now remembers the `files/<name>`, so the next attempt
polls the file that this attempt stored. The memo cannot save the local read,
because `handle_for` loads the blob before it calls an uploader at all. It saves
the sending.

So this number is the patience of **one attempt**, not a total. A file that needs
ten minutes of processing is reached by whichever step comes after the file is
ready. The steps in between degrade to a pointer and say so."""


MAX_TRANSFERS = 3
"""How many times the bytes of one attachment are sent before this route stops trying.

**Nothing above this number limits an upload.** `load_handles` catches every
failure and sends the block inline, so an upload never reaches `llm-retry` and
its `max_attempts`. That setting governs the model *request* and not the upload.
What remains is one attempt for each model step, for as long as the attachment is
in the conversation. For a hard failure, each attempt is a full transfer: a
300 MB clip that a route refuses would be sent on every step of a fifty-step
session.

The number counts **transfers, not attempts**, and the design depends on that
distinction. A run that timed out with a resumable file remembered costs one
`GET` on the next step, and waiting makes it cheaper. A cap on that would abandon
a legitimately slow transcode, which is the case that `_pending` exists to
survive. The cap is on paying for the bytes again, which is the cost that does
not improve with repetition.

The value is three because the failures that are worth a retry are transient
ones: a 5xx, or a dropped connection. A fault that survives three sends is a fact
about the file or the account. After that the block degrades to a pointer. A
route with no file API gives the same answer, and the model can read it.

The count is kept for each process, and deliberately not persisted. A restart is
a new judgement, for the reason that the note on `_pending` gives."""


class Config(WireModel):
    """Row config for the Google route."""

    provider: str = "google"
    """The name that this route registers under. A request that names this
    provider goes to this adapter. The file handles of the route are kept under
    the same name."""
    base_url: str = BASE_URL
    """The root address of the API that this route calls. The adapter appends
    `/models/<model>:streamGenerateContent` for a model request. For an upload
    it replaces `/v1beta` in this address with `/upload/v1beta`."""
    api_key_env: str = "GEMINI_API_KEY"
    """The name of the environment variable that holds the API key. It is a
    name and never the key itself. The adapter resolves the name when it builds
    a request and sends the value in the `x-goog-api-key` header. A request
    fails with `MISSING_CREDENTIAL` when the variable is not set."""
    context_window: int | None = 1_048_576
    """The window that pH budgets and compacts against, in tokens. `None` means
    that the route declares no window, and then nothing budgets or compacts
    against one."""
    default_max_tokens: int | None = None
    """The `maxOutputTokens` that a request carries when the request itself
    states none. `None`, the default, means that such a request carries no
    `maxOutputTokens`. The route also reports this value to the layers above
    (`ResolvedModel.default_max_tokens`)."""
    accepts: tuple[str, ...] = ACCEPTED_MEDIA
    """The MIME types that this route takes as message content (P7-01)."""
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES
    """The largest attachment that this route sends, in bytes. The
    `media-degrade` row replaces a larger attachment with a text pointer before
    the request reaches this adapter. The default is 20 MiB, because this
    provider limits a whole request to 20 MB."""
    uploads: tuple[str, ...] = UPLOADED_MEDIA
    """The MIME types that this route sends by reference through the Files API (P7-03)."""
    upload_ready_ms: int = UPLOAD_READY_MS
    """How long one upload attempt waits for a stored file to become usable, in
    milliseconds. The provider processes a video after it stores it, and the
    adapter polls until the file is no longer `PROCESSING`. When the time runs
    out, the attachment goes inline on that step. The next attempt polls the
    file that this attempt stored and does not send the bytes again. The
    default is two minutes."""
    max_image_edge: int | None = None
    """The longest edge of an image that this route accepts, in pixels, when a
    deployment knows it. The `media-degrade` row replaces a larger image with a
    text pointer before the request reaches this adapter.

    The default is `None`, like that of the OpenAI row and unlike the stated
    number of the Anthropic row. `None` means that no image is refused for its
    pixel size."""
    usable_image_edge: int | None = None
    """The longest edge of an image that this route uses, in pixels, when a
    deployment knows it. A larger image is still sent whole, and the
    `media-degrade` row warns that the extra pixels are uploaded and discarded.

    The default is `None`, like that of the OpenAI row and unlike the stated
    number of the Anthropic row. This provider tiles large images and does not
    publish one edge that it scales to. A number invented here would warn the
    user about an overpayment that is not happening. `None` means that no such
    warning is given."""


_MISSING_FILE_PHRASES = ("is not in an active state", "was not found", "not found", "permission")
"""How this provider says that a referenced file is gone or unusable.

The tuple is separate, like the tuple of each sibling, because this is a fact
about the prose of one provider. This one also differs in kind. A file here can
be *present and not ready*. That reads as a different sentence and needs the same
remedy, because the cached handle is one that this request cannot use.

`permission` is in the list because a file that was deleted from another session
answers `PERMISSION_DENIED` and not a not-found. As everywhere, the match is used
together with the check that the message names an id that this request sent."""


def _is_missing_file(message: str) -> bool:
    lowered = message.lower()
    return any(phrase in lowered for phrase in _MISSING_FILE_PHRASES)


def _is_overflow(body: str) -> bool:
    lowered = body.lower()
    return "token count" in lowered or "exceeds the maximum" in lowered


_FINISH_KINDS: dict[str, FinishKind] = {"MAX_TOKENS": "max-tokens", "STOP": "stop"}
"""The finish reasons of this wire, and what pH calls them.

Everything that is not named here (`SAFETY`, `RECITATION`, `OTHER`) maps to
`stop`. That is the correct mapping and not a lazy one: the model stopped, the
transcript holds what it produced, and the kinds of pH have no member that means
"refused". To invent `error` for a completed response would make the retry policy
run again a request that the provider will answer in the same way."""


@dataclass(slots=True)
class GoogleAdapter:
    """Streams one Gemini route."""

    ctx: Context
    config: Config
    http: HttpClient = field(default_factory=HttpClient)
    _pending: dict[str, str] = field(default_factory=dict)
    """Maps `attachment_id → files/<name>` for a transfer that finished and whose
    file is not usable yet. See `_resume`.

    It limits itself and has no cap. An entry is written when the bytes are
    stored. It is removed when the file is usable or turns out to be gone. So it
    holds what this process is still waiting on and nothing else. A restart loses
    it and pays for the transfer again, which is the behavior that this replaced."""
    _spent: dict[str, int] = field(default_factory=dict)
    """How many transfers each attachment has cost without producing a handle.

    The count is cleared when a transfer succeeds, so it counts *consecutive*
    failures and not a lifetime total. An attachment that worked, expired and is
    uploaded again starts from zero. That is the ordinary path and not a fault.
    See `MAX_TRANSFERS`."""

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

        **This is the only uploader of the three that is not finished when the
        bytes have arrived.** A video is processed after it is stored, and a
        `fileUri` that is referenced before its file reaches `ACTIVE` is refused.
        An uploader that returned at the end of the transfer would return a
        handle whose first use fails. To the cache, that handle would look the
        same as one that the provider expired.

        The transfer itself is the resumable protocol of this provider. Its two
        steps are the reason why `post_raw` exists. The first step is a JSON body
        that is answered with an empty body and a destination in
        `X-Goog-Upload-URL`. The second step is the bytes of the file with no
        form encoding at all.

        **A transfer that already happened is not paid for twice.** That is the
        purpose of `_pending`. Before it, running out of the readiness budget
        discarded the stored file too, so the next step sent every byte again
        and waited again. Now the name is remembered and the next attempt polls
        it.

        `load_handles` catches a failure of any kind here, and the attachment
        goes inline. So the worst outcome of a slow provider is the behavior that
        every route had before the row.
        """
        spent = self._spent.get(ref.attachment_id, 0)
        if spent >= MAX_TRANSFERS:
            # Refused before the bytes move. A route that cannot take this file
            # then costs one dictionary lookup for each step and not a transfer.
            raise LlmError(
                f"{ref.name or ref.attachment_id} failed to upload {spent} times; "
                "sending it inline instead",
                "REQUEST_FAILED",
            )
        resumed = await self._resume(ref)
        try:
            record = resumed if resumed is not None else await self._transfer(ref, content)
            uri = as_str(record.get("uri"))
            if not uri:
                raise LlmError("the files API returned no uri", "REQUEST_FAILED")
            # The name is remembered when the bytes are stored, not when the wait
            # for them runs out. Otherwise a turn that stopped while the provider
            # processed the file (an interrupt, a release) forgot a transfer that
            # had happened, and the next step sent every byte again.
            if name := as_str(record.get("name")):
                self._pending[ref.attachment_id] = name
            record = await self._ready(record, ref.attachment_id)
        except Exception:
            # A failure is counted only when this attempt has nothing to resume
            # from. A timeout that left a remembered file did *not* pay for the
            # bytes, and the next step will not pay either. To count it here
            # would abandon a slow transcode because it is slow.
            if ref.attachment_id not in self._pending:
                self._spent[ref.attachment_id] = spent + 1
            raise
        # The file is ready, and the seam is about to cache it. Neither record
        # has anything more to say about this attachment.
        self._pending.pop(ref.attachment_id, None)
        self._spent.pop(ref.attachment_id, None)
        return FileHandle(
            provider=self.config.provider,
            attachment_id=ref.attachment_id,
            handle=uri,
            uploaded_at=now_ms(),
            expires_at=_expiry(record.get("expirationTime")),
        )

    async def _resume(self, ref: AttachmentRef) -> dict[str, Any] | None:
        """The file that a previous attempt already stored for these bytes, or `None`.

        **This is the purpose of the memo, and the reason why the memo is here
        and not in `ctx.uploads`.** That seam caches a `FileHandle`, which is a
        *usable* id. A file that is still `PROCESSING` is not usable. To store it
        would give the next request a reference that the provider refuses, and
        would turn a repeated upload into a retry loop. What is safe to keep is
        the weaker fact that a transfer happened. That fact never has to cross
        the seam: `upload` owns the whole transfer and wait, and the row
        registers one long-lived adapter, so this side of the boundary can hold
        it.

        Anything other than a readable record drops the entry and returns `None`,
        which pays for the transfer again. That covers a file that was deleted
        from elsewhere, an expiry, and a revoked key. All of them answer here and
        not in the middle of a turn, because this is the first request that asks
        about the file.

        The memo does **not** save the local read. `handle_for` loads the blob
        before it calls an uploader at all, so the bytes are in memory in both
        cases. It saves the sending, which is the cost that matters for the
        format that this exists to serve.
        """
        name = self._pending.get(ref.attachment_id)
        if name is None:
            return None
        try:
            record = _file_record(await self._file(name))
        except Exception:
            log.info("ph_app.adapters.google: %s is no longer there; re-uploading", name)
            self._pending.pop(ref.attachment_id, None)
            return None
        return record

    async def _transfer(self, ref: AttachmentRef, content: bytes) -> dict[str, Any]:
        """The two steps of the resumable protocol: start, then the bytes."""
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
        return _file_record(stored)

    async def _file(self, name: str) -> dict[str, Any]:
        """One `GET files/<id>`, the only way to ask what state a file is in."""
        return await self.http.get_json(
            f"{self.config.base_url.rstrip('/')}/{name}",
            headers=self._headers(),
            is_overflow=_is_overflow,
        )

    async def _ready(self, record: dict[str, Any], attachment_id: str) -> dict[str, Any]:
        """Poll until the file is `ACTIVE`, or give up inside the budget.

        Giving up leaves the file **remembered**, as `upload` recorded it when it
        was stored. The next attempt then resumes against it and does not send it
        again. `_resume` is the other half. `FAILED` forgets the file, and the
        difference is deliberate. The provider has said that this file will never
        work, so the useful action is to fall back to the bytes. A slow file is
        worth waiting for on the next step.
        """
        name = as_str(record.get("name"))
        waited = 0
        while as_str(record.get("state"), "ACTIVE") == "PROCESSING":
            if waited >= self.config.upload_ready_ms:
                raise LlmError(f"{name} was still processing after the upload budget", TIMEOUT)
            await anyio.sleep(UPLOAD_POLL_MS / 1000)
            waited += UPLOAD_POLL_MS
            record = _file_record(await self._file(name))
        if as_str(record.get("state")) == "FAILED":
            self._pending.pop(attachment_id, None)
            raise LlmError(f"{name} could not be processed", "REQUEST_FAILED")
        return record

    # ----------------------------------------------------------- streaming --

    async def _body(self, options: GenerateOptions) -> tuple[dict[str, Any], dict[str, str]]:
        """The request, and the provider file ids it was built from. See `_body` in
        `anthropic`."""
        handles = await load_handles(
            self.ctx.get(UPLOADS),
            options.messages,
            provider=self.config.provider,
            mimes=frozenset(self.config.uploads),
            session_id=options.session_id,
        )
        media = await load_media(self.ctx.get(ATTACHMENTS), options.messages, skip=handles.keys())
        names: dict[str, str] = {}
        contents: list[dict[str, Any]] = []
        for message in options.messages:
            _remember_calls(message, names)
            if (content := _to_google(message, media, handles, names)) is not None:
                contents.append(content)
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
        generation.update(options.sampling.on_wire(SAMPLING_WIRE))
        max_tokens = options.max_tokens or self.config.default_max_tokens
        if max_tokens is not None:
            generation["maxOutputTokens"] = max_tokens
        if options.stop:
            generation["stopSequences"] = list(options.stop)
        if options.response_schema is not None:
            # **The mime type is sent and the schema is not.** That is a decision
            # and not an omission. The `responseSchema` of this provider is an
            # OpenAPI subset that rejects keywords that ordinary JSON Schema
            # carries. To forward the schema of pH would fail requests for a
            # capability that `resolve_model`, below, says this route does not
            # have. Asking for JSON is the half that is true. The caller
            # validates in both cases, and `structured_output: False` exists to
            # tell it so.
            generation["responseMimeType"] = "application/json"
        if generation:
            body["generationConfig"] = generation
        wire_effort(body, options)
        return body, handles

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
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
                    if is_error_finish(chunk):
                        # See the same return in the other two adapters. The wire
                        # has said that the request failed, and `finish()` would
                        # follow that with a second `Finish` that reads "stop".
                        return
        except LlmError as error:
            raise forget_named_handle(
                self.ctx, error, referenced, provider=self.config.provider
            ) from error
        for chunk in state.finish():
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        # See `_body`. This route is asked for JSON and is not held to a schema,
        # so it does not enforce one. To claim otherwise would make a caller skip
        # the validation that does the actual work.
        return resolved(
            self.config,
            structured_output=False,
            credential=self.config.api_key_env,
            sampling=frozenset(SAMPLING_WIRE),
            max_decision_tokens=None,
        )


def _file_record(payload: dict[str, Any]) -> dict[str, Any]:
    """The fields of the file itself, with or without a `file` envelope around them.

    Both spellings occur. The resumable finalize answers `{"file": {...}}`, and a
    `GET` on the file answers the record itself. A caller that guessed wrong
    would read every field as absent. For `state`, absent means "already ACTIVE",
    and the caller would return a handle whose first use fails.
    """
    inner = payload.get("file")
    return inner if isinstance(inner, dict) else payload


def _expiry(stated: object) -> int | None:
    """`expirationTime` as epoch milliseconds, or `None` if it was not stated.

    The stamp is RFC 3339 with a `Z`, which `fromisoformat` learned to read in
    3.11. A stamp that cannot be parsed answers `None` and does not raise.
    `FileHandle` already has a meaning for "no expiry announced", and a date
    string is not a reason to fail a turn.
    """
    if not isinstance(stated, str) or not stated:
        return None
    try:
        return int(datetime.fromisoformat(stated.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _remember_calls(message: Message, names: dict[str, str]) -> None:
    """Record the calls of this message, so a later result resolves to the nearest one.

    A `functionResponse` carries the **name** of the function. The
    `ToolResultBlock` of pH carries the **id** of the call, because every other
    wire pairs on the id.

    **The map is updated along the conversation and is not built over it first**
    (G1). A flat map that is built in one pass keeps the *last* binding for each
    id. A session that carried the duplicate ids that this wire used to mint
    then resolved every earlier result to the name of the newest call. It did so
    silently, and in the direction that tells the model that a result came from a
    tool that it did not call. When the map is updated as each message is
    converted, a result sees the calls that precede it, which is the pairing that
    the transcript describes. `named_call` (`ph.llm.assembler`) stops new
    duplicates. This function makes the duplicates that are already in stored
    sessions resolve correctly.
    """
    for block in message.content:
        if isinstance(block, ToolCallBlock):
            names[block.id] = block.name


def _to_google(
    message: Message,
    media: dict[str, str],
    handles: dict[str, str],
    names: dict[str, str],
) -> dict[str, Any] | None:
    """One pH message as one `contents` entry, or `None` if it says nothing here.

    The answer is `None` and not an empty entry. A message whose only content was
    a reasoning block has nothing that this wire can carry, and an entry with no
    parts is a request error and not a no-op.
    """
    parts: list[dict[str, Any]] = []
    for block in message.content:
        match block:
            case TextBlock():
                parts.append({"text": block.text})
            case ToolCallBlock():
                try:
                    arguments = json.loads(block.arguments) if block.arguments else {}
                except json.JSONDecodeError:
                    arguments = {}
                parts.append({"functionCall": {"name": block.name, "args": arguments}})
            case ToolResultBlock():
                parts.append(
                    {
                        "functionResponse": {
                            "name": names.get(block.tool_call_id, block.tool_call_id),
                            "response": {"output": text_of(block.content)},
                        }
                    }
                )
            case ReasoningBlock():
                # Dropped deliberately. See the module docstring. There is no
                # input shape for a thought here, and to render one as text would
                # make the transcript claim that the model said what it was only
                # considering.
                pass
            case MediaBlock():
                parts.append(_media_part(block.attachment, media, handles))
            case _ as unhandled:
                assert_never(unhandled)
    if not parts:
        return None
    return {"role": "model" if message.role == "assistant" else "user", "parts": parts}


def _media_part(
    attachment: AttachmentRef, media: dict[str, str], handles: dict[str, str]
) -> dict[str, Any]:
    """One attachment as `fileData` or `inlineData`, or a pointer if neither.

    The function is total over its own vocabulary, like every sibling renderer.
    The vocabulary of this wire is already unusually complete: `inlineData` takes
    any MIME that the route accepts. So the only way to the pointer is a blob
    that could not be read, which is the race that `media_pointer` exists for.
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
    """One function call that this wire streamed.

    It has named fields and is not a 3-tuple, because `finish` unpacks it a
    screen away from where it is built. It is a dataclass and not a `NamedTuple`,
    because `index` is a field name that a tuple already uses for a method.

    It has no `id`. This wire does not issue one, and this adapter no longer
    invents one (G10). The block goes up with an empty id, and `named_call`
    (`ph.llm.assembler`) fills it from the turn, the step, the attempt and the
    block index. The absent ids of the other two wires are filled in there too.
    """

    index: int
    name: str
    arguments: str


@dataclass(slots=True)
class _StreamState:
    """Turns the chunks of this wire into pH chunks, and tracks block indexes.

    Text arrives here as whole parts and not as deltas. That is not a reason to
    emit whole blocks, because `assistant/chunk` promises token-level replay. So
    each part is emitted as a delta on the block that it continues. That is the
    same fidelity that the other two adapters keep, in bigger pieces.
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

    def consume(self, payload: dict[str, Any]) -> list[StreamChunk]:
        error = payload.get("error")
        if isinstance(error, dict):
            # **A failed request inside a 200** (L2). This wire reports one as a
            # top-level `error` object mid-stream, exactly as the other two do.
            # Reading only `usageMetadata`/`candidates` dropped it. `finish()`
            # then emitted `Finish(stop)`, and the turn recorded a clean end for
            # a request that failed. This was the third shipped wire, and the
            # fix had never reached it. The fix has two halves here: this branch
            # and the return in `stream`. Both are needed because `finish()` is
            # unconditional, so a caller that kept reading got the `stop` in any
            # case.
            #
            # This wire spells the same field `status` and not `type`
            # (`RESOURCE_EXHAUSTED`, `UNAVAILABLE`). With it, the shared map has
            # what it needs to say whether a retry could help.
            return [
                wire_error_finish(
                    error,
                    kind=error.get("status"),
                    is_overflow=_is_overflow,
                    is_missing_file=_is_missing_file,
                )
            ]
        out: list[StreamChunk] = []
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

    def _part(self, part: dict[str, Any]) -> list[StreamChunk]:
        out: list[StreamChunk] = []
        call = part.get("functionCall")
        if isinstance(call, dict):
            # The id is left empty for `named_call` (`ph.llm.assembler`) to fill
            # (G10). This wire issues none, and a counter kept here could only
            # span one request, which is the collision that G1 was filed for.
            # `named_call` builds the id from the turn, the step, the attempt and
            # the block index. It is the one place where the unnamed calls of all
            # three wires get a name.
            streamed = _Call(
                index=self._claim(),
                name=as_str(call.get("name")),
                arguments=json.dumps(call.get("args") or {}),
            )
            self.calls.append(streamed)
            out.append(BlockStart(index=streamed.index, block_type="tool-call"))
            out.append(
                ToolCallDelta(
                    index=streamed.index,
                    id="",
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

    def finish(self) -> list[StreamChunk]:
        out: list[StreamChunk] = []
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
                    block=ToolCallBlock(id="", name=streamed.name, arguments=streamed.arguments),
                )
            )
        if self.usage is not None:
            out.append(UsageChunk(usage=self.usage))
        # A tool call is `STOP` on this wire, so the finish kind comes from what
        # was streamed and not from what the provider called it. Elsewhere
        # `_finish_kind` reads the same fact from `finish_reason`. It is derived
        # here because this provider does not report it.
        out.append(Finish(reason=FinishReason(kind=self._kind())))
        return out


def _to_usage(raw: dict[str, Any]) -> TokenUsage:
    """`usageMetadata` as the disjoint counts of pH (D15).

    There are two corrections, and both are of the kind that reports a smaller
    bill than the real one. Cached input is *inside* `promptTokenCount` here, so
    it is subtracted, in the way that the cached input of DeepSeek is.
    `thoughtsTokenCount` is *outside* `candidatesTokenCount`, and the
    `reasoning_tokens` of pH is documented as a subset of `output_tokens`. So it
    is added in. Otherwise the output of every thinking turn is under-reported by
    the part that did the work.
    """
    prompt = as_int(raw.get("promptTokenCount"))
    cached = as_int(raw.get("cachedContentTokenCount"))
    thoughts = as_int(raw.get("thoughtsTokenCount"))
    return TokenUsage(
        input_tokens=max(0, prompt - cached),
        output_tokens=as_int(raw.get("candidatesTokenCount")) + thoughts,
        cache_read_tokens=cached or None,
        reasoning_tokens=thoughts or None,
    )


@plugin("llm-google", affects="environment", config=Config, inject=[LLM, CREDENTIALS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the Google route."""
    adapter = GoogleAdapter(ctx=ctx, config=config)
    handle = ctx.require(LLM).register_adapter([config.provider], adapter)
    ctx.add_disposer(handle.dispose, label=f"llm({config.provider})")
    uploads = ctx.get(UPLOADS)
    if uploads is not None and config.uploads:
        ctx.add_disposer(
            uploads.register_uploader(config.provider, adapter),
            label=f"uploader({config.provider})",
        )
    ctx.add_disposer(adapter.http.aclose, label=f"http({config.provider})")
