"""`llm-openai-compatible`: the OpenAI chat-completions wire, DeepSeek included.

One adapter covers OpenAI, DeepSeek, and every gateway that speaks the same
shape, because the differences that matter are two fields:

* **`reasoning_content`**. DeepSeek streams thinking separately from text. pH
  has a distinct `reasoning` block for this, so the field is mapped and not
  concatenated. To put thinking into visible text would make the transcript
  claim that the model said things that it was only considering.
* **`prompt_cache_hit_tokens`**. DeepSeek counts cache hits inside
  `prompt_tokens`. The counts of the `TokenUsage` of pH are **disjoint** (D15),
  so the cached part is subtracted. If it stayed in, the accounting of pH would
  bill every cache hit twice.

Tool calls stream as incremental `arguments` deltas, and the adapter keeps that
fidelity. `assistant/chunk` promises token-level replay, and an adapter that
emitted only completed calls would silently break that promise.

The credential is resolved **here and in no layer above** (I-3). The seam passes
a `CredentialRef`, and only `_headers` turns one into a header.

A route may serve a *decision* model and not a chat model. `decide` posts
`/systemone`, which is the request that llama.cpp and Ollama answer beside this
wire.

@module ph_app.adapters.openai_compatible
"""

from __future__ import annotations

import logging
import math
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard, assert_never

import anyio

from ph.cordis import Context, plugin
from ph.json import as_int, as_obj, as_seq, as_str
from ph.keys import ATTACHMENTS, CREDENTIALS, LLM, UPLOADS
from ph.llm.adapter import (
    Answer,
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    Decisions,
    LlmError,
    Noul,
    NoulAnswer,
    Question,
    ResolvedModel,
    Score,
    ScoreAnswer,
    input_too_long,
    resolved,
)
from ph.llm.types import (
    BAD_DECISION,
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
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.uploads import FileHandle
from ph.session import now_ms
from ph.wire import WireModel

from ._http import HttpClient, resolve_secret, wire_error_finish
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
"""The default for `ProviderProfile.accepts`: the OpenAI baseline.

Images go as `image_url` data URIs and audio goes as `input_audio`. Those are two
wire shapes for two MIME families, which is the branching that a block union with
one member for each medium would have duplicated one layer up.

**A PDF can be expressed, and this default still does not claim it.** This wire
carries a document by `file_id` or inline as a `file` part (P7-03), so a route
that takes one sets `accepts: [application/pdf]`. But one `openai-compatible` row
serves gateways, local llama.cpp and Groq, and most of them do not take a PDF. A
default that claimed it would produce requests that the far end rejects for the
majority of routes."""

MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
"""The default per-attachment ceiling."""

UPLOAD_PURPOSE = "user_data"
"""What the Files API of this wire is told the bytes are for.

It is row config, below, and not a constant at the call site. `purpose` is the
one required form field of this API, and the gateways in front of it do not agree
about the vocabulary. `user_data` is what the chat completions of OpenAI itself
read as an input file. A compatible server may want `assistants`, or nothing at
all."""

SAMPLING_WIRE: dict[SamplingSetting, str] = {
    "temperature": "temperature",
    "topP": "top_p",
    "presencePenalty": "presence_penalty",
    "frequencyPenalty": "frequency_penalty",
    "topK": "top_k",
    "minP": "min_p",
}
"""The body field of each sampling setting on this wire (A1).

`top_k` and `min_p` are not fields of OpenAI. llama.cpp and vLLM take them as
extra body fields, and OpenAI refuses a field that it does not know. So a route
carries them only when its profile says that its server takes them
(`ProviderProfile.sampling_extras`)."""

OPENAI_SAMPLING: frozenset[SamplingSetting] = frozenset(
    {"temperature", "topP", "presencePenalty", "frequencyPenalty"}
)
"""What every server speaking this wire takes."""


class WindowProbe(WireModel):
    """One request that asks a server for its context window, as a profile states it.

    `path` is joined to the server root. That is *beside* `/v1` and not under it,
    because the endpoint that answers need not be part of the OpenAI wire.
    `field` is the path through the reply to the number.

    **A step is a key or an index.** `["data", 0, "meta", "n_ctx"]` reads the
    `/v1/models` of llama.cpp, and the same shape reaches the
    `data[0].max_model_len` of vLLM. `["default_generation_settings", "n_ctx"]`
    reads the `/props` of llama.cpp. An `int` step indexes a list and a `str`
    step reads an object, so the two kinds of JSON that a server answers with
    need no second field to tell them apart.

    The class is defined above `ProviderProfile` because that model annotates
    it. For a forward reference that it cannot resolve yet, pydantic leaves
    `__pydantic_complete__` false and rebuilds lazily at the first validation,
    which moves a structural error onto the mount path.
    `test_every_wire_model_is_built_at_import` checks this.
    """

    path: str
    """The path of the endpoint to ask, for example `/props`. It is joined to
    the server root, which is the `base_url` of the route without a trailing
    `/v1`."""
    field: tuple[str | int, ...]
    """The path through the reply to the number. Each step is a key of an
    object (`str`) or an index into a list (`int`), for example
    `["default_generation_settings", "n_ctx"]`. A reply that has no positive
    number at this path counts as no answer."""
    measures: Literal["request", "model"] = "request"
    """Whether the number is what one request may use, or what the model allows.

    **The path alone cannot make this distinction**, and it is the reason why the
    adapter does not take the first probe that answers. A `request` measurement
    is what this server will actually serve one caller, and it is the number that
    pH budgets and compacts against. The `/props` of llama.cpp reports the
    `n_ctx` of its slot, and the `max_model_len` of vLLM reports its own
    `--max-model-len`. A `model` measurement is the ceiling that the weights were
    built with, and a server is free to run far below it. A llama.cpp started
    with `-c 32768` still publishes `n_ctx_train` 262 144 at `/v1/models`, and to
    believe that number overflows every slot on the machine.

    This was measured against a llama.cpp that served Qwen3.8-27B, started with
    `-c 524288 --parallel 2`. `/props` and both `/slots` entries report 262 144,
    which is the total divided, as llama.cpp documents. `/v1/models` reports
    `meta.n_ctx` 262 144 beside `meta.n_ctx_train` 262 144. Its neighbors there
    are `n_vocab`, `n_embd`, `n_params` and `ftype`.

    **That server settles the question.** Its total was 524 288, and the number
    at `/v1/models` was 262 144. So `meta` is not the undivided total that this
    file once called it. It is the header of the GGUF itself. That is the reason
    why it matches `n_ctx_train` exactly, and why a server that runs below the
    ceiling of its model keeps publishing the ceiling.

    **So it is not a divisor.** A first design had this field say `slot | server`
    and divided a server-wide number by the `total_slots` of `/props`. On that
    machine the arithmetic turns a correct 262 144 into 131 072. The two probes
    disagree about *authority*, not about scale."""


class ProviderProfile(WireModel):
    """One route this adapter serves."""

    provider: str
    """The name that this route registers under. A request that names this
    provider goes to this route. The file handles of the route are kept under
    the same name."""
    base_url: str = "https://api.openai.com/v1"
    """The root address of the API that this route calls. The adapter appends
    `/chat/completions` for a model request, `/files` for an upload, and
    `/systemone` for a decision request. A probe of `context_window_probes`
    goes to this address without a trailing `/v1`."""
    api_key_env: str = "OPENAI_API_KEY"
    """The name of the environment variable that holds the API key. It is a
    name and never the key itself. The adapter resolves the name when it builds
    a request and sends the value as a `Bearer` token. A request fails with
    `MISSING_CREDENTIAL` when the variable is not set, also on a server that
    ignores the key."""
    context_window: int | None = None
    """The window that pH budgets and compacts against, in tokens.

    **On a llama.cpp route this is what a *slot* holds, not what the server
    holds.** Nobody can be expected to keep that number right by hand, and that
    is the reason for the field below. `context_window_probes` says where to
    ask, and this field is the fallback for a server that does not answer."""
    context_window_probes: tuple[WindowProbe, ...] = ()
    """Where to *ask* this server for its window at mount, **in order of trust.**

    **The profile declares the probes, and this file does not name them.** This
    adapter serves OpenAI, DeepSeek, llama.cpp, vLLM and every gateway in
    between, and the endpoints that answer this question belong to those servers.
    A boolean here could only mean "do the llama thing", so a second server would
    have arrived as an `elif` in shared code. With a path and a field, a second
    server arrives as four lines of YAML and no edit here.

    **It is a list, because one `baseUrl` outlives one server.** `LLAMA_BASE_URL`
    names a port, not an implementation. The same route is llama.cpp this morning
    and vLLM this afternoon. With a single probe, the second server silently
    keeps the fallback. That is the 32 768-against-262 144 mistake that this
    field exists to end, and it would return through the assumption about which
    server answers.

    **`measures` decides, and order only breaks a tie.** Every `request` probe
    outranks every `model` probe, wherever each one is in the list. What
    separates the `/props` of llama.cpp from its `/v1/models` is which question
    the number answers, and a reader of this file cannot see that in a path.
    Before, the order mattered and was invisible. To append a probe was safe, and
    to prepend one silently believed a model ceiling over a live slot. That is
    the same 32 768-against-262 144 failure, returning by a second route.

    This is a second field and not a keyword on the field above, because the
    case that matters is discovery *with* a fallback. "Ask, and here is what to
    do when the answer does not come" is two facts, and a single value cannot
    carry both.

    A server that answers none of the probes leaves `context_window` in force.
    Nothing refuses to start because of it. A window that pH guessed low is a
    session that compacts early, and a mount that refused is a user who cannot
    work."""
    default_max_tokens: int | None = None
    """The `max_tokens` that a request carries when the request itself states
    none. `None`, the default, means that such a request carries no
    `max_tokens`. The route also reports this value to the layers above
    (`ResolvedModel.default_max_tokens`)."""
    accepts: tuple[str, ...] = ACCEPTED_MEDIA
    """The MIME types that this route takes as message content (P7-01).

    The setting is for each *profile*, and that is its purpose. One
    `openai-compatible` row serves many routes (a hosted gateway, Groq, Together,
    a local llama.cpp), and they do not agree about media. A module constant made
    every one of them declare images and a 20 MB ceiling, so a text-only local
    model was promised capabilities that it does not have. A route that takes no
    media sets `accepts: []`."""
    max_attachment_bytes: int = MAX_ATTACHMENT_BYTES
    """The largest attachment that this route sends, in bytes. The
    `media-degrade` row replaces a larger attachment with a text pointer before
    the request reaches this adapter. The default is 20 MiB."""
    sampling_extras: tuple[Literal["topK", "minP"], ...] = ()
    """The sampling settings that this server takes beyond those of OpenAI (A1).
    llama.cpp and vLLM take `topK` and `minP`. The default is empty, because
    OpenAI refuses a body field that it does not know. The `topK` of a route is
    then dropped with a warning."""
    uploads: tuple[str, ...] = ()
    """The MIME types that this route sends by *reference* through the Files API
    and not inline (P7-03).

    The default is empty, like that of the Anthropic row. The reason belongs to
    this wire and is not a beta header: a route here may be anything that speaks
    the shape, and most gateways implement `/chat/completions` without
    implementing `/files` at all. A declared MIME that the wire cannot
    *reference* is removed by an intersection and is not uploaded. See `_body`.

    Where it applies, it is worth it: a 4 MB PDF is 5.5 MB of base64 on every
    step of a session, and the Files API is one upload and an id after that."""
    upload_purpose: str = UPLOAD_PURPOSE
    """The `purpose` form field that an upload to the Files API carries. It is
    the one form field that this API requires, and the gateways in front of it
    do not agree about the vocabulary. The default, `user_data`, is what the
    chat completions of OpenAI itself read as an input file. A compatible
    server may want `assistants`, or nothing at all."""
    max_image_edge: int | None = None
    """The longest edge of an image that this route accepts, in pixels, when
    the route publishes one (P7-03). The `media-degrade` row replaces a larger
    image with a text pointer before the request reaches this adapter.

    The default is `None`, and that is the correct value. Unlike `accepts`,
    where a wrong guess silently drops a block, an unknown pixel ceiling has no
    safe assumption. `None` means that no image is refused for its pixel size.
    A deployment that knows its route sets the number."""
    usable_image_edge: int | None = None
    """The longest edge of an image that this route uses, in pixels, when the
    route publishes one (P7-03). A larger image is still sent whole, and the
    `media-degrade` row warns that the extra pixels are uploaded and discarded.

    The default is `None`, and that is the correct value. Unlike `accepts`,
    where a wrong guess silently drops a block, an unknown pixel ceiling has no
    safe assumption. One route scales at 2048 and another does not scale at all.
    A profile that invented a number would warn the user about an overpayment
    that is not happening. `None` means that no such warning is given. A
    deployment that knows its route sets the number."""
    max_decision_tokens: int | None = None
    """The most tokens that one decision request may carry, for a route that
    serves a decision model (`ResolvedModel.max_decision_tokens`).

    It is written here because the server does not report it. A llama.cpp reads
    a decision in one physical batch, so this is its `--ubatch-size`, which is
    512 unless the server was started with more. Its `/props` publishes the
    context window and not the batch. When the setting is left out, an asker
    assumes the default and learns a smaller limit from the first refusal. Set
    it to show an asker more of what it decides about."""


class Config(WireModel):
    """Row config: the routes to register."""

    profiles: list[ProviderProfile] = field(default_factory=list)
    """The routes that this row registers. Each entry is one route: one
    provider name with its own address, credential name and limits. A profile
    has one `openai-compatible` row, so a second route is a second entry in
    this list and not a second row. The default is empty, and the row then
    registers nothing."""


_OVERFLOW_PHRASES = (
    "context length",
    "context window",
    "context size",
    "exceeds the available context",
    "exceed context",
    "too many tokens",
)
"""How the servers on this wire say that the request did not fit (G5).

These are phrases, and not "context" plus one of two words. That conjunction read
as a general rule, and was in fact a list of two spellings written apart.
llama.cpp says *"the request exceeds the available context size"*, which contains
"context" and neither "length" nor "window". So the shipped `llama` profile never
produced `CONTEXT_WINDOW_EXCEEDED`. That code triggers compaction, and it is the
only thing between an overlong conversation and a turn that fails every attempt.

The phrases are spelled out because a miss and a false positive have different
costs, and both are expensive. A missed overflow retries a request that cannot
work, and a false overflow compacts a conversation that fit. The docstring of
`failure_from_status` makes that argument for keeping the judgement in each wire.

This is not verified against a live llama.cpp. `./test.sh smoke` is the gate that
would verify it. The phrase is taken from the source of the server and not from a
session.
"""


def _is_overflow(body: str) -> bool:
    lowered = body.lower()
    return any(phrase in lowered for phrase in _OVERFLOW_PHRASES)


_MISSING_FILE_PHRASES = ("no such file", "file_not_found", "not_found", "not found")
"""How this provider says that a referenced file is gone.

The tuple is separate and not shared, although it currently agrees with the tuple
of the Anthropic row almost word for word. What a provider *says* is a fact about
the prose of that provider. A shared tuple would have to be the union of the
phrases of every wire, and that is how a `not_found` about a model name comes to
be classified as a missing file. What is really shared is what happens next
(`forget_named_handle`), and that is shared.

The match is used together with a check that the message names an id that this
request sent. That check is the half that makes the guess safe."""


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

    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle:
        """Give the bytes to the Files API and keep the id that it returns (P7-03).

        This is the `Uploader` half of `ctx.uploads` for this wire. `purpose` is
        the one form field that the endpoint requires. That is the reason why
        `post_multipart` got a `data` argument, and why this method does not
        build its own request.

        **The expiry is read when the provider states one.** The Anthropic row
        differs, because its provider publishes none for each file. `expires_at`
        is unix *seconds* here, and `FileHandle` is in milliseconds. The cheap
        mistake in that conversion would treat every handle as expired in 1970
        and upload again on every request. An absent value means "no expiry
        announced" and not "never expires". The mid-turn invalidation covers a
        file that goes away, and it does not depend on a truthful expiry.
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
        handle = as_str(reply.get("id"))
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
        """The request, and the provider file ids it was built from. See `_body` in
        `anthropic`."""
        handles = await load_handles(
            self.ctx.get(UPLOADS),
            options.messages,
            provider=self.profile.provider,
            # **Intersected with what this renderer can reference.** Here this
            # row diverges from the Anthropic row, deliberately. There, a
            # declared MIME with no wire shape (video) is uploaded and then
            # degrades to a pointer. Anthropic has no video content block *at
            # all*, so the pointer was the outcome in both cases and only the
            # upload was wasted. Here every MIME that can be declared is already
            # expressible inline. To upload one that this wire cannot reference
            # would turn a picture that works today into a pointer, because
            # `load_media` skips the bytes of anything that has a handle. A
            # configuration mistake must not be able to remove a capability.
            mimes=frozenset(self.profile.uploads) & _FILE_MIMES,
            session_id=options.session_id,
        )
        media = await load_media(self.ctx.get(ATTACHMENTS), options.messages, skip=handles.keys())
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
        body.update(
            options.sampling.on_wire({setting: SAMPLING_WIRE[setting] for setting in self.sampling})
        )
        max_tokens = options.max_tokens or self.profile.default_max_tokens
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if options.stop:
            body["stop"] = list(options.stop)
        if options.response_schema is not None:
            # `strict` makes this a constraint and not a hint. A server that
            # supports the field builds a grammar from the schema, and the reply
            # cannot come back in any other shape (P7-17). The caller validates
            # anyway. `structured_output` says whether this route enforces the
            # schema, and a caller must not have to know which kind it got.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "reply",
                    "schema": options.response_schema,
                    "strict": True,
                },
            }
        wire_effort(body, options)
        return body, handles

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
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
                    if is_error_finish(chunk):
                        # See the same return in the Anthropic adapter. The wire
                        # has said that the request failed, and `finish()` would
                        # follow that with a second `Finish` that reads "stop".
                        return
        except LlmError as error:
            # A handle that this request referenced is gone: it expired early,
            # was deleted from another session, or was never honored. Forget it
            # first, then let the request retry. `FILE_EXPIRED` is in
            # `TRANSIENT_CODES` because the state that caused it is already
            # cleared when the retry runs. The next attempt therefore uploads
            # again and does not repeat a request that cannot work.
            raise forget_named_handle(
                self.ctx, error, referenced, provider=self.profile.provider
            ) from error
        for chunk in state.finish():
            yield chunk

    async def decide(self, request: DecisionRequest) -> Decisions:
        """`/systemone`, the request that a decision model answers
        (`ph.llm.adapter.DecisionAdapter`).

        It is beside `/chat/completions` under the same base address, on the
        servers that have one: llama.cpp and Ollama. A server without it answers
        with a status and its own sentence, and that is the error. A server whose
        model is a chat model does the same. One request carries every question.
        The model scores and writes nothing, so the reply is whole and there is
        no stream.

        An input that the server will not take is `CONTEXT_WINDOW_EXCEEDED`, in
        the words that an embeddings server uses (`input_too_long`), because the
        limit is the same one. A llama.cpp reads a decision in one batch, and it
        refuses what does not fit the batch, whatever its context window is."""
        reply, _ = await self.http.post_raw(
            f"{self.profile.base_url.rstrip('/')}/systemone",
            headers=self._headers(),
            json=_systemone_body(request),
            is_overflow=input_too_long,
        )
        usage = as_obj(reply.get("usage")).get("input_tokens")
        return Decisions(
            answers=_answers(as_obj(reply.get("answers")), request.questions),
            input_tokens=usage if isinstance(usage, int) else None,
        )

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        # This wire takes `response_format: {"type": "json_schema", …}` with
        # `strict`, so the server builds a grammar and the reply cannot come back
        # another shape (P7-17).
        return resolved(
            self.profile,
            structured_output=True,
            credential=self.profile.api_key_env,
            sampling=self.sampling,
            max_decision_tokens=self.profile.max_decision_tokens,
        )

    @property
    def sampling(self) -> frozenset[SamplingSetting]:
        """The sampling settings that this route carries: those of OpenAI, and the
        extras of its server."""
        return OPENAI_SAMPLING | frozenset(self.profile.sampling_extras)


def _systemone_body(request: DecisionRequest) -> dict[str, Any]:
    """The `/systemone` request. `model` is left out when it is empty, because a
    server that serves one model needs no name. `state` is sent as text, which
    every server that answers this request takes."""
    body: dict[str, Any] = {
        "state": request.state,
        "questions": {name: _question(one) for name, one in request.questions.items()},
    }
    if request.model:
        body["model"] = request.model
    return body


def _question(question: Question) -> dict[str, Any]:
    """One question as this wire spells it. A `noul` takes no criteria. An option
    that has no text is sent as `null`."""
    match question:
        case Noul():
            return {"type": "noul", "instructions": question.instructions}
        case Choice():
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": {key: said or None for key, said in question.criteria.items()},
            }
        case Score():
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": list(question.criteria),
            }
        case _:
            assert_never(question)


def _answers(answers: Mapping[str, object], questions: Mapping[str, Question]) -> dict[str, Answer]:
    """The answer to each question, read as the kind that its question requires.

    A question that the reply does not answer is left out, and the seam refuses
    the whole reply because of it (`LlmRuntime.decide`). An answer in another
    shape is refused here, where the reply is available to say what was wrong
    with it."""
    found: dict[str, Answer] = {}
    for name, question in questions.items():
        answer = as_obj(answers.get(name))
        if not answer:
            continue
        match question:
            case Noul():
                found[name] = NoulAnswer(noul=_share(answer.get("noul"), name, "noul"))
            case Choice():
                choice = as_str(answer.get("choice"))
                if choice not in question.criteria:
                    raise _bad(name, f'chose "{choice}", which is not one of its criteria')
                found[name] = ChoiceAnswer(
                    choice=choice,
                    probabilities=_shares(answer, name, question.criteria),
                    confidence=_confidence(answer),
                )
            case Score():
                levels = [str(index) for index in range(len(question.criteria))]
                score = answer.get("score")
                if not _is_number(score):
                    raise _bad(name, 'has no number under "score"')
                found[name] = ScoreAnswer(
                    score=float(score),
                    probabilities=tuple(_shares(answer, name, levels).values()),
                    confidence=_confidence(answer),
                )
            case _:
                assert_never(question)
    return found


def _shares(answer: Mapping[str, object], name: str, keys: Iterable[str]) -> dict[str, float]:
    """An answer's `probabilities`, one share for each of `keys`, in their order."""
    said = as_obj(answer.get("probabilities"))
    return {key: _share(said.get(key), name, f"probabilities.{key}") for key in keys}


def _share(value: object, name: str, field: str) -> float:
    """A probability, or a refusal. Anything but a number from 0 to 1 means that
    the server answered in another shape. A rule that depends on the value would
    then be applied to a number that nobody gave."""
    if not _is_number(value) or not 0.0 <= value <= 1.0:
        raise _bad(name, f'has no probability under "{field}"')
    return float(value)


def _confidence(answer: Mapping[str, object]) -> float | None:
    """An answer's `confidence`, where the server gave one."""
    value = answer.get("confidence")
    return float(value) if _is_number(value) else None


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _bad(name: str, what: str) -> LlmError:
    return LlmError(f'the decision API\'s answer to "{name}" {what}', BAD_DECISION)


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

    def consume(self, payload: dict[str, Any]) -> list[StreamChunk]:
        # An error arrives here as an ordinary data frame. The response was a 200
        # and `failure_from_status` never saw it, so a stream that dies in the
        # middle of an answer looks exactly like one that ended early. When the
        # frame was ignored, the turn was recorded as a `stop` with whatever text
        # had arrived, and there was no retry. The wire said "over capacity", and
        # pH said "done".
        error = payload.get("error")
        if isinstance(error, dict):
            return [
                wire_error_finish(
                    error,
                    kind=error.get("type"),
                    is_overflow=_is_overflow,
                    is_missing_file=_is_missing_file,
                )
            ]
        out: list[StreamChunk] = []
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
                position = as_int(call.get("index"))
                if position not in self.tool_indexes:
                    self.tool_indexes[position] = self._claim()
                    self.tool_calls[position] = {"id": "", "name": "", "arguments": ""}
                    out.append(
                        BlockStart(index=self.tool_indexes[position], block_type="tool-call")
                    )
                record = self.tool_calls[position]
                if call.get("id"):
                    record["id"] = as_str(call["id"])
                function = call.get("function") or {}
                if function.get("name"):
                    record["name"] = as_str(function["name"])
                fragment = function.get("arguments") or ""
                record["arguments"] += fragment
                out.append(
                    ToolCallDelta(
                        index=self.tool_indexes[position],
                        id=record["id"],
                        name=record["name"] or None,
                        arguments_delta=fragment,
                    )
                )
        return out

    def finish(self) -> list[StreamChunk]:
        out: list[StreamChunk] = []
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
                        id=record["id"],
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


def _finish_kind(reason: str | None) -> FinishKind:
    if reason == "tool_calls":
        return "tool-calls"
    if reason == "length":
        return "max-tokens"
    return "stop"


def _to_usage(raw: dict[str, Any]) -> TokenUsage:
    """The usage of this wire, made disjoint (D15).

    **The cached prefix has three spellings, and here all of them are counted
    inside `prompt_tokens`.** That is the whole difference from the Anthropic
    adapter, where the same three counts arrive already separated and map across
    unchanged. DeepSeek says `prompt_cache_hit_tokens`. OpenAI and vLLM say
    `prompt_tokens_details.cached_tokens`. vLLM alone also says
    `created_cache_tokens`, for the part of *this* prompt that it just wrote into
    the prefix cache. Each is subtracted, because `TokenUsage.total` adds them
    back, and a count that stays inside `prompt_tokens` is billed twice. On a
    warm turn that is most of the window.

    **`created_cache_tokens` is a write, not a miss.** That distinction keeps the
    `prompt_cache_miss_tokens` of DeepSeek out of this function. Misses are the
    uncached remainder, which is already what `input_tokens` means. To read a
    miss as a write would set the input to zero and call the whole prompt a cache
    write.
    """
    prompt = as_int(raw.get("prompt_tokens"))
    prompt_details = raw.get("prompt_tokens_details") or {}
    cached = as_int(raw.get("prompt_cache_hit_tokens") or prompt_details.get("cached_tokens"))
    written = as_int(prompt_details.get("created_cache_tokens"))
    details = raw.get("completion_tokens_details") or {}
    return TokenUsage(
        input_tokens=max(0, prompt - cached - written),
        output_tokens=as_int(raw.get("completion_tokens")),
        cache_read_tokens=cached or None,
        cache_write_tokens=written or None,
        reasoning_tokens=as_int(details.get("reasoning_tokens")) or None,
    )


_AUDIO_FORMATS = {"audio/wav": "wav", "audio/mpeg": "mp3"}
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_FILE_MIMES = frozenset({"application/pdf"})
"""The MIMEs that this wire carries as a `file` part: by `file_id`, or inline as
`file_data`.

This is the third shape beside `image_url` and `input_audio`, and the only one
with two spellings. That is the reason why `uploads` is intersected with it. A
`file` part is the only place where a `file_id` can go on this wire, so a MIME
outside this set has no reference form, however it was configured."""


def _media_part(
    attachment: AttachmentRef, data: str | None, handle: str | None
) -> dict[str, Any] | None:
    """One attachment in the shape of this wire, or `None` if it has none here.

    The function is total over its own vocabulary, and is not a second copy of
    the accept policy. `media-degrade` decides what may be sent, but this
    renderer must still be exact about what it can *express*. A MIME that it has
    no shape for becomes a pointer and is not presented as an image.

    `handle` wins over `data` where both are offered, which is the purpose of
    uploading. In practice they never both arrive, because `load_media` skips
    the bytes of anything that `load_handles` claimed. `data` is therefore
    `None` for a referenced file, and the signature says so, so that the caller
    does not branch.
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
        # This is the inline spelling. It makes the fallback contract of
        # `load_handles` true here: an upload that fails for any reason leaves
        # the id out, and the attachment goes inline. Without it, a route whose
        # file API was down would degrade a document that it is able to send.
        return {
            "type": "file",
            "file": {
                "filename": attachment.name or "attachment.pdf",
                "file_data": f"data:{attachment.mime};base64,{data}",
            },
        }
    return None


def _to_openai(
    message: Message,
    media: dict[str, str],
    handles: dict[str, str],
) -> list[dict[str, Any]]:
    """One pH message as the messages of the wire, which are sometimes several.

    A user message becomes a *content list* as soon as it carries media, and a
    plain string otherwise. This wire accepts both. The string form is kept for
    the far more common case, so that every existing request stays byte-for-byte
    what it was. The prefix cache depends on that (A12).
    """
    if message.role == "assistant":
        text = text_of(message.content, separator="")
        calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": block.arguments},
            }
            for block in message.content
            if isinstance(block, ToolCallBlock)
        ]
        if not text and not calls:
            # **Nothing that this wire can carry** (G9). `text_of` skips
            # reasoning, because reasoning is the scratch of the model and not
            # conversation. An assistant turn that produced only a
            # `ReasoningBlock` therefore renders as `{"content": null}` with no
            # `tool_calls`. Servers on this wire reject that outright, so the
            # whole request fails on a message that said nothing. A max-tokens
            # step that stopped inside its thinking is the ordinary way to
            # produce one. The message stays in history for the rest of the
            # session, so every later request fails too.
            #
            # It is dropped and not sent as `""`. An empty assistant turn is a
            # message that the model did not write, and nothing references it. A
            # reasoning-only message has no calls for a later result to pair
            # with, so its removal cannot unbalance the history.
            return []
        entry: dict[str, Any] = {"role": "assistant", "content": text or None}
        if calls:
            entry["tool_calls"] = calls
        return [entry]
    results = [block for block in message.content if isinstance(block, ToolResultBlock)]
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
        match block:
            case TextBlock():
                parts.append({"type": "text", "text": block.text})
            case MediaBlock():
                carries_media = True
                attachment = block.attachment
                part = _media_part(
                    attachment,
                    media.get(attachment.attachment_id),
                    handles.get(attachment.attachment_id),
                )
                parts.append(part if part is not None else media_pointer(attachment))
            case ToolCallBlock() | ToolResultBlock() | ReasoningBlock():
                # Cannot reach here: an assistant message returned above with its
                # calls, a tool result returned above as its own wire role, and a
                # reasoning block is never sent back.
                pass
            case _ as unhandled:
                assert_never(unhandled)
    if carries_media:
        # The test is that the message *has* media, not that the rendered parts
        # still look like media. A degraded attachment renders as a text part.
        # The question "are any parts non-text" therefore sent the list form
        # only when the attachment succeeded. For the case that the pointer
        # exists to cover, it fell back to `text_of`, which reads `TextBlock`s
        # alone. That is the silent drop this row is here to end, reintroduced
        # one branch lower down.
        return [{"role": "user", "content": parts}]
    return [{"role": "user", "content": text_of(message.content)}]


PROBE_TIMEOUT = 2.0
"""How long a mount waits for a server to describe itself, in seconds.

The value is short. Nothing is blocked on the *answer*, because a route that does
not reply keeps its configured window. Everything is blocked on the *asking*:
this runs during `apply`, so the budget is a user who waits for a TUI to open.
Two seconds is generous for a server on localhost, which is the only kind that
answers this at all."""


async def discover_window(adapter: OpenAiCompatibleAdapter) -> tuple[int, str] | None:
    """The context window of one slot and the path that reported it, or `None`.

    The answer is `None` for every way that this can fail: no probe declared, the
    endpoint not published, the server down, a field of the wrong shape. The
    caller does one thing with all of them. A route that cannot be asked is the
    ordinary case here. This adapter also serves OpenAI, DeepSeek and every
    gateway in between, and most of them publish no window anywhere.

    **The path that answered is returned with the answer.** `phern doctor` prints
    which endpoint the number came from. With a list of probes, the profile can
    no longer answer that afterwards: to derive it again would name the first
    probe, whichever probe answered.

    **The probes are asked at the same time, and the winner is chosen by what
    each measures.** A `request` measurement beats a `model` measurement wherever
    the two are in the list (see `WindowProbe`), and position decides only among
    equals. So the winner is picked from the collected answers and not by
    arrival. A race that the first answer wins would let a model ceiling win by
    being quicker. Given that, there is no reason to wait between the probes. In
    sequence, the cost was `PROBE_TIMEOUT` for each probe, and a mount is a user
    who waits for a TUI to open. Two probes against a server that accepts a
    connection and then hangs cost four seconds of that wait. The price is one
    extra localhost GET against a server that would have answered the first
    probe anyway.
    """
    probes = adapter.profile.context_window_probes
    if not probes:
        return None
    root = adapter.profile.base_url.rstrip("/").removesuffix("/v1").rstrip("/")
    headers = _probe_headers(adapter)
    found: list[int | None] = [None] * len(probes)

    async def ask(index: int, probe: WindowProbe) -> None:
        found[index] = await _ask(adapter, probe, root, headers)

    async with anyio.create_task_group() as tasks:
        for index, probe in enumerate(probes):
            tasks.start_soon(ask, index, probe)
    answers = [(window, probe) for window, probe in zip(found, probes, strict=True) if window]
    best = min(answers, key=lambda answer: answer[1].measures != "request", default=None)
    return (best[0], best[1].path) if best else None


def _probe_headers(adapter: OpenAiCompatibleAdapter) -> dict[str, str]:
    """Headers with the credential when there is one, and plain headers when there is none.

    `/props` needs no key on a llama.cpp server that was started without
    `--api-key`. `phern doctor` is the command that someone runs *before*
    exporting anything. A probe that failed there would report the fallback to
    the user who is least able to tell it from the true value. A server that
    does want a key answers 401, and the fallback stands.

    The headers are resolved one time for each route and not for each probe.
    They are a property of the adapter, and every probe of one route sends the
    same header.
    """
    try:
        return adapter._headers()
    except Exception:
        # Any reason at all, and deliberately not only `LlmError`. The purpose
        # of this branch is to continue without a header. A narrower clause
        # would turn some *other* failure to build a header into a probe that
        # does not happen, which reads as "the server did not say".
        return {"Content-Type": "application/json"}


async def _ask(
    adapter: OpenAiCompatibleAdapter, probe: WindowProbe, root: str, headers: dict[str, str]
) -> int | None:
    """One probe, answered or not."""
    try:
        answer: object = await adapter.http.get_json(
            f"{root}/{probe.path.lstrip('/')}",
            headers=headers,
            is_overflow=_is_overflow,
            timeout=PROBE_TIMEOUT,
        )
    except Exception:
        log.debug("ph_app.adapters: %s did not answer %s", root, probe.path, exc_info=True)
        return None
    for step in probe.field:
        answer = _step(answer, step)
    window = as_int(answer)
    return window if window > 0 else None


def _step(value: object, step: str | int) -> object:
    """One step of a probe's field path: a key into an object, an index into a list.

    It goes through the `as_*` family and not a hand-written `isinstance` walk.
    That family is the standard narrowing of this codebase, and it is the reason
    why this stays checked. `as_obj` and `as_seq` both answer empty for the wrong
    shape. A server that replies with something unexpected therefore yields
    `None` here, and not an `AttributeError` two lines later. `object` also
    stays `object` and does not become `Any`.
    """
    if isinstance(step, int):
        items = as_seq(value)
        return items[step] if -len(items) <= step < len(items) else None
    return as_obj(value).get(step)


@dataclass(slots=True)
class _Windows:
    """What `phern doctor` says about the window of each route, read when it is asked.

    This is a small object and not a closure over `apply`. A `Diagnostic`
    outlives the mount that registered it, so a lambda here would hold `config`,
    every `ProviderProfile` and the locals of the loop for the life of the row.
    This object holds the adapters that it reads and nothing else.
    """

    adapters: list[OpenAiCompatibleAdapter] = field(default_factory=list)
    asked: dict[str, str] = field(default_factory=dict)
    """Provider → the path that answered, for the routes whose server reported a
    window.

    It holds the path and not a bare "yes". Otherwise `describe` would derive
    the path again from the profile. That would be a projection deriving again
    what the probe had already established, and it needed a type suppression to
    do it."""

    async def discover(self, adapter: OpenAiCompatibleAdapter) -> None:
        """Ask one route for its window and adopt the answer, or keep the config.

        It is a method and not a closure in `apply`, because it is what the task
        group spawns. A bound method carries this object and the adapter that it
        was given. A lambda would carry the whole scope of the mount.
        """
        found = await discover_window(adapter)
        if found is None:
            return
        window, path = found
        adapter.profile = adapter.profile.model_copy(update={"context_window": window})
        self.asked[adapter.profile.provider] = path

    def describe(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for adapter in self.adapters:
            route = adapter.profile
            window = route.context_window
            answered = self.asked.get(route.provider)
            if window is None:
                source = "not declared; nothing budgets or compacts against a window"
            elif answered is not None:
                source = f"from {answered}"
            elif route.context_window_probes:
                source = "configured; the server did not say"
            else:
                source = "configured"
            rows.append((route.provider, f"{window if window is not None else '—'} ({source})"))
        return rows


@plugin("llm-openai-compatible", affects="environment", config=Config, inject=[LLM, CREDENTIALS])
async def apply(ctx: Context, config: Config) -> None:
    """Register every configured OpenAI-compatible route."""
    uploads = ctx.get(UPLOADS)
    routes = _Windows(
        adapters=[OpenAiCompatibleAdapter(ctx=ctx, profile=profile) for profile in config.profiles]
    )
    # **At mount and one time, so that `resolve_model` stays synchronous.**
    # Every layer above reads a window for each request, and a probe there would
    # be an HTTP round trip inside a projection. A failure keeps the configured
    # number: see `context_window_probes`.
    #
    # **Together and not in turn**, because the routes are independent and a
    # mount is a user who waits for a TUI to open. Asking three routes in
    # sequence spends three timeouts on a bad network, and asking them at the
    # same time spends one. `discover_window` answers `None` for every failure,
    # so nothing here can raise into the group.
    async with anyio.create_task_group() as tasks:
        for adapter in routes.adapters:
            if adapter.profile.context_window_probes:
                tasks.start_soon(routes.discover, adapter)
    llm = ctx.require(LLM)
    for adapter in routes.adapters:
        profile = adapter.profile
        handle = llm.register_adapter([profile.provider], adapter)
        ctx.add_disposer(handle.dispose, label=f"llm({profile.provider})")
        if uploads is not None and profile.uploads:
            # Only for a route that references files. Most servers that speak
            # this shape implement `/chat/completions` and not `/files`. An
            # uploader registered for every profile would put a file API behind
            # a provider name that has none, and the first attachment would
            # spend a failed round trip to discover it.
            ctx.add_disposer(
                uploads.register_uploader(profile.provider, adapter),
                label=f"uploader({profile.provider})",
            )
        ctx.add_disposer(adapter.http.aclose, label=f"http({profile.provider})")
    if routes.adapters:
        # Every route that this row serves, not only the routes that probed. If
        # `phern doctor` answered for one of five, the user would read the report
        # as "the others have none". The value is read live from the adapter and
        # is not a snapshot taken at mount, like every other `Diagnostic` here.
        # The number that `resolve_model` answers with is the number that this
        # has to print, and a copy taken at mount is a second number that is
        # free to drift.
        contribute(
            ctx,
            Diagnostic(
                id="llm-context-window", title="Context window", read=routes.describe, order=6
            ),
        )
