"""The provider-neutral message and streaming vocabulary.

Ported from dsh `packages/llm/llm/src/{types,message}.ts`. Adapters alone
translate provider wire formats; everything above this line — the loop, the
session log, every plugin, the TUI — speaks only these types.

The one field worth calling out is `Message.source`. It is what lets a consumer
tell *typed human input* from *injected context* without a second channel: a
file-change notice, a skill body and a subagent's reply are all user-role
messages, and only `source` distinguishes them.

@module ph.llm.types
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Final, Literal, TypeAlias, TypeGuard, cast

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from ..json import JsonObject, thaw_json
from ..wire import WireDataclass, WireModel, validation_errors, wire_alias

__all__ = [
    "DEFAULT_EFFORT",
    "AssistantMessage",
    "AttachmentRef",
    "BlockEnd",
    "BlockStart",
    "CallSetting",
    "ContentBlock",
    "ContextForm",
    "ContextSnapshotSection",
    "EffortFields",
    "Finish",
    "FinishKind",
    "FinishReason",
    "GenerateOptions",
    "LlmCallConfig",
    "LlmFailure",
    "MediaBlock",
    "Message",
    "MessageContent",
    "MessageSource",
    "ModelSource",
    "PluginSource",
    "ReasoningBlock",
    "ReasoningDelta",
    "Sampling",
    "SamplingSetting",
    "StreamChunk",
    "TextBlock",
    "TextDelta",
    "TokenUsage",
    "ToolCallBlock",
    "ToolCallDelta",
    "ToolResultBlock",
    "ToolResultMessage",
    "ToolSchema",
    "ToolSource",
    "UsageChunk",
    "UserMessage",
    "UserSource",
    "attachment_of",
    "chunk_from_wire",
    "content_from_wire",
    "create_assistant_message",
    "create_message",
    "create_tool_result_message",
    "create_user_message",
    "is_error_finish",
    "is_token_delta",
    "new_message_id",
    "text_of",
    "unmapped_effort",
    "wire_effort",
]


def new_message_id() -> str:
    """A stable message identity preserved across every representation boundary."""
    return str(uuid.uuid4())


# ------------------------------------------------------------ content blocks --


class TextBlock(WireModel):
    """Plain text visible to the end user."""

    type: Literal["text"] = "text"
    text: str


class ReasoningBlock(WireModel):
    """Reasoning content, distinct from visible text."""

    type: Literal["reasoning"] = "reasoning"
    text: str
    provider_state: dict[str, Any] | None = None
    """Whatever the producing adapter must send back with this block (G8).

    `dict[str, Any]` rather than `JsonObject`, which `ph.json` rules out as a
    pydantic annotation: the alias expands eagerly and never finishes building
    its validator. This is the boundary that module names as the one place the
    `Any` stays.

    **Opaque, and read only by the adapter that wrote it** — the same contract
    `ModelSource.replay_state` states, one level down, because this is per-*block*
    state and that one is per-message. A single assistant message can hold several
    thinking blocks, so a map beside them would need an index, and that index
    desynchronizes the moment the list is filtered: `blocks()` drops tool calls
    under max-tokens and `interrupted_blocks()` keeps a prefix.

    It has to live on the block rather than in the adapter because the log is the
    state: a resumed session rebuilds its history from these events, and anything
    the adapter held in memory is gone by then. Anthropic signs every thinking
    block and **rejects a conversation that sends one back without its
    signature**, which makes this the difference between a session with extended
    thinking that continues and one whose every later request fails.

    A dict rather than a named `signature` field so the next provider costs
    nothing here: Gemini's `thought_signature` and the OpenAI Responses API's
    `encrypted_content` are the same shape of secret, and naming each one on this
    neutral type would make every other adapter learn to ignore it.

    `None` for a provider that needs nothing back, and for reasoning logged
    before this existed — the adapter decides what to do without it, which for
    Anthropic is to leave the block out rather than have the request refused.
    """
    redacted: bool = False
    """Whether `text` is an opaque blob the provider encrypted rather than prose.

    Anthropic sends `redacted_thinking` when its own safety systems withhold the
    reasoning; the payload is ciphertext that only it can read, and it must still
    be passed back for the conversation to continue.

    **Not adapter-private, which is why it is a named field and not part of
    `provider_state`:** it is the one thing a *neutral* reader has to know.
    Readers show `ph.text.redacted_marker`, which says why.
    """


class AttachmentRef(WireModel):
    """Where an attachment's bytes are, and what a reader needs without opening them.

    The log is lossless JSON (A1), so bytes never enter it — a message carries
    this reference and `ctx.attachments` resolves it. `attachment_id` is
    `sha256:<hex>` of the content, which is what makes two sessions attaching one
    photo share one file and what lets a fork reference its parent's media
    without owning a directory.

    The measurement fields stay *optional* — a video's duration and a PDF's page
    count are still facts an ingester happened to know — but `width` and `height`
    are no longer among them: `ph.llm.dimensions` reads them out of the file
    header with no dependency at all, and `AttachmentStore.save_bytes` fills them
    in. This docstring used to say that measuring an image "means decoding it,
    and the decoder is exactly the dependency `media-transform` (P7-02) exists to
    keep optional", which conflated resizing with reading four integers at a
    fixed offset (P7-03). Present, they make the token estimate accurate and let
    a route say the picture is larger than it can use; absent, the estimate falls
    back to a flat figure.
    """

    attachment_id: str
    """`sha256:<hex>` — the content digest, and the store's whole index."""
    mime: str
    bytes: int
    name: str | None = None
    """The filename a person would recognize. Never used to locate the blob."""
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    pages: int | None = None


class MediaBlock(WireModel):
    """Durable, MIME-typed media the model may be shown.

    One block keyed on MIME rather than a family of per-medium blocks. Providers
    *do* distinguish them on the wire — Anthropic `image` against `document`,
    OpenAI `image_url` against `input_audio` against `file` — but that branching
    belongs in the adapter, where the differences actually live, rather than in a
    union that grows a member every time a provider learns a format. It is also
    the shape pH's other media path already has: the kernel's `display` frame is
    `{mime, data}`, and the two are meant to become one.

    Deliberately *not* a general file block. What a provider ingests as content —
    images, PDFs at some routes, audio at a few — belongs here; a CSV or an
    archive belongs in the workspace with a path the model can `read`, which the
    spill store, the `read` tool and `rlm-context-loader` already answer three
    ways. A block that accepted anything would claim a capability no provider has.

    Role-neutral by design: user content carries one today, and a route that
    generates images will carry one back.
    """

    type: Literal["media"] = "media"
    attachment: AttachmentRef


class ToolCallBlock(WireModel):
    """A tool invocation requested by the model."""

    type: Literal["tool-call"] = "tool-call"
    id: str
    """Provider-issued call id; correlates with the matching tool result."""
    name: str
    arguments: str
    """Raw JSON string exactly as the model produced it — never re-serialized."""


class ToolResultBlock(WireModel):
    """The result of a tool invocation, sent back to the model."""

    type: Literal["tool-result"] = "tool-result"
    tool_call_id: str
    content: list[ContentBlock]
    is_error: bool | None = None


ContentBlock: TypeAlias = Annotated[
    "TextBlock | ReasoningBlock | MediaBlock | ToolCallBlock | ToolResultBlock",
    Field(discriminator="type"),
]

ToolResultBlock.model_rebuild()

MessageContent: TypeAlias = Sequence[ContentBlock | JsonObject]
"""What a message is built from: declared blocks, their wire form, or a mix.

Both, because the constructors below validate through `Message.model_validate`,
which coerces a raw block dict and passes an instance through — and callers use
each: an adapter holds `ContentBlock`s it just assembled, while a row writing one
message inline spells the dict.

The wire half is the arm to watch. Every site in this tree that uses it passes a
single text block, which `user_text` and `TextBlock` already name — so the union
describes what callers happen to do rather than a contract worth keeping, and it
narrows to `Sequence[ContentBlock]` the day they stop.
"""

_CONTENT_BLOCKS: TypeAdapter[list[ContentBlock]] = TypeAdapter(list[ContentBlock])


def content_from_wire(blocks: object) -> list[ContentBlock]:
    """Validate a list of content blocks read back from the log."""
    return _CONTENT_BLOCKS.validate_python(blocks)


def text_of(
    blocks: Sequence[ContentBlock],
    *,
    placeholder: Callable[[str], str] | None = None,
    separator: str = "\n",
) -> str:
    """The text of a block list, joined by `separator` — newlines by default.

    Non-text blocks are skipped, or rendered through `placeholder(type)` when a
    caller wants a marker — the one join every mode and adapter needs.

    `separator` is for the one wire that cannot take the default: an assistant
    message on the OpenAI wire concatenates its text fragments with no join, and
    changing those bytes would invalidate the cached prefix (A12). A parameter
    rather than a second hand-rolled comprehension at that call site, so the
    "which blocks count as text" rule stays in one place.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif placeholder is not None:
            parts.append(placeholder(block.type))
    return separator.join(parts)


def attachment_of(block: ContentBlock) -> AttachmentRef | None:
    """The attachment a block carries, or `None` — for readers that want only that.

    Beside `text_of` for the same reason that exists: a walk that asks one
    question about every block should not spell the answer by hand. Its callers
    are the loops that collect or measure attachments and ignore the rest.

    **Not for a dispatch that must stay exhaustive.** Returning `Optional` erases
    the union, so a renderer using it narrows nothing and a variant it forgot
    falls through silently — which is what hid the missing `media` arms. Those
    match `MediaBlock` directly and close on `assert_never`;
    `test_exhaustive_dispatch` enforces it.
    """
    return block.attachment if isinstance(block, MediaBlock) else None


# ------------------------------------------------------------------ sources --

ContextForm: TypeAlias = Literal[
    "instructions", "catalog", "snapshot", "notice", "relay", "recall", "compaction"
]
"""What kind of thing a piece of injected context is.

Deliberately semantic, never visual: a value says the content is a file's
instructions or a catalog, and the consumer decides what that looks like.
Colors, icons and collapse defaults are the consumer's business and must not
enter this union.

`compaction` is the one form that is also a *claim about the surface*: this text
stands in for conversation that has left the derivation (A3). It is what lets a
reader tell a compaction summary from the other plugin-authored replacement — an
offloaded paste — without guessing from the shape of the replace op.
"""

CONTEXT_SUMMARY_MAX_CHARS = 120


class ContextSnapshotSection(WireModel):
    """One named contribution to a `snapshot`-form context, in assembly order."""

    name: str
    text: str


class UserSource(WireModel):
    """Text a human typed."""

    kind: Literal["user"] = "user"


class PluginSource(WireModel):
    """Content a plugin injected, and what kind of content it is."""

    kind: Literal["plugin"] = "plugin"
    plugin: str
    form: ContextForm | None = None
    summary: Annotated[str, Field(max_length=CONTEXT_SUMMARY_MAX_CHARS)] | None = None
    """Required for `notice`: the one-line account shown without expanding."""
    sections: list[ContextSnapshotSection] | None = None
    """Required for `snapshot`: the named contributions, in order."""

    @model_validator(mode="after")
    def _form_carries_its_fields(self) -> PluginSource:
        # Discriminated by `form` so a producer cannot select a form without the
        # fields needed to present it.
        if self.form == "notice" and self.summary is None:
            raise ValueError('a "notice" context source must carry a summary')
        if self.form == "snapshot" and self.sections is None:
            raise ValueError('a "snapshot" context source must carry sections')
        return self


class ModelSource(WireModel):
    """Provider identity and adapter-private replay data."""

    kind: Literal["model"] = "model"
    provider: str
    model: str
    replay_state: dict[str, Any] | None = None


class ToolSource(WireModel):
    """The call this result answers."""

    kind: Literal["tool"] = "tool"
    call_id: str


MessageSource: TypeAlias = Annotated[
    "UserSource | PluginSource | ModelSource | ToolSource", Field(discriminator="kind")
]


# ----------------------------------------------------------------- messages --


class Message(WireModel):
    """One immutable message shared by delivery, durable history and requests."""

    id: str
    role: Literal["system", "user", "assistant"]
    content: list[ContentBlock]
    source: MessageSource


UserMessage: TypeAlias = Message
AssistantMessage: TypeAlias = Message
ToolResultMessage: TypeAlias = Message


def create_message(
    *,
    role: Literal["system", "user", "assistant"],
    content: MessageContent,
    source: object,
    message_id: str | None = None,
) -> Message:
    """Create one identified message: `message_id` when the caller derives one, else a
    fresh id."""
    return Message.model_validate(
        {"id": message_id or new_message_id(), "role": role, "content": content, "source": source}
    )


def create_user_message(
    *, content: MessageContent, source: object, message_id: str | None = None
) -> Message:
    return create_message(role="user", content=content, source=source, message_id=message_id)


def user_text(text: str) -> Message:
    """One plain-text message from the person — the commonest user message there is.

    The three-key incantation it replaces was written out verbatim at four sites
    across three packages: a prompt from the TUI, a prompt over the socket, an
    approval's redirecting `reason`. Each is the same claim — *this text came from
    a human, at a step boundary* — and `source={"kind": "user"}` is the part that
    makes it true, so it is the part worth having one copy of.
    """
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})


def create_assistant_message(
    *,
    content: MessageContent,
    provider: str,
    model: str,
    replay_state: object = None,
) -> Message:
    source: dict[str, Any] = {"kind": "model", "provider": provider, "model": model}
    if replay_state is not None:
        source["replayState"] = replay_state
    return create_message(role="assistant", content=content, source=source)


def create_tool_result_message(*, call_id: str, content: MessageContent, is_error: bool) -> Message:
    return create_user_message(
        content=[
            ToolResultBlock(
                tool_call_id=call_id, content=content_from_wire(content), is_error=is_error
            )
        ],
        source={"kind": "tool", "callId": call_id},
    )


# ---------------------------------------------------------------- accounting --


class TokenUsage(WireModel):
    """Token accounting for one model call.

    Counts are DISJOINT: `input_tokens` is uncached input only, and cached input
    is reported separately. An adapter whose provider folds cache hits into one
    prompt total subtracts them out — otherwise every cache hit is billed twice
    in pH's own accounting.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def total(self) -> int:
        """Everything the provider billed for this call.

        The four terms, in one place, which `TokenMeter.baseline` and the TUI's
        footer both read. Each once spelled them out — the footer's docstring saying
        "same four terms as `TokenMeter.baseline`", which is a comment doing a
        type's job — and P5-07's budget arrived as a third definition with only *two*,
        so a cache-heavy run spent most of its input outside the budget while
        the footer showed a larger number for the same word.

        `reasoning_tokens` is deliberately out: providers that report it count
        it inside `output_tokens`, so adding it bills the same tokens twice —
        which is the mistake this model's own docstring warns about for cached
        input.
        """
        return (
            self.input_tokens
            + self.output_tokens
            + (self.cache_read_tokens or 0)
            + (self.cache_write_tokens or 0)
        )


class LlmFailure(WireModel):
    """Serializable provider or transport failure facts.

    Facts only: policy decides whether they are retryable (Phase 1's retry
    plugin), and compaction keys off `CONTEXT_WINDOW_EXCEEDED`.
    """

    message: str
    code: str
    status: int | None = None
    provider_retry_after_ms: int | None = None
    request_id: str | None = None


CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
EMPTY_RESPONSE = "EMPTY_RESPONSE"
FILE_EXPIRED = "FILE_EXPIRED"
"""A request referenced an uploaded file the provider no longer has (P7-03).

Its own code because the remedy is specific and automatic: the handle has already
been forgotten by the time this is raised, so the next attempt re-uploads and
succeeds. `CONTEXT_WINDOW_EXCEEDED` is the opposite case and the reason this
codebase classifies rather than blanket-retries — that one will not fit on the
second attempt either."""


SamplingSetting: TypeAlias = Literal[
    "temperature", "topP", "topK", "minP", "presencePenalty", "frequencyPenalty"
]
"""A sampling setting, spelled as a route sets it (`ModelRoute`) and as a warning or
`phern doctor` names it: what model makers' recommended settings consist of. Each API
takes some of them (`ResolvedModel.sampling`), and one a route's API cannot carry is
dropped with a warning rather than refused."""


class Sampling(WireModel):
    """The sampling settings, as one value every layer carries whole: a route sets it
    (`ModelRoute.sampling`), an agent's options and the logged call config carry it, and
    a request sends it. A setting added here reaches all of them, where six fields on
    each layer had to be added on each and copied between them by hand.

    Each field's alias is its `SamplingSetting`, so a route's YAML, a warning and
    `phern doctor` spell a setting one way (`topP`). Unset is `None`, and absent from
    what an adapter sends.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def is_empty(self) -> bool:
        """Whether this sets nothing: what a route or a header that sets none carries,
        and leaves off its wire form, which is then as it was before there was one."""
        return not self.settings()

    def settings(self) -> dict[SamplingSetting, float | int]:
        """The settings this sets, by name. Unset is absent, so an adapter sends nothing
        for it."""
        return cast("dict[SamplingSetting, float | int]", self.to_wire())

    def uncarried(self, carried: frozenset[SamplingSetting]) -> dict[SamplingSetting, float | int]:
        """The settings this sets that a route carrying `carried` cannot send, by the value
        asked: what the `llm` seam drops and `phern doctor` names, by one rule."""
        return {name: value for name, value in self.settings().items() if name not in carried}

    def without(self, settings: Collection[str]) -> Sampling:
        """This, with `settings` unset: the one drop rule, for the call config the loop
        logs and for a request that reaches the seam another way."""
        kept = {name: value for name, value in self.settings().items() if name not in settings}
        return Sampling.model_validate(kept)

    def on_wire(self, wire: Mapping[SamplingSetting, str]) -> dict[str, float | int]:
        """The settings this sets that a wire names, under its names: an adapter's
        request fields, from its own map of what its API takes."""
        return {wire[name]: value for name, value in self.settings().items() if name in wire}


DEFAULT_EFFORT: Final = "default"
"""The reasoning effort that asks for the provider's own: nothing is sent, so the
provider's default applies. Not a setting of pH's, so no profile defines it
(`ModelRoute.efforts`) and nothing warns about it; a route, a spawn or a skill names it
to run at the provider's default over an effort its route sets."""

EffortFields: TypeAlias = JsonObject
"""The request body fields one reasoning effort sends on a route's API, as its profile
wrote them (`ModelRoute.efforts`): opaque to all but the adapter that merges them
(`wire_effort`)."""

CallSetting: TypeAlias = SamplingSetting | Literal["reasoningEffort"]
"""A call setting a route may be unable to carry, spelled as a route sets it: a
sampling setting its API does not take, or a reasoning effort its profile does not say
how to send (`ModelRoute.efforts`). What `EpochHeader.withheld` is keyed by: a sampling
setting's alias on `Sampling`, and the effort's on `LlmCallConfig`."""


def unmapped_effort(effort: str | None, efforts: Collection[str]) -> TypeGuard[str]:
    """Whether `effort` is asked for and a route whose profile defines `efforts` cannot
    send it: what the `llm` seam withholds and `phern doctor` names, by one rule. Unset
    and `DEFAULT_EFFORT` ask for nothing to be sent, so neither is."""
    return effort is not None and effort != DEFAULT_EFFORT and effort not in efforts


class LlmCallConfig(WireModel):
    """The conversation's call configuration: route plus its sampling.

    Separated from `GenerateOptions` because this is the part that is *durable* —
    it is what `request/header` snapshots and what header equality compares, so
    a request that differs only in its message list appends no new header.
    """

    provider: str
    model: str
    reasoning_effort: str | None = None
    sampling: Sampling = Field(default_factory=Sampling, exclude_if=Sampling.is_empty)
    max_tokens: int | None = None
    stop: list[str] | None = None

    def without(self, settings: Collection[CallSetting]) -> LlmCallConfig:
        """This, with `settings` unset: what a route that cannot carry them is sent.

        `asking` is its inverse, and the two are the one place a `CallSetting` is told
        apart: a sampling setting is `sampling`'s, and `reasoningEffort` is this model's
        own."""
        effort = None if "reasoningEffort" in settings else self.reasoning_effort
        return self.model_copy(
            update={"sampling": self.sampling.without(settings), "reasoning_effort": effort}
        )

    def asking(self, withheld: Mapping[CallSetting, float | int | str]) -> LlmCallConfig:
        """This, with what `without` took off asked for again (`EpochHeader.withheld`):
        validated, since what comes back from a log is only a value."""
        restored = {name: value for name, value in withheld.items() if name != "reasoningEffort"}
        data = {**self.to_wire(), "sampling": {**self.sampling.settings(), **restored}}
        if "reasoningEffort" in withheld:
            data["reasoningEffort"] = withheld["reasoningEffort"]
        return LlmCallConfig.model_validate(data)


class ToolSchema(WireModel):
    """A tool as the model sees it."""

    name: str
    description: str
    parameters: dict[str, Any]


# ------------------------------------------------------------ stream chunks --
#
# Chunks are per-token hot-path values (D4), so they are frozen dataclasses on
# `WireDataclass` rather than pydantic models. Their wire form still follows the
# one alias rule.


FinishKind: TypeAlias = Literal["stop", "tool-calls", "max-tokens", "aborted", "error"]
"""Why a response stopped, as the five answers every consumer branches on.

A name rather than an inline literal because an adapter has to *return* one, and
the third one to be written spelled the union out again to say so — which is the
same list in a second place, free to grow a sixth member nothing else handles.

`aborted` is the one no adapter in this repo produces: a canceled run reaches
the loop as an exception, never as a chunk, so `TurnEndReason` carries that
answer instead. It stays here for an adapter whose provider can report a
cancellation on the wire, and `driver.py` routes it alongside `error`."""


@dataclass(frozen=True, slots=True)
class FinishReason(WireDataclass):
    """Why a model response stopped."""

    kind: FinishKind
    failure: LlmFailure | None = None


@dataclass(frozen=True, slots=True)
class BlockStart(WireDataclass):
    index: int
    block_type: str
    type: Literal["block-start"] = "block-start"


@dataclass(frozen=True, slots=True)
class TextDelta(WireDataclass):
    index: int
    text: str
    type: Literal["text-delta"] = "text-delta"


@dataclass(frozen=True, slots=True)
class ReasoningDelta(WireDataclass):
    index: int
    text: str
    type: Literal["reasoning-delta"] = "reasoning-delta"


@dataclass(frozen=True, slots=True)
class ToolCallDelta(WireDataclass):
    index: int
    id: str
    arguments_delta: str
    name: str | None = None
    type: Literal["tool-call-delta"] = "tool-call-delta"


@dataclass(frozen=True, slots=True)
class BlockEnd(WireDataclass):
    index: int
    block: ContentBlock
    type: Literal["block-end"] = "block-end"


@dataclass(frozen=True, slots=True)
class UsageChunk(WireDataclass):
    usage: TokenUsage
    type: Literal["usage"] = "usage"


@dataclass(frozen=True, slots=True)
class Finish(WireDataclass):
    reason: FinishReason
    replay_state: dict[str, Any] | None = None
    type: Literal["finish"] = "finish"


StreamChunk: TypeAlias = (
    "BlockStart | TextDelta | ReasoningDelta | ToolCallDelta | BlockEnd | UsageChunk | Finish"
)
"""The raw streaming protocol adapters emit.

Contract: block indexes correlate interleaved deltas; `block-end` carries the
assembled block; `usage` precedes the terminal `finish` and nothing follows it;
tool arguments stay raw JSON strings until `block-end`. An adapter may raise,
and the runtime normalizes that into a terminal `finish{error}` before any
consumer sees it — so a consumer never has to handle both shapes.
"""


_CHUNKS: TypeAdapter[StreamChunk] = TypeAdapter(
    Annotated[StreamChunk, Field(discriminator="type")],
    config=ConfigDict(alias_generator=wire_alias, populate_by_name=True, extra="ignore"),
)
"""The tagged union, decoded the way `ContentBlock` already is.

**The read side is not hot** — `ph.session.events` says the same of its own
dataclass, and for the same reason: D4 is about *constructing* a chunk per token,
which an adapter does per streamed message and this never touches. There is one
caller, `ph.llm.replay`, reading a log back.

So the eight-branch ladder this replaces was the second answer in one file to a
problem `content_from_wire` had already solved. It also had a hole: the chunks
are frozen dataclasses and validate nothing, so a missing key raised but a
*mis-typed* one built silently — `{"index": "x"}` produced a `BlockStart` whose
index was a string, and handed it to an assembler that indexes and concatenates.

`extra="ignore"` rather than `WireModel`'s `forbid`: a log is read by builds older
than the one that wrote it, so a key added later is skipped rather than condemning
the step. That is the ladder's behavior, kept.
"""


def chunk_from_wire(wire: JsonObject) -> StreamChunk:
    """Rebuild one stream chunk from its logged JSON form (replay fidelity).

    A malformed chunk is reported as `ValueError` naming the chunk kind and the
    field path, so a replay reader can say which line was wrong rather than
    surfacing a `KeyError` from three frames down.
    """
    try:
        # `thaw_json` first: the log freezes payloads into `MappingProxyType` and
        # `tuple`, and pydantic refuses those for a *dataclass* target — `WireModel`
        # thaws them in `_thaw_frozen_input`, which a `WireDataclass` has no
        # equivalent of. A whole-tree thaw, not the shallow `dict(...)` this used
        # to carry, which left nested frozen containers in `replay_state`.
        return _CHUNKS.validate_python(thaw_json(wire))
    except ValidationError as error:
        raise ValueError(f"malformed stream chunk: {validation_errors(error)}") from error


def is_token_delta(chunk: StreamChunk) -> bool:
    """Whether a chunk carries visible output — the first-token boundary.

    Empty deltas (heartbeats, empty tool-call frames) do not count.
    """
    if isinstance(chunk, (TextDelta, ReasoningDelta)):
        return chunk.text != ""
    if isinstance(chunk, ToolCallDelta):
        return chunk.arguments_delta != "" or chunk.name is not None
    return False


def is_error_finish(chunk: StreamChunk) -> bool:
    """Whether this chunk is a stream saying it failed — nothing after it counts.

    Both wires report a failed request *inside* a 200 response, and an adapter
    that kept reading past one went on to close its half-written blocks and
    append a second `Finish` reading "stop", which is what the turn was then
    recorded as. Each adapter carried a `failed` flag to notice; the flag was a
    round trip through mutable state for a fact the chunk already states, and a
    second code path emitting an error `Finish` would have had to remember to
    set it. Derived here instead, once, for every adapter including ones this
    repo does not own.
    """
    return isinstance(chunk, Finish) and chunk.reason.kind == "error"


# ------------------------------------------------------------- request shape --


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    """A single model request, fully assembled."""

    provider: str
    model: str
    messages: tuple[Message, ...]
    system: str | None = None
    tools: tuple[ToolSchema, ...] = ()
    effort_fields: EffortFields | None = None
    """The reasoning effort asked for, as this route's API spells it: the request body
    fields its profile gives that effort (`ModelRoute.efforts`), which the adapter merges
    into its body (`wire_effort`). No adapter spells an effort of its own, so `None`
    sends none and the provider's default applies. Request body like any other, so an
    `llm/stream` listener that moves a request to another route owns them, as it owns
    `max_tokens`: the request names no effort to look up on the new one."""
    sampling: Sampling = field(default_factory=Sampling)
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    response_schema: dict[str, Any] | None = None
    """A JSON Schema the reply's *text* must satisfy (P7-17).

    For the calls that want a value rather than prose — `rlm-harness`'s review
    gate and its planner ask for JSON today and then hope, with a tolerant parser
    whose own docstring concedes that *"return only JSON is an instruction and
    not a guarantee"*. A schema turns that into something the wire can enforce.

    **Not a `LlmCallConfig` field**, deliberately: that class is what
    `request/header` snapshots and what header equality compares, so a schema
    there would put one caller's shape into the cached prefix of a conversation
    that has nothing to do with it (A12). This is per request, like `tools`.

    **Not combined with tools.** A tool call is not a JSON document in the
    schema's shape, so a route asked for both would have to break one promise;
    callers that want structure ask for it on a call that offers no tools, and
    `structured` refuses the combination rather than letting a provider decide
    which to honor."""
    session_id: str | None = None
    purpose: Literal["compaction", "session-title", "refine"] | None = None
    """Why an auxiliary call is being made. Ordinary conversation requests leave
    it unset — so a session-bound request with no `purpose` *is* a loop request,
    and the "model-visible means logged" invariant holds it to the log. Nothing
    outside the loop can opt a conversation request out by leaving a flag at its
    default; it would have to name a purpose it does not have.

    A closed set rather than a free string, for that reason: opting out has to be
    a declaration this file can enumerate. `refine` is the Continual Harness's
    planner and its review gate (P3-16) — calls *about* the conversation, whose
    prompt is deliberately not its derivation."""

    @property
    def is_loop_request(self) -> bool:
        return self.session_id is not None and self.purpose is None


def wire_effort(body: dict[str, Any], options: GenerateOptions) -> None:
    """Merge into an adapter's request `body` the fields `options`' reasoning effort
    means on its route (`GenerateOptions.effort_fields`), after the adapter's own: an
    object merges key by key, so a profile can reach into `generationConfig` beside the
    fields the adapter put there, and anything else replaces what was there."""
    if options.effort_fields is not None:
        _merge(body, options.effort_fields)


def _merge(into: dict[str, Any], fields: EffortFields) -> None:
    for key, value in fields.items():
        here = into.get(key)
        if isinstance(value, Mapping) and isinstance(here, dict):
            _merge(here, value)
        else:
            into[key] = thaw_json(value)
