"""The provider-neutral message and streaming vocabulary.

Ported from dsh `packages/llm/llm/src/{types,message}.ts`. Only an adapter
translates a provider wire format. Everything above the adapters speaks only
these types: the loop, the session log, every plugin, and the TUI.

`Message.source` is the field to know. It lets a consumer tell *typed human
input* from *injected context* without a second channel. A file-change notice,
the body of a skill and the reply of a subagent are all user-role messages, and
only `source` distinguishes them.

@module ph.llm.types
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Final, Literal, TypeAlias, TypeGuard, cast

from pydantic import (
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ..json import JsonObject, thaw_json
from ..text import one_line
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
    """What the adapter that produced this block must send back with it (G8).

    The type is `dict[str, Any]` and not `JsonObject`. `ph.json` rules out
    `JsonObject` as a pydantic annotation: pydantic expands the alias eagerly,
    and the model never finishes building its validator. `ph.json` names this
    boundary as the one place where the `Any` stays.

    **The value is opaque, and only the adapter that wrote it reads it.**
    `ModelSource.replay_state` has the same contract for a whole message. This
    field holds state for one *block*. One assistant message can hold several
    thinking blocks. A map beside the blocks would need an index, and the index
    goes out of step as soon as the list is filtered: `blocks()` drops tool
    calls under max-tokens, and `interrupted_blocks()` keeps a prefix.

    The state is on the block and not in the adapter, because the log is the
    state. A resumed session rebuilds its history from these events, and by then
    everything that the adapter held in memory is gone. Anthropic signs every
    thinking block and **rejects a conversation that sends one back without its
    signature**. With this field, a session with extended thinking continues.
    Without it, every later request of that session fails.

    The field is a dict and not a named `signature` field, so that the next
    provider needs no change here. The `thought_signature` of Gemini and the
    `encrypted_content` of the OpenAI Responses API are secrets of the same
    shape. A named field for each one on this neutral type would make every
    other adapter learn to ignore it.

    The value is `None` for a provider that needs nothing back, and for
    reasoning that was logged before this field existed. The adapter decides
    what to do without it. The Anthropic adapter leaves the block out, so that
    the request is not refused.
    """
    redacted: bool = False
    """Whether `text` is an opaque blob that the provider encrypted, and not prose.

    Anthropic sends `redacted_thinking` when its own safety systems withhold the
    reasoning. The payload is ciphertext that only Anthropic can read, and it
    must still be passed back for the conversation to continue.

    **This fact is not private to the adapter. That is why it is a named field
    and not part of `provider_state`.** It is the one thing that a *neutral*
    reader must know. Readers show `ph.text.redacted_marker`, which says why.
    """


class AttachmentRef(WireModel):
    """Where an attachment's bytes are, and what a reader needs without opening them.

    The log is lossless JSON (A1), so bytes never enter it. A message carries
    this reference, and `ctx.attachments` resolves it. `attachment_id` is
    `sha256:<hex>` of the content. So two sessions that attach one photo share
    one file, and a fork can reference the media of its parent without owning a
    directory.

    The measurement fields are *optional*. The duration of a video and the page
    count of a PDF are still facts that an ingester supplies only when it knows
    them. `width` and `height` are no longer such facts: `ph.llm.dimensions`
    reads them from the file header with no dependency at all, and
    `AttachmentStore.save_bytes` fills them in (P7-03). An earlier version of
    this docstring said that measuring an image "means decoding it, and the
    decoder is exactly the dependency `media-transform` (P7-02) exists to keep
    optional". That confused resizing with reading two integers from a
    header. When `width` and `height` are present, the token estimate is
    accurate, and a route can say that the picture is larger than it can use.
    When they are absent, the estimate falls back to a flat figure.
    """

    attachment_id: str
    """`sha256:<hex>`: the content digest. It is the only index of the store."""
    mime: str
    bytes: int
    name: str | None = None
    """The filename a person would recognize. Never used to locate the blob."""
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    pages: int | None = None


class MediaBlock(WireModel):
    """Durable, MIME-typed media that the model may be shown.

    There is one block keyed on MIME, and not a family of blocks with one for
    each medium. Providers *do* distinguish media on the wire: Anthropic has
    `image` and `document`, and OpenAI has `image_url`, `input_audio` and
    `file`. That branching belongs in the adapter, where the differences are. A
    union here would grow a member every time a provider learns a format. The
    other media path of pH already has this shape: the `display` frame of the
    kernel is `{mime, data}`. The two are meant to become one.

    This is deliberately *not* a general file block. It holds what a provider
    ingests as content: images, PDFs at some routes, audio at a few. A CSV or an
    archive belongs in the workspace, with a path that the model can `read`. The
    spill store, the `read` tool and `rlm-context-loader` already handle that
    case in three ways. A block that accepted anything would claim a capability
    that no provider has.

    The block is role-neutral by design. User content carries one today, and a
    route that generates images will carry one back.
    """

    type: Literal["media"] = "media"
    attachment: AttachmentRef


class ToolCallBlock(WireModel):
    """A tool invocation requested by the model."""

    type: Literal["tool-call"] = "tool-call"
    id: str
    """The call id that the provider issued. It correlates with the matching tool result."""
    name: str
    arguments: str
    """The raw JSON string, exactly as the model produced it. It is never re-serialized."""


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

Both forms are allowed because the constructors below validate through
`Message.model_validate`. That call coerces a raw block dict and passes an
instance through, and callers use each form. An adapter holds the
`ContentBlock`s that it just assembled. A row that writes one message inline
writes the dict.

The wire form is the arm to watch. Every site in this tree that uses it passes
only text blocks, which `user_text` and `TextBlock` already name. So the union
describes what callers do today, and not a contract that is worth keeping. It
narrows to `Sequence[ContentBlock]` on the day that they stop.
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
    """The text of a block list, joined by `separator`. The default is a newline.

    Non-text blocks are skipped. A caller that wants a marker passes
    `placeholder`, and each non-text block is then rendered through
    `placeholder(type)`. Every mode and adapter needs this one join.

    `separator` is for the one wire that cannot take the default. An assistant
    message on the OpenAI wire concatenates its text fragments with no join, and
    a change to those bytes would invalidate the cached prefix (A12). It is a
    parameter here, and not a second comprehension written by hand at that call
    site, so that the rule for which blocks count as text stays in one place.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif placeholder is not None:
            parts.append(placeholder(block.type))
    return separator.join(parts)


def attachment_of(block: ContentBlock) -> AttachmentRef | None:
    """The attachment that a block carries, or `None`, for a reader that wants only that.

    It is beside `text_of` for the same reason that `text_of` exists: a walk
    that asks one question about every block should not write the answer by
    hand. Its callers are the loops that collect or measure attachments and
    ignore the rest.

    **Do not use it for a dispatch that must stay exhaustive.** A return type of
    `Optional` erases the union. A renderer that uses it narrows nothing, and a
    variant that the renderer forgot falls through silently. That is what hid
    the missing `media` arms. Those renderers match `MediaBlock` directly and
    close on `assert_never`, and `test_exhaustive_dispatch` enforces it.
    """
    return block.attachment if isinstance(block, MediaBlock) else None


# ------------------------------------------------------------------ sources --

ContextForm: TypeAlias = Literal[
    "instructions", "catalog", "snapshot", "notice", "relay", "recall", "compaction"
]
"""What kind of thing a piece of injected context is.

The value is deliberately semantic and never visual. It says that the content is
the instructions of a file, or that it is a catalog, and the consumer decides
what that looks like. Colors, icons and collapse defaults belong to the consumer
and must not enter this union.

`compaction` is the one form that is also a *claim about the surface*: this text
stands in for conversation that has left the derivation (A3). With it, a reader
can tell a compaction summary from the other replacement that a plugin writes,
which is an offloaded paste, without guessing from the shape of the replace op.
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
    summary: str | None = None
    """Required for `notice`: the one-line account that is shown without expanding.
    The validator `_one_line` puts it on one line and cuts it to
    `CONTEXT_SUMMARY_MAX_CHARS` when it is set."""
    sections: list[ContextSnapshotSection] | None = None
    """Required for `snapshot`: the named contributions, in order."""

    @field_validator("summary")
    @classmethod
    def _one_line(cls, value: str | None) -> str | None:
        """Put a summary on one line and cut it to its bound (`ph.text.one_line`).

        A summary is a row. This validator does the work one time, here, so that
        no producer has to. Four producers did it by hand, in two ways. One
        producer did not, and its notice was refused because a sentence was a
        little too long. The function is idempotent, so a summary that is read
        back from a log is the summary that was written."""
        return None if value is None else one_line(value, CONTEXT_SUMMARY_MAX_CHARS)

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
    """Create one identified message. The id is `message_id` when the caller derives
    one, and a fresh id otherwise."""
    return Message.model_validate(
        {"id": message_id or new_message_id(), "role": role, "content": content, "source": source}
    )


def create_user_message(
    *, content: MessageContent, source: object, message_id: str | None = None
) -> Message:
    return create_message(role="user", content=content, source=source, message_id=message_id)


def user_text(text: str) -> Message:
    """One plain-text message from the user: the most common user message.

    It replaces three keys that were written out verbatim at four sites across
    three packages: a prompt from the TUI, a prompt over the socket, the
    redirecting `reason` of an approval. Each site makes the same claim: *this
    text came from a human, at a step boundary*. `source={"kind": "user"}` is
    the part that makes the claim true, so it is the part that should have one
    copy.
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

    Counts are DISJOINT. `input_tokens` is uncached input only, and cached input
    is reported separately. An adapter whose provider folds cache hits into one
    prompt total subtracts them out. Otherwise the accounting of pH would bill
    every cache hit twice.
    """

    input_tokens: int
    output_tokens: int
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def total(self) -> int:
        """Everything the provider billed for this call.

        The four terms are in one place, and `TokenMeter.baseline` and the footer
        of the TUI both read it. Each of them used to spell the terms out. The
        docstring of the footer said "same four terms as `TokenMeter.baseline`",
        which is a comment that does the job of a type. The budget of P5-07 then
        arrived as a third definition with only *two* terms. So a cache-heavy
        run spent most of its input outside the budget, while the footer showed
        a larger number for the same word.

        `reasoning_tokens` is deliberately left out. Providers that report it
        count it inside `output_tokens`, so adding it would bill the same tokens
        twice. The docstring of this model warns about the same mistake for
        cached input.
        """
        return (
            self.input_tokens
            + self.output_tokens
            + (self.cache_read_tokens or 0)
            + (self.cache_write_tokens or 0)
        )


class LlmFailure(WireModel):
    """Serializable facts about a provider or transport failure.

    Facts only. Policy decides whether a failure is retryable (the `llm-retry`
    row), and compaction keys off `CONTEXT_WINDOW_EXCEEDED`.
    """

    message: str
    code: str
    status: int | None = None
    provider_retry_after_ms: int | None = None
    request_id: str | None = None
    sent_tokens: int | None = None
    """For a request refused as too long (`CONTEXT_WINDOW_EXCEEDED`): how many
    tokens the provider counted in it, when its refusal gave the number."""
    limit_tokens: int | None = None
    """For the same refusal: how many tokens the provider reads, when its refusal
    gave the number. A shorter request is aimed at this limit. Whoever builds the
    failure reads both numbers one time, from the whole answer of the provider
    (`ph.llm.adapter.too_long_facts`). Nothing reads them again from a message
    that may have been cut down for showing."""


CONTEXT_WINDOW_EXCEEDED = "CONTEXT_WINDOW_EXCEEDED"
TIMEOUT = "TIMEOUT"
"""A request that did not finish in the time it was given. The limit can be the
read or connect timeout of a transport, the deadline of a provider, or the time
that a caller said it would wait (`LlmRuntime.decide`)."""
NO_DECISIONS = "NO_DECISIONS"
"""The adapter registered for a provider does not decide."""
BAD_DECISION = "BAD_DECISION"
"""A route answered a decision in a shape other than the one it was asked in: a
question with no answer, or an answer of another kind than its question."""
MISSING_CREDENTIAL = "MISSING_CREDENTIAL"
"""The key of a route is not set in the place where it is resolved. The code is
named here, beside the other codes that a caller routes on, because the
embedders that resolve a key outside an adapter must report it in the way that
an adapter does."""
EMPTY_RESPONSE = "EMPTY_RESPONSE"
FILE_EXPIRED = "FILE_EXPIRED"
"""A request referenced an uploaded file that the provider no longer has (P7-03).

It has its own code because the remedy is specific and automatic. The handle has
already been forgotten when this is raised, so the next attempt uploads the file
again and succeeds. `CONTEXT_WINDOW_EXCEEDED` is the opposite case: that request
will not fit on the second attempt either. That case is the reason that this
codebase classifies failures and does not retry all of them."""


SamplingSetting: TypeAlias = Literal[
    "temperature", "topP", "topK", "minP", "presencePenalty", "frequencyPenalty"
]
"""A sampling setting, spelled as a route sets it (`ModelRoute`) and as a warning
or `phern doctor` names it. The recommended settings of a model maker consist of
these. Each API takes some of them (`ResolvedModel.sampling`). A setting that the
API of a route cannot carry is dropped with a warning, and not refused."""


class Sampling(WireModel):
    """The sampling settings, as one value that every layer carries whole. A route
    sets it (`ModelRoute.sampling`), the options of an agent and the logged call
    config carry it, and a request sends it. A setting that is added here reaches
    all of them. Before this model, each layer had six fields, and a new setting
    had to be added on each layer and copied between them by hand.

    The alias of each field is its `SamplingSetting`, so the YAML of a route, a
    warning and `phern doctor` spell a setting one way (`topP`). An unset setting
    is `None`, and it is absent from what an adapter sends.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def is_empty(self) -> bool:
        """Whether this sets nothing. A route or a header that sets no sampling
        carries an empty value and leaves it off its wire form. The wire form is
        then the same as before this model existed."""
        return not self.settings()

    def settings(self) -> dict[SamplingSetting, float | int]:
        """The settings that this sets, by name. An unset setting is absent, so an
        adapter sends nothing for it."""
        return cast("dict[SamplingSetting, float | int]", self.to_wire())

    def uncarried(self, carried: frozenset[SamplingSetting]) -> dict[SamplingSetting, float | int]:
        """The settings that this sets and that a route carrying `carried` cannot send,
        with the values that were asked. The `llm` seam drops these settings and
        `phern doctor` names them, by this one rule."""
        return {name: value for name, value in self.settings().items() if name not in carried}

    def without(self, settings: Collection[str]) -> Sampling:
        """This value with `settings` unset. It is the one drop rule, for the call
        config that the loop logs and for a request that reaches the seam another
        way."""
        kept = {name: value for name, value in self.settings().items() if name not in settings}
        return Sampling.model_validate(kept)

    def on_wire(self, wire: Mapping[SamplingSetting, str]) -> dict[str, float | int]:
        """The settings that this sets and that a wire names, under the names of the
        wire. An adapter builds its request fields with it, from its own map of what
        its API takes."""
        return {wire[name]: value for name, value in self.settings().items() if name in wire}


DEFAULT_EFFORT: Final = "default"
"""The reasoning effort that asks for the default of the provider. Nothing is sent,
so the provider default applies. It is not a setting of pH, so no profile defines
it (`ModelRoute.efforts`) and nothing warns about it. A route, a spawn or a skill
names it to run at the provider default in place of an effort that its route
sets."""

EffortFields: TypeAlias = JsonObject
"""The request body fields that one reasoning effort sends on the API of a route,
as its profile wrote them (`ModelRoute.efforts`). They are opaque to everything
except the adapter that merges them (`wire_effort`)."""

CallSetting: TypeAlias = SamplingSetting | Literal["reasoningEffort"]
"""A call setting that a route may be unable to carry, spelled as a route sets it.
It is a sampling setting that the API of the route does not take, or a reasoning
effort that its profile does not say how to send (`ModelRoute.efforts`).
`EpochHeader.withheld` is keyed by it: by the alias of a sampling setting on
`Sampling`, and by the alias of the effort on `LlmCallConfig`."""


def unmapped_effort(effort: str | None, efforts: Collection[str]) -> TypeGuard[str]:
    """Whether `effort` is asked for and a route whose profile defines `efforts`
    cannot send it. The `llm` seam withholds such an effort and `phern doctor`
    names it, by this one rule. Unset and `DEFAULT_EFFORT` ask for nothing to be
    sent, so neither is unmapped."""
    return effort is not None and effort != DEFAULT_EFFORT and effort not in efforts


class LlmCallConfig(WireModel):
    """The call configuration of the conversation: the route plus its sampling.

    It is separate from `GenerateOptions` because this is the part that is
    *durable*. `request/header` snapshots it, and header equality compares it.
    So a request that differs only in its message list appends no new header.
    """

    provider: str
    model: str
    reasoning_effort: str | None = None
    sampling: Sampling = Field(default_factory=Sampling, exclude_if=Sampling.is_empty)
    max_tokens: int | None = None
    stop: list[str] | None = None

    def without(self, settings: Collection[CallSetting]) -> LlmCallConfig:
        """This config with `settings` unset: what a route that cannot carry them is sent.

        `asking` is its inverse. These two methods are the one place that tells
        the two kinds of `CallSetting` apart: a sampling setting belongs to
        `sampling`, and `reasoningEffort` belongs to this model."""
        effort = None if "reasoningEffort" in settings else self.reasoning_effort
        return self.model_copy(
            update={"sampling": self.sampling.without(settings), "reasoning_effort": effort}
        )

    def asking(self, withheld: Mapping[CallSetting, float | int | str]) -> LlmCallConfig:
        """This config with what `without` removed asked for again
        (`EpochHeader.withheld`). The result is validated, because what comes back
        from a log is only a value."""
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
# Chunks are values on the per-token hot path (D4), so they are frozen
# dataclasses on `WireDataclass` and not pydantic models. Their wire form still
# follows the one alias rule.


FinishKind: TypeAlias = Literal["stop", "tool-calls", "max-tokens", "aborted", "error"]
"""Why a response stopped, as the five answers that every consumer branches on.

It is a name and not an inline literal because an adapter must *return* one. The
third adapter to be written spelled the union out again to say so. That put the
same list in a second place, where it could grow a sixth member that nothing
else handles.

`aborted` is the one answer that no adapter in this repo produces. A canceled
run reaches the loop as an exception and never as a chunk, so `TurnEndReason`
carries that answer. The value stays here for an adapter whose provider can
report a cancellation on the wire, and `ph.agent_loop.driver` routes it together
with `error`."""


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
"""The raw streaming protocol that adapters emit.

Contract:

* Block indexes correlate interleaved deltas.
* `block-end` carries the assembled block.
* `usage` precedes the terminal `finish`, and nothing follows `finish`.
* Tool arguments stay raw JSON strings until `block-end`.

An adapter may raise. The runtime normalizes that into a terminal
`finish{error}` before any consumer sees it, so a consumer never has to handle
both shapes.
"""


_CHUNKS: TypeAdapter[StreamChunk] = TypeAdapter(
    Annotated[StreamChunk, Field(discriminator="type")],
    config=ConfigDict(alias_generator=wire_alias, populate_by_name=True, extra="ignore"),
)
"""The tagged union, decoded in the way that `ContentBlock` already is.

**The read side is not hot.** `ph.session.events` says the same of its own
dataclass, for the same reason. D4 is about *constructing* a chunk for each
token. An adapter does that as it streams a message, and this decoder is never
on that path. It has one caller, `ph.llm.replay`, which reads a log back.

This replaced a ladder of eight branches. The ladder was the second answer in
one file to a problem that `content_from_wire` had already solved. It also had a
hole. The chunks are frozen dataclasses and validate nothing. A missing key
raised, but a *mis-typed* key built silently: `{"index": "x"}` produced a
`BlockStart` whose index was a string, and handed it to an assembler that
indexes and concatenates.

The config says `extra="ignore"` and not the `forbid` of `WireModel`. A log is
read by builds older than the build that wrote it, so a key that was added later
is skipped and does not condemn the step. The ladder behaved this way, and that
behavior is kept.
"""


def chunk_from_wire(wire: JsonObject) -> StreamChunk:
    """Rebuild one stream chunk from its logged JSON form (replay fidelity).

    A malformed chunk is reported as a `ValueError` that names the chunk kind and
    the field path. A replay reader can then say which line was wrong, and does
    not surface a `KeyError` from three frames down.
    """
    try:
        # `thaw_json` comes first. The log freezes payloads into
        # `MappingProxyType` and `tuple`, and pydantic refuses those for a
        # *dataclass* target. `WireModel` thaws them in `_thaw_frozen_input`, and
        # a `WireDataclass` has no equivalent. The thaw covers the whole tree.
        # The shallow `dict(...)` that was here before left nested frozen
        # containers in `replay_state`.
        return _CHUNKS.validate_python(thaw_json(wire))
    except ValidationError as error:
        raise ValueError(f"malformed stream chunk: {validation_errors(error)}") from error


def is_token_delta(chunk: StreamChunk) -> bool:
    """Whether a chunk carries visible output. The first such chunk is the
    first-token boundary.

    Empty deltas (heartbeats, empty tool-call frames) do not count.
    """
    if isinstance(chunk, (TextDelta, ReasoningDelta)):
        return chunk.text != ""
    if isinstance(chunk, ToolCallDelta):
        return chunk.arguments_delta != "" or chunk.name is not None
    return False


def is_error_finish(chunk: StreamChunk) -> bool:
    """Whether this chunk is a stream reporting that it failed. Nothing after it counts.

    A provider wire can report a failed request *inside* a 200 response. An
    adapter that kept reading past such a report went on to close its
    half-written blocks and to append a second `Finish` that read "stop", and
    the turn was then recorded as "stop". Each adapter carried a `failed` flag
    to notice the report. The flag was a round trip through mutable state for a
    fact that the chunk already states, and a second code path that emitted an
    error `Finish` would have had to remember to set it. The fact is derived
    here, in one place, for every adapter, including adapters that this repo
    does not own.
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
    """The reasoning effort that is asked for, as the API of this route spells it.
    These are the request body fields that the profile gives that effort
    (`ModelRoute.efforts`), and the adapter merges them into its body
    (`wire_effort`). No adapter spells an effort of its own, so `None` sends none
    and the provider default applies. The fields are request body like any
    other. So an `llm/stream` listener that moves a request to another route owns
    them, as it owns `max_tokens`: the request names no effort to look up on the
    new route."""
    sampling: Sampling = field(default_factory=Sampling)
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    response_schema: dict[str, Any] | None = None
    """A JSON Schema that the *text* of the reply must satisfy (P7-17).

    It is for the calls that want a value and not prose. The review gate and the
    planner of `rlm-harness` are such calls. Before this field, they asked for
    JSON in the prompt and read the reply with a tolerant parser. An instruction
    to return only JSON is not a guarantee. With a schema, the wire can enforce
    the shape.

    **This is deliberately not a `LlmCallConfig` field.** `request/header`
    snapshots that class, and header equality compares it. A schema there would
    put the shape of one caller into the cached prefix of a conversation that
    has nothing to do with it (A12). The schema is per request, like `tools`.

    **It is not combined with tools.** A tool call is not a JSON document in the
    shape of the schema, so a route that was asked for both would have to break
    one promise. A caller that wants structure asks for it on a call that offers
    no tools. `ph.llm.structured.ask_for_shape` refuses the combination, and
    does not let a provider decide which to honor."""
    session_id: str | None = None
    purpose: Literal["compaction", "refine"] | None = None
    """Why an auxiliary call is being made. An ordinary conversation request leaves
    it unset. So a session-bound request with no `purpose` *is* a loop request,
    and the "model-visible means logged" invariant holds it to the log. Nothing
    outside the loop can opt a conversation request out by leaving a flag at its
    default. It would have to name a purpose that it does not have.

    For that reason the set is closed and is not a free string: opting out must
    be a declaration that this file can enumerate. `refine` is the planner and
    the review gate of the Continual Harness (P3-16). Those calls are *about*
    the conversation, and their prompt is deliberately not its derivation."""

    @property
    def is_loop_request(self) -> bool:
        return self.session_id is not None and self.purpose is None


def wire_effort(body: dict[str, Any], options: GenerateOptions) -> None:
    """Merge the effort fields of `options` (`GenerateOptions.effort_fields`) into
    the request `body` of an adapter. These are the fields that its reasoning
    effort means on its route. The merge runs after the adapter has set its own
    fields. An object merges key by key, so a profile can reach into
    `generationConfig` beside the fields that the adapter put there. Any other
    value replaces what was there."""
    if options.effort_fields is not None:
        _merge(body, options.effort_fields)


def _merge(into: dict[str, Any], fields: EffortFields) -> None:
    for key, value in fields.items():
        here = into.get(key)
        if isinstance(value, Mapping) and isinstance(here, dict):
            _merge(here, value)
        else:
            into[key] = thaw_json(value)
