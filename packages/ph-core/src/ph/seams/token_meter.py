"""`ctx.token_meter`: the provider's count is the truth, and ours is for pressure.

There are two numbers, and confusing them causes real bugs.

* **The `usage` that the provider reports is authoritative.** It is what is
  billed, and it is what the context window is actually measured against.
* **An estimate exists only to decide "should we compact *before* asking".** A
  request that has not been made yet has no usage number, so something has to
  guess. A guess that is 15% off is fine for a threshold.

So the baseline switches from the estimate to the reported usage when the first
response arrives (D15), and it never goes back. **From there it follows the
surface until the next response.** What was appended since then is estimated
and added. What a replacement took off the surface since then is estimated and
subtracted: a compaction summary, a context edit by the model itself, an elided
argument. So the trigger of compaction, the gate of a context edit and its
readouts all read one number. That number moves when the context moves. It is
in the units of the provider, and it includes the system prompt and the tool
schemas. `tiktoken` is used when it is installed, and `len/4` otherwise. The
fallback is deliberately crude. A harness that refused to start without an
optional tokenizer would be worse than a harness that sometimes compacts a turn
early.

@module ph.seams.token_meter
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, assert_never

from pydantic import ValidationError

from ..cordis import Context, plugin
from ..keys import TOKEN_METER, TUI_STATUS
from ..llm.types import (
    AttachmentRef,
    MediaBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    text_of,
)
from ..session import Session, SessionEvent, SessionTable, derive_event_message, shadowed_by
from ..text import thousands
from ._registry import contribute_item
from .tui_status import StatusField, StatusReading

__all__ = [
    "CHARS_PER_TOKEN",
    "IMAGE_TOKENS_UNKNOWN",
    "MEDIA_TOKENS_UNKNOWN",
    "PDF_TOKENS_PER_PAGE",
    "SPENDING_TYPES",
    "TokenBaseline",
    "TokenMeter",
    "apply",
    "estimate_media_tokens",
    "event_tokens",
    "reported_usage",
]

log = logging.getLogger("ph.seams.token_meter")

CHARS_PER_TOKEN = 4
"""The `len/4` fallback ratio, matching dsh and Deep Agents."""

IMAGE_PIXELS_PER_TOKEN = 750
"""The published approximation of Anthropic for an image: about
`width x height / 750`.

Of the figures here, it is the only one that comes from the documentation of a
provider and not from judgment. It needs dimensions, and an ingester supplies
them only when it already knew them. To decode an image to find them is the
dependency that P7-02 keeps optional. So the unknown case falls back to a flat
number."""

IMAGE_TOKENS_UNKNOWN = 1_600
PDF_TOKENS_PER_PAGE = 2_000
PDF_TOKENS_UNKNOWN = 4_000
AUDIO_TOKENS_PER_SECOND = 25
MEDIA_TOKENS_UNKNOWN = 1_000
"""Order-of-magnitude costs for media whose exact price this module cannot know.

The figures are deliberately rough. That is defensible here in a way that it
would not be for billing. This estimate exists to answer "should we compact
*before* asking", and a guess that is 15% off is fine for a threshold. What is
**not** fine is a media block that matches none of the branches of `measure`
and contributes zero. To be wrong by a factor is a rounding error beside being
wrong by everything.
"""


def estimate_media_tokens(attachment: AttachmentRef) -> int:
    """What one attachment plausibly costs in a request.

    For each kind, it uses the facts that the ingester recorded when it has
    them, and a flat figure when it does not. The result is never zero. A block
    that the model is shown must cost something, or the pressure trigger
    measures a different conversation from the one that is sent.
    """
    mime = attachment.mime
    if mime.startswith("image/"):
        if attachment.width and attachment.height:
            return max(1, (attachment.width * attachment.height) // IMAGE_PIXELS_PER_TOKEN)
        return IMAGE_TOKENS_UNKNOWN
    if mime == "application/pdf":
        return attachment.pages * PDF_TOKENS_PER_PAGE if attachment.pages else PDF_TOKENS_UNKNOWN
    if (mime.startswith("audio/") or mime.startswith("video/")) and attachment.duration_ms:
        return max(1, (attachment.duration_ms * AUDIO_TOKENS_PER_SECOND) // 1000)
    return MEDIA_TOKENS_UNKNOWN


@dataclass(frozen=True, slots=True)
class TokenBaseline:
    """The best available count of what the next request will cost."""

    tokens: int
    source: str
    """`"usage"` after a provider has reported, and `"estimate"` before that."""
    context_window: int | None = None

    @property
    def pressure(self) -> float | None:
        """Fraction of the window in use, when the window is known."""
        if not self.context_window:
            return None
        return self.tokens / self.context_window


@dataclass(slots=True)
class TokenMeter:
    """The service published as `ctx.token_meter`."""

    ctx: Context
    _encoder: Any = None
    _encoder_tried: bool = False
    _nodes: SessionTable[dict[int, int]] = field(default_factory=SessionTable)
    """The estimate of each surface node, by session and seq (`node_tokens`).
    The event of a node never changes, so its estimate does not change either.
    A long result is therefore tokenized one time, and not on every check until
    the next response, nor again by each reader. An estimate is kept only for a
    session that the store holds, and it goes when the session goes."""

    def _encode(self, text: str) -> int:
        if not self._encoder_tried:
            self._encoder_tried = True
            try:
                # `tiktoken` is not a declared dependency. This import is the
                # check for it: when the import fails, the meter uses `len/4`.
                import tiktoken  # type: ignore[import-not-found]  # noqa: PLC0415

                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                return max(1, len(text) // CHARS_PER_TOKEN)
        return max(1, len(text) // CHARS_PER_TOKEN)

    def measure_text(self, text: str) -> int:
        return self._encode(text) if text else 0

    def measure(self, message: Message) -> int:
        """Estimate one message. This is the measurement for each node that
        compaction sorts by.

        There is one arm for each variant, and not a `getattr` walk over
        `.text`, `.arguments` and `.content`. The duck-typed version read the
        same union by the presence of an attribute. A kind of block that nobody
        had written a probe for then cost **zero** tokens. It did so in
        silence, and in the direction that keeps oversized content inside the
        window, because this number is what compaction sorts and clips by.
        `assert_never` makes that a build failure instead.
        """
        total = 0
        for block in message.content:
            # The arms are in frequency order, which is the only lever left for
            # a `match`. A hit costs ~37 ns plus ~22 ns for each arm that it
            # passed, and a miss costs what the `isinstance` ladder did. An
            # agent transcript is ~55% text and ~36% tool call or result, and
            # media is rarest.
            match block:
                case TextBlock():
                    total += self._encode(block.text)
                case ToolCallBlock():
                    total += self._encode(block.arguments)
                case ToolResultBlock():
                    total += self._encode(text_of(block.content))
                case ReasoningBlock():
                    total += self._encode(block.text)
                case MediaBlock():
                    total += estimate_media_tokens(block.attachment)
                case _ as unhandled:
                    assert_never(unhandled)
        return total

    def estimate_messages(self, messages: Sequence[Message]) -> int:
        return sum(self.measure(message) for message in messages)

    def last_usage(self, session: Session) -> TokenUsage | None:
        """The most recent reported usage in this log.

        This is an incremental fold, and not a walk back through
        `session.events`. The walk materialized a snapshot of the whole log to
        read one field, and it cost more the longer a conversation ran.
        `baseline` reads the same fold on every pressure check, so the cost
        fell exactly where the log was longest.

        The parser stays *here*, with the row that understands `TokenUsage`,
        and `Session` is asked to fold with it. For a short time the fold was
        on `Session` itself, as a method, a slot and a private parser. The log
        model then had to learn the type of a seam so that the seam could read
        it back (I5).
        """
        found = session.projection("assistant/message", _reported_at)
        return None if found is None else found[1]

    def reasoning_reading(self, session: Session) -> StatusReading | None:
        """What the route asks the model to spend on thinking, or nothing.

        The value comes from `Session.request_header`. That is the typed
        accessor, folded incrementally, for the event whose whole payload is
        the call config. Before this, the daemon walked
        `header.config.reasoningEffort` out of a dict by its camelCase alias,
        so a rename returned `""` forever and did not fail.

        **The reading is here and not on the `llm` row, which owns the fact.**
        `ph.session` imports `ph.llm.types`, so `ph.llm` cannot import the seam
        layer at module scope: a `StatusField` that is contributed from there
        is an import cycle. This row is the nearest correct place. It already
        answers "what did this request cost", and effort is the setting that
        changes that answer most.

        It is a *reading* and not a field on every status frame, for the same
        reason as the posture reading beside it. The field that it replaced
        cost an entry on two payload models, a resolver in the supervisor, a
        line in the status sink of the client and a line in `TuiState`.
        """
        header = session.request_header()
        effort = header.config.reasoning_effort if header is not None else None
        return StatusReading(text=str(effort)) if effort else None

    def cache_reading(self, session: Session) -> StatusReading | None:
        """What the last request did with the prompt cache of the provider (P7-14).

        The footer already shows what the conversation *costs*. This reading is
        the other half of that number: how much of it the provider did not have
        to ingest again. It is worth a field because a falling cache hit rate
        is the visible symptom of a moved prefix (A12). A section that changes
        every turn bills the whole conversation again, and nothing else on this
        line would show that.

        **Silence when the provider says nothing.** For a route that reports no
        cache figures at all, this returns `None`, and every route is in that
        case on the first request of a session. The footer renders `None` as
        nothing and not as `cache 0%`. This is the rule of `StatusField.read`:
        a line that always carries every field hides the one field that
        matters.

        The reading uses `last_usage`, the same last reported usage that the
        gauge budgets against, and that keeps the two consistent. Both describe
        the *last request*, and a message that is rewritten in place is not a
        new request. This method used to read `session.latest` directly, while
        `last_usage` was a walk of the whole log. So it deliberately said
        nothing after a rewrite, and did not repeat a figure. Both are now the
        one incremental fold. The inconsistency to avoid is a footer that shows
        a cache rate beside a context percentage that was computed from a
        different message.

        **One compact phrase**, as `StatusReading.text` asks, because the field
        is in the `context` slot and a front end draws it onto the gauge. It
        has no separator of its own, and no space between the count and the
        share of the prompt that the count was: `28.8k(98%)`, then
        `+2.0k stored` when the request also paid to store a prefix.
        """
        usage = self.last_usage(session)
        if usage is None:
            return None
        read, written = usage.cache_read_tokens or 0, usage.cache_write_tokens or 0
        if not read and not written:
            return None
        parts = []
        if read:
            # The share is of the prompt, not of the request. Output tokens are
            # never cacheable, so including them would make a perfect hit rate
            # read as a falling one on a long answer.
            #
            # It goes through `total` and does not add the terms again, which
            # is what that property exists for. A fifth billed term then
            # reaches this percentage and does not silently fall out of it.
            prompt = usage.total - usage.output_tokens
            share = f"({read * 100 // prompt}%)" if prompt else ""
            parts.append(f"{thousands(read)}{share}")
        if written:
            # What this request paid to store. This is the shape of a first
            # turn on a route that bills cache writes. It is not a hit and must
            # not be counted as one.
            parts.append(f"{thousands(written)} stored")
        return StatusReading(text="cache " + "+".join(parts))

    def baseline(self, session: Session, *, pending: Sequence[Message] = ()) -> TokenBaseline:
        """What the next request will cost, from usage when there is any.

        Reported usage, adjusted by what the surface did since, is closer than
        either count alone. The provider counted the request that it answered
        exactly, system prompt and tool schemas included, and only the change
        has to be guessed. **Every surface event after the reply that reported
        adds a node**, and a replacement takes off the nodes that it shadows. A
        shadowed node that the provider counted is estimated and subtracted. A
        shadowed node that was added since then never counts. So a tool result
        that was appended after the reply is in the number. A summary, a
        context edit or a rewrite in place moves the number when it lands, and
        not at the next response. `pending` is added on top, as messages that
        are not logged yet.

        The method walks only the events since that reply, which is the work of
        one step. It estimates each node one time (`node_tokens`), as the
        estimate before any reply does.
        """
        window = None
        context = session.request_context()
        if context is not None:
            window = context.context_window
        found = session.projection("assistant/message", _reported_at)
        if found is None:
            surface = sum(self.node_tokens(session, seq) for seq in session.surface.nodes)
            return TokenBaseline(
                tokens=surface + self.estimate_messages(pending),
                source="estimate",
                context_window=window,
            )
        reported, usage = found
        added: set[int] = set()
        shadowed: set[int] = set()
        for event in session.events_from(reported + 1):
            if event.surface_op is None:
                continue
            added.add(event.seq)
            for seq in shadowed_by(event):
                if seq in added:
                    added.discard(seq)
                else:
                    shadowed.add(seq)
        change = sum(self.node_tokens(session, seq) for seq in added)
        change -= sum(self.node_tokens(session, seq) for seq in shadowed)
        return TokenBaseline(
            tokens=max(0, usage.total + change) + self.estimate_messages(pending),
            source="usage",
            context_window=window,
        )

    def node_tokens(self, session: Session, seq: int) -> int:
        """The estimate of one surface node. It is measured the first time that
        anything asks: the baseline, a section map, a context edit that prices
        what it replaces, or compaction. It is kept for the life of the
        session, because the event of a node never changes. The result is `0`
        for a node that projects to no message."""
        # An estimate is kept only for a session that the store still holds. A
        # reader may still hold a session that has gone, such as a status line
        # that is drawn one more time. That reader gets its answer and nothing
        # is filed (`ph.session.tables`).
        known = self._nodes.setdefault(session, dict)
        tokens = known.get(seq)
        if tokens is None:
            event = session.at(seq)
            message = None if event is None else derive_event_message(event)
            tokens = known[seq] = 0 if message is None else self.measure(message)
        return tokens


def _reported_at(event: SessionEvent) -> tuple[int, TokenUsage] | None:
    """The usage of an event and the seq where it was reported, for the fold that
    `baseline` and `last_usage` share. After that seq, the count of the provider
    is no longer the whole count."""
    usage = reported_usage(event)
    return None if usage is None else (event.seq, usage)


def reported_usage(event: SessionEvent) -> TokenUsage | None:
    """The usage of one event, or `None` when it carries none.

    The function is public and here, and not private in four places. It was the
    fourth spelling of "read `TokenUsage` off an event", and the three before
    it disagreed about a missing value, `null` and a malformed value. A payload
    that does not parse answers `None`, the same as a payload that carries
    nothing. The alternative is a footer that raises on a frame that it could
    have skipped. `fold_latest` then keeps the last usage that did parse.
    """
    usage = event.data.get("usage")
    if not usage:
        return None
    try:
        return TokenUsage.model_validate(usage)
    except ValidationError:
        log.warning("ph.seams.token_meter: %s carries unparseable usage", event.type)
        return None


SPENDING_TYPES = frozenset({"assistant/message", "compaction/summarized", "llm/decided"})
"""The records in a log that spend tokens: the replies of a model, the summarize
calls that shortened its context, and the questions that a row had a decision
model answer. What the log of a goal charges and what the log of a sub-agent
says that it spent both read this one set, so a record that is added to the set
reaches both."""


def event_tokens(event: SessionEvent) -> int:
    """The spend of one record as `TokenUsage.total`, or `0` when the record
    carries no usage that parses. This is the one definition of "tokens" that a
    budget and the account of a child share.

    It is `total` and not `inputTokens + outputTokens`. A sum of two terms left
    most of the input of a cache-heavy run outside the budget, and it disagreed
    with the footer, which showed the same word.
    """
    usage = reported_usage(event)
    return 0 if usage is None else usage.total


@plugin("token-meter", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the token meter, and contribute the two readings that it answers
    for a footer: `reasoning` and `cache`."""
    meter = TokenMeter(ctx=ctx)
    ctx.provide(TOKEN_METER, meter)
    # This uses `contribute_item` and not an `inject`, for the same reason as
    # `diagnostics`. This row must activate in a profile that mounts no front
    # end at all. A dependency on `ctx.tui_status` would make the meter, which
    # compaction needs, wait on a registry that a headless run never mounts.
    for status_field in (
        # `model`: the reading qualifies the model name that the line opens
        # with, because one model at two efforts is two costs and two answers.
        # So a front end draws it onto the name (`flash/high`), and not
        # somewhere further down the line.
        StatusField(id="reasoning", read=meter.reasoning_reading, order=5, slot="model"),
        # `context`: the other half of the number of the gauge, drawn onto the
        # gauge (`context 12%/cache 28.8k(98%)`). `cache_reading` says how it
        # is spelled.
        StatusField(id="cache", read=meter.cache_reading, order=14, slot="context"),
    ):
        contribute_item(ctx, TUI_STATUS, status_field, label=f"token-meter({status_field.id})")
