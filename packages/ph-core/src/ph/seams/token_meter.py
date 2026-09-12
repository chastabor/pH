"""`ctx.token_meter` — the provider's count is the truth; ours is for pressure.

Two numbers, and conflating them causes real bugs.

* **Provider-reported `usage` is authoritative.** It is what gets billed and
  what the context window is actually measured against.
* **An estimate exists only to decide "should we compact *before* asking".**
  There is no usage number for a request that has not been made yet, so
  something has to guess, and a guess that is 15% off is fine for a threshold.

So the baseline switches from estimate to reported usage the moment the first
response lands (D15), and never drifts back. `tiktoken` is used when installed
and `len/4` otherwise — the fallback is deliberately crude, because a
harness that refused to start without an optional tokenizer would be worse than
one that occasionally compacts a turn early.

@module ph.seams.token_meter
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..cordis import Context, plugin
from ..keys import TOKEN_METER, TUI_STATUS
from ..llm.types import AttachmentRef, Message, TokenUsage, attachment_of
from ..session import Session, SessionEvent
from ..text import thousands
from ._registry import contribute_item
from .tui_status import StatusField, StatusReading

__all__ = [
    "CHARS_PER_TOKEN",
    "IMAGE_TOKENS_UNKNOWN",
    "MEDIA_TOKENS_UNKNOWN",
    "PDF_TOKENS_PER_PAGE",
    "TokenBaseline",
    "TokenMeter",
    "apply",
    "estimate_media_tokens",
    "reported_usage",
]

log = logging.getLogger("ph.seams.token_meter")

CHARS_PER_TOKEN = 4
"""The `len/4` fallback ratio, matching dsh and Deep Agents."""

IMAGE_PIXELS_PER_TOKEN = 750
"""Anthropic's published approximation for an image: about `width x height / 750`.

The one figure below that comes from a provider's own documentation rather than
from judgement. It needs dimensions, which an ingester supplies only when it
already knew them — decoding an image to find out is the dependency P7-02 keeps
optional — so the unknown case falls back to a flat number."""

IMAGE_TOKENS_UNKNOWN = 1_600
PDF_TOKENS_PER_PAGE = 2_000
PDF_TOKENS_UNKNOWN = 4_000
AUDIO_TOKENS_PER_SECOND = 25
MEDIA_TOKENS_UNKNOWN = 1_000
"""Order-of-magnitude costs for media whose exact price this cannot know.

Deliberately rough, and defensible here in a way it would not be for billing: this
estimate exists to answer "should we compact *before* asking", and a guess 15% off
is fine for a threshold. What is **not** fine is a media block matching none of
`measure`'s branches and contributing zero — being wrong by a factor is a rounding
error against being wrong by everything.
"""


def estimate_media_tokens(attachment: AttachmentRef) -> int:
    """What one attachment plausibly costs in a request.

    Uses the facts the ingester recorded when it has them and a flat figure when
    it does not, per kind. Never zero: a block the model is shown must cost
    something, or the pressure trigger is measuring a different conversation than
    the one being sent.
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
    """`"usage"` once a provider has reported; `"estimate"` before that."""
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

    def _encode(self, text: str) -> int:
        if not self._encoder_tried:
            self._encoder_tried = True
            try:
                import tiktoken  # type: ignore[import-not-found]

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
        """Estimate one message — the per-node measurement compaction sorts by."""
        total = 0
        for block in message.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                total += self._encode(text)
                continue
            attachment = attachment_of(block)
            if attachment is not None:
                total += estimate_media_tokens(attachment)
                continue
            arguments = getattr(block, "arguments", None)
            if isinstance(arguments, str):
                total += self._encode(arguments)
                continue
            nested = getattr(block, "content", None)
            if isinstance(nested, list):
                for inner in nested:
                    inner_text = getattr(inner, "text", None)
                    if isinstance(inner_text, str):
                        total += self._encode(inner_text)
        return total

    def estimate_messages(self, messages: Sequence[Message]) -> int:
        return sum(self.measure(message) for message in messages)

    def last_usage(self, session: Session) -> TokenUsage | None:
        """The most recent reported usage in this log.

        An incremental fold, not a walk back through `session.events` — which
        materialised a snapshot of the whole log to read one field, and grew
        more expensive the longer a conversation ran. `baseline` asks on every
        pressure check, so the cost landed exactly where the log was longest.

        The parser stays *here*, with the row that understands `TokenUsage`, and
        `Session` is asked to fold with it. It briefly lived on `Session`
        itself — a method, a slot and a private parser — which made the log
        model learn a seam's type so the seam could read it back (I5).
        """
        return session.projection("assistant/message", reported_usage)

    def reasoning_reading(self, session: Session) -> StatusReading | None:
        """What the route asks the model to spend on thinking, or nothing.

        Off `Session.request_header` — the typed, incrementally folded accessor
        for the event whose whole payload is the call config. The daemon walked
        `header.config.reasoningEffort` out of a dict by its camelCase alias
        before this, so a rename returned `""` forever instead of failing.

        **Here rather than on the `llm` row, which owns the fact.** `ph.session`
        imports `ph.llm.types`, so `ph.llm` cannot import the seam layer at
        module scope — a `StatusField` contributed from there is an import
        cycle. This row is the nearest honest home: it already answers "what did
        this request cost", and effort is the setting that most changes the
        answer.

        A *reading* rather than a field on every status frame, for the reason
        the posture beside it is one. The field it replaced cost an entry on two
        payload models, a resolver in the supervisor, a line in the client's
        status sink and one in `TuiState`.
        """
        header = session.request_header()
        effort = header.config.reasoning_effort if header is not None else None
        return StatusReading(text=str(effort)) if effort else None

    def cache_reading(self, session: Session) -> StatusReading | None:
        """What the last request did with the provider's prompt cache (P7-14).

        The footer already shows what the conversation *costs*; this is the
        other half of that number — how much of it the provider did not have to
        re-ingest. It is worth a field because a cache hit rate that falls is
        the visible symptom of a moved prefix (A12): a section that changes
        every turn re-bills the whole conversation, and nothing else on this
        line would show it happening.

        **Silence when the provider says nothing.** A route that reports no
        cache figures at all — and every route does on the first request of a
        session — has this return `None`, which the footer renders as nothing
        rather than as `cache 0%`. `StatusField.read`'s own rule: a line that
        always carries every field is a line where the one that matters cannot
        be seen.

        The same `last_usage` the gauge budgets against, which is what keeps the
        two consistent: both describe the *last request*, and a message rewritten
        in place is not a new one. This read `session.latest` directly while
        `last_usage` was a whole-log walk — so it deliberately said nothing
        after a rewrite rather than repeat a figure — and now that both are the
        one incremental fold, a footer showing a cache rate beside a context
        percentage computed from a different message is the inconsistency worth
        avoiding.
        """
        usage = self.last_usage(session)
        if usage is None:
            return None
        read, written = usage.cache_read_tokens or 0, usage.cache_write_tokens or 0
        if not read and not written:
            return None
        parts = []
        if read:
            # The share of the prompt, not of the request: output tokens are
            # never cacheable, so including them would make a perfect hit rate
            # read as a falling one on a long answer.
            #
            # Through `total` rather than by re-adding its terms, which is what
            # that property exists for: a fifth billed term reaches this
            # percentage instead of silently falling out of it.
            prompt = usage.total - usage.output_tokens
            share = f" ({read * 100 // prompt}%)" if prompt else ""
            parts.append(f"{thousands(read)} hit{share}")
        if written:
            # What this request paid to store, which is the first-turn shape on
            # a route that bills cache writes: it is not a hit and must not be
            # counted as one.
            parts.append(f"{thousands(written)} stored")
        return StatusReading(text="cache " + " · ".join(parts))

    def baseline(self, session: Session, *, pending: Sequence[Message] = ()) -> TokenBaseline:
        """What the next request will cost, from usage when there is any.

        Reported usage plus an estimate of anything appended since is closer
        than either alone: the provider counted the prefix exactly, and only the
        new tail has to be guessed.
        """
        window = None
        context = session.request_context()
        if context is not None:
            window = context.context_window
        usage = self.last_usage(session)
        if usage is None:
            return TokenBaseline(
                tokens=self.estimate_messages(session.derive_messages())
                + self.estimate_messages(pending),
                source="estimate",
                context_window=window,
            )
        counted = (
            usage.input_tokens
            + usage.output_tokens
            + (usage.cache_read_tokens or 0)
            + (usage.cache_write_tokens or 0)
        )
        return TokenBaseline(
            tokens=counted + self.estimate_messages(pending),
            source="usage",
            context_window=window,
        )


def reported_usage(event: SessionEvent) -> TokenUsage | None:
    """One `assistant/message`'s usage, or `None` when it carries none.

    Public and here rather than private in four places: this was the fourth
    spelling of "read `TokenUsage` off an event" and the three before it
    disagreed about missing, `null` and malformed. A payload that will not parse
    answers `None` — the same as one that carries nothing — because the
    alternative is a footer that raises on a frame it could have skipped, and
    `fold_latest` then keeps the last usage that did parse.
    """
    usage = event.data.get("usage")
    if not usage:
        return None
    try:
        return TokenUsage.model_validate(usage)
    except ValidationError:
        log.warning("ph.seams.token_meter: %s carries unparseable usage", event.type)
        return None


@plugin("token-meter")
async def apply(ctx: Context, config: None) -> None:
    """Mount the token meter, and the one reading it can answer for a footer."""
    meter = TokenMeter(ctx=ctx)
    ctx.provide(TOKEN_METER, meter)
    # `contribute_item` rather than an `inject`, for `diagnostics`' reason: this
    # row must activate in a profile that mounts no front end at all, and a
    # dependency on `ctx.tui_status` would make the meter — which compaction
    # needs — wait on a registry a headless run never mounts.
    for field_id, read, order in (
        # Ahead of the others, because it qualifies the model name the line
        # opens with: one model at two efforts is two costs and two answers.
        ("reasoning", meter.reasoning_reading, 5),
        ("cache", meter.cache_reading, 14),
    ):
        contribute_item(
            ctx,
            TUI_STATUS,
            StatusField(id=field_id, read=read, order=order),
            label=f"token-meter({field_id})",
        )
