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

from ..cordis import Context, plugin
from ..llm.types import Message, TokenUsage
from ..session import Session

__all__ = ["CHARS_PER_TOKEN", "TokenBaseline", "TokenMeter", "apply"]

log = logging.getLogger("ph.seams.token_meter")

CHARS_PER_TOKEN = 4
"""The `len/4` fallback ratio, matching dsh and Deep Agents."""


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
        """The most recent reported usage in this log."""
        for event in reversed(session.events):
            if event.type == "assistant/message" and "usage" in event.data:
                return TokenUsage.model_validate(event.data["usage"])
        return None

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


@plugin("token-meter")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the token meter."""
    ctx.provide("token_meter", TokenMeter(ctx=ctx))
