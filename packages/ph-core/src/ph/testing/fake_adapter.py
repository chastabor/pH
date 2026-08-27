"""`llm-fake` — a scripted adapter, so Phase 0 is testable without a provider.

Emits the same chunk protocol a real adapter does — `block-start`, deltas,
`block-end`, `usage`, `finish` — so the loop, the assembler and the log are
exercised against the real contract rather than a shortcut. Phase 1's
`llm-replay` (P1-24) replaces the *script* with a recorded session and keeps
this shape.

@module ph.testing.fake_adapter
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, plugin
from ..llm.adapter import ResolvedModel
from ..llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    TextBlock,
    TextDelta,
    TokenUsage,
    UsageChunk,
)

__all__ = ["FakeAdapter", "apply", "text_script"]


def text_script(*replies: str) -> Callable[[GenerateOptions], str]:
    """Answer each request with the next scripted reply, repeating the last."""
    pending = list(replies) or ["(no scripted reply)"]

    def respond(_request: GenerateOptions) -> str:
        return pending.pop(0) if len(pending) > 1 else pending[0]

    return respond


@dataclass(slots=True)
class FakeAdapter:
    """A deterministic adapter that streams a scripted text reply."""

    respond: Callable[[GenerateOptions], str] = field(default_factory=lambda: text_script("ok"))
    chunk_size: int = 8
    context_window: int | None = 8192
    requests: list[GenerateOptions] = field(default_factory=list)

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        self.requests.append(options)
        text = self.respond(options)
        yield BlockStart(index=0, block_type="text")
        for start in itertools.count(0, self.chunk_size):
            piece = text[start : start + self.chunk_size]
            if not piece:
                break
            yield TextDelta(index=0, text=piece)
        yield BlockEnd(index=0, block=TextBlock(text=text))
        yield UsageChunk(
            usage=TokenUsage(
                input_tokens=_estimate_tokens(options),
                output_tokens=max(1, len(text) // 4),
            )
        )
        yield Finish(reason=FinishReason(kind="stop"))

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(context_window=self.context_window)


def _estimate_tokens(options: GenerateOptions) -> int:
    """A four-chars-per-token estimate over visible text, as dsh's meter does."""
    characters = sum(
        len(block.text)
        for message in options.messages
        for block in message.content
        if isinstance(block, TextBlock)
    )
    return characters // 4


@plugin("llm-fake", inject=["llm"])
async def apply(ctx: Context, config: Any) -> None:
    """Register the fake adapter for the routes a profile names."""
    providers: Sequence[str] = ("fake",)
    replies: Sequence[str] = ()
    if isinstance(config, dict):
        providers = tuple(config.get("providers") or providers)
        replies = tuple(config.get("replies") or ())
    adapter = FakeAdapter(respond=text_script(*replies) if replies else text_script("ok"))
    handle = ctx.llm.register_adapter(providers, adapter)
    ctx.provide("llm_fake", adapter)
    ctx.add_disposer(handle.dispose, label="llm-fake")
