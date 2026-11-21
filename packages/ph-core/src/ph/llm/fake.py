"""`llm-fake`: a scripted adapter, so that pH can be tested without a provider.

The adapter emits the same chunk protocol that a real adapter does:
`block-start`, deltas, `block-end`, `usage`, `finish`. The loop, the assembler
and the log are therefore exercised against the real contract and not against a
shortcut. `llm-replay` (P1-24) replaces the *script* with a recorded session and
keeps this shape.

@module ph.llm.fake
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Annotated

from pydantic import Field

from ..cordis import Context, plugin
from ..keys import LLM, LLM_FAKE
from ..wire import WireModel
from .adapter import ResolvedModel
from .types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    StreamChunk,
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
    route: ResolvedModel | None = None
    """What `resolve_model` answers, for a test that needs this route to declare
    media (P7-01, P7-03).

    `None`, the default, is text-only. That is what a provider declares when it
    has said nothing. It is also the safe answer, and `media-degrade` exists to
    give it at no cost. The field is a whole `ResolvedModel` and not one field
    for each capability, so a route capability that is added there needs no
    second declaration here."""
    requests: list[GenerateOptions] = field(default_factory=list)

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
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
        if self.route is None:
            return ResolvedModel(context_window=self.context_window)
        # The window stays the window of this adapter unless the override names
        # one. Every limits test and compaction test is sized against it, and a
        # route that is set to declare an image should not move it silently.
        return replace(self.route, context_window=self.route.context_window or self.context_window)


def _estimate_tokens(options: GenerateOptions) -> int:
    """An estimate of four characters per token over visible text, as the meter of dsh does."""
    characters = sum(
        len(block.text)
        for message in options.messages
        for block in message.content
        if isinstance(block, TextBlock)
    )
    return characters // 4


class Config(WireModel):
    """Row config for `llm-fake`: which routes it answers, and with what."""

    providers: Annotated[tuple[str, ...], Field(min_length=1)] = ("fake",)
    """The provider names that this row answers for. `min_length` refuses
    `providers: []`, which would otherwise be read silently as the default. The
    raw-dict reading that this replaced substituted the default for any falsy
    value. That is the silent no-op that the config rules of this row now
    reject."""
    replies: tuple[str, ...] = ()


@plugin("llm-fake", affects="environment", inject=[LLM], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register the fake adapter for the routes a profile names."""
    adapter = FakeAdapter(
        respond=text_script(*config.replies) if config.replies else text_script("ok")
    )
    handle = ctx.require(LLM).register_adapter(config.providers, adapter)
    ctx.provide(LLM_FAKE, adapter)
    ctx.add_disposer(handle.dispose, label="llm-fake")
