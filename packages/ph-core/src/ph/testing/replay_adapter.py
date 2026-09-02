"""`llm-replay` — re-run a recorded session without a provider.

The loop logs every raw chunk, so a stored session already contains everything
a model call produced. Replay reads them back in order, which buys two things
that are hard to get any other way:

* **a regression test with real model output** and no network, no key, and no
  nondeterminism;
* **the prefix-stability assertion** (A12). Whether consecutive requests share
  a cached prefix is a property of what the *harness* builds, not of the
  provider — so it can be checked exactly, on a real conversation, offline.
  Without this it is checkable only by reading an invoice.

Replay is strict on purpose: running out of recorded steps is an error rather
than a fallback to a canned reply. A replay that quietly invented output would
make a passing test meaningless.

@module ph.testing.replay_adapter
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, plugin
from ..llm.adapter import LlmError, ResolvedModel
from ..llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    chunk_from_wire,
)
from ..session import SessionEvent

__all__ = [
    "REPLAY_ROW",
    "RecordedStep",
    "ReplayAdapter",
    "apply",
    "recorded_steps",
    "shared_prefix",
    "text_chunks",
    "tool_call_chunks",
]

REPLAY_ROW: dict[str, Any] = {"insert": [{"id": "llm-replay", "name": "llm-replay"}]}
"""The row a test inserts to route a replay. Here, beside the adapter it names,
so renaming the plugin is one edit rather than one per test module."""


def shared_prefix(previous: GenerateOptions, current: GenerateOptions) -> int | None:
    """How many leading messages of `current` a provider could serve from `previous`'s prefix.

    A12's definition of a cache hit, written once: the system prompt must be
    byte-identical — it precedes every message, so a change there is a miss
    before the first message — and then the longest run of messages that are
    the same message, by id, in the same position. `test_prefix_stability`
    asserts this equals the whole of `previous`; the P6-03 benchmark prices it.
    Two spellings of one rule is how the structural test and the priced one
    come to disagree about what a hit is.

    **`None` for a changed system prompt, `0` for a matching one with no shared
    messages** — and the distinction is worth an `Optional`. A provider caches
    the byte prefix, so an identical system prompt is a hit even when the first
    message differs; a first draft returned `0` for both cases and the benchmark
    lost the system prompt's credit on every compaction turn, misreporting one
    row's hit rate by half.
    """
    if (current.system or "") != (previous.system or ""):
        return None
    shared = 0
    for before, after in zip(previous.messages, current.messages, strict=False):
        if before.id != after.id:
            break
        shared += 1
    return shared


def tool_call_chunks(call_id: str, name: str, arguments: str) -> tuple[Any, ...]:
    """The chunk triple a model emits for one tool call, ending the step on `tool-calls`."""
    return (
        BlockStart(index=0, block_type="tool-call"),
        BlockEnd(index=0, block=ToolCallBlock(id=call_id, name=name, arguments=arguments)),
        Finish(reason=FinishReason(kind="tool-calls")),
    )


def text_chunks(text: str) -> tuple[Any, ...]:
    """The chunk quartet a model emits for a text reply, ending the step on `stop`."""
    return (
        BlockStart(index=0, block_type="text"),
        TextDelta(index=0, text=text),
        BlockEnd(index=0, block=TextBlock(text=text)),
        Finish(reason=FinishReason(kind="stop")),
    )


@dataclass(frozen=True, slots=True)
class RecordedStep:
    """One recorded model call: its position, and the chunks it produced."""

    turn: int
    step: int
    chunks: tuple[Any, ...]


def recorded_steps(events: Sequence[SessionEvent]) -> list[RecordedStep]:
    """Group a log's `assistant/chunk` events into per-step streams, in order."""
    grouped: dict[tuple[int, int], list[Any]] = {}
    order: list[tuple[int, int]] = []
    for event in events:
        if event.type != "assistant/chunk":
            continue
        key = (int(event.data["turn"]), int(event.data["step"]))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(chunk_from_wire(dict(event.data["chunk"])))
    return [
        RecordedStep(turn=turn, step=step, chunks=tuple(grouped[(turn, step)]))
        for turn, step in order
    ]


@dataclass(slots=True)
class ReplayAdapter:
    """Serves recorded chunk streams, one per request, in recorded order."""

    steps: list[RecordedStep] = field(default_factory=list)
    cursor: int = 0
    requests: list[GenerateOptions] = field(default_factory=list)
    context_window: int | None = 8192

    @classmethod
    def from_events(cls, events: Sequence[SessionEvent]) -> ReplayAdapter:
        return cls(steps=recorded_steps(events))

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.steps)

    async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
        self.requests.append(options)
        if self.exhausted:
            # Strict: inventing output here would make a replay test pass while
            # proving nothing about the recording.
            raise LlmError(
                f"replay exhausted after {len(self.steps)} recorded steps; the loop "
                "made more requests than the recording contains",
                "REPLAY_EXHAUSTED",
            )
        recorded = self.steps[self.cursor]
        self.cursor += 1
        for chunk in recorded.chunks:
            yield chunk

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(context_window=self.context_window)


@plugin("llm-replay", inject=["llm"])
async def apply(ctx: Context, config: Any) -> None:
    """Register a replay adapter; a test loads its recording."""
    providers = ("replay",)
    if isinstance(config, dict) and config.get("providers"):
        providers = tuple(config["providers"])
    adapter = ReplayAdapter()
    handle = ctx.llm.register_adapter(providers, adapter)
    ctx.provide("llm_replay", adapter)
    ctx.add_disposer(handle.dispose, label="llm-replay")
