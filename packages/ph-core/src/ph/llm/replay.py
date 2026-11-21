"""`llm-replay`: run a recorded session again without a provider.

The loop logs every raw chunk, so a stored session already contains everything
that a model call produced. Replay reads the chunks back in order. That gives
two things that are hard to get in any other way:

* **A regression test with real model output**, with no network, no key and no
  nondeterminism.
* **The prefix-stability assertion** (A12). Whether consecutive requests share
  a cached prefix is a property of what the *harness* builds, and not of the
  provider. It can therefore be checked exactly, on a real conversation,
  offline. Without replay it can be checked only by reading an invoice.

Replay is strict on purpose. When the recorded steps run out, that is an error,
and replay does not fall back to a canned reply. A replay that invented output
without saying so would make a passing test meaningless.

@module ph.llm.replay
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

from pydantic import Field

from ..cordis import Context, plugin
from ..json import as_int, as_obj
from ..keys import LLM, LLM_REPLAY
from ..session import SessionEvent
from ..wire import WireModel
from .adapter import LlmError, ResolvedModel
from .types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    StreamChunk,
    TextBlock,
    TextDelta,
    ToolCallBlock,
    chunk_from_wire,
)

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
"""The row that a test inserts to route a replay. It is here, beside the adapter
that it names, so that renaming the plugin is one edit and not one edit in each
test module."""


def shared_prefix(previous: GenerateOptions, current: GenerateOptions) -> int | None:
    """The number of leading messages of `current` that the prefix of `previous` could serve.

    This is the definition of a cache hit in A12, written one time. The system
    prompt must be byte-identical. It precedes every message, so a change there
    is a miss before the first message. After that, the count is the longest
    run of messages that are the same message, by id, in the same position.
    `test_prefix_stability` asserts that this equals the whole of `previous`,
    and the P6-03 benchmark puts a price on it. With two spellings of one rule,
    the structural test and the priced test come to disagree about what a hit
    is.

    **`None` means a changed system prompt, and `0` means a matching system
    prompt with no shared messages.** The distinction is worth an `Optional`. A
    provider caches the byte prefix, so an identical system prompt is a hit even
    when the first message differs. A first draft returned `0` for both cases.
    The benchmark then lost the credit for the system prompt on every compaction
    turn, and reported the hit rate of one row wrong by half.
    """
    if (current.system or "") != (previous.system or ""):
        return None
    shared = 0
    for before, after in zip(previous.messages, current.messages, strict=False):
        if before.id != after.id:
            break
        shared += 1
    return shared


def tool_call_chunks(call_id: str, name: str, arguments: str) -> tuple[StreamChunk, ...]:
    """The three chunks that a model emits for one tool call. The step ends on `tool-calls`."""
    return (
        BlockStart(index=0, block_type="tool-call"),
        BlockEnd(index=0, block=ToolCallBlock(id=call_id, name=name, arguments=arguments)),
        Finish(reason=FinishReason(kind="tool-calls")),
    )


def text_chunks(text: str) -> tuple[StreamChunk, ...]:
    """The four chunks that a model emits for a text reply. The step ends on `stop`."""
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
    chunks: tuple[StreamChunk, ...]


def recorded_steps(events: Sequence[SessionEvent]) -> list[RecordedStep]:
    """Group the `assistant/chunk` events of a log into one stream per model call, in order.

    **There is one recorded stream for each *call*, not for each `(turn, step)`.**
    A retried step keeps its turn and step, because that is what makes
    `llm/retry` readable in the log. Grouping by the pair therefore concatenated
    every attempt into one stream with several `Finish` chunks in it. The replay
    then served the whole concatenation to the first request and had one step
    fewer for the rest of the run. A recording of a turn that hit a rate limit
    replayed as a *different* conversation. It ended with the report that the
    loop had made more requests than the recording contains. `REPLAY_EXHAUSTED`
    exists so that this condition cannot be missed, and here it pointed at the
    loop and not at the grouping.

    **The attempt is read, not inferred** (G12). The driver records which
    attempt at `(turn, step)` each chunk belongs to. The boundary between two
    attempts is therefore a fact in the log, and it is not recovered from the
    shape of the stream. The `Finish` split below stays, for two reasons. A log
    that was written before the driver recorded the attempt has no attempt to
    read. And a `Finish` ends a stream by contract in both cases: the adapter
    that emitted one has nothing more to say for that call.
    """
    calls: list[tuple[tuple[int, int, int], list[StreamChunk]]] = []
    for event in events:
        if event.type != "assistant/chunk":
            continue
        key = (
            as_int(event.data["turn"]),
            as_int(event.data["step"]),
            as_int(event.data.get("attempt", 0)),
        )
        chunk = chunk_from_wire(as_obj(event.data["chunk"]))
        # A new call starts when the coordinates move, and also when the last
        # call was closed. `calls[-1][1]` is never empty, because a group is
        # appended only together with the chunk that opens it.
        if not calls or calls[-1][0] != key or isinstance(calls[-1][1][-1], Finish):
            calls.append((key, []))
        calls[-1][1].append(chunk)
    return [
        RecordedStep(turn=turn, step=step, chunks=tuple(chunks))
        for (turn, step, _attempt), chunks in calls
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

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
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


class Config(WireModel):
    """Row config for `llm-replay`: the routes the recording answers for."""

    providers: Annotated[tuple[str, ...], Field(min_length=1)] = ("replay",)
    """The provider names that this row answers for. `min_length` is here for the
    reason that `llm-fake` states: an empty list is a mistake to report, and not
    a falsy value to read as the default."""


@plugin("llm-replay", affects="environment", inject=[LLM], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register a replay adapter. A test loads the recording into it."""
    adapter = ReplayAdapter()
    handle = ctx.require(LLM).register_adapter(config.providers, adapter)
    ctx.provide(LLM_REPLAY, adapter)
    ctx.add_disposer(handle.dispose, label="llm-replay")
