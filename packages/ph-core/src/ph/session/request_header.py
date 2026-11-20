"""Request-header reconstruction: what any request was built under.

`request/header` is a full snapshot and not a delta. So anyone who holds the log
can reconstruct the header in force at any point: take the latest one. The loop
uses the same equality so that it does not log an unchanged header. That keeps
the cached prefix stable (A12).

Ported from dsh `packages/core/session/src/request-header.ts`.

@module ph.session.request_header
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..llm.types import CallSetting, LlmCallConfig, ToolSchema
from ..wire import WireModel
from .events import SessionEvent

__all__ = [
    "EpochHeader",
    "RequestContext",
    "canonical_header",
    "fold_latest",
    "fold_request_context",
    "fold_request_header",
    "header_equals",
]


class EpochHeader(WireModel):
    """Logged request state outside derived history."""

    config: LlmCallConfig
    """The call configuration as sent. A setting that the route cannot carry is not in it."""
    withheld: dict[CallSetting, float | int | str] | None = None
    """The settings that were asked for and that the route cannot carry, so the
    request went without them (`LlmRuntime.withhold`). Such a setting is a
    sampling setting that the API of the route does not take, or a reasoning
    effort that the profile of the route does not say how to send. They are kept
    by value. So the next request asks for them again and is checked against the
    route at that time, and a reader of the log sees what was asked for and not
    sent."""
    system: str | None = None
    tools: list[ToolSchema] | None = None


class RequestContext(WireModel):
    """Registration-bound metadata for one resolved model route.

    It is logged only when it changes, and it is deliberately excluded from
    header equality. A provider that advertises a different context window does
    not change what the model was asked, so it must not invalidate the prefix.
    """

    provider: str
    model: str
    context_window: int | None = None


def canonical_header(header: EpochHeader) -> EpochHeader:
    """Normalize to the one representation that both logging and comparison use.

    An empty system prompt, an empty tool list and an empty `withheld` become
    absent, which matches how requests are really built. Otherwise `[]` and
    `None` would compare unequal and append a header on every step.
    """
    return EpochHeader(
        config=header.config,
        withheld=header.withheld or None,
        system=header.system if header.system else None,
        tools=list(header.tools) if header.tools else None,
    )


def header_equals(a: EpochHeader, b: EpochHeader) -> bool:
    """Equality over canonical headers.

    The models are frozen `WireModel`s, so pydantic already gives them
    field-wise equality: tool schemas in order, and dicts independent of key
    order.
    """
    return canonical_header(a) == canonical_header(b)


def fold_latest[T](
    events: Sequence[SessionEvent],
    event_type: str,
    parse: Callable[[SessionEvent], T | None],
    start: T | None = None,
    *,
    since: int = 0,
) -> T | None:
    """The latest event of one type, parsed. Every snapshot fold has this shape.

    **When `parse` returns `None`, the fold keeps what it had.** That is why
    `None` is in the signature and not only in this paragraph. An event of the
    right type that carries nothing of this projection does not erase the
    projection. For example, an `assistant/message` that is rewritten in place
    drops its `usage`, and the most recent *reported* usage is still the one
    before it. That reading also lets `parse` act as a filter. So "latest event
    of type X that actually says Y" is an incremental fold and not a scan.

    **The walk goes backwards and stops at the first answer.** A forward walk
    parsed every matching event in the range to keep the last one. That is free
    for a type that appears occasionally. For `assistant/message`, where nearly
    every event is a match, it is N `model_validate` calls. The old code that
    this function replaced walked backwards and parsed exactly one event. This
    function iterates the same way, so it keeps that property and gains the
    incremental cursor. `start` is the answer when the range holds nothing,
    which is the common case with one append at a time.

    **`since` makes this one function and not two.** `_LatestFold.read` kept a
    cursor. To avoid the `events[since:]` copy that a slice would make per read
    per fold, it repeated this loop by index: the same walk, the same
    stop-at-the-first-answer rule and the same treatment of `None`, in a second
    place, with the justification left as a comment that pointed here. One
    keyword removes the duplicate. An index into a `Sequence` copies nothing, so
    the fold gets its cursor and the rule is written one time.
    """
    for index in range(len(events) - 1, since - 1, -1):
        event = events[index]
        if event.type == event_type and (parsed := parse(event)) is not None:
            return parsed
    return start


def fold_request_header(
    events: Sequence[SessionEvent], start: EpochHeader | None = None
) -> EpochHeader | None:
    """Fold header events into the header in force after the last snapshot."""
    return fold_latest(events, "request/header", parse_request_header, start)


def fold_request_context(
    events: Sequence[SessionEvent], start: RequestContext | None = None
) -> RequestContext | None:
    return fold_latest(events, "request/context", parse_request_context, start)


def parse_request_header(event: SessionEvent) -> EpochHeader:
    return canonical_header(EpochHeader.model_validate(event.data["header"]))


def parse_request_context(event: SessionEvent) -> RequestContext:
    return RequestContext.model_validate(event.data)
