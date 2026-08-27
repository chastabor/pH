"""Request-header reconstruction: what any request was built under.

`request/header` is a full snapshot, not a delta, so anyone holding the log can
reconstruct the header in force at any point by taking the latest one. The loop
uses the same equality to avoid logging an unchanged header — which is what
keeps the cached prefix stable (A12).

Ported from dsh `packages/core/session/src/request-header.ts`.

@module ph.session.request_header
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..llm.types import LlmCallConfig, LlmCallConfigAdapterDefaults, ToolSchema
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
    adapter_defaults: LlmCallConfigAdapterDefaults | None = None
    system: str | None = None
    tools: list[ToolSchema] | None = None


class RequestContext(WireModel):
    """Registration-bound metadata for one resolved model route.

    Logged only when it changes, and deliberately excluded from header equality:
    a provider advertising a different context window does not change what the
    model was asked, so it must not invalidate the prefix.
    """

    provider: str
    model: str
    context_window: int | None = None


def canonical_header(header: EpochHeader) -> EpochHeader:
    """Normalize to the one representation logging and comparison both use.

    An empty system prompt and an empty tool list become absent, matching how
    requests are actually built — otherwise `[]` and `None` would compare
    unequal and append a header on every step.
    """
    defaults = header.adapter_defaults
    keep_defaults = defaults is not None and (
        defaults.reasoning_effort is True or defaults.max_tokens is True
    )
    return EpochHeader(
        config=header.config,
        adapter_defaults=defaults if keep_defaults else None,
        system=header.system if header.system else None,
        tools=list(header.tools) if header.tools else None,
    )


def header_equals(a: EpochHeader, b: EpochHeader) -> bool:
    """Equality over canonical headers.

    The models are frozen `WireModel`s, so field-wise equality — tool schemas in
    order, dicts key-order-independent — is what pydantic already gives them.
    """
    return canonical_header(a) == canonical_header(b)


def fold_latest[T](
    events: Sequence[SessionEvent],
    event_type: str,
    parse: Callable[[SessionEvent], T],
    start: T | None = None,
) -> T | None:
    """The latest event of one type, parsed — the shape of every snapshot fold."""
    state = start
    for event in events:
        if event.type == event_type:
            state = parse(event)
    return state


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
