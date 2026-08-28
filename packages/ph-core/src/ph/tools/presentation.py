"""UI render intents: what a card *is*, not what it looks like.

A tool says "this call is a diff" or "this is a terminal run"; the front-end
decides typography. The vocabulary is deliberately semantic (the same rule as
`ContextForm`) and every view is computed from `args` plus the durable result —
never from live execution state — because a UI renders these during streaming
*and* during a log replay, and the two must agree.

@module ph.tools.presentation
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypedDict

from ..wire import WireModel

__all__ = ["CardKind", "ToolCallView", "ToolResultView", "ToolViews", "simple_views"]

CardKind: TypeAlias = Literal["generic", "terminal", "diff", "search", "read", "web"]
"""The card families a front-end must know how to draw."""


class ToolCallView(WireModel):
    """How to present a call that has not settled yet."""

    card: CardKind = "generic"
    title: str
    subtitle: str | None = None
    input: str | None = None
    """The call's salient input as one line — a path, a command, a query."""
    body: str | None = None
    """The call's full input, when a card is meant to show it — the program of a
    code cell, a whole command. `input` is the one-line header; this is what a
    widget renders underneath, and leaving it unset keeps the header-only card
    every other tool gets (P3-19)."""


class ToolResultView(WireModel):
    """How to present a settled call.

    `meta` is the tool's own durable payload (a diff, a match count), threaded
    verbatim from the `tool/result` event so replay reproduces the card exactly.
    """

    card: CardKind = "generic"
    title: str
    subtitle: str | None = None
    body: str | None = None
    is_error: bool = False
    meta: dict[str, Any] | None = None


class ToolViews(TypedDict):
    """The two presentation hooks, shaped to unpack into `define_tool(**views)`."""

    present_call: Callable[[Any], ToolCallView | None]
    present_result: Callable[[Any, Any], ToolResultView | None]


def simple_views(card: CardKind, title: str, key: str) -> ToolViews:
    """`present_call`/`present_result` for the common "one salient argument" tool.

    Most tools present as a fixed title over one argument — a path, a pattern.
    Returned as kwargs so a definition reads `**simple_views("read", "Read", "path")`.
    """

    def salient(args: Any) -> str:
        return str(args.get(key, "")) if hasattr(args, "get") else ""

    return ToolViews(
        present_call=lambda args: ToolCallView(card=card, title=title, input=salient(args)),
        present_result=lambda args, result: ToolResultView(
            card=card, title=title, subtitle=salient(args), is_error=result.is_error
        ),
    )
