"""UI render intents: what a card *is*, not what it looks like.

A tool says "this call is a diff" or "this is a terminal run"; the front-end
decides typography. The vocabulary is deliberately semantic (the same rule as
`ContextForm`) and every view is computed from `args` plus the durable result —
never from live execution state — because a UI renders these during streaming
*and* during a log replay, and the two must agree.

@module ph.tools.presentation
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, TypeAlias, TypedDict

from ..cordis import DEPLOYMENT
from ..wire import WireModel
from .json_schema import parse_arguments

log = logging.getLogger("ph.tools.presentation")

__all__ = [
    "CardKind",
    "ToolCallView",
    "ToolResultView",
    "ToolViews",
    "render_call_view",
    "render_result_view",
    "simple_views",
]

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


def render_call_view(tools: Any, name: str, arguments: Any) -> ToolCallView | None:
    """Ask the tool how its pending call looks. `None` when it cannot say.

    **Here rather than in either front end**, because there are now two callers
    and they must not drift: the terminal renders in process, and the daemon
    renders the same view to send to a browser that has no registry to ask. The
    split between them is exactly what the sidecar exists to close, so stating
    the rule twice would reopen it one edit later.

    `None` covers every ordinary absence — a tool this deployment no longer
    mounts, a definition with no hook, a hook that raised — and a caller then
    renders the generic card built from the log alone. A presentation hook is a
    row's body and may do anything; a card that cannot be rendered is a plainer
    card, never a dropped event.
    """
    definition = _presentable(tools, name)
    if definition is None or definition.present_call is None:
        return None
    try:
        view: ToolCallView | None = definition.present_call(parse_arguments(arguments))
    except Exception:
        log.debug("ph.tools: %s could not present its call", name)
        return None
    return view


def render_result_view(tools: Any, name: str, arguments: Any, result: Any) -> ToolResultView | None:
    """Ask the tool how its settled call looks. `None` when it cannot say.

    `arguments` are the *call's*, not the result's: a tool presents its outcome
    against what it was asked to do, and `tool/call` is what recorded that (B4).
    """
    definition = _presentable(tools, name)
    if definition is None or definition.present_result is None:
        return None
    try:
        view: ToolResultView | None = definition.present_result(parse_arguments(arguments), result)
    except Exception:
        log.debug("ph.tools: %s could not present its result", name)
        return None
    return view


def _presentable(tools: Any, name: str) -> Any:
    """The definition behind a card, for presentation only.

    `DEPLOYMENT` and not an agent's scope (P6-32): this renders a call the *log*
    already recorded, and `tool/call` events carry no agent, so "which agent's
    presentation" is unanswerable from what the log holds.

    **Best available, not exact, in two ways** (§5 rule 6): a name that was
    agent-*shadowed* at execution time renders under the global definition — a
    wrong card, not a blank one — and an agent-*scoped* tool renders with no
    definition at all, because `DEPLOYMENT` is the mount's chain, not a union
    over agents. Both are presentation-only, and both have the same real fix:
    record the executing agent on `tool/call`, which is a log-schema row rather
    than a boundary choice here.

    Nothing is gated on the answer; it supplies a title and a renderer.
    """
    if tools is None:
        return None
    try:
        return tools.get(name, scope=DEPLOYMENT)
    except Exception:
        return None


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
