"""UI render intents: what a card *is*, and not how it looks.

A tool says "this call is a diff" or "this is a terminal run", and the front end
decides the typography. The vocabulary is deliberately semantic, which is the
same rule as for `ContextForm`. Every view is computed from `args` and the
durable result, and never from live execution state. A UI renders these views
during streaming *and* during a log replay, and the two must agree.

@module ph.tools.presentation
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

from ..cordis import DEPLOYMENT
from ..json import JsonObject, as_obj
from ..wire import WireModel
from .json_schema import parse_arguments

if TYPE_CHECKING:
    # `definition` imports this module for `ToolCallView` and `ToolResultView`,
    # so the import in this direction exists only for the type checker. The
    # names are used only in the annotations of function signatures, and not in
    # a runtime alias, so nothing resolves them at run time.
    from .definition import ToolDefinition, ToolResult
    from .registry import ToolRuntime

log = logging.getLogger("ph.tools.presentation")

__all__ = [
    "CARD_VIEWS",
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
    """The salient input of the call as one line: a path, a command, a query."""
    body: str | None = None
    """The full input of the call, when a card is meant to show it: the program
    of a code cell, or a whole command. `input` is the one-line header, and a
    widget renders this field underneath. When this field is unset, the card has
    only the header, as for every other tool (P3-19)."""


class ToolResultView(WireModel):
    """How to present a settled call.

    `meta` is the durable payload of the tool itself (a diff, a match count). It
    is passed through verbatim from the `tool/result` event, so that a replay
    reproduces the card exactly.
    """

    card: CardKind = "generic"
    title: str
    subtitle: str | None = None
    body: str | None = None
    is_error: bool = False
    meta: dict[str, Any] | None = None
    """The durable payload of the tool itself. The values are `Any` and not
    `JsonValue` as in the `ToolResult.meta` that it comes from. This is a
    pydantic field, and `JsonValue` is a *recursive* alias whose schema pydantic
    cannot build. `ApprovalRequest.arguments` keeps `Any` for the same reason.
    `test_every_wire_model_is_built_at_import` checks this."""


def render_call_view(tools: ToolRuntime | None, name: str, arguments: str) -> ToolCallView | None:
    """Ask the tool how its pending call looks. `None` when the tool cannot say.

    `arguments` is a `str` because that is what the writer records: the bytes
    from the model, unparsed (`batch._append_call`). A log that another build
    wrote could hold something else there. The reader of that log narrows, so
    such a call arrives here as no arguments, and not as a shape that
    `parse_arguments` would refuse.

    **The function is here and not in either front end**, because there are two
    callers and they must not drift. The terminal renders in process. The daemon
    renders the same view and sends it to a browser that has no registry to ask.
    The sidecar exists to close exactly that split between them, so a rule that
    was stated twice would reopen the split one edit later.

    `None` covers every ordinary absence: a tool that this deployment no longer
    mounts, a definition with no hook, a hook that raised. A caller then renders
    the generic card that is built from the log alone. A presentation hook is
    the body of a row and may do anything. A card that cannot be rendered
    becomes a plainer card, and never a dropped event.
    """
    definition = _presentable(tools, name)
    if definition is None or definition.present_call is None:
        return None
    try:
        view: ToolCallView | None = definition.present_call(as_obj(parse_arguments(arguments)))
    except Exception:
        log.debug("ph.tools: %s could not present its call", name)
        return None
    return view


def render_result_view(
    tools: ToolRuntime | None,
    name: str,
    arguments: str,
    result: ToolResult,
) -> ToolResultView | None:
    """Ask the tool how its settled call looks. `None` when the tool cannot say.

    `arguments` belongs to the *call* and not to the result. A tool presents its
    outcome against what it was asked to do, and `tool/call` is the record of
    that (B4).
    """
    definition = _presentable(tools, name)
    if definition is None or definition.present_result is None:
        return None
    try:
        view: ToolResultView | None = definition.present_result(
            as_obj(parse_arguments(arguments)), result
        )
    except Exception:
        log.debug("ph.tools: %s could not present its result", name)
        return None
    return view


def _presentable(tools: ToolRuntime | None, name: str) -> ToolDefinition | None:
    """The definition behind a card, for presentation only.

    The lookup uses `DEPLOYMENT` and not the scope of an agent (P6-32). The
    function renders a call that the *log* already recorded, and `tool/call`
    events carry no agent. So the log cannot answer "the presentation of which
    agent".

    **The answer is the best available and not exact, in two ways** (§5 rule 6).
    A name that an agent *shadowed* at execution time renders under the global
    definition, which gives a wrong card and not a blank one. An agent-*scoped*
    tool renders with no definition at all, because `DEPLOYMENT` is the chain of
    the mount and not a union over agents. Both affect presentation only, and
    both have the same real fix: record the executing agent on `tool/call`. That
    is a change to the log schema, and not a choice of boundary here.

    Nothing is gated on the answer. It supplies a title and a renderer.
    """
    if tools is None:
        return None
    try:
        return tools.get(name, scope=DEPLOYMENT)
    except Exception:
        return None


CARD_VIEWS: Mapping[str, type[ToolCallView] | type[ToolResultView]] = {
    "tool/call": ToolCallView,
    "tool/result": ToolResultView,
}
"""The session event types that carry a card view, and the model that each one
carries.

This is one table because the same fact used to be written in three places: a
set of event types in the daemon, an `if` in the renderer of the daemon, and an
`if/elif` in the reader of the client. A third card event needed three edits,
and one forgotten edit was a silent miss."""


class ToolViews(TypedDict):
    """The two presentation hooks, shaped to unpack into `define_tool(**views)`."""

    present_call: Callable[[JsonObject], ToolCallView | None]
    present_result: Callable[[JsonObject, ToolResult], ToolResultView | None]


def simple_views(card: CardKind, title: str, key: str) -> ToolViews:
    """`present_call` and `present_result` for the common tool that has one
    salient argument.

    Most tools present as a fixed title over one argument, such as a path or a
    pattern. The result is returned as keyword arguments, so a definition reads
    `**simple_views("read", "Read", "path")`.
    """

    def salient(args: JsonObject) -> str:
        # `str` and not `as_str`, because two of these keys hold a *list*:
        # `code_index` and `text_index` are both `simple_views(..., "paths")`
        # over a `list[str]`. The line of the card is the repr of that value.
        # Narrowing made those lines blank, so `test_json_narrowing.py` exempts
        # this expression.
        return str(args.get(key, ""))

    return ToolViews(
        present_call=lambda args: ToolCallView(card=card, title=title, input=salient(args)),
        present_result=lambda args, result: ToolResultView(
            card=card, title=title, subtitle=salient(args), is_error=result.is_error
        ),
    )
