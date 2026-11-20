"""Render a tool card for a front end that cannot ask the tool (P5-14).

A `ToolCard` gets its title, its kind and its one-line summary from the
`present_call`/`present_result` of the tool itself. Those are Python callables,
held in the registry that the harness mounted. The adapter uses `ctx.tools` for
that and for nothing else. Over a socket there is no `ctx.tools` to use. The two
alternatives are both worse than this module. One is to ship a client that
renders every card as `generic`/`bash --version`. The other is to teach the
client to guess from tool names, which puts the tool set of the daemon in the
source of the client.

The daemon therefore renders the **view** and sends it beside the event, never
inside it. The view goes beside the event because it is *derived*: it is
recomputed from the definitions that are mounted now, and an event is what the
log recorded. A view in `data` would make a presentation choice look like a fact
that somebody appended. `_EventWire` forbids the extra key so that this cannot
happen by accident.

**The rendering itself is not here.** `ph.tools.presentation` owns it, because
the terminal renders the same views in process. A second copy of "which hook,
which scope, what a raised hook means" would reopen the split that the sidecar
exists to close. This module keeps the part that belongs to the daemon: it
follows the link in the log to find the call of a result.

**The only subtle part is where the arguments come from.** `present_result`
takes the arguments of the call, and a `tool/result` event does not carry them.
`tool/call` recorded them before the body ran (B4), and two copies of one fact
can disagree. The link is already in the log: `batch.py` appends the result with
`source_event_seqs = (call_seq,)`, so one lookup finds the call and no scan is
needed. A result can have no link: a repair closer for a call that was never
recorded. Such a result renders without a view. That is the same "best
available, not exact" contract that `render_call_view` already states.

@module ph_app.daemon.cards
"""

from __future__ import annotations

from typing import Any

from ph.json import as_bool, as_obj, as_str
from ph.session import Session, SessionEvent
from ph.tools import ToolResult
from ph.tools.presentation import CARD_VIEWS, render_call_view, render_result_view
from ph.tools.registry import ToolRuntime

from ..wire import result_block

__all__ = ["CARD_EVENTS", "presentation_of"]

CARD_EVENTS = frozenset(CARD_VIEWS)
"""The event types that carry a view, derived from the one table that names them.

**The callers check this set, and `presentation_of` does not.** The relay runs
one time for each appended event, and that includes every streamed assistant
chunk. An argument is evaluated before the function that would reject it. A gate
inside the function would therefore still pay a service lookup for each chunk to
reach a `return None`."""


def presentation_of(
    tools: ToolRuntime | None,
    session: Session,
    event: SessionEvent,
) -> dict[str, Any] | None:
    """Return the rendered view for one event, or `None` when there is none.

    `None` covers every ordinary absence: a tool that this deployment no longer
    mounts, or a definition with no presentation hooks. The client then renders
    what it renders today without a daemon: the generic card built from the log
    alone.
    """
    if event.type == "tool/call":
        view = render_call_view(
            tools, as_str(event.data.get("name")), as_str(event.data.get("arguments"))
        )
        return None if view is None else view.to_wire()
    call = _call_of(session, event)
    if call is None:
        return None
    settled = render_result_view(
        tools,
        as_str(call.data.get("name")),
        as_str(call.data.get("arguments")),
        ToolResult(
            content=(),
            is_error=as_bool(result_block(as_obj(event.data.get("message"))).get("isError")),
            meta=as_obj(event.data.get("meta")),
        ),
    )
    return None if settled is None else settled.to_wire()


def _call_of(session: Session, event: SessionEvent) -> SessionEvent | None:
    """Return the `tool/call` that this result settles, through the link that the
    log already keeps."""
    for seq in event.source_event_seqs or ():
        found = session.at(seq)
        if found is not None and found.type == "tool/call":
            return found
    return None
