"""Rendering a tool card for a front end that cannot ask the tool (P5-14).

A `ToolCard` gets its title, its kind and its one-line summary from the tool's
own `present_call`/`present_result` — Python callables held in the registry the
harness mounted. The adapter reaches for `ctx.tools` for that and for nothing
else. Over a socket there is no `ctx.tools` to reach, and the two ways out are
both worse than this one: ship a client that renders every card as
`generic`/`bash --version`, or teach the client to guess from tool names, which
puts the daemon's tool set in the client's source.

So the daemon renders and sends the **view**, beside the event and never inside
it. Beside, because the view is *derived* — recomputed from the definitions
mounted right now — and an event is what the log said. Putting it in `data`
would make a presentation choice look like a fact somebody appended, and
`_EventWire` forbids the extra key precisely so that cannot happen by accident.

**The rendering itself is not here**: `ph.tools.presentation` owns it, because
the terminal renders the same views in process and a second copy of "which hook,
which scope, what a raised hook means" would reopen the very split the sidecar
exists to close. What is left is the part that is genuinely daemon-side —
following the log's own link to find a result's call.

**Where the arguments come from is the only subtle part.** `present_result`
takes the call's arguments, and a `tool/result` event does not carry them —
`tool/call` recorded them before the body ran (B4), and two copies of one fact
are two that can disagree. The link is already in the log: `batch.py` appends the
result with `source_event_seqs = (call_seq,)`, so the call is one lookup away
rather than a scan. A result whose link is missing — a repair closer for a call
that was never recorded — renders without a view, which is the same "best
available, not exact" contract `render_call_view` already states.

@module ph_app.daemon.cards
"""

from __future__ import annotations

from typing import Any

from ph.session import Session, SessionEvent
from ph.tools import ToolResult
from ph.tools.presentation import render_call_view, render_result_view

from ..wire import obj, result_block

__all__ = ["CARD_EVENTS", "presentation_of"]

CARD_EVENTS = frozenset({"tool/call", "tool/result"})
"""The event types that carry a view.

**Checked by the callers**, not only in here: the relay runs once per appended
event — every streamed assistant chunk included — and an argument is evaluated
before the function that would reject it, so gating inside would still pay a
service lookup per chunk to reach a `return None`."""


def presentation_of(tools: Any, session: Session, event: SessionEvent) -> dict[str, Any] | None:
    """The rendered view for one event, or `None` when there is nothing to say.

    `None` covers every ordinary absence — a tool this deployment no longer
    mounts, a definition with no presentation hooks — and the client then renders
    exactly what it renders today without a daemon: the generic card built from
    the log alone.
    """
    if event.type == "tool/call":
        view = render_call_view(tools, str(event.data.get("name", "")), event.data.get("arguments"))
        return None if view is None else view.to_wire()
    call = _call_of(session, event)
    if call is None:
        return None
    settled = render_result_view(
        tools,
        str(call.data.get("name", "")),
        call.data.get("arguments"),
        ToolResult(
            content=(),
            is_error=bool(result_block(obj(event.data.get("message"))).get("isError")),
            meta=event.data.get("meta"),
        ),
    )
    return None if settled is None else settled.to_wire()


def _call_of(session: Session, event: SessionEvent) -> SessionEvent | None:
    """The `tool/call` this result settles, through the link the log already keeps."""
    for seq in event.source_event_seqs or ():
        found = session.at(seq)
        if found is not None and found.type == "tool/call":
            return found
    return None
