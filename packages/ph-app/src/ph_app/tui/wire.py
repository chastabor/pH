"""Reading the log's plain JSON without a model in the way.

Every reader in the TUI — the transcript adapter, the session lister — sees
payloads in one of two shapes: **frozen**, when the event is in memory (a
`MappingProxyType` over tuples), or **plain**, when it was persisted and read
back (`dict`s over `list`s). The helpers here accept both, and the distinction
is load-bearing: a reader that tested for `dict` would work on resume and
silently see nothing live. Phase 1 hit the same trap with `run_code` arguments.

Absence is normal too. The log is JSON, every field is optional to a reader, and
a missing one must cost a row rather than the transcript.

@module ph_app.tui.wire
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["first", "obj", "one_line", "seq", "text_of_wire"]


def obj(value: Any) -> Mapping[str, Any]:
    """A wire object, or an empty one."""
    return value if isinstance(value, Mapping) else {}


def seq(value: Any) -> Sequence[Any]:
    """A wire list, or an empty one. A tuple in memory, a list on disk."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return ()
    return value


def first(value: Any) -> Mapping[str, Any]:
    """The first object of a wire list, or an empty one."""
    items = seq(value)
    return obj(items[0]) if items else {}


def text_of_wire(blocks: Any, *, kind: str = "text") -> str:
    """Join the text of wire-form content blocks of one kind.

    The wire-form twin of `ph.llm.types.text_of`, which needs models. `kind`
    selects `text` or `reasoning`; anything else in the list is skipped.
    """
    return "\n".join(
        str(block.get("text", ""))
        for block in seq(blocks)
        if isinstance(block, Mapping)
        if block.get("type") == kind
    )


def one_line(text: str, limit: int = 120) -> str:
    """Whitespace collapsed and truncated with an ellipsis — a card subtitle."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[: limit - 1]}…"
