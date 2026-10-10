"""The one door for revising the surface, and the one reading of what a revision stood for.

Every producer that changes what the model sees after the fact — compaction's
summary, its argument elision and overflow clip, input-offload's paste preview, a
model's own context edit — appends a surface `replace`. The rules a replacement
must keep are the same for all of them, so they are kept here, once, and tested
here, once:

* **`substitute`** — text standing in for a run of nodes. Always a `user/message`
  carrying a message of its own (a fresh id), citing every node it shadows.
* **`rewrite`** — one node replaced by a near-copy of itself. An
  `assistant/message` or a `tool/result` only, keeping the message id and every
  field but the message, except that an assistant reply loses `usage`, `turn` and
  `step` (D8, below).

The commit holds the part of this every reader relies on — a replacement in
assistant or tool-result role rewrites exactly one node of its own type and keeps
its id (`surface._assert_assistant_rewrite`, `_assert_tool_result_rewrite`) —
which is what makes `is_in_place_rewrite` exact by type, for this door's writes
and any other. The door adds what a producer would otherwise have to remember: the
payload built from the original, D8's fields left behind, the citation. And the
writers walk (`test_log_writers.py`) holds that no other shipped module constructs
a `SurfaceReplace`, so it stays the one place those are remembered.

Both write into the caller's log *or batch*, through this module's own writer, so
a producer's record of why — `compaction/summarized`, `offload/input-spilled`,
`clm/revised` — lands in the same batch as the revision it describes.

**D8, why a rewritten assistant reply drops `usage`, `turn` and `step`.** The
replacement is appended at the tail. `TokenMeter.last_usage` folds to the newest
`assistant/message` carrying usage, so a copied usage would become the meter's
baseline and say the session had shrunk; a copied `turn`/`step` would announce a
finished step as the newest thing in the log, and every reader keyed on "the
latest assistant message" would read a closed step as the open one. Both belong
to the original, which still has them — and `replaces` names it, a stronger link
than a pair of integers a reader has to match up.

@module ph.session.revise
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ..json import JsonValue, thaw_json
from ..llm.types import Message
from .derive import derive_event_message
from .events import SessionEvent, SurfaceIntent, SurfaceReplace
from .surface import REWRITTEN_IN_PLACE, SurfaceError, is_in_place_rewrite, shadowed_by
from .writers import log_writer

if TYPE_CHECKING:
    from .session import Session, SessionBatch

_LOG = log_writer(__name__)

__all__ = ["editable_message", "origin_of", "originals", "rewrite", "substitute"]


def substitute(
    log: Session | SessionBatch, shadowed: Sequence[int], message: Message
) -> SessionEvent:
    """Append `message` in place of the nodes `shadowed` names.

    The surface lands it where the earliest of them was and takes the rest out; the
    log keeps every one of them (I4). `message` is a user message of its own — a
    summary, a preview, a revision — never one the shadowed nodes already carry.
    """
    if message.role != "user":
        raise SurfaceError(
            "a substitution is a user message: text standing for several turns is no "
            f"one turn's speaker, and this one is {message.role!r}"
        )
    seqs = tuple(shadowed)
    return _LOG.append(
        log,
        "user/message",
        message.to_wire(),
        SurfaceIntent(surface_op=SurfaceReplace(replaces=seqs), source_event_seqs=seqs),
    )


def rewrite(
    log: Session | SessionBatch, event: SessionEvent, message: Mapping[str, JsonValue]
) -> SessionEvent:
    """Append a near-copy of `event` carrying `message`, in place of it.

    `message` is the replacement message's wire form — the original's, changed
    (`editable_message`). Its id must stay the original's, which the commit checks:
    that is what "near-copy" means, and what lets a reader update the row it
    already has rather than draw a second one.
    """
    if event.type not in REWRITTEN_IN_PLACE:
        raise SurfaceError(
            f'only an assistant reply or a tool result is rewritten in place, not "{event.type}"'
        )
    dropped = ("usage", "turn", "step") if event.type == "assistant/message" else ()
    # The small fields thawed one by one; the message is the caller's, already thawed.
    payload: dict[str, Any] = {
        key: thaw_json(value)
        for key, value in event.data.items()
        if key != "message" and key not in dropped
    }
    payload["message"] = message
    intent = SurfaceIntent(
        surface_op=SurfaceReplace(replaces=(event.seq,)), source_event_seqs=(event.seq,)
    )
    # Spelled per type rather than `event.type`: the writers table is read off the
    # literal, so the walk sees exactly which types this door writes.
    if event.type == "assistant/message":
        return _LOG.append(log, "assistant/message", payload, intent)
    return _LOG.append(log, "tool/result", payload, intent)


def editable_message(event: SessionEvent) -> dict[str, Any]:
    """`event`'s message as plain JSON, for a producer to change and hand to `rewrite`."""
    return cast("dict[str, Any]", thaw_json(event.data["message"]))


# ------------------------------------------------------------------ reading --


def origin_of(session: Session, seq: int) -> int:
    """The seq a node was first appended at, through any in-place rewrites of it.

    A rewrite is the same message, so it is the same node to anyone naming it; a
    substitution is a new one, and the walk stops there.
    """
    event = session.at(seq)
    while event is not None and is_in_place_rewrite(event):
        seq = shadowed_by(event)[0]
        event = session.at(seq)
    return seq


def originals(session: Session, seq: int) -> Iterator[Message]:
    """The messages a surface node stands for, back to what was first appended.

    A node that replaced nothing is its own original. A replacement is expanded
    through every node it shadowed, recursively — a revision of a revision reads
    back to the conversation it all started from.
    """
    event = session.at(seq)
    if event is None:
        return
    replaced = shadowed_by(event)
    if replaced:
        for shadowed in replaced:
            yield from originals(session, shadowed)
        return
    message = derive_event_message(event)
    if message is not None:
        yield message
