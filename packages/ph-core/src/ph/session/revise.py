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

**Every revision carries its producer's record of why**, and the door writes it:
`compaction/summarized`, `offload/input-spilled`, `clm/revised`, handed in as a
`RevisionRecord` with the producer's own writer, since the type is the producer's. The
door writes it right after the replacement, into the caller's log *or batch*, stamped
`revises: [seq, ...]` with the replacements it explains, a payload key the writers keep
for this door alone (`writers.REVISES`). So `Session.records_of(seq)` answers "why is
this node here" by lookup, for every producer alike, where readers used to infer the
pairing from where a record sat in its batch. A record for several replacements at once,
as argument elision writes one for every argument it cut, goes through `rewrite_many`,
which names them all. A record may carry the revision's size (`RevisionSize`), which the
door writes as its `size`: tokens where the producer counts them, bytes where it moves
them.

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

import difflib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, assert_never, cast

from ..json import JsonValue, as_int, as_obj, as_str, thaw_json
from ..llm.types import (
    MediaBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    text_of,
)
from ..text import block_marker
from .derive import derive_event_message
from .events import SessionEvent, SurfaceIntent, SurfaceReplace
from .revision_index import RevisionIndex, revises_of
from .session import Session, SessionBatch
from .surface import (
    REWRITTEN_IN_PLACE,
    SurfaceError,
    is_in_place_rewrite,
    is_stand_in,
    shadowed_by,
)
from .writers import REVISES, LogWriter, log_writer

_LOG = log_writer(__name__)

__all__ = [
    "RevisionIndex",
    "RevisionRecord",
    "RevisionShape",
    "RevisionSize",
    "editable_message",
    "messages_of",
    "origin_of",
    "originals",
    "producer_data",
    "render_messages",
    "revises_of",
    "rewrite",
    "rewrite_many",
    "shape_of",
    "substitute",
    "unified_diff",
]


SIZE = "size"
"""The payload key the door writes a record's `RevisionSize` under."""

SizeUnit: TypeAlias = Literal["tokens", "bytes", "chars"]
"""Tokens where a producer counts them, bytes where it moves them; characters where a
reader that has no meter counts them itself."""

RevisionShape: TypeAlias = Literal["substitution", "stand-in", "rewrite"]
"""How a revision changed the surface: a **substitution** stands for a run of nodes
with text of its own, and a **stand-in** is the substitution that claims the run left
the surface as condensed conversation (`is_stand_in`), a summary or a model's
revision; a **rewrite** replaces one node with a near-copy of itself
(`is_in_place_rewrite`)."""


def shape_of(event: SessionEvent) -> RevisionShape:
    """How the replacement `event` changed the surface: one answer for `/diff`, the
    revision export and ph-clm's `context_diff`."""
    if is_in_place_rewrite(event):
        return "rewrite"
    return "stand-in" if is_stand_in(event) else "substitution"


@dataclass(frozen=True, slots=True)
class RevisionSize:
    """How big what a revision replaced was, and what it put there, in `unit`. For a
    record of several replacements, all of them together."""

    unit: SizeUnit
    before: int
    after: int

    def to_wire(self) -> dict[str, JsonValue]:
        return {"unit": self.unit, "before": self.before, "after": self.after}

    def label(self) -> str:
        return f"{self.before:,}→{self.after:,} {self.unit}"

    @classmethod
    def of(cls, record: SessionEvent) -> RevisionSize | None:
        """The size the door wrote on `record`, or `None` for none, or for a unit this
        build does not know."""
        size = as_obj(record.data.get(SIZE))
        unit = as_str(size.get("unit"))
        if unit not in ("tokens", "bytes", "chars"):
            return None
        return cls(cast("SizeUnit", unit), as_int(size.get("before")), as_int(size.get("after")))


def producer_data(record: SessionEvent) -> dict[str, JsonValue]:
    """What a record's producer wrote, without what the door stamped on it (`revises`,
    `size`): a record's own fields, as a reader lists them."""
    return {key: value for key, value in record.data.items() if key not in (REVISES, SIZE)}


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    """A producer's record of why it revised the surface, for the door to write.

    `writer` is the producer's own, which owns `type`; the door adds `revises`, and
    `size` when there is one.
    """

    writer: LogWriter
    type: str
    data: Mapping[str, JsonValue]
    size: RevisionSize | None = None

    def __post_init__(self) -> None:
        if SIZE in self.data:
            raise ValueError(f'"{SIZE}" is the door\'s to write: hand it over as `size`')

    def _write(self, log: SessionBatch, revises: Sequence[int]) -> SessionEvent:
        data = self.data if self.size is None else {**self.data, SIZE: self.size.to_wire()}
        return self.writer._append_revising(log, self.type, data, revises)


@contextmanager
def _together(log: Session | SessionBatch) -> Iterator[SessionBatch]:
    """`log` as a batch, so a replacement and its record land together or not at all
    (P10-14): the caller's own, or one opened here for a caller that wrote nothing
    else beside them."""
    if isinstance(log, SessionBatch):
        yield log
        return
    with log.batch() as batch:
        yield batch


def substitute(
    log: Session | SessionBatch,
    shadowed: Sequence[int],
    message: Message,
    record: RevisionRecord,
) -> SessionEvent:
    """Append `message` in place of the nodes `shadowed` names, then `record`; the
    replacement is returned.

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
    with _together(log) as batch:
        replacement = _LOG.append(
            batch,
            "user/message",
            message.to_wire(),
            SurfaceIntent(surface_op=SurfaceReplace(replaces=seqs), source_event_seqs=seqs),
        )
        record._write(batch, (replacement.seq,))
    return replacement


def rewrite(
    log: Session | SessionBatch,
    event: SessionEvent,
    message: Mapping[str, JsonValue],
    record: RevisionRecord,
) -> SessionEvent:
    """Append a near-copy of `event` carrying `message`, in place of it, then `record`;
    the replacement is returned.

    `message` is the replacement message's wire form — the original's, changed
    (`editable_message`). Its id must stay the original's, which the commit checks:
    that is what "near-copy" means, and what lets a reader update the row it
    already has rather than draw a second one.
    """
    return rewrite_many(log, ((event, message),), record)[0]


def rewrite_many(
    log: Session | SessionBatch,
    rewrites: Sequence[tuple[SessionEvent, Mapping[str, JsonValue]]],
    record: RevisionRecord,
) -> tuple[SessionEvent, ...]:
    """`rewrite` for each `(event, message)`, then one `record` naming every replacement:
    for a producer whose one record explains them all, as argument elision's does."""
    if not rewrites:
        raise ValueError("a record of rewrites names at least one")
    with _together(log) as batch:
        replacements = tuple(_rewritten(batch, event, message) for event, message in rewrites)
        record._write(batch, tuple(one.seq for one in replacements))
    return replacements


def _rewritten(
    log: SessionBatch, event: SessionEvent, message: Mapping[str, JsonValue]
) -> SessionEvent:
    """The near-copy itself, with no record: `rewrite_many`'s half."""
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


def messages_of(session: Session, seqs: Iterable[int]) -> Iterator[Message]:
    """The messages the events at `seqs` carry, in that order — skipping an event that
    carries none, as an empty assistant turn does. What a revision's two sides are made
    of, before `render_messages` reads them."""
    for seq in seqs:
        event = session.at(seq)
        if event is not None and (message := derive_event_message(event)) is not None:
            yield message


def unified_diff(before: str, after: str, labels: tuple[str, str]) -> tuple[str, ...]:
    """Two renderings as a unified diff, every line of it, under `labels`: the one diff
    `/diff` and ph-clm's `context_diff` draw. A reader that shows it bounds it."""
    return tuple(
        difflib.unified_diff(before.splitlines(), after.splitlines(), *labels, lineterm="")
    )


def render_messages(messages: Iterable[Message]) -> str:
    """Messages as plain text, one role-headed block each: how a revision and what it
    stands for read side by side.

    One rendering for every reader that compares a revision with what it replaced, so
    each shows it the same way. Calls keep their name and arguments and results their
    text; reasoning and media are markers.
    """
    blocks: list[str] = []
    for message in messages:
        lines: list[str] = []
        for block in message.content:
            match block:
                case TextBlock():
                    lines.append(block.text)
                case ToolCallBlock():
                    lines.append(f"[call {block.name} {block.arguments}]")
                case ToolResultBlock():
                    lines.append(f"[result {block.tool_call_id}]")
                    lines.append(text_of(block.content, placeholder=block_marker))
                case ReasoningBlock() | MediaBlock():
                    lines.append(block_marker(block.type))
                case _ as unhandled:
                    assert_never(unhandled)
        blocks.append(f"[{message.role}]\n" + "\n".join(lines))
    return "\n\n".join(blocks)
