"""The one place that revises the surface, and the one reading of what a revision stood for.

Every producer that changes what the model sees after the fact appends a surface
`replace`. The producers are the summary of compaction, its argument elision and
its overflow clip, the relocation of a large tool result, the paste preview of
input-offload, and a context edit by the model itself. A replacement must keep
the same rules for all of them. So the rules are kept here, one time, and tested
here, one time. The module has two kinds of revision function:

* **`substitute`**: text that stands in for a run of nodes. It is always a
  `user/message` that carries a message of its own (a fresh id) and cites every
  node that it shadows.
* **`rewrite`**: one node replaced by a near-copy of itself. Only an
  `assistant/message` or a `tool/result` can be rewritten. The near-copy keeps
  the message id and every field except the message. There is one exception: an
  assistant reply loses `usage`, `turn` and `step` (D8, below).

The commit holds the part of these rules that every reader relies on: a
replacement in the assistant role or the tool-result role rewrites exactly one
node of its own type and keeps its id (`surface._assert_assistant_rewrite`,
`_assert_tool_result_rewrite`). That makes `is_in_place_rewrite` exact by type,
for the writes of this module and for any other write. The revision functions
add what a producer would otherwise have to remember: the payload built from the
original, the fields that D8 leaves behind, and the citation. The writers walk
(`test_log_writers.py`) checks that no other shipped module constructs a
`SurfaceReplace`. So this module stays the one place where those rules are
remembered.

**Every revision carries the record of its producer that says why**, and the
revision function writes it. Examples are `compaction/summarized`,
`offload/input-spilled` and `clm/revised`. The producer passes the record in as
a `RevisionRecord` with its own writer, because the type belongs to the
producer. The revision function writes the record immediately after the
replacement, into the log *or batch* of the caller. It stamps the record with
`revises: [seq, ...]`, which names the replacements that the record explains.
The writers keep that payload key for this module alone (`writers.REVISES`). So
`Session.records_of(seq)` answers "why is this node here" by lookup, in the same
way for every producer. Readers formerly inferred the pairing from the position
of a record in its batch. A record for several replacements at one time goes
through `rewrite_many`, which names them all. Argument elision writes one such
record for all the arguments that it cut. A record may carry the size of the
revision (`RevisionSize`), which the revision function writes as its `size`:
tokens where the producer counts them, and bytes where it moves them.

**D8: why a rewritten assistant reply drops `usage`, `turn` and `step`.** The
replacement is appended at the tail. `TokenMeter.last_usage` folds to the newest
`assistant/message` that carries usage. So a copied usage would become the
baseline of the meter and would say that the session had shrunk. A copied
`turn`/`step` would announce a finished step as the newest thing in the log.
Every reader keyed on "the latest assistant message" would then read a closed
step as the open one. Both belong to the original, which still has them. And
`replaces` names the original, which is a stronger link than a pair of integers
that a reader must match up.

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
"""The payload key under which a revision function writes the `RevisionSize` of a record."""

SizeUnit: TypeAlias = Literal["tokens", "bytes", "chars"]
"""Tokens where a producer counts them, and bytes where it moves them. Characters
where a reader that has no meter counts them itself."""

RevisionShape: TypeAlias = Literal["substitution", "stand-in", "rewrite"]
"""How a revision changed the surface:

- A **substitution** stands for a run of nodes with text of its own.
- A **stand-in** is the substitution that claims that the run left the surface
  as condensed conversation (`is_stand_in`): a summary, or a revision by the
  model.
- A **rewrite** replaces one node with a near-copy of itself
  (`is_in_place_rewrite`)."""


def shape_of(event: SessionEvent) -> RevisionShape:
    """How the replacement `event` changed the surface. `/diff`, the revision
    export and the `context_diff` of ph-clm all use this one answer."""
    if is_in_place_rewrite(event):
        return "rewrite"
    return "stand-in" if is_stand_in(event) else "substitution"


@dataclass(frozen=True, slots=True)
class RevisionSize:
    """The size of what a revision replaced, and of what it put there, in `unit`.
    For a record of several replacements, the sizes cover all of them together."""

    unit: SizeUnit
    before: int
    after: int

    def to_wire(self) -> dict[str, JsonValue]:
        return {"unit": self.unit, "before": self.before, "after": self.after}

    def label(self) -> str:
        return f"{self.before:,}→{self.after:,} {self.unit}"

    @classmethod
    def of(cls, record: SessionEvent) -> RevisionSize | None:
        """The size that a revision function wrote on `record`. `None` when there
        is no size, or when the unit is one that this build does not know."""
        size = as_obj(record.data.get(SIZE))
        unit = as_str(size.get("unit"))
        if unit not in ("tokens", "bytes", "chars"):
            return None
        return cls(cast("SizeUnit", unit), as_int(size.get("before")), as_int(size.get("after")))


def producer_data(record: SessionEvent) -> dict[str, JsonValue]:
    """What the producer of a record wrote, without the keys that the revision
    function stamped on it (`revises`, `size`). These are the own fields of the
    record, as a reader lists them."""
    return {key: value for key, value in record.data.items() if key not in (REVISES, SIZE)}


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    """The record of a producer that says why it revised the surface, for a
    revision function to write.

    `writer` is the writer of the producer, and it owns `type`. The revision
    function adds `revises`, and adds `size` when there is one.
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
    """`log` as a batch, so that a replacement and its record land together or
    not at all (P10-14). The batch is the batch of the caller, or one that is
    opened here for a caller that wrote nothing else beside them."""
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
    """Append `message` in place of the nodes that `shadowed` names, then append
    `record`. Returns the replacement.

    The surface puts the replacement where the earliest of the nodes was and
    takes the others out. The log keeps every one of them (I4). `message` is a
    user message of its own, such as a summary, a preview or a revision. It is
    never a message that the shadowed nodes already carry.
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
    """Append a near-copy of `event` that carries `message`, in place of `event`,
    then append `record`. Returns the replacement.

    `message` is the wire form of the replacement message: the message of the
    original, changed (`editable_message`). Its id must stay the id of the
    original, and the commit checks that. That is the meaning of "near-copy",
    and it lets a reader update the row that it already has and not draw a
    second one.
    """
    return rewrite_many(log, ((event, message),), record)[0]


def rewrite_many(
    log: Session | SessionBatch,
    rewrites: Sequence[tuple[SessionEvent, Mapping[str, JsonValue]]],
    record: RevisionRecord,
) -> tuple[SessionEvent, ...]:
    """`rewrite` for each `(event, message)`, then one `record` that names every
    replacement. It is for a producer whose one record explains them all, as the
    record of argument elision does."""
    if not rewrites:
        raise ValueError("a record of rewrites names at least one")
    with _together(log) as batch:
        replacements = tuple(_rewritten(batch, event, message) for event, message in rewrites)
        record._write(batch, tuple(one.seq for one in replacements))
    return replacements


def _rewritten(
    log: SessionBatch, event: SessionEvent, message: Mapping[str, JsonValue]
) -> SessionEvent:
    """The near-copy itself, with no record. `rewrite_many` writes the record."""
    if event.type not in REWRITTEN_IN_PLACE:
        raise SurfaceError(
            f'only an assistant reply or a tool result is rewritten in place, not "{event.type}"'
        )
    dropped = ("usage", "turn", "step") if event.type == "assistant/message" else ()
    # The small fields are thawed one by one. The message comes from the caller,
    # already thawed.
    payload: dict[str, Any] = {
        key: thaw_json(value)
        for key, value in event.data.items()
        if key != "message" and key not in dropped
    }
    payload["message"] = message
    intent = SurfaceIntent(
        surface_op=SurfaceReplace(replaces=(event.seq,)), source_event_seqs=(event.seq,)
    )
    # The type is spelled per branch and not passed as `event.type`. The writers
    # table is read from the literal, so the walk sees exactly which types this
    # module writes.
    if event.type == "assistant/message":
        return _LOG.append(log, "assistant/message", payload, intent)
    return _LOG.append(log, "tool/result", payload, intent)


def editable_message(event: SessionEvent) -> dict[str, Any]:
    """The message of `event` as plain JSON, for a producer to change and pass to `rewrite`."""
    return cast("dict[str, Any]", thaw_json(event.data["message"]))


# ------------------------------------------------------------------ reading --


def origin_of(session: Session, seq: int) -> int:
    """The seq at which a node was first appended, through any in-place rewrites of it.

    A rewrite is the same message, so it is the same node to anyone that names
    it. A substitution is a new node, and the walk stops there.
    """
    event = session.at(seq)
    while event is not None and is_in_place_rewrite(event):
        seq = shadowed_by(event)[0]
        event = session.at(seq)
    return seq


def originals(session: Session, seq: int) -> Iterator[Message]:
    """The messages that a surface node stands for, back to what was first appended.

    A node that replaced nothing is its own original. A replacement is expanded
    through every node that it shadowed, recursively. So a revision of a
    revision reads back to the conversation that it all started from.
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
    """The messages that the events at `seqs` carry, in that order. An event that
    carries no message is skipped, as an empty assistant turn is. The two sides of
    a revision are made of these messages, before `render_messages` reads them."""
    for seq in seqs:
        event = session.at(seq)
        if event is not None and (message := derive_event_message(event)) is not None:
            yield message


def unified_diff(before: str, after: str, labels: tuple[str, str]) -> tuple[str, ...]:
    """Two renderings as a unified diff, every line of it, under `labels`. `/diff`
    and the `context_diff` of ph-clm both draw this one diff. A reader that shows
    the diff limits its length."""
    return tuple(
        difflib.unified_diff(before.splitlines(), after.splitlines(), *labels, lineterm="")
    )


def render_messages(messages: Iterable[Message]) -> str:
    """Messages as plain text, one block per message with the role as its heading.
    A revision and what it stands for are read side by side in this form.

    There is one rendering for every reader that compares a revision with what
    it replaced, so each reader shows it the same way. Calls keep their name and
    arguments, and results keep their text. Reasoning and media become markers.
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
