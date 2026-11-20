"""The revisions of a log, read with nothing mounted: one reading of each, for `/diff`
and for training rows (`phern revisions`).

`/diff` (`ph_app.tui.diffs`) shows each revision to the user. The export writes
the same revisions as JSON lines, one line for each change, for diagnosis and
training data. Both read a revision here (`read_revision`), so a row and a
screen agree about what a change was.

**A row is one change, whoever made it.** It is read through core alone, because
the app may not import a bundle to learn its types (`test_app_layering.py`). A
row holds:

- what the change replaced and what it is now, as `render_messages` renders
  both;
- the records that its producer wrote through the revision door
  (`Session.records_of`), whole;
- the size that the door wrote;
- the tool calls that those records name by `callId`. For a context edit by the
  model itself, that is the call that asked for the edit. For the overflow clip,
  it is the call whose result was clipped.

This module does not import Textual, like `ph_app.stored`, so an export of a log
does not load a terminal UI.

@module ph_app.revisions
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from ph.json import JsonObject, as_str, thaw_json
from ph.llm.types import ToolCallBlock
from ph.session import (
    RevisionShape,
    RevisionSize,
    Session,
    SessionEvent,
    derive_event_message,
    is_replacement_surface_event,
    messages_of,
    render_messages,
    revises_of,
    shadowed_by,
    shape_of,
)

from .wire import source_of

__all__ = ["Revision", "read_revision", "recorded_size", "revision_rows"]


@dataclass(frozen=True, slots=True)
class Revision:
    """One revision as every reader here sees it."""

    session: Session
    event: SessionEvent
    shape: RevisionShape
    replaced: tuple[int, ...]
    producer: str | None
    """The plugin that its message names as its author, `None` when it names none."""
    records: tuple[SessionEvent, ...]
    size: RevisionSize | None

    def texts(self) -> tuple[str, str]:
        """`(before, after)`: what it replaced, and itself, rendered on each call."""
        return (
            render_messages(messages_of(self.session, self.replaced)),
            render_messages(messages_of(self.session, (self.event.seq,))),
        )


def read_revision(session: Session, event: SessionEvent) -> Revision:
    """The replacement `event`, read one time for every reader."""
    kind, plugin, _ = source_of(event.data)
    records = session.records_of(event.seq)
    return Revision(
        session=session,
        event=event,
        shape=shape_of(event),
        replaced=shadowed_by(event),
        producer=plugin if kind == "plugin" else None,
        records=records,
        size=recorded_size(records),
    )


def recorded_size(records: Sequence[SessionEvent]) -> RevisionSize | None:
    """The size that a record of this change alone gave, as the door wrote it.

    Exactly one such size across the records is accepted. Two would be two
    claims about one change, and to choose between them is to interpret a
    producer. A record of several changes gives no size."""
    sizes = [
        size
        for record in records
        if len(revises_of(record)) == 1 and (size := RevisionSize.of(record)) is not None
    ]
    return sizes[0] if len(sizes) == 1 else None


def revision_rows(session: Session) -> Iterator[JsonObject]:
    """One row for each revision in the log, in log order."""
    revisions = [
        read_revision(session, event)
        for event in session.events
        if is_replacement_surface_event(event)
    ]
    named = {
        call_id
        for revision in revisions
        for record in revision.records
        if (call_id := as_str(record.data.get("callId")))
    }
    calls = _calls(session, named)
    for revision in revisions:
        event = revision.event
        before, after = revision.texts()
        wanted = dict.fromkeys(as_str(record.data.get("callId")) for record in revision.records)
        yield {
            "session": session.id,
            "seq": event.seq,
            "time": event.time,
            "shape": revision.shape,
            "replaces": list(revision.replaced),
            "producer": revision.producer,
            "before": before,
            "after": after,
            "records": [
                {"type": record.type, "seq": record.seq, "data": thaw_json(record.data)}
                for record in revision.records
            ],
            "size": None if revision.size is None else revision.size.to_wire(),
            "calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call_id in wanted
                if (call := calls.get(call_id)) is not None
            ],
        }


def _calls(session: Session, named: set[str]) -> dict[str, ToolCallBlock]:
    """The tool calls `named`, by id, as first written. They are read off the
    replies of the model only while some are still to find, and not at all when
    none is named."""
    found: dict[str, ToolCallBlock] = {}
    for event in session.events if named else ():
        if event.type != "assistant/message" or is_replacement_surface_event(event):
            continue
        message = derive_event_message(event)
        for block in message.content if message is not None else ():
            if isinstance(block, ToolCallBlock) and block.id in named:
                found.setdefault(block.id, block)
        if len(found) == len(named):
            break
    return found
