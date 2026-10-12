"""A log's revisions, read with nothing mounted: one reading of each, for `/diff` and for
training rows (`phern revisions`).

`/diff` (`ph_app.tui.diffs`) shows each revision to a person; the export writes the
same revisions as JSON lines, one per change, for diagnosis and training data. Both read
a revision here (`read_revision`), so a row and a screen agree about what a change was.

**A row is one change, whoever made it**, read through core alone, since the app may
not import a bundle to learn its types (`test_app_layering.py`): what it replaced and
what it is now, as `render_messages` renders both; the records its producer wrote
through the revision door (`Session.records_of`), whole; the size the door wrote; and
the tool calls those records name by `callId`. For a model's own context edit that is
the call that asked for it; for the overflow clip, the call whose result was clipped.

Textual-free, like `ph_app.stored`, so exporting a log does not load a terminal UI.

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
    producer: str
    """The plugin its message names as its author, `""` when it names none."""
    records: tuple[SessionEvent, ...]
    size: RevisionSize | None

    def texts(self) -> tuple[str, str]:
        """`(before, after)`: what it replaced, and itself, rendered on each call."""
        return (
            render_messages(messages_of(self.session, self.replaced)),
            render_messages(messages_of(self.session, (self.event.seq,))),
        )


def read_revision(session: Session, event: SessionEvent) -> Revision:
    """The replacement `event`, read once for every reader."""
    kind, plugin, _ = source_of(event.data)
    records = session.records_of(event.seq)
    return Revision(
        session=session,
        event=event,
        shape=shape_of(event),
        replaced=shadowed_by(event),
        producer=plugin if kind == "plugin" else "",
        records=records,
        size=recorded_size(records),
    )


def recorded_size(records: Sequence[SessionEvent]) -> RevisionSize | None:
    """The size a record of this change alone gave, as the door wrote it. One across
    them: two would be two claims about one change, and choosing between them is
    interpreting a producer. A record of several changes sizes none."""
    sizes = [
        size
        for record in records
        if len(revises_of(record)) == 1 and (size := RevisionSize.of(record)) is not None
    ]
    return sizes[0] if len(sizes) == 1 else None


def revision_rows(session: Session) -> Iterator[JsonObject]:
    """One row per revision in the log, in log order."""
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
            "producer": revision.producer or None,
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
    """The tool calls `named`, by id, as first written: read off the model's replies
    only while some are still to find, and not at all when none is named."""
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
