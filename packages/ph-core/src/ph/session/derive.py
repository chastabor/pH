"""The per-node projection rule: one event gives one message or none.

This is THE projection. `Session.derive_messages()` folds it over the live
surface. An external reconstructor folds the same function over the surface of
a stored log, to rebuild exactly the messages that any request was built from.
Two implementations would give two answers to "what did the model see", so
there is one.

`derive_transcript` is the *other* projection, the human one. The surface
shadows compacted ranges. That is right for the model and wrong for a person
who already saw the conversation, so the transcript keeps every append-origin
message. Both projections are here so that a consumer never chooses one by
accident.

Ported from dsh `deriveEventMessage` in `surface.ts`.

@module ph.session.derive
"""

from __future__ import annotations

from collections.abc import Iterable

from ..json import as_bool, as_obj, as_seq, as_str
from ..llm.types import Message
from .events import SessionEvent
from .surface import is_append_surface_event

__all__ = ["derive_event_message", "derive_transcript", "settle_of"]


def derive_event_message(event: SessionEvent) -> Message | None:
    """Project one event into the message that it derives to, or `None`.

    Injected context projects in user role with its content **verbatim**. The
    framing belongs to the producer: a plugin that wants `<system-reminder>`
    around its text puts it into `content` before it appends. If this function
    added framing, the log would no longer say what the model saw.

    Pydantic accepts the frozen payload directly, so nothing is copied before
    validation.
    """
    if event.type == "user/message":
        return Message.model_validate(event.data)
    if event.type == "assistant/message":
        message = as_obj(event.data.get("message"))
        # An `assistant/message` with empty content exists only to carry the
        # usage of a max-tokens step. Several providers reject a provider
        # transcript that contains an assistant turn with no content.
        if not message or not message.get("content"):
            return None
        return Message.model_validate(message)
    if event.type == "tool/result":
        return Message.model_validate(event.data["message"])
    # A non-surface event projects to no message. The event map is
    # merge-extensible, so this is deliberately non-exhaustive.
    return None


def derive_transcript(events: Iterable[SessionEvent]) -> tuple[Message, ...]:
    """Every append-origin message, in log order. This is the human transcript."""
    messages: list[Message] = []
    for event in events:
        if not is_append_surface_event(event):
            continue
        message = derive_event_message(event)
        if message is not None:
            messages.append(message)
    return tuple(messages)


def settle_of(record: SessionEvent) -> tuple[str, bool] | None:
    """The call that a settle answers, and whether the call failed.

    This works for either record that a call settles with. The result is `None`
    for any other record, and for a record that is too damaged to read.

    **This is the one reader of that shape.** Five readers each reached the
    call id by their own route, so a change to the shape of the result left
    some of them counting nothing, with no error. A `tool/result` is read at
    its `tool-result` block, which is the block that both
    `batch._append_result` and the closer of repair write. It is read from the
    wire and not through `derive_event_message`. Repair folds every record of a
    log on each resume, and a projection that validated each result cost 8 µs
    per result and raised on the damage that repair exists to read past. A Code
    Mode dispatch settles as `tool/code-dispatch`, which has a shape of its
    own.
    """
    if record.type == "tool/code-dispatch":
        return as_str(record.data.get("subCallId")), as_bool(record.data.get("isError"))
    if record.type != "tool/result":
        return None
    content = as_seq(as_obj(record.data.get("message")).get("content"))
    block = next((as_obj(one) for one in content if as_obj(one).get("type") == "tool-result"), None)
    return (
        None if block is None else (as_str(block.get("toolCallId")), as_bool(block.get("isError")))
    )
