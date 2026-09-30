"""The per-node projection rule: one event, one message or none.

THE projection. `Session.derive_messages()` folds it over the live surface;
an external reconstructor folds the same function over a stored log's surface
to rebuild exactly the messages any request was built from. Two implementations
would be two answers to "what did the model see", so there is one.

`derive_transcript` is the *other* projection — the human one. The surface
shadows compacted ranges, which is right for the model and wrong for a person
who already saw the conversation; the transcript keeps every append-origin
message. Both live here so a consumer never has to choose by accident.

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
    """Project one event into the message it derives to, or `None`.

    Injected context projects in user role with its content **verbatim**. Framing
    is the producer's: a plugin that wants `<system-reminder>` around its text
    bakes it into `content` before appending. Re-adding framing here would mean
    the log no longer says what the model saw.

    Pydantic accepts the frozen payload directly, so nothing is copied before
    validation.
    """
    if event.type == "user/message":
        return Message.model_validate(event.data)
    if event.type == "assistant/message":
        message = as_obj(event.data.get("message"))
        # An empty-content assistant/message exists only to host a max-tokens
        # step's usage; injecting a content-less assistant turn into the
        # provider transcript is an error at several providers.
        if not message or not message.get("content"):
            return None
        return Message.model_validate(message)
    if event.type == "tool/result":
        return Message.model_validate(event.data["message"])
    # A non-surface event projects to no message. The event map is
    # merge-extensible, so this is deliberately non-exhaustive.
    return None


def derive_transcript(events: Iterable[SessionEvent]) -> tuple[Message, ...]:
    """Every append-origin message, in log order — the human transcript."""
    messages: list[Message] = []
    for event in events:
        if not is_append_surface_event(event):
            continue
        message = derive_event_message(event)
        if message is not None:
            messages.append(message)
    return tuple(messages)


def settle_of(record: SessionEvent) -> tuple[str, bool] | None:
    """The call a settle answers, and whether it failed — for either record a call
    settles with, and `None` for any other record, or one too damaged to say.

    **The one reader of that shape.** Five readers each reached the call id by a
    route of their own, so a change to the result's shape left some of them counting
    nothing, with no error. A `tool/result` is read at its `tool-result` block, the
    one `batch._append_result` and repair's closer both write, off the wire rather
    than through `derive_event_message`: repair folds every record of a log on each
    resume, and a projection that validated each result paid 8 µs apiece and raised
    on the damage repair exists to read past. A Code Mode dispatch settles as
    `tool/code-dispatch`, a shape of its own.
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
