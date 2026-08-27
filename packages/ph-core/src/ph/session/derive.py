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

from ..llm.types import Message
from .events import SessionEvent
from .surface import is_append_surface_event

__all__ = ["derive_event_message", "derive_transcript"]


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
        message = event.data.get("message")
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
