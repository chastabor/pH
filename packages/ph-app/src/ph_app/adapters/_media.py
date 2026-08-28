"""Attachment loading and the degrade rule, once for every adapter (P7-01).

The sibling of `_http.py`, and here for the reason its docstring gives: written
twice, the copies drift. This half of media handling is not wire-shaped at all —
deciding whether a route can take an attachment, loading it, and saying so when
it cannot are the same decisions on every provider. What *is* wire-shaped is the
block a usable attachment becomes (`image` against `document`, `image_url`
against `input_audio`), and that stays in the adapter.

**The route's own declaration is the input.** `ResolvedModel.accepts` and
`max_attachment_bytes` are what an adapter already answers for a route, so
policy is read from the capability rather than restated beside it — otherwise
the declaration is decorative and the real rule lives in whichever adapter you
happen to read.

@module ph_app.adapters._media
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ph.llm.adapter import ResolvedModel
from ph.llm.types import AttachmentRef, Message, attachment_of

__all__ = ["load_media", "media_pointer", "unusable_reason"]

log = logging.getLogger("ph_app.adapters.media")


def unusable_reason(attachment: AttachmentRef, store: Any, route: ResolvedModel) -> str | None:
    """Why this attachment cannot be sent inline, or `None` if it can.

    Three unrelated situations answered with one branch on purpose — a MIME the
    route does not take, a file over its ceiling, and bytes that are no longer on
    disk all have the same remedy, because the model can act on a pointer and can
    do nothing with a wire error.
    """
    if store is None:
        return "no attachment store is mounted"
    if attachment.mime not in route.accepts:
        return f"this route does not accept {attachment.mime}"
    if route.max_attachment_bytes is not None and attachment.bytes > route.max_attachment_bytes:
        return f"{attachment.bytes} bytes is over the {route.max_attachment_bytes}-byte limit"
    if not store.exists(attachment):
        return "the stored bytes are gone"
    return None


def media_pointer(attachment: AttachmentRef) -> dict[str, Any]:
    """What the model is shown when the bytes cannot go.

    A sentence it can act on — the kind of file, its name, and that this route
    could not read it — rather than a message that quietly lost a block. Shared
    so the two wires cannot come to describe the same failure differently.
    """
    name = attachment.name or attachment.attachment_id
    return {
        "type": "text",
        "text": (
            f'[{attachment.mime} attachment "{name}" was not sent: this model cannot read it]'
        ),
    }


async def load_media(
    store: Any, messages: Sequence[Message], route: ResolvedModel, *, provider: str
) -> dict[str, str]:
    """Base64 for every attachment this route will actually take.

    Absent from the map means "render a pointer instead". Loaded in one pass
    before rendering so the message renderers stay pure functions of what they
    are given, and through `AttachmentStore.load_b64`, which encodes once per
    process — a media block lives in derived history for the session's life, so
    re-encoding per request is a cost paid on every step.
    """
    loaded: dict[str, str] = {}
    for message in messages:
        for block in message.content:
            attachment = attachment_of(block)
            if attachment is None or attachment.attachment_id in loaded:
                continue
            reason = unusable_reason(attachment, store, route)
            if reason is not None:
                # Not silent. A wire dropping media without a word is the bug
                # this whole path exists to end.
                log.warning(
                    "ph_app.adapters: %s is sending %s as a pointer: %s",
                    provider,
                    attachment.name or attachment.attachment_id,
                    reason,
                )
                continue
            loaded[attachment.attachment_id] = await store.load_b64(attachment)
    return loaded
