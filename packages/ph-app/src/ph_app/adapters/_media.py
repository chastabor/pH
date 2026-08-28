"""Loading the bytes for media that is going out (P7-01).

The sibling of `_http.py`, and all that is left here once `ph.llm.media` answers
the *policy* question above every adapter: by the time a request reaches one, any
`MediaBlock` still on it is one this route said it accepts, so there is nothing
left to decide — only bytes to fetch and a wire shape to build, and the shape is
the adapter's own.

@module ph_app.adapters._media
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ph.llm.media import media_pointer_text
from ph.llm.types import Message, attachment_of

__all__ = ["load_media", "media_pointer"]

log = logging.getLogger("ph_app.adapters.media")


def media_pointer(attachment: Any) -> dict[str, Any]:
    """The text block that stands in for media that could not be loaded.

    Reached only on a race — `media-degrade` already checked the blob was there,
    so arriving here means it went away between that check and this read. The
    wording is `ph.llm.media`'s, so the model reads one sentence for one
    situation however it was arrived at.
    """
    return {"type": "text", "text": media_pointer_text(attachment)}


async def load_media(store: Any, messages: Sequence[Message]) -> dict[str, str]:
    """Base64 for every attachment still on the request, keyed by id.

    Absent from the map means the read failed after `media-degrade` had approved
    it, which is a race rather than a policy outcome — the caller renders a
    pointer either way, so one branch covers both.

    Through `AttachmentStore.load_b64`, which encodes once per process: a media
    block lives in derived history for the session's life, so re-encoding per
    request is a cost paid on every step.
    """
    loaded: dict[str, str] = {}
    if store is None:
        return loaded
    for message in messages:
        for block in message.content:
            attachment = attachment_of(block)
            if attachment is None or attachment.attachment_id in loaded:
                continue
            try:
                loaded[attachment.attachment_id] = await store.load_b64(attachment)
            except OSError:
                log.warning(
                    "ph_app.adapters: could not read %s; sending a pointer",
                    attachment.name or attachment.attachment_id,
                )
    return loaded
