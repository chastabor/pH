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
from collections.abc import Collection, Sequence
from typing import Any

from ph.llm.media import media_pointer_text
from ph.llm.types import Message, attachment_of

__all__ = ["load_handles", "load_media", "media_pointer"]

log = logging.getLogger("ph_app.adapters.media")


def media_pointer(attachment: Any) -> dict[str, Any]:
    """The text block that stands in for media that could not be loaded.

    Reached only on a race — `media-degrade` already checked the blob was there,
    so arriving here means it went away between that check and this read. The
    wording is `ph.llm.media`'s, so the model reads one sentence for one
    situation however it was arrived at.
    """
    return {"type": "text", "text": media_pointer_text(attachment)}


async def load_handles(
    uploads: Any,
    messages: Sequence[Message],
    *,
    provider: str,
    mimes: frozenset[str],
    session: Any = None,
) -> dict[str, str]:
    """Provider file ids for the attachments this route references rather than inlines.

    Keyed by attachment id like `load_media`, and consulted first by the
    renderer: an id present here is sent as a reference, absent as bytes. The
    two maps rather than one union type because the fallback has to be silent and
    total — an upload that fails for any reason leaves the id out and the
    attachment goes inline, which is what every route did before this row.

    `mimes` is the route's own list rather than "everything large": which formats
    are worth a round trip is a fact about a provider's file API, and video is
    the case where it is not a choice.
    """
    handles: dict[str, str] = {}
    if uploads is None or not mimes:
        return handles
    for message in messages:
        for block in message.content:
            attachment = attachment_of(block)
            if attachment is None or attachment.attachment_id in handles:
                continue
            if attachment.mime not in mimes:
                continue
            try:
                handle = await uploads.handle_for(attachment, provider=provider, session=session)
            except Exception:
                # Deliberately broad and deliberately not fatal: an upload is an
                # optimisation, and a route that can take the bytes inline must
                # not lose a turn because a file API was down.
                log.warning(
                    "ph_app.adapters: could not upload %s to %s; sending it inline",
                    attachment.name or attachment.attachment_id,
                    provider,
                    exc_info=True,
                )
                continue
            if handle is not None:
                handles[attachment.attachment_id] = handle.handle
    return handles


async def load_media(
    store: Any, messages: Sequence[Message], *, skip: Collection[str] = ()
) -> dict[str, str]:
    """Base64 for every attachment still on the request, keyed by id.

    Absent from the map means the read failed after `media-degrade` had approved
    it, which is a race rather than a policy outcome — the caller renders a
    pointer either way, so one branch covers both.

    Through `AttachmentStore.load_b64`, which encodes once per process: a media
    block lives in derived history for the session's life, so re-encoding per
    request is a cost paid on every step.

    **`skip` is what makes an upload actually cheaper** (P7-03). Without it a
    referenced file was still read and base64-encoded here and then discarded by
    the renderer — so the wire payload shrank and nothing else did, while a
    5.5 MB string sat in the store's encode cache for the life of the process.
    Uploading is supposed to remove that work, not move it.
    """
    loaded: dict[str, str] = {}
    if store is None:
        return loaded
    for message in messages:
        for block in message.content:
            attachment = attachment_of(block)
            if attachment is None or attachment.attachment_id in loaded:
                continue
            if attachment.attachment_id in skip:
                continue
            try:
                loaded[attachment.attachment_id] = await store.load_b64(attachment)
            except OSError:
                log.warning(
                    "ph_app.adapters: could not read %s; sending a pointer",
                    attachment.name or attachment.attachment_id,
                )
    return loaded
