"""Loading the bytes for media that is going out (P7-01).

This module is the sibling of `_http.py`. `ph.llm.media` answers the *policy*
question above every adapter, and this module holds what is left. When a request
reaches an adapter, any `MediaBlock` still on it is one that the route said it
accepts. One decision is left, for a file of a type that the route uploads: it
goes by reference, or inline when the upload gave no file and it fits under the
inline limit, or as a pointer (`load_media`). Apart from that, the adapter only
fetches bytes and builds a wire shape, and the shape belongs to the adapter.

The module also holds the *upload* half of that work. It was added when a second
wire got an upload API (P7-03). The docstring of `_http` gives the argument: the
adapters differ in message shape and in how usage is reported, and they do
**not** differ in what a dead file handle means. So `forget_named_handle` is
here, and is not a third copy of twelve lines whose bug would be invisible.

The route projection (`MediaRoute`, `resolved`) is *not* here, although the three
adapters share it too. Its content is the field list of `ResolvedModel`, so it
belongs beside that dataclass in `ph.llm` and not in an application package.
Everything in this module needs `ctx.uploads`, the attachment store, or the
meaning of `FILE_EXPIRED`. The route projection needs none of them.

@module ph_app.adapters._media
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence

from ph.cordis import Context
from ph.keys import UPLOADS
from ph.llm.adapter import LlmError
from ph.llm.media import media_pointer_text
from ph.llm.types import FILE_EXPIRED, AttachmentRef, Message, attachment_of
from ph.seams.attachments import AttachmentStore
from ph.seams.uploads import UploadRegistry

__all__ = [
    "forget_named_handle",
    "load_handles",
    "load_media",
    "media_pointer",
    "referenced_mimes",
]

log = logging.getLogger("ph_app.adapters.media")


def forget_named_handle(
    ctx: Context, error: LlmError, referenced: Sequence[str], *, provider: str
) -> LlmError:
    """Drop the one handle that this failure named, or leave the failure alone (P7-03).

    `_http` has already decided that the provider is talking about a missing
    file. Only an adapter knows whether it is a file that *we* sent. A `not_found`
    that names no id of ours is the 404 of something else, such as a stale route
    or a gateway. To retry it would be the "unknown failure billed twice" that the
    retry policy exists to refuse. So a `FILE_EXPIRED` failure that names no
    handle of this request is returned as `REQUEST_FAILED`.

    **The named handle, not every handle.** A first draft invalidated all of them
    on a match. On a request that carried twenty files, that discarded nineteen
    live uploads to replace one dead upload.

    This function is shared and not copied, because the *classification* is
    already shared. `failure_from_status` takes the phrases of each wire and
    answers one code, and this function is what a caller does with that code. A
    second adapter that wrote its own version would be free to disagree about
    which half of the check matters. The half that matters is the one that keeps
    `FILE_EXPIRED` out of an infinite retry.
    """
    uploads = ctx.get(UPLOADS)
    message = str(error.failure.message)
    named = [handle for handle in referenced if handle in message]
    if uploads is None or not named:
        return (
            LlmError(
                message,
                "REQUEST_FAILED",
                error.failure.model_copy(update={"code": "REQUEST_FAILED"}),
            )
            if error.code == FILE_EXPIRED
            else error
        )
    for handle in named:
        uploads.invalidate_handle(provider, handle)
    return error


def media_pointer(attachment: AttachmentRef) -> dict[str, str]:
    """The text block that stands in for media that could not be loaded.

    There are two ways to reach it. One is a race: `media-degrade` already
    checked that the blob was there, and the blob went away between that check
    and this read. The other is a file of an uploaded type that has no handle
    and is too large to go inline (`load_media`). The wording comes from
    `ph.llm.media`, so the model reads one sentence for one situation, whichever
    path led to it.
    """
    return {"type": "text", "text": media_pointer_text(attachment)}


def referenced_mimes(ctx: Context, mimes: Collection[str]) -> frozenset[str]:
    """The MIME types that an adapter will send by reference: `mimes`, or nothing
    when no upload service is mounted.

    An adapter reports this set as `ResolvedModel.uploads`, and `media-degrade`
    then leaves a large file of such a type to the adapter. The set must
    therefore be what `load_handles` will try, and not what the config lists. A
    route whose deployment mounts no `ctx.uploads` uploads nothing, and its
    large files must still degrade before the request.
    """
    return frozenset(mimes) if ctx.get(UPLOADS) is not None else frozenset()


async def load_handles(
    uploads: UploadRegistry | None,
    messages: Sequence[Message],
    *,
    provider: str,
    mimes: frozenset[str],
    session_id: str | None = None,
) -> dict[str, str]:
    """Provider file ids for the attachments that this route references and does not inline.

    The map is keyed by attachment id, like the map of `load_media`. The renderer
    consults it first: an id that is present is sent as a reference, and an id
    that is absent is sent as bytes. There are two maps and not one union type,
    because the fallback must be silent and total. An upload that fails for any
    reason leaves the id out, and the attachment goes inline, which is what every
    route did before this row.

    `mimes` is the list of the route itself and not "everything large". Which
    formats are worth a round trip is a fact about the file API of a provider,
    and video is the case where it is not a choice.
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
                handle = await uploads.handle_for(
                    attachment, provider=provider, session_id=session_id
                )
            except Exception:
                # Deliberately broad and deliberately not fatal. An upload is an
                # optimization, and a route that can take the bytes inline must
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
    store: AttachmentStore | None,
    messages: Sequence[Message],
    *,
    skip: Collection[str] = (),
    inline_limit: int | None,
) -> dict[str, str]:
    """Base64 for every attachment that goes inline, keyed by id.

    An id that is absent from the map is rendered as a pointer by the caller.
    There are two ways to be absent. The read failed after `media-degrade` had
    approved the block, which is a race and not a policy outcome. Or the file
    is over `inline_limit` and has no uploaded handle.

    **`inline_limit` is the `max_attachment_bytes` of the route.** `media-degrade`
    does not hold a type that the route uploads to that ceiling, because the
    file goes by reference. When the upload gives no handle (the file API
    failed, or the file is not ready yet), the fallback is inline, and a file
    over the ceiling cannot go inline: the provider refuses the whole request.
    So that file is left out, and the model reads the pointer. A later step
    tries the upload again.

    The read goes through `AttachmentStore.load_b64`, which encodes one time for
    each process. A media block stays in derived history for the life of the
    session, so encoding again for each request is a cost paid on every step.

    **`skip` is what makes an upload cheaper in fact** (P7-03). Without it, a
    referenced file was still read and base64-encoded here, and then the renderer
    discarded it. The wire payload shrank and nothing else did, and a 5.5 MB
    string stayed in the encode cache of the store for the life of the process.
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
            if inline_limit is not None and attachment.bytes > inline_limit:
                log.warning(
                    "ph_app.adapters: %s has no uploaded file and its %d bytes are over "
                    "the %d-byte inline limit; sending a pointer",
                    attachment.name or attachment.attachment_id,
                    attachment.bytes,
                    inline_limit,
                )
                continue
            try:
                loaded[attachment.attachment_id] = await store.load_b64(attachment)
            except OSError:
                log.warning(
                    "ph_app.adapters: could not read %s; sending a pointer",
                    attachment.name or attachment.attachment_id,
                )
    return loaded
