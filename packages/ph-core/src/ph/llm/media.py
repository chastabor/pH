"""`media-degrade`: one answer to "can this route read this file" (P7-01).

The row is above every adapter, on `llm/stream`, because the question does not
depend on the wire. Whether a route takes a MIME type, whether a file is under
the ceiling of the route, and whether the bytes are still on disk are the same
for Anthropic and for an OpenAI-compatible gateway. When each adapter answered
the question, the *fake* adapter did not answer it at all. A text-only route
declared nothing and received media anyway, without an error. This row exists
to end exactly that failure.

So a `MediaBlock` that cannot go is replaced here with a sentence that the
model can act on. What is left for an adapter is the part that does differ:
turning an acceptable attachment into `image` or `document`, `image_url` or
`input_audio`.

**The row rewrites the request, and that is allowed.** `agent-loop-invariant`
is registered with `prepend=True` on this same waterfall, and a comment there
gives the reason: the invariant is about what the *loop* built. The invariant
has therefore already run and passed when this listener sees the request. What
the model is shown still traces to the log, because a pointer names an
attachment that the log holds a reference to.

**The notice is recorded one time.** Every request derives the same history
again, so a refused attachment is refused again on every step for the life of
the session. Appending on every request would fill the conversation with one
repeated sentence.

@module ph.llm.media
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Protocol

from ..cordis import Context, Next, plugin
from ..json import JsonObject, as_seq
from ..keys import ATTACHMENTS, LLM, SESSIONS
from ..session import Session
from ..session.writers import log_writer
from .adapter import ResolvedModel
from .types import (
    AttachmentRef,
    ContentBlock,
    GenerateOptions,
    Message,
    StreamChunk,
    TextBlock,
    attachment_of,
)

_LOG = log_writer(__name__)

__all__ = [
    "ATTACHABLE",
    "MediaPresence",
    "apply",
    "degrade_media",
    "is_attachable",
    "media_pointer_text",
    "oversized_notices",
    "record_degraded",
    "record_oversized",
    "untaken",
    "unusable_reason",
]

log = logging.getLogger("ph.llm.media")


ATTACHABLE: tuple[str, ...] = ("image/", "audio/", "video/", "application/pdf")
"""What a `MediaBlock` may carry at all. This question comes before the question
of any one route.

The value is a set of prefixes and not an enumeration, because the exact list
belongs to the *route* (`ResolvedModel.accepts`). That list differs for each
deployment, and `degrade_media` below checks it on each request. A closed list
here would refuse `image/heic` on a route that takes it, and no configuration
could lift the refusal. What can be stated without knowing the route is the
coarser fact that is true everywhere: a provider ingests pictures, sound, video
and documents as content, and does not ingest archives.

**The constant is here and not in the tool that first needed it** (P7-01). It
is a statement about `MediaBlock` and not about `tool-attach`, and this module
already owns that vocabulary (`unusable_reason`, `media_pointer_text`). In the
tools package it would also have been in a place that this layer cannot import
from. The *other* producer, the `--attach` option of the user, could then not
have used the same sentence.
"""


def is_attachable(mime: str) -> bool:
    """Whether this is media a provider could ingest as content."""
    return mime.startswith(ATTACHABLE)


def _longest_edge(attachment: AttachmentRef) -> int | None:
    """The larger of the two sides of an image, or `None` if it was never measured.

    Both limits below are stated for an *edge* and not for a pixel count. That
    is how providers state them, and it is the number that a person can act on:
    "your screenshot is 3840 wide" is a fact about the file that they made.
    """
    if attachment.width is None or attachment.height is None:
        return None
    return max(attachment.width, attachment.height)


def oversized_notices(messages: Sequence[Message], route: ResolvedModel) -> list[JsonObject]:
    """Images that will be *sent* and then scaled down by the provider (P7-03).

    This is not a degradation, and it is deliberately not on the same path.
    Nothing is replaced, the model sees the picture, and the turn is exactly as
    correct as it would have been. What is wrong is the cost: every request of
    the session uploads again the pixels that the provider discards. The only
    remedy is that a person knows, because no shipped row resizes an image
    (`media-transform`, P7-02, is only planned).

    The images are read after degradation and not before. An image over the
    *accept* limit is already gone. Telling someone that their unsent file is
    also too detailed would be two answers to one problem.
    """
    if route.usable_image_edge is None:
        return []
    notices: list[JsonObject] = []
    seen: set[str] = set()
    for message in messages:
        for block in message.content:
            attachment = attachment_of(block)
            if attachment is None or attachment.attachment_id in seen:
                continue
            edge = _longest_edge(attachment)
            if edge is None or edge <= route.usable_image_edge:
                continue
            seen.add(attachment.attachment_id)
            notices.append(
                {
                    "attachmentId": attachment.attachment_id,
                    "mime": attachment.mime,
                    "name": attachment.name,
                    "width": attachment.width,
                    "height": attachment.height,
                    "usableEdge": route.usable_image_edge,
                }
            )
    return notices


class MediaPresence(Protocol):
    """What degradation asks of an attachment store: whether the bytes are still there.

    This is a Protocol and not `AttachmentStore`, because `exists` is all that
    the two readers below use. Narrowing to it makes the stand-ins in
    `test_media_degrade` fair doubles and not a shortcut past a type. Naming the
    class would not have cost an import edge: `ph.keys` already carries it under
    `TYPE_CHECKING`, and this module imports `ATTACHMENTS` from there. The
    reason is therefore the contract and not the layering.
    """

    def exists(self, ref: AttachmentRef) -> bool: ...


def untaken(needed: Sequence[str], accepts: frozenset[str]) -> tuple[str, ...]:
    """The media in `needed` that a route which takes `accepts` cannot take.

    An entry is not taken when it is a type that the route does not list, or a
    family (`image/*`) of which the route lists no member. This function is for
    a requirement that is stated before the bytes exist, as a skill states what
    it hands a child (`Skill.accepts`). A block that exists is checked exactly
    (`unusable_reason`).
    """
    return tuple(
        media
        for media in needed
        if not (
            media in accepts
            or (media.endswith("/*") and any(one.startswith(media[:-1]) for one in accepts))
        )
    )


def unusable_reason(
    attachment: AttachmentRef,
    store: MediaPresence | None,
    route: ResolvedModel,
) -> str | None:
    """Why this attachment cannot be sent, or `None` if it can.

    One function answers five unrelated situations on purpose: no store, a MIME
    type that the route does not take, a file over the byte ceiling of the
    route, an image over the edge limit of the route, and bytes that are no
    longer on disk. All five have the same remedy, because the model can act on
    a pointer and can do nothing with a wire error.

    The byte ceiling is the ceiling of an *inline* block. A file of a type that
    the route sends by reference (`ResolvedModel.uploads`) is not held to it
    here. Its adapter uploads it, and decides what to send when the upload
    gives no handle.
    """
    if store is None:
        return "no attachment store is mounted"
    if attachment.mime not in route.accepts:
        return f"this route does not accept {attachment.mime}"
    if (
        route.max_attachment_bytes is not None
        and attachment.bytes > route.max_attachment_bytes
        and attachment.mime not in route.uploads
    ):
        return f"{attachment.bytes} bytes is over the {route.max_attachment_bytes}-byte limit"
    edge = _longest_edge(attachment)
    if route.max_image_edge is not None and edge is not None and edge > route.max_image_edge:
        return (
            f"{attachment.width}x{attachment.height} is over this route's "
            f"{route.max_image_edge}-pixel limit"
        )
    if not store.exists(attachment):
        return "the stored bytes are gone"
    return None


def media_pointer_text(attachment: AttachmentRef) -> str:
    """What the model reads in place of media that it was not shown.

    The text gives the kind of the media, the name that the user gave it, and
    the fact that this route could not read it. That is enough to ask about it,
    or to use a tool. A message never loses a block without saying so.
    """
    name = attachment.name or attachment.attachment_id
    return f'[{attachment.mime} attachment "{name}" was not sent: this model cannot read it]'


def degrade_media(
    messages: Sequence[Message],
    store: MediaPresence | None,
    route: ResolvedModel,
) -> tuple[tuple[Message, ...], list[JsonObject]]:
    """The messages that an adapter should see, and a record of what was replaced.

    When nothing had to go, the function returns the originals unchanged. The
    most common case therefore allocates nothing and leaves the request
    byte-identical, which the prefix cache depends on (A12).
    """
    degraded: list[JsonObject] = []
    rewritten: list[Message] = []
    changed = False
    for message in messages:
        blocks: list[ContentBlock] = []
        touched = False
        for block in message.content:
            attachment = attachment_of(block)
            reason = None if attachment is None else unusable_reason(attachment, store, route)
            if attachment is None or reason is None:
                blocks.append(block)
                continue
            touched = True
            blocks.append(TextBlock(text=media_pointer_text(attachment)))
            degraded.append(
                {
                    "attachmentId": attachment.attachment_id,
                    "mime": attachment.mime,
                    "name": attachment.name,
                    "reason": reason,
                }
            )
        if not touched:
            rewritten.append(message)
            continue
        changed = True
        rewritten.append(message.model_copy(update={"content": blocks}))
    return (tuple(rewritten) if changed else tuple(messages)), degraded


def _record_once(session: Session, event_type: str, provider: str, items: list[JsonObject]) -> bool:
    """Append a media notice, but only when it says something new.

    One fold serves both notices, because they share the *mechanism* and differ
    only in the event name. Every request derives the same history again, so an
    attachment that was refused, or sent too large, is refused again on every
    step for the life of the session. Appending on every request would fill the
    conversation with one repeated sentence.

    The previous notice is read through `Session.latest`, which is the
    incremental fold and not a scan. This function runs on every request, and
    the answer that it compares against is almost always the one that it just
    wrote. It returns whether it appended. That is also the right condition for
    logging, because a warning on every request is the same flood in the
    channel of the operator.
    """
    if not items:
        return False
    ids = [item["attachmentId"] for item in items]
    previous = session.latest(event_type)
    if (
        previous is not None
        and [str(one) for one in as_seq(previous.data.get("attachmentIds"))] == ids
    ):
        return False
    _LOG.append(
        session, event_type, {"provider": provider, "attachmentIds": ids, "attachments": items}
    )
    return True


def record_degraded(session: Session, provider: str, degraded: list[JsonObject]) -> bool:
    """Media a route would not take at all."""
    return _record_once(session, "attachment/degraded", provider, degraded)


def record_oversized(session: Session, provider: str, notices: list[JsonObject]) -> bool:
    """Media that was sent and is larger than the route can use (P7-03)."""
    return _record_once(session, "attachment/oversized", provider, notices)


@plugin("media-degrade", affects="environment", inject=[LLM, SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Replace media the routed model cannot read, before any adapter sees it."""

    async def degrade(
        options: GenerateOptions,
        next_: Next[AsyncIterator[StreamChunk]],
    ) -> AsyncIterator[StreamChunk]:
        store = ctx.get(ATTACHMENTS)
        route = ctx.require(LLM).resolve_model(options.provider, options.model)
        messages, degraded = degrade_media(options.messages, store, route)
        oversized = oversized_notices(messages, route)
        if not degraded and not oversized:
            return await next_()
        session = ctx.require(SESSIONS).get(options.session_id) if options.session_id else None
        # A warning is logged only when the notice was *new*, which is the same
        # condition that the append is under. A warning that repeats on every
        # step for the life of the session is the flood that the fold exists to
        # prevent. It would only have moved into the channel of the operator,
        # where nobody would notice that it was a duplicate.
        if session is None or record_degraded(session, options.provider, degraded):
            for item in degraded:
                log.warning(
                    "ph.llm.media: %s is sending %s as a pointer: %s",
                    options.provider,
                    item["name"] or item["attachmentId"],
                    item["reason"],
                )
        if session is None or record_oversized(session, options.provider, oversized):
            for notice in oversized:
                log.warning(
                    'ph.llm.media: "%s" is %sx%s; %s uses at most %s pixels on the long edge, '
                    "so the rest is uploaded on every request and discarded",
                    notice["name"] or notice["attachmentId"],
                    notice["width"],
                    notice["height"],
                    options.provider,
                    notice["usableEdge"],
                )
        if not degraded:
            return await next_()
        return await next_(replace(options, messages=messages))

    ctx.on("llm/stream", degrade)
