"""`media-degrade` — one answer to "can this route read this file" (P7-01).

Above every adapter, on `llm/stream`, because the question is not wire-shaped.
Whether a route takes a MIME type, whether a file is under its ceiling, whether
the bytes are still on disk — none of that differs between Anthropic and an
OpenAI-compatible gateway, and answering it per adapter meant the *fake* adapter
answered it not at all: a text-only route declared nothing and silently received
media anyway, which is the exact failure this row exists to end.

So a `MediaBlock` that cannot go is replaced, here, with a sentence the model can
act on, and what is left for an adapter is the part that genuinely differs —
turning an acceptable attachment into `image` or `document`, `image_url` or
`input_audio`.

**It rewrites the request, and that is allowed.** `agent-loop-invariant` is
registered `prepend=True` on this same waterfall and its docstring says why: the
invariant is about what the *loop* built, so it has already run and passed by the
time this listener sees the request. What the model is shown still traces to the
log — a pointer names an attachment the log holds a reference to.

**The notice is recorded once.** Every request re-derives the same history, so a
refused attachment is refused again on every step for the session's life;
appending per request would bury the conversation in one repeated sentence.

@module ph.llm.media
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from ..cordis import Context, plugin
from ..session import Session
from .adapter import ResolvedModel
from .types import AttachmentRef, GenerateOptions, Message, TextBlock, attachment_of

__all__ = [
    "ATTACHABLE",
    "apply",
    "degrade_media",
    "is_attachable",
    "media_pointer_text",
    "oversized_notices",
    "record_degraded",
    "record_oversized",
    "unusable_reason",
]

log = logging.getLogger("ph.llm.media")


ATTACHABLE: tuple[str, ...] = ("image/", "audio/", "video/", "application/pdf")
"""What a `MediaBlock` may carry at all — the question above every route's own.

Prefixes rather than an enumeration, because the exact list is the *route's*
(`ResolvedModel.accepts`), differs per deployment and is checked per request by
`degrade_media` below: a closed list here would refuse `image/heic` on a route
that takes it, and no configuration could lift it. What can be said without
knowing the route is the coarser thing that is true everywhere — a provider
ingests pictures, sound, video and documents as content, and does not ingest
archives.

**Here rather than in the tool that first needed it** (P7-01). It is a statement
about `MediaBlock`, not about `tool-attach`, and this module already owns that
vocabulary (`unusable_reason`, `media_pointer_text`). In the tools package it
would also have sat where this layer cannot import it, so the *other* producer —
a person's `--attach` — could not have used the same sentence.
"""


def is_attachable(mime: str) -> bool:
    """Whether this is media a provider could ingest as content."""
    return mime.startswith(ATTACHABLE)


def _longest_edge(attachment: AttachmentRef) -> int | None:
    """The larger of an image's two sides, or `None` if it was never measured.

    Both limits below are stated per *edge* rather than per pixel count because
    that is how providers state them, and because it is the number a person can
    act on: "your screenshot is 3840 wide" is a fact about the file they made.
    """
    if attachment.width is None or attachment.height is None:
        return None
    return max(attachment.width, attachment.height)


def oversized_notices(messages: Sequence[Message], route: ResolvedModel) -> list[dict[str, Any]]:
    """Images that will be *sent* and then scaled down at the far end (P7-03).

    Not a degradation and deliberately not on the same path: nothing is replaced,
    the model sees the picture, and the turn is exactly as correct as it would
    have been. What is wrong is the bill — every request of the session re-uploads
    pixels the provider throws away — and the only thing that fixes it is a person
    knowing, because the row that could resize is optional (P7-02) and may not be
    mounted.

    Read after degradation rather than before: an image over the *accept* limit is
    already gone, and telling someone their unsent file is also too detailed would
    be two answers to one problem.
    """
    if route.usable_image_edge is None:
        return []
    notices: list[dict[str, Any]] = []
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


def unusable_reason(attachment: AttachmentRef, store: Any, route: ResolvedModel) -> str | None:
    """Why this attachment cannot be sent, or `None` if it can.

    Four unrelated situations answered with one branch on purpose — no store, a
    MIME the route does not take, a file over its ceiling, and bytes no longer on
    disk all have the same remedy, because the model can act on a pointer and can
    do nothing with a wire error.
    """
    if store is None:
        return "no attachment store is mounted"
    if attachment.mime not in route.accepts:
        return f"this route does not accept {attachment.mime}"
    if route.max_attachment_bytes is not None and attachment.bytes > route.max_attachment_bytes:
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
    """What the model reads in place of media it was not shown.

    Its kind, the name the person gave it, and the fact that this route could not
    read it — enough to ask about, or to reach for a tool. Never a message that
    quietly lost a block.
    """
    name = attachment.name or attachment.attachment_id
    return f'[{attachment.mime} attachment "{name}" was not sent: this model cannot read it]'


def degrade_media(
    messages: Sequence[Message], store: Any, route: ResolvedModel
) -> tuple[tuple[Message, ...], list[dict[str, Any]]]:
    """The messages an adapter should see, and an account of what was replaced.

    Returns the originals unchanged when nothing had to go, so the overwhelmingly
    common case allocates nothing and leaves the request byte-identical — which
    is what the prefix cache is counting on (A12).
    """
    degraded: list[dict[str, Any]] = []
    rewritten: list[Message] = []
    changed = False
    for message in messages:
        blocks: list[Any] = []
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


def _record_once(
    session: Session, event_type: str, provider: str, items: list[dict[str, Any]]
) -> bool:
    """Append a media notice, but only when it says something new.

    One fold for both notices, because the *mechanism* is what they share and the
    event name is all that differs: every request re-derives the same history, so
    an attachment refused — or sent too large — is refused again on every step for
    the life of the session, and appending per request would bury the
    conversation in one repeated sentence.

    Read through `Session.latest`, the incremental fold rather than a scan: this
    runs on every request and the answer it compares against is almost always the
    one it just wrote. Returns whether it appended, which is also the right
    condition for logging — a warning per request is the same flood in the
    operator's channel.
    """
    if not items:
        return False
    ids = [item["attachmentId"] for item in items]
    previous = session.latest(event_type)
    if (
        previous is not None
        and [str(one) for one in previous.data.get("attachmentIds") or ()] == ids
    ):
        return False
    session.append(event_type, {"provider": provider, "attachmentIds": ids, "attachments": items})
    return True


def record_degraded(session: Session, provider: str, degraded: list[dict[str, Any]]) -> bool:
    """Media a route would not take at all."""
    return _record_once(session, "attachment/degraded", provider, degraded)


def record_oversized(session: Session, provider: str, notices: list[dict[str, Any]]) -> bool:
    """Media that was sent and is larger than the route can use (P7-03)."""
    return _record_once(session, "attachment/oversized", provider, notices)


@plugin("media-degrade", inject=["llm", "sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Replace media the routed model cannot read, before any adapter sees it."""

    async def degrade(options: GenerateOptions, next_: Callable[..., Any]) -> Any:
        store = ctx.get("attachments")
        route = ctx.llm.resolve_model(options.provider, options.model)
        messages, degraded = degrade_media(options.messages, store, route)
        oversized = oversized_notices(messages, route)
        if not degraded and not oversized:
            return await next_()
        raw = ctx.sessions.get(options.session_id) if options.session_id else None
        session = raw if isinstance(raw, Session) else None
        # Logged only when the notice was *new*, which is the same condition the
        # append is under: a warning repeated on every step for the life of the
        # session is the flood the fold exists to prevent, moved into the
        # operator's channel where nobody would notice it was a duplicate.
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
