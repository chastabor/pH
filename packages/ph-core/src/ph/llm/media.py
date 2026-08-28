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
    "apply",
    "degrade_media",
    "media_pointer_text",
    "record_degraded",
    "unusable_reason",
]

log = logging.getLogger("ph.llm.media")


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


def record_degraded(session: Session, provider: str, degraded: list[dict[str, Any]]) -> None:
    """Append the notice, but only when it says something new.

    Read through `Session.latest`, which is the incremental fold rather than a
    scan: this runs on every request, and the answer it is comparing against is
    almost always the one it just wrote.
    """
    if not degraded:
        return
    ids = [item["attachmentId"] for item in degraded]
    previous = session.latest("attachment/degraded")
    if (
        previous is not None
        and [str(one) for one in previous.data.get("attachmentIds") or ()] == ids
    ):
        return
    session.append(
        "attachment/degraded",
        {"provider": provider, "attachmentIds": ids, "attachments": degraded},
    )


@plugin("media-degrade", inject=["llm", "sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Replace media the routed model cannot read, before any adapter sees it."""

    async def degrade(options: GenerateOptions, next_: Callable[..., Any]) -> Any:
        store = ctx.get("attachments")
        route = ctx.llm.resolve_model(options.provider, options.model)
        messages, degraded = degrade_media(options.messages, store, route)
        if not degraded:
            return await next_()
        for item in degraded:
            log.warning(
                "ph.llm.media: %s is sending %s as a pointer: %s",
                options.provider,
                item["name"] or item["attachmentId"],
                item["reason"],
            )
        session = ctx.sessions.get(options.session_id) if options.session_id else None
        if isinstance(session, Session):
            record_degraded(session, options.provider, degraded)
        return await next_(replace(options, messages=messages))

    ctx.on("llm/stream", degrade)
