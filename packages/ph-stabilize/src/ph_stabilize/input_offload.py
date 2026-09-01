"""`input-offload` — a pasted blob relocated, not lost (P4-02, G3).

The sibling of `tool-result-offload`, for the other direction: someone pastes a
2 MB log into the prompt.

**One statement, two readers.** The original `user/message` is logged as the person
sent it. A second `user/message` carrying the preview is appended with
`surface_op: replace` citing it. From then on:

* `derive_messages()` — what the model reads — yields the preview;
* `transcript()` — what the person reads — yields the original, in full.

Nothing is written twice and nothing is deleted, which is what makes the
substitution reversible reading rather than an edit (I4). Rewriting the message
*before* it is logged was the alternative, and is refused: the log would then
attribute harness-authored text to the human, which is a false statement in an
append-only record rather than merely a lossy one.

**Where it runs, and why there.** `agent/request` is the only seam between the
driver appending `user/message` and `_build_request` calling `derive_messages()`.
A listener here appends the replacement and returns the config untouched; the loop
then derives, sees the replacement, and sends the preview. There is no second copy
of the substitution to keep in step — the loop's own re-derivation applies it.

**Only the newest message, and only once.** `Session.latest("user/message")` and
`is_replacement_surface_event` make the pass idempotent: the replacement is itself
a `user/message`, so the next request finds it, sees a replacement, and stops.

@module ph_stabilize.input_offload
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ph.cordis import Context, plugin
from ph.llm.types import PluginSource, create_user_message, text_of
from ph.session import (
    Session,
    SessionEvent,
    SurfaceIntent,
    derive_event_message,
    is_replacement_surface_event,
)
from ph.session.events import SurfaceReplace
from ph.text import count_of
from ph.wire import WireModel

from .offload import HISTORY_PREFIX, content_preview, over_token_limit

__all__ = [
    "HUMAN_TOKEN_LIMIT_BEFORE_EVICT",
    "TOO_LARGE_HUMAN_MSG",
    "Config",
    "apply",
]

HUMAN_TOKEN_LIMIT_BEFORE_EVICT = 50_000
"""`human_message_token_limit_before_evict`, so the char threshold is 200 000.

Two and a half times the tool-result limit, and deliberately: a person does not
reach this by typing. It is the paste of a build log or a dumped table, which is
the case the row exists for."""

TOO_LARGE_HUMAN_MSG = """Message content too large and was saved to the filesystem at: {file_path}

You can read the full content using the read_file tool with pagination (offset and limit parameters).

Here is a preview showing the head and tail of the content:

{content_sample}
"""  # noqa: E501
"""Verbatim from `deepagents/middleware/filesystem.py`."""


class Config(WireModel):
    """Row config. `None` disables offloading, as upstream's `None` does."""

    token_limit: int | None = HUMAN_TOKEN_LIMIT_BEFORE_EVICT


def _pending(session: Session, config: Config) -> tuple[SessionEvent, str] | None:
    """The newest human message that should be offloaded, and its text.

    `None` when there is nothing to do — no message, one already replaced, or
    one that fits. Read from the log rather than from the derived list because
    the substitution needs the *event*: a derived `Message` carries no seq, and
    the seq is what a surface replace cites.
    """
    event = session.latest("user/message")
    if event is None or is_replacement_surface_event(event):
        return None
    message = derive_event_message(event)
    if message is None:
        return None
    # Text blocks only. An image is not what exhausts a context window, and it
    # is the half of a message that reading a file cannot give back.
    text = text_of(message.content)
    return (event, text) if over_token_limit(text, config.token_limit) else None


@plugin("input-offload", inject=["spill_store"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Replace an oversized pasted message on the surface, not in the log."""

    async def offload(proposal: Any, next_: Any) -> Any:
        session: Session | None = getattr(proposal.agent, "session", None)
        pending = _pending(session, config) if session is not None else None
        if session is None or pending is None:
            return await next_(proposal)
        event, text = pending
        ref = await ctx.spill_store.try_save_text(
            owner=session.id,
            source="pasted message",
            # Named by the seq it replaces rather than upstream's uuid: the
            # store already dedupes by content digest, so a random name buys
            # nothing a traceable one does not.
            suggested_name=f"{HISTORY_PREFIX}/{event.seq}.md",
            content=text,
        )
        if ref is None:
            # Fail open, like the tool-result side: an offload that cannot store
            # the content must not be the reason the model loses it. The seam
            # logs why.
            return await next_(proposal)
        session.append(
            "offload/input-spilled",
            {"seq": event.seq, "locator": ref.locator, "bytes": ref.bytes},
        )
        preview = TOO_LARGE_HUMAN_MSG.format(
            file_path=ref.locator, content_sample=content_preview(text)
        )
        session.append(
            "user/message",
            create_user_message(
                content=[{"type": "text", "text": preview}],
                # A plugin's notice, not the person's words. Attributing the
                # preview to `user` would be the same lie one layer down from
                # the one this design rejected — and `PluginSource` is the
                # repo's idiom for injected context, so the transcript already
                # renders it as such.
                source=PluginSource(
                    plugin="input-offload",
                    form="notice",
                    summary=f"{count_of(ref.bytes, 'byte')} offloaded to {Path(ref.locator).name}",
                ),
            ).to_wire(),
            SurfaceIntent(
                surface_op=SurfaceReplace(replaces=(event.seq,)),
                source_event_seqs=(event.seq,),
            ),
        )
        # The config is returned untouched. `_build_request` calls
        # `derive_messages()` *after* this waterfall, so the loop picks the
        # replacement up on its own — which is what keeps one statement of the
        # substitution rather than two that have to agree.
        return await next_(proposal)

    ctx.on("agent/request", offload)
