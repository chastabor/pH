"""P7-01 — `media-degrade`: one answer to "can this route read this file".

The row exists because the question is not wire-shaped, and answering it per
adapter meant the *fake* adapter answered it not at all — a route declaring no
media received media anyway, silently, which is the failure the whole attachment
path is built to prevent. Above every adapter, on `llm/stream`, so a route that
has said nothing gets the safe answer for free.

The properties: an unusable block becomes a sentence rather than nothing, an
acceptable request is left byte-identical, and the notice lands once rather than
once per step.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.llm.adapter import ResolvedModel
from ph.llm.media import degrade_media, media_pointer_text, unusable_reason
from ph.llm.types import AttachmentRef, MediaBlock, TextBlock, create_user_message
from ph.seams.attachments import AttachmentStore

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 32


class _Store:
    """A store that says everything it is asked about is present."""

    def exists(self, _ref: Any) -> bool:
        return True


def _ref(mime: str = "image/png", **facts: Any) -> AttachmentRef:
    return AttachmentRef(attachment_id="sha256:x", mime=mime, bytes=1024, name="shot.png", **facts)


def _message(*blocks: Any) -> Any:
    return create_user_message(content=list(blocks), source={"kind": "user"})


def _takes(*mimes: str, limit: int | None = None) -> ResolvedModel:
    return ResolvedModel(accepts=frozenset(mimes), max_attachment_bytes=limit)


# ------------------------------------------------------------- the reasons --


def test_every_way_an_attachment_can_be_unusable_has_a_sentence() -> None:
    """Four unrelated situations, one remedy — the model can act on a pointer
    and can do nothing with a wire error."""
    assert "no attachment store" in (unusable_reason(_ref(), None, _takes("image/png")) or "")
    assert "does not accept" in (unusable_reason(_ref(), _Store(), _takes()) or "")
    assert "over the" in (unusable_reason(_ref(), _Store(), _takes("image/png", limit=8)) or "")
    assert unusable_reason(_ref(), _Store(), _takes("image/png")) is None


def test_a_missing_blob_is_a_reason_not_a_crash() -> None:
    """A session copied without its attachments still opens and still runs."""

    class Gone:
        def exists(self, _ref: Any) -> bool:
            return False

    assert "bytes are gone" in (unusable_reason(_ref(), Gone(), _takes("image/png")) or "")


# ---------------------------------------------------------- the rewriting --


def test_an_unusable_block_becomes_a_sentence_in_its_place() -> None:
    """Replaced, not dropped — the position in the message is kept, so a model
    reading "look at this" still finds something where the image was."""
    message = _message({"type": "text", "text": "look at this"}, MediaBlock(attachment=_ref()))

    (rewritten,), degraded = degrade_media([message], _Store(), _takes())

    kinds = [block.type for block in rewritten.content]
    assert kinds == ["text", "text"], "the block was dropped rather than replaced"
    assert isinstance(rewritten.content[1], TextBlock)
    assert rewritten.content[1].text == media_pointer_text(_ref())
    assert [item["reason"] for item in degraded] == ["this route does not accept image/png"]


def test_a_request_with_nothing_to_degrade_is_left_identical() -> None:
    """The overwhelmingly common case allocates nothing and changes no bytes,
    which is what the prefix cache is counting on (A12)."""
    messages = [_message({"type": "text", "text": "hello"})]

    rewritten, degraded = degrade_media(messages, _Store(), _takes())

    assert degraded == []
    assert rewritten[0] is messages[0], "an untouched message was copied"


def test_an_acceptable_attachment_is_left_alone(tmp_path: Any) -> None:
    """Only what cannot go is rewritten; the rest reaches the adapter as media."""
    message = _message(MediaBlock(attachment=_ref()))

    (rewritten,), degraded = degrade_media([message], _Store(), _takes("image/png"))

    assert degraded == []
    assert rewritten is message


# ------------------------------------------------------------- the notice --


async def test_the_row_degrades_and_records_once(mount: Any, tmp_path: Any) -> None:
    """End to end through the waterfall, on the fake route — which declares no
    media, and so is every text-only provider's path too.

    The notice lands once, not once per step: every request re-derives the same
    history, so the refusal repeats for the life of the session.
    """
    ctx: Context = await mount()
    store: AttachmentStore = ctx.attachments
    ref = await store.save_bytes(content=PNG, mime="image/png", name="shot.png")
    session = ctx.sessions.create("degraded")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="fake-1"))

    agent.followup(_message({"type": "text", "text": "what is this?"}, MediaBlock(attachment=ref)))
    await agent.run()
    await agent.prompt("and again")

    notices = [event for event in session.events if event.type == "attachment/degraded"]
    assert len(notices) == 1
    assert notices[0].ignorable
    (request,) = [one for one in ctx.llm_fake.requests if one.is_loop_request][:1]
    assert "was not sent" in str(request.messages[0].content)
