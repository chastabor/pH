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
from test_dimensions import png

from ph.agent.types import AgentOptions
from ph.cordis import Context
from ph.llm.adapter import ResolvedModel
from ph.llm.media import (
    degrade_media,
    media_pointer_text,
    oversized_notices,
    unusable_reason,
)
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


# ---------------------------------------------------------- pixel limits --


def _pixels(*, accepts: int | None = None, uses: int | None = None) -> ResolvedModel:
    return ResolvedModel(
        accepts=frozenset({"image/png"}), max_image_edge=accepts, usable_image_edge=uses
    )


def test_an_image_over_the_routes_pixel_limit_is_a_reason() -> None:
    """The hard ceiling, on the same path as the byte ceiling (P7-03).

    Anthropic refuses over 8000 px, so a request carrying one fails outright —
    which is precisely the outcome `unusable_reason` exists to convert into a
    sentence the model can read. The measurement comes from the header, so this
    is a limit pH can actually check before sending.
    """
    big = _ref(width=9000, height=1000)

    reason = unusable_reason(big, _Store(), _pixels(accepts=8000))

    assert reason is not None and "9000x1000" in reason and "8000-pixel" in reason
    assert unusable_reason(_ref(width=8000, height=8000), _Store(), _pixels(accepts=8000)) is None


def test_an_unmeasured_image_is_never_refused_for_its_size() -> None:
    """`None` dimensions mean "not measured", never "zero".

    A format `ph.llm.dimensions` cannot read — an ingester that supplied nothing,
    a MIME added to a route before the probe learned it — must keep working
    exactly as it did before there was a pixel limit at all.
    """
    assert unusable_reason(_ref(), _Store(), _pixels(accepts=8000)) is None
    assert oversized_notices([_message(MediaBlock(attachment=_ref()))], _pixels(uses=1568)) == []


def test_a_route_that_publishes_no_pixel_limit_warns_about_nothing() -> None:
    """The default on every OpenAI-compatible profile, and the honest one.

    Unlike `accepts`, an unknown pixel ceiling has no safe assumption: one
    gateway scales at 2048 and another not at all, so a guessed number would warn
    a person about an overpayment that is not happening.
    """
    huge = _message(MediaBlock(attachment=_ref(width=4000, height=3000)))

    assert oversized_notices([huge], _takes("image/png")) == []


def test_an_image_larger_than_the_route_uses_is_sent_and_flagged() -> None:
    """The distinction the two limits exist to draw.

    Over `max_image_edge` nothing is sent. Over `usable_image_edge` everything
    works — the model sees the picture, the turn is correct — and the surplus
    pixels are uploaded on every request of the session and discarded at the far
    end. Nothing about the conversation is wrong, which is exactly why it needs
    saying: the failure otherwise lasts the whole session unannounced.
    """
    message = _message(MediaBlock(attachment=_ref(width=4000, height=3000)))
    route = _pixels(accepts=8000, uses=1568)

    messages, degraded = degrade_media([message], _Store(), route)
    notices = oversized_notices(messages, route)

    assert degraded == [], "an image the route accepts must still be sent"
    assert messages[0].content[0].attachment.width == 4000, "and sent unchanged"
    (notice,) = notices
    assert (notice["width"], notice["height"], notice["usableEdge"]) == (4000, 3000, 1568)
    assert notice["name"] == "shot.png", "and names the file a person would recognise"


async def test_the_oversized_notice_lands_once_and_names_the_picture(
    mount: Any, tmp_path: Any
) -> None:
    """End to end, and `record_degraded`'s rule applied to its sibling.

    Derived history replays the attachment on every step for the life of the
    session, so an append per request would bury the conversation in one repeated
    sentence. The fake route is given pixel limits here because it is the one
    that runs in a test; the shipped numbers live on the Anthropic row.
    """
    ctx: Context = await mount()
    ctx.llm_fake.route = ResolvedModel(
        accepts=frozenset({"image/png"}), max_image_edge=8000, usable_image_edge=1568
    )
    store: AttachmentStore = ctx.attachments
    ref = await store.save_bytes(
        content=PNG, mime="image/png", name="shot.png", width=4000, height=3000
    )
    session = ctx.sessions.create("oversized")
    agent = ctx.agents.create(session, AgentOptions(provider="fake", model="fake-1"))

    agent.followup(_message({"type": "text", "text": "look"}, MediaBlock(attachment=ref)))
    await agent.run()
    await agent.prompt("and again")

    notices = [event for event in session.events if event.type == "attachment/oversized"]
    assert len(notices) == 1
    assert notices[0].ignorable
    assert notices[0].data["attachments"][0]["name"] == "shot.png"
    assert not [one for one in session.events if one.type == "attachment/degraded"]


async def test_the_store_measures_an_image_it_is_given(mount: Any) -> None:
    """P7-03's enabling change: `width` is no longer a fact someone had to know.

    The header carries it, so the store reads it — which is what makes every
    pixel limit above checkable rather than decorative. A caller that already
    knows still wins, because an ingester that decoded the image knows at least
    as much as its header does.
    """
    ctx: Context = await mount()
    store: AttachmentStore = ctx.attachments
    # `test_dimensions.png` rather than a second hand-rolled header: that module
    # is where the byte layout is stated and asserted, and a private copy here
    # would drift from it silently — this test would keep passing against a PNG
    # the probe had stopped being able to read.
    real = png(1280, 720)

    measured = await store.save_bytes(content=real, mime="image/png", name="real.png")
    told = await store.save_bytes(
        content=real, mime="image/png", name="real.png", width=1, height=2
    )
    unreadable = await store.save_bytes(content=PNG, mime="image/png", name="stub.png")

    assert (measured.width, measured.height) == (1280, 720)
    assert (told.width, told.height) == (1, 2), "a caller that knows is not overruled"
    assert unreadable.width is None, "and a header that cannot be read stays unmeasured"
