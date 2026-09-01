"""P7-01 — `ctx.attachments`: media the log points at but cannot rebuild.

The log is lossless JSON, so an image never enters it; a `MediaBlock` carries a
reference and the bytes live in a content-addressed store. The properties worth
holding are the ones that follow from *content addressing* rather than from the
writing: the same file twice is one blob, two sessions share it, and a fork
references exactly what its parent does — which is why this store has no owner
directory and why nothing here collects automatically.

The other half is the token estimate. A media block used to contribute **zero**,
which meant a conversation of forty images reported no context pressure at all;
the tests below care that the number is non-zero and proportionate, not that it
is right to the token — it exists to answer "compact before asking?".

## What the base64 cache saves

A media block stays in derived history for the life of the session, so a **4 MB PDF
attached at the first turn was re-read from disk and re-encoded into a 5.5 MB
string for every request after it**.

The cache cannot go stale, and that is a property of the naming rather than a
promise it has to keep: `attachment_id` is the SHA-256 of the content, so two refs
with one id have one body by definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.llm.types import AttachmentRef, MediaBlock, Message, create_user_message
from ph.seams.attachments import AttachmentStore, digest_of
from ph.seams.token_meter import (
    IMAGE_TOKENS_UNKNOWN,
    MEDIA_TOKENS_UNKNOWN,
    PDF_TOKENS_PER_PAGE,
    TokenMeter,
    estimate_media_tokens,
)

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"pretend pixels" * 64


def _store(tmp_path: Path) -> AttachmentStore:
    return AttachmentStore(ctx=Context(), root=tmp_path / "attachments")


# ------------------------------------------------------------- the storing --


async def test_the_same_bytes_are_stored_once(tmp_path: Path) -> None:
    """Content addressing, and the reason a fork needs no copy.

    Two sessions attaching one photo share a blob, so a child that references
    its parent's digests is not referencing a directory the parent owns — which
    is what stops deleting a parent from breaking its children.
    """
    store = _store(tmp_path)

    first = await store.save_bytes(content=PNG, mime="image/png", name="a.png")
    second = await store.save_bytes(content=PNG, mime="image/png", name="b.png")

    assert first.attachment_id == second.attachment_id == digest_of(PNG)
    assert store.path_for(first) == store.path_for(second)
    assert len(list((tmp_path / "attachments").iterdir())) == 1


async def test_a_reference_resolves_without_a_directory_scan(tmp_path: Path) -> None:
    """`path_for` is a pure function of the reference.

    The digest is the name and the suffix comes from the ref's own MIME, so a
    reference read back out of a stored log resolves to the same path the write
    produced — on a machine that has never seen the session before.
    """
    store = _store(tmp_path)
    ref = await store.save_bytes(content=PNG, mime="image/png", name="shot.png")

    travelled = AttachmentRef.model_validate(ref.to_wire())

    assert store.path_for(travelled) == store.path_for(ref)
    assert store.path_for(travelled).suffix == ".png"
    assert await store.load_bytes(travelled) == PNG


async def test_a_missing_blob_is_a_question_that_can_be_asked(tmp_path: Path) -> None:
    """`exists` is what lets a caller degrade instead of failing.

    A reference whose bytes are gone — a session copied without its attachments —
    must become a pointer the model can read, not a wire error it cannot act on.
    """
    store = _store(tmp_path)
    ref = await store.save_bytes(content=PNG, mime="image/png")
    assert store.exists(ref)

    store.path_for(ref).unlink()

    assert not store.exists(ref)


async def test_ingesting_a_path_records_the_name_and_sniffs_the_type(tmp_path: Path) -> None:
    """The human door. A person attaches a file they can already open, and the
    name travels so a reader recognizes it later."""
    store = _store(tmp_path)
    source = tmp_path / "diagram.png"
    source.write_bytes(PNG)

    ref = await store.save_path(source)

    assert ref.mime == "image/png"
    assert ref.name == "diagram.png"
    assert ref.bytes == len(PNG)


async def test_the_store_never_derives_measurements(tmp_path: Path) -> None:
    """Dimensions are facts an ingester *had*, never facts this went and got.

    Decoding an image to measure it is the optional dependency `media-transform`
    exists to keep optional (P7-02), so a caller that does not know leaves them
    absent and the estimate falls back — rather than the store growing a decoder.
    """
    store = _store(tmp_path)

    plain = await store.save_bytes(content=PNG, mime="image/png")
    measured = await store.save_bytes(content=PNG, mime="image/png", width=800, height=600)

    assert plain.width is None and plain.height is None
    assert (measured.width, measured.height) == (800, 600)


async def test_the_encoding_is_paid_once_per_process(tmp_path: Path) -> None:
    """A media block lives in derived history for the session's life, so the
    encode was being paid on every model step.

    Safe to cache because the id *is* the content digest — two refs with one id
    have one body by definition, so there is no invalidation to get wrong.
    """
    store = _store(tmp_path)
    ref = await store.save_bytes(content=PNG, mime="image/png")

    first = await store.load_b64(ref)
    store.path_for(ref).unlink()
    second = await store.load_b64(ref)

    assert first == second, "the second call went back to a file that is gone"


# ------------------------------------------------------------ the estimate --


def _ref(mime: str, **facts: Any) -> AttachmentRef:
    """A placeholder reference, so each estimate test shows only what varies."""
    return AttachmentRef(attachment_id="sha256:x", mime=mime, bytes=1024, **facts)


def test_media_never_costs_zero_tokens() -> None:
    """The bug this replaces, stated as a property.

    `TokenMeter.measure` reads `.text`, `.arguments` and nested `.content`; a
    media block has none of them, so it contributed nothing and an image-heavy
    conversation reported no pressure. Every kind now costs something.
    """
    for mime in ("image/png", "application/pdf", "audio/wav", "video/mp4", "model/gltf+json"):
        assert estimate_media_tokens(_ref(mime)) > 0, mime


def test_a_measured_image_is_priced_from_its_pixels() -> None:
    """Anthropic's published `w x h / 750`, used when the facts are there and a
    flat figure when they are not — the difference being what an ingester knew,
    never what this module decoded."""
    assert estimate_media_tokens(_ref("image/png", width=1000, height=750)) == 1000
    assert estimate_media_tokens(_ref("image/png")) == IMAGE_TOKENS_UNKNOWN


def test_a_pdf_is_priced_by_its_pages() -> None:
    assert estimate_media_tokens(_ref("application/pdf", pages=3)) == 3 * PDF_TOKENS_PER_PAGE


def test_an_unknown_kind_still_costs_the_floor() -> None:
    """Being wrong by a factor is a rounding error against being wrong by
    everything, which is what zero was."""
    assert estimate_media_tokens(_ref("application/x-invented")) == MEDIA_TOKENS_UNKNOWN


def test_a_message_carrying_media_measures_more_than_its_text() -> None:
    """End to end through the meter the compaction trigger actually reads."""
    ctx = Context()
    meter = TokenMeter(ctx=ctx)
    ref = _ref("image/png", name="a.png")
    text_only: Message = create_user_message(
        content=[{"type": "text", "text": "look at this"}], source={"kind": "user"}
    )
    with_media = create_user_message(
        content=[{"type": "text", "text": "look at this"}, MediaBlock(attachment=ref)],
        source={"kind": "user"},
    )

    assert meter.measure(with_media) == meter.measure(text_only) + IMAGE_TOKENS_UNKNOWN


# --------------------------------------------------------------- the mount --


async def test_the_row_provides_the_store(mount: Any, tmp_path: Path) -> None:
    """`attachments-local` ships in `ph-base`, beside the spill store and
    deliberately not inside it: clearing a cache must never delete conversation."""
    ctx = await mount()

    store = ctx.attachments

    assert store.root == tmp_path / "attachments"
    assert store.root != ctx.spill_store.root
