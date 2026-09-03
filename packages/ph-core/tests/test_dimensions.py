"""P7-03 — an image's size from its header, with nothing installed.

The claim under test is narrow and load-bearing: **measuring an image is not
decoding one.** `AttachmentRef.width` sat empty for a phase because its docstring
said otherwise, and a route could not say "that picture is bigger than I can use"
without a number nobody was reading.

Every fixture here is **built byte by byte** rather than loaded from a file, and
that is the point twice over. It keeps the suite free of the dependency the
module exists to avoid — there is no Pillow here to generate a PNG with — and it
makes each test state the layout it is asserting, so a wrong offset reads as a
wrong constant rather than as a mysterious number from an opaque blob.

The formats do not agree with each other, and the disagreements are where the
bugs live: JPEG writes height before width, WebP has three incompatible chunk
layouts, and two of them bias their values by one.
"""

from __future__ import annotations

import pytest

from ph.llm.dimensions import HEADER_BUDGET, IMAGE_MIMES, image_dimensions


def png(width: int, height: int) -> bytes:
    """Signature, then the mandatory IHDR chunk at a fixed offset."""
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


def jpeg(width: int, height: int, *, marker: int = 0xC0, decoy: bool = False) -> bytes:
    """An APP0 segment, optionally a Huffman table, then a frame header.

    `decoy` writes a `0xC4` segment — a Huffman table, which sits *inside* the
    start-of-frame marker range — before the real frame. A reader that took the
    range wholesale returns two bytes of that table as the picture's height.
    """
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    huffman = b"\xff\xc4" + (10).to_bytes(2, "big") + b"\x11" * 8 if decoy else b""
    frame = (
        b"\xff"
        + bytes([marker])
        + (17).to_bytes(2, "big")
        + b"\x08"
        + height.to_bytes(2, "big")
        + width.to_bytes(2, "big")
        + b"\x03"
        + b"\x00" * 9
    )
    return b"\xff\xd8" + app0 + huffman + frame


def gif(width: int, height: int) -> bytes:
    """The logical screen descriptor, little-endian, straight after the magic."""
    return b"GIF89a" + width.to_bytes(2, "little") + height.to_bytes(2, "little") + b"\x00" * 7


def webp_lossy(width: int, height: int) -> bytes:
    return (
        b"RIFF"
        + (0).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8 "
        + (0).to_bytes(4, "little")
        + b"\x00\x00\x00"
        + b"\x9d\x01\x2a"
        + width.to_bytes(2, "little")
        + height.to_bytes(2, "little")
    )


def webp_lossless(width: int, height: int) -> bytes:
    bits = (width - 1) | ((height - 1) << 14)
    return (
        b"RIFF"
        + (0).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8L"
        + (0).to_bytes(4, "little")
        + b"\x2f"
        + bits.to_bytes(4, "little")
        + b"\x00" * 5
    )


def webp_extended(width: int, height: int) -> bytes:
    return (
        b"RIFF"
        + (0).to_bytes(4, "little")
        + b"WEBP"
        + b"VP8X"
        + (10).to_bytes(4, "little")
        + b"\x10"
        + b"\x00\x00\x00"
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )


@pytest.mark.parametrize(
    "build",
    [png, jpeg, gif, webp_lossy, webp_lossless, webp_extended],
    ids=["png", "jpeg", "gif", "webp-lossy", "webp-lossless", "webp-extended"],
)
def test_every_accepted_format_measures(build: object) -> None:
    """One assertion per layout, with an asymmetric size on purpose.

    3840x2160 rather than a square: width and height are swapped in JPEG relative
    to every other format here, and a square fixture agrees with a reader that has
    them backwards.
    """
    assert image_dimensions(build(3840, 2160)) == (3840, 2160)  # type: ignore[operator]


def test_a_huffman_table_is_not_a_frame_header() -> None:
    """`0xC4` is inside the SOF range and is not one. See `_SOF_MARKERS`."""
    assert image_dimensions(jpeg(1920, 1080, decoy=True)) == (1920, 1080)


@pytest.mark.parametrize("marker", [0xC1, 0xC2], ids=["extended-sequential", "progressive"])
def test_the_other_frame_headers_are_read_too(marker: int) -> None:
    """A progressive JPEG is an ordinary thing for a person to attach, and it
    carries its size under `0xC2` rather than `0xC0`."""
    assert image_dimensions(jpeg(800, 600, marker=marker)) == (800, 600)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x89PNG\r\n\x1a\n",  # a signature and nothing after it
        b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"gAMA" + b"\x00" * 8,
        b"\xff\xd8\xff\xe0" + (16).to_bytes(2, "big"),  # truncated mid-segment
        b"RIFF" + (0).to_bytes(4, "little") + b"WEBPXXXX" + b"\x00" * 20,
        b"GIF89a",
        b"%PDF-1.7\n",
        b"not an image at all",
    ],
    ids=[
        "empty",
        "png-signature-only",
        "png-without-ihdr",
        "jpeg-truncated",
        "webp-unknown-chunk",
        "gif-truncated",
        "pdf",
        "text",
    ],
)
def test_what_cannot_be_read_is_none_and_never_a_guess(data: bytes) -> None:
    """A wrong number is worse than no number.

    Every caller already handles `None` — the measurement fields have been
    optional since P7-01 — so declining costs nothing, while a guess would put a
    fabricated size in the log and refuse a picture for being a shape it is not.
    """
    assert image_dimensions(data) is None


def test_the_magic_bytes_decide_not_the_filename() -> None:
    """A `.png` that is really a JPEG measures correctly rather than not at all.

    Worth pinning because the MIME on an `AttachmentRef` comes from
    `mimetypes.guess_type` in the human-attach path, which reads the *name*.
    """
    assert image_dimensions(jpeg(120, 340)) == (120, 340)


def test_the_measurable_set_gates_what_is_probed() -> None:
    """`IMAGE_MIMES` is what `AttachmentStore.save_bytes` asks before reading.

    Held against the parser rather than restated as a literal: a format added to
    the set with no branch behind it would make the store probe a file it can
    never measure, and a branch added without the entry would measure nothing
    because the caller never asks.
    """
    for build, mime in (
        (png, "image/png"),
        (jpeg, "image/jpeg"),
        (gif, "image/gif"),
        (webp_lossy, "image/webp"),
    ):
        assert mime in IMAGE_MIMES
        assert image_dimensions(build(10, 20)) == (10, 20)  # type: ignore[operator]
    assert "application/pdf" not in IMAGE_MIMES, "a PDF must never reach the JPEG walk"


def test_a_corrupt_jpeg_cannot_walk_the_whole_file() -> None:
    """The bound, and the case that needs it.

    A file that opens like a JPEG and contains no frame header and no scan marker
    walks four bytes at a time — a second of event loop for a 20 MB file, before
    `save_bytes` reaches its thread. Real headers sit within kilobytes of the
    start, so the budget costs nothing and removes the pathological case.

    Asserted as a *duration* rather than by counting iterations, because the
    property is "this cannot become slow" and an iteration count would pass for a
    bound raised to something useless.
    """
    from time import perf_counter

    padding = b"\xff\xe0" + (65533).to_bytes(2, "big") + b"\x00" * 65529
    corrupt = b"\xff\xd8" + padding * 320  # ~21 MB of segments and no frame

    started = perf_counter()
    assert image_dimensions(corrupt) is None
    assert perf_counter() - started < 0.05, "the walk is not bounded by HEADER_BUDGET"
    assert len(corrupt) > HEADER_BUDGET, "the fixture has to be bigger than the budget"
