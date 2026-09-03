"""An image's pixel dimensions, from its header and nothing else (P7-03).

**Reading a header is not decoding an image**, and conflating the two is what
kept `AttachmentRef.width` empty until now: its own docstring said measuring an
image "means decoding it, and the decoder is exactly the dependency
`media-transform` (P7-02) exists to keep optional". That is true of resizing and
false of measuring. Every format pH accepts as an image writes its dimensions in
the first few dozen bytes, in fixed positions, as plain integers — so this is
`int.from_bytes` four times and no dependency at all: no Pillow, no ImageMagick,
no `ffmpeg`, nothing to install and nothing to sandbox.

What it buys, now that something reads it: a route can say how many pixels it can
actually use, and a person who attaches a 4000-pixel screenshot to a model that
will scale it to 1568 can be *told* — rather than paying for the upload on every
turn of the session and never learning why. `ph.llm.media` is where that judgement
lives; this file only answers the measurement.

**Never raises and never guesses.** A truncated file, a format not listed here,
or bytes that are not an image at all all return `None`, which every caller
already handles — `AttachmentRef`'s measurement fields have been optional since
P7-01 and the token estimate falls back to a flat figure. A wrong number would be
worse than no number, so anything this file is not certain of, it declines.

Deliberately not `imghdr`: it was removed in Python 3.13, and it answers "what
format is this" rather than "how big is it".

@module ph.llm.dimensions
"""

from __future__ import annotations

__all__ = ["HEADER_BUDGET", "IMAGE_MIMES", "image_dimensions"]

HEADER_BUDGET = 1 << 20
"""How far into a file a dimension may be looked for.

Only JPEG can need a search at all — every other format here writes its size at a
fixed offset — and a megabyte is orders of magnitude past any real frame header.
It bounds the pathological case instead: bytes that open like a JPEG and contain
no frame."""

IMAGE_MIMES: frozenset[str] = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
"""The formats this file can measure.

`AttachmentStore.save_bytes` gates on it rather than probing every blob, which is
what keeps this an answer a caller can act on instead of an affordance nobody
uses — and what keeps a 40 MB video out of the JPEG walk below."""

_SOF_MARKERS: frozenset[int] = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
"""JPEG start-of-frame markers, which are the ones carrying the frame size.

`0xC4`, `0xC8` and `0xCC` sit inside this range and are *not* frame headers —
they are the Huffman table, an extension, and the arithmetic-coding table. Taking
the range wholesale reads two bytes of a Huffman table as a picture's height."""


def _png(data: bytes) -> tuple[int, int] | None:
    # IHDR is mandated to be the first chunk, so its width and height are always
    # at these offsets: 8 signature + 4 length + 4 type.
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _gif(data: bytes) -> tuple[int, int] | None:
    # The logical screen descriptor, little-endian, immediately after the magic.
    if len(data) < 10:
        return None
    return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")


def _webp(data: bytes) -> tuple[int, int] | None:
    """The three WebP chunk layouts, which do not agree with each other.

    A container that says `WEBP` can hold lossy (`VP8 `), lossless (`VP8L`) or
    extended (`VP8X`, which is what an animation or an image with alpha gets), and
    each writes its size differently — different offsets, different widths,
    different biases. Reading one layout and assuming the others match is how a
    1920-pixel animation measures as 14 pixels.
    """
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Lossy: a 3-byte start code, then 14-bit width and height.
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        # Lossless: one signature byte, then both sizes packed into 32 bits as
        # 14 bits each, biased by one.
        if data[20] != 0x2F:
            return None
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        # Extended: 24-bit canvas sizes, biased by one.
        return (
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    return None


def _jpeg(data: bytes) -> tuple[int, int] | None:
    """Walk the marker segments to the frame header.

    JPEG is the one format here without a fixed offset: the frame header follows
    however many application and quantization segments the encoder wrote, so it
    has to be walked. Bounded three ways — the loop cannot pass the end, a segment
    length below its own two bytes is refused rather than looped on, and the scan
    stops at start-of-scan, after which the bytes are compressed image data that
    would resynchronize on anything.
    """
    # Bounded by a header budget, not by the file. A real frame header is within
    # a few kilobytes of the start; a *corrupt* JPEG with no frame and no scan
    # marker would otherwise walk to the end four bytes at a time — 1.1 s for a
    # 20 MB file, on the event loop, before `save_bytes` even reaches its thread.
    end = min(len(data), HEADER_BUDGET)
    index = 2
    while index + 9 <= end:
        if data[index] != 0xFF:
            return None
        marker = data[index + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            index += 2  # standalone markers carry no length
            continue
        if marker == 0xDA:
            return None  # entropy-coded data begins; a frame header cannot follow
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        if length < 2:
            return None
        if marker in _SOF_MARKERS:
            # precision(1) then height(2) then width(2) — height first, which is
            # the opposite of every other format in this file.
            return (
                int.from_bytes(data[index + 7 : index + 9], "big"),
                int.from_bytes(data[index + 5 : index + 7], "big"),
            )
        index += 2 + length
    return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """`(width, height)` in pixels, or `None` when they cannot be read.

    Dispatched on the magic bytes rather than on a declared MIME: the MIME on an
    `AttachmentRef` comes from a filename in the human-attach path, and a `.png`
    that is really a JPEG should measure correctly rather than not at all.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _png(data)
    if data.startswith(b"\xff\xd8\xff"):
        return _jpeg(data)
    if data.startswith((b"GIF87a", b"GIF89a")):
        return _gif(data)
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return _webp(data)
    return None
