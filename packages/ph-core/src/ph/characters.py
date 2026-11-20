"""Cutting text between characters, never through one.

A cut that lands inside a character leaves half of it on each side: an accent
without its letter, a skin tone without its hand, one letter of a flag, a family
split into its members. Python indices count code points, so slicing never
breaks a byte sequence. But a slice can still land inside one of these
characters, which are several code points each. Half of the cut points in such
text do.

So every cut that sizes text for a reader or a model goes through `cut_before`
and `cut_after`. They stop only between characters as a reader sees them: the
extended grapheme clusters of Unicode (UAX #29). The helper to use depends on
what the caller holds:

* **All of the text.** `head`, `tail` and `pieces` cut by code points, and
  `head_bytes` and `tail_bytes` cut by a byte budget. They see what follows a
  cut, so a cut never leaves an accent or a modifier on the wrong side of it.
* **Only the first bytes of the text**, in a buffer whose byte cap discarded the
  rest. `capped_text` reads such a buffer. What followed is gone, so it can
  promise a whole code point and no more.
* **A byte count.** `utf8_size` is the one measure that every byte cap here
  uses.
* **Bytes that arrive in pieces.** `stream_decoder` joins a character that is
  split between two reads, so that it is not shown as two U+FFFDs.

`ph_runtime` keeps a copy of this module, because the guest may not import
ph-core. `test_protocol_mirror.py` holds the copy equal to this source.

**The module is written with the standard library**, which has no grapheme
segmentation. It errs in one direction only. It may refuse a cut that UAX #29
allows: it never breaks at a control, it joins a little wide after a virama, and
it never cuts after a ZWJ. It never allows a cut that UAX #29 refuses.
`test_characters.py` holds it to the `\\X` of the `regex` package on that. This
is its own module and not part of `ph.text`, because it is a Unicode property
table with a version behind it, and not prose formatting.

@module ph.characters
"""

from __future__ import annotations

import codecs
import unicodedata

__all__ = [
    "capped_text",
    "cut_after",
    "cut_before",
    "head",
    "head_bytes",
    "pieces",
    "stream_decoder",
    "tail",
    "tail_bytes",
    "utf8_size",
]

_ZWJ = "‍"
_ZWNJ = "‌"
_EXTENDING_LETTERS = frozenset({_ZWJ, _ZWNJ, "ำ", "ຳ", "ﾞ", "ﾟ"})
"""The joiners, and the letters that UAX #29 also treats as extending: Thai and
Lao AM, which are spacing marks to it, and the half-width katakana sound
marks."""

_PREPEND = frozenset(
    map(
        chr,
        (
            *range(0x0600, 0x0606),
            0x06DD,
            0x070F,
            0x0890,
            0x0891,
            0x08E2,
            0x0D4E,
            0x110BD,
            0x110CD,
            0x111C2,
            0x111C3,
            0x113D1,
            0x1193F,
            0x11941,
            0x11A3A,
            *range(0x11A84, 0x11A8A),
            0x11D46,
            0x11F02,
        ),
    )
)
"""The Prepend class of UAX #29: the marks that join the character *after* them."""


def _extends(char: str) -> bool:
    """Whether `char` belongs to the character before it and does not start
    one: a combining, enclosing or spacing mark, a joiner, a variation selector,
    a skin-tone modifier or an emoji tag. These are the Extend, SpacingMark and
    ZWJ classes of UAX #29, by the properties that the standard library can
    see."""
    code = ord(char)
    return (
        unicodedata.category(char) in ("Mn", "Me", "Mc")
        or char in _EXTENDING_LETTERS
        or 0x1F3FB <= code <= 0x1F3FF
        or 0xE0020 <= code <= 0xE007F
    )


_REGIONALS = "".join(map(chr, range(0x1F1E6, 0x1F200)))
"""Every regional indicator, for counting a run of them in one C-level strip."""


def _regional(char: str) -> bool:
    """A regional indicator: two of them are one flag."""
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _hangul(char: str) -> str:
    """The part that a Hangul jamo or syllable takes in a syllable block, or `""`."""
    code = ord(char)
    if 0x1100 <= code <= 0x115F or 0xA960 <= code <= 0xA97C:
        return "L"
    if 0x1160 <= code <= 0x11A7 or 0xD7B0 <= code <= 0xD7C6:
        return "V"
    if 0x11A8 <= code <= 0x11FF or 0xD7CB <= code <= 0xD7FB:
        return "T"
    if 0xAC00 <= code <= 0xD7A3:
        return "LV" if (code - 0xAC00) % 28 == 0 else "LVT"
    return ""


_JAMO_JOINS = frozenset(
    {
        ("L", "L"),
        ("L", "V"),
        ("L", "LV"),
        ("L", "LVT"),
        ("LV", "V"),
        ("LV", "T"),
        ("V", "V"),
        ("V", "T"),
        ("LVT", "T"),
        ("T", "T"),
    }
)


def _after_virama(text: str, at: int) -> bool:
    """Whether the run of marks just before `at` holds a virama. A virama makes
    a conjunct: it joins the consonant that follows, across any marks between
    them (GB9c of UAX #29)."""
    index = at - 1
    while index >= 0 and _extends(text[index]):
        if unicodedata.combining(text[index]) == 9:
            return True
        index -= 1
    return False


def _between_characters(text: str, at: int, floor: int = 0) -> bool:
    """Whether a cut at `at` falls between two characters as a reader sees them.

    The ends of the text always do. Otherwise the rules of UAX #29 apply, in its
    order, without the rules that only ever allow a cut:

    * CR LF is one character.
    * A mark, joiner or modifier joins what is before it.
    * A prepended mark joins what follows.
    * What follows a ZWJ or a virama joins it (an emoji sequence, an Indic
      conjunct). This is wider than UAX #29, which is the safe side for a cut.
    * Jamo join into a syllable.
    * Regional indicators pair off into flags. The pairs are counted from
      `floor`, a place that is known to fall between characters. A walk along a
      long line of flags therefore does not count them all again at every cut.
    """
    if at <= floor or at >= len(text):
        return True
    before, after = text[at - 1], text[at]
    if before == "\r" and after == "\n":
        return False
    if _extends(after) or before in _PREPEND:
        return False
    if before == _ZWJ or _after_virama(text, at):
        return False
    if (_hangul(before), _hangul(after)) in _JAMO_JOINS:
        return False
    if _regional(before) and _regional(after):
        # The run before `at`, back to `floor`, is counted by `rstrip` and not
        # by a Python loop. A cut inside a long line of flags walked all of it.
        since = text[floor:at]
        return (len(since) - len(since.rstrip(_REGIONALS))) % 2 == 0
    return True


def cut_before(text: str, at: int, *, floor: int = 0) -> int:
    """The last place at or before `at` that falls between characters, and no
    earlier than `floor`. `floor` is itself such a place, and it is `0` by
    default. The answer is `floor` when the character there runs past `at`. A
    caller that must make progress then takes `cut_after(text, floor + 1)`,
    which is the whole of that character."""
    at = max(floor, min(at, len(text)))
    while not _between_characters(text, at, floor):
        at -= 1
    return at


def cut_after(text: str, at: int, *, floor: int = 0) -> int:
    """The first place at or after `at` that falls between characters, with
    `floor` as for `cut_before`."""
    at = max(floor, min(at, len(text)))
    while not _between_characters(text, at, floor):
        at += 1
    return at


def head(text: str, size: int) -> str:
    """As much of the start of `text` as is at most `size` code points, ending
    between characters. Empty when the first character alone is longer."""
    return text[: cut_before(text, size)]


def tail(text: str, size: int) -> str:
    """As much of the end of `text` as is at most `size` code points, starting
    between characters. An example is the end of a path, which is the part that
    names it."""
    return text[cut_after(text, len(text) - size) :]


def utf8_size(text: str) -> int:
    """How many bytes `text` is in UTF-8. A lone surrogate is counted as the
    three bytes that it takes (`surrogatepass`), so any `str` can be measured:
    the `print` of a guest can pass one, and a strict encode would raise. ASCII
    is the common case. It is its own length and is not encoded at all."""
    return len(text) if text.isascii() else len(text.encode("utf-8", "surrogatepass"))


def _fitting(text: str, room: int) -> int:
    """How many code points from the start of `text` fit in `room` bytes of
    UTF-8, measured as `utf8_size` measures. Only the first `room` code points
    are looked at. No code point is less than a byte, so they hold the bytes
    that `room` covers. A code point that the byte cut went through is left
    undecoded, as the decoder leaves an unfinished one."""
    raw = text[:room].encode("utf-8", "surrogatepass")[:room]
    return len(codecs.utf_8_decode(raw, "surrogatepass", False)[0])


def head_bytes(text: str, room: int) -> str:
    """As much of the start of `text` as is at most `room` bytes of UTF-8,
    ending between characters."""
    return head(text, _fitting(text, room))


def tail_bytes(text: str, room: int) -> str:
    """As much of the end of `text` as is at most `room` bytes of UTF-8,
    starting between characters. This is what a byte budget leaves of a message
    whose end is the part that matters. The measure is taken on the last `room`
    code points, reversed, because the bytes of a code point do not depend on
    the direction in which it is read."""
    fits = _fitting(text[: -room - 1 : -1], room) if room > 0 else 0
    return text[cut_after(text, len(text) - fits) :]


def pieces(text: str, size: int, *, in_bytes: bool = False) -> list[str]:
    """`text` in pieces of at most `size` code points, or UTF-8 bytes with
    `in_bytes`, each cut between characters. A character longer than `size` (a
    base under a run of marks) is a piece of its own and is not cut through.

    Each cut is looked for from the last cut, which is between characters. A
    long line therefore costs one walk along it, and not a walk back to its
    start for each piece."""
    found: list[str] = []
    start = 0
    while start < len(text):
        room = _fitting(text[start : start + size], size) if in_bytes else size
        end = cut_before(text, start + room, floor=start)
        if end <= start:
            end = cut_after(text, start + 1, floor=start)
        found.append(text[start:end])
        start = end
    return found


def stream_decoder() -> codecs.IncrementalDecoder:
    """A UTF-8 decoder for bytes that arrive in pieces. A character that is
    split between two pieces is held back until the rest arrives, and bytes that
    are not UTF-8 are replaced. Call its `decode(piece)` for each read.
    `decode(b"", final=True)` at the end gives what an unfinished character
    left."""
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def capped_text(kept: bytes | bytearray, cap: int) -> str:
    """What a buffer kept of a byte stream, as text, where it kept at most `cap`
    bytes and discarded the rest. A full buffer ended where the cap fell, which
    may be inside a character. That character is left off and is not shown as
    U+FFFD, because what followed is gone. A buffer short of the cap holds all
    that there was, and a bad last byte belongs to the stream and is shown. The
    buffer is decoded in place, with no copy, because the buffer of a subprocess
    is megabytes."""
    if len(kept) >= cap:
        return codecs.utf_8_decode(kept, "replace", False)[0]
    return kept.decode("utf-8", "replace")
