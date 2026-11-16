"""The cuts `ph_runtime` needs from `ph.characters` — a copy, pinned by a test.

The guest bounds what a cell prints by bytes, and a bound that cuts text has to
cut between characters: an accent left without its letter, half an emoji
sequence, a byte sequence broken in two. `ph.characters` is that rule, and this
package ships into the guest venv with `dill` as its only dependency, so it
cannot import it — the module lives in the ph-core wheel.

**Copied whole, compared character for character below this docstring by
`test_protocol_mirror.py`** — the arrangement `_json.py` has, taken a step
further because the rule here is tables as well as functions. When the test
fails, copy `ph/characters.py` across again, below this docstring.

Private: this is ph-core's code, not the guest's API.

@module ph_runtime._characters
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
"""The joiners, and the letters UAX #29 extends with all the same: Thai and Lao
AM, which are spacing marks to it, and the half-width katakana sound marks."""

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
"""UAX #29's Prepend: the marks that join the character *after* them."""


def _extends(char: str) -> bool:
    """Whether `char` belongs to the character before it rather than starting
    one: a combining, enclosing or spacing mark, a joiner, a variation selector,
    a skin-tone modifier or an emoji tag — UAX #29's Extend, SpacingMark and ZWJ
    by the properties the standard library can see."""
    code = ord(char)
    return (
        unicodedata.category(char) in ("Mn", "Me", "Mc")
        or char in _EXTENDING_LETTERS
        or 0x1F3FB <= code <= 0x1F3FF
        or 0xE0020 <= code <= 0xE007F
    )


def _regional(char: str) -> bool:
    """A regional indicator: two of them are one flag."""
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _hangul(char: str) -> str:
    """A Hangul jamo or syllable's part in a syllable block, or `""`."""
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
    """Whether the run of marks just before `at` holds a virama: a conjunct, which
    joins the consonant that follows across any marks between (UAX #29's GB9c)."""
    index = at - 1
    while index >= 0 and _extends(text[index]):
        if unicodedata.combining(text[index]) == 9:
            return True
        index -= 1
    return False


def _between_characters(text: str, at: int, floor: int = 0) -> bool:
    """Whether a cut at `at` falls between two characters as a reader sees them.

    The ends of the text always do. Otherwise UAX #29's rules in its order, less
    the ones that only ever allow a cut: CR LF is one; a mark, joiner or
    modifier joins what is before it; a prepended mark joins what follows; what
    follows a ZWJ or a virama joins it (an emoji sequence, an Indic conjunct —
    wider than UAX #29, which is the safe side for a cut); jamo join into a
    syllable; and regional indicators pair off into flags, counted from `floor`
    — a place known to fall between characters, so a walk along a long line of
    flags does not count them all again at every cut.
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
        run = 0
        while at - 1 - run >= floor and _regional(text[at - 1 - run]):
            run += 1
        return run % 2 == 0
    return True


def cut_before(text: str, at: int, *, floor: int = 0) -> int:
    """The last place at or before `at` that falls between characters, and no
    earlier than `floor` — itself such a place, `0` by default. `floor` when the
    character there runs past `at`: a caller that must make progress takes
    `cut_after(text, floor + 1)`, the whole of that character, instead."""
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
    between characters: the end of a path, say, which is the part that names it."""
    return text[cut_after(text, len(text) - size) :]


def utf8_size(text: str) -> int:
    """How many bytes `text` is in UTF-8 — a lone surrogate counted as the three
    it takes (`surrogatepass`), so any `str` can be measured: a guest's `print`
    can hand over one, and a strict encode would raise. ASCII, the common case,
    is its own length and is not encoded at all."""
    return len(text) if text.isascii() else len(text.encode("utf-8", "surrogatepass"))


def _fitting(text: str, room: int) -> int:
    """How many code points from the start of `text` fit in `room` bytes of
    UTF-8, measured as `utf8_size` measures. Only the first `room` code points
    are looked at — no code point is less than a byte, so they hold the bytes
    `room` covers — and a code point the byte cut went through is left
    undecoded, which is what the decoder does with an unfinished one."""
    raw = text[:room].encode("utf-8", "surrogatepass")[:room]
    return len(codecs.utf_8_decode(raw, "surrogatepass", False)[0])


def head_bytes(text: str, room: int) -> str:
    """As much of the start of `text` as is at most `room` bytes of UTF-8,
    ending between characters."""
    return head(text, _fitting(text, room))


def tail_bytes(text: str, room: int) -> str:
    """As much of the end of `text` as is at most `room` bytes of UTF-8,
    starting between characters: what a byte budget leaves of a message whose
    end is the part that matters. Measured on the last `room` code points
    reversed, since a code point's bytes do not depend on which way it is read."""
    fits = _fitting(text[: -room - 1 : -1], room) if room > 0 else 0
    return text[cut_after(text, len(text) - fits) :]


def pieces(text: str, size: int, *, in_bytes: bool = False) -> list[str]:
    """`text` in pieces of at most `size` code points — or UTF-8 bytes,
    `in_bytes` — each cut between characters. A character longer than `size` — a
    base under a run of marks — is a piece of its own rather than cut through.

    Each cut is looked for from the last one, which is between characters, so a
    long line costs a walk along it and not a walk back to its start per piece."""
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
    """A UTF-8 decoder for bytes that arrive in pieces: a character split between
    two of them is held back until the rest arrives, and bytes that are not UTF-8
    are replaced. Its `decode(piece)` per read; `decode(b"", final=True)` at the
    end gives what an unfinished character left."""
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


def capped_text(kept: bytes | bytearray, cap: int) -> str:
    """What a buffer kept of a byte stream, as text, where it kept at most `cap`
    bytes and threw the rest away. Full, it ended where the cap fell, which may
    be inside a character: that one is left off rather than shown as U+FFFD,
    since what followed is gone. Short of the cap it is all there was, and a bad
    last byte is the stream's own and shown. Decoded in place, with no copy of
    the buffer — a subprocess's is megabytes."""
    if len(kept) >= cap:
        return codecs.utf_8_decode(kept, "replace", False)[0]
    return kept.decode("utf-8", "replace")
