"""`ph.characters` — cuts that fall between characters, never through one.

The rule is written with the standard library, which has no grapheme
segmentation, so it is held here to the `regex` package's `\\X`, which has.
`regex` is not a ph-core dependency — it reaches the lockfile through
text-index's local model — so the comparison skips where it is missing; the
cuts themselves are checked against characters spelled out by hand.
"""

from __future__ import annotations

import random
import unicodedata
from functools import cache
from itertools import accumulate
from typing import Any

import pytest

from ph.characters import (
    _between_characters,
    capped_text,
    cut_after,
    cut_before,
    head,
    head_bytes,
    pieces,
    stream_decoder,
    tail,
    tail_bytes,
    utf8_size,
)
from ph.testing import EMOJI_FAMILY

FLAG = "\U0001f1fa\U0001f1f8"
THUMB = "\U0001f44d\U0001f3fd"
ACCENTED = "e\u0301"
CONJUNCT = "\u0915\u094d\u0937"
MIXED = " ".join([ACCENTED, EMOJI_FAMILY, FLAG, THUMB, CONJUNCT, "\uac01", "plain"] * 12)
"""Characters of several code points each — an accent, a family, a flag, a skin
tone, an Indic conjunct — between ordinary words, where half the code-point cuts
land inside one."""


@cache
def _clusters() -> Any:  # noqa: ANN401
    """The `regex` package's grapheme-cluster pattern, compiled once: the sweep
    below asks it of a million and a half short texts."""
    return pytest.importorskip("regex").compile(r"\X")


def _characters(text: str) -> list[str]:
    """`text` as Unicode's extended grapheme clusters, by the `regex` package."""
    found: list[str] = _clusters().findall(text)
    return found


def _whole(piece: str, of: str) -> bool:
    """Whether `piece` is made of whole characters of `of`."""
    return set(_characters(piece)) <= set(_characters(of))


def _wrong(text: str) -> list[int]:
    """The places the rule calls between characters that Unicode does not."""
    allowed = set(accumulate(map(len, _characters(text)), initial=0))
    return [at for at in range(1, len(text)) if _between_characters(text, at) and at not in allowed]


def test_no_cut_falls_inside_a_character_as_unicode_defines_one() -> None:
    """A place the rule calls between characters is one UAX #29 does. It may
    refuse a cut Unicode allows — a control, a conjunct joined a little wide, a
    ZWJ never cut after — which only moves a cut one character over.

    Every assigned code point but private use (which is never more than itself
    in any Unicode version), before, after and beside itself and inside an emoji
    sequence; then mixed sequences of the code points the rules turn on, which
    is what catches a rule about a run rather than a pair.

    Sabotage: drop `_after_virama`, or the Thai AM from `_EXTENDING_LETTERS`,
    and a Devanagari conjunct or a Thai syllable is cut in half.
    """
    pytest.importorskip("regex")
    for code in range(0x110000):
        char = chr(code)
        if 0xD800 <= code <= 0xDFFF or unicodedata.category(char) in ("Cn", "Co"):
            continue
        for text in ("a" + char, char + "a", char + char, EMOJI_FAMILY[:2] + char, char + "\u0301"):
            assert not _wrong(text), f"U+{code:04X} in {text!r}"

    pool = [*"aZ 1\t\r\n", *MIXED, "\u0e33", "\u0600", "\u0d4e", "\uff9e", "\u20e3", "\ufe0f"]
    rng = random.Random(7)
    for _ in range(3000):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(1, 16)))
        assert not _wrong(text), [hex(ord(char)) for char in text]


def test_the_cuts_keep_every_character_whole() -> None:
    """What text-index, the decision model's state and an adapter's error are
    cut with. Each piece is whole characters, nothing is lost or duplicated, and
    each stays inside its size — code points, or bytes for `head_bytes` and
    `tail_bytes` — except a single character longer than the size, which is
    kept whole rather than cut through.

    Sabotage: slice by index in `pieces` and a family is split across two chunks.
    """
    for size in (1, 2, 5, 11, 40):
        parts = pieces(MIXED, size)
        assert "".join(parts) == MIXED
        assert all(_whole(part, MIXED) for part in parts)
        assert all(len(part) <= size or len(_characters(part)) == 1 for part in parts)

        first = head(MIXED, size)
        assert MIXED.startswith(first) and len(first) <= size and _whole(first, MIXED)

        opening = head_bytes(MIXED, size * 3)
        assert MIXED.startswith(opening) and len(opening.encode()) <= size * 3
        assert _whole(opening, MIXED)

        last = tail_bytes(MIXED, size * 3)
        assert MIXED.endswith(last) and len(last.encode()) <= size * 3 and _whole(last, MIXED)

    for size in (1, 4, 9, 20):
        parts = pieces(MIXED, size, in_bytes=True)
        assert "".join(parts) == MIXED and all(_whole(part, MIXED) for part in parts)
        assert all(utf8_size(part) <= size or len(_characters(part)) == 1 for part in parts)

    assert cut_before(EMOJI_FAMILY, 3) == 0, "inside the first character: nowhere to cut"
    assert cut_after(EMOJI_FAMILY, 1) == len(EMOJI_FAMILY), "so the whole of it"
    assert pieces(EMOJI_FAMILY, 2) == [EMOJI_FAMILY]
    assert tail_bytes(MIXED, 0) == "" and head_bytes(MIXED, 0) == ""


def test_a_long_line_of_flags_is_cut_from_the_last_cut() -> None:
    """Flags pair off from wherever their run starts, so a rule that counted the
    run from the start of the text recounted a whole line of flags at every
    piece: 1.7 s for 200,000 code points, and four times that for twice as many.
    Counted from the last cut (`floor`), which is between characters, the count
    is the same and the walk is along the line once.

    Sabotage: drop `floor=start` from `pieces` and the pieces still match, at
    that cost.
    """
    line = FLAG * 50_000

    parts = pieces(line, 1_201)

    assert "".join(parts) == line
    assert all(len(part) % 2 == 0 for part in parts), "every flag whole"
    assert {cut_before(line, 1_201, floor=at) for at in (0, 2, 600)} == {1_200}


def test_the_end_of_a_path_is_kept_whole() -> None:
    """`tail` keeps the end — the part of a path that names it — starting between
    characters. A filesystem that stores names decomposed, as macOS does, writes
    "é" as two code points, and a cut by index left the accent at the start."""
    path = "/Users/someone/projects/re\u0301sume\u0301/notes.md"

    for size in range(1, len(path)):
        end = tail(path, size)
        assert path.endswith(end) and len(end) <= size
        assert not end or not unicodedata.combining(end[0]), "no accent without its letter"
    assert tail(path, 0) == "" and tail("short", 40) == "short"


def test_bytes_arriving_in_pieces_join_into_characters() -> None:
    """A stream read in pieces splits characters between reads; decoded piece
    by piece each half is a U+FFFD. One `stream_decoder` per stream joins them."""
    said = "café ✓".encode()
    decoder = stream_decoder()

    text = "".join(decoder.decode(said[at : at + 1]) for at in range(len(said)))

    assert text == "café ✓"
    assert stream_decoder().decode(b"ok \xff") == "ok \ufffd", "bad bytes are still replaced"


def test_a_capped_buffer_leaves_off_only_what_its_cap_cut() -> None:
    """A buffer that kept at most `cap` bytes ended where the cap fell when it is
    full, which may be inside a character — left off. Short of the cap it is all
    the stream wrote, and a bad last byte is the stream's own: shown."""
    said = "ok ✓".encode()

    assert capped_text(said[:-1], len(said) - 1) == "ok ", "full: the cut character left off"
    assert capped_text(said, 64) == "ok ✓"
    assert capped_text(said[:-1], 64) == "ok \ufffd", "short of the cap: the stream's own"


def test_a_byte_budget_measures_any_string() -> None:
    """A guest's `print` can hand over a lone surrogate, which a strict encode
    refuses. The byte budgets count it as the three bytes it takes and cut on."""
    text = "a\ud800b" * 20

    assert len(head_bytes(text, 10).encode("utf-8", "surrogatepass")) <= 10
    assert len(tail_bytes(text, 10).encode("utf-8", "surrogatepass")) <= 10
    assert head_bytes(text, 10) == "a\ud800ba\ud800b"
    assert utf8_size(text) == len(text.encode("utf-8", "surrogatepass")) and utf8_size("ab") == 2
