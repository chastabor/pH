"""Cutting a document into passages that keep their place in it.

**A chunk that does not know its line range is nearly useless here.** The purpose
of this row is to give an agent a pointer that it can act on, such as
`read docs/x.md offset 120`, and not a wall of prose that it must then locate.
Every chunk therefore carries the 1-based line span that it came from. The
boundary rule is chosen to keep that span meaningful: cuts are made between
paragraphs, never in the middle of a sentence, so a span names something that a
person would recognize as a unit.

The unit is the paragraph, not a fixed character stride, for the same reason. A
stride is simpler, and most naive indexers use one. It reliably splits the one
sentence that answers the query across two chunks, so neither chunk retrieves
it. A blank line is where the author of a document already said "new idea", in
Markdown, in plain text, in reStructuredText, and in code comments.

The overlap exists because that rule is imperfect. A paragraph whose meaning
depends on the paragraph before it is common. Carrying the end of the previous
chunk forward is therefore worth the duplicated tokens.

**Two bounds, in two units.** `max_chars` is how a profile says how large a
passage should be. It is not what an embedder counts. A model reads a limited
number of tokens. A character is a quarter of a token in English prose, a third
of a token in code, and one token or more in a script that takes three bytes for
each character. A caller that knows what its embedder counts therefore passes a
`Weight`: what a text weighs, and the most that a chunk may weigh. Every cut here
keeps to both bounds. Each chunk is then as large as it may be: a table in a page
of prose is cut small, and the prose around it is not.

@module ph_text_index._chunk
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from ph.characters import cut_after, cut_before, pieces
from ph.indexable import Paragraph, paragraphs

__all__ = ["Chunk", "Weight", "chunk_paragraphs", "chunk_text", "head_of"]

_LAST_WORD = re.compile(r"\s+\S*\Z")
"""The last run of whitespace in a text, and the word that begins after it."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """One passage, and where in the file it was."""

    text: str
    start_line: int
    """1-based, inclusive. The `offset` of `read` is one less than this."""
    end_line: int
    """1-based, inclusive."""


@dataclass(slots=True)
class Weight:
    """A second bound on a chunk, in a unit that the caller chooses. For the text
    index, the unit is the tokens of its embedder. See the module docstring."""

    of: Callable[[Sequence[str]], Sequence[float]]
    """What each text weighs by itself. It is called with one batch at a time (the
    paragraphs of a document, or the lines of a paragraph). The function that
    counts is a tokenizer, and a call to a tokenizer costs more than the
    counting."""
    most: float
    """The most a chunk may weigh."""
    bound: bool = False
    """Set when this weight, and not `max_chars`, ended a chunk or split a
    paragraph. The caller reads it after packing, so that it can report that."""


@dataclass(frozen=True, slots=True)
class _Bounds:
    """Both bounds on a chunk, and what joining two pieces adds to its weight."""

    chars: int
    weight: Weight | None
    joint: float
    """The weight of a separator, as a careful estimate. It is never under one. A
    tokenizer that drops a blank line when it counts the line alone may keep it
    when the line stands between two words."""

    @classmethod
    def of(cls, max_chars: int, weight: Weight | None) -> _Bounds:
        if weight is None:
            return cls(max_chars, None, 0.0)
        return cls(max_chars, weight, max(1.0, weight.of(["\n\n"])[0]))

    def weigh(self, texts: Sequence[str]) -> Sequence[float]:
        return [0.0] * len(texts) if self.weight is None else self.weight.of(texts)

    def past(self, chars: int, heavy: float) -> bool:
        """Whether a piece of this size is past either bound. It records when the
        weight alone was the reason."""
        if chars > self.chars:
            return True
        if self.weight is not None and heavy > self.weight.most:
            self.weight.bound = True
            return True
        return False

    def spare(self, piece: _Block) -> float:
        """How much weight a chunk that will hold `piece` has for the blocks that
        are carried into it. This is what `piece` leaves, and never more than a
        quarter of a chunk. A bound that leaves room for one sentence then does
        not fill that room with the last sentence of the chunk before it."""
        if self.weight is None:
            return math.inf
        most = self.weight.most
        return min(most / 4, most - piece.weight - self.joint)


@dataclass(frozen=True, slots=True)
class _Block:
    """A run of non-blank lines: the smallest unit of the document itself."""

    text: str
    start_line: int
    end_line: int
    weight: float = 0.0
    """What it weighs, when a `Weight` is given. Zero when none is given."""

    def __len__(self) -> int:
        return len(self.text)


def _fitting(text: str, heavy: float, weight: Weight, least: int) -> tuple[int, float]:
    """How many code points at the start of `text` weigh no more than
    `weight.most`, and what they weigh. `text` weighs `heavy`. The answer is all
    of `text` when all of it fits.

    The text is cut in proportion and then weighed, because weight is not even
    along a line. The start of a line that is half prose and half base64 does not
    weigh what the average of the line predicts. It is then cut again, smaller,
    until it fits. Every cut is between characters
    (`ph.characters.cut_before`), so an accent is never on one side of a cut and
    its letter on the other. The cut goes no smaller than one whole character:
    `least`, where the first character ends. `text` itself must end between
    characters: it is all of a line, or a stretch of a line that `cut_after`
    cut."""
    size, own = len(text), heavy
    while own > weight.most and size > least:
        size = max(least, cut_before(text, int(size * weight.most / own)))
        [own] = weight.of([text[:size]])
    return size, own


def _lightened(text: str, heavy: float, at: int, weight: Weight) -> Iterator[_Block]:
    """`text`, which weighs `heavy`, in pieces no heavier than `weight.most`. Each
    piece is as much of the remaining text as fits (`_fitting`).

    **Each cut is aimed by the stretch just ahead of it**, which is twice the
    length of the last piece, and not by all the remaining text. When the rest of
    the line was weighed again at every cut, the tokenizer made one pass over the
    line for each piece: 18 s for one 100,000-character line of CJK. On a line
    whose weight is not even, the stretch just ahead is also the better guide to
    where the next cut falls. A stretch that fits whole is taken whole, and the
    next stretch is aimed by it."""
    ahead, heavy_ahead = len(text), heavy
    first = cut_after(text, 1)
    while first < len(text) and (ahead < len(text) or heavy_ahead > weight.most):
        size, own = _fitting(text[:ahead], heavy_ahead, weight, first)
        yield _Block(text[:size], at, at, own)
        text = text[size:]
        ahead = cut_after(text, 2 * size)
        first = cut_after(text, 1)
        [heavy_ahead] = weight.of([text[:ahead]]) if text else [0.0]
    if text:
        # All that is left, so what it weighs is known.
        yield _Block(text, at, at, heavy_ahead)


def _of_one_line(one: _Block, other: _Block) -> bool:
    """Whether two blocks are pieces of one line that was cut inside itself.
    Blocks are joined by a blank line, and a line must not get a blank line that
    it never had. Such pieces therefore stay in chunks of their own."""
    return one.start_line == one.end_line == other.start_line == other.end_line


def head_of(text: str, weight: Weight) -> str:
    """As much of the start of `text` as weighs no more than `weight.most`. That
    is all of `text` when it fits. Otherwise `text` is cut where the weight runs
    out, and then back to the end of a word if there is one.

    This is for a text that is one unit and is wanted as nearly whole as
    possible, such as a question. `chunk_paragraphs` is for a text that is wanted
    in parts. If a head were packed as paragraphs, it would stop at the first
    paragraph that the next paragraph did not fit after. For a title and a page,
    that is the title."""
    [heavy] = weight.of([text])
    size, _ = _fitting(text, heavy, weight, cut_after(text, 1))
    if size == len(text):
        return text
    first = text[:size]
    if not text[size].isspace():
        # The cut is inside a word, and a piece of a word is a token that the
        # whole word was not.
        first = _LAST_WORD.sub("", first) or first
    return first.rstrip()


def _split_line(line: str, heavy: float, at: int, bounds: _Bounds) -> list[_Block]:
    """One line that is too long to be a chunk, cut on characters (X5).

    **This is the last resort, and it costs the exact span that cutting by lines
    otherwise protects.** A cut by lines keeps the line range of every chunk
    true, and a pointer into a 400-line file needs that. But a line is only a
    unit when something wrote it as one. A minified bundle, a base64 blob and a
    long CSV row are each one "line" of thousands of characters. For them the
    line-only rule was not a bound at all: `max_chars` held for the number of
    lines packed together and not for the chunk. A single line therefore went to
    the embedder whole and was silently truncated there. That is the failure of
    X4, by the route that X4 did not cover.

    The cost is that the pieces all claim the same line. That cost is right
    here. A pointer that says "line 812" has value when line 812 is a sentence,
    and has none when it is 40 KB of minified JavaScript. The span is approximate
    for exactly those lines, so the precision that is lost was already notional.

    The line is cut to `max_chars` first, between characters
    (`ph.characters.pieces`). Each piece that is still too heavy is cut again
    (`_lightened`). `heavy` is what the line weighs. It is the weight of the one
    piece of a line that `max_chars` does not cut.
    """
    parts = pieces(line, bounds.chars)
    if bounds.weight is None:
        return [_Block(part, at, at) for part in parts]
    weights = [heavy] if len(parts) == 1 else bounds.weigh(parts)
    return [
        block
        for part, own in zip(parts, weights, strict=True)
        for block in _lightened(part, own, at, bounds.weight)
    ]


def _split_block(block: _Block, bounds: _Bounds) -> list[_Block]:
    """A paragraph that is longer than one chunk, cut on line boundaries.

    Such a paragraph is a code block, a table, or a wrapped-at-nothing paragraph
    of prose. It is cut by lines and not by characters, so that the span stays
    exact. A character cut would leave two chunks that claim the same line. A
    model does not trust a pointer a second time after it was off by one line in
    a 400-line file. A line that is *itself* over a bound has no such cut to
    make, and goes to `_split_line` (X5).
    """
    if not bounds.past(len(block), block.weight):
        return [block]
    pieces: list[_Block] = []
    lines = block.text.split("\n")
    # A paragraph of one line was weighed with the document's paragraphs.
    weights = [block.weight] if len(lines) == 1 else bounds.weigh(lines)
    held: list[str] = []
    chars, heavy = 0, 0.0
    start = block.start_line
    for offset, (line, own) in enumerate(zip(lines, weights, strict=True)):
        at = block.start_line + offset
        # The size of `held` with this line added at its end, and a newline
        # between.
        if held and bounds.past(chars + 1 + len(line), heavy + bounds.joint + own):
            pieces.append(_Block("\n".join(held), start, at - 1, heavy))
            held, chars, heavy = [], 0, 0.0
            start = at
        if bounds.past(len(line), own):
            # `held` is already empty. This line alone is over a bound, so the
            # flush above it certainly ran.
            pieces.extend(_split_line(line, own, at, bounds))
            start = at + 1
            continue
        if held:
            chars, heavy = chars + 1 + len(line), heavy + bounds.joint + own
        else:
            chars, heavy = len(line), own
        held.append(line)
    if held:
        pieces.append(_Block("\n".join(held), start, block.end_line, heavy))
    return pieces


def _tail(blocks: list[_Block], overlap: int, spare: float, joint: float) -> list[_Block]:
    """The trailing blocks that are worth carrying into the next chunk, in order.

    The result is never the whole chunk. A chunk that carried all of itself
    forward would make the next chunk identical whenever a single block filled
    it, and the walk would not advance.

    **`overlap` is a width in the same unit that `chunk_text` measures in**: each
    block plus its two separator characters. The caller can therefore pass a
    budget, not a wish, and get back something that fits. The first attempt at
    X4 left this function summing raw `len(block)`, and put a second trimming
    loop in the caller that used `width()`. That made two owners of one decision
    that counted differently, and the next edit to either would make them drift.
    A budget of zero or less gives `[]`, which is the guarantee that the loop
    provided.

    `spare` is the same budget by weight (`_Bounds.spare`). Each block is counted
    with the `joint` that follows it.
    """
    if overlap <= 0 or len(blocks) < 2:
        return []
    carried: list[_Block] = []
    total, heavy = 0, 0.0
    for block in reversed(blocks[1:]):
        total += len(block) + 2
        heavy += block.weight + joint
        if total > overlap or heavy > spare:
            break
        carried.append(block)
    carried.reverse()
    return carried


def chunk_text(text: str, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """`chunk_paragraphs` over every paragraph of `text`, with no policy applied.

    This is the direct entry to the packer, for a caller that has no opinion
    about what is worth indexing. The text index is not that caller. It asks
    `ph.indexable.triage` first and passes only what survived. That is why this
    module does not drop machine output (X6 review). For a short time it did, and
    a packer whose tests had to know indexing policy was the sign that the job
    was in the wrong place.
    """
    return chunk_paragraphs(paragraphs(text), max_chars=max_chars, overlap_chars=overlap_chars)


def chunk_paragraphs(
    blocks: Sequence[Paragraph],
    *,
    max_chars: int,
    overlap_chars: int,
    weight: Weight | None = None,
) -> list[Chunk]:
    """Pack `blocks` into chunks of at most `max_chars`, each with its line span.
    When a `weight` is given, no chunk is heavier than it allows.

    The packing is greedy. Blocks accumulate until the next block would overflow.
    The chunk is then emitted, and the next chunk starts with the carry-over from
    `_tail`. A single block that is past a bound is split first, so both bounds
    hold for every chunk and not only for most of them.

    **The weight of a chunk is the sum of the weights of its blocks, and a
    `joint` between each of them.** That sum is exact where weight is bytes, and
    a careful estimate where it is the count of a tokenizer. The caller that must
    not be over the bound therefore checks what it sends. This function gets it
    there in one cut.

    This function blocks where `weight.of` blocks. With the `weight.of` of a
    tokenizer, call it in a worker thread.

    **The carry-over is a budget, not a wish** (X4). Before, the carried tail was
    never checked again against the block that caused the flush, so a chunk could
    reach `overlap_chars + max_chars`. The input limit of an embedder is a hard
    limit. A chunk over it is silently truncated at the far end, which indexes
    text that the search can never match. `_tail` now receives what is really
    left. The overlap therefore gives way to the bound, in the one function that
    owns how much is carried.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if weight is not None and weight.most <= 0:
        raise ValueError("weight.most must be positive")
    bounds = _Bounds.of(max_chars, weight)
    packed: list[_Block] = []
    weights = bounds.weigh([block.text for block in blocks])
    for block, own in zip(blocks, weights, strict=True):
        packed.extend(
            _split_block(_Block(block.text, block.start_line, block.end_line, own), bounds)
        )
    chunks: list[Chunk] = []
    held: list[_Block] = []

    def flush() -> None:
        if held:
            chunks.append(
                Chunk(
                    text="\n\n".join(one.text for one in held),
                    start_line=held[0].start_line,
                    end_line=held[-1].end_line,
                )
            )

    def measured(blocks: list[_Block]) -> tuple[int, float]:
        """What the joined blocks would occupy and weigh, separators included."""
        return (
            sum(len(one) + 2 for one in blocks),
            sum(one.weight + bounds.joint for one in blocks),
        )

    held_width, held_weight = 0, 0.0
    for piece in packed:
        same_line = bool(held) and _of_one_line(held[-1], piece)
        if same_line or (held and bounds.past(held_width + len(piece), held_weight + piece.weight)):
            flush()
            # The carry-over gets what is left after the block that caused the
            # flush, so it cannot push the next chunk past a bound.
            # `_split_block` holds every block to both bounds, so this budget can
            # go to zero, and `_tail` then answers `[]`. No part of a line is
            # carried into the next piece of that line.
            held = (
                []
                if same_line
                else _tail(
                    held,
                    min(overlap_chars, max_chars - len(piece) - 2),
                    bounds.spare(piece),
                    bounds.joint,
                )
            )
            held_width, held_weight = measured(held)
        held.append(piece)
        held_width += len(piece) + 2
        held_weight += piece.weight + bounds.joint
    flush()
    return chunks
