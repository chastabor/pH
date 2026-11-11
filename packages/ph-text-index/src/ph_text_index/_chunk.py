"""Cutting a document into passages that keep their place in it.

**A chunk that does not know its line range is nearly useless here.** The point
of this row is to hand an agent a pointer it can act on — `read docs/x.md offset
120` — rather than a wall of prose it must then locate. So every chunk carries
the 1-based line span it came from, and the boundary rule is chosen to keep that
span meaningful: cuts land between paragraphs, never mid-sentence, so a span
names something a person would recognize as a unit.

Paragraphs, not a fixed character stride, for the same reason. A stride is
simpler and is what most naive indexers do, and it reliably splits the one
sentence that answers the query across two chunks, so neither retrieves. Blank
lines are where a document's own author already said "new idea", in Markdown, in
plain text, in reStructuredText, and in code comments.

The overlap is the concession to that rule being imperfect: a paragraph whose
meaning depends on the one before it is common enough that carrying the tail of
the previous chunk forward is worth the duplicated tokens.

**Two bounds, in two units.** `max_chars` is how a profile says how big a
passage should be. It is not what an embedder counts: a model reads so many
tokens, and a character is a quarter of a token in English prose, a third in
code and a whole one or more in a script that is three bytes a character. So a
caller that knows what its embedder counts hands over a `Weight` — what a text
weighs and the most a chunk may — and every cut here keeps to both. Each chunk
is then as large as it may be: a table in a page of prose is cut small and the
prose around it is not.

@module ph_text_index._chunk
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from ph.indexable import Paragraph, paragraphs

__all__ = ["Chunk", "Weight", "chunk_paragraphs", "chunk_text", "head_of"]

_LAST_WORD = re.compile(r"\s+\S*\Z")
"""The last run of whitespace in a text, and the word begun after it."""


@dataclass(frozen=True, slots=True)
class Chunk:
    """One passage, and where in the file it was."""

    text: str
    start_line: int
    """1-based, inclusive — the spelling `read`'s `offset` is one less than."""
    end_line: int
    """1-based, inclusive."""


@dataclass(slots=True)
class Weight:
    """A second bound on a chunk, in a unit that is the caller's: to the text
    index, its embedder's tokens. See the module docstring."""

    of: Callable[[Sequence[str]], Sequence[float]]
    """What each text weighs by itself. Asked a batch at a time — a document's
    paragraphs, a paragraph's lines — because the one that counts is a tokenizer,
    and a call of one costs more than the counting."""
    most: float
    """The most a chunk may weigh."""
    bound: bool = False
    """Set once this, and not `max_chars`, is what ended a chunk or split a
    paragraph: the caller's to read after packing, to say that it did."""


@dataclass(frozen=True, slots=True)
class _Bounds:
    """Both bounds on a chunk, and what joining two pieces adds to its weight."""

    chars: int
    weight: Weight | None
    joint: float
    """A separator's weight, on the careful side: never under one, because a
    tokenizer that drops a blank line when it counts it alone may not when the
    line stands between two words."""

    @classmethod
    def of(cls, max_chars: int, weight: Weight | None) -> _Bounds:
        if weight is None:
            return cls(max_chars, None, 0.0)
        return cls(max_chars, weight, max(1.0, weight.of(["\n\n"])[0]))

    def weigh(self, texts: Sequence[str]) -> Sequence[float]:
        return [0.0] * len(texts) if self.weight is None else self.weight.of(texts)

    def past(self, chars: int, heavy: float) -> bool:
        """Whether a piece of this size is past either bound, noting when the
        weight alone was why."""
        if chars > self.chars:
            return True
        if self.weight is not None and heavy > self.weight.most:
            self.weight.bound = True
            return True
        return False

    def spare(self, piece: _Block) -> float:
        """How much weight a chunk that will hold `piece` has for what is carried
        into it: what `piece` leaves, and never more than a quarter of a chunk —
        so a bound that leaves room for a sentence does not fill it with the last
        one."""
        if self.weight is None:
            return math.inf
        most = self.weight.most
        return min(most / 4, most - piece.weight - self.joint)


@dataclass(frozen=True, slots=True)
class _Block:
    """A run of non-blank lines: the document's own smallest unit."""

    text: str
    start_line: int
    end_line: int
    weight: float = 0.0
    """What it weighs, where a `Weight` says; nothing where none does."""

    def __len__(self) -> int:
        return len(self.text)


def _fitting(text: str, heavy: float, weight: Weight) -> tuple[int, float]:
    """How many characters of the start of `text`, which weighs `heavy`, weigh no
    more than `weight.most`, and what they weigh: all of it, where it does.

    Cut in proportion and weighed, because weight is not even along a line: the
    start of a line that is half prose and half base64 does not weigh what the
    line's average says, and is cut again, smaller, until it fits. A single
    character is as far as it goes."""
    size, own = len(text), heavy
    while own > weight.most and size > 1:
        size = max(1, int(size * weight.most / own))
        [own] = weight.of([text[:size]])
    return size, own


def _lightened(text: str, heavy: float, at: int, weight: Weight) -> Iterator[_Block]:
    """`text`, which weighs `heavy`, in pieces no heavier than `weight.most`,
    each as much of what is left as fits (`_fitting`).

    **Each cut is aimed by the stretch just ahead of it** — twice the last
    piece — and not by all that is left. Weighing the rest of the line again at
    every cut was a tokenizer's pass over the line for each piece: 18 s for one
    100,000-character line of CJK. What is just ahead is also the better guide to
    where the next cut falls, on a line whose weight is not even; a stretch that
    fits whole is taken whole, and the next one is aimed by it."""
    ahead, heavy_ahead = len(text), heavy
    while len(text) > 1 and (ahead < len(text) or heavy_ahead > weight.most):
        size, own = _fitting(text[:ahead], heavy_ahead, weight)
        yield _Block(text[:size], at, at, own)
        text = text[size:]
        ahead = min(len(text), 2 * size)
        [heavy_ahead] = weight.of([text[:ahead]]) if text else [0.0]
    if text:
        # All that is left, so what it weighs is known.
        yield _Block(text, at, at, heavy_ahead)


def _of_one_line(one: _Block, other: _Block) -> bool:
    """Whether two blocks are pieces of one line that was cut inside itself.
    Blocks are joined by a blank line, and a line is not given one it never had:
    such pieces stay in chunks of their own."""
    return one.start_line == one.end_line == other.start_line == other.end_line


def head_of(text: str, weight: Weight) -> str:
    """As much of the start of `text` as weighs no more than `weight.most`: all
    of it when it does, and otherwise cut where the weight runs out, back to the
    end of a word where there is one.

    For a text that is one thing and wanted as nearly whole as it can be had — a
    question — where `chunk_paragraphs` is for one wanted in parts: packed as
    paragraphs, a head would stop at the first paragraph the next did not fit
    after, which for a title and a page is the title."""
    [heavy] = weight.of([text])
    size, _ = _fitting(text, heavy, weight)
    if size == len(text):
        return text
    first = text[:size]
    if not text[size].isspace():
        # Inside a word: a piece of one is a token the whole word was not.
        first = _LAST_WORD.sub("", first) or first
    return first.rstrip()


def _split_line(line: str, heavy: float, at: int, bounds: _Bounds) -> list[_Block]:
    """One line too long to be a chunk, cut on characters (X5).

    **The last resort, and it costs the exact span this function otherwise
    protects.** Cutting by lines keeps every chunk's line range true, which is
    what a pointer into a 400-line file needs — but a line is only a unit when
    something wrote it as one. A minified bundle, a base64 blob and a long CSV
    row are each one "line" of thousands of characters, and for them the
    line-only rule was not a bound at all: `max_chars` held for the number of
    lines packed together and not for the chunk, so a single line went to the
    embedder whole and was silently truncated there — X4's failure by the route
    X4 did not cover.

    The pieces all claim the same line, which is the price. It is the right one
    here: a pointer that says "line 812" is worth something when line 812 is a
    sentence and nothing when it is 40 KB of minified JavaScript, so the span
    being approximate for exactly those lines costs a precision that was
    already notional.

    Cut to `max_chars` first, and each piece that is still too heavy cut again
    (`_lightened`). `heavy` is what the line weighs: the one piece of a line
    that `max_chars` does not cut.
    """
    pieces = [line[cut : cut + bounds.chars] for cut in range(0, len(line), bounds.chars)]
    if bounds.weight is None:
        return [_Block(piece, at, at) for piece in pieces]
    weights = [heavy] if len(pieces) == 1 else bounds.weigh(pieces)
    return [
        block
        for piece, own in zip(pieces, weights, strict=True)
        for block in _lightened(piece, own, at, bounds.weight)
    ]


def _split_block(block: _Block, bounds: _Bounds) -> list[_Block]:
    """A paragraph longer than one chunk, cut on line boundaries.

    A code block, a table, or a wrapped-at-nothing paragraph of prose. Cut by
    lines rather than by characters so the span stays exact — a character cut
    would leave two chunks claiming the same line, and a pointer that is off by
    a line in a 400-line file is a pointer a model will not trust twice. A line
    that is *itself* over a bound has no such cut to make, and goes to
    `_split_line` (X5).
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
        # What `held` would be with this line on its end: a newline between.
        if held and bounds.past(chars + 1 + len(line), heavy + bounds.joint + own):
            pieces.append(_Block("\n".join(held), start, at - 1, heavy))
            held, chars, heavy = [], 0, 0.0
            start = at
        if bounds.past(len(line), own):
            # `held` is already empty: this line alone is over a bound, so the
            # flush above it fired for certain.
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
    """The trailing blocks worth carrying into the next chunk, in order.

    Never the whole chunk: a chunk that carried all of itself forward would make
    the next one identical whenever a single block filled it, and the walk would
    not advance.

    **`overlap` is a width in the same unit `chunk_text` measures in** — each
    block plus its two separator characters — so the caller can hand it a budget
    rather than a wish and get back something that fits. The first cut of X4 left
    this function summing raw `len(block)` and put a second trimming loop in the
    caller using `width()`: two owners of one decision, counting differently, and
    the next edit to either drifts. A budget of zero or less is `[]`, which is
    the guarantee that loop provided.

    `spare` is the same budget by weight (`_Bounds.spare`), each block counted
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

    The packer's own door, for a caller that has no opinion about what is worth
    indexing. The text index is not that caller: it asks `ph.indexable.triage`
    first and hands over only what survived, which is why dropping machine
    output is not this module's job (X6 review — it briefly was, and a packer
    whose tests had to know indexing policy was the sign it was in the wrong
    place).
    """
    return chunk_paragraphs(paragraphs(text), max_chars=max_chars, overlap_chars=overlap_chars)


def chunk_paragraphs(
    blocks: Sequence[Paragraph],
    *,
    max_chars: int,
    overlap_chars: int,
    weight: Weight | None = None,
) -> list[Chunk]:
    """Pack `text` into chunks of at most `max_chars`, and no heavier than
    `weight` allows where one is given, each knowing its lines.

    Greedy: blocks accumulate until the next one would overflow, at which point
    the chunk is emitted and the next is seeded with `_tail`'s carry-over. A
    single block past a bound is split first, so both hold for every chunk
    rather than for most of them.

    **A chunk's weight is the sum of its blocks' and a `joint` between each**,
    which is exact where weight is bytes and on the careful side where it is a
    tokenizer's count. So the caller that must not be over checks what it sends;
    this gets it there in one cut.

    Blocking where `weight.of` is: a tokenizer's, in a worker thread.

    **The carry-over is a budget, not a wish** (X4). The seeded tail was never
    re-checked against the block that caused the flush, so a chunk could reach
    `overlap_chars + max_chars` — an embedder's input limit is a hard one, and a
    chunk over it is silently truncated at the far end, indexing text the search
    can never match. `_tail` is handed what is actually left, so the overlap
    gives way to the bound in the one function that owns how much is carried.
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
            # `_split_block` holds every block to both, so this budget can go to
            # zero and `_tail` answers `[]`. None of a line is carried into the
            # next piece of it.
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
