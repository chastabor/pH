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

@module ph_text_index._chunk
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ph.indexable import Paragraph, paragraphs

__all__ = ["Chunk", "chunk_text"]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One passage, and where in the file it was."""

    text: str
    start_line: int
    """1-based, inclusive — the spelling `read`'s `offset` is one less than."""
    end_line: int
    """1-based, inclusive."""


@dataclass(frozen=True, slots=True)
class _Block:
    """A run of non-blank lines: the document's own smallest unit."""

    text: str
    start_line: int
    end_line: int

    def __len__(self) -> int:
        return len(self.text)


def _split_line(line: str, limit: int, at: int) -> list[_Block]:
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
    """
    return [_Block(line[cut : cut + limit], at, at) for cut in range(0, len(line), limit)]


def _split_block(block: _Block, limit: int) -> list[_Block]:
    """A paragraph longer than one chunk, cut on line boundaries.

    A code block, a table, or a wrapped-at-nothing paragraph of prose. Cut by
    lines rather than by characters so the span stays exact — a character cut
    would leave two chunks claiming the same line, and a pointer that is off by
    a line in a 400-line file is a pointer a model will not trust twice. A line
    that is *itself* over the limit has no such cut to make, and goes to
    `_split_line` (X5).
    """
    if len(block) <= limit:
        return [block]
    pieces: list[_Block] = []
    lines = block.text.split("\n")
    held: list[str] = []
    start = block.start_line
    for offset, line in enumerate(lines):
        at = block.start_line + offset
        candidate = len("\n".join([*held, line]))
        if held and candidate > limit:
            pieces.append(_Block("\n".join(held), start, at - 1))
            held = []
            start = at
        if len(line) > limit:
            # `held` is already empty: this line alone is over the limit, so the
            # flush above it fired for certain.
            pieces.extend(_split_line(line, limit, at))
            start = at + 1
            continue
        held.append(line)
    if held:
        pieces.append(_Block("\n".join(held), start, block.end_line))
    return pieces


def _tail(blocks: list[_Block], overlap: int) -> list[_Block]:
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
    """
    if overlap <= 0 or len(blocks) < 2:
        return []
    carried: list[_Block] = []
    total = 0
    for block in reversed(blocks[1:]):
        total += len(block) + 2
        if total > overlap:
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
    blocks: Sequence[Paragraph], *, max_chars: int, overlap_chars: int
) -> list[Chunk]:
    """Pack `text` into chunks of at most `max_chars`, each knowing its lines.

    Greedy: blocks accumulate until the next one would overflow, at which point
    the chunk is emitted and the next is seeded with `_tail`'s carry-over. A
    single block bigger than `max_chars` is split first, so the bound holds for
    every chunk rather than for most of them.

    **The carry-over is a budget, not a wish** (X4). The seeded tail was never
    re-checked against the block that caused the flush, so a chunk could reach
    `overlap_chars + max_chars` — an embedder's input limit is a hard one, and a
    chunk over it is silently truncated at the far end, indexing text the search
    can never match. `_tail` is handed what is actually left, so the overlap
    gives way to the bound in the one function that owns how much is carried.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    packed: list[_Block] = []
    for block in blocks:
        packed.extend(_split_block(_Block(block.text, block.start_line, block.end_line), max_chars))
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

    def width(blocks: list[_Block]) -> int:
        """Characters the joined blocks would occupy, separators included."""
        return sum(len(one) for one in blocks) + 2 * len(blocks)

    held_width = 0
    for piece in packed:
        if held and held_width + len(piece) > max_chars:
            flush()
            # The carry-over gets what is left after the block that caused the
            # flush, so it cannot push the next chunk past the bound.
            # `_split_block` bounds every block by `max_chars`, so this budget
            # can go to zero and `_tail` answers `[]`.
            held = _tail(held, min(overlap_chars, max_chars - len(piece) - 2))
            held_width = width(held)
        held.append(piece)
        held_width += len(piece) + 2
    flush()
    return chunks
