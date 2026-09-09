"""Cutting a document into passages that keep their place in it.

**A chunk that does not know its line range is nearly useless here.** The point
of this row is to hand an agent a pointer it can act on — `read docs/x.md offset
120` — rather than a wall of prose it must then locate. So every chunk carries
the 1-based line span it came from, and the boundary rule is chosen to keep that
span meaningful: cuts land between paragraphs, never mid-sentence, so a span
names something a person would recognise as a unit.

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

from dataclasses import dataclass

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


def _blocks(text: str) -> list[_Block]:
    """Paragraphs, with the line numbers they occupied."""
    found: list[_Block] = []
    current: list[str] = []
    start = 1
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not current:
                start = number
            current.append(line)
            continue
        if current:
            found.append(_Block("\n".join(current), start, number - 1))
            current = []
    if current:
        found.append(_Block("\n".join(current), start, start + len(current) - 1))
    return found


def _split_block(block: _Block, limit: int) -> list[_Block]:
    """A paragraph longer than one chunk, cut on line boundaries.

    A code block, a table, or a wrapped-at-nothing paragraph of prose. Cut by
    lines rather than by characters so the span stays exact — a character cut
    would leave two chunks claiming the same line, and a pointer that is off by
    a line in a 400-line file is a pointer a model will not trust twice.
    """
    if len(block) <= limit:
        return [block]
    pieces: list[_Block] = []
    lines = block.text.split("\n")
    held: list[str] = []
    start = block.start_line
    for offset, line in enumerate(lines):
        candidate = len("\n".join([*held, line]))
        if held and candidate > limit:
            pieces.append(_Block("\n".join(held), start, block.start_line + offset - 1))
            held = []
            start = block.start_line + offset
        held.append(line)
    if held:
        pieces.append(_Block("\n".join(held), start, block.end_line))
    return pieces


def _tail(blocks: list[_Block], overlap: int) -> list[_Block]:
    """The trailing blocks worth carrying into the next chunk, in order.

    Never the whole chunk: a chunk that carried all of itself forward would make
    the next one identical whenever a single block filled it, and the walk would
    not advance.
    """
    if overlap <= 0 or len(blocks) < 2:
        return []
    carried: list[_Block] = []
    total = 0
    for block in reversed(blocks[1:]):
        total += len(block)
        if total > overlap:
            break
        carried.append(block)
    carried.reverse()
    return carried


def chunk_text(text: str, *, max_chars: int, overlap_chars: int) -> list[Chunk]:
    """Pack `text` into chunks of at most `max_chars`, each knowing its lines.

    Greedy: blocks accumulate until the next one would overflow, at which point
    the chunk is emitted and the next is seeded with `_tail`'s carry-over. A
    single block bigger than `max_chars` is split first, so the bound holds for
    every chunk rather than for most of them.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    blocks: list[_Block] = []
    for block in _blocks(text):
        blocks.extend(_split_block(block, max_chars))
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
    for block in blocks:
        if held and held_width + len(block) > max_chars:
            flush()
            held = _tail(held, overlap_chars)
            held_width = width(held)
        held.append(block)
        held_width += len(block) + 2
    flush()
    return chunks
