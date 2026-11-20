"""What in a document is worth putting in an index, and what is waste (X6).

An embedding index answers by similarity, so what goes in decides what can come
out. A vector for 1 200 characters of minified JavaScript matches nothing that a
person will ever ask. It occupies the same memory as a paragraph, and it costs
the same embedder call to make. Garbage in gives more than garbage out here. It
also gives a slower, larger index for the passages that were worth keeping.

**Two questions, deliberately separate.** "Is this document worth indexing at
all" is answered for each file, by the `reason` of `triage`. The caller reports
the sentence in the same way as it already reports a size skip. A corpus that is
quiet because everything was skipped must not look like a corpus that is quiet
because nothing matched. "Is this *part* worth indexing" is answered for each
block, by `machine_block`, so a README with an embedded base64 image still
indexes its prose. The second question makes it safe for the first to be
strict. A document is skipped only when what is left, after the machine parts
are dropped, is not worth the walk.

**These are heuristics, and they are named as such.** No test separates text
that a person wrote from text that a program emitted. There is only a set of
properties that prose reliably has and machine output reliably lacks. Each
heuristic below is stated with the shape that it looks for and the reason for
the value of its threshold, so that a false positive can be argued with and not
guessed at. The heuristics are tuned to refuse only the obvious cases. A
document that is *arguably* prose is indexed, because the cost of indexing junk
is spent one time, and the cost of dropping a real passage is paid on every
search that needed it.

@module ph.indexable
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Paragraph",
    "Triage",
    "machine_block",
    "machine_line",
    "paragraphs",
    "triage",
]

_MINIFIED_CHARS = 200
"""How long a line must be before its shape is evidence of anything.

A hand-written line runs to about 120 characters, the width that every style
guide argues over. A long sentence, a table row or a URL can pass that without
being machine output. At 200 characters, a line is no longer something that a
person laid out, and the two tests below decide what it is."""

_SYMBOL_SHARE = 0.20
"""The punctuation fraction above which a long line is code or structured data.

These values were measured on the shapes in this repo. English prose is ~2%
characters that are neither alphanumeric nor space. A markdown table row is
12.5%, and a comma-separated row is 12.6%. Minified JavaScript is 28%, minified
CSS is 24%, and a line of packed JSON is 59%. The output of a minifier is mostly
the punctuation that survives when everything else is squeezed out. That makes
it legible to a parser and useless to an embedder.

**The gap is between 13% and 24%**, and the threshold is in that gap. Keeping
the comma-separated row on the prose side is deliberate. The character cut of X5
exists so that a long data row is still indexed, and a row of names and places
is text that somebody may search for."""

_ENCODED = re.compile(r"^[A-Za-z0-9+/=_-]+$")
"""One line of base64, base64url, or hex.

The pattern is anchored at both ends and covers the whole line, so a line of
prose that contains a token does not match. The signal is a line that is
*nothing but* the alphabet."""

_RUN = re.compile(r"[A-Za-z0-9+/=]{64,}")
"""An unbroken alphanumeric run that is longer than any word.

The longest word in ordinary English is under 30 characters, and the longest
identifier that a person types is not much more. So 64 is past anything that was
written and not generated. A run of that length is a hash, a key, a data URI, or
a tracking parameter."""

_RUN_SHARE = 0.5
"""How much of a line one such run must cover before the line *is* the run.

A sentence that quotes a 64-character digest is still a sentence, and it is
worth indexing. A 300-character line that is 290 characters of base64 is a
payload with a few characters of punctuation around it. Half separates the two.
The value does not need to be precise, because real cases are near 5% or near
100%."""

_ENCODED_SHARE = 0.8
"""How much of a block must be the alphabet before the block is a payload.

Not *all* of it, because a payload in a document almost never arrives bare. A
markdown image is `![alt](data:image/png;base64,`, then the blob, then a
parenthesis. An attachment dump has a header line. A rule that every line must
be the alphabet misses exactly the shape that this value is for. Four fifths
leaves room for the wrapper. It also keeps a paragraph of prose, where the share
is zero, well clear."""

_ENCODED_CHARS = 160
"""How much encoded text makes a block a blob and not a checksum.

Base64 wraps at 76 characters, so a blob is many short lines, and no single line
is long enough for the line test to see. Two full lines are a 120-byte payload.
That is longer than a hash, a key fingerprint or a short signature. All of
those are worth keeping in a document, because a person may search for them by
name."""

_REPLACEMENT = "\ufffd"
"""What a byte that is not UTF-8 decodes to.

`_decode` in `ph.seams.fs` falls back to `errors="replace"`, so a binary file
arrives here as text full of these characters. This module reads the fact from
the text. It is not passed down from the `lossy` flag of the decoder. The reason
is worth stating, because the flag is the obvious design. `FileSlice` is a
**tool result**, so every field on it is in the schema of the `read` tool. The
model is charged for that schema, and `offload` measures it. A first attempt
added the flag there. That moved the prefix-cache benchmark and an offload
threshold, in tests that have nothing to do with indexing. The replacement
character *is* the answer of the decoder, written into the text that it
returned. Reading it here therefore costs nothing that any other code pays
for."""

_REPLACEMENT_SHARE = 0.01
"""How many replacement characters mean that the bytes were never text.

A document can legitimately contain one: a quoted mojibake example, or a
troubleshooting note about encodings. A file that was never UTF-8 decodes to
them in bulk. There is one for each malformed byte, and for binary content that
is most of the bytes. 1% is far above the first case and far below the
second."""

_PROSE_FLOOR = 200
"""How much prose must survive the drop for a document to be worth indexing.

**The measure is what is left, not what was thrown away.** It applies only to a
document that something was taken out of. The difference decides two common
cases.

A README with a base64 image in it is mostly machine output *by weight*: the
blob outweighs the paragraphs around it several times. But it still holds every
paragraph that somebody would search for. Counting the garbage refuses that
document. Counting the remainder keeps it, and drops only the blob.

So this is the floor under "index what is left". A document whose surviving
prose would not fill even a fraction of one passage has nothing to answer with.
The walk, the read and the embedder call would be spent to index a header
comment. 200 characters is about two sentences. That is far under `max_chars`,
because the question is whether anything survived, and not whether it fills a
chunk. A document with no machine content never meets this number at all: a
two-sentence note is short, not waste."""


def machine_line(line: str) -> bool:
    """Whether one line is machine output and not something that a person wrote.

    There are two shapes, because minifiers and encoders leave opposite marks.
    A minifier squeezes out everything but punctuation, so its output is *dense
    in symbols*. An encoder emits one alphabet and no punctuation at all, so its
    output is *one enormous run*. Prose is neither. A long comma-separated data
    row is also neither, and that is important: the character cut of X5 exists
    to keep such a row indexable.

    Length is the gate and not the evidence. A hand-written line runs to about
    120 characters, and below `_MINIFIED_CHARS` the shape of a line says nothing
    worth acting on.
    """
    if len(line) < _MINIFIED_CHARS:
        return False
    symbols = sum(not character.isalnum() and not character.isspace() for character in line)
    if symbols / len(line) > _SYMBOL_SHARE:
        return True
    longest = max(map(len, _RUN.findall(line)), default=0)
    return longest / len(line) > _RUN_SHARE


def machine_block(text: str) -> bool:
    """Whether a block of a document is machine output, and should not be indexed.

    There are two shapes, because the two arrive differently. A minified bundle
    is one enormous line, and `machine_line` catches it. An encoded payload is
    wrapped at 76 characters, so no single line is remarkable, and only the
    block is. Most of the block is the alphabet of the encoding and nothing
    else, and there is enough of it to be a payload and not a fingerprint.

    *Most*, not all: a payload in a document arrives wrapped in something. A
    markdown image opens with `![alt](data:image/png;base64,` and closes with a
    parenthesis. A rule that every line must be the alphabet would miss the most
    common shape.
    """
    lines = [line for line in text.splitlines() if line]
    if not lines:
        return False
    if all(machine_line(line) for line in lines):
        return True
    encoded = sum(len(line) for line in lines if _ENCODED.match(line))
    total = sum(len(line) for line in lines)
    return encoded >= _ENCODED_CHARS and encoded / total > _ENCODED_SHARE


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A run of non-blank lines, and the lines it occupied (1-based, inclusive)."""

    text: str
    start_line: int
    end_line: int


def paragraphs(text: str) -> list[Paragraph]:
    """The blocks of a document, separated by blank lines. This is the one
    definition of a block.

    **There is one splitter, because two had already disagreed.** The document
    gate split paragraphs with a blank-line regex and lines with a bare newline
    split. The chunker used `splitlines()`. On a CRLF file the gate kept each
    carriage return, and the end anchor of `_ENCODED` stopped matching. So the
    gate said "index it" about a blob that the chunker then dropped. Every
    caller that asks what a block is now asks here. `splitlines()` is the
    answer, because it is the one that handles a carriage return.
    """
    found: list[Paragraph] = []
    current: list[str] = []
    start = 1
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip():
            if not current:
                start = number
            current.append(line)
            continue
        if current:
            found.append(Paragraph("\n".join(current), start, number - 1))
            current = []
    if current:
        found.append(Paragraph("\n".join(current), start, start + len(current) - 1))
    return found


@dataclass(frozen=True, slots=True)
class Triage:
    """A document that is split and classified one time: what is worth indexing,
    and whether any of it is."""

    kept: list[Paragraph]
    reason: str
    """Why the document is not worth indexing, or `""` when `kept` should be
    indexed.

    The answer is a sentence and not a bool, for the same reason as in
    `FsService.skip_reason`: a caller that never learns *why* a file was passed
    over cannot tell a corpus that is quiet from one that is broken.

    Bytes that were never UTF-8 are not a document. `_REPLACEMENT` explains why
    that is read from the text and not from the flag of the decoder."""


def triage(text: str) -> Triage:
    """Split a document one time, classify every block one time, and judge what
    is left.

    **One time, because the caller needs both answers and each costs a pass.**
    The document gate and the chunker used to classify the same blocks
    independently, and each did its own splitting first. That measured at
    roughly half of the whole cost of this module on a 2 MiB file. The caller
    passes `kept` to the packer, which then does only packing again.

    **Only a document that something was taken *out* of can fall below the
    floor.** A short note is not waste. It is short. To refuse it would be the
    same mistake in the other direction: an index that quietly lacks the file
    in which somebody wrote two sentences. An empty document has nothing taken
    out either, so it too is left to its caller, which can say "is empty" more
    accurately than this function could.
    """
    if text.count(_REPLACEMENT) > len(text) * _REPLACEMENT_SHARE:
        return Triage(kept=[], reason="is not text")
    blocks = paragraphs(text)
    kept = [block for block in blocks if not machine_block(block.text)]
    if len(kept) < len(blocks) and sum(len(block.text) for block in kept) < _PROSE_FLOOR:
        return Triage(kept=[], reason="holds no prose to index (minified, encoded, or generated)")
    return Triage(kept=kept, reason="")
