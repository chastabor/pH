"""What in a document is worth putting in an index, and what is waste (X6).

An embedding index answers by similarity, so what goes in decides what can come
out: a vector for 1 200 characters of minified JavaScript matches nothing a
person will ever ask, occupies the same memory as a paragraph, and costs the
same embedder call to make. Garbage in is not merely garbage out here — it is
garbage out *and* a slower, larger index for the passages that were worth
keeping.

**Two questions, deliberately separate.** "Is this document worth indexing at
all" is answered per file, by `unindexable_reason`, and the caller reports the
sentence the way it already reports a size skip — a corpus that is quiet because
everything was skipped must not look like a corpus that is quiet because nothing
matched. "Is this *part* worth indexing" is answered per block by
`machine_block`, so a README with a base64 image embedded in it still indexes
its prose. The second is what makes the first safe to be strict: a document is
only skipped when what is left after dropping the machine parts is not worth the
walk.

**Heuristics, and named as such.** There is no test that separates text a person
wrote from text a program emitted; what there is is a set of properties prose
reliably has and machine output reliably lacks. Each one below is stated with
the shape it is looking for and the reason its threshold sits where it does, so
a false positive can be argued with rather than guessed at. They are tuned to
refuse only the obvious — a document that is *arguably* prose is indexed, since
the cost of indexing junk is spent once and the cost of dropping a real passage
is paid on every search that needed it.

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
    "unindexable_reason",
]

_MINIFIED_CHARS = 200
"""How long a line must be before its shape is evidence of anything.

A hand-written line runs to about 120 characters — the width every style guide
argues over — and a long sentence, a table row or a URL can pass that without
being machine output. At 200 a line has stopped being something a person laid
out, and the two tests below decide what it is instead."""

_SYMBOL_SHARE = 0.20
"""The punctuation fraction above which a long line is code or structured data.

Measured on this repo's own shapes: English prose is ~2% non-alphanumeric
non-space characters, a markdown table row 12.5%, a comma-separated row 12.6% —
and minified JavaScript 28%, minified CSS 24%, a line of packed JSON 59%. A
minifier's output is mostly the punctuation that survives when everything else
is squeezed out, and that is what makes it legible to a parser and useless to an
embedder.

**The gap is between 13% and 24%**, which is where the threshold sits. Keeping
the comma-separated row on the prose side is deliberate: X5's character cut
exists so a long data row is still indexed, and a row of names and places is
text somebody may search for."""

_ENCODED = re.compile(r"^[A-Za-z0-9+/=_-]+$")
"""One line of base64, base64url, or hex.

Anchored and total, so a line of prose containing a token does not match — the
signal is a line that is *nothing but* the alphabet."""

_RUN = re.compile(r"[A-Za-z0-9+/=]{64,}")
"""An unbroken alphanumeric run no word is.

The longest word in ordinary English is under 30 characters and the longest
identifier a person types is not much more, so 64 is past anything written
rather than generated: a hash, a key, a data URI, a tracking parameter."""

_RUN_SHARE = 0.5
"""How much of a line one such run must cover before the line *is* the run.

A sentence that quotes a 64-character digest is still a sentence and is worth
indexing; a 300-character line that is 290 characters of base64 is a payload
with a few characters of punctuation around it. Half separates them and does not
need to be precise, because real cases sit near 5% or near 100%."""

_ENCODED_SHARE = 0.8
"""How much of a block must be the alphabet before the block is a payload.

Not *all* of it, because a payload in a document almost never arrives bare: a
markdown image is `![alt](data:image/png;base64,` and then the blob and then a
paren, and an attachment dump has a header line. Requiring every line to be the
alphabet misses exactly the shape this is for. Four fifths leaves room for the
wrapper while keeping a paragraph of prose — where the share is zero — well
clear."""

_ENCODED_CHARS = 160
"""How much encoded text makes a block a blob rather than a checksum.

Base64 wraps at 76 characters, so a blob is many short lines and no single line
is long enough for the line test to see. Two full lines is a 120-byte payload:
past a hash, a key fingerprint or a short signature, all of which are worth
keeping in a document a person may search for them by name."""

_REPLACEMENT = "\ufffd"
"""What a byte that is not UTF-8 decodes to.

`FsService._decode` falls back to `errors="replace"`, so a binary file arrives
here as text full of these. Read from the text rather than threaded from the
decoder's own `lossy` flag, and the reason is worth stating because the obvious
design is the other one: `FileSlice` is a **tool result**, so every field on it
is in the `read` tool's schema that the model is charged for and that `offload`
measures. A first attempt added the flag there and moved the prefix-cache
benchmark and an offload threshold in tests with nothing to do with indexing.
The replacement character *is* the decoder's answer, written into the text it
returned, so reading it here costs nothing anybody else pays for."""

_REPLACEMENT_SHARE = 0.01
"""How many replacement characters mean the bytes were never text.

A document can legitimately contain one — a quoted mojibake example, a
troubleshooting note about encodings. A file that was never UTF-8 decodes to
them in bulk: one per malformed byte, which for binary content is most of it.
1% is far above the former and far below the latter."""

_PROSE_FLOOR = 200
"""How much prose must survive the drop for a document to be worth indexing.

**Measured on what is left, not on what was thrown away**, and only for a
document something was taken out of. The difference decides two common cases.

A README with a base64 image in it is mostly machine output *by weight* — the
blob outweighs the paragraphs around it several times over — while still
holding every paragraph somebody would search for. Counting the garbage refuses
that document; counting the remainder keeps it, and drops only the blob.

So this is the floor under "index what is left": a document whose surviving
prose would not fill even a fraction of one passage has nothing to answer with,
and the walk, the read and the embedder call are spent to index a header
comment. 200 characters is about two sentences — under `max_chars` by a wide
margin, because the question is whether anything survived rather than whether it
fills a chunk. A document with no machine content in it never meets this number
at all: a two-sentence note is short, not waste."""


def machine_line(line: str) -> bool:
    """Whether one line is machine output rather than something a person wrote.

    Two shapes, because minifiers and encoders leave opposite fingerprints. A
    minifier squeezes out everything but punctuation, so its output is *dense in
    symbols*. An encoder emits one alphabet and no punctuation at all, so its
    output is *one enormous run*. Prose is neither, and — importantly — neither
    is a long comma-separated data row, which X5's character cut exists to keep
    indexable.

    Length is the gate rather than evidence: a hand-written line runs to about
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

    Two shapes, because the two arrive differently. A minified bundle is one
    enormous line, caught by `machine_line`. An encoded payload is wrapped at
    76 characters, so no single line is remarkable and only the block is — most
    of it is the encoding's alphabet and nothing else, and there is enough of it
    to be a payload rather than a fingerprint.

    *Most*, not all: a payload in a document arrives wrapped in something. A
    markdown image opens `![alt](data:image/png;base64,` and closes with a
    paren, and requiring every line to be the alphabet would miss the commonest
    shape there is.
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
    """A document's blank-line-separated blocks — the one definition of a block.

    **One splitter, because two had already disagreed.** The document gate split
    paragraphs with a blank-line regex and lines with a bare newline split,
    while the chunker used `splitlines()`; on a CRLF file the gate kept each
    carriage return and `_ENCODED`'s end anchor stopped matching, so the gate
    said "index it" about a blob the chunker then dropped. Every caller that
    asks what a block is now asks here, and `splitlines()` is the answer because
    it is the one that knows about a carriage return.
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
    """A document split and classified once: what is worth indexing, and whether any is."""

    kept: list[Paragraph]
    reason: str
    """Why the document is not worth indexing, or `""` when `kept` should be."""


def triage(text: str) -> Triage:
    """Split a document once, classify every block once, and judge what is left.

    **Once, because the caller needs both answers and each costs a pass.** The
    document gate and the chunker used to classify the same blocks
    independently — measured at roughly half of this module's whole cost on a
    2 MiB file — and each did its own splitting to get there. The caller hands
    `kept` to the packer, which is then back to doing only packing.

    **Only a document something was taken *out* of can fall below the floor.** A
    short note is not waste, it is short — and refusing it would be the same
    mistake in the other direction, an index quietly missing the file somebody
    wrote two sentences in. An empty document has nothing taken out either, so
    it too is left to its caller, which says "is empty" more truly than this
    could.
    """
    if text.count(_REPLACEMENT) > len(text) * _REPLACEMENT_SHARE:
        return Triage(kept=[], reason="is not text")
    blocks = paragraphs(text)
    kept = [block for block in blocks if not machine_block(block.text)]
    if len(kept) < len(blocks) and sum(len(block.text) for block in kept) < _PROSE_FLOOR:
        return Triage(kept=[], reason="holds no prose to index (minified, encoded, or generated)")
    return Triage(kept=kept, reason="")


def unindexable_reason(text: str) -> str:
    """Why a whole document is not worth indexing. `""` when it is.

    A sentence rather than a bool, for `FsService.skip_reason`'s reason: a
    caller that never learns *why* a file was passed over cannot tell a corpus
    that is quiet from one that is broken. `triage` when the caller also wants
    the blocks that survived, which the text index does.

    Bytes that were never UTF-8 are not a document, and `_REPLACEMENT` explains
    why that is read off the text rather than off the decoder's flag.
    """
    return triage(text).reason
