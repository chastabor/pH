"""X6 — what an index is worth putting a vector on, and what is waste.

Gate: *the shapes a person writes stay indexable, and the shapes a program
emits do not.*

These are heuristics, so the test that matters is the one that pins the
**boundary** rather than the obvious cases: a comma-separated data row and a
minified stylesheet both have almost no whitespace, and a document that is
mostly blob by weight can still be mostly prose by worth. Each case below is a
real shape, measured, with the number that separates it from its neighbour.
"""

from __future__ import annotations

import base64
import textwrap

import pytest

from ph.indexable import machine_block, machine_line, triage, unindexable_reason

PROSE = (
    "The chunker cuts on line boundaries so the span stays exact, which is what a "
    "pointer into a large file needs to be worth trusting, and that is the argument."
)
MINIFIED_JS = (
    '!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?t(exports):'
    '"function"==typeof define&&define.amd?define(["exports"],t):t(e.lib={})}(this,'
    'function(e){"use strict";function t(e,t){return e.map(function(n){return n.x+t})}e.t=t});'
) * 3
MINIFIED_CSS = (
    ".a{margin:0;padding:0}.b{display:flex;align-items:center}.c{color:#fff;background:#000}"
) * 6
CSV_ROW = ",".join(f"value{number}" for number in range(60))
CSV_WORDS = ",".join(f"Some Name {number},Austin TX,{number * 7}" for number in range(30))
MARKDOWN_ROW = "| name | description | default |" * 12
BLOB = "\n".join(textwrap.wrap(base64.b64encode(b"x" * 400).decode(), 76))


@pytest.mark.parametrize(
    ("label", "line", "machine"),
    [
        # Prose, and the shapes that look like prose to the measurements that
        # matter: ~2% punctuation, no run longer than a word.
        ("prose", PROSE, False),
        ("a markdown table row", MARKDOWN_ROW, False),
        # **The boundary.** A comma-separated row has *no whitespace at all*,
        # which is why a whitespace test would refuse it — and X5's character
        # cut exists precisely so a long data row stays indexable. Punctuation
        # tells them apart: 12.6% here against 24% for the stylesheet below.
        ("a long comma-separated row", CSV_ROW, False),
        ("a row of names and places", CSV_WORDS, False),
        # A sentence that quotes a digest is still a sentence: the run is there
        # but it is a fraction of the line, not the whole of it.
        (
            "a sentence quoting a digest",
            f"The digest is {'a' * 64} and you can verify it "
            "against the signature published beside the tarball on the downloads page.",
            False,
        ),
        # Machine output, by the two opposite fingerprints.
        ("minified javascript", MINIFIED_JS, True),
        ("minified css", MINIFIED_CSS, True),
        ("one long base64 line", base64.b64encode(b"x" * 200).decode(), True),
        # Short lines say nothing whatever their shape: the length gate first.
        ("a short dense line", ".a{b:0}.c{d:1}", False),
    ],
)
def test_the_line_shapes_a_person_writes_are_kept(label: str, line: str, machine: bool) -> None:
    """One line at a time, at the boundary rather than at the extremes."""
    assert machine_line(line) is machine, label


def test_an_encoded_blob_is_caught_by_the_block_it_makes_not_the_lines() -> None:
    """Base64 wraps at 76 characters, so no single line is remarkable.

    This is why there are two detectors rather than one: the line test needs
    length to say anything, and an encoder never gives it any. What gives the
    payload away is that *every* line is the alphabet and there is enough of it
    to be a payload rather than a fingerprint.
    """
    assert all(not machine_line(line) for line in BLOB.split("\n")), (
        "a wrapped blob's lines are short; the line test cannot be what catches this"
    )
    assert machine_block(BLOB)

    # And a lone digest in a document is a fingerprint worth keeping, not a blob.
    assert not machine_block("a" * 64)


@pytest.mark.parametrize(
    ("label", "block", "machine"),
    [
        ("a paragraph", PROSE, False),
        ("two data rows", f"{CSV_ROW}\n{CSV_ROW}", False),
        ("a minified bundle", MINIFIED_JS, True),
        ("an encoded payload", BLOB, True),
    ],
)
def test_a_block_is_dropped_only_when_every_line_of_it_is_machine(
    label: str, block: str, machine: bool
) -> None:
    assert machine_block(block) is machine, label


def test_a_document_is_judged_on_what_survives_not_on_what_is_dropped() -> None:
    """The case the obvious rule gets wrong, and the reason for the floor (X6).

    A README with an embedded image is mostly machine output *by weight* — the
    blob outweighs the paragraphs several times over — while still holding every
    paragraph somebody would search for. A rule counting the garbage refuses
    that document; a rule counting the remainder keeps it and drops the blob,
    which is what "index what is left" means.

    Sabotage: measure the machine share instead of the remainder and the mixed
    document below is skipped.
    """
    mixed = f"{PROSE}\n\n{BLOB}\n\n{PROSE}"
    assert len(BLOB) > 2 * len(PROSE), "the fixture does not reproduce the weight imbalance"
    assert unindexable_reason(mixed) == "", "a document with prose in it was refused"

    # Nothing survives: the walk, the read and the embedder call would be spent
    # to index a header comment.
    assert unindexable_reason(MINIFIED_JS)
    assert unindexable_reason(BLOB)
    assert unindexable_reason(f"// vendor bundle\n\n{MINIFIED_JS}")


def test_bytes_that_were_never_text_are_read_as_such_from_what_they_decoded_to() -> None:
    """A binary file arrives as replacement characters, in bulk.

    Read from the text rather than from `_decode`'s own `lossy` flag, and the
    reason is the interesting part: `FileSlice` is a *tool result*, so a field
    on it lands in the `read` tool's schema that the model is charged for. A
    first attempt put the flag there and moved the prefix-cache benchmark and an
    offload threshold in suites with nothing to do with indexing.

    One replacement character is not evidence — a troubleshooting note about
    encodings may quote one — so the test is a share, and the two cases sit
    orders of magnitude apart.
    """
    binary = bytes(range(256)) * 20
    assert unindexable_reason(binary.decode("utf-8", errors="replace")) == "is not text"

    quoting_one = (
        "When a file is not UTF-8 the decoder writes \ufffd in place of the byte, which is "
        "how the read tool reports it rather than raising, and that is what this note is about."
    )
    assert unindexable_reason(quoting_one) == "", "a note quoting one was mistaken for binary"


def test_an_empty_document_is_left_to_its_callers_own_sentence() -> None:
    """Empty is not waste, and "is empty" is the truer thing to report."""
    assert unindexable_reason("") == ""
    assert unindexable_reason("   \n\n  ") == ""


def test_a_windows_line_ending_does_not_change_what_a_block_is() -> None:
    """X6 review — two definitions of a paragraph had already disagreed.

    The document gate split paragraphs with a blank-line regex and lines with a
    plain newline split, keeping each carriage return; the chunker used
    `splitlines()`, which drops it. So on a CRLF file `_ENCODED`'s end anchor
    stopped matching the blob's lines, the gate said "index it", and the chunker
    then dropped the blob the gate had just counted as prose. One splitter,
    built on `splitlines()`, makes the two answers the same answer.

    Sabotage: split `machine_block`'s lines on a bare newline again and the
    direct assertion fails; split `paragraphs` that way and the `triage` ones do.
    """
    lf = f"{PROSE}\n\n{BLOB}\n\n{PROSE}"
    crlf = lf.replace("\n", "\r\n")

    assert [p.text for p in triage(crlf).kept] == [p.text for p in triage(lf).kept]
    assert len(triage(crlf).kept) == 2, "the blob survived on a CRLF file"
    assert unindexable_reason(BLOB.replace("\n", "\r\n")), "a CRLF blob read as prose"

    # And `machine_block` on its own, because it is public and a caller may hand
    # it raw text: `triage` is protected by `paragraphs` stripping the `\r`
    # first, which is exactly why a test through `triage` alone could not see
    # this line go wrong.
    assert machine_block(BLOB.replace("\n", "\r\n")), "machine_block kept the \\r"
