"""Prose helpers for text that a person or a model reads.

The module is small on purpose. It holds formatting that appears in *more than
one* package and has a wrong answer. Such formatting is retyped correctly four
times and wrongly the fifth time. `count_of` is here because that fifth time
already happened. A tool card shipped that read "1 governed calls" for a phase
and a half. It was the one place among four that had written the conditional
expression inline by hand.

@module ph.text
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Literal

from .characters import cut_after, cut_before, head, head_bytes, tail, tail_bytes, utf8_size
from .json import JsonValue

__all__ = [
    "BRIEFLY",
    "NO_OUTPUT",
    "around",
    "block_marker",
    "brief_value",
    "count_of",
    "duration",
    "elide",
    "first_lines",
    "one_line",
    "redacted_marker",
    "thousands",
    "truncation_marker",
]

BRIEFLY = 8
"""How many items of a list a sentence or a note names before it says how many
more there are. It is one number, so that the notes of two tools about the same
kind of run agree."""

NO_OUTPUT = "(no output)"
"""What stands in for a command that printed nothing.

It has a name for the same reason as `truncation_marker`. `!` and `tool-bash`
render the same child process, and a reader who compares a transcript with a log
must not find two sentences for one event (D4). The shell card of the TUI gives
a third answer, and it is deliberately a different one. An empty card body
already reads as "it said nothing", and a card does not need to say in words
what its own emptiness says.
"""


def first_lines[T](
    items: Sequence[T], shown: int, line: Callable[[T], str], *, rest: str = "more"
) -> list[str]:
    """`line` of each of the first `shown` of `items`, then one line that counts
    the rest: `[and 12 more skipped]`. Each line is indented two spaces, as a
    list under a sentence is. Only the lines shown are made, so a list of
    thousands costs `shown`."""
    kept = [f"  {line(item)}" for item in items[:shown]]
    if len(items) > shown:
        kept.append(f"  [and {len(items) - shown} {rest}]")
    return kept


def block_marker(kind: str) -> str:
    """`[media]`, `[tool-call]`: the name of a content block that a renderer
    skipped.

    Every renderer that shows blocks to a person owes the same answer: name what
    was not rendered and do not drop it, so that a reader can see that an image
    was there (`ph_app.tui.trajectory._text` states the rule). Before this
    function existed, five callers wrote this inline, and one of them already
    said something else.
    """
    return f"[{kind}]"


def redacted_marker() -> str:
    """What stands in for reasoning that the provider encrypted (G8).

    A `redacted_thinking` block carries ciphertext that only the provider can
    read, and it is still a `ReasoningBlock`. So every renderer that shows
    `block.text` showed a wall of base64 as something that the assistant had
    said. The marker has a name here, and is not written at each renderer, for
    the same reason as `block_marker`: the three renderers that show reasoning
    to a person owe the same answer. It is built *from* `block_marker` and not
    beside it. A sixth inline `f"[{kind}]"`, three lines under the function
    whose docstring counts the five before it, would repeat the mistake that
    the docstring records.
    """
    return block_marker("redacted reasoning")


def brief_value(value: JsonValue, *, nested: bool = False) -> str:
    """One JSON value on one line, with containers named and not dumped.

    It is here because the mistake that this module exists for had already
    happened two times. `str()` on a payload value is correct for a scalar and
    garbage for anything else, because it emits the Python literal. Two readers
    in two packages both did it. `ph_app.wire.describe` renders 56 event types
    for the auditor view and `phern agents attach`. `ph.commands.revert._brief`
    names the calls that a restore did *not* undo, in the one report that a
    person reads to decide whether the restore was enough. The first rendered a
    whole `Message` (uuid, role, nested content blocks), truncated in the middle
    of a token.

    **The test names the concrete shapes, never the ABCs.** `as_obj` measured
    this: a `MappingProxyType` is the worst input for `isinstance(x, Mapping)`,
    at 220 ns against 56 ns for the pair, and this function runs for each field
    of each event. The same paragraph is the reason that `tuple` is named beside
    `list`. The log freezes arrays into tuples, so a test for `list` alone works
    on decoded wire frames and silently does nothing on the live path. The first
    version of this function shipped with exactly that defect.

    A list is counted, because a list never fits on one line. A mapping is
    expanded one level, because the readable facts are usually there:
    `spent={turns=9}` is the answer that somebody wanted. Below that level a
    mapping is counted. A container is named and not dropped, which is the rule
    of `block_marker` one layer up.
    """
    if isinstance(value, (list, tuple)):
        return f"[{count_of(len(value), 'item')}]"
    if isinstance(value, (dict, MappingProxyType)):
        if nested:
            return f"{{{count_of(len(value), 'field')}}}"
        return "{" + ", ".join(f"{k}={brief_value(v, nested=True)}" for k, v in value.items()) + "}"
    return str(value)


def count_of(count: int, noun: str, plural: str = "") -> str:
    """`1 replacement`, `3 replacements`: the count and its noun, in agreement.

    `plural` is for nouns that English does not pluralize with a suffix (`entry`
    → `entries`). The default covers the regular case, which is almost every
    current caller.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {plural or f'{noun}s'}"


def duration(milliseconds: float) -> str:
    """`3d 4h`, `1h 30m`, `12m`, `8s`: the two largest units that are not zero.

    Zeros are dropped, and are not kept for shape. A sweep every sixty seconds
    reads as `1m`, not `1m 0s`. An hour and a half of quiet is `1h 30m` whether
    or not the seconds are round.

    **The argument is a number, not an `object`.** The function carried an
    `isinstance` guard, and a `—` for "that was not a number". Both were written
    when its callers read loose wire dictionaries. Every caller now reads a
    typed field. The guard was therefore a runtime check in a place where a
    signature does the same work earlier and at no cost. A caller that does hold
    an `object` has a better place to decide what it is than inside a formatter.

    The function is here and not in `ph_app.agents`, where it was written,
    because the sidebar renders the same kind of thing (P9-07). A second copy of
    a duration format is two sentences for one fact. `truncation_marker` below
    is in this module by the same argument.
    """
    seconds = int(milliseconds // 1000)
    parts: list[str] = []
    for name, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count}{name}")
    return " ".join(parts[:2]) or "0s"


def one_line(text: str, limit: int = 120) -> str:
    """Whitespace collapsed, and the text truncated with an ellipsis: a
    subtitle, a preview, a summary that must fit a row.

    The function is here and not in `ph_app.wire`, where it was written, because
    a plugin needs the same row-sized line and may not import the app. ph-clm
    uses it for its section previews and for the summaries that its revisions
    carry. `elide` makes the cut.
    """
    return elide(" ".join(text.split()), limit)


ELLIPSIS = "…"


def elide(
    text: str, size: int, *, keep: Literal["head", "tail"] = "head", in_bytes: bool = False
) -> str:
    """`text` whole when it fits in `size`, which counts code points, or UTF-8
    bytes with `in_bytes`. Otherwise, as much of its start (`keep="head"`) or its
    end as fits beside an ellipsis that stands for the rest.

    This is the one way to mark a cut. The ellipsis is one glyph and is counted
    inside `size`, so a 40-wide column is 40 wide. The cut falls between
    characters (`ph.characters`). Whitespace is left as it is. `one_line` is
    this function on one line."""
    measure = utf8_size if in_bytes else len
    if measure(text) <= size:
        return text
    room = max(size - measure(ELLIPSIS), 0)
    if keep == "tail":
        return ELLIPSIS + (tail_bytes(text, room) if in_bytes else tail(text, room))
    return (head_bytes(text, room) if in_bytes else head(text, room)) + ELLIPSIS


def around(text: str, start: int, size: int, *, stop: int | None = None) -> str:
    """As much of `text` as fits in `size` code points around `text[start:stop]`
    (a match, a signature), with an ellipsis on each side that was cut.

    This is for the excerpt that must show what it was found by. The first few
    hundred characters of a line do not hold a match that is further along a
    minified line. The window is centered on the span where the text allows. A
    span longer than the room keeps its start. Each ellipsis is counted inside
    `size`, and every cut falls between characters, as in `elide`. `text` is
    whole when it fits.
    """
    if len(text) <= size:
        return text
    # The span is widened to whole characters first. A match that ends inside a
    # character is shown with all of that character, and does not stop at the
    # character before it.
    stop = cut_after(text, start if stop is None else max(stop, start))
    start = cut_before(text, start)
    room = max(size - 2 * len(ELLIPSIS), 0)
    lo = max(start - max(room - (stop - start), 0) // 2, 0)
    if lo == 0:
        return elide(text, size)
    if lo + size - len(ELLIPSIS) >= len(text):
        return elide(text, size, keep="tail")
    # `lo` is moved back to a place between characters, but never so far that
    # the end of the span falls out of the room. It also never passes the start
    # of the span, for a span that is wider than the room.
    lo = min(start, max(cut_before(text, lo), cut_after(text, stop - room)))
    hi = cut_before(text, lo + room)
    if hi <= lo:
        # A room narrower than the one character there holds none of it, as in
        # `elide`: the result is never more than `size`.
        return ELLIPSIS
    # An ellipsis marks each side where the window was cut. Moving the window to
    # places between characters can move a cut to an end of the text.
    left = ELLIPSIS if lo > 0 else ""
    right = ELLIPSIS if hi < len(text) else ""
    return f"{left}{text[lo:hi]}{right}"


def thousands(count: int) -> str:
    """`1777` → `1.8k`, `900` → `900`. One abbreviation for a token count.

    This follows the rule of this module, and the mistake had already happened
    on one screen. The subagent panel truncated (`row.tokens // 1000` → `1k`)
    while the cache field of the footer rounded (`1.8k`). The same magnitude
    was therefore printed in two ways a few rows apart. Rounding is the form
    that is kept, because the figure is read as a size and is not counted with.
    """
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def truncation_marker(dropped: int | None, cap: int) -> str:
    """The text that stands in for output that a cap discarded (D4).

    It is here because four things now discard output against a cap: the RLM
    kernel, the guest runner, `!!` and `tool-bash`. The rule is *"a reader
    comparing a transcript to a log must not find two different sentences for
    the same event"*. The first two were already written to keep that rule:
    they are byte-identical, and `test_protocol_mirror` asserts it. The last two
    arrived with P7-13, and each invented a wording. That is exactly the mistake
    that this module exists to prevent.

    `ph_runtime.protocol` keeps a deliberate copy. That package ships into the
    guest venv with `dill` as its only dependency, so it cannot import this
    module. It is one of three copies. `ph_runtime._json` copies the one
    narrowing that the guest imports from `ph.json`, and
    `ph_runtime._characters` copies `ph.characters`. `test_protocol_mirror`
    checks all three.

    **`dropped` is `None` where the cut is made on a stream that is still
    running**: the pipes of the kernel, the prints of the guest. Such a stream
    cannot know yet how much will follow. The marker then says where the cut
    fell and gives no number. The stream reports its total after it has ended,
    in this same sentence with the count. A count written at the cut was only
    the size of the write that crossed it, and it was read as the whole.
    """
    if dropped is None:
        return f"\n[ph: output truncated — the rest is not kept, cap {cap} bytes]\n"
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
