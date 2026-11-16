"""Prose helpers for text a person or a model reads.

Small on purpose. What lives here is formatting that appears in *more than one*
package and has a wrong answer — the kind of thing that is retyped correctly
four times and wrongly the fifth. `count_of` is here because that fifth time
already happened: a tool card shipped reading "1 governed calls" for a phase and
a half, in the one place among four that had inlined the ternary by hand.

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
"""How many of a list a sentence or a note names before it says how many more
there are. One number, so two tools' notes about the same kind of run agree."""

NO_OUTPUT = "(no output)"
"""What stands in for a command that printed nothing.

Named for the same reason `truncation_marker` is: `!` and `tool-bash` render
the same child process, and a reader comparing a transcript to a log must not
find two sentences for one event (D4). The TUI's shell card is a third answer
and deliberately a different one — an empty card body already reads as "it
said nothing", and a card need not say in words what its own emptiness says.
"""


def first_lines[T](
    items: Sequence[T], shown: int, line: Callable[[T], str], *, rest: str = "more"
) -> list[str]:
    """`line` of each of the first `shown` of `items`, then one counting the rest:
    `[and 12 more skipped]`. Each indented two spaces, as a list under a sentence
    is; only the lines shown are made, so a list of thousands costs `shown`."""
    kept = [f"  {line(item)}" for item in items[:shown]]
    if len(items) > shown:
        kept.append(f"  [and {len(items) - shown} {rest}]")
    return kept


def block_marker(kind: str) -> str:
    """`[media]`, `[tool-call]` — a content block a renderer skipped, named.

    Every renderer that shows blocks to a person owes the same answer: name what
    was not rendered rather than drop it, so a reader can see an image was there
    (`ph_app.tui.trajectory._text` states the rule). Five callers spelled this
    inline, and one of them already says something else.
    """
    return f"[{kind}]"


def redacted_marker() -> str:
    """What stands in for reasoning the provider encrypted (G8).

    A `redacted_thinking` block carries ciphertext only the provider can read,
    and it is still a `ReasoningBlock` — so every renderer that shows `block.text`
    showed a wall of base64 as something the assistant had said. Named here
    rather than spelled at each renderer for `block_marker`'s reason: the three
    that show reasoning to a person owe the same answer. Built *from*
    `block_marker` rather than beside it — a sixth inline `f"[{kind}]"` three
    lines under the function whose docstring counts the five that preceded it
    would be a joke at its expense.
    """
    return block_marker("redacted reasoning")


def brief_value(value: JsonValue, *, nested: bool = False) -> str:
    """One JSON value on one line, with containers named rather than dumped.

    Here because the fifth-time-wrongly had already happened twice. `str()` on a
    payload value is correct for a scalar and garbage for anything else — it
    emits the Python literal — and two readers in two packages both did it:
    `ph_app.wire.describe`, which renders 56 event types for the auditor view and
    `phern agents attach`, and `ph.commands.revert._clip`, which names the calls a
    restore did *not* undo in the one report a person reads while deciding
    whether the restore was enough. The first rendered a whole `Message` — uuid,
    role, nested content blocks — truncated mid-token.

    **The concrete shapes, never the ABCs.** `as_obj` measured this: a
    `MappingProxyType` is `isinstance(x, Mapping)`'s worst input at 220 ns
    against 56 ns for the pair, and this runs per field per event. The same
    paragraph is why `tuple` is named beside `list` — the log freezes arrays into
    tuples, so a `list`-only test is one that works on decoded wire frames and
    silently does nothing on the live path, which is exactly how the first
    version of this shipped.

    A list is counted, because a list is never one-line material. A mapping is
    expanded one level, because that is where the readable facts usually are —
    `spent={turns=9}` is the answer somebody wanted — and counted below that.
    Named rather than dropped, which is `block_marker`'s rule one layer up.
    """
    if isinstance(value, (list, tuple)):
        return f"[{count_of(len(value), 'item')}]"
    if isinstance(value, (dict, MappingProxyType)):
        if nested:
            return f"{{{count_of(len(value), 'field')}}}"
        return "{" + ", ".join(f"{k}={brief_value(v, nested=True)}" for k, v in value.items()) + "}"
    return str(value)


def count_of(count: int, noun: str, plural: str = "") -> str:
    """`1 replacement`, `3 replacements` — the count and its noun, agreeing.

    `plural` is for nouns English does not pluralize by suffix (`entry` →
    `entries`); the default covers the regular case, which is every current
    caller.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {plural or f'{noun}s'}"


def duration(milliseconds: float) -> str:
    """`3d 4h`, `1h 30m`, `12m`, `8s` — the two largest units that are not zero.

    Zeros are dropped rather than kept for shape: a sweep every sixty seconds
    reads as `1m`, not `1m 0s`, and an hour and a half of quiet is `1h 30m`
    whether or not the seconds happen to be round.

    **A number, not an `object`.** It carried an `isinstance` guard and a `—` for
    "that was not a number", written when its callers were reading loose wire
    dictionaries; every one of them now reads a typed field, so the guard was a
    runtime check standing where a signature does the same work earlier and for
    free. A caller who really does hold an `object` has somewhere better to
    decide what it is than inside a formatter.

    Here rather than in `ph_app.agents`, where it was written, because the
    sidebar renders the same kind of thing (P9-07) and a second copy of a
    duration format is two sentences for one fact — which is the argument
    `truncation_marker` above makes for itself.
    """
    seconds = int(milliseconds // 1000)
    parts: list[str] = []
    for name, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count}{name}")
    return " ".join(parts[:2]) or "0s"


def one_line(text: str, limit: int = 120) -> str:
    """Whitespace collapsed and truncated with an ellipsis — a subtitle, a preview,
    a summary that must fit a row.

    Here rather than in `ph_app.wire`, where it was written, because a plugin needs
    the same row-sized line (ph-clm's section previews and the summaries its
    revisions carry) and may not import the app. The cut is `elide`'s.
    """
    return elide(" ".join(text.split()), limit)


ELLIPSIS = "…"


def elide(
    text: str, size: int, *, keep: Literal["head", "tail"] = "head", in_bytes: bool = False
) -> str:
    """`text` whole when it fits in `size` — code points, or UTF-8 bytes
    `in_bytes` — and otherwise as much of its start (`keep="head"`) or its end as
    fits beside an ellipsis that stands for the rest.

    The one way a cut is said: the ellipsis is one glyph and is counted inside
    `size`, so a 40-wide column is 40 wide, and the cut falls between characters
    (`ph.characters`). Whitespace is left as it is; `one_line` is this on one
    line."""
    measure = utf8_size if in_bytes else len
    if measure(text) <= size:
        return text
    room = max(size - measure(ELLIPSIS), 0)
    if keep == "tail":
        return ELLIPSIS + (tail_bytes(text, room) if in_bytes else tail(text, room))
    return (head_bytes(text, room) if in_bytes else head(text, room)) + ELLIPSIS


def around(text: str, start: int, size: int, *, stop: int | None = None) -> str:
    """As much of `text` as fits in `size` code points around `text[start:stop]`
    — a match, a signature — with an ellipsis on each side that was cut.

    For the excerpt that has to show what it was found by: a line's first few
    hundred characters do not hold a match that sits further along a minified
    one. Centered on the span where the text allows; a span longer than the
    room keeps its start. Each ellipsis is counted inside `size` and every cut
    falls between characters, as `elide`'s do, and `text` is whole when it fits.
    """
    if len(text) <= size:
        return text
    # The span widened to whole characters first: a match that ends inside one
    # is shown with all of it, rather than stopping at the character before.
    stop = cut_after(text, start if stop is None else max(stop, start))
    start = cut_before(text, start)
    room = max(size - 2 * len(ELLIPSIS), 0)
    lo = max(start - max(room - (stop - start), 0) // 2, 0)
    if lo == 0:
        return elide(text, size)
    if lo + size - len(ELLIPSIS) >= len(text):
        return elide(text, size, keep="tail")
    # Snapped back to a character, but never so far the span's end falls out of
    # the room — and never past the span's start, for a span wider than it.
    lo = min(start, max(cut_before(text, lo), cut_after(text, stop - room)))
    hi = cut_before(text, lo + room)
    if hi <= lo:
        # A room narrower than the one character there holds none of it, as
        # `elide`'s does: never more than `size`.
        return ELLIPSIS
    # Marked where the window was cut, which snapping it to characters can
    # move to an end of the text.
    left = ELLIPSIS if lo > 0 else ""
    right = ELLIPSIS if hi < len(text) else ""
    return f"{left}{text[lo:hi]}{right}"


def thousands(count: int) -> str:
    """`1777` → `1.8k`, `900` → `900`. One abbreviation for a token count.

    This module's own rule, and the fifth-time-wrongly had already happened on
    one screen: the subagent panel truncated (`row.tokens // 1000` → `1k`) while
    the footer's cache field rounded (`1.8k`), so the same magnitude was printed
    two ways a few rows apart. Rounding is the one kept, because the figure is
    read as a size rather than counted with.
    """
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def truncation_marker(dropped: int | None, cap: int) -> str:
    """The text that stands in for output a cap discarded (D4).

    Here because four things now discard output against a cap — the RLM kernel,
    the guest runner, `!!` and `tool-bash` — and *"a reader comparing a
    transcript to a log must not find two different sentences for the same
    event"* is the rule the first two were already written to keep, byte-identical
    and asserted so by `test_protocol_mirror`. The last two arrived with P7-13
    and each invented a wording, which is precisely the fifth-time-wrongly this
    module exists for.

    `ph_runtime.protocol` keeps a deliberate copy: that package ships into the
    guest venv with no dependencies at all, so it cannot import this. It is one
    of two — `ph_runtime._json` copies the one narrowing the guest imports out
    of `ph.json` — and `test_protocol_mirror` pins both.

    **`dropped` is `None` where the cut is made on a stream still running** —
    the kernel's pipes, the guest's prints — which cannot know yet how much will
    follow. The marker then says where the cut fell and no number, and the
    stream says its total once it has ended, in this same sentence with the
    count: a count written at the cut was only the write that crossed it, and
    read as the whole.
    """
    if dropped is None:
        return f"\n[ph: output truncated — the rest is not kept, cap {cap} bytes]\n"
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
