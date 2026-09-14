"""Prose helpers for text a person or a model reads.

Small on purpose. What lives here is formatting that appears in *more than one*
package and has a wrong answer — the kind of thing that is retyped correctly
four times and wrongly the fifth. `count_of` is here because that fifth time
already happened: a tool card shipped reading "1 governed calls" for a phase and
a half, in the one place among four that had inlined the ternary by hand.

@module ph.text
"""

from __future__ import annotations

from types import MappingProxyType

from .json import JsonValue

__all__ = [
    "NO_OUTPUT",
    "block_marker",
    "brief_value",
    "count_of",
    "thousands",
    "truncation_marker",
]

NO_OUTPUT = "(no output)"
"""What stands in for a command that printed nothing.

Named for the same reason `truncation_marker` is: `!` and `tool-bash` render
the same child process, and a reader comparing a transcript to a log must not
find two sentences for one event (D4). The TUI's shell card is a third answer
and deliberately a different one — an empty card body already reads as "it
said nothing", and a card need not say in words what its own emptiness says.
"""


def block_marker(kind: str) -> str:
    """`[media]`, `[tool-call]` — a content block a renderer skipped, named.

    Every renderer that shows blocks to a person owes the same answer: name what
    was not rendered rather than drop it, so a reader can see an image was there
    (`ph_app.tui.trajectory._text` states the rule). Five callers spelled this
    inline, and one of them already says something else.
    """
    return f"[{kind}]"


def brief_value(value: JsonValue, *, nested: bool = False) -> str:
    """One JSON value on one line, with containers named rather than dumped.

    Here because the fifth-time-wrongly had already happened twice. `str()` on a
    payload value is correct for a scalar and garbage for anything else — it
    emits the Python literal — and two readers in two packages both did it:
    `ph_app.wire.describe`, which renders 56 event types for the auditor view and
    `ph agents attach`, and `ph.commands.revert._clip`, which names the calls a
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


def thousands(count: int) -> str:
    """`1777` → `1.8k`, `900` → `900`. One abbreviation for a token count.

    This module's own rule, and the fifth-time-wrongly had already happened on
    one screen: the subagent panel truncated (`row.tokens // 1000` → `1k`) while
    the footer's cache field rounded (`1.8k`), so the same magnitude was printed
    two ways a few rows apart. Rounding is the one kept, because the figure is
    read as a size rather than counted with.
    """
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def truncation_marker(dropped: int, cap: int) -> str:
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
    """
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
