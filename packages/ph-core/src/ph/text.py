"""Prose helpers for text a person or a model reads.

Small on purpose. What lives here is formatting that appears in *more than one*
package and has a wrong answer — the kind of thing that is retyped correctly
four times and wrongly the fifth. `count_of` is here because that fifth time
already happened: a tool card shipped reading "1 governed calls" for a phase and
a half, in the one place among four that had inlined the ternary by hand.

@module ph.text
"""

from __future__ import annotations

__all__ = ["count_of", "truncation_marker"]


def count_of(count: int, noun: str, plural: str = "") -> str:
    """`1 replacement`, `3 replacements` — the count and its noun, agreeing.

    `plural` is for nouns English does not pluralize by suffix (`entry` →
    `entries`); the default covers the regular case, which is every current
    caller.
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {plural or f'{noun}s'}"


def truncation_marker(dropped: int, cap: int) -> str:
    """The text that stands in for output a cap discarded (D4).

    Here because four things now discard output against a cap — the RLM kernel,
    the guest runner, `!!` and `tool-bash` — and *"a reader comparing a
    transcript to a log must not find two different sentences for the same
    event"* is the rule the first two were already written to keep, byte-identical
    and asserted so by `test_protocol_mirror`. The last two arrived with P7-13
    and each invented a wording, which is precisely the fifth-time-wrongly this
    module exists for.

    `ph_runtime.protocol` keeps the one deliberate copy: that package ships into
    the guest venv with no dependencies at all, so it cannot import this.
    """
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
