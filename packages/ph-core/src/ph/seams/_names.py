"""What a registrable name may be, for the seams whose names a human types.

Two seams take a **slug**: `ctx.skills` (a skill's name) and `ctx.tui_screens`
(a screen's id). Same format, for the same reason — both become an addressable
token someone types, an entry in a catalog the model reads and a `/<id>` at the
prompt, and neither may contain a space, because a name with one in it is a
command whose argument is part of its name. Only the *bound* differs, so the
bound is the only parameter; two hand-written copies of one regex is how the
two come to disagree about what a name is, and how a person meets two different
sentences refusing the same mistake.

**Deliberately not every name under `ph.seams`.** `code_runtime`'s binding names
are a portable *language identifier* with a reserved-word list — a different
rule that happens to also be about names. Folding it in would give one function
two vocabularies and its callers a flag to pick between them, which is a worse
trade than the four lines it saves.

@module ph.seams._names
"""

from __future__ import annotations

import re

__all__ = ["SLUG_CHARACTERS", "require_slug", "slug_pattern"]

SLUG_CHARACTERS = "a-z0-9-"
"""Lowercase, digits, hyphen. Named so the rule and the sentence that reports a
violation cannot drift apart."""


def slug_pattern(maximum: int) -> re.Pattern[str]:
    """The compiled slug rule for one bound.

    `re` caches compiled patterns internally, so a seam may hold the result as a
    module constant (`skills.NAME_PATTERN` does, because a reader that *tests*
    rather than raises needs it) and a caller may ask again per registration.
    """
    return re.compile(rf"^[{SLUG_CHARACTERS}]{{1,{maximum}}}$")


def require_slug(value: str, *, maximum: int, kind: str) -> None:
    """Refuse `value` unless it is a slug of 1..`maximum` characters.

    `kind` names what was being registered — "skill name", "screen id" — so the
    refusal says which vocabulary was violated while the rule itself stays one.
    """
    if slug_pattern(maximum).match(value) is None:
        raise ValueError(
            f'"{value}" is not a {kind}: 1..{maximum} of lowercase [{SLUG_CHARACTERS}]'
        )
