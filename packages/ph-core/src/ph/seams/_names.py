"""What a registrable name may be, for the seams whose names a human types.

The format is a **slug**. Two seams took it first: `ctx.skills` (the name of a
skill) and `ctx.tui_screens` (the id of a screen). They share the format for one
reason. Each name becomes an addressable token that someone types: an entry in a
catalog that the model reads, and a `/<id>` at the prompt. Neither may contain a
space, because a name with a space in it is a command whose argument is part of
its name. `ctx.tui_status`, `ctx.diagnostics`, `ctx.invariants` and
`ph.seams.models` now hold their ids and keys to the same format.

Only the *bound* differs, so the bound is the only parameter. With two
hand-written copies of one regex, the two seams come to disagree about what a
name is, and a person meets two different sentences that refuse the same
mistake.

**The rule deliberately does not cover every name under `ph.seams`.** The
binding names of `code_runtime` are a portable *language identifier* with a list
of reserved words. That is a different rule that is also about names. To fold it
in would give one function two vocabularies, and give its callers a flag to
choose between them. That is a worse trade than the four lines that it saves.

@module ph.seams._names
"""

from __future__ import annotations

import re

__all__ = ["SLUG_CHARACTERS", "require_slug", "slug_pattern", "slugify"]

SLUG_CHARACTERS = "a-z0-9-"
"""Lowercase letters, digits and the hyphen. The set has a name so that the rule
and the sentence that reports a violation cannot drift apart."""


def slug_pattern(maximum: int) -> re.Pattern[str]:
    """The compiled slug rule for one bound.

    `re` caches compiled patterns internally. A seam may therefore hold the
    result as a module constant, and a caller may ask again for each
    registration. `skills.NAME_PATTERN` is such a constant, because a reader
    that *tests* a name and does not raise needs it.
    """
    return re.compile(rf"^[{SLUG_CHARACTERS}]{{1,{maximum}}}$")


def require_slug(value: str, *, maximum: int, kind: str) -> None:
    """Refuse `value` unless it is a slug of 1..`maximum` characters.

    `kind` names what was being registered ("skill name", "screen id"). The
    refusal then says which vocabulary was violated, and the rule itself stays
    one rule.
    """
    if slug_pattern(maximum).match(value) is None:
        raise ValueError(
            f'"{value}" is not a {kind}: 1..{maximum} of lowercase [{SLUG_CHARACTERS}]'
        )


def slugify(value: str, *, maximum: int) -> str:
    """`value` made a slug of at most `maximum` characters, or `""` when nothing is left.

    A run of characters outside `SLUG_CHARACTERS` becomes one hyphen, and no
    hyphen leads or trails. So the result passes `require_slug` whenever it is
    not empty.
    """
    slug = re.sub(rf"[^{SLUG_CHARACTERS}]+", "-", value.lower()).strip("-")
    return slug[:maximum].rstrip("-")
