"""Naming one event vocabulary or the other, and one namespace within it (P6-33).

pH has **two** event vocabularies that share a `namespace/name` spelling and are
otherwise unrelated:

* the **cordis bus** — in-process pub/sub between plugins, carrying live Python
  objects, gone when the process is. `ph events` lists it.
* the **session log** — durable, frozen-JSON, append-only. The evidence I3 and I4
  are about.

They overlap in **zero** names, yet six roots appear in both (`agent`,
`approval`, `fs`, `harness`, `llm`, `session`) and the log's `tool/*` sits one
letter from the bus's `tools/*`. So a bare `workspace` or `tool` typed into a
filter does not say which vocabulary is meant, and `tool` typed into a substring
matcher silently catches `tools/` as well.

A selector says both things at once:

```
log:workspace            every workspace/* session-log type
log:workspace/acquired   exactly one
bus:tools/*              every tools/* bus event
log:*                    every session-log type
workspace                whichever vocabulary the surface serves
```

**Namespaced in the selector, not in the data.** Prefixing the stored types with
`event/` was the alternative and was refused: 944 literal type strings across 107
files, a session-format bump making every stored log unreadable, and it would not
fix the collision it is aimed at — `event/tool` is still one letter from `tools`.
Nothing anywhere holds one type string and asks "bus or log?", so the scheme
belongs to the *question*, which is what this module is.

**Matching is segment-aware, and that is the whole point.** A substring test is
exactly what makes `tool` catch `tools/`; comparing whole segments is what stops
it. `workspace` and `workspace/*` mean the same thing, because a namespace with
nothing after it can only mean all of it.

@module ph.selectors
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

__all__ = [
    "SCHEMES",
    "Scheme",
    "Selector",
    "SelectorError",
    "matches_any",
    "parse",
    "parse_all",
    "unknown_namespaces",
]

Scheme: TypeAlias = Literal["log", "bus"]
"""Which vocabulary a selector is about.

Two, and closed: they are the two that exist, and a third would be a new kind of
event rather than a new name for one of these. `log` is the durable session log
(`KNOWN_SESSION_EVENT_TYPES`); `bus` is the cordis dispatch registry
(`EventRegistry`).
"""

SCHEMES: tuple[Scheme, ...] = ("log", "bus")
"""The vocabulary, for an error message that names what *is* accepted.

A refusal that says only "unknown scheme" makes the reader guess; one that lists
the two costs nothing and ends the guessing.
"""

_ANY = "*"


class SelectorError(ValueError):
    """A selector is malformed, or names a vocabulary the caller cannot serve."""


@dataclass(frozen=True, slots=True)
class Selector:
    """One vocabulary, and a namespace prefix within it.

    `segments` empty means *everything in this scheme* — the parsed form of `*`.
    A frozen value because a surface parses its selectors once and then asks them
    per event, thousands of times; nothing about a selector changes after parsing.
    """

    scheme: Scheme
    segments: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.scheme}:{'/'.join(self.segments) or _ANY}"

    def matches(self, name: str) -> bool:
        """Whether `name` is at or under this selector's namespace.

        **Segment-aware, never a substring test.** `log:tool` must not match
        `tools/change`, and a `startswith` would — the two vocabularies really do
        differ by one letter there, which is the case this method exists for.

        The scheme is *not* checked here, deliberately: a bare type string does
        not say which vocabulary it came from, so a `matches` that pretended to
        check would be guessing. `parse_all` settles the scheme once, against the
        surface that knows, and refuses a mismatch there — where it can say why.
        """
        # A slice shorter than `segments` can never compare equal to it, so no
        # length guard is needed; and `parts[:0] == ()` already answers the
        # match-everything case. `parse` refuses an empty segment, so the one
        # input that would read oddly here — `("",)` — cannot be constructed.
        return tuple(name.split("/"))[: len(self.segments)] == self.segments


def parse(text: str, *, scheme: Scheme | None = None) -> Selector:
    """One selector, with `scheme` supplying the vocabulary when the text omits it.

    An explicit `log:` or `bus:` in the text wins. `scheme` is the surface's own
    vocabulary — the trajectory view shows session-log events only, so `workspace`
    typed there means `log:workspace` and the terse form stays terse.

    A trailing `/*` is sugar for the namespace itself: `workspace/*` and
    `workspace` select the same set, because a namespace with nothing under it
    could not mean anything else.
    """
    raw = text.strip()
    if not raw:
        raise SelectorError("a selector cannot be empty")
    head, colon, tail = raw.partition(":")
    if colon:
        if head not in SCHEMES:
            raise SelectorError(
                f'unknown selector scheme "{head}" in {text!r}; '
                f"expected one of {', '.join(SCHEMES)}"
            )
        # `head not in SCHEMES` narrows `str` to `Scheme`, so the explicit
        # prefix simply replaces the fallback rather than needing a second name.
        scheme, raw = head, tail.strip()
    if scheme is None:
        raise SelectorError(
            f"{text!r} names no vocabulary and none was supplied; "
            f"write {SCHEMES[0]}:{raw} or {SCHEMES[1]}:{raw}"
        )
    if not raw:
        raise SelectorError(f"{text!r} names a vocabulary and no namespace; write {scheme}:{_ANY}")
    # No early return for a bare `*`: it falls through as a one-element list
    # whose trailing star is popped, leaving the empty segments that mean "all".
    parts = raw.split("/")
    if parts[-1] == _ANY:
        parts.pop()
    for part in parts:
        if not part:
            raise SelectorError(f"{text!r} has an empty path segment")
        if _ANY in part:
            # Refused rather than treated as a wildcard: a selector is a
            # namespace prefix, and `work*/acquired` would be a glob — a second,
            # richer language whose only user would be the person who typed it by
            # accident.
            raise SelectorError(
                f'{text!r}: "{_ANY}" is only a whole trailing segment, not part of one'
            )
    return Selector(scheme=scheme, segments=tuple(parts))


def parse_all(patterns: Iterable[str], *, vocabulary: Scheme) -> list[Selector]:
    """Parse for one surface: default the scheme, and refuse a foreign one.

    **The one call a surface makes**, because it fuses the two things a surface
    must not get wrong. `vocabulary` is what this surface can actually answer
    about, so it serves as the default for a bare pattern *and* as the check on an
    explicit one.

    An explicit scheme the surface cannot serve is a **refusal, not an empty
    result**: `bus:tools` typed into the trajectory is a person asking a question
    of the wrong view, and answering "no matching records" would let them
    conclude there are none.
    """
    selectors = [parse(pattern, scheme=vocabulary) for pattern in patterns]
    foreign = [one for one in selectors if one.scheme != vocabulary]
    if foreign:
        raise SelectorError(
            f"{', '.join(str(one) for one in foreign)}: this view does not serve that "
            f'vocabulary; it holds "{vocabulary}" events'
        )
    return selectors


def matches_any(name: str, selectors: Sequence[Selector]) -> bool:
    """Whether any selector covers `name`. **No selectors means no filter.**

    An empty list is "the caller asked for nothing", which is every unfiltered
    read — so it admits everything rather than nothing. The opposite default
    would make an absent `--type` return an empty log.
    """
    return not selectors or any(one.matches(name) for one in selectors)


def unknown_namespaces(selectors: Iterable[Selector], known: Collection[str]) -> list[str]:
    """Selectors whose namespace no known name occupies — a typo report.

    **Separate from `matches`, and never called by it.** A stored log may
    legitimately carry types this build does not know: that is what `ignorable`
    exists for, and a matcher that refused an unrecognised namespace would break
    reading a log written by a newer harness. So matching stays mechanical and
    this is the opt-in half, for a command that would rather say "no such
    namespace" than return nothing and let the reader conclude there is nothing
    there.

    Returns the selectors' own strings, in order, deduplicated — a caller printing
    them wants what the person typed, not a reconstruction.
    """
    return list(
        dict.fromkeys(
            str(one)
            for one in selectors
            # `one.matches`, not a second copy of it. The rule this module exists
            # to state once was written out again here, and a matcher that drifts
            # from its own typo reporter is exactly the divergence the module
            # argues against. A catch-all names no namespace, so it cannot name a
            # wrong one.
            if one.segments and not any(one.matches(name) for name in known)
        )
    )
