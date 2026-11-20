"""Naming one event vocabulary or the other, and one namespace within it (P6-33).

pH has two unrelated event vocabularies that share a `namespace/name` spelling:
the **cordis bus** (in-process pub/sub between plugins, listed by `phern events`)
and the **session log** (durable, frozen-JSON, append-only). A selector names
the vocabulary and the namespace at once:

```
log:workspace            every workspace/* session-log type
log:workspace/acquired   exactly one
bus:tools/*              every tools/* bus event
log:*                    every session-log type
workspace                whichever vocabulary the surface serves
```

This module enforces these invariants:

* **Matching compares whole segments, never substrings.** The `tool/*` of the
  log is one letter from the `tools/*` of the bus, and `log:tool` must not reach
  it.
* `workspace` and `workspace/*` select the same set.
* An explicit scheme that a surface cannot serve is refused, and is not answered
  with an empty result.
* An empty list of selectors admits everything.

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
"""Which vocabulary a selector is about: the durable session log
(`KNOWN_SESSION_EVENT_TYPES`) or the cordis dispatch registry (`EventRegistry`).
"""

SCHEMES: tuple[Scheme, ...] = ("log", "bus")
"""The vocabulary, so a refusal can name what *is* accepted."""

_ANY = "*"


class SelectorError(ValueError):
    """A selector is malformed, or names a vocabulary the caller cannot serve."""


@dataclass(frozen=True, slots=True)
class Selector:
    """One vocabulary, and a namespace prefix within it. Empty `segments` means
    *everything in this scheme*, which is the parsed form of `*`.
    """

    scheme: Scheme
    segments: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.scheme}:{'/'.join(self.segments) or _ANY}"

    def matches(self, name: str) -> bool:
        """Whether `name` is at or under the namespace of this selector.

        **The test compares segments, and is never a substring test**: `log:tool`
        must not match `tools/change`. The scheme is deliberately *not* checked
        here. A bare type string does not say which vocabulary it came from, so
        `parse_all` settles that one time, against the surface that knows.
        """
        # A slice shorter than `segments` never compares equal to it, so no
        # length guard is needed. `parts[:0] == ()` answers the match-everything
        # case.
        return tuple(name.split("/"))[: len(self.segments)] == self.segments


def parse(text: str, *, scheme: Scheme | None = None) -> Selector:
    """One selector. `scheme` supplies the vocabulary when the text omits it.

    An explicit `log:` or `bus:` in the text wins. A trailing `/*` is another
    way to write the namespace itself. The function raises `SelectorError` on an
    empty, unscoped or malformed pattern, and does not guess at it.
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
        # The guard above narrows `str` to `Scheme`, so the prefix can replace
        # the fallback, and a second name is not needed.
        scheme, raw = head, tail.strip()
    if scheme is None:
        raise SelectorError(
            f"{text!r} names no vocabulary and none was supplied; "
            f"write {SCHEMES[0]}:{raw} or {SCHEMES[1]}:{raw}"
        )
    if not raw:
        raise SelectorError(f"{text!r} names a vocabulary and no namespace; write {scheme}:{_ANY}")
    # There is no early return for a bare `*`. It falls through as a one-element
    # list whose trailing star is popped. That leaves the empty segments that
    # mean "all".
    parts = raw.split("/")
    if parts[-1] == _ANY:
        parts.pop()
    for part in parts:
        if not part:
            raise SelectorError(f"{text!r} has an empty path segment")
        if _ANY in part:
            # A selector is a namespace prefix, not a glob. `work*/acquired` is
            # refused, and is not given a second, richer matching language.
            raise SelectorError(
                f'{text!r}: "{_ANY}" is only a whole trailing segment, not part of one'
            )
    return Selector(scheme=scheme, segments=tuple(parts))


def parse_all(patterns: Iterable[str], *, vocabulary: Scheme) -> list[Selector]:
    """Parse for one surface: default the scheme, and refuse a foreign one.

    `vocabulary` is what this surface can answer about, so it is both the
    default for a bare pattern and the check on an explicit one. A scheme that
    the surface cannot serve is a **refusal, not an empty result**. One refusal
    lists every offender.
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
    """Whether any selector covers `name`. **No selectors means no filter**: an empty
    list admits everything, so an absent `--type` does not return an empty log.
    """
    return not selectors or any(one.matches(name) for one in selectors)


def unknown_namespaces(selectors: Iterable[Selector], known: Collection[str]) -> list[str]:
    """Selectors whose namespace no known name occupies. This is a report of
    typos.

    **`matches` never consults this.** A stored log may legitimately carry types
    that this build does not know. So matching stays mechanical, and this check
    is the optional part. The function returns the strings of the selectors, in
    order, without duplicates. A catch-all names no namespace, so it is never
    reported.
    """
    return list(
        dict.fromkeys(
            str(one)
            for one in selectors
            # `one.matches`, never a second copy of the segment rule.
            if one.segments and not any(one.matches(name) for name in known)
        )
    )
