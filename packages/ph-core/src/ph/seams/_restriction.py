"""One allow/deny rule, for every registry that scopes what a caller may reach.

Two registries ask this question — `ctx.tools` and `ctx.skills` — and they ask
it identically: a set that may be named, a set that may not, and the answer
composed by **intersection** so that a narrowing can never be undone by a
narrower scope adding one of its own.

The value type is shared and the machinery is not, deliberately. `ToolRuntime`
resolves through layers keyed by isolation with name shadowing, a presentation
mode and a reserved transport; `SkillService` is a flat table where installation
is global and only reach is scoped. Those differ for real reasons. What must not
differ is what `allow` and `deny` *mean* — `_names.py` exists one file over
because "two hand-written copies of one regex is how the two come to disagree
about what a name is", and this is that argument about a rule rather than a
pattern.

@module ph.seams._restriction
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NameFilter"]


@dataclass(frozen=True, slots=True)
class NameFilter:
    """Which names a scope may reach. Filters intersect; neither field widens.

    `None` means "no opinion" in both directions, which is what makes
    intersection the whole composition rule: a filter can only ever remove a
    name another filter allowed, never restore one another removed.
    """

    allow: frozenset[str] | None = None
    deny: frozenset[str] | None = None

    def admits(self, name: str) -> bool:
        if self.allow is not None and name not in self.allow:
            return False
        return not (self.deny is not None and name in self.deny)
