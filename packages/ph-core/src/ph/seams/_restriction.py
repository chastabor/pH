"""One allow/deny rule, for every registry that scopes what a caller may reach.

Two registries ask this question, `ctx.tools` and `ctx.skills`, and they ask it
identically: a set that may be named, and a set that may not. The answer is
composed by **intersection**, so a narrower scope can never undo a narrowing by
adding a filter of its own.

The value type is shared and the machinery is not, deliberately. `ToolRuntime`
resolves through layers keyed by isolation, with name shadowing, a presentation
mode and a reserved transport. `SkillService` is a flat table where installation
is global and only reach is scoped. Those differ for real reasons. What must not
differ is what `allow` and `deny` *mean*. `_names.py` is beside this file
because two hand-written copies of one regex come to disagree about what a name
is. This module makes the same argument about a rule and not about a pattern.

@module ph.seams._restriction
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NameFilter"]


@dataclass(frozen=True, slots=True)
class NameFilter:
    """Which names a scope may reach. Filters intersect, and neither field widens.

    `None` means "no opinion" in both directions. That makes intersection the
    whole composition rule: a filter can only remove a name that another filter
    allowed, and can never restore a name that another filter removed.
    """

    allow: frozenset[str] | None = None
    deny: frozenset[str] | None = None

    def admits(self, name: str) -> bool:
        if self.allow is not None and name not in self.allow:
            return False
        return not (self.deny is not None and name in self.deny)
