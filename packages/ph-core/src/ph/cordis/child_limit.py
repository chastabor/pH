"""How a row says a child holds less of it: `plugin(..., narrows=)` (session profiles, S7b).

A child runs on the mount of its parent (decision 21). A profile that its parent
assigns to it is therefore read and not mounted. Each row that the child runs is
read **by the plugin that owns the row**, because the owner knows what its config
means and what "less" means for it. A plugin declares that beside its body, as
`narrows=` on the decorator. The narrowing seam (`ph.seams.subagent_profiles`)
asks each row that the assigned profile runs. A new row joins by saying so where
it is written, as long as what it holds back is one of the kinds of
`ChildLimit`. A new kind is a field here and on the grant of the child.

This module is in the package of the loader because the declaration is part of
the identity of a plugin (`PluginSpec.narrows`), as its config model is. The
values are plain names and paths, so nothing here knows a seam.

@module ph.cordis.child_limit
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context

__all__ = ["ChildLimit", "ChildReach", "NarrowingRefused", "Narrows"]


class NarrowingRefused(ValueError):
    """An assigned profile asks for more than the parent holds. The message names what."""


@dataclass(frozen=True, slots=True)
class ChildReach:
    """The side of the parent in one narrowing: what the narrower of a row reads it from."""

    ctx: Context
    """The mount of the parent: every seam that a narrower asks."""
    boundary: Context
    """The scope that the delegation is made from: the scope of the parent agent."""
    agent: str
    """The id of the parent agent, for a seam that answers per agent (its sandbox)."""
    skills: tuple[str, ...]
    """The skills that the parent holds. A narrower may withhold some of them."""
    row: str = ""
    """The id of the row that is asked. The narrowing sets it for each row that it
    asks, so that a plugin mounted as two rows can tell its own registrations
    apart."""


@dataclass(frozen=True, slots=True)
class ChildLimit:
    """What one row keeps back from a child. The default keeps nothing back."""

    model_key: str | None = None
    """The model that the child runs on, as a key that its parent lists. `None` runs
    the child on the model of the parent."""
    read_only: bool = False
    withheld_skills: frozenset[str] = frozenset()
    writable_paths: tuple[str, ...] | None = None
    """The extra directories that the sandbox of the child binds writable, in
    canonical form. `None` binds the directories of the parent."""

    def join(self, other: ChildLimit) -> ChildLimit:
        """Both limits at once: the tighter of each.

        :raises NarrowingRefused: when two rows name different models, because a
            child runs on one.
        """
        if self.model_key and other.model_key and self.model_key != other.model_key:
            raise NarrowingRefused(
                f"it names two models for its child, {self.model_key} and {other.model_key}"
            )
        paths = self.writable_paths
        if other.writable_paths is not None:
            paths = (
                other.writable_paths
                if paths is None
                else tuple(path for path in paths if path in other.writable_paths)
            )
        return ChildLimit(
            model_key=self.model_key or other.model_key,
            read_only=self.read_only or other.read_only,
            withheld_skills=self.withheld_skills | other.withheld_skills,
            writable_paths=paths,
        )


type Narrows[C] = Callable[[C, C, ChildReach], ChildLimit]
"""The narrower of a plugin. It takes the row as the mount of the parent runs it,
the same row as the assigned profile configures it, and the side of the parent.
It returns what the child is held to, or raises `NarrowingRefused` with a message
that says what the assigned row asks beyond its parent."""
