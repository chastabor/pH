"""How a row says a child holds less of it: `plugin(..., narrows=)` (session profiles, S7b).

A child runs on its parent's mount (decision 21), so a profile its parent assigns it is
read rather than mounted, and each row it runs is read **by the plugin that owns the
row**: the owner knows what its config means, and what "less" means for it. A plugin
declares that beside its body, as `narrows=` on the decorator, and the narrowing seam
(`ph.seams.subagent_profiles`) asks each row the assigned profile runs. A new row joins
by saying so where it is written, as long as what it holds back is one of
`ChildLimit`'s kinds; a new kind is a field here and on the child's grant.

Here, in the loader's package, because the declaration is part of a plugin's identity
(`PluginSpec.narrows`) the way its config model is; the values are plain names and
paths, so nothing here knows a seam.

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
    """An assigned profile asks for more than the parent holds; the message names what."""


@dataclass(frozen=True, slots=True)
class ChildReach:
    """The parent's side of one narrowing: what a row's narrower reads it from."""

    ctx: Context
    """The parent's mount — every seam a narrower asks."""
    boundary: Context
    """The scope the delegation is made from: the parent agent's."""
    agent: str
    """The parent agent's id, for a seam that answers per agent (its sandbox)."""
    skills: tuple[str, ...]
    """The skills the parent holds, which a narrower may withhold some of."""
    row: str = ""
    """The id of the row being asked — set by the narrowing for each row it asks,
    so a plugin mounted as two rows tells its own registrations apart."""


@dataclass(frozen=True, slots=True)
class ChildLimit:
    """What one row keeps back from a child. The default keeps nothing back."""

    model_key: str | None = None
    """The model the child runs on, a key its parent lists; `None` runs it on the
    parent's."""
    read_only: bool = False
    withheld_skills: frozenset[str] = frozenset()
    writable_paths: tuple[str, ...] | None = None
    """The extra directories the child's sandbox binds writable, canonical; `None`
    binds the parent's."""

    def join(self, other: ChildLimit) -> ChildLimit:
        """Both limits at once — the tighter of each.

        :raises NarrowingRefused: when two rows name different models, since a child
            runs on one.
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
"""A plugin's narrower: its row as the parent's mount runs it, the same row as the
assigned profile configures it, and the parent's side. Returns what the child is held
to, or raises `NarrowingRefused` saying what the assigned row asks beyond its parent."""
