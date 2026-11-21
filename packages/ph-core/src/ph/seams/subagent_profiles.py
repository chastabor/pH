"""A profile that a parent assigns to its child, read as a narrowing (session
profiles, S7b).

A child runs inside the scope of its parent and **on the mount of its parent**
(decision 21). It never gets a mount of its own, so a profile that is assigned
to it cannot be mounted. The profile is read instead, row by row, against what
the mount of the parent runs:

* **What it runs.** A row that the assigned profile runs and the mount of the
  parent does not run is refused, and the refusal names the row. There is no
  mount to run it on, and a child can hold less than its parent and never more.
* **What a dropped row gave.** A dropped row is a row that the parent runs and
  the assigned profile does not. Its tools and skills go with it. They are
  found by the row that registered each (`ToolRuntime.registrants`,
  `SkillService.registrants`).
* **What a kept row says.** The plugin of the row answers this, because the
  plugin declares how a child holds less of the row (`plugin(..., narrows=)`,
  `ph.cordis.child_limit`). The narrowers cover the default of the `models`
  row, the posture of `sandbox-policy`, the paths of `skills-progressive` and
  the writable directories of `sandbox-allow`. A row joins by declaring a
  narrower, and nothing here names the row, as long as what it holds back is
  one of the kinds of `ChildLimit`.

For a row with no narrower, what the row says for the parent applies to the
child, because the child runs on the mount of the parent. A retry ceiling or a
compaction threshold cannot differ for one agent of a mount. The admission
records the resolved narrowing. So the reach of a child is fixed when the child
is admitted, and it does not follow what the named file says later.

`NamedProfiles` is how the seam gets a named profile composed. The host that
mounted the parent (`ph_app`) knows where named profiles are kept, and ph-core
does not.

@module ph.seams.subagent_profiles
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from pydantic import BaseModel, ValidationError

from ..cordis import (
    ChildLimit,
    ChildReach,
    Context,
    ForkScope,
    NarrowingRefused,
    Profile,
    Row,
    interpolate,
)
from ..keys import MOUNT, SKILLS, TOOLS
from ..wire import validation_summary

__all__ = ["NamedProfiles", "Narrowing", "narrowing"]


class NamedProfiles(Protocol):
    """Composes a named profile by name. The host provides it as `ctx.named_profiles`."""

    def compose(self, name: str) -> Profile:
        """Return `name` composed as the profile of a session would be, or raise
        with the reason why not."""
        ...


@dataclass(frozen=True, slots=True)
class Narrowing:
    """What a child holds under an assigned profile, within its parent's reach."""

    tools: tuple[str, ...]
    skills: tuple[str, ...]
    """What the parent holds, less what the dropped rows gave and what a kept row
    withheld."""
    limit: ChildLimit
    """The limits of every row that the child keeps, joined: its model, posture
    and writable directories."""


def narrowing(assigned: Profile, reach: ChildReach, *, held_tools: Sequence[str]) -> Narrowing:
    """Read `assigned` against the mount of the parent, which is `reach.ctx`. The
    module docstring describes the reading.

    `held_tools` and `reach.skills` are the reach of the parent itself
    (`SubagentService.held_by`). That reach is narrower than the reach of the
    mount when the parent is itself a child.

    :raises NarrowingRefused: for anything that the parent does not hold. The
        message names the row.
    """
    mount = reach.ctx.require(MOUNT)
    running = _enabled(mount.profile)
    asked = _enabled(assigned)
    limit = ChildLimit()
    for row_id, row in asked.items():
        have = running.get(row_id)
        if have is None or have.name != row.name:
            raise NarrowingRefused(
                f"it runs {row_id}, which its parent does not; a child runs on its "
                "parent's mount, so it can hold less than its parent and never more"
            )
        fork = mount.forks.get(row_id)
        if fork is not None and fork.spec.narrows is not None:
            asking = replace(reach, row=row_id)
            limit = limit.join(fork.spec.narrows(*_configs(fork, row), asking))
    dropped = set(running) - set(asked)
    return Narrowing(
        tools=_given(reach.ctx.require(TOOLS).registrants(), held_tools, dropped),
        skills=tuple(
            name
            for name in _skills(reach.ctx, reach.skills, dropped)
            if name not in limit.withheld_skills
        ),
        limit=limit,
    )


def _configs(fork: ForkScope, row: Row) -> tuple[BaseModel | None, BaseModel | None]:
    """The config of one row as the mount of the parent runs it, and as the
    assigned profile states it. The config model of the row reads each of them.
    The narrower of the row compares the two."""
    try:
        asked = fork.spec.resolve_config(interpolate(row.config))
    except ValidationError as error:
        said = validation_summary(error, root=row.id)
        raise NarrowingRefused(f"its {row.id} row does not validate: {said}") from error
    return fork.spec.resolve_config(fork.config), asked


def _enabled(profile: Profile) -> dict[str, Row]:
    """A profile's environment rows that run, by id."""
    return {row.id: row for row in profile.rows_of({"environment"}) if not row.disabled}


def _given(registrants: dict[str, str], held: Sequence[str], dropped: set[str]) -> tuple[str, ...]:
    """What the parent holds, less what each dropped row gave. A dropped row is a
    row that the assigned profile does not run."""
    return tuple(name for name in held if registrants.get(name) not in dropped)


def _skills(ctx: Context, held: Sequence[str], dropped: set[str]) -> tuple[str, ...]:
    """The skills of the parent, less those that each dropped row gave. They are
    found by the row that registered each (`SkillService.registrants`), as tools
    are."""
    service = ctx.get(SKILLS)
    return tuple(held) if service is None else _given(service.registrants(), held, dropped)
