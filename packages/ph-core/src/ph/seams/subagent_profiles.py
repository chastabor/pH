"""A profile a parent assigns its child, read as a narrowing (session profiles, S7b).

A child runs inside its parent's scope and **on its parent's mount** (decision 21):
it never gets a mount of its own, so a profile assigned to it cannot be mounted. It
is read instead, row by row, against what the parent's mount runs:

* **what it runs** — a row the assigned profile runs that the parent's mount does
  not is refused, naming the row: there is no mount to run it on, and a child can
  hold less than its parent and never more;
* **what a row it drops gave** — the tools and skills of each row the parent runs
  and the assigned profile does not go with it, found by the row that registered
  each (`ToolRuntime.registrants`, `SkillService.registrants`);
* **what a row it keeps says** — asked of the row's own plugin, which declares how a
  child holds less of it (`plugin(..., narrows=)`, `ph.cordis.child_limit`): the
  `models` row's default, `sandbox-policy`'s posture, `skills-progressive`'s paths
  and `sandbox-allow`'s writable directories. A row joins by declaring one, and
  nothing here names it, as long as what it holds back is one of `ChildLimit`'s
  kinds.

What a row with no narrower says is the parent's, because it is the parent's mount
the child runs on: a retry ceiling or a compaction threshold cannot differ for one
agent of a mount. The resolved narrowing is what the admission records, so a child's
reach is fixed when it is admitted — not whatever the named file says later.

`NamedProfiles` is how the seam gets a named profile composed: the host that mounted
the parent knows where named profiles live (`ph_app`), and ph-core does not.

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
    """Compose a named profile by name — the host's, provided as `ctx.named_profiles`."""

    def compose(self, name: str) -> Profile:
        """`name` composed as a session's would be, or raise saying why not."""
        ...


@dataclass(frozen=True, slots=True)
class Narrowing:
    """What a child holds under an assigned profile, within its parent's reach."""

    tools: tuple[str, ...]
    skills: tuple[str, ...]
    """The parent's, less the dropped rows' and what a kept row withheld."""
    limit: ChildLimit
    """Every row it keeps, joined: its model, posture and writable directories."""


def narrowing(assigned: Profile, reach: ChildReach, *, held_tools: Sequence[str]) -> Narrowing:
    """`assigned` read against the parent's mount, `reach.ctx` — see the module
    docstring.

    `held_tools` and `reach.skills` are the parent's own reach (`SubagentService.
    held_by`), which is narrower than the mount's when the parent is a child itself.

    :raises NarrowingRefused: naming the row, for anything the parent does not hold.
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
    """One row's config as the parent's mount runs it and as the assigned profile
    says it, each read by the row's own model — what its narrower compares."""
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
    """What the parent holds, less what a row the assigned profile does not run gave."""
    return tuple(name for name in held if registrants.get(name) not in dropped)


def _skills(ctx: Context, held: Sequence[str], dropped: set[str]) -> tuple[str, ...]:
    """The parent's skills, less those a row the assigned profile does not run gave —
    by the row that registered each (`SkillService.registrants`), as tools are."""
    service = ctx.get(SKILLS)
    return tuple(held) if service is None else _given(service.registrants(), held, dropped)
