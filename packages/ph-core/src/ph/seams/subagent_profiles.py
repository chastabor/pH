"""A profile a parent assigns its child, read as a narrowing (session profiles, S7b).

A child runs inside its parent's scope and **on its parent's mount** (decision 21):
it never gets a mount of its own, so a profile assigned to it cannot be mounted. It
is read instead, row by row, against what the parent's mount runs, for the parts of
an environment a child can hold less of than its parent:

* **what it runs** — a row the assigned profile runs that the parent's mount does
  not is refused, naming the row: there is no mount to run it on, and a child can
  hold less than its parent and never more;
* **tools** — the tools of each row the parent runs and the assigned profile does
  not are taken away (by the row that registered them, `ToolRuntime.registrants`);
* **skills** — the `skills-progressive` row's paths, a subset of the parent's, and
  the skills found under them; without the row, none of those;
* **its model** — the `models` row's default, which must be a key the parent's own
  list holds, since the child runs on the parent's adapters;
* **its sandbox** — a `read-only` default makes the child read-only; a default wider
  than the posture the parent runs in is refused.

Everything else an assigned profile says is the parent's, because it is the parent's
mount the child runs on: a retry ceiling or a compaction threshold cannot differ for
one agent of a mount. The resolved narrowing is what the admission records, so a
child's reach is fixed when it is admitted — not whatever the named file says later.

`NamedProfiles` is how the seam gets a named profile composed: the host that mounted
the parent knows where named profiles live (`ph_app`), and ph-core does not.

@module ph.seams.subagent_profiles
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..cordis import Context, Profile, Row
from ..cordis.loader import Mount
from ..keys import MODELS, MOUNT, SKILLS, TOOLS
from .models import ModelChoice, ModelChoiceError, ModelList
from .sandbox import SandboxMode, default_mode_of, narrower
from .skills import progressive_paths

__all__ = ["NamedProfiles", "Narrowing", "NarrowingRefused", "narrowing"]


class NamedProfiles(Protocol):
    """Compose a named profile by name — the host's, provided as `ctx.named_profiles`."""

    def compose(self, name: str) -> Profile:
        """`name` composed as a session's would be, or raise saying why not."""
        ...


class NarrowingRefused(ValueError):
    """An assigned profile asks for more than the parent holds; the message names what."""


@dataclass(frozen=True, slots=True)
class Narrowing:
    """What a child holds under an assigned profile, within its parent's reach."""

    tools: tuple[str, ...]
    skills: tuple[str, ...]
    model_key: str | None
    """The assigned profile's default model, a key its parent lists; `None` when it
    lists none, so the child runs on its parent's."""
    read_only: bool


def narrowing(
    ctx: Context,
    assigned: Profile,
    *,
    held_tools: Sequence[str],
    held_skills: Sequence[str],
    boundary: Context,
    parent_mode: SandboxMode,
) -> Narrowing:
    """`assigned` read against the mount of `ctx` — see the module docstring.

    `held_tools` and `held_skills` are the parent's own reach (`SubagentService.
    held_by`), which is narrower than the mount's when the parent is a child itself.

    :raises NarrowingRefused: naming the row, for anything the parent does not hold.
    """
    mount = ctx.require(MOUNT)
    running = _enabled(mount.profile)
    asked = _enabled(assigned)
    for row_id, row in asked.items():
        have = running.get(row_id)
        if have is None or have.name != row.name:
            raise NarrowingRefused(
                f"it runs {row_id}, which its parent does not; a child runs on its "
                "parent's mount, so it can hold less than its parent and never more"
            )
    dropped = set(running) - set(asked)
    return Narrowing(
        tools=_tools(ctx, mount, held_tools, dropped),
        skills=_skills(ctx, mount.profile, assigned, held_skills, boundary),
        model_key=_model(ctx, assigned),
        read_only=_read_only(assigned, parent_mode),
    )


def _enabled(profile: Profile) -> dict[str, Row]:
    """A profile's environment rows that run, by id."""
    return {row.id: row for row in profile.rows_of({"environment"}) if not row.disabled}


def _tools(ctx: Context, mount: Mount, held: Sequence[str], dropped: set[str]) -> tuple[str, ...]:
    """The parent's tools, less those a row the assigned profile does not run gave."""
    rows = {fork.ctx: row_id.split("/")[0] for row_id, fork in mount.forks.items() if fork.ctx}
    registrants = ctx.require(TOOLS).registrants()

    def row_of(name: str) -> str | None:
        scope: Context | None = registrants.get(name)
        while scope is not None and scope not in rows:
            scope = scope.parent
        return rows.get(scope) if scope is not None else None

    return tuple(name for name in held if row_of(name) not in dropped)


def _skills(
    ctx: Context, running: Profile, assigned: Profile, held: Sequence[str], boundary: Context
) -> tuple[str, ...]:
    """The parent's skills, less those found under paths the assigned profile lacks."""
    theirs, mine = progressive_paths(running), progressive_paths(assigned)
    service = ctx.get(SKILLS)
    if theirs is None or service is None or mine == theirs:
        return tuple(held)
    kept = mine or []
    extra = [path for path in kept if path not in theirs]
    if extra:
        raise NarrowingRefused(
            f"its skills-progressive reads {', '.join(map(str, extra))}, which its parent's "
            "does not"
        )

    def found_under_kept(name: str) -> bool:
        skill = service.get(name, boundary)
        if skill is None or skill.source != "skills-progressive" or skill.path is None:
            return True
        where = Path(skill.path).resolve()
        return any(where.is_relative_to(path) for path in kept)

    return tuple(name for name in held if found_under_kept(name))


def _model(ctx: Context, assigned: Profile) -> str | None:
    """The assigned profile's default model, which the parent's list must hold."""
    key = ModelList.of(assigned).default
    if not key:
        return None
    try:
        (ctx.get(MODELS) or ModelList()).resolve(ModelChoice(key=key))
    except ModelChoiceError as error:
        raise NarrowingRefused(
            f"its models row runs on {key}, which its parent does not list: {error}"
        ) from error
    return key


def _read_only(assigned: Profile, parent_mode: SandboxMode) -> bool:
    """Whether the assigned sandbox makes the child read-only — refused when wider."""
    mode = default_mode_of(assigned)
    if mode is None:
        return False
    if narrower(mode, parent_mode) != mode:
        raise NarrowingRefused(
            f"its sandbox-policy lets a child run {mode}, where its parent runs {parent_mode}"
        )
    return mode == "read-only"
