"""`/profile`: show what a session runs on and how its named profile moved, and change both.

This command is the user's half of session profiles (S7). The environment of a
session is its base and the overrides logged since. The command reads the
environment from the log of the session. It changes the environment only through
the functions of `ph.session_profile`, so the log records every change before
the mount makes it.

The verbs:

- `show [--full]` prints the base, the overrides in the order in which they
  apply, and each setting that the session runs with and that its base does not
  state. `--full` prints every environment row as it runs.
- `diff` compares the named profile as it composes now with the version that
  this session runs on (the listing of S6).
- `save <name> [--replace]` merges the base with the overrides, writes the
  result as the sparse named profile `name` (decision 5), and appends a
  `profile/saved` record.
- `use <name> [--clear]` makes another named profile the base. The overrides are
  kept over it or cleared (decision 11). A base changes rows that a live mount
  cannot follow, so the root starts again from its log after the command is
  durable (`Root.restart_wanted`). `use` with the name of the current base takes
  the current version of that profile.
- `clear [row]` stops applying one override, or all of them.

**The supervisor registers this command, and a row does not.** The ask desk is
registered the same way. Saving, switching and restarting belong to this host,
which has the named-profile store and the roots. A profile does not contribute
them. The command unwinds with the root that registered it.

@module ph_app.daemon.profile_command
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from ph.cordis import LoaderError
from ph.json import thaw_json
from ph.seams.commands import CommandDefinition, CommandVerb, Verbs
from ph.session_profile import (
    OverrideNotRecorded,
    ProfileBase,
    clear_overrides,
    environment_listing,
    listing,
    logged_environment,
    profile_change,
    record_adopted,
    record_saved,
    resolved_environment,
)
from ph.text import count_of
from ph.wire import validation_summary

from ..console import detail
from ..profiles import SaveRefused, named_version, save_session

if TYPE_CHECKING:
    # `supervisor` imports this module to register the command. The import in
    # the other direction is therefore for the type checker only.
    from .supervisor import Root

__all__ = ["profile_command"]

USAGE = (
    "usage: /profile [show [--full] | diff | save <name> [--replace] | "
    "use <name> [--clear] | clear [row]]"
)
HINT = USAGE.removeprefix("usage: /profile ")


class _Refused(Exception):
    """A `/profile` line that changes nothing. Its message says why."""


def profile_command(root: Root) -> CommandDefinition:
    """Build `/profile` for one root. The command reads and restarts that root."""

    def named(rest: str) -> str:
        """Return the profile that a verb acts on, which is the word after the
        verb. Refuse with the usage line when there is no such word."""
        if not rest:
            raise _Refused(USAGE)
        return rest.split()[0]

    showing = CommandVerb(
        lambda rest, _invocation: _show(root, full="--full" in rest.split()), record="after"
    )
    run = Verbs(
        {
            "": showing,
            "show": showing,
            "diff": CommandVerb(lambda _rest, _invocation: _diff(root), record="after"),
            "save": CommandVerb(
                lambda rest, invocation: _save(
                    root, named(rest), invocation.line, replace="--replace" in rest.split()
                )
            ),
            "use": CommandVerb(
                lambda rest, _invocation: _use(root, named(rest), clear="--clear" in rest.split())
            ),
            "clear": CommandVerb(
                lambda rest, invocation: _clear(
                    root, rest.split()[0] if rest else None, invocation.line
                )
            ),
        },
        otherwise=USAGE,
        refused=(_Refused, SaveRefused, OverrideNotRecorded),
    )

    return CommandDefinition(
        name="profile",
        summary="Show this session's profile, how it moved, and save or switch it.",
        argument_hint=HINT,
        run=run,
    )


def _show(root: Root, *, full: bool) -> str:
    env = logged_environment(root.session)
    if full:
        rows = thaw_json(resolved_environment(env))
        return yaml.safe_dump(rows, sort_keys=False, default_flow_style=False).rstrip()
    return "\n".join(environment_listing(env))


def _diff(root: Root) -> str:
    env = logged_environment(root.session)
    on = env.starts_on
    if on is None:
        return "This session has no recorded base yet."
    if not on.name:
        return "This session started on a profile file, so there is no named profile to compare."
    change = profile_change(env, _version(on.name))
    if change is None:
        return f"{on.name} is as this session runs it."
    return "\n".join(
        [*listing(change), f"`/profile use {on.name}` runs this session on the new version."]
    )


async def _save(root: Root, name: str, line: str, *, replace: bool) -> str:
    env = logged_environment(root.session)
    stamp = f"Saved from session {root.id} by `{line}` on {date.today().isoformat()}."
    path, entries = save_session(name, env, comment=stamp, replace=replace)
    await record_saved(root.ctx, root.session, name=name, path=str(path), entries=entries)
    return (
        f"Saved this session's environment as {name} ({count_of(len(entries), 'row')} "
        f"differ from the profile it extends): {path}. `/profile use {name}` runs this "
        "session on it; `phern --profile` starts others on it."
    )


async def _use(root: Root, name: str, *, clear: bool) -> str:
    """Adopt the current version of `name`, then have the root start again on it.

    **This is an adoption, as every base change after the first is.** It is the
    same as a "yes" to a moved profile, or as `phern profiles adopt`. The start
    that applies the adoption is therefore where the base changes. A version that
    will not mount is withdrawn there, and does not strand the session
    (`runtime.mount_session`). The overrides are cleared with the old base or
    apply over the new base, as the user chose (decision 11).
    """
    if root.held_on_profile:
        raise _Refused("This session is waiting on its profile question; answer that first.")
    if root.agent.status != "idle":
        raise _Refused("The agent is working; `/profile use` when it is idle.")
    version = _version(name)
    if not await record_adopted(root.ctx, root.session, version, clear=clear):
        raise _Refused(f"{name} was not adopted: the session log could not be written.")
    root.restart_wanted = True
    kept = len(logged_environment(root.session).overrides)
    said = [f"Starting this session again on {name}."]
    if kept and clear:
        said.append(f"Its {count_of(kept, 'override')} go with the old profile.")
    elif kept:
        said.append(
            f"Its {count_of(kept, 'override')} still apply over it, less any it already "
            "says; `/profile clear` drops them."
        )
    return " ".join(said)


async def _clear(root: Root, row: str | None, line: str) -> str:
    cleared, left = await clear_overrides(
        root.ctx, root.session, None if row is None else [row], command=line
    )
    if not cleared:
        return f"No override of {row} in this session." if row else "This session has no overrides."
    said = f"Cleared {count_of(len(cleared), 'override')}: {', '.join(cleared)}."
    if left:
        # An override turned a row on or off, or added a row. A live mount
        # cannot follow that change.
        root.restart_wanted = True
        return f"{said} Starting the session again, since {', '.join(left)} cannot change live."
    return f"{said} The session runs on its base for them now."


def _version(name: str) -> ProfileBase:
    """Return `name` as the base of a session would record it now. Refuse with
    the reason when the profile does not resolve or does not compose."""
    try:
        return named_version(name)
    except ValidationError as error:
        said = validation_summary(error, root="config")
        raise _Refused(f'profile "{name}" does not resolve: {said}') from error
    except (LoaderError, OSError, ValueError) as error:
        raise _Refused(f'profile "{name}" does not compose: {detail(error)}') from error
