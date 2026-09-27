"""`phern profiles` — what a named profile sets, and folding the old layers into it (S2).

`show` answers "what does this profile say" two ways (decision 16): what the
person's file sets — sparse, the rows that differ — and, with `--full`, every
setting a session on it runs with, defaults included. The second is for debugging
and for a first session, deciding what to change.

`diff` and `adopt` are S6's: a named profile that moved since a session started is
kept by that session until a person takes the new version on purpose — ahead of
its next start, for one session or every one on the profile (decisions 10, 18).

`fold` is item 0 of `plans/Session_Profiles_Plan.md`, run on purpose: a profile's
drop-ins, and a file still in the list format before S2, become the one named
profile file. It composes the profile before and after and keeps the change only
when the two agree, so a session that starts on it next notices nothing.

@module ph_app.profiles_cli
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from functools import partial
from pathlib import Path
from typing import Annotated

import anyio
import typer
import yaml
from pydantic import ValidationError

from ph.cordis import LoaderError, sparse_entries
from ph.documents import decode_document
from ph.json import as_seq, as_str, thaw_json
from ph.paths import RuntimeDirError, resolve_roots, write_atomic
from ph.persistence import LineageError, SessionBusy, materialize, stored_session
from ph.persistence.jsonl import read_stored
from ph.seams.skills import READ as SKILL_READ
from ph.seams.skills import read_summary
from ph.session_profile import (
    LoggedEnvironment,
    ProfileBase,
    ProfileChange,
    base_of,
    environment_listing,
    fold_environment,
    listing,
    profile_change,
    record_adopted,
    resolved_environment,
)
from ph.text import count_of
from ph.wire import validation_errors

from . import verbs
from .console import detail, emit, fail
from .daemon.client import DaemonClient, connected
from .named_profiles import append_rows, render_named_profile
from .params import AdoptParams
from .profiles import (
    compose_profile,
    log_host,
    profile_or_exit,
    profile_plan,
    sparse_text,
    unfolded_profiles,
)
from .protocol import DaemonError, DaemonGone
from .runtime import mounted
from .sessions import recorded_environment, stored_on

__all__ = ["FoldRefused", "adopt_version", "fold_profile", "profiles_app"]

profiles_app = typer.Typer(
    help="Named profiles: what they set, and folding the older layers into them.",
    no_args_is_help=True,
)

ProfileName = Annotated[str, typer.Argument(help="A profile name, or a path to a .yaml.")]
SessionChoice = Annotated[
    str,
    typer.Option("--session", help="One stored session; every one on the profile when left out."),
]


@profiles_app.command()
def show(
    name: ProfileName,
    full: Annotated[
        bool,
        typer.Option("--full", help="Every setting a session on it runs with, defaults included."),
    ] = False,
) -> None:
    """Print what a named profile sets — or, with `--full`, everything it runs with.

    `--full` is the session's environment: its rows only, each through its
    plugin's own model, so a row the profile never mentions still shows every
    setting it would run with. The host's and the TUI's rows are not a session's;
    `phern config` and `--dump-config` show those.
    """
    if full:
        rows = _resolved_or_exit(
            profile_or_exit(name), lambda composed: composed.resolved({"environment"})
        )
        emit(yaml.safe_dump(thaw_json(rows), sort_keys=False, default_flow_style=False).rstrip())
        return
    try:
        plan = profile_plan(name)
    except (LoaderError, OSError, ValueError) as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
    named, dropins = plan.named, plan.dropins
    if named is None:
        emit(
            f'"{name}" is a shipped profile, and no file of yours layers over it; '
            "`--full` lists every setting it runs with."
        )
        return
    notes = [f"{name or named.path.name}: {named.path}"]
    if named.legacy:
        notes.append(f"In the list format before S2: `phern profiles fold {name}` converts it.")
    if dropins:
        notes.append(
            f"And {len(dropins)} drop-in(s) under {dropins[0].parent}, "
            f"read after it until `phern profiles fold {name}`."
        )
    entries = [entry for entry in as_seq(named.rows) if isinstance(entry, dict)]
    emit(render_named_profile(named.extends, entries, comment="\n".join(notes)).rstrip())


@profiles_app.command()
def session(
    session_id: Annotated[str, typer.Argument(help="A stored session's id.")],
    at: Annotated[
        int | None,
        typer.Option("--at", help="The seq to read it at; the log's end when left out."),
    ] = None,
    full: Annotated[
        bool, typer.Option("--full", help="Every environment row as it ran, defaults included.")
    ] = False,
) -> None:
    """The environment a session ran in, at any point of its log (S3, S8).

    A fold of the session's own log (item 11): the base in force at `--at`, the
    overrides logged up to it and what they change, and each skill read by then
    with the hash of what was read. Read from disk through the log's lineage, so a
    session nothing is running is readable too, and a fork answers with the base its
    prefix holds. `--full` prints every environment row as it ran at that point,
    defaults included.
    """
    sessions_dir = resolve_roots().sessions_dir()
    try:
        _header, events = materialize(partial(read_stored, sessions_dir), session_id)
    except FileNotFoundError:
        fail(f"[red]no session {session_id!r} under {sessions_dir}[/red]", code=2)
    except (LineageError, ValueError) as error:
        fail(f"[red]{session_id}'s log does not read: {detail(error)}[/red]", code=2, cause=error)
    end = events[-1].seq if events else 0
    point = end if at is None else at
    upto = [event for event in events if event.seq <= point]
    env = fold_environment((event.type, event.data) for event in upto)
    if env.base is None:
        fail(
            f"[red]{session_id} has no recorded base by seq {point}: it had not started "
            "since pH began recording one[/red]",
            code=2,
        )
    if full:
        rows = _resolved_or_exit(env, resolved_environment)
        emit(yaml.safe_dump(thaw_json(rows), sort_keys=False, default_flow_style=False).rstrip())
        return
    lines = [f"{session_id} at seq {point} of {end}:", *environment_listing(env)]
    read = {as_str(event.data.get("name")): event for event in upto if event.type == SKILL_READ}
    if read:
        lines.append("Skills read by then, each as it was last read:")
        lines += [
            f"  {read_summary(event.data)} at seq {event.seq}"
            for _name, event in sorted(read.items())
        ]
    emit("\n".join(lines))


@profiles_app.command()
def diff(name: ProfileName, session: SessionChoice = "") -> None:
    """List what taking a named profile's current version would change (S6).

    For each stored session on `name` — or the one `--session` names — every
    setting that differs between the version it runs on and `name` as it composes
    now, with its old and new value and whose it is: the person's file, or pH's.
    Sessions with the same account are listed together.
    """
    now = _version_or_exit(name)
    emit("\n".join(_report(name, _changes(name, session, now))))


@profiles_app.command()
def adopt(
    name: ProfileName,
    session: SessionChoice = "",
    yes: Annotated[bool, typer.Option("--yes", help="Adopt without asking.")] = False,
) -> None:
    """Take a named profile's current version for its sessions' next start (S6).

    Shows what it changes (`diff`), asks, and records the version in each session
    that differs: its next start makes it the base, and the session's overrides
    still apply over it. A running session is not interrupted. A stored session is
    written under its lease; one a daemon holds, through that daemon.
    """
    now = _version_or_exit(name)
    changes = _changes(name, session, now)
    moving = [session_id for session_id, change in changes if change is not None]
    if not moving:
        emit(f"{name}: every session on it runs on its current version")
        return
    emit("\n".join(_report(name, changes)))
    asking = f"Adopt this version of {name} for {count_of(len(moving), 'session')}?"
    if not yes and not typer.confirm(asking):
        fail("[yellow]nothing adopted[/yellow]", code=1)
    lines, missed = anyio.run(partial(adopt_version, name, moving, now))
    emit("\n".join(lines))
    if missed:
        raise typer.Exit(code=1)


def _version_or_exit(name: str) -> ProfileBase:
    """`name` as a session's base would record it now, or exit 2 saying why not."""
    return _resolved_or_exit(profile_or_exit(name), base_of)


def _resolved_or_exit[S, T](composed: S, resolve: Callable[[S], T]) -> T:
    """Every row of `composed` through its model, or exit 2 naming what refused —
    `show --full`'s listing, `session --full`'s and `diff`'s version all resolve,
    and all refuse alike."""
    try:
        return resolve(composed)
    except ValidationError as error:
        fail(f"[red]{'; '.join(validation_errors(error, root='config'))}[/red]", code=2)
    except LoaderError as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)


def _changes(name: str, session: str, now: ProfileBase) -> list[tuple[str, ProfileChange | None]]:
    """Each session on `name`, or the one named, with how `now` differs from it."""
    sessions_dir = resolve_roots().sessions_dir()
    targets: list[tuple[str, LoggedEnvironment]]
    if session:
        env = recorded_environment(sessions_dir, session)
        if env.base is None:
            fail(f"[red]{session} has no recorded base under {sessions_dir}[/red]", code=2)
        if env.base.name != name:
            started = env.base.name or "a profile file"
            fail(f"[red]{session} started on {started}, not {name}[/red]", code=2)
        targets = [(session, env)]
    else:
        targets = stored_on(sessions_dir, name)
    return [(session_id, profile_change(env, now)) for session_id, env in targets]


def _report(name: str, changes: Sequence[tuple[str, ProfileChange | None]]) -> list[str]:
    """The accounts, one per distinct change, each under the sessions it is theirs."""
    if not changes:
        return [f"no stored session runs on {name}"]
    grouped: dict[tuple[str, ...], list[str]] = {}
    current: list[str] = []
    for session_id, change in changes:
        if change is None:
            current.append(session_id)
        else:
            grouped.setdefault(tuple(listing(change)), []).append(session_id)
    lines: list[str] = []
    for account, ids in grouped.items():
        lines.append(f"{', '.join(ids)}:")
        lines += [f"  {line}" for line in account]
    if current:
        lines.append(f"on the current version: {', '.join(current)}")
    return lines


async def adopt_version(
    name: str, session_ids: Sequence[str], version: ProfileBase
) -> tuple[list[str], bool]:
    """Record `version` in each session for its next start. `(lines, any missed)`.

    Each under its own lease, on a mount of the host's rows alone — the store and
    nothing that runs (`log_host`) — so nothing is started to write it, and the
    lease is given back before the next. One another process holds is asked of the
    daemon, which holds it if anything does.
    """
    host = log_host(name)
    lines: list[str] = []
    busy: list[str] = []
    missed = False
    for session_id in session_ids:
        try:
            async with mounted(host) as ctx:
                session = await stored_session(ctx, session_id)
                written = await record_adopted(ctx, session, version)
        except SessionBusy:
            busy.append(session_id)
            continue
        missed = missed or not written
        lines.append(
            f"{session_id}: adopted; its next start runs on it"
            if written
            else f"{session_id}: not adopted — its log could not be written"
        )
    if busy:
        through, refused = await _through_daemon(busy, version)
        lines += through
        missed = missed or refused
    return lines, missed


async def _through_daemon(
    session_ids: Sequence[str], version: ProfileBase
) -> tuple[list[str], bool]:
    """Ask the running daemon to record `version` in the sessions it holds."""

    def held_elsewhere(session_id: str, why: str) -> str:
        return f"{session_id}: not adopted — held by another process ({why})"

    async def work(client: DaemonClient) -> tuple[list[str], bool]:
        lines: list[str] = []
        refused = False
        for session_id in session_ids:
            try:
                await client.call(
                    verbs.SESSION_ADOPT,
                    AdoptParams(session_id=session_id, version=dict(version.to_wire())),
                )
            except DaemonError as error:
                lines.append(held_elsewhere(session_id, detail(error)))
                refused = True
                continue
            lines.append(f"{session_id}: adopted through the daemon; its next start runs on it")
        return lines, refused

    try:
        return await connected(resolve_roots().daemon_socket(), work)
    except (RuntimeDirError, DaemonGone, OSError) as error:
        return [
            held_elsewhere(one, f"no daemon to ask: {detail(error)}") for one in session_ids
        ], True


@profiles_app.command()
def fold(
    name: Annotated[
        str | None, typer.Argument(help="One profile; every unfolded one when left out.")
    ] = None,
) -> None:
    """Fold a profile's drop-ins, and a file in the old list format, into its named file.

    Composed before and after, and kept only when the two agree — so nothing a
    session runs with changes, and its next start has nothing to ask. The drop-in
    directory is moved aside, to `<name>.d.folded-<date>/`, which nothing reads.
    """
    names = [name] if name else unfolded_profiles()
    if not names:
        emit("nothing to fold")
        return
    for one in names:
        try:
            emit(fold_profile(one))
        except FoldRefused as error:
            fail(f"[red]{error}[/red]", code=1, cause=error)
        except (LoaderError, OSError, ValueError) as error:
            fail(f"[red]{one}: {detail(error)}[/red]", code=2, cause=error)


class FoldRefused(RuntimeError):
    """Folding would have changed what the profile composes, so nothing was changed."""


def fold_profile(name: str, *, today: date | None = None) -> str:
    """Fold `name`'s drop-ins and old-format file into one named profile, or refuse.

    Item 0's steps: compose as it composes now; write the file — appended as text
    when there is one, so the person's comments stay, and created sparse when there
    is not; move the drop-ins aside; compose again. A difference, or a file that no
    longer reads, puts all of it back and says which rows moved.
    """
    roots = resolve_roots()
    overlay = roots.profile_overlay(name)
    plan = profile_plan(name)
    named, dropins = plan.named, plan.dropins
    if not dropins and (named is None or not named.legacy):
        return f"{name}: nothing to fold"
    before = compose_profile(name)
    when = (today or date.today()).isoformat()
    folded = [
        entry
        for path in dropins
        for entry in as_seq(decode_document(path))
        if isinstance(entry, dict)
    ]
    # Each drop-in's own header comes with its rows: `/sandbox`'s says the host list
    # it pins will not follow pH's defaults, which stays true once folded.
    headers = [
        line.lstrip("#").strip()
        for path in dropins
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    ]
    folded_from = roots.profile_dropins(name).name
    stamp = f"Folded from {folded_from}/ by `phern profiles fold` on {when}."
    comment = "\n".join([stamp, *headers]) if dropins else ""
    original = overlay.read_text(encoding="utf-8") if named is not None else None
    if named is None or original is None:
        text = sparse_text(
            before,
            extends=name,
            comment=f"{name}, over the shipped profile of that name.\n{comment}",
        )
    else:
        try:
            text = append_rows(original, named, comment, folded)
        except LoaderError as error:
            raise FoldRefused(
                f"{name}: not folded — {detail(error)}; {overlay} is as it was"
            ) from error
    aside = _aside(roots.profile_dropins(name), when) if dropins else None
    write_atomic(overlay, text)
    if aside is not None:
        roots.profile_dropins(name).rename(aside)
    try:
        after = compose_profile(name)
        moved = [as_str(entry.get("id")) for entry in sparse_entries(before.rows, after.rows)]
        failure = ""
    except (LoaderError, OSError, ValueError) as error:
        moved, failure = [], detail(error)
    if failure or moved:
        if aside is not None:
            aside.rename(roots.profile_dropins(name))
        if original is None:
            overlay.unlink()
        else:
            write_atomic(overlay, original)
        why = failure or f"it would change {', '.join(moved)}"
        raise FoldRefused(
            f"{name}: not folded — {why}. {overlay} is as it was; fold it by hand, "
            "or remove what it cannot read."
        )
    said = f"{len(dropins)} drop-in(s)" if dropins else "the old list format"
    return f"{name}: folded {said} into {overlay}" + (
        f"; the drop-ins are in {aside}" if aside else ""
    )


def _aside(directory: Path, when: str) -> Path:
    """Where a folded drop-in directory goes: dated, and never over an earlier one."""
    candidate = directory.with_name(f"{directory.name}.folded-{when}")
    count = 2
    while candidate.exists():
        candidate = directory.with_name(f"{directory.name}.folded-{when}-{count}")
        count += 1
    return candidate
