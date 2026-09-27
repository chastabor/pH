"""`phern profiles` — what a named profile sets, and folding the old layers into it (S2).

`show` answers "what does this profile say" two ways (decision 16): what the
person's file sets — sparse, the rows that differ — and, with `--full`, every
setting a session on it runs with, defaults included. The second is for debugging
and for a first session, deciding what to change.

`fold` is item 0 of `plans/Session_Profiles_Plan.md`, run on purpose: a profile's
drop-ins, and a file still in the list format before S2, become the one named
profile file. It composes the profile before and after and keeps the change only
when the two agree, so a session that starts on it next notices nothing.

@module ph_app.profiles_cli
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from ph.cordis import LoaderError, sparse_entries
from ph.documents import decode_document
from ph.json import as_seq, as_str, thaw_json
from ph.paths import resolve_roots, write_atomic
from ph.persistence import read_session
from ph.persistence.jsonl import locate_session
from ph.session_profile import BASE
from ph.wire import validation_errors

from .console import detail, emit, fail
from .named_profiles import append_rows, render_named_profile
from .profiles import (
    compose_profile,
    profile_or_exit,
    profile_plan,
    sparse_text,
    unfolded_profiles,
)

__all__ = ["FoldRefused", "fold_profile", "profiles_app"]

profiles_app = typer.Typer(
    help="Named profiles: what they set, and folding the older layers into them.",
    no_args_is_help=True,
)

ProfileName = Annotated[str, typer.Argument(help="A profile name, or a path to a .yaml.")]


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
        composed = profile_or_exit(name)
        try:
            rows = composed.resolved({"environment"})
        except ValidationError as error:
            fail(f"[red]{'; '.join(validation_errors(error, root='config'))}[/red]", code=2)
        except LoaderError as error:
            fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
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
) -> None:
    """Print the profile a session started in — its `profile/base`, from its log (S3).

    Read from the file on disk, so a session nothing is running is readable too.
    The rows are the full environment it was mounted with, defaults included; the
    sources are the person's own layers it was composed from, as they were then.
    """
    path = locate_session(resolve_roots().sessions_dir(), session_id)
    if path is None:
        fail(f"[red]no session {session_id!r} under {resolve_roots().sessions_dir()}[/red]", code=2)
    _header, events = read_session(path)
    recorded = [event for event in events if event.type == BASE]
    if not recorded:
        fail(
            f"[red]{session_id} has no recorded base: it has not started since pH began "
            "recording one[/red]",
            code=2,
        )
    shown = yaml.safe_dump(thaw_json(recorded[-1].data), sort_keys=False, default_flow_style=False)
    emit(shown.rstrip())


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
