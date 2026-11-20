"""`phern attachments gc`: the only thing that may delete conversation (P7-01).

The attachment store never collects on its own. That is a decision and not an
omission. A digest is addressed globally with no owner directory. Two sessions
that attach one photo therefore share a file, and a fork references exactly the
digests of its parent. That is what stops the deletion of a parent from breaking
its children. It also makes "is anyone still using this" a question that only a
fold over *every* stored session can answer. The spill sweep of F7 can run at
session open because a spilled result has an owner. This sweep cannot, and it is
a command for the same reason that `phern workspaces gc` is one.

**The fold obeys one rule, which the plan wrote down before any of this
existed: a blob that any stored log still references must not be collected,
however old it is.** Age is the obvious predicate, and it is the wrong one.
`ctx.uploads` is what made it wrong. After P7-03 the local blob is the last
copy, because the copy of the provider is behind a handle that expires.
Collecting a referenced attachment therefore does not degrade a session. It ends
the session. `media-degrade` will report "the stored bytes are gone" forever.
*Running the same conversation again against a different model* also stops being
possible, because a second route is a cache miss that uploads again from disk.

Age therefore appears here only in the direction that is safe. `--min-age`
**refuses** to collect something new, because a file that a user dropped on the
composer sits on disk with nothing referencing it until they send the prompt.

**There are two stores, and this command must not confuse them.**
`$PH_HOME/attachments` is content and is not safe to delete. `$PH_CACHE/uploads`
is a cache and is safe to delete wholesale. Both are swept here from one
reference set, which is what `ctx.uploads` said this command should do. Only the
first store needs the survey to be complete, and the report says so.

@module ph_app.attachments
"""

from __future__ import annotations

from functools import partial
from typing import Annotated

import anyio
import typer
from rich.filesize import decimal

from ph.cordis import Profile
from ph.keys import ATTACHMENTS, SESSION_PERSISTENCE, UPLOADS
from ph.seams.attachments import MIN_AGE, AttachmentSurvey, collect_attachments, survey_attachments

from .console import emit, fail_unmounted
from .profiles import DEFAULT_PROFILE, PatchOption, ProfileOption, profile_or_exit
from .runtime import mounted

__all__ = ["attachments_app"]

attachments_app = typer.Typer(
    help="Account for the media stored across every session.",
    add_completion=False,
)

DEFAULT_MIN_AGE_DAYS = MIN_AGE / 86_400.0
"""The default in days, from the constant of the seam itself and not from a
second number here.

A default that is written twice disagrees with itself after the first edit. This
default matters: it is the window in which a staged file has no log that
references it."""

LISTED = 20
"""How many collectable blobs are listed before the report stops naming them.

A digest and a size do not help a user decide much after the first screen. The
total, and whether the survey was complete, decide `--remove`. Without this
limit, a store with four thousand dead blobs would print four thousand lines of
hex."""


def _report(survey: AttachmentSurvey, *, removed: tuple[int, int] | None) -> str:
    """Return the whole answer as one block: what is there, then what happened to it.

    The block is built as a list in which the summary is always present. The
    alternative is an early return on the good counts, and that is how the
    refusal came to be hidden: a store with a torn log and nothing collectable
    answered "nothing to collect" and never mentioned the log.
    """
    listed = [
        # `rich.filesize.decimal`, which the session picker already renders
        # sizes with. A local formatter here did 1024-based math under `KB` and
        # `MB` labels, so the same file measured differently on each surface
        # that a user read.
        f"  {blob.digest[:19]}…  {decimal(blob.bytes):>9}  {blob.age / 86400:>5.1f}d"
        for blob in survey.collect[:LISTED]
    ]
    if len(survey.collect) > LISTED:
        listed.append(f"  … and {len(survey.collect) - LISTED} more")
    lines = [
        *listed,
        f"{chr(10) if listed else ''}{len(survey.kept)} referenced by "
        f"{survey.sessions} session(s), {len(survey.collect)} collectable "
        f"({decimal(survey.collectable_bytes)}), {len(survey.recent)} too new, "
        f"{len(survey.uploads)} stale upload handle(s)",
    ]
    if not survey.safe:
        # The refusal, and the *reason* for it. A user who reads only "nothing
        # collected" would reasonably try `--remove` again.
        reason = (
            f"{len(survey.unreadable)} session(s) would not read"
            if survey.unreadable
            else "the session listing was cut short"
        )
        lines.append(
            f"refusing to collect anything: {reason}, so a blob nothing appears to "
            "reference may still be needed"
        )
    elif removed is not None:
        blobs, entries = removed
        lines.append(f"collected {blobs} blob(s) and {entries} upload handle(s)")
    elif survey.collect or survey.uploads:
        lines.append("re-run with --remove to collect them")
    return "\n".join(lines)


async def _collect(profile: Profile, *, min_age: float, remove: bool) -> str:
    """Mount, fold every stored session, and either report or collect.

    The mount supplies both roots. Row config can point a store anywhere. A
    collector that resolved `$PH_HOME/attachments` itself would sweep the default
    directory of a deployment that does not use it.
    """
    async with mounted(profile) as ctx:
        store = ctx.get(ATTACHMENTS)
        persistence = ctx.get(SESSION_PERSISTENCE)
        if store is None or persistence is None:
            missing = "attachment store" if store is None else "session store"
            return f"this profile mounts no {missing}, so there is nothing to account for"
        uploads = ctx.get(UPLOADS)
        survey = survey_attachments(store, persistence, uploads=uploads, min_age=min_age)
        # **One summary sentence, always.** An early "nothing to collect" return
        # here read the counts and not `safe`. A store with a torn log and
        # nothing collectable therefore answered "nothing to collect" and never
        # mentioned the log. The refusal that `_report` exists to print was
        # hidden by the better-sounding of the two sentences.
        return _report(survey, removed=collect_attachments(survey, uploads) if remove else None)


@attachments_app.command()
def gc(
    profile: ProfileOption = DEFAULT_PROFILE,
    patch: PatchOption = [],  # noqa: B006 - typer reads the default as the option's
    min_age: Annotated[
        float,
        typer.Option("--min-age", help="Days a blob must have existed before it may be collected."),
    ] = DEFAULT_MIN_AGE_DAYS,
    remove: Annotated[
        bool, typer.Option("--remove", help="Actually collect; the default only reports.")
    ] = False,
) -> None:
    """Report — or with `--remove`, collect — media no stored session references.

    Reporting is the default, `phern workspaces gc`'s way round and for a sharper
    reason: what is on the other side here is not evidence somebody might want,
    it is content a session cannot be opened without. A person who runs this
    because the disk is full should be able to see the size of the answer before
    anything acts on it.

    The fold reads **every** stored session, not a page of recent ones. A bounded
    answer to "does anyone still need this" is not a smaller answer, it is a wrong
    one — the log below the cut is the old conversation nobody has opened lately,
    which is exactly the one whose pictures are worth keeping. If the store is too
    large to survey, or any log will not parse, this collects nothing and says so.
    """
    # `--patch`, like every other command that composes a profile. Both stores
    # that this command folds are row config. Only a command that takes a patch
    # can sweep a deployment that put the stores on another volume. Otherwise
    # the answer is "nothing to collect" against the default directory, which
    # is a confident wrong answer. `phern workspaces gc` takes `--patch` for the
    # same reason.
    composed = profile_or_exit(profile, patch)
    try:
        report = anyio.run(partial(_collect, composed, min_age=min_age * 86_400.0, remove=remove))
    except Exception as error:
        fail_unmounted(profile, error)
    emit(report)
