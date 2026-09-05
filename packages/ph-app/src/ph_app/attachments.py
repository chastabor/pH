"""`ph attachments gc` — the only thing allowed to delete conversation (P7-01).

The attachment store never collects on its own, and that is a decision rather than
an omission. A digest is addressed globally with no owner directory, so two
sessions attaching one photo share a file and a fork references exactly its
parent's digests — which is what stops deleting a parent from breaking its
children, and which makes "is anyone still using this" a question only a fold over
*every* stored session can answer. F7's spill sweep can run at session open
because a spilled result has an owner; this cannot, and it is a command for the
same reason `ph workspaces gc` is.

**The rule the fold obeys, written down in the plan before any of this existed: a
blob any stored log still references must not be collected, however old it is.**
Age is the obvious predicate and it is the wrong one. What makes it wrong is what
`ctx.uploads` changed: after P7-03 the local blob is the last copy — the
provider's is behind a handle that expires — so collecting a referenced
attachment does not degrade a session, it ends it. `media-degrade` will report
"the stored bytes are gone" for ever, and the thing that made *re-running the same
conversation against a different model* possible goes with it, since a second
route is a cache miss that re-uploads from disk.

So age appears here only in the direction that is safe: `--min-age` **refuses** to
collect something new, because a file a person dropped on the composer sits on
disk with nothing referencing it until they send the prompt.

**Two stores, and they must not be confused while doing this.**
`$PH_HOME/attachments` is content and is not safe to delete; `$PH_CACHE/uploads`
is a cache and is safe to delete wholesale. Both are swept here off one reference
set, which is what `ctx.uploads` said this command should do — but only the first
needs the survey to be complete, and the report says so.

@module ph_app.attachments
"""

from __future__ import annotations

from functools import partial
from typing import Annotated

import anyio
import typer
from rich.filesize import decimal

from ph.cordis import Profile
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
"""Days, from the seam's own constant rather than a second number here.

A default written twice is a default that disagrees with itself after the first
edit, and this one is load-bearing: it is the window in which a staged file has
no log referencing it."""

LISTED = 20
"""How many collectable blobs are listed before the report stops naming them.

A digest and a size do not help a person decide much past the first screenful —
what decides `--remove` is the total and whether the survey was complete — and a
store with four thousand dead blobs would otherwise print four thousand lines of
hex."""


def _report(survey: AttachmentSurvey, *, removed: tuple[int, int] | None) -> str:
    """The whole answer as one block: what is there, then what happened to it.

    Built as a list with the summary always present, because the alternative —
    returning early on the happy counts — is how the refusal came to be shadowed:
    a store with a torn log and nothing collectable answered "nothing to collect"
    and never mentioned the log.
    """
    listed = [
        # `rich.filesize.decimal`, which the session picker already renders sizes
        # with: a local formatter here did 1024-based maths under `KB`/`MB`
        # labels, so the same file measured differently depending on which
        # surface a person happened to be reading.
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
        # The refusal, and *why*: a person who reads only "nothing collected"
        # would reasonably try `--remove` again.
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

    Mounting is what supplies both roots — a store can be pointed anywhere by row
    config, and a collector that resolved `$PH_HOME/attachments` itself would
    sweep the default directory of a deployment that does not use it.
    """
    async with mounted(profile) as ctx:
        store = ctx.get("attachments")
        persistence = ctx.get("session_persistence")
        if store is None or persistence is None:
            missing = "attachment store" if store is None else "session store"
            return f"this profile mounts no {missing}, so there is nothing to account for"
        uploads = ctx.get("uploads")
        survey = survey_attachments(store, persistence, uploads=uploads, min_age=min_age)
        # **One summary sentence, always.** An early "nothing to collect" return
        # here read the counts and not `safe`, so a store with a torn log and
        # nothing collectable answered "nothing to collect" and never mentioned
        # the log — the refusal `_report` exists to print, shadowed by the
        # happier of the two sentences.
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

    Reporting is the default, `ph workspaces gc`'s way round and for a sharper
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
    # `--patch` like every other command that composes a profile: both stores
    # this folds are row config, so a deployment that put them on another volume
    # can only be swept by a command that can say so — otherwise the answer is
    # "nothing to collect" against the default directory, which is the confident
    # wrong answer. `ph workspaces gc` takes it for the same reason.
    composed = profile_or_exit(profile, patch)
    try:
        report = anyio.run(partial(_collect, composed, min_age=min_age * 86_400.0, remove=remove))
    except Exception as error:
        fail_unmounted(profile, error)
    emit(report)
