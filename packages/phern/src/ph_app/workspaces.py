"""`phern workspaces gc`: collect the evidence that nobody came back for (P6-28).

**This is the half that makes retention affordable.** The worktree of a settled
child is what a parent needs to diagnose a run that failed. The kind that a
failed child gets is the kind that discards even a dirty tree. P6-28 therefore
lets a tree be *retained* with a reason. The benefit is evidence. The cost is an
unbounded pile of checkouts, one for each child that ever ended badly, on a disk
that nobody is watching. This command bounds that cost, and the row would not
have been allowed to land the policy without it.

**Never automatic**, on the precedent of P7-01: `phern attachments gc` collects
only what no session references, and is deliberately unlike the open-time sweep
of F7. A harness that swept retained trees at startup would delete the evidence
of the failure of last night exactly when the user sat down to read it.

**Three refusals and a revocation.** A tree is refused if a live session holds
it, if its log was written inside the age bound, or if the directory is already
gone. A tree that passes all three is not deleted *by this command*. The
retention is revoked, and the tree goes back to the disposal policy that would
have run at release time if nobody had retained it. An ephemeral checkout is
therefore discarded, and an ordinary worktree keeps its dirty tree for review.
Nothing here can destroy more than an ordinary release would have destroyed.
That property lets the age bound be a default and not a decision.

@module ph_app.workspaces
"""

from __future__ import annotations

from collections import Counter
from functools import partial
from time import time
from typing import Annotated

import anyio
import typer

from ph.cordis import Profile
from ph.keys import SESSION_PERSISTENCE, WORKSPACE
from ph.seams.workspace import Collectable, stored_survivors

from .console import emit, fail_unmounted, nonblank
from .profiles import DEFAULT_PROFILE, PatchOption, ProfileOption, profile_or_exit
from .runtime import mounted

__all__ = ["workspaces_app"]

workspaces_app = typer.Typer(
    help="Account for the branches agents left behind, across every stored session.",
    add_completion=False,
)

DEFAULT_AGE_DAYS = 7.0
"""How long a retained tree is kept before it may be collected.

The unit is days, where the passivation of P5-05 is in minutes, and the units
are the argument. A user resumes a passivated root by typing at it. A person
reads a retained tree only after they notice that it exists, find it, and open
it. A week is one working cycle. It is long enough that the failure of Friday is
still there on Monday, and short enough that a month of failures is not.
"""


def _rows(collectables: list[Collectable]) -> list[str]:
    """Return one line for each tree, verdict first, in the order that the fold produced."""
    return [
        f"  {row.verdict:<8} {row.record.agent_id:<16} "
        f"{row.age / 86400:>5.1f}d  {row.record.root}  {row.record.reason}"
        for row in collectables
        # `gone` is a tree that somebody already removed by hand. It is counted
        # in the summary and never listed. It is the one verdict that grows
        # without bound in a log that keeps its `retained` forever. People stop
        # reading a report that is mostly directories that no longer exist.
        if row.verdict != "gone"
    ]


async def _collect(profile: Profile, *, older_than: float, remove: bool, family: str | None) -> str:
    """Mount, fold every stored session, and either report or collect.

    `family` narrows the fold to one session and the children that it spawned.
    `None` folds every stored session.

    The mount is required, and it is the cost that needs a reason. The *provider*
    is what knows how to end a tree. A collector that ran `git worktree remove`
    itself would be a second disposal policy. The docstring of `reclaim` says
    that such a policy must not exist, because it would be free to remove more
    than a release would.
    """
    async with mounted(profile) as ctx:
        store = ctx.get(SESSION_PERSISTENCE)
        seam = ctx.get(WORKSPACE)
        if store is None or seam is None:
            # Two rows, one sentence. Without a store there is nothing to fold,
            # and without the seam there is no provider to end a tree with. A
            # profile that lacks either one never made one of these trees.
            missing = "session store" if store is None else "workspace seam"
            return f"this profile mounts no {missing}, so there is nothing to account for"
        # The seam spells "every session" as an argument that is left out.
        survivors, touched = (
            stored_survivors(store) if family is None else stored_survivors(store, family=family)
        )
        rows = seam.collectable(survivors, older_than=older_than, now=time(), touched=touched)
        if not rows:
            whose = f" under {family}" if family is not None else ""
            return f"no retained trees{whose}, across the {len(touched)} most recent session(s)"
        lines = _rows(rows)
        # `Counter`, and not a dict comprehension over a hand-written list of
        # verdicts. That list was a second spelling of `CollectVerdict` with
        # nothing to keep the two in step, and it walked `rows` one time for
        # each verdict.
        counts = Counter(row.verdict for row in rows)
        if not remove:
            lines.append(
                f"\n{counts['collect']} collectable, {counts['held']} held, "
                f"{counts['recent']} within the age bound, {counts['gone']} already gone"
            )
            lines.append("re-run with --remove to collect them")
            return "\n".join(lines)
        removed = await seam.collect(rows)
        kept = counts["collect"] - len(removed)
        lines.append(
            f"\ncollected {counts['collect']} tree(s): {len(removed)} discarded, {kept} kept; "
            f"{counts['held']} held, {counts['recent']} within the age bound"
        )
        if kept:
            # The gap is usually the disposal policy doing its job and not a
            # failure. Work that a retained tree still held goes to its branch,
            # so the tree ends and the artifact does not. Both reasons are
            # offered because this code cannot tell them apart. `reclaim`
            # answers a bool, and a tier that declined to collect looks exactly
            # like a tier that saved the work first. Naming only the good reason
            # would be the overstatement that E1 forbids.
            lines.append(
                "a kept tree's work is on its branch, or the mounted tier could not end it — "
                "`/workspaces list` tells them apart"
            )
        return "\n".join(lines)


@workspaces_app.command()
def gc(
    profile: ProfileOption = DEFAULT_PROFILE,
    patch: PatchOption = [],  # noqa: B006 - typer reads the default as the option's
    older_than: Annotated[
        float,
        typer.Option("--older-than", help="Days a retained tree is kept before collecting."),
    ] = DEFAULT_AGE_DAYS,
    remove: Annotated[
        bool, typer.Option("--remove", help="Actually collect; the default only reports.")
    ] = False,
    session: Annotated[
        str | None,
        typer.Option(
            "--session", help="Only this session and the children it spawned.", callback=nonblank
        ),
    ] = None,
) -> None:
    """Report — or with `--remove`, collect — the trees retained as evidence.

    **Reporting is the default and removing is the flag**, which is the opposite
    way round from most `gc`. The reason is what is on the other side: every tree
    here was kept *because a run went wrong*, and the person who most needs this
    command is the one who just discovered the disk is full and does not yet know
    what these directories are. They should be able to find that out by typing
    the obvious thing.

    `--session` is the enumeration half of P6-28 as a person asks it: a child's
    workspace events are in the *child's* log, so "what did that run and its
    children leave" was reduced to opening each transcript by hand and knowing
    which fields to read. It narrows to that session and everything spawned
    beneath it — **descent, not the messaging family**, since a sibling's
    checkout is not this run's to collect.
    """
    composed = profile_or_exit(profile, patch)
    try:
        report = anyio.run(
            partial(
                _collect,
                composed,
                older_than=older_than * 86400.0,
                remove=remove,
                family=session,
            )
        )
    except Exception as error:
        # There is no `except typer.Exit: raise` above this, unlike `doctor`.
        # That guard is there because `typer.Exit` subclasses `RuntimeError`,
        # and it is needed only where the block can raise one. The profile is
        # resolved one line above this `try`, and `_collect` returns its
        # refusals as strings.
        fail_unmounted(profile, error)
    # `emit`, and not `console.print`. This is a table of paths and reasons. A
    # console that decides width from the terminal wraps the path of a checkout
    # onto a second line and ellipsizes the reason. The reason is the one column
    # that a user reads to decide whether to pass `--remove`.
    emit(report)
