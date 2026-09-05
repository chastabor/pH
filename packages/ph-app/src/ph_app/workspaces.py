"""`ph workspaces gc` — collect the evidence nobody came back for (P6-28).

**The half that makes retention affordable.** A settled child's worktree is what
a parent needs to diagnose a run that failed, and the kind a failed child gets is
the kind that discards even a dirty tree — so P6-28 lets a tree be *retained*
with a reason. What that buys is evidence; what it sells is an unbounded pile of
checkouts, one per child that ever ended badly, on a disk nobody is watching.
This is the command that closes that trade, and the row would not have been
allowed to land the policy without it.

**Never automatic**, on P7-01's precedent: `ph attachments gc` collects only what
no session references and deliberately unlike F7's open-time sweep. A harness
that swept retained trees at startup would be deleting the evidence of last
night's failure exactly as the person sat down to read it.

**Three refusals and a revocation.** A tree is refused if a live session holds
it, if its log was written inside the age bound, or if the directory is already
gone. What survives all three is not deleted *by this command* — the retention is
revoked and the tree handed back to the disposal policy that would have run at
release time had nobody retained it. So an ephemeral checkout is discarded, an
ordinary worktree keeps its dirty tree for review, and nothing here can destroy
more than an ordinary release would have. That is the property that lets the age
bound be a default rather than a decision.

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
from ph.seams.workspace import Collectable, stored_survivors

from .console import emit, fail_unmounted
from .profiles import DEFAULT_PROFILE, PatchOption, ProfileOption, profile_or_exit
from .runtime import mounted

__all__ = ["workspaces_app"]

workspaces_app = typer.Typer(
    help="Account for the branches agents left behind, across every stored session.",
    add_completion=False,
)

DEFAULT_AGE_DAYS = 7.0
"""How long a retained tree is kept before it may be collected.

Days, where P5-05's passivation is minutes, and the units are the argument: a
passivated root is resumed by typing at it, while a retained tree is read by a
person who has to notice it exists, find it, and open it. A week is one working
cycle — long enough that Friday's failure is still there on Monday, short enough
that a month of them is not.
"""


def _rows(collectables: list[Collectable]) -> list[str]:
    """One line per tree, verdict first, in the order the fold produced them."""
    return [
        f"  {row.verdict:<8} {row.record.agent_id:<16} "
        f"{row.age / 86400:>5.1f}d  {row.record.root}  {row.record.reason}"
        for row in collectables
        # `gone` is a tree somebody already removed by hand. Counted in the
        # summary, never listed: it is the one verdict that grows without bound
        # in a log that keeps its `retained` forever, and a report whose bulk is
        # directories that no longer exist is one people stop reading.
        if row.verdict != "gone"
    ]


async def _collect(profile: Profile, *, older_than: float, remove: bool, family: str) -> str:
    """Mount, fold every stored session, and either report or collect.

    Mounting is not optional and is the interesting cost: the *provider* is what
    knows how to end a tree, and a collector that shelled out to `git worktree
    remove` itself would be a second disposal policy — the exact thing
    `reclaim`'s docstring says must not exist, since it would be free to remove
    more than a release would.
    """
    async with mounted(profile) as ctx:
        store = ctx.get("session_persistence")
        seam = ctx.get("workspace")
        if store is None or seam is None:
            # Two rows, one sentence: without a store there is nothing to fold,
            # and without the seam there is no provider to end a tree with. A
            # profile missing either is a profile that never made one of these.
            missing = "session store" if store is None else "workspace seam"
            return f"this profile mounts no {missing}, so there is nothing to account for"
        survivors, touched = stored_survivors(store, family=family)
        rows = seam.collectable(survivors, older_than=older_than, now=time(), touched=touched)
        if not rows:
            whose = f" under {family}" if family else ""
            return f"no retained trees{whose}, across the {len(touched)} most recent session(s)"
        lines = _rows(rows)
        # `Counter`, not a dict comprehension over a hand-written verdict list:
        # that list was a second spelling of `CollectVerdict` with nothing
        # keeping the two in step, and it walked `rows` once per verdict.
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
            # The gap is usually the disposal policy doing its job rather than a
            # failure — work a retained tree still held goes to its branch, so the
            # tree ends and the artifact does not. Both reasons are offered because
            # this cannot tell them apart: `reclaim` answers a bool, and a tier that
            # declined to collect at all looks exactly like one that saved the work
            # first. Naming only the happy one would be the overstatement E1 forbids.
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
        str,
        typer.Option("--session", help="Only this session and the children it spawned."),
    ] = "",
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
        # No `except typer.Exit: raise` above it, unlike `doctor`: that guard is
        # there because `typer.Exit` subclasses `RuntimeError`, and it only earns
        # its place where the block can raise one. Resolving the profile happens
        # a line above this `try`, and `_collect` returns its refusals as strings.
        fail_unmounted(profile, error)
    # `emit`, not `console.print`: this is a table of paths and reasons, and a
    # console that decides width from the terminal wraps a checkout's path onto
    # a second line and ellipsizes the reason — which is the one column a person
    # deciding whether to pass `--remove` is reading.
    emit(report)
