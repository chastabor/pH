"""`/workspaces` — the human half of the disposal policy (E15).

The `worktree` tier commits an agent's work to `ph/<session>/<agent>` and takes
the checkout back: the branch is the artifact, a directory is a resource the
agent borrowed. That is the right default and it *creates* an accumulation
problem — one branch per agent that wrote something, with nothing in the harness
to see or finish them. `wtp`'s README opens with exactly this complaint about
bare git ("remove worktree, forget to delete the branch, orphaned branches
accumulate") and answers it with `remove --with-branch`; this is that answer,
scoped to the branches pH made.

**Rows are branches.** They were checkouts for one round, which meant this
command could see exactly the two states that are *not* the ordinary one — a
tree a live agent holds, and a tree disposal failed to remove — and reported
every agent that finished cleanly as nothing at all. A checkout is now extra
information attached to a row rather than the thing being listed.

**Three refusals, and each is the interesting part.**

A workspace a *live agent holds* is refused. The seam is asked, not the
filesystem, because a checkout that is clean this instant belongs to an agent
that may write to it in the next — and it is matched by **ref**, the one name
every tier puts on a `Workspace` and carries, rather than by inverting a
directory name back into an agent id, because `sanitize_ref` is lossy and an id
that does not sanitize to itself would read as unheld and lose its protection.

**Nothing here spells `git` any more.** This command listed branches, joined them
against `git worktree list`, merged and deleted with `git`, which made every verb
git-shaped twice over. A `workspace-jj` stray had no path, no `dirty` and no way to
be removed, and a live one was not even protected — and worse, **jj embeds its own
git**, so a deployment running that tier need not have the binary at all, and there
the branch listing returned nothing while this command reported that there was
nothing left behind. `ArtifactProvider` is every one of those verbs, asked of
whichever tier is mounted.

A branch is deleted with `-d`, never `-D`, unless `--force-branch` says so, and
a row with no checkout refuses a bare `remove` outright. Every disposed agent is
such a row, so deleting the branch anyway would make `--with-branch` decorative
and throw away the artifact disposal went to the trouble of saving.

Nothing outside `BRANCH_PREFIX` is listed or touched. That prefix is the whole
of the guard now that rows are branches — a person's own `feature/x` never
becomes a row, so `remove` can never be aimed at it.

@module ph.commands.workspaces
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cordis import Context, plugin
from ..keys import COMMANDS, FS, SESSION_PERSISTENCE, WORKSPACE
from ..seams.commands import CommandDefinition
from ..seams.workspace import BRANCH_PREFIX as PREFIX
from ..seams.workspace import stored_survivors

__all__ = ["KeptWorktree", "apply"]

log = logging.getLogger("ph.commands.workspaces")

USAGE = (
    "usage: /workspaces [list | export <agent> | merge <agent> "
    "| remove <agent> [--with-branch] [--force-branch]]"
)
HINT = USAGE.removeprefix("usage: /workspaces ")
"""The palette's hint is the usage line, so the two cannot drift — they already
had, with `--force-branch` in one and not the other."""


@dataclass(frozen=True, slots=True)
class KeptWorktree:
    """One artifact pH left behind: a branch, and whatever checkout is still on it."""

    branch: str
    agent_id: str
    session_id: str
    path: Path | None
    """The checkout, when one still exists — which is no longer the normal case.

    Disposal commits and removes, so a directory here means one of exactly two
    things: a live agent is working in it, or disposal could not remove it. Both
    are worth seeing; neither is the artifact."""
    dirty: bool
    held: bool
    """Whether a live agent still has it. A held workspace is a *current* one, not
    a leftover, and is listed as such rather than hidden — an operator wondering
    where the disk went should see all of them."""

    def describe(self) -> str:
        state = (
            "held"
            if self.held
            else "branch"
            if self.path is None
            else ("stray-dirty" if self.dirty else "stray")
        )
        where = str(self.path) if self.path is not None else "-"
        return f"{self.agent_id:<16} {state:<11} {self.session_id:<14} {self.branch:<24} {where}"


@plugin("workspace-commands", inject=[COMMANDS, WORKSPACE, FS])
async def apply(ctx: Context, config: None) -> None:
    """Register `/workspaces`.

    No `root` setting any more. It existed to filter the worktree join and had to be
    kept equal to `workspace-git-worktree`'s own — one fact in two places, where a
    deployment that moved one and not the other got a command that could see none of
    its checkouts. The tier answers now, and it knows where it puts them.
    """

    async def workspaces(argument: str, invocation: Any) -> str:  # noqa: ANN401
        verb, _, rest = argument.strip().partition(" ")
        view = _Workspaces(ctx=ctx, base=ctx.require(FS).root)
        try:
            if verb in ("", "list"):
                return await view.list()
            if verb == "export":
                return await view.export(rest.strip())
            if verb == "merge":
                return await view.merge(rest.strip())
            if verb == "remove":
                return await view.remove(rest)
        except _Refused as refusal:
            return str(refusal)
        return USAGE

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="workspaces",
            summary="List, export, merge or remove the branches agents left behind.",
            argument_hint=HINT,
            run=workspaces,
        ),
        scope=ctx,
    )


class _Refused(Exception):
    """A refusal raised where it has to escape a caller.

    Only `_find` raises it — every other refusal is `return`ed, because the
    verbs already return the sentence a person reads and an exception caught two
    lines below its raise is a `return` spelled longer.
    """


@dataclass(frozen=True, slots=True)
class _Workspaces:
    """One dispatch's view of the branches pH left behind.

    A value rather than five functions threading `(ctx, base)`: both are fixed for
    the whole of one `/workspaces` invocation.
    """

    ctx: Context
    base: Path
    _refs: list[list[str]] = field(default_factory=list)
    """One invocation's ref listing, asked once.

    `merge <name>` asked for it twice — once through `kept()` and again in
    `_branch_for`'s fallback — which is two `git branch --list` / `jj bookmark list`
    spawns for one question. The value is fixed for the whole of one dispatch, which
    is the argument this frozen value object already makes about `base`; a one-slot
    list is how a frozen dataclass holds a memo.
    """

    async def kept(self, *, with_status: bool = True) -> list[KeptWorktree]:
        """Every artifact pH left in this repository: one row per `ph/*` branch.

        **Branches, not directories, because a branch is what disposal leaves.** A
        checkout is a resource an agent borrows and disposal takes back; the branch
        is what survives it, and it is what `export`, `merge` and `remove` already
        name. Enumerating checkouts listed only the two states that are *not* the
        ordinary one — a tree a live agent still holds, and a tree disposal failed
        to remove — and reported every successfully disposed agent as nothing at
        all.

        **The checkout join is the tier's**, not this command's. It used to be `git
        worktree list --porcelain` here, and **a jj workspace is not a git
        worktree** — so a stray one had no path in the listing, no `dirty`, and no
        verb that could remove it, while a live one read as unheld and
        `remove --with-branch --force-branch` deleted the branch of an agent still
        working in it. Asking the seam is also what let this command stop carrying a
        `root` setting that had to be kept equal to the tier's own.

        `held` is the seam's too, matched on `ref`: it is the one name every tier
        puts on a `Workspace` and *carries*, where a directory name has been through
        a lossy `sanitize_ref` and an id that does not sanitize to itself would read
        as unheld and lose its protection.

        **The ref listing is the tier's too**, and it was the half that failed
        *silently*. It was `git branch --list ph/*`, and jj embeds its own git — so a
        jj deployment need not have the binary, and there this returned nothing and
        the command answered "no agent workspaces are left behind" with the bookmarks
        sitting right there. The tier is asked, and it filters nothing.

        `BRANCH_PREFIX` is applied **here**, because pH's own prefix is the whole of
        what keeps this from being a command that offers to delete a person's
        `feature/x` — a guard belongs with the verb it guards, not with the tier that
        would have to be trusted to apply it.

        `with_status=False` for the verbs that only need a name: measuring a stray
        costs a subprocess per checkout and `merge`/`remove` never read `dirty`.
        """
        branches = [ref for ref in await self._all_refs() if ref.startswith(PREFIX)]
        if not branches:
            return []
        checkouts = await self.ctx.require(WORKSPACE).strays(self.base, with_status=with_status)
        live = {workspace.ref for workspace in self.ctx.require(WORKSPACE).live() if workspace.ref}
        rows = []
        for branch in branches:
            agent_id, session_id = _identify(branch)
            stray = checkouts.get(branch)
            rows.append(
                KeptWorktree(
                    branch=branch,
                    agent_id=agent_id,
                    session_id=session_id,
                    path=None if stray is None else stray.path,
                    dirty=stray is not None and stray.dirty,
                    held=branch in live,
                )
            )
        return rows

    async def _all_refs(self) -> list[str]:
        """Every ref the mounted tier knows, once per dispatch."""
        if not self._refs:
            self._refs.append(await self.ctx.require(WORKSPACE).refs(self.base))
        return self._refs[0]

    async def find(self, name: str, *, with_status: bool = False) -> KeptWorktree:
        if not name:
            raise _Refused(USAGE)
        kept = await self.kept(with_status=with_status)
        for row in kept:
            if name in (row.agent_id, row.branch):
                return row
        known = ", ".join(row.agent_id for row in kept) or "none"
        raise _Refused(f"refusing: no workspace named {name!r} (known: {known})")

    async def list(self) -> str:
        kept = await self.kept()
        if not kept:
            return "no agent workspaces are left behind"
        return "\n".join(row.describe() for row in kept)

    async def export(self, name: str) -> str:
        """Put an agent's work on a branch, whichever tier it worked in.

        **Asks the seam, not the tier.** A worktree agent has been committing to
        its branch all along and the answer is that branch's name; an overlay
        agent's work is in a delta until somebody assembles a commit from it. One
        verb covers both because `ExportingProvider` is what answers, so this
        command never learns which provider is mounted.

        Found through the *records* rather than through `git worktree list`,
        which is what `list` and `merge` use: an overlay has no worktree to
        enumerate, so the git-shaped lookup cannot see one. The records are the
        log's own, which is the only enumeration both tiers appear in.
        """
        if not name:
            raise _Refused(USAGE)
        store = self.ctx.get(SESSION_PERSISTENCE)
        if store is None:
            return "refusing: nothing is storing sessions, so there are no records to export"
        survivors, _ = stored_survivors(store)
        row = next((one for one in survivors if one.agent_id == name), None)
        if row is None:
            known = ", ".join(sorted({one.agent_id for one in survivors})) or "none"
            return f"refusing: no workspace named {name!r} (known: {known})"
        ref = await self.ctx.require(WORKSPACE).export(row)
        if ref is None:
            return f"refusing: the mounted tier cannot export {name}"
        return f"exported {name} to {ref} — merge it with: /workspaces merge {ref}"

    async def merge(self, name: str) -> str:
        """Merge an agent's branch into the checkout the person is standing in.

        Deliberately *not* `--no-ff` or squashed or rebased: which of those a
        project wants is a project's policy, and a management command that
        picked one would be making it. This is the plain merge a person would
        type, offered where they already are.
        """
        branch = await self._branch_for(name)
        # The tier's own sentence when it is not a clean merge, and that is not always
        # a failure: jj records conflicts in the commit and exits zero, so "merged,
        # with conflicts at these paths" is a true answer only it can give.
        trouble = await self.ctx.require(WORKSPACE).merge(self.base, branch)
        return trouble or f"merged {branch}"

    async def _branch_for(self, name: str) -> str:
        """The branch to merge: an agent's, or a ref `export` just made.

        The agent lookup first, because that is what a person names most of the
        time. Falling through to a literal ref is what makes `export`'s own closing
        sentence true: an overlay's branch is not under `BRANCH_PREFIX`, so the
        prefixed enumeration cannot find it and the name a person was just handed
        would refuse.
        """
        try:
            return (await self.find(name)).branch
        except _Refused:
            if name in await self._all_refs():
                return name
            raise

    async def remove(self, argument: str) -> str:
        parts = argument.split()
        flags = {part for part in parts if part.startswith("--")}
        names = [part for part in parts if not part.startswith("--")]
        unknown = flags - {"--with-branch", "--force-branch"}
        if unknown:
            return f"refusing: unknown flag(s) {', '.join(sorted(unknown))}\n{USAGE}"
        if "--force-branch" in flags and "--with-branch" not in flags:
            return "refusing: --force-branch only means something with --with-branch"

        row = await self.find(names[0] if names else "")
        if row.held:
            return (
                f"refusing: {row.agent_id} still holds this workspace; "
                "it is released when that agent is disposed"
            )

        if row.path is None:
            # The ordinary row now: disposal already took the checkout back, so the
            # branch is all there is and removing it cannot be the default. The two
            # flags keep the meanings they had — this only stops the verb reading
            # "remove nothing" for every successfully disposed agent.
            if "--with-branch" not in flags:
                return (
                    f"refusing: {row.agent_id} has no checkout to remove — its work is on "
                    f"{row.branch}\npass --with-branch to delete the branch too"
                )
            removed = f"{row.agent_id} had no checkout"
        else:
            refused = await self.ctx.require(WORKSPACE).discard(row.path)
            if refused:
                return f"could not remove {row.path}: {refused}"
            removed = f"removed {row.path}"
            if "--with-branch" not in flags:
                return removed

        force = "--force-branch" in flags
        refused = await self.ctx.require(WORKSPACE).delete_ref(self.base, row.branch, force=force)
        if refused:
            # The refusal is the mechanism working, so it reads as a fact plus the
            # flag that overrides it — not as an error to decode. Both tiers refuse a
            # ref holding work nothing else has; only the wording is theirs.
            return (
                f"{removed}, but kept branch {row.branch}: {refused}\n"
                "pass --force-branch to delete it anyway"
            )
        return f"{removed} and branch {row.branch}"


def _identify(branch: str) -> tuple[str, str]:
    """`ph/<session>/<agent>` into the two ids, as `(agent, session)`.

    `sanitize_ref` collapses `/` out of both components at acquire, so a
    well-formed branch has exactly three parts. Anything else under the prefix is
    somebody's own branch and is reported whole rather than mis-attributed — a
    wrong agent id here is a `remove` aimed at the wrong row.
    """
    parts = branch.split("/")
    if len(parts) != 3:
        return branch, ""
    return parts[2], parts[1]
