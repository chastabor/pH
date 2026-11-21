"""`/workspaces`: the human side of the disposal policy (E15).

The `worktree` tier commits the work of an agent to `ph/<session>/<agent>` and
takes the checkout back. The branch is the artifact, and a directory is a
resource that the agent borrowed. That is the right default, and it *creates* an
accumulation problem: one branch for each agent that wrote something, and
nothing in the harness to see the branches or finish them. The README of `wtp`
opens with exactly this complaint about bare git ("remove worktree, forget to
delete the branch, orphaned branches accumulate") and answers it with
`remove --with-branch`. This command is that answer, limited to the branches
that pH made.

**Rows are branches.** For one round they were checkouts. This command could
then see exactly the two states that are *not* the ordinary state: a tree that a
live agent holds, and a tree that disposal failed to remove. It reported every
agent that finished cleanly as nothing. A checkout is now extra information on a
row, and not the thing that is listed.

**Three refusals, and each is the interesting part.**

A workspace that a *live agent holds* is refused. The command asks the seam and
not the filesystem, because a checkout that is clean at this instant belongs to
an agent that may write to it in the next. The workspace is matched by **ref**,
which is the one name that every tier puts on a `Workspace` and carries. It is
not matched by turning a directory name back into an agent id. `sanitize_ref` is
lossy, and an id that does not sanitize to itself would read as unheld and lose
its protection.

**Nothing here names `git` any more.** This command listed branches, joined them
against `git worktree list`, and merged and deleted with `git`, so every verb
was git-shaped in two ways. A `workspace-jj` stray had no path, no `dirty` and
no way to be removed, and a live one was not even protected. Worse, **jj embeds
its own git**, so a deployment that runs that tier need not have the binary.
There the branch listing returned nothing, and this command reported that
nothing was left behind. `ArtifactProvider` and `EnumeratingProvider` are those
verbs, asked of whichever tier is mounted.

A branch is deleted with `-d`, never `-D`, unless `--force-branch` says so. A
row with no checkout refuses a bare `remove` outright. Every disposed agent is
such a row, so to delete the branch anyway would make `--with-branch` decorative
and would discard the artifact that disposal saved.

Nothing outside `BRANCH_PREFIX` is listed or touched. Now that rows are
branches, that prefix is the whole of the guard. The own `feature/x` of a user
never becomes a row, so `remove` can never be aimed at it.

@module ph.commands.workspaces
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..cordis import Context, plugin
from ..keys import COMMANDS, FS, SESSION_PERSISTENCE, WORKSPACE
from ..seams.commands import (
    CommandContext,
    CommandDefinition,
    CommandRecord,
    CommandVerb,
    Verbs,
)
from ..seams.workspace import BRANCH_PREFIX as PREFIX
from ..seams.workspace import stored_survivors

__all__ = ["KeptWorktree", "apply"]


USAGE = (
    "usage: /workspaces [list | export <agent> | merge <agent> "
    "| remove <agent> [--with-branch] [--force-branch]]"
)
HINT = USAGE.removeprefix("usage: /workspaces ")
"""The hint in the palette is the usage line, so the two cannot drift. They had
already drifted: `--force-branch` was in one and not in the other."""


@dataclass(frozen=True, slots=True)
class KeptWorktree:
    """One artifact that pH left behind: a branch, and any checkout that is still
    on it."""

    branch: str
    agent_id: str
    session_id: str
    path: Path | None
    """The checkout, when one still exists. That is no longer the normal case.

    Disposal commits and removes. A directory here therefore means one of
    exactly two things: a live agent is working in it, or disposal could not
    remove it. Both are worth showing, and neither is the artifact."""
    dirty: bool
    held: bool
    """Whether a live agent still has it. A held workspace is a *current*
    workspace and not a leftover. It is listed as held and not hidden, because
    an operator who wonders where the disk space went should see all of them."""

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


@plugin("workspace-commands", affects="environment", inject=[COMMANDS, WORKSPACE, FS])
async def apply(ctx: Context, config: None) -> None:
    """Register `/workspaces`.

    There is no `root` setting any more. The setting existed to filter the
    worktree join, and it had to be kept equal to the `root` of
    `workspace-git-worktree`. That was one fact in two places. A deployment that
    moved one and not the other got a command that could see none of its
    checkouts. The tier answers now, and the tier knows where it puts them.
    """

    def verb(
        act: Callable[[_Workspaces, str], Awaitable[str]], *, record: CommandRecord = "before"
    ) -> CommandVerb:
        """One `/workspaces` verb: `act` on a fresh view, with a refusal returned
        as text and not raised."""

        async def run(rest: str, _invocation: CommandContext) -> str:
            try:
                return await act(_Workspaces(ctx=ctx, base=ctx.require(FS).root), rest)
            except _Refused as refusal:
                return str(refusal)

        return CommandVerb(run, record=record)

    listing = verb(lambda view, _rest: view.list(), record="after")
    workspaces = Verbs(
        {
            "": listing,
            "list": listing,
            "export": verb(_Workspaces.export),
            "merge": verb(_Workspaces.merge),
            "remove": verb(_Workspaces.remove),
        },
        otherwise=USAGE,
    )

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
    """A refusal that is raised where it must escape a caller.

    `find` raises it, and `export` raises it for a missing name. Every other
    refusal is returned. The verbs already return the sentence that the user
    reads, and an exception that is caught two lines below its raise is a longer
    way to write `return`.
    """


@dataclass(frozen=True, slots=True)
class _Workspaces:
    """The view that one dispatch has of the branches that pH left behind.

    This is a value and not five functions that each pass `(ctx, base)` along.
    Both are fixed for the whole of one `/workspaces` invocation.
    """

    ctx: Context
    base: Path
    _refs: list[list[str]] = field(default_factory=list)
    """The ref listing of one invocation, asked for one time.

    `merge <name>` asked for it twice: through `kept()`, and again in the
    fallback of `_branch_for`. That was two `git branch --list` or
    `jj bookmark list` spawns for one question. The value is fixed for the whole
    of one dispatch, which is the argument that this frozen value object already
    makes about `base`. A one-slot list is how a frozen dataclass holds a memo.
    """

    async def kept(self, *, with_status: bool = True) -> list[KeptWorktree]:
        """Every artifact that pH left in this repository: one row for each `ph/*`
        branch.

        **Branches, not directories, because a branch is what disposal leaves.**
        A checkout is a resource that an agent borrows and that disposal takes
        back. The branch is what survives, and it is what `export`, `merge` and
        `remove` already name. A listing of checkouts showed only the two states
        that are *not* the ordinary state: a tree that a live agent still holds,
        and a tree that disposal failed to remove. It reported every
        successfully disposed agent as nothing.

        **The checkout join belongs to the tier**, not to this command. It used
        to be `git worktree list --porcelain` here, and **a jj workspace is not
        a git worktree**. A stray jj workspace had no path in the listing, no
        `dirty`, and no verb that could remove it. A live one read as unheld,
        and `remove --with-branch --force-branch` deleted the branch of an agent
        that was still working in it. Because the seam is asked, this command
        also no longer carries a `root` setting that had to be kept equal to the
        setting of the tier.

        `held` also comes from the seam, matched on `ref`. The ref is the one
        name that every tier puts on a `Workspace` and *carries*. A directory
        name has been through a lossy `sanitize_ref`, and an id that does not
        sanitize to itself would read as unheld and lose its protection.

        **The ref listing belongs to the tier too**, and it was the half that
        failed *silently*. It was `git branch --list ph/*`, and jj embeds its
        own git. A jj deployment need not have the binary. There the listing
        returned nothing, and the command answered "no agent workspaces are left
        behind" while the bookmarks existed. The tier is asked, and it filters
        nothing.

        `BRANCH_PREFIX` is applied **here**. The prefix of pH is the whole of
        what keeps this from being a command that offers to delete the
        `feature/x` of a user. A guard belongs with the verb that it guards, and
        not with the tier, which would have to be trusted to apply it.

        `with_status=False` is for the verbs that need only a name. To measure a
        stray costs a subprocess for each checkout, and `merge` and `remove`
        never read `dirty`.
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
        """Every ref that the mounted tier knows, asked one time for each dispatch."""
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
        """Put the work of an agent on a branch, whichever tier it worked in.

        **The command asks the seam, not the tier.** A worktree agent has
        committed to its branch all along, and the answer is the name of that
        branch. The work of an overlay agent is in a delta until somebody
        assembles a commit from it. One verb covers both because
        `ExportingProvider` is what answers, so this command never learns which
        provider is mounted.

        The agent is found through the *records* and not through the ref
        listing that `list` and `merge` use. An overlay has no branch to list
        until it is exported, so the ref listing cannot see one. The records
        come from the stored session logs, which are the only enumeration in
        which both a worktree agent and an overlay agent appear.
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
        """Merge the branch of an agent into the checkout that the user is in.

        The merge is deliberately *not* `--no-ff`, squashed or rebased. Which of
        those a project wants is the policy of the project, and a management
        command that picked one would be making that policy. This is the plain
        merge that a person would type, offered where the user already is.
        """
        branch = await self._branch_for(name)
        # When the merge is not clean, the tier gives its own sentence, and that
        # is not always a failure. jj records conflicts in the commit and exits
        # zero, so "merged, with conflicts at these paths" is a true answer that
        # only the tier can give.
        trouble = await self.ctx.require(WORKSPACE).merge(self.base, branch)
        return trouble or f"merged {branch}"

    async def _branch_for(self, name: str) -> str:
        """The branch to merge: the branch of an agent, or a literal ref such as
        the one that `export` just made.

        The agent lookup comes first, because that is what a user names most of
        the time. If it finds nothing, a literal ref that the mounted tier lists
        is accepted. The fallback exists so that the closing sentence of
        `export` is true: a ref outside `BRANCH_PREFIX` is not in the prefixed
        enumeration, and without the fallback the name that the user was just
        given would be refused.
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
            # This is now the ordinary row. Disposal already took the checkout
            # back, so the branch is all that exists, and its removal cannot be
            # the default. The two flags keep the meanings that they had. This
            # case only stops the verb from meaning "remove nothing" for every
            # successfully disposed agent.
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
            # The refusal is the mechanism at work. So the reply is a fact plus
            # the flag that overrides it, and not an error to decode. Every tier
            # refuses a ref that holds work nothing else has, and only the
            # wording belongs to the tier.
            return (
                f"{removed}, but kept branch {row.branch}: {refused}\n"
                "pass --force-branch to delete it anyway"
            )
        return f"{removed} and branch {row.branch}"


def _identify(branch: str) -> tuple[str, str]:
    """`ph/<session>/<agent>` split into the two ids, as `(agent, session)`.

    `sanitize_ref` replaces `/` in both components at acquire, so a well-formed
    branch has exactly three parts. Anything else under the prefix is a branch
    that somebody else made. It is reported whole and not attributed to the
    wrong agent, because a wrong agent id here is a `remove` aimed at the wrong
    row.
    """
    parts = branch.split("/")
    if len(parts) != 3:
        return branch, ""
    return parts[2], parts[1]
