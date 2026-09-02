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
that may write to it in the next — and it is matched by **root path**, not by
inverting a directory name back into an agent id, because `sanitize_ref` is
lossy and an id that does not sanitize to itself would read as unheld and lose
its protection.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path, is_under
from ..seams.commands import CommandDefinition
from ..seams.workspace import stored_survivors
from ..seams.workspace_git import BRANCH_PREFIX as PREFIX
from ..seams.workspace_git import git
from ..wire import WireModel

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


class Config(WireModel):
    """Row config for the command."""

    root: str | None = None
    """Where checkouts live; must match `workspace-git-worktree`'s own setting.
    Both default to `$PH_HOME/worktrees`, so a deployment that moved one has to
    move the other."""


@plugin("workspace-commands", inject=["commands", "workspace", "subprocess", "fs"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register `/workspaces`."""
    root = default_home_path(config.root, "worktrees")

    async def workspaces(argument: str, invocation: Any) -> str:
        verb, _, rest = argument.strip().partition(" ")
        view = _Workspaces(ctx=ctx, root=root, base=ctx.fs.root)
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

    ctx.commands.register(
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

    A value rather than five functions threading `(ctx, root, base)`: all three
    are fixed for the whole of one `/workspaces` invocation.
    """

    ctx: Context
    root: Path
    base: Path

    async def kept(self, *, with_status: bool = True) -> list[KeptWorktree]:
        """Every artifact pH left in this repository: one row per `ph/*` branch.

        **Branches, not directories, because a branch is what disposal leaves.** A
        checkout is a resource an agent borrows and disposal takes back; the branch
        is what survives it, and it is what `export`, `merge` and `remove` already
        name. Enumerating checkouts listed only the two states that are *not* the
        ordinary one — a tree a live agent still holds, and a tree disposal failed
        to remove — and reported every successfully disposed agent as nothing at
        all.

        `BRANCH_PREFIX` rather than a walk of `refs/heads`: pH's own prefix is the
        whole of what keeps this from being a command that offers to delete a
        person's `feature/x`.

        The worktree join is what remains of the old enumeration, and it still earns
        its subprocess — it is how `held` and a stray checkout's `path` are known.
        Filtered by `root` for the same reason as before: a `ph/*` branch checked
        out somewhere else is not a tree this command made.

        `with_status=False` for the verbs that only need a name: `git status` is one
        subprocess per checkout and `merge`/`remove` never read `dirty`.
        """
        code, out, _ = await git(
            self.ctx, self.base, "branch", "--list", "--format=%(refname:short)", f"{PREFIX}*"
        )
        if code != 0:
            return []
        branches = [line.strip() for line in out.splitlines() if line.strip()]
        if not branches:
            return []
        code, out, _ = await git(self.ctx, self.base, "worktree", "list", "--porcelain")
        checkouts = {
            branch: path
            for path, branch in (_parse(out) if code == 0 else [])
            if branch and is_under(path, self.root)
        }
        live = {workspace.root for workspace in self.ctx.workspace.live()}
        dirty = dict.fromkeys(branches, False)
        pending = [
            branch
            for branch in branches
            if with_status and branch in checkouts and checkouts[branch] not in live
        ]
        if pending:
            # Concurrent, because this is one subprocess per stray checkout, and a
            # run that has stranded one has usually stranded several.
            async def measure(branch: str) -> None:
                dirty[branch] = await self._dirty(checkouts[branch])

            async with anyio.create_task_group() as group:
                for branch in pending:
                    group.start_soon(measure, branch)
        rows = []
        for branch in branches:
            agent_id, session_id = _identify(branch)
            path = checkouts.get(branch)
            rows.append(
                KeptWorktree(
                    branch=branch,
                    agent_id=agent_id,
                    session_id=session_id,
                    path=path,
                    dirty=dirty[branch],
                    held=path is not None and path in live,
                )
            )
        return rows

    async def _dirty(self, path: Path) -> bool:
        """`--untracked-files=normal`, because the answer is a boolean.

        `all` enumerates every file under a provisioned `node_modules` to say what
        one `?? node_modules/` line says. Measured only for a *stray* checkout —
        one disposal could not remove — where the question is whether removing the
        directory by hand loses anything the branch does not already have. A held
        tree is never measured at all, since `describe` reports it as held.
        """
        code, out, _ = await git(self.ctx, path, "status", "--porcelain")
        return code != 0 or bool(out.strip())

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
        store = self.ctx.get("session_persistence")
        if store is None:
            return "refusing: nothing is storing sessions, so there are no records to export"
        survivors, _ = stored_survivors(store)
        row = next((one for one in survivors if one.agent_id == name), None)
        if row is None:
            known = ", ".join(sorted({one.agent_id for one in survivors})) or "none"
            return f"refusing: no workspace named {name!r} (known: {known})"
        ref = await self.ctx.workspace.export(row)
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
        code, out, err = await git(self.ctx, self.base, "merge", "--no-edit", branch)
        if code != 0:
            detail = (err.strip() or out.strip()).splitlines()
            return f"could not merge {branch}: {detail[0] if detail else f'git exited {code}'}"
        return f"merged {branch}"

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
            code, _, _ = await git(
                self.ctx, self.base, "rev-parse", "--verify", f"refs/heads/{name}"
            )
            if code == 0:
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
            code, _, err = await git(
                self.ctx, self.base, "worktree", "remove", "--force", str(row.path)
            )
            if code != 0:
                return f"could not remove {row.path}: {err.strip() or f'git exited {code}'}"
            removed = f"removed {row.path}"
            if "--with-branch" not in flags:
                return removed

        force = "--force-branch" in flags
        code, _, err = await git(self.ctx, self.base, "branch", "-D" if force else "-d", row.branch)
        if code != 0:
            # `-d` refusing is the mechanism working, so it reads as a fact plus
            # the flag that overrides it — not as an error to decode.
            return (
                f"{removed}, but kept branch {row.branch}: {err.strip() or 'git refused'}\n"
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


def _parse(porcelain: str) -> list[tuple[Path, str]]:
    """`git worktree list --porcelain` into `(path, branch)` pairs.

    The main checkout has no `branch` line when detached, and a linked worktree
    always names one; a record with no path is not a record.
    """
    rows: list[tuple[Path, str]] = []
    path: Path | None = None
    branch = ""
    for line in porcelain.splitlines():
        if line.startswith("worktree "):
            if path is not None:
                rows.append((path, branch))
            path, branch = Path(line[len("worktree ") :].strip()), ""
        elif line.startswith("branch refs/heads/"):
            branch = line[len("branch refs/heads/") :].strip()
    if path is not None:
        rows.append((path, branch))
    return rows
