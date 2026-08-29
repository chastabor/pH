"""`/workspaces` — the human half of the keep-dirty policy (E15).

The `worktree` tier keeps a dirty worktree on disposal, on purpose: an agent
that did work leaves it for a person to inspect and merge. That is the right
default and it *creates* an accumulation problem — trees under
`$PH_HOME/worktrees/<session>/<agent>` on branches `ph/<session>/<agent>`, one
per agent that wrote something, with nothing in the harness to see or finish
them. `wtp`'s README opens with exactly this complaint about bare git
("remove worktree, forget to delete the branch, orphaned branches accumulate")
and answers it with `remove --with-branch`; this is that answer, scoped to the
trees pH made.

**Three refusals, and each is the interesting part.**

A tree a *live agent holds* is refused. The seam is asked, not the filesystem,
because a worktree that is clean this instant belongs to an agent that may write
to it in the next — and it is matched by **root path**, not by inverting the
directory name back into an agent id, because `sanitize_ref` is lossy and an id
that does not sanitize to itself would read as unheld and lose its protection.

A branch is deleted with `-d`, never `-D`, unless `--force-branch` says so. A
clean worktree is *not* evidence that its branch was merged: an agent that
committed its work leaves nothing in `git status`, and force-deleting there
discards exactly what the keep-dirty rule exists to protect. Git refuses, and
the refusal names the flag that would override it.

Nothing outside pH's own worktree root is listed or touched. The command reads
`git worktree list`, which reports the user's own worktrees too, and a
management command that offered to delete those would be a different and much
worse tool.

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
from ..seams.workspace_git import git
from ..wire import WireModel

__all__ = ["KeptWorktree", "apply"]

log = logging.getLogger("ph.commands.workspaces")

USAGE = (
    "usage: /workspaces [list | merge <agent> | remove <agent> [--with-branch] [--force-branch]]"
)
HINT = USAGE.removeprefix("usage: /workspaces ")
"""The palette's hint is the usage line, so the two cannot drift — they already
had, with `--force-branch` in one and not the other."""


@dataclass(frozen=True, slots=True)
class KeptWorktree:
    """One worktree pH made and left behind."""

    path: Path
    branch: str
    agent_id: str
    session_id: str
    dirty: bool
    held: bool
    """Whether a live agent still has it. A held tree is a *current* workspace,
    not a leftover, and is listed as such rather than hidden — an operator
    wondering where the disk went should see all of them."""

    def describe(self) -> str:
        state = "held" if self.held else ("dirty" if self.dirty else "clean")
        return f"{self.agent_id:<16} {state:<6} {self.session_id:<14} {self.branch:<24} {self.path}"


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
            summary="List, merge or remove the worktrees agents left behind.",
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
    """One dispatch's view of the worktrees pH left behind.

    A value rather than five functions threading `(ctx, root, base)`: all three
    are fixed for the whole of one `/workspaces` invocation.
    """

    ctx: Context
    root: Path
    base: Path

    async def kept(self, *, with_status: bool = True) -> list[KeptWorktree]:
        """Every pH-made worktree the repository still knows about.

        Read from `git worktree list --porcelain` rather than by walking `root`,
        because git's registration is what makes a directory a worktree — a
        stale directory git has pruned is not one, and a tree someone moved
        still is.

        `with_status=False` for the verbs that only need a name: `git status` is
        one subprocess per tree and `merge`/`remove` never read `dirty`.
        """
        code, out, _ = await git(self.ctx, self.base, "worktree", "list", "--porcelain")
        if code != 0:
            return []
        live = {workspace.root: workspace for workspace in self.ctx.workspace.live()}
        rows = [(path, branch) for path, branch in _parse(out) if is_under(path, self.root)]
        dirty = dict.fromkeys((path for path, _ in rows), False)
        pending = [path for path, _ in rows if with_status and path not in live]
        if pending:
            # Concurrent, because this is one subprocess per leftover and the
            # thing the keep-dirty policy is designed to accumulate is leftovers.
            async def measure(path: Path) -> None:
                dirty[path] = await self._dirty(path)

            async with anyio.create_task_group() as group:
                for path in pending:
                    group.start_soon(measure, path)
        return [
            KeptWorktree(
                path=path,
                branch=branch,
                agent_id=path.name,
                session_id=path.parent.name,
                dirty=dirty[path],
                held=path in live,
            )
            for path, branch in rows
        ]

    async def _dirty(self, path: Path) -> bool:
        """`--untracked-files=normal`, because the answer is a boolean.

        `all` enumerates every file under a provisioned `node_modules` to say
        what one `?? node_modules/` line says. A tree in this list is one the
        disposal policy *kept*, which it does only when the agent's own work is
        in it, so no exclusion is needed here — a held tree is never measured at
        all, since `describe` reports it as held.
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
            return "no agent worktrees are left behind"
        return "\n".join(row.describe() for row in kept)

    async def merge(self, name: str) -> str:
        """Merge an agent's branch into the checkout the person is standing in.

        Deliberately *not* `--no-ff` or squashed or rebased: which of those a
        project wants is a project's policy, and a management command that
        picked one would be making it. This is the plain merge a person would
        type, offered where they already are.
        """
        row = await self.find(name)
        if not row.branch:
            return f"refusing: {row.agent_id} is on a detached HEAD; there is no branch to merge"
        code, out, err = await git(self.ctx, self.base, "merge", "--no-edit", row.branch)
        if code != 0:
            detail = (err.strip() or out.strip()).splitlines()
            return f"could not merge {row.branch}: {detail[0] if detail else f'git exited {code}'}"
        return f"merged {row.branch}"

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

        code, _, err = await git(
            self.ctx, self.base, "worktree", "remove", "--force", str(row.path)
        )
        if code != 0:
            return f"could not remove {row.path}: {err.strip() or f'git exited {code}'}"
        removed = f"removed {row.path}"
        if "--with-branch" not in flags or not row.branch:
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
