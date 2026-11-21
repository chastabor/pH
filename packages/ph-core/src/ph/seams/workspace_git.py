"""`workspace-git-worktree`: the `worktree` containment tier (D21, §4.8).

This tier is the middle rung of the ladder. It gives an agent its own checkout
on the branch `ph/<session>/<agent>`. The checkout shares the object store of
the repository, so it is cheap to create. The tier points the root of `ctx.fs`
and the cwd of `ctx.subprocess` at the checkout.

**What it bounds, exactly.** It bounds every write that a tool makes and every
raw write to a relative path, because both resolve against the cwd of the
agent. It does **not** bound `open("/etc/passwd", "w")`, which never reads a
cwd. Only the `sandbox` tier refuses that write, at the kernel. The property
that this tier gives is **collision isolation and revertibility**, not
confinement. Any sentence here, in `phern doctor`, or in a config comment that
confuses the two is a defect (§12 Q10, E13).

**`access="read"` is a different kind, not a different permission.**
"Read-only" is an enforcement claim that this tier cannot make. So `read` gives
`worktree-ephemeral`: a full checkout that the child may write, **discarded on
disposal and never merged**. `repo_writable` stays `True`, because the writes
happen. They reach nobody.

**Disposal is a policy, and the redirection env keeps it meaningful.** Disposal
removes the checkout of an unchanged worktree together with its branch. It
commits the work of a changed worktree to its branch, and keeps the branch to
inspect and merge. That rule has a meaning only if "changed" means the work of
the agent. `pytest` writes `.pytest_cache/` and `__pycache__/` into the tree
that it runs against. Without redirection, every worktree ends up dirty, every
branch is kept, and the policy becomes "keep everything". The env points build
caches inside `scratch` (E12), which is outside the worktree and survives
disposal. The redirection is best-effort by construction. A toolchain that
insists on writing beside its sources still does so. The answer for that agent
is `access="write"`, not a weaker tier.

**Per-run restore points are here too (E7, §12 Q9c).** They are `git add -A`,
`git write-tree` and a ref namespace from end to end. That is why they are a
capability of *this* tier and not of the seam. A denial settles the whole run
(C3), which limits partial state to about one cell. Before a run, this tier
captures the worktree as a git tree object under a hidden ref, and `/revert`
restores it.

@module ph.seams.workspace_git
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..keys import SUBPROCESS, WORKSPACE
from ..paths import canonical, default_home_path, is_under, write_atomic
from ..wire import WireModel
from .subprocess import SubprocessSpawnSpec
from .workspace import (
    EXCLUDE,
    Backend,
    ContainmentTier,
    DeclineReason,
    Stray,
    Workspace,
    WorkspaceAccess,
    WorkspaceDeclined,
    WorkspaceRecord,
    WorkspaceSite,
    checkout_site,
    discards_writes,
    fresh_root,
    measure_strays,
    redirection_env,
    sanitize_ref,
    work_pathspec,
)

__all__ = [
    "GitWorktreeProvider",
    "apply",
    "delete_branch",
    "git",
    "git_lines",
    "list_branches",
    "merge_branch",
    "pre_run_ref",
    "sanitize_ref",
    "tree_hash",
]

log = logging.getLogger("ph.seams.workspace_git")


async def git(
    ctx: Context, cwd: Path, *args: str, env: Mapping[str, str] | None = None
) -> tuple[int, str, str]:
    """Run one git command through `ctx.subprocess`, never through `os.system`.

    The seam scrubs the credential-shaped environment (F1) and reaps the child
    in a `finally` (F4). A bare `subprocess.run` here would take this module out
    of both and gain nothing.

    `env` *adds to* the scrubbed parent environment and does not replace it.
    The case with a reason to pass one is `GIT_INDEX_FILE`. A git that inherited
    nothing else would not find its own configuration.

    **The seam drops the inherited git location** (J10, `subprocess.LOCATION`).
    A pH that was started from a hook therefore does not give every call here a
    repository that outranks the `cwd` above. This is the job of the seam and
    not of this function, because `shell.run` spawns git too: the git that a
    *model* types. A rule written here would not have reached it.

    `LC_ALL=C` is set because pH reads the output of git. `--porcelain` is
    stable by contract. But git translates its stderr with gettext, and
    `ctx.subprocess` passes `LANG` through. Without `LC_ALL=C`, a decline
    reason, and anything else that pH reads from the words of git, would depend
    on the locale of the operator.
    """
    spec = SubprocessSpawnSpec(
        argv=("git", *args),
        cwd=cwd,
        env=ctx.require(SUBPROCESS).env(extra={"LC_ALL": "C", **(env or {})}),
    )
    outcome = await ctx.require(SUBPROCESS).run(spec)
    return outcome.exit_code, outcome.stdout, outcome.stderr


async def git_lines(
    ctx: Context, cwd: Path, *args: str, env: Mapping[str, str] | None = None
) -> list[str]:
    """Return the output of git as a list of lines. Never use `split()` (J5).

    A path may hold a space, and `rev-parse --git-dir --show-toplevel` answers
    with two paths. When the output was split on whitespace, a workspace under
    `~/My Projects` turned two lines into three tokens. The length check failed
    and `_git_dir` reported "not a git checkout". Everything above it believed
    that: there were no checkpoints, `/revert` raised, and an overlay refused
    with `no-base-commit`. The only thing wrong on that machine was the name of
    a directory.

    This is a named function and not `out.splitlines()` at each reader, because
    the split was written by hand at three sites and was wrong at all three.
    Position carries meaning here. That is why this function is not placed
    beside `subprocess.first_line`: `first_line` strips lines and drops blank
    ones, and a blank line in a `rev-parse` reply has to *shift* the line after
    it, not disappear.

    A non-zero exit returns `[]`. A caller then checks the shape that it
    expected one time, and does not check a code and then a length.
    """
    code, out, _ = await git(ctx, cwd, *args, env=env)
    return [] if code != 0 else out.splitlines()


EXCLUDE_FILE = "ph-provisioned-exclude"
"""The file where the seam lists what it provisioned. git reads it as excludes (L9)."""


async def _excludes(git_dir: Path, provisioned: Sequence[str]) -> Path:
    """Write the materials of the seam where `core.excludesFile` will find them.

    **git is told the rule, and nothing is undone afterwards** (L9). Staging
    used to be `add -A` and then `reset -- <provisioned>`: stage everything,
    then spawn a second git to take part of it back. For a path that the project
    already gitignores, that `reset` removed nothing at all. That is the
    ordinary case, because nearly every project that has a provisioned
    `node_modules` or `.venv` gitignores it. The `reset` still cost a process,
    measured at ~2.5 ms of a ~14 ms checkpoint, on the path that a code cell
    takes before every run.

    `core.excludesFile` follows the approach that `workspace_jj.auto_track`
    already uses on the jj tier: state the rule one time, in the vocabulary of
    the tool, and let the tool apply it. One `add` now does what two commands
    used to do. git also reads this file again on every invocation, so the
    answer cannot go stale, unlike anything that pH could remember.

    **Not `info/exclude`.** `ProvisionReport.provisioned` rejects it for a
    reason that still holds: git resolves it against the *common* directory, so
    to write it would hide these paths in every other worktree and in the own
    checkout of the user. This file is per workspace, git reads it per
    invocation, and it mutates nothing.

    The file is written again on each call and is not seeded one time. It is
    one short file in the git dir, and the spawn that follows costs orders of
    magnitude more. A file that is always written cannot disagree with
    `workspace.provisioned`.
    """
    path = git_dir / EXCLUDE_FILE
    # Each entry is anchored with a leading `/`, because these are paths
    # relative to the workspace root and not patterns: a provisioned `deps` must
    # not also exclude `src/deps`. The anchor also makes escaping unnecessary.
    # A leading `#` or `!` would read as a comment or a negation, and neither
    # can be leading when a `/` is in front of it.
    body = "".join(f"/{entry.lstrip('/')}\n" for entry in provisioned)
    await anyio.to_thread.run_sync(lambda: path.write_text(body, encoding="utf-8"))
    return path


async def _mark_tracked(
    ctx: Context, root: Path, provisioned: Sequence[str], env: Mapping[str, str]
) -> None:
    """Set `--skip-worktree` on a provisioned path that the repository tracks.

    **This is the one thing that an exclude cannot do.** `core.excludesFile`
    governs untracked files. A provisioned path that the project has committed
    (a `config.yaml` that the seam overwrites with real credentials) would still
    have its modification staged by `add -A`. The old `reset` covered that case,
    and an exclude alone would silently lose it.

    `skip-worktree` says the same thing to git in the vocabulary of the index:
    the worktree content of this path is not yours to look at. The flag is kept
    in the index that is staged into, so it is set one time per index and is not
    derived again per checkpoint. It is state of git, in a file of git, which is
    the purpose of the change.

    `ls-files` runs first, because `update-index --skip-worktree` exits 128 on a
    path that the index does not already hold. A provisioned path is almost
    never tracked, so the ordinary answer is empty and the second command never
    runs.
    """
    tracked = await _lines(ctx, root, "ls-files", "-z", "--", *provisioned, env=env)
    if not tracked:
        return
    code, _, err = await git(ctx, root, "update-index", "--skip-worktree", "--", *tracked, env=env)
    if code != 0:
        log.warning(
            "ph.seams.workspace_git: could not exclude %d tracked provisioned path(s) in %s (%s)",
            len(tracked),
            root,
            err.strip() or f"git exited {code}",
        )


def _staging_steps(excludes: Path) -> list[tuple[str, ...]]:
    """One `add`, with the materials of the seam excluded by git itself (J3, L9).

    `git add -A -- . ':(exclude)deps'` exits 1 with "the following paths are
    ignored by one of your .gitignore files". To name a path in a pathspec is a
    request to add it, and `:(exclude)` does not exempt the path from that
    check. The command exits 1 even when the exclusion *worked*, because some
    other named path was ignored. So the exclusions cannot be given to `add` as
    a pathspec at all. J3 found this and answered it with a second `reset`
    command.

    The exclusions can be given to `add` as **excludes**, and this function
    does that. Nothing is named to `add`, nothing has to be undone afterwards,
    and the tool that owns the rule applies it. `_excludes` and `_mark_tracked`
    are the two halves. This function is only the shape of the command.

    **What the exclusion is for**: a provisioned `.env` that the project does
    not gitignore. `add -A` alone would hash it into every tree that pH writes,
    and `_commit` would publish it to a ref that somebody merges.
    """
    return [("-c", f"core.excludesFile={excludes}", "add", "-A")]


COMMIT_AS_PH = (
    "-c",
    "user.name=pH",
    "-c",
    "user.email=ph@localhost",
    "commit",
    "--no-verify",
    "--no-gpg-sign",
)
"""The prefix of a commit that pH makes on its own behalf. It ends at the message.

pH commits in two places: the disposal of a worktree and the export of an
overlay. Both are *saves*, not contributions. The identity is supplied and not
read, because a workspace runs with `GIT_CONFIG_GLOBAL` redirected into its
scratch (`redirection_env`), where there is no `user.email` to find. The two
`--no-` flags are here for the same reason as the identity: nothing about this
commit may block on the setup of an operator. A failing hook or a signing
prompt would strand work that git was about to make durable. Both checks belong
on the merge that publishes the work.
"""


@dataclass(slots=True)
class GitWorktreeProvider:
    """The `worktree` tier: one checkout per agent, on its own branch."""

    ctx: Context
    root: Path
    """Where worktrees are checked out: `$PH_HOME/worktrees/<session>/<agent>`.
    This is outside the repository, so a checkout is never a candidate for the
    walk that the agent runs over its own tree."""
    tier: ContainmentTier = field(default="worktree", init=False)
    vcs: Backend = field(default="git", init=False)
    """This tier is git-backed, so an indexer can ask git what changed.

    This is the `ph.seams.workspace.VersionedProvider` capability. It is
    declared and not probed, for the reason that the Protocol states. It cannot
    be inferred from `Workspace.kind`, which is `worktree` for the jj tier too."""
    _toplevels: dict[Path, Path | None] = field(default_factory=dict, init=False)
    """`base` → its repository, asked one time.

    Every sibling in a fan-out is given the *same* `base`, which is the root of
    the parent. Without this memo, `rev-parse` runs one time per child and gives
    the same answer each time. The memo is per process, and it caches a `None`
    too. A directory that is not a repository does not become one while pH is
    running. To ask again per agent would make a profile that declines pay for
    a tier that it is not using.
    """
    _pinned: dict[Path, str] = field(default_factory=dict, init=False)
    """Workspace root → the tree that its last restore point pinned.

    `capture` runs before **every** code cell, and `pre_run_ref` is named by the
    tree that it pins. A cell that changed nothing would therefore write the
    identical ref again with the identical value. The memo turns that spawn into
    a dict lookup. This matters because most cells only inspect or print.
    """

    def locate(self, *, session_id: str, agent_id: str, access: WorkspaceAccess) -> WorkspaceSite:
        """Where the checkout of this agent goes, and on which branch (`LocatingProvider`).

        `acquire` uses this one derivation too. So the `workspace/acquiring` that
        the seam writes before `worktree add` names the directory that the
        command then makes.
        """
        return checkout_site(self.root, session_id, agent_id, access)

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None:
        """Return a checkout for this agent, or decline with a named reason.

        To decline is the normal answer for half the directories that a user
        runs pH in. The fallback of the seam makes a decline a notice, not a
        refusal to start. This method declines by *raising* `WorkspaceDeclined`
        and not by returning `None`, so that the reason survives to
        `workspace/acquired` and to `phern doctor`. A fallback that cannot say
        why looks the same as no tier at all, and E15 exists to remove that
        confusion.
        """
        toplevel = await self._toplevel(base)
        if toplevel is None:
            raise WorkspaceDeclined("not-a-repository", f"{base} is not a git repository")

        site = self.locate(session_id=session_id, agent_id=agent_id, access=access)
        path, ref = site.root, site.ref
        assert ref is not None, "a checkout is always on a branch"
        reused = await self._add(toplevel, path, ref)

        return Workspace(
            root=path,
            scratch=scratch,
            kind=site.kind,
            reused=reused,
            # True for both kinds, on purpose. An ephemeral child writes freely,
            # and its writes reach nobody. `False` here would be a confinement
            # claim that only the sandbox tier can make.
            repo_writable=True,
            ref=ref,
            env=redirection_env(scratch),
            # `workspace.retained` is read at teardown and is not captured here.
            # Whether this tree is evidence depends on how the agent *ended*,
            # and nobody knows that at acquire time (P6-28).
            release=lambda workspace: self._release(
                toplevel,
                path,
                ref,
                discard=discards_writes(workspace.kind) and not workspace.retained,
                pathspec=workspace.agent_work_pathspec(),
            ),
        )

    async def capture(self, workspace: Workspace) -> str | None:
        """Return the tree that the work of this agent hashes to, pinned against `gc`.

        `CheckpointingProvider`. The hash is the token. The pin is the second
        half of the job and not a detail: `write-tree` leaves an object that
        nothing references, and a restore point that a `gc` collected is worse
        than none, because `/revert` had already listed it.

        A workspace with no branch cannot be pinned into a namespace that is
        derived from a branch. It gets no restore point, which is better than an
        unpinned promise.
        """
        tree = await tree_hash(self.ctx, workspace)
        if tree is None or workspace.ref is None:
            return None
        if self._pinned.get(workspace.root) == tree:
            # Already pinned. The ref is named by the tree, so this would write
            # the same value to the same name. Most cells change nothing, and
            # for them this is the difference between four spawns and three.
            return tree
        code, _, err = await self._git(
            workspace.root, "update-ref", pre_run_ref(workspace.ref, tree), tree
        )
        if code != 0:
            # The tree exists in both cases. What is missing is the guarantee
            # that it survives. So the caller gets `None`, and not a token that
            # may fail to resolve later.
            log.warning("ph.seams.workspace_git: could not pin %s (%s)", tree, err.strip())
            return None
        self._pinned[workspace.root] = tree
        return tree

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]:
        """Put the worktree back to `token`. The other half of `CheckpointingProvider`."""
        return await restore_tree(self.ctx, workspace, token)

    async def refs(self, base: Path) -> list[str]:
        """`ArtifactProvider`. Every branch, through the shared implementation."""
        return await list_branches(self.ctx, base)

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str:
        return await delete_branch(self.ctx, base, ref, force=force)

    async def merge(self, base: Path, ref: str) -> str:
        return await merge_branch(self.ctx, base, ref)

    async def strays(
        self, base: Path, *, with_status: bool = True, skip: Container[str] = ()
    ) -> list[Stray]:
        """Return the checkouts that this tier still has on disk (`EnumeratingProvider`).

        The source is `git worktree list --porcelain`, filtered to the own root
        of this tier. A `ph/*` branch that is checked out somewhere else is not
        a tree that this tier made. The filter is here and not in `/workspaces`,
        because the root is a setting of the provider and the command was
        carrying a copy of it.

        `with_status=False` is for the verbs that need only a name. `git status`
        is one subprocess per checkout, and `merge` and `remove` never read
        `dirty`.

        **`dirty` counts every untracked file**, a provisioned one included.
        The seam passes the refs of live agents as `skip`, so only a checkout
        that no live agent holds is asked, and this method has no list of what
        was provisioned into such a tree. A leftover tree with a provisioned
        `.env` and no work therefore reads as dirty here. Disposal and crash
        reclaim have the list and leave the materials out.
        """
        code, out, _ = await self._git(base, "worktree", "list", "--porcelain")
        if code != 0:
            return []
        found = [
            (ref, path) for path, ref in _parse_worktrees(out) if ref and is_under(path, self.root)
        ]
        return await measure_strays(
            found,
            lambda path: self._dirty(path, ()),
            with_status=with_status,
            skip=skip,
        )

    async def discard(self, path: Path) -> str:
        """Remove one checkout and leave its branch alone (`EnumeratingProvider`).

        The tree locates its own repository, which is the rule of `reclaim`. A
        path that was recorded against a different checkout resolves the wrong
        toplevel, and `worktree remove` cannot run from inside the tree that it
        removes.
        """
        toplevel = await self._common_root(path)
        if toplevel is None:
            return f"{path} is not a git worktree"
        code, _, err = await self._git(toplevel, "worktree", "remove", "--force", str(path))
        return "" if code == 0 else (err.strip() or f"git exited {code}")

    async def export(self, record: WorkspaceRecord) -> str:
        """Return the branch that this agent has been committing to all along.

        There is nothing to build: a worktree *is* its branch, so to export one
        is to name it. The method exists so that `/workspaces` can ask the seam
        one question and does not have to ask which tier answered. The overlay
        tier is the one that makes this verb non-trivial, because its work stays
        in a delta until somebody assembles a commit.
        """
        if record.ref is None:
            raise WorkspaceDeclined(
                "provider-failed",
                f"{record.agent_id} is on a detached HEAD; there is no branch to export",
            )
        return record.ref

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        """Release a tree that this process never acquired (F6).

        This runs the same disposal policy as an orderly release: commit what is
        uncommitted, then remove the checkout. A crash is not a reason to throw
        work away. A reconciliation that discarded more than a normal exit would
        make a crash *worse* than the leak that it fixes. **This is what makes a
        crash cost a directory and not a day.** The branch ends up holding what
        the tree held when the process died, and a worktree can be added again
        from a branch.

        **The record locates its own repository.** To take the base from the
        `ctx.fs.root` of the reconciling process is a guess: a pair that was
        recorded against a different checkout resolves the wrong toplevel. A
        linked worktree knows its own common directory, so the tree answers the
        question about itself.

        **What was provisioned comes from the record** (S13). It is the one fact
        of the live `Workspace` that a crash would otherwise take with it.
        Everything else in the tree counts as the work of the agent and is
        committed, which errs toward keeping. A provisioned `.env` is the
        exception, because the branch is something that somebody merges.

        **A retention is an exception to `discard`, exactly as it is at
        release** (P6-28). It is not a separate rule here. With two rules for
        one word, a tree is deleted by whichever path reaches it first. A
        retained tree is therefore removed on both paths and loses nothing.
        Everything that a child did is on its **branch**, and `-d` declines to
        delete that branch for exactly this reason. The benefit of retention is
        the *uncommitted* work that an ephemeral tree would have thrown away.
        That work becomes a commit on the branch, and not a directory that
        somebody has to look for.
        """
        if record.ref is None or not record.root.exists():
            # git pruned it, or a person removed it. There is nothing to
            # reclaim. The pair still needs closing, so that the next open stops
            # reporting it again.
            return False
        toplevel = await self._common_root(record.root)
        if toplevel is None:
            return True
        return await self._release(
            toplevel,
            record.root,
            record.ref,
            discard=discards_writes(record.kind) and not record.reason,
            pathspec=work_pathspec(record.provisioned),
        )

    async def _common_root(self, tree: Path) -> Path | None:
        """Return the main worktree of the repository that `tree` is linked into.

        This uses `--git-common-dir` and not `--show-toplevel`. It is the one
        place where this module wants the *shared* directory. Inside a linked
        worktree, `--show-toplevel` answers with that worktree, and `worktree
        remove` cannot remove the tree that it stands in.
        """
        code, out, _ = await self._git(
            tree, "rev-parse", "--path-format=absolute", "--git-common-dir"
        )
        if code != 0 or not out.strip():
            return None
        return Path(out.strip()).parent

    async def _toplevel(self, base: Path) -> Path | None:
        if base not in self._toplevels:
            self._toplevels[base] = await self._ask_toplevel(base)
        return self._toplevels[base]

    async def _ask_toplevel(self, base: Path) -> Path | None:
        """Return the repository that `base` is in, or `None`.

        This asks git and does not look for a `.git` entry. The `.git` of a
        worktree is a file, a bare checkout has none, and a subdirectory is a
        valid `base`. A directory probe gets those three cases wrong, in the
        direction of a false positive.
        """
        code, out, _ = await self._git(base, "rev-parse", "--show-toplevel")
        if code != 0:
            log.info(
                "ph.seams.workspace_git: %s is not a git repository; declining so the "
                "seam falls back to a shared workspace",
                base,
            )
            return None
        return Path(out.strip())

    async def _add(self, toplevel: Path, path: Path, ref: str) -> bool:
        """Run `git worktree add`, tolerating the two states that a resume can find.
        Returns whether the worktree was already there (`Workspace.reused`).

        A worktree that is already checked out at this path is *reused*. The
        agent id is the key, so a worktree found here is the own tree of this
        agent. To create it again would discard the uncommitted work that
        disposal exists to put on the branch. A branch that exists without a
        worktree is attached and not reset, for the same reason: `-B` would
        silently drop it.
        """
        if (path / ".git").exists():
            log.info("ph.seams.workspace_git: reusing the worktree already at %s", path)
            return True
        code, _, err = await self._git(toplevel, "worktree", "add", "-b", ref, str(path), "HEAD")
        if code != 0:
            # The recovery runs only after the failure, which keeps it free for
            # the common case. A first acquire succeeds in one subprocess, and a
            # prune up front would have spent one more on every acquire.
            #
            # **Attach before prune, because the branch-exists case is no longer
            # rare.** Disposal commits the tree to the branch and removes only
            # the checkout, and `branch -d` refuses an unmerged branch. So a
            # child that is given its workspace back (`rehydrate`), or that is
            # readmitted after a restart, *always* arrives here with the branch
            # present and the checkout gone. A probe with `prune` + `_has_ref`
            # first cost that path two extra subprocesses (~10 ms) to learn what
            # the attempt would have told it.
            code, _, err = await self._git(toplevel, "worktree", "add", str(path), ref)
        if code != 0:
            # Neither worked. A stale registration from a crash is the other
            # reason that `add` refuses, and it is the one that needs a prune.
            await self._git(toplevel, "worktree", "prune")
            retry = (
                ("worktree", "add", str(path), ref)
                if await self._has_ref(toplevel, ref)
                else ("worktree", "add", "-b", ref, str(path), "HEAD")
            )
            code, _, err = await self._git(toplevel, *retry)
        if code != 0:
            detail = err.strip() or f"git exited {code}"
            reason = await self._why(toplevel, path, ref)
            log.warning(
                "ph.seams.workspace_git: could not create a worktree at %s (%s); declining as %s",
                path,
                detail,
                reason,
            )
            raise WorkspaceDeclined(reason, detail)
        return False

    async def _why(self, toplevel: Path, path: Path, ref: str) -> DeclineReason:
        """Return which decline this was, from the *state* of git and not from its messages.

        git translates its stderr with gettext, and `ctx.subprocess` passes
        `LANG` and `LC_ALL` through because they are not credential-shaped. A
        match on the message would give the generic code for every decline on a
        non-English host, in the row whose only purpose is to tell an operator
        why. (`git` above sets `LC_ALL=C` for the same reason.) Both facts can
        be read from the state, in the same way that `_add` already reads
        `_has_ref`.
        """
        if await self._checked_out(toplevel, ref):
            return "branch-in-use"
        if path.exists():
            return "path-exists"
        return "provider-failed"

    async def _checked_out(self, toplevel: Path, ref: str) -> bool:
        """Whether some worktree already holds this branch.

        This reads `--porcelain` and not the human listing, so the answer is the
        same in every locale. The check that it replaced was not.
        """
        code, out, _ = await self._git(toplevel, "worktree", "list", "--porcelain")
        return code == 0 and f"branch refs/heads/{ref}" in out.splitlines()

    async def _release(
        self,
        toplevel: Path,
        path: Path,
        ref: str,
        *,
        discard: bool,
        pathspec: Sequence[str],
    ) -> bool:
        """Commit, remove, and report whether anything was kept. The disposal policy.

        **The checkout always goes, and the branch survives.** Uncommitted work
        is committed to the branch first. So `kept` means "this branch holds
        work" and not "a directory is still on disk". The two were the same
        claim while disposal left dirty trees behind, and only the first one is
        durable.

        Every path reports what happened and not what was intended. A commit or
        a removal that failed leaves the tree on disk and returns `True`. That
        keeps the field a record and not a statement of policy.
        """
        # The pin memo goes with the tree that it names. Otherwise it would keep
        # one entry for every worktree that the mount ever made.
        self._pinned.pop(path, None)
        if not path.exists():
            # Disposed already. Reconciliation and the scope that acquired the
            # tree can both reach this. Now that the ordinary path *removes* the
            # checkout, the second arrival is the common case and not the odd
            # one. It decides nothing. It only reports what the first arrival
            # left, which is the branch.
            code, out, _ = await self._git(toplevel, "branch", "--list", ref)
            return code == 0 and bool(out.strip())
        if (
            not discard
            and await self._dirty(path, pathspec)
            and not await self._commit(path, ref, pathspec)
        ):
            log.warning(
                "ph.seams.workspace_git: could not commit %s to %s — keeping the tree, "
                "because the only thing worse than an orphaned checkout is deleting one "
                "whose work never reached the branch",
                path,
                ref,
            )
            return True
        # `--force` even for the clean case. A tree that was measured clean a
        # moment ago can still hold an untracked file that git would object to.
        # And an ephemeral tree is discarded *even if dirty*, which is the
        # promise of the kind.
        #
        # `Workspace.retained` is the exception. It is an exception to `discard`
        # and not to this removal: a retained ephemeral tree takes the same
        # commit-then-remove path as a `worktree`. So retention gives a branch
        # to read and not a checkout to trip over (P6-28).
        code, _, err = await self._git(toplevel, "worktree", "remove", "--force", str(path))
        if code != 0:
            log.warning(
                "ph.seams.workspace_git: could not remove the worktree at %s (%s)",
                path,
                err.strip() or f"git exited {code}",
            )
            return True
        if discard:
            await delete_branch(self.ctx, toplevel, ref, force=True)
            return False
        # `-d`, not `-D`. A clean worktree is not evidence that its branch was
        # merged. By this line, everything that the agent did is *on* that
        # branch, whether the agent committed it or `_commit` just did. To force
        # here would delete exactly what disposal set out to preserve. git
        # refuses instead, the branch survives, and `kept` says so.
        return bool(await delete_branch(self.ctx, toplevel, ref, force=False))

    async def _commit(self, path: Path, ref: str, pathspec: Sequence[str]) -> bool:
        """Put the uncommitted work of the tree on its own branch. Returns whether it landed.

        **The branch is the artifact, and the checkout is a resource that the
        agent borrowed.** A directory that is left behind is an orphan. The
        listing of `/workspaces` shows it (`strays`), but `merge` and `export`
        name branches, so its contents reach neither until they are on one. A
        commit makes the work durable in the one place that already survives a
        crash. It also makes recovery cheap: a worktree can be added again from
        a branch, so a process that dies mid-run loses a directory and not a
        day.

        **A failure here cancels the removal, not the work.** The caller keeps
        the tree. That is the behavior from before disposal committed to a
        branch, and it is the right fallback: an orphan is worse than a clean
        disposal and better than a deletion.

        **`_staging_steps` stages the tree with the provisioned paths
        excluded.** Its docstring says why.

        `--no-verify` and `--no-gpg-sign` are set because this commit is a save,
        not a contribution. A pre-commit hook that fails, or a signing prompt
        with no terminal to answer it, would strand the work on disk because of
        a check that belongs on the merge.
        """
        provisioned = [one[len(EXCLUDE) :] for one in pathspec if one.startswith(EXCLUDE)]
        git_dir = await _git_dir(self.ctx, path)
        if git_dir is None:  # pragma: no cover - `_release` only reaches a checkout
            return False
        # These are the same two halves that the checkpoint path uses. Here they
        # run against the own index of this worktree and not the index of pH.
        # The tree is removed immediately afterwards, so the flag goes with it.
        # The commit is the one thing that must not carry a provisioned secret
        # onto a branch that somebody merges.
        await _mark_tracked(self.ctx, path, provisioned, {})
        steps = [
            *_staging_steps(await _excludes(git_dir, provisioned)),
            (*COMMIT_AS_PH, "-m", f"{ref}: work at disposal"),
        ]
        for args in steps:
            code, _, err = await self._git(path, *args)
            if code != 0:
                log.warning(
                    "ph.seams.workspace_git: git %s failed in %s (%s)",
                    args[0],
                    path,
                    err.strip() or f"git exited {code}",
                )
                return False
        return True

    async def _dirty(self, path: Path, pathspec: Sequence[str]) -> bool:
        """Whether this worktree holds anything worth keeping.

        The command is `--porcelain` with untracked files included. A new file
        that nobody staged is exactly the work that a discarded worktree would
        lose.

        The pathspec comes from the caller and is not built here. Disposal
        passes `Workspace.agent_work_pathspec()`, and crash reclaim passes
        `work_pathspec` over the provisioned list of the record. When each
        caller built its own pathspec, they disagreed about the same tree.
        `strays` passes none, for the reason that its docstring gives.

        `--untracked-files=all` is used only when there is an **exclusion** to
        refine. `normal` reports a directory as one `?? node_modules/` line.
        That line hides a file of the agent inside a provisioned directory
        behind the entry that the pathspec removes. With nothing excluded, the
        answer is a boolean in both modes, and `all` would list every file under
        that `node_modules` to say what one line says.
        """
        untracked = "all" if any(one.startswith(EXCLUDE) for one in pathspec) else "normal"
        code, out, _ = await self._git(
            path,
            "status",
            "--porcelain",
            f"--untracked-files={untracked}",
            "--",
            *pathspec,
        )
        if code != 0:
            # An unreadable status is treated as dirty. To keep a tree that
            # nobody wanted costs disk. To remove a tree that held work costs
            # the work.
            return True
        return bool(out.strip())

    async def _has_ref(self, toplevel: Path, ref: str) -> bool:
        code, _, _ = await self._git(
            toplevel, "show-ref", "--verify", "--quiet", f"refs/heads/{ref}"
        )
        return code == 0

    async def _git(self, cwd: Path, *args: str) -> tuple[int, str, str]:
        return await git(self.ctx, cwd, *args)


class Config(WireModel):
    """Row config for the worktree tier."""

    root: str | None = None
    """Where checkouts are kept. The default is `$PH_HOME/worktrees`, which is
    outside the repository on purpose: a worktree inside `base` would be walked
    by the agent's own `glob`, committed by its own `git add -A`, and nested one
    level deeper by every child."""


@plugin(
    "workspace-git-worktree", affects="environment", inject=[WORKSPACE, SUBPROCESS], config=Config
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the worktree tier as the workspace provider.

    A profile that wants isolation layers this row. `ph-base` never does. The
    tier costs a checkout per agent, and P4-11 decides which profiles pay for
    it. This module does not.
    """
    provider = GitWorktreeProvider(ctx=ctx, root=default_home_path(config.root, "worktrees"))
    ctx.require(WORKSPACE).register_provider(provider, scope=ctx)


# ------------------------------------------------- per-run restore points --


_INDEX = "ph-checkpoint-index"
"""The own index of pH, in the git dir of the worktree beside the real index.

It is per worktree, because `rev-parse --git-dir` inside a linked worktree
answers with the own directory of that worktree. Two agents that checkpoint at
the same time therefore do not share a staging area, and neither touches the
index that the `git` commands of the agent use.

**It is seeded from the real index of the worktree the first time.** `git
worktree add` has just written an index full of valid stat data. To start from
an empty file would make the first `add -A` hash every file in the repository
again, and each agent would pay that on its first cell. The copy costs half a
millisecond.
"""


def pre_run_ref(branch: str, tree: str) -> str:
    """`refs/<branch>/pre-run/<tree>`: hidden, outside `refs/heads`, named by its tree.

    It is not a branch. `git branch` does not list it, `git log` does not walk
    it, and the `git push` of a user does not carry it. It exists for exactly
    one reason: to keep the tree object from being garbage-collected between the
    run and the revert.

    **It is named by the tree that it pins. It used to be named by the `seq` of
    the checkpoint event.** There are two consequences, and both are wanted. A
    capture can pin *before* it records, because it no longer needs a number
    that only `append` could give it. And repeated captures of an unchanged tree
    write **one** ref and not one per run. That bounds the namespace: it holds
    as many refs as the agent made distinct trees. The refs outlive disposal,
    and nothing prunes them today. That is tolerable at one ref per distinct
    tree, and it would not have been at one ref per code cell.
    """
    return f"refs/{branch}/pre-run/{tree}"


async def _checkpoint_index(
    ctx: Context, root: Path, git_dir: Path, provisioned: Sequence[str]
) -> Path:
    """Return the index of pH, seeded one time from the own index of the worktree.
    The seed makes the first cell cheap.

    **Seeded whole or not at all.** An index under this name is never seeded
    again. A copy that was cut off part-way (a full disk, a kill) used to leave
    a torn index there for good, and every later checkpoint staged against it.
    The write is not synced: git rewrites this index at once (`add -A`), by
    rename and without an `fsync` of its own.

    **And marked one time, while it is new** (L9). `_mark_tracked` is a property
    of an index and not of a checkpoint. So the moment when this file comes into
    existence is the one time that the question has to be asked. Every later
    `add -A` against the file reads the flag that git already holds.
    """
    index = git_dir / _INDEX

    def seed() -> bool:
        if index.exists():
            return False
        live = git_dir / "index"
        if live.exists():
            write_atomic(index, live.read_bytes(), durable=False)
        return True

    if await anyio.to_thread.run_sync(seed) and provisioned:
        await _mark_tracked(ctx, root, provisioned, {"GIT_INDEX_FILE": str(index)})
    return index


async def restore_tree(ctx: Context, workspace: Workspace, tree: str) -> tuple[str, ...]:
    """Put the worktree back to `tree`. Returns the paths that the run had added.

    **`read-tree --reset -u` runs against a *seeded* scratch index.** The seed
    is the important part. The scratch index is a copy of the checkpoint index,
    which is refreshed first so that its stat data is current. git then touches
    only the paths that differ. `checkout-index -a -f` rewrites *every* file in
    the tree. That is slower. Worse, it stamps a new mtime on every unchanged
    file, and so invalidates every mtime-keyed cache that the user has: pytest,
    mypy, ruff, the index of the editor.

    The index is a scratch index and not the index of the agent. A file that was
    untracked before the run is therefore untracked after the restore, and is
    not silently staged. `.gitignore`d paths were never in the checkpoint, so
    they are never considered in either direction. A `/revert` that wiped a
    build cache would turn a recovery into a rebuild.
    """
    git_dir = await _git_dir(ctx, workspace.root)
    if git_dir is None:
        raise FileNotFoundError(f"{workspace.root} is not a git checkout")

    checkpoint_index = await _checkpoint_index(ctx, workspace.root, git_dir, workspace.provisioned)
    environ = {"GIT_INDEX_FILE": str(checkpoint_index)}
    excludes = await _excludes(git_dir, workspace.provisioned)
    for args in _staging_steps(excludes):
        await git(ctx, workspace.root, *args, env=environ)
    index = git_dir / "ph-restore-index"
    await anyio.to_thread.run_sync(lambda: shutil.copyfile(checkpoint_index, index))
    environ = {"GIT_INDEX_FILE": str(index)}

    # What the run added. This is asked one time, before the reset takes it away.
    added = await _lines(
        ctx,
        workspace.root,
        "diff-index",
        "--cached",
        "--name-only",
        "--diff-filter=A",
        "-z",
        tree,
        env=environ,
    )
    code, _, err = await git(ctx, workspace.root, "read-tree", "--reset", "-u", tree, env=environ)
    if code != 0:
        raise FileNotFoundError(f"checkpoint tree {tree} is gone: {err.strip() or code}")
    await anyio.to_thread.run_sync(lambda: _prune_empty(workspace.root, added))
    return tuple(sorted(added))


def _prune_empty(root: Path, paths: list[str]) -> None:
    """`read-tree -u` removes files, never the directories they were alone in."""
    for name in paths:
        parent = (root / name).parent
        while parent != root and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent


async def _lines(
    ctx: Context, cwd: Path, *args: str, env: Mapping[str, str] | None = None
) -> list[str]:
    """Return a `-z` listing, split as git wrote it.

    The listing is NUL-separated because a path may contain a newline. A restore
    that skipped such a file would leave exactly the thing that it promised to
    remove.
    """
    code, out, _ = await git(ctx, cwd, *args, env=env)
    return [] if code != 0 else [item for item in out.split("\0") if item]


async def _git_dir(ctx: Context, root: Path) -> Path | None:
    """Return the own git directory of this workspace, or `None` if it has none.

    **`--show-toplevel` is asked in the same command, and the answer is refused
    unless it is `root` itself.** `rev-parse` walks *up* from its cwd. So a
    workspace that is not a git checkout does not fail: it finds whichever
    repository is an ancestor. Under `$PH_HOME` that is nothing on most machines
    and the dotfiles repository of a user on some. The callers here stage a tree
    and write objects. A checkpoint would have hashed the workspace of one tier
    into the store of another repository, silently, on exactly the setup that
    nobody tests on.

    This case became live when a `worktree`-kind workspace stopped implying a
    git checkout, because `workspace-jj` produces one. But the walk was always
    the bug, and this is the depth where the check belongs: one gate, shared by
    `tree_hash`, `restore_tree`, `_commit` and the checkpoint policy, and not a
    provider test added to each of them.

    The two paths are compared in canonical form, because git answers with a
    real path and `$PH_HOME` is often reached through a symlink. `root` is
    canonical by construction, so only the answer of git is resolved here.
    """
    lines = await git_lines(
        ctx, root, "rev-parse", "--path-format=absolute", "--git-dir", "--show-toplevel"
    )
    if len(lines) != 2:
        return None
    git_dir, toplevel = Path(lines[0]), Path(lines[1])
    # `root` is canonical by construction, as every workspace root and provider
    # root is (`ph.paths.canonical`). So only the answer of git needs resolving.
    # To resolve `root` again here would be the private re-resolution that the
    # rule of the seam ended.
    if await anyio.to_thread.run_sync(lambda: canonical(toplevel) != root):
        log.info(
            "ph.seams.workspace_git: %s is not a git checkout — the nearest repository is "
            "%s, which is not this workspace, so nothing here will touch it",
            root,
            toplevel,
        )
        return None
    return git_dir


async def tree_hash(ctx: Context, workspace: Workspace) -> str | None:
    """Return what the work of this agent currently hashes to, or `None` if it cannot say.

    It runs `git add -A && git write-tree` against the own index of pH. That
    makes the capture invisible: the branch history, the working tree and the
    own staging area of the agent are all untouched (P4-09). The result is a
    content address. Two identical trees hash identically, and any edit changes
    the hash.

    It is a separate function because the same answer has two uses. P4-09
    stores it as a restore point. P5-07 uses it as a **fingerprint**, to decide
    that a quality gate which failed against this exact tree does not need to
    run again. There is one derivation, so a gate memo and a checkpoint can
    never disagree about whether the work changed.
    """
    if not fresh_root(workspace.kind):
        # **Never the base.** The seam gates the *capability*. This is the
        # narrower rule, and the function that would touch the tree owes it. The
        # root of a `shared` workspace is the own checkout of the user. To hash
        # their uncommitted work, and then offer to restore it, is the one thing
        # that this function must never do.
        return None
    git_dir = await _git_dir(ctx, workspace.root)
    if git_dir is None:
        return None
    index = await _checkpoint_index(ctx, workspace.root, git_dir, workspace.provisioned)
    environ = {"GIT_INDEX_FILE": str(index)}
    excludes = await _excludes(git_dir, workspace.provisioned)
    for args in _staging_steps(excludes):
        code, _, err = await git(ctx, workspace.root, *args, env=environ)
        if code != 0:
            log.warning("ph.seams.workspace_git: could not stage %s (%s)", workspace.root, err)
            return None
    code, out, err = await git(ctx, workspace.root, "write-tree", env=environ)
    if code != 0:
        log.warning("ph.seams.workspace_git: could not write a tree (%s)", err)
        return None
    return out.strip()


def _parse_worktrees(porcelain: str) -> list[tuple[Path, str]]:
    """Parse `git worktree list --porcelain` into `(path, branch)` pairs.

    The main checkout has no `branch` line when it is detached, and a linked
    worktree always names one. A record with no path is not a record.

    This function was in `/workspaces` and is now here. It parses the own format
    of git, and the command that used to hold it no longer uses git for this
    question at all.
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


# ------------------------------------------------------- git as an artifact --
#
# The three ref verbs of `ArtifactProvider`, as functions and not methods. The
# reason is that **two tiers make git branches and neither of them is the
# other**: the worktree tier commits the work of an agent to a branch, and the
# overlay tier builds a branch out of its delta at export. The verbs have one
# implementation here and not a copy in each tier. `workspace_agentfs` already
# imports `git` and `COMMIT_AS_PH` from this module for the same reason.
#
# `workspace-jj` implements the same three verbs itself, and that is the purpose
# of the Protocol. Its artifact is also a git ref, but jj embeds its own git. So
# a deployment that runs that tier need not have the binary that these functions
# spawn.


async def list_branches(ctx: Context, base: Path) -> list[str]:
    """Every branch in this repository, unfiltered, for `ArtifactProvider.refs`."""
    code, out, _ = await git(ctx, base, "branch", "--list", "--format=%(refname:short)")
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


async def delete_branch(ctx: Context, base: Path, ref: str, *, force: bool) -> str:
    """Run `git branch -d`, and `-D` only when the caller says so.

    **A refusal from `-d` is the mechanism, not an error to decode.** A clean
    worktree is not evidence that its branch was merged, and by disposal
    everything that the agent did is *on* that branch. So the own refusal of git
    is the last protection between a user and the work that they meant to keep.
    """
    code, _, err = await git(ctx, base, "branch", "-D" if force else "-d", ref)
    return "" if code == 0 else (err.strip() or "git refused")


async def merge_branch(ctx: Context, base: Path, ref: str) -> str:
    """Run `git merge --no-edit`, and nothing more.

    It is deliberately not `--no-ff`, squashed or rebased. Which of those a
    project wants is the policy of the project, and a management command that
    picked one would be making that policy.
    """
    code, out, err = await git(ctx, base, "merge", "--no-edit", ref)
    if code == 0:
        return ""
    detail = (err.strip() or out.strip()).splitlines()
    return f"could not merge {ref}: {detail[0] if detail else f'git exited {code}'}"
