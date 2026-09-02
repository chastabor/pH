"""`workspace-git-worktree` — the `worktree` containment tier (D21, §4.8).

The middle rung of the ladder. It gives an agent its own checkout on
`ph/<session>/<agent>`, sharing the repository's object store so creation is
cheap, and points `ctx.fs`'s root and `ctx.subprocess`'s cwd at it.

**What it bounds, exactly.** Every tool-mediated write and every relative-path
raw write, because both resolve against the agent's cwd. **Not**
`open("/etc/passwd", "w")`, which never consults a cwd — only the `sandbox` tier
refuses that, at the kernel. The property bought here is **collision isolation
and revertibility**, not confinement, and any sentence here, in `ph doctor`, or in
a config comment that blurs the two is a defect (§12 Q10, E13).

**`access="read"` is a different kind, not a different permission.** "Read-only"
is an enforcement claim this tier cannot make, so `read` yields
`worktree-ephemeral`: a full checkout the child may write, **discarded on
disposal and never merged**. `repo_writable` stays `True`, because the writes
happen; they simply reach nobody.

**Disposal is a policy, and the redirection env is what keeps it meaningful.** An
unchanged worktree is removed and a changed one is kept to inspect and merge — a
rule that only says something if "changed" means the agent's work. `pytest`
writes `.pytest_cache/` and `__pycache__/` into the tree it runs against, so
without redirection every worktree ends up dirty, every one is kept, and the
policy decays into "keep everything". The env points build caches inside
`scratch` (E12), which is outside the worktree and survives disposal. Best-effort
by construction: a toolchain that insists on writing beside its sources still
will, and the answer is `access="write"` for that agent, not a weaker tier.

**Per-run restore points live here too (E7, §12 Q9c)** — `git add -A`, `git
write-tree` and a ref namespace end to end, which is why they are a capability of
*this* tier and not of the seam. A denial settles the whole run (C3), which bounds
partial state to about one cell; before a run this tier captures the worktree as a
git tree object under a hidden ref, and `/revert` restores it.

@module ph.seams.workspace_git
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path
from ..session import Session
from ..tools.definition import ToolExecution
from ..wire import WireModel
from .subprocess import SubprocessSpawnSpec, scrub_env
from .workspace import (
    EXCLUDE,
    ContainmentTier,
    DeclineReason,
    Workspace,
    WorkspaceAccess,
    WorkspaceDeclined,
    WorkspaceRecord,
    discards_writes,
    redirection_env,
    restorable,
    workspace_of,
)

__all__ = [
    "CHECKPOINT",
    "GitWorktreeProvider",
    "apply",
    "checkpoint",
    "checkpoint_policy",
    "checkpoints",
    "git",
    "latest_checkpoint",
    "ref_for",
    "restore",
    "sanitize_ref",
]

log = logging.getLogger("ph.seams.workspace_git")


async def git(
    ctx: Context, cwd: Path, *args: str, env: Mapping[str, str] | None = None
) -> tuple[int, str, str]:
    """One git invocation, through `ctx.subprocess` — never `os.system`.

    The seam scrubs the credential-shaped environment (F1) and reaps the child in
    a `finally` (F4); a bare `subprocess.run` here would opt this module out of
    both for no gain.

    `env` *adds to* the scrubbed parent environment rather than replacing it —
    `GIT_INDEX_FILE` is the caller with a reason, and a git that inherited
    nothing else would not find its own configuration.

    `LC_ALL=C` because pH reads what git says. `--porcelain` is stable by
    contract, but stderr is gettext-translated and `ctx.subprocess` passes
    `LANG` through — so without this a decline reason, and anything else read
    off git's words, would depend on the operator's locale.
    """
    spec = SubprocessSpawnSpec(
        argv=("git", *args), cwd=cwd, env=scrub_env(extra={"LC_ALL": "C", **(env or {})})
    )
    result: tuple[int, str, str] = await ctx.subprocess.run(spec)
    return result


BRANCH_PREFIX = "ph/"
"""What every branch this tier makes is named under.

Shared with `/workspaces`, which enumerates the prefix to find the artifacts
disposal leaves. A management command that offered to delete a person's own
branches would be a different and much worse tool, and this prefix is the whole
of what keeps it from being one — so the two must not drift.
"""


COMMIT_AS_PH = (
    "-c",
    "user.name=pH",
    "-c",
    "user.email=ph@localhost",
    "commit",
    "--no-verify",
    "--no-gpg-sign",
)
"""Prefix for a commit pH makes on its own behalf, ending at the message.

pH commits in two places — a worktree's disposal and an overlay's export — and
both are *saves*, not contributions. The identity is supplied rather than read
because a workspace runs with `GIT_CONFIG_GLOBAL` redirected into its scratch
(`redirection_env`), where there is no `user.email` to find; the two `--no-`
flags are here for the same reason the identity is, which is that nothing about
this commit should be able to block on an operator's setup. A hook or a signing
prompt failing would strand work that git was about to make durable, and both
checks belong on the merge that publishes it.
"""


_UNSAFE_REF = re.compile(r"[^A-Za-z0-9._-]+")
"""Everything git refuses in a ref component, plus `/`, which would nest.

Session and agent ids are pH's, not a user's, but a ref name is a filesystem
path under `.git/refs` on most setups — so this collapses rather than trusts.
"""


def sanitize_ref(component: str) -> str:
    """One ref path component, safe by construction.

    Git's own rules are a deny-list (`git check-ref-format`); this is the
    allow-list, because a branch name that fails validation *after* a worktree
    has been created is a half-made artifact to clean up.
    """
    cleaned = _UNSAFE_REF.sub("-", component).strip("-.")
    return cleaned or "agent"


@dataclass(slots=True)
class GitWorktreeProvider:
    """The `worktree` tier: one checkout per agent, on its own branch."""

    ctx: Context
    root: Path
    """Where worktrees are checked out — `$PH_HOME/worktrees/<session>/<agent>`,
    outside the repository, so a checkout is never itself a candidate for the
    walk the agent runs over its own tree."""
    tier: ContainmentTier = field(default="worktree", init=False)
    _toplevels: dict[Path, Path | None] = field(default_factory=dict, init=False)
    """`base` → its repository, asked once.

    Every sibling in a fan-out is handed the *same* `base` — the parent's root —
    so without this, `rev-parse` runs once per child to be told the same thing.
    Per process, and a `None` is cached too: a directory that is not a repository
    does not become one while pH is running, and re-asking per agent is how a
    profile that declines pays for the tier it is not using.
    """

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None:
        """A checkout for this agent, or a named decline.

        Declining is the normal answer for half the directories a person runs pH
        in, and the seam's fallback makes it a notice rather than a refusal to
        start. It declines by *raising* `WorkspaceDeclined` rather than returning
        `None`, so the reason survives to `workspace/acquired` and to `ph doctor`
        — a fallback that cannot say why is indistinguishable from no tier at
        all, which is the confusion E15 exists to remove.
        """
        toplevel = await self._toplevel(base)
        if toplevel is None:
            raise WorkspaceDeclined("not-a-repository", f"{base} is not a git repository")

        ref = f"{BRANCH_PREFIX}{sanitize_ref(session_id)}/{sanitize_ref(agent_id)}"
        path = self.root / sanitize_ref(session_id) / sanitize_ref(agent_id)
        await self._add(toplevel, path, ref)

        ephemeral = access == "read"
        return Workspace(
            root=path,
            scratch=scratch,
            kind="worktree-ephemeral" if ephemeral else "worktree",
            # True for both, and deliberately: an ephemeral child writes freely,
            # its writes simply reach nobody. `False` here would be a
            # confinement claim only the sandbox tier can make.
            repo_writable=True,
            ref=ref,
            env=redirection_env(scratch),
            # `workspace.retained`, read at teardown rather than captured here:
            # whether this tree is evidence is decided by how the agent *ended*,
            # which nobody knows at acquire time (P6-28).
            release=lambda workspace: self._release(
                toplevel,
                path,
                ref,
                discard=ephemeral and not workspace.retained,
                pathspec=workspace.agent_work_pathspec(),
            ),
        )

    async def export(self, record: WorkspaceRecord) -> str:
        """The branch this agent has been committing to all along.

        Nothing to build: a worktree *is* its branch, so exporting one is naming
        it. The method exists so `/workspaces` can ask the seam one question
        instead of asking which tier answered — the overlay tier, whose work
        lives in a delta until somebody assembles a commit, is the one that makes
        this verb non-trivial.
        """
        if record.ref is None:
            raise WorkspaceDeclined(
                "provider-failed",
                f"{record.agent_id} is on a detached HEAD; there is no branch to export",
            )
        return record.ref

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        """Release a tree this process never acquired (F6).

        The same disposal policy an orderly release runs — commit what is uncommitted,
        then remove the checkout — because a crash is not a reason to throw away work,
        and reconciliation that discarded more than a normal exit would make crashing
        *worse* than the leak it is fixing. **This is what makes a crash cost a directory
        instead of a day**: the branch ends up holding what the tree held when the
        process died, and a worktree is re-addable from a branch.

        **The record locates its own repository.** Taking the base from the reconciling
        process's `ctx.fs.root` is a guess: a pair recorded against a different checkout
        resolves the wrong toplevel. A linked worktree knows its own common directory, so
        the tree answers the question about itself.

        There is no `Workspace`, so nothing is known to have been *provisioned* and every
        file counts as the agent's work — which errs toward committing, where the cost
        is a commit somebody can drop rather than work nobody can recover.

        **A retention is an exception to `discard`, exactly as it is at release**
        (P6-28), rather than a rule of its own here: two rules for one word is how a tree
        gets deleted by whichever path reached it first. A retained tree is
        therefore removed on both paths and loses nothing — everything a child did lives
        on its **branch**, which `-d` declines to delete for precisely this reason. What
        retention buys is the *uncommitted* work an ephemeral tree would have thrown away,
        which is a commit on that branch rather than a directory to go looking for.
        """
        if record.ref is None or not record.root.exists():
            # git pruned it, or a person removed it. Nothing to reclaim, and the
            # pair still wants closing so the next open stops re-reporting it.
            return False
        toplevel = await self._common_root(record.root)
        if toplevel is None:
            return True
        return await self._release(
            toplevel,
            record.root,
            record.ref,
            discard=discards_writes(record.kind) and not record.reason,
            pathspec=(),
        )

    async def _common_root(self, tree: Path) -> Path | None:
        """The main worktree of the repository `tree` is linked into.

        `--git-common-dir` rather than `--show-toplevel`, which is the one place
        this module wants the *shared* directory: inside a linked worktree
        `--show-toplevel` answers with that worktree, and `worktree remove`
        cannot remove the tree it is standing in.
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
        """The repository `base` is in, or `None`.

        Asked of git rather than by looking for a `.git` entry, because a
        worktree's `.git` is a file, a bare checkout has none, and a
        subdirectory is a perfectly good `base` — three cases a directory probe
        gets wrong in the direction of a false positive.
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

    async def _add(self, toplevel: Path, path: Path, ref: str) -> None:
        """`git worktree add`, tolerating the two states a resume can find.

        A worktree already checked out at this path is *reused*: the agent id is
        the key, so finding one means finding this agent's own tree, and
        recreating it would discard the uncommitted work disposal exists to
        put on the branch. A branch that exists without a worktree is attached rather
        than reset, for the same reason — `-B` would silently drop it.
        """
        if (path / ".git").exists():
            log.info("ph.seams.workspace_git: reusing the worktree already at %s", path)
            return
        code, _, err = await self._git(toplevel, "worktree", "add", "-b", ref, str(path), "HEAD")
        if code != 0:
            # Both recoveries hang off the failure, which is what makes them
            # free: a stale registration from a crash is *why* `add` refuses, and
            # an existing branch is the other why. Pruning up front spent a
            # subprocess per acquire to prepare for a case that had not happened.
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

    async def _why(self, toplevel: Path, path: Path, ref: str) -> DeclineReason:
        """Which decline this was, asked of git's *state* rather than its prose.

        git's stderr is gettext-translated — `ctx.subprocess` passes `LANG`/`LC_ALL`
        through, since they are not credential-shaped — so a message match would collapse
        every decline to the generic code on a non-English host, in the row whose entire
        purpose is telling an operator why. Both facts are available structurally, and one
        of them (`_has_ref`) is already asked two lines above.
        """
        if await self._checked_out(toplevel, ref):
            return "branch-in-use"
        if path.exists():
            return "path-exists"
        return "provider-failed"

    async def _checked_out(self, toplevel: Path, ref: str) -> bool:
        """Whether some worktree already holds this branch.

        `--porcelain` rather than the human listing, so this reads the same on
        every locale as the thing it replaced did not.
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

        **The checkout always goes; the branch is what survives.** Uncommitted work is
        committed to it first, so `kept` means "this branch holds work" rather than "a
        directory is still on disk" — the two were the same claim while disposal left
        dirty trees behind, and only the first one is durable.

        Every path reports what actually happened rather than what was intended: a
        commit or a removal that failed leaves the tree on disk and says `True`, which
        keeps the field a record instead of a statement of policy.
        """
        if not path.exists():
            # Disposed already. Reconciliation and the scope that acquired the tree
            # can both reach this — and now that the ordinary path *removes*, the
            # second arrival is the common case rather than the odd one. It decides
            # nothing; it only reports what the first left, which is the branch.
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
        # `--force` even for the clean case: a tree measured clean a moment ago
        # can still hold an untracked file git would object to, and an ephemeral
        # tree is discarded *even if dirty*, which is the kind's whole promise.
        #
        # `Workspace.retained` is the exception, and it is an exception to
        # `discard` rather than to this branch: a retained ephemeral tree takes
        # the same commit-then-remove path a `worktree` does, so retention buys
        # a branch to read and not a checkout to trip over (P6-28).
        code, _, err = await self._git(toplevel, "worktree", "remove", "--force", str(path))
        if code != 0:
            log.warning(
                "ph.seams.workspace_git: could not remove the worktree at %s (%s)",
                path,
                err.strip() or f"git exited {code}",
            )
            return True
        if discard:
            await self._git(toplevel, "branch", "-D", ref)
            return False
        # `-d`, not `-D`: a clean worktree is not evidence that its branch was
        # merged — by this line everything the agent did is *on* that branch,
        # whether the agent committed it or `_commit` just did, and forcing here
        # would delete precisely what disposal set out to preserve. Git refuses
        # instead, the branch survives, and `kept` says so.
        code, _, _ = await self._git(toplevel, "branch", "-d", ref)
        return code != 0

    async def _commit(self, path: Path, ref: str, pathspec: Sequence[str]) -> bool:
        """Put the tree's uncommitted work on its own branch. Reports whether it landed.

        **The branch is the artifact; the checkout is a resource the agent borrowed.**
        A directory left behind is an orphan — nothing enumerates it, nothing collects
        it, and its contents are invisible to every verb `/workspaces` offers, all of
        which name branches. Committing makes the work durable in the one place that
        already survives a crash, which is also what makes recovery cheap: a worktree
        can be re-added from a branch, so a process that dies mid-run loses a
        directory rather than a day.

        **A failure here cancels the removal, not the work.** The caller keeps the
        tree, which is the pre-branch behaviour and the right fallback: an orphan is
        worse than a clean disposal and better than a deletion.

        **Staged wide and then narrowed**, rather than by handing `add` the same
        exclusions `_dirty` reads. `git add -A -- . ':(exclude)deps'` exits 1 with "the
        following paths are ignored by one of your .gitignore files": naming an ignored
        path in a pathspec is a request to add it, and `:(exclude)` does not exempt it
        from that check. It stages the rest anyway, so tolerating the failure would
        work and would also swallow every real one. `reset` narrows without ever
        naming a path *to* `add`, and is a no-op for a provisioned entry the project
        already ignores.

        **The narrowing is what keeps a secret off the branch.** A provisioned `.env`
        the project does *not* gitignore is the case: it was dirt the old policy could
        only mistake for the agent's work, and it is a file `add -A` would now publish
        to a ref somebody merges.

        `--no-verify` and `--no-gpg-sign` because this commit is a save, not a
        contribution: a pre-commit hook that fails, or a signing prompt with no
        terminal to answer it, would strand the work on disk over a check that belongs
        on the merge.
        """
        provisioned = [one[len(EXCLUDE) :] for one in pathspec if one.startswith(EXCLUDE)]
        steps: list[tuple[str, ...]] = [("add", "-A")]
        if provisioned:
            steps.append(("reset", "--quiet", "--", *provisioned))
        steps.append((*COMMIT_AS_PH, "-m", f"{ref}: work at disposal"))
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

        `--porcelain` with untracked files included: a new file nobody staged is exactly
        the work a discarded worktree would lose.

        The pathspec comes from `Workspace.agent_work_pathspec()` rather than being built
        here, because `/workspaces list` and `/revert` ask the same question and when
        they each built their own they disagreed about the same tree.

        An **empty** pathspec is the reconciliation case (F6): a crash leaves a log record
        and no `Workspace`, so every file counts as the agent's work. The caller states
        that rather than the callee decoding a sentinel — and with no exclusion to refine,
        `--untracked-files=normal` is the right mode, because the answer is a boolean and
        `all` enumerates every file under a provisioned `node_modules` to say what one
        `?? node_modules/` line says.
        """
        untracked = "all" if pathspec else "normal"
        code, out, _ = await self._git(
            path,
            "status",
            "--porcelain",
            f"--untracked-files={untracked}",
            "--",
            *pathspec,
        )
        if code != 0:
            # Unreadable is treated as dirty: keeping a tree nobody wanted costs
            # disk, and removing one that held work costs the work.
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
    """Where checkouts live. `$PH_HOME/worktrees` by default, and outside the
    repository on purpose: a worktree inside `base` would be walked by the
    agent's own `glob`, committed by its own `git add -A`, and nested one level
    deeper by every child."""


@plugin("workspace-git-worktree", inject=["workspace", "subprocess"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register the worktree tier as the workspace provider.

    Layered by a profile that wants isolation, never by `ph-base`: the tier costs
    a checkout per agent, and which profiles pay for it is P4-11's decision, not
    this module's.
    """
    provider = GitWorktreeProvider(ctx=ctx, root=default_home_path(config.root, "worktrees"))
    ctx.workspace.register_provider(provider, scope=ctx)


# ------------------------------------------------- per-run restore points --


CHECKPOINT = "workspace/checkpoint"

_INDEX = "ph-checkpoint-index"
"""pH's own index, beside the worktree's git dir.

Per worktree, because `rev-parse --git-dir` inside a linked worktree answers with
that worktree's own directory — so two agents checkpointing at once do not share a
staging area, and neither touches the index the agent's `git` commands use.

**Seeded from the worktree's real index the first time.** `git worktree add` has
just written one full of valid stat data; starting from an empty file makes the
first `add -A` re-hash every file in the repository, paid per agent on the first
cell. Copying it costs half a millisecond.
"""


def ref_for(session_id: str, agent_id: str, seq: int) -> str:
    """`refs/ph/<session>/<agent>/pre-run/<seq>` — hidden, and outside `refs/heads`.

    Not a branch: `git branch` does not list it, `git log` does not walk it, and
    a person's `git push` does not carry it. It exists for exactly one reason —
    to keep the tree object from being garbage-collected between the run and the
    revert — which is why disposal prunes the whole `pre-run` namespace.
    """
    return f"refs/ph/{sanitize_ref(session_id)}/{sanitize_ref(agent_id)}/pre-run/{seq}"


async def checkpoint(
    ctx: Context, workspace: Workspace, *, session: Session, agent_id: str, call_id: str
) -> str | None:
    """Capture the worktree and record the restore point. `None` when not applicable.

    Only for a workspace *this tier* made: `ref` is a git ref and `tree` is a git
    object, so a kind this provider did not produce has no restore point to take
    — and a `shared` workspace is the person's own checkout, where offering to
    overwrite their uncommitted work with whatever an agent found is the one
    thing this must never do.

    What is captured is `agent_work_pathspec()` — the tree *minus* what the seam
    provisioned (E14). A copied `.env` or a hardlinked `node_modules` is not the
    agent's work, so hashing it into every cell's tree would be both wrong and
    the most expensive thing here.
    """
    # Both guards live in `tree_hash`, which returns `None` for exactly these
    # two conditions. Keeping copies here costs a *fifth* `git` spawn per code
    # cell — `rev-parse --absolute-git-dir` — which is the waste
    # `_cached_checkpoint`'s own docstring names: "one of four spawns spent
    # learning a constant".
    tree = await tree_hash(ctx, workspace)
    if tree is None:
        return None

    # Write-ahead (A10): the event precedes the ref that keeps the tree alive, so
    # a crash in between leaves a checkpoint that is *unavailable* and says so,
    # never a ref nobody recorded. The event's own seq is the address `/revert`
    # takes, so nothing here predicts what `append` will assign.
    event = session.append(CHECKPOINT, {"agentId": agent_id, "tree": tree, "callId": call_id})
    ref = ref_for(session.id, agent_id, event.seq)
    code, _, err = await git(ctx, workspace.root, "update-ref", ref, tree)
    if code != 0:
        log.warning("ph.seams.workspace_git: could not write %s (%s)", ref, err)
    return ref


async def _checkpoint_index(git_dir: Path) -> Path:
    """pH's index, seeded once from the worktree's own so the first cell is cheap."""
    index = git_dir / _INDEX

    def seed() -> None:
        if index.exists():
            return
        live = git_dir / "index"
        if live.exists():
            shutil.copyfile(live, index)

    await anyio.to_thread.run_sync(seed)
    return index


async def restore(ctx: Context, workspace: Workspace, tree: str) -> tuple[str, ...]:
    """Put the worktree back to `tree`. Returns the paths the run had added.

    **`read-tree --reset -u` against a *seeded* scratch index**, which is the whole
    trick: seeding from the checkpoint index — refreshed first, so its stat data is
    current — lets git touch only the paths that actually differ. `checkout-index -a
    -f` rewrites *every* file in the tree, which is slower and, worse, stamps a new
    mtime on every unchanged file and so invalidates every mtime-keyed cache the
    person has — pytest, mypy, ruff, the editor's index.

    Scratch, not the agent's index, so a file that was untracked before the run is
    untracked after the restore rather than silently staged. `.gitignore`d paths were
    never in the checkpoint, so they are never considered in either direction — a
    `/revert` that wiped a build cache would turn a recovery into a rebuild.
    """
    git_dir = await _git_dir(ctx, workspace.root)
    if git_dir is None:
        raise FileNotFoundError(f"{workspace.root} is not a git checkout")

    checkpoint_index = await _checkpoint_index(git_dir)
    environ = {"GIT_INDEX_FILE": str(checkpoint_index)}
    await git(ctx, workspace.root, "add", "-A", "--", *workspace.agent_work_pathspec(), env=environ)
    index = git_dir / "ph-restore-index"
    await anyio.to_thread.run_sync(lambda: shutil.copyfile(checkpoint_index, index))
    environ = {"GIT_INDEX_FILE": str(index)}

    # What the run added, asked once and before the reset takes it away.
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
    """A `-z` listing, split the way git wrote it.

    NUL-separated because a path may contain a newline, and a restore that
    skipped such a file would leave exactly the thing it promised to remove.
    """
    code, out, _ = await git(ctx, cwd, *args, env=env)
    return [] if code != 0 else [item for item in out.split("\0") if item]


async def _git_dir(ctx: Context, root: Path) -> Path | None:
    code, out, _ = await git(ctx, root, "rev-parse", "--absolute-git-dir")
    return Path(out.strip()) if code == 0 and out.strip() else None


def checkpoints(session: Session) -> dict[int, dict[str, Any]]:
    """Every restore point in this session, by the event's own seq — a fold.

    A checkpoint is a fact in the log, so a resumed or forked session finds the
    same restore points a live one has, without anything having remembered them.
    """
    return {event.seq: dict(event.data) for event in session.events if event.type == CHECKPOINT}


async def tree_hash(ctx: Context, workspace: Workspace) -> str | None:
    """What this agent's work currently hashes to, or `None` if it cannot say.

    `git add -A && git write-tree` against pH's own index, which is what makes
    the capture invisible: branch history, the working tree and the agent's own
    staging area are all untouched (P4-09). What comes back is a content
    address — two identical trees hash identically, and any edit changes it.

    Extracted because a second caller wanted the same answer for a different
    reason: P4-09 stores it as a restore point, and P5-07 uses it as a
    **fingerprint**, to decide that a quality gate which failed against this
    exact tree does not need running again. One derivation, so a gate memo and
    a checkpoint can never disagree about whether the work changed.
    """
    if not restorable(workspace.kind):
        return None
    git_dir = await _git_dir(ctx, workspace.root)
    if git_dir is None:
        return None
    index = await _checkpoint_index(git_dir)
    environ = {"GIT_INDEX_FILE": str(index)}
    pathspec = workspace.agent_work_pathspec()
    code, _, err = await git(ctx, workspace.root, "add", "-A", "--", *pathspec, env=environ)
    if code != 0:
        log.warning("ph.seams.workspace_git: could not stage %s (%s)", workspace.root, err)
        return None
    code, out, err = await git(ctx, workspace.root, "write-tree", env=environ)
    if code != 0:
        log.warning("ph.seams.workspace_git: could not write a tree (%s)", err)
        return None
    return out.strip()


def latest_checkpoint(session: Session, agent_id: str) -> str:
    """The newest restore point *this agent* took, or `""` if it has none.

    A reverse scan rather than `checkpoints()` plus `max()`: the caller that wants one
    restore point does not need a dict of every restore point, and building it copies
    each payload to discard all but the last — on a crash path that runs once per
    retry.

    Scoped to the agent, which is the rule `/revert` already states: a restore point
    belongs to the agent that took it. Only one agent writes into a root session today,
    so this is a latent difference rather than a live one — it is here so the two
    readers of this fold cannot disagree about it later.
    """
    for event in reversed(session.events):
        if event.type == CHECKPOINT and str(event.data.get("agentId", "")) == agent_id:
            return str(event.data.get("tree", ""))
    return ""


@plugin("workspace-checkpoint", inject=["tools", "workspace", "subprocess"])
async def checkpoint_policy(ctx: Context, _config: Any) -> None:
    """Take a restore point before every code run that has a worktree to save.

    Around the *transport*, because a run is the unit that can be denied with
    work already done (Q9a) — a native call that is denied never ran, so there is
    nothing to restore it to. The transport is identified by the view's own
    `transport_name`, since a profile may present it as `ipython`.
    """
    git_dirs: dict[Path, Path | None] = {}

    async def around(execution: ToolExecution, next_: Callable[..., Any]) -> Any:
        # A failure here never blocks the run: a missing restore point is worse
        # than no restore point only if it is believed in, and the log records
        # which runs have one. The guard is inside the `try` on purpose — reading
        # the tool view is itself a call that must not take a cell down.
        try:
            view = ctx.tools.view(execution.scope)
            workspace = workspace_of(ctx, execution.agent)
            if (
                execution.session is not None
                and execution.name == view.transport_name
                and workspace is not None
            ):
                await _cached_checkpoint(ctx, git_dirs, workspace, execution)
        except Exception:
            log.warning("ph.seams.workspace_git: no restore point for this run", exc_info=True)
        return await next_()

    ctx.on("tools/execute", around)


async def _cached_checkpoint(
    ctx: Context,
    git_dirs: dict[Path, Path | None],
    workspace: Workspace,
    execution: ToolExecution,
) -> None:
    """`checkpoint`, with the one immutable answer in it remembered.

    A worktree's git directory does not move, so asking `rev-parse` per cell is
    one of four spawns spent learning a constant — the same finding
    `_toplevels` records for `--show-toplevel` one class up.
    """
    if not restorable(workspace.kind):
        # The gate `checkpoint` applies anyway, hoisted above the `rev-parse`
        # that would otherwise run — a subprocess in the person's own checkout,
        # for a `shared` workspace that can never have a restore point. Newly
        # reachable once a profile layers this row and puts the root agent on
        # `advisory`, which the shipped `rlm` bundle does.
        return
    if workspace.root not in git_dirs:
        git_dirs[workspace.root] = await _git_dir(ctx, workspace.root)
    if git_dirs[workspace.root] is None:
        return
    assert execution.session is not None
    await checkpoint(
        ctx,
        workspace,
        session=execution.session,
        agent_id=getattr(execution.agent, "id", ""),
        call_id=execution.call_id,
    )
