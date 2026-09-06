"""`workspace-jj` — the `worktree` tier over Jujutsu, where a child sees work in progress.

A third workspace provider, and it exists for one property the git tier cannot
offer. **jj's working copy is a commit**: there is no uncommitted state, because
every `jj` command snapshots the tree it runs in before doing anything else.

That is not a nicety here, it is the gap P4-11 left open. A child under
`workspace-git` branches from its parent's last *commit*, and a live parent
commits only at disposal — so for the whole of a session the parent's branch tip
is the base commit, and every child starts from context the parent moved past
hours ago. The parent's actual work is sitting uncommitted, invisible. Under jj
the same spawn starts from the parent's working copy, uncommitted work included,
which is what makes a fan-out inherit the state it was fanned out *from*.

**A peer of `workspace-git`, not a rung above it.** `tier` is `worktree` and the
kinds are the worktree kinds, because the guarantee is identical — cwd-relative
and tool-mediated writes are bounded, an absolute-path raw write is not. Nothing
here confines a process. `describe_tier` corrects the one column where the rung's
stock text would overstate this tier: it has no per-run checkpoints.

**The fork point is frozen, and that is the whole of `_fork_point`.** Handing
`workspace add` the parent's `@` looks right and is a trap: jj rewrites the
working-copy commit on every snapshot and **auto-rebases its descendants**, so
each thing the parent did afterwards would rebase the child and leave the child's
workspace *stale* — every `jj` command in it refusing until someone runs
`workspace update-stale`, which would then pull the parent's newer work into a
tree the child is in the middle of editing. Measured, not reasoned about. So the
parent's work is frozen into a commit nothing will rewrite and children fork from
that. Which is the shared spawn point in jj's own terms: one fork point, every
child that spawns off it, and a parent free to keep working.

**It never converts a repository.** `jj git init --colocate` writes `.jj/` into
somebody's project and changes what their own `git` commands report; a harness
doing that because a row was mounted would be deciding something about a person's
repo that the person did not decide. So a base jj does not already manage is
declined, and the deployment adopts jj rather than pH adopting it for them.

**Colocation is what keeps `git` working**, and it is why this is a provider swap
rather than a migration: a bookmark exports to a real `refs/heads` entry, so the
ref this tier hands back is one `git show` resolves, `/workspaces` can enumerate
under `BRANCH_PREFIX`, and a person merges with the git they already know.

@module ph.seams.workspace_jj
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path
from ..wire import WireModel
from .containment import TIERS, TierDescription
from .diagnostics import Diagnostic, contribute
from .subprocess import SubprocessSpawnSpec, first_line, scrub_env
from .workspace import (
    ContainmentTier,
    DeclineReason,
    Workspace,
    WorkspaceAccess,
    WorkspaceDeclined,
    WorkspaceRecord,
    discards_writes,
    redirection_env,
)
from .workspace_git import BRANCH_PREFIX, sanitize_ref

__all__ = ["JjWorkspaceProvider", "apply", "jj"]

log = logging.getLogger("ph.seams.workspace_jj")


async def jj(
    ctx: Context, cwd: Path, *args: str, env: Mapping[str, str] | None = None
) -> tuple[int, str, str]:
    """One `jj` invocation, through `ctx.subprocess` — `workspace_git.git`'s twin.

    Through the seam for that function's reasons rather than for tidiness: the
    seam scrubs the credential-shaped environment (F1) and reaps the child in a
    `finally` (F4), and a bare `subprocess.run` here would opt this module out of
    both for no gain.

    `LC_ALL=C` for the same reason it is there: pH reads what the tool says.

    **`cwd` is never incidental.** jj snapshots the workspace the command runs in
    and no other, so which directory a call is made from decides whose work the
    command sees — the finding that cost this module a lost child tree in
    testing, when a bookmark set from the parent's directory named the child's
    *previous* commit and the child's work went nowhere.
    """
    spec = SubprocessSpawnSpec(
        argv=("jj", *args), cwd=cwd, env=scrub_env(extra={"LC_ALL": "C", **(env or {})})
    )
    outcome = await ctx.subprocess.run(spec)
    return outcome.exit_code, outcome.stdout, outcome.stderr


_WORK = ("log", "--no-graph", "-r", "@ & ~empty()", "-T", "commit_id.short()")
"""Does this workspace's working copy hold anything? Empty output means no.

One command doing two jobs, which is why it is spelled once and used twice.
Running it **snapshots the workspace first**, so at release it is both the
question and the act of capturing the answer — the child's work lands in its
commit, the bookmark follows, and no separate `jj status` is spent making that
happen.
"""


@dataclass(slots=True)
class JjWorkspaceProvider:
    """The `worktree` tier over jj: one workspace per agent, forked from the parent."""

    ctx: Context
    root: Path
    """Where workspaces are created — `$PH_HOME/jj/<session>/<agent>`, outside the
    repository for `workspace_git`'s reason: a checkout inside `base` is walked by
    the agent's own `glob` and nested one level deeper by every child."""
    tier: ContainmentTier = field(default="worktree", init=False)
    _roots: dict[Path, Path | None] = field(default_factory=dict, init=False)
    """`base` → the root of the jj workspace containing it, asked once.

    Every sibling in a fan-out is handed the *same* `base`, so without this the
    probe runs once per child to be told the same thing. A `None` is cached too:
    a directory jj does not manage does not start being managed while pH runs.
    """

    def describe_tier(self) -> TierDescription:
        """What this tier actually bounds — which is not quite what its rung's row says.

        Two columns are the worktree tier's verbatim, because the bounding is the
        same mechanism and copying it is the measured result rather than a
        convenience. `buys` is where the stock text would be false in both
        directions: `TIERS["worktree"]` sells "per-run checkpoints, /revert", and
        `workspace-checkpoint` is a git row — it hashes a tree with `git
        write-tree`, and a jj workspace is not a git checkout, so no restore point
        is ever written. What this tier buys instead is the thing the git tier
        cannot do, so the row says that rather than staying silent about both.
        """
        return TierDescription(
            bounds=TIERS["worktree"].bounds,
            does_not_bound=TIERS["worktree"].does_not_bound,
            buys=(
                "collision isolation, and a child that starts from its parent's work in "
                "progress rather than from the parent's last commit — but no per-run "
                "checkpoints and no /revert"
            ),
        )

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None:
        """A workspace for this agent, forked from `base`'s work in progress.

        The bookmark is set **here, at acquire**, not at release, and it is not
        bookkeeping run early: a bookmark follows the commit it points at through
        every rewrite, and a jj working copy is rewritten on every snapshot — so
        one `bookmark set` now leaves a name that tracks this child's work for the
        rest of its life. That is what makes `ref` as honest on the
        `workspace/acquired` event as the git tier's branch, which exists from
        `worktree add` onward.

        Declines by raising, like every tier, so the reason survives to
        `workspace/acquired` and `ph doctor` rather than becoming an unexplained
        `shared` (E15).
        """
        managed = await self._managed(base)
        if managed is None:
            raise WorkspaceDeclined(
                "not-a-repository", f"{base} is not inside a workspace jj manages"
            )
        # One name, two uses, so neither can be derived wrongly from the other:
        # `reclaim` recovers the workspace name from the ref, having nothing else
        # left after a crash. jj accepts `/` in a workspace name and resolves
        # `<name>@` as that workspace's working copy, so the shapes can match.
        name = f"{sanitize_ref(session_id)}/{sanitize_ref(agent_id)}"
        ref = f"{BRANCH_PREFIX}{name}"
        path = self.root / sanitize_ref(session_id) / sanitize_ref(agent_id)
        await self._add(managed, base, name, path)

        code, _, err = await jj(self.ctx, managed, "bookmark", "set", ref, "-r", f"{name}@")
        if code != 0:
            # Not fatal, because release names the work again before exporting it
            # and is the authority on whether `kept` is true. What is lost is the
            # *live* view: `ref` on `workspace/acquired` would name something that
            # does not exist yet, and nobody could see which bookmark is which
            # child while they are running. Worth a line, not a decline.
            log.warning(
                "ph.seams.workspace_jj: %s has no bookmark yet (%s); its work is still "
                "named at disposal, but nothing shows it until then",
                path,
                first_line(err) or f"jj exited {code}",
            )

        ephemeral = access == "read"
        return Workspace(
            root=path,
            scratch=scratch,
            kind="worktree-ephemeral" if ephemeral else "worktree",
            # True for both, deliberately, and for the git tier's reason: an
            # ephemeral child writes freely and its writes simply reach nobody.
            # `False` would be a confinement claim only a sandbox can make.
            repo_writable=True,
            ref=ref,
            env=redirection_env(scratch),
            # `workspace.retained` read at teardown rather than captured here:
            # whether this tree is evidence is decided by how the agent *ended*,
            # which nobody knows at acquire time (P6-28).
            release=lambda workspace: self._release(
                name,
                path,
                ref,
                discard=ephemeral and not workspace.retained,
                repo=managed,
            ),
        )

    async def export(self, record: WorkspaceRecord) -> str:
        """The bookmark this agent's work has been tracking all along.

        Nothing to build, exactly as the git tier has nothing to build: the
        bookmark moved with the child's working copy through every snapshot, and
        colocation exported it, so the name handed back is a `refs/heads` entry
        `git show` resolves. The method exists so `/workspaces` asks the seam one
        question rather than asking which tier answered.
        """
        if record.ref is None:
            raise WorkspaceDeclined(
                "provider-failed",
                f"{record.agent_id} has no bookmark; there is nothing to export",
            )
        return record.ref

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        """Release a workspace this process never acquired (F6).

        The same disposal an orderly release runs, for the git tier's reason: a
        crash is not grounds to throw work away, and a reconciliation that
        discarded more than a clean exit would make crashing worse than the leak
        it repairs.

        **The record locates its own repository**, and here that costs nothing —
        every jj command a release needs runs in the child's own directory, which
        the record names, so there is no toplevel to guess at and no reconciling
        process's cwd to mistake for one.

        The workspace name comes back off `ref`, which is why `acquire` spends a
        line making the two the same string. Nothing else survives a crash that
        could carry it.

        **A retention is an exception to `discard`, exactly as it is at release**
        (P6-28), rather than a second rule here: two rules for one word is how a
        tree gets deleted by whichever path reached it first.
        """
        if record.ref is None:
            return False
        return await self._release(
            record.ref.removeprefix(BRANCH_PREFIX),
            record.root,
            record.ref,
            discard=discards_writes(record.kind) and not record.reason,
        )

    async def _add(self, managed: Path, base: Path, name: str, path: Path) -> None:
        """`jj workspace add`, tolerating the two states a resume can find.

        A workspace already at this path is **reused**, the git tier's rule and
        for its reason: the agent id is the key, so finding one means finding this
        agent's own tree, and recreating it would discard the work disposal exists
        to name. A registration whose directory is gone — what a crash leaves — is
        forgotten and retried, which is the one recoverable failure.

        **The recovery hangs off the failure**, so the common case stays one
        subprocess: a first acquire succeeds, and probing the workspace list up
        front would have spent a call on every one of them to learn nothing.
        """
        if (path / ".jj").exists():
            log.info("ph.seams.workspace_jj: reusing the jj workspace already at %s", path)
            return
        # `jj workspace add` refuses a path whose parent is missing, where `git
        # worktree add` creates the chain — so the first agent of a session pays
        # one mkdir rather than a decline that reads as "jj is broken here".
        await anyio.to_thread.run_sync(lambda: path.parent.mkdir(parents=True, exist_ok=True))
        fork = await self._fork_point(base)
        # From `base`, not from `managed`: `@` is per workspace, so a fan-out from
        # a child forks that child's work, which is the whole point of the tier.
        add = ("workspace", "add", "--name", name, "-r", fork, str(path))
        code, _, err = await jj(self.ctx, base, *add)
        registered = code != 0 and await self._registered(managed, name)
        if registered and not path.exists():
            await jj(self.ctx, managed, "workspace", "forget", name)
            code, _, err = await jj(self.ctx, base, *add)
            registered = code != 0
        if code != 0:
            reason = self._why(registered, path)
            detail = first_line(err) or f"jj exited {code}"
            log.warning(
                "ph.seams.workspace_jj: could not create a workspace at %s (%s); declining as %s",
                path,
                detail,
                reason,
            )
            raise WorkspaceDeclined(reason, detail)

    def _why(self, registered: bool, path: Path) -> DeclineReason:
        """Which decline this was, from jj's *state* rather than its prose.

        The two facts are already in hand, and reading them is what keeps this
        module out of the trap `workspace_git._why` documents: a message match
        would be a guess about wording nobody promised to keep.

        A live workspace holding this name is `branch-in-use` — the nearest thing
        the shared vocabulary has to "another checkout has claimed this", which is
        exactly what it means one tier over.
        """
        if registered:
            return "branch-in-use"
        return "path-exists" if path.exists() else "provider-failed"

    async def _fork_point(self, base: Path) -> str:
        """Freeze the parent's work into a commit nothing will rewrite, and name it.

        `jj new` is what does the freezing, and it is the ordinary thing a person
        does before branching off: the parent's working-copy commit stops being
        the working copy, so snapshots stop rewriting it, and a fresh empty commit
        takes its place. **The parent's files are untouched** — the tree on disk
        is identical either way, which is what makes this safe to do in somebody's
        own checkout.

        Skipped when `@` is already empty, and that is the fan-out case rather
        than an optimisation: the first child of a spawn freezes the parent's
        work, and its siblings find nothing new to freeze and fork from the *same*
        commit. Without the guard, eight children would leave eight empty commits
        stacked in the parent's history and each fork from a different one.
        """
        code, out, _ = await jj(self.ctx, base, *_WORK)
        if code == 0 and out.strip():
            await jj(self.ctx, base, "new")
        return "@-"

    async def _registered(self, managed: Path, name: str) -> bool:
        """Whether jj still knows a workspace by this name.

        `jj workspace list` prints `<name>: <commit> …` a line at a time, so the
        test is the prefix. Structural, for `_why`'s reason.
        """
        code, out, _ = await jj(self.ctx, managed, "workspace", "list")
        return code == 0 and any(line.startswith(f"{name}: ") for line in out.splitlines())

    async def _release(
        self, name: str, path: Path, ref: str, *, discard: bool, repo: Path | None = None
    ) -> bool:
        """Snapshot, name, export, forget, remove. Report whether anything was **kept**.

        **The checkout always goes; the bookmark is what survives** — the git
        tier's policy, reached by different means. There is no commit step,
        because there is nothing uncommitted to commit: the first call snapshots
        the working copy into its own commit and the bookmark, which has been
        following that commit since acquire, comes with it. So one command both
        captures the child's work and answers whether it did any.

        **Every jj call runs in the child's own directory**, which is what lets
        this serve `reclaim` unchanged: `git export` and `workspace forget` both
        reach the shared repo from a secondary workspace, so nothing here needs
        to know where the repository is.

        **The export is explicit rather than incidental.** A bookmark becomes a
        `refs/heads` entry when a jj command touches the colocated repo, and
        `workspace forget` would usually do it on the way past — usually is not a
        guarantee, and `kept` is a claim that a person can find this work with the
        git they already use.

        Every path reports what happened rather than what was intended, which is
        what keeps the field a record instead of a statement of policy.
        """
        if not path.exists():
            # Disposed already. Reconciliation and the scope that acquired the
            # tree can both reach this, and the second arrival decides nothing —
            # it reports what the first left, which is the bookmark.
            return repo is not None and await self._named(repo, ref)
        # `None` is "could not tell", and it counts as work for the git tier's
        # reason: keeping a tree nobody wanted costs disk, and discarding one that
        # held work costs the work.
        kept = not discard and await self._has_work(path) is not False
        if kept and not await self._name(path, ref):
            # **A failure to name cancels the removal, not the work** — the git
            # tier's rule at the same point. An orphaned directory is worse than a
            # clean disposal and far better than deleting a tree whose work never
            # reached a ref.
            log.warning(
                "ph.seams.workspace_jj: could not name %s in %s — keeping the tree, because "
                "the only thing worse than an orphaned workspace is deleting one whose work "
                "reached nothing",
                ref,
                path,
            )
            return True
        if not kept:
            await jj(self.ctx, path, "bookmark", "delete", ref)
        code, _, err = await jj(self.ctx, path, "git", "export")
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: %s did not reach git (%s)",
                ref,
                first_line(err) or f"jj exited {code}",
            )
        code, _, err = await jj(self.ctx, path, "workspace", "forget", name)
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: could not forget the workspace %s (%s)",
                name,
                first_line(err) or f"jj exited {code}",
            )
        # jj deliberately leaves the directory — right for a person switching
        # between workspaces, wrong for a harness that promised one per agent and
        # must not accumulate them.
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(path, ignore_errors=True))
        return kept

    async def _name(self, path: Path, ref: str) -> bool:
        """Point the bookmark at what this workspace holds now. Reports whether it landed.

        **Release names the work rather than trusting the name acquire wrote.** The
        bookmark does follow its commit through every snapshot, so on the ordinary
        path this is confirming what is already true — and `kept` is a claim that a
        person can find this work, which is not a claim to make on the strength of a
        call made at a different time that may have failed.
        """
        code, _, err = await jj(self.ctx, path, "bookmark", "set", ref, "-r", "@")
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: jj bookmark set failed in %s (%s)",
                path,
                first_line(err) or f"jj exited {code}",
            )
        return code == 0

    async def _has_work(self, path: Path) -> bool | None:
        """Whether this workspace's working copy holds anything. `None` if jj could not say."""
        code, out, _ = await jj(self.ctx, path, *_WORK)
        return None if code != 0 else bool(out.strip())

    async def _named(self, repo: Path, ref: str) -> bool:
        """Whether a bookmark by this name still points at something."""
        code, out, _ = await jj(self.ctx, repo, "bookmark", "list", ref)
        return code == 0 and bool(out.strip())

    async def _managed(self, base: Path) -> Path | None:
        if base not in self._roots:
            self._roots[base] = await self._ask_root(base)
        return self._roots[base]

    async def _ask_root(self, base: Path) -> Path | None:
        """The root of the jj workspace containing `base`, or `None`.

        Asked of jj rather than by looking for a `.jj` directory, for the reason
        the git tier asks git: a subdirectory is a perfectly good `base`, and a
        probe that stats one path answers wrongly for it.
        """
        code, out, _ = await jj(self.ctx, base, "workspace", "root")
        if code != 0 or not out.strip():
            log.info(
                "ph.seams.workspace_jj: %s is not inside a jj workspace; declining so the "
                "seam falls back to a shared workspace",
                base,
            )
            return None
        root = Path(out.strip())
        if not (root / ".git").exists():
            # Served anyway — the isolation is real and the bookmark is a real
            # handle. But said out loud once per repository, because the merge
            # story this tier advertises is a git one, and a person who cannot
            # `git show` what a child produced should hear why before disposal.
            log.warning(
                "ph.seams.workspace_jj: %s is not colocated with git, so bookmarks this "
                "tier writes will not appear as git branches",
                root,
            )
        return root


class Config(WireModel):
    """Row config for the jj tier."""

    root: str | None = None
    """Where workspaces live. `$PH_HOME/jj` by default, and outside the repository
    on purpose — the same reason the worktree tier gives."""


@plugin("workspace-jj", inject=["workspace", "subprocess"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Claim the workspace slot, but only where jj is installed.

    **The probe gates registration, not acquisition**, which is
    `workspace-agentfs`' rule and is easy to get backwards: `register_provider` is
    `claim_slot` — exclusive — so a row that took the slot and then declined every
    acquire would not fall back to the git tier, it would fall back to `shared`,
    and a deployment that asked for isolation would silently have none.

    Only the *binary* is probed here. Whether a given directory is one jj manages
    is a per-`base` question that `acquire` answers by name, so the reason reaches
    the log and doctor; asking it at mount would be asking about the process's own
    directory, which is not where agents work.

    **Nothing is ever installed, and no repository is ever converted.** A missing
    binary is a decline that says so, in `ph doctor`, through the diagnostics
    seam — the standing provisioning rule, and the reason this row is safe to
    layer in a profile that runs on hosts without jj.
    """
    installed = shutil.which("jj")
    contribute(
        ctx,
        Diagnostic(
            id="workspace-jj",
            title="Jujutsu workspaces",
            read=lambda: [
                ("jj", installed or "not installed — this row declined"),
                (
                    "a child starts from",
                    "its parent's work in progress, uncommitted included"
                    if installed
                    else "nothing; another tier is answering",
                ),
            ],
            order=31,
        ),
    )
    if installed is None:
        log.info("ph.seams.workspace_jj: declining — jj is not installed")
        return
    ctx.workspace.register_provider(
        JjWorkspaceProvider(ctx=ctx, root=default_home_path(config.root, "jj")), scope=ctx
    )
