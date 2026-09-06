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
here confines a process. `describe_tier` adds the one thing the rung's stock text
cannot know about: what a child inherits.

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

**Restore points come from the operation log** (`CheckpointingProvider`). Every jj
command records an operation, and a commit an operation refers to is not collected
until that operation is — so `capture` names the working-copy commit and has
nothing to pin, where the git tier must write a hidden ref to keep a tree from
being garbage-collected. The revert itself is `jj restore --from`, deliberately
*not* `jj op restore`: the operation log is the right retention mechanism and the
wrong restore unit, since its unit is the whole repository and undoing one child's
cell that way would take every sibling's work back with it.

**The non-guarantee, and it is the mirror image of what this tier is for.** A file
sitting untracked in the base is already in the base's working-copy commit — jj put
it there — so children fork from it and it is reachable from their bookmarks. "A
child starts from its parent's work in progress" and "an uncommitted file in the
base does not reach a child's branch" are one sentence with opposite signs, and the
git tier is the one that keeps the second. What pH *does* keep out is the material
it copied in itself (`auto_track`): a provisioned `.env` is not the agent's work,
and putting it on a ref somebody merges is P6-35's finding rather than bookkeeping.

@module ph.seams.workspace_jj
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..paths import default_home_path, is_under
from ..wire import WireModel
from .containment import TIERS, TierDescription
from .diagnostics import Diagnostic, contribute
from .subprocess import SubprocessSpawnSpec, first_line, scrub_env
from .workspace import (
    ContainmentTier,
    DeclineReason,
    Stray,
    Workspace,
    WorkspaceAccess,
    WorkspaceDeclined,
    WorkspaceRecord,
    discards_writes,
    redirection_env,
)
from .workspace_git import BRANCH_PREFIX, sanitize_ref

__all__ = ["JjWorkspaceProvider", "apply", "auto_track", "jj"]

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


_BOOKMARKS = 'if(remote, "", name ++ "\n")'
"""`jj bookmark list`'s local names, one per line.

A template rather than the human `name: commit description` line, and filtered on
`remote` because the listing's unit is a bookmark-*remote* pair: a colocated repo
has a `git` remote, so a bookmark that has moved since the last export renders
twice under the same name.
"""

_LISTING = 'name ++ "\t" ++ root ++ "\n"'
"""`jj workspace list`'s two machine-readable fields: the name and where it is.

A template rather than the human format, which is a sentence — `name: path commit
bookmark | description` — that nobody promised to keep stable and that would need
splitting on a space a path may contain.
"""

_STATE_PARENT = ("log", "--no-graph", "-r", "@-", "-T", "commit_id")
"""The commit a workspace created here would fork from, resolved once."""

_STATE = ("log", "--no-graph", "-r", "@", "-T", "commit_id")
"""This workspace's working-copy commit — its state, named.

`@` unconditionally, where `_WORK` asks `@ & ~empty()`: an *empty* working copy is
a perfectly good restore point, and it is the interesting one — "the run created
three files, put it back" is exactly the revert a person asks for after a bad
cell. The full id rather than the short prefix, because this is stored in a log
and read back by a later process.
"""

_WORK = ("log", "--no-graph", "-r", "@ & ~empty()", "-T", "commit_id.short()")
"""Does this workspace's working copy hold anything? Empty output means no.

One command doing two jobs, which is why it is spelled once and used twice.
Running it **snapshots the workspace first**, so at release it is both the
question and the act of capturing the answer — the child's work lands in its
commit, the bookmark follows, and no separate `jj status` is spent making that
happen.
"""


def auto_track(provisioned: Sequence[str]) -> tuple[str, ...]:
    """`--config` keeping the seam's own materials out of the agent's commit (E14).

    **jj snapshots everything the project does not gitignore, and provisioning is
    what puts non-gitignored things in a workspace.** A `.env` copied in so the
    tests can run is not the agent's work; here it lands in the working-copy commit
    by default, which puts it on the bookmark, which puts it in the git ref somebody
    merges. That is P6-35's finding arriving at this tier, where it stops being
    bookkeeping and becomes a credential on a branch — and it is silent, because
    every command still succeeds.

    `snapshot.auto-track` is a **fileset**, so this is the exact analogue of the git
    tier's `agent_work_pathspec()`: everything, minus what the seam put there.
    Passed per call rather than written to the repository's config, because the list
    is a property of one *workspace* and a repo config is shared by all of them.

    It governs what becomes tracked, so it only works while the file has never been
    snapshotted — which holds because every jj command pH runs in a child carries
    this, and `workspace add` runs before provisioning does.

    A path holding a `"` is skipped rather than escaped: the string is parsed as a
    fileset, and a quote inside one changes what it selects — so a wrong guess reads
    as "track something we meant to exclude" and says nothing. Nothing pH provisions
    is named like that, and the skip is logged.
    """
    wanted = [one for one in provisioned if '"' not in one]
    if len(wanted) != len(provisioned):
        log.warning(
            "ph.seams.workspace_jj: %d provisioned path(s) hold a quote and cannot be "
            "excluded from the agent's commit by name",
            len(provisioned) - len(wanted),
        )
    if not wanted:
        return ()
    paths = " | ".join(f'"{one}"' for one in wanted)
    # TOML on the outside, fileset on the inside: the value jj parses is a string,
    # and a bare `all() ~ (...)` is not one.
    return ("--config", f"snapshot.auto-track='all() ~ ({paths})'")


def _colocated(root: Path) -> bool:
    """Whether the repository behind this workspace has a git repo beside it.

    **Not `(root / ".git").exists()`**, which is what this was and which is wrong for
    every workspace but the first: a secondary workspace holds no `.git`, so a
    fan-out *from a child* warned that a perfectly colocated repository was not.
    jj records the way back in `.jj/repo` — a directory in the main workspace, a
    file naming it everywhere else — so the question is asked of the repository the
    way jj itself asks it.
    """
    pointer = root / ".jj" / "repo"
    if pointer.is_dir():
        return (root / ".git").exists()
    try:
        # **Relative to `.jj`, and it is written that way**: `../../../../repo/.jj/repo`
        # resolved against the process's cwd is a different repository or none, which
        # is how this managed to report a colocated repo as not colocated.
        named = pointer.parent / pointer.read_text(encoding="utf-8").strip()
        return (named.resolve().parent.parent / ".git").exists()
    except OSError:
        return False


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
        """The rung's own bargain, plus the one thing its stock text cannot know.

        Two columns are the worktree tier's verbatim, because the bounding is the
        same mechanism and copying it is the measured result rather than a
        convenience. `buys` keeps every word of the stock text — the checkpoints and
        `/revert` it sells are **real here**, through the operation log rather than
        through a hidden git ref — and adds what a child inherits, which is the only
        reason to run this tier instead of the git one.

        It said "no per-run checkpoints and no /revert" while that was true, which is
        the direction E1 is usually about. Leaving it there once the capability
        landed would have been the same failure the other way round: a person
        checking whether `/revert` works here would have been told no by the tool
        while the mechanism sat behind it.
        """
        return TierDescription(
            bounds=TIERS["worktree"].bounds,
            does_not_bound=TIERS["worktree"].does_not_bound,
            buys=(
                f"{TIERS['worktree'].buys}, and a child that starts from its parent's "
                "work in progress rather than from the parent's last commit"
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
        fork = await self._add(managed, base, name, path)

        # Through `_in` like every other call made in a directory that is not ours:
        # for a fan-out *from a child*, `managed` is that child's own workspace.
        code, _, err = await self._in(managed, "bookmark", "set", ref, "-r", f"{name}@")
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
                provisioned=workspace.provisioned,
                fork=fork,
                repo=managed,
            ),
        )

    async def capture(self, workspace: Workspace) -> str | None:
        """The commit this workspace's working copy is at — its state, named (P4-09).

        `CheckpointingProvider`. One command, and it is as much a snapshot as a
        question: jj commits the working copy before answering anything, so the id
        that comes back *is* the state and there is nothing left to pin.

        **The operation log is what keeps it reachable**, which is this tier's answer
        to the hidden ref the git tier has to write. Every jj command records an
        operation, and a commit an operation refers to is not collected until that
        operation is — so a restore point survives by the same mechanism that lets a
        person undo anything jj did, rather than by a ref pH has to remember to
        write and would then have to remember to prune.

        **Equal tokens mean equal content**, which is the whole of what a token must
        promise. The converse does not hold here where it does for a git tree: a jj
        commit id covers the committer timestamp as well as the tree, so work that
        returns to a state it held before gets a *new* id. What that costs is a
        quality gate re-run against content it has already judged — the safe
        direction, and the reason `unchanged_failure` is written as "already failed
        against this exact state" rather than as a cache to be trusted.
        """
        code, out, err = await self._jj(workspace, *_STATE)
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: no restore point for %s (%s)",
                workspace.root,
                first_line(err) or f"jj exited {code}",
            )
            return None
        return out.strip() or None

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]:
        """Put the working copy back to `token`. Returns the paths the run had added.

        `jj restore --from <commit>`, and deliberately **not** `jj op restore`: the
        operation log is what keeps the token alive, but its unit is the whole
        repository, so undoing a run that way would take every sibling workspace's
        work back with it. `restore` rewrites only this workspace's working-copy
        commit, which is the unit a per-run revert is about.

        Ignored files are never considered in either direction because they were
        never in the commit — a `/revert` that wiped a build cache would turn a
        recovery into a rebuild.

        **What the run added is asked before the restore takes it away**, the git
        tier's ordering and for its reason. `--summary`'s `A` rows are the answer,
        and it is newline-separated with no `-z` mode — so a path containing a
        newline is *reported* wrongly here. jj is what does the restoring, so the
        cost is a miscounted line in a message rather than a file left behind.

        `FileNotFoundError` for a token that will not resolve, matching the git
        tier: it is what `/revert` catches to say "no longer available" rather than
        showing a person a traceback for a restore point it offered them.
        """
        code, out, err = await self._jj(workspace, "diff", "--summary", "--from", token)
        if code != 0:
            raise FileNotFoundError(
                f"restore point {token} is gone: {first_line(err) or f'jj exited {code}'}"
            )
        added = tuple(sorted(line[2:] for line in out.splitlines() if line.startswith("A ")))
        code, _, err = await self._jj(workspace, "restore", "--from", token)
        if code != 0:
            raise FileNotFoundError(
                f"could not restore {token}: {first_line(err) or f'jj exited {code}'}"
            )
        return added

    async def _jj(self, workspace: Workspace, *args: str) -> tuple[int, str, str]:
        """One jj call in this workspace, with the seam's materials held out of it."""
        return await jj(self.ctx, workspace.root, *auto_track(workspace.provisioned), *args)

    async def refs(self, base: Path) -> list[str]:
        """Every bookmark in this repository (`ArtifactProvider`).

        **This is the verb the user's question was about, and it was the one that
        failed silently.** `/workspaces` listed branches with `git branch --list`,
        and jj embeds its own git implementation — the whole acquire, bookmark and
        export flow was measured running against a `git` on `PATH` that exits 127 —
        so on a jj deployment without the binary that listing returned nothing and
        the command answered "no agent workspaces are left behind" with the
        bookmarks sitting right there.

        `if(remote, "", ...)` is what makes it a list of *bookmarks* rather than of
        bookmark-remote pairs, and it is load-bearing rather than tidy: a colocated
        repo has a `git` remote, and any bookmark that has moved since the last
        export — **which is every live child, because the bookmark tracks the working
        copy** — renders once for itself and once for `@git`. Without it a person
        sees each agent twice, as if the work were two.

        Deduplicating instead would also produce the right list and was the first
        shape this took. It went because it hid the filter: with both, removing the
        filter changed nothing any test could see, and the mechanism that was
        actually correct could not be told from the one that was covering for it.
        """
        code, out, _ = await self._in(base, "bookmark", "list", "--all-remotes", "-T", _BOOKMARKS)
        if code != 0:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str:
        """Delete a bookmark, refusing one that holds work nothing else has.

        **jj has no `-d`/`-D` distinction, so the refusal is built here** rather than
        borrowed: `jj bookmark delete` always succeeds. The question git answers with
        "not fully merged" is a revset — commits on this bookmark that are not in the
        ancestry of where the person is standing — and it is the same question,
        because by disposal everything the agent did is on that bookmark.
        """
        if not force:
            code, out, _ = await self._in(
                base, "log", "--no-graph", "-r", f"{ref} ~ ::@", "-T", "commit_id.short()"
            )
            if code == 0 and out.strip():
                return f"the bookmark {ref} holds work that is not in this workspace's history"
        code, _, err = await self._in(base, "bookmark", "delete", ref)
        return "" if code == 0 else (first_line(err) or f"jj exited {code}")

    async def merge(self, base: Path, ref: str) -> str:
        """`jj new @ <ref>` — the merge a jj user would type (`ArtifactProvider`).

        A commit with two parents, which is what a merge *is* in jj; there is no
        index and nothing to commit afterwards.

        **Success has to be verified rather than assumed**, and this is where the
        two tiers differ most. git exits non-zero on a conflict and refuses; jj exits
        **zero**, records the conflict inside the commit, and writes markers into the
        files. A caller reading the exit code would tell a person the merge was clean
        and let them find out by opening the file, so the conflict is asked for by
        revset and reported as what it is: a merge that happened and is not finished.
        """
        code, _, err = await self._in(base, "new", "@", ref)
        if code != 0:
            return f"could not merge {ref}: {first_line(err) or f'jj exited {code}'}"
        code, out, _ = await self._in(
            base, "log", "--no-graph", "-r", "@ & conflicts()", "-T", "commit_id.short()"
        )
        if code != 0 or not out.strip():
            return ""
        code, out, _ = await self._in(base, "resolve", "--list")
        paths = [line.split()[0] for line in out.splitlines() if line.strip()]
        return (
            f"merged {ref}, with conflicts jj recorded in the commit at: "
            f"{', '.join(paths) or 'some paths'} — resolve them and describe the merge"
        )

    async def strays(self, base: Path, *, with_status: bool = True) -> list[Stray]:
        """The workspaces this tier still has on disk (`EnumeratingProvider`).

        **`jj workspace list` with a template**, which is why this is a listing and
        not a parse: `name ++ "\t" ++ root` is two fields jj computes, where the
        human format is a sentence whose shape nobody promised to keep. The name is
        the one acquire wrote, so the ref comes back by the same construction that
        made it rather than by inverting a directory name.

        Filtered to this tier's own root, the git tier's rule: a workspace jj knows
        about that lives somewhere else is a person's own, not a tree pH made.

        **Asking whether one holds work also snapshots it**, and that is worth
        saying rather than hiding: a jj command commits the working copy before
        answering and the bookmark follows, so a `dirty` reported here is work that
        is now *on the ref*. `/workspaces list` on this tier is therefore mildly
        curative, where on the git tier it is purely a read.
        """
        code, out, _ = await self._in(base, "workspace", "list", "-T", _LISTING)
        if code != 0:
            return []
        found: list[tuple[str, Path]] = []
        for line in out.splitlines():
            name, tab, where = line.partition("\t")
            path = Path(where)
            if tab and is_under(path, self.root):
                found.append((f"{BRANCH_PREFIX}{name}", path))
        dirty = dict.fromkeys((ref for ref, _ in found), False)
        if with_status and found:

            async def measure(ref: str, path: Path) -> None:
                dirty[ref] = await self._has_work(path) is True

            async with anyio.create_task_group() as group:
                for ref, path in found:
                    group.start_soon(measure, ref, path)
        return [Stray(ref=ref, path=path, dirty=dirty[ref]) for ref, path in found]

    async def discard(self, path: Path) -> str:
        """Remove one workspace, leaving its bookmark alone (`EnumeratingProvider`).

        The name comes back off the path by the same construction `acquire` used to
        build it — `<root>/<session>/<agent>` and `<session>/<agent>` are the same
        two sanitised components — so nothing is inverted out of a lossy name. A
        path from anywhere but `strays` is refused rather than guessed at.

        jj deliberately leaves the directory after `forget`, so removing it is ours
        to do, exactly as it is at release.
        """
        try:
            name = path.relative_to(self.root).as_posix()
        except ValueError:
            return f"{path} is not a workspace this tier made"
        code, _, err = await jj(self.ctx, path, "workspace", "forget", name)
        if code != 0:
            return first_line(err) or f"jj exited {code}"
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(path, ignore_errors=True))
        return "" if not path.exists() else f"jj forgot {name} but {path} is still on disk"

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

        **Nothing is known to have been provisioned**, because a `WorkspaceRecord`
        does not carry it — so on this path a material the seam put in the tree
        counts as the agent's work and rides the bookmark. That is the git tier's
        deliberate trade at the same point, kept rather than re-decided: the crash
        path errs toward *keeping*, where the cost is a commit somebody can drop
        rather than work nobody can recover. The ordinary path excludes them, which
        is where the credential-on-a-branch case actually lives.
        """
        if record.ref is None:
            return False
        return await self._release(
            record.ref.removeprefix(BRANCH_PREFIX),
            record.root,
            record.ref,
            discard=discards_writes(record.kind) and not record.reason,
        )

    async def _add(self, managed: Path, base: Path, name: str, path: Path) -> str:
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
            # The fork point of a workspace that already exists, asked of the workspace
            # rather than remembered: a rehydrated child is a *new process* half the
            # time, and its release owes the same "what has this done since it forked"
            # the first one did.
            code, out, _ = await jj(self.ctx, path, *_STATE_PARENT)
            return out.strip() if code == 0 else ""
        # `jj workspace add` refuses a path whose parent is missing, where `git
        # worktree add` creates the chain — so the first agent of a session pays
        # one mkdir rather than a decline that reads as "jj is broken here".
        await anyio.to_thread.run_sync(lambda: path.parent.mkdir(parents=True, exist_ok=True))
        fork = await self._fork_point(base)
        # From `base`, not from `managed`: `@` is per workspace, so a fan-out from
        # a child forks that child's work, which is the whole point of the tier.
        add = ("workspace", "add", "--name", name, "-r", fork, str(path))
        code, _, err = await self._in(base, *add)
        registered = code != 0 and await self._registered(managed, name)
        if registered and not path.exists():
            await jj(self.ctx, managed, "workspace", "forget", name)
            code, _, err = await self._in(base, *add)
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
        return fork

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

    async def _in(self, cwd: Path, *args: str) -> tuple[int, str, str]:
        """One jj call in a directory that may be somebody's live workspace.

        **Every jj command snapshots the workspace it runs in**, which is the trap
        this exists to close and it closed it twice: `workspace root` — a *probe*,
        which reads nothing and changes nothing as far as its name goes — and
        `workspace add`, both of which run in a **parent's** tree and both of which
        tracked a provisioned material on the way past. Once tracked, no later
        `auto_track` can untrack it, so the exclusion at release was correct and
        already too late.

        So the rule is not "pass the config where it matters" but "there is one door
        into somebody else's workspace". A call that forgets it is a call that does
        not compile, rather than a leak nothing reports.
        """
        return await jj(self.ctx, cwd, *auto_track(self._base_materials(cwd)), *args)

    def _base_materials(self, base: Path) -> tuple[str, ...]:
        """What the seam provisioned into `base`, if `base` is itself a workspace.

        `jj new` snapshots the parent before freezing it, so a material sitting in
        the *parent's* tree would land in the commit every child forks from — and
        from there into each child's bookmark history, which is the same
        credential-on-a-branch this tier excludes at its own release.

        Asked of `live()` and matched by root, which is how `collectable` asks the
        same question: a `base` that is somebody's fresh workspace is in that list,
        and one that is not (the person's own checkout, which is `shared`) was never
        provisioned into at all — so the empty answer is the right one rather than a
        gap.
        """
        seam = self.ctx.get("workspace")
        if seam is None:
            return ()
        for workspace in seam.live():
            if workspace.root == base:
                return tuple(workspace.provisioned)
        return ()

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
        code, out, _ = await self._in(base, *_WORK)
        if code == 0 and out.strip():
            await self._in(base, "new")
        # **Resolved, not left as `@-`.** The workspace needs a revision and `@-` would
        # do; what needs the concrete commit is *release*, which has to ask "what has
        # this workspace done since it forked" and cannot ask it of a revset that means
        # something different in every workspace and moves under both of them.
        code, out, _ = await self._in(base, *_STATE_PARENT)
        return out.strip() or "@-"

    async def _registered(self, managed: Path, name: str) -> bool:
        """Whether jj still knows a workspace by this name.

        `jj workspace list` prints `<name>: <commit> …` a line at a time, so the
        test is the prefix. Structural, for `_why`'s reason.
        """
        code, out, _ = await jj(self.ctx, managed, "workspace", "list")
        return code == 0 and any(line.startswith(f"{name}: ") for line in out.splitlines())

    async def _release(
        self,
        name: str,
        path: Path,
        ref: str,
        *,
        discard: bool,
        provisioned: Sequence[str] = (),
        fork: str = "",
        repo: Path | None = None,
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
        held = auto_track(provisioned)
        kept = not discard and await self._has_work(path, held, fork) is not False
        if kept and not await self._name(path, ref, held):
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
            await jj(self.ctx, path, *held, "bookmark", "delete", ref)
        code, _, err = await jj(self.ctx, path, *held, "git", "export")
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: %s did not reach git (%s)",
                ref,
                first_line(err) or f"jj exited {code}",
            )
        code, _, err = await jj(self.ctx, path, *held, "workspace", "forget", name)
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

    async def _name(self, path: Path, ref: str, held: Sequence[str] = ()) -> bool:
        """Point the bookmark at what this workspace holds now. Reports whether it landed.

        **Release names the work rather than trusting the name acquire wrote.** The
        bookmark does follow its commit through every snapshot, so on the ordinary
        path this is confirming what is already true — and `kept` is a claim that a
        person can find this work, which is not a claim to make on the strength of a
        call made at a different time that may have failed.
        """
        code, _, err = await jj(self.ctx, path, *held, "bookmark", "set", ref, "-r", "@")
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: jj bookmark set failed in %s (%s)",
                path,
                first_line(err) or f"jj exited {code}",
            )
        return code == 0

    async def _has_work(self, path: Path, held: Sequence[str] = (), fork: str = "") -> bool | None:
        """Whether this workspace has done anything since it forked. `None` if jj cannot say.

        **Since it forked, not "is `@` non-empty"**, and the difference is a bug that
        threw work away. Freezing a parent's work for a child moves that parent's `@`
        to a fresh empty commit — its work is now in `@-` — so a parent that spawned
        anything looked, at its own release, exactly like a parent that had done
        nothing: the bookmark was deleted and the tree removed.

        An empty `fork` is the reclaim path, which has only a `WorkspaceRecord` and so
        cannot know where the workspace started. It falls back to the tip, which
        under-reports in exactly the case above — and under-reporting there means
        *keeping* a tree whose work is already on its bookmark, because reclaim errs
        toward keeping.
        """
        revset = f"{fork}..@ & ~empty()" if fork else "@ & ~empty()"
        code, out, _ = await jj(
            self.ctx, path, *held, "log", "--no-graph", "-r", revset, "-T", "commit_id.short()"
        )
        return None if code != 0 else bool(out.strip())

    async def _named(self, repo: Path, ref: str) -> bool:
        """Whether a bookmark by this name still points at something."""
        code, out, _ = await self._in(repo, "bookmark", "list", ref)
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
        code, out, _ = await self._in(base, "workspace", "root")
        if code != 0 or not out.strip():
            log.info(
                "ph.seams.workspace_jj: %s is not inside a jj workspace; declining so the "
                "seam falls back to a shared workspace",
                base,
            )
            return None
        root = Path(out.strip())
        if not await anyio.to_thread.run_sync(lambda: _colocated(root)):
            # Served anyway — the isolation is real and the bookmark is a real
            # handle. But said out loud once per repository, because the merge
            # story this tier advertises is a git one, and a person who cannot
            # `git show` what a child produced should hear why before disposal.
            log.warning(
                "ph.seams.workspace_jj: the repository behind %s is not colocated with git, "
                "so bookmarks this tier writes will not appear as git branches",
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
