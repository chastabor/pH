"""`workspace-jj`: the `worktree` tier over Jujutsu, where a child sees work in progress.

This is the third provider at the `worktree` tier, after `workspace-git` and
`workspace-agentfs`. It exists for one property that the git tier cannot offer.
**The working copy of jj is a commit.** There is no uncommitted state, because
a `jj` command commits the tree it runs in before it does anything else. A
command can be told not to do that, and `_read` and `_commit` exist to make
that distinction.

This property closes the gap that P4-11 left open. A child under
`workspace-git` branches from the last *commit* of its parent, and a live
parent commits only at disposal. So for a whole session the branch tip of the
parent is the base commit, and every child starts from context that the parent
moved past hours ago. The actual work of the parent is uncommitted and
invisible. Under jj the same spawn starts from the working copy of the parent,
uncommitted work included. So a fan-out inherits the state that it was fanned
out *from*.

**A peer of `workspace-git`, not a rung above it.** `tier` is `worktree` and
the kinds are the worktree kinds, because the guarantee is identical. Writes
relative to the cwd and writes through a tool are bounded. A raw write to an
absolute path is not. Nothing here confines a process. `describe_tier` adds
the one thing that the stock text of the rung cannot know: what a child
inherits.

**The fork point is frozen, and `_fork_point` does only that.** To hand
`workspace add` the `@` of the parent looks right, but it is a trap. jj
rewrites the working-copy commit on every snapshot and **auto-rebases its
descendants**. So each later action of the parent would rebase the child and
leave the child's workspace *stale*. Every `jj` command in a stale workspace
refuses until someone runs `workspace update-stale`. That command would then
pull the newer work of the parent into a tree that the child is editing. This
was measured, not only reasoned about. So the work of the parent is frozen
into a commit that nothing will rewrite, and children fork from that commit.
In the terms of jj this is the shared spawn point: one fork point, every child
that spawns from it, and a parent that is free to continue its work.

**It never converts a repository.** `jj git init --colocate` writes `.jj/`
into the project of a user and changes what that user's own `git` commands
report. A harness that did so because a row was mounted would decide something
about the repository that the user did not decide. So this tier declines a
base that jj does not already manage. The deployment adopts jj. pH does not
adopt it for the deployment.

**Colocation keeps `git` working.** It is also the reason why this is a change
of provider and not a migration. A bookmark exports to a real `refs/heads`
entry. So `git show` resolves the ref that this tier returns, `/workspaces`
can enumerate under `BRANCH_PREFIX`, and a user merges with the git that they
already know.

**Restore points come from the operation log** (`CheckpointingProvider`).
Every jj command records an operation. A commit that an operation refers to is
not collected until that operation is collected. So `capture` names the
working-copy commit and has nothing to pin. The git tier must write a hidden
ref to keep a tree from garbage collection. The revert itself is
`jj restore --from`, and deliberately *not* `jj op restore`. The operation log
is the right mechanism for retention and the wrong unit for a restore. Its
unit is the whole repository, so to undo one cell of one child that way would
also take back the work of every sibling.

**The non-guarantee is the mirror image of the purpose of this tier.** A file
that is untracked in the base is already in the working-copy commit of the
base, because jj put it there. So children fork from it, and it is reachable
from their bookmarks. "A child starts from its parent's work in progress" and
"an uncommitted file in the base does not reach a child's branch" are the same
statement with opposite signs. The git tier is the one that keeps the second.
What pH *does* keep out is the material that it copied in itself
(`auto_track`). A provisioned `.env` is not the work of the agent. To put it
on a ref that somebody merges is the finding of P6-35 and not a bookkeeping
matter.

@module ph.seams.workspace_jj
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
from ..paths import default_home_path, is_under
from ..wire import WireModel
from .containment import TIERS, TierDescription
from .diagnostics import Diagnostic, contribute
from .subprocess import SubprocessSpawnSpec, first_line
from .workspace import (
    BRANCH_PREFIX,
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
    measure_strays,
    redirection_env,
)

__all__ = ["JjWorkspaceProvider", "apply", "auto_track", "jj"]

log = logging.getLogger("ph.seams.workspace_jj")

_CONFLICTS = 'self.conflicted_files().map(|e| e.path() ++ "\\n").join("")'
"""Which paths in a commit are conflicted, as jj computes them.

**This is a template and not a parse.** The docstring of `strays` states the
same rule. `jj resolve --list` prints a padded table of two columns, and the
human format is a sentence whose shape nobody promised to keep. A parser of
that table cut a path with a space at the first column boundary (the fourth
site of J5). The obvious repair is to split on a run of whitespace. That fails
on the *longest* path, which jj pads with exactly one space.
`conflicted_files()` returns the paths themselves, one per line, in the same
single spawn.

It also answers *whether* there is a conflict: a clean revision prints nothing
and exits 0. The old `@ & conflicts()` revset needed a second spawn for that
half of the answer."""


async def jj(
    ctx: Context, cwd: Path, *args: str, env: Mapping[str, str] | None = None
) -> tuple[int, str, str]:
    """One `jj` invocation, through `ctx.subprocess`. The twin of `workspace_git.git`.

    It goes through the seam for the reasons that function gives, not for
    tidiness. The seam scrubs the credential-shaped environment (F1) and reaps
    the child in a `finally` (F4). A bare `subprocess.run` here would opt this
    module out of both for no gain.

    `LC_ALL=C` is set for the same reason as there: pH reads what the tool
    says. The seam drops the inherited git location (`subprocess.LOCATION`).
    That matters here too. jj reads and writes the git repository of a
    colocated workspace, so a `GIT_DIR` that pH was started with would redirect
    that backend in the same way.

    **`cwd` is never incidental.** jj commits the workspace that the command
    runs in and no other. So the directory of a call decides whose work the
    command sees. Testing found this when a child tree was lost: a bookmark
    that was set from the directory of the parent named the *previous* commit
    of the child, and the work of the child went nowhere.

    Callers use `JjWorkspaceProvider._read` or `._commit` and not this
    function, because every call site must state explicitly whether the call
    commits that tree.
    """
    spec = SubprocessSpawnSpec(
        argv=("jj", *args),
        cwd=cwd,
        env=ctx.require(SUBPROCESS).env(extra={"LC_ALL": "C", **(env or {})}),
    )
    outcome = await ctx.require(SUBPROCESS).run(spec)
    return outcome.exit_code, outcome.stdout, outcome.stderr


_BOOKMARKS = 'if(remote, "", name ++ "\n")'
"""The local names of `jj bookmark list`, one per line.

This is a template and not the human `name: commit description` line. It
filters on `remote` because the unit of the listing is a bookmark-*remote*
pair. A colocated repository has a `git` remote, so a bookmark that has moved
since the last export renders two times under the same name.
"""

_LISTING = 'name ++ "\t" ++ root ++ "\n"'
"""The two machine-readable fields of `jj workspace list`: the name and the root.

This is a template and not the human format. The human format is a sentence,
`name: path commit bookmark | description`. Nobody promised to keep it stable,
and a reader would have to split it on a space that a path may contain.
"""


_NEEDS_THE_TREE = (("new",), ("restore",), ("diff",), ("workspace", "add"))
"""jj verbs whose answer or effect *is* the working copy, so reading them is a bug.

**This was measured, and it corrects the first argument for this split.** The
claim was that jj refuses `--ignore-working-copy`, *by name*, on any command
that must write, so a wrong choice would fail loudly. jj does that for
`workspace add` alone. Even there it refuses only *after* it registers the
workspace, and it leaves an empty directory. `new`, `restore` and `diff` all
**exit 0 and answer from a stale tree**. `jj --ignore-working-copy new`
silently produced a fork point with none of the work of the parent in it. That
work is the one property that this provider exists to offer.

So pH must make the failure loud. The argv cannot *derive* the choice: `log`
and `bookmark set` are each correct on both sides, and the revset or the tree
that the call runs in decides. But the argv can **veto** the combination that
is never right, and `_read` asserts that. The check has one direction and sits
at the runner. Otherwise each of 25 call sites would have to remember it.
"""


def _needs_the_tree(args: Sequence[str]) -> bool:
    """Whether this argv is one `_read` must refuse."""
    return any(tuple(args[: len(verb)]) == verb for verb in _NEEDS_THE_TREE)


def auto_track(provisioned: Sequence[str]) -> tuple[str, ...]:
    """The `--config` that keeps the seam's own materials out of the agent's commit (E14).

    **jj commits everything that the project does not gitignore, and
    provisioning puts non-gitignored things in a workspace.** A `.env` that was
    copied in so that the tests can run is not the work of the agent. Here it
    lands in the working-copy commit by default. That puts it on the bookmark,
    and so in the git ref that somebody merges. This is the finding of P6-35 at
    this tier. Here it is no longer a bookkeeping matter: it is a credential on
    a branch. It is also silent, because every command still succeeds.

    `snapshot.auto-track` is a **fileset**. So this is the exact analogue of
    the `agent_work_pathspec()` that the git tier uses: everything, minus what
    the seam put there.

    The value is passed per call and is not written to the config of the
    repository. The list is a property of one *workspace*, and all workspaces
    share a repository config.

    **Precondition: only a call that commits needs this.** It governs what
    *becomes* tracked, so it works only while the file has never been
    committed. That holds because every committing call that pH makes goes
    through `_commit`, and `workspace add` runs before provisioning does.

    A path that holds a `"` is skipped and not escaped. The string is parsed as
    a fileset, and a quote inside a fileset changes what it selects. A wrong
    guess would track something that pH meant to exclude, and nothing would
    report it. Nothing that pH provisions has such a name, and the skip is
    logged.
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
    # TOML on the outside, fileset on the inside. The value that jj parses is a
    # string, and a bare `all() ~ (...)` is not a string.
    return ("--config", f"snapshot.auto-track='all() ~ ({paths})'")


def _colocated(root: Path) -> bool:
    """Whether the repository behind this workspace has a git repo beside it.

    **Not `(root / ".git").exists()`.** That was the earlier check, and it is
    wrong for every workspace but the first. A secondary workspace holds no
    `.git`, so a fan-out *from a child* warned that a correctly colocated
    repository was not colocated. jj records the way back in `.jj/repo`. That
    is a directory in the main workspace, and a file that names that directory
    everywhere else. So this function asks the question of the repository in
    the way that jj itself asks it.
    """
    pointer = root / ".jj" / "repo"
    if pointer.is_dir():
        return (root / ".git").exists()
    try:
        # **The path is relative to `.jj`, and jj writes it that way.**
        # `../../../../repo/.jj/repo` resolved against the cwd of the process is
        # a different repository or none. That is how an earlier version of
        # this function reported a colocated repository as not colocated.
        named = pointer.parent / pointer.read_text(encoding="utf-8").strip()
        return (named.resolve().parent.parent / ".git").exists()
    except OSError:
        return False


@dataclass(slots=True)
class JjWorkspaceProvider:
    """The `worktree` tier over jj: one workspace per agent, forked from the parent."""

    ctx: Context
    root: Path
    """Where workspaces are created: `$PH_HOME/jj/<session>/<agent>`. It is outside
    the repository for the reason that `workspace_git` gives. The agent's own
    `glob` walks a checkout that is inside `base`, and every child nests it one
    level deeper."""
    tier: ContainmentTier = field(default="worktree", init=False)
    vcs: Backend = field(default="jj", init=False)
    """This tier is jj-backed. See the same field in `workspace_git`, and
    `VersionedProvider` in `ph.seams.workspace`. `Workspace.kind` is `worktree`
    for both tiers, so only the provider can say which one backs the tree."""
    _roots: dict[Path, Path | None] = field(default_factory=dict, init=False)
    """`base` → the root of the jj workspace that contains it, asked one time.

    Every sibling in a fan-out receives the *same* `base`. Without this cache
    the probe would run one time per child and return the same answer. A `None`
    is cached too: a directory that jj does not manage does not become managed
    while pH runs.
    """

    def describe_tier(self) -> TierDescription:
        """The stock description of the rung, plus the one thing that it cannot know.

        Two columns are copied from the worktree tier word for word, because
        the bounding is the same mechanism. The copy is the measured result
        and not a convenience. `buys` keeps every word of the stock text. The
        checkpoints and the `/revert` that it names are **real here**, through
        the operation log and not through a hidden git ref. `buys` then adds
        what a child inherits, which is the only reason to run this tier and
        not the git tier.

        The text said "no per-run checkpoints and no /revert" while that was
        true, which is the direction that E1 is usually about. To leave it
        after the capability arrived would have been the same failure in the
        other direction. A user who checked whether `/revert` works here would
        have read "no" from the tool while the mechanism was in place.
        """
        return TierDescription(
            bounds=TIERS["worktree"].bounds,
            does_not_bound=TIERS["worktree"].does_not_bound,
            buys=(
                f"{TIERS['worktree'].buys}, and a child that starts from its parent's "
                "work in progress rather than from the parent's last commit"
            ),
        )

    def locate(self, *, session_id: str, agent_id: str, access: WorkspaceAccess) -> WorkspaceSite:
        """Where the workspace of this agent goes, and its bookmark (`LocatingProvider`).

        **One name has two uses**, so that neither can be derived wrongly from
        the other. The name of the workspace is the bookmark without its
        prefix. `reclaim` recovers the name from the ref, because nothing else
        is left after a crash. jj accepts `/` in a workspace name and resolves
        `<name>@` as the working copy of that workspace, so the two shapes can
        match. `acquire` derives its own site from here, so the
        `workspace/acquiring` event that is written first names this path.
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
        """A workspace for this agent, forked from the work in progress of `base`.

        The bookmark is set **here, at acquire**, and not at release. This is
        not bookkeeping that runs early. A bookmark follows the commit that it
        points at through every rewrite, and a jj working copy is rewritten on
        every snapshot. So one `bookmark set` now leaves a name that tracks the
        work of this child for the rest of its life. That makes `ref` on the
        `workspace/acquired` event as accurate as the branch of the git tier,
        which exists from `worktree add` onward.

        It declines by raising, as the other tiers do. So the reason reaches
        `workspace/acquired` and `phern doctor`, and does not become an
        unexplained `shared` (E15).
        """
        managed = await self._managed(base)
        if managed is None:
            raise WorkspaceDeclined(
                "not-a-repository", f"{base} is not inside a workspace jj manages"
            )
        site = self.locate(session_id=session_id, agent_id=agent_id, access=access)
        path, ref = site.root, site.ref
        assert ref is not None, "a jj workspace is always bookmarked"
        name = ref.removeprefix(BRANCH_PREFIX)
        fork, reused = await self._add(managed, base, name, path)

        # Through `_read`, like every other call that runs in a directory that is
        # not ours. For a fan-out *from a child*, `managed` is that child's own
        # workspace.
        code, _, err = await self._read(managed, "bookmark", "set", ref, "-r", f"{name}@")
        if code != 0:
            # Not fatal, because release names the work again before it exports
            # the bookmark, and release decides whether `kept` is true. What is
            # lost is the *live* view: `ref` on `workspace/acquired` would name
            # something that does not exist yet, and nobody could see which
            # bookmark belongs to which child while the children run. This is
            # worth a log line and not a decline.
            log.warning(
                "ph.seams.workspace_jj: %s has no bookmark yet (%s); its work is still "
                "named at disposal, but nothing shows it until then",
                path,
                first_line(err) or f"jj exited {code}",
            )

        return Workspace(
            root=path,
            scratch=scratch,
            kind=site.kind,
            reused=reused,
            # True for both kinds, deliberately, and for the reason that the git
            # tier gives: an ephemeral child writes freely and its writes reach
            # nobody. `False` would be a claim of confinement that only a
            # sandbox can make.
            repo_writable=True,
            ref=ref,
            fork=fork,
            env=redirection_env(scratch),
            # `workspace.retained` is read at teardown and not captured here.
            # How the agent *ended* decides whether this tree is evidence, and
            # nobody knows that at acquire time (P6-28).
            release=lambda workspace: self._release(
                name,
                path,
                ref,
                discard=discards_writes(workspace.kind) and not workspace.retained,
                provisioned=workspace.provisioned,
                fork=fork,
                repo=managed,
            ),
        )

    async def capture(self, workspace: Workspace) -> str | None:
        """The commit of this workspace's working copy: its state, named (P4-09).

        `CheckpointingProvider`. This is one command, and it is a snapshot as
        much as a question. jj commits the working copy before it answers
        anything, so the id that comes back *is* the state and nothing is left
        to pin.

        **The operation log keeps it reachable.** This replaces the hidden ref
        that the git tier must write. Every jj command records an operation,
        and a commit that an operation refers to is not collected until that
        operation is collected. So a restore point survives by the same
        mechanism that lets a user undo anything that jj did. It does not
        survive by a ref that pH must remember to write and then must remember
        to prune.

        **Equal tokens mean equal content.** A token must promise that and
        nothing more. The converse holds for a git tree and does not hold
        here. A jj commit id covers the committer timestamp as well as the
        tree, so work that returns to an earlier state gets a *new* id. The
        cost is that a quality gate runs again against content that it has
        already judged. That is the safe direction. It is also the reason why
        `unchanged_failure` asks whether a gate already failed against this
        exact state, and is not a cache to be trusted.
        """
        found = await self._log(
            workspace.root, "@", full=True, snapshot=True, provisioned=workspace.provisioned
        )
        if found is None:
            log.warning("ph.seams.workspace_jj: no restore point for %s", workspace.root)
            return None
        return found or None

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]:
        """Put the working copy back to `token`. Returns the paths the run had added.

        This is `jj restore --from <commit>`, and deliberately **not**
        `jj op restore`. The operation log keeps the token alive, but its unit
        is the whole repository. To undo a run that way would also take back
        the work of every sibling workspace. `restore` rewrites only the
        working-copy commit of this workspace, which is the unit of a per-run
        revert.

        Ignored files are never considered in either direction, because they
        were never in the commit. A `/revert` that wiped a build cache would
        turn a recovery into a rebuild.

        **This asks what the run added before the restore removes it.** The
        git tier uses the same order for the same reason. The `A` rows of
        `--summary` are the answer. That output is separated by newlines and
        has no `-z` mode, so a path that contains a newline is *reported*
        wrongly here. jj does the restoring, so the cost is a miscounted line
        in a message and not a file that is left behind.

        It raises `FileNotFoundError` for a token that does not resolve, as
        the git tier does. `/revert` catches that error and says "no longer
        available". Otherwise it would show a user a traceback for a restore
        point that it offered to them.
        """
        code, out, err = await self._commit(
            workspace.root,
            "diff",
            "--summary",
            "--from",
            token,
            provisioned=workspace.provisioned,
        )
        if code != 0:
            raise FileNotFoundError(
                f"restore point {token} is gone: {first_line(err) or f'jj exited {code}'}"
            )
        added = tuple(sorted(line[2:] for line in out.splitlines() if line.startswith("A ")))
        code, _, err = await self._commit(
            workspace.root, "restore", "--from", token, provisioned=workspace.provisioned
        )
        if code != 0:
            raise FileNotFoundError(
                f"could not restore {token}: {first_line(err) or f'jj exited {code}'}"
            )
        return added

    async def refs(self, base: Path) -> list[str]:
        """Every bookmark in this repository (`ArtifactProvider`).

        **The user asked about this verb, and it was the one that failed
        silently.** `/workspaces` listed branches with `git branch --list`.
        But jj embeds its own git implementation: the whole acquire, bookmark
        and export flow was measured with a `git` on `PATH` that exits 127. So
        on a jj deployment without the binary that listing returned nothing.
        The command answered "no agent workspaces are left behind" while the
        bookmarks existed.

        The `if(remote, "", ...)` in the template makes this a list of
        *bookmarks* and not of bookmark-remote pairs. The result depends on
        it, so it is not there for tidiness. A colocated repository has a
        `git` remote. Any bookmark that has moved since the last export
        renders one time for itself and one time for `@git`. **That is every
        live child, because the bookmark tracks the working copy.** Without
        the filter a user sees each agent two times, as if the work were two
        pieces of work.

        To deduplicate would also produce the right list, and the first
        version did that. It was removed because it hid the filter. With both
        in place, no test could see a change when the filter was removed. The
        mechanism that was correct could not be told from the one that covered
        for it.
        """
        code, out, _ = await self._read(base, "bookmark", "list", "--all-remotes", "-T", _BOOKMARKS)
        if code != 0:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str:
        """Delete a bookmark, but refuse one that holds work nothing else has.

        **jj has no `-d`/`-D` distinction, so the refusal is built here** and
        not borrowed: `jj bookmark delete` always succeeds. The question that
        git answers with "not fully merged" is a revset here: the commits on
        this bookmark that are not in the ancestry of the working copy that
        the user is in. It is the same question, because by disposal
        everything that the agent did is on that bookmark.
        """
        if not force and await self._log(base, f"{ref} ~ ::@", snapshot=False):
            return f"the bookmark {ref} holds work that is not in this workspace's history"
        code, _, err = await self._read(base, "bookmark", "delete", ref)
        return "" if code == 0 else (first_line(err) or f"jj exited {code}")

    async def merge(self, base: Path, ref: str) -> str:
        """`jj new @ <ref>`: the merge that a jj user would type (`ArtifactProvider`).

        The result is a commit with two parents, which is what a merge *is* in
        jj. There is no index and nothing to commit afterwards.

        **Success must be verified and not assumed.** This is where the git
        tier and this tier differ most. git exits non-zero on a conflict and
        refuses. jj exits **zero**, records the conflict inside the commit, and
        writes markers into the files. A caller that read the exit code would
        tell a user that the merge was clean, and the user would find out by
        opening the file. So this method asks jj for the conflicted paths with
        a template (`_CONFLICTS`) and reports the result as what it is: a merge
        that happened and is not finished.
        """
        code, _, err = await self._commit(base, "new", "@", ref)
        if code != 0:
            return f"could not merge {ref}: {first_line(err) or f'jj exited {code}'}"
        # One spawn gives both halves of the answer. See `_CONFLICTS`.
        code, out, _ = await self._read(base, "log", "-r", "@", "--no-graph", "-T", _CONFLICTS)
        if code != 0 or not out.strip():
            return ""
        paths = [line for line in out.splitlines() if line]
        return (
            f"merged {ref}, with conflicts jj recorded in the commit at: "
            f"{', '.join(paths) or 'some paths'} — resolve them and describe the merge"
        )

    async def strays(
        self, base: Path, *, with_status: bool = True, skip: Container[str] = ()
    ) -> list[Stray]:
        """The workspaces that this tier still has on disk (`EnumeratingProvider`).

        **`jj workspace list` with a template**, so this is a listing and not a
        parse. `name ++ "\t" ++ root` is two fields that jj computes. The human
        format is a sentence whose shape nobody promised to keep. The name is
        the one that `acquire` wrote. So the ref is built in the same way that
        it was first made, and not by inverting a directory name.

        The list is filtered to this tier's own root, which is the rule of the
        git tier. A workspace that jj knows about and that is somewhere else
        belongs to a user. pH did not make that tree.

        **To ask whether a workspace holds work also snapshots it.** This is
        stated here and not hidden. A jj command commits the working copy
        before it answers, and the bookmark follows. So a `dirty` that this
        method reports is work that is now *on the ref*. On the git tier
        `/workspaces list` only reads. On this tier it also puts that work on
        the ref.
        """
        code, out, _ = await self._read(base, "workspace", "list", "-T", _LISTING)
        if code != 0:
            return []
        found: list[tuple[str, Path]] = []
        for line in out.splitlines():
            name, tab, where = line.partition("\t")
            path = Path(where)
            if tab and is_under(path, self.root):
                found.append((f"{BRANCH_PREFIX}{name}", path))

        async def probe(path: Path) -> bool:
            # `is True` because `_has_work` returns `None` when jj could not say.
            # Here that means an unreadable tree and not a dirty one.
            return await self._has_work(path) is True

        return await measure_strays(found, probe, with_status=with_status, skip=skip)

    async def discard(self, path: Path, *, provisioned: Sequence[str] | None = None) -> str:
        """Remove one workspace and leave its bookmark alone (`EnumeratingProvider`).

        The name is built from the path in the same way that `acquire` built
        the path from the name. `<root>/<session>/<agent>` and
        `<session>/<agent>` are the same two sanitized components, so nothing
        is inverted out of a lossy name. A path that is not under the root of
        this tier, which is where `strays` finds its paths, is refused and not
        guessed at.

        jj deliberately leaves the directory after `forget`. So this method
        removes it, and `_release` calls this method for the same two steps.
        """
        try:
            name = path.relative_to(self.root).as_posix()
        except ValueError:
            return f"{path} is not a workspace this tier made"
        code, _, err = await self._commit(
            path, "workspace", "forget", name, provisioned=provisioned
        )
        if code != 0:
            return first_line(err) or f"jj exited {code}"
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(path, ignore_errors=True))
        return "" if not path.exists() else f"jj forgot {name} but {path} is still on disk"

    async def export(self, record: WorkspaceRecord) -> str:
        """The bookmark that has tracked the work of this agent from the start.

        There is nothing to build, as on the git tier. The bookmark moved with
        the working copy of the child through every snapshot, and colocation
        exported it. So the name that is returned is a `refs/heads` entry that
        `git show` resolves. The method exists so that `/workspaces` asks the
        seam one question and does not ask which tier answered.
        """
        if record.ref is None:
            raise WorkspaceDeclined(
                "provider-failed",
                f"{record.agent_id} has no bookmark; there is nothing to export",
            )
        return record.ref

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        """Release a workspace that this process never acquired (F6).

        This runs the same disposal as an orderly release, for the reason that
        the git tier gives. A crash is not grounds to throw work away. A
        reconciliation that discarded more than a clean exit would make a crash
        worse than the leak that it repairs.

        **The record locates its own repository**, and here that costs nothing.
        Every jj command that a release needs runs in the child's own
        directory, which the record names. So there is no toplevel to guess
        at, and no cwd of a reconciling process to mistake for one.

        The workspace name is taken from `ref`, which is why `acquire` spends a
        line to make the two the same string. Nothing else that could carry
        the name survives a crash.

        **A retention is an exception to `discard`, exactly as it is at
        release** (P6-28), and not a second rule here. With two rules for one
        word, a tree is deleted by whichever path reaches it first.

        **What was provisioned is read from the record** (S13), as on the git
        tier. `workspace/acquired` names it, and every command below runs with
        it untracked. So a provisioned `.env` stays off the bookmark on the one
        path that nobody watches.
        """
        if record.ref is None:
            return False
        return await self._release(
            record.ref.removeprefix(BRANCH_PREFIX),
            record.root,
            record.ref,
            discard=discards_writes(record.kind) and not record.reason,
            provisioned=record.provisioned,
            fork=record.fork,
        )

    async def _add(self, managed: Path, base: Path, name: str, path: Path) -> tuple[str, bool]:
        """`jj workspace add`, tolerating the two states that a resume can find. Returns
        the fork point, and whether the workspace was already there
        (`Workspace.reused`).

        A workspace that is already at this path is **reused**. This is the
        rule of the git tier, for the same reason. The agent id is the key, so
        a workspace found here is this agent's own tree, and to recreate it
        would discard the work that disposal exists to name. A registration
        whose directory is gone is what a crash leaves. It is forgotten and the
        add is retried. That is the one recoverable failure.

        **The recovery runs only after a failure**, so the common case stays
        one subprocess. A first acquire succeeds. A probe of the workspace list
        up front would have spent a call on every acquire and learned nothing.
        """
        if (path / ".jj").exists():
            log.info("ph.seams.workspace_jj: reusing the jj workspace already at %s", path)
            # The fork point of a workspace that already exists is asked of the
            # workspace and not remembered. A rehydrated child is a *new process*
            # half the time, and its release must answer the same question as
            # the first one: what has this workspace done since it forked.
            return await self._log(path, "@-", full=True, snapshot=False) or "", True
        # `jj workspace add` refuses a path whose parent is missing, where `git
        # worktree add` creates the chain. So the first agent of a session pays
        # one mkdir, and does not get a decline that reads as "jj is broken here".
        await anyio.to_thread.run_sync(lambda: path.parent.mkdir(parents=True, exist_ok=True))
        fork = await self._fork_point(base)
        # From `base`, not from `managed`. `@` is per workspace, so a fan-out
        # from a child forks the work of that child, which is the purpose of the
        # tier.
        add = ("workspace", "add", "--name", name, "-r", fork, str(path))
        code, _, err = await self._commit(base, *add)
        registered = code != 0 and await self._registered(managed, name)
        if registered and not path.exists():
            await self._read(managed, "workspace", "forget", name)
            code, _, err = await self._commit(base, *add)
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
        return fork, False

    def _why(self, registered: bool, path: Path) -> DeclineReason:
        """Which decline this was, from the *state* of jj and not from its prose.

        The two facts are already available. Reading them keeps this module out
        of the trap that `workspace_git._why` documents: a match on a message
        would be a guess about wording that nobody promised to keep.

        A live workspace that holds this name is `branch-in-use`. That is the
        nearest term in the shared vocabulary to "another checkout has claimed
        this", which is exactly what it means on the git tier.
        """
        if registered:
            return "branch-in-use"
        return "path-exists" if path.exists() else "provider-failed"

    async def _read(self, cwd: Path, *args: str) -> tuple[int, str, str]:
        """A jj call that must not touch the tree it runs in.

        It passes `--ignore-working-copy` and no fileset: nothing can be
        adopted into a commit that is not being made. This is *stronger* than
        the `auto_track` exclusion that `_commit` needs. That exclusion keeps a
        material from becoming tracked. This flag keeps the command from
        reading the tree at all.

        **It closed a hole that no fileset could close.** `_managed` runs
        `workspace root` in the *user's own* checkout, where `_base_materials`
        correctly returns `()` because nothing was provisioned there. So a
        probe whose name says that it reads was committing the working copy of
        the user and adopting whatever they had left untracked. No exclusion
        could have fixed that, because the material belonged to the user.

        There are two methods and not one `snapshot=` flag, so that the pairing
        that means nothing, a read *plus* a fileset, cannot be written. Before,
        it could be written, and it was silently ignored. The change that
        introduced it had already paid for that two times by hand, when it
        deleted a `provisioned=` that had quietly stopped applying.
        """
        assert not _needs_the_tree(args), f"jj {' '.join(args[:2])} cannot answer from a stale tree"
        return await jj(self.ctx, cwd, "--ignore-working-copy", *args)

    async def _commit(
        self, cwd: Path, *args: str, provisioned: Sequence[str] | None = None
    ) -> tuple[int, str, str]:
        """A jj call whose answer or effect is the tree, so it commits the tree first.

        These are the only calls that let the snapshot happen. So they are the
        only calls that need `auto_track` to keep the seam's own materials out
        of the commit that they cause.

        `provisioned=None` means "look it up". That is right whenever the seam
        still holds the workspace. Disposal is the case that must pass the
        list: `_release` runs *after* the seam has dropped the workspace, so a
        lookup would return nothing exactly when the materials still need to be
        held out.
        """
        materials = self._base_materials(cwd) if provisioned is None else provisioned
        return await jj(self.ctx, cwd, *auto_track(materials), *args)

    async def snapshotting(self, cwd: Path, *args: str) -> tuple[int, str, str]:
        """`SnapshottingProvider`: a call that may commit this tree, fileset applied.

        This is `_commit` under a name that a caller outside this module can
        use. The change filter (`ph.seams.changes`) needs the snapshot, because
        the snapshot makes one `jj diff` cover uncommitted work. The filter
        needed a way to get it that does not bypass `auto_track`, which is all
        of E14 at this tier. The method is public because the capability is
        public. It is thin because the decision that it carries was already
        made one time, in `_commit`.
        """
        return await self._commit(cwd, *args)

    async def _log(
        self,
        cwd: Path,
        revset: str,
        *,
        snapshot: bool,
        full: bool = False,
        provisioned: Sequence[str] | None = None,
    ) -> str | None:
        """One revset, answered as a commit id. `None` when jj could not say.

        The `log --no-graph -r <revset> -T commit_id` shape was written six
        times, as three module constants and three inline calls. They differed
        only in the revset and in the short or full id. `full` is for a value
        that goes into a **log** and that a later process reads back. The short
        form is for a presence test.

        This is the one shape that correctly belongs on both sides. So it keeps
        a boolean where every other call picks a method: a revset can ask about
        the tree or about the repository. `@ & ~empty()` means nothing until
        the working copy is committed. `@-` and the ancestry of a bookmark are
        answers that the tree cannot change.
        """
        args = (
            "log",
            "--no-graph",
            "-r",
            revset,
            "-T",
            "commit_id" if full else "commit_id.short()",
        )
        code, out, _ = await (
            self._commit(cwd, *args, provisioned=provisioned)
            if snapshot
            else self._read(cwd, *args)
        )
        return None if code != 0 else out.strip()

    def _base_materials(self, base: Path) -> tuple[str, ...]:
        """What the seam provisioned into `base`, if `base` is itself a workspace.

        `jj new` snapshots the parent before it freezes it. So a material in
        the tree of the *parent* would land in the commit that every child
        forks from, and from there in the bookmark history of each child. That
        is the same credential on a branch that this tier excludes at its own
        release.

        The question is asked of `live()` and matched by root, which is how
        `collectable` asks the same question. A `base` that is the fresh
        workspace of some agent is in that list. A `base` that is not in the
        list (the user's own checkout, which is `shared`) was never provisioned
        into at all. So the empty answer is correct and is not a gap.
        """
        seam = self.ctx.get(WORKSPACE)
        if seam is None:
            return ()
        for workspace in seam.live():
            if workspace.root == base:
                return tuple(workspace.provisioned)
        return ()

    async def _fork_point(self, base: Path) -> str:
        """Freeze the parent's work into a commit that nothing will rewrite, and name it.

        `jj new` does the freezing, and it is the ordinary thing that a user
        does before they branch off. The working-copy commit of the parent
        stops being the working copy, so snapshots stop rewriting it, and a
        fresh empty commit takes its place. **The files of the parent are
        untouched.** The tree on disk is identical either way, which makes this
        safe to do in a user's own checkout.

        It is skipped when `@` is already empty. That is the fan-out case and
        not an optimization. The first child of a spawn freezes the work of the
        parent. Its siblings find nothing new to freeze and fork from the
        *same* commit. Without the guard, eight children would leave eight
        empty commits stacked in the history of the parent, and each child
        would fork from a different one.
        """
        if await self._log(base, "@ & ~empty()", snapshot=True):
            await self._commit(base, "new")
        # **Resolved, not left as `@-`.** The workspace needs a revision, and `@-`
        # would do. *Release* is what needs the concrete commit. It must ask
        # "what has this workspace done since it forked", and it cannot ask that
        # of a revset that means something different in every workspace and that
        # moves under both of them.
        return await self._log(base, "@-", full=True, snapshot=False) or "@-"

    async def _registered(self, managed: Path, name: str) -> bool:
        """Whether jj still knows a workspace by this name.

        It goes through `_LISTING` and `_read`, like `strays`. An earlier
        version read the human `<name>: <commit> …` line, which the docstring
        of that template calls a sentence that nobody promised to keep. It also
        ran bare `jj` in a directory that may be the workspace of somebody
        else.
        """
        code, out, _ = await self._read(managed, "workspace", "list", "-T", _LISTING)
        return code == 0 and any(line.partition("\t")[0] == name for line in out.splitlines())

    async def _release(
        self,
        name: str,
        path: Path,
        ref: str,
        *,
        discard: bool,
        provisioned: Sequence[str] = (),
        fork: str | None = None,
        repo: Path | None = None,
    ) -> bool:
        """Snapshot, name, export, forget, remove. Report whether anything was **kept**.

        **The checkout always goes, and the bookmark survives.** This is the
        policy of the git tier, reached by different means. There is no commit
        step, because nothing is uncommitted. The first call snapshots the
        working copy into its own commit. The bookmark has followed that commit
        since acquire, so it comes with it. So one command both captures the
        work of the child and answers whether the child did any.

        **Every jj call runs in the child's own directory.** That lets this
        method serve `reclaim` unchanged. `git export` and `workspace forget`
        both reach the shared repository from a secondary workspace, so nothing
        here needs to know where the repository is.

        **The export is explicit and not incidental.** A bookmark becomes a
        `refs/heads` entry when a jj command touches the colocated repository.
        `workspace forget` would usually do that as a side effect, but
        "usually" is not a guarantee. `kept` is a claim that a user can find
        this work with the git that they already use.

        Every path reports what happened and not what was intended. That keeps
        the field a record and not a statement of policy.
        """
        if not path.exists():
            # Disposed already. Reconciliation and the scope that acquired the
            # tree can both reach this. The second arrival decides nothing. It
            # reports what the first left, which is the bookmark.
            return repo is not None and await self._named(repo, ref)
        # `None` means "could not tell". It counts as work for the reason that the
        # git tier gives: to keep a tree that nobody wanted costs disk, and to
        # discard one that held work costs the work.
        kept = not discard and await self._has_work(path, fork, provisioned) is not False
        if kept and not await self._name(path, ref, provisioned):
            # **A failure to name cancels the removal, not the work.** The git
            # tier has the same rule at the same point. An orphaned directory is
            # worse than a clean disposal and far better than the deletion of a
            # tree whose work never reached a ref.
            log.warning(
                "ph.seams.workspace_jj: could not name %s in %s — keeping the tree, because "
                "the only thing worse than an orphaned workspace is deleting one whose work "
                "reached nothing",
                ref,
                path,
            )
            return True
        if not kept:
            await self._read(path, "bookmark", "delete", ref)
        code, _, err = await self._read(path, "git", "export")
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: %s did not reach git (%s)",
                ref,
                first_line(err) or f"jj exited {code}",
            )
        # These are the same two steps that `discard` runs. An earlier version
        # said so only in a comment, and now the code calls `discard`: forget the
        # registration, then remove the directory that jj deliberately leaves
        # behind. To leave it is right for a user who switches workspaces. It is
        # wrong for a harness that promised one workspace per agent and must not
        # accumulate them.
        refused = await self.discard(path, provisioned=provisioned)
        if refused:
            log.warning("ph.seams.workspace_jj: could not take back %s (%s)", path, refused)
        return kept

    async def _name(self, path: Path, ref: str, provisioned: Sequence[str]) -> bool:
        """Point the bookmark at what this workspace holds now. Reports whether it landed.

        **Release names the work and does not trust the name that acquire
        wrote.** The bookmark does follow its commit through every snapshot, so
        on the ordinary path this confirms what is already true. But `kept` is
        a claim that a user can find this work. A call that was made at a
        different time, and that may have failed, is not enough to support that
        claim.

        The call goes through `_commit`, which snapshots, for that same reason.
        `-r @` is the work only if the working copy has been committed, and
        this must not depend on whether `_has_work` happened to run first.
        """
        code, _, err = await self._commit(
            path, "bookmark", "set", ref, "-r", "@", provisioned=provisioned
        )
        if code != 0:
            log.warning(
                "ph.seams.workspace_jj: jj bookmark set failed in %s (%s)",
                path,
                first_line(err) or f"jj exited {code}",
            )
        return code == 0

    async def _has_work(
        self, path: Path, fork: str | None = None, provisioned: Sequence[str] = ()
    ) -> bool | None:
        """Whether this workspace has done anything since it forked. `None` if
        jj cannot say, or if nobody knows where the workspace forked.

        **The question is "since it forked" and not "is `@` non-empty".** The
        difference was a bug that threw work away. To freeze the work of a
        parent for a child moves the `@` of that parent to a fresh empty
        commit, and its work is then in `@-`. So at its own release a parent
        that had spawned anything looked exactly like a parent that had done
        nothing: the bookmark was deleted and the tree removed.

        **With no `fork`, an empty tip proves nothing.** A crash reclaim reads
        the fork point from its `WorkspaceRecord`, and a record can lack it
        (`WorkspaceRecord.fork`). A non-empty `@` is still work. An empty `@` is
        either a tree that did nothing or the parent in the case above, and
        this method cannot tell which, so it answers `None`. `_release` counts
        `None` as work and keeps the bookmark, because reclaim errs toward
        keeping. This path used to answer "no work" for an empty tip, and the
        reclaim of such a parent deleted the bookmark that held its work.
        """
        found = await self._log(
            path,
            f"{fork}..@ & ~empty()" if fork else "@ & ~empty()",
            snapshot=True,
            provisioned=provisioned,
        )
        if found is None:
            return None
        if found:
            return True
        return False if fork else None

    async def _named(self, repo: Path, ref: str) -> bool:
        """Whether a bookmark by this name still points at something."""
        code, out, _ = await self._read(repo, "bookmark", "list", ref)
        return code == 0 and bool(out.strip())

    async def _managed(self, base: Path) -> Path | None:
        if base not in self._roots:
            self._roots[base] = await self._ask_root(base)
        return self._roots[base]

    async def _ask_root(self, base: Path) -> Path | None:
        """The root of the jj workspace that contains `base`, or `None`.

        This asks jj and does not look for a `.jj` directory, for the reason
        that the git tier asks git: a subdirectory is a valid `base`, and a
        probe that stats one path answers wrongly for it.
        """
        code, out, _ = await self._read(base, "workspace", "root")
        if code != 0 or not out.strip():
            log.info(
                "ph.seams.workspace_jj: %s is not inside a jj workspace; declining so the "
                "seam falls back to a shared workspace",
                base,
            )
            return None
        root = Path(out.strip())
        if not await anyio.to_thread.run_sync(lambda: _colocated(root)):
            # The workspace is served anyway: the isolation is real and the
            # bookmark is a real handle. But this is logged one time per
            # repository, because the merge path that this tier advertises is a
            # git one. A user who cannot `git show` what a child produced should
            # learn why before disposal.
            log.warning(
                "ph.seams.workspace_jj: the repository behind %s is not colocated with git, "
                "so bookmarks this tier writes will not appear as git branches",
                root,
            )
        return root


class Config(WireModel):
    """Row config for the jj tier."""

    root: str | None = None
    """Where workspaces live. The default is `$PH_HOME/jj`. It is outside the
    repository on purpose, for the same reason that the worktree tier gives."""


@plugin("workspace-jj", affects="environment", inject=[WORKSPACE, SUBPROCESS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Claim the workspace slot, but only where jj is installed.

    **The probe gates registration, not acquisition.** This is the rule of
    `workspace-agentfs`, and it is easy to get backwards. `register_provider`
    is `claim_slot`, which is exclusive. A row that took the slot and then
    declined every acquire would not fall back to the git tier. It would fall
    back to `shared`, and a deployment that asked for isolation would silently
    have none.

    Only the *binary* is probed here. Whether jj manages a given directory is
    a question per `base`. `acquire` answers it with a named decline, so the
    reason reaches the log and `phern doctor`. To ask it at mount would be to
    ask about the process's own directory, which is not where agents work.

    **Nothing is ever installed, and no repository is ever converted.** A
    missing binary is a decline that says so in `phern doctor`, through the
    diagnostics seam. That is the standing provisioning rule. It is also the
    reason why this row is safe to layer in a profile that runs on hosts
    without jj.
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
    ctx.require(WORKSPACE).register_provider(
        JjWorkspaceProvider(ctx=ctx, root=default_home_path(config.root, "jj")), scope=ctx
    )
