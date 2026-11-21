"""Ask the version control what changed, without reading every file.

An indexer over a tree, such as a code graph or a document corpus, must answer
"which of these files is different since I last looked". To hash the content of
every file is correct, and it costs a full read of the whole tree. Measured over
this repository, that is 5.2 ms of reads plus 3.0 ms of sha256 for 136 files. So
it is about **1.2 s and 40 000 thread hops for a 20 000-file tree** before the
indexer finds a single changed file.

Git and jj already hold that answer. Both maintain a content-addressed tree, so
"what changed" is a lookup and not a scan:

| | Measured on this repository (509 files) |
|---|---|
| `git ls-files --full-name -s`: a blob id per file, **no file reads** | **3 ms** |
| `git status --porcelain -uall`: the paths that git cannot vouch for | **9 ms** |
| `git rev-parse --show-prefix`: the one spelling that both are read in | **~2 ms** |
| `jj diff --from <token> --name-only`: the changed set, one call | **one call** |

## A filter, not a digest

This does **not** replace the content hash of the caller, and that is
deliberate. A consumer continues to hash and to store its own digest. It uses
this module only to decide *which files to open*. So the guarantee that a
consumer makes about its index is unchanged: a skip is justified by the content
hash of git or of jj, and never by a timestamp. That is why this filter is worth
having where an mtime check was not.

## The two backends have different shapes

The working tree of git is not a commit, so `ls-files` reports the **index**. A
file that is modified and not staged still shows its old blob id. This was
measured: the id was `4b48dee…` while the content hashed to `e9d25ac…`. So there
is a `suspect` set, which the caller must read again whatever the ids say. There
is also a third call, because the two answers do not arrive in the same
*spelling* unless git is asked for that. `_git_state` gives the argument.

jj **snapshots the working copy on every command**, so `@` always describes the
tree on disk. An edit to a file changes the commit id of `@` with no command
run. So one `jj diff --from <the token of the caller>` is the complete changed
set, with uncommitted work included. There are no per-file ids to compare.

`TreeState` is the union of those two shapes. `vouches_for` is the one predicate
that a consumer needs, so the consumer does not have to know which backend
answered.

## Which backend, and why the workspace decides

**The workspace provider is the authority.** `Workspace.kind` cannot answer the
question, because `workspace-git-worktree` and `workspace-jj` both return
`worktree`. So the provider declares it, through `VersionedProvider`.
`ph.seams.workspace` defines that Protocol, and this module re-exports it. It is
an optional capability as its own Protocol, and never a `getattr` probe.
`ReclaimingProvider` argues that rule, and `ExportingProvider` and
`RehydratableProvider` take the same shape. The reason is that a probe reports a
provider whose attribute is *misnamed* as one that cannot answer.

A provider that declares nothing falls through to a probe of the root itself,
and the answer is cached per directory. The built-in `shared` provider is such a
provider, and it is the one that the default profile mounts. That case is not an
afterthought. It is the common case, and a filter that did nothing for it would
do nothing in the profile that most users run. The question is asked of the
mounted provider and not of the tier that one agent got. So in a profile that
layers a git or jj provider and keeps its root agent on
`containment.tier: advisory`, the provider answers for the root agent too.

## The overlay tier also declares nothing, and that is correct

An AgentFS overlay keeps its own log of every change that it has layered over
the base. That log looks like the obvious third backend. **It cannot be read
while the overlay is mounted, and that is exactly when an indexer runs.** The
delta database takes a single writer. So with a live mount, every way into it
fails in the same way. This was measured on a real FUSE mount:

| | With the overlay mounted |
|---|---|
| `agentfs diff <id>` | `Locking error: File is locked by another process` |
| `agentfs timeline`, `agentfs fs ls` | the same locking error |
| `sqlite3 'file:…?mode=ro'` | `database is locked` |
| `sqlite3 'file:…?mode=ro&immutable=1'` | **no error and zero tables** |

The last row is the dangerous one. `immutable=1` reads the 4 KB main database
while every change is in a 321 KB WAL. So it answers "nothing has changed" for a
tree that has changed entirely. That is a wrong `True`, which is exactly what
`vouches_for` must never produce. `export_overlay` reads the log only while the
overlay is not mounted. An indexer cannot arrange that for a workspace that an
agent is working in.

The mechanism that *does* work is the one that is already here. The root of an
overlay is its mountpoint, and the mount serves the `.git` of the base together
with everything else. So `_probed` finds it, and **git answers from inside the
overlay**. It reports the agent's own uncommitted writes:

    $ echo 'a = 999' > <mountpoint>/a.py     # a write into the delta layer
    $ git -C <mountpoint> status --porcelain -uall
     M a.py
    $ git -C <mountpoint> ls-files -s        # the base's ids, to vouch with
    100644 1337a53…  a.py
    100644 a678603…  b.py

That is exactly the git shape: the blob ids of the base to vouch with, and every
overlay write in `suspect`. It is the case of a file that is modified and not
staged, which `suspect` was built for in the first place. The host tree stays
clean throughout. So reading through the overlay does not weaken the isolation
that the tier exists for.

So `AgentFsProvider` deliberately declares no `vcs`. A declared `vcs` would
*override* the probe (see `backend_for`). It would replace a backend that works
with a backend that cannot be read. This is the rare case where the obvious
feature makes the result worse.

@module ph.seams.changes
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import anyio

from ..cordis import Context
from ..keys import WORKSPACE
from .workspace import Backend, SnapshottingProvider, VersionedProvider, WorkspaceProvider
from .workspace_git import git
from .workspace_jj import jj

__all__ = [
    "Backend",
    "TreeState",
    "VersionedProvider",
    "backend_for",
    "tree_state",
]

log = logging.getLogger("ph.seams.changes")


@dataclass(frozen=True, slots=True)
class TreeState:
    """What the version control says about one tree, right now.

    It is the union of the shapes of git and jj. The module docstring says why
    they differ. A consumer asks `vouches_for` and never has to branch on
    `backend`.
    """

    backend: Backend = ""
    token: str = ""
    """An opaque marker to store now and to pass back as `since` next time.

    For jj it is the commit id of `@`, which makes the one-call diff possible.
    It is empty for git, whose per-file ids carry the same information without a
    marker."""
    diffed: bool = False
    """Whether `suspect` is a real answer, and not "nothing was compared".

    It is a field of its own because `token` could not do both jobs. The first
    jj run has a good working-copy id to *store*, and it has compared nothing.
    With the proof resting on `bool(token)`, a first run either vouched for
    every file or discarded the token that the next run needs. The field was
    written out after the test for that case caught the code doing the second.

    It has no meaning for git, whose per-file ids carry their own proof."""
    ids: Mapping[str, str] = field(default_factory=dict)
    """Workspace-relative path → the backend's own content id. Empty for jj."""
    suspect: frozenset[str] = frozenset()
    """Paths that the backend will not vouch for. The caller must read them.

    For git, it is everything that `status` reports as modified or untracked.
    For jj, it is everything that changed since the `since` token of the
    caller. So a path that is *absent* from this set is unchanged, and that is
    the whole answer of jj."""

    def vouches_for(self, path: str, stored_id: str) -> bool:
        """Whether `path` is provably unchanged since the caller stored `stored_id`.

        **One predicate serves both backends**, so a consumer never branches on
        which one answered:

        * No backend, or a suspect path: `False`, and the caller reads the file.
        * git: the stored id must equal the id that the index reports now.
        * jj: a path that is not suspect is unchanged, provided that a diff
          produced `suspect` at all (`diffed`). The reason is that the diff
          started from the caller's own token.

        The predicate is conservative in exactly one direction. A `False` costs
        a read that the caller was going to do anyway. A wrong `True` would
        leave a stale index. Every branch that cannot prove freshness answers
        `False`.
        """
        if not self.backend or path in self.suspect:
            return False
        if self.ids:
            return bool(stored_id) and self.ids.get(path) == stored_id
        # jj has no per-file ids, so `suspect` is the whole answer. It is an
        # answer only if a diff produced it. See `diffed`.
        return self.diffed

    def id_for(self, path: str) -> str:
        """The id a consumer should store for `path`. `""` when there is none."""
        return self.ids.get(path, "")


def backend_for(ctx: Context, root: Path) -> Backend:
    """Which version control backs `root`: the answer of the workspace, then a probe.

    **The provider is asked first.** Its answer is the statement of the
    deployment itself, and `Workspace.kind` cannot carry it: git and jj
    worktrees are both `worktree`. A provider that declares nothing falls
    through to `_probed`. That is the built-in `shared` provider, and therefore
    the common case.
    """
    provider = _tier(ctx)
    if isinstance(provider, VersionedProvider) and provider.vcs:
        return provider.vcs
    return _probed(str(root))


def _tier(ctx: Context) -> WorkspaceProvider | None:
    """The mounted workspace provider, or `None`. This is the one place that asks.

    It uses `ctx.get` and not `getattr(ctx, "workspace", None)`, because the
    seam may not be mounted at all. To reach a service by attribute name was the
    only such probe in this tree. A rename of the key would then have cost every
    indexer a silent full re-read, with nothing failing. These Protocols exist
    to stop that from happening by accident.
    """
    seam = ctx.get(WORKSPACE)
    return None if seam is None else seam.provider


@lru_cache(maxsize=256)
def _probed(root: str) -> Backend:
    """What the directory itself says, walking upward. Cached per root.

    It is a filesystem check and not a subprocess. `jj root` or `git rev-parse`
    would each cost a spawn to answer what a marker directory already states,
    and this runs one time per tree per process. `.jj` is tested first because a
    colocated repository has both markers, and jj is then the one that snapshots
    the working copy.

    The cache is bounded by `lru_cache`, for the same reason as
    `ph.seams.fs._compiled`. The argument comes from a workspace root, and a
    long-lived daemon that hands out ephemeral worktrees would otherwise
    accumulate one entry per tree forever.
    """
    here = Path(root)
    for directory in (here, *here.parents):
        if (directory / ".jj").is_dir():
            return "jj"
        if (directory / ".git").exists():
            return "git"
    return ""


async def tree_state(ctx: Context, root: Path, *, since: str = "") -> TreeState:
    """What version control says about `root`, for a caller last seen at `since`.

    It never raises, and it never blocks on a missing tool. A backend that is
    not installed, a directory that is not a repository, and a command that
    fails all give the same empty `TreeState`, whose `vouches_for` is `False`
    for everything. A consumer that ignored the result entirely would still be
    correct, only slower. That property makes this safe to put in front of an
    indexer.
    """
    backend = backend_for(ctx, root)
    try:
        if backend == "git":
            return await _git_state(ctx, root)
        if backend == "jj":
            return await _jj_state(ctx, root, since=since)
    except Exception:
        # Logged, not raised. This is an optimization. A caller that loses it
        # reads every file, which is what it did before this module existed.
        log.warning("ph.seams.changes: %s could not describe %s", backend, root, exc_info=True)
    return TreeState()


async def _git_state(ctx: Context, root: Path) -> TreeState:
    """Blob ids from the index, plus everything that `status` will not vouch for.

    **There are three calls, and the third is what makes this correct.** The
    first two answers arrive in *different spellings*. `status --porcelain`
    names paths from the repository root, and `ls-files` names them from the
    directory that it ran in. On a workspace below the repository root, such as
    `packages/ph-core`, the difference is not cosmetic. It is a collision.
    `README.md` means the file of this package to one command and the file of
    the repository to the other. So `suspect` marks a path that `ids` never had,
    and a file that really is modified keeps its stale blob id. That is a wrong
    `True`, which is the one thing that `vouches_for` must never produce.

    So both answers are put in the spelling of the repository, and are then
    re-spelled against `root` with `rev-parse --show-prefix`. `--full-name` does
    the first step. It keeps the subtree scoping of `ls-files` and changes only
    how it spells what it found. That order is deliberate. A prefix that this
    function got *wrong* leaves keys that match nothing that the caller stored,
    which costs a re-read. To trust that two spellings agree costs a stale
    index.
    """
    answers = await _gathered(
        listed=git(ctx, root, "ls-files", "--full-name", "-s"),
        dirty=git(ctx, root, "status", "--porcelain", "--untracked-files=all"),
        prefix=git(ctx, root, "rev-parse", "--show-prefix"),
    )
    if any(code != 0 for code, _, _ in answers.values()):
        return TreeState()
    prefix = answers["prefix"][1].strip()

    ids: dict[str, str] = {}
    for line in answers["listed"][1].splitlines():
        # The format is `<mode> <sha> <stage>\t<path>`. A path with a tab in it
        # cannot appear here, because git quotes such names. A quoted name
        # matches no stored path, so it can never cause a false skip.
        head, _, path = line.partition("\t")
        parts = head.split()
        if path and len(parts) == 3:
            named = _under(path, prefix)
            if named:
                ids[named] = parts[1]
    suspect = frozenset(_git_paths(answers["dirty"][1], prefix))
    return TreeState(backend="git", ids=ids, suspect=suspect)


def _git_paths(porcelain: str, prefix: str) -> list[str]:
    """The paths in `status --porcelain` output, spelled against the tree that was asked about.

    A rename counts at both ends, because the caller may hold a record under
    either name. Anything outside `prefix` is dropped and not kept. A
    repository holds files that this workspace cannot name. A suspect entry
    that the caller can never match is one more string to compare on every
    path, forever.
    """
    found: list[str] = []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        # `R  old -> new`: both sides are suspect, because the caller may hold a
        # record under either name.
        before, arrow, after = line[3:].partition(" -> ")
        for name in (before, after) if arrow else (before,):
            named = _under(name.strip('"'), prefix)
            if named:
                found.append(named)
    return found


def _under(path: str, prefix: str) -> str:
    """`path` re-spelled against a tree that is `prefix` deep in the repository.

    It returns `""` when the path lies outside, which every caller treats as
    "not mine". It is a string slice and not `Path.relative_to`, for the same
    reason as `ph.paths.is_under`: this runs one time per tracked file, and git
    has already spelled both sides in posix form.
    """
    if not prefix:
        return path
    return path[len(prefix) :] if path.startswith(prefix) else ""


async def _jj_state(ctx: Context, root: Path, *, since: str) -> TreeState:
    """The working-copy commit id, and what changed since the token of the caller.

    jj snapshots the working copy as part of running any command. So `@`
    reflects the tree on disk, and one diff is the whole answer. The order
    matters. `@` is read **first**, so that the token that a caller stores can
    only be older than or equal to the tree that the diff describes. If `@` were
    read afterwards, the token could name a snapshot that the diff had not
    covered.
    """
    ask = _snapshotting(ctx, root)

    code, out, _ = await ask("log", "-r", "@", "--no-graph", "-T", "commit_id")
    token = out.strip()
    if code != 0 or not token:
        return TreeState()
    if not since:
        # There is nothing to diff from. The token is still reported, which is
        # the purpose of a first run. `diffed` stays false, so this run proves
        # nothing and the next run is the cheap one.
        return TreeState(backend="jj", token=token)

    code, out, _ = await ask("diff", "--from", since, "--to", "@", "--name-only")
    if code != 0:
        # An unknown revision. The token predates a `jj op abandon`, or the
        # repository was rebuilt. This is not an error. The caller reads
        # everything again one time and stores a token that works.
        return TreeState(backend="jj", token=token)
    changed = frozenset(one.strip() for one in out.splitlines() if one.strip())
    return TreeState(backend="jj", token=token, diffed=True, suspect=changed)


def _snapshotting(ctx: Context, root: Path) -> Callable[..., Awaitable[tuple[int, str, str]]]:
    """How to run a jj call that is allowed to commit the tree that it reads.

    **The snapshot is the whole advantage of this backend, and its one
    hazard.** jj commits the working copy on any command. That is why one `diff`
    covers uncommitted work. It is also why a call that is made directly adopts
    whatever is untracked. `auto_track` (E14) keeps the provisioned materials of
    the seam out of that commit. Only the tier can supply it, because only the
    tier knows what it put there. So a declared `SnapshottingProvider` is asked.
    The raw `jj` helper is the fallback for the case with nothing to keep out.
    The docstring of that helper sends callers through the provider for exactly
    this reason. An undeclared tier is the user's own checkout, where the seam
    provisioned nothing and `auto_track(())` is `()` anyway.
    """
    tier = _tier(ctx)
    if isinstance(tier, SnapshottingProvider):
        return lambda *args: tier.snapshotting(root, *args)
    return lambda *args: jj(ctx, root, *args)


async def _gathered(
    **calls: Awaitable[tuple[int, str, str]],
) -> dict[str, tuple[int, str, str]]:
    """Await independent VCS calls concurrently, keyed by name.

    The calls do not depend on each other, and each is a process spawn. The git
    calls measured 3 ms and 9 ms, so to run them in series would add the time of
    the shorter calls for no benefit.

    The results are keyed by name and not by position, because the caller reads
    three of them. `listed[1]` and `dirty[0]` said nothing about which command
    answered, and the third call was added as one more subscript on top of that.
    """
    results: dict[str, tuple[int, str, str]] = {}

    async def run(name: str, awaitable: Awaitable[tuple[int, str, str]]) -> None:
        results[name] = await awaitable

    async with anyio.create_task_group() as group:
        for name, awaitable in calls.items():
            group.start_soon(run, name, awaitable)
    return results
