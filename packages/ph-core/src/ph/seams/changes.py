"""Asking the version control what changed, instead of reading every file.

An indexer over a tree — a code graph, a document corpus — has to answer "which
of these files is different since I last looked". Content-hashing every file is
correct and costs a full read of the whole tree: measured over this repository,
5.2 ms of reads plus 3.0 ms of sha256 for 136 files, so about **1.2 s and 40 000
thread hops for a 20 000-file tree** before a single file is found to have
changed.

Git and jj already hold that answer. Both maintain a content-addressed tree, so
"what changed" is a lookup rather than a scan:

| | measured on this repository (509 files) |
|---|---|
| `git ls-files --full-name -s` — a blob id per file, **no file reads** | **3 ms** |
| `git status --porcelain -uall` — the paths git cannot vouch for | **9 ms** |
| `git rev-parse --show-prefix` — the one vocabulary both are read in | **~2 ms** |
| `jj diff --from <token> --name-only` — the changed set, one call | **one call** |

## A filter, not a digest

This does **not** replace the caller's own content hash, and that is the point.
A consumer keeps hashing, keeps storing its own digest, and uses this only to
decide *which files to open*. So the guarantee a consumer makes about its index
is unchanged — a skip is justified by git's or jj's own content hash, never by a
timestamp — which is why this is worth having where an mtime check was not.

## The two backends are genuinely different shapes

Git's working tree is not a commit, so `ls-files` reports the **index**: a
modified-unstaged file still shows its old blob id (measured — `4b48dee…` while
the content hashes to `e9d25ac…`). Hence a `suspect` set the caller must re-read
regardless of ids — and a third call, because the two answers do not even arrive
in the same *spelling* unless asked to. `_git_state` is where that is argued.

jj **snapshots the working copy on every command**, so `@` always describes the
tree on disk — editing a file changes `@`'s commit id with no command run. One
`jj diff --from <the caller's own token>` is therefore the complete changed set,
uncommitted work included, and there are no per-file ids to compare.

`TreeState` is the union of those two shapes, and `vouches_for` is the one
predicate a consumer needs so it does not have to know which backend answered.

## Which backend, and why the workspace decides

**The workspace provider is the authority.** `Workspace.kind` cannot answer it —
`workspace-git-worktree` and `workspace-jj` both hand back `worktree` — so the
provider declares it, through `VersionedProvider`. An optional capability as its
own Protocol, never a `getattr` probe: the rule this tree states three times
(`ReclaimingProvider`, `ExportingProvider`, `RehydratableProvider`) and argues
in the first of them, because a probe reports a provider whose attribute is
*misnamed* as one that cannot answer.

A tier that declares nothing — `shared`, which is what the default profile and
`containment.tier: advisory` give the root agent — falls through to a probe of
the root itself, cached per directory. That case is not an afterthought: it is
the common one, and a filter that stood down for it would do nothing in the
profile most people run.

## The overlay tier declares nothing either, and that is the answer

An AgentFS overlay keeps its own log of every change it has layered over the
base, and reaching for it is the obvious third backend. **It cannot be read
while the overlay is mounted, which is precisely when an indexer runs.** The
delta database takes a single writer, so with a live mount every route into it
fails alike — measured, on a real FUSE mount:

| | with the overlay mounted |
|---|---|
| `agentfs diff <id>` | `Locking error: File is locked by another process` |
| `agentfs timeline`, `agentfs fs ls` | the same locking error |
| `sqlite3 'file:…?mode=ro'` | `database is locked` |
| `sqlite3 'file:…?mode=ro&immutable=1'` | **no error and zero tables** |

That last row is the trap: `immutable=1` reads the 4 KB main database while
every change sits in a 321 KB WAL, so it answers "nothing has changed" for a
tree that has changed entirely — a wrong `True` of exactly the kind
`vouches_for` exists to never produce. `export_overlay` copes by unmounting
first, which an indexer cannot do to a workspace an agent is working in.

The mechanism that *does* work is the one already here. An overlay's root is its
mountpoint, and the mount serves the base's `.git` along with everything else —
so `_probed` finds it and **git answers, from inside the overlay**, reporting the
agent's own uncommitted writes:

    $ echo 'a = 999' > <mountpoint>/a.py     # a write into the delta layer
    $ git -C <mountpoint> status --porcelain -uall
     M a.py
    $ git -C <mountpoint> ls-files -s        # the base's ids, to vouch with
    100644 1337a53…  a.py
    100644 a678603…  b.py

Which is the git shape exactly: base blob ids to vouch with, and every overlay
write in `suspect` — the modified-unstaged case `suspect` was built for in the
first place. The host tree stays clean throughout, so the isolation the tier
exists for is not weakened by reading through it.

So `AgentFsProvider` deliberately declares no `vcs`. Declaring one would
*override* the probe (see `backend_for`) and replace a backend that works with a
backend that cannot be read — the rare case where adding the obvious feature
subtracts.

@module ph.seams.changes
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..cordis import Context
from ..keys import WORKSPACE
from .workspace import Backend, SnapshottingProvider, VersionedProvider

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

    The union of git's and jj's shapes; see the module docstring for why they
    differ. A consumer asks `vouches_for` and never has to branch on `backend`.
    """

    backend: Backend = ""
    token: str = ""
    """Opaque marker to store now and hand back as `since` next time.

    jj's `@` commit id, which is what makes its one-call diff possible. Empty for
    git, whose per-file ids carry the same information without a marker."""
    diffed: bool = False
    """Whether `suspect` is a real answer rather than "nothing was compared".

    Its own field because `token` was doing both jobs and could not: the first
    jj run has a perfectly good working-copy id to *store* and has compared
    nothing, so resting the proof on `bool(token)` either vouched for every file
    on a first run or threw away the token the next run needs. Written out after
    the test for that case caught it doing the second.

    Meaningless for git, whose per-file ids carry their own proof."""
    ids: Mapping[str, str] = field(default_factory=dict)
    """Workspace-relative path → the backend's own content id. Empty for jj."""
    suspect: frozenset[str] = frozenset()
    """Paths the backend will not vouch for, and the caller must therefore read.

    For git: everything `status` reports as modified or untracked. For jj:
    everything that changed since the caller's `since` token — so a path *absent*
    from this set is unchanged, which is the whole of jj's answer."""

    def vouches_for(self, path: str, stored_id: str) -> bool:
        """Whether `path` is provably unchanged since the caller stored `stored_id`.

        **One predicate for both backends**, so a consumer never branches on
        which answered:

        * no backend, or a suspect path → `False`, and the caller reads it;
        * git → the stored id must equal the id the index reports now;
        * jj → not being suspect *is* the answer, provided a diff produced
          `suspect` at all (`diffed`), because that diff started from the
          caller's own token.

        Conservative in exactly one direction: a `False` costs a read the caller
        was going to do anyway, while a wrong `True` would leave a stale index.
        Every branch that cannot prove freshness answers `False`.
        """
        if not self.backend or path in self.suspect:
            return False
        if self.ids:
            return bool(stored_id) and self.ids.get(path) == stored_id
        # jj: no per-file ids, so `suspect` is the whole answer — and it is only
        # an answer if a diff produced it. See `diffed`.
        return self.diffed

    def id_for(self, path: str) -> str:
        """The id a consumer should store for `path`. `""` when there is none."""
        return self.ids.get(path, "")


def backend_for(ctx: Context, root: Path) -> Backend:
    """Which version control backs `root` — the workspace's answer, then a probe.

    **The provider first**, because that is the deployment's own statement and
    `Workspace.kind` cannot carry it: git and jj worktrees are both `worktree`.
    A tier that declares nothing falls through to `_probed`, which is the
    `shared` case and therefore the common one.
    """
    provider = _tier(ctx)
    if isinstance(provider, VersionedProvider) and provider.vcs:
        return provider.vcs
    return _probed(str(root))


def _tier(ctx: Context) -> Any:
    """The mounted workspace provider, or `None` — the one place that asks.

    `ctx.get`, not `getattr(ctx, "workspace", None)`: the seam may not be mounted
    at all, and reaching a service by attribute name was the only such probe in
    this tree. Renaming the key would then have cost every indexer a silent
    full re-read, with nothing failing — which is what these Protocols exist to
    stop happening by accident.
    """
    seam = ctx.get(WORKSPACE)
    return None if seam is None else seam.provider


@lru_cache(maxsize=256)
def _probed(root: str) -> Backend:
    """What the directory itself says, walking upward. Cached per root.

    A filesystem check rather than a subprocess: `jj root`/`git rev-parse` would
    each cost a spawn to answer what a marker directory already states, and this
    runs once per tree per process. `.jj` is tested first because a colocated
    repository has both and jj is then the one that snapshots the working copy.

    Bounded by `lru_cache` for `ph.seams.fs._compiled`'s reason: the argument
    comes from a workspace root, and a long-lived daemon handing out ephemeral
    worktrees would otherwise accumulate one entry per tree forever.
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

    Never raises and never blocks on a missing tool: a backend that is not
    installed, a directory that is not a repository, or a command that fails all
    answer the same empty `TreeState`, whose `vouches_for` is `False` for
    everything. A consumer that ignored the result entirely would still be
    correct — just slower — which is the property that makes this safe to put in
    front of an indexer.
    """
    backend = backend_for(ctx, root)
    try:
        if backend == "git":
            return await _git_state(ctx, root)
        if backend == "jj":
            return await _jj_state(ctx, root, since=since)
    except Exception:
        # Logged, not raised. This is an optimisation: a caller who loses it
        # reads every file, which is what it did before this module existed.
        log.warning("ph.seams.changes: %s could not describe %s", backend, root, exc_info=True)
    return TreeState()


async def _git_state(ctx: Context, root: Path) -> TreeState:
    """Blob ids from the index, plus everything `status` will not vouch for.

    **Three calls, and the third one is why this is correct.** The two answers
    arrive in *different vocabularies*: `status --porcelain` names paths from the
    repository root, while `ls-files` names them from the directory it ran in.
    On a workspace below the repo root — `packages/ph-core`, say — that is not a
    cosmetic difference, it is a collision: `README.md` means this package's file
    to one command and the repository's to the other, so `suspect` marks a path
    `ids` never had and a genuinely modified file keeps its stale blob id. A
    wrong `True`, which is the one thing `vouches_for` must never produce.

    So both are put in the repository's vocabulary (`--full-name`, which keeps
    `ls-files`' subtree scoping and only changes how it spells what it found) and
    then re-spelled against `root` with `rev-parse --show-prefix`. That order is
    deliberate: a prefix this got *wrong* leaves keys that match nothing the
    caller stored, which costs a re-read — where trusting two vocabularies to
    agree costs a stale index.
    """
    from .workspace_git import git

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
        # `<mode> <sha> <stage>\t<path>`; a path with a tab in it is impossible
        # here because git quotes such names, and a quoted name simply does not
        # match a stored path — so it lands in neither `ids` nor a false skip.
        head, _, path = line.partition("\t")
        parts = head.split()
        if path and len(parts) == 3:
            named = _under(path, prefix)
            if named:
                ids[named] = parts[1]
    suspect = frozenset(_git_paths(answers["dirty"][1], prefix))
    return TreeState(backend="git", ids=ids, suspect=suspect)


def _git_paths(porcelain: str, prefix: str) -> list[str]:
    """The paths in `status --porcelain` output, spelled against the asked-about tree.

    Renames counted at both ends, because the caller may hold a record under
    either name. Anything outside `prefix` is dropped rather than kept: a repo
    holds files this workspace cannot name, and a suspect entry the caller can
    never match is one more string to compare on every path, forever.
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
    """`path` re-spelled against a tree `prefix` deep in the repository.

    `""` when it lies outside, which every caller treats as "not mine". A string
    slice rather than `Path.relative_to`, for `ph.paths.is_under`'s reason: this
    runs once per tracked file, and both sides are already posix-spelled by git.
    """
    if not prefix:
        return path
    return path[len(prefix) :] if path.startswith(prefix) else ""


async def _jj_state(ctx: Context, root: Path, *, since: str) -> TreeState:
    """The working-copy commit id, and what changed since the caller's token.

    jj snapshots the working copy as part of running any command, so `@` reflects
    the tree on disk and one diff is the whole answer. The order matters: `@` is
    read **first** so the token a caller stores can only be older than or equal
    to the tree the diff describes — reading it afterwards could hand back a
    token for a snapshot the diff had not covered.
    """
    ask = _snapshotting(ctx, root)

    code, out, _ = await ask("log", "-r", "@", "--no-graph", "-T", "commit_id")
    token = out.strip()
    if code != 0 or not token:
        return TreeState()
    if not since:
        # Nothing to diff from. The token is still reported — that is the whole
        # point of a first run — while `diffed` stays false, so this run proves
        # nothing and the next one is the cheap one.
        return TreeState(backend="jj", token=token)

    code, out, _ = await ask("diff", "--from", since, "--to", "@", "--name-only")
    if code != 0:
        # An unknown revision — the token predates a `jj op abandon`, or the
        # repository was rebuilt. Not an error: the caller re-reads everything
        # once and stores a token that works.
        return TreeState(backend="jj", token=token)
    changed = frozenset(one.strip() for one in out.splitlines() if one.strip())
    return TreeState(backend="jj", token=token, diffed=True, suspect=changed)


def _snapshotting(ctx: Context, root: Path) -> Any:
    """How to run a jj call that is allowed to commit the tree it reads.

    **The snapshot is this backend's whole advantage and its one hazard.** jj
    commits the working copy on any command, which is why one `diff` covers
    uncommitted work — and why a call made the naive way adopts whatever is
    untracked. `auto_track` (E14) is what holds the seam's own provisioned
    materials out of that commit, and only the tier can supply it, because only
    the tier knows what it put there. So a declared `SnapshottingProvider` is
    asked, and the raw helper — whose docstring sends callers through the
    provider for exactly this reason — is the fallback for the case with nothing
    to hold out: an undeclared tier is the person's own checkout, where the seam
    provisioned nothing and `auto_track(())` is `()` anyway.
    """
    from .workspace_jj import jj

    tier = _tier(ctx)
    if isinstance(tier, SnapshottingProvider):
        return lambda *args: tier.snapshotting(root, *args)
    return lambda *args: jj(ctx, root, *args)


async def _gathered(**calls: Any) -> dict[str, tuple[int, str, str]]:
    """Await independent VCS calls concurrently, keyed by name.

    They do not depend on each other, and each is a process spawn — the git
    calls measured 3 ms and 9 ms, so serialising them spends the shorter ones'
    latency for nothing.

    Keyed by name rather than by position because the caller reads three of them:
    `listed[1]` and `dirty[0]` said nothing about which command answered, and the
    third call arrived by adding a subscript to that.
    """
    import anyio

    results: dict[str, tuple[int, str, str]] = {}

    async def run(name: str, awaitable: Any) -> None:
        results[name] = await awaitable

    async with anyio.create_task_group() as group:
        for name, awaitable in calls.items():
            group.start_soon(run, name, awaitable)
    return results
