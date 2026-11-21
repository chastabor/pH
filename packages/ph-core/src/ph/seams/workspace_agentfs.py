"""`workspace-agentfs`: a copy-on-write overlay as a second workspace provider (P6-21).

AgentFS presents the host tree as a **read-only base** and puts every change in
a delta layer that belongs to one agent. So an agent gets an isolated view of
the whole directory without a checkout.

**It is a peer of `worktree`, not a rung above it.** An overlay that is made
with `mount` does not confine the *process*. A command whose cwd is inside the
mount can write to an absolute path outside it, and the write really lands on
the host. What the overlay bounds is exactly what the worktree tier bounds. It
bounds tool-mediated writes and relative-path writes, because both resolve
against the cwd. It does not bound a raw write to an absolute path. So `tier` is
`worktree` here, and the rung between `worktree` and `sandbox` is left for
something that really confines.

**Where the overlay differs from a worktree, the difference counts against the
overlay.** A relative write into a worktree *lands*, and git can see it. The
same write into an overlay lands in the delta. So the agent believes that work
is durable which nobody else will ever see. `discards_writes` is therefore true
of `overlay-ephemeral`. An `overlay` keeps its delta so that the work can be
exported afterwards.

**The mode is `mount`, not `run` and not `exec`.** `run` is the mode that would
confine. It is unavailable on any host with
`kernel.apparmor_restrict_unprivileged_userns=1`, which is the default on
Ubuntu 23.10 and later. `exec` mounts and runs in one step, which would suit
`acquire` exactly. But it fails on a new agent that has nothing mounted. So this
provider owns a mount lifecycle and does not wrap a command.

**The probe gates registration, not acquisition.** It also asserts *isolation*
and not availability. `register_provider` is `claim_slot`, which is exclusive. A
row that claimed the slot and then declined every `acquire` would fall back to
`shared`, which is advisory. A deployment that asked for more isolation would
silently get none. Availability is the wrong question in any case, because
`agentfs run --experimental-sandbox` exits 0, prints nothing, and writes
straight through to the host. The probe writes a canary through a throwaway
overlay, and the row registers only if the host copy is untouched.

**Nothing is ever installed.** A missing binary or an unavailable backend is a
decline. `phern doctor` reports it, through the diagnostics seam.

@module ph.seams.workspace_agentfs
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import anyio

from ..cordis import Context, plugin, releasing
from ..keys import SUBPROCESS, WORKSPACE
from ..paths import default_home_path
from ..wire import WireModel
from .containment import TIERS, TierDescription
from .diagnostics import Diagnostic, contribute
from .subprocess import SubprocessSpawnSpec, first_line
from .workspace import (
    ContainmentTier,
    Workspace,
    WorkspaceAccess,
    WorkspaceDeclined,
    WorkspaceRecord,
    WorkspaceSite,
    discards_writes,
    redirection_env,
    sanitize_ref,
    tree_path,
)
from .workspace_git import (
    COMMIT_AS_PH,
    delete_branch,
    git,
    git_lines,
    list_branches,
    merge_branch,
)

__all__ = [
    "ORIGIN",
    "AgentFsProvider",
    "ExportRefused",
    "OverlayProbe",
    "apply",
    "export_overlay",
    "fs_id",
    "is_mount",
    "open_overlay",
    "probe_overlay",
    "store_for",
    "unmount",
]

log = logging.getLogger("ph.seams.workspace_agentfs")


_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
"""AgentFS refuses an id that holds anything but alphanumerics, hyphens and
underscores.

It refuses and does not sanitize: `Invalid agent ID` is an error and not a
warning. So the mapping happens here, and is not discovered at the first
acquire.
"""


def fs_id(session_id: str, agent_id: str) -> str:
    """One AgentFS id per (session, agent), in the characters that AgentFS accepts."""
    return _UNSAFE.sub("_", f"{session_id}-{agent_id}")


def store_for(root: Path, session_id: str, agent_id: str) -> Path:
    """Where the delta and the mountpoint of one agent live.

    It is the `tree_path` of the seam, which is also where the git tier and the
    jj tier put their checkouts. So one lossy mapping from id to path serves
    every tier. `fs_id` answers a different question, because AgentFS refuses
    ids that git would accept.
    """
    return tree_path(root, session_id, agent_id)


async def open_overlay(ctx: Context, store: Path, identifier: str, mountpoint: Path) -> str:
    """Mount one overlay. Return `""` when it is up, or else the reason why it is not.

    **Both halves of the success rule are in one place.** A mount is up when the
    command exited zero *and* something is really mounted there.
    `agentfs run --experimental-sandbox` shows that an exit code alone proves
    nothing. Three callers needed this rule. Each of them wrote out the pair of
    checks, and the two operator-facing strings, for itself.

    The answer is a reason and not an exception, because the three callers turn
    it into different things: a probe result, a `WorkspaceDeclined`, and an
    `ExportRefused`.
    """
    await anyio.to_thread.run_sync(lambda: mountpoint.mkdir(parents=True, exist_ok=True))
    code, _, err = await agentfs(ctx, store, "mount", identifier, str(mountpoint))
    if code == 0 and await is_mount(mountpoint):
        return ""
    return first_line(err) or "the overlay did not mount"


async def run(ctx: Context, program: str, cwd: Path, *args: str) -> tuple[int, str, str]:
    """One invocation of `agentfs` or of an unmount tool, through `ctx.subprocess`.
    It never goes through `os.system`.

    The reasons are those of `workspace_git.git`: the seam scrubs the
    credential-shaped environment (F1) and reaps the child in a `finally` (F4).

    `LC_ALL=C` is set because stderr is translated with gettext and the seam
    passes `LANG` through. Without it, a refusal reason that is read from the
    words of a tool would depend on the locale of the operator.

    **`cwd` matters here.** `agentfs init` writes its delta database to
    `./.agentfs/<id>.db`, relative to the working directory. So the directory
    that this runs from decides where the overlay of an agent is stored. To run
    it from the agent's own directory keeps the deltas of two agents apart on
    disk, as well as in the filesystem that they see.
    """
    spec = SubprocessSpawnSpec(
        argv=(program, *args), cwd=cwd, env=ctx.require(SUBPROCESS).env(extra={"LC_ALL": "C"})
    )
    outcome = await ctx.require(SUBPROCESS).run(spec)
    return outcome.exit_code, outcome.stdout, outcome.stderr


async def agentfs(ctx: Context, cwd: Path, *args: str) -> tuple[int, str, str]:
    """One AgentFS invocation. It is `run`, with the program named in one place."""
    return await run(ctx, "agentfs", cwd, *args)


async def unmount(ctx: Context, mountpoint: Path) -> bool:
    """Drop a FUSE mount. Return `True` if the path is no longer mounted afterwards.

    It uses `fusermount3` because that is what an unprivileged user has.
    `umount(8)` needs `CAP_SYS_ADMIN` for anything that is not in `/etc/fstab`,
    and the purpose of the FUSE path is that it needs no privilege.
    `fusermount` is the same tool on a host that still ships FUSE 2. The
    fallback that was here before named `umount`, which cannot work for the
    reason above.

    It does not raise on failure, because this runs at teardown. A mount that
    will not release is a leak that is worth a log line. It is not a reason to
    fail a disposal that has other work to do.
    """
    for argv in (("fusermount3", "-u", str(mountpoint)), ("fusermount", "-u", str(mountpoint))):
        if shutil.which(argv[0]) is None:
            continue
        code, _, err = await run(ctx, argv[0], mountpoint.parent, *argv[1:])
        if code == 0 or not await is_mount(mountpoint):
            return True
        log.debug("ph.seams.workspace_agentfs: %s left %s mounted (%s)", argv[0], mountpoint, err)
    return not await is_mount(mountpoint)


async def is_mount(path: Path) -> bool:
    """Whether a FUSE filesystem is really mounted here.

    It uses `os.path.ismount` and does not ask "does the directory have anything
    in it". An empty base is legal, and a mount that failed silently leaves an
    empty directory that reads in exactly the same way.

    It runs off the event loop, because this is the one call in this module that
    is most likely to block. It stats a FUSE mountpoint whose server may be
    stuck. Everything cheaper in this module already ran in a thread.
    """

    def check() -> bool:
        try:
            return os.path.ismount(path)
        except OSError:  # pragma: no cover - a path that vanished mid-check
            return False

    return await anyio.to_thread.run_sync(check)


@dataclass(frozen=True, slots=True)
class OverlayProbe:
    """What one canary run learned about this host.

    Both outcomes carry a reason, because the interesting case for an operator
    is the decline. "Why am I on worktrees" can be answered only if the row that
    declined said what it tried.
    """

    isolates: bool
    because: str


async def probe_overlay(ctx: Context, scratch: Path) -> OverlayProbe:
    """Write a canary through a throwaway overlay and see whether the host copy changed.

    **It tests isolation, not availability.** A backend can exit 0, print no
    error, and write straight through to the host, because it sandboxes only its
    own mount. A probe that checked the exit code would claim the tier while the
    agent had no isolation. This probe reports isolation only if the host copy
    is unchanged after a write through the overlay.

    It runs one time, when the row is mounted, because the answer is a property
    of the host. A kernel that refuses namespaces, or a FUSE backend that will
    not start, gives the same answer for every agent. With a probe at each
    acquire, a profile that declines would pay for a tier that it does not use.
    """
    if shutil.which("agentfs") is None:
        return OverlayProbe(False, "agentfs is not installed")

    work = scratch / "agentfs-probe"
    base, mount = work / "base", work / "mnt"

    def prepare() -> None:
        # This guards against a predecessor that died in the middle of a probe.
        # The `finally` below cleans up after *this* probe.
        shutil.rmtree(work, ignore_errors=True)
        base.mkdir(parents=True)
        (base / "canary").write_text("host", encoding="utf-8")

    def write_and_read() -> str:
        (mount / "canary").write_text("agent", encoding="utf-8")
        return (base / "canary").read_text(encoding="utf-8")

    await anyio.to_thread.run_sync(prepare)
    try:
        code, _, err = await agentfs(ctx, work, "init", "--base", str(base), "ph_probe")
        if code != 0:
            return OverlayProbe(False, first_line(err) or "agentfs init failed")
        refused = await open_overlay(ctx, work, "ph_probe", mount)
        if refused:
            return OverlayProbe(False, refused)
        if await anyio.to_thread.run_sync(write_and_read) != "host":
            # The `--experimental-sandbox` case: the command succeeded and the
            # host copy changed anyway. The probe exists to refuse this case.
            return OverlayProbe(False, "a write through the overlay reached the host")
        return OverlayProbe(True, "writes land in a delta layer; the host base is unchanged")
    except Exception as error:  # pragma: no cover - a host that fails in a new way
        return OverlayProbe(False, f"{type(error).__name__}: {error}")
    finally:
        with releasing():
            if await is_mount(mount):
                await unmount(ctx, mount)
            await anyio.to_thread.run_sync(lambda: shutil.rmtree(work, ignore_errors=True))


@dataclass(slots=True)
class AgentFsProvider:
    """The `overlay` kind: one copy-on-write view of the tree for each agent.

    **It declares no `vcs`, on purpose.** See `ph.seams.changes`. The change log
    of the overlay itself looks like the obvious backend for the change filter.
    But it cannot be read while the overlay is mounted, and the overlay is
    mounted whenever an agent uses it. What works is this: the mountpoint serves
    the `.git` of the base, so the probe of the filter finds git, and git
    reports the writes in the delta layer as the uncommitted changes that they
    are. A `vcs` declared here would override that probe, and would replace a
    working backend with one that cannot be read.
    """

    ctx: Context
    root: Path
    """Where deltas and mountpoints live: `$PH_HOME/overlays/<session>/<agent>`.
    It is outside the tree that is overlaid. So a mountpoint is never itself a
    candidate for the walk that the agent runs over its own workspace."""
    tier: ContainmentTier = field(default="worktree", init=False)
    """**A peer of the worktree tier, not a rung above it.**

    An overlay bounds exactly what a worktree bounds: writes that resolve
    against the cwd. It misses exactly what a worktree misses: a raw write to an
    absolute path. The rung between `worktree` and `sandbox` belongs to
    something that confines the process. To name that rung here would make
    `phern doctor` overstate this tier.
    """

    def describe_tier(self) -> TierDescription:
        """What an overlay really bounds, which is not what the stock row of its rung says.

        `bounds` is the text of the worktree tier, word for word. That is the
        measured result and not a convenience. The other two columns are where
        the stock text of the rung was false. The writes of an overlay reach
        nobody until somebody exports them. And `TIERS["worktree"]` offers
        "per-run checkpoints, /revert", and an overlay has neither. `write-tree`
        against a FUSE mountpoint has nothing to hash, so no restore point is
        ever written. To advertise a mechanism that the mounted tier does not
        have, in the one place where a user looks to check, is the failure that
        E1 exists to prevent.
        """
        return TierDescription(
            bounds=TIERS["worktree"].bounds,
            does_not_bound=(
                'an absolute-path raw write — open("/etc/passwd", "w") never consults a '
                "cwd; and every write it does bound lands in a delta layer nobody else "
                "sees until it is exported"
            ),
            buys=(
                "collision isolation, the tree as it actually is (untracked and ignored "
                "files included), and instant discard — but no per-run checkpoints and "
                "no /revert"
            ),
        )

    async def refs(self, base: Path) -> list[str]:
        """`ArtifactProvider`, and only that. An overlay leaves refs, not checkouts.

        **The artifact of an overlay is a git branch, like that of any other
        tier.** `export_overlay` builds one from the delta, and the reply of
        `/workspaces export` tells the user to merge it with
        `/workspaces merge`. So this tier answers the ref verbs. Otherwise that
        reply would name a command that refuses.

        The tier does *not* implement `EnumeratingProvider`, which holds
        `strays` and `discard`. That is the reason why the two are separate
        Protocols. A mountpoint is not a checkout that `/workspaces` can hand
        back. This tier used to say so with two stubs, and `isinstance` could
        not tell those stubs from the methods of a tier that meant them.
        """
        return await list_branches(self.ctx, base)

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str:
        return await delete_branch(self.ctx, base, ref, force=force)

    async def merge(self, base: Path, ref: str) -> str:
        return await merge_branch(self.ctx, base, ref)

    def locate(self, *, session_id: str, agent_id: str, access: WorkspaceAccess) -> WorkspaceSite:
        """Where the overlay of this agent is mounted. `LocatingProvider`.

        The site is the mountpoint and not the store. `root` is what a record
        names and what `reclaim` reads, and the delta is in its parent. The
        site is stated before the tier acts, because the acts of this tier are
        the ones that a crash leaves in the worst state. A FUSE mount outlives
        the process that made it, and a mount that no record named was never
        unmounted.
        """
        return WorkspaceSite(
            root=store_for(self.root, session_id, agent_id) / "mnt",
            kind="overlay-ephemeral" if access == "read" else "overlay",
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
        """A private view of `base` for this agent, or a named decline.

        **`write` and `read` divide exactly as the worktree tier divides them.**
        A writer keeps its delta at release, so that the work can be exported
        onto a branch. The delta of a reader is discarded. That mapping is the
        only reason why `write` can be served at all. An overlay with no way to
        merge back would give a root agent a tree whose work silently
        disappears.

        **The base commit is recorded here, not at export.** That makes the
        export a real merge and not an overwrite. A branch rooted at whatever
        `HEAD` is when somebody exports would present the work of the agent as
        though it had been written against a tree that the agent never saw.
        There is nothing to record when the base is not a repository.
        `export_overlay` refuses such an overlay, and says so.
        """
        site = self.locate(session_id=session_id, agent_id=agent_id, access=access)
        identifier = fs_id(session_id, agent_id)
        mount = site.root
        store = mount.parent
        await anyio.to_thread.run_sync(lambda: store.mkdir(parents=True, exist_ok=True))

        code, _, err = await agentfs(self.ctx, store, "init", "--base", str(base), identifier)
        if code != 0:
            raise WorkspaceDeclined("overlay-failed", first_line(err) or "agentfs init failed")
        refused = await open_overlay(self.ctx, store, identifier, mount)
        if refused:
            raise WorkspaceDeclined("overlay-failed", refused)

        await self._record_base(store, base)
        return Workspace(
            root=mount,
            scratch=scratch,
            kind=site.kind,
            # True, for the same reason as `worktree-ephemeral`: the agent
            # writes the whole tree freely. That the writes reach nobody is for
            # `kind` to say, not for this flag. `False` here would be a
            # confinement claim that only a sandbox backend can make.
            repo_writable=True,
            env=redirection_env(scratch),
            # `discards_writes(kind)`, and not a flag derived from `access` a
            # second time. The retention policy reads the predicate of the
            # seam, and a second spelling of the rule here is how the two would
            # come to disagree. The git tier and the jj tier release in the
            # same way.
            release=lambda workspace: self._release(
                store, discard=discards_writes(workspace.kind) and not workspace.retained
            ),
        )

    async def _record_base(self, store: Path, base: Path) -> None:
        """Remember where this overlay came from, if it came from a repository.

        A `stat` gates the call, and git is not asked unconditionally. On a
        base that is not a repository, the spawn fails as slowly as it
        succeeds. That would be one process per agent, spent on the one branch
        where the call can return nothing.
        """
        if not await anyio.to_thread.run_sync((base / ".git").exists):
            return
        lines = await git_lines(self.ctx, base, "rev-parse", "--show-toplevel", "HEAD")
        if len(lines) != 2:
            return
        await anyio.to_thread.run_sync(
            lambda: (store / ORIGIN).write_text("\n".join(lines), encoding="utf-8")
        )

    async def export(self, record: WorkspaceRecord) -> str:
        """Assemble the delta of this overlay into a branch, and return its name.

        The record locates everything. `root` is the mountpoint, so the delta
        is in its parent, and the repository and the commit were written beside
        it at acquire. The branch is `ph/<session>/<agent>`, which is how the
        worktree tier names its own. An operator who reads `git branch` later
        cannot tell which tier produced which branch, and does not need to.
        """
        return await export_overlay(
            self.ctx,
            store=Path(record.root).parent,
            identifier=fs_id(record.session_id, record.agent_id),
            ref=f"ph/{sanitize_ref(record.session_id)}/{sanitize_ref(record.agent_id)}",
        )

    async def _release(self, store: Path, *, discard: bool) -> bool:
        """Drop the mount. Remove the delta too, unless the tree is evidence.

        The mount always goes, and the delta is kept conditionally. A live FUSE
        mount outlives the process that made it, so a mount left behind leaks a
        mount table entry and a server process. A delta database left behind
        costs disk. The two are not the same kind of leak, and they are not
        decided together.

        `retained` is read from the workspace at teardown and is not captured
        at acquire, for the reason of P6-28: whether this tree is evidence
        depends on how the agent *ended*, and nobody knows that when it starts.
        """
        mount = store / "mnt"
        if not await unmount(self.ctx, mount):
            log.warning("ph.seams.workspace_agentfs: %s is still mounted; not removing", mount)
            return True
        if not discard:
            return True
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(store, ignore_errors=True))
        return False

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        """Release an overlay that this process never acquired (F6).

        **Without this method the leak is reported forever and never closed.**
        An overlay folds into `workspace_survivors` like any fresh-root kind, so
        reconciliation finds the pair. Without `reclaim`,
        `isinstance(provider, ReclaimingProvider)` would be False. The seam
        would log "no mounted tier can reclaim" and leave the pair open for
        every future session to report again. The retention summary would
        continue to tell users to run a `gc` that collects nothing.

        It is worse here than for a checkout. A crash skips `_release`
        entirely, and a live FUSE mount outlives the process that made it. So
        what is left behind is a mount table entry and a server, and not a
        directory.

        The record locates itself: `root` is the mountpoint, so the delta is in
        its parent. The disposal policy is the same as that of an orderly
        release, for the reason that the worktree tier gives.
        """
        return await self._release(
            Path(record.root).parent,
            discard=discards_writes(record.kind) and not record.reason,
        )


class Config(WireModel):
    """Row config for the overlay provider."""

    root: str | None = None
    """Where deltas and mountpoints live. The default is `$PH_HOME/overlays`.
    It is outside the tree that is overlaid, for the same reason that a
    worktree is outside the repository: the agent's own `glob` would walk a
    mountpoint inside `base`, and every child would nest it one level deeper."""


@plugin("workspace-agentfs", affects="environment", inject=[WORKSPACE, SUBPROCESS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Probe the host, and claim the workspace slot only if the overlay isolates.

    **The probe gates registration, not acquisition.** This is easy to get
    backwards, and the mistake is expensive. `register_provider` is
    `claim_slot`: exclusive, one provider. A row that took the slot and then
    declined every acquire would not fall back to the worktree tier. It would
    fall back to `shared`, which is advisory. So a deployment that asked for
    *more* isolation would silently end up with none.

    The result reaches `phern doctor` in both cases. "Why am I on worktrees" is
    a question that a user asks of the tool and not of the source. And a row
    that declined without saying so cannot be told apart from a row that nobody
    mounted.
    """
    root = default_home_path(config.root, "overlays")
    result = await probe_overlay(ctx, root / "probe")

    contribute(
        ctx,
        Diagnostic(
            id="workspace-agentfs",
            title="Overlay workspaces",
            read=lambda: [
                ("isolates", "yes" if result.isolates else "no"),
                ("because", result.because),
                # The accurate columns of the tier are in the Containment
                # section now, through `describe_tier`. A caveat that is
                # repeated in a second section is the drift that the docstring
                # of `TIERS` warns about.
            ],
            order=30,
        ),
    )
    if not result.isolates:
        log.info("ph.seams.workspace_agentfs: declining — %s", result.because)
        return
    ctx.require(WORKSPACE).register_provider(AgentFsProvider(ctx=ctx, root=root))


ORIGIN = "origin"
"""The name of the file that says where an overlay came from: the repository,
then the commit.

It is beside the delta and not on the `Workspace`, because export happens
*after* release. The value must outlive the object, and after a crash it must
outlive the process. It is two lines of text next to the database that they
describe.

**It holds both facts, not only the commit.** Export is reached from a
`WorkspaceRecord`. The `root` of that record is the mountpoint, and the record
has never carried the tree that the overlay was taken from. Without the
repository written down here, the one caller that matters cannot ask. The
commit alone was enough only while the sole caller was a test that already had
the path.
"""

_DIFF_LINE = re.compile(r"^([AMD]) (\w) (/.*)$")
"""A row of `agentfs diff`: an operation, a type, and an absolute path in the overlay.

**The pattern accepts any type letter, not a list of the letters that were
known when this was written.** A row that does not match is treated as absent.
So a narrower pattern would silently drop whatever AgentFS learns to report
next. `_apply` refuses a letter that it does not understand. That turns a new
type into a refusal that names it, and not into work that quietly fails to
arrive.
"""


ExportRefusal: TypeAlias = Literal[
    "no-base-commit", "not-a-repository", "branch-exists", "nothing-to-export", "git", "overlay"
]
"""Why an export could not run, as a code and not as prose.

The argument is that of `DeclineReason`, applied to the operation beside it: a
caller that must branch on the answer cannot parse an English sentence. This is
a vocabulary of its own, and its members are not added to `DeclineReason`. The
one name that they would have shared already means something else there. The
`branch-in-use` of the worktree tier means "a checkout holds this branch". The
`branch-exists` here means "the ref exists at all". One code with two meanings
is worse than two codes.
"""


class ExportRefused(Exception):
    """The work of an overlay cannot be put on a branch, and why.

    It carries a named reason, for the same reason that `WorkspaceDeclined`
    does. A failed export could mean that the base was not a repository, that
    the branch was taken, or that the delta was already discarded. An export
    that did not say which would leave an operator with work that is visible in
    a database, and with no idea how to get it out.
    """

    def __init__(self, message: str, *, reason: ExportRefusal) -> None:
        super().__init__(message)
        self.reason = reason


async def export_overlay(ctx: Context, *, store: Path, identifier: str, ref: str) -> str:
    """Put the delta of an overlay on a git branch rooted at the commit it was taken from.

    **Git detects the conflicts, and the design rests on that.** To compare
    file mtimes and refuse when a target has changed underneath is a worse
    `git merge`, written by hand. It cannot tell an edit to a different
    function from an edit to the same line. So it refuses work that would have
    merged cleanly, and it accepts work that silently loses the work of
    somebody else. To root a branch at the base commit and commit the delta
    onto it gives the question to a three-way merge. It also leaves the
    operator a branch, which is what the worktree tier gives. So there is one
    way to merge work back and not two.

    **It is deliberate, and it never runs on release.** Disposal is for ending
    things, not for applying them (P6-28).

    **There are two phases, because `diff` and the mount cannot coexist.** The
    delta database takes a single writer, so `agentfs diff` fails with a
    locking error while the overlay is mounted. The changeset is read first,
    while the overlay is not mounted. The content is copied afterwards, through
    a temporary mount. That also keeps the copy binary-safe, where
    `agentfs fs cat` would send every file through a text pipe and back.

    Returns the branch that it created. The caller merges it, or does not.
    """
    origin = store / ORIGIN
    if not await anyio.to_thread.run_sync(origin.is_file):
        raise ExportRefused(
            "this overlay was not taken from a git repository, so there is no commit "
            "to root a branch at",
            reason="no-base-commit",
        )
    # Read line by line, to match how `_record_base` writes it. A file that
    # some older build truncated is refused, and does not raise `IndexError`
    # (J5).
    recorded = (await anyio.to_thread.run_sync(origin.read_text)).splitlines()
    if len(recorded) != 2:
        raise ExportRefused(
            f"the origin recorded for this overlay is unreadable ({origin})",
            reason="no-base-commit",
        )
    repo, commit = Path(recorded[0]), recorded[1]

    code, _, _ = await git(ctx, repo, "rev-parse", "--show-toplevel")
    if code != 0:
        raise ExportRefused(f"{repo} is no longer a git repository", reason="not-a-repository")

    code, _, _ = await git(ctx, repo, "rev-parse", "--verify", f"refs/heads/{ref}")
    if code == 0:
        raise ExportRefused(
            f"branch {ref} already exists; merge or delete it before exporting again",
            reason="branch-exists",
        )

    changes = await _changeset(ctx, store, identifier)
    if not changes:
        raise ExportRefused("the overlay has no changes to export", reason="nothing-to-export")

    tree = store / "export"
    await anyio.to_thread.run_sync(lambda: shutil.rmtree(tree, ignore_errors=True))
    code, _, err = await git(ctx, repo, "worktree", "add", "-b", ref, str(tree), commit)
    if code != 0:
        raise ExportRefused(first_line(err) or "could not create the export branch", reason="git")

    # **One owner for the whole life of the checkout.** The removal used to be
    # written at two exits and was missing from a third. An `_apply` that
    # raised left both the worktree *and* the branch behind. The existing
    # branch then made every retry refuse with `branch-exists`. `git branch -D`
    # refused too, because a registration still pointed at the tree. The
    # operator was told to "merge or delete it" and could do neither.
    try:
        mount = store / "export-mnt"
        refused = await open_overlay(ctx, store, identifier, mount)
        if refused:
            raise ExportRefused(refused, reason="overlay")
        try:
            await anyio.to_thread.run_sync(lambda: _apply(changes, mount, tree))
        finally:
            with releasing():
                await unmount(ctx, mount)

        await git(ctx, tree, "add", "-A")
        await git(ctx, tree, *COMMIT_AS_PH, "-m", f"{ref}: overlay export")
    finally:
        # The worktree goes and the branch stays. The branch *is* the
        # deliverable. A checkout left behind is a second copy of work that is
        # now in git, and also a registration that blocks the branch.
        with releasing():
            await git(ctx, repo, "worktree", "remove", "--force", str(tree))
    return ref


async def _changeset(ctx: Context, store: Path, identifier: str) -> list[tuple[str, str, str]]:
    """`agentfs diff`, parsed. It works only while the overlay is not mounted,
    because the database takes one writer."""
    code, out, err = await agentfs(ctx, store, "diff", identifier)
    if code != 0:
        raise ExportRefused(
            first_line(err) or "could not read the overlay's changes", reason="overlay"
        )
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        found = _DIFF_LINE.match(line.strip())
        if found is not None:
            rows.append((found.group(1), found.group(2), found.group(3).lstrip("/")))
    return rows


def _apply(changes: list[tuple[str, str, str]], mount: Path, tree: Path) -> None:
    """Copy one changeset out of a mounted overlay and into a checkout.

    Deletions come last, so a path that is removed and re-added lands in that
    order. Within each group the paths ascend, so an `A d` exists before its
    entries do. A `D d` therefore runs *before* the files beneath it. That is
    harmless: `rmtree` takes the subtree, and the `unlink(missing_ok=True)` of
    the child then does nothing.
    """
    for operation, entry, relative in sorted(changes, key=lambda row: (row[0] == "D", row[2])):
        target = tree / relative
        if operation == "D":
            if entry == "d":
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            continue
        if entry == "d":
            target.mkdir(parents=True, exist_ok=True)
            continue
        if entry not in ("f", "l"):
            raise ExportRefused(
                f"agentfs diff reported {relative!r} as type {entry!r}, which this "
                "export does not know how to carry",
                reason="overlay",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        # `follow_symlinks=False` matters here, and `workspace_provision`
        # copies with the same argument. The default copies a symlink as a
        # *copy of its target* and raises on a dangling one. An export would
        # then flatten links that it should preserve and abort on links that
        # it should carry. With the argument, `l` and `f` need no separate
        # branch, because the link is recreated.
        shutil.copy2(mount / relative, target, follow_symlinks=False)
