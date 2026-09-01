"""`workspace-agentfs` — a copy-on-write overlay as a second workspace provider (P6-21).

AgentFS presents the host tree as a **read-only base** and lands every change in a
per-agent delta layer, so an agent gets an isolated view of the whole directory
without a checkout.

**It is a peer of `worktree`, not a rung above it.** A `mount`-based overlay does
not confine the *process*: a command whose cwd is inside the mount can write to
an absolute path outside it and the write lands on the host for real. Its honest
columns are the worktree tier's verbatim — it bounds tool-mediated and
relative-path writes, because both resolve against cwd, and does not bound an
absolute-path raw write. So `tier` is `worktree` here, and the rung between
`worktree` and `sandbox` is left for something that actually confines.

**The one column that differs cuts against the overlay**: a relative write into a
worktree *lands*, and git can see it; the same write into an overlay lands in the
delta, so the agent believes work is durable that no one else will ever see.
`discards_writes` is therefore true of `overlay-ephemeral`; an `overlay` keeps its
delta so the work can be exported afterwards.

**`mount`, not `run`, and not `exec`.** `run` is the mode that would confine, and
it is unavailable on any host with
`kernel.apparmor_restrict_unprivileged_userns=1` (the Ubuntu 23.10+ default).
`exec` mounts and runs in one shot, which would suit `acquire` exactly, and fails
on a brand-new agent with nothing mounted. So this provider owns a mount lifecycle
rather than wrapping a command.

**The probe gates registration, not acquisition**, and it asserts *isolation*
rather than availability. `register_provider` is `claim_slot`: exclusive, so a row
that claims the slot and then declines every `acquire` falls back to `shared` —
advisory — and a deployment that asked for more isolation would silently get
none. Availability is the wrong question anyway, because
`agentfs run --experimental-sandbox` exits 0, prints nothing, and writes straight
through to the host. The probe writes a canary through a throwaway overlay and
registers only if the host copy is untouched.

**Nothing is ever installed.** A missing binary or an unavailable backend is a
decline that says so, in `ph doctor`, through the diagnostics seam.

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

from ..cordis import Context, plugin
from ..paths import default_home_path
from ..wire import WireModel
from .containment import TIERS, TierDescription
from .diagnostics import Diagnostic, contribute
from .subprocess import SubprocessSpawnSpec, first_line, scrub_env
from .workspace import (
    ContainmentTier,
    Workspace,
    WorkspaceAccess,
    WorkspaceDeclined,
    WorkspaceRecord,
    discards_writes,
    redirection_env,
)
from .workspace_git import git, sanitize_ref

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
"""AgentFS refuses an id that is not alphanumeric, hyphen or underscore.

Refuses, not sanitizes — `Invalid agent ID` is an error, not a warning — so the
mapping happens here rather than being discovered at the first acquire.
"""


def fs_id(session_id: str, agent_id: str) -> str:
    """One AgentFS id per (session, agent), inside its own alphabet."""
    return _UNSAFE.sub("_", f"{session_id}-{agent_id}")


def store_for(root: Path, session_id: str, agent_id: str) -> Path:
    """Where one agent's delta and mountpoint live — `<root>/<session>/<agent>`.

    **`sanitize_ref`, not a second alphabet.** The worktree tier lays its
    checkouts out exactly this way with exactly this mapping, and `/workspaces`
    already records that the id→path mapping is lossy and cannot be inverted —
    two different lossy rules for one layout would be two inversions to be wrong
    about. `fs_id` next door is a genuinely different question, because AgentFS
    refuses ids git would accept.
    """
    return root / sanitize_ref(session_id) / sanitize_ref(agent_id)


async def open_overlay(ctx: Context, store: Path, identifier: str, mountpoint: Path) -> str:
    """Mount one overlay. `""` when it is up, else the reason it is not.

    **Both halves of the success rule, in one place.** A mount is up when the
    command exited zero *and* something is actually mounted there — `agentfs run
    --experimental-sandbox` is the standing reminder that an exit code alone
    proves nothing — and the three callers that needed this were each spelling
    the pair, and the two operator-facing strings, for themselves.

    A reason rather than an exception, because the three callers raise different
    things: a probe result, a `WorkspaceDeclined`, an `ExportRefused`.
    """
    await anyio.to_thread.run_sync(lambda: mountpoint.mkdir(parents=True, exist_ok=True))
    code, _, err = await agentfs(ctx, store, "mount", identifier, str(mountpoint))
    if code == 0 and await is_mount(mountpoint):
        return ""
    return first_line(err) or "the overlay did not mount"


async def run(ctx: Context, program: str, cwd: Path, *args: str) -> tuple[int, str, str]:
    """One AgentFS invocation, through `ctx.subprocess` — never `os.system`.

    `git`'s reasoning one module over: the seam scrubs the credential-shaped
    environment (F1) and reaps the child in a `finally` (F4).

    `LC_ALL=C` because stderr is gettext-translated and the seam passes `LANG`
    through, so a refusal reason read off a tool's words would otherwise depend on the
    operator's locale.

    **`cwd` is load-bearing rather than incidental.** `agentfs init` writes its delta
    database to `./.agentfs/<id>.db`, relative to the working directory — so where
    this is run from decides where an agent's overlay is stored, and running it from
    the agent's own directory is what keeps two agents' deltas apart on disk as well
    as in the filesystem they see.
    """
    spec = SubprocessSpawnSpec(argv=(program, *args), cwd=cwd, env=scrub_env(extra={"LC_ALL": "C"}))
    result: tuple[int, str, str] = await ctx.subprocess.run(spec)
    return result


async def agentfs(ctx: Context, cwd: Path, *args: str) -> tuple[int, str, str]:
    """One AgentFS invocation. `run` with the program named once."""
    return await run(ctx, "agentfs", cwd, *args)


async def unmount(ctx: Context, mountpoint: Path) -> bool:
    """Drop a FUSE mount. `True` if the path is no longer mounted afterwards.

    `fusermount3` because that is what an unprivileged user has: `umount(8)`
    needs `CAP_SYS_ADMIN` for anything that is not in `/etc/fstab`, and the whole
    point of the FUSE path is that it needs no privilege. `fusermount` is the
    same tool on a host still shipping FUSE 2 — the fallback that was here named
    `umount`, which the sentence above says cannot work.

    Not raising on failure, because this runs at teardown: a mount that will not
    release is a leak worth logging and not a reason to fail a disposal that has
    other work to do.
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
    """Whether a FUSE filesystem is actually mounted here.

    `os.path.ismount` rather than "does the directory have anything in it": an
    empty base is legal, and a mount that failed silently leaves an empty
    directory that reads exactly the same way.

    Off the event loop, because this is the one call in this module most likely
    to block for real: it stats a FUSE mountpoint whose server may be wedged, and
    everything cheaper here was already being threaded.
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

    A reason on both outcomes, because the interesting case for an operator is
    the decline: "why am I on worktrees" is answerable only if the row that
    declined said what it tried.
    """

    isolates: bool
    because: str


async def probe_overlay(ctx: Context, scratch: Path) -> OverlayProbe:
    """Write a canary through a throwaway overlay and see whether the host moved.

    **Isolation, not availability.** A backend can exit 0, print no error, and write
    straight through to the host — sandboxing only its own mount — so a probe that
    checked the exit code would claim the tier while the agent had none. This claims
    the tier only if the host copy is unchanged after a write through the overlay.

    Runs once, at mount, because the answer is a property of the host: a kernel that
    refuses namespaces or a FUSE backend that will not start does not change its mind
    between agents, and re-probing per acquire is how a profile that declines pays for
    the tier it is not using.
    """
    if shutil.which("agentfs") is None:
        return OverlayProbe(False, "agentfs is not installed")

    work = scratch / "agentfs-probe"
    base, mount = work / "base", work / "mnt"

    def prepare() -> None:
        # Defensive against a predecessor that died mid-probe; the `finally`
        # below is what cleans up after *this* one.
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
            # The `--experimental-sandbox` shape: the command succeeded and the
            # host moved anyway. Refusing here is the whole point of the probe.
            return OverlayProbe(False, "a write through the overlay reached the host")
        return OverlayProbe(True, "writes land in a delta layer; the host base is unchanged")
    except Exception as error:  # pragma: no cover - a host that fails in a new way
        return OverlayProbe(False, f"{type(error).__name__}: {error}")
    finally:
        if await is_mount(mount):
            await unmount(ctx, mount)
        await anyio.to_thread.run_sync(lambda: shutil.rmtree(work, ignore_errors=True))


@dataclass(slots=True)
class AgentFsProvider:
    """The `overlay` kind: one copy-on-write view of the tree per agent."""

    ctx: Context
    root: Path
    """Where deltas and mountpoints live — `$PH_HOME/overlays/<session>/<agent>`,
    outside the tree being overlaid, so a mountpoint is never itself a candidate
    for the walk the agent runs over its own workspace."""
    probe: OverlayProbe | None = None
    """What the registration probe found, kept so nobody has to re-run it.

    On the provider because `apply` has already paid for it and the answer is a
    property of the host: a caller that wants to explain why the overlay is or is
    not in use should read this rather than mounting a second throwaway one.
    """
    tier: ContainmentTier = field(default="worktree", init=False)
    """**A peer of the worktree tier, not a rung above it.**

    An overlay bounds exactly what a worktree bounds — writes that resolve against cwd
    — and misses exactly what a worktree misses, an absolute-path raw write. The rung
    between `worktree` and `sandbox` belongs to something that confines the process;
    naming it here would make `ph doctor` overstate this.
    """

    def describe_tier(self) -> TierDescription:
        """What an overlay actually bounds — which is not what its rung's row says.

        `bounds` is the worktree tier's verbatim, which is the measured result rather
        than a convenience. The other two columns are where the rung's stock text was
        false: an overlay's writes reach nobody until somebody exports them, and
        `TIERS["worktree"]` sells "per-run checkpoints, /revert", which an overlay has
        none of — `write-tree` against a FUSE mountpoint has nothing to hash, so no
        restore point is ever written. Advertising a mechanism the mounted tier does not
        have, in the one place a person looks to check, is the failure E1 is about.
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

        **`write` and `read` split exactly as the worktree tier splits them.** A writer
        keeps its delta at release so the work can be exported onto a branch; a reader's
        delta is thrown away. That mapping is the whole reason `write` is servable at all
        — an overlay with no merge-back would hand a root agent a tree whose work
        silently evaporates.

        **The base commit is recorded here, not at export**, and that is what makes the
        export a real merge rather than an overwrite: a branch rooted at whatever `HEAD`
        happens to be when somebody exports would present the agent's work as though it
        had been written against a tree it never saw. Nothing to record when the base is
        not a repository — `export_overlay` refuses those, and says so.
        """
        ephemeral = access == "read"
        identifier = fs_id(session_id, agent_id)
        store = store_for(self.root, session_id, agent_id)
        mount = store / "mnt"
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
            kind="overlay-ephemeral" if ephemeral else "overlay",
            # True, and for `worktree-ephemeral`'s reason: the agent writes the
            # whole tree freely. That the writes reach nobody is `kind`'s job to
            # say, not this flag's — `False` here would be a confinement claim
            # only a sandbox backend can make.
            repo_writable=True,
            env=redirection_env(scratch),
            # `discards_writes(kind)`, not the captured `ephemeral`: the seam's
            # own predicate is what the retention policy reads, and spelling the
            # rule a second way here is how the two come to disagree. It is also
            # `GitWorktreeProvider`'s polarity rather than its De Morgan twin.
            release=lambda workspace: self._release(
                store, discard=discards_writes(workspace.kind) and not workspace.retained
            ),
        )

    async def _record_base(self, store: Path, base: Path) -> None:
        """Remember where this overlay came from, if it came from a repository.

        Gated on a `stat` rather than asked of git unconditionally: on a base that
        is not a repository the spawn fails just as slowly as it succeeds, which is a
        process per agent bought on the one branch where it can return nothing.
        """
        if not await anyio.to_thread.run_sync((base / ".git").exists):
            return
        code, out, _ = await git(self.ctx, base, "rev-parse", "--show-toplevel", "HEAD")
        lines = out.split()
        if code != 0 or len(lines) != 2:
            return
        await anyio.to_thread.run_sync(
            lambda: (store / ORIGIN).write_text("\n".join(lines), encoding="utf-8")
        )

    async def export(self, record: WorkspaceRecord) -> str:
        """Assemble this overlay's delta into a branch, and name it.

        The record locates everything: `root` is the mountpoint, so the delta is
        its parent, and the repository and commit were written beside it at
        acquire. The branch is named the way the worktree tier names its own, so
        an operator reading `git branch` after the fact cannot tell which tier
        produced which — and does not need to.
        """
        return await export_overlay(
            self.ctx,
            store=Path(record.root).parent,
            identifier=fs_id(record.session_id, record.agent_id),
            ref=f"ph/{sanitize_ref(record.session_id)}/{sanitize_ref(record.agent_id)}",
        )

    async def _release(self, store: Path, *, discard: bool) -> bool:
        """Drop the mount, and the delta with it unless the tree is evidence.

        Unmounting always, keeping conditionally: a live FUSE mount outlives the
        process that made it, so leaving one behind leaks a mount table entry and
        a server process, where leaving a delta database behind costs disk. The
        two are not the same kind of leak and are not decided together.

        `retained` is read off the workspace at teardown rather than captured at
        acquire, for P6-28's reason: whether this tree is evidence depends on how
        the agent *ended*, which nobody knows when it starts.
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
        """Release an overlay this process never acquired (F6).

        **Without this the leak is reported forever and never closed.** An overlay folds
        into `workspace_survivors` like any fresh-root kind, so reconciliation finds the
        pair — and then `isinstance(provider, ReclaimingProvider)` is False, so it logs
        "no mounted tier can reclaim" and leaves the pair open for every future session to
        report again, while the retention summary goes on telling people to run a `gc`
        that collects nothing.

        Worse here than for a checkout: a crash skips `_release` entirely, and a live FUSE
        mount outlives the process that made it — so what is left behind is a mount table
        entry and a server, not a directory.

        The record locates itself: `root` is the mountpoint, so the delta is its parent.
        Same disposal policy as an orderly release, for the reason the worktree tier
        gives.
        """
        return await self._release(
            Path(record.root).parent,
            discard=discards_writes(record.kind) and not record.reason,
        )


class Config(WireModel):
    """Row config for the overlay provider."""

    root: str | None = None
    """Where deltas and mountpoints live. `$PH_HOME/overlays` by default, and
    outside the tree being overlaid for the same reason a worktree is outside the
    repository: a mountpoint inside `base` would be walked by the agent's own
    `glob` and nested one level deeper by every child."""


@plugin("workspace-agentfs", inject=["workspace", "subprocess"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Probe the host, and claim the workspace slot only if the overlay isolates.

    **The probe gates registration, not acquisition**, which is easy to get
    backwards and expensive when it is. `register_provider` is `claim_slot`:
    exclusive, one provider. A row that took the slot and then declined every
    acquire would not fall back to the worktree tier — it would fall back to
    `shared`, which is advisory, so a deployment that asked for *more* isolation
    would silently end up with none.

    The result reaches `ph doctor` either way, because "why am I on worktrees" is
    a question a person asks of the tool rather than of the source, and a row that
    declined without saying so is indistinguishable from a row nobody mounted.
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
                # The tier's honest columns live in the Containment section now,
                # through `describe_tier` — a caveat repeated in a second section
                # is the drift `TIERS`'s own docstring warns about.
            ],
            order=30,
        ),
    )
    if not result.isolates:
        log.info("ph.seams.workspace_agentfs: declining — %s", result.because)
        return
    ctx.workspace.register_provider(AgentFsProvider(ctx=ctx, root=root, probe=result))


ORIGIN = "origin"
"""Filename holding where an overlay came from: the repository, then the commit.

Beside the delta rather than on the `Workspace`, because export happens *after*
release — the value has to outlive the object, and on a crash it has to outlive
the process. Two lines of text next to the database they describe.

**Both facts, not just the commit.** Export is reached from a `WorkspaceRecord`,
whose `root` is the mountpoint and which has never carried the tree the overlay
was taken from; without the repository written down here, the one caller that
matters cannot ask. The commit alone was enough only while the sole caller was a
test that already had the path in hand.
"""

_DIFF_LINE = re.compile(r"^([AMD]) (\w) (/.*)$")
"""`agentfs diff`'s rows: an operation, a type, and an absolute path in the overlay.

**Any type letter, not a list of the ones known when this was written.** A row
that does not match is a row that is not there, so a narrower pattern drops
whatever AgentFS learns to report next — silently. `_apply` refuses a letter it
does not understand instead, which turns a new type into a refusal naming it
rather than into work that quietly fails to arrive.
"""


ExportRefusal: TypeAlias = Literal[
    "no-base-commit", "not-a-repository", "branch-exists", "nothing-to-export", "git", "overlay"
]
"""Why an export could not run, as a code rather than prose.

`DeclineReason`'s argument, for the operation next door: a caller that has to
branch on the answer cannot parse an English sentence. A vocabulary of its own
rather than members bolted onto `DeclineReason`, because the one name they would
have shared already means something else there — the worktree tier's
`branch-in-use` is "a checkout holds this branch", and this one's is "the ref
exists at all". One code with two meanings is worse than two codes.
"""


class ExportRefused(Exception):
    """An overlay's work cannot be put on a branch, and why.

    A named reason for `WorkspaceDeclined`'s reason: an export that failed
    without saying whether the base was not a repository, the branch was taken,
    or the delta was already discarded leaves an operator with work they can see
    in a database and no idea how to get it out.
    """

    def __init__(self, message: str, *, reason: ExportRefusal) -> None:
        super().__init__(message)
        self.reason = reason


async def export_overlay(ctx: Context, *, store: Path, identifier: str, ref: str) -> str:
    """Put an overlay's delta on a git branch rooted at the commit it was taken from.

    **Git detects the conflicts, and that is the whole design.** Comparing file mtimes
    and refusing when a target has moved underneath is a worse `git merge` written by
    hand: it cannot tell an edit to a different function from an edit to the same
    line, so it refuses work that would have merged cleanly and accepts work that
    silently loses somebody's. Rooting a branch at `base-commit` and committing the
    delta onto it hands the question to a three-way merge, and leaves the operator a
    branch — which is what the worktree tier gives them, so the merge-back story is
    one story rather than two.

    **Deliberate, never on release**: disposal is for ending things, not for applying
    them (P6-28).

    **Two phases, because `diff` and the mount cannot coexist.** The delta database
    takes a single writer, so `agentfs diff` fails with a locking error while the
    overlay is mounted. The changeset is read first, unmounted, and the content copied
    afterwards through a temporary mount — which also keeps the copy binary-safe,
    where `agentfs fs cat` would round-trip every file through a text pipe.

    Returns the branch it created. The caller merges it, or does not.
    """
    origin = store / ORIGIN
    if not await anyio.to_thread.run_sync(origin.is_file):
        raise ExportRefused(
            "this overlay was not taken from a git repository, so there is no commit "
            "to root a branch at",
            reason="no-base-commit",
        )
    recorded = (await anyio.to_thread.run_sync(origin.read_text)).split()
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

    # **One owner for the checkout's whole life.** The removal was written at
    # two exits and missing from a third: an `_apply` that raised left both the
    # worktree *and* the branch behind, and the branch's existence then made
    # every retry refuse `branch-exists` — while `git branch -D` refused too,
    # because a registration still pointed at the tree. The operator was told to
    # "merge or delete it" and could do neither.
    try:
        mount = store / "export-mnt"
        refused = await open_overlay(ctx, store, identifier, mount)
        if refused:
            raise ExportRefused(refused, reason="overlay")
        try:
            await anyio.to_thread.run_sync(lambda: _apply(changes, mount, tree))
        finally:
            await unmount(ctx, mount)

        await git(ctx, tree, "add", "-A")
        await git(
            ctx,
            tree,
            "-c",
            "user.name=pH",
            "-c",
            "user.email=ph@localhost",
            "commit",
            "--no-verify",
            "-m",
            f"{ref}: overlay export",
        )
    finally:
        # The worktree goes, the branch stays — the branch *is* the deliverable,
        # and a checkout left behind is both a second copy of work that now lives
        # in git and a registration that wedges the branch.
        await git(ctx, repo, "worktree", "remove", "--force", str(tree))
    return ref


async def _changeset(ctx: Context, store: Path, identifier: str) -> list[tuple[str, str, str]]:
    """`agentfs diff`, parsed. Unmounted only — the database takes one writer."""
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

    Deletions last, so a path that is removed and re-added lands in that order,
    and ascending within each group so an `A d` exists before its entries do.
    A `D d` therefore runs *before* the files beneath it, which is harmless:
    `rmtree` takes the subtree and the child's `unlink(missing_ok=True)` is then
    a no-op.
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
        # `follow_symlinks=False`, which `workspace_provision` next door calls
        # "the load-bearing argument": the default copies a symlink as a *copy of
        # its target* and raises outright on a dangling one, so an export would
        # flatten links it should preserve and abort on links it should carry.
        # With it, `l` and `f` need no separate branch — the link is recreated.
        shutil.copy2(mount / relative, target, follow_symlinks=False)
