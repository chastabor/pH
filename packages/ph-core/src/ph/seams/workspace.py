"""`ctx.workspace`: where an agent's writes land, and how accurately that is stated.

The containment ladder is built on this seam (D21, §4.8). Its consumer is the
**agent lifecycle**, and not a tool. An agent acquires a workspace. The root of
`ctx.fs` and the cwd of `ctx.subprocess` then resolve to `workspace.root`. That
is what makes a tier bound *authored* code, where it would otherwise only
observe it.

Invariants that this seam holds:

* **`repo_writable` records which guarantee was obtained, and never which was
  requested.** A caller that asks for `access="read"` gets the strongest kind
  that the mounted tier can really provide. `False` means that a tier enforces
  it. Any wording that blurs request and guarantee is a defect (§12 Q10),
  whether it is here, in `phern doctor`, or in a config comment.
* **There is always a workspace.** `acquire` never fails and never returns
  `None`. A provider that cannot serve a request *declines*, and the seam falls
  back to `shared` with a logged notice.
* **A workspace is an effect of the scope that took it (I2).** `acquire`
  registers its teardown through `ctx.effect`, so a disposed agent scope
  unwinds the workspace. That is the in-process half of cleanup. The
  `workspace/acquired`/`disposed` pair is the crash half, and it is reconciled
  at session open (§4.9).
* **`scratch` is always present and always writable**, on every kind and every
  tier. The *seam* creates it, so there is one implementation and not one for
  each provider. It is in the state directory of pH and not in the workspace,
  so it survives disposal as a session artifact. A provider receives the path
  and may substitute its own, but it never has to invent the layout.
* **The kind predicates below are exhaustive `match` statements, and never
  membership tests.** A seventh `WorkspaceKind` therefore fails to type-check.
  It cannot be classified silently.

@module ph.seams.workspace
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

import anyio
from pydantic import Field, ValidationError

from ..agent.types import AgentHandle, PreStepDecision, PreStepRequest
from ..cordis import (
    Context,
    Disposer,
    InactiveScopeError,
    Next,
    Releaser,
    Running,
    maybe_await,
    plugin,
    releasing,
    running,
    safe_yaml_load,
)
from ..json import JsonValue, as_seq, as_str
from ..keys import AGENTS, CONTAINMENT, FS, SESSION_PERSISTENCE, SESSIONS, TOOLS, WORKSPACE
from ..paths import canonical, default_home_path
from ..persistence.families import descendants
from ..session import Session, SessionEvent, intents_of, session_written
from ..session.kinds import WORKSPACE_RESTORE, restore_settled
from ..session.writers import log_writer
from ..tools.definition import ToolExecution, ToolExecutionResult
from ..tools.errors import HarnessError, error_message
from ..wire import WireModel, literal_lookup, validation_summary

if TYPE_CHECKING:
    # For annotations only. `ph/persistence/__init__.py` imports the JSONL
    # backend. An unguarded import would therefore load that backend, `filelock`
    # and `sqlite3` into every process that mounts a workspace.
    from ..persistence.protocol import SessionArchive

from . import workspace_provision
from ._registry import claim_entry, claim_slot
from .diagnostics import Diagnostic, contribute
from .sandbox import SandboxPolicy
from .telemetry import ops_record
from .workspace_provision import ProvisionEntry, ProvisionReport, ProvisionStop

_LOG = log_writer(__name__)

__all__ = [
    "BRANCH_PREFIX",
    "CHECKPOINT",
    "PROJECT_PROVISION_FILE",
    "ArtifactProvider",
    "Backend",
    "CheckpointingProvider",
    "ChildWorkspaceMissing",
    "CollectVerdict",
    "Collectable",
    "ContainmentTier",
    "DeclineReason",
    "EnumeratingProvider",
    "ExportingProvider",
    "LifecycleConfig",
    "LocatingProvider",
    "ReclaimingProvider",
    "SharedWorkspaceProvider",
    "SnapshottingProvider",
    "Stray",
    "VersionedProvider",
    "Workspace",
    "WorkspaceAccess",
    "WorkspaceDeclined",
    "WorkspaceKind",
    "WorkspaceOutcome",
    "WorkspaceProvider",
    "WorkspaceRecord",
    "WorkspaceSeam",
    "WorkspaceSite",
    "apply",
    "checkout_site",
    "checkpoint_policy",
    "checkpoints",
    "confined_env",
    "discards_writes",
    "discover_provisioning",
    "family_survivors",
    "fresh_root",
    "has_restore_point",
    "latest_checkpoint",
    "lifecycle",
    "measure_strays",
    "project_access",
    "redirection_env",
    "sanitize_ref",
    "scratch_of",
    "stored_survivors",
    "tree_path",
    "workspace_leaks",
    "workspace_of",
    "workspace_policy",
    "workspace_survivors",
    "writable_roots",
]

log = logging.getLogger("ph.seams.workspace")

ContainmentTier: TypeAlias = Literal["advisory", "worktree", "sandbox"]
"""The ladder, named in one place. `phern doctor` prints it (P4-12), and
`containment.tier` selects it (P4-11).
"""

WorkspaceKind: TypeAlias = Literal[
    "shared",
    "worktree",
    "worktree-ephemeral",
    "readonly-scratch",
    "overlay",
    "overlay-ephemeral",
]
"""What an agent really got.

* `shared`: one checkout, no isolation. It is the only kind whose root *is* the
  base.
* `worktree`: a branch that belongs to that agent, and that is merged back
  deliberately.
* `worktree-ephemeral`: a full checkout that the agent may write, and whose
  writes **reach nobody**. They are discarded on disposal and never merged.
* `readonly-scratch`: the repository is really unwritable. Only a sandbox
  backend can deliver this kind.
* `overlay` / `overlay-ephemeral`: copy-on-write *views* of the tree. The agent
  may write anywhere and the host sees none of it, because every change lands
  in a delta layer. They divide as the worktree kinds do. `overlay` keeps its
  delta at release, so that the work can be exported onto a branch.
  `overlay-ephemeral` discards it.

An overlay is **not** a flavor of `worktree-ephemeral`. It contains the tree as
it is, with untracked and ignored files included, and its history is not the
history of git. So `workspace_git` must decline it, and must not run
`write-tree` against a mountpoint.
"""

WorkspaceAccess: TypeAlias = Literal["write", "read"]
"""What the *caller* needs of `base`. It is a request and not a guarantee. The
tier decides whether a read-only claim can be enforced. The answer comes back as
`kind` and `repo_writable`.
"""


def project_access(kind: WorkspaceKind) -> WorkspaceAccess:
    """What a workspace of this kind grants of the **project** (E3).

    The answer is about the project and not about the directory. A
    `worktree-ephemeral` may be written freely and merges nothing, so its holder
    was granted `read` of the project. A spawn records the answer as
    `granted_access`, and the parent reads it in the admission of the child.
    `phern doctor` prints the kind and `repo_writable` for each agent, not this
    answer.
    """
    match kind:
        case "shared" | "worktree" | "overlay":
            # `overlay`: its delta survives release and can be exported onto a
            # branch, so what the holder wrote can reach the project. That is
            # deliberate, exactly as a merge is: granting write promises nobody
            # ran it.
            return "write"
        case "worktree-ephemeral" | "readonly-scratch" | "overlay-ephemeral":
            return "read"


def fresh_root(kind: WorkspaceKind) -> bool:
    """Whether this kind gives the agent a directory that is not the base.

    `shared` is the only kind whose root *is* the base. For that reason nothing
    is provisioned into it. Every material is already there, and a copy of
    `.env` onto itself would destroy the file that the provisioning exists to
    provide.
    """
    match kind:
        case "shared":
            return False
        case (
            "worktree" | "worktree-ephemeral" | "readonly-scratch" | "overlay" | "overlay-ephemeral"
        ):
            return True


def discards_writes(kind: WorkspaceKind) -> bool:
    """Whether release discards the writes of the agent, even from a dirty tree
    (P6-28).

    The retention policy reads this predicate. An ordinary release can lose
    *evidence* only for the kinds that answer `True`. So those are the only
    kinds that a policy has reason to retain by default.
    """
    match kind:
        case "shared" | "worktree" | "readonly-scratch" | "overlay":
            return False
        case "worktree-ephemeral" | "overlay-ephemeral":
            return True


def redirection_env(scratch: Path) -> dict[str, str]:
    """Where the toolchain writes its caches and temporary files, in place of the
    workspace (E12).

    Every entry is a cache or a temp location that a build tool writes *beside
    the sources* by default. Each one is pointed inside `scratch`, which is
    outside the workspace and survives disposal. At the `worktree` tier, this is
    what makes `git status` report the work of the agent and not the files of
    `pytest`. So "remove a clean worktree, keep a dirty one" keeps its meaning.

    This is a property of *scratch* and not of worktrees. So it is beside
    `Workspace.env`, and not in the git tier that needed it first. §4.8 gives
    the same env to `readonly-scratch`.

    `PYTEST_ADDOPTS` disables the cache provider completely and also moves
    `--basetemp`. `.pytest_cache/` is written next to `rootdir`, and no
    environment variable relocates it. `TMPDIR` is `scratch` itself. It must
    exist before the first `tempfile` call, and `scratch` is the one directory
    that the seam guarantees.
    """
    return {
        "TMPDIR": str(scratch),
        "PYTHONPYCACHEPREFIX": str(scratch / "pycache"),
        "PYTEST_ADDOPTS": f"-p no:cacheprovider --basetemp={scratch / 'pytest'}",
        "PIP_CACHE_DIR": str(scratch / "pip"),
        "UV_CACHE_DIR": str(scratch / "uv"),
        "GIT_CONFIG_GLOBAL": str(scratch / "gitconfig"),
    }


def writable_roots(workspace: Workspace) -> tuple[Path, ...]:
    """Where this agent may write without being asked (E6).

    This is the one definition of the set. The default rule of `permissions-fs`
    prompts about what falls outside it, and `workspace_policy` gives the same
    set to a backend to enforce. If two spellings drifted apart, the name of a
    tier would promise what its policy does not do.

    `scratch` is always in the set. It is outside the worktree by design (E5),
    and it is the one place where a read-only or ephemeral agent is *told* that
    it may write. A set that named only `root` would prompt on exactly the
    writes that the design invites.

    **The set is canonical by construction, and this function does not resolve
    it.** The kernel matches the path that it resolves. Seatbelt refused a
    workspace spelled `/var/folders/…` its own writes, because that path is
    `/private/var/…` to the kernel. A backend that re-spelled the set privately
    would have made the enforced boundary and the prompted boundary two
    different strings. So the seam makes canonical every input from which it
    builds roots (`acquire`), and this function returns what the seam gave out.
    """
    return (workspace.root, workspace.scratch)


def workspace_policy(workspace: Workspace) -> SandboxPolicy:
    """The workspace as a confinement request: write here, ask about elsewhere.

    The policy is derived from `writable_roots` and does not restate it. So the
    boundary that `ctx.shell` enforces and the boundary that the
    `outside-workspace` rule of `permissions-fs` prompts about cannot drift.

    **No mode** (N1). This function knows *where* the agent writes. It does not
    know whether the user allowed writing at all. When it said
    `workspace-write`, it answered the second question as well as the first.
    That capped a logged `danger-full-access` and floored a
    `defaultMode: read-only`. `SandboxSeam.effective` fills the mode from the
    resolved posture of the session. The roots stand whatever that mode is,
    because `writable_paths` consults them only under `workspace-write`.
    """
    first, *extra = writable_roots(workspace)
    return SandboxPolicy(
        workspace_root=str(first),
        writable_extra=[str(path) for path in extra],
    )


def confined_env(workspace: Workspace) -> dict[str, str]:
    """The redirection that a confined process gets: the redirection of the
    workspace, and a `TMPDIR` in the scratch directory whatever the tier.

    **A confined process has no `/tmp` of its own.** The sandbox lets it write
    only the workspace (under `workspace-write`) and the scratch directory. A
    workspace that is the user's own checkout carries no redirection, so the
    first `tempfile` call failed. That was measured with a model library under a
    read-only sandbox. The scratch directory is the one directory that every
    mode lets the agent write, and it exists. This is one rule for everything
    that confines by `workspace_policy`: a command that `ctx.shell` runs, and a
    code kernel. So neither is left without a `TMPDIR`."""
    return {"TMPDIR": str(workspace.scratch), **workspace.env}


def scratch_of(ctx: Context, session_id: str, agent: AgentHandle | str) -> Path:
    """The scratch directory of this agent: the one that it holds, or the place
    where the seam would put it.

    This is for a row that keeps a per-agent artifact that the tools of the
    agent must reach and write under every tier (the context file of ph-clm).
    `scratch` is that place. A caller may ask before a workspace is acquired, or
    with no workspace row mounted at all. The answer is then the layout of the
    seam under its configured root, or under the default root. So a reader never
    spells the layout a second time.
    """
    held = workspace_of(ctx, agent)
    if held is not None:
        return held.scratch
    seam = ctx.get(WORKSPACE)
    root = seam.scratch_root if seam is not None else default_home_path(None, "scratch")
    agent_id = agent if isinstance(agent, str) else agent.id
    return _scratch_layout(root, session_id, agent_id)


def _scratch_layout(root: Path, session_id: str, agent_id: str) -> Path:
    """One directory for each session *and* each agent. Without the agent part,
    two children of one session would write notes into one directory. The path
    is canonical, for the reason that `_scratch_for` gives."""
    return canonical(root / session_id / agent_id)


def workspace_of(ctx: Context, agent: AgentHandle | str | None) -> Workspace | None:
    """The workspace of this agent, asked of a seam that may not be mounted.

    **A handle, a bare id, or nothing.** The body always accepted all three:
    `rlm-prompt` passes the id that it was given, and the fs resolver passes
    whatever it holds. `Any` hid the fact that this one function serves two
    calling conventions (plan P8-03, issue 3).

    The question is written in one place for all of its callers. The first five
    were the prompt line, `bash`, the kernel, the spawn path and the fs
    resolver.

    **It fails soft on purpose.** A caller may ask "where does this agent write"
    during a teardown, or in a profile that layers no workspace row. It gets
    `None` and continues with the directory of the process. A seam that raised
    here would make an absent optional row fatal.
    """
    seam = ctx.get(WORKSPACE)
    if seam is None or agent is None:
        return None
    agent_id = agent if isinstance(agent, str) else getattr(agent, "id", "")
    try:
        found: Workspace | None = seam.of(agent_id)
    except Exception:
        log.warning("ph.seams.workspace: lookup failed for %s", agent_id, exc_info=True)
        return None
    return found


BRANCH_PREFIX = "ph/"
"""The prefix under which every ref that pH makes is named.

It is here, and not in the git tier that wrote it first, for the same reason
that `CHECKPOINT` moved: **it is the ref namespace of the seam, and not of one
tier**. Three tiers and one command must agree on it. The jj tier names its
bookmarks with it, the overlay tier names its exports with it, and
`/workspaces` enumerates it. While it was in `workspace_git`, it was a
vocabulary that the others borrowed. A deployment that never layers the git
tier also had to import that module for a string constant.

It is shared with `/workspaces`, which enumerates the prefix to find the
artifacts that disposal leaves. A management command that offered to delete a
user's own branches would be a different and much worse tool. This prefix is
all that keeps `/workspaces` from being that tool, so the two must not drift.
"""


_UNSAFE_REF = re.compile(r"[^A-Za-z0-9._-]+")
"""Everything that git refuses in a ref component, plus `/`, which would nest.

Session ids and agent ids come from pH and not from a user. But on most setups
a ref name is a filesystem path under `.git/refs`. So this pattern collapses
the unsafe characters and does not trust the ids.
"""


def sanitize_ref(component: str) -> str:
    """One ref path component, safe by construction.

    The rules of git itself are a deny-list (`git check-ref-format`). This is
    the allow-list. A branch name that fails validation *after* a worktree has
    been created leaves a half-made artifact to clean up.
    """
    cleaned = _UNSAFE_REF.sub("-", component).strip("-.")
    return cleaned or "agent"


EXCLUDE = ":(exclude)"
"""The pathspec magic of git for "everything but this".

It has a name because `workspace_git._commit` reads it back off the pathspec to
learn which entries were provisioned. The two uses are one fact. If the marker
were spelled in two places, a secret would reach a branch on the day that one
of them changed.
"""


def work_pathspec(provisioned: Sequence[str]) -> list[str]:
    """A `git` pathspec that selects a tree *minus* what the seam put in it.

    It is a function of the provisioned list alone. So a live `Workspace` and a
    `WorkspaceRecord` that is read back after a crash build their pathspec in
    one way. When reclaim staged with a pathspec of its own, a provisioned
    `.env` was committed to the branch, on exactly the path that no person
    watches (S13). The two lists differ in one direction: the list of the record
    is what was configured, and the live list is what arrived. To exclude a
    configured path that never arrived is the safe side.

    The positive `.` is required. Exclusions alone match everything, which is
    the opposite of what they appear to say.
    """
    return [".", *(f"{EXCLUDE}{entry}" for entry in provisioned)]


@dataclass(frozen=True, slots=True)
class Workspace:
    """The working directory of one agent, and the facts about what it bounds.

    It is a value. It holds what the provider decided, and nothing about who
    holds it. The seam keeps the bookkeeping: which agent, which session, and
    how to end it. So a provider cannot implement only half of the lifecycle.
    """

    root: Path
    """The agent's cwd: `ctx.fs`'s root and `ctx.subprocess`'s default cwd."""
    scratch: Path
    """Always writable, on every kind and tier, and outside `root` on purpose.
    So it survives disposal even when the workspace itself is discarded. The
    provider receives it already created. A provider substitutes its own path
    only if its tier needs the path somewhere else."""
    kind: WorkspaceKind
    repo_writable: bool
    """Whether `root` can really be written. **It is `False` only when a tier
    enforces it.** It is never a statement of intent, and it is never inferred
    from `access`."""
    ref: str | None = None
    """The git branch, when the kind has one."""
    env: Mapping[str, str] = field(default_factory=dict)
    """The environment that a runner should apply, for a kind that needs
    redirection.

    Empty for `shared`. Every shipped tier that makes a fresh root points
    `TMPDIR`, `PYTEST_ADDOPTS` and the other variables of `redirection_env`
    inside `scratch`, because build tools write into the tree that they run
    against. It is best-effort by construction. A toolchain that insists on
    writing beside its sources still fails. The answer to that is
    `access="write"` for that agent, and not a weaker tier.
    """
    provisioned: tuple[str, ...] = ()
    """Paths that the seam put in this workspace (E14). They are not the work of
    the agent."""
    reused: bool = False
    """Whether the tier returned a tree that it found already made for this
    agent, and did not make one now. Examples are a resume, and a child that
    gets its workspace back.

    The tier sets it, because the tier is the one party that knows. What the
    seam provisioned when the tree was made is still in it, and the agent may
    have edited it since. So the seam provisions only what is missing
    (`provision(fill=True)`). A full copy overwrote those edits."""
    provision_failures: tuple[str, ...] = ()
    """Materials that the seam could not put in place (E14).

    They are on the value and not in a log line, because the party that must
    know that `.env` is missing is the *agent*, which is about to wonder why the
    tests fail. The workspace prompt line reads them directly. Empty is the
    ordinary case, and that includes "this profile provisions nothing"."""
    retained: str = ""
    """Why this tree is kept. Whoever learns the outcome sets it late (P6-28).

    It is late state, and not a field set at acquire or an argument of
    `release`. Nobody knows at acquire how the child will end. And `release`
    runs as a *scope disposer*, so it is told nothing about the reason.

    It is a **reason** and not a flag. That lets the fold tell apart three
    states that all leave a tree on disk. A deliberate keep gives its reason
    here. A dirty-tree keep is `kept` with no reason. A leak has no `disposed`
    event at all.
    """
    fork: str | None = None
    """The commit that this tree started from, for a tier whose disposal has to
    ask what the tree has done *since* (the jj tier). `None` for a tier that
    does not need it.

    It is on the value so that the seam can put it on `workspace/acquired`. A
    crash reclaim has only the record, and without the fork point the jj tier
    cannot tell a tree that did nothing from a tree whose work was frozen into
    a commit for a child. See `WorkspaceRecord.fork`.
    """
    release: Callable[[Workspace], Awaitable[bool]] | None = None
    """The teardown of the provider. It returns whether anything was **kept**.

    It takes the whole workspace and not one field. A teardown policy needs what
    was provisioned *and* what kind the workspace holds. The shipped tiers also
    read `retained` from it. The checkpoint refs of P4-09 were expected to be a
    further such fact.

    Only the teardown can know the answer. The policy of P4-08 is "commit, then
    remove, discard ephemeral even if dirty", so `kind` cannot be asked in its
    place. The seam records the answer on `workspace/disposed`. A reader can
    then tell "nothing changed, so it was removed" from "these writes were
    discarded by design".
    """

    def agent_work_pathspec(self) -> list[str]:
        """A `git` pathspec that selects this tree *minus* what the seam put in it.

        This is the one definition of "the work of the agent". The disposal
        policy of the git tier uses it: `workspace_git._release` gives it to
        `_dirty` and `_commit`. Crash reclaim builds the same pathspec through
        `work_pathspec`, from the list that `workspace/acquired` records. The
        `/revert` of P4-09 reads `provisioned` directly and must not disagree
        with it. Its "restore tracked + untracked-not-ignored" is exactly the
        set that must not clobber a provisioned `node_modules`.
        """
        return work_pathspec(self.provisioned)


DeclineReason: TypeAlias = Literal[
    "not-a-repository",
    "branch-in-use",
    "path-exists",
    "provider-failed",
    "overlay-failed",
]
"""Why a tier could not serve a request, as a code and not as prose.

`phern doctor` prints it (P4-12). An operator who set `worktree` and got
`shared` must be told the reason. And the consumer that must branch on a durable
event cannot parse an English sentence in it.
"""


class WorkspaceDeclined(Exception):
    """A provider that declines *with* a reason. It is never fatal, and the seam
    falls back.

    It is raised and not returned, so that the `acquire` protocol keeps one
    shape. A bare `None` is still a decline. It only cannot say why.
    """

    def __init__(self, reason: DeclineReason, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason: DeclineReason = reason


class ChildWorkspaceMissing(HarnessError):
    """A child reached its first step with no workspace. This is refused, and the
    refusal is loud (§6.5).

    **The lazy acquire in `lifecycle` below is for the user's own agent. If it
    answered for a child, it would widen that child.** The workspace of a child
    is the decision of its *parent*. The base to fork from and the `access` to
    grant both arrive with the spawn and are recorded on its admission. So the
    only thing that this row could do for a child is invent them: the directory
    of the process as `base`, and whatever the profile configured for the *root*
    as `access`, which defaults to `write`. A research child admitted as `read`
    would come back with a writable checkout of a tree that is not its own.
    Every event about it would report that accurately, and would still be wrong
    about what was asked for.

    So the absence is treated as the bug that it is. A spawn path that forgot to
    acquire has a **missing call**. The loud version costs a failed turn that
    names the child. The quiet version costs a child that holds more than its
    admission recorded, and nothing downstream can detect that.
    `workspace/acquired` reports what the child got and not what it should have
    got, and §6.5 is checked at admission, where this never appeared.

    This is not `WorkspaceDeclined`. That is a *tier* that says it cannot serve
    a request, and the seam is right to fall back for it. This is a caller that
    asks a question which this row was never the one to answer.

    It is a `HarnessError`, so that the code survives to the record. Raised from
    `agent/pre-step`, it becomes a recorded `turn/end` with
    `reason.kind: "error"`, which is loud and durable. `error_info` is what puts
    a name in it. A bare exception reaches that record as `UNKNOWN`, and that is
    the prose-only failure that the code exists to avoid. `HarnessError` sets
    `self.code`. That is also the attribute that the `respond` function of the
    daemon reads, so one spelling serves both readers.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, "CHILD_WORKSPACE_MISSING")


@runtime_checkable
class WorkspaceProvider(Protocol):
    """The implementation of a tier. `None` declines, and a decline is normal.

    The provider is typed and not duck-typed, and `tier` is a member and not a
    `getattr` probe. Otherwise a provider whose method drifted would fail at
    runtime inside the `except` of the seam. The operator would see it reported
    as `shared`, and the containment would be lost silently.

    `tier` is a property for the same reason as in `CodeRuntime`: every tier
    declares it as a frozen field, and a settable Protocol attribute refused all
    of them.
    """

    @property
    def tier(self) -> ContainmentTier: ...

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None: ...


@dataclass(frozen=True, slots=True)
class WorkspaceSite:
    """Where a tier will put the tree of an agent, stated before the tier makes
    one (S12)."""

    root: Path
    kind: WorkspaceKind
    ref: str | None = None


@runtime_checkable
class LocatingProvider(Protocol):
    """A tier that can say where a tree will go before it makes the tree (S12).

    **This is what lets the record come first.** The seam writes
    `workspace/acquiring` from this answer and flushes it before the tier acts.
    So a crash inside `worktree add`, the init of an overlay, or its mount
    leaves a record that names the root, and reconcile then reclaims it. Without
    the record, the act ran with nothing in the log, and a mounted overlay that
    nothing named was never unmounted.

    `acquire` must put the tree exactly at this site. For that reason every tier
    that implements this Protocol derives its own `acquire` from `locate`, and
    does not compute the site a second time. `locate` is pure, with no I/O. So a
    tier that is about to decline has cost a record and not a subprocess. This
    is an optional capability, with the shape and the reason of
    `ReclaimingProvider`.
    """

    def locate(
        self, *, session_id: str, agent_id: str, access: WorkspaceAccess
    ) -> WorkspaceSite: ...


def tree_path(root: Path, session_id: str, agent_id: str) -> Path:
    """Where a tier puts the tree of one agent under its own `root`:
    `<root>/<session>/<agent>`.

    **One lossy mapping from id to path for every tier**, through
    `sanitize_ref`. Reconcile and `/workspaces` invert it. A second spelling in
    a second tier would be a second inversion that could be wrong.
    """
    return root / sanitize_ref(session_id) / sanitize_ref(agent_id)


def checkout_site(
    root: Path, session_id: str, agent_id: str, access: WorkspaceAccess
) -> WorkspaceSite:
    """The site of a version-controlled tier: a checkout at `tree_path`, on the
    branch that is named the same way. The checkout is ephemeral when the caller
    asked only to read. The git tier and the jj tier share this function, so a
    record that either tier writes names what the other tier would name."""
    return WorkspaceSite(
        root=tree_path(root, session_id, agent_id),
        kind="worktree-ephemeral" if access == "read" else "worktree",
        ref=f"{BRANCH_PREFIX}{sanitize_ref(session_id)}/{sanitize_ref(agent_id)}",
    )


@runtime_checkable
class ReclaimingProvider(Protocol):
    """A provider that can release a workspace that it did not create (F6).

    **An optional capability as its own Protocol, never a `getattr` probe.**
    `ExportingProvider` and `RehydratableProvider` take the same shape. Not
    every tier can reclaim (an in-memory tier has nothing to reclaim), and a
    probe would report a provider whose method is *misnamed* as one that cannot.
    That hides a leak when it should close it.

    It returns whether anything was **kept**, as `Workspace.release` does. So
    the `workspace/disposed` that a reconciliation writes says what an orderly
    release would say.
    """

    async def reclaim(self, record: WorkspaceRecord) -> bool: ...


@runtime_checkable
class ExportingProvider(Protocol):
    """A provider that can put the work of an agent where the project can see it.

    It is an optional capability, for the reason that `ReclaimingProvider`
    gives. `shared` has nothing to export, and a tier that isolates by
    *discarding* has nothing to offer.

    It returns the git ref that the work is on, so one verb serves all three
    tiers that export: git, agentfs and jj. A worktree answers with the branch
    that it has committed to. A jj workspace answers with the bookmark that has
    tracked its work. An overlay first builds a branch out of its delta.
    `/workspaces` asks the seam, and does not ask which tier it is talking to.
    """

    async def export(self, record: WorkspaceRecord) -> str: ...


@dataclass(frozen=True, slots=True)
class Stray:
    """One checkout that a tier still has on disk, and what its removal would cost.

    The name describes the interesting case and not the common one. Disposal
    commits and takes the checkout back. So a directory that is still there
    means that a live agent is working in it, or that disposal could not remove
    it.
    """

    ref: str
    """The branch or bookmark that this checkout is on. `/workspaces` joins on
    it, because it is the one name that the git tier and the jj tier both put on
    a `Workspace` and carry. They do not derive it from a path."""
    path: Path
    dirty: bool
    """Whether removal of the directory would lose something that the ref does
    not have.

    **Each tier answers in its own terms, and the questions are not the same.**
    A git worktree is dirty when it holds uncommitted changes. The working copy
    of a jj workspace *is* a commit, and its bookmark follows it. So the act of
    asking is also what makes the answer safe. A `True` there means "there is
    work, and it is now on the ref", which is a better answer than git can give.
    """


async def measure_strays(
    found: Sequence[tuple[str, Path]],
    probe: Callable[[Path], Awaitable[bool]],
    *,
    with_status: bool,
    skip: Container[str] = (),
) -> list[Stray]:
    """Turn `(ref, path)` pairs into `Stray`s, and measure `dirty` concurrently.

    It is beside `Stray` because it was the same twelve lines in the git tier
    and in the jj tier: the `dict.fromkeys`, the `with_status` guard, the task
    group and the comprehension. Only the probe differed, and the probe is the
    one part that really belongs to the tier. A change to the fan-out or to
    `Stray` had to be made twice, in files that already share code deliberately.

    It is concurrent because each probe is a subprocess, and a run that left one
    checkout behind has usually left several.

    **`skip` is the refs that a live agent holds. To leave it out cost more
    than time.** `/workspaces` never reads `dirty` for a held row: `describe`
    reports it as `held` and stops. So every probe against a held row was a
    subprocess spent on an answer that nobody reads, one for each live child in
    each listing. On the jj tier it was worse than waste. The probe snapshots,
    so *listing* mutated the working-copy commit of every agent that was in the
    middle of a turn. The old command filtered these rows out, and the filter
    was lost when the code moved behind a Protocol. The filter is here now,
    where both tiers get it.
    """
    dirty = dict.fromkeys((ref for ref, _ in found), False)
    wanted = [(ref, path) for ref, path in found if ref not in skip] if with_status else []
    if wanted:

        async def measure(ref: str, path: Path) -> None:
            dirty[ref] = await probe(path)

        async with anyio.create_task_group() as group:
            for ref, path in wanted:
                group.start_soon(measure, ref, path)
    return [Stray(ref=ref, path=path, dirty=dirty[ref]) for ref, path in found]


@runtime_checkable
class ArtifactProvider(Protocol):
    """A provider that can list, merge and delete the refs that it leaves behind
    (E15).

    It is an optional capability as its own Protocol, with the shape and the
    reason of `ReclaimingProvider`. It exists because `/workspaces` did this
    work itself, **in git**: `git branch --list ph/*`, `git merge`,
    `git branch -d`. But `jj` embeds its own git implementation. *Measured*:
    the whole acquire, bookmark and export flow runs with a `git` on `PATH` that
    exits 127, and the binary is never invoked. So a jj deployment need not have
    the binary. There the branch listing failed, and `/workspaces` answered "no
    agent workspaces are left behind" while the bookmarks were present.

    `refs` returns **everything**, and not only what pH made. The prefix guard
    belongs to the command that would otherwise offer to delete the `feature/x`
    of a user. The second caller is `merge`, which must accept a ref outside the
    prefix.

    Both acting verbs return `""` for "it happened", and otherwise **the
    sentence that a person reads**. They do not return a bool, because the tier
    is the one to say what went wrong, and the git-backed tiers and the jj tier
    do not fail in the same way. A conflicted merge is the case that proves it.
    git exits non-zero and refuses. jj exits **zero** and records the conflict
    in the commit. So a caller that tests an exit code reports a clean merge,
    and a person then discovers the conflict by opening the file.
    """

    async def refs(self, base: Path) -> list[str]: ...

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str: ...

    async def merge(self, base: Path, ref: str) -> str: ...


@runtime_checkable
class EnumeratingProvider(Protocol):
    """A provider that can find and take back the checkouts that it left on disk
    (E15).

    **It is separate from `ArtifactProvider`, and the overlay tier is the
    reason.** The two were one Protocol of five verbs. The justification was
    that the five are the whole surface of `/workspaces`, so a tier serves that
    command or it does not. `workspace-agentfs` showed at once that this was
    false. Its artifact is a git branch like that of any other tier, so it
    answers the ref verbs. But a *mountpoint is not a checkout* that this
    command can hand back. So the tier had to write two stubs to be admitted,
    and `isinstance` could not tell a tier that meant them from a tier that had
    only typed them. There are now two Protocols along the line that the code
    already drew, and the stubs are gone.

    `strays` filters by the **own root** of the provider, and that is all that
    makes `discard` safe. `jj workspace list` reports the user's own `default`
    workspace at the repository root, and a `remove` that trusted the list would
    delete the repository of the user. The filter is also what let the command
    stop carrying a copy of that setting.

    `discard` removes a checkout and nothing else. The ref is the artifact, and
    `/workspaces` deletes that separately and deliberately, behind its own flag.
    """

    async def strays(
        self, base: Path, *, with_status: bool = True, skip: Container[str] = ()
    ) -> list[Stray]: ...

    async def discard(self, path: Path) -> str: ...


@runtime_checkable
class CheckpointingProvider(Protocol):
    """A provider that can name the state of a workspace and put it back
    (P4-09).

    It is an optional capability as its own Protocol, with the shape and the
    reason of `ReclaimingProvider`. Not every tier has a restore mechanism.
    `shared` is the user's own checkout. The delta of an overlay is a good
    restore point, but it is not a git tree. A `getattr` probe would report a
    provider whose method drifted as one that cannot checkpoint. That silently
    loses `/revert` for a tier that has it.

    **`capture` returns one token that serves two readers.** The git tier
    already argued this rule for its tree hash. P4-09 stores the token as a
    restore point. P5-07 uses it as a *fingerprint*, to decide that a quality
    gate which failed against this exact state need not run again. There is one
    derivation, so a gate memo and a restore point can never disagree about
    whether the work changed.

    So what a token must promise is narrow and exact: **equal tokens mean equal
    content**. The converse is *not* required. A tier whose token changes while
    the content stays the same only makes a gate run again, which is the safe
    direction. `workspace-jj` is such a tier, and its own docstring says so.

    `capture` also **pins** what it names, for as long as the workspace lives.
    An unreferenced git tree is eligible for `gc`. A restore point that
    disappears is worse than none, because `/revert` listed it.
    """

    async def capture(self, workspace: Workspace) -> str | None: ...

    async def restore(self, workspace: Workspace, token: str) -> tuple[str, ...]: ...


Backend: TypeAlias = Literal["git", "jj", ""]
"""Which version control backs a tier, or `""` for none of them.

It is here and not in `ph.seams.changes`, which is the module that consumes it.
A tier that states what backs it describes *itself*. To do so, the two tiers had
to import the change filter, which made them depend on an indexing helper. That
dependency pointed the wrong way, and it is the reason why this moved.
"""


@runtime_checkable
class VersionedProvider(Protocol):
    """A provider that knows which version control backs the tree that it gives
    out.

    It is an optional capability as its own Protocol, with the shape and the
    reason of `ReclaimingProvider`. **`Workspace.kind` cannot answer this.** The
    git tier and the jj tier both return `worktree`, so only the provider can
    say. A tier that declares nothing falls through to the filesystem probe of
    `ph.seams.changes`. That is the `shared` case, and therefore the common
    case.
    """

    vcs: Backend


@runtime_checkable
class SnapshottingProvider(Protocol):
    """A tier whose version control commits the tree in order to *read* it.

    jj is the one that does. It snapshots the working copy on any command. That
    is what makes the jj backend of `ph.seams.changes` a single call. It also
    means that a read adopts whatever is untracked. `auto_track` exists to keep
    the provisioned materials of the seam (E14) out of that commit. Only the
    tier can apply it, because only the tier knows what it provisioned.

    So a caller that needs the snapshot asks for it **here**, and does not use
    the raw `jj` helper. The docstring of that helper sends callers through the
    provider for exactly this reason. A tier that does not declare this
    capability is one whose reads cost the tree nothing.
    """

    async def snapshotting(self, cwd: Path, *args: str) -> tuple[int, str, str]: ...


@dataclass(frozen=True, slots=True)
class SharedWorkspaceProvider:
    """`workspace-shared`: the behavior with no tier, and the floor under every
    tier.

    It returns `base` itself. To mount the seam changes nothing: no checkout, no
    copy, no cost. It is also the fallback for a provider that declines. For
    that reason it is beside the seam and not in a row of its own. "There is
    always a workspace" cannot be a promise kept by a row that a profile might
    not layer.

    A request for `access="read"` is answered truthfully and is not enforced.
    The kind stays `shared` and `repo_writable` stays `True`, because nothing
    here enforces anything.
    """

    tier: ContainmentTier = "advisory"

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace:
        return Workspace(root=base, scratch=scratch, kind="shared", repo_writable=True)


@dataclass(slots=True)
class _Held:
    """One live workspace and the disposer that ends it.

    The two are paired here and not on `Workspace`, because the disposer belongs
    to the *scope* and not to the provider. A value that is given to a caller
    should not carry the bookkeeping of the seam.
    """

    workspace: Workspace
    dispose: Releaser | None = None
    session: Session | None = None
    """Where the closing event of this workspace goes. It is set when the
    acquisition is logged. The release closure reads it here and does not
    capture it, so the two halves of the pair cannot disagree about which log
    they belong to.
    """
    copying: _Copy | None = None
    """The provisioning copy running into this tree, while it runs (W1)."""


@dataclass(slots=True, eq=False)
class _Copy:
    """A provisioning copy in progress, as a release in the middle of the copy
    needs it (W1).

    The workspace is held before it is provisioned. So its scope can release it
    while the copy runs on a thread that no cancel reaches. The release then
    read a workspace that said nothing was provisioned. It committed the
    half-copied materials to the branch of the agent as its work, and it removed
    the tree under the copy. Now the release stops the copy, waits for it, and
    counts every one of these targets as provisioned.
    """

    targets: tuple[str, ...]
    """What is being provisioned, whether or not it arrived. All of it belongs
    to the seam, as reclaim after a crash counts it (S13)."""
    stop: ProvisionStop = field(default_factory=ProvisionStop)
    done: anyio.Event = field(default_factory=anyio.Event)


@dataclass(slots=True)
class WorkspaceSeam:
    """The service published as `ctx.workspace`."""

    ctx: Context
    shared: SharedWorkspaceProvider
    scratch_root: Path
    provider: WorkspaceProvider | None = None
    provider_by: Running | None = None
    """Who registered the tier (P6-29). The seam enters it around every call
    into the provider. See `CompactionSeam.engine_by` for the reason why the
    layer stays that of the registration."""
    _provisioning: list[ProvisionEntry] = field(default_factory=list)
    """Materials to put in a fresh workspace, in registration order (E14).

    This is a list and not a `claim_slot`, because two sources legitimately
    compose: the own row of the profile, and the `.ph-workspace.yml` of a
    repository. A later entry wins a collision.
    """
    _held: dict[str, _Held] = field(default_factory=dict)
    """Live workspaces by agent id. The key is the id because the callers that
    ask have an agent id and no agent object (the workspace line of the prompt,
    the per-agent report of `phern doctor`). The effect disposer empties it. So
    an entry that outlives its agent is the same leak that a
    `workspace/acquired` without a `disposed` records durably.
    """

    _claimed: dict[str, anyio.Event] = field(default_factory=dict)
    """Agent ids whose tree is being acquired or reclaimed right now. Each has
    the event that is set when that act is done (J6). **One act on the tree of
    an agent at a time.**

    A reclaim runs a `git worktree remove --force` for each leaked tree.
    Meanwhile an `acquire` can ask the tier for the tree of the *same* agent,
    because the catch-up of the `workspace-reconcile` row and `collect` run
    beside agents that are already working. Either order lost the tree. In one,
    an acquire returned a root that the reclaim was deleting. In the other, a
    reclaim that found no holder deleted the tree that an acquire had not yet
    held: `acquire` awaits the scratch directory, its `acquiring` flush and the
    tier before `_held` names the tree. So an acquire waits for any claim on its
    agent to end and then takes its own. A reclaim leaves any claimed tree
    alone, and takes a claim while it deletes. Each side checks and takes with
    no await between the two steps, so under one event loop each side sees the
    other.

    The key is the agent id and not the root, because that is the identity that
    both sides already have. `acquire` knows it before the tier computes a root,
    and a record carries it. So the wait happens *before* the tree is built and
    not after.
    """

    def of(self, agent_id: str) -> Workspace | None:
        """The workspace that this agent holds, if it has acquired one. `None` is
        a real answer and the common one: nothing acquires until the agent
        lifecycle does (P4-08).
        """
        held = self._held.get(agent_id)
        return None if held is None else held.workspace

    def provision(
        self, entries: Sequence[ProvisionEntry], *, scope: Context | None = None
    ) -> Disposer:
        """Contribute materials for every fresh workspace that this seam gives out.

        This is on the *seam* and not on the tier. So `readonly-scratch` (P6-05)
        and any later fresh-root kind inherit the guards and do not implement
        them again. The same argument put `scratch` here. Nothing is
        provisioned into a `shared` workspace, whose root *is* the base.

        `scope=` registers on the lifetime of *someone else*, and that is all
        that it now means (P6-12, P6-25). A registration made from the `apply`
        of a row, or from a listener that the row wrote, already unwinds with
        the row.

        It goes through `claim_entry` because a `ProvisionEntry` is a
        **value**. Two rows that contribute `{source: .env}` compare equal, and
        with `list.remove` the disposer of one row would remove the entry of
        the other.
        """
        disposers = [
            claim_entry(
                self.ctx.owner_for(scope), self._provisioning, entry, label="workspace.provision"
            )
            for entry in entries
        ]

        def release() -> None:
            for disposer in disposers:
                disposer()

        return self.ctx.owner_for(scope).add_disposer(release, label="workspace.provision")

    def live(self) -> list[Workspace]:
        """Every workspace that an agent currently holds. `/workspaces` needs
        this before it offers to delete a directory.

        Callers match these by ref (`/workspaces`, `strays`) or by root
        (`collectable`). They do not turn a directory name back into an agent
        id. `sanitize_ref` is lossy, so an id that does not sanitize to itself
        would read as unheld and lose the refusal that protects it.
        """
        return [held.workspace for held in self._held.values()]

    def register_provider(
        self, provider: WorkspaceProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim the tier. One provider holds it at a time, and `shared` remains
        the fallback."""
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "provider",
            provider,
            label="workspace.provider",
        )

    def effective_tier(self, *, child: bool) -> ContainmentTier:
        """What one role really gets, with the provider and the choice reconciled.

        **It is effective and not configured, in both directions.** A `worktree`
        row over a directory that is not a repository declines on every
        acquire. And the shipped `rlm` profile layers the git provider while it
        chooses `advisory` for the user's own agent. To read either half alone
        names containment that somebody does not have.

        Each of the two halves can only *lower* the answer, and neither can
        raise it. `acquire` makes the same reconciliation by *doing* it. For
        that reason this is the only other place that may state it.
        """
        if self.provider is None:
            return "advisory"
        containment = self.ctx.get(CONTAINMENT)
        chosen = None if containment is None else containment.for_role(child=child)
        if chosen == "advisory":
            return "advisory"
        return self.provider.tier

    def describe(self) -> list[tuple[str, str]]:
        """What `phern doctor` prints about workspaces (E10).

        **One row for each agent, and not one for the profile.** Since P4-11
        there is no single answer: the shipped `rlm` posture puts the user's own
        agent in the checkout of the user, and its children in worktrees. An
        agent that has acquired nothing prints nothing. The alternative would
        invent a row for each configured agent, for workspaces that nobody
        holds.
        """
        provider = self.provider
        rows: list[tuple[str, str]] = [
            (
                "provider",
                "none — every agent works in place" if provider is None else provider.tier,
            ),
            ("scratch root", str(self.scratch_root)),
        ]
        if self._provisioning:
            materials = ", ".join(entry.dest or entry.source for entry in self._provisioning)
            rows.append(("provisions", materials))
        for agent_id, held in sorted(self._held.items()):
            workspace = held.workspace
            writable = "writable" if workspace.repo_writable else "read-only (enforced)"
            detail = f"{workspace.kind}, {writable}, at {workspace.root}"
            if workspace.ref:
                detail += f" on {workspace.ref}"
            rows.append((f"agent {agent_id}", detail))
        rows.append(("retained trees", self._retained_summary()))
        return rows

    def _retained_summary(self) -> str:
        """How many trees are kept as evidence, across stored sessions (P6-28).

        This is the row that makes the pile visible. The per-agent rows above
        cannot show it. `doctor` mounts a profile with no agents, so every
        retained tree is by definition one that nobody holds any more.

        It is printed **even when the answer is none**, by rule 6. When a row is
        absent, a reader assumes that nothing is accumulating, and this row
        exists to check that assumption.

        It is bounded by whatever `stored()` lists, and does not walk every log
        ever written. So it is a **floor and not a census**, and the row says
        so. `stored_survivors` catches broadly, for the reason that `doctor`
        itself has: a profile that cannot answer one question must still answer
        the rest.
        """
        store = self.ctx.get(SESSION_PERSISTENCE)
        if store is None:
            return "unknown — no session store is mounted"
        survivors, touched = stored_survivors(store)
        found = [record for record in survivors if record.outcome == "retained"]
        if not found:
            return f"none, across the {len(touched)} most recent session(s)"
        sessions = len({record.session_id for record in found})
        # Two lines. This text renders into a two-column table, and a sentence
        # that wraps at 80 columns looks like a stray line.
        return (
            f"{len(found)} across {sessions} of the {len(touched)} most recent session(s)\n"
            "collect them with `phern workspaces gc`"
        )

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        access: WorkspaceAccess = "write",
        session: Session | None = None,
        scope: Context | None = None,
        tier: ContainmentTier | None = None,
    ) -> Workspace:
        """Take a workspace for one agent. Never fails, and never returns `None`.

        **The rung is derived, and the caller is not asked for it** (P4-11).
        The role is already known, because the session of a child says so
        (`origin: "subagent"`). So no caller can forget it and get the provider
        where the shipped profile says that a *root* agent must ask for the
        escalation.

        `tier` overrides that derivation for a caller that means something
        specific, exactly as `cwd` overrides the default of `shell.run`.
        `advisory` declines a registered provider. Any other value consults it.

        `scope` bounds the life of the workspace. It is the own scope of the
        agent. To dispose it releases the workspace and writes the closing
        event, so an error path that never reaches an explicit `dispose` is not
        a leak. **When `scope` is omitted, the workspace still belongs to the
        agent** if `ctx.agents` knows `agent_id` (P4-16). The caller has already
        said whose workspace this is, and the only lifetime that can own the
        checkout of a live agent is the lifetime of that agent. An id that the
        registry has never seen keeps the fallback of `owner_for`.

        **Every root that this gives out is canonical** (`ph.paths.canonical`),
        because the inputs are. `base` is resolved here, `scratch` is resolved
        in `_scratch_for`, and the own root of a provider comes from
        `default_home_path`. The kernel matches resolved paths, and
        `permissions-fs` prompts about the same set. One spelling at the source
        is what keeps those two boundaries one boundary (E6, `writable_roots`).
        """
        base = canonical(base)
        # This comes before anything is built. An act that is in progress on the
        # tree of this agent owns the tree until the act is done (J6).
        while (claimed := self._claimed.get(agent_id)) is not None:
            await claimed.wait()
        done = self._claimed[agent_id] = anyio.Event()
        try:
            return await self._acquire(
                session_id=session_id,
                agent_id=agent_id,
                base=base,
                access=access,
                session=session,
                scope=scope,
                tier=tier,
            )
        finally:
            del self._claimed[agent_id]
            done.set()

    async def _acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        access: WorkspaceAccess,
        session: Session | None,
        scope: Context | None,
        tier: ContainmentTier | None,
    ) -> Workspace:
        """`acquire`, under this agent's claim (`_claimed`)."""
        scratch = await self._scratch_for(session_id, agent_id)
        chosen = self._chosen_tier(session) if tier is None else tier
        workspace = None
        declined: DeclineReason | None = None
        site: WorkspaceSite | None = None
        if self.provider is not None and chosen != "advisory":
            site = await self._acquiring(session, session_id, agent_id, access)
            try:
                with running(self.provider_by):
                    workspace = await self.provider.acquire(
                        session_id=session_id,
                        agent_id=agent_id,
                        base=base,
                        scratch=scratch,
                        access=access,
                    )
            except WorkspaceDeclined as refusal:
                # This is not an error path. Half of the directories in which a
                # user runs pH are not repositories.
                declined = refusal.reason
                log.info(
                    "ph.seams.workspace: tier declined %s for agent %s (%s); using a shared "
                    "workspace, so this agent is not contained",
                    base,
                    agent_id,
                    refusal.reason,
                )
            except Exception:
                # A tier that broke is a tier that is not in force, so
                # `workspace/acquired` says `shared`.
                declined = "provider-failed"
                log.exception("ph.seams.workspace: provider failed; falling back to shared")
                # The operator must know this as much as the log must (E1): an
                # agent that was meant to be contained is not contained.
                await ops_record(
                    self.ctx,
                    "workspace provider failed; the agent is not contained",
                    severity="error",
                    agent_id=agent_id,
                    base=str(base),
                )
                if site is not None:
                    await self._take_back(site, agent_id, session_id)
            except BaseException:
                # Stopped inside the tier (W2): a turn was stopped while the tier
                # was making its worktree. The seam takes back now what the tier
                # made at the site that it named, as it does for a tier that
                # raised. It does not leave that for the next `session/loaded` of
                # this session, which may never come for a child.
                if site is not None:
                    with releasing():
                        await self._take_back(site, agent_id, session_id)
                raise
            else:
                if workspace is None:
                    # The seam invents no reason. A provider that declined
                    # without a reason has not said why.
                    log.info(
                        "ph.seams.workspace: provider declined %s for agent %s; using a shared "
                        "workspace, so this agent is not contained",
                        base,
                        agent_id,
                    )
        if workspace is None:
            workspace = await self.shared.acquire(
                session_id=session_id,
                agent_id=agent_id,
                base=base,
                scratch=scratch,
                access=access,
            )
        # The workspace is owned *before* it is provisioned. To materialize a
        # dependency directory takes thousands of syscalls. If that ran before
        # the `ctx.effect` registration, the worktree would exist with nothing
        # to unwind it. That would break I2 in the module that argues I2.
        if scope is None:
            scope = self._agent_scope(agent_id)
        try:
            held = await self._track(workspace, agent_id, scope)
        except InactiveScopeError:
            # The scope that was to own the workspace went away while the tier
            # made it. The seam gives it back now, as `effect` gives back an
            # artifact whose scope died in the middle of an acquire. Nothing
            # holds it, so nothing else would ever release it.
            if workspace.release is not None:
                with releasing():
                    await workspace.release(workspace)
            raise
        fresh = fresh_root(workspace.kind)
        materials = self._provisioning if fresh else []
        self._log(workspace, agent_id, session, declined, materials)
        # When the tier made the tree at the site that it named, the `acquiring`
        # record on disk already names this tree and what is about to be
        # provisioned into it.
        covered = site is not None and (site.root, site.kind, site.ref) == (
            workspace.root,
            workspace.kind,
            workspace.ref,
        )
        if session is not None and fresh and not covered:
            # **The record is on disk before the seam provisions into the tree**
            # (S12), and so before the agent receives the tree. The copy takes
            # thousands of syscalls. A crash inside it used to leave a tree that
            # no record named, and reconcile reads only recorded pairs, so the
            # tree was never reclaimed. The flush is best effort, as is every
            # flush whose failure a seam cannot turn into a refusal: the tree
            # already exists, and to withhold it would not un-make it. A `shared`
            # root leaves no directory to reclaim, so its record waits for the
            # own barrier of the step.
            await session_written(self.ctx, session)
        copy = _Copy(targets=tuple(entry.target for entry in materials)) if materials else None
        held.copying = copy
        try:
            provisioned = await self._provision(
                workspace, base, materials, copy.stop if copy is not None else None
            )
        finally:
            held.copying = None
            if copy is not None:
                copy.done.set()
        if self._held.get(agent_id) is not held:
            # The workspace was released while it was provisioned (W1). Its
            # release stopped the copy and gave the tree back, and
            # `workspace/disposed` is already in the log. The seam does not give
            # the workspace out, and it records nothing more about a tree that
            # is gone.
            raise InactiveScopeError(
                f"agent {agent_id}'s workspace was released while it was being provisioned"
            )
        held.workspace = provisioned
        if session is not None and held.workspace.provision_failures:
            _LOG.append(
                session,
                "workspace/provisioned",
                {"agentId": agent_id, "failed": list(held.workspace.provision_failures)},
            )
        return held.workspace

    async def _acquiring(
        self,
        session: Session | None,
        session_id: str,
        agent_id: str,
        access: WorkspaceAccess,
    ) -> WorkspaceSite | None:
        """Record where the tier is about to put a fresh tree, on disk before it does.

        This is the opening half of the pair, moved ahead of the act of the tier
        (S12). That act is `worktree add`, or the `init` and FUSE mount of an
        overlay. It used to run with nothing in the log, so a crash inside it
        left a tree that reconcile could not find. `workspace/acquired` still
        follows and supersedes this record, and it carries what only the
        finished tree knows. A decline supersedes this record with a `shared`
        one.

        The flush is best effort, as the flush after `acquired` is. A log that
        cannot take this record loses the pre-record, and not the workspace of
        the agent.

        The record also names what is about to be provisioned, which is the list
        that `acquired` repeats. So when the record is on disk, a tree that the
        tier made at the site that it named needs no second flush before
        provisioning. Returns the site that was recorded, or `None`.
        """
        provider = self.provider
        if session is None or not isinstance(provider, LocatingProvider):
            return None
        site = provider.locate(session_id=session_id, agent_id=agent_id, access=access)
        if not fresh_root(site.kind):
            return None
        data = pair_payload(agent_id, site.ref, kind=site.kind, root=str(site.root))
        if self._provisioning:
            data["provisioned"] = [entry.target for entry in self._provisioning]
        _LOG.append(session, ACQUIRING, data)
        await session_written(self.ctx, session)
        return site

    async def _take_back(self, site: WorkspaceSite, agent_id: str, session_id: str) -> None:
        """Reclaim whatever a tier that *raised* made at the site that it named.

        The `shared` acquire that follows supersedes the `acquiring` record, so
        after it nothing names this root. And a tier can raise part-way through:
        an overlay is mounted, and then its base is not recorded. So the seam
        does now what reconcile would do after a crash, through the reclaim of
        the tier itself. This is not done for a tier that *declined*. A decline
        is decided before the tree is made, and `path-exists` names a directory
        that is not for this agent to remove.
        """
        record = WorkspaceRecord(
            agent_id=agent_id, kind=site.kind, root=site.root, ref=site.ref, session_id=session_id
        )
        provider = self._reclaimer([record], "reclaim")
        if provider is not None:
            await self._reclaim(provider, record, "reclaim", own=True)

    async def _provision(
        self,
        workspace: Workspace,
        base: Path,
        materials: Sequence[ProvisionEntry],
        stop: ProvisionStop | None,
    ) -> Workspace:
        """Put the configured materials in a *fresh* root (E14).

        A root that the tier reused (`Workspace.reused`) gets only what it
        lacks. The rest arrived when the tree was made, and the agent may have
        edited it since.
        """
        if not materials:
            return workspace
        # The name is qualified. `provision` on this class is the
        # *registration*, and the module function does the work.
        report: ProvisionReport = await workspace_provision.provision(
            materials,
            base=base,
            root=workspace.root,
            fill=workspace.reused,
            stop=stop,
        )
        if report.failed:
            log.warning(
                "ph.seams.workspace: %d material(s) did not reach %s",
                len(report.failed),
                workspace.root,
            )
        return replace(workspace, provisioned=report.provisioned, provision_failures=report.failed)

    async def dispose(self, agent_id: str) -> None:
        """Release the workspace of this agent early.

        It is "early" because the scope owns the workspace in either case (I2).
        This is the same teardown, reached deliberately and not by unwinding. A
        second call does nothing, because the disposer deregisters itself.
        """
        held = self._held.get(agent_id)
        if held is None or held.dispose is None:
            return
        await maybe_await(held.dispose())

    async def _scratch_for(self, session_id: str, agent_id: str) -> Path:
        """The scratch directory for one session *and* one agent. It is created,
        and not only named.

        The seam owns it, so the layout has one implementation. The collision
        that this avoids is two children of one session that write notes into
        one directory.

        The path is made canonical here, and `scratch_root` is not trusted to be
        canonical already. The shipped row gets `scratch_root` from
        `default_home_path`, which is canonical. But the guarantee that
        `acquire` makes belongs to the seam, so it does not depend on how the
        seam was built.
        """
        scratch = _scratch_layout(self.scratch_root, session_id, agent_id)
        await anyio.to_thread.run_sync(lambda: scratch.mkdir(parents=True, exist_ok=True))
        return scratch

    def _chosen_tier(self, session: Session | None) -> ContainmentTier | None:
        """Which rung this acquisition gets, read from the choice of the deployment.

        The role comes from the session and not from an argument. The header of
        a child carries `origin: "subagent"`, so "is this a child" is a fact
        that the seam holds.

        `None` means that there is no containment row, so nobody chose. A
        deployment that layered a provider and never mentioned containment gets
        that provider, because layering it *was* the choice.
        """
        containment = self.ctx.get(CONTAINMENT)
        if containment is None:
            return None
        child = session is not None and session.header.is_subagent
        chosen: ContainmentTier | None = containment.for_role(child=child)
        return chosen

    def _agent_scope(self, agent_id: str) -> Context | None:
        """The own scope of the live agent `agent_id`, or `None` when nobody knows
        that id.

        `acquire` with no `scope=` used to fall straight through to
        `owner_for(None)`. Outside the `apply` of a row, that is the context of
        the seam itself. A worktree taken for a real agent then outlived that
        agent for the whole life of the process, and the containment-ladder
        tests had to remember `scope=agent.ctx` by hand (P4-16). The registry
        already knows the answer, so the seam asks it
        (`AgentRegistry.scope_of`).
        """
        agents = self.ctx.get(AGENTS)
        return None if agents is None else agents.scope_of(agent_id)

    async def _track(self, workspace: Workspace, agent_id: str, scope: Context | None) -> _Held:
        """Register the teardown as an effect, so that the workspace has an owner.

        The release closure reads `held.workspace` and does not capture a
        workspace. Provisioning replaces the value a moment later, and the
        teardown policy needs the *final* one. What was put in the tree is what
        the policy must not mistake for the work of the agent.
        """
        held = _Held(workspace=workspace)

        def enter() -> Disposer:
            # The entry is written here. `effect` reaches this point only after
            # its check that the owner is up, and nothing suspends between here
            # and the registration of the release. When the entry was written
            # ahead of the call, a refusal left the entry for good, and it had
            # already dropped the release of the entry that it replaced.
            replaced = self._held.get(agent_id)
            if replaced is not None and replaced.dispose is not None:
                # The agent acquired again. The previous release is dropped
                # without being run, so it cannot evict the live workspace from
                # `_held` when its scope goes.
                replaced.dispose.drop()
            self._held[agent_id] = held

            async def release() -> None:
                del self._held[agent_id]
                current = held.workspace
                copy = held.copying
                if copy is not None:
                    # A copy is running (W1). It is stopped before its next file
                    # and waited for, so the tree is not removed under the
                    # thread of the copy. Everything that the copy was putting
                    # there belongs to the seam, and is never the work of the
                    # agent.
                    copy.stop.set()
                    await copy.done.wait()
                    current = replace(current, provisioned=copy.targets)
                kept = True if current.release is None else await current.release(current)
                if held.session is not None:
                    # The record comes after the act, which is idempotent. An
                    # open `acquired` is the intent of the act, and reconcile
                    # drives again a release whose record never landed. The
                    # scope of the agent flushes its log as the last thing that
                    # it unwinds (`AgentRegistry.create`). That is what gets
                    # this record to disk on every path by which a child ends
                    # (S12).
                    _LOG.append(held.session, DISPOSED, self._payload(current, agent_id, kept=kept))

            return release

        held.dispose = await self.ctx.owner_for(scope).effect(enter, label=f"workspace({agent_id})")
        return held

    def _log(
        self,
        workspace: Workspace,
        agent_id: str,
        session: Session | None,
        declined: DeclineReason | None,
        materials: Sequence[ProvisionEntry],
    ) -> None:
        """Record the opening half of the durable pair. The seam writes both
        halves.

        A pair reconciles only if one place owns both halves. A provider that
        forgot the second half would leave every workspace looking leaked.

        The act of the tier itself is `worktree add`, or the init and mount of
        an overlay. By now that act is already on record as
        `workspace/acquiring` (`_acquiring`). This record supersedes that one
        with what only the finished tree knows.
        """
        if session is None:
            return
        self._held[agent_id].session = session
        data = self._payload(
            workspace,
            agent_id,
            kind=workspace.kind,
            root=str(workspace.root),
            repoWritable=workspace.repo_writable,
        )
        if declined is not None:
            # This key is set only when a tier was asked and could not serve.
            # An absent key means "no tier configured". That is a different
            # fact, and `phern doctor` must not confuse the two (E15).
            data["declined"] = declined
        if materials:
            # The list is on the opening half, because reclaim has nothing else
            # to read it from. Without it, the provisioned secret of a crashed
            # tree is staged as the work of the agent and committed to a branch
            # that somebody merges (S13). The list is what is *about* to be
            # provisioned, because the copy has not run yet. So a crash in the
            # middle of the copy still names every path that the copy may have
            # written, and a material that never arrived keeps its path off the
            # branch too. No target here climbs out of the tree, because
            # `ProvisionEntry` refuses such a target when the config is loaded.
            # That matters for more than tidiness: git exits 128 on a pathspec
            # that names a path outside the repository.
            data["provisioned"] = [entry.target for entry in materials]
        if workspace.fork is not None:
            # On the opening half for the reason that `provisioned` is: reclaim
            # has nothing else to read it from (`WorkspaceRecord.fork`).
            data["fork"] = workspace.fork
        _LOG.append(session, ACQUIRED, data)

    def retain(self, agent_id: str, reason: str) -> bool:
        """Keep the tree of this agent past disposal, and say why (P6-28).

        Whoever learns how an agent ended calls this, at any point before the
        scope unwinds. The teardown policy reads `Workspace.retained` and skips
        the discard.

        **An empty `reason` clears the mark.** It is the same call because it is
        the same decision, revisited. The shipped policy retains *by default*
        for the kind that discards, so a clean settle is a caller that withdraws
        the mark. The withdrawal must be as durable as the mark.

        Returns whether anything was marked. The answer is `False` for an agent
        that holds no workspace, so a caller that retains speculatively needs no
        `hasattr` probe and no exception.
        """
        held = self._held.get(agent_id)
        if held is None:
            return False
        held.workspace = replace(held.workspace, retained=reason)
        if held.session is not None:
            # The mark is recorded when it is made, and not only on the closing
            # half. The worst way for a run to go wrong writes no `disposed`.
            _LOG.append(
                held.session, RETAINED, pair_payload(agent_id, held.workspace.ref, retained=reason)
            )
        return True

    def _payload(
        self,
        workspace: Workspace,
        agent_id: str,
        **extra: JsonValue,
    ) -> dict[str, JsonValue]:
        payload = pair_payload(agent_id, workspace.ref, **extra)
        # The reason is added to the closing half, because that is the half
        # that a fold reads to tell a deliberate keep from a dirty-tree keep
        # (P6-28).
        if workspace.retained:
            payload["retained"] = workspace.retained
        return payload

    async def reconcile(self, session: Session) -> None:
        """Close the pairs that a crash left open in the log of this session (F6).

        It is on the seam because what it needs is here. `_held` says whether a
        tree belongs to anybody. `pair_payload` holds the shape of the pair, so
        the `disposed` that a reconciliation writes is the one that an orderly
        release would have written.

        A leak that this profile cannot reclaim is **reported and left alone**.
        The tree belongs to a tier that is not mounted here. To remove a
        directory because of a record that a different configuration wrote is
        the one way in which this could destroy the work that it exists to
        protect.
        """
        leaks = [
            one
            for one in workspace_leaks(session)
            if one.agent_id not in self._held and one.agent_id not in self._claimed
        ]
        provider = self._reclaimer(leaks, "reclaim")
        if provider is None:
            return

        async def reclaim(record: WorkspaceRecord) -> None:
            kept = await self._reclaim(provider, record, "reclaim")
            if kept is None:
                return
            # The pair closes whether or not anything was kept. A leak that is
            # left open is reported at every later open.
            _LOG.append(
                session,
                DISPOSED,
                pair_payload(record.agent_id, record.ref, kept=kept, reconciled=True),
            )

        # Concurrent, because each leaked tree takes several subprocesses.
        async with anyio.create_task_group() as group:
            for record in leaks:
                group.start_soon(reclaim, record)
        if leaks:
            # The closing halves are flushed with the acts that they record. No
            # agent scope owns these records, so nothing else flushes them. And
            # a second reclaim of a tree that the first one removed can no
            # longer tell what reached the branch: it says `kept: false`.
            await session_written(self.ctx, session)

    def collectable(
        self,
        survivors: Iterable[WorkspaceRecord],
        *,
        older_than: float,
        now: float,
        touched: Mapping[str, float],
    ) -> list[Collectable]:
        """Which retained trees may be removed, and why the others may not.

        **Only `retained` trees are considered, and that boundary is the safety
        argument.** A `kept` tree is work that the disposal policy kept for a
        person to inspect, and `/workspaces remove` is the deliberate way to end
        it. A `leaked` tree belongs to `reconcile`, which is the only
        thing that can tell "the process died" from "the process is running".
        This may collect only what a policy retained without anybody asking.

        There are three refusals, in the order in which they are cheap to test:

        * `open`: an unclosed pair is a live process or a crash, and in either
          case this mechanism does not settle it. Such a record is listed in
          `survivors` so that an enumeration can show it. It is never
          collected.
        * `held`: this process holds the tree. The match is by **root path** and
          not by agent id. An agent that retained a tree and then acquired
          another is the ordinary case, and an id match would refuse to collect
          the first tree for as long as the agent lives. The `_is_held` of
          `_reclaim` is deliberately the stricter test: it matches the id or
          the root. It guards a teardown, where it costs nothing to refuse too
          often.
        * `recent`: the tree is inside the age bound, dated from `touched`. A
          session id that is *absent* from `touched` is refused as `recent` and
          is not collected. "I could not date this" and "this is old" are
          different answers, and only one of them may delete a checkout.

        The age is the last write of the **log**, and not the mtime of the tree.
        A person who reads a retained tree without editing it changes neither,
        so no clock here detects interest.
        """
        rows: list[Collectable] = []
        held = {workspace.root for workspace in self.live()}
        for record in survivors:
            if record.outcome != "retained" or not record.closed:
                continue
            age = now - touched.get(record.session_id, now)
            if record.root in held:
                verdict: CollectVerdict = "held"
            elif not record.root.exists():
                verdict = "gone"
            elif age < older_than:
                verdict = "recent"
            else:
                verdict = "collect"
            rows.append(Collectable(record=record, verdict=verdict, age=age))
        return rows

    async def collect(self, rows: Iterable[Collectable]) -> list[WorkspaceRecord]:
        """Remove the trees that `collectable` cleared, and return the records of
        the trees that went.

        **Retention is revoked, and not overridden.** This gives the record back
        to `reclaim` with its reason cleared. So what runs is the disposal
        policy that would have run at release if nobody had retained the tree.
        Nothing here can destroy more than an ordinary disposal would have
        destroyed. That is what lets the age bound be a *default* and not a
        decision.

        The removals are sequential, unlike the fan-out of `reconcile`. Here a
        user watches a command that they typed, and a failure halfway through a
        fan-out is a report that they cannot act on.

        Nothing is appended, because the pair is already closed. `reclaim`
        answers `False` for a directory that is not there, so a second run
        changes nothing.
        """
        wanted = [row.record for row in rows if row.verdict == "collect"]
        provider = self._reclaimer(wanted, "collect")
        if provider is None:
            return []
        removed: list[WorkspaceRecord] = []
        for record in wanted:
            # Revoked, and not overridden. `reason=""` is what makes `reclaim`
            # run the disposal policy that it would have run if nobody had
            # retained the tree.
            if await self._reclaim(provider, replace(record, reason=""), "collect") is False:
                removed.append(record)
        return removed

    async def export(self, record: WorkspaceRecord) -> str | None:
        """The ref that the work of this agent is on, or `None` if no tier can say.

        The answer is `None` and not a raise, as in `_reclaimer`. A profile
        whose tier cannot export is not a broken deployment. It is one where the
        answer is "there is nothing to move".
        """
        provider = self.provider
        if not isinstance(provider, ExportingProvider):
            log.warning("ph.seams.workspace: no mounted tier can export %s", record.agent_id)
            return None
        return await provider.export(record)

    def _artifacts(self) -> ArtifactProvider | None:
        """The ref verbs of the mounted tier, or `None`. It has the shape of
        `_reclaimer`, in one line."""
        provider = self.provider
        return provider if isinstance(provider, ArtifactProvider) else None

    def _checkouts(self) -> EnumeratingProvider | None:
        provider = self.provider
        return provider if isinstance(provider, EnumeratingProvider) else None

    def _checkpointer(self, workspace: Workspace) -> CheckpointingProvider | None:
        """The tier that can checkpoint *this* workspace, or `None` (P6-20).

        Both halves of `can_checkpoint` are in one place, so the predicate and
        the narrowing cannot come apart. They were the same `isinstance` written
        twice: one time to decide, and one time to convince the type checker.

        `fresh_root` is the **kind** half. The root of a `shared` workspace is
        the user's own checkout. To offer to overwrite the uncommitted work of
        the user with whatever an agent found is the one thing that this must
        never do. The Protocol test is the **tier** half. It replaced a table of
        provider facts that was keyed by kind and kept where no provider could
        see it.
        """
        provider = self.provider
        if not fresh_root(workspace.kind) or not isinstance(provider, CheckpointingProvider):
            return None
        return provider

    async def refs(self, base: Path) -> list[str]:
        """Every ref in this repository, as the mounted tier lists them.

        The list is unfiltered on purpose. `/workspaces` applies `BRANCH_PREFIX`
        itself, because that prefix is all that keeps it from offering to delete
        a user's own branch. And `merge` deliberately accepts a ref outside the
        prefix.
        """
        provider = self._artifacts()
        if provider is None:
            return []
        with running(self.provider_by):
            return await provider.refs(base)

    async def delete_ref(self, base: Path, ref: str, *, force: bool) -> str:
        """Delete one ref. Returns `""` when it went, and otherwise the reason
        that the tier gives.

        `force` is the decision of the caller and not of the tier. Each of the
        three tiers that delete refs (git, agentfs and jj) refuses a ref that
        holds work which nothing else has. A user who overrides that refusal
        answers a question that only the user can answer.
        """
        provider = self._artifacts()
        if provider is None:
            return f"no mounted tier can delete {ref}"
        with running(self.provider_by):
            return await provider.delete_ref(base, ref, force=force)

    async def merge(self, base: Path, ref: str) -> str:
        """Merge `ref` into the checkout that the user is in. Returns `""` when it
        merged cleanly.

        Any other answer is the sentence of the tier, and it is not always a
        failure. jj records conflicts *in the commit* and exits zero. So
        "merged, with conflicts at these paths" is a true answer that this can
        give and that a bool could not.
        """
        provider = self._artifacts()
        if provider is None:
            return f"no mounted tier can merge {ref}"
        with running(self.provider_by):
            return await provider.merge(base, ref)

    async def strays(self, base: Path, *, with_status: bool = True) -> dict[str, Stray]:
        """The checkouts that the mounted tier still has on disk, by ref. `{}` if
        no tier can say.

        The answer is keyed by ref and is not a list, because every caller joins
        it against something. `/workspaces` joins it against the branches that
        pH made. With a list, each caller would build the same index.
        """
        provider = self._checkouts()
        if provider is None:
            return {}
        # `held` is the refs that live agents hold. With it, no tier spends a
        # probe on a `dirty` that the caller discards: `describe` reports a held
        # row as held and reads no further. The seam supplies the set because
        # `live()` belongs to the seam.
        held = frozenset(workspace.ref for workspace in self.live() if workspace.ref)
        with running(self.provider_by):
            found = await provider.strays(base, with_status=with_status, skip=held)
        return {one.ref: one for one in found}

    async def discard(self, path: Path) -> str:
        """Remove one checkout. Returns `""` when it went, and otherwise the
        sentence that says why not.

        The answer is a message and not a bool. The only caller is a user who
        typed `remove`, and that user is owed the reason of the tier itself. A
        profile whose tier cannot enumerate cannot have produced this path in
        the first place. So the refusal is a guard and not a case.
        """
        provider = self._checkouts()
        if provider is None:
            return f"no mounted tier can remove {path}"
        with running(self.provider_by):
            return await provider.discard(path)

    def can_checkpoint(self, workspace: Workspace) -> bool:
        """Whether a restore point can be taken for this workspace at all (P6-20).

        `/revert` asks this before it offers a restore point. So a workspace
        that can never have a restore point **refuses**, and does not report "no
        restore points in this session". That report is true and useless, and
        it cannot be told apart from a run that has not checkpointed yet.

        Both halves are in `_checkpointer`, and `capture` narrows through the
        same function. The predicate and the narrowing were the same
        `isinstance` written twice, and two spellings of one gate can come
        apart.
        """
        return self._checkpointer(workspace) is not None

    async def capture(self, workspace: Workspace) -> str | None:
        """A token that names the state of this workspace now, or `None` if no
        tier can say.

        The answer is `None` and not a raise, as in `export`. A profile whose
        tier cannot checkpoint is not a broken deployment. The consumer of the
        fingerprint reads an empty answer as "always run again", which is the
        safe direction.
        """
        provider = self._checkpointer(workspace)
        if provider is None:
            return None
        with running(self.provider_by):
            return await provider.capture(workspace)

    async def checkpoint(
        self, workspace: Workspace, *, session: Session, agent_id: str, call_id: str
    ) -> str | None:
        """Capture a restore point and record it. Returns `None` when the tier has
        none to give.

        The event belongs to the *seam*, for the reason that `_log` gives. Two
        tiers take restore points and three consumers fold them, so the shape of
        the payload belongs to neither tier.

        **The restore point is pinned before it is recorded.** That reverses the
        order that the git tier used while it owned this. The old order had a
        mechanical reason: the ref was named by the `seq` of the event, so a pin
        written first had no name to write. A token now names its own pin. What
        changes is the direction in which a crash between the two steps can
        fail, and the new direction is better. It loses the *record* of a
        restore point whose state is safely pinned. The old order kept a record
        of state that had not been pinned yet. A pin that nobody recorded is
        idempotent garbage, and the next capture writes over it.

        `tree` is the payload key, and it is historical and not descriptive. The
        token of the git tier is a tree, and the token of jj is a commit. To
        rename the key would break every reader of a log that was written
        before the rename, for no gain that a user can see.
        """
        token = await self.capture(workspace)
        if token is None:
            return None
        _LOG.append(session, CHECKPOINT, {"agentId": agent_id, "tree": token, "callId": call_id})
        return token

    async def restore(self, agent_id: str, token: str, *, session: Session) -> tuple[str, ...]:
        """Put the workspace of this agent back to `token`. Returns the paths that
        the run had added.

        This raises and does not answer `None`, unlike every other optional
        capability here. Each caller acts on a restore point that a user or a
        retry ladder asked for **by name**. "It silently did nothing" is the one
        answer that none of them may mistake for success.

        **The restore is always recorded** (`WORKSPACE_RESTORE`). A restore
        rewrites and deletes files. If the process dies inside one, the tree is
        neither the restore point nor what was there. Repair closes that pair as
        `outcome-unknown`, which is what a reader is owed. The record is on disk
        before the tree is touched, and a restore that the log cannot record is
        not run (`IntentNotDurable`). A restore that raised is settled
        `ok: false`, and the error is then raised again.

        The record is written here and not at each caller, and `session` is
        required as `checkpoint` requires it. So `/revert`, the retry of a
        supervisor and any third caller record it in one way, and none can leave
        it out. The workspace is the own workspace of the agent, looked up here.
        So a caller cannot give one tree and record another agent.
        """
        workspace = self.of(agent_id)
        if workspace is None:
            raise FileNotFoundError(f"agent {agent_id!r} holds no workspace to restore")
        provider = self.provider
        if not isinstance(provider, CheckpointingProvider):
            raise FileNotFoundError(f"no mounted tier can restore {workspace.root}")
        journal = intents_of(self.ctx)
        opening: dict[str, JsonValue] = {"agentId": agent_id, "tree": token}
        async with journal.claim(session, WORKSPACE_RESTORE, opening) as held:
            try:
                with running(self.provider_by):
                    removed = await provider.restore(workspace, token)
            except Exception as error:
                failed = restore_settled(held.opened, ok=False, detail=error_message(error))
                journal.settle(session, held, failed)
                raise
            journal.settle(session, held, restore_settled(held.opened, ok=True))
        return removed

    def _reclaimer(
        self, records: Sequence[WorkspaceRecord], verb: str
    ) -> ReclaimingProvider | None:
        """The mounted tier, or `None` after a warning that names the trees that
        nobody can end.

        The warning comes one time for each batch, and not one time for each
        record. A profile with no reclaiming tier owes one sentence that names
        what it cannot touch, and not one sentence for each directory. The
        refusal is the purpose. To remove a directory because of a record that
        a different configuration wrote is the one way in which a caller could
        destroy the work that it exists to protect.
        """
        provider = self.provider
        if isinstance(provider, ReclaimingProvider):
            return provider
        if records:
            log.warning(
                "ph.seams.workspace: no mounted tier can %s %s",
                verb,
                ", ".join(str(one.root) for one in records),
            )
        return None

    def _is_held(self, record: WorkspaceRecord) -> bool:
        """Whether this process uses the tree that this record names, by agent id
        or by root.

        **This is the strict test, and only `_reclaim` wants it.** Its two
        siblings ask narrower questions on purpose. `reconcile` folds leaks for
        each agent, so the id is the identity there. `collectable` matches
        roots, because an agent that retained one tree and acquired another must
        not keep the first tree alive. `_reclaim` is the function that
        *deletes*. There, to refuse one tree too many costs a directory that
        somebody can remove by hand, and to refuse one too few costs the work in
        it.
        """
        if record.agent_id in self._held:
            return True
        return any(held.workspace.root == record.root for held in self._held.values())

    async def _reclaim(
        self,
        provider: ReclaimingProvider,
        record: WorkspaceRecord,
        verb: str,
        *,
        own: bool = False,
    ) -> bool | None:
        """One call into the teardown of a tier. Returns whether the tier **kept**
        anything, or `None` if the tier raised or an agent took the tree.

        The answer has three states and is not an exception, because a caller
        that works through a batch may not abort the batch. `reconcile` would
        leave the rest of the pairs of a crash open, and `collect` would stop at
        the first tree on which git failed.

        `verb` reaches the log only. It tells an operator whether a warning came
        from a reconciliation at session open or from a `gc` that the operator
        typed.

        **`reconcile` and `collect` both decided that this tree was nobody's,
        and then awaited something** (J6). `reconcile` folds its leaks before it
        starts a task for each tree, and `collect` asks `collectable` before it
        removes any tree. An `acquire` that lands in that gap gets a tree that
        this is about to delete. So the question is asked again here, where a
        second answer is worth something, because this is the last line before
        the removal. This also asks whether an act is in progress on the tree
        (`_claimed`), and it takes a claim while it deletes.

        `own` is an `acquire` that takes back its own half-made tree, under the
        claim that it already has.
        """
        if self._is_held(record) or (not own and record.agent_id in self._claimed):
            log.info(
                "ph.seams.workspace: not reclaiming %s — an agent took it while the "
                "%s was being prepared",
                record.root,
                verb,
            )
            return None
        done = None if own else anyio.Event()
        if done is not None:
            self._claimed[record.agent_id] = done
        try:
            with running(self.provider_by):
                return await provider.reclaim(record)
        except Exception:
            log.warning("ph.seams.workspace: could not %s %s", verb, record.root, exc_info=True)
            return None
        finally:
            if done is not None:
                del self._claimed[record.agent_id]
                done.set()


WorkspaceOutcome: TypeAlias = Literal["leaked", "kept", "retained"]
"""Why a tree is still on disk. `git worktree list` reports the three cases
identically (P6-28).

* `leaked`: an `acquired` that the log never saw closed. It is the only outcome
  that nobody decided.
* `kept`: the disposal policy kept the work of a dirty tree for review. The git
  tier and the jj tier remove the checkout and keep the work on its ref, so
  for them the record outlives the directory.
* `retained`: somebody named a reason. This outcome **wins over the other two**.
  A tree that was asked for stays asked for, whether or not the process that
  held it exited cleanly.
"""


@dataclass(frozen=True, slots=True)
class WorkspaceRecord:
    """One tree that a session left on disk, and why.

    It holds what survives a crash, and only that: the fields of the log. It is
    not a `Workspace`. `scratch`, `env` and the `release` closure are process
    state that died with the process. A value that carried empty versions of
    them would invite a caller to use them.
    """

    agent_id: str
    kind: WorkspaceKind
    root: Path
    ref: str | None = None
    reason: str = ""
    """Why it was retained. Empty for the other two outcomes."""
    closed: bool = False
    """Whether the durable pair reconciled. This is a separate axis from `reason`.

    A retention is marked *while the agent is live*, so a process that then dies
    leaves a record that is both retained and unclosed. One axis would force a
    choice between two wrong answers. If the record counts as leaked,
    reconciliation discards the evidence that it was told to keep. If it counts
    as retained, the pair never closes, so the tree is reported again at every
    open.
    """
    provisioned: tuple[str, ...] = ()
    """What the seam provisions into the tree, as the opening record
    (`workspace/acquiring` or `workspace/acquired`) recorded it.

    This is the one piece of the live `Workspace` that a reclaim cannot do
    without. Every other file in the tree is the work of the agent, and these
    are the files that must stay off its branch. The list is recorded before the
    copy runs, so it is what was *configured* and not what arrived. That is a
    superset, and it is the safe one to exclude.
    """
    fork: str | None = None
    """The commit that the tree started from, as `workspace/acquired` recorded
    it (`Workspace.fork`). `None` when the record does not say: the tier does
    not use one, the process died before `acquired` was written, or an older
    build wrote the log.

    The jj tier reads it at a crash reclaim. A parent that spawned a child has
    an empty working copy, because its work was frozen into a commit for the
    child. Without the fork point, a reclaim read that parent as a tree that
    had done nothing, deleted its bookmark and removed the tree. With `None`,
    the tier keeps the bookmark, because it cannot tell.
    """
    session_id: str = ""
    """The session whose log this record came from. It cannot be derived from
    `agent_id`. A reader that crosses sessions (the family fold, the collector)
    must get back to the log. To infer the session from a sanitized directory
    name is the lossy round trip that `/workspaces` refuses to make.
    """

    @property
    def outcome(self) -> WorkspaceOutcome:
        """Which of the three outcomes this is. It is **derived from the two facts
        and never stored**.

        A third name for what `reason` and `closed` already say is a third
        chance to disagree with them. `collect` revokes a retention by clearing
        `reason`, and a stored field went on reporting `retained` after the
        decision was withdrawn.
        """
        if self.reason:
            return "retained"
        return "kept" if self.closed else "leaked"


CollectVerdict: TypeAlias = Literal["collect", "held", "recent", "gone"]
"""What the collector decided about one retained tree. Each verdict is a
sentence that a user reads. `held` and `recent` are refusals that the user may
want to argue with. `gone` is a tree that somebody already removed by hand.
"""


@dataclass(frozen=True, slots=True)
class Collectable:
    """One retained tree, with the verdict of the collector and the age of the
    tree.

    There is a verdict for each record and not a filtered list, because the
    refusals are the useful half. "Nothing to collect" and "three trees, all
    still held by live sessions" are very different answers to
    `phern workspaces gc`.
    """

    record: WorkspaceRecord
    verdict: CollectVerdict
    age: float
    """Seconds since the log of the owning session was last written."""


def workspace_survivors(session: Session) -> list[WorkspaceRecord]:
    """Every tree that this session left on disk, and which of three reasons put
    it there.

    **Only the log can tell the three apart.** They all leave a worktree that
    `git worktree list` reports identically, so `/workspaces` cannot. Two are
    features, and the third is the leak that F6 exists to close.

    The answer is folded and not tracked, as the subagents seam folds the state
    of a child (`ChildState`). The answer must be computable from a log that a
    process reads off disk, when that process was not running while the log was
    written.

    **The fold starts at `seed_length`, and not at the beginning.** A fork seeds
    the child with the transcript of the parent. A fold over the whole log
    would report the worktrees that the parent still holds as those of the
    child, and reconciliation would then remove a tree in which an agent is
    working. What a session inherited is not what it acquired.

    Only a kind with a fresh root can leave a directory behind. The root of a
    `shared` workspace *is* the base, so an unclosed pair there records a crash
    and no stray.

    **`acquiring` opens a record as `acquired` does** (S12). It is written
    before the tier makes the tree, so an `acquiring` with nothing after it
    names a tree that a crash may have left half-made. The `acquired` that
    follows supersedes it. A `shared` one supersedes it too: that is a tier that
    declined and made nothing at the site that it named.
    """
    open_records: dict[str, WorkspaceRecord] = {}
    closed: list[WorkspaceRecord] = []
    for event in session.own_events():
        # The type test comes first. A long log is mostly `assistant/chunk`. To
        # read `agentId` from every one of them, only to find it absent, costs a
        # mapping get and a string for each event.
        if event.type not in _SURVIVOR_TYPES:
            continue
        data = event.data
        agent_id = as_str(data.get("agentId"))
        if not agent_id:
            continue
        if event.type == RETAINED:
            # The mark was made while the agent was live, so the record is
            # already open. The exception is a session that only *seeded* the
            # acquire. Then there is nothing here to mark, and nothing on disk
            # that this session may claim.
            marked = open_records.get(agent_id)
            if marked is not None:
                # An empty reason is a withdrawal. It returns the record to the
                # outcome that an open acquire has by default. If it were read
                # as a retention with a blank reason, a clean settle would keep
                # a tree forever.
                open_records[agent_id] = replace(marked, reason=as_str(data.get("retained")))
            continue
        if event.type == DISPOSED:
            record = open_records.pop(agent_id, None)
            if record is None:
                continue
            # A reason wins over `kept`, and over a `kept: false` too. The
            # closing half repeats the reason so that an orderly release states
            # it. And a policy that discarded a tree that it had been asked to
            # keep would have written `kept: false` about a directory that is
            # still there.
            reason = as_str(data.get("retained") or record.reason)
            if reason or data.get("kept"):
                closed.append(replace(record, closed=True, reason=reason))
            continue
        # A Literal that is read from JSON is a claim to check, and not a cast
        # to make. The lookup is built one time at import, and not for each
        # event: `get_args` is not memoized and rebuilds its tuple on every
        # call.
        kind = _WORKSPACE_KINDS.get(as_str(data.get("kind")), "shared")
        if not fresh_root(kind):
            # A `shared` acquire answers an `acquiring` that the tier then
            # declined. Nothing was made at the site that the tier named, so
            # nothing there belongs to this agent.
            open_records.pop(agent_id, None)
            continue
        ref = data.get("ref")
        open_records[agent_id] = WorkspaceRecord(
            agent_id=agent_id,
            kind=kind,
            root=Path(as_str(data.get("root"))),
            ref=str(ref) if ref else None,
            provisioned=tuple(filter(None, map(as_str, as_seq(data.get("provisioned"))))),
            fork=as_str(data.get("fork")) or None,
            session_id=session.id,
        )
    return [*open_records.values(), *closed]


def workspace_leaks(session: Session) -> list[WorkspaceRecord]:
    """The workspaces that this session took and never released (F6).

    This is a filter over `workspace_survivors`, and not a fold of its own. The
    two questions differ by one predicate, and they share every rule that is
    easy to get wrong: the seed offset, the fresh-root test, and which acquire
    is the live one. A second implementation here would disagree about whether a
    directory may be deleted (A11).

    The predicate is **`closed`, and not `outcome == "leaked"`**. A tree that
    somebody retained before the process died is a survivor with a reason *and*
    an open pair. Reconciliation still owes it the closing event. `reclaim` is
    what knows that a reason means that the work is not discarded.
    """
    return [one for one in workspace_survivors(session) if not one.closed]


def has_restore_point(session: Session, call_id: str) -> bool:
    """Whether the run `call_id` took a restore point: whether the latest restore
    point that this session recorded belongs to that run (S4).

    The `session-checkpoint-policy` row asks this before it lets a nested
    dispatch skip its barrier because a restore covers it. A restore point that
    does not exist covers nothing. There is none on a profile whose tier cannot
    take one, and none for a run whose capture failed (the
    `workspace-checkpoint` row only warns). The latest one is the right one to
    test, because that row records it immediately before the run that it
    covers, and the cells of a session run one at a time.
    """
    latest = session.latest(CHECKPOINT)
    return latest is not None and as_str(latest.data.get("callId")) == call_id


def checkpoints(session: Session) -> dict[int, dict[str, Any]]:
    """Every restore point in this session, by the seq of its event. This is a
    fold.

    A checkpoint is a fact in the log. So a resumed or forked session finds the
    same restore points that a live one has, and nothing had to remember them.
    """
    return {event.seq: dict(event.data) for event in session.events if event.type == CHECKPOINT}


@dataclass(frozen=True, slots=True)
class _CheckpointOf:
    """The restore point of one agent, read from a `workspace/checkpoint` event.

    It is frozen, so it hashes by value. That is what gets each agent its own
    fold out of one parser class. `Session.projection` keys on
    `(event_type, parse)`, so `_CheckpointOf("a")` and `_CheckpointOf("b")` are
    two keys. A closure is a new object on each call and hashes by identity, so
    it would be a new fold on each call.
    """

    agent_id: str

    def __call__(self, event: SessionEvent) -> str | None:
        """The tree, or `None` when some other agent took this checkpoint.

        `""` is a value here, and not a miss. A checkpoint that recorded no tree
        is still the newest one that this agent took. To answer with an *older*
        tree would revert further than the log says to.
        """
        if as_str(event.data.get("agentId")) != self.agent_id:
            return None
        return as_str(event.data.get("tree"))


def latest_checkpoint(session: Session, agent_id: str) -> str:
    """The newest restore point that *this agent* took, or `""` if it has none.

    This is an incremental fold, and not `checkpoints()` plus `max()`. The
    caller that wants one restore point does not need a dict of every restore
    point. To build that dict copies each payload and then discards all but the
    last, on a crash path that runs one time for each retry. It is not a reverse
    scan of `session.events` either. That scan materialized a snapshot of the
    whole log to read one field. `Session.projection` keeps the fold and parses
    only what has arrived since.

    The answer is scoped to the agent, which is the rule that `/revert` already
    states: a restore point belongs to the agent that took it. Only one agent
    writes into a root session today, so this is a latent difference and not a
    live one. It is here so that the two readers of this fold cannot disagree
    about it later. The agent id is in the *parser*, which is half of the key of
    `Session.projection`. So two agents get two folds, and this function does
    not have to invent a name for either.
    """
    return session.projection(CHECKPOINT, _CheckpointOf(agent_id)) or ""


def stored_survivors(
    store: SessionArchive,
    *,
    limit: int = 50,
    family: str = "",
) -> tuple[list[WorkspaceRecord], dict[str, float]]:
    """Every tree that the *store* can still account for, and when each log was
    written.

    This is the deployment-wide half of the fold, and `family_survivors` is the
    half for one parent. Three callers use it: the count of `phern doctor`, the
    collector, and the `export` of `/workspaces`. The first two need both
    answers. A second loop over `stored()` is where a listing limit and a
    tolerance rule would diverge without notice.

    **A session that cannot be read is skipped, and is not fatal.** The
    direction is the safe one for the count and for the collector: doctor counts
    too few, and the collector removes nothing. That is the right result for a
    half-written file. The failure is logged, because a store that cannot read
    most of what it listed has a real problem that shows only as a small
    number. Since reference-forking there are two ways in which a log cannot be
    read. A *good* file whose ancestor was removed takes every descendant with
    it, so this count can fall by more than the number of damaged files. The
    "Session lineage" section of `phern doctor` says which of the two it is.

    `family` narrows the answer to one agent and its descendants. It goes
    through `descendants`, so this function and `family_survivors` cannot
    disagree about who counts. It is applied to the **listing**, before anything
    is read: `StoredSession` already carries `delegating_parent`, and descent
    needs nothing else.

    `touched` is **not** narrowed with it. The caller reports how much of the
    store it looked at. A denominator that shrank with the filter would say that
    the trees of a family came from every session on disk.

    **Each log is folded and discarded, one at a time.** A stored log is seeded
    through the same surface validation that a resume makes. That stops this
    function from counting a tree in a log that the harness would refuse to
    reopen. The fold reads four event types and keeps nothing else.

    `limit` is whatever `stored()` will show, which makes the answer a **floor**
    and not a census. The count and the collector both say so in their own
    words.
    """
    survivors: list[WorkspaceRecord] = []
    touched: dict[str, float] = {}
    try:
        listed = store.stored(limit=limit)
    except Exception:
        log.warning("ph.seams.workspace: could not list stored sessions", exc_info=True)
        return [], {}
    for entry in listed:
        touched[entry.session_id] = entry.modified
    wanted = (
        set(touched)
        if not family
        else set(descendants(((one.session_id, one.delegating_parent) for one in listed), family))
    )
    for entry in listed:
        if entry.session_id not in wanted:
            continue
        try:
            filed = entry.family or None
            # The records of this log that make a tree are read first, which is
            # a fraction of the full read. A log that never made a tree of its
            # own leaves none, whether or not it would reopen. So it is not read
            # and seeded whole to learn that.
            _header, own = store.read_own(entry.session_id, family=filed, types=_OPENERS)
            if not any(_makes_a_tree(event) for event in own):
                continue
            header, events = store.read(entry.session_id, family=filed)
            survivors.extend(workspace_survivors(Session(entry.session_id, events, header)))
        except Exception:
            log.warning(
                "ph.seams.workspace: could not read session %s", entry.session_id, exc_info=True
            )
    return survivors, touched


def family_survivors(sessions: Sequence[Session], agent_id: str) -> list[WorkspaceRecord]:
    """What one agent and everything beneath it left on disk (P6-28).

    **A parent cannot make this fold from its own log**, because the workspace
    events of a child are in the log of the *child*. The link already exists:
    the session of a child names its parent in its own header. So this is a walk
    over state, and not an index to maintain.

    The order is **parent first, then by descent**: what the agent that was
    asked about left, and then what it delegated. `descendants` is
    breadth-first, and this function preserves that order.

    Sessions are passed in and not fetched, as in `reachable_family`. The caller
    has already decided which logs it may open. A session that `agent_id` names
    and that is not in `sessions` yields nothing and does not raise, because a
    truncated listing is an ordinary answer.
    """
    by_id = {session.id: session for session in sessions}
    lineage = [(session.id, session.header.delegating_parent) for session in sessions]
    return [
        record
        for one in descendants(lineage, agent_id)
        if (session := by_id.get(one)) is not None
        for record in workspace_survivors(session)
    ]


def pair_payload(agent_id: str, ref: str | None, **extra: JsonValue) -> dict[str, JsonValue]:
    """The keys that both halves of the durable pair share, spelled one time.

    `ref` is on both halves, so that a reader can say which branch a turn ran
    against without inspecting the repository. It is omitted, and not sent as
    `null`, for a kind that has none. This is a module function because the pair
    has two writers, an orderly release and a reconciliation, and the second is
    the half that nobody watches.
    """
    data: dict[str, JsonValue] = {"agentId": agent_id, **extra}
    if ref is not None:
        data["ref"] = ref
    return data


ACQUIRING = "workspace/acquiring"
"""Where a tier is about to put a fresh tree, before it makes one (S12). It
opens the pair as `ACQUIRED` does. An `ACQUIRED` for the same agent supersedes
it."""
ACQUIRED = "workspace/acquired"
DISPOSED = "workspace/disposed"
"""The durable pair, named one time. The fold (`workspace_survivors`) and both
producers must agree on these names exactly.
"""

RETAINED = "workspace/retained"
"""A tree marked as evidence, recorded at the moment that it is marked (P6-28).

**It is not part of the pair, and the reason is the crash.** The decision
records that a run went wrong, and the most complete way for a run to go wrong
is for the process to die. That writes no `disposed` at all. A retention held
only in memory would be lost by exactly the failure that it exists to survive,
and reconciliation would discard the tree that it was told to keep.

It is ignorable. An older build that skips it reads a deliberate keep as an
ordinary keep.
"""

CHECKPOINT = "workspace/checkpoint"
"""One restore point, recorded at the moment that it is taken (P4-09).

It is here, and not in the git tier that first wrote it, because the capability
now belongs to the seam. The seam appends this event for the two tiers that
take restore points, and three consumers fold it. An event name spelled in one
tier would be a vocabulary that the other tier borrowed.

It is not part of the acquire/dispose pair. A checkpoint opens nothing and
closes nothing, and a fold over the pair must ignore it.
"""


_SURVIVOR_TYPES = frozenset({ACQUIRING, ACQUIRED, DISPOSED, RETAINED})

_OPENERS = frozenset({ACQUIRING, ACQUIRED})
"""The records that name a tree, which `stored_survivors` reads first."""


def _makes_a_tree(event: SessionEvent) -> bool:
    """Whether `event` puts a tree on disk: an opening record of a kind whose root
    is not the base."""
    kind = _WORKSPACE_KINDS.get(as_str(event.data.get("kind")))
    return event.type in _OPENERS and kind is not None and fresh_root(kind)


_WORKSPACE_KINDS: Mapping[str, WorkspaceKind] = literal_lookup(WorkspaceKind)
"""Every `WorkspaceKind` by its own spelling. It is the read-side check for a
kind that is named in a stored log, which a different build may have written.
See `literal_lookup`.

It is built here and not in the fold, because the fold tests it one time for
each event in its hot loop."""

PROJECT_PROVISION_FILE = ".ph-workspace.yml"
"""Where a repository states what its worktrees need (E14).

The file is found by a walk up from the project directory, and it is read for
**data only**: a `copy`, `symlink` or `hardlink` entry, and nothing that
executes. That is what makes it safe to clone a repository and start pH. It is
also why the `command` hook was refused and not gated on trust.

The file is read one time, at mount. A file that appears later needs a restart.
"""


class LifecycleConfig(WireModel):
    """Row config for the lifecycle."""

    access: WorkspaceAccess = "write"
    """What the *root* agent needs of the project directory.

    The access of a child is decided by its parent and arrives with the spawn
    (E4, Q11). This setting is for the user at the keyboard, who asked for a
    harness in their own repository.
    """
    provision: list[ProvisionEntry] = Field(default_factory=list)
    """Materials that every fresh workspace gets: `.env`, a dependency directory,
    or a local config that the project gitignores (E14). The list is empty by
    default, and a profile that names none provisions none. A `shared` workspace
    is never provisioned at all, because its root already *is* the base."""


@plugin(
    "workspace-lifecycle", affects="environment", inject=[WORKSPACE, FS], config=LifecycleConfig
)
async def lifecycle(ctx: Context, config: LifecycleConfig) -> None:
    """Give every agent a workspace, and point `ctx.fs` at it.

    **The seam alone changes nothing, and this row is what puts a tier in
    force.** It is separate from the row of the seam, because the two answer
    different questions: "what happens when someone acquires" and "who acquires,
    and when". A deployment that drives the lifecycle itself wants the first
    without the second.

    Acquisition is *lazy and idempotent*, at the first `agent/pre-step`.
    `agent/created` is an `emit`. A listener there that must
    `await git worktree add` could not hold the agent up, and the first tool
    call would race the checkout. And the workspace of a child is the decision
    of its parent, both the base and the `access`. So the spawn path acquires
    first, and this row must find that workspace and not overwrite it.
    """

    def root_of(agent: AgentHandle) -> Path | None:
        workspace = workspace_of(ctx, agent)
        return None if workspace is None else workspace.root

    ctx.require(FS).rebase(root_of, scope=ctx)

    # The list of the profile and the list of the project are composed here, and
    # not by two registrations. `provision()` accepts many contributors, and
    # this row is the one that knows about both sources.
    entries = [*config.provision, *discover_provisioning(ctx.require(FS).root)]
    if entries:
        ctx.require(WORKSPACE).provision(entries, scope=ctx)

    async def ensure(
        request: PreStepRequest,
        next_: Next[PreStepDecision],
    ) -> PreStepDecision:
        agent = request.agent
        if ctx.require(WORKSPACE).of(agent.id) is None:
            if request.session.header.is_subagent:
                # Refused, and not answered: see `ChildWorkspaceMissing`. The
                # test is the one that the seam uses. `_chosen_tier` reads the
                # same field to decide which rung a child gets. So "is this a
                # child" has one spelling here and cannot come apart from the
                # tier decision.
                raise ChildWorkspaceMissing(
                    f"agent {agent.id} is a subagent with no workspace: its base and access "
                    "are its parent's to decide and arrive with the spawn, so acquiring one "
                    "here would grant it more than its admission recorded"
                )
            await ctx.require(WORKSPACE).acquire(
                session_id=request.session.id,
                agent_id=agent.id,
                # The directory of the process, and never `fs.root_for(agent)`.
                # That is the workspace that is about to be taken, and a
                # worktree branched from the previous one would nest a checkout
                # for each turn.
                base=ctx.require(FS).root,
                access=config.access,
                session=request.session,
                # The own scope of the agent, so the worktree is released when
                # the agent is. That is the in-process half of cleanup (I2). The
                # event pair covers the crash that the scope cannot (§4.9).
                scope=agent.ctx,
            )
        return await next_()

    # Outermost, so a listener that reads or writes files during the step sees
    # the own root of the agent and not the root of the process. Examples are
    # the summarizer of compaction and a permissions row.
    ctx.on("agent/pre-step", ensure, prepend=True)


def discover_provisioning(start: Path) -> list[ProvisionEntry]:
    """The materials list of the project itself, found by a walk up from `start`.

    The nearest file comes first, as in `memory-agents-md`, and here the *first
    file wins*. The file beside the code knows what the code needs, and a
    monorepo root should not override a package that states its own list.

    The file is read with `safe_yaml_load`, the reader that every profile row
    goes through. This is the *least* trusted config that the harness opens, so
    it is the last one that should have parsing rules of its own.

    **Every failure is logged and ignored**: a malformed file, an unknown key,
    or a `source` that names a place outside the tree. To refuse to start
    because the optional config of a repository is wrong would make this list
    load-bearing. The refusal is **for each entry**, in the way that
    `resolve_entry` refuses at use. One line that names `../../etc/passwd` is
    dropped, and the `.env` beside it still arrives.
    """
    for directory in (start, *start.parents):
        candidate = directory / PROJECT_PROVISION_FILE
        if not candidate.is_file():
            continue
        try:
            document = (
                safe_yaml_load(candidate.read_text(encoding="utf-8"), origin=str(candidate)) or {}
            )
        except Exception:
            log.warning("ph.seams.workspace: ignoring %s", candidate, exc_info=True)
            return []
        raw = document.get("provision", []) if isinstance(document, dict) else []
        if not isinstance(raw, list):
            log.warning("ph.seams.workspace: ignoring %s: `provision` is not a list", candidate)
            return []
        return [entry for item in raw if (entry := _discovered(item, candidate)) is not None]
    return []


def _discovered(item: object, candidate: Path) -> ProvisionEntry | None:
    """One entry of the list of a project, or `None` after a warning that says why
    the entry was dropped."""
    try:
        return ProvisionEntry.model_validate(item)
    except ValidationError as refused:
        log.warning(
            "ph.seams.workspace: ignoring an entry of %s: %s",
            candidate,
            validation_summary(refused, root="entry"),
        )
        return None


class Config(WireModel):
    """Row config for the shared provider."""

    scratch: str | None = None
    """Where scratch directories are kept. The default is `$PH_HOME/scratch`,
    which is the idiom that `default_home_path` exists for. It is outside the
    workspace on purpose."""


@plugin("workspace-shared", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam with the shared provider as its floor.

    The row is named for the provider and not for the seam, because a "seam with
    no behavior" state is not useful here. `sandbox-policy` can mount a seam
    whose `confine()` refuses, because a refusal is a real answer. An agent with
    no working directory is not.
    """
    seam = WorkspaceSeam(
        ctx=ctx,
        shared=SharedWorkspaceProvider(),
        scratch_root=default_home_path(config.scratch, "scratch"),
    )
    ctx.provide(WORKSPACE, seam)

    contribute(ctx, Diagnostic(id="workspaces", title="Workspaces", read=seam.describe, order=20))


@plugin("workspace-reconcile", affects="environment", inject=[WORKSPACE])
async def reconcile(ctx: Context, config: None) -> None:
    """Run the reconciliation of the seam whenever a stored session is read (F6).

    **It runs on `session/loaded`, which the reader awaits.**
    `SessionStore.loaded` says why the event is not `session/created`. So a
    session is handed out with its crash already settled, and with its
    `workspace/disposed` records ahead of anything that the new start writes.

    A tree that git cannot remove and a log that cannot be written are both
    handled inside the seam (`_reclaim`, `session_written`). So anything that
    still raises is a bug, and the reader logs it and does not refuse the open
    (`SessionStore.loaded`).
    """
    seam = ctx.require(WORKSPACE)
    # Catch-up, for the reason that `ph.persistence.protocol.attach` gives: a
    # row that is activated after sessions already exist owes them what a loaded
    # session gets. A failure is logged, as the reader would log it, and does
    # not take the row down.
    for session in ctx.require(SESSIONS).list():
        try:
            await seam.reconcile(session)
        except Exception:
            log.exception("ph.seams.workspace: could not reconcile %s", session.id)
    ctx.on("session/loaded", seam.reconcile)


@plugin("workspace-checkpoint", affects="environment", inject=[TOOLS, WORKSPACE])
async def checkpoint_policy(ctx: Context, config: None) -> None:
    """Take a restore point before every code run that has a workspace to save.

    The row wraps the *transport*, because a run is the unit that can be denied
    with work already done (Q9a). A native call that is denied never ran, so
    there is nothing to restore it to. The transport is identified by the
    `transport_name` of the view, because a profile may present it as `python`.

    **The row does not depend on a tier, since the capability became the
    seam's.** This was a git row that cached git directories and asked a
    predicate keyed by kind. A second tier with restore points would have needed
    a second copy of the policy to get them, and it would have had to be added
    to that predicate to be allowed to. The row now asks the seam. `subprocess`
    is gone from `inject`, because nothing here spawns anything any more.
    """

    async def around(
        execution: ToolExecution,
        next_: Next[ToolExecutionResult],
    ) -> ToolExecutionResult:
        # A failure here never blocks the run. A missing restore point is worse
        # than no restore point only if somebody believes that it exists, and
        # the log records which runs have one. The guard is inside the `try` on
        # purpose: to read the tool view is itself a call that must not take a
        # cell down.
        try:
            view = ctx.require(TOOLS).view(execution.scope)
            workspace = workspace_of(ctx, execution.agent)
            if (
                execution.session is not None
                and execution.agent is not None
                and execution.name == view.transport_name
                and workspace is not None
            ):
                await ctx.require(WORKSPACE).checkpoint(
                    workspace,
                    session=execution.session,
                    agent_id=execution.agent.id,
                    call_id=execution.call_id,
                )
        except Exception:
            log.warning("ph.seams.workspace: no restore point for this run", exc_info=True)
        return await next_()

    # **Outermost** (F5), so the snapshot stays out of the `tools-timeout`
    # budget of the cell. That budget belongs to the model and not to the
    # harness. The checkpoint barrier makes the restore point durable before the
    # cell runs, whatever the order: the barrier is on `tools/body`, after every
    # wrapper.
    ctx.on("tools/execute", around, prepend=True)
